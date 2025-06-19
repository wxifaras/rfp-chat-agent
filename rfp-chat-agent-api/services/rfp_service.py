import json
from typing import List, Optional
from fastapi import File, Form, UploadFile
from models.decision_log import DecisionLog
from models.process_rfp_response import ProcessRfpResponse
from services.azure_storage_service import AzureStorageService
from services.azure_doc_intel_service import AzureDocIntelService
from services.azure_openai_service import AzureOpenAIService
from services.azure_ai_search_service import AzureAISearchService, SearchResult
from pathlib import Path
from models.rfp_conversation import RfpConversation, ConversationResult, ReviewDecision, SearchPromptResponse
from core.settings import settings
import uuid
import logging
import tiktoken
import time

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# Console handler (prints to terminal)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)

# Add handler
logger.addHandler(ch)

MAX_ATTEMPTS = 3

class RfpService:
    def __init__(self):
        if not all([
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP
        ]):
            raise ValueError("Required settings are missing")

    async def process_rfp(
            self, 
            pursuit_name: str = Form(...),
            data: Optional[str] = Form(None),
            files: List[UploadFile] = File(...)
        ):
        """
            Process an uploaded RFP end-to-end.

            - Extract plain text from the uploaded RFP using Azure Document Intelligence.
            - Use the LLM to extract structured metadata from the RFP content.
            - Persist the original file, extracted text, and metadata in Azure Blob Storage.
            - Ensure the Azure AI Search index exists; create it if not present.
            - Chunk the RFP text, generate embeddings, and upload them to the AI Search index.
            - Return the merged metadata for further use.
        """
        
        rfp_id = str(uuid.uuid4())
        decision_log_dict = {}
        
        # Parse decision log data if provided
        if data:
            try:
                decision_log_data = json.loads(data)
                decision_log_dict = DecisionLog(**decision_log_data).model_dump(exclude_none=True)
            except Exception as e:
                logger.error(f"Error validating decision log data: {e}")
                raise ValueError("Invalid decision log data provided")
        
        new_rfp_parts = []  # Store tuples of (content, filename)
        
        # Process each uploaded file
        for file in files:
            file_content = await file.read()
            blob_path, uploaded = AzureStorageService().upload_file_with_dup_check(
                pursuit_name,
                file_content,
                file.filename
            )

            if not uploaded:
                logger.info(f"Blob '{blob_path}' already exists. Skipping processing.")
                continue

            logger.info(f"Uploaded '{blob_path}'.")

            # Extract text from the uploaded file
            sas_url = AzureStorageService().generate_blob_sas_url(blob_path)
            logger.info(f"SAS URL: {sas_url}")
            time.sleep(5)  # Sleep to ensure SAS URL is generated correctly
            
            content = AzureDocIntelService().extract_text_from_url(sas_url)
            logger.info(f"Extracted content: {content[:100]}...")

            # Save extracted text as .txt file
            p = Path(file.filename)
            blob_name_with_txt = p.with_suffix(".txt").name
            text_blob_path = AzureStorageService().upload_file(
                pursuit_name,
                content,
                blob_name_with_txt
            )

            logger.info(f"Uploaded text content to '{text_blob_path}'.")
            
            # Add content with filename as tuple
            new_rfp_parts.append((content, file.filename))

        # Check if we have any NEW content to process
        if not new_rfp_parts:
            logger.info("No new RFP content to process. All files were duplicates or no files provided.")
            
            return ProcessRfpResponse(
                Pursuit_Name=pursuit_name,
                Decision_Log=None
            )

        new_content_only = "".join([content for content, filename in new_rfp_parts])
        
        # Update the combined RFP file with existing + new content
        existing_rfp = AzureStorageService().get_blob(f"{pursuit_name}/{pursuit_name}.txt") or ""
        final_rfp = existing_rfp + new_content_only
        
        final_blob_path = AzureStorageService().upload_file(
            pursuit_name,
            final_rfp,
            pursuit_name + ".txt",
        )
        logger.info(f"Updated RFP uploaded to '{final_blob_path}'.")

        # Call LLM ONLY for NEW content
        logger.info(f"Calling LLM for NEW RFP content only with: {len(new_content_only)} characters.")
        llm_response = AzureOpenAIService().get_rfp_decision_log(new_content_only)
        logger.info(f"LLM response received for new content: {llm_response}")

        # Create search index
        index_name = AzureAISearchService().create_index()
        logger.info(f"Index created: {index_name}")

        # Index chunks for each file separately to maintain file name association
        total_indexed = 0
        for content, filename in new_rfp_parts:
            chunks = self.chunk_text(content)
            logger.info(f"Chunked file '{filename}' into {len(chunks)} parts.")
            
            indexing_result = AzureAISearchService().index_rfp_chunks(
                pursuit_name=pursuit_name,
                rfp_id=rfp_id,
                chunks=chunks,
                file_name=filename
            )
            
            total_indexed += len(indexing_result)
            logger.info(f"Indexed {len(indexing_result)} chunks for file '{filename}' in pursuit: {pursuit_name}")
        
        logger.info(f"Total indexed {total_indexed} chunks across all files uploaded for Pursuit: {pursuit_name}")

        # Prepare metadata for storage
        llm_response_dict = llm_response.model_dump()
        
        # Merge decision log and LLM response (from NEW content only)
        merged = {**llm_response_dict, **{k: v for k, v in decision_log_dict.items() if v not in (None, "")}}     
        merged["Rfp_Id"] = rfp_id

        # Store metadata
        AzureStorageService().add_metadata(
            folder=pursuit_name,
            metadata=merged
        )
        logger.info(f"Metadata stored for Pursuit: {pursuit_name}")

        # Validate and create final decision log
        try:
            # Filter merged data to only include valid DecisionLog fields
            valid_fields = set(DecisionLog.model_fields.keys())
            filtered_data = {k: v for k, v in merged.items() if k in valid_fields}
            
            final_decision_log = DecisionLog.model_validate(filtered_data)
            logger.info("Successfully validated DecisionLog model")
            
        except Exception as e:
            logger.error(f"Error validating DecisionLog: {e}")
            logger.error(f"Available data: {merged}")
            
            final_decision_log = None
            logger.info("Set decision log to None due to validation error")

        # Create and return the response
        process_rfp_response = ProcessRfpResponse(
            Pursuit_Name=pursuit_name,
            Decision_Log=final_decision_log
        )

        return process_rfp_response
    
    async def process_capabilities(self, files: list[UploadFile] = File(...)):
        new_parts = []
        existing_capabilities = AzureStorageService().get_blob("capabilities/capabilities.txt") or ""
        logger.info(f"Loaded existing capabilities: {existing_capabilities[:100]}...")

        for file in files:
            content_bytes = await file.read()
            blob_path, uploaded= AzureStorageService().upload_file_with_dup_check("capabilities", content_bytes, file.filename)

            if not uploaded:        
                logger.info(f"Blob '{blob_path}' already exists. Notifying user.")
                continue  # Skip to next file if already exists

            logger.info(f"Uploaded raw file '{file.filename}' to blob storage.")

            sas_url = AzureStorageService().generate_blob_sas_url(f"capabilities/{file.filename}")
            logger.info(f"SAS URL: {sas_url}")
            time.sleep(5)  # wait for blob propagation (if needed)

            extracted = AzureDocIntelService().extract_text_from_url(sas_url)
            logger.info(f"Extracted content (first 100 chars): {extracted[:100]}...")
            new_parts.append(extracted)

        if not new_parts:
            logger.info("No new capabilities extracted from files.")
            return

        combined_text = "\n".join(filter(None, [existing_capabilities] + new_parts))

        blob_path = AzureStorageService().upload_file(
            "capabilities",
            combined_text,
            "capabilities.txt"
        )

        logger.info(f"Updated capabilities blob uploaded at '{blob_path}'.")
    
    @staticmethod
    def chunk_text(text):
        enc = tiktoken.get_encoding("o200k_base")
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + settings.CHUNK_SIZE, len(tokens))
            chunk = enc.decode(tokens[start:end])
            chunks.append(chunk)
            start += settings.CHUNK_SIZE - settings.CHUNK_OVERLAP

        return chunks
    
    async def chat_with_rfp(self, user_query: str, pursuit_name: Optional[str] = None):
        try:
            # initialize conversation state
            conversation = RfpConversation(
                user_query=user_query,
                pursuit_name=pursuit_name,
                max_attempts=MAX_ATTEMPTS
            )

            # Execute conversation workflow
            result = await self.execute_conversation_workflow(conversation)

            return {
                "answer": result.final_answer,
                "citations": result.citations,
                "thought_process": result.thought_process,
                "attempts": result.attempts,
                "search_queries": result.search_queries
            }

        except Exception as e:
            logger.error(f"Chat workflow failed: {str(e)}")
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "citations": [],
                "thought_process": [],
                "attempts": 0,
                "search_queries": []
            }
    
    async def execute_conversation_workflow(self, conversation: RfpConversation) -> ConversationResult:
        """Executes the agentic rag workflow"""
        azure_openai = AzureOpenAIService()
        azure_search = AzureAISearchService()
        
        # continue if we have not exceeded max attempts and conversation is not finalized
        while conversation.should_continue():
            # Generate and execute search
            search_query = await self.generate_search_query(conversation, azure_openai)
            search_results = await self.execute_search(search_query, conversation, azure_search)
            
            # Review results
            await self.review_search_results(conversation, search_results, azure_openai)

        # Generate final answer by synthesizing vetted results
        final_answer = await self.generate_final_answer(conversation, azure_openai)

        return conversation.to_result(final_answer)
    
    async def generate_search_query(self, conversation: RfpConversation, azure_openai: AzureOpenAIService) -> str:
        """Generate search query using the LLM based on the conversation history"""

        logger.info(f"Generating search query for attempt {conversation.attempts + 1}")

        from prompts.core_prompts import SEARCH_PROMPT
        
        # Build context more clearly
        context_parts = [f"User Question: {conversation.user_query}"]
        
        if conversation.has_search_history():
            context_parts.append("### Previous Search Attempts ###")
            for i, (search, review) in enumerate(zip(conversation.search_history, conversation.reviews), 1):
                context_parts.append(f"<Attempt {i}>\n")
                context_parts.append(f"   search_query: {search['query']}\n")
                context_parts.append(f"   review: {review}\n")
        
        context = "\n".join(context_parts)
        
        messages = [
            {"role": "system", "content": SEARCH_PROMPT},
            {"role": "user", "content": context}
        ]
        
        try:
            response = azure_openai.get_chat_response(messages, SearchPromptResponse)
            conversation.add_search_attempt(response.search_query)
            return response.search_query
        except Exception as e:
            logger.error(f"Search query generation failed: {str(e)}")
            # Fallback to user query
            return conversation.user_query
    
    async def execute_search(self, query: str, conversation: RfpConversation, azure_search: AzureAISearchService) -> List[SearchResult]:
        """Execute search with proper error handling"""
        try:
            results = azure_search.run_search(
                search_query=query,
                processed_ids=conversation.processed_ids,
                pursuit_name=conversation.pursuit_name
            )

            conversation.current_results = results
            
            conversation.thought_process.append({
                "step": "retrieve",
                "details": {
                    "user_query": conversation.user_query,
                    "generated_search_query": query,
                    "pursuit_name": conversation.pursuit_name,
                    "results_summary": [
                        # The chunk ID may not be super useful here, but it can help track which chunks were returned. If we change the indexing to include source_file, we should use that here instead
                        {"pursuit_name": result["pursuit_name"], "chunk_id": result["chunk_id"]}
                        for result in results
                    ]
                }
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Search execution failed for query '{query}': {str(e)}")
            return []  # Return empty results to continue workflow

    async def review_search_results(self, conversation: RfpConversation, search_results: List[SearchResult], azure_openai: AzureOpenAIService):
        """
        Review search results and determine which are valid/invalid for answering the user's question.
        Uses Azure OpenAI to analyze relevance and make decisions about continuing or finalizing.
        """

        logger.info(f"Reviewing search results.")

        try:
            from prompts.core_prompts import SEARCH_REVIEW_PROMPT

            # Format current search results for review
            current_results_formatted = self.format_search_results_for_review(conversation.current_results)
            
            # Format previously vetted results (don't review these again)
            vetted_results_formatted = self.format_search_results_for_review(conversation.vetted_results)
            
            # Format search history for context
            search_history_formatted = self.format_search_history_for_review(conversation)
            
            # Construct the review prompt with all context
            llm_input = f"""
                User Question: {conversation.user_query}

                <Current Search Results to review>
                {current_results_formatted}
                <end current search results to review>

                <previously vetted results, do not review>
                {vetted_results_formatted}
                <end previously vetted results, do not review>

                <Previous Attempts>
                {search_history_formatted}
                <end Previous Attempts>
                """
            
            messages = [
                {"role": "system", "content": SEARCH_REVIEW_PROMPT},
                {"role": "user", "content": llm_input}
            ]

            # Get review decision from Azure OpenAI
            review_decision = azure_openai.get_chat_response(messages, ReviewDecision)

            conversation.thought_process.append({
                "step": "review",
                "details": {
                    "review_thought_process": review_decision.thought_process,
                    "valid_results": [
                        {
                            "chunk_id": conversation.current_results[idx]["chunk_id"],
                            "pursuit_name": conversation.current_results[idx]["pursuit_name"],
                        }
                        for idx in review_decision.valid_results
                    ],
                    "invalid_results": [ 
                        {
                            "chunk_id": conversation.current_results[idx]["chunk_id"],
                            "pursuit_name": conversation.current_results[idx]["pursuit_name"],
                        }
                        for idx in review_decision.invalid_results
                    ],
                    "decision": review_decision.decision
                }
            })

            # Validate indices before using them as sometimes Azure OpenAI can return indices like [0,1,2,3,4] when you only have 2 results, causing an index error
            current_results_count = len(conversation.current_results)
            
            # Filter out invalid indices to prevent IndexError
            valid_indices = [idx for idx in review_decision.valid_results if 0 <= idx < current_results_count]
            invalid_indices = [idx for idx in review_decision.invalid_results if 0 <= idx < current_results_count]
            
            # Log warnings if Azure OpenAI returned invalid indices
            if len(valid_indices) != len(review_decision.valid_results):
                invalid_valid_indices = [idx for idx in review_decision.valid_results if idx not in valid_indices]
                logger.warning(f"Azure OpenAI returned invalid valid_results indices: {invalid_valid_indices}. Current results count: {current_results_count}")
            
            if len(invalid_indices) != len(review_decision.invalid_results):
                invalid_invalid_indices = [idx for idx in review_decision.invalid_results if idx not in invalid_indices]
                logger.warning(f"Azure OpenAI returned invalid invalid_results indices: {invalid_invalid_indices}. Current results count: {current_results_count}")

            conversation.reviews.append(review_decision.thought_process)
            conversation.decisions.append(review_decision.decision)

            # add all valid results from this review to the vetted results list of the overall conversation
            for idx in review_decision.valid_results:
                result = conversation.current_results[idx]
                conversation.vetted_results.append(result)
                conversation.processed_ids.add(result["chunk_id"])
            
            # add all invalid results from this review to the discarded results list of the overall conversation
            for idx in review_decision.invalid_results:
                result = conversation.current_results[idx]
                conversation.discarded_results.append(result)
                conversation.processed_ids.add(result["chunk_id"])
            
            # resest the current results to empty for the next search
            conversation.current_results = []
            
        except Exception as e:
            logger.error(f"Search results review failed: {str(e)}")

    def format_search_results_for_review(self, results: List[SearchResult]) -> str:
        """Format search results for the review prompt with clear structure"""
        if not results:
            return "No results available."
        
        output_parts = ["\n=== Search Results ==="]
        for i, result in enumerate(results, 0):
            result_section = [
                f"\nResult #{i}",
                "=" * 80,
                f"Chunk ID: {result.get('chunk_id', 'Unknown')}",
                f"Source: {result.get('pursuit_name', 'Unknown')}",
                "\n--- Content ---",
                result.get('chunk_content', 'No content available'),
                "--- End Content ---"
            ]
            output_parts.extend(result_section)
        
        return "\n".join(output_parts)

    def format_search_history_for_review(self, conversation: RfpConversation) -> str:
        """Format search history for context in the review prompt"""
        if not conversation.search_history:
            return "No previous search attempts."
        
        history_parts = ["\n=== Search History ==="]
        for i, (search, review) in enumerate(zip(conversation.search_history, conversation.reviews), 1):
            history_parts.extend([
                f"<Attempt {i}>",
                f"   Query: {search['query']}",
                f"   Review: {review}",
                "</Attempt>"
            ])
        
        return "\n".join(history_parts)

    async def generate_final_answer(self, conversation: RfpConversation, azure_openai: AzureOpenAIService) -> str:
        """Generate final answer using Azure OpenAI with proper error handling"""
        
        logger.info(f"Generating final answer.")
        
        try:
            if not conversation.vetted_results:
                return "I couldn't find relevant information in the RFP documents to answer your question. Please try rephrasing your question or check if the information exists in the uploaded documents."
            
            # Format vetted results in the same way as review node
            vetted_results_formatted = "\n=== Vetted Results ===\n"
            for i, result in enumerate(conversation.vetted_results, 0):
                result_parts = [
                    f"\nResult #{i}",
                    "=" * 80,
                    f"ID: {result.get('chunk_id')}",
                    f"Pursuit Name: {result.get('pursuit_name')}",
                    "\n<Start Content>",
                    "-" * 80,
                    result.get('chunk_content'),
                    "-" * 80,
                    "<End Content>"
                ]
                vetted_results_formatted += "\n".join(result_parts)

            final_prompt = """Create a comprehensive answer to the user's question using the vetted results."""
            
            llm_input = f"""Create a comprehensive answer to the user's question using the vetted results.

                User Question: {conversation.user_query}

                Vetted Results:
                {vetted_results_formatted}

                Synthesize these results into a clear, complete answer. If there were no vetted results, say you couldn't find any relevant information to answer the question.

                Guidance:
                - Always use valid markdown syntax. Try to use level 1 or level 2 headers for your sections. 
                - Cite your sources using the following format: some text <cit>pursuit name - chunk id</cit> , some more text <cit>pursuit name - chunk id> , etc.
                - Only cite sources that are actually used in the answer."""

            messages = [
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": llm_input}
            ]
            
            final_answer = azure_openai.get_chat_response_text(messages)
            
            conversation.thought_process.append({
                "step": "response",
                "details": {
                    "final_answer": final_answer
                }
            })

            logger.info(f"Sending final payload.")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Final answer generation failed: {str(e)}")
            return f"I encountered an error generating the final answer. Error: {str(e)}. Please try rephrasing your question."
