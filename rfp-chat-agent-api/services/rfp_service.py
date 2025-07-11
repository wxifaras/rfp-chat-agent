import json
from typing import Dict, List, Optional
from fastapi import File, Form, UploadFile
from models.decision_log import DecisionLog
from models.process_rfp_response import ProcessRfpResponse
from services.azure_ai_search_service import SearchResult
from pathlib import Path
from models.rfp_conversation import RfpConversation, ConversationResult, ReviewDecision, SearchPromptResponse
from core.settings import settings
import uuid
import logging
import tiktoken
import time
from services.service_registry import get_azure_openai_service
from services.service_registry import get_azure_ai_search_service
from services.service_registry import get_azure_storage_service
from services.service_registry import get_azure_doc_intel_service
from services.service_registry import get_chat_history_manager
from models.chat_history import ChatMessage, Role
from prompts.core_prompts import SEARCH_PROMPT

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
            settings.PAGE_OVERLAP
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
        
        new_rfp_parts = []  # Store tuples of (allContent, filename)
        
        # Process each uploaded file
        for file in files:
            file_content = await file.read()
            blob_path, uploaded = get_azure_storage_service().upload_file_with_dup_check(
                pursuit_name,
                file_content,
                file.filename
            )

            if not uploaded:
                logger.info(f"Blob '{blob_path}' already exists. Skipping processing.")
                continue

            logger.info(f"Uploaded '{blob_path}'.")

            # Extract text from the uploaded file
            sas_url = get_azure_storage_service().generate_blob_sas_url(blob_path)
            logger.info(f"SAS URL: {sas_url}")
            time.sleep(5)  # Sleep to ensure SAS URL is generated correctly
            
            allContent = get_azure_doc_intel_service().extract_text_from_url(sas_url)
            content = allContent.content
            logger.info(f"Extracted content: {content[:100]}...")

            # Save extracted text as .txt file
            p = Path(file.filename)
            blob_name_with_txt = p.with_suffix(".txt").name
            text_blob_path = get_azure_storage_service().upload_file(
                pursuit_name,
                content,
                blob_name_with_txt
            )

            logger.info(f"Uploaded text content to '{text_blob_path}'.")
            
            # Add content with filename as tuple
            new_rfp_parts.append((allContent, file.filename))

        # Check if we have any NEW content to process
        if not new_rfp_parts:
            logger.info("No new RFP content to process. All files were duplicates or no files provided.")
            
            return ProcessRfpResponse(
                Pursuit_Name=pursuit_name,
                Decision_Log=None
            )

        new_content_only = "".join([allContent.content for allContent, filename in new_rfp_parts])
        
        # Update the combined RFP file with existing + new content
        existing_rfp = get_azure_storage_service().get_blob(f"{pursuit_name}/{pursuit_name}.txt") or ""
        final_rfp = existing_rfp + new_content_only
        
        final_blob_path = get_azure_storage_service().upload_file(
            pursuit_name,
            final_rfp,
            pursuit_name + ".txt",
        )
        logger.info(f"Updated RFP uploaded to '{final_blob_path}'.")

        # Call LLM ONLY for NEW content
        logger.info(f"Calling LLM for NEW RFP content only with: {len(new_content_only)} characters.")
        llm_response = get_azure_openai_service().get_rfp_decision_log(new_content_only)
        logger.info(f"LLM response received for new content: {llm_response}")

        # Create search index
        index_name = get_azure_ai_search_service().create_index()
        logger.info(f"Index created: {index_name}")

        # Index chunks for each file separately to maintain file name association
        total_indexed = 0
        for allContent, filename in new_rfp_parts:
            chunks = self.chunk_text(allContent)
            logger.info(f"Chunked file '{filename}' into {len(chunks)} parts.")
            
            indexing_result = get_azure_ai_search_service().index_rfp_chunks(
                pursuit_name=pursuit_name,
                rfp_id=rfp_id,
                chunks=[chunk['chunked_text'] for chunk in chunks],
                page_number= [chunk['pages'] for chunk in chunks],
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
        get_azure_storage_service().add_metadata(
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
        existing_capabilities = get_azure_storage_service().get_blob("capabilities/capabilities.txt") or ""
        logger.info(f"Loaded existing capabilities: {existing_capabilities[:100]}...")

        for file in files:
            content_bytes = await file.read()
            blob_path, uploaded= get_azure_storage_service().upload_file_with_dup_check("capabilities", content_bytes, file.filename)

            if not uploaded:        
                logger.info(f"Blob '{blob_path}' already exists. Notifying user.")
                continue  # Skip to next file if already exists

            logger.info(f"Uploaded raw file '{file.filename}' to blob storage.")

            sas_url = get_azure_storage_service().generate_blob_sas_url(f"capabilities/{file.filename}")
            logger.info(f"SAS URL: {sas_url}")
            time.sleep(5)  # wait for blob propagation (if needed)

            extractedAll = get_azure_doc_intel_service().extract_text_from_url(sas_url)
            extracted = extractedAll.content

            logger.info(f"Extracted content (first 100 chars): {extracted[:100]}...")
            new_parts.append(extracted)

        if not new_parts:
            logger.info("No new capabilities extracted from files.")
            return

        combined_text = "\n".join(filter(None, [existing_capabilities] + new_parts))

        blob_path = get_azure_storage_service().upload_file(
            "capabilities",
            combined_text,
            "capabilities.txt"
        )

        logger.info(f"Updated capabilities blob uploaded at '{blob_path}'.")
    
    @staticmethod
    def chunk_text(allContent):
        enc = tiktoken.get_encoding("o200k_base")
        pages = allContent.pages
        page_tokens = []
        page_numbers = []

        # Precompute tokens for each page
        for page in pages:
            page_text = " ".join(w['content'] for w in page.get('words', []))
            tokens = enc.encode(page_text)
            page_tokens.append(tokens)
            page_numbers.append(page.get('pageNumber'))

        chunks = []
        
        page_overlap_percent = float(settings.PAGE_OVERLAP) / 100

        for idx, tokens in enumerate(page_tokens):

            prev_tokens = []
            if idx > 0:
                prev = page_tokens[idx - 1]
                prev_len = max(1, int(len(prev) * page_overlap_percent))
                prev_tokens = prev[-prev_len:]

            next_tokens = []
            if idx < len(page_tokens) - 1:
                nxt = page_tokens[idx + 1]
                next_len = max(1, int(len(nxt) * page_overlap_percent))
                next_tokens = nxt[:next_len]

            combined_tokens = prev_tokens + tokens + next_tokens
            combined_page_map = (
                [page_numbers[idx - 1]] * len(prev_tokens) if idx > 0 else []
            ) + [page_numbers[idx]] * len(tokens) + (
                [page_numbers[idx + 1]] * len(next_tokens) if idx < len(page_tokens) - 1 else []
            )

            chunked_text = enc.decode(combined_tokens)
            pages_in_chunk = sorted(set(p for p in combined_page_map if p is not None))

            chunks.append({
                'chunk_tokens': combined_tokens,
                'chunked_text': chunked_text,
                'pages': pages_in_chunk,
                'page_token_map': combined_page_map
            })

        return chunks
    
    async def chat_with_rfp(self, 
                            user_query: str, 
                            pursuit_name: Optional[str] = None, 
                            session_id: Optional[str] = None, 
                            user_id: Optional[str] = None):
        try:

            if not session_id:
                session_id = str(uuid.uuid4())
                logger.info(f"Generated new session ID: {session_id}")

            # initialize conversation state
            conversation = RfpConversation(
                user_query=user_query,
                pursuit_name=pursuit_name,
                max_attempts=MAX_ATTEMPTS,
                user_id=user_id,
                session_id=session_id
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
        # continue if we have not exceeded max attempts and conversation is not finalized

        # Initialize chat message for user query
        chat_message = ChatMessage(
            user_id= conversation.user_id,
            session_id=conversation.session_id,
            role=Role.USER,
            message=conversation.user_query
        )
                
        get_chat_history_manager().add_message(chat_message)

        while conversation.should_continue():
            # Generate and execute search
            search_query = await self.generate_search_query(conversation)
            await self.execute_search(search_query, conversation)
            
            # Review results
            await self.review_search_results(conversation)

        # Generate final answer by synthesizing vetted results
        final_answer = await self.generate_final_answer(conversation)

        return conversation.to_result(final_answer)
    
    async def generate_search_query(self, conversation: RfpConversation) -> str:
        """Generate search query using the LLM based on the conversation history"""

        logger.info(f"Generating search query for attempt {conversation.attempts + 1}")

        # Build context more clearly
        context_parts = [f"User Question: {conversation.user_query}"]
        
        if conversation.has_search_history():
            context_parts.append("### Previous Search Attempts ###")
            for i, (search, review) in enumerate(zip(conversation.search_history, conversation.reviews), 1):
                context_parts.append(f"<Attempt {i}>\n")
                context_parts.append(f"   search_query: {search['query']}\n")
                context_parts.append(f"   review: {review}\n")
        
        context = "\n".join(context_parts)

        # No reason to query chat history because context is already built from search history and user query
        messages = [
            {"role": "system", "content": SEARCH_PROMPT},
            {"role": "user", "content": context}
        ]
                
        try:
            response = get_azure_openai_service().get_chat_response(messages, SearchPromptResponse)
            
            chat_message = ChatMessage(
                user_id=conversation.user_id,
                session_id=conversation.session_id,
                role=Role.ASSISTANT,
                message=response.search_query)
            
            get_chat_history_manager().add_message(chat_message)

            conversation.add_search_attempt(response.search_query)
            return response.search_query
        except Exception as e:
            logger.error(f"Search query generation failed: {str(e)}")
            # Fallback to user query
            return conversation.user_query
    
    async def execute_search(self, query: str, conversation: RfpConversation):
        """Execute search with proper error handling"""
        try:
            results = get_azure_ai_search_service().run_search(
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
                    "file_name": conversation.file_name,
                    "results_summary": [
                        # The chunk ID may not be super useful here, but it can help track which chunks were returned. If we change the indexing to include source_file, we should use that here instead
                        {
                            "pursuit_name": result.pursuit_name,
                            "file_name": result.source_file,
                            "chunk_id": result.chunk_id
                        }
                        for result in results
                    ]
                }
            })
            
            logger.info(f"Search executed successfully: {len(results)} results found for query '{query}'")

        except Exception as e:
            logger.error(f"Search execution failed for query '{query}': {str(e)}")
            conversation.current_results = []

    async def review_search_results(self, conversation: RfpConversation):
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
            review_decision = get_azure_openai_service().get_chat_response(messages, ReviewDecision)

            conversation.thought_process.append({
                "step": "review",
                "details": {
                    "review_thought_process": review_decision.thought_process,
                    "valid_results": [
                        {
                            "chunk_id": conversation.current_results[idx].chunk_id,
                            "pursuit_name": conversation.current_results[idx].pursuit_name,
                            "source_file": conversation.current_results[idx].source_file
                        }
                        for idx in review_decision.valid_results
                    ],
                    "invalid_results": [ 
                        {
                            "chunk_id": conversation.current_results[idx].chunk_id,
                            "pursuit_name": conversation.current_results[idx].source_file,
                            "source_file": conversation.current_results[idx].source_file
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
                conversation.processed_ids.add(result.chunk_id)
            
            # add all invalid results from this review to the discarded results list of the overall conversation
            for idx in review_decision.invalid_results:
                result = conversation.current_results[idx]
                conversation.discarded_results.append(result)
                conversation.processed_ids.add(result.chunk_id)
            
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
                f"Chunk ID: {result.chunk_id}",
                f"Source: {result.source_file}",
                "\n--- Content ---",
                result.chunk_content,
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

    async def generate_final_answer(self, conversation: RfpConversation) -> str:
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
                    f"ID: {result.chunk_id}",
                    f"Pursuit Name: {result.pursuit_name}",
                    f"Source File: {result.source_file}",
                    "\n<Start Content>",
                    "-" * 80,
                    result.chunk_content,
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

            chat_history = get_chat_history_manager().get_history(conversation.session_id)

            # New Messages
            messages = [
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": llm_input}
            ]

            for msg in messages:
                chat_message = ChatMessage(
                    user_id=conversation.user_id,
                    session_id=conversation.session_id,
                    role=Role(msg["role"]),
                    message=msg["content"]
                )
                get_chat_history_manager().add_message(chat_message)

            for msg in chat_history:
                messages.append({"role": msg.role, "content": msg.message})
            
            final_answer = get_azure_openai_service().get_chat_response_text(messages)
            
            chat_message = ChatMessage(
                user_id=conversation.user_id,
                session_id=conversation.session_id,
                role=Role.ASSISTANT,
                message=final_answer
            )

            get_chat_history_manager().add_message(chat_message)

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