from core.settings import settings
from openai import AzureOpenAI
from typing import List, Optional
from prompts.core_prompts import TRIGGER_RESEARCH_PROMPT
from services.azure_openai_service import AzureOpenAIService
from services.azure_ai_search_service import AzureAISearchService, SearchResult
from models.rfp_conversation import CapabilitiesConversation, ConversationResult, ReviewDecision, SearchPromptResponse
import logging
from azure.storage.blob import BlobServiceClient

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

class AgentIntent:
    def __init__(self):
        if not all([
            settings.AZURE_OPENAI_ENDPOINT,
            settings.AZURE_OPENAI_API_KEY,
            settings.AZURE_OPENAI_API_VERSION,
            settings.AZURE_OPENAI_DEPLOYMENT_NAME
        ]):
            raise ValueError("Required Azure OpenAI settings are missing")

        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME

    async def classify_intent(self, user_input: str) -> ConversationResult:
        """Classifies the intent as 'capability' or 'process' using Azure OpenAI."""

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": TRIGGER_RESEARCH_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )

        result = response.choices[0].message.content.strip().lower()
        if "capability" in result:
            conversation = CapabilitiesConversation(
                user_query=user_input,
                max_attempts=MAX_ATTEMPTS
            )
            conversation_result = await self.execute_conversation_workflow(conversation)
            return conversation_result
        else:
            return result

    async def execute_conversation_workflow(self, conversation: CapabilitiesConversation) -> ConversationResult:
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
    
    async def generate_search_query(self, conversation: CapabilitiesConversation, azure_openai: AzureOpenAIService) -> str:
        """Generate search query using the LLM based on the conversation history"""

        logger.info(f"Generating search query for attempt {conversation.attempts + 1}")

        from prompts.core_prompts import CAPABILITIES_SEARCH_PROMPT
        
        # Build context more clearly
        context_parts = [f"User Question: {conversation.user_query}\n"]
        
        if conversation.has_search_history():
            context_parts.append("### Previous Search Attempts ###")
            for i, (search, review) in enumerate(zip(conversation.search_history, conversation.reviews), 1):
                context_parts.append(f"<Attempt {i}>\n")
                context_parts.append(f"   search_query: {search['query']}\n")
                context_parts.append(f"   review: {review}\n")
        
        context = "\n".join(context_parts)
        
        messages = [
            {"role": "system", "content": CAPABILITIES_SEARCH_PROMPT},
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
    
    async def execute_search(self, query: str, conversation: CapabilitiesConversation, azure_search: AzureAISearchService) -> List[SearchResult]:
        """Execute search with proper error handling"""
        try:
            results = azure_search.run_search(
                search_query=query,
                processed_ids=conversation.processed_ids
                
            )

            conversation.current_results = results
            
            conversation.thought_process.append({
                "step": "retrieve",
                "details": {
                    "user_query": conversation.user_query,
                    "generated_search_query": query,
                    "results_summary": [
                         {"pursuit_name": result["pursuit_name"], "chunk_id": result["chunk_id"]}
                        for result in results
                    ]
                }
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Search execution failed for query '{query}': {str(e)}")
            return []  # Return empty results to continue workflow
    
    async def review_search_results(self, conversation: CapabilitiesConversation, search_results: List[SearchResult], azure_openai: AzureOpenAIService):
        """
        Review search results and determine which are valid/invalid for answering the user's question.
        Uses Azure OpenAI to analyze relevance and make decisions about continuing or finalizing.
        """

        logger.info(f"Reviewing search results.")

        try:
            from prompts.core_prompts import REVIEW_CAPABILITY_DOCUMENT_PROMPT

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
                {"role": "system", "content": REVIEW_CAPABILITY_DOCUMENT_PROMPT},
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

    def format_search_history_for_review(self, conversation: CapabilitiesConversation) -> str:
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

    async def generate_final_answer(self, conversation: CapabilitiesConversation, azure_openai: AzureOpenAIService) -> str:
        """Generate final answer using Azure OpenAI with proper error handling and blob file context"""

        logger.info("Generating final answer.")

        try:
            if not conversation.vetted_results:
                return "I couldn't find relevant information in the RFP documents to answer your question. Please try rephrasing your question or check if the information exists in the uploaded documents."

            # --- Load Blob File Content for Additional Context ---
            blob_context = ""
            try:
                blob_service = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
                container_client = blob_service.get_container_client(settings.AZURE_STORAGE_RFP_CONTAINER_NAME)
                blob_client = container_client.get_blob_client("capabilities/capabilities.txt")
                stream =  blob_client.download_blob()
                blob_context = stream.readall().decode("utf-8")
                logger.info("Blob context loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load blob context file: {str(e)}")

            # --- Format vetted results ---
            vetted_results_formatted = "\n=== Vetted Results ===\n"
            for i, result in enumerate(conversation.vetted_results, 0):
                result_parts = [
                    f"\nResult #{i}",
                    "=" * 80,
                    f"Chunk ID: {result.get('chunk_id')}",
                    f"Document Name: {result.get('pursuit_name')}",
                    "\n<Start Content>",
                    "-" * 80,
                    result.get('chunk_content'),
                    "-" * 80,
                    "<End Content>"
                ]
                vetted_results_formatted += "\n".join(result_parts)

            # --- Final Prompt to LLM ---
            final_prompt = """You are an RFP analysis expert. Your task is to extract the capability answer based on the user's question.

                Return the answer with:
                1. A comprehensive response based on the document chunks.
                2. A markdown-formatted citation for each referenced chunk.
                3. Each citation should include: **Document Name**, **Chunk ID**, and **Quoted Text**.

                Also, use the external document context if needed to provide better guidance.

                Ensure the final answer includes:
                - a clear narrative response
                - cited evidence with exact source
                """

            llm_input = f"""
                User Question: {conversation.user_query}

                <Blob Reference Content>
                {blob_context}
                </Blob Reference Content>

                <Vetted RFP Results>
                {vetted_results_formatted}
                </Vetted RFP Results>

                Now create the answer.
                """

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

            logger.info("Returning final payload.")
            return final_answer
        except Exception as e:
            logger.error(f"Final answer generation failed: {str(e)}")
            return f"I encountered an error generating the final answer. Error: {str(e)}. Please try rephrasing your question."

if __name__ == "__main__":
    classifier = AgentIntent()
    question = "Do we have the required certifications for this project?"
    result = classifier.classify_intent(question)
    print(f"Intent classified as: {result}")
    


