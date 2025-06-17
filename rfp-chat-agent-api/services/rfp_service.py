import json
from typing import List, Optional
from fastapi import File, Form, UploadFile
from models.decision_log import DecisionLog
from models.process_rfp_response import ProcessRfpResponse
from services.azure_storage_service import AzureStorageService
from services.azure_doc_intel_service import AzureDocIntelService
from services.azure_openai_service import AzureOpenAIService
from services.azure_ai_search_service import AzureAISearchService
from pathlib import Path
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
        
        new_rfp_parts = []  # Only new content for LLM processing and indexing
        
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
            
            # Add only new content
            new_rfp_parts.append(content)

        # Check if we have any NEW content to process
        if not new_rfp_parts:
            logger.info("No new RFP content to process. All files were duplicates or no files provided.")
            
            # Return a default response when no new content is available
            default_decision_log = DecisionLog(
                Details="No new RFP content provided",
                Est_TCV="Not specified",
                Division="Not specified",
                Pursuit_Sponsor="Not specified",
                CDD_Booking_Date="2025-01-01",
                Resources_Required="Not specified",
                Pursuit_Due_Date="2025-01-01",
                Client_Ministry_Name="Not specified",
                I_have_reviewed_the_Opportunity_Review_Policy_the_pursuit_lead_has_the_following_triggers="Not specified",
                Deal_type_current="Not specified",
                Expected_Gross_Margin="Not specified",
                Does_Maximus_Canada_have_the_qualifications="Not specified",
                Rfp_Id=rfp_id
            )
            
            return ProcessRfpResponse(
                Pursuit_Name=pursuit_name,
                Decision_Log=default_decision_log
            )

        # Process only NEW content
        new_content_only = "".join(new_rfp_parts)
        
        # Update the combined RFP file with existing + new content
        existing_rfp = AzureStorageService().get_blob(f"{pursuit_name}/{pursuit_name}.txt") or ""
        
        # Add proper separation between existing and new content
        if existing_rfp and new_content_only:
            final_rfp = existing_rfp + "\n\n" + new_content_only
        else:
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

        # Index only NEW content
        chunks = self.chunk_text(new_content_only)
        logger.info(f"Chunked NEW RFP content into {len(chunks)} parts for indexing.")

        # Create search index and index only the new chunks
        index_name = AzureAISearchService().create_index()
        logger.info(f"Index created: {index_name}")

        indexing_result = AzureAISearchService().index_rfp_chunks(
            pursuit_name=pursuit_name,
            rfp_id=rfp_id,
            chunks=chunks
        )
        logger.info(f"Indexed {len(indexing_result)} NEW chunks for Pursuit: {pursuit_name}")

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
            
            # Create a fallback decision log with required fields
            fallback_decision_log = DecisionLog(
                Details=merged.get("Details", "New RFP content processed"),
                Est_TCV=merged.get("Est_TCV", "Not specified"),
                Division=merged.get("Division", "Not specified"),
                Pursuit_Sponsor=merged.get("Pursuit_Sponsor", "Not specified"),
                CDD_Booking_Date=merged.get("CDD_Booking_Date", "2025-01-01"),
                Resources_Required=merged.get("Resources_Required", "Not specified"),
                Pursuit_Due_Date=merged.get("Pursuit_Due_Date", "2025-01-01"),
                Client_Ministry_Name=merged.get("Client_Ministry_Name", "Not specified"),
                I_have_reviewed_the_Opportunity_Review_Policy_the_pursuit_lead_has_the_following_triggers=merged.get("I_have_reviewed_the_Opportunity_Review_Policy_the_pursuit_lead_has_the_following_triggers", "Not specified"),
                Deal_type_current=merged.get("Deal_type_current", "Not specified"),
                Expected_Gross_Margin=merged.get("Expected_Gross_Margin", "Not specified"),
                Does_Maximus_Canada_have_the_qualifications=merged.get("Does_Maximus_Canada_have_the_qualifications", "Not specified"),
                Rfp_Id=rfp_id
            )
            
            final_decision_log = fallback_decision_log
            logger.info("Created fallback DecisionLog model")

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
