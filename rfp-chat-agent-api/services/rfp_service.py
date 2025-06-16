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
            - Chunk the RFP text, generate embeddings, and upload them to theAI Search index.
            - Return the merged metadata for further use.
        """
        
        rfp_id = str(uuid.uuid4())
        decision_log_dict = {}
        
            try:
            decision_log_data = json.loads(data)
            decision_log_dict = DecisionLog(**decision_log_data).model_dump(exclude_none=True)
        except Exception as e:
            logger.error(f"Error validating decision log data: {e}")
            raise ValueError("Invalid decision log data provided")
        
        rfp_parts = []
        for file in files:
            file_content = await file.read()
        
            blob_path = AzureStorageService().upload_file(
                pursuit_name,
                file_content,
                file.filename
            )

            logger.info(f"Uploaded '{blob_path}'.")

            sas_url = AzureStorageService().generate_blob_sas_url(blob_path)
            logger.info(f"SAS URL: {sas_url}")
            time.sleep(5)  # Sleep to ensure SAS URL is generated correctly
            content = AzureDocIntelService().extract_text_from_url(sas_url)
            logger.info(f"Extracted content: {content[:100]}...")  # Print first 100 characters for brevity

            p = Path(file.filename)
            blob_name_with_txt = p.with_suffix(".txt").name
            blob_path = AzureStorageService().upload_file(
                pursuit_name,
                content,
                blob_name_with_txt
            )

            logger.info(f"Uploaded text content to '{blob_path}'.")

            rfp_parts.append(content)
        
        final_rfp = "".join(rfp_parts)

        blob_path = AzureStorageService().upload_file(
            pursuit_name,
            final_rfp,
            pursuit_name + ".txt",
        )

        logger.info(f"Final RFP uploaded to '{blob_path}'.")

        llm_response = AzureOpenAIService().get_rfp_decision_log(final_rfp)
        logger.info(f"LLM Response: {llm_response}")

        llm_response_dict = llm_response.model_dump()

        # Merge decision log and LLM response and use that as metadata
        merged = {**llm_response_dict, **{k: v for k, v in decision_log_dict.items() if v not in (None, "")}}     
        merged["rfp_id"] = rfp_id
        
        AzureStorageService().add_metadata(
            folder=pursuit_name,
            metadata=merged
        )

        logger.info(f"Metadata stored for Pursuit: {pursuit_name}")

        chunks = self.chunk_text(final_rfp)
        logger.info(f"Chunked RFP into {len(chunks)} parts.")

        index_name = AzureAISearchService().create_index()
        logger.info(f"Index created: {index_name}")

        indexing_result = AzureAISearchService().index_rfp_chunks(
            pursuit_name=pursuit_name,
            rfp_id=rfp_id,
            chunks=chunks
        )

        logger.info(f"Indexed {len(indexing_result)} chunks for Pursuit: {pursuit_name}")

        # validate and use the DecisionLog model in the repsonse
        final_decision_log = DecisionLog.model_validate(merged)

        process_rfp_response = ProcessRfpResponse(
            Pursuit_Name=pursuit_name,
            Decision_Log=final_decision_log
        )

        return process_rfp_response
    
    async def process_capabilities(self, files: list[UploadFile] = File(...)):
        new_parts = []
        existing_capabilities = AzureStorageService().get_capabilities() or ""
        logger.info(f"Loaded existing capabilities: {existing_capabilities[:100]}...")

        for file in files:
            content_bytes = await file.read()
            AzureStorageService().upload_file("capabilities", content_bytes, file.filename)
            logger.info(f"Uploaded raw file '{file.filename}' to blob storage.")

            sas_url = AzureStorageService().generate_blob_sas_url(f"capabilities/{file.filename}")
            logger.info(f"SAS URL: {sas_url}")
            time.sleep(5)  # wait for blob propagation (if needed)

            extracted = AzureDocIntelService().extract_text_from_url(sas_url)
            logger.info(f"Extracted content (first 100 chars): {extracted[:100]}...")
            new_parts.append(extracted)

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
