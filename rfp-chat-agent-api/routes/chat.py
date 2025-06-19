from typing_extensions import Annotated
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from typing import Dict, Any, Optional
from services.rfp_service import RfpService
from models.process_rfp_response import ProcessRfpResponse
import logging

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

@router.post("/rfp/chat")
async def chat_with_rfp(
    session_id: Annotated[str, Form(...)],
    user_id: Annotated[str, Form(...)],
    question: Annotated[str, Form(...)],
    pursuit_name: Annotated[Optional[str], Form()] = None
):
    try:
        logger.info(f"Received chat request for RFP pursuit: {pursuit_name} with question: {question}")
        search_response = await RfpService().chat_with_rfp(question, pursuit_name, session_id, user_id)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    return search_response

@router.post("/rfp/upload", response_model=ProcessRfpResponse)
async def upload_rfp(
    pursuit_name: Annotated[str, Form(...)] ,
    files: Annotated[list[UploadFile], File(...)],
    data: Annotated[Optional[str], Form()] = None
):
    try:
        logger.info(f"Received RFP upload request for pursuit: {pursuit_name}")
        process_rfp_response = await RfpService().process_rfp(pursuit_name, data, files)           
        return process_rfp_response
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/capabilities/upload")
async def upload_capabilities(files: Annotated[list[UploadFile], File(...)]):
    try:
        logger.info("Received capabilities upload request")
        await RfpService().process_capabilities(files)
    except Exception as e:
        logger.error(f"Error processing capabilities upload: {e}")
        return {"error": str(e)}