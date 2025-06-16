from typing_extensions import Annotated
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import ValidationError
from typing import Dict, Any, Optional
from services.rfp_service import RfpService
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

@router.get("/rfp/chat", response_model=Dict[str, Any])
async def chat_with_rfp():
    return {"message": "Hello World"}

@router.post("/rfp/upload")
async def upload_rfp(
    pursuit_name: Annotated[str, Form(...)] ,
    files: Annotated[list[UploadFile], File(...)],
    data: Annotated[Optional[str], Form()] = None
):
    try:
        logger.info(f"Received RFP upload request for pursuit: {pursuit_name}")
        process_rfp_response = await RfpService().process_rfp(pursuit_name, data, files)           
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    return process_rfp_response

@router.post("/capabilities/upload")
async def upload_capabilities(files: Annotated[list[UploadFile], File(...)]):
    try:
        logger.info("Received capabilities upload request")
        await RfpService().process_capabilities(files)
    except Exception as e:
        logger.error(f"Error processing capabilities upload: {e}")
        return {"error": str(e)}