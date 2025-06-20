"""
Centralized singleton management for Azure services
"""

from functools import lru_cache
from services.azure_openai_service import AzureOpenAIService
from services.azure_ai_search_service import AzureAISearchService
from services.azure_storage_service import AzureStorageService
from services.azure_doc_intel_service import AzureDocIntelService
from services.chat_history_manager import ChatHistoryManager
import logging
from core.settings import settings

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

@lru_cache(maxsize=1)
def get_azure_openai_service():
    """
    Create and cache a single instance of AzureOpenAIService
    """
    logger.info("Creating new AzureOpenAIService singleton instance")

    return AzureOpenAIService()

@lru_cache(maxsize=1)
def get_azure_ai_search_service():
    """
    Create and cache a single instance of AzureAISearchService
    """
    logger.info("Creating new AzureAISearchService singleton instance")
    
    return AzureAISearchService()

@lru_cache(maxsize=1)
def get_azure_storage_service():
    """
    Create and cache a single instance of AzureStorageService
    """
    logger.info("Creating new AzureStorageService singleton instance")
    
    return AzureStorageService()

@lru_cache(maxsize=1)
def get_azure_doc_intel_service():
    """
    Create and cache a single instance of AzureDocIntelService
    """
    logger.info("Creating new AzureDocIntelService singleton instance")
    
    return AzureDocIntelService()

@lru_cache(maxsize=1)
def get_chat_history_manager():
    """
    Create and cache a single instance of ChatHistoryManager
    """
    logger.info("Creating new ChatHistoryManager singleton instance")
    
    return ChatHistoryManager()