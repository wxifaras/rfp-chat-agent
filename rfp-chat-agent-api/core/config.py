from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import List
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Project settings
    PROJECT_NAME: str = "RFP Chat API"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_RFP_CONTAINER_NAME: str
    AZURE_DOCUMENTINTELLIGENCE_ENDPOINT: str
    AZURE_DOCUMENTINTELLIGENCE_API_KEY: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    AZURE_AI_SEARCH_SERVICE_ENDPOINT: str
    AZURE_AI_SEARCH_SERVICE_KEY: str
    AZURE_AI_SEARCH_INDEX_NAME: str
    COSMOS_DATABASE_NAME: str
    COSMOS_CONTAINER_NAME: str
    COSMOS_ENDPOINT: str
    
    # Optional logging settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    @model_validator(mode='after')
    def check_required_fields(self) -> 'Settings':
        for field_name, field in self.__class__.model_fields.items():
            if field.is_required() and getattr(self, field_name) is None:
                raise ValueError(f"{field_name} environment variable is required")
        return self

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()