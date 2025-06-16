from openai import AzureOpenAI
from prompts.core_prompts import RFP_INGESTION_SYSTEM_PROMPT
from prompts.core_prompts import SYSTEM_PROMPT
from core.settings import settings
from models.decision_log import DecisionLog
import json

class AzureOpenAIService:
    def __init__(self):
        if not all([
            settings.AZURE_OPENAI_ENDPOINT,
            settings.AZURE_OPENAI_API_KEY,
            settings.AZURE_OPENAI_API_VERSION,
            settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME
        ]):
            raise ValueError("Required Azure OpenAI settings are missing")

        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME

    def get_rfp_decision_log(
            self, 
            rfp_content: str
        ):

        response = self.client.beta.chat.completions.parse(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": RFP_INGESTION_SYSTEM_PROMPT},
                {"role": "user", "content": rfp_content}
            ],
            response_format=DecisionLog
        )

        message_content = response.choices[0].message.parsed
        return message_content
    
    def create_embedding(
            self, 
            text: str
        ):

        response = self.client.embeddings.create(
            model=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
            input=text
        )
        
        embeddings = response.data[0].embedding
        return embeddings