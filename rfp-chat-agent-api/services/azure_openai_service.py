from openai import AzureOpenAI
from prompts.core_prompts import RFP_INGESTION_SYSTEM_PROMPT
from core.settings import settings
from models.decision_log import DecisionLog
import json
from typing import List, Dict, Generator

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
    
    # This method is used to get a chat response from the Azure OpenAI service using the supplied response format for a structured output response
    def get_chat_response(
            self,
            messages: List[Dict[str, str]],
            response_format: type
        ):
        response = self.client.beta.chat.completions.parse(
            model=self.deployment_name,
            messages=messages,
            response_format=response_format
        )

        message_content = response.choices[0].message.parsed
        return message_content
    
    def get_chat_response_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Gets a simple text response from Azure OpenAI (non-streaming, no structured output)
        """
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages
        )
        
        message_content = response.choices[0].message.content
        
        return message_content

    def get_chat_response_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Get streaming chat response from Azure OpenAI
        Yields content chunks as they arrive
        """
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            #if chunk.choices[0].delta.content is not None:
            #    yield chunk.choices[0].delta.content
            # Check if chunk has choices and if the first choice has delta content
            if (chunk.choices and 
                len(chunk.choices) > 0 and 
                chunk.choices[0].delta and 
                chunk.choices[0].delta.content is not None):
                yield chunk.choices[0].delta.content
                    

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