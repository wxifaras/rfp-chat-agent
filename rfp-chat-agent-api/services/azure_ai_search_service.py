import uuid
from azure.search.documents.indexes.models import SearchIndex, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential 
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
import logging
from typing import List
from core.settings import settings
from services.azure_openai_service import AzureOpenAIService

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

class AzureAISearchService:
    def __init__(self): 
        if not all([
            settings.AZURE_AI_SEARCH_SERVICE_ENDPOINT,
            settings.AZURE_AI_SEARCH_SERVICE_KEY,
            settings.AZURE_AI_SEARCH_INDEX_NAME
        ]):
            raise ValueError("Required Azure AI Search settings are missing")
        
        self.search_index_client = SearchIndexClient(settings.AZURE_AI_SEARCH_SERVICE_ENDPOINT, AzureKeyCredential(settings.AZURE_AI_SEARCH_SERVICE_KEY))
        self.search_client = SearchClient(settings.AZURE_AI_SEARCH_SERVICE_ENDPOINT, settings.AZURE_AI_SEARCH_INDEX_NAME, AzureKeyCredential(settings.AZURE_AI_SEARCH_SERVICE_KEY))

    def create_index(self) -> str:
        # Check if index exists, return if so
        try:
            self.search_index_client.get_index(settings.AZURE_AI_SEARCH_INDEX_NAME)
            print("Index already exists")
            return
        except:
            pass

        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="rfp_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="pursuit_name", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="chunk_content", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SearchField(
                name="chunk_content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=3072,
                vector_search_profile_name="rpf-vector-config",
                retrievable=False
            )
        ]

        vector_search = VectorSearch(
            algorithms=[ HnswAlgorithmConfiguration(name="rpf-vector-config", kind="hnsw", parameters={"m":4, "efConstruction":400}) ],
            profiles=[ VectorSearchProfile(name="rpf-vector-config", algorithm_configuration_name="rpf-vector-config", vectorizer_name="rfp-vectorizer") ],
            vectorizers=[ AzureOpenAIVectorizer(
                vectorizer_name="rfp-vectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                resource_url=settings.AZURE_OPENAI_ENDPOINT,
                deployment_name=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
                model_name=settings.AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME,
                api_key=settings.AZURE_OPENAI_API_KEY
                )
            )]
        )
        
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="chunk_content")],
                title_field=SemanticField(field_name="pursuit_name")
            )
        )
        idx = SearchIndex(
            name=settings.AZURE_AI_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=SemanticSearch(configurations=[semantic_config])
        )

        result = self.search_index_client.create_or_update_index(idx)
        logger.info(f"Index created: {result.name}")
        return result.name
    
    def index_rfp_chunks(self, pursuit_name: str, rfp_id: str, chunks: List[str]) -> List[str]:
        """Embed each chunk and upload to vector index."""
        documents = []
        for chunk in chunks:
            embedding = AzureOpenAIService().create_embedding(chunk)
            chunk_id = str(uuid.uuid4())
            logger.info(f"Generated chunk ID: {chunk_id} for pursuit: {pursuit_name} Chunk content: {chunk[:50]}...")
            documents.append({
                "chunk_id": chunk_id,
                "rfp_id": rfp_id,
                "pursuit_name": pursuit_name,
                "chunk_content": chunk,
                "chunk_content_vector": embedding  # this must match the Collection(Edm.Single) field
            })
            logger.info(f"Prepared document for chunk: {chunk_id}")

        result = self.search_client.upload_documents(documents=documents)
        uploaded = [str(r.key) for r in result if r.succeeded]
        failed = [str(r.key) for r in result if not r.succeeded]

        if failed:
            logger.error(f"Failed to upload chunks: {failed}")
        else:
            logger.info(f"Successfully uploaded {len(uploaded)} chunks.")
        
        return uploaded