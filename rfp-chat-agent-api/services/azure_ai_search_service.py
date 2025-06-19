import uuid
from azure.search.documents.indexes.models import SearchIndex, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential 
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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
from typing import List, TypedDict, Set, Optional
from core.settings import settings
from services.azure_openai_service import AzureOpenAIService

NUM_SEARCH_RESULTS = 5
K_NEAREST_NEIGHBORS = 30

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

# Type Definitions
class SearchResult(TypedDict):
    id: str
    content: str
    source_file: str
    source_pages: int
    score: float

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
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True, key=True),
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
            profiles=[ VectorSearchProfile(name="rpf-vector-config", algorithm_configuration_name="rpf-vector-config") ]
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
    
    def run_search(
            self,
            search_query: str,
            processed_ids: Set[str],
            #category_filter: str | None = None,
            pursuit_name: Optional[str] = None
        ) -> List[SearchResult]:
        """
        Perform a search using Azure Cognitive Search with both semantic and vector queries.
        """
        query_vector = AzureOpenAIService().create_embedding(search_query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=K_NEAREST_NEIGHBORS,
            fields="chunk_content_vector",
        )
        filter_parts = []
        if processed_ids:
            ids_string = ','.join(processed_ids)
            filter_parts.append(f"not search.in(chunk_id, '{ids_string}')")
        #if category_filter:
        #    filter_parts.append(f"({category_filter})")
        if pursuit_name:
            filter_parts.append(f"(pursuit_name eq '{pursuit_name}')")
        filter_str = " and ".join(filter_parts) if filter_parts else None

        results = self.search_client.search(
            search_text=search_query,
            vector_queries=[vector_query],
            filter=filter_str,
            select=["chunk_id", "chunk_content", "pursuit_name", "file_name"],
            top=NUM_SEARCH_RESULTS
        )
        search_results = []
        for result in results:
            search_result: SearchResult = {
                "chunk_id": result["chunk_id"],
                "chunk_content": result["chunk_content"],
                "pursuit_name": result["pursuit_name"],
                "score": result["@search.score"]
            }
            search_results.append(search_result)
        return search_results