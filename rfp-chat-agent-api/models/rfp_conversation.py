from dataclasses import dataclass, field
from typing import List, Set, Optional
from typing import Literal
from services.azure_ai_search_service import SearchResult
from pydantic import BaseModel

class SearchPromptResponse(BaseModel):
    search_query: str
    #filter: str | None

@dataclass
class RfpConversation:
    user_query: str
    pursuit_name: Optional[str]
    max_attempts: int = 3
    
    # State maintained while processing the agent rag workflow
    attempts: int = field(default=0)
    # this should probably have it's own dictionary type; we can update this later
    search_history: List[dict] = field(default_factory=list)
    current_results: List[SearchResult] = field(default_factory=list)
    vetted_results: List[SearchResult] = field(default_factory=list)
    discarded_results: List[SearchResult] = field(default_factory=list)
    processed_ids: Set[str] = field(default_factory=set)
    thought_process: List[dict] = field(default_factory=list)
    reviews: List[str] = field(default_factory=list)      # Thought processes from reviews
    decisions: List[str] = field(default_factory=list)    # Store the actual decisions
    
    def should_continue(self) -> bool:
        return self.attempts < self.max_attempts and not self.has_sufficient_results()
    
    def has_sufficient_results(self) -> bool:
        return "finalize" in self.decisions
    
    def has_search_history(self) -> bool:
        return len(self.search_history) > 0
    
    def has_valid_results(self) -> bool:
        return len(self.vetted_results) > 0
    
    def add_search_attempt(self, query: str):
        self.attempts += 1
        self.search_history.append({
            "query": query
        })
    
    def to_result(self, final_answer: str) -> 'ConversationResult':
        """Convert conversation to final result"""
        return ConversationResult(
            final_answer=final_answer,
            citations=self.vetted_results,
            thought_process=self.thought_process,
            attempts=self.attempts,
            search_queries=[search["query"] for search in self.search_history]
        )
@dataclass
class CapabilitiesConversation:
    user_query: str
    max_attempts: int = 3
    
    # State maintained while processing the agent rag workflow
    attempts: int = field(default=0)
    # this should probably have it's own dictionary type; we can update this later
    search_history: List[dict] = field(default_factory=list)
    current_results: List[SearchResult] = field(default_factory=list)
    vetted_results: List[SearchResult] = field(default_factory=list)
    discarded_results: List[SearchResult] = field(default_factory=list)
    processed_ids: Set[str] = field(default_factory=set)
    thought_process: List[dict] = field(default_factory=list)
    reviews: List[str] = field(default_factory=list)      # Thought processes from reviews
    decisions: List[str] = field(default_factory=list)    # Store the actual decisions
    
    def should_continue(self) -> bool:
        return self.attempts < self.max_attempts and not self.has_sufficient_results()
    
    def has_sufficient_results(self) -> bool:
        return "finalize" in self.decisions
    
    def has_search_history(self) -> bool:
        return len(self.search_history) > 0
    
    def has_valid_results(self) -> bool:
        return len(self.vetted_results) > 0
    
    def add_search_attempt(self, query: str):
        self.attempts += 1
        self.search_history.append({
            "query": query
        })
    
    def to_result(self, final_answer: str) -> 'ConversationResult':
        """Convert conversation to final result"""
        return ConversationResult(
            final_answer=final_answer,
            citations=self.vetted_results,
            thought_process=self.thought_process,
            attempts=self.attempts,
            search_queries=[search["query"] for search in self.search_history]
        )
    
@dataclass
class ConversationResult:
    final_answer: str
    citations: List[SearchResult]
    thought_process: List[dict]
    attempts: int
    search_queries: List[str]


NUM_SEARCH_RESULTS = 5

# Create a type for indices from 0 to NUM_SEARCH_RESULTS-1
SearchResultIndex = Literal[tuple(range(NUM_SEARCH_RESULTS))]

class ReviewDecision(BaseModel):
    """Schema for review agent decisions"""
    thought_process: str
    valid_results: List[SearchResultIndex]  # Indices of valid results
    invalid_results: List[SearchResultIndex]  # Indices of invalid results
    decision: Literal["retry", "finalize"]