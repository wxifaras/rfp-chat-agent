**Primary Purpose:** 

An intelligent RFP (Request for Proposal) processing and chat system that combines document ingestion with conversational AI.

**Two Main Functions:**

1. **RFP Processing (`process_rfp`):**
   - Ingests uploaded RFP documents
   - Extracts text using Azure Document Intelligence
   - Uses LLM to extract structured metadata (decision logs)
   - Indexes content in Azure AI Search for retrieval
   - Stores everything in Azure Blob Storage

2. **Conversational RFP Chat with Agentic RAG (`chat_with_rfp`):**
   - Allows users to ask questions about uploaded RFPs
   - Uses agentic RAG (Retrieval-Augmented Generation) workflow
   - Iteratively searches indexed RFP content
   - LLM reviews search results for relevance
   - Generates contextual answers with citations
   - Maintains chat history for conversational context

**Core Value:** Transforms static RFP documents into an intelligent, searchable knowledge base that users can query conversationally, getting accurate answers with proper source citations.

**Architecture:** Cloud-native using Azure services (OpenAI, Search, Storage, Document Intelligence, Cosmos DB) with FastAPI backend.
