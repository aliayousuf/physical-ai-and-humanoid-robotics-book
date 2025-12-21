# Quickstart: Qdrant Cloud Integration

## Overview
This guide provides a quick setup for integrating Qdrant Cloud into your RAG-based ChatKit chatbot system. The integration will replace the current vector storage with Qdrant Cloud while maintaining compatibility with existing frontend and ChatKit setup.

## Prerequisites
- Python 3.11+
- Existing Gemini embedding model configuration
- Qdrant Cloud account and API key
- Book content in the `docs/` folder

## Installation
```bash
pip install qdrant-client google-generativeai fastapi
```

## Configuration
1. Set up environment variables:
```bash
export QDRANT_URL="your-qdrant-cloud-url"
export QDRANT_API_KEY="your-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

2. Configure the Qdrant client in your application:
```python
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=True  # Optional: for better performance
)
```

## Core Implementation Steps

### 1. Document Ingestion
```python
from services.ingestion_service import IngestionService

# Initialize the ingestion service
ingestion_service = IngestionService(qdrant_client=client, gemini_model="embedding-001")

# Process all documents in the docs/ folder
ingestion_service.ingest_documents_from_folder("docs/")
```

### 2. Semantic Search
```python
from services.vector_db_service import VectorDBService

# Initialize the vector database service
vector_service = VectorDBService(qdrant_client=client)

# Perform semantic search
search_results = vector_service.search_similar_chunks(query_text="your query", top_k=5)
```

### 3. Chat Integration
```python
from services.chat_service import ChatService

# Initialize the chat service with vector retrieval
chat_service = ChatService(
    vector_service=vector_service,
    gemini_client=gemini_client
)

# Get response based on retrieved context
response = chat_service.get_rag_response(user_query="What does the book say about AI?", top_k=3)
```

## Key Components

### Ingestion Service (`ingestion_service.py`)
- Reads documents from `docs/` folder
- Splits documents into semantic chunks
- Generates embeddings using Gemini model
- Stores chunks with embeddings in Qdrant Cloud

### Vector DB Service (`vector_db_service.py`)
- Manages interactions with Qdrant Cloud
- Performs semantic search operations
- Handles error cases and fallbacks
- Maintains connection pooling for efficiency

### Chat Service (`chat_service.py`)
- Integrates vector retrieval with chatbot logic
- Ensures responses are grounded in book content
- Returns "Not found in the book" when no relevant content exists
- Maintains compatibility with existing ChatKit setup

## Environment Variables
- `QDRANT_URL`: Your Qdrant Cloud instance URL
- `QDRANT_API_KEY`: Your Qdrant Cloud API key
- `GEMINI_API_KEY`: Your Google Gemini API key
- `DOCS_FOLDER`: Path to book content (defaults to "docs/")

## Error Handling
The system includes robust error handling:
- Connection retries with exponential backoff
- Graceful degradation when Qdrant is unavailable
- Fallback responses ("Not found in the book") when retrieval fails
- Comprehensive logging for debugging

## Testing
Run the following to verify the integration:
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Contract tests
pytest tests/contract/
```

## Next Steps
1. Review the API contracts in the `contracts/` directory
2. Implement the data models as defined in `data-model.md`
3. Follow the tasks in `tasks.md` for complete implementation