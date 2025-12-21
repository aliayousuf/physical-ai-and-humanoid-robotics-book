# Physical AI and Humanoid Robotics Book - Backend

This is the backend service for the Physical AI and Humanoid Robotics book project. It provides a RAG (Retrieval Augmented Generation) chatbot that can answer questions about the book content using Google's Gemini model.

## Features

- RAG-powered Q&A for book content
- Contextual queries based on selected text
- Session management
- Vector search for relevant content retrieval
- Rate limiting and input sanitization
- Document ingestion from various formats (PDF, Markdown, Text)
- Job tracking for ingestion processes

## Tech Stack

- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Vector Database**: Qdrant Cloud
- **Embeddings**: Google Generative AI (text-embedding-004)
- **LLM**: Google Gemini 2.5 Flash
- **Database**: Neon Postgres

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Copy the example environment file and update the values:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `QDRANT_URL`: Your Qdrant vector database URL
   - `QDRANT_API_KEY`: Your Qdrant API key (if using cloud)
   - `DOCS_PATH`: Path to the book content folder (default: "./docs")

3. **Run the Application**:
   ```bash
   uvicorn main:app --reload
   ```

## Usage Instructions

### Ingest Book Content
To ingest the book content from the docs folder into the vector database:

```bash
curl -X POST http://localhost:8000/api/v1/ingestion/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "force_reprocess": false,
    "file_patterns": ["*.md", "*.pdf", "*.txt"]
  }'
```

To check the status of an ingestion job:
```bash
curl -X GET http://localhost:8000/api/v1/ingestion/status/{job_id}
```

To get a list of processed documents:
```bash
curl -X GET http://localhost:8000/api/v1/ingestion/documents
```

### Query the Chatbot
Once the content is ingested, you can query the chatbot:

```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the book say about humanoid robotics?",
    "max_results": 5,
    "similarity_threshold": 0.7
  }'
```

Or use the simple query endpoint:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/query_simple?query=What%20is%20AI?"
```

### Stream Chat Responses
For real-time chat responses with progress updates:

```bash
curl -X POST "http://localhost:8000/api/v1/chat/stream?query=What%20is%20AI?&max_results=5&similarity_threshold=0.3" \
  -H "Accept: text/plain"
```

### Semantic Search
Perform direct semantic search on the book content:

```bash
curl -X POST "http://localhost:8000/api/v1/search?query=What%20is%20AI?&top_k=5&score_threshold=0.3" \
  -H "Content-Type: application/json"
```

### Health Check
Check if the service is running:
```bash
curl -X GET http://localhost:8000/api/v1/health
```

Check if the service is ready to handle requests:
```bash
curl -X GET http://localhost:8000/api/v1/ready
```

## API Endpoints

- `POST /api/v1/ingestion/trigger` - Trigger content ingestion from docs folder
- `GET /api/v1/ingestion/status/{job_id}` - Get ingestion job status
- `GET /api/v1/ingestion/documents` - Get list of processed documents
- `POST /api/v1/chat/query` - Query the chatbot with book content
- `POST /api/v1/chat/query_simple` - Simple query endpoint for the chatbot
- `POST /api/v1/chat/stream` - Stream chat responses with progress updates
- `POST /api/v1/search` - Perform semantic search on book content
- `GET /api/v1/search/health` - Health check for search service
- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/ready` - Readiness check endpoint

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key
- `GEMINI_EMBEDDING_MODEL_NAME` - Google embedding model name (default: text-embedding-004)
- `GEMINI_MODEL_NAME` - Google Gemini model name (default: gemini-2.5-flash)
- `QDRANT_URL` - Qdrant Cloud URL
- `QDRANT_API_KEY` - Qdrant Cloud API key
- `DOCS_PATH` - Path to the book content folder (default: "./docs")
- `ENVIRONMENT` - Environment (default: development)
- `LOG_LEVEL` - Logging level (default: info)
- `RAG_TOP_K` - Number of results to retrieve (default: 5)
- `RAG_SCORE_THRESHOLD` - Minimum similarity score (default: 0.3)
- `MAX_QUERY_LENGTH` - Maximum query length (default: 2000)
- `MAX_RESPONSE_TOKENS` - Maximum response tokens (default: 1000)
- `CHUNK_SIZE` - Size of text chunks (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)

## Testing

Run the tests:

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests
pytest
```

## Architecture

The backend follows a service-oriented architecture:

- `models/` - Pydantic models for data validation
- `services/` - Business logic and external service integration
- `api/` - FastAPI endpoints
- `config/` - Configuration and settings
- `scripts/` - Utility scripts for data ingestion
- `middleware/` - Request processing middleware
- `utils/` - Utility functions
- `tests/` - Test files