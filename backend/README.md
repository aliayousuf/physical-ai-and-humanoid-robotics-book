# RAG Chatbot Backend

This is the backend service for the RAG (Retrieval-Augmented Generation) chatbot integrated with the Physical AI and Humanoid Robotics documentation.

## Features

- RAG-powered Q&A for book content
- Contextual queries based on selected text
- Session management
- Vector search for relevant content retrieval
- Rate limiting and input sanitization

## Tech Stack

- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Vector Database**: Qdrant Cloud
- **Embeddings**: Cohere
- **LLM**: Google Gemini 2.5 Flash
- **Database**: Neon Postgres

## Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or if using pyproject.toml:
   pip install poetry
   poetry install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Initialize the vector database**:
   ```bash
   python -m src.scripts.initialize_vector_db
   ```

5. **Ingest book content** (optional, for testing):
   ```bash
   python -m src.scripts.ingest_docs
   ```

## Running the Service

```bash
# Using uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or directly
python -m src.api.main
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/chat/session` - Create new chat session
- `POST /api/v1/chat/query` - General RAG query
- `POST /api/v1/chat/selected-text-query` - Query with selected text context
- `GET /api/v1/chat/session/{session_id}/history` - Get conversation history
- `DELETE /api/v1/chat/session/{session_id}/context` - Clear selected text context

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key
- `COHERE_API_KEY` - Cohere API key
- `DATABASE_URL` - Neon Postgres connection string
- `QDRANT_URL` - Qdrant Cloud URL
- `QDRANT_API_KEY` - Qdrant Cloud API key
- `ENVIRONMENT` - Environment (default: development)
- `LOG_LEVEL` - Logging level (default: info)
- `SESSION_EXPIRATION_HOURS` - Session expiration in hours (default: 24)
- `RAG_TOP_K` - Number of results to retrieve (default: 5)
- `RAG_SCORE_THRESHOLD` - Minimum similarity score (default: 0.7)
- `MAX_QUERY_LENGTH` - Maximum query length (default: 2000)
- `MAX_RESPONSE_TOKENS` - Maximum response tokens (default: 1000)

## Testing

Run the tests:

```bash
# Unit tests
pytest tests/unit/

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