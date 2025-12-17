# Quickstart Guide: RAG Chatbot for Physical AI and Humanoid Robotics Documentation

## Overview
This guide provides step-by-step instructions to set up and run the RAG chatbot backend service for the Physical AI and Humanoid Robotics book documentation.

## Prerequisites

### System Requirements
- Python 3.11 or higher
- Node.js 18+ (for Docusaurus frontend)
- Git
- Access to API keys for:
  - Google Gemini API
  - Cohere API
  - Qdrant Cloud
  - Neon Postgres

### External Services Setup
1. **Qdrant Cloud Free Tier**
   - Sign up at [qdrant.io](https://qdrant.io)
   - Create a new cloud instance
   - Note your URL and API key

2. **Neon Postgres Serverless**
   - Sign up at [neon.tech](https://neon.tech)
   - Create a new project
   - Note your connection string

3. **Google Gemini API**
   - Get API key from [Google AI Studio](https://aistudio.google.com)

4. **Cohere API**
   - Get API key from [Cohere](https://cohere.com)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd physical-ai-and-humanoid-robotics-book
```

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Or if using pyproject.toml:
   pip install poetry
   poetry install
   ```

### 3. Environment Configuration
Create a `.env` file in the `backend` directory with the following content:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Database Configuration
DATABASE_URL=your_neon_postgres_connection_string

# Vector Database Configuration
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=book_content_embeddings

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=info
SESSION_EXPIRATION_HOURS=24

# RAG Configuration
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.7
MAX_QUERY_LENGTH=2000
MAX_RESPONSE_TOKENS=1000
```

### 4. Database Setup
1. Run database migrations (if using Alembic):
   ```bash
   alembic upgrade head
   ```

2. Initialize vector database:
   ```bash
   python -m src.scripts.initialize_vector_db
   ```

### 5. Index Book Content
1. Process and index the book content from the `docs/` directory:
   ```bash
   python -m src.scripts.ingest_docs
   ```

2. This will:
   - Parse all markdown files in the `docs/` directory
   - Chunk the content appropriately
   - Generate embeddings using Cohere
   - Store embeddings in Qdrant
   - Store metadata in Neon Postgres

### 6. Run the Backend Service
```bash
# Using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the run script if available
python -m src.api.main
```

The backend will be available at `http://localhost:8000`.

### 7. Frontend Integration
1. From the project root, ensure Docusaurus is set up:
   ```bash
   cd ..
   npm install
   ```

2. The RAG chatbot component should already be integrated into the Docusaurus layout. To verify:
   - Check that the chatbot component is included in the Docusaurus layout
   - Verify the component makes API calls to your backend

3. Run Docusaurus in development mode:
   ```bash
   npm run start
   ```

## API Testing

### Test Health Endpoint
```bash
curl http://localhost:8000/health
```

### Test Chat Functionality
1. Create a new session:
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat/session \
     -H "Content-Type: application/json" \
     -d '{"initial_context": "Physical AI chat session"}'
   ```

2. Send a query:
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat/query \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "your-session-id",
       "query": "What are the key principles of physical AI?",
       "context": {
         "page_url": "/docs/introduction",
         "selected_mode": false
       }
     }'
   ```

## Development Workflow

### Adding New Features
1. Update the API contracts in `specs/1-rag-chatbot-docusaurus/contracts/`
2. Implement the backend functionality
3. Update tests
4. Test integration with frontend

### Working with Book Content
- New content in the `docs/` directory will need to be re-indexed
- Use the ingestion script to update the vector database with new content
- Monitor the content summary endpoint to verify indexing

### Testing
Run backend tests:
```bash
pytest tests/
```

Run specific test suites:
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/
```

## Troubleshooting

### Common Issues
1. **API Keys Not Working**: Verify all API keys are correctly set in the `.env` file
2. **Database Connection Errors**: Check the Neon Postgres connection string format
3. **Vector Database Errors**: Verify Qdrant URL and API key
4. **Rate Limiting**: Check if you're exceeding free tier limits

### Useful Commands
- Check service health: `curl http://localhost:8000/health`
- Get content summary: `curl http://localhost:8000/api/v1/content/summary`
- View active sessions: Check your Neon Postgres database directly

## Next Steps
1. Review the full API documentation
2. Set up monitoring and logging
3. Configure production deployment
4. Implement authentication (for future releases)