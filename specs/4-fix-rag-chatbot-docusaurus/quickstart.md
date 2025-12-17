# Quickstart Guide: RAG Chatbot for Docusaurus Documentation

**Feature**: 4-fix-rag-chatbot-docusaurus
**Date**: 2025-12-17
**Status**: Draft

## Overview

This guide provides step-by-step instructions to set up, configure, and run the RAG (Retrieval-Augmented Generation) chatbot for the Physical AI and Humanoid Robotics book documentation. The system includes a FastAPI backend, Qdrant vector database, Neon Postgres for session management, and Docusaurus frontend integration.

## Prerequisites

- Python 3.9+ installed
- Node.js 16+ installed
- Access to OpenAI API (API key)
- Access to Qdrant Cloud (API key and URL)
- Access to Neon Postgres (connection string)
- Git installed for version control

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd physical-ai-and-humanoid-robotics-book
```

### 2. Set Up Backend Environment

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
DATABASE_URL=your_neon_postgres_connection_string
DEBUG=true
LOG_LEVEL=info
```

## Backend Setup

### 1. Install Dependencies

```bash
cd backend
pip install fastapi uvicorn langchain openai qdrant-client sqlalchemy asyncpg python-multipart pydantic
```

### 2. Initialize Database

```bash
cd backend
python -c "
from database import init_db
import asyncio
asyncio.run(init_db())
"
```

### 3. Index Book Content

```bash
cd backend
python -c "
from indexing import index_book_content
import asyncio
asyncio.run(index_book_content())
"
```

### 4. Start Backend Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup

### 1. Navigate to Docusaurus Directory

```bash
cd ..
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Configure Docusaurus

Update `docusaurus.config.ts` to include the chatbot component:

```typescript
// Add to plugins array in docusaurus.config.ts
plugins: [
  // ... existing plugins
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'chatbot',
      path: 'src/components/Chatbot',
      routeBasePath: 'chatbot',
    },
  ],
],
```

### 4. Build and Serve Documentation

```bash
npm run build
npm run serve
```

## Running the System

### 1. Start Backend (in one terminal)

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Frontend (in another terminal)

```bash
npm run start
```

### 3. Access the Documentation

- Documentation: `http://localhost:3000`
- API Documentation: `http://localhost:8000/docs`
- API Status: `http://localhost:8000/api/status`

## API Usage Examples

### Creating a Session

```bash
curl -X POST http://localhost:8000/api/chat/session
```

### Querying the Chatbot

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key principles of humanoid robotics?",
    "mode": "general"
  }'
```

### Querying with Selected Text Context

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain this concept further",
    "mode": "selected_text",
    "selectedText": "The key principles of humanoid robotics include..."
  }'
```

## Testing the System

### 1. Unit Tests

```bash
cd backend
python -m pytest tests/
```

### 2. API Tests

```bash
cd backend
python -c "
import asyncio
from tests.api_test import test_api_endpoints
asyncio.run(test_api_endpoints())
"
```

### 3. Integration Tests

```bash
cd backend
python -c "
import asyncio
from tests.integration_test import test_rag_integration
asyncio.run(test_rag_integration())
"
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Verify environment variables are set correctly
2. **Vector DB Connection**: Check QDRANT_URL and QDRANT_API_KEY
3. **Database Connection**: Verify DATABASE_URL format
4. **Content Indexing**: Ensure the `/docs` directory has content to index

### Debugging Commands

```bash
# Check backend status
curl http://localhost:8000/api/status

# Check if content is indexed
curl "http://localhost:8000/api/content/search?query=test"

# View backend logs
tail -f backend/logs/app.log
```

## Deployment

### 1. Production Build

```bash
# Backend
cd backend
pip install -r requirements.txt --no-cache-dir
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
npm run build
```

### 2. Docker Deployment (Optional)

```bash
# Build Docker images
docker build -t rag-chatbot-backend -f backend/Dockerfile .
docker build -t rag-chatbot-frontend -f frontend/Dockerfile .

# Run containers
docker-compose up -d
```

## Next Steps

1. Implement the data models and API contracts as defined in this specification
2. Add authentication and authorization mechanisms
3. Implement monitoring and logging
4. Add comprehensive error handling and fallback mechanisms
5. Optimize for performance and scalability