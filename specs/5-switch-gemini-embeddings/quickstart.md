# Quickstart Guide: Switch to Google's Gemini Embedding Model

**Feature**: 5-switch-gemini-embeddings
**Created**: 2025-12-17
**Status**: Draft

## Overview

This guide provides step-by-step instructions to switch the RAG chatbot system from using Cohere embeddings to Google's free Gemini embedding model. This resolves rate limit issues while maintaining the same functionality.

## Prerequisites

- Python 3.9+ installed
- Google Cloud account with access to Gemini API
- Existing backend infrastructure (Qdrant vector database, Neon Postgres)
- Access to the project repository

## Environment Setup

### 1. Get Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create an account or sign in
3. Create a new API key for the Gemini API
4. Note your API key for configuration

### 2. Update Environment Variables

Update your `.env` file with the Google Gemini API configuration:

```bash
# Update your .env file with Google's API key instead of Cohere
GEMINI_API_KEY="your_google_gemini_api_key_here"

# Keep other configurations the same
DATABASE_URL="postgresql://..."
QDRANT_URL="https://..."
QDRANT_API_KEY="your_qdrant_api_key"
QDRANT_COLLECTION_NAME="book_content"
```

## Project Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The requirements.txt should include the Google-specific dependencies:
- `langchain-google-genai` - for Google Gemini integration
- `google-generativeai` - Google's official SDK

### 2. Update Configuration

The system configuration needs to be updated to use Google's embedding service instead of Cohere:

1. Update the embedding service configuration in `src/config/settings.py`
2. Ensure the embedding model is set to use Google's text-embedding model
3. Verify that the embedding dimensions remain compatible (768 dimensions for both models)

### 3. Run Database Migrations

```bash
cd backend
python -m src.scripts.migrate_db
```

## Implementation Steps

### Step 1: Update Embedding Service

The embedding service needs to be modified to use Google's API instead of Cohere's:

1. In `src/services/embedding_service.py`, update the client initialization to use Google's client
2. Update the embedding generation method to call Google's API
3. Ensure error handling works with Google's API responses

### Step 2: Re-index Content

After switching to Google embeddings, you'll need to re-index your content:

```bash
# Run the ingestion script to index content with Google embeddings
cd backend
python -m src.scripts.ingest_docs ../docs
```

### Step 3: Test the Integration

1. Start the backend server:
```bash
cd backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

2. Test the API endpoints directly:
```bash
# Create a session
curl -X POST http://localhost:8000/api/v1/chat/session \
  -H "Content-Type: application/json" \
  -d '{"initial_context": "Physical AI and Humanoid Robotics"}'

# Test a query
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "query": "What are the key principles of humanoid robotics?",
    "mode": "general"
  }'
```

## API Usage Examples

### Creating a Session

```python
import requests

response = requests.post("http://localhost:8000/api/v1/chat/session", json={
    "initial_context": "Physical AI and Humanoid Robotics documentation"
})

session_data = response.json()
session_id = session_data["session_id"]
```

### Querying the Chatbot

```python
import requests

response = requests.post("http://localhost:8000/api/v1/chat/query", json={
    "session_id": session_id,
    "query": "What does the book say about humanoid locomotion?",
    "mode": "general"
})

result = response.json()
print(result["response"])
print("Sources:", result["sources"])
```

### Using Selected Text Mode

```python
import requests

response = requests.post("http://localhost:8000/api/v1/chat/query", json={
    "session_id": session_id,
    "query": "Explain this concept further",
    "mode": "selected_text",
    "selected_text": "The key principle of humanoid locomotion is dynamic balance control..."
})

result = response.json()
print(result["response"])
```

## Verification

### 1. Test Content Indexing

Verify that content is being indexed with Google embeddings:

```bash
# Check if indexing is working
curl -X GET "http://localhost:8000/api/v1/content/search?query=test&limit=5"
```

### 2. Test Chat Functionality

Test that the chatbot responds to queries:

1. Navigate to your Docusaurus site
2. Interact with the chatbot widget
3. Ask questions about book content
4. Verify responses are relevant and properly sourced

### 3. Check Embedding Quality

Verify that Google embeddings provide similar or better semantic search quality compared to Cohere:

1. Perform searches with various query types
2. Compare relevance of results to the query
3. Ensure context is maintained properly in conversations

## Troubleshooting

### Common Issues

1. **API Rate Limits**: While Google's free tier has higher limits than Cohere, ensure you're within quota
2. **Environment Variables**: Double-check that GEMINI_API_KEY is properly set
3. **Embedding Dimensions**: Verify that the embedding dimensions match between old and new models (768 for compatibility)

### Error Messages

- `429: Rate limit exceeded` - Check your Google API quota
- `401: Unauthorized` - Verify your GEMINI_API_KEY is correct
- `500: Embedding generation failed` - Check network connectivity and API endpoint

## Next Steps

1. Deploy the updated backend to your production environment
2. Monitor performance metrics after the switch
3. Collect user feedback on response quality
4. Optimize the system based on usage patterns