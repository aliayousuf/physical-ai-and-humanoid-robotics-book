# RAG Chatbot API Documentation

## Overview
The RAG Chatbot API provides endpoints for interacting with the Retrieval-Augmented Generation chatbot for the Physical AI and Humanoid Robotics book documentation.

## Base URL
```
https://your-backend-domain.com/api/v1
```

## Authentication
Currently, no authentication is required for the MVP. In future releases, Bearer token authentication will be implemented.

## Common Headers
- `Content-Type: application/json`
- `Accept: application/json`

## Error Format
All error responses follow this format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details if applicable"
  }
}
```

## Endpoints

### Health Check
#### GET /health
Check the health status of the backend service.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-17T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "vector_db": "connected",
    "gemini_api": "reachable"
  }
}
```

### Session Management
#### POST /chat/session
Create a new chat session.

**Request Body:**
```json
{
  "initial_context": "Optional initial context for the session"
}
```

**Response (201 Created):**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "created_at": "2025-12-17T10:30:00Z",
  "expires_at": "2025-12-18T10:30:00Z"
}
```

### General RAG Queries
#### POST /chat/query
Submit a query about general book content using RAG.

**Request Body:**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "query": "What are the key principles of physical AI?",
  "context": {
    "page_url": "/docs/introduction/physical-ai",
    "selected_mode": false
  }
}
```

**Response (200 OK):**
```json
{
  "response_id": "b2c3d4e5-f6g7-8901-2345-67890abcdef1",
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "query": "What are the key principles of physical AI?",
  "response": "The key principles of physical AI include...",
  "sources": [
    {
      "content_id": "c3d4e5f6-g7h8-9012-3456-7890abcdef12",
      "title": "Introduction to Physical AI",
      "page_reference": "/docs/introduction/physical-ai",
      "relevance_score": 0.92
    }
  ],
  "timestamp": "2025-12-17T10:30:05Z",
  "query_mode": "general"
}
```

### Selected Text Queries
#### POST /chat/selected-text-query
Submit a query about user-selected text.

**Request Body:**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "query": "Can you explain this concept in simpler terms?",
  "selected_text": "The complex mathematical framework of...",
  "context": {
    "page_url": "/docs/advanced-topics/mathematical-framework",
    "section_context": "In the context of humanoid robotics, the mathematical framework..."
  }
}
```

**Response (200 OK):**
```json
{
  "response_id": "c3d4e5f6-g7h8-9012-3456-7890abcdef12",
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "query": "Can you explain this concept in simpler terms?",
  "response": "In simpler terms, this concept means...",
  "sources": [
    {
      "content_id": "d4e5f6g7-h8i9-0123-4567-890abcdef123",
      "title": "Mathematical Framework",
      "page_reference": "/docs/advanced-topics/mathematical-framework",
      "relevance_score": 0.87
    }
  ],
  "timestamp": "2025-12-17T10:31:12Z",
  "query_mode": "selected_text"
}
```

### Conversation History
#### GET /chat/session/{session_id}/history
Retrieve conversation history for a session.

**Query Parameters:**
- `limit` (optional): Number of messages to return (default: 20, max: 50)
- `offset` (optional): Number of messages to skip (default: 0)

**Response (200 OK):**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "messages": [
    {
      "id": "e5f6g7h8-i9j0-1234-5678-90abcdef1234",
      "role": "user",
      "content": "What are the key principles of physical AI?",
      "timestamp": "2025-12-17T10:30:00Z",
      "sources": []
    },
    {
      "id": "f6g7h8i9-j0k1-2345-6789-0abcdef12345",
      "role": "assistant",
      "content": "The key principles of physical AI include...",
      "timestamp": "2025-12-17T10:30:05Z",
      "sources": [
        {
          "content_id": "c3d4e5f6-g7h8-9012-3456-7890abcdef12",
          "title": "Introduction to Physical AI",
          "page_reference": "/docs/introduction/physical-ai",
          "relevance_score": 0.92
        }
      ]
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 2
  }
}
```

### Context Management
#### DELETE /chat/session/{session_id}/context
Clear the selected text context and return to general mode.

**Response (200 OK):**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "message": "Session context cleared",
  "new_mode": "general"
}
```

### Content Summary
#### GET /content/summary
Get a summary of indexed book content.

**Response (200 OK):**
```json
{
  "total_pages": 150,
  "total_content_segments": 1250,
  "total_tokens": 450000,
  "last_indexed": "2025-12-17T09:00:00Z",
  "content_coverage": {
    "introduction": 100,
    "physical_ai": 85,
    "humanoid_robotics": 92,
    "advanced_topics": 78
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_REQUEST | 400 | Request body validation failed |
| INVALID_SESSION | 400 | Session not found or expired |
| RAG_PROCESSING_ERROR | 500 | Error during RAG pipeline processing |
| VECTOR_SEARCH_ERROR | 500 | Error during vector similarity search |
| LLM_GENERATION_ERROR | 500 | Error during LLM response generation |
| SERVICE_UNAVAILABLE | 503 | Backend service unavailable |
| RATE_LIMIT_EXCEEDED | 429 | Request rate limit exceeded |
| CONTENT_NOT_FOUND | 200 | No relevant content found for query (not an error, returns informative response) |
| MALFORMED_QUERY | 400 | Query is malformed or contains invalid content |

## Rate Limiting
All endpoints are subject to rate limiting:
- Unauthenticated requests: 100 requests per hour per IP
- Future authenticated requests: 1000 requests per hour per user

## Usage Monitoring
The API monitors usage of external services (Cohere, Gemini, Qdrant) to stay within free tier limits. When limits are approached (80% usage), warnings are logged. When limits are exceeded (100% usage), API calls will fail with appropriate error messages.