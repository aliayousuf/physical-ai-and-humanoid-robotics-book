# API Contracts: RAG Chatbot Backend

## Overview
This document defines the API contracts for the RAG chatbot backend service, including endpoints for chat interactions, health checks, and management functions.

## Base Configuration
- **Base URL**: `/api/v1`
- **Content-Type**: `application/json`
- **Authentication**: Bearer token (for future authentication expansion)
- **Error Format**: Standard error response format

### Standard Error Response
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

### 1. Health Check
**Endpoint**: `GET /health`
**Description**: Check the health status of the backend service
**Authentication**: None required

#### Response
**Success (200 OK)**:
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

**Error (503 Service Unavailable)**:
```json
{
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "One or more services are unavailable",
    "details": "Database connection failed"
  }
}
```

### 2. Start New Chat Session
**Endpoint**: `POST /chat/session`
**Description**: Create a new chat session
**Authentication**: None required (for MVP)

#### Request Body
```json
{
  "initial_context": "Optional initial context for the session"
}
```

#### Response
**Success (201 Created)**:
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "created_at": "2025-12-17T10:30:00Z",
  "expires_at": "2025-12-18T10:30:00Z"
}
```

**Error (400 Bad Request)**:
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid request body",
    "details": "initial_context too long"
  }
}
```

### 3. General Book Content Query (RAG)
**Endpoint**: `POST /chat/query`
**Description**: Submit a query about general book content using RAG
**Authentication**: None required (for MVP)

#### Request Body
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

#### Response
**Success (200 OK)**:
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

**Error (400 Bad Request)**:
```json
{
  "error": {
    "code": "INVALID_SESSION",
    "message": "Session not found or expired",
    "details": "Session ID is invalid or has expired"
  }
}
```

**Error (500 Internal Server Error)**:
```json
{
  "error": {
    "code": "RAG_PROCESSING_ERROR",
    "message": "Error processing RAG query",
    "details": "Vector search or LLM generation failed"
  }
}
```

### 4. Selected Text Query
**Endpoint**: `POST /chat/selected-text-query`
**Description**: Submit a query about user-selected text
**Authentication**: None required (for MVP)

#### Request Body
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

#### Response
**Success (200 OK)**:
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

### 5. Get Conversation History
**Endpoint**: `GET /chat/session/{session_id}/history`
**Description**: Retrieve conversation history for a session
**Authentication**: None required (for MVP)

#### Path Parameters
- `session_id`: The session identifier

#### Query Parameters
- `limit` (optional): Number of messages to return (default: 20, max: 50)
- `offset` (optional): Number of messages to skip (default: 0)

#### Response
**Success (200 OK)**:
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

### 6. Clear Session Context
**Endpoint**: `DELETE /chat/session/{session_id}/context`
**Description**: Clear the selected text context and return to general mode
**Authentication**: None required (for MVP)

#### Path Parameters
- `session_id`: The session identifier

#### Response
**Success (200 OK)**:
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "message": "Session context cleared",
  "new_mode": "general"
}
```

### 7. Get Book Content Summary
**Endpoint**: `GET /content/summary`
**Description**: Get a summary of indexed book content
**Authentication**: None required

#### Response
**Success (200 OK)**:
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

## Rate Limiting
All endpoints are subject to rate limiting:
- Unauthenticated requests: 100 requests per hour per IP
- Future authenticated requests: 1000 requests per hour per user

## Error Codes
- `INVALID_REQUEST`: Request body validation failed
- `INVALID_SESSION`: Session ID is invalid or expired
- `RAG_PROCESSING_ERROR`: Error during RAG pipeline processing
- `VECTOR_SEARCH_ERROR`: Error during vector similarity search
- `LLM_GENERATION_ERROR`: Error during LLM response generation
- `SERVICE_UNAVAILABLE`: Backend service unavailable
- `RATE_LIMIT_EXCEEDED`: Request rate limit exceeded
- `CONTENT_NOT_FOUND`: No relevant content found for query
- `MALFORMED_QUERY`: Query is malformed or contains invalid content