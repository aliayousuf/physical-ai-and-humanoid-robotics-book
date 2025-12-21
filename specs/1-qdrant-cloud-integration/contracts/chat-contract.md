# API Contract: RAG Chat Service

## Overview
Contract for the RAG-based chat service that retrieves relevant book content and generates responses strictly based on book content.

## Endpoints

### POST /api/chat
**Description**: Get a response from the chatbot based on book content using RAG

**Request**:
```
POST /api/chat
Content-Type: application/json
Authorization: Bearer {token}

{
  "message": "What does the book say about artificial intelligence?",
  "context_chunks": 3,      // Optional: number of context chunks to retrieve (default: 3)
  "temperature": 0.1,       // Optional: response randomness (default: 0.1 for factual responses)
  "session_id": "sess_123"  // Optional: for conversation context
}
```

**Response**:
- `200 OK`: Chat response generated successfully
```
{
  "request_id": "req_45678",
  "message": "According to the book, artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence.",
  "retrieved_chunks": 3,
  "sources": [
    {
      "document": "docs/chapter3.md",
      "chunk_id": 1,
      "score": 0.87
    }
  ],
  "timestamp": "2025-12-20T10:30:00Z",
  "session_id": "sess_123"
}
```

- `200 OK`: No relevant content found
```
{
  "request_id": "req_45679",
  "message": "Not found in the book.",
  "retrieved_chunks": 0,
  "sources": [],
  "timestamp": "2025-12-20T10:30:01Z",
  "session_id": "sess_123"
}
```

- `400 Bad Request`: Invalid request parameters
```
{
  "error": "InvalidParameter",
  "message": "Message cannot be empty"
}
```

- `500 Internal Server Error`: Processing failed
```
{
  "error": "ProcessingFailed",
  "message": "Failed to retrieve content from Qdrant Cloud"
}
```

**Authentication**: Required - API key or JWT token

**Authorization**: Any authenticated user can use chat service

---

### POST /api/chat/stream
**Description**: Get a streaming response from the chatbot based on book content using RAG

**Request**:
```
POST /api/chat/stream
Content-Type: application/json
Authorization: Bearer {token}

{
  "message": "Explain the concept of machine learning from the book",
  "context_chunks": 3,
  "session_id": "sess_123"
}
```

**Response**:
- `200 OK`: Streaming response with Content-Type: text/plain
```
data: {"type": "retrieval_start", "timestamp": "2025-12-20T10:30:00Z"}

data: {"type": "retrieval_complete", "retrieved_chunks": 2, "sources": ["docs/chapter4.md", "docs/chapter7.md"], "timestamp": "2025-12-20T10:30:01Z"}

data: {"type": "chunk", "content": "Machine learning is", "timestamp": "2025-12-20T10:30:02Z"}

data: {"type": "chunk", "content": " a subset of artificial intelligence", "timestamp": "2025-12-20T10:30:02Z"}

data: {"type": "chunk", "content": " that focuses on algorithms", "timestamp": "2025-12-20T10:30:03Z"}

data: {"type": "complete", "request_id": "req_45680", "timestamp": "2025-12-20T10:30:03Z"}
```

---

### GET /api/chat/health
**Description**: Check the health of the chat service and its dependencies

**Request**:
```
GET /api/chat/health
```

**Response**:
- `200 OK`: Service is healthy
```
{
  "status": "healthy",
  "qdrant_connected": true,
  "gemini_connected": true,
  "last_heartbeat": "2025-12-20T10:30:00Z",
  "active_sessions": 5
}
```

- `503 Service Unavailable`: Service is unhealthy
```
{
  "status": "unhealthy",
  "qdrant_connected": false,
  "gemini_connected": true,
  "message": "Cannot connect to Qdrant Cloud"
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| `QDRANT_CONNECTION_FAILED` | Unable to connect to Qdrant Cloud for retrieval |
| `GEMINI_CONNECTION_FAILED` | Unable to connect to Gemini API for response generation |
| `NO_RELEVANT_CONTENT` | No relevant content found in book (results in "Not found in the book" response) |
| `RESPONSE_GENERATION_FAILED` | Failed to generate response based on retrieved content |
| `INVALID_MESSAGE` | User message is empty or invalid |
| `SESSION_EXPIRED` | Session ID is no longer valid |

## RAG Behavior Rules
- All responses must be grounded only in retrieved book data
- No hallucinations or outside knowledge allowed
- If no relevant content is found, respond with "Not found in the book."
- Responses should be clear and concise

## Performance Requirements
- Response time: <3 seconds for 95% of requests
- Maximum context length: 4096 tokens to prevent overflow
- Support for streaming responses to improve user experience

## Security Considerations
- Message content should be sanitized to prevent prompt injection
- Session IDs should be securely generated and managed
- Rate limits should be implemented to prevent abuse
- Response content should not expose sensitive document metadata