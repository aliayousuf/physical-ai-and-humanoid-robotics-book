# API Contract: Semantic Search Service

## Overview
Contract for the semantic search service that performs similarity search against Qdrant Cloud to retrieve relevant book content chunks.

## Endpoints

### POST /api/search
**Description**: Perform semantic search against stored book content using Qdrant Cloud

**Request**:
```
POST /api/search
Content-Type: application/json
Authorization: Bearer {token}

{
  "query": "What does the book say about artificial intelligence?",
  "top_k": 5,           // Optional: number of results to return (default: 3)
  "score_threshold": 0.5 // Optional: minimum similarity score (default: 0.3)
}
```

**Response**:
- `200 OK`: Search completed successfully
```
{
  "query": "What does the book say about artificial intelligence?",
  "results": [
    {
      "id": "chunk_12345",
      "content": "Artificial intelligence is a branch of computer science...",
      "source_document": "docs/chapter3.md",
      "chunk_id": 1,
      "position": 1500,
      "score": 0.87,
      "metadata": {
        "author": "Book Author",
        "title": "Chapter 3 Title"
      }
    },
    {
      "id": "chunk_12346",
      "content": "Machine learning is a subset of artificial intelligence...",
      "source_document": "docs/chapter5.md",
      "chunk_id": 2,
      "position": 800,
      "score": 0.76,
      "metadata": {
        "author": "Book Author",
        "title": "Chapter 5 Title"
      }
    }
  ],
  "search_time_ms": 45,
  "timestamp": "2025-12-20T10:30:00Z"
}
```

- `400 Bad Request`: Invalid search parameters
```
{
  "error": "InvalidParameter",
  "message": "Query cannot be empty"
}
```

- `500 Internal Server Error`: Search failed
```
{
  "error": "SearchFailed",
  "message": "Failed to connect to Qdrant Cloud"
}
```

**Authentication**: Required - API key or JWT token

**Authorization**: Any authenticated user can perform search

---

### GET /api/search/health
**Description**: Check the health of the search service and Qdrant Cloud connection

**Request**:
```
GET /api/search/health
```

**Response**:
- `200 OK`: Service is healthy
```
{
  "status": "healthy",
  "qdrant_connected": true,
  "collection_exists": true,
  "last_heartbeat": "2025-12-20T10:30:00Z",
  "collection_name": "book_content",
  "total_chunks": 120
}
```

- `503 Service Unavailable`: Service is unhealthy
```
{
  "status": "unhealthy",
  "qdrant_connected": false,
  "message": "Cannot connect to Qdrant Cloud"
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| `QDRANT_CONNECTION_FAILED` | Unable to connect to Qdrant Cloud |
| `SEARCH_FAILED` | Semantic search operation failed |
| `EMPTY_QUERY` | Query string is empty or invalid |
| `NO_RESULTS_FOUND` | No relevant content found (not an error, but a valid outcome) |
| `SCORE_BELOW_THRESHOLD` | All results have scores below the threshold |
| `INVALID_PARAMETERS` | Invalid search parameters provided |

## Performance Requirements
- Search response time: <200ms for 95% of requests
- Maximum concurrent searches: 100
- Minimum similarity score: Configurable threshold (default 0.3)

## Security Considerations
- Query content should be sanitized to prevent injection attacks
- API rate limits should be implemented to prevent abuse
- Search results should not expose sensitive metadata unnecessarily