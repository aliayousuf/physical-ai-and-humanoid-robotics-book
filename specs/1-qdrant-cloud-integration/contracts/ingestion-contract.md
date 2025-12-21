# API Contract: Document Ingestion Service

## Overview
Contract for the document ingestion service that processes book content from the `docs/` folder and stores embeddings in Qdrant Cloud.

## Endpoints

### POST /api/ingest
**Description**: Process all documents in the docs/ folder and store their embeddings in Qdrant Cloud

**Request**:
```
POST /api/ingest
Content-Type: application/json
Authorization: Bearer {token}

{
  "docs_folder": "docs/",  // Optional: defaults to "docs/"
  "force_reprocess": false // Optional: if true, reprocesses all documents
}
```

**Response**:
- `200 OK`: Ingestion completed successfully
```
{
  "status": "completed",
  "processed_documents": 15,
  "processed_chunks": 120,
  "timestamp": "2025-12-20T10:30:00Z",
  "collection_name": "book_content"
}
```

- `400 Bad Request`: Invalid request parameters
```
{
  "error": "InvalidParameter",
  "message": "docs_folder path does not exist"
}
```

- `500 Internal Server Error`: Processing failed
```
{
  "error": "ProcessingFailed",
  "message": "Failed to connect to Qdrant Cloud"
}
```

**Authentication**: Required - API key or JWT token

**Authorization**: Admin or system user role required

---

### GET /api/ingest/status
**Description**: Get the status of the ingestion process

**Request**:
```
GET /api/ingest/status
Authorization: Bearer {token}
```

**Response**:
- `200 OK`: Status retrieved successfully
```
{
  "status": "processing|completed|failed",
  "progress": 0.75,  // Percentage complete
  "total_documents": 20,
  "processed_documents": 15,
  "total_chunks": 150,
  "stored_chunks": 112,
  "last_updated": "2025-12-20T10:25:00Z"
}
```

---

### POST /api/ingest/single
**Description**: Process a single document and store its embeddings in Qdrant Cloud

**Request**:
```
POST /api/ingest/single
Content-Type: application/json
Authorization: Bearer {token}

{
  "document_path": "docs/chapter1.md",
  "metadata": {
    "author": "Book Author",
    "title": "Chapter 1 Title"
  }
}
```

**Response**:
- `200 OK`: Document processed successfully
```
{
  "status": "completed",
  "document_path": "docs/chapter1.md",
  "processed_chunks": 8,
  "timestamp": "2025-12-20T10:30:00Z",
  "collection_name": "book_content"
}
```

- `400 Bad Request`: Document not found or invalid format
```
{
  "error": "DocumentError",
  "message": "Document not found at path docs/chapter1.md"
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| `INGESTION_NOT_STARTED` | Ingestion process has not started |
| `INGESTION_IN_PROGRESS` | Ingestion is currently running |
| `QDRANT_CONNECTION_FAILED` | Unable to connect to Qdrant Cloud |
| `EMBEDDING_GENERATION_FAILED` | Failed to generate embeddings |
| `DOCUMENT_PARSING_FAILED` | Unable to parse the document format |
| `INSUFFICIENT_CONTENT` | Document has no meaningful text content |
| `QDRANT_STORAGE_FAILED` | Failed to store in Qdrant Cloud |

## Security Considerations
- Only authorized users can trigger ingestion
- Document paths must be validated to prevent directory traversal
- API rate limits should be implemented to prevent abuse