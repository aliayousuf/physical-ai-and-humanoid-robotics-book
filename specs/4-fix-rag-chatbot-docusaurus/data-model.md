# Data Model: RAG Chatbot for Docusaurus Documentation

**Feature**: 4-fix-rag-chatbot-docusaurus
**Date**: 2025-12-17
**Status**: Draft

## Overview

This document defines the data models for the RAG chatbot system, including user sessions, book content, content representations, query history, and selection contexts. The models are designed to support both general book content queries and selected text mode operations.

## Entity Models

### User Session

**Description**: Represents an active chat session with a user, containing conversation history and metadata.

**Fields**:
- `session_id` (string, primary key, required): Unique identifier for the session
- `created_at` (timestamp, required): When the session was created
- `last_interaction` (timestamp, required): Timestamp of last user interaction
- `conversation_history` (array of Message objects, optional): Complete conversation history
- `metadata` (JSON object, optional): Additional session metadata

**Relationships**:
- One-to-many with Query History (via session_id)
- One-to-many with User Selection Context (via session_id)

**Validation Rules**:
- session_id must be a UUID string
- created_at must be a valid ISO 8601 timestamp
- last_interaction must be >= created_at
- conversation_history length must not exceed 100 messages

**State Transitions**:
- Active: When user is actively chatting
- Idle: When no interaction for more than 30 minutes
- Expired: When session exceeds 24 hours

### Book Content

**Description**: Represents the documentation content from the book, including text segments and page references.

**Fields**:
- `content_id` (string, primary key, required): Unique identifier for content chunk
- `title` (string, required): Title of the content section
- `section_ref` (string, required): Reference to the book section
- `content_text` (string, required): The actual content text (max 1000 characters per chunk)
- `embedding_vector` (array of floats, required): Vector representation for semantic search
- `page_reference` (string, optional): Specific page or URL reference
- `created_at` (timestamp, required): When content was indexed
- `updated_at` (timestamp, optional): When content was last updated

**Relationships**:
- One-to-many with Content Representation (via content_id)

**Validation Rules**:
- content_id must be unique
- content_text must not exceed 1000 characters per chunk
- embedding_vector must have consistent dimension (1536 for OpenAI ada-002)
- section_ref must follow book's hierarchical structure

### Content Representation (Vector)

**Description**: Mathematical representation of book content segments for semantic search operations.

**Fields**:
- `content_id` (string, foreign key to Book Content, required): Reference to original content
- `vector_data` (array of floats, required): The embedding vector data
- `metadata` (JSON object, optional): Additional metadata for search filtering
- `created_at` (timestamp, required): When vector was created

**Relationships**:
- Many-to-one with Book Content (via content_id)

**Validation Rules**:
- vector_data must match expected embedding dimensions
- content_id must reference an existing Book Content record
- metadata must be valid JSON

### Query History

**Description**: Record of user queries and system responses for context maintenance and analytics.

**Fields**:
- `query_id` (string, primary key, required): Unique identifier for the query
- `session_id` (string, foreign key to User Session, required): Associated session
- `query_text` (string, required): Original user query text
- `response_text` (string, required): System's response text
- `timestamp` (timestamp, required): When query was processed
- `source_references` (array of strings, optional): References to book sections used
- `query_mode` (string, required): Either "general" or "selected_text"
- `selected_text_context` (string, optional): Text that was selected (for selected_text mode)
- `response_tokens` (integer, optional): Number of tokens in response
- `processing_time_ms` (integer, optional): Time taken to process query

**Relationships**:
- Many-to-one with User Session (via session_id)

**Validation Rules**:
- query_id must be a UUID string
- session_id must reference an existing User Session
- query_text and response_text must not exceed 10,000 characters
- query_mode must be either "general" or "selected_text"
- timestamp must be a valid ISO 8601 timestamp

### User Selection Context

**Description**: Temporary data structure containing selected/highlighted text for contextual mode.

**Fields**:
- `selection_id` (string, primary key, required): Unique identifier for selection
- `session_id` (string, foreign key to User Session, required): Associated session
- `selected_text` (string, required): The actual selected/highlighted text
- `page_context` (string, required): Context around the selected text
- `created_at` (timestamp, required): When selection was made
- `expires_at` (timestamp, required): When selection context expires (30 minutes after creation)

**Relationships**:
- Many-to-one with User Session (via session_id)

**Validation Rules**:
- selection_id must be a UUID string
- session_id must reference an existing User Session
- selected_text must not exceed 2000 characters
- expires_at must be 30 minutes after created_at
- created_at must be a valid ISO 8601 timestamp

### System Status

**Description**: Information about the operational state of various services (vector DB, LLM, API, etc.).

**Fields**:
- `status_id` (string, primary key, required): Unique identifier for status record
- `service_name` (string, required): Name of the service being monitored
- `status` (string, required): Current status ("healthy", "degraded", "unavailable")
- `last_checked` (timestamp, required): When status was last checked
- `details` (JSON object, optional): Additional status details
- `error_message` (string, optional): Error message if service is not healthy

**Validation Rules**:
- status_id must be a UUID string
- service_name must be one of predefined services
- status must be one of "healthy", "degraded", "unavailable"
- last_checked must be a valid ISO 8601 timestamp

## Message Object Definition

### Message

**Description**: A single message in a conversation history.

**Fields**:
- `message_id` (string, required): Unique identifier for message
- `role` (string, required): Either "user" or "assistant"
- `content` (string, required): The message content
- `timestamp` (timestamp, required): When message was created
- `references` (array of strings, optional): Book section references in response

**Validation Rules**:
- role must be either "user" or "assistant"
- content must not exceed 5000 characters
- timestamp must be a valid ISO 8601 timestamp

## Indexes and Performance

### Required Indexes

1. **User Session**:
   - Index on `session_id` (primary)
   - Index on `last_interaction` for session cleanup

2. **Book Content**:
   - Index on `content_id` (primary)
   - Index on `section_ref` for content lookup

3. **Query History**:
   - Index on `session_id` for session queries
   - Index on `timestamp` for chronological access

4. **User Selection Context**:
   - Index on `session_id` for session queries
   - Index on `expires_at` for cleanup

### Vector Database Considerations

- Content Representation entities will be stored in Qdrant vector database
- Vector similarity search will be performed using cosine similarity
- Metadata filtering will be available for content categorization
- HNSW indexing for efficient similarity search

## Data Relationships

```
User Session (1) <---> (Many) Query History
User Session (1) <---> (Many) User Selection Context
Book Content (1) <---> (Many) Content Representation
Query History (Many) --> (1) User Session
User Selection Context (Many) --> (1) User Session
Content Representation (Many) --> (1) Book Content
```

## Data Validation Strategy

1. **Input Validation**: All incoming data validated using Pydantic models
2. **Database Constraints**: Primary keys, foreign keys, and field constraints
3. **Application Logic**: Business rule validation in service layer
4. **Data Integrity**: Transactional operations for consistency

## Security Considerations

- All sensitive data fields are encrypted at rest
- User session data is scoped to individual users
- Content indexing sanitizes input to prevent injection
- Query history does not store sensitive user information