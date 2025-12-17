# Data Model: RAG Chatbot for Physical AI and Humanoid Robotics Documentation

## Overview
This document defines the data models for the RAG chatbot system, including entities for user sessions, book content, content representations, query history, and user selection context.

## Entity Models

### 1. User Session
**Purpose**: Represents an active chat session with a user, containing conversation history and metadata

```python
class UserSession:
    id: UUID
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    conversation_history: List[Message]
    current_mode: str  # "general" or "selected_text"
    selected_text_context: Optional[SelectedTextContext]
    metadata: Dict[str, Any]  # Additional session data
```

**Fields**:
- `id`: Unique identifier for the session
- `created_at`: Timestamp when session was created
- `updated_at`: Timestamp of last activity
- `expires_at`: Expiration time for session cleanup
- `conversation_history`: List of messages in the conversation
- `current_mode`: Current query mode (general or selected text)
- `selected_text_context`: Context when in selected text mode
- `metadata`: Additional session-specific data

### 2. Message
**Purpose**: Represents a single message in a conversation

```python
class Message:
    id: UUID
    session_id: UUID
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: List[ContentReference]  # Citations for assistant responses
    metadata: Dict[str, Any]
```

**Fields**:
- `id`: Unique identifier for the message
- `session_id`: Reference to the parent session
- `role`: Role of the message sender
- `content`: The actual message content
- `timestamp`: When the message was created
- `sources`: References to book content used in the response
- `metadata`: Additional message-specific data

### 3. Book Content
**Purpose**: Represents the documentation content from the book, including text segments and page references

```python
class BookContent:
    id: UUID
    title: str
    content: str
    page_reference: str  # Path to the Docusaurus page
    section: str  # Section within the page
    hash: str  # Hash of content for change detection
    created_at: datetime
    updated_at: datetime
    embedding_id: str  # Reference to vector in Qdrant
    metadata: Dict[str, Any]  # Additional content metadata
```

**Fields**:
- `id`: Unique identifier for the content segment
- `title`: Title of the content segment
- `content`: The actual text content
- `page_reference`: Reference to the Docusaurus page
- `section`: Section within the page (for context)
- `hash`: Hash of content for change detection
- `created_at`: When content was first indexed
- `updated_at`: When content was last updated
- `embedding_id`: Reference to vector in Qdrant
- `metadata`: Additional content-specific metadata

### 4. Content Representation (Vector Embedding)
**Purpose**: Mathematical representation of book content segments for semantic search operations

*Note: This is primarily stored in Qdrant vector database, with references in Postgres*

```python
class ContentRepresentation:
    embedding_id: str  # ID in Qdrant
    content_id: UUID  # Reference to BookContent
    vector: List[float]  # The actual embedding vector
    metadata: Dict[str, Any]  # Additional embedding metadata
```

**Fields**:
- `embedding_id`: Unique identifier in the vector database
- `content_id`: Reference to the original BookContent
- `vector`: The actual embedding vector (dimension depends on model)
- `metadata`: Additional metadata for the embedding

### 5. Query History
**Purpose**: Record of user queries and system responses for context maintenance and analytics

```python
class QueryHistory:
    id: UUID
    session_id: UUID
    query_text: str
    response_text: str
    query_mode: str  # "general" or "selected_text"
    selected_text: Optional[str]  # Text that was selected (if applicable)
    response_sources: List[str]  # IDs of content used in response
    response_time_ms: int
    timestamp: datetime
    is_successful: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]
```

**Fields**:
- `id`: Unique identifier for the query record
- `session_id`: Reference to the session
- `query_text`: Original user query
- `response_text`: System response
- `query_mode`: Mode in which query was processed
- `selected_text`: Text that was selected (for selected text mode)
- `response_sources`: References to content used in response
- `response_time_ms`: Time taken to process the query
- `timestamp`: When the query was processed
- `is_successful`: Whether the query was processed successfully
- `error_message`: Error message if query failed
- `metadata`: Additional query-specific data

### 6. User Selection Context
**Purpose**: Temporary data structure containing selected/highlighted text for contextual mode

```python
class UserSelectionContext:
    id: UUID
    session_id: UUID
    selected_text: str
    page_url: str
    section_context: str  # Surrounding text for context
    created_at: datetime
    expires_at: datetime
```

**Fields**:
- `id`: Unique identifier for the selection context
- `session_id`: Reference to the user session
- `selected_text`: The actual selected text
- `page_url`: URL where text was selected
- `section_context`: Surrounding text for additional context
- `created_at`: When selection was made
- `expires_at`: When selection context expires

## Relationships

### Session-Message Relationship
- One UserSession can have many Messages (1:N)
- Messages are linked to sessions via session_id foreign key
- Messages are automatically deleted when session is deleted (cascade)

### Content-Representation Relationship
- One BookContent maps to one ContentRepresentation (1:1)
- ContentRepresentation has reference to BookContent via content_id
- Both are updated together when content changes

### Session-QueryHistory Relationship
- One UserSession can have many QueryHistory records (1:N)
- QueryHistory records are linked to sessions via session_id
- Records are retained for analytics even after session deletion

## Validation Rules

### UserSession Validation
- `expires_at` must be in the future
- `current_mode` must be either "general" or "selected_text"
- `conversation_history` length must not exceed 50 messages (configurable)

### Message Validation
- `role` must be either "user" or "assistant"
- `content` length must be between 1 and 10000 characters
- `timestamp` must not be in the future

### BookContent Validation
- `content` must not be empty
- `page_reference` must be a valid path
- `hash` must be a valid SHA-256 hash
- `embedding_id` must exist in Qdrant

### QueryHistory Validation
- `query_mode` must be either "general" or "selected_text"
- `response_time_ms` must be positive
- `timestamp` must not be in the future

## State Transitions

### UserSession States
- Active: Session is created and can accept queries
- Expired: Session has exceeded expiration time, becomes read-only
- Cleared: Session data is removed after cleanup process

### Query Processing States
- Pending: Query is received and being processed
- Successful: Query processed successfully
- Failed: Query processing failed with error

## Indexing Strategy

### Postgres Indexes
- UserSession: Index on `expires_at` for cleanup, `updated_at` for active session queries
- Message: Index on `session_id` and `timestamp` for conversation retrieval
- BookContent: Index on `page_reference` and `hash` for content management
- QueryHistory: Index on `session_id` and `timestamp` for analytics

### Qdrant Indexes
- ContentRepresentation: Vector index on embedding vectors for similarity search
- Metadata index on content_id for reference lookups