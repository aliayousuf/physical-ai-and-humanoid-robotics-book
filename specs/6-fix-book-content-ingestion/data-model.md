# Data Model: Book Content Ingestion System

## Entities

### Document
**Description**: Represents a document from the docs folder that needs to be ingested
**Fields**:
- id: string (unique identifier)
- filename: string (original file name)
- filepath: string (path in docs folder)
- format: string (file format: markdown, pdf, txt)
- size: integer (file size in bytes)
- created_at: datetime (timestamp when file was discovered)
- updated_at: datetime (timestamp when file was last processed)
- status: string (processing status: pending, processing, completed, failed)

### DocumentChunk
**Description**: Represents a chunk of text from a document that has been processed into embeddings
**Fields**:
- id: string (unique identifier)
- document_id: string (foreign key to Document)
- chunk_index: integer (order of chunk in document)
- content: string (text content of the chunk)
- embedding: vector (vector embedding of the content)
- metadata: object (additional metadata like page number, section)
- created_at: datetime (timestamp when chunk was created)

### VectorEmbedding
**Description**: Represents the vector embedding stored in the vector database
**Fields**:
- id: string (unique identifier, usually matches DocumentChunk.id)
- vector: array (numerical vector representation from Gemini embeddings)
- collection_name: string (name of the collection in vector DB)
- metadata: object (document metadata for filtering)
- created_at: datetime (timestamp when embedding was stored)
- model_used: string (the model used to generate the embedding, e.g., "embedding-001" for Gemini)

### IngestionJob
**Description**: Represents a job for processing documents from the docs folder
**Fields**:
- id: string (unique identifier)
- status: string (status: queued, running, completed, failed)
- total_documents: integer (number of documents to process)
- processed_documents: integer (number of documents processed)
- started_at: datetime (timestamp when job started)
- completed_at: datetime (timestamp when job completed)
- error_message: string (error if job failed)

## Relationships
- Document has many DocumentChunks
- DocumentChunk has one VectorEmbedding
- IngestionJob processes many Documents

## Validation Rules
- Document.filename must not be empty
- Document.format must be one of ['markdown', 'pdf', 'txt']
- DocumentChunk.content must not exceed 2000 characters
- DocumentChunk.embedding must be a valid vector
- DocumentChunk.document_id must reference an existing Document

## State Transitions
- Document: pending → processing → completed/failed
- IngestionJob: queued → running → completed/failed