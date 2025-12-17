# Data Model: Switch to Google's Gemini Embedding Model

**Feature**: 5-switch-gemini-embeddings
**Date**: 2025-12-17
**Status**: Draft

## Overview

This document defines the data models for the migration from Cohere to Google's Gemini embedding model. The models are designed to support both embedding providers during the transition period while maintaining compatibility with the existing architecture.

## Entity Models

### Embedding Service Configuration

**Description**: Configuration for the embedding service that supports both Cohere and Google providers.

**Fields**:
- `service_type` (string, required): Either "cohere" or "google" to indicate the embedding provider
- `model_name` (string, required): Name of the specific embedding model being used
- `api_key` (string, required): API key for the embedding service
- `api_base_url` (string, optional): Base URL for the API (for Google, this may include project/location)
- `embedding_dimensions` (integer, required): Number of dimensions in the embeddings (should be 768 for both providers)
- `rate_limit_requests_per_minute` (integer, optional): Rate limit for the service
- `created_at` (timestamp, required): When this configuration was created
- `updated_at` (timestamp, required): When this configuration was last updated

**Validation Rules**:
- service_type must be either "cohere" or "google"
- model_name must be a valid model identifier for the service_type
- embedding_dimensions must be 768 for compatibility with existing Qdrant setup
- api_key must not be empty

### Content Representation (Vector)

**Description**: Mathematical representation of book content segments for semantic search operations, now supporting multiple embedding providers.

**Fields**:
- `content_id` (string, primary key, required): Reference to original content in Book Content entity
- `vector_data` (array of floats, required): The embedding vector data (768-dimensional for both providers)
- `embedding_provider` (string, required): Indicates which service was used to generate the embedding ("cohere" or "google")
- `embedding_model` (string, required): Specific model used to generate this embedding
- `metadata` (JSON object, optional): Additional metadata for search filtering and migration tracking
- `created_at` (timestamp, required): When vector was created
- `last_accessed_at` (timestamp, optional): When vector was last used in a search

**Relationships**:
- Many-to-one with Book Content (via content_id)

**Validation Rules**:
- vector_data must have exactly 768 elements (for compatibility)
- embedding_provider must be either "cohere" or "google"
- embedding_model must be a valid model for the provider
- content_id must reference an existing Book Content record
- metadata must be valid JSON

### Migration State Tracker

**Description**: Tracks the migration progress from Cohere to Google embeddings.

**Fields**:
- `migration_id` (string, primary key, required): Unique identifier for this migration
- `source_provider` (string, required): Source embedding provider ("cohere")
- `target_provider` (string, required): Target embedding provider ("google")
- `status` (string, required): Current migration status ("not_started", "in_progress", "completed", "failed")
- `total_documents` (integer, required): Total number of documents to migrate
- `processed_documents` (integer, required): Number of documents processed
- `failed_documents` (integer, optional): Number of documents that failed to process
- `start_time` (timestamp, optional): When migration started
- `end_time` (timestamp, optional): When migration completed
- `error_details` (JSON object, optional): Details about any errors encountered

**Validation Rules**:
- source_provider must be "cohere"
- target_provider must be "google"
- status must be one of the allowed values
- processed_documents must not exceed total_documents
- failed_documents must not exceed total_documents

### Embedding Performance Metrics

**Description**: Metrics to compare performance between Cohere and Google embeddings during migration.

**Fields**:
- `metric_id` (string, primary key, required): Unique identifier for this metric
- `embedding_provider` (string, required): Provider being measured ("cohere" or "google")
- `operation_type` (string, required): Type of operation ("embedding_generation", "semantic_search", "overall")
- `average_response_time_ms` (float, required): Average response time in milliseconds
- `success_rate` (float, required): Success rate as decimal (0.0 to 1.0)
- `timestamp` (timestamp, required): When these metrics were recorded
- `sample_size` (integer, required): Number of samples used to calculate metrics

**Validation Rules**:
- embedding_provider must be either "cohere" or "google"
- operation_type must be one of the allowed values
- average_response_time_ms must be positive
- success_rate must be between 0.0 and 1.0
- sample_size must be positive

## Indexes and Performance

### Required Indexes

1. **Content Representation**:
   - Index on `content_id` (primary)
   - Index on `embedding_provider` for filtering by provider
   - Index on `created_at` for chronological access

2. **Migration State Tracker**:
   - Index on `migration_id` (primary)
   - Index on `status` for status-based queries
   - Index on `start_time` for time-based queries

### Vector Database Considerations

- Content Representation entities will be stored in Qdrant vector database
- Vector similarity search will be performed using cosine similarity
- Metadata filtering will be available for content categorization
- HNSW indexing for efficient similarity search
- The embedding dimension remains consistent (768) between providers

## Data Relationships

```
Book Content (1) <---> (Many) Content Representation
Migration State Tracker (1) <---> (Many) Content Representation (during migration)
Embedding Performance Metrics (Many) <---> (Many) Content Representation (for comparison)
```

## Migration Data Flow

During the transition from Cohere to Google embeddings:

1. New content will be indexed using Google embeddings
2. Existing Cohere embeddings will be retained initially
3. A background process will migrate Cohere embeddings to Google embeddings
4. The system will gradually shift to using Google embeddings primarily
5. Cohere embeddings will eventually be phased out after successful validation

## Security Considerations

- API keys for both services will be stored securely in environment variables
- Embedding provider metadata will be logged for audit purposes
- Migration processes will be secured with appropriate authentication
- Performance metrics will not contain sensitive information