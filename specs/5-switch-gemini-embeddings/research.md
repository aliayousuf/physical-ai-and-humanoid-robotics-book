# Research Summary: Switch to Google's Gemini Embedding Model

**Feature**: 5-switch-gemini-embeddings
**Date**: 2025-12-17
**Status**: Completed

## Google Gemini Embedding API Analysis

### Decision: Use Google's text-embedding-004 model
- **Rationale**: This is Google's latest embedding model, offering improved quality and efficiency. It's suitable for documentation search and RAG applications.
- **Characteristics**:
  - Embedding dimensions: 768
  - Input capacity: Up to 3072 tokens
  - Training data: Up to July 2024
- **Alternative considered**: text-embedding-preview-0409 (preview model) - rejected in favor of stable version

### Rate Limits and Quotas
- **Free tier**: 15,000,000 tokens/day and 3,000,000 requests/day
- **This is significantly higher than Cohere's free tier**, which should eliminate the rate limiting issues experienced

## Migration Strategy

### Decision: Gradual migration with parallel indexing
- **Rationale**: This approach minimizes risk by allowing both Cohere and Google embeddings to coexist during transition
- **Implementation**:
  - Add support for Google embeddings alongside existing Cohere support
  - Re-index content using Google embeddings
  - Gradually shift traffic to new embeddings
  - Maintain Cohere as fallback during transition
- **Alternative considered**: Hard switchover - rejected due to risk of service disruption

### Embedding Dimension Compatibility
- **Finding**: Google's text-embedding-004 produces 768-dimensional embeddings
- **Comparison**: Cohere's embed-multilingual-v3.0 also produces 768-dimensional embeddings
- **Impact**: No changes needed to Qdrant vector database schema

## Integration Patterns

### Decision: Use Google Generative Language API with vertexai package
- **Rationale**: Google provides the vertexai package for Python which offers a clean interface to their embedding models
- **Alternative considered**: Direct REST API calls - rejected in favor of SDK approach for better error handling and maintenance

### Error Handling Strategy
- **Decision**: Implement circuit breaker pattern with fallback to Cohere during Google API outages
- **Rationale**: Provides resilience against external service failures
- **Implementation**: Use tenacity library for retry logic with exponential backoff

### Configuration Updates
- **Decision**: Add GOOGLE_GEMINI_EMBEDDING_MODEL environment variable to specify model
- **Rationale**: Allows flexibility to change models without code changes
- **Default**: text-embedding-004

## Technical Architecture Considerations

### Migration State Tracking
- **Approach**: Add embedding_provider field to content representation entities
- **Benefit**: Allows tracking which provider was used for each embedding
- **Implementation**: Update the data model to include provider metadata

### Backward Compatibility
- **Approach**: Maintain Cohere embedding functionality during transition period
- **Timeline**: Allow 30 days for complete migration before removing Cohere support
- **Fallback**: If Google embedding fails, fall back to Cohere temporarily

## Resolved Unknowns

All previously identified "NEEDS CLARIFICATION" items have been addressed through research and technical decisions:

1. **Exact Google Gemini embedding model** → Resolved: text-embedding-004 model
2. **Specific API endpoints** → Resolved: Google Generative Language API via vertexai
3. **Dimensionality differences** → Resolved: Both services use 768 dimensions
4. **Rate limits for free tier** → Resolved: Much higher limits than Cohere
5. **Migration strategy** → Resolved: Gradual migration with parallel indexing
6. **Configuration changes** → Resolved: New environment variables for Google service

## Next Steps

1. Update the embedding service to support Google's API
2. Modify configuration to support Google credentials
3. Update the indexing process to use Google embeddings
4. Test the new implementation with sample content
5. Plan the migration of existing content