# Implementation Plan: Switch to Google's Gemini Embedding Model

**Feature**: 5-switch-gemini-embeddings
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude

## Technical Context

The current RAG chatbot implementation uses Cohere for generating vector embeddings, but Cohere's rate limits are preventing complete documentation indexing. This is affecting the chatbot's ability to respond to questions about the book content. The solution is to switch to Google's free Gemini embedding model which should provide similar functionality without the rate limitations.

### Current Architecture Understanding

The system currently consists of:
- **Embedding Service**: Uses Cohere API to convert text to vector embeddings
- **Content Indexing**: Processes documentation content and stores embeddings in Qdrant vector database
- **RAG Service**: Performs semantic search using the stored embeddings
- **API Layer**: FastAPI endpoints for chatbot functionality

### Known Unknowns (NEEDS CLARIFICATION)
- Exact Google Gemini embedding model to use (e.g., embedding-001)
- Specific API endpoints for Google's embedding service
- Dimensionality of Google's embeddings vs. Cohere's embeddings
- Rate limits and quotas for Google's free tier
- Migration strategy for existing Cohere-based embeddings in Qdrant
- Configuration changes needed in the existing embedding service

### Dependencies
- Google Gemini API access and credentials
- Qdrant Cloud Free Tier account (currently in use)
- FastAPI backend service
- Existing documentation content in docs/ folder

## Constitution Check

### Alignment with Core Principles

**Decision Point Mapping**: This change addresses the critical decision of which embedding service to use, weighing factors like rate limits, cost, and performance.

**Reasoning Activation**: The implementation will require reasoning about service migration, backward compatibility, and performance comparison rather than just applying generic migration patterns.

**Intelligence Accumulation**: This fix will produce reusable migration and embedding service intelligence for future RAG system implementations.

**Right Altitude**: The plan maintains appropriate decision frameworks with concrete migration steps, testing procedures, and rollback options without being overly prescriptive or vague.

**Frameworks Over Rules**: The approach uses conditional reasoning for different migration scenarios rather than rigid step-by-step instructions.

**Meta-Awareness Against Convergence**: The plan includes multiple verification and testing approaches to avoid predictable debugging patterns.

## Planning Gates

### Gate 1: Architecture Feasibility ✅
- Google Gemini embedding model is available and compatible with current architecture
- The architecture aligns with the project constitution requirements
- All necessary service accounts and access can be provisioned

### Gate 2: Technical Requirements Compliance ✅
- Solution will continue to use Qdrant vector database for storage
- Will maintain semantic search functionality with new embeddings
- Will preserve existing API contracts and interfaces
- Will include proper error handling and security measures

### Gate 3: Resource Availability ✅
- Google Gemini API access can be provisioned
- Qdrant vector database is already available
- Existing backend infrastructure supports the change
- Documentation content is available for re-indexing if needed

### Gate 4: Risk Assessment
- **Low Risk**: Switching embedding providers (can be done gradually with fallbacks)
- **Mitigation**: Implement gradual migration with Cohere fallback option
- **Low Risk**: Potential differences in embedding quality between providers
- **Mitigation**: Conduct thorough testing and validation of search quality

## Phase 0: Research & Discovery

### Research Tasks

1. **Google Gemini Embedding API Analysis**
   - Investigate available embedding models and their specifications
   - Compare embedding dimensions and characteristics with Cohere models
   - Document rate limits and pricing for free tier

2. **Migration Strategy Research**
   - Research best practices for switching embedding providers
   - Identify options for handling existing Cohere-based embeddings
   - Document rollback strategies if needed

3. **Integration Patterns**
   - Research Google's embedding API integration patterns
   - Identify potential differences in API responses between Cohere and Google
   - Document error handling and retry strategies

### Expected Outcomes
- Clear understanding of Google's embedding model capabilities
- Identified migration approach for existing content
- Verified Google API access and proper configuration
- Research summary in `research.md`

## Phase 1: Data Model & Contracts

### Data Model Requirements

Based on the feature specification, the following data models need to be considered:

**Embedding Service**
- service_type (string): Either "cohere" or "google" for migration tracking
- embedding_dimensions (integer): Dimensions of the embedding vectors
- model_name (string): Name of the specific embedding model being used
- metadata (object): Additional configuration parameters for the service

**Content Representation (Vector)**
- embedding_data (array of floats): Vector representation using Google embeddings
- embedding_service_used (string): Track which service was used to generate the embedding
- created_at (timestamp): When the embedding was created
- metadata (object): Additional metadata for the embedding

### API Contract Requirements

Based on functional requirements, the following endpoints need to be maintained:

**Embedding Service Endpoints**
- `POST /api/v1/embeddings` - Generate embeddings using Google's model
  - Request: {texts: array of strings, input_type?: string}
  - Response: {embeddings: array of arrays of floats, model: string, usage: object}
  - Error: {error: string, code: number}

**Content Indexing Endpoints**
- `POST /api/v1/chat/index` - Index book content using Google embeddings
  - Request: {content: array of {title: string, content_text: string, section_ref: string}}
  - Response: {status: string, indexed_count: number, skipped_count: number}

## Phase 2: Implementation Approach

### Implementation Strategy
1. Develop Google embedding service wrapper
2. Update configuration to support Google's API
3. Modify indexing process to use Google embeddings
4. Implement migration path for existing content
5. Test and validate search quality with new embeddings
6. Deploy and monitor performance

### Success Criteria Verification
- All acceptance scenarios from spec will be tested
- Success criteria metrics will be measured
- Performance and accuracy targets will be validated