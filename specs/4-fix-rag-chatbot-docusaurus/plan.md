# Implementation Plan: Fix RAG Chatbot for Docusaurus Documentation

**Feature**: 4-fix-rag-chatbot-docusaurus
**Created**: 2025-12-17
**Status**: Draft
**Author**: Claude

## Technical Context

The RAG (Retrieval-Augmented Generation) chatbot for the Physical AI and Humanoid Robotics book documentation is currently non-functional. Based on the project constitution, the system should include an embedded RAG chatbot built with OpenAI Agents/ChatKit SDKs, FastAPI backend, Neon Serverless Postgres, and Qdrant Cloud Free Tier.

### Current Architecture Understanding

The system likely consists of:
- **Frontend**: Embedded chatbot component in Docusaurus pages
- **Backend**: FastAPI service handling chat queries and RAG operations
- **Vector DB**: Qdrant for semantic search of book content
- **Relational DB**: Neon Postgres for session management and metadata
- **LLM Service**: OpenAI API for generating responses

### Known Unknowns (NEEDS CLARIFICATION)
- Current error logs showing specific failure points
- Current backend implementation status and which components are broken
- Current frontend component implementation and integration with Docusaurus
- Current vector database state and whether book content has been indexed
- Current API endpoint configurations and authentication setup
- Current deployment configuration and environment variables

### Dependencies
- OpenAI API access and credentials
- Qdrant Cloud Free Tier account and collection setup
- Neon Serverless Postgres database and schema
- Docusaurus documentation site integration points
- FastAPI backend service

## Constitution Check

### Alignment with Core Principles

**Decision Point Mapping**: This fix addresses critical decisions about what components need to be working for a functional RAG system, distinguishing between infrastructure setup (agent-executable) versus complex debugging (requires human reasoning).

**Reasoning Activation**: The implementation will force reasoning about system architecture, error diagnosis, and integration patterns rather than just applying generic fixes.

**Intelligence Accumulation**: This fix will produce reusable debugging and deployment intelligence for future RAG system implementations.

**Right Altitude**: The plan maintains appropriate decision frameworks with concrete debugging steps, error handling patterns, and verification checkpoints without being overly prescriptive or vague.

**Frameworks Over Rules**: The approach uses conditional reasoning for different failure scenarios rather than rigid step-by-step instructions.

**Meta-Awareness Against Convergence**: The plan includes multiple verification and testing approaches to avoid predictable debugging patterns.

## Planning Gates

### Gate 1: Architecture Feasibility ✅
- The required technologies (FastAPI, Qdrant, Neon Postgres, OpenAI) are available and compatible
- The architecture aligns with the project constitution requirements
- All necessary service accounts and access can be provisioned

### Gate 2: Technical Requirements Compliance ✅
- Solution will include embedded RAG chatbot in documentation
- Will support general book content queries and selected text mode
- Will maintain conversation context as required
- Will include proper error handling and security measures

### Gate 3: Resource Availability ✅
- Backend folder exists for implementation
- Docusaurus integration points available
- Book content in docs folder available for indexing
- Required dependencies can be installed

### Gate 4: Risk Assessment
- **Medium Risk**: Complex system integration may have multiple failure points
- **Mitigation**: Implement verification checkpoints at each integration level
- **Medium Risk**: External service dependencies (OpenAI, Qdrant, Neon) may have rate limits
- **Mitigation**: Implement proper rate limiting and error handling

## Phase 0: Research & Discovery

### Research Tasks

1. **Current System Analysis**
   - Investigate existing backend implementation in `/backend` folder
   - Identify current error logs and failure points
   - Document current architecture and integration points

2. **Dependency Verification**
   - Verify OpenAI API access and credentials
   - Verify Qdrant Cloud collection exists and is populated
   - Verify Neon Postgres database connectivity and schema
   - Verify Docusaurus integration mechanisms

3. **Integration Patterns**
   - Research best practices for FastAPI-Docusaurus integration
   - Identify optimal RAG implementation patterns for documentation sites
   - Document error handling and fallback strategies

### Expected Outcomes
- Clear understanding of current system state
- Identification of specific failure points
- Verified access to all required services
- Research summary in `research.md`

## Phase 1: Data Model & Contracts

### Data Model Requirements

Based on the feature specification, the following data models are required:

**User Session**
- session_id (string, unique, required)
- created_at (timestamp, required)
- last_interaction (timestamp, required)
- conversation_history (array of messages, optional)
- metadata (object, optional)

**Book Content**
- content_id (string, unique, required)
- title (string, required)
- section_ref (string, required)
- content_text (string, required)
- embedding_vector (array, required)
- page_reference (string, optional)

**Content Representation (Vector)**
- content_id (string, foreign key to Book Content)
- vector_data (array of floats, required)
- metadata (object, optional)

**Query History**
- query_id (string, unique, required)
- session_id (string, foreign key to User Session)
- query_text (string, required)
- response_text (string, required)
- timestamp (timestamp, required)
- source_references (array of strings, optional)

**User Selection Context**
- selection_id (string, unique, required)
- session_id (string, foreign key to User Session)
- selected_text (string, required)
- page_context (string, required)
- created_at (timestamp, required)

### API Contract Requirements

Based on functional requirements, the following endpoints are needed:

**Chat Service Endpoints**
- `POST /api/chat/query` - Process user queries with RAG
  - Request: {query: string, session_id?: string, mode: "general"|"selected_text", selected_text?: string}
  - Response: {response: string, references: array, session_id: string, context: object}
  - Error: {error: string, code: number}

- `GET /api/chat/session/{session_id}` - Get session details
  - Response: {session_id: string, created_at: timestamp, last_interaction: timestamp, history: array}

- `POST /api/chat/session` - Create new session
  - Response: {session_id: string}

**Content Service Endpoints**
- `GET /api/content/search` - Semantic search of book content
  - Request: {query: string, limit?: number}
  - Response: {results: array of {content_id, title, content_text, relevance_score}}

- `POST /api/content/index` - Index book content for RAG
  - Request: {content: array of {title, content_text, section_ref}}
  - Response: {status: string, indexed_count: number}

## Phase 2: Implementation Approach

### Implementation Strategy
1. Diagnose current system failures
2. Set up proper backend services
3. Implement RAG functionality
4. Integrate with Docusaurus frontend
5. Add error handling and monitoring
6. Test and validate all functionality

### Success Criteria Verification
- All acceptance scenarios from spec will be tested
- Success criteria metrics will be measured
- Performance and reliability targets will be validated