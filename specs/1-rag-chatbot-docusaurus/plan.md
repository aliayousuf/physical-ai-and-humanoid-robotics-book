# Implementation Plan: RAG Chatbot for Physical AI and Humanoid Robotics Documentation

**Branch**: `1-rag-chatbot-docusaurus` | **Date**: 2025-12-17 | **Spec**: [specs/1-rag-chatbot-docusaurus/spec.md](specs/1-rag-chatbot-docusaurus/spec.md)
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

## Summary

Implementation of a RAG chatbot integrated with the Docusaurus documentation site for the Physical AI and Humanoid Robotics book. The system will use OpenAI Agents/ChatKit SDKs customized to use Gemini 2.5 Flash model, FastAPI backend, Neon Serverless Postgres, Qdrant Cloud Free Tier for vector storage, and Cohere embeddings. The chatbot will support general book content Q&A via RAG and Q&A limited to user-selected text, with persistent UI across all pages.

## Technical Context

**Language/Version**: Python 3.11, TypeScript/JavaScript for frontend
**Primary Dependencies**: FastAPI, OpenAI SDK (customized for Gemini), Cohere API, Qdrant client, Neon Postgres driver, Docusaurus
**Storage**: Neon Serverless Postgres for metadata/sessions, Qdrant Cloud Free Tier for embeddings
**Testing**: pytest, Jest for frontend components
**Target Platform**: Linux server (backend), Web browser (frontend)
**Project Type**: Web application (backend + frontend integration)
**Performance Goals**: 95% of queries respond within 5 seconds, 95% uptime during business hours
**Constraints**: <200ms p95 for internal operations, stay within free tier limits, <85% accuracy threshold for responses
**Scale/Scope**: Support up to 100 concurrent users, handle book content with up to 1M tokens

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the constitution, this implementation must:
1. Follow Spec-Driven Development (SDD) methodology
2. Include decision frameworks and reasoning activation
3. Accumulate reusable intelligence across components
4. Maintain "right altitude" - not too low (rigid) or too high (vague)
5. Use conditional reasoning frameworks
6. Disrupt predictable patterns with adaptive variability

## Project Structure

### Documentation (this feature)

```text
specs/1-rag-chatbot-docusaurus/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── session.py
│   │   ├── content.py
│   │   └── query.py
│   ├── services/
│   │   ├── rag_service.py
│   │   ├── embedding_service.py
│   │   ├── qdrant_service.py
│   │   └── gemini_service.py
│   ├── api/
│   │   ├── main.py
│   │   ├── chat.py
│   │   └── health.py
│   └── config/
│       ├── settings.py
│       └── database.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── requirements.txt
└── pyproject.toml

src/
├── components/
│   └── RagChatbot/
│       ├── RagChatbot.tsx
│       ├── ChatInterface.tsx
│       ├── Message.tsx
│       └── TextSelectionHandler.tsx
└── css/
    └── rag-chatbot.css

docs/
└── (existing book content - will be processed for RAG)

static/
└── (for chatbot assets if needed)
```

**Structure Decision**: Web application with separate backend service and frontend integration. The backend handles RAG processing, embeddings, and API endpoints, while the frontend provides the persistent UI component that integrates with Docusaurus.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple services (FastAPI, Qdrant, Neon) | Required for proper RAG architecture | Single service approach insufficient for vector search requirements |
| Custom OpenAI SDK modification | Needed to use Gemini instead of OpenAI | Direct API calls would be less maintainable |

## Implementation Phases

### Phase 0: Setup and Configuration
- Provision Neon Postgres and Qdrant Cloud Free Tier accounts
- Set up environment with API keys (Gemini, Cohere)
- Configure FastAPI backend with proper security
- Set up development environment and dependencies

### Phase 1: Data Preparation
- Ingest book content from Docusaurus Markdown files in docs/
- Chunk text appropriately for embedding
- Generate embeddings with Cohere models
- Store embeddings in Qdrant vector database
- Define Postgres schema for sessions and metadata

### Phase 2: Backend Development
- Build FastAPI endpoints for chatbot interactions
- Implement /query endpoint for general RAG
- Implement /selected-text-query endpoint for limited scope queries
- Integrate customized OpenAI/ChatKit SDK with Gemini 2.5 Flash
- Implement RAG pipeline: retrieve from Qdrant, augment prompt, generate response

### Phase 3: Frontend Integration
- Create React component for chatbot UI
- Embed chatbot in Docusaurus layout
- Add text selection event listeners
- Implement state management for conversation persistence
- Ensure responsive design and accessibility

### Phase 4: Testing and Deployment
- Write unit/integration tests for RAG accuracy
- Test text selection handling and error cases
- Deploy backend to cloud platform (e.g., Render, Railway)
- Deploy frontend via Docusaurus build to static hosting
- Set up monitoring for free tier limits