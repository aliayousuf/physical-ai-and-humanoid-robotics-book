# Implementation Plan: Fix Book Content Ingestion into Vector Database

**Branch**: `6-fix-book-content-ingestion` | **Date**: 2025-12-20 | **Spec**: specs/6-fix-book-content-ingestion/spec.md
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The primary requirement is to fix the book content ingestion into the vector database so that the chatbot can properly answer questions about the book content. The technical approach involves implementing a document ingestion pipeline that scans the docs folder, processes documentation files into vector embeddings, and stores them in the vector database for semantic search. This will enable the RAG chatbot to retrieve relevant content when answering user queries.

## Technical Context

**Language/Version**: Python 3.11 (based on ChatKit SDK requirements)
**Primary Dependencies**: OpenAI Agents SDK (customized for Gemini), ChatKit SDKs, FastAPI, Qdrant (vector database), PyPDF2/Markdown libraries, Google Generative AI SDK
**Storage**: Qdrant Cloud Free Tier (vector database), Neon Serverless Postgres (for structured knowledge)
**Testing**: pytest for unit and integration tests
**Target Platform**: Linux server (web application backend)
**Project Type**: Web application (backend service for RAG chatbot)
**Performance Goals**: Process documentation within 5 minutes, respond to queries within 2 seconds
**Constraints**: <200ms p95 latency for query responses, handle common doc formats (Markdown, PDF, text)
**Scale/Scope**: Support book content with multiple chapters and sections, handle concurrent user queries

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the constitution, this implementation must:
1. Use OpenAI Agents / ChatKit SDKs as specified in the constitution (Section 35) - MODIFIED to use customized ChatKit SDKs with Gemini API
2. Use FastAPI backend as specified in the constitution (Section 35)
3. Use Qdrant Cloud Free Tier for vector retrieval as specified in the constitution (Section 35)
4. Use Neon Serverless Postgres for structured knowledge as specified in the constitution (Section 35)
5. Follow Spec-Driven Development (SDD) methodology as specified in the constitution (Section 39)

**Note**: The implementation uses OpenAI Agents/ChatKit SDKs customized to use Gemini 2.5 Flash model with Gemini API key, which is a modification of the original constitution requirement but maintains the ChatKit SDK framework.

## Project Structure

### Documentation (this feature)

```text
specs/6-fix-book-content-ingestion/
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
│   │   ├── document.py
│   │   ├── embedding.py
│   │   └── chat.py
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── vector_db_service.py
│   │   └── chat_service.py
│   ├── api/
│   │   ├── ingestion_api.py
│   │   └── chat_api.py
│   └── utils/
│       ├── file_parser.py
│       └── text_splitter.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/
```

**Structure Decision**: Selected web application backend structure to support the RAG chatbot functionality. The backend will handle document ingestion and chat interactions as specified in the constitution.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |