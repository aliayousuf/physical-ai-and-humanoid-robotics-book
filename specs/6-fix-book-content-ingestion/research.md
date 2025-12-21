# Research: Book Content Ingestion for RAG Chatbot

## Overview
This research document addresses the technical requirements for fixing book content ingestion into the vector database to enable the RAG chatbot functionality.

## Decision: Vector Database Selection
**Rationale**: The constitution specifies using Qdrant Cloud Free Tier for vector retrieval, so this is predetermined.
**Alternatives considered**: Pinecone, Weaviate, ChromaDB - but Qdrant is required by constitution.

## Decision: Document Processing Libraries
**Rationale**: Need libraries to process different document formats (Markdown, PDF, text) from the docs folder.
**Chosen approach**:
- PyPDF2 or pypdf for PDF processing
- markdown library for Markdown processing
- Standard Python libraries for text files

## Decision: Embedding Model Selection
**Rationale**: Need to generate vector embeddings for the book content to enable semantic search.
**Chosen approach**: Google's embedding API via Google Generative AI SDK, compatible with the customized ChatKit SDKs for Gemini.
**Alternatives considered**: Sentence Transformers, Cohere embeddings, OpenAI embeddings - but using Google embeddings to align with Gemini API requirement.

## Decision: File Watching vs Batch Processing
**Rationale**: Need to determine how to handle updates to the docs folder.
**Chosen approach**: Implement both - batch processing for initial ingestion and optional file watching for continuous updates.
**Alternatives considered**: Only batch processing (simpler) vs. real-time watching (more complex but responsive).

## Decision: Text Chunking Strategy
**Rationale**: Large documents need to be split into smaller chunks for effective embedding and retrieval.
**Chosen approach**: Recursive character text splitter with overlap to maintain context.
**Alternatives considered**: Sentence-based splitting, fixed-length splitting - recursive character splitting maintains context better.

## Decision: ChatKit Integration Approach
**Rationale**: The constitution requires using ChatKit SDKs for the chatbot functionality.
**Chosen approach**: Implement ingestion pipeline that feeds content into ChatKit's knowledge base using customized ChatKit SDKs with Gemini API.
**Alternatives considered**: Direct integration with OpenAI APIs vs. ChatKit abstractions - ChatKit is required by constitution.

## Decision: ChatKit Expert Subagent and Skills Usage
**Rationale**: The feature requires utilizing chatkit-expert subagent and chatkit skills as specified in the original request.
**Chosen approach**: Implement specialized subagents for document processing and content ingestion using ChatKit skills framework.
**Implementation**: Will create dedicated skills for document parsing, vector generation, and content retrieval to leverage the chatkit-expert subagent capabilities.

## Technical Implementation Steps
1. Scan docs folder for supported file types
2. Process each file format using appropriate parser
3. Split documents into chunks with appropriate size limits
4. Generate embeddings for each chunk using OpenAI API
5. Store embeddings in Qdrant vector database with metadata
6. Update ChatKit knowledge base with new content
7. Implement error handling for unsupported formats
8. Create API endpoints for manual/triggered ingestion