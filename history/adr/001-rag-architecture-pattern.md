# ADR-001: RAG Architecture Pattern for Physical AI Book Documentation

## Status
Accepted

## Date
2025-12-17

## Context
We need to implement a RAG (Retrieval-Augmented Generation) chatbot for the Physical AI and Humanoid Robotics book documentation. The system must:
- Answer general questions about the entire book content
- Answer questions based on user-selected text only
- Be embedded in the Docusaurus documentation site
- Support persistent conversations across pages
- Operate within free tier constraints of cloud services
- Ensure security and proper input sanitization

The architecture must balance performance, cost, and scalability while providing accurate, cited responses from the book content.

## Decision
We will implement a multi-service RAG architecture using:

**Backend Services:**
- FastAPI: As the primary backend framework for handling API requests with excellent performance and automatic documentation
- Google Gemini 2.5 Flash: As the LLM for response generation due to cost-effectiveness and performance
- Cohere Embeddings: For generating vector representations of book content
- Qdrant Cloud Free Tier: As the vector database for similarity search operations
- Neon Serverless Postgres: For storing session data, query history, and metadata

**Frontend Integration:**
- React Components: Embedded directly in Docusaurus layout for seamless integration
- Text Selection Handlers: Event-driven approach for contextual queries
- Session State Management: Cross-page conversation persistence

**Architecture Pattern:**
- Retrieve-Augment-Generate pipeline for processing queries
- Separate query endpoints for general vs selected-text modes
- Asynchronous processing for optimal performance

## Alternatives Considered

1. **Monolithic Architecture**: Single service handling all RAG operations
   - Pros: Simpler deployment, fewer services to manage
   - Cons: Insufficient for vector search requirements, harder to scale components independently

2. **Different Vector Database Options**:
   - Pinecone: More commercial focus, potentially insufficient free tier
   - Weaviate: Good alternative but more complex setup
   - FAISS: Requires self-hosting, not cloud-native
   - Qdrant: Selected for its free tier, Python client, and vector search optimization

3. **Different Embedding Services**:
   - OpenAI Embeddings: Higher cost, may exceed free tier limits
   - Self-hosted Sentence Transformers: Requires more resources and maintenance
   - Google Embeddings: Different pricing model and integration complexity
   - Cohere: Selected for quality, documentation, and free tier availability

4. **Different LLM Options**:
   - OpenAI GPT models: Higher cost than Gemini
   - Open-source models: Require more infrastructure and maintenance
   - Google Gemini: Selected for cost-effectiveness and performance

5. **Frontend Integration Approaches**:
   - Iframe embedding: Less integrated, potential styling issues
   - External widget: Less control over UI/UX
   - React component: Selected for seamless Docusaurus integration

## Consequences

**Positive:**
- Scalable architecture with independent components
- Cost-effective solution using free tier services
- High performance with specialized tools for each function
- Seamless integration with existing Docusaurus documentation
- Proper separation of concerns between vector search and metadata storage
- Support for both general and contextual querying modes

**Negative:**
- More complex infrastructure with multiple services
- Dependency on multiple external APIs
- Potential for increased latency due to multiple service calls
- Complexity in managing multiple service configurations
- Need for careful monitoring of free tier usage

## References
- specs/1-rag-chatbot-docusaurus/plan.md
- specs/1-rag-chatbot-docusaurus/research.md
- specs/1-rag-chatbot-docusaurus/data-model.md
- specs/1-rag-chatbot-docusaurus/contracts/api-contracts.md