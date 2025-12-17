# RAG Chatbot Setup and Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Production Deployment](#production-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Testing](#testing)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Overview

This guide provides step-by-step instructions for setting up and deploying the RAG (Retrieval-Augmented Generation) chatbot for the Physical AI and Humanoid Robotics book documentation. The system consists of a FastAPI backend with vector search capabilities and a React frontend component for Docusaurus integration.

## Prerequisites

### System Requirements
- Python 3.11+
- Node.js 18+ (for Docusaurus frontend)
- Git
- Docker and Docker Compose (for containerized deployment)
- Access to the following API keys:
  - Google Gemini API key
  - Cohere API key
  - Qdrant Cloud account
  - Neon Postgres account (for session storage)

### External Services Setup
1. **Qdrant Cloud Free Tier**
   - Sign up at [qdrant.io](https://qdrant.io)
   - Create a new cloud instance
   - Note your URL and API key

2. **Neon Postgres Serverless**
   - Sign up at [neon.tech](https://neon.tech)
   - Create a new project
   - Note your connection string

3. **Google Gemini API**
   - Get API key from [Google AI Studio](https://aistudio.google.com)

4. **Cohere API**
   - Get API key from [Cohere](https://cohere.com)

## Local Development Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd physical-ai-and-humanoid-robotics-book
```

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Configuration
1. Create a `.env` file in the `backend` directory:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys and configuration:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   DATABASE_URL=your_neon_postgres_connection_string
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION_NAME=book_content
   ENVIRONMENT=development
   LOG_LEVEL=info
   SESSION_EXPIRATION_HOURS=24
   RAG_TOP_K=5
   RAG_SCORE_THRESHOLD=0.7
   MAX_QUERY_LENGTH=2000
   MAX_RESPONSE_TOKENS=1000
   ```

### 4. Initialize Services
1. Initialize the vector database:
   ```bash
   python -m src.scripts.initialize_vector_db
   ```

2. Ingest book content from the docs directory:
   ```bash
   python -m src.scripts.ingest_docs
   ```

### 5. Run the Backend Service
```bash
# Using uvicorn for development
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the start script
./start.sh  # On Windows: start.bat
```

The backend API will be available at `http://localhost:8000`.

### 6. Frontend Integration with Docusaurus
1. From the project root, install dependencies:
   ```bash
   npm install
   ```

2. The RAG chatbot component is located in `src/components/RagChatbot/` and can be integrated into Docusaurus layouts.

3. Run Docusaurus in development mode:
   ```bash
   npm run start
   ```

## Production Deployment

### Option 1: Docker Deployment

#### 1. Build and Run with Docker Compose
1. Ensure you have Docker and Docker Compose installed
2. Create your `.env` file with production configuration
3. Run the services:
   ```bash
   docker-compose up -d
   ```

#### 2. Using the Deployment Script
1. Make the script executable:
   ```bash
   chmod +x deploy.sh
   ```

2. Run the deployment:
   ```bash
   ./deploy.sh
   ```

### Option 2: Kubernetes Deployment

#### 1. Prepare Kubernetes Secrets
```bash
kubectl create secret generic rag-chatbot-secrets \
  --from-literal=gemini-api-key=YOUR_GEMINI_KEY \
  --from-literal=cohere-api-key=YOUR_COHERE_KEY \
  --from-literal=database-url=YOUR_DATABASE_URL \
  --from-literal=qdrant-url=YOUR_QDRANT_URL \
  --from-literal=qdrant-api-key=YOUR_QDRANT_KEY
```

#### 2. Deploy to Kubernetes
```bash
kubectl apply -f k8s-deployment.yaml
```

### Option 3: Cloud Platform Deployment

#### For Render.com:
1. Create a new Web Service
2. Set the repository to your fork
3. Set the environment variables in Render dashboard
4. Set the build command to: `pip install -r requirements-prod.txt`
5. Set the start command to: `gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:$PORT src.api.main:app`

#### For Railway:
1. Import your repository
2. Add the environment variables in Railway dashboard
3. Deploy the application

## Environment Configuration

### Required Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key
- `COHERE_API_KEY`: Cohere API key
- `DATABASE_URL`: Neon Postgres connection string
- `QDRANT_URL`: Qdrant Cloud URL
- `QDRANT_API_KEY`: Qdrant Cloud API key

### Optional Environment Variables
- `ENVIRONMENT`: Set to "production" for production environments (default: "development")
- `LOG_LEVEL`: Logging level (default: "info")
- `LOG_FILE`: Path to log file (optional)
- `SESSION_EXPIRATION_HOURS`: Session expiration time in hours (default: 24)
- `RAG_TOP_K`: Number of results to retrieve (default: 5)
- `RAG_SCORE_THRESHOLD`: Minimum similarity score (default: 0.7)
- `MAX_QUERY_LENGTH`: Maximum query length (default: 2000)
- `MAX_RESPONSE_TOKENS`: Maximum response tokens (default: 1000)

## Testing

### Running Unit Tests
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/
```

### Running Specific Test Suites
```bash
# Test the RAG service
pytest tests/unit/test_rag_service.py

# Test the API endpoints
pytest tests/integration/test_rag_integration.py
```

### End-to-End Testing
The integration tests in `tests/integration/test_rag_integration.py` cover the complete flow:
- Session creation
- General RAG queries
- Selected text queries
- Conversation history retrieval
- Error handling scenarios

## Monitoring and Maintenance

### API Monitoring
- The system tracks usage of external services (Cohere, Gemini, Qdrant) to stay within free tier limits
- Performance metrics are collected for each endpoint
- Security events are logged

### Log Files
- In development: logs appear in the console
- In production: logs are written to the configured log file or container logs

### Health Checks
- The `/health` endpoint provides system status information
- Configure your load balancer or monitoring service to check this endpoint

### Free Tier Monitoring
The system includes monitoring for free tier usage:
- Cohere: Monitored for embedding requests
- Gemini: Monitored for generation requests
- Qdrant: Monitored for vector operations
- Neon: Monitored for database operations

When usage reaches 80% of limits, warnings are logged. At 100%, appropriate error responses are returned.

### Backup and Recovery
- Regular backups of the Postgres database should be configured in Neon
- The vector database in Qdrant should also be backed up according to your needs
- Configuration files should be version controlled

## Troubleshooting

### Common Issues

1. **API Keys Not Working**: Verify all API keys are correctly set in the environment
2. **Database Connection Errors**: Check the Neon Postgres connection string format
3. **Vector Database Errors**: Verify Qdrant URL and API key
4. **Rate Limiting**: Check if you're exceeding free tier limits

### Useful Commands
- Check service health: `curl http://localhost:8000/api/v1/health`
- Get content summary: `curl http://localhost:8000/api/v1/chat/content/summary`
- View container logs: `docker logs rag-chatbot-api`

## Scaling Considerations

### Horizontal Scaling
- The application is designed to be stateless (except for in-memory session cache)
- Multiple instances can be run behind a load balancer
- Consider using Redis for distributed session storage in production

### Performance Optimization
- Embeddings are cached to reduce API calls
- Vector database is optimized for similarity search
- Responses are logged for performance analysis

## Security Considerations

### API Security
- Input sanitization is performed on all user inputs
- Rate limiting prevents abuse
- Security headers are added to all responses
- Potential XSS and SQL injection attempts are logged

### Data Security
- API keys are stored in environment variables
- Sensitive data is not logged
- Connection to external services is encrypted

## Updating the System

### Backend Updates
1. Pull the latest code
2. Install any new dependencies: `pip install -r requirements.txt`
3. Restart the service

### Content Updates
1. Update the documentation in the `docs/` directory
2. Re-run the ingestion script: `python -m src.scripts.ingest_docs`
3. The new content will be available for RAG queries

This guide provides a complete setup and deployment process for the RAG chatbot. For additional support, refer to the individual README files in the backend and frontend directories.