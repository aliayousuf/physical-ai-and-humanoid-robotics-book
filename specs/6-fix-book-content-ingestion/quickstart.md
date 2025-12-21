# Quickstart Guide: Book Content Ingestion

## Overview
This guide explains how to set up and run the book content ingestion system to populate the vector database for the RAG chatbot using Gemini 2.5 Flash model.

## Prerequisites
- Python 3.11+
- Google Gemini API key
- Qdrant vector database (local or cloud)
- Access to the docs folder with book content

## Setup

### 1. Environment Configuration
```bash
# Set up environment variables
export GEMINI_API_KEY="your-gemini-api-key"
export QDRANT_URL="your-qdrant-url"
export QDRANT_API_KEY="your-qdrant-api-key"  # if using cloud
export DOCS_PATH="./docs"  # Path to the book content folder
export GEMINI_MODEL="gemini-2.5-flash"  # Default model to use
```

### 2. Install Dependencies
```bash
pip install google-generativeai fastapi uvicorn qdrant-client PyPDF2 markdown python-multipart
```

### 3. Verify Document Formats
Ensure your docs folder contains supported formats:
- Markdown files (.md)
- PDF files (.pdf)
- Text files (.txt)

## Running Ingestion

### 1. Manual Ingestion
Trigger the ingestion process via API:

```bash
curl -X POST http://localhost:8000/api/v1/ingestion/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "force_reprocess": false,
    "file_patterns": ["*.md", "*.pdf", "*.txt"]
  }'
```

### 2. Check Ingestion Status
Monitor the progress of your ingestion job:

```bash
curl -X GET http://localhost:8000/api/v1/ingestion/status/{job_id}
```

## Using the Chat Interface

Once ingestion is complete, you can query the chatbot:

```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the book say about humanoid robotics?",
    "max_results": 5,
    "similarity_threshold": 0.7
  }'
```

## Configuration Options

### Environment Variables
- `DOCS_PATH`: Path to the documentation folder (default: "./docs")
- `QDRANT_COLLECTION_NAME`: Name of the vector collection (default: "book_content")
- `CHUNK_SIZE`: Size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)
- `GEMINI_MODEL`: Google Gemini model to use (default: "gemini-2.5-flash")

### Processing Options
- `force_reprocess`: Set to true to reprocess all documents even if already in database
- `file_patterns`: Specify which file types to process

## Troubleshooting

### Common Issues
1. **No content found**: Verify the docs folder path and file permissions
2. **API errors**: Check that your Gemini and Qdrant credentials are correct
3. **Poor results**: Adjust similarity_threshold or check that content was properly ingested

### Verification Steps
1. Confirm files exist in the docs folder
2. Check that the ingestion job completed successfully
3. Verify content appears in the Qdrant collection
4. Test a simple query to the chatbot