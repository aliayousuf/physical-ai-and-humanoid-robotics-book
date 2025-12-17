#!/bin/bash

# Start the RAG Chatbot backend service
echo "Starting RAG Chatbot backend service..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

# Start the FastAPI application
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload