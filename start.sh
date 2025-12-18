#!/bin/bash

# Install Python dependencies for the backend
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start the backend service in the background
echo "Starting backend service..."
python -m src.scripts.initialize_vector_db 2>/dev/null || echo "Vector DB initialization skipped or failed"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for the backend to start
sleep 10

# Go back to the root directory
cd ..

# Build the frontend
npm run build

# Start the frontend proxy server
node server.js