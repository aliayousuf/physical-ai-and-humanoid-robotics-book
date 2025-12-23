#!/bin/bash

# Install Python dependencies for the backend
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start the backend service in the background
echo "Starting backend service..."
python -m src.scripts.initialize_vector_db 2>/dev/null || echo "Vector DB initialization skipped or failed"

# Start the backend on port 8000 in the background
PORT_BACKEND=${BACKEND_PORT:-8000}
echo "Starting backend on port $PORT_BACKEND"
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT_BACKEND &

# Wait for backend to start
sleep 10

# Simple check if backend is running (optional, since nc may not be available)
echo "Waiting to ensure backend has started..."
sleep 5

# Go back to the root directory
cd ..

# Build the frontend if not already built
if [ ! -d "build" ]; then
  echo "Building frontend..."
  npm install
  npm run build
fi

# Set the backend URL for the proxy
export BACKEND_URL="http://localhost:${BACKEND_PORT:-8000}"

# Start the frontend proxy server on the main port
MAIN_PORT=${PORT:-3000}
echo "Starting proxy server on port $MAIN_PORT"
node server.js