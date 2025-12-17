#!/bin/bash

# Deployment script for RAG Chatbot API

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
SERVICE_NAME="rag-chatbot-api"
IMAGE_NAME="rag-chatbot-api"
ENV_FILE=".env"

# Check if required files exist
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE file not found. Please create it with your API keys and configuration."
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    echo "Error: Dockerfile not found."
    exit 1
fi

echo "Starting deployment of $SERVICE_NAME..."

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

# Stop existing containers if running
if [ "$(docker ps -q -f name=$SERVICE_NAME)" ]; then
    echo "Stopping existing container..."
    docker stop $SERVICE_NAME
fi

# Remove old containers if they exist
if [ "$(docker ps -aq -f name=$SERVICE_NAME)" ]; then
    echo "Removing old container..."
    docker rm $SERVICE_NAME
fi

# Run the new container
echo "Starting $SERVICE_NAME container..."
docker run -d \
    --name $SERVICE_NAME \
    --env-file $ENV_FILE \
    -p 8000:8000 \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    $IMAGE_NAME

echo "Deployment completed successfully!"
echo "Service is running at http://localhost:8000"
echo "Check logs with: docker logs $SERVICE_NAME"

# Wait a moment for the service to start
sleep 5

# Health check
echo "Performing health check..."
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✓ Service is healthy"
else
    echo "⚠ Service may not be healthy. Check logs with: docker logs $SERVICE_NAME"
fi