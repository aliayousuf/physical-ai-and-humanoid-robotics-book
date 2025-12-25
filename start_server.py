#!/usr/bin/env python3
"""
Python-based startup script that avoids shell commands
"""
import os
import sys
import subprocess
import asyncio
from pathlib import Path


def initialize_vector_db():
    """Initialize the vector database"""
    try:
        # Change to the backend directory
        backend_path = Path("/app/backend")
        os.chdir(backend_path)

        # Add the backend directory to Python path
        sys.path.insert(0, str(backend_path))

        # Import and run the initialization script
        from src.scripts.initialize_vector_db import initialize_vector_db as init_func
        asyncio.run(init_func())
        print("Vector database initialized successfully!")
    except Exception as e:
        print(f"Vector DB initialization skipped or failed: {e}")


def start_server():
    """Start the uvicorn server"""
    import uvicorn

    # Change to the backend directory
    backend_path = Path("/app/backend")
    os.chdir(backend_path)

    # Add the backend directory to Python path
    sys.path.insert(0, str(backend_path))

    # Get the port from environment - Railway always sets this
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting server on port {port}")

    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    print("Starting backend server...")

    # Initialize vector database
    initialize_vector_db()

    # Start the server
    start_server()