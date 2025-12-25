#!/usr/bin/env python3
"""
Python-based startup script that avoids shell commands and ensures proper port binding
"""
import os
import sys
from pathlib import Path


def initialize_vector_db():
    """Initialize the vector database"""
    try:
        # Change to the backend directory
        backend_path = Path("/app/backend")
        os.chdir(backend_path)

        # Add the backend directory to Python path
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        # Import and run the initialization script
        import asyncio
        from src.scripts.initialize_vector_db import initialize_vector_db as init_func
        asyncio.run(init_func())
        print("Vector database initialized successfully!")
    except Exception as e:
        print(f"Vector DB initialization skipped or failed: {e}")


def start_server():
    """Start the uvicorn server with proper port binding"""
    import uvicorn

    # Change to the backend directory
    backend_path = Path("/app/backend")
    os.chdir(backend_path)

    # Add the backend directory to Python path
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    # Get the port from environment - Railway always sets this
    # If not set, this is an error in Railway deployment
    port_env = os.environ.get("PORT")
    if port_env is None:
        print("ERROR: PORT environment variable not set by Railway")
        sys.exit(1)

    port = int(port_env)
    print(f"Environment PORT variable: {port_env}")
    print(f"Starting server on port {port}")
    print(f"Host: 0.0.0.0 (binding to all interfaces)")

    # Start the server with explicit configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Bind to all interfaces
        port=port,        # Use the port provided by Railway
        reload=False,
        log_level="info",
        access_log=True,
        use_colors=True
    )


if __name__ == "__main__":
    print("Starting backend server...")
    print(f"Current working directory: {os.getcwd()}")

    # Initialize vector database
    initialize_vector_db()

    # Start the server
    start_server()