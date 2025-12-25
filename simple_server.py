#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
"""
import os
import sys
from pathlib import Path
import uvicorn
import asyncio

def run_initialization():
    """Run vector database initialization"""
    try:
        # Change to backend directory
        backend_path = Path("/app/backend")
        os.chdir(backend_path)

        # Add to Python path
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        # Import and run initialization
        from src.scripts.initialize_vector_db import initialize_vector_db
        asyncio.run(initialize_vector_db())
        print("Vector database initialized successfully!")
    except Exception as e:
        print(f"Vector DB initialization failed or skipped: {e}")

def main():
    print("Starting server initialization...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment variables: {dict(os.environ)}")

    # Get port from environment - Railway should set this
    port_env = os.environ.get("PORT")
    if port_env is None:
        print("ERROR: PORT environment variable not set by Railway")
        sys.exit(1)

    port = int(port_env)
    print(f"Using Railway-assigned port: {port}")

    # Run initialization
    run_initialization()

    # Change to backend directory for running the app
    backend_path = Path("/app/backend")
    os.chdir(backend_path)
    print(f"Changed to backend directory: {os.getcwd()}")

    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    print(f"Starting FastAPI app on 0.0.0.0:{port}")

    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()