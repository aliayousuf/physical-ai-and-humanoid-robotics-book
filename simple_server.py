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
        sys.path.insert(0, str(backend_path))

        # Import and run initialization
        from src.scripts.initialize_vector_db import initialize_vector_db
        asyncio.run(initialize_vector_db())
        print("Vector database initialized successfully!")
    except Exception as e:
        print(f"Vector DB initialization failed or skipped: {e}")

def main():
    print("Starting server initialization...")

    # Run initialization
    run_initialization()

    # Get port from environment
    port = int(os.environ.get("PORT", "8000"))
    print(f"Using port: {port}")

    # Change to backend directory for running the app
    backend_path = Path("/app/backend")
    os.chdir(backend_path)
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