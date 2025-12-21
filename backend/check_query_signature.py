#!/usr/bin/env python3
"""
Test script to understand the correct Qdrant query method signature
"""
from qdrant_client import QdrantClient
from src.config.settings import settings

def check_query_method_signature():
    print("Checking Qdrant query method signature...")

    # Initialize Qdrant client
    try:
        if settings.qdrant_api_key:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=True
            )
        else:
            client = QdrantClient(url=settings.qdrant_url)

        print(f"Qdrant client initialized successfully")

        # Get the query method signature
        import inspect
        if hasattr(client, 'query'):
            sig = inspect.signature(client.query)
            print(f"query() method signature: {sig}")
        else:
            print("query() method not found")

        if hasattr(client, 'search'):
            sig = inspect.signature(client.search)
            print(f"search() method signature: {sig}")
        else:
            print("search() method not found")

        # Also check for http-based methods if they exist
        if hasattr(client, '_client'):
            http_client = client._client
            print(f"HTTP client type: {type(http_client)}")

    except Exception as e:
        print(f"Failed to initialize Qdrant client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_query_method_signature()