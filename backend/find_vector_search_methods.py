#!/usr/bin/env python3
"""
Test script to find all available Qdrant methods and understand vector search
"""
from qdrant_client import QdrantClient
from src.config.settings import settings

def find_vector_search_methods():
    print("Finding all Qdrant methods related to vector search...")

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
        print(f"Client type: {type(client)}")

        # List all methods that might be related to vector search
        all_methods = [method for method in dir(client) if not method.startswith('_')]
        vector_search_methods = []

        for method in all_methods:
            if any(keyword in method.lower() for keyword in ['search', 'query', 'find', 'retrieve', 'vector']):
                vector_search_methods.append(method)

        print(f"Vector search related methods: {vector_search_methods}")

        # Check for the actual search method that accepts vectors
        # In newer versions, it might be called something else
        if hasattr(client, 'search_points'):
            print("search_points method exists")
        if hasattr(client, 'search'):
            print("search method exists")
        if hasattr(client, 'query'):
            print("query method exists (but appears to be for text queries)")

        # Let's check the client object more deeply
        print("\nChecking for low-level API methods...")
        if hasattr(client, '_client'):
            low_level_client = client._client
            low_level_methods = [method for method in dir(low_level_client) if not method.startswith('_')]
            low_level_search_methods = []

            for method in low_level_methods:
                if any(keyword in method.lower() for keyword in ['search', 'query', 'find', 'retrieve', 'vector']):
                    low_level_search_methods.append(method)

            print(f"Low-level search methods: {low_level_search_methods}")

        # Check if the client has the http-based methods
        if hasattr(client, 'http'):
            print("HTTP API is available")
            if hasattr(client.http, 'points_api'):
                print("Points API is available")
                # The search method might be in the http API
                points_api_methods = [method for method in dir(client.http.points_api) if not method.startswith('_')]
                points_search_methods = [method for method in points_api_methods if 'search' in method.lower() or 'query' in method.lower()]
                print(f"Points API search methods: {points_search_methods}")

    except Exception as e:
        print(f"Failed to initialize Qdrant client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    find_vector_search_methods()