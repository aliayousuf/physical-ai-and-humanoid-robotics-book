#!/usr/bin/env python3
"""
Check what methods are available on the Qdrant client
"""
from qdrant_client import QdrantClient
from src.config.settings import settings

def check_available_methods():
    print("Checking available methods on Qdrant client...")

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

        # List all methods that contain 'search' in the name
        all_attrs = dir(client)
        search_methods = [attr for attr in all_attrs if 'search' in attr.lower()]

        print(f"Methods containing 'search': {search_methods}")

        # List all methods that might be related to search
        search_related = [attr for attr in all_attrs if any(keyword in attr.lower() for keyword in ['search', 'find', 'query', 'retrieve'])]

        print(f"Search-related methods: {search_related}")

        # Print all methods to see what's available
        print("\nAll available methods (first 20):")
        for i, attr in enumerate(all_attrs):
            if not attr.startswith('_'):  # Skip private methods
                print(f"  {attr}")
                if i >= 30:  # Limit output
                    print("  ... (truncated)")
                    break

    except Exception as e:
        print(f"Failed to initialize Qdrant client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_available_methods()