#!/usr/bin/env python3
"""
Test script to verify Qdrant client search functionality
"""
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config.settings import settings

def test_qdrant_client():
    print("Testing Qdrant client...")

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
        print(f"Qdrant client version: {getattr(client, '__version__', 'Unknown')}")

        # Check if search method exists
        if hasattr(client, 'search'):
            print("[OK] search() method exists")
        else:
            print("[ERROR] search() method does NOT exist")

        if hasattr(client, 'search_points'):
            print("[OK] search_points() method exists (older API)")
        else:
            print("[ERROR] search_points() method does NOT exist")

        # Test collection exists
        collection_name = settings.qdrant_collection_name
        try:
            collection_info = client.get_collection(collection_name)
            print(f"[OK] Collection '{collection_name}' exists")
            print(f"  - Vector size: {collection_info.config.params.vectors.size}")
            print(f"  - Distance: {collection_info.config.params.vectors.distance}")
        except Exception as e:
            print(f"[ERROR] Collection '{collection_name}' does not exist or error occurred: {e}")

        # Try a simple search operation with dummy data
        try:
            # Create a dummy collection for testing if needed
            test_collection = "test_search_functionality"
            try:
                client.get_collection(test_collection)
            except:
                print(f"Creating test collection: {test_collection}")
                client.create_collection(
                    collection_name=test_collection,
                    vectors_config=models.VectorParams(
                        size=4,  # Small vector for testing
                        distance=models.Distance.COSINE
                    )
                )

            # Insert a dummy point
            client.upsert(
                collection_name=test_collection,
                points=[
                    models.PointStruct(
                        id=1,
                        vector=[0.1, 0.2, 0.3, 0.4],
                        payload={"test": "data"}
                    )
                ]
            )

            # Try search
            if hasattr(client, 'search'):
                search_results = client.search(
                    collection_name=test_collection,
                    query_vector=[0.1, 0.2, 0.3, 0.4],
                    limit=1
                )
                print(f"[OK] search() method works - found {len(search_results)} results")
            else:
                print("[ERROR] search() method not available")

            if hasattr(client, 'search_points'):
                search_results = client.search_points(
                    collection_name=test_collection,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    limit=1
                )
                print(f"[OK] search_points() method works - found {len(search_results)} results")
            else:
                print("[ERROR] search_points() method not available")

            # Clean up test collection
            client.delete_collection(test_collection)

        except Exception as e:
            print(f"✗ Search test failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Failed to initialize Qdrant client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qdrant_client()