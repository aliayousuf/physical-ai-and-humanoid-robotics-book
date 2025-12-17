#!/usr/bin/env python3
"""
Test script to check Qdrant vector database status and content
"""
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config.settings import settings

async def test_qdrant_connection():
    """Test connection to Qdrant and check collection status"""
    print("Testing Qdrant connection...")

    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False
        )

        print(f"Connected to Qdrant at {settings.qdrant_url}")

        # List all collections
        collections = client.get_collections()
        print(f"\nAvailable collections: {[col.name for col in collections.collections]}")

        # Check the book_content collection specifically
        collection_name = "book_content"
        try:
            collection_info = client.get_collection(collection_name)
            print(f"\nCollection '{collection_name}' exists:")
            print(f"  - Points count: {collection_info.points_count}")
            print(f"  - Config: {collection_info.config}")

            # Try to get some points to see what content exists
            if collection_info.points_count > 0:
                print(f"\nSample points from collection:")
                points = client.scroll(
                    collection_name=collection_name,
                    limit=3,
                    with_payload=True
                )

                for point in points[0]:
                    print(f"  - ID: {point.id}")
                    print(f"    Title: {point.payload.get('title', 'N/A')}")
                    print(f"    Page Reference: {point.payload.get('page_reference', 'N/A')}")
                    print(f"    Content snippet: {point.payload.get('content', '')[:100]}...")
                    print()
            else:
                print(f"\nCollection '{collection_name}' is empty.")

        except Exception as e:
            print(f"\nCollection '{collection_name}' does not exist: {e}")

        return True

    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_qdrant_connection())