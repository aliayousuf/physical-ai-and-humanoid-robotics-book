#!/usr/bin/env python3
"""
Test script to verify the updated Qdrant client search functionality
"""
from src.services.vector_db_service import vector_db_service

def test_updated_search():
    print("Testing updated Qdrant search functionality...")

    try:
        # Test that the vector_db_service is initialized
        if vector_db_service.client:
            print(f"[OK] Qdrant client initialized: {type(vector_db_service.client)}")

            # Check if query method exists (new API)
            if hasattr(vector_db_service.client, 'query'):
                print("[OK] query() method exists (new API)")
            else:
                print("[INFO] query() method does not exist")

            # Check if search method exists (old API)
            if hasattr(vector_db_service.client, 'search'):
                print("[INFO] search() method exists (old API)")
            else:
                print("[INFO] search() method does not exist")

            # Check if search_points method exists (older API)
            if hasattr(vector_db_service.client, 'search_points'):
                print("[INFO] search_points() method exists (older API)")
            else:
                print("[INFO] search_points() method does not exist")
        else:
            print("[ERROR] Qdrant client not initialized")
            return

        # Try a minimal search with dummy data to test the search_similar method
        dummy_embedding = [0.1] * 768  # Create a dummy embedding with 768 dimensions
        results = vector_db_service.search_similar(
            query_embedding=dummy_embedding,
            top_k=1,
            threshold=0.0  # Very low threshold to get results even with dummy data
        )

        print(f"[OK] search_similar method executed successfully - found {len(results)} results")
        if results:
            print(f"Sample result keys: {list(results[0].keys()) if results else 'No results'}")

    except Exception as e:
        print(f"[ERROR] Error during search test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_search()