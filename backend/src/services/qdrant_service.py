import asyncio
from typing import List, Dict, Any, Optional
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..config.settings import settings
from ..utils.monitoring import check_service_usage, ServiceType


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False  # Using HTTP for simplicity
        )
        self.collection_name = "book_content"

    async def initialize_collection(self):
        """
        Initialize the Qdrant collection for book content
        """
        try:
            # Check usage limits before making API call
            if not check_service_usage(ServiceType.QDRANT):
                raise Exception("Qdrant API usage limit exceeded. Please try again later.")

            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size for Gemini embeddings
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,  # Using 768 dimensions to match Gemini embedding dimensions
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                # Check the existing collection's configuration and recreate if dimensions don't match
                collection_info = self.client.get_collection(self.collection_name)
                if collection_info.config.params.vectors.size != 768:
                    print(f"Collection dimension mismatch. Expected 768, got {collection_info.config.params.vectors.size}. Recreating collection...")
                    self.client.delete_collection(self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=768,  # Using 768 dimensions to match Gemini embedding dimensions
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"Recreated Qdrant collection: {self.collection_name} with correct dimensions")
                else:
                    print(f"Qdrant collection {self.collection_name} already exists with correct dimensions")
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")
            raise

    async def store_embedding(self, content_id: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Store an embedding in Qdrant
        """
        try:
            # Check usage limits before making API call
            if not check_service_usage(ServiceType.QDRANT):
                raise Exception("Qdrant API usage limit exceeded. Please try again later.")

            point_id = str(uuid4())

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "content_id": content_id,
                            **metadata
                        }
                    )
                ]
            )

            return point_id
        except Exception as e:
            print(f"Error storing embedding in Qdrant: {e}")
            raise

    async def search_similar(self, query_embedding: List[float], top_k: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar content based on embedding
        """
        try:
            # Check usage limits before making API call
            if not check_service_usage(ServiceType.QDRANT):
                raise Exception("Qdrant API usage limit exceeded. Please try again later.")

            # Handle different Qdrant client versions
            if hasattr(self.client, 'search'):
                # Modern Qdrant client API
                # First try with score threshold
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold
                )
                # If no results with threshold, try without threshold to debug
                if not results:
                    results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=top_k
                    )
            elif hasattr(self.client, 'search_points'):
                # Older Qdrant client API
                results = self.client.search_points(
                    collection_name=self.collection_name,
                    vector=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold
                )
                # If no results with threshold, try without threshold to debug
                if not results:
                    results = self.client.search_points(
                        collection_name=self.collection_name,
                        vector=query_embedding,
                        limit=top_k
                    )
            else:
                # Fallback: return empty results instead of failing completely
                print("Warning: Qdrant client doesn't have expected search method. Returning empty results.")
                return []

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content_id": result.payload.get("content_id") if hasattr(result, 'payload') else result.get('payload', {}).get('content_id'),
                    "score": getattr(result, 'score', 0) if hasattr(result, 'score') else result.get('score', 0),
                    "payload": getattr(result, 'payload', {}) if hasattr(result, 'payload') else result.get('payload', {})
                })

            return formatted_results
        except Exception as e:
            print(f"Error searching in Qdrant: {e}")
            # Return empty results as fallback to allow the system to continue working
            return []

    async def delete_embedding(self, point_id: str):
        """
        Delete an embedding from Qdrant
        """
        try:
            # Check usage limits before making API call
            if not check_service_usage(ServiceType.QDRANT):
                raise Exception("Qdrant API usage limit exceeded. Please try again later.")

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
        except Exception as e:
            print(f"Error deleting embedding from Qdrant: {e}")
            raise

    async def update_embedding(self, point_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """
        Update an existing embedding in Qdrant
        """
        try:
            # Check usage limits before making API call
            if not check_service_usage(ServiceType.QDRANT):
                raise Exception("Qdrant API usage limit exceeded. Please try again later.")

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )
        except Exception as e:
            print(f"Error updating embedding in Qdrant: {e}")
            raise


# Global instance
qdrant_service = QdrantService()