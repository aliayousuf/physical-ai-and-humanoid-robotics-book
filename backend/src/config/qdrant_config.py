import os
from typing import Optional
from qdrant_client import QdrantClient


class QdrantConfig:
    """Configuration class for Qdrant Cloud client"""

    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content")

    # Vector configuration
    VECTOR_SIZE: int = 768  # Default size for Gemini embeddings
    DISTANCE_METRIC: str = "Cosine"  # Cosine similarity for text embeddings

    @classmethod
    def get_client(cls) -> Optional[QdrantClient]:
        """Create and return Qdrant client instance"""
        if not cls.QDRANT_URL or not cls.QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")

        try:
            client = QdrantClient(
                url=cls.QDRANT_URL,
                api_key=cls.QDRANT_API_KEY,
                prefer_grpc=True  # Use gRPC for better performance when available
            )
            return client
        except Exception as e:
            print(f"Error creating Qdrant client: {e}")
            return None

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        return bool(cls.QDRANT_URL and cls.QDRANT_API_KEY)