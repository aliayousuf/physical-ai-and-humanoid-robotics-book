from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from src.config.settings import settings
from src.models.vector_embedding import VectorEmbedding
from src.models.document_chunk import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Service for interacting with the Qdrant vector database
    """
    def __init__(self):
        # Initialize Qdrant client
        try:
            if settings.qdrant_api_key:
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    prefer_grpc=True
                )
            else:
                self.client = QdrantClient(url=settings.qdrant_url)

            self.collection_name = settings.qdrant_collection_name
            self._ensure_collection_exists()
        except Exception as e:
            print(f"Error: Could not initialize Qdrant client: {e}")
            print("Please ensure Qdrant server is running before starting the application.")
            print("For local setup, install Docker and run: docker run -p 6333:6333 qdrant/qdrant")
            print("Alternatively, use Qdrant Cloud: https://cloud.qdrant.io/")
            raise e  # Re-raise the exception to prevent the application from starting with broken vector DB

    def _ensure_collection_exists(self):
        """
        Ensure the collection exists in Qdrant
        """
        if not self.client:
            logger.error("Qdrant client not initialized, cannot ensure collection exists")
            return

        try:
            # Check if collection exists
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")

            # Create index for document_id field if it doesn't exist
            # Check if payload has document_id field and create index if needed
            try:
                # Attempt to create index for document_id field to optimize filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info("Created index for document_id field")
            except Exception as e:
                logger.info(f"Index for document_id field may already exist or error occurred: {e}")

        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Gemini embeddings are typically 768 dimensions
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection {self.collection_name}")

            # Create index for document_id field after collection creation
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info("Created index for document_id field")
            except Exception as e:
                logger.warning(f"Could not create index for document_id field: {e}")

    def store_embedding(self, chunk: DocumentChunk, embedding: List[float]) -> bool:
        """
        Store a document chunk embedding in the vector database
        """
        if not self.client:
            logger.error("Qdrant client not initialized, cannot store embedding")
            return False

        try:
            # Prepare the point to be stored
            point = PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "metadata": chunk.metadata or {},
                    "created_at": chunk.created_at.isoformat()
                }
            )

            # Store the embedding
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Stored embedding for chunk {chunk.id}")
            return True
        except Exception as e:
            logger.error(f"Error storing embedding for chunk {chunk.id}: {str(e)}")
            return False

    def search_similar(self, query_embedding: List[float], top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for similar content in the vector database
        """
        if not self.client:
            logger.error("Qdrant client not initialized, cannot search for similar content")
            return []

        try:
            # Search for similar vectors using the modern query_points method
            if hasattr(self.client, 'query_points'):
                # Modern Qdrant client API (1.9.0+) - uses query_points for vector search
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
                    with_payload=True
                )
                # Convert QueryResponse to list of results
                if hasattr(search_results, 'points'):
                    search_results = search_results.points
            elif hasattr(self.client, 'search'):
                # Older Qdrant client API (for compatibility)
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
                    with_payload=True
                )
            else:
                # Fallback if no expected methods exist
                logger.error("Qdrant client doesn't have expected query_points or search method")
                return []

            results = []
            for result in search_results:
                # The search result should be a ScoredPoint object with id, score, and payload attributes
                if hasattr(result, 'payload') and result.payload:
                    payload = result.payload
                elif isinstance(result, dict) and 'payload' in result:
                    payload = result['payload']
                else:
                    payload = {}

                results.append({
                    "id": getattr(result, 'id', result.get('id') if isinstance(result, dict) else None),
                    "content": payload.get("content", ""),
                    "metadata": payload.get("metadata", {}),
                    "similarity_score": getattr(result, 'score', result.get('score') if isinstance(result, dict) else 0),
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index")
                })

            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error searching for similar content: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def delete_document_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document
        """
        if not self.client:
            logger.error("Qdrant client not initialized, cannot delete document chunks")
            return False

        try:
            # Find all points with the given document_id
            points = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="payload.document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=10000  # Assuming a reasonable limit for document chunks
            )

            if points[0]:  # If there are points to delete
                point_ids = [point.id for point in points[0]]
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                logger.info(f"Deleted {len(point_ids)} chunks for document {document_id}")
                return True

            logger.info(f"No chunks found for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document chunks for {document_id}: {str(e)}")
            return False

    def get_all_document_ids(self) -> List[str]:
        """
        Get all document IDs stored in the vector database
        """
        if not self.client:
            logger.error("Qdrant client not initialized, cannot retrieve document IDs")
            return []

        try:
            # Get all points and extract document IDs
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Assuming a reasonable limit
            )

            document_ids = set()
            for point in points:
                doc_id = point.payload.get("document_id")
                if doc_id:
                    document_ids.add(doc_id)

            logger.info(f"Retrieved {len(document_ids)} unique document IDs")
            return list(document_ids)
        except Exception as e:
            logger.error(f"Error retrieving document IDs: {str(e)}")
            return []

# Create a singleton instance
vector_db_service = VectorDBService()