from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List

class VectorEmbedding(BaseModel):
    """
    Represents the vector embedding stored in the vector database
    """
    id: str  # unique identifier, usually matches DocumentChunk.id
    vector: List[float]  # numerical vector representation from Gemini embeddings
    collection_name: str  # name of the collection in vector DB
    metadata: Optional[Dict[str, Any]] = None  # document metadata for filtering
    created_at: datetime
    model_used: str  # the model used to generate the embedding, e.g., "embedding-001" for Gemini