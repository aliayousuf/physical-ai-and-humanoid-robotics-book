from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class DocumentChunk(BaseModel):
    """
    Represents a chunk of text from a document that has been processed into embeddings
    """
    id: str
    document_id: str  # foreign key to Document
    chunk_index: int  # order of chunk in document
    content: str  # text content of the chunk
    embedding: Optional[list] = None  # vector embedding of the content
    metadata: Optional[Dict[str, Any]] = None  # additional metadata like page number, section
    created_at: datetime