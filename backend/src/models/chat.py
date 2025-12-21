from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class SourceMetadata(BaseModel):
    """
    Metadata for a source document
    """
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None

class Source(BaseModel):
    """
    A source document that contributed to the response
    """
    filename: str
    content: str
    similarity_score: float
    metadata: SourceMetadata

class ChatResponse(BaseModel):
    """
    Response from the chatbot
    """
    response: str
    sources: List[Source]
    confidence: float

class ChatQuery(BaseModel):
    """
    Query to the chatbot
    """
    query: str
    max_results: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.3