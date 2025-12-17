from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request schema for chat queries."""
    query: str
    session_id: Optional[str] = None
    mode: str = "general"  # Either "general" or "selected_text"
    selected_text: Optional[str] = None


class QueryResponse(BaseModel):
    """Response schema for chat queries."""
    response: str
    session_id: str
    references: List[str]
    context: Optional[Dict[str, Any]] = None


class QueryHistoryCreate(BaseModel):
    """Request schema for creating query history."""
    session_id: str
    query_text: str
    response_text: str
    query_mode: str
    selected_text_context: Optional[str] = None
    source_references: List[str] = []
    response_tokens: Optional[int] = None
    processing_time_ms: Optional[int] = None


class QueryHistoryResponse(BaseModel):
    """Response schema for query history."""
    query_id: str
    session_id: str
    query_text: str
    response_text: str
    timestamp: datetime
    source_references: List[str] = []
    query_mode: str
    selected_text_context: Optional[str] = None
    response_tokens: Optional[int] = None
    processing_time_ms: Optional[int] = None


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str
    code: int
    details: Optional[Dict[str, Any]] = None