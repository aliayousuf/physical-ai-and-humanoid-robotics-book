from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class Message(BaseModel):
    """A single message in a conversation history."""
    id: str
    role: str  # Either "user" or "assistant"
    content: str
    timestamp: datetime
    references: Optional[List[str]] = None  # Book section references in response


class UserSessionCreate(BaseModel):
    """Request schema for creating a new session."""
    pass  # No specific fields needed for session creation


class UserSessionResponse(BaseModel):
    """Response schema for a created session."""
    session_id: str
    created_at: datetime


class SessionDetailsResponse(BaseModel):
    """Response schema for session details."""
    session_id: str
    created_at: datetime
    last_interaction: Optional[datetime] = None
    history: List[Message] = []
    metadata: Dict[str, Any] = {}