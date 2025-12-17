from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pydantic import BaseModel
from enum import Enum


class QueryMode(str, Enum):
    GENERAL = "general"
    SELECTED_TEXT = "selected_text"


class Message(BaseModel):
    id: str
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: List[Dict[str, Any]] = []  # Citations for assistant responses
    metadata: Dict[str, Any] = {}


class UserSelectionContext(BaseModel):
    id: str
    session_id: str
    selected_text: str
    page_url: str
    section_context: str  # Surrounding text for context
    created_at: datetime
    expires_at: datetime


class UserSession(BaseModel):
    id: str = str(uuid4())
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    expires_at: datetime
    conversation_history: List[Message] = []
    current_mode: QueryMode = QueryMode.GENERAL
    selected_text_context: Optional[UserSelectionContext] = None
    metadata: Dict[str, Any] = {}  # Additional session data