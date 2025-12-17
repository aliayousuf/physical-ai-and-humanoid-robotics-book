from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class UserSelectionContextCreate(BaseModel):
    """Request schema for creating user selection context."""
    session_id: str
    selected_text: str
    page_context: str


class UserSelectionContextResponse(BaseModel):
    """Response schema for user selection context."""
    selection_id: str
    session_id: str
    selected_text: str
    page_context: str
    created_at: datetime
    expires_at: datetime