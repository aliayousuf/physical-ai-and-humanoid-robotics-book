from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class BookContentCreate(BaseModel):
    """Request schema for creating book content."""
    title: str
    content_text: str
    section_ref: str
    page_reference: Optional[str] = None


class BookContentResponse(BaseModel):
    """Response schema for book content."""
    content_id: str
    title: str
    section_ref: str
    content_text: str
    page_reference: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class ContentResult(BaseModel):
    """Response schema for content search results."""
    content_id: str
    title: str
    content_text: str
    section_ref: str
    relevance_score: float