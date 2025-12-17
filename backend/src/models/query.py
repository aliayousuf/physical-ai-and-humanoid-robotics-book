from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pydantic import BaseModel
from .session import QueryMode


class QueryHistory(BaseModel):
    id: str = str(uuid4())
    session_id: str
    query_text: str
    response_text: str
    query_mode: QueryMode
    selected_text: Optional[str] = None  # Text that was selected (if applicable)
    response_sources: List[str] = []  # IDs of content used in response
    response_time_ms: int = 0
    timestamp: datetime = datetime.now()
    is_successful: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class UserSelectionContext(BaseModel):
    id: str = str(uuid4())
    session_id: str
    selected_text: str
    page_url: str
    section_context: str = ""  # Surrounding text for context
    created_at: datetime = datetime.now()
    expires_at: datetime