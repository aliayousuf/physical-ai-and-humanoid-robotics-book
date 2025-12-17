from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4
from pydantic import BaseModel


class Message(BaseModel):
    id: str = str(uuid4())
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()
    sources: List[Dict[str, Any]] = []  # Citations for assistant responses
    metadata: Dict[str, Any] = {}