from datetime import datetime
from typing import Dict, Any
from uuid import uuid4
from pydantic import BaseModel


class BookContent(BaseModel):
    id: str = str(uuid4())
    title: str
    content: str
    page_reference: str  # Path to the Docusaurus page
    section: str = ""  # Section within the page
    hash: str = ""  # Hash of content for change detection
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    embedding_id: str = ""  # Reference to vector in Qdrant
    metadata: Dict[str, Any] = {}  # Additional content metadata