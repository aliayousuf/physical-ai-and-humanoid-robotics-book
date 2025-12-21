from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from enum import Enum

class DocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class Document(BaseModel):
    """
    Represents a document from the docs folder that needs to be ingested
    """
    id: str
    filename: str
    filepath: str
    format: str  # file format: markdown, pdf, txt
    size: int  # file size in bytes
    created_at: datetime
    updated_at: datetime
    status: DocumentStatus
    checksum: Optional[str] = None  # to detect changes

    class Config:
        use_enum_values = True