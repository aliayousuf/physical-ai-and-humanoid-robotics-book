from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from enum import Enum

class IngestionJobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"

class IngestionJob(BaseModel):
    """
    Represents a job for processing documents from the docs folder
    """
    id: str
    status: IngestionJobStatus
    total_documents: int  # number of documents to process
    processed_documents: int  # number of documents processed
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        use_enum_values = True