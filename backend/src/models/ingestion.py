from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum

class IngestionJobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"

class IngestionRequest(BaseModel):
    """
    Request to trigger content ingestion
    """
    force_reprocess: bool = False
    file_patterns: List[str] = ["*.md", "*.pdf", "*.txt"]

class IngestionJobResponse(BaseModel):
    """
    Response with ingestion job status
    """
    job_id: str
    status: IngestionJobStatus
    total_documents: int
    processed_documents: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None