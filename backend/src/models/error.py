from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ErrorResponse(BaseModel):
    """
    Standard error response model for consistent error handling
    """
    error: str
    message: str
    timestamp: datetime = datetime.now()
    details: Optional[Dict[str, Any]] = None

class ValidationErrorResponse(BaseModel):
    """
    Error response for validation errors
    """
    error: str = "validation_error"
    message: str
    timestamp: datetime = datetime.now()
    validation_errors: List[Dict[str, Any]]