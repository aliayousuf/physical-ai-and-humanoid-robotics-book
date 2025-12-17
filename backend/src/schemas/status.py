from pydantic import BaseModel
from typing import Dict, Optional, Any


class ServiceStatus(BaseModel):
    """Response schema for individual service status."""
    status: str  # Either "healthy", "degraded", or "unavailable"
    last_checked: str  # ISO 8601 timestamp
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class StatusResponse(BaseModel):
    """Response schema for system status."""
    overall_status: str  # Either "healthy", "degraded", or "unavailable"
    services: Dict[str, ServiceStatus]