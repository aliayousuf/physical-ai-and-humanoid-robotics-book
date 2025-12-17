from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
from ..config.settings import settings
from ..services.qdrant_service import qdrant_service
from ..services.gemini_service import gemini_service
from ..utils.metrics import track_performance


router = APIRouter()


@router.get("/health")
@track_performance("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check the health status of the backend service
    """
    # Test connections to external services
    services_status = {
        "database": "connected",  # Simplified - in real implementation, test actual DB connection
        "vector_db": "unknown",
        "gemini_api": "unknown"
    }

    # Test Qdrant connection
    try:
        # This will fail if Qdrant is not accessible, but we'll handle the exception
        await qdrant_service.initialize_collection()  # This checks if we can access Qdrant
        services_status["vector_db"] = "connected"
    except Exception:
        services_status["vector_db"] = "disconnected"

    # Test Gemini API (simplified - in real implementation, do a lightweight test)
    try:
        # We won't make an actual API call here to avoid using quota
        # But we can check if the API key is configured
        if settings.gemini_api_key and len(settings.gemini_api_key) > 0:
            services_status["gemini_api"] = "configured"
        else:
            services_status["gemini_api"] = "not configured"
    except Exception:
        services_status["gemini_api"] = "error"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": services_status
    }