from fastapi import APIRouter, Request, HTTPException
from typing import Dict
import logging

from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check(request: Request) -> Dict[str, str]:
    """
    Health check endpoint to verify the service is running
    """
    try:
        return {"status": "healthy", "message": "Book Content Ingestion API is running"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/ready")
async def readiness_check(request: Request) -> Dict[str, str]:
    """
    Readiness check endpoint to verify the service is ready to handle requests
    """
    try:
        # Check if required configurations are present
        missing_configs = []

        if not settings.gemini_api_key:
            missing_configs.append("GEMINI_API_KEY")
        if not settings.qdrant_url:
            missing_configs.append("QDRANT_URL")

        if missing_configs:
            return {
                "status": "not_ready",
                "message": f"Missing required configurations: {', '.join(missing_configs)}"
            }

        return {"status": "ready", "message": "Service is ready to handle requests"}
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Readiness check failed")