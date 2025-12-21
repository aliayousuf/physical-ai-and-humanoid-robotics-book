from fastapi import APIRouter, HTTPException, Request
from typing import List
import logging

from ..services.ingestion_service import ingestion_service
from ..models.ingestion import IngestionRequest, IngestionJobResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ingestion/trigger", response_model=IngestionJobResponse)
async def trigger_ingestion_endpoint(
    request: Request,
    ingestion_request: IngestionRequest
):
    """
    Trigger content ingestion from docs folder
    """
    try:
        # Validate request data
        if not ingestion_request.file_patterns:
            ingestion_request.file_patterns = ["*.md", "*.pdf", "*.txt"]

        # Trigger the ingestion process
        job_id = ingestion_service.trigger_ingestion(
            force_reprocess=ingestion_request.force_reprocess,
            file_patterns=ingestion_request.file_patterns
        )

        # Get the job status
        job = ingestion_service.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=500, detail="Failed to create ingestion job")

        # Create response
        response = IngestionJobResponse(
            job_id=job.id,
            status=job.status,
            total_documents=job.total_documents,
            processed_documents=job.processed_documents,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message
        )

        logger.info(f"Ingestion job {job_id} started successfully with {job.total_documents} documents")
        return response
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error starting ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during ingestion")

@router.get("/ingestion/status/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_status_endpoint(
    request: Request,
    job_id: str
):
    """
    Get the status of an ingestion job
    """
    try:
        job = ingestion_service.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ingestion job not found")

        # Create response
        response = IngestionJobResponse(
            job_id=job.id,
            status=job.status,
            total_documents=job.total_documents,
            processed_documents=job.processed_documents,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message
        )

        logger.info(f"Retrieved status for ingestion job {job_id}")
        return response
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error getting ingestion status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving status")

@router.get("/ingestion/documents")
async def get_processed_documents(
    request: Request
):
    """
    Get list of processed documents
    """
    try:
        documents = ingestion_service.get_processed_documents()
        logger.info(f"Retrieved {len(documents)} processed documents")
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error getting processed documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving documents")