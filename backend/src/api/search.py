from fastapi import APIRouter, HTTPException, Request, Query
from typing import Optional, List, Dict, Any
import logging
import time
import datetime

from src.services.vector_db_service import vector_db_service
from src.services.embedding_service import embedding_service
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search")
async def semantic_search_endpoint(
    request: Request,
    query: str = Query(..., description="The search query", min_length=1, max_length=2000),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity score")
):
    """
    Perform semantic search against stored book content using Qdrant Cloud
    """
    try:
        # Validate query length
        if len(query) > 2000:  # Max query length
            raise HTTPException(status_code=400, detail="Query too long. Maximum 2000 characters allowed.")

        # Validate parameters
        if top_k < 1 or top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

        if score_threshold < 0.0 or score_threshold > 1.0:
            raise HTTPException(status_code=400, detail="score_threshold must be between 0.0 and 1.0")

        # Generate embedding for the query
        start_time = time.time()

        query_embedding = await embedding_service.embed_single_text(query)

        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for query")

        # Perform semantic search
        search_results = vector_db_service.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=score_threshold
        )

        search_time_ms = (time.time() - start_time) * 1000

        # Format response
        response = {
            "query": query,
            "results": search_results,
            "search_time_ms": round(search_time_ms, 2),
            "timestamp": datetime.datetime.now().isoformat()
        }

        logger.info(f"Semantic search completed for query: {query[:50]}... Found {len(search_results)} results in {search_time_ms:.2f}ms")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error performing semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during search")


@router.get("/search/health")
async def search_health_check():
    """
    Check the health of the search service and Qdrant Cloud connection
    """
    try:
        # Test Qdrant connection by attempting a minimal search
        test_embedding = [0.0] * 768  # Use standard Gemini embedding size (768 dimensions)

        # Try to perform a search with the dummy embedding
        # This will test if the Qdrant connection is working
        test_results = vector_db_service.search_similar(
            query_embedding=test_embedding,
            top_k=1
        )

        # If we get here without exception, the search service is working
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collection_exists": vector_db_service.collection_name is not None,
            "last_heartbeat": datetime.datetime.now().isoformat(),
            "collection_name": vector_db_service.collection_name
        }
    except Exception as e:
        logger.error(f"Search health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "qdrant_connected": False,
            "message": str(e)
        }