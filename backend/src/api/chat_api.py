from fastapi import APIRouter, HTTPException, Query, Request
from typing import Optional
import logging
from fastapi.responses import StreamingResponse
import json
import datetime
import uuid

from src.services.chat_service import chat_service
from src.models.chat import ChatQuery, ChatResponse
from src.models.error import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat/query", response_model=ChatResponse)
async def query_chat_endpoint(
    request: Request,
    query_data: ChatQuery
):
    """
    Query the chatbot with book content using RAG approach
    """
    try:
        # Validate query length
        if len(query_data.query) > 2000:  # Max query length from settings
            raise HTTPException(status_code=400, detail="Query too long. Maximum 2000 characters allowed.")

        # Call the chat service
        result = await chat_service.query_chat(
            query=query_data.query,
            max_results=query_data.max_results or 5,
            similarity_threshold=query_data.similarity_threshold or 0.3
        )

        # Create response
        response = ChatResponse(
            response=result["response"],
            sources=result["sources"],
            confidence=result["confidence"]
        )

        logger.info(f"Chat query processed successfully: {query_data.query[:50]}...")
        return response
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")

@router.post("/chat/query_simple")
async def query_chat_simple(
    request: Request,
    query: str = Query(..., description="The user's question/query", max_length=2000),
    max_results: Optional[int] = Query(5, ge=1, le=20, description="Maximum number of results to return"),
    similarity_threshold: Optional[float] = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity threshold for results")
):
    """
    Simple query endpoint for the chatbot with book content
    """
    try:
        # Validate query length
        if len(query) > 2000:
            raise HTTPException(status_code=400, detail="Query too long. Maximum 2000 characters allowed.")

        # Call the chat service
        result = await chat_service.query_chat(
            query=query,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )

        logger.info(f"Simple chat query processed successfully: {query[:50]}...")
        return result
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error processing simple chat query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")


async def generate_streaming_response(query: str, max_results: int = 5, similarity_threshold: float = 0.3):
    """
    Generator function to create streaming response for chat
    """
    try:
        # Yield the retrieval start event
        yield f"data: {json.dumps({'type': 'retrieval_start', 'timestamp': datetime.datetime.now().isoformat()})}\n\n"

        # Get relevant content first
        relevant_content = await chat_service.get_relevant_content(
            query=query,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )

        # Yield the retrieval complete event
        sources = [item.get("metadata", {}).get("filename", "Unknown") for item in relevant_content]
        yield f"data: {json.dumps({'type': 'retrieval_complete', 'retrieved_chunks': len(relevant_content), 'sources': sources, 'timestamp': datetime.datetime.now().isoformat()})}\n\n"

        # Check if any relevant content was found
        if not relevant_content:
            # Yield the complete response with "Not found in the book" message
            yield f"data: {json.dumps({'type': 'chunk', 'content': 'Not found in the book.', 'timestamp': datetime.datetime.now().isoformat()})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'request_id': str(uuid.uuid4()), 'timestamp': datetime.datetime.now().isoformat()})}\n\n"
            return

        # For streaming with Gemini, we need to get the full response first and then simulate streaming
        # In a real implementation, we'd use Gemini's streaming capabilities
        result = await chat_service.query_chat(
            query=query,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )

        # Simulate streaming by breaking the response into chunks
        response_text = result["response"]
        chunk_size = 20  # characters per chunk for simulation

        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'timestamp': datetime.datetime.now().isoformat()})}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'type': 'complete', 'request_id': str(uuid.uuid4()), 'timestamp': datetime.datetime.now().isoformat()})}\n\n"

    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Error processing streaming response', 'timestamp': datetime.datetime.now().isoformat()})}\n\n"


@router.post("/chat/stream")
async def stream_chat_endpoint(
    request: Request,
    query: str = Query(..., description="The user's question/query", max_length=2000),
    max_results: Optional[int] = Query(5, ge=1, le=20, description="Maximum number of results to return"),
    similarity_threshold: Optional[float] = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity threshold for results")
):
    """
    Stream the chat response with retrieval and generation events
    """
    try:
        # Validate query length
        if len(query) > 2000:
            raise HTTPException(status_code=400, detail="Query too long. Maximum 2000 characters allowed.")

        return StreamingResponse(
            generate_streaming_response(
                query=query,
                max_results=max_results or 5,
                similarity_threshold=similarity_threshold or 0.3
            ),
            media_type="text/plain"
        )
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error processing streaming chat query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during streaming query processing")