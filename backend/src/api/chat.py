from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import logging
from ..services.rag_service import rag_service
from ..utils.validation import sanitize_input, validate_query_length, is_malformed_query
from ..utils.monitoring import track_performance

logger = logging.getLogger(__name__)


router = APIRouter()


class CreateSessionRequest(BaseModel):
    initial_context: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime


@router.post("/session", response_model=CreateSessionResponse)
@track_performance("/chat/session")
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """
    Create a new chat session
    """
    try:
        session = await rag_service.create_session(request.initial_context or "")
        return CreateSessionResponse(
            session_id=session.id,
            created_at=session.created_at,
            expires_at=session.expires_at
        )
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        from ..utils.logging import logger
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create chat session")


class GetSessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_interaction: Optional[datetime] = None
    history: list
    current_mode: str
    metadata: Dict[str, Any] = {}


@router.get("/session/{session_id}", response_model=GetSessionResponse)
@track_performance("/chat/session/{session_id}")
async def get_session(session_id: str) -> GetSessionResponse:
    """
    Retrieve details about a specific user session
    """
    try:
        session = await rag_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail={
                "error": {
                    "code": "SESSION_NOT_FOUND",
                    "message": "Session not found or expired",
                    "details": "Session ID is invalid or has expired"
                }
            })

        # Convert session to response format
        return GetSessionResponse(
            session_id=session.id,
            created_at=session.created_at,
            last_interaction=session.updated_at,
            history=[{
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "sources": getattr(msg, 'sources', [])
            } for msg in session.conversation_history],
            current_mode=session.current_mode.value if hasattr(session.current_mode, 'value') else str(session.current_mode),
            metadata=session.metadata
        )
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        from ..utils.logging import logger
        logger.error(f"Error retrieving session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


class QueryRequest(BaseModel):
    session_id: str
    query: str
    mode: str = "general"  # "general" or "selected_text"
    selected_text: Optional[str] = None  # Only used in selected_text mode
    context: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    response_id: str
    session_id: str
    query: str
    response: str
    sources: list
    timestamp: datetime
    query_mode: str


@router.post("/query", response_model=QueryResponse)
@track_performance("/chat/query")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Submit a query about book content using RAG - supports both general and selected text modes
    """
    # Sanitize and validate input
    sanitized_query = sanitize_input(request.query)

    length_error = validate_query_length(sanitized_query)
    if length_error:
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Query validation failed",
                "details": length_error
            }
        })

    if is_malformed_query(sanitized_query):
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "MALFORMED_QUERY",
                "message": "Query appears to be malformed or potentially malicious",
                "details": "Please submit a valid query"
            }
        })

    try:
        page_url = request.context.get("page_url") if request.context else None
        section_context = request.context.get("section_context") if request.context else None

        if request.mode == "selected_text":
            # Process with selected text context
            sanitized_selected_text = sanitize_input(request.selected_text) if request.selected_text else ""

            result = await rag_service.process_selected_text_query(
                session_id=request.session_id,
                query=sanitized_query,
                selected_text=sanitized_selected_text,
                page_url=page_url,
                section_context=section_context
            )
        else:
            # Process as general query
            result = await rag_service.process_general_query(
                session_id=request.session_id,
                query=sanitized_query,
                page_url=page_url
            )

        return QueryResponse(**result)
    except ValueError as e:
        if "Invalid or expired session ID" in str(e):
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "INVALID_SESSION",
                    "message": "Session not found or expired",
                    "details": "Session ID is invalid or has expired"
                }
            })
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        from ..utils.logging import logger
        logger.error(f"Error processing query in {request.mode} mode: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": {
                "code": "RAG_PROCESSING_ERROR",
                "message": f"Error processing RAG query in {request.mode} mode",
                "details": str(e)
            }
        })




class GetHistoryResponse(BaseModel):
    session_id: str
    messages: list
    pagination: Dict[str, int]


@router.get("/session/{session_id}/history", response_model=GetHistoryResponse)
@track_performance("/chat/session/history")
async def get_conversation_history(
    session_id: str,
    limit: int = 20,
    offset: int = 0
) -> GetHistoryResponse:
    """
    Retrieve conversation history for a session
    """
    try:
        result = await rag_service.get_conversation_history(session_id, limit, offset)
        return GetHistoryResponse(**result)
    except ValueError as e:
        if "Invalid or expired session ID" in str(e):
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "INVALID_SESSION",
                    "message": "Session not found or expired",
                    "details": "Session ID is invalid or has expired"
                }
            })
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        from ..utils.logging import logger
        logger.error(f"Error retrieving conversation history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")


class ClearContextResponse(BaseModel):
    session_id: str
    message: str
    new_mode: str


@router.delete("/session/{session_id}/context", response_model=ClearContextResponse)
@track_performance("/chat/session/context")
async def clear_session_context(session_id: str) -> ClearContextResponse:
    """
    Clear the selected text context and return to general mode
    """
    try:
        result = await rag_service.clear_selected_text_context(session_id)
        return ClearContextResponse(**result)
    except ValueError as e:
        if "Invalid or expired session ID" in str(e):
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "INVALID_SESSION",
                    "message": "Session not found or expired",
                    "details": "Session ID is invalid or has expired"
                }
            })
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        from ..utils.logging import logger
        logger.error(f"Error clearing session context for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear session context")


class ContentSummaryResponse(BaseModel):
    total_pages: int
    total_content_segments: int
    total_tokens: int
    last_indexed: datetime
    content_coverage: Dict[str, int]


@router.get("/content/summary", response_model=ContentSummaryResponse)
@track_performance("/chat/content/summary")
async def get_content_summary() -> ContentSummaryResponse:
    """
    Get a summary of indexed book content
    """
    # This is a simplified implementation - in a real implementation,
    # this would calculate actual statistics from the vector database
    return ContentSummaryResponse(
        total_pages=150,
        total_content_segments=1250,
        total_tokens=450000,
        last_indexed=datetime.now(),
        content_coverage={
            "introduction": 100,
            "physical_ai": 85,
            "humanoid_robotics": 92,
            "advanced_topics": 78
        }
    )