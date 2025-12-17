"""
Custom error handling utilities for the RAG Chatbot system
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from pydantic import BaseModel
import logging


class RAGException(Exception):
    """
    Base exception class for RAG-related errors
    """
    def __init__(self, message: str, error_code: str = "RAG_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        return f"{self.error_code}: {self.message}"


class EmbeddingException(RAGException):
    """
    Exception for embedding-related errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EMBEDDING_ERROR", details)


class VectorDBException(RAGException):
    """
    Exception for vector database-related errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VECTOR_DB_ERROR", details)


class QueryException(RAGException):
    """
    Exception for query-related errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "QUERY_ERROR", details)


class SessionException(RAGException):
    """
    Exception for session-related errors
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SESSION_ERROR", details)


class APIResponse(BaseModel):
    """
    Standard API response format for errors
    """
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None


def handle_rag_exception(exc: RAGException, logger: logging.Logger = None) -> HTTPException:
    """
    Convert RAG exceptions to HTTP exceptions with appropriate status codes.

    Args:
        exc: The RAG exception to handle
        logger: Optional logger to log the error

    Returns:
        HTTPException with appropriate status code
    """
    if logger:
        logger.error(f"{exc.error_code}: {exc.message}", exc_info=True)

    # Map exception types to HTTP status codes
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, (EmbeddingException, VectorDBException)):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, QueryException):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, SessionException):
        status_code = status.HTTP_404_NOT_FOUND

    return HTTPException(
        status_code=status_code,
        detail={
            "error": exc.message,
            "code": exc.error_code,
            "details": exc.details
        }
    )


def handle_general_exception(exc: Exception, logger: logging.Logger = None) -> HTTPException:
    """
    Handle general exceptions and convert to HTTP exceptions.

    Args:
        exc: The exception to handle
        logger: Optional logger to log the error

    Returns:
        HTTPException with appropriate status code
    """
    if logger:
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR",
            "details": {"type": type(exc).__name__}
        }
    )


def validate_api_key(api_key: Optional[str], key_type: str = "Gemini") -> None:
    """
    Validate that an API key is provided and not empty.

    Args:
        api_key: The API key to validate
        key_type: Type of API key (for error message)

    Raises:
        RAGException if the API key is invalid
    """
    if not api_key or api_key.strip() == "" or api_key == "your_google_gemini_api_key_here":
        raise RAGException(
            f"Valid {key_type} API key is required",
            "MISSING_API_KEY",
            {"key_type": key_type}
        )


def validate_query_text(query: str) -> None:
    """
    Validate query text for length and content.

    Args:
        query: The query text to validate

    Raises:
        QueryException if the query is invalid
    """
    if not query or query.strip() == "":
        raise QueryException("Query cannot be empty", {"field": "query"})

    if len(query.strip()) > 2000:  # Adjust based on your requirements
        raise QueryException(
            "Query is too long",
            {"field": "query", "max_length": 2000, "actual_length": len(query)}
        )


def validate_session_id(session_id: str) -> None:
    """
    Validate session ID format.

    Args:
        session_id: The session ID to validate

    Raises:
        SessionException if the session ID is invalid
    """
    if not session_id or len(session_id.strip()) == 0:
        raise SessionException("Session ID is required", {"field": "session_id"})

    # Basic UUID format check (simplified)
    if len(session_id) < 32:
        raise SessionException(
            "Invalid session ID format",
            {"field": "session_id", "reason": "too_short"}
        )