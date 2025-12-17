"""
Error handling utilities for the RAG Chatbot system
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from ..schemas import ErrorResponse
import traceback
import logging


logger = logging.getLogger(__name__)


class RAGChatbotError(Exception):
    """Base exception class for RAG Chatbot errors"""
    def __init__(self, message: str, error_code: str = "GENERIC_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationError(RAGChatbotError):
    """Exception raised for validation errors"""
    def __init__(self, message: str, field: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details or {})
        self.field = field
        if field:
            self.details["field"] = field


class DatabaseError(RAGChatbotError):
    """Exception raised for database errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details or {})


class ExternalServiceError(RAGChatbotError):
    """Exception raised for external service errors (e.g., OpenAI, Qdrant)"""
    def __init__(self, message: str, service: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", details or {})
        self.service = service
        self.details["service"] = service


class RateLimitError(RAGChatbotError):
    """Exception raised when rate limits are exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details or {})


def handle_error(exception: Exception, context: str = "") -> ErrorResponse:
    """
    Convert an exception to an error response
    """
    error_details = {
        "context": context,
        "exception_type": type(exception).__name__,
        "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else None
    }

    if isinstance(exception, RAGChatbotError):
        # It's already a custom error, just add context
        error_details.update(exception.details)
        return ErrorResponse(
            error=exception.message,
            code=getattr(exception, 'status_code', 500),
            details=error_details
        )
    elif isinstance(exception, HTTPException):
        # It's an HTTP exception from FastAPI
        return ErrorResponse(
            error=exception.detail,
            code=exception.status_code,
            details=error_details
        )
    else:
        # It's an unexpected error
        logger.error(f"Unexpected error in {context}: {str(exception)}", exc_info=True)
        return ErrorResponse(
            error="An unexpected error occurred",
            code=500,
            details=error_details
        )


def create_error_response(error: RAGChatbotError, status_code: int = 500) -> JSONResponse:
    """
    Create a JSON response for an error
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error.message,
            "code": status_code,
            "error_code": error.error_code,
            "details": error.details
        }
    )


def validate_session_id(session_id: str) -> None:
    """
    Validate a session ID
    """
    if not session_id:
        raise ValidationError("Session ID is required", field="session_id")

    # Additional validation can be added here
    if len(session_id) < 10:
        raise ValidationError("Session ID is too short", field="session_id")


def validate_query_text(query: str) -> None:
    """
    Validate query text
    """
    if not query or not query.strip():
        raise ValidationError("Query text is required", field="query")

    if len(query.strip()) > 10000:  # Using the max_query_length from settings
        raise ValidationError("Query text is too long", field="query")


def log_error(exception: Exception, context: str = "") -> None:
    """
    Log an error with context
    """
    logger.error(f"Error in {context}: {str(exception)}", exc_info=True)


def handle_external_service_error(service_name: str, original_error: Exception) -> ExternalServiceError:
    """
    Handle errors from external services and wrap them in a consistent format
    """
    error_msg = f"Error calling {service_name}: {str(original_error)}"
    return ExternalServiceError(error_msg, service=service_name, details={
        "original_error": str(original_error),
        "service": service_name
    })


# Custom exception handlers for FastAPI
def http_exception_handler(request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    return create_error_response(
        RAGChatbotError(str(exc.detail), details={"status_code": exc.status_code}),
        exc.status_code
    )


def validation_exception_handler(request, exc: ValidationError):
    """Handler for validation exceptions"""
    return create_error_response(
        exc,
        status.HTTP_422_UNPROCESSABLE_ENTITY
    )


def generic_exception_handler(request, exc: Exception):
    """Handler for generic exceptions"""
    return create_error_response(
        RAGChatbotError("Internal server error", details={"original_error": str(exc)}),
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )