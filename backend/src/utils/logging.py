import logging
from datetime import datetime
from typing import Any, Dict
from ..config.settings import settings
import sys


def setup_logging():
    """
    Set up logging configuration based on settings
    """
    # Create a custom logger
    logger = logging.getLogger("rag_chatbot")
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))

    # Create file handler if configured
    file_handler = None
    if hasattr(settings, 'log_file') and settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(getattr(logging, settings.log_level.upper()))

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler.setFormatter(simple_formatter)
    if file_handler:
        file_handler.setFormatter(detailed_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    if file_handler:
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


# Global logger instance
logger = setup_logging()


def log_api_call(
    endpoint: str,
    method: str,
    user_id: str = None,
    session_id: str = None,
    query: str = None,
    response_time: float = None,
    status_code: int = 200,
    error: str = None
):
    """
    Log API calls with relevant information
    """
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "session_id": session_id,
        "response_time_ms": response_time,
        "status_code": status_code,
        "error": error
    }

    # Only include query if it's not sensitive
    if query and len(query) < 200:  # Don't log very long queries
        log_data["query"] = query

    if status_code >= 400:
        logger.error(f"API call failed: {log_data}")
    else:
        logger.info(f"API call completed: {log_data}")


def log_rag_query(
    session_id: str,
    query: str,
    response: str,
    sources: list,
    query_mode: str,
    response_time: float = None
):
    """
    Log RAG query details
    """
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "query_mode": query_mode,
        "query_length": len(query),
        "response_length": len(response),
        "num_sources": len(sources),
        "response_time_ms": response_time
    }

    logger.info(f"RAG query processed: {log_data}")


def log_error(error: Exception, context: str = "", extra_info: Dict[str, Any] = None):
    """
    Log errors with context and extra information
    """
    error_msg = f"Error in {context}: {str(error)}"
    if extra_info:
        error_msg += f" | Extra info: {extra_info}"

    logger.error(error_msg, exc_info=True)


def alert_on_error(error: Exception, context: str = ""):
    """
    Enhanced error logging with alert capabilities
    """
    error_msg = f"CRITICAL ERROR in {context}: {str(error)}"
    logger.critical(error_msg, exc_info=True)

    # In a real implementation, this would trigger external alerts
    # (e.g., send to monitoring service, email admin, etc.)
    print(f"ALERT: {error_msg}")


def log_performance_issue(endpoint: str, response_time: float, threshold: float = 1.0):
    """
    Log performance issues when response time exceeds threshold
    """
    if response_time > threshold:
        logger.warning(f"Performance issue detected: {endpoint} took {response_time:.2f}s (threshold: {threshold}s)")


def log_security_event(event_type: str, details: Dict[str, Any] = None, severity: str = "MEDIUM"):
    """
    Log security-related events
    """
    security_msg = f"SECURITY EVENT [{severity}]: {event_type}"
    if details:
        security_msg += f" | Details: {details}"

    if severity == "HIGH" or severity == "CRITICAL":
        logger.critical(security_msg)
    else:
        logger.warning(security_msg)