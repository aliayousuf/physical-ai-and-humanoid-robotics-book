import re
from typing import Optional
from ..config.settings import settings


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    """
    if not text:
        return text

    # Remove potentially dangerous characters/sequences
    # This is a basic implementation - in production, use a more comprehensive sanitization library
    sanitized = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)

    return sanitized.strip()


def validate_query_length(query: str) -> Optional[str]:
    """
    Validate query length against configured maximum
    """
    if len(query) > settings.max_query_length:
        return f"Query exceeds maximum length of {settings.max_query_length} characters"
    return None


def validate_content_length(content: str) -> Optional[str]:
    """
    Validate content length
    """
    max_content_length = settings.max_query_length * 2  # Allow for longer content
    if len(content) > max_content_length:
        return f"Content exceeds maximum length of {max_content_length} characters"
    return None


def is_malformed_query(query: str) -> bool:
    """
    Check if query appears to be malformed or potentially malicious
    """
    # Check for excessive special characters that might indicate injection attempts
    special_char_ratio = len(re.findall(r'[^\w\s]', query)) / len(query) if query else 0
    if special_char_ratio > 0.5:  # More than 50% special characters
        return True

    # Check for common SQL injection patterns
    injection_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|WAITFOR|SLEEP)\b)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",  # e.g., OR 1=1
        r"('--|;--|;|/\*|\*/|--|\bEXEC\b|\bUNION\b|\bSELECT\b)"
    ]

    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True

    return False