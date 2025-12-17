"""
Schemas for the RAG Chatbot API
"""
from .session import *
from .content import *
from .query import *
from .selection_context import *
from .status import *

__all__ = [
    "UserSessionCreate",
    "UserSessionResponse",
    "SessionDetailsResponse",
    "Message",
    "BookContentCreate",
    "BookContentResponse",
    "ContentResult",
    "QueryRequest",
    "QueryResponse",
    "QueryHistoryCreate",
    "QueryHistoryResponse",
    "UserSelectionContextCreate",
    "UserSelectionContextResponse",
    "StatusResponse",
    "ServiceStatus",
    "ErrorResponse",
]