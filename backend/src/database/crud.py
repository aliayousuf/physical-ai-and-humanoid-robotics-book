"""
CRUD operations for the RAG Chatbot system
"""
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import uuid4, UUID
from datetime import datetime, timedelta
import json

from ..models.database import (
    UserSession, BookContent, ContentRepresentation,
    QueryHistory, UserSelectionContext, SystemStatus
)
from ..schemas import (
    UserSessionCreate, UserSessionResponse,
    BookContentCreate, QueryHistoryCreate,
    UserSelectionContextCreate
)


def create_session(db: Session) -> UserSession:
    """Create a new user session."""
    session_id = uuid4()
    db_session = UserSession(
        session_id=session_id,
        created_at=datetime.now(),
        last_interaction=datetime.now()
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def get_session(db: Session, session_id: UUID) -> Optional[UserSession]:
    """Get a user session by ID."""
    return db.query(UserSession).filter(UserSession.session_id == session_id).first()


def update_session(db: Session, session_id: UUID, **kwargs) -> Optional[UserSession]:
    """Update a user session."""
    db_session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if db_session:
        for key, value in kwargs.items():
            setattr(db_session, key, value)
        db_session.last_interaction = datetime.now()
        db.commit()
        db.refresh(db_session)
    return db_session


def create_book_content(db: Session, content: BookContentCreate) -> BookContent:
    """Create book content entry."""
    db_content = BookContent(
        content_id=str(uuid4()),
        title=content.title,
        section_ref=content.section_ref,
        content_text=content.content_text,
        page_reference=content.page_reference,
        created_at=datetime.now()
    )
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content


def get_book_content(db: Session, content_id: str) -> Optional[BookContent]:
    """Get book content by ID."""
    return db.query(BookContent).filter(BookContent.content_id == content_id).first()


def search_book_content(db: Session, query: str) -> List[BookContent]:
    """Search book content (basic implementation - would use vector search in practice)."""
    # This is a basic text search - in a real implementation, this would use vector search
    # against the Qdrant database rather than the SQL database
    return db.query(BookContent).filter(
        BookContent.content_text.contains(query) |
        BookContent.title.contains(query)
    ).limit(10).all()


def create_content_representation(db: Session, content_id: str, vector_data: List[float], metadata: dict = None) -> ContentRepresentation:
    """Create content representation (vector) entry."""
    db_representation = ContentRepresentation(
        content_id=content_id,
        vector_data=json.dumps(vector_data),  # Store as JSON string
        metadata=metadata or {},
        created_at=datetime.now()
    )
    db.add(db_representation)
    db.commit()
    db.refresh(db_representation)
    return db_representation


def get_content_representation(db: Session, content_id: str) -> Optional[ContentRepresentation]:
    """Get content representation by content ID."""
    return db.query(ContentRepresentation).filter(ContentRepresentation.content_id == content_id).first()


def create_query_history(db: Session, query_history: QueryHistoryCreate) -> QueryHistory:
    """Create query history entry."""
    db_query = QueryHistory(
        query_id=str(uuid4()),
        session_id=query_history.session_id,
        query_text=query_history.query_text,
        response_text=query_history.response_text,
        timestamp=datetime.now(),
        source_references=query_history.source_references,
        query_mode=query_history.query_mode,
        selected_text_context=query_history.selected_text_context,
        response_tokens=query_history.response_tokens,
        processing_time_ms=query_history.processing_time_ms
    )
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query


def get_query_history(db: Session, session_id: str, limit: int = 10) -> List[QueryHistory]:
    """Get query history for a session."""
    return db.query(QueryHistory).filter(
        QueryHistory.session_id == session_id
    ).order_by(QueryHistory.timestamp.desc()).limit(limit).all()


def create_user_selection_context(db: Session, selection: UserSelectionContextCreate) -> UserSelectionContext:
    """Create user selection context."""
    expires_at = datetime.now() + timedelta(minutes=30)  # 30 minutes expiry
    db_selection = UserSelectionContext(
        selection_id=str(uuid4()),
        session_id=selection.session_id,
        selected_text=selection.selected_text,
        page_context=selection.page_context,
        created_at=datetime.now(),
        expires_at=expires_at
    )
    db.add(db_selection)
    db.commit()
    db.refresh(db_selection)
    return db_selection


def get_user_selection_context(db: Session, selection_id: str) -> Optional[UserSelectionContext]:
    """Get user selection context by ID."""
    return db.query(UserSelectionContext).filter(
        UserSelectionContext.selection_id == selection_id
    ).first()


def create_system_status(db: Session, service_name: str, status: str, details: dict = None, error_message: str = None) -> SystemStatus:
    """Create or update system status."""
    # Check if status record already exists for this service
    existing_status = db.query(SystemStatus).filter(SystemStatus.service_name == service_name).first()

    if existing_status:
        # Update existing record
        existing_status.status = status
        existing_status.last_checked = datetime.now()
        existing_status.details = details or {}
        existing_status.error_message = error_message
        db.commit()
        db.refresh(existing_status)
        return existing_status
    else:
        # Create new record
        db_status = SystemStatus(
            status_id=str(uuid4()),
            service_name=service_name,
            status=status,
            last_checked=datetime.now(),
            details=details or {},
            error_message=error_message
        )
        db.add(db_status)
        db.commit()
        db.refresh(db_status)
        return db_status


def get_system_status(db: Session, service_name: str) -> Optional[SystemStatus]:
    """Get system status for a specific service."""
    return db.query(SystemStatus).filter(SystemStatus.service_name == service_name).first()


def update_system_status(db: Session, service_name: str, status: str, details: dict = None, error_message: str = None) -> Optional[SystemStatus]:
    """Update system status for a specific service."""
    db_status = db.query(SystemStatus).filter(SystemStatus.service_name == service_name).first()
    if db_status:
        db_status.status = status
        db_status.last_checked = datetime.now()
        db_status.details = details or db_status.details
        db_status.error_message = error_message
        db.commit()
        db.refresh(db_status)
    return db_status