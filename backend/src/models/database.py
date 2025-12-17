"""
SQLAlchemy database models for the RAG Chatbot system
"""
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class UserSession(Base):
    """
    Represents an active chat session with a user, containing conversation history and metadata.
    """
    __tablename__ = "user_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_interaction = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    conversation_history = Column(JSON, default=list)  # Array of Message objects
    metadata = Column(JSON, default=dict)  # Additional session metadata

    # Relationships
    query_histories = relationship("QueryHistory", back_populates="session")
    selection_contexts = relationship("UserSelectionContext", back_populates="session")


class BookContent(Base):
    """
    Represents the documentation content from the book, including text segments and page references.
    """
    __tablename__ = "book_content"

    content_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    section_ref = Column(String, nullable=False)  # Reference to the book section
    content_text = Column(Text, nullable=False)  # The actual content text (max 1000 characters per chunk)
    embedding_vector = Column(String)  # Vector representation for semantic search (stored as JSON string)
    page_reference = Column(String)  # Specific page or URL reference
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    content_representations = relationship("ContentRepresentation", back_populates="book_content")


class ContentRepresentation(Base):
    """
    Mathematical representation of book content segments for semantic search operations.
    """
    __tablename__ = "content_representations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column(String, ForeignKey("book_content.content_id"), nullable=False)  # Reference to original content
    vector_data = Column(String, nullable=False)  # The embedding vector data (stored as JSON string)
    metadata = Column(JSON, default=dict)  # Additional metadata for search filtering
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    book_content = relationship("BookContent", back_populates="content_representations")


class QueryHistory(Base):
    """
    Record of user queries and system responses for context maintenance and analytics.
    """
    __tablename__ = "query_histories"

    query_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.session_id"), nullable=False)  # Associated session
    query_text = Column(Text, nullable=False)  # Original user query text
    response_text = Column(Text, nullable=False)  # System's response text
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  # When query was processed
    source_references = Column(JSON, default=list)  # References to book sections used
    query_mode = Column(String(20), nullable=False)  # Either "general" or "selected_text"
    selected_text_context = Column(Text)  # Text that was selected (for selected_text mode)
    response_tokens = Column(Integer)  # Number of tokens in response
    processing_time_ms = Column(Integer)  # Time taken to process query

    # Relationships
    session = relationship("UserSession", back_populates="query_histories")


class UserSelectionContext(Base):
    """
    Temporary data structure containing selected/highlighted text for contextual mode.
    """
    __tablename__ = "user_selection_contexts"

    selection_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.session_id"), nullable=False)  # Associated session
    selected_text = Column(Text, nullable=False)  # The actual selected/highlighted text
    page_context = Column(Text, nullable=False)  # Context around the selected text
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  # When selection was made
    expires_at = Column(DateTime(timezone=True), nullable=False)  # When selection context expires (30 minutes after creation)

    # Relationships
    session = relationship("UserSession", back_populates="selection_contexts")


class SystemStatus(Base):
    """
    Information about the operational state of various services (vector DB, LLM, API, etc.).
    """
    __tablename__ = "system_statuses"

    status_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    service_name = Column(String, nullable=False)  # Name of the service being monitored
    status = Column(String(20), nullable=False)  # Current status ("healthy", "degraded", "unavailable")
    last_checked = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  # When status was last checked
    details = Column(JSON, default=dict)  # Additional status details
    error_message = Column(Text)  # Error message if service is not healthy