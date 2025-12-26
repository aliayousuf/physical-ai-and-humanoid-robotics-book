from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4
from ..models.session import UserSession, Message, QueryMode
from ..models.query import QueryHistory, UserSelectionContext
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Import services with error handling to prevent initialization failures
try:
    from .qdrant_service import qdrant_service
    QDRANT_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import qdrant_service: {e}")
    QDRANT_AVAILABLE = False
    qdrant_service = None

try:
    from .gemini_service import gemini_service
    GEMINI_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import gemini_service: {e}")
    GEMINI_AVAILABLE = False
    gemini_service = None

try:
    from .embedding_service import embedding_service
    EMBEDDING_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import embedding_service: {e}")
    EMBEDDING_AVAILABLE = False
    embedding_service = None

from ..utils.logging import log_error, log_rag_query, alert_on_error


class RAGService:
    def __init__(self):
        self.sessions = {}  # In-memory session storage (use DB in production)

    async def create_session(self, initial_context: Optional[str] = None) -> UserSession:
        """
        Create a new chat session
        """
        expires_at = datetime.now() + timedelta(hours=settings.session_expiration_hours)

        session = UserSession(
            expires_at=expires_at,
            metadata={"initial_context": initial_context} if initial_context else {}
        )

        self.sessions[session.id] = session
        return session

    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Retrieve a session by ID
        """
        session = self.sessions.get(session_id)
        if session and datetime.now() > session.expires_at:
            # Session expired, remove it
            del self.sessions[session_id]
            return None
        return session

    async def process_general_query(self, session_id: str, query: str, page_url: str = None) -> Dict[str, Any]:
        """
        Process a general query about book content using RAG
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError("Invalid or expired session ID")

        # Validate query length
        if len(query) > settings.max_query_length:
            raise ValueError(f"Query exceeds maximum length of {settings.max_query_length} characters")

        # Check if required services are available
        if not EMBEDDING_AVAILABLE or embedding_service is None:
            logger.error("Embedding service not available")
            # Return a response without RAG functionality
            response_text = (
                "I can answer your question, but the RAG (Retrieval Augmented Generation) functionality is not available. "
                "This means I cannot search through the book content to provide specific answers. "
                "I'll do my best to answer based on my general knowledge: "
                "Please provide more context or check if the topic is covered in the documentation."
            )

            # Create response message
            response_message = Message(
                id=str(uuid4()),
                session_id=session_id,
                role="assistant",
                content=response_text,
                timestamp=datetime.now(),
                sources=[]
            )

            # Add user query to session history
            user_message = Message(
                id=str(uuid4()),
                session_id=session_id,
                role="user",
                content=query,
                timestamp=datetime.now()
            )

            session.conversation_history.append(user_message)
            session.conversation_history.append(response_message)

            # Return the response
            return {
                "response_id": response_message.id,
                "session_id": session_id,
                "query": query,
                "response": response_text,
                "sources": [],
                "timestamp": datetime.now(),
                "query_mode": "general"
            }

        # Generate embedding for the query
        start_time = datetime.now()
        query_embedding = await embedding_service.embed_single_text(query, input_type="search_query")

        # Search for similar content in the vector database
        if not QDRANT_AVAILABLE or qdrant_service is None:
            logger.error("Qdrant service not available")
            # Return a response without RAG functionality
            response_text = (
                "I can answer your question, but the search functionality is not available. "
                "This means I cannot search through the book content to provide specific answers. "
                "I'll do my best to answer based on my general knowledge: "
                "Please provide more context or check if the topic is covered in the documentation."
            )

            # Create response message
            response_message = Message(
                id=str(uuid4()),
                session_id=session_id,
                role="assistant",
                content=response_text,
                timestamp=datetime.now(),
                sources=[]
            )

            # Add user query to session history
            user_message = Message(
                id=str(uuid4()),
                session_id=session_id,
                role="user",
                content=query,
                timestamp=datetime.now()
            )

            session.conversation_history.append(user_message)
            session.conversation_history.append(response_message)

            # Return the response
            return {
                "response_id": response_message.id,
                "session_id": session_id,
                "query": query,
                "response": response_text,
                "sources": [],
                "timestamp": datetime.now(),
                "query_mode": "general"
            }

        search_results = await qdrant_service.search_similar(
            query_embedding,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold
        )

        # Retrieve content for the top results
        context_content = []
        for result in search_results:
            # In a real implementation, we would fetch the actual content using content_id
            # For now, we'll use the payload data
            context_content.append({
                "content_id": result["content_id"],
                "title": result["payload"].get("title", ""),
                "content": result["payload"].get("content", ""),
                "page_reference": result["payload"].get("page_reference", ""),
                "relevance_score": result["score"]
            })

        # Generate response using the LLM
        if not context_content:
            # No relevant content found, inform the user
            response_text = "I couldn't find any relevant content in the book to answer your question. Please try rephrasing your question or check if the topic is covered in the documentation."
        else:
            if not GEMINI_AVAILABLE or gemini_service is None:
                logger.error("Gemini service not available")
                # Graceful degradation: provide a helpful message
                response_text = (
                    "I found relevant content in the book, but encountered an issue generating a response. "
                    "Please try again, or check the following potentially relevant sections:\n" +
                    "\n".join([f"- {ctx['title']} (Score: {ctx['relevance_score']:.2f})" for ctx in context_content])
                )
            else:
                try:
                    response_text = await gemini_service.generate_response(query, context_content)
                except Exception as e:
                    logger.error(f"Error generating response with Gemini: {e}")
                    # Graceful degradation: provide a helpful message
                    response_text = (
                        "I found relevant content in the book, but encountered an issue generating a response. "
                        "Please try again, or check the following potentially relevant sections:\n" +
                        "\n".join([f"- {ctx['title']} (Score: {ctx['relevance_score']:.2f})" for ctx in context_content])
                    )

        # Log the RAG query with performance metrics
        response_time = (datetime.now() - start_time).total_seconds()
        log_rag_query(
            session_id=session_id,
            query=query,
            response=response_text,
            sources=context_content,
            query_mode=QueryMode.GENERAL,
            response_time=response_time * 1000  # Convert to milliseconds
        )

        # Create response message
        response_message = Message(
            id=str(uuid4()),
            session_id=session_id,
            role="assistant",
            content=response_text,
            timestamp=datetime.now(),
            sources=[{
                "content_id": ctx["content_id"],
                "title": ctx["title"],
                "page_reference": ctx["page_reference"],
                "relevance_score": ctx["relevance_score"]
            } for ctx in context_content]
        )

        # Add user query to session history
        user_message = Message(
            id=str(uuid4()),
            session_id=session_id,
            role="user",
            content=query,
            timestamp=datetime.now()
        )

        session.conversation_history.append(user_message)
        session.conversation_history.append(response_message)

        # Create query history record
        query_history = QueryHistory(
            session_id=session_id,
            query_text=query,
            response_text=response_text,
            query_mode=QueryMode.GENERAL,
            response_sources=[ctx["content_id"] for ctx in context_content if ctx.get("content_id")],
            response_time_ms=0,  # Would be calculated in a real implementation
            is_successful=True
        )

        # Return the response with sources
        return {
            "response_id": response_message.id,
            "session_id": session_id,
            "query": query,
            "response": response_text,
            "sources": [{
                "content_id": ctx["content_id"],
                "title": ctx["title"],
                "page_reference": ctx["page_reference"],
                "relevance_score": ctx["relevance_score"]
            } for ctx in context_content],
            "timestamp": datetime.now(),
            "query_mode": "general"
        }

    async def process_selected_text_query(self, session_id: str, query: str, selected_text: str, page_url: str = None, section_context: str = None) -> Dict[str, Any]:
        """
        Process a query about selected text only
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError("Invalid or expired session ID")

        # Validate query length
        if len(query) > settings.max_query_length:
            raise ValueError(f"Query exceeds maximum length of {settings.max_query_length} characters")

        start_time = datetime.now()

        # For selected text mode, we use the selected text as context instead of searching
        if not selected_text.strip():
            # No text was selected, inform the user
            response_text = "No text was selected to ask a question about. Please select some text on the page and try again."
            context_content = []
        else:
            context_content = [{
                "content_id": "selected_text",
                "title": "Selected Text Context",
                "content": selected_text,
                "page_reference": page_url or "",
                "relevance_score": 1.0
            }]

            # Create selected text context for the session
            selection_context = UserSelectionContext(
                session_id=session_id,
                selected_text=selected_text,
                page_url=page_url or "",
                section_context=section_context or ""
            )
            session.selected_text_context = selection_context

            # Generate response using the LLM with selected text as context
            if not GEMINI_AVAILABLE or gemini_service is None:
                logger.error("Gemini service not available for selected text query")
                # Graceful degradation: provide a helpful message
                response_text = (
                    "I have the selected text context, but encountered an issue generating a response. "
                    "Please try again, or consider rephrasing your question about the selected text."
                )
            else:
                try:
                    response_text = await gemini_service.generate_response(query, context_content)
                except Exception as e:
                    logger.error(f"Error generating response with Gemini for selected text: {e}")
                    # Graceful degradation: provide a helpful message
                    response_text = (
                        "I have the selected text context, but encountered an issue generating a response. "
                        "Please try again, or consider rephrasing your question about the selected text."
                    )

        # Log the RAG query with performance metrics
        response_time = (datetime.now() - start_time).total_seconds()
        log_rag_query(
            session_id=session_id,
            query=query,
            response=response_text,
            sources=context_content,
            query_mode=QueryMode.SELECTED_TEXT,
            response_time=response_time * 1000  # Convert to milliseconds
        )

        # Create response message
        response_message = Message(
            id=str(uuid4()),
            session_id=session_id,
            role="assistant",
            content=response_text,
            timestamp=datetime.now(),
            sources=[{
                "content_id": "selected_text",
                "title": "Selected Text Context",
                "page_reference": page_url or "",
                "relevance_score": 1.0
            }]
        )

        # Add user query to session history
        user_message = Message(
            id=str(uuid4()),
            session_id=session_id,
            role="user",
            content=query,
            timestamp=datetime.now()
        )

        session.conversation_history.append(user_message)
        session.conversation_history.append(response_message)

        # Update session mode to selected text
        session.current_mode = QueryMode.SELECTED_TEXT

        # Create query history record
        query_history = QueryHistory(
            session_id=session_id,
            query_text=query,
            response_text=response_text,
            query_mode=QueryMode.SELECTED_TEXT,
            selected_text=selected_text,
            response_sources=["selected_text"],
            response_time_ms=response_time * 1000,  # Now we have the actual time
            is_successful=True
        )

        # Return the response with sources
        return {
            "response_id": response_message.id,
            "session_id": session_id,
            "query": query,
            "response": response_text,
            "sources": [{
                "content_id": "selected_text",
                "title": "Selected Text Context",
                "page_reference": page_url or "",
                "relevance_score": 1.0
            }],
            "timestamp": datetime.now(),
            "query_mode": "selected_text"
        }

    async def get_conversation_history(self, session_id: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        Retrieve conversation history for a session
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError("Invalid or expired session ID")

            # Get messages with pagination
            total_messages = len(session.conversation_history)
            start_idx = min(offset, total_messages)
            end_idx = min(offset + limit, total_messages)

            messages = session.conversation_history[start_idx:end_idx]

            return {
                "session_id": session_id,
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "sources": msg.sources
                    } for msg in messages
                ],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total_messages
                }
            }
        except ValueError:
            # Re-raise ValueError as they are expected
            raise
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            # Return empty history as fallback
            return {
                "session_id": session_id,
                "messages": [],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 0
                }
            }

    async def clear_selected_text_context(self, session_id: str) -> Dict[str, Any]:
        """
        Clear the selected text context and return to general mode
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError("Invalid or expired session ID")

            session.selected_text_context = None
            session.current_mode = QueryMode.GENERAL

            return {
                "session_id": session_id,
                "message": "Session context cleared",
                "new_mode": "general"
            }
        except ValueError:
            # Re-raise ValueError as they are expected
            raise
        except Exception as e:
            logger.error(f"Error clearing selected text context: {e}")
            # Return appropriate error response
            raise ValueError(f"Error clearing selected text context: {str(e)}")


# Global instance with error handling
try:
    rag_service = RAGService()
except Exception as e:
    logger.error(f"Failed to initialize RAG service: {e}")
    # Create a mock service that indicates it's not available
    class MockRAGService:
        def __init__(self):
            self.sessions = {}

        async def create_session(self, initial_context=None):
            logger.error("RAG service not available, cannot create session")
            raise Exception("RAG service is not available")

        async def get_session(self, session_id):
            logger.error("RAG service not available, cannot get session")
            return None

        async def process_general_query(self, session_id, query, page_url=None):
            logger.error("RAG service not available, cannot process general query")
            raise Exception("RAG service is not available")

        async def process_selected_text_query(self, session_id, query, selected_text, page_url=None, section_context=None):
            logger.error("RAG service not available, cannot process selected text query")
            raise Exception("RAG service is not available")

        async def get_conversation_history(self, session_id, limit=20, offset=0):
            logger.error("RAG service not available, cannot get conversation history")
            return {"session_id": session_id, "messages": [], "pagination": {"limit": limit, "offset": offset, "total": 0}}

        async def clear_selected_text_context(self, session_id):
            logger.error("RAG service not available, cannot clear selected text context")
            raise Exception("RAG service is not available")

    rag_service = MockRAGService()