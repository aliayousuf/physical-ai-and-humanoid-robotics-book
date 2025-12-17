import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime
from src.services.rag_service import RAGService
from src.models.session import QueryMode


@pytest.mark.asyncio
async def test_create_session():
    """Test creating a new session"""
    rag_service = RAGService()

    # Create a session
    session = await rag_service.create_session("Test context")

    # Verify session was created
    assert session is not None
    assert session.id is not None
    assert session.metadata["initial_context"] == "Test context"
    assert session.current_mode == QueryMode.GENERAL


@pytest.mark.asyncio
async def test_get_session():
    """Test retrieving an existing session"""
    rag_service = RAGService()

    # Create a session first
    session = await rag_service.create_session()
    session_id = session.id

    # Retrieve the session
    retrieved_session = await rag_service.get_session(session_id)

    # Verify session was retrieved
    assert retrieved_session is not None
    assert retrieved_session.id == session_id


@pytest.mark.asyncio
async def test_get_expired_session():
    """Test retrieving an expired session"""
    rag_service = RAGService()

    # Create a session with a past expiration time
    session = await rag_service.create_session()
    session.expires_at = datetime.now().replace(year=2020)  # Set to past date
    rag_service.sessions[session.id] = session

    # Try to retrieve the expired session
    retrieved_session = await rag_service.get_session(session.id)

    # Verify session was removed and None returned
    assert retrieved_session is None
    assert session.id not in rag_service.sessions


@pytest.mark.asyncio
async def test_process_general_query():
    """Test processing a general query"""
    rag_service = RAGService()

    # Create a session
    session = await rag_service.create_session()
    session_id = session.id

    # Mock the embedding and LLM services
    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

        # Set up mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]  # Mock embedding
        mock_search.return_value = [
            {
                "content_id": "test_id",
                "score": 0.9,
                "payload": {
                    "title": "Test Title",
                    "content": "Test content",
                    "page_reference": "/test/page"
                }
            }
        ]
        mock_generate.return_value = "Test response"

        # Process a query
        result = await rag_service.process_general_query(session_id, "Test query")

        # Verify the result
        assert result["query"] == "Test query"
        assert result["response"] == "Test response"
        assert result["session_id"] == session_id
        assert result["query_mode"] == "general"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["content_id"] == "test_id"