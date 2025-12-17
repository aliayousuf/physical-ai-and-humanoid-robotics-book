import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.api.main import app
from src.services.rag_service import rag_service


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.mark.asyncio
async def test_end_to_end_session_creation_and_query():
    """Test the complete flow: session creation -> general query -> selected text query -> history retrieval"""
    with TestClient(app) as client:
        # Mock external services
        with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
             patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
             patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

            # Set up mocks
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
            mock_search.return_value = [
                {
                    "content_id": "test_id_1",
                    "score": 0.9,
                    "payload": {
                        "title": "Test Title 1",
                        "content": "Test content for the first result",
                        "page_reference": "/test/page1"
                    }
                },
                {
                    "content_id": "test_id_2",
                    "score": 0.8,
                    "payload": {
                        "title": "Test Title 2",
                        "content": "Test content for the second result",
                        "page_reference": "/test/page2"
                    }
                }
            ]
            mock_generate.return_value = "This is a test response from the LLM."

            # Step 1: Create a session
            response = client.post("/api/v1/chat/session", json={"initial_context": "Test context"})
            assert response.status_code == 200
            data = response.json()
            session_id = data["session_id"]
            assert session_id is not None

            # Step 2: Make a general query
            query_payload = {
                "session_id": session_id,
                "query": "What are the key principles of physical AI?",
                "context": {"page_url": "/docs/introduction", "selected_mode": False}
            }
            response = client.post("/api/v1/chat/query", json=query_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert data["query"] == "What are the key principles of physical AI?"
            assert "This is a test response" in data["response"]
            assert len(data["sources"]) > 0

            # Step 3: Make a selected text query
            selected_query_payload = {
                "session_id": session_id,
                "query": "Can you explain this concept?",
                "selected_text": "The complex mathematical framework for physical AI",
                "context": {
                    "page_url": "/docs/advanced/concepts",
                    "section_context": "In the context of humanoid robotics, the mathematical framework..."
                }
            }
            response = client.post("/api/v1/chat/selected-text-query", json=selected_query_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert data["query_mode"] == "selected_text"
            assert "This is a test response" in data["response"]

            # Step 4: Retrieve conversation history
            response = client.get(f"/api/v1/chat/session/{session_id}/history")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert len(data["messages"]) >= 2  # Should have at least user query and assistant response


@pytest.mark.asyncio
async def test_error_handling_scenarios():
    """Test error handling for various scenarios"""
    with TestClient(app) as client:
        # Test invalid session ID
        query_payload = {
            "session_id": "invalid_session_id",
            "query": "Test query",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }
        response = client.post("/api/v1/chat/query", json=query_payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

        # Test rate limiting (mock the rate limiter to trigger limit)
        with patch('src.middleware.rate_limit.rate_limiter.is_allowed', return_value=False):
            response = client.get("/api/v1/health")
            assert response.status_code == 429
            data = response.json()
            assert "RATE_LIMIT_EXCEEDED" in data.get("error", {}).get("code", "")


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health endpoint"""
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        # Note: In real implementation, services status would depend on actual connectivity


@pytest.mark.asyncio
async def test_content_summary_endpoint():
    """Test the content summary endpoint"""
    with TestClient(app) as client:
        response = client.get("/api/v1/chat/content/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_pages" in data
        assert "total_content_segments" in data
        assert "total_tokens" in data
        assert "last_indexed" in data
        assert "content_coverage" in data


@pytest.mark.asyncio
async def test_session_context_management():
    """Test session context management including clearing selected text context"""
    with TestClient(app) as client:
        # First create a session
        response = client.post("/api/v1/chat/session", json={})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Mock the services for a selected text query
        with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
             patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate:

            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_generate.return_value = "Test response for selected text."

            # Make a selected text query to set the context
            selected_query_payload = {
                "session_id": session_id,
                "query": "Explain this text",
                "selected_text": "Sample selected text",
                "context": {"page_url": "/docs/test", "section_context": "Context..."}
            }
            response = client.post("/api/v1/chat/selected-text-query", json=selected_query_payload)
            assert response.status_code == 200

            # Now clear the context
            response = client.delete(f"/api/v1/chat/session/{session_id}/context")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert data["new_mode"] == "general"