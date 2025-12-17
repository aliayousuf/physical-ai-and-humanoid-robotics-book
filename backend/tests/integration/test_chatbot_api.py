"""
Integration tests for the RAG Chatbot API
Tests the communication between frontend components and backend services
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.api.main import app
from src.services.rag_service import rag_service
from src.models.session import QueryMode


@pytest.fixture
def client():
    """Test client for the FastAPI app"""
    return TestClient(app)


@pytest.mark.asyncio
async def test_create_session_endpoint(client):
    """Test the session creation endpoint"""
    response = client.post("/api/v1/chat/session", json={"initial_context": "test context"})

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "created_at" in data
    assert "expires_at" in data

    # Verify session was created in the service
    session_id = data["session_id"]
    session = await rag_service.get_session(session_id)
    assert session is not None


@pytest.mark.asyncio
async def test_general_query_endpoint(client):
    """Test the general book content query endpoint"""
    # First create a session
    session_response = client.post("/api/v1/chat/session", json={})
    assert session_response.status_code == 200
    session_data = session_response.json()
    session_id = session_data["session_id"]

    # Test general query
    query_payload = {
        "session_id": session_id,
        "query": "What is the main topic of the book?",
        "mode": "general"
    }

    with patch.object(rag_service, 'process_general_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = {
            "response_id": "test_response_id",
            "session_id": session_id,
            "query": "What is the main topic of the book?",
            "response": "The main topic is Physical AI and Humanoid Robotics.",
            "sources": [{"content_id": "test_id", "title": "Test Title", "page_reference": "/docs/test", "relevance_score": 0.9}],
            "timestamp": "2025-12-17T10:00:00",
            "query_mode": "general"
        }

        response = client.post("/api/v1/chat/query", json=query_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "response" in data
        assert data["query_mode"] == "general"


@pytest.mark.asyncio
async def test_selected_text_query_endpoint(client):
    """Test the selected text query endpoint"""
    # First create a session
    session_response = client.post("/api/v1/chat/session", json={})
    assert session_response.status_code == 200
    session_data = session_response.json()
    session_id = session_data["session_id"]

    # Test selected text query
    query_payload = {
        "session_id": session_id,
        "query": "Explain this concept further",
        "mode": "selected_text",
        "selected_text": "The key concept is that humanoid robots must integrate sensory perception with motor control."
    }

    with patch.object(rag_service, 'process_selected_text_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = {
            "response_id": "test_response_id",
            "session_id": session_id,
            "query": "Explain this concept further",
            "response": "This concept refers to the integration of sensory perception with motor control in humanoid robots.",
            "sources": [{"content_id": "selected_text", "title": "Selected Text Context", "page_reference": "/docs/test", "relevance_score": 1.0}],
            "timestamp": "2025-12-17T10:00:00",
            "query_mode": "selected_text"
        }

        response = client.post("/api/v1/chat/query", json=query_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "response" in data
        assert data["query_mode"] == "selected_text"


@pytest.mark.asyncio
async def test_get_session_details(client):
    """Test the session details retrieval endpoint"""
    # First create a session
    session_response = client.post("/api/v1/chat/session", json={})
    assert session_response.status_code == 200
    session_data = session_response.json()
    session_id = session_data["session_id"]

    # Retrieve session details
    response = client.get(f"/api/v1/chat/session/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "created_at" in data
    assert "history" in data


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "services" in data


@pytest.mark.asyncio
async def test_error_handling_invalid_session(client):
    """Test error handling for invalid session ID"""
    query_payload = {
        "session_id": "invalid_session_id",
        "query": "Test query",
        "mode": "general"
    }

    response = client.post("/api/v1/chat/query", json=query_payload)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_error_handling_missing_query(client):
    """Test error handling for missing query"""
    # First create a session
    session_response = client.post("/api/v1/chat/session", json={})
    assert session_response.status_code == 200
    session_data = session_response.json()
    session_id = session_data["session_id"]

    query_payload = {
        "session_id": session_id,
        "query": "",  # Empty query
        "mode": "general"
    }

    response = client.post("/api/v1/chat/query", json=query_payload)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_mode_switching_functionality(client):
    """Test that mode switching works correctly between general and selected text"""
    # First create a session
    session_response = client.post("/api/v1/chat/session", json={})
    assert session_response.status_code == 200
    session_data = session_response.json()
    session_id = session_data["session_id"]

    # Test general query mode
    general_payload = {
        "session_id": session_id,
        "query": "What is Physical AI?",
        "mode": "general"
    }

    with patch.object(rag_service, 'process_general_query', new_callable=AsyncMock) as mock_general:
        mock_general.return_value = {
            "response_id": "test_response_id",
            "session_id": session_id,
            "query": "What is Physical AI?",
            "response": "Physical AI combines principles of robotics and AI.",
            "sources": [{"content_id": "test_id", "title": "Test Title", "page_reference": "/docs/test", "relevance_score": 0.9}],
            "timestamp": "2025-12-17T10:00:00",
            "query_mode": "general"
        }

        general_response = client.post("/api/v1/chat/query", json=general_payload)
        assert general_response.status_code == 200
        general_data = general_response.json()
        assert general_data["query_mode"] == "general"

    # Test selected text mode
    selected_payload = {
        "session_id": session_id,
        "query": "Explain this concept",
        "mode": "selected_text",
        "selected_text": "Physical AI is a field that combines robotics and artificial intelligence."
    }

    with patch.object(rag_service, 'process_selected_text_query', new_callable=AsyncMock) as mock_selected:
        mock_selected.return_value = {
            "response_id": "test_response_id2",
            "session_id": session_id,
            "query": "Explain this concept",
            "response": "This concept refers to the combination of robotics and AI.",
            "sources": [{"content_id": "selected_text", "title": "Selected Text Context", "page_reference": "/docs/test", "relevance_score": 1.0}],
            "timestamp": "2025-12-17T10:01:00",
            "query_mode": "selected_text"
        }

        selected_response = client.post("/api/v1/chat/query", json=selected_payload)
        assert selected_response.status_code == 200
        selected_data = selected_response.json()
        assert selected_data["query_mode"] == "selected_text"


@pytest.mark.asyncio
async def test_conversation_history_persistence(client):
    """Test that conversation history is maintained across queries"""
    # First create a session
    session_response = client.post("/api/v1/chat/session", json={})
    assert session_response.status_code == 200
    session_data = session_response.json()
    session_id = session_data["session_id"]

    # Make a query
    query_payload = {
        "session_id": session_id,
        "query": "What is the first principle?",
        "mode": "general"
    }

    with patch.object(rag_service, 'process_general_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = {
            "response_id": "response_1",
            "session_id": session_id,
            "query": "What is the first principle?",
            "response": "The first principle is integration of perception and action.",
            "sources": [{"content_id": "test_id", "title": "Test Title", "page_reference": "/docs/test", "relevance_score": 0.9}],
            "timestamp": "2025-12-17T10:00:00",
            "query_mode": "general"
        }

        response = client.post("/api/v1/chat/query", json=query_payload)
        assert response.status_code == 200

    # Retrieve session history to verify persistence
    history_response = client.get(f"/api/v1/chat/session/{session_id}")
    assert history_response.status_code == 200
    history_data = history_response.json()

    # Verify that the conversation history contains the query and response
    assert len(history_data["history"]) >= 2  # User query + Assistant response


def test_cors_configuration():
    """Test that CORS is properly configured for Docusaurus integration"""
    # This test verifies that the CORS middleware is properly configured
    # in the main app, which was done in src/api/main.py
    from src.api.main import app

    # Check if CORS middleware is in the middleware stack
    cors_found = False
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            cors_found = True
            break

    assert cors_found, "CORS middleware should be configured for Docusaurus integration"