import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_general_rag_functionality(client):
    """
    T031: Test general RAG functionality with sample queries
    """
    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

        # Mock the services
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_search.return_value = [
            {
                "content_id": "test_content",
                "score": 0.85,
                "payload": {
                    "title": "Test Content",
                    "content": "This is test content for the RAG functionality",
                    "page_reference": "/docs/test"
                }
            }
        ]
        mock_generate.return_value = "This is a test response based on the retrieved content."

        # Create a session
        response = client.post("/api/v1/chat/session", json={"initial_context": "Test context"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Make a general query
        query_payload = {
            "session_id": session_id,
            "query": "What is the main concept?",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=query_payload)
        assert response.status_code == 200

        data = response.json()
        assert "test response" in data["response"].lower()
        assert len(data["sources"]) > 0
        assert data["sources"][0]["relevance_score"] >= 0.8

        print("✓ T031: General RAG functionality test passed")


def test_conversation_context_maintenance(client):
    """
    T032: Test conversation context maintenance across multiple queries
    """
    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

        # Mock the services
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_search.return_value = [
            {
                "content_id": "context_content",
                "score": 0.9,
                "payload": {
                    "title": "Context Content",
                    "content": "This content is relevant for maintaining context",
                    "page_reference": "/docs/context"
                }
            }
        ]
        mock_generate.return_value = "This response maintains context from previous queries."

        # Create a session
        response = client.post("/api/v1/chat/session", json={"initial_context": "Context testing"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # First query
        query1_payload = {
            "session_id": session_id,
            "query": "What is the first concept?",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response1 = client.post("/api/v1/chat/query", json=query1_payload)
        assert response1.status_code == 200

        # Second query (follow-up)
        query2_payload = {
            "session_id": session_id,
            "query": "Can you elaborate on that?",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response2 = client.post("/api/v1/chat/query", json=query2_payload)
        assert response2.status_code == 200

        # Verify both messages are in history
        history_response = client.get(f"/api/v1/chat/session/{session_id}/history")
        assert history_response.status_code == 200
        history_data = history_response.json()

        # Should have at least 2 user messages and 2 assistant responses
        assert len(history_data["messages"]) >= 4

        print("✓ T032: Conversation context maintenance test passed")


def test_source_citation_accuracy(client):
    """
    T033: Test source citation accuracy in responses
    """
    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

        # Mock the services
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected_content = {
            "content_id": "accurate_source",
            "score": 0.95,
            "payload": {
                "title": "Accurate Source Title",
                "content": "This is the exact content that should be cited",
                "page_reference": "/docs/accurate/source"
            }
        }
        mock_search.return_value = [expected_content]
        mock_generate.return_value = "Based on the source, the answer is derived from the content."

        # Create a session
        response = client.post("/api/v1/chat/session", json={"initial_context": "Source citation testing"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Make a query
        query_payload = {
            "session_id": session_id,
            "query": "What does the source say?",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=query_payload)
        assert response.status_code == 200

        data = response.json()
        assert len(data["sources"]) > 0

        # Verify the source has proper information
        source = data["sources"][0]
        assert source["content_id"] == "accurate_source"
        assert source["relevance_score"] == 0.95
        assert "/docs/accurate/source" in source["page_reference"]
        assert len(source["title"]) > 0

        print("✓ T033: Source citation accuracy test passed")