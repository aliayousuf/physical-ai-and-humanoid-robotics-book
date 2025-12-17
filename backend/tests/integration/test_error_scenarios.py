import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app
from src.utils.monitoring import ServiceType


def test_error_handling_no_content_found():
    """
    T066: Test error handling for no content found scenarios
    """
    client = TestClient(app)

    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate:

        # Mock services to simulate no content found
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_search.return_value = []  # Empty results - no content found
        mock_generate.return_value = "I couldn't find any relevant content in the book to answer your question. Please try rephrasing your question or check if the topic is covered in the documentation."

        # Create a session first
        response = client.post("/api/v1/chat/session", json={"initial_context": "No content test"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Make a query that will result in no content found
        query_payload = {
            "session_id": session_id,
            "query": "What is the capital of Mars?",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=query_payload)
        assert response.status_code == 200

        data = response.json()
        assert "couldn't find any relevant content" in data["response"].lower()
        assert "rephrasing" in data["response"].lower()

        print("✓ T066: Error handling for no content found scenarios test passed")


def test_rate_limiting_functionality():
    """
    T067: Test rate limiting functionality
    """
    client = TestClient(app)

    # Test rate limiting by mocking the rate limiter to return False
    with patch('src.middleware.rate_limit.rate_limiter.is_allowed', return_value=False):
        response = client.get("/api/v1/health")
        assert response.status_code == 429  # Too Many Requests

        error_data = response.json()
        assert "RATE_LIMIT_EXCEEDED" in error_data["error"]["code"]
        assert "rate limit" in error_data["error"]["message"].lower()

        print("✓ T067: Rate limiting functionality test passed")


def test_api_error_responses_and_frontend_display():
    """
    T068: Test API error responses and frontend display
    """
    client = TestClient(app)

    # Test invalid session error
    invalid_query_payload = {
        "session_id": "nonexistent_session_id",
        "query": "Test query",
        "context": {"page_url": "/docs/test", "selected_mode": False}
    }

    response = client.post("/api/v1/chat/query", json=invalid_query_payload)
    assert response.status_code == 400

    error_data = response.json()
    assert "error" in error_data
    assert "INVALID_SESSION" in error_data["error"]["code"]
    assert "Session not found or expired" in error_data["error"]["message"]

    # Test malformed query error
    long_query_payload = {
        "session_id": "some_session_id",
        "query": "This is a very long query. " * 1000,  # Exceeds max length
        "context": {"page_url": "/docs/test", "selected_mode": False}
    }

    response = client.post("/api/v1/chat/query", json=long_query_payload)
    assert response.status_code == 400

    error_data = response.json()
    assert "error" in error_data
    assert "INVALID_REQUEST" in error_data["error"]["code"]
    assert "exceeds maximum length" in error_data["error"]["details"].lower()

    # Test selected text query error
    invalid_selected_payload = {
        "session_id": "nonexistent_session_id",
        "query": "Test query",
        "selected_text": "Test selected text",
        "context": {"page_url": "/docs/test", "section_context": "test"}
    }

    response = client.post("/api/v1/chat/selected-text-query", json=invalid_selected_payload)
    assert response.status_code == 400

    error_data = response.json()
    assert "error" in error_data
    assert "INVALID_SESSION" in error_data["error"]["code"]

    print("✓ T068: API error responses and frontend display test passed")


if __name__ == "__main__":
    test_error_handling_no_content_found()
    test_rate_limiting_functionality()
    test_api_error_responses_and_frontend_display()
    print("\n✓ All error scenario tests passed!")