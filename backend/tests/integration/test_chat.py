import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.chat_api import router
from src.services.chat_service import chat_service
from main import app

client = TestClient(app)

def test_chat_query_endpoint():
    """
    Test the chat query endpoint with a sample query
    """
    # Mock the chat service response
    mock_response = {
        "response": "This is a test response based on the book content.",
        "sources": [
            {
                "filename": "test.md",
                "content": "Test content from the book...",
                "similarity_score": 0.85,
                "metadata": {
                    "page": 1,
                    "section": "Introduction",
                    "chunk_index": 0
                }
            }
        ],
        "confidence": 0.85
    }

    with patch.object(chat_service, 'query_chat', return_value=mock_response):
        response = client.post(
            "/api/v1/chat/query",
            json={
                "query": "What is humanoid robotics?",
                "max_results": 5,
                "similarity_threshold": 0.3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert "confidence" in data
        assert data["response"] == "This is a test response based on the book content."


def test_chat_query_simple_endpoint():
    """
    Test the simple chat query endpoint
    """
    # Mock the chat service response
    mock_response = {
        "response": "This is a simple test response.",
        "sources": [],
        "confidence": 0.0
    }

    with patch.object(chat_service, 'query_chat', return_value=mock_response):
        response = client.post(
            "/api/v1/chat/query_simple",
            params={
                "query": "What is AI?",
                "max_results": 3,
                "similarity_threshold": 0.5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "This is a simple test response."


def test_chat_query_too_long():
    """
    Test that the endpoint rejects queries that are too long
    """
    long_query = "A" * 2001  # Exceeds the 2000 character limit

    response = client.post(
        "/api/v1/chat/query",
        json={
            "query": long_query,
            "max_results": 5,
            "similarity_threshold": 0.3
        }
    )

    assert response.status_code == 400


def test_chat_query_simple_too_long():
    """
    Test that the simple endpoint rejects queries that are too long
    """
    long_query = "A" * 2001  # Exceeds the 2000 character limit

    response = client.post(
        "/api/v1/chat/query_simple",
        params={
            "query": long_query,
            "max_results": 3,
            "similarity_threshold": 0.5
        }
    )

    assert response.status_code == 400


def test_chat_stream_endpoint():
    """
    Test the chat streaming endpoint
    """
    # Mock the chat service response
    mock_response = {
        "response": "This is a test response based on the book content.",
        "sources": [
            {
                "filename": "test.md",
                "content": "Test content from the book...",
                "similarity_score": 0.85,
                "metadata": {
                    "page": 1,
                    "section": "Introduction",
                    "chunk_index": 0
                }
            }
        ],
        "confidence": 0.85
    }

    # Mock the get_relevant_content method for the streaming function
    mock_relevant_content = [
        {
            "id": "chunk_12345",
            "content": "Test content from the book...",
            "metadata": {"filename": "test.md", "page": 1, "section": "Introduction"},
            "similarity_score": 0.85,
            "document_id": "doc1",
            "chunk_index": 0
        }
    ]

    with patch.object(chat_service, 'query_chat', return_value=mock_response), \
         patch.object(chat_service, 'get_relevant_content', return_value=mock_relevant_content):

        response = client.post(
            "/api/v1/chat/stream",
            params={
                "query": "What is humanoid robotics?",
                "max_results": 5,
                "similarity_threshold": 0.3
            }
        )

        assert response.status_code == 200
        # Check that response is a streaming response
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


def test_chat_stream_endpoint_no_content():
    """
    Test the chat streaming endpoint when no relevant content is found
    """
    # Mock empty relevant content (should trigger "Not found in the book" response)
    mock_relevant_content = []

    with patch.object(chat_service, 'get_relevant_content', return_value=mock_relevant_content):

        response = client.post(
            "/api/v1/chat/stream",
            params={
                "query": "What is a completely unrelated topic?",
                "max_results": 5,
                "similarity_threshold": 0.3
            }
        )

        assert response.status_code == 200
        response_text = response.text
        # Verify that "Not found in the book" appears in the response
        assert "Not found in the book" in response_text


def test_chat_stream_endpoint_too_long():
    """
    Test that the streaming endpoint rejects queries that are too long
    """
    long_query = "A" * 2001  # Exceeds the 2000 character limit

    response = client.post(
        "/api/v1/chat/stream",
        params={
            "query": long_query,
            "max_results": 3,
            "similarity_threshold": 0.5
        }
    )

    assert response.status_code == 400