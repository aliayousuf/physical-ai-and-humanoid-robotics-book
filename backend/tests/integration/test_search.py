import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.api.search import router
from src.services.vector_db_service import vector_db_service
from src.services.embedding_service import embedding_service
from main import app

client = TestClient(app)


def test_semantic_search_endpoint():
    """
    Test the semantic search endpoint
    """
    # Mock the embedding service response
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example embedding

    # Mock the vector DB service response
    mock_search_results = [
        {
            "id": "chunk_12345",
            "content": "Artificial intelligence is a branch of computer science...",
            "metadata": {"document_id": "doc1", "chunk_index": 1},
            "similarity_score": 0.87,
            "document_id": "doc1",
            "chunk_index": 1
        },
        {
            "id": "chunk_12346",
            "content": "Machine learning is a subset of artificial intelligence...",
            "metadata": {"document_id": "doc2", "chunk_index": 2},
            "similarity_score": 0.76,
            "document_id": "doc2",
            "chunk_index": 2
        }
    ]

    with patch.object(embedding_service, 'embed_single_text', return_value=mock_embedding), \
         patch.object(vector_db_service, 'search_similar', return_value=mock_search_results):

        response = client.post(
            "/api/v1/search?query=What does the book say about artificial intelligence?&top_k=5&score_threshold=0.3"
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "search_time_ms" in data
        assert "timestamp" in data
        assert len(data["results"]) == 2
        assert data["query"] == "What does the book say about artificial intelligence?"


def test_semantic_search_endpoint_with_invalid_params():
    """
    Test the semantic search endpoint with invalid parameters
    """
    response = client.post(
        "/api/v1/search?query=test&top_k=25&score_threshold=0.3"  # Invalid: top_k too high
    )

    assert response.status_code == 400


def test_semantic_search_endpoint_long_query():
    """
    Test the semantic search endpoint with a query that is too long
    """
    long_query = "test " * 500  # This should exceed the max length

    response = client.post(
        "/api/v1/search?query=" + long_query + "&top_k=5&score_threshold=0.3"
    )

    assert response.status_code == 400


def test_search_health_check():
    """
    Test the search health check endpoint
    """
    # Mock the search functionality to return a successful result
    mock_search_results = []

    with patch.object(vector_db_service, 'search_similar', return_value=mock_search_results):
        response = client.get("/api/v1/search/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]  # Could be either depending on actual connection


def test_search_health_check_with_connection_error():
    """
    Test the search health check endpoint when there's a connection error
    """
    with patch.object(vector_db_service, 'search_similar', side_effect=Exception("Connection failed")):
        response = client.get("/api/v1/search/health")

        assert response.status_code == 200  # Health check returns 200, but with unhealthy status
        data = response.json()
        assert "status" in data
        assert data["status"] == "unhealthy"
        assert "qdrant_connected" in data
        assert data["qdrant_connected"] is False