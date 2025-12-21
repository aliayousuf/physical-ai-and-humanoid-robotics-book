import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from main import app
from src.services.ingestion_service import ingestion_service
from src.services.chat_service import chat_service

client = TestClient(app)

def test_end_to_end_ingestion_and_chat():
    """
    End-to-end test to verify the complete flow:
    1. Trigger ingestion
    2. Check ingestion status
    3. Query the chatbot
    """
    # Mock ingestion service
    mock_job_id = "test-e2e-job-123"
    mock_job_status = {
        "id": mock_job_id,
        "status": "completed",
        "total_documents": 1,
        "processed_documents": 1,
        "started_at": datetime.now(),
        "completed_at": datetime.now(),
        "error_message": None
    }

    # Mock chat service response
    mock_chat_response = {
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

    with patch.object(ingestion_service, 'trigger_ingestion', return_value=mock_job_id), \
         patch.object(ingestion_service, 'get_job_status', return_value=mock_job_status), \
         patch.object(chat_service, 'query_chat', return_value=mock_chat_response):

        # Step 1: Trigger ingestion
        ingestion_response = client.post(
            "/api/v1/ingestion/trigger",
            json={
                "force_reprocess": False,
                "file_patterns": ["*.md"]
            }
        )
        assert ingestion_response.status_code == 200
        ingestion_data = ingestion_response.json()
        assert ingestion_data["job_id"] == mock_job_id
        assert ingestion_data["status"] == "completed"

        # Step 2: Check ingestion status
        status_response = client.get(f"/api/v1/ingestion/status/{mock_job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == mock_job_id
        assert status_data["status"] == "completed"

        # Step 3: Query the chatbot
        chat_response = client.post(
            "/api/v1/chat/query",
            json={
                "query": "What is humanoid robotics?",
                "max_results": 5,
                "similarity_threshold": 0.3
            }
        )
        assert chat_response.status_code == 200
        chat_data = chat_response.json()
        assert "response" in chat_data
        assert "sources" in chat_data
        assert chat_data["response"] == "This is a test response based on the book content."


def test_ingestion_and_document_listing():
    """
    Test the ingestion flow and document listing
    """
    mock_job_id = "test-doc-list-job-456"
    mock_job_status = {
        "id": mock_job_id,
        "status": "completed",
        "total_documents": 2,
        "processed_documents": 2,
        "started_at": datetime.now(),
        "completed_at": datetime.now(),
        "error_message": None
    }
    mock_documents = ["doc1.md", "doc2.pdf"]

    with patch.object(ingestion_service, 'trigger_ingestion', return_value=mock_job_id), \
         patch.object(ingestion_service, 'get_job_status', return_value=mock_job_status), \
         patch.object(ingestion_service, 'get_processed_documents', return_value=mock_documents):

        # Trigger ingestion
        ingestion_response = client.post(
            "/api/v1/ingestion/trigger",
            json={"force_reprocess": False}
        )
        assert ingestion_response.status_code == 200

        # Get processed documents
        docs_response = client.get("/api/v1/ingestion/documents")
        assert docs_response.status_code == 200
        docs_data = docs_response.json()
        assert "documents" in docs_data
        assert len(docs_data["documents"]) == 2
        assert all(doc in docs_data["documents"] for doc in mock_documents)


def test_health_endpoints():
    """
    Test health and readiness endpoints
    """
    # Test health endpoint
    health_response = client.get("/api/v1/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data["status"] == "healthy"

    # Test readiness endpoint
    readiness_response = client.get("/api/v1/ready")
    assert readiness_response.status_code == 200
    readiness_data = readiness_response.json()
    assert "status" in readiness_data