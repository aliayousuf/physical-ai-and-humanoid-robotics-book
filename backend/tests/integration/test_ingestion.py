import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.api.ingestion import router
from src.services.ingestion_service import ingestion_service
from main import app

client = TestClient(app)

def test_ingestion_trigger_endpoint():
    """
    Test the ingestion trigger endpoint
    """
    # Mock the ingestion service response
    mock_job_id = "test-job-123"
    mock_job_status = {
        "id": mock_job_id,
        "status": "queued",
        "total_documents": 2,
        "processed_documents": 0,
        "started_at": datetime.now(),
        "completed_at": None,
        "error_message": None
    }

    with patch.object(ingestion_service, 'trigger_ingestion', return_value=mock_job_id), \
         patch.object(ingestion_service, 'get_job_status', return_value=mock_job_status):

        response = client.post(
            "/api/v1/ingestion/trigger",
            json={
                "force_reprocess": False,
                "file_patterns": ["*.md", "*.pdf"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["job_id"] == mock_job_id
        assert data["status"] == "queued"
        assert data["total_documents"] == 2


def test_ingestion_status_endpoint():
    """
    Test the ingestion status endpoint
    """
    # Mock the ingestion service response
    mock_job_id = "test-job-123"
    mock_job_status = {
        "id": mock_job_id,
        "status": "completed",
        "total_documents": 2,
        "processed_documents": 2,
        "started_at": datetime.now(),
        "completed_at": datetime.now(),
        "error_message": None
    }

    with patch.object(ingestion_service, 'get_job_status', return_value=mock_job_status):
        response = client.get(f"/api/v1/ingestion/status/{mock_job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == mock_job_id
        assert data["status"] == "completed"
        assert data["total_documents"] == 2
        assert data["processed_documents"] == 2


def test_ingestion_status_not_found():
    """
    Test the ingestion status endpoint when job doesn't exist
    """
    with patch.object(ingestion_service, 'get_job_status', return_value=None):
        response = client.get("/api/v1/ingestion/status/non-existent-job")

        assert response.status_code == 404


def test_get_processed_documents():
    """
    Test the endpoint to get processed documents
    """
    # Mock the ingestion service response
    mock_documents = ["doc1", "doc2", "doc3"]

    with patch.object(ingestion_service, 'get_processed_documents', return_value=mock_documents):
        response = client.get("/api/v1/ingestion/documents")

        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) == 3
        assert all(doc in data["documents"] for doc in mock_documents)