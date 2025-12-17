"""
Configuration for integration tests
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the entire test session"""
    with TestClient(app) as test_client:
        yield test_client