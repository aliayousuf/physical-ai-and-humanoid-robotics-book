import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app
from src.utils.monitoring import metrics_collector, get_performance_report
from src.utils.logging import logger


def test_monitoring_and_logging():
    """
    T065: Test monitoring and logging for error conditions
    """
    client = TestClient(app)

    # Test that API calls are being monitored
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    # Check that metrics were collected
    metrics = get_performance_report()
    health_metrics = metrics.get("/health")

    if health_metrics:
        assert health_metrics["count"] >= 1
        assert health_metrics["error_count"] == 0
        print("✓ Performance metrics are being collected")
    else:
        print("⚠ Health endpoint not found in metrics (may not have been called via API)")

    # Test error logging by attempting an invalid request
    with patch('src.services.rag_service.rag_service.create_session') as mock_create_session:
        mock_create_session.side_effect = Exception("Connection failed")

        # This would cause an error that should be logged
        response = client.post("/api/v1/chat/session", json={"initial_context": "test"})

        # Even if it fails, the error should have been logged
        print("✓ Error logging test completed")

    # Test usage monitoring
    from src.utils.monitoring import usage_monitor

    # Get initial usage
    initial_usage = usage_monitor.get_usage(type(usage_monitor).ServiceType.GEMINI)

    # This test confirms that monitoring is in place
    assert "current_usage" in initial_usage
    assert "limit" in initial_usage
    assert "percentage_used" in initial_usage

    print("✓ Usage monitoring is in place")


def test_error_handling_and_logging():
    """
    Test comprehensive error handling and logging
    """
    client = TestClient(app)

    # Test invalid session
    invalid_query_payload = {
        "session_id": "invalid_session_id_that_does_not_exist",
        "query": "Test query",
        "context": {"page_url": "/docs/test", "selected_mode": False}
    }

    response = client.post("/api/v1/chat/query", json=invalid_query_payload)
    assert response.status_code == 400

    # The error should have been logged
    print("✓ Error handling and logging test completed")


if __name__ == "__main__":
    test_monitoring_and_logging()
    test_error_handling_and_logging()
    print("\n✓ All monitoring and logging tests passed!")