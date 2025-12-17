import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app


def test_selected_text_query_functionality():
    """
    T054: Test selected text query functionality
    """
    client = TestClient(app)

    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

        # Mock the services
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_search.return_value = [
            {
                "content_id": "selected_text_content",
                "score": 1.0,  # Perfect match for selected text
                "payload": {
                    "title": "Selected Text Context",
                    "content": "The selected text that the user wants explained",
                    "page_reference": "/docs/test"
                }
            }
        ]
        mock_generate.return_value = "Based on the selected text, the explanation is quite straightforward."

        # Create a session
        response = client.post("/api/v1/chat/session", json={"initial_context": "Selected text testing"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Make a selected text query
        selected_text_payload = {
            "session_id": session_id,
            "query": "Can you explain this concept?",
            "selected_text": "The selected text that the user wants explained",
            "context": {
                "page_url": "/docs/test",
                "section_context": "In the context of this documentation page..."
            }
        }

        response = client.post("/api/v1/chat/selected-text-query", json=selected_text_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["query_mode"] == "selected_text"
        assert "selected_text" in str(data["sources"]).lower()
        assert "straightforward" in data["response"].lower()

        print("✓ T054: Selected text query functionality test passed")


def test_mode_switching_between_general_and_selected():
    """
    T055: Test mode switching between general and selected text
    """
    client = TestClient(app)

    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

        # Mock the services
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_search.return_value = [
            {
                "content_id": "general_content",
                "score": 0.85,
                "payload": {
                    "title": "General Content",
                    "content": "General content for the book",
                    "page_reference": "/docs/general"
                }
            }
        ]
        mock_generate.return_value = "This is a response based on general book content."

        # Create a session
        response = client.post("/api/v1/chat/session", json={"initial_context": "Mode switching testing"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Step 1: Make a general query
        general_query_payload = {
            "session_id": session_id,
            "query": "What is the main topic?",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=general_query_payload)
        assert response.status_code == 200
        general_data = response.json()
        assert general_data["query_mode"] == "general"

        # Step 2: Make a selected text query
        selected_text_payload = {
            "session_id": session_id,
            "query": "Explain this specific part",
            "selected_text": "Specific text that needs explanation",
            "context": {"page_url": "/docs/test", "section_context": "Section context..."}
        }

        response = client.post("/api/v1/chat/selected-text-query", json=selected_text_payload)
        assert response.status_code == 200
        selected_data = response.json()
        assert selected_data["query_mode"] == "selected_text"

        # Step 3: Make another general query to confirm switching back
        response = client.post("/api/v1/chat/query", json=general_query_payload)
        assert response.status_code == 200
        switched_back_data = response.json()
        assert switched_back_data["query_mode"] == "general"

        print("✓ T055: Mode switching between general and selected text test passed")


def test_selected_text_context_clearing():
    """
    T056: Test selected text context clearing
    """
    client = TestClient(app)

    # Create a session
    response = client.post("/api/v1/chat/session", json={"initial_context": "Context clearing testing"})
    assert response.status_code == 200
    session_data = response.json()
    session_id = session_data["session_id"]

    # Clear the context
    response = client.delete(f"/api/v1/chat/session/{session_id}/context")
    assert response.status_code == 200

    clear_data = response.json()
    assert clear_data["session_id"] == session_id
    assert clear_data["new_mode"] == "general"

    print("✓ T056: Selected text context clearing test passed")


if __name__ == "__main__":
    test_selected_text_query_functionality()
    test_mode_switching_between_general_and_selected()
    test_selected_text_context_clearing()
    print("\n✓ All selected text functionality tests passed!")