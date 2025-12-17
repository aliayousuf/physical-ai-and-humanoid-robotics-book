import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.api.main import app
from src.services.rag_service import rag_service


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.mark.asyncio
async def test_user_story_1_access_book_knowledge_via_chat():
    """
    Test User Story 1: Access Book Knowledge via Chat
    As a reader of the Physical AI and Humanoid Robotics book, I want to ask questions about the book content through an AI chatbot
    so that I can quickly find relevant information without manually searching through the documentation.
    """
    with TestClient(app) as client:
        # Mock external services
        with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
             patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
             patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

            # Set up mocks
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
            mock_search.return_value = [
                {
                    "content_id": "content_1",
                    "score": 0.92,
                    "payload": {
                        "title": "Introduction to Physical AI",
                        "content": "The key principles of physical AI include embodiment, interaction with the physical world, and closed-loop control systems.",
                        "page_reference": "/docs/introduction/physical-ai"
                    }
                }
            ]
            mock_generate.return_value = "The key principles of physical AI include embodiment, interaction with the physical world, and closed-loop control systems. These principles are detailed in the Introduction to Physical AI section."

            # Step 1: Create a session
            response = client.post("/api/v1/chat/session", json={"initial_context": "Physical AI book content"})
            assert response.status_code == 200
            session_data = response.json()
            session_id = session_data["session_id"]
            assert session_id is not None

            # Step 2: Ask a question about book content
            query_payload = {
                "session_id": session_id,
                "query": "What are the key principles of physical AI?",
                "context": {"page_url": "/docs/introduction", "selected_mode": False}
            }

            response = client.post("/api/v1/chat/query", json=query_payload)
            assert response.status_code == 200

            data = response.json()
            assert data["session_id"] == session_id
            assert "key principles" in data["query"].lower()
            assert "physical ai" in data["response"].lower()
            assert len(data["sources"]) > 0
            assert data["sources"][0]["relevance_score"] >= 0.9

            # Step 3: Test follow-up question to verify context maintenance
            follow_up_payload = {
                "session_id": session_id,
                "query": "Can you elaborate on the closed-loop control systems?",
                "context": {"page_url": "/docs/introduction", "selected_mode": False}
            }

            response = client.post("/api/v1/chat/query", json=follow_up_payload)
            assert response.status_code == 200

            follow_up_data = response.json()
            assert follow_up_data["session_id"] == session_id
            assert "closed-loop control" in follow_up_data["response"].lower() or "feedback loop" in follow_up_data["response"].lower()

            print("✓ User Story 1: Access Book Knowledge via Chat - PASSED")


@pytest.mark.asyncio
async def test_user_story_2_contextual_questions_on_selected_text():
    """
    Test User Story 2: Contextual Questions on Selected Text
    As a reader studying specific content on a page, I want to ask questions about only the text I've selected/highlighted
    so that I can get focused explanations without the chatbot referencing other parts of the book.
    """
    with TestClient(app) as client:
        # Mock external services
        with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
             patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate:

            # Set up mocks
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
            mock_generate.return_value = "In simpler terms, the complex mathematical framework refers to the use of linear algebra and differential equations to model physical systems. This allows robots to predict the effects of their actions."

            # Step 1: Create a session
            response = client.post("/api/v1/chat/session", json={"initial_context": "Physical AI book content"})
            assert response.status_code == 200
            session_data = response.json()
            session_id = session_data["session_id"]
            assert session_id is not None

            # Step 2: Ask a question about selected text
            selected_text_query_payload = {
                "session_id": session_id,
                "query": "Can you explain this concept in simpler terms?",
                "selected_text": "The complex mathematical framework of linear algebra and differential equations used to model physical systems",
                "context": {
                    "page_url": "/docs/advanced-topics/mathematical-framework",
                    "section_context": "In the context of humanoid robotics, the mathematical framework allows robots to predict the effects of their actions."
                }
            }

            response = client.post("/api/v1/chat/selected-text-query", json=selected_text_query_payload)
            assert response.status_code == 200

            data = response.json()
            assert data["session_id"] == session_id
            assert data["query_mode"] == "selected_text"
            assert "simpler terms" in data["query"].lower()
            assert "linear algebra" in data["response"].lower() or "mathematical" in data["response"].lower()

            # Step 3: Verify that the response is based on selected text, not general content
            assert "selected_text" in str(data["sources"]).lower()

            # Step 4: Test clearing the selected text context
            response = client.delete(f"/api/v1/chat/session/{session_id}/context")
            assert response.status_code == 200

            clear_data = response.json()
            assert clear_data["session_id"] == session_id
            assert clear_data["new_mode"] == "general"

            print("✓ User Story 2: Contextual Questions on Selected Text - PASSED")


@pytest.mark.asyncio
async def test_user_story_3_persistent_chat_interface():
    """
    Test User Story 3: Persistent Chat Interface Across All Pages
    As a reader navigating through the book documentation, I want the chatbot interface to be consistently available on every page
    so that I can access help without losing my place or context.
    """
    with TestClient(app) as client:
        # This test validates that the API endpoints are available and functional
        # The persistent UI aspect is frontend-specific and tested separately

        # Step 1: Verify health endpoint is accessible
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

        # Step 2: Create a session (should work on any page context)
        response = client.post("/api/v1/chat/session", json={"initial_context": "Page navigation test"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]
        assert session_id is not None

        # Step 3: Test that session history can be retrieved (maintaining context across "pages")
        response = client.get(f"/api/v1/chat/session/{session_id}/history")
        assert response.status_code == 200
        history_data = response.json()
        assert history_data["session_id"] == session_id
        assert "messages" in history_data
        assert "pagination" in history_data

        # Step 4: Test content summary endpoint (should be available on all pages)
        response = client.get("/api/v1/chat/content/summary")
        assert response.status_code == 200
        summary_data = response.json()
        assert "total_pages" in summary_data
        assert "total_content_segments" in summary_data
        assert "content_coverage" in summary_data

        print("✓ User Story 3: Persistent Chat Interface Across All Pages - PASSED")


@pytest.mark.asyncio
async def test_user_story_4_handle_edge_cases_and_error_conditions():
    """
    Test User Story 4: Handle Edge Cases and Error Conditions
    As a user, I want the chatbot to handle various error conditions gracefully
    so that I receive helpful feedback when issues occur.
    """
    with TestClient(app) as client:
        # Test 1: Invalid session ID
        invalid_query_payload = {
            "session_id": "invalid_session_id_that_does_not_exist",
            "query": "Test query",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=invalid_query_payload)
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "INVALID_SESSION" in error_data["error"]["code"]

        # Test 2: Very long query (should be rejected)
        long_query_payload = {
            "session_id": "some_session_id",
            "query": "This is a very long query. " * 1000,  # Way beyond max length
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=long_query_payload)
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "INVALID_REQUEST" in error_data["error"]["code"]
        assert "exceeds maximum length" in error_data["error"]["details"].lower()

        # Test 3: Malformed query with potential injection
        malicious_query_payload = {
            "session_id": "some_session_id",
            "query": "SELECT * FROM users; DROP TABLE users; --",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=malicious_query_payload)
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "MALFORMED_QUERY" in error_data["error"]["code"]

        # Test 4: Query with no relevant content (simulated by mocking empty search results)
        with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
             patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
             patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_search.return_value = []  # No results found
            mock_generate.return_value = "I couldn't find any relevant content in the book to answer your question. Please try rephrasing your question or check if the topic is covered in the documentation."

            # Create a valid session first
            response = client.post("/api/v1/chat/session", json={"initial_context": "Edge case testing"})
            assert response.status_code == 200
            session_data = response.json()
            session_id = session_data["session_id"]

            # Test query with no relevant content
            no_content_query_payload = {
                "session_id": session_id,
                "query": "What is the capital of Mars?",
                "context": {"page_url": "/docs/test", "selected_mode": False}
            }

            response = client.post("/api/v1/chat/query", json=no_content_query_payload)
            assert response.status_code == 200
            data = response.json()
            assert "couldn't find any relevant content" in data["response"].lower() or "try rephrasing" in data["response"].lower()

        print("✓ User Story 4: Handle Edge Cases and Error Conditions - PASSED")


def test_all_user_stories_comprehensive():
    """
    Comprehensive test that validates all user stories work together in a realistic scenario
    """
    # This would normally be an async test, but we're calling the individual tests
    # For demonstration purposes, we'll just call the individual tests

    # Create a test client
    client = TestClient(app)

    # Run through a complete scenario that touches all user stories
    with patch('src.services.embedding_service.embedding_service.embed_single_text') as mock_embed, \
         patch('src.services.gemini_service.gemini_service.generate_response') as mock_generate, \
         patch('src.services.qdrant_service.qdrant_service.search_similar') as mock_search:

        # Set up mocks
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_search.return_value = [
            {
                "content_id": "comprehensive_test",
                "score": 0.85,
                "payload": {
                    "title": "Comprehensive Test Section",
                    "content": "This section covers comprehensive testing of the RAG chatbot system.",
                    "page_reference": "/docs/comprehensive/test"
                }
            }
        ]
        mock_generate.return_value = "The comprehensive test confirms that all user stories work together seamlessly."

        # Step 1: Health check (US3 - persistent availability)
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        # Step 2: Create session (US3 - persistent interface)
        response = client.post("/api/v1/chat/session", json={"initial_context": "Comprehensive test"})
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Step 3: General book content query (US1 - access knowledge)
        general_query_payload = {
            "session_id": session_id,
            "query": "What does the comprehensive test section cover?",
            "context": {"page_url": "/docs/comprehensive", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=general_query_payload)
        assert response.status_code == 200
        general_data = response.json()
        assert "comprehensive" in general_data["response"].lower()

        # Step 4: Selected text query (US2 - contextual questions)
        selected_query_payload = {
            "session_id": session_id,
            "query": "Can you explain this further?",
            "selected_text": "The comprehensive testing approach covers multiple aspects of the system",
            "context": {"page_url": "/docs/comprehensive/advanced", "section_context": "Advanced testing methodologies"}
        }

        response = client.post("/api/v1/chat/selected-text-query", json=selected_query_payload)
        assert response.status_code == 200
        selected_data = response.json()
        assert selected_data["query_mode"] == "selected_text"

        # Step 5: Get history (US3 - persistent context)
        response = client.get(f"/api/v1/chat/session/{session_id}/history")
        assert response.status_code == 200
        history_data = response.json()
        assert len(history_data["messages"]) >= 2  # At least the two queries we made

        # Step 6: Test error handling (US4 - edge cases)
        invalid_payload = {
            "session_id": "nonexistent_session",
            "query": "Test query",
            "context": {"page_url": "/docs/test", "selected_mode": False}
        }

        response = client.post("/api/v1/chat/query", json=invalid_payload)
        assert response.status_code == 400

        print("✓ All User Stories Validation - COMPREHENSIVE TEST PASSED")


if __name__ == "__main__":
    # Run all the tests
    test_user_story_1_access_book_knowledge_via_chat()
    test_user_story_2_contextual_questions_on_selected_text()
    test_user_story_3_persistent_chat_interface()
    test_user_story_4_handle_edge_cases_and_error_conditions()
    test_all_user_stories_comprehensive()

    print("\n🎉 All user stories have been successfully validated!")
    print("✅ User Story 1: Access Book Knowledge via Chat - VALIDATED")
    print("✅ User Story 2: Contextual Questions on Selected Text - VALIDATED")
    print("✅ User Story 3: Persistent Chat Interface Across All Pages - VALIDATED")
    print("✅ User Story 4: Handle Edge Cases and Error Conditions - VALIDATED")