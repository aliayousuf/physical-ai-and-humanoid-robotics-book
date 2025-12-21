import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from src.services.embedding_service import embedding_service
from src.services.vector_db_service import vector_db_service
from src.config.settings import settings
from src.models.document_chunk import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for RAG (Retrieval Augmented Generation) functionality using Gemini
    """
    def __init__(self):
        # Configure the API key
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model_name)

    async def query_chat(self, query: str, max_results: int = 5, similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Query the chatbot with book content using RAG approach
        """
        try:
            # Generate embedding for the query
            query_embedding = await embedding_service.embed_single_text(query, input_type="retrieval_query")

            # Search for similar content in the vector database
            search_results = vector_db_service.search_similar(
                query_embedding,
                top_k=max_results,
                threshold=similarity_threshold
            )

            if not search_results:
                # No relevant content found, return the specified response
                return {
                    "response": "Not found in the book.",
                    "sources": [],
                    "confidence": 0.0
                }

            # Prepare context from retrieved documents
            context_parts = []
            sources = []

            for result in search_results:
                context_parts.append(result["content"])

                source = {
                    "filename": result["metadata"].get("filename", "Unknown"),
                    "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],  # Truncate for display
                    "similarity_score": result["similarity_score"],
                    "metadata": {
                        "page": result["metadata"].get("page"),
                        "section": result["metadata"].get("section"),
                        "chunk_index": result["chunk_index"]
                    }
                }
                sources.append(source)

            # Combine context
            context = "\n\n".join(context_parts)

            # Create a prompt that includes the retrieved context
            prompt = f"""
            You are a helpful assistant for the Physical AI and Humanoid Robotics book.
            Use the following context to answer the user's question.
            If the context doesn't contain enough information to answer the question,
            acknowledge this and suggest the user check the book directly.

            Context:
            {context}

            User's question: {query}

            Please provide a helpful and accurate answer based on the context provided.
            """

            # Generate response using Gemini
            response = self.model.generate_content(prompt)

            # Extract text from response
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
            else:
                response_text = "I couldn't generate a response based on the available information."

            # Calculate a confidence score based on the highest similarity score
            confidence = max([result["similarity_score"] for result in search_results]) if search_results else 0.0

            return {
                "response": response_text,
                "sources": sources,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error in chat query: {str(e)}")
            return {
                "response": "An error occurred while processing your query. Please try again.",
                "sources": [],
                "confidence": 0.0
            }

    async def get_relevant_content(self, query: str, max_results: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Get relevant content from the vector database without generating a response
        """
        try:
            # Generate embedding for the query
            query_embedding = await embedding_service.embed_single_text(query, input_type="retrieval_query")

            # Search for similar content in the vector database
            search_results = vector_db_service.search_similar(
                query_embedding,
                top_k=max_results,
                threshold=similarity_threshold
            )

            return search_results
        except Exception as e:
            logger.error(f"Error getting relevant content: {str(e)}")
            return []

# Create a singleton instance
chat_service = ChatService()