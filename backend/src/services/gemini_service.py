import google.generativeai as genai
from typing import List, Dict, Any, Optional
import asyncio
from ..config.settings import settings
from ..utils.monitoring import check_service_usage, ServiceType


class GeminiService:
    def __init__(self):
        if settings.gemini_api_key:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.model = genai.GenerativeModel(settings.gemini_model_name)
                self.generation_config = genai.GenerationConfig(
                    max_output_tokens=settings.max_response_tokens,
                    temperature=0.7,
                )
                self.is_available = True
            except Exception as e:
                print(f"Warning: Could not configure Gemini service: {e}")
                self.is_available = False
                self.model = None
                self.generation_config = None
        else:
            print("Warning: Gemini API key not provided, service will be unavailable")
            self.is_available = False
            self.model = None
            self.generation_config = None

    async def generate_response(self, prompt: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the Gemini model
        """
        if not self.is_available:
            print("Gemini service not available, cannot generate response")
            raise Exception("Gemini service not available")

        # Check usage limits before making API call
        if not check_service_usage(ServiceType.GEMINI):
            raise Exception("Gemini API usage limit exceeded. Please try again later.")

        try:
            # Build the full prompt with context if provided
            full_prompt = prompt
            if context:
                context_str = "\n\nRelevant context from the book:\n"
                for i, ctx in enumerate(context, 1):
                    context_str += f"{i}. {ctx.get('title', '')}: {ctx.get('content', '')[:500]}...\n"
                full_prompt = f"{context_str}\n\nQuestion: {prompt}\n\nPlease provide a helpful response based on the above context, citing specific sections when possible."

            # Set a timeout for the API call (e.g., 30 seconds)
            # Note: The genai library doesn't have a direct timeout parameter,
            # so we wrap the call in asyncio.wait_for for timeout control
            response = await asyncio.wait_for(
                self._make_generation_call(full_prompt),
                timeout=30.0  # 30-second timeout
            )

            return response.text
        except asyncio.TimeoutError:
            raise Exception("Timeout: Gemini API call took too long to respond. Please try again.")
        except Exception as e:
            print(f"Error generating response with Gemini: {e}")
            raise

    async def _make_generation_call(self, full_prompt: str):
        """
        Helper method to make the actual generation call
        """
        response = self.model.generate_content(
            full_prompt,
            generation_config=self.generation_config
        )
        return response

    async def embed_content(self, content: str) -> List[float]:
        """
        Generate embedding for content using Gemini's embedding model
        """
        if not self.is_available:
            print("Gemini service not available, cannot generate embedding")
            raise Exception("Gemini service not available")

        # Check usage limits before making API call
        if not check_service_usage(ServiceType.GEMINI):
            raise Exception("Gemini API usage limit exceeded. Please try again later.")

        try:
            # Use the Google Generative AI embeddings
            import google.generativeai as genai
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_model_name,
                google_api_key=settings.gemini_api_key
            )

            result = embeddings.embed_query(content)
            return result
        except Exception as e:
            print(f"Error generating embedding with Gemini: {e}")
            raise


# Global instance with error handling
try:
    gemini_service = GeminiService()
except Exception as e:
    print(f"Warning: Could not initialize Gemini service: {e}")
    # Create a mock service that indicates it's not available
    class MockGeminiService:
        def __init__(self):
            self.is_available = False

        async def generate_response(self, prompt, context=None):
            print("Gemini service not available, cannot generate response")
            raise Exception("Gemini service not available")

        async def embed_content(self, content):
            print("Gemini service not available, cannot generate embedding")
            raise Exception("Gemini service not available")

    gemini_service = MockGeminiService()