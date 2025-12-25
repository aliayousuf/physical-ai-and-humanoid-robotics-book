"""
OpenAI Service configured to work with Google's Gemini 2.5 Flash model
This service uses the OpenAI SDK pattern but routes requests to Google's Gemini API
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..config.settings import settings
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_GENAI_AVAILABLE = False
    print("Warning: google-generativeai not available in openai_service")
from ..utils.monitoring import check_service_usage, ServiceType
from ..utils.logging import logger


class OpenAIService:
    """
    Service that uses OpenAI SDK patterns but configured for Google's Gemini API
    This allows us to leverage the OpenAI SDK's interface while using Gemini's capabilities
    """

    def __init__(self):
        # Configure the Google Generative AI client as the underlying engine
        if GOOGLE_GENAI_AVAILABLE and settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            self.is_available = True
        else:
            self.gemini_model = None
            self.is_available = False
            print("Warning: OpenAIService not available due to missing dependencies or API key")

        # Initialize OpenAI client with Gemini-compatible configuration
        # We'll use this to maintain the OpenAI SDK interface
        self.client = None  # Not actually used since we're routing through Gemini

        # Store configuration
        self.model_name = settings.model_name
        self.max_tokens = settings.max_response_tokens
        self.temperature = 0.7

    async def generate_response(self, prompt: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the configured model (Gemini via OpenAI patterns)
        """
        if not self.is_available or not GOOGLE_GENAI_AVAILABLE:
            raise Exception("OpenAIService not available")

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

            # Generate content using Gemini (the actual backend)
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            )

            if response.text:
                return response.text
            else:
                raise Exception("No response generated from model")

        except Exception as e:
            logger.log_error(e, "OpenAIService.generate_response")
            raise

    async def create_embedding(self, text: str, model: str = "text-embedding-004") -> List[float]:
        """
        Create embedding for text
        Note: For embeddings, we're using Cohere as specified in the architecture
        This method is provided for OpenAI SDK compatibility
        """
        # In the actual implementation, we'd use Cohere for embeddings
        # This is just a placeholder to maintain OpenAI interface compatibility
        raise NotImplementedError("Use Cohere EmbeddingService for embeddings instead of this service")

    def chat_completions_create(self, messages: List[Dict[str, str]], **kwargs):
        """
        Mock method to maintain OpenAI SDK interface compatibility
        """
        # This would route to the actual Gemini model in a real implementation
        # For now, this is a placeholder to maintain interface compatibility
        pass


# Global instance
openai_service = OpenAIService()