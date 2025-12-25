"""
OpenAI Client configured to work with Google's Gemini 2.5 Flash model
This client uses the OpenAI SDK but routes requests to Google's Gemini API
"""
from typing import List, Dict, Any, Optional
import openai
from ..config.settings import settings
try:
    from google.generativeai import configure as configure_genai
    import google.generativeai as genai
    from google.ai import generativelanguage as glm
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    configure_genai = None
    genai = None
    glm = None
    GOOGLE_GENAI_AVAILABLE = False
    print("Warning: google-generativeai not available in openai_client")
import requests
import json


class OpenAIGeminiAdapter:
    """
    Adapter to use OpenAI SDK with Google's Gemini API
    This allows us to use OpenAI's familiar interface while leveraging Gemini's capabilities
    """

    def __init__(self):
        # Configure the Google Generative AI client
        if GOOGLE_GENAI_AVAILABLE and settings.gemini_api_key:
            configure_genai(api_key=settings.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            self.is_available = True
        else:
            self.gemini_model = None
            self.is_available = False
            print("Warning: OpenAIGeminiAdapter not available due to missing dependencies or API key")

        # For compatibility with OpenAI patterns, we'll also set the OpenAI API key
        # but this won't be used since we're routing through Gemini
        openai.api_key = settings.gemini_api_key

        # We'll also store the base URL for potential direct API calls
        self.gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.api_key = settings.gemini_api_key

    def chat_completions_create(self, model: str = "gemini-2.5-flash", messages: List[Dict[str, str]],
                                max_tokens: Optional[int] = None, temperature: Optional[float] = 0.7,
                                **kwargs) -> Dict[str, Any]:
        """
        Adapter method that mimics OpenAI's chat.completions.create method
        but uses Google's Gemini API
        """
        if not self.is_available or not GOOGLE_GENAI_AVAILABLE:
            raise Exception("OpenAIGeminiAdapter not available")

        # Convert OpenAI-style messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            parts = [msg.get("content", "")]

            gemini_contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": parts
            })

        # Generate content using Gemini
        generation_config = {
            "max_output_tokens": max_tokens or settings.max_response_tokens,
            "temperature": temperature
        }

        # Use the Gemini model to generate content
        response = self.gemini_model.generate_content(
            contents=gemini_contents,
            generation_config=genai.GenerationConfig(**generation_config)
        )

        # Format response to match OpenAI's structure
        openai_style_response = {
            "id": "chatcmpl-" + str(hash(response.text))[:8],  # Simple ID generation
            "object": "chat.completion",
            "created": int(__import__('time').time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(str(gemini_contents)),
                "completion_tokens": len(response.text.split()),
                "total_tokens": len(str(gemini_contents)) + len(response.text.split())
            }
        }

        return openai_style_response

    def embeddings_create(self, input: str, model: str = "text-embedding-004") -> Dict[str, Any]:
        """
        Create embeddings using Google's embedding API
        """
        # Use Google's embedding API
        embedding_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "content": {
                "parts": [
                    {
                        "text": input
                    }
                ]
            },
            "outputDimensionality": 768  # Google's embedding dimension
        }

        response = requests.post(
            f"{embedding_url}?key={self.api_key}",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            embedding = result["embedding"]["values"]

            # Format response to match OpenAI's structure
            openai_style_response = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": embedding
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": len(input.split()),
                    "total_tokens": len(input.split())
                }
            }

            return openai_style_response
        else:
            raise Exception(f"Embedding API call failed: {response.text}")


# Global instance
openai_client = OpenAIGeminiAdapter()

# Monkey-patch the openai module to use our adapter
# This allows existing OpenAI code to work with Gemini
class OpenAIAdapter:
    def __init__(self, client):
        self._client = client

    def chat(self):
        return self

    def completions(self):
        return self

    def create(self, **kwargs):
        return self._client.chat_completions_create(**kwargs)


# Set up the adapter to work with OpenAI patterns
openai.ChatCompletion = OpenAIAdapter(openai_client)