import google.generativeai as genai
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ..config.settings import settings
from ..utils.cache import embeddings_cache
from ..utils.monitoring import check_service_usage, ServiceType


class EmbeddingService:
    def __init__(self):
        # Configure the Google Generative AI
        genai.configure(api_key=settings.gemini_api_key)

        # Initialize the Google Generative AI Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key
        )

    async def generate_embeddings(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Google Gemini
        With caching for frequently accessed embeddings
        """
        # Check usage limits before making API call
        if not check_service_usage(ServiceType.GEMINI):
            raise Exception("Gemini API usage limit exceeded. Please try again later.")

        try:
            # Create cache keys for each text
            cache_keys = [f"embedding_{text[:50]}_{input_type}" for text in texts]
            results = []

            # Check cache first for each text
            for i, text in enumerate(texts):
                cache_key = cache_keys[i]
                cached_result = embeddings_cache.get(cache_key)

                if cached_result is not None:
                    results.append(cached_result)
                else:
                    # If not in cache, we'll generate them all at once
                    # (Google API is more efficient with batch requests)
                    break
            else:
                # All results were in cache
                return results

            # Some or all results not in cache, make API call using Google embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # Cache the results
            for i, embedding in enumerate(embeddings):
                cache_keys[i] = f"embedding_{texts[i][:50]}_{input_type}"
                embeddings_cache.set(cache_keys[i], embedding, ttl=3600)  # Cache for 1 hour

            return embeddings
        except Exception as e:
            print(f"Error generating embeddings with Google Gemini: {e}")
            raise

    async def embed_single_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding for a single text using Google Gemini
        """
        # Check cache first
        cache_key = f"embedding_{text[:50]}_{input_type}"
        cached_result = embeddings_cache.get(cache_key)

        if cached_result is not None:
            return cached_result

        # Not in cache, generate using Google embeddings
        try:
            # Use the Google embeddings directly for single text
            result = self.embeddings.embed_query(text)

            if result:
                embeddings_cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour

            return result
        except Exception as e:
            print(f"Error generating single embedding with Google Gemini: {e}")
            raise


# Global instance
embedding_service = EmbeddingService()