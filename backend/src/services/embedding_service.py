try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_GENAI_AVAILABLE = False
    print("Warning: google-generativeai not available, some features may be disabled")
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ..config.settings import settings
from ..utils.cache import embeddings_cache
from ..utils.monitoring import check_service_usage, ServiceType


class EmbeddingService:
    def __init__(self):
        if settings.gemini_api_key and GOOGLE_GENAI_AVAILABLE:
            try:
                # Configure the Google Generative AI
                genai.configure(api_key=settings.gemini_api_key)

                # Initialize the Google Generative AI Embeddings
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=settings.gemini_embedding_model,
                    google_api_key=settings.gemini_api_key
                )
                self.is_available = True
            except Exception as e:
                print(f"Warning: Could not configure EmbeddingService: {e}")
                self.is_available = False
                self.embeddings = None
        else:
            if not GOOGLE_GENAI_AVAILABLE:
                print("Warning: Google Generative AI package not available, embedding service will be unavailable")
            else:
                print("Warning: Gemini API key not provided to EmbeddingService")
            self.is_available = False
            self.embeddings = None

    async def generate_embeddings(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Google Gemini
        With caching for frequently accessed embeddings
        """
        if not self.is_available:
            print("EmbeddingService not available, returning empty results")
            return [[] for _ in texts]  # Return empty embeddings for each text

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
        if not self.is_available:
            print("EmbeddingService not available, returning empty embedding")
            return []  # Return empty embedding

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


# Global instance with error handling
try:
    embedding_service = EmbeddingService()
except Exception as e:
    print(f"Warning: Could not initialize EmbeddingService: {e}")
    # Create a mock service that indicates it's not available
    class MockEmbeddingService:
        def __init__(self):
            self.is_available = False

        async def generate_embeddings(self, texts, input_type="search_document"):
            print("EmbeddingService not available, returning empty results")
            return [[] for _ in texts]

        async def embed_single_text(self, text, input_type="search_document"):
            print("EmbeddingService not available, returning empty embedding")
            return []

    embedding_service = MockEmbeddingService()