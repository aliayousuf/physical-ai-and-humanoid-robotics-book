from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # cohere_api_key: str = os.getenv("COHERE_API_KEY", "")  # Removed - using Gemini embeddings instead

    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./rag_chatbot.db")
    neon_postgres_connection_string: Optional[str] = os.getenv("NEON_POSTGRES_CONNECTION_STRING")

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content")

    # Application Configuration
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "info")
    log_file: Optional[str] = os.getenv("LOG_FILE")  # Path to log file, if desired
    session_expiration_hours: int = int(os.getenv("SESSION_EXPIRATION_HOURS", "24"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Rate Limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # in seconds

    # Session Configuration
    session_expiry_hours: int = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
    max_query_length: int = int(os.getenv("MAX_QUERY_LENGTH", "10000"))
    max_selected_text_length: int = int(os.getenv("MAX_SELECTED_TEXT_LENGTH", "2000"))

    # Vector Search Configuration
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "768"))  # Using Gemini embedding dimensions
    search_limit: int = int(os.getenv("SEARCH_LIMIT", "5"))
    search_threshold: float = float(os.getenv("SEARCH_THRESHOLD", "0.7"))

    # RAG Configuration
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    rag_score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.7"))
    max_response_tokens: int = int(os.getenv("MAX_RESPONSE_TOKENS", "1000"))

    # Model Configuration
    model_name: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")  # Using Gemini 2.5 Flash
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")  # Using Gemini 2.5 Flash
    gemini_embedding_model_name: str = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "text-embedding-004")  # Using embedding model

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single instance of settings
settings = Settings()