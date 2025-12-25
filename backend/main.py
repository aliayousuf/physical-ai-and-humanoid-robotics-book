from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.chat_api import router as chat_router
from src.api.ingestion_api import router as ingestion_router
from src.api.health_api import router as health_router
from src.api.search import router as search_router  # Added for semantic search functionality
from src.config.settings import settings, get_allowed_origins

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for application startup and shutdown
    """
    logger.info("Application starting up...")
    # Add any startup logic here
    yield
    # Add any shutdown logic here
    logger.info("Application shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Book Content Ingestion API",
    description="API for ingesting book content into the vector database for the RAG chatbot using Gemini 2.5 Flash model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(ingestion_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(search_router, prefix="/api/v1", tags=["search"])  # Added for semantic search functionality
app.include_router(health_router, prefix="/api/v1", tags=["health"])

@app.get("/")
async def root():
    return {"message": "Book Content Ingestion API is running"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Disable reload in production
    )