from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.requests import Request
import time

from .chat import router as chat_router
from .chat_api import router as chat_api_router  # Added for RAG chat functionality
from .health import router as health_router
from .ingestion import router as ingestion_router  # Added for book content ingestion
from .search import router as search_router  # Added for semantic search functionality
from ..middleware.rate_limit import rate_limit_middleware
from ..utils.logging import logger, log_security_event


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses
    """
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"  # or "SAMEORIGIN" if you need frames
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';"

        return response


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log security-related events
    """
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Log potential security issues
        user_agent = request.headers.get("user-agent", "")
        if "sqlmap" in user_agent.lower() or "nikto" in user_agent.lower() or "nessus" in user_agent.lower():
            log_security_event(
                "Potential security scanner detected",
                {"user_agent": user_agent, "path": str(request.url.path), "method": request.method},
                "MEDIUM"
            )

        # Log requests with potential malicious content
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
                if isinstance(body, dict):
                    for key, value in body.items():
                        if isinstance(value, str):
                            if "<script" in value.lower() or "javascript:" in value.lower():
                                log_security_event(
                                    "Potential XSS attempt detected",
                                    {"field": key, "path": str(request.url.path), "method": request.method},
                                    "HIGH"
                                )
                            elif "union select" in value.lower() or "drop table" in value.lower():
                                log_security_event(
                                    "Potential SQL injection detected",
                                    {"field": key, "path": str(request.url.path), "method": request.method},
                                    "HIGH"
                                )
            except:
                # If we can't parse the body, that's okay - just continue
                pass

        return response


app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG chatbot integrated with Physical AI and Humanoid Robotics documentation",
    version="1.0.0"
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SecurityLoggingMiddleware)

# Add custom rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Add CORS middleware for Docusaurus integration
# Allow environment-specific origins
import os
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8080",
    "https://your-docusaurus-site.com"
]

# Add production frontend URL from environment variable if available
production_frontend_url = os.getenv("FRONTEND_URL")
if production_frontend_url:
    allowed_origins.append(production_frontend_url.rstrip('/'))  # Remove trailing slash if present
else:
    # Default production URLs - update these to match your actual deployments
    allowed_origins.extend([
        "https://physical-ai-and-humanoid-robotics-b-chi.vercel.app",
        "https://physical-ai-and-humanoid-robotics-book.vercel.app"  # common Vercel deployment URL
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(chat_api_router, prefix="/api/v1", tags=["chat_api"])  # Added for RAG chat functionality
app.include_router(ingestion_router, prefix="/api/v1", tags=["ingestion"])  # Added for book content ingestion
app.include_router(search_router, prefix="/api/v1", tags=["search"])  # Added for semantic search functionality
app.include_router(health_router, prefix="/api/v1", tags=["health"])

# Log startup
logger.info("RAG Chatbot API started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)