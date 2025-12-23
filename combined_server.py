from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from pathlib import Path
import logging

# Import the main backend app
from backend.main import app as backend_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the main combined app
main_app = FastAPI(
    title="Physical AI & Humanoid Robotics Platform",
    description="Combined frontend and backend server",
    version="1.0.0"
)

# Mount the backend API routes under /api/v1
main_app.include_router(backend_app.router, prefix="/api/v1")

# Serve static files from the build directory
static_dir = Path(__file__).parent / "build"

if static_dir.exists():
    main_app.mount("/static", StaticFiles(directory=static_dir / "static"), name="static")

    @main_app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        # Serve the main index.html file
        index_path = static_dir / "index.html"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            return HTMLResponse(content="<h1>Build directory not found</h1>")

    @main_app.get("/{full_path:path}")
    async def serve_static(request: Request, full_path: str):
        # Try to serve the requested path from static files
        file_path = static_dir / full_path

        # If it's a directory, try to serve index.html from that directory
        if file_path.is_dir():
            dir_index = file_path / "index.html"
            if dir_index.exists():
                with open(dir_index, 'r', encoding='utf-8') as f:
                    content = f.read()
                return HTMLResponse(content=content)

        # If it's a file, serve it directly
        if file_path.exists() and file_path.is_file():
            # Determine content type based on file extension
            if full_path.endswith('.js'):
                from fastapi.responses import Response
                return Response(content=file_path.read_text(encoding='utf-8'), media_type='application/javascript')
            elif full_path.endswith('.css'):
                from fastapi.responses import Response
                return Response(content=file_path.read_text(encoding='utf-8'), media_type='text/css')
            elif full_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico')):
                from fastapi.responses import FileResponse
                return FileResponse(file_path)
            else:
                from fastapi.responses import Response
                return Response(content=file_path.read_text(encoding='utf-8'), media_type='text/html')

        # If file doesn't exist, serve the main index.html (for client-side routing)
        index_path = static_dir / "index.html"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            return HTMLResponse(content="<h1>Page not found</h1>", status_code=404)
else:
    @main_app.get("/")
    async def read_root():
        return {"message": "Build directory not found. Please run 'npm run build' first."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting combined server on port {port}")
    uvicorn.run(main_app, host="0.0.0.0", port=port)