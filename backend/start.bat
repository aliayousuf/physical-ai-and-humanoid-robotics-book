@echo off
echo Starting RAG Chatbot backend service...

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)

REM Start the FastAPI application
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

pause