"""
Database connection and initialization for the RAG Chatbot system
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from ..config.settings import settings


# Create the database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections after 5 minutes
    echo=False           # Set to True for SQL debugging
)

# Create a configured "Session" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create a Base class for declarative models
Base = declarative_base()


def get_db():
    """
    Dependency function that provides database sessions for FastAPI endpoints.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables.
    """
    Base.metadata.create_all(bind=engine)


# Optional: Function to test database connection
def test_connection():
    """
    Test the database connection.
    """
    try:
        db = SessionLocal()
        # Try a simple query to test the connection
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception:
        return False