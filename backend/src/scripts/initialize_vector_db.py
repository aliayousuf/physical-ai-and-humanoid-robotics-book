import asyncio
from ..services.qdrant_service import qdrant_service


async def initialize_vector_db():
    """
    Initialize the vector database by creating the required collections
    """
    print("Initializing vector database...")

    try:
        await qdrant_service.initialize_collection()
        print("Vector database initialized successfully!")
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(initialize_vector_db())