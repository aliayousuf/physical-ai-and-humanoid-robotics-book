"""
Test script to verify the book content ingestion and chat functionality with real content from docs folder
"""
import os
import sys
from pathlib import Path

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from src.services.ingestion_service import ingestion_service
from src.services.chat_service import chat_service
from src.config.settings import settings

def test_with_real_docs():
    """
    Test the system with real documentation from the docs folder
    """
    print("Testing with real documentation from docs folder...")

    # Check if docs folder exists
    docs_path = settings.docs_path
    if not os.path.exists(docs_path):
        print(f"Docs folder does not exist: {docs_path}")
        print("Creating a sample docs folder with test content...")
        os.makedirs(docs_path, exist_ok=True)

        # Create a sample document
        sample_doc_path = os.path.join(docs_path, "sample_robotics.md")
        with open(sample_doc_path, 'w', encoding='utf-8') as f:
            f.write("""
# Introduction to Humanoid Robotics

Humanoid robotics is a branch of robotics that focuses on creating robots with human-like characteristics. These robots are designed to resemble the human body structure and often mimic human behavior.

## Key Components

- **Actuators**: Enable movement and control of the robot's joints
- **Sensors**: Allow the robot to perceive its environment
- **Control Systems**: Coordinate the robot's movements and actions
- **AI Systems**: Enable decision-making and learning capabilities

## Applications

Humanoid robots have various applications including:
- Healthcare assistance
- Customer service
- Education
- Research and development
- Entertainment

## Challenges

Creating effective humanoid robots faces several challenges:
- Balance and locomotion
- Natural language processing
- Emotional recognition
- Power efficiency
- Cost of production
            """)
        print(f"Created sample document: {sample_doc_path}")

    # List documents in the folder
    print(f"Scanning docs folder: {docs_path}")
    documents = ingestion_service.scan_and_get_documents()
    print(f"Found {len(documents)} documents to process:")
    for doc in documents:
        print(f"  - {doc}")

    if not documents:
        print("No documents found to process. Please add some documents to the docs folder.")
        return

    # Trigger ingestion
    print("\nTriggering ingestion process...")
    job_id = ingestion_service.trigger_ingestion(force_reprocess=False)
    print(f"Ingestion job started with ID: {job_id}")

    # Get job status
    job_status = ingestion_service.get_job_status(job_id)
    if job_status:
        print(f"Job status: {job_status.status}")
        print(f"Processed {job_status.processed_documents}/{job_status.total_documents} documents")
        if job_status.error_message:
            print(f"Error: {job_status.error_message}")

    # Test chat functionality
    print("\nTesting chat functionality...")

    test_questions = [
        "What is humanoid robotics?",
        "What are the key components of humanoid robots?",
        "What applications do humanoid robots have?",
        "What challenges exist in humanoid robotics?"
    ]

    for question in test_questions:
        print(f"\nQuery: {question}")
        response = chat_service.query_chat(
            query=question,
            max_results=3,
            similarity_threshold=0.1
        )

        print(f"Response: {response['response'][:200]}...")
        print(f"Sources: {len(response['sources'])}")
        print(f"Confidence: {response['confidence']:.2f}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_with_real_docs()