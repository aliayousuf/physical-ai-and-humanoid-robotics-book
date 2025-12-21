from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import uuid
from pathlib import Path
import asyncio

from src.models.ingestion_job import IngestionJob, IngestionJobStatus
from src.models.document import Document, DocumentStatus
from src.models.document_chunk import DocumentChunk
from src.utils.file_parser import parse_file, scan_docs_folder
from src.utils.text_splitter import split_document_content
from src.services.embedding_service import embedding_service
from src.services.vector_db_service import vector_db_service
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Service for processing documents from the docs folder and ingesting them into the vector database
    """
    def __init__(self):
        self.jobs: Dict[str, IngestionJob] = {}

    def trigger_ingestion(self, force_reprocess: bool = False, file_patterns: List[str] = None) -> str:
        """
        Trigger the ingestion process for documents in the docs folder
        Returns the job ID for tracking the ingestion process
        NOTE: This starts the ingestion process in a background thread
        """
        import asyncio
        import threading

        if file_patterns is None:
            file_patterns = ["*.md", "*.pdf", "*.txt"]

        # Create a new ingestion job
        job_id = str(uuid.uuid4())
        files = scan_docs_folder(settings.docs_path, file_patterns)

        job = IngestionJob(
            id=job_id,
            status=IngestionJobStatus.queued,
            total_documents=len(files),
            processed_documents=0,
            started_at=datetime.now()
        )

        self.jobs[job_id] = job
        logger.info(f"Created ingestion job {job_id} for {len(files)} documents")

        # Start processing in a background thread
        thread = threading.Thread(
            target=self._process_ingestion_in_background,
            args=(job_id, files, force_reprocess)
        )
        thread.daemon = True  # Dies when main process dies
        thread.start()

        return job_id

    def _process_ingestion_in_background(self, job_id: str, files: List[str], force_reprocess: bool):
        """
        Process ingestion in a background thread
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found in background processing")
            return

        # Process documents
        job.status = IngestionJobStatus.running
        processed_count = 0

        try:
            for file_path in files:
                # For background processing, we'll create a new event loop
                import asyncio
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)

                try:
                    # Run the async function in the new event loop
                    success = new_loop.run_until_complete(
                        self._process_single_document(file_path, force_reprocess)
                    )
                    if success:
                        processed_count += 1
                finally:
                    new_loop.close()

                # Update job progress
                job.processed_documents = processed_count
                self.jobs[job_id] = job

            job.status = IngestionJobStatus.completed
            job.completed_at = datetime.now()
            self.jobs[job_id] = job

            logger.info(f"Ingestion job {job_id} completed successfully. Processed {processed_count}/{len(files)} documents")
        except Exception as e:
            job.status = IngestionJobStatus.failed
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self.jobs[job_id] = job

            logger.error(f"Ingestion job {job_id} failed: {str(e)}")

    async def _process_single_document(self, file_path: str, force_reprocess: bool) -> bool:
        """
        Process a single document: parse, chunk, embed, and store
        """
        try:
            # Create document record
            doc_id = str(uuid.uuid4())
            file_stat = os.stat(file_path)

            document = Document(
                id=doc_id,
                filename=Path(file_path).name,
                filepath=file_path,
                format=Path(file_path).suffix.lower().strip('.'),
                size=file_stat.st_size,
                created_at=datetime.fromtimestamp(file_stat.st_ctime),
                updated_at=datetime.fromtimestamp(file_stat.st_mtime),
                status=DocumentStatus.processing
            )

            logger.info(f"Processing document: {document.filename}")

            # Parse the document
            content = parse_file(file_path)

            # Split content into chunks
            chunks_data = split_document_content(
                content,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )

            # Remove old chunks from vector database if reprocessing
            if force_reprocess:
                vector_db_service.delete_document_chunks(doc_id)

            # Process each chunk
            for i, chunk_data in enumerate(chunks_data):
                # Create document chunk with proper UUID
                chunk_uuid = str(uuid.uuid4())
                chunk = DocumentChunk(
                    id=chunk_uuid,
                    document_id=doc_id,
                    chunk_index=i,
                    content=chunk_data["content"],
                    metadata=chunk_data.get("metadata", {}),
                    created_at=datetime.now()
                )

                # Generate embedding for the chunk
                embedding = await embedding_service.embed_single_text(chunk.content)

                # Store in vector database
                success = vector_db_service.store_embedding(chunk, embedding)
                if not success:
                    logger.error(f"Failed to store embedding for chunk {chunk_id}")
                    document.status = DocumentStatus.failed
                    return False

            document.status = DocumentStatus.completed
            logger.info(f"Successfully processed document: {document.filename}")
            return True
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False

    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """
        Get the status of an ingestion job
        """
        return self.jobs.get(job_id)

    def scan_and_get_documents(self, file_patterns: List[str] = None) -> List[str]:
        """
        Scan the docs folder for supported documents
        """
        if file_patterns is None:
            file_patterns = ["*.md", "*.pdf", "*.txt"]

        return scan_docs_folder(settings.docs_path, file_patterns)

    def get_processed_documents(self) -> List[str]:
        """
        Get all document IDs that have been processed and stored in the vector database
        """
        return vector_db_service.get_all_document_ids()

# Create a singleton instance
ingestion_service = IngestionService()