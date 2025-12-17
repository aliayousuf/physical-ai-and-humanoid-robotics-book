#!/usr/bin/env python3
"""
Script to ingest all book content into Qdrant with rate limit handling
"""
import asyncio
import time
from pathlib import Path
from typing import List
import hashlib

from src.models.content import BookContent
from src.services.qdrant_service import qdrant_service
from src.services.embedding_service import embedding_service


async def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If this isn't the last chunk, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the end
            snippet = text[start:end]
            last_period = snippet.rfind('.')
            last_exclamation = snippet.rfind('!')
            last_question = snippet.rfind('?')
            last_sentence_end = max(last_period, last_exclamation, last_question)

            if last_sentence_end > chunk_size // 2:  # Only if it's reasonably close to the end
                end = start + last_sentence_end + 1

        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end

        # Ensure we're making progress
        if start == end:
            start += chunk_size

    return [chunk.strip() for chunk in chunks if chunk.strip()]


async def process_markdown_file(file_path: Path, relative_path: str) -> List[BookContent]:
    """
    Process a markdown file and extract content chunks
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple extraction - in a real implementation you might want to parse frontmatter, etc.
    title = file_path.stem  # Use filename as title

    # Chunk the content
    chunks = await chunk_text(content)

    book_contents = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) == 0:
            continue

        # Create a hash of the content
        content_hash = hashlib.sha256(chunk.encode()).hexdigest()

        book_content = BookContent(
            title=f"{title} - Part {i+1}" if len(chunks) > 1 else title,
            content=chunk,
            page_reference=relative_path,
            section=f"part_{i+1}" if len(chunks) > 1 else "main",
            hash=content_hash
        )

        book_contents.append(book_content)

    return book_contents


async def ingest_docs_with_retry(docs_dir: str = "../docs"):
    """
    Ingest all markdown files from the docs directory into the vector database
    with retry logic to handle API rate limits
    """
    print("Starting document ingestion with rate limit handling...")

    # Initialize the Qdrant collection
    await qdrant_service.initialize_collection()

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Docs directory {docs_dir} does not exist!")
        return

    # Find all markdown files
    md_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))

    total_processed = 0
    total_embeddings = 0

    print(f"Found {len(md_files)} markdown files to process")

    for i, md_file in enumerate(md_files):
        print(f"\nProcessing file {i+1}/{len(md_files)}: {md_file}")

        try:
            # Calculate relative path from docs directory
            relative_path = str(md_file.relative_to(docs_path))

            # Process the file
            book_contents = await process_markdown_file(md_file, relative_path)

            for j, book_content in enumerate(book_contents):
                success = False
                retry_count = 0
                max_retries = 5

                while not success and retry_count < max_retries:
                    try:
                        print(f"  Processing chunk {j+1}/{len(book_contents)}: '{book_content.title[:50]}...'")

                        # Generate embedding for the content
                        embedding = await embedding_service.embed_single_text(
                            book_content.content,
                            input_type="search_document"
                        )

                        # Store in Qdrant
                        embedding_id = await qdrant_service.store_embedding(
                            content_id=book_content.id,
                            embedding=embedding,
                            metadata={
                                "title": book_content.title,
                                "content": book_content.content[:200] + "..." if len(book_content.content) > 200 else book_content.content,  # Truncate for storage
                                "page_reference": book_content.page_reference,
                                "section": book_content.section,
                                "hash": book_content.hash
                            }
                        )

                        # Update the content with the embedding ID
                        book_content.embedding_id = embedding_id
                        total_embeddings += 1

                        success = True
                        print(f"    [SUCCESS] Stored embedding with ID: {embedding_id}")

                    except Exception as e:
                        retry_count += 1
                        if "rate limit" in str(e).lower() or "429" in str(e) or "Too Many Requests" in str(e):
                            wait_time = 2 ** retry_count  # Exponential backoff
                            print(f"    [WARN] Rate limit hit, waiting {wait_time}s before retry {retry_count}/{max_retries}")
                            time.sleep(wait_time)
                        else:
                            print(f"    [ERROR] Error processing chunk: {e}")
                            break  # Don't retry for non-rate-limit errors

                if not success:
                    print(f"    [ERROR] Failed to process chunk after {max_retries} retries")

            total_processed += len(book_contents)
            print(f"  Processed {len(book_contents)} content chunks from {md_file.name}")

        except Exception as e:
            print(f"  [ERROR] Error processing file {md_file}: {e}")
            continue

    print(f"\nIngestion complete! Processed {total_processed} content chunks with {total_embeddings} embeddings stored.")


if __name__ == "__main__":
    asyncio.run(ingest_docs_with_retry())