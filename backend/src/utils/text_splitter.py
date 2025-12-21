from typing import List, Dict, Any
import re

class TextSplitter:
    """
    Utility class for splitting documents into chunks with overlap to maintain context
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap to maintain context
        Returns a list of dictionaries with 'content' and 'metadata' keys
        """
        if len(text) <= self.chunk_size:
            return [{"content": text, "metadata": {"start_index": 0, "end_index": len(text)}}]

        chunks = []
        start = 0

        while start < len(text):
            # Determine the end position
            end = start + self.chunk_size

            # If we're at the end, adjust the end position
            if end > len(text):
                end = len(text)

            # Extract the chunk
            chunk = text[start:end]

            # If we have overlap and we're not at the end, try to split at a sentence boundary
            if end < len(text) and self.chunk_overlap > 0:
                # Look for sentence endings in the overlap region
                overlap_text = text[end - self.chunk_overlap:end]
                sentence_endings = [m.end() for m in re.finditer(r'[.!?]+\s+', overlap_text)]

                if sentence_endings:
                    # Adjust end to the last sentence ending in the overlap
                    last_sentence_end = sentence_endings[-1]
                    end = end - self.chunk_overlap + last_sentence_end
                    chunk = text[start:end]

            chunks.append({
                "content": chunk,
                "metadata": {
                    "start_index": start,
                    "end_index": end
                }
            })

            # Move start position forward
            start = end

            # If we've reached the end, break
            if start >= len(text):
                break

        return chunks

def split_document_content(content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split document content into chunks with specified size and overlap
    """
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(content)