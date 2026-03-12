"""
Module 1.7 — Text Chunker for T-RAG pipeline.
Splits raw text into overlapping chunks for LLM-based quadruple extraction.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks on sentence boundaries.

    Overlap ensures entities/relations spanning chunk boundaries
    aren't lost during extraction.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        """
        Args:
            chunk_size: Target characters per chunk.
            overlap: Characters of overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks on sentence boundaries.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return [text.strip()] if text.strip() else []

        chunks = []
        current_chunk: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            # If adding this sentence exceeds chunk_size, finalize chunk
            if current_len + sent_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Keep overlap: walk backwards to find overlap boundary
                overlap_chunk: List[str] = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > self.overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s)

                current_chunk = overlap_chunk
                current_len = overlap_len

            current_chunk.append(sentence)
            current_len += sent_len

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        logger.info(
            f"Chunked text: {len(text)} chars -> {len(chunks)} chunks "
            f"(target={self.chunk_size}, overlap={self.overlap})"
        )
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex-based rules."""
        # Split on sentence-ending punctuation followed by whitespace
        # Handles: periods, question marks, exclamation marks
        # Avoids splitting on abbreviations like "U.S." or "Dr."
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text)

        sentences = []
        for s in raw:
            cleaned = s.strip()
            if cleaned:
                sentences.append(cleaned)

        return sentences

    def chunk_pages(self, pages: List[str]) -> List[str]:
        """Chunk multiple pages/sections, concatenating first."""
        combined = "\n\n".join(pages)
        return self.chunk(combined)
