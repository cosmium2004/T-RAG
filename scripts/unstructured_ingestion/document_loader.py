"""
Module 1.6 — Document Loader for T-RAG pipeline.
Loads text from PDF, TXT, and URL sources for quadruple extraction.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads raw text from various document formats.

    Supported sources:
        - .txt / .md  — plain text files
        - .pdf        — PDF documents (via PyPDF2)
        - http(s)://  — web pages (via requests + BeautifulSoup)
    """

    def load(self, source: str) -> List[str]:
        """
        Auto-detect source format and load text.

        Args:
            source: File path or URL.

        Returns:
            List of text strings (one per page/section).
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self.load_url(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(source)
        else:
            # Treat everything else as plain text (.txt, .md, .csv, etc.)
            return self.load_text(source)

    def load_text(self, filepath: str) -> List[str]:
        """Load a plain text file as a single-element list."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = path.read_text(encoding="utf-8", errors="replace")
        logger.info(f"Loaded text file: {filepath} ({len(text)} chars)")
        return [text]

    def load_pdf(self, filepath: str) -> List[str]:
        """Load a PDF file, returning one string per page."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF loading. "
                "Install with: pip install PyPDF2"
            )

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        reader = PdfReader(filepath)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

        logger.info(
            f"Loaded PDF: {filepath} ({len(reader.pages)} pages, "
            f"{len(pages)} with text)"
        )
        return pages

    def load_url(self, url: str) -> List[str]:
        """Fetch a web page and extract readable text."""
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text from paragraphs and headings
        blocks = []
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "article"]):
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 20:  # Skip very short fragments
                blocks.append(text)

        if not blocks:
            # Fallback: get all text
            text = soup.get_text(separator="\n", strip=True)
            blocks = [text] if text else []

        combined = "\n\n".join(blocks)
        logger.info(f"Loaded URL: {url} ({len(combined)} chars)")
        return [combined]
