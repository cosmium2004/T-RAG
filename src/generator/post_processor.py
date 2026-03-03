"""
Module 5.3 — Post-Processor for T-RAG Generator.
Cleans and formats LLM responses.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PostProcessor:
    """Cleans and normalises LLM-generated responses."""

    def process(self, raw_response: str) -> str:
        """Apply all post-processing steps."""
        text = raw_response.strip()
        text = self._remove_preamble(text)
        text = self._fix_whitespace(text)
        return text

    @staticmethod
    def _remove_preamble(text: str) -> str:
        """Remove common LLM preambles like 'Based on the context...'."""
        preambles = [
            r"^Based on the (provided )?context[,:]?\s*",
            r"^According to the (provided )?information[,:]?\s*",
            r"^Answer:\s*",
        ]
        for pattern in preambles:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _fix_whitespace(text: str) -> str:
        """Collapse multiple whitespace into single spaces."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
