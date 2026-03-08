"""
Module 5.3 — Post-Processor for T-RAG Generator.
Cleans, normalises, and enriches LLM responses.
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class PostProcessor:
    """Cleans and enriches LLM-generated responses."""

    def process(
        self,
        raw_response: str,
        source_facts: Optional[List[dict]] = None,
    ) -> str:
        """Apply all post-processing steps."""
        text = raw_response.strip()
        text = self._remove_preamble(text)
        text = self._normalize_temporal_markers(text)
        text = self._fix_whitespace(text)

        if source_facts:
            text = self._add_confidence_note(text, source_facts)

        return text

    @staticmethod
    def _remove_preamble(text: str) -> str:
        """Remove common LLM preambles."""
        preambles = [
            r"^Based on the (provided |retrieved )?context[,:]?\s*",
            r"^According to the (provided |retrieved )?information[,:]?\s*",
            r"^From the (provided |retrieved )?facts[,:]?\s*",
            r"^Answer:\s*",
            r"^Here is my answer:\s*",
            r"^Sure[,!]?\s*",
        ]
        for pattern in preambles:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _normalize_temporal_markers(text: str) -> str:
        """Normalize date formats and temporal language."""
        # Convert ISO-like timestamps to readable dates
        text = re.sub(
            r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}[+\-Z][\d:]*",
            r"\1",
            text,
        )
        # Normalize "to present" variations
        text = re.sub(
            r"to\s*(the\s*)?present(\s*day)?",
            "to present",
            text,
            flags=re.IGNORECASE,
        )
        return text

    @staticmethod
    def _fix_whitespace(text: str) -> str:
        """Collapse multiple whitespace into single spaces."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _add_confidence_note(
        text: str, source_facts: List[dict]
    ) -> str:
        """Append a brief confidence note based on source quality."""
        if not source_facts:
            return text

        avg_fvs = sum(f.get("fvs", 0) for f in source_facts) / len(source_facts)
        avg_conf = sum(f.get("confidence", 0) for f in source_facts) / len(source_facts)
        n_facts = len(source_facts)

        if avg_fvs < 0.3 or n_facts < 2:
            note = (
                "\n\n---\n"
                f"⚠️ *Caution: This answer is based on {n_facts} source(s) "
                f"with average freshness {avg_fvs:.0%}. "
                "Some information may be outdated.*"
            )
            return text + note

        return text
