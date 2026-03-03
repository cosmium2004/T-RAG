"""
Module 4.5 — Context Assembler for T-RAG retriever.
Formats retrieved facts into a prompt-ready context string.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Formats top-k facts into a structured context for LLM prompts."""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    def format(
        self,
        facts: List[Dict[str, Any]],
        query_time: Optional[datetime] = None,
    ) -> str:
        """
        Build a numbered context string from ranked facts.

        Returns a string like:
            1. [2018-01-01 to present] Kim Jong-Un make statement ...
            2. [2018-02-15 to present] ...
        """
        lines = []
        for i, fact in enumerate(facts, 1):
            start = fact.get("start_time", "unknown")
            end = fact.get("end_time", "present")
            if end is None:
                end = "present"
            text = fact.get("text", "")
            source = fact.get("source", "unknown")
            conf = fact.get("confidence", 0)
            fvs = fact.get("fvs", 0)
            wrs = fact.get("wrs", 0)

            line = (
                f"{i}. [Valid: {start} to {end}] "
                f"{text} "
                f"(Source: {source}, Confidence: {conf:.0%}, "
                f"Freshness: {fvs:.2f}, Relevance: {wrs:.3f})"
            )
            lines.append(line)

        context = "\n".join(lines)

        # Rough token estimate (4 chars ≈ 1 token)
        est_tokens = len(context) // 4
        if est_tokens > self.max_tokens:
            logger.warning(
                f"Context may exceed token limit: ~{est_tokens} tokens "
                f"(max {self.max_tokens})"
            )

        return context
