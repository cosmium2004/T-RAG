"""
Module 5.1 — Prompt Builder for T-RAG LLM Generator.
Creates time-aware prompts with structured context.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds time-aware prompts for LLM generation."""

    SYSTEM_PROMPT = (
        "You are a helpful assistant specialising in temporally-accurate information. "
        "Answer based ONLY on the provided context. "
        "Always cite timestamps when stating facts. "
        "If the context is insufficient, say so honestly."
    )

    def build_prompt(
        self,
        query: str,
        context: str,
        query_time: Optional[datetime] = None,
    ) -> str:
        """Build a complete prompt with query, context, and instructions."""
        if query_time is None:
            query_time = datetime.utcnow()

        return f"""Question: {query}

Context (valid as of {query_time.strftime('%Y-%m-%d %H:%M UTC')}):
{context}

Instructions:
- Use ONLY the provided context to answer
- Prioritise facts marked "to present" (most current)
- If conflicting facts exist, prefer the most recent
- State "I don't have current information" if context is insufficient
- Always include source citations with timestamps
- Use appropriate tenses (past events in past tense)

Answer:"""

    def build_messages(
        self,
        query: str,
        context: str,
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, str]]:
        """Build chat-style messages for the LLM API."""
        user_prompt = self.build_prompt(query, context, query_time)
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
