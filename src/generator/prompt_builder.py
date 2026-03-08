"""
Module 5.1 — Prompt Builder for T-RAG LLM Generator.
Creates time-aware prompts with structured context and conflict awareness.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds time-aware prompts for LLM generation."""

    SYSTEM_PROMPT = (
        "You are a geopolitical analysis assistant that provides temporally-accurate answers. "
        "You answer questions based ONLY on the retrieved factual context provided to you.\n\n"
        "Rules:\n"
        "1. ONLY use information from the provided context — never invent facts.\n"
        "2. Always cite dates when mentioning events (e.g., 'In January 2018, ...').\n"
        "3. If facts conflict, acknowledge both and prefer the most recent one.\n"
        "4. If a fact is marked [OUTDATED], note that it may no longer be current.\n"
        "5. If the context is insufficient, say: 'The available evidence does not cover this.'\n"
        "6. Use past tense for events with end dates, present tense for ongoing facts.\n"
        "7. Keep your answer concise but informative."
    )

    def build_prompt(
        self,
        query: str,
        context: str,
        query_time: Optional[datetime] = None,
    ) -> str:
        """Build a complete prompt with query, context, and instructions."""
        if query_time is None:
            query_time = datetime.now(timezone.utc)

        time_str = query_time.strftime("%Y-%m-%d %H:%M UTC")

        return f"""Question: {query}

Reference Time: {time_str}

Retrieved Context:
{context}

Instructions:
- Answer the question using ONLY the facts above.
- Cite dates and sources from the context.
- Facts marked [OUTDATED] may no longer be current — mention this caveat.
- If conflicting facts exist, discuss both and explain which is more recent.
- If context is insufficient, state that clearly.

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
