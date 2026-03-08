"""
Module 4.5 — Context Assembler for T-RAG retriever.
Formats retrieved facts into a prompt-ready context string.
Includes entity grouping, conflict detection, and token-aware truncation.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Formats top-k facts into a structured context for LLM prompts."""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self._conflicts: List[str] = []

    def format(
        self,
        facts: List[Dict[str, Any]],
        query_time: Optional[datetime] = None,
    ) -> str:
        """
        Build a structured context string from ranked facts.

        Groups facts by entity, flags temporal conflicts, and
        truncates to stay within the token budget.
        """
        self._conflicts = []

        if not facts:
            return "(No relevant facts found.)"

        # Detect conflicts before formatting
        self._detect_conflicts(facts)

        # Build fact lines
        lines = []
        for i, fact in enumerate(facts, 1):
            line = self._format_fact(i, fact)
            lines.append(line)

        context_parts = []

        # Add entity summary header
        entity_summary = self._entity_summary(facts)
        if entity_summary:
            context_parts.append(f"Entities mentioned: {entity_summary}")

        # Add conflict warnings
        if self._conflicts:
            warnings = "; ".join(self._conflicts[:3])
            context_parts.append(f"⚠ Potential conflicts: {warnings}")

        context_parts.append("")  # blank line
        context_parts.extend(lines)

        context = "\n".join(context_parts)

        # Token-aware truncation
        context = self._truncate(context)

        return context

    @property
    def conflicts(self) -> List[str]:
        """Return any temporal conflicts detected during last format call."""
        return self._conflicts

    def _format_fact(self, index: int, fact: Dict[str, Any]) -> str:
        """Format a single fact into a context line."""
        start = fact.get("start_time", "unknown")
        end = fact.get("end_time", "present")
        if end is None:
            end = "present"

        # Trim ISO timestamps to dates for readability
        if isinstance(start, str) and "T" in start:
            start = start[:10]
        if isinstance(end, str) and "T" in end:
            end = end[:10]

        head = fact.get("head", "")
        relation = fact.get("relation", "")
        tail = fact.get("tail", "")
        text = fact.get("text", f"{head} {relation} {tail}".strip())

        source = fact.get("source", "unknown")
        conf = fact.get("confidence", 0)
        fvs = fact.get("fvs", 0)
        wrs = fact.get("wrs", 0)

        deprecated = " [OUTDATED]" if not fact.get("is_valid", True) else ""

        return (
            f"{index}. [{start} → {end}]{deprecated} "
            f"{text} "
            f"(Source: {source}, Conf: {conf:.0%}, "
            f"Fresh: {fvs:.2f}, WRS: {wrs:.3f})"
        )

    def _detect_conflicts(self, facts: List[Dict[str, Any]]) -> None:
        """Detect contradictory facts about the same entity pair."""
        # Group by (head, tail)
        pairs: Dict[tuple, List[Dict]] = defaultdict(list)
        for fact in facts:
            h = fact.get("head", "").lower()
            t = fact.get("tail", "").lower()
            if h and t:
                key = tuple(sorted([h, t]))
                pairs[key].append(fact)

        for pair_key, pair_facts in pairs.items():
            if len(pair_facts) < 2:
                continue

            # Check for conflicting relations on the same entity pair
            relations = set()
            for f in pair_facts:
                rel = f.get("relation", "").lower()
                if rel:
                    relations.add(rel)

            if len(relations) > 1:
                entities = " & ".join(w.title() for w in pair_key)
                self._conflicts.append(
                    f"{entities}: {', '.join(relations)}"
                )

    @staticmethod
    def _entity_summary(facts: List[Dict[str, Any]]) -> str:
        """Build a brief summary of entities in the context."""
        entities = set()
        for fact in facts:
            h = fact.get("head", "")
            t = fact.get("tail", "")
            if h:
                entities.add(h)
            if t:
                entities.add(t)

        if not entities:
            return ""

        # Show up to 8 entities
        sorted_ents = sorted(entities)
        if len(sorted_ents) > 8:
            return ", ".join(sorted_ents[:8]) + f" (+{len(sorted_ents) - 8} more)"
        return ", ".join(sorted_ents)

    def _truncate(self, context: str) -> str:
        """Truncate context to fit within token budget."""
        est_tokens = len(context) // 4
        if est_tokens <= self.max_tokens:
            return context

        # Truncate line by line from the end
        lines = context.split("\n")
        while est_tokens > self.max_tokens and len(lines) > 3:
            lines.pop()
            est_tokens = len("\n".join(lines)) // 4

        logger.warning(
            f"Context truncated to ~{est_tokens} tokens "
            f"(limit: {self.max_tokens})"
        )
        return "\n".join(lines) + "\n[... truncated]"
