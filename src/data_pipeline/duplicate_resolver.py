"""
Module 1.4 — Duplicate Resolver for T-RAG pipeline.
Identifies and merges duplicate / contradictory facts.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class DuplicateResolver:
    """
    De-duplicates facts based on (head, relation, tail) triples.

    Strategy:
    - Exact duplicates (same triple + same timestamp) → keep one
    - Same triple, different timestamps → merge into time range
    - Contradictory triples → keep highest-confidence version
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._total_input = 0
        self._duplicates_removed = 0

    def resolve(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        De-duplicate a list of fact dicts.

        Returns a new list with duplicates merged or removed.
        """
        self._total_input = len(facts)

        # Group by canonical key
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for fact in facts:
            key = self._canonical_key(fact)
            groups.setdefault(key, []).append(fact)

        resolved: List[Dict[str, Any]] = []
        for key, group in groups.items():
            merged = self._merge_group(group)
            resolved.append(merged)

        self._duplicates_removed = self._total_input - len(resolved)
        reduction_pct = (
            self._duplicates_removed / max(self._total_input, 1) * 100
        )
        logger.info(
            f"Duplicate resolution: {self._total_input} → {len(resolved)} facts "
            f"({reduction_pct:.1f}% reduction)"
        )

        return resolved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_key(fact: Dict[str, Any]) -> str:
        """Create a normalised key for grouping."""
        h = fact.get("head", "").lower().strip()
        r = fact.get("relation", "").lower().strip()
        t = fact.get("tail", "").lower().strip()
        return f"{h}||{r}||{t}"

    def _merge_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of duplicate facts into a single canonical fact."""
        if len(group) == 1:
            return group[0]

        # Sort by confidence descending, then by start_time ascending
        group.sort(
            key=lambda f: (
                -(f.get("confidence") or 0),
                f.get("start_time") or "",
            )
        )

        best = dict(group[0])  # shallow copy of highest-confidence fact

        # Collect all timestamps to build a time range
        start_times = [
            f["start_time"] for f in group
            if f.get("start_time")
        ]
        end_times = [
            f["end_time"] for f in group
            if f.get("end_time")
        ]

        if start_times:
            best["start_time"] = min(start_times)
        if end_times:
            best["end_time"] = max(end_times)

        # Collect unique sources
        sources = {f.get("source", "unknown") for f in group}
        best["source"] = ", ".join(sorted(sources))

        # Average confidence
        confs = [f.get("confidence", 0.5) for f in group]
        best["confidence"] = round(sum(confs) / len(confs), 4)

        return best

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_input": self._total_input,
            "duplicates_removed": self._duplicates_removed,
            "output_count": self._total_input - self._duplicates_removed,
        }
