"""
Module 4.4 — Weighted Relevance Score (WRS) for T-RAG retriever.

WRS = α × Sim(q, d) + (1-α) × FVS

Combines semantic similarity with temporal freshness.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class WRSScorer:
    """Computes Weighted Relevance Score for retrieved facts."""

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for semantic similarity (0-1).
                   Higher = prioritise relevance over freshness.
        """
        self.alpha = alpha

    def score(self, fact: Dict[str, Any]) -> float:
        """Compute WRS for a single fact (must have 'similarity' and 'fvs')."""
        sim = fact.get("similarity", 0.0)
        fvs = fact.get("fvs", 0.5)
        return self.alpha * sim + (1 - self.alpha) * fvs

    def rank(
        self,
        facts: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Score and rank facts, return top-k."""
        for fact in facts:
            fact["wrs"] = self.score(fact)

        ranked = sorted(facts, key=lambda f: f["wrs"], reverse=True)
        return ranked[:top_k]
