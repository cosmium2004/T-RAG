"""
Module 6.3 — Confidence Scorer for T-RAG Validator.
Calculates an overall confidence score for a T-RAG response.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Computes a composite confidence score for a generated response.

    Factors:
    - Consistency score from validation
    - Average FVS of source facts
    - Number of supporting facts
    - Average source confidence
    """

    def score(
        self,
        validation: Dict[str, Any],
        source_facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute composite confidence.

        Returns dict with ``confidence``, ``breakdown``, and ``rating``.
        """
        consistency = validation.get("consistency_score", 0.5)
        avg_fvs = self._avg(source_facts, "fvs", 0.5)
        avg_conf = self._avg(source_facts, "confidence", 0.5)
        coverage = min(len(source_facts) / 5.0, 1.0)  # normalise to max 5 facts

        # Weighted composite
        confidence = (
            0.35 * consistency
            + 0.25 * avg_fvs
            + 0.20 * avg_conf
            + 0.20 * coverage
        )

        rating = (
            "HIGH" if confidence >= 0.8 else
            "MEDIUM" if confidence >= 0.5 else
            "LOW"
        )

        return {
            "confidence": round(confidence, 4),
            "rating": rating,
            "breakdown": {
                "consistency": round(consistency, 4),
                "temporal_freshness": round(avg_fvs, 4),
                "source_confidence": round(avg_conf, 4),
                "fact_coverage": round(coverage, 4),
            },
        }

    @staticmethod
    def _avg(facts: List[Dict], key: str, default: float) -> float:
        vals = [f.get(key, default) for f in facts]
        return sum(vals) / max(len(vals), 1)
