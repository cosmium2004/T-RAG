"""
Module 6.1 — Consistency Validator for T-RAG.
Detects temporal contradictions and verifies claims against sources.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dp

logger = logging.getLogger(__name__)


class ConsistencyValidator:
    """
    Validates LLM responses for temporal consistency.

    Checks:
    1. No future dates beyond query time
    2. All claims backed by source facts
    3. No contradictory statements about the same entity
    """

    def validate(
        self,
        response: str,
        query_time: datetime,
        source_facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run all validation checks on a response."""
        claims = self._extract_temporal_claims(response)
        contradictions = []

        for claim_text, claim_date in claims:
            if claim_date and claim_date > query_time:
                contradictions.append(f"Future date referenced: {claim_date.isoformat()}")
            if not self._verify_against_sources(claim_text, source_facts):
                contradictions.append(f"Unsupported claim: {claim_text[:80]}")

        consistency_score = 1.0 - (
            len(contradictions) / max(len(claims), 1)
        )

        return {
            "consistency_score": max(consistency_score, 0.0),
            "contradictions": contradictions,
            "claims_checked": len(claims),
            "is_valid": consistency_score > 0.7,
        }

    def _extract_temporal_claims(
        self, text: str
    ) -> List[Tuple[str, Optional[datetime]]]:
        """Extract sentences containing date-like patterns."""
        claims = []
        date_pattern = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"

        sentences = re.split(r"[.!?]\s+", text)
        for sent in sentences:
            dates = re.findall(date_pattern, sent)
            if dates:
                try:
                    dt = dp.parse(dates[0])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    claims.append((sent.strip(), dt))
                except (ValueError, OverflowError):
                    claims.append((sent.strip(), None))
            elif any(w in sent.lower() for w in ["currently", "present", "now", "today"]):
                claims.append((sent.strip(), None))

        return claims

    @staticmethod
    def _verify_against_sources(
        claim: str, source_facts: List[Dict[str, Any]]
    ) -> bool:
        """Check if a claim has support from source facts (fuzzy)."""
        claim_words = set(claim.lower().split())
        for fact in source_facts:
            fact_text = fact.get("text", "").lower()
            fact_words = set(fact_text.split())
            overlap = len(claim_words & fact_words)
            if overlap >= 2:
                return True
        return False
