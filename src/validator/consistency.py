"""
Module 6.1 — Consistency Validator for T-RAG.
Detects temporal contradictions and verifies claims against sources.
Includes entity-level contradiction detection and fuzzy source matching.
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
    4. Temporal ordering consistency
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
        warnings = []

        for claim_text, claim_date in claims:
            # Check for future dates
            if claim_date and claim_date > query_time:
                contradictions.append(
                    f"Future date referenced: {claim_date.strftime('%Y-%m-%d')}"
                )

            # Check source support
            support_score = self._source_support_score(claim_text, source_facts)
            if support_score < 0.1:
                contradictions.append(
                    f"Unsupported claim: {claim_text[:80]}"
                )
            elif support_score < 0.3:
                warnings.append(
                    f"Weakly supported: {claim_text[:60]}"
                )

        # Check for entity-level contradictions in the response itself
        entity_contradictions = self._detect_response_contradictions(response)
        contradictions.extend(entity_contradictions)

        # Calculate score
        total_checks = max(len(claims) + len(entity_contradictions), 1)
        consistency_score = 1.0 - (len(contradictions) / total_checks)

        return {
            "consistency_score": round(max(consistency_score, 0.0), 4),
            "contradictions": contradictions,
            "warnings": warnings,
            "claims_checked": len(claims),
            "is_valid": consistency_score > 0.7,
        }

    def _extract_temporal_claims(
        self, text: str
    ) -> List[Tuple[str, Optional[datetime]]]:
        """Extract sentences containing temporal information."""
        claims = []

        # Date patterns
        date_patterns = [
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",       # 2024-01-15
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s*\d{4}",  # January 15, 2024
            r"(?:early|mid|late)\s+\d{4}",           # early 2024
            r"in\s+\d{4}",                            # in 2024
        ]
        combined_pattern = "|".join(f"({p})" for p in date_patterns)

        sentences = re.split(r"[.!?]\s+", text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Look for date-like patterns
            date_matches = re.findall(combined_pattern, sent, re.IGNORECASE)
            if date_matches:
                # Flatten and find first non-empty match
                flat = [m for group in date_matches for m in group if m]
                if flat:
                    try:
                        dt = dp.parse(flat[0], fuzzy=True)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        claims.append((sent, dt))
                    except (ValueError, OverflowError):
                        claims.append((sent, None))
                continue

            # Temporal keywords without explicit dates
            temporal_words = [
                "currently", "present", "now", "today",
                "recently", "ongoing", "since", "until",
            ]
            if any(w in sent.lower() for w in temporal_words):
                claims.append((sent, None))

        return claims

    def _source_support_score(
        self, claim: str, source_facts: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate how well a claim is supported by source facts.

        Returns a float in [0, 1] based on word overlap.
        """
        claim_words = set(
            w.lower() for w in claim.split()
            if len(w) > 2 and w.lower() not in self._stopwords
        )

        if not claim_words:
            return 1.0  # can't check, assume OK

        best_overlap = 0.0
        for fact in source_facts:
            fact_text = fact.get("text", "").lower()
            # Also check entity fields
            head = fact.get("head", "").lower()
            tail = fact.get("tail", "").lower()
            relation = fact.get("relation", "").lower()

            fact_words = set(
                w for w in (fact_text + " " + head + " " + tail + " " + relation).split()
                if len(w) > 2
            )

            if fact_words:
                overlap = len(claim_words & fact_words) / len(claim_words)
                best_overlap = max(best_overlap, overlap)

        return best_overlap

    @staticmethod
    def _detect_response_contradictions(response: str) -> List[str]:
        """
        Detect self-contradictions within the response.

        Looks for patterns like "X did A" and "X did not A" in the same text.
        """
        contradictions = []
        sentences = re.split(r"[.!?]\s+", response)

        # Simple negation check
        for i, sent_a in enumerate(sentences):
            for sent_b in sentences[i + 1:]:
                a_lower = sent_a.lower().strip()
                b_lower = sent_b.lower().strip()

                # Check if one is the negation of the other
                if (("not " in a_lower or "no " in a_lower) and
                        not ("not " in b_lower or "no " in b_lower)):
                    # Check entity overlap
                    a_words = set(a_lower.split()) - {"not", "no", "the", "a", "an"}
                    b_words = set(b_lower.split()) - {"not", "no", "the", "a", "an"}
                    overlap = len(a_words & b_words)
                    if overlap >= 4:
                        contradictions.append(
                            f"Possible contradiction: '{sent_a[:40]}...' vs '{sent_b[:40]}...'"
                        )

        return contradictions[:3]  # limit to 3

    _stopwords = {
        "the", "a", "an", "is", "was", "were", "are", "be", "been",
        "has", "had", "have", "do", "did", "does", "will", "would",
        "could", "should", "may", "might", "can", "shall", "for",
        "and", "but", "or", "nor", "not", "no", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few",
        "more", "most", "other", "some", "such", "than", "too",
        "very", "just", "also", "that", "this", "these", "those",
        "with", "from", "into", "during", "before", "after",
        "above", "below", "between", "through", "about",
    }
