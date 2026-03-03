"""
Module 3.3 — Deprecation Classifier for T-RAG.
Binary classifier: determines if a fact is valid or deprecated.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.deprecation.decay import DecayFunction

logger = logging.getLogger(__name__)


class DeprecationClassifier:
    """
    Classifies facts as VALID or DEPRECATED based on FVS threshold
    and temporal validity window.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        default_lambda: float = 0.01,
    ):
        self.threshold = threshold
        self.decay = DecayFunction(default_lambda=default_lambda)

    def classify(
        self,
        fact: Dict[str, Any],
        query_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Classify a single fact. Adds ``deprecated``, ``fvs``,
        and ``deprecation_reason`` keys.
        """
        from dateutil import parser as dp

        if query_time is None:
            query_time = datetime.now(timezone.utc)

        result = dict(fact)
        reasons = []

        # Check temporal window
        start = fact.get("start_time")
        end = fact.get("end_time")

        if start:
            st = dp.parse(start) if isinstance(start, str) else start
            if st > query_time:
                reasons.append(f"Fact starts in the future ({start})")

        if end:
            et = dp.parse(end) if isinstance(end, str) else end
            if et < query_time:
                reasons.append(f"Fact ended ({end})")

        # Check FVS
        lv = fact.get("last_verified")
        if lv:
            lv_dt = dp.parse(lv) if isinstance(lv, str) else lv
            fvs = self.decay.calculate_fvs(lv_dt, query_time)
            result["fvs"] = fvs
            if fvs < self.threshold:
                reasons.append(f"FVS too low ({fvs:.4f} < {self.threshold})")
        else:
            result["fvs"] = 0.0
            reasons.append("No verification date")

        result["deprecated"] = len(reasons) > 0
        result["deprecation_reason"] = "; ".join(reasons) if reasons else None
        return result

    def classify_batch(
        self,
        facts: List[Dict[str, Any]],
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a list of facts."""
        return [self.classify(f, query_time) for f in facts]

    def filter_valid(
        self,
        facts: List[Dict[str, Any]],
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return only non-deprecated facts."""
        classified = self.classify_batch(facts, query_time)
        valid = [f for f in classified if not f["deprecated"]]
        logger.info(f"Deprecation filter: {len(facts)} -> {len(valid)} valid facts")
        return valid
