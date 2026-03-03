"""
Module 4.3 — Temporal Filter for T-RAG retriever.
Filters facts based on temporal validity at query time.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dateutil import parser as dp

logger = logging.getLogger(__name__)


class TemporalFilter:
    """Removes facts that are not temporally valid at query time."""

    def __init__(self, deprecation_threshold: float = 0.3):
        self.deprecation_threshold = deprecation_threshold

    def filter(
        self,
        facts: List[Dict[str, Any]],
        query_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Keep only facts valid at ``query_time``.

        Checks:
        1. start_time <= query_time (fact has started)
        2. end_time is None OR end_time >= query_time (fact hasn't ended)
        3. FVS >= threshold (if fvs key present)
        """
        if query_time is None:
            query_time = datetime.now(timezone.utc)

        valid = []
        for fact in facts:
            if self._is_valid(fact, query_time):
                valid.append(fact)

        logger.debug(
            f"Temporal filter: {len(facts)} -> {len(valid)} "
            f"(query_time={query_time.isoformat()})"
        )
        return valid

    def _is_valid(self, fact: Dict[str, Any], query_time: datetime) -> bool:
        start = fact.get("start_time")
        end = fact.get("end_time")

        if start:
            st = dp.parse(start) if isinstance(start, str) else start
            if st > query_time:
                return False

        if end:
            et = dp.parse(end) if isinstance(end, str) else end
            if et < query_time:
                return False

        # Check FVS if present
        fvs = fact.get("fvs")
        if fvs is not None and fvs < self.deprecation_threshold:
            return False

        return True
