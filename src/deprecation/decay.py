"""
Module 3.1 — Decay Function (FVS) for T-RAG Deprecation Detection.
Calculates Fact Validity Score using half-life exponential decay.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DecayFunction:
    """
    Computes Fact Validity Score (FVS) using exponential decay.

    FVS = exp(-λ × Δt)

    Where:
        λ = decay rate (higher = faster deprecation)
        Δt = days since last verification
    """

    def __init__(self, default_lambda: float = 0.01):
        self.default_lambda = default_lambda

    def calculate_fvs(
        self,
        last_verified: datetime,
        current_time: Optional[datetime] = None,
        lambda_val: Optional[float] = None,
    ) -> float:
        """
        Calculate Fact Validity Score.

        Returns a float in [0, 1] where 1 = perfectly fresh.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        if lambda_val is None:
            lambda_val = self.default_lambda

        delta_days = max((current_time - last_verified).total_seconds() / 86400, 0)
        fvs = float(np.exp(-lambda_val * delta_days))
        return round(fvs, 6)

    def is_valid(
        self,
        last_verified: datetime,
        current_time: Optional[datetime] = None,
        threshold: float = 0.3,
        lambda_val: Optional[float] = None,
    ) -> bool:
        """Check if a fact is still valid (FVS >= threshold)."""
        return self.calculate_fvs(last_verified, current_time, lambda_val) >= threshold

    def score_facts(
        self,
        facts: list,
        query_time: Optional[datetime] = None,
        threshold: float = 0.3,
    ) -> list:
        """Add FVS score and validity flag to each fact dict."""
        from dateutil import parser as dp

        if query_time is None:
            query_time = datetime.now(timezone.utc)

        for fact in facts:
            lv = fact.get("last_verified")
            if lv:
                if isinstance(lv, str):
                    lv = dp.parse(lv)
                fact["fvs"] = self.calculate_fvs(lv, query_time)
                fact["is_valid"] = fact["fvs"] >= threshold
            else:
                fact["fvs"] = 0.0
                fact["is_valid"] = False
        return facts
