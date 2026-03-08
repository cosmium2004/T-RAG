"""
Module 3.1 — Decay Function (FVS) for T-RAG Deprecation Detection.
Calculates Fact Validity Score using half-life exponential decay.
Supports per-relation learned decay rates via MLE.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RelationDecayRates:
    """
    Learns per-relation decay rates (λ) from fact update intervals.

    Uses MLE: for exponentially distributed inter-update times,
    the MLE for λ is simply 1 / mean(Δt).
    """

    def __init__(self, rates_path: str = "data/cache/decay_rates.json"):
        self.rates_path = Path(rates_path)
        self.rates: Dict[str, float] = {}
        if self.rates_path.exists():
            self._load()

    def learn_rates(
        self, facts: List[Dict[str, Any]], default_lambda: float = 0.01
    ) -> Dict[str, float]:
        """
        Learn λ per relation type from fact timestamps.

        Groups facts by relation, computes mean inter-update interval,
        then sets λ = 1 / mean_interval_days.

        Args:
            facts: List of fact dicts with 'relation' and timestamp fields.
            default_lambda: Fallback λ for relations with insufficient data.

        Returns:
            Dict mapping relation -> learned λ.
        """
        from dateutil import parser as dp

        # Group timestamps by relation
        relation_times: Dict[str, List[datetime]] = defaultdict(list)
        for fact in facts:
            rel = fact.get("relation", "unknown").lower().strip()
            ts = fact.get("start_time") or fact.get("last_verified")
            if ts:
                try:
                    if isinstance(ts, str):
                        ts = dp.parse(ts)
                    relation_times[rel].append(ts)
                except (ValueError, OverflowError):
                    continue

        # Compute λ per relation
        self.rates = {}
        for rel, times in relation_times.items():
            if len(times) < 3:
                # Not enough data points for meaningful estimation
                self.rates[rel] = default_lambda
                continue

            times.sort()
            intervals = []
            for i in range(1, len(times)):
                delta = (times[i] - times[i - 1]).total_seconds() / 86400
                if delta > 0:
                    intervals.append(delta)

            if intervals:
                mean_interval = np.mean(intervals)
                # λ = 1 / mean_interval (MLE for exponential distribution)
                # Clamp to reasonable range [0.001, 1.0]
                learned_lambda = 1.0 / max(mean_interval, 1.0)
                self.rates[rel] = round(
                    float(np.clip(learned_lambda, 0.001, 1.0)), 6
                )
            else:
                self.rates[rel] = default_lambda

        self._save()
        logger.info(
            f"Learned decay rates for {len(self.rates)} relations "
            f"(range: {min(self.rates.values()):.4f} - "
            f"{max(self.rates.values()):.4f})"
        )
        return self.rates

    def get_lambda(self, relation: str, default: float = 0.01) -> float:
        """Look up λ for a relation, falling back to default."""
        return self.rates.get(relation.lower().strip(), default)

    def _save(self) -> None:
        self.rates_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rates_path, "w") as f:
            json.dump(self.rates, f, indent=2)

    def _load(self) -> None:
        with open(self.rates_path) as f:
            self.rates = json.load(f)
        logger.info(f"Loaded decay rates for {len(self.rates)} relations")


class DecayFunction:
    """
    Computes Fact Validity Score (FVS) using exponential decay.

    FVS = exp(-λ × Δt)

    Where:
        λ = decay rate (higher = faster deprecation)
        Δt = days since last verification

    Supports per-relation λ via RelationDecayRates.
    """

    def __init__(self, default_lambda: float = 0.01):
        self.default_lambda = default_lambda
        self._relation_rates: Optional[RelationDecayRates] = None

        # Try to load learned rates
        rates_path = Path("data/cache/decay_rates.json")
        if rates_path.exists():
            self._relation_rates = RelationDecayRates(str(rates_path))

    def calculate_fvs(
        self,
        last_verified: datetime,
        current_time: Optional[datetime] = None,
        lambda_val: Optional[float] = None,
        relation: Optional[str] = None,
    ) -> float:
        """
        Calculate Fact Validity Score.

        Args:
            last_verified: When the fact was last verified.
            current_time: Reference time (default: now UTC).
            lambda_val: Explicit λ override.
            relation: Relation type for per-relation λ lookup.

        Returns a float in [0, 1] where 1 = perfectly fresh.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        if lambda_val is None:
            if relation and self._relation_rates:
                lambda_val = self._relation_rates.get_lambda(
                    relation, self.default_lambda
                )
            else:
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
            lv = fact.get("last_verified") or fact.get("start_time")
            if lv:
                if isinstance(lv, str):
                    lv = dp.parse(lv)
                relation = fact.get("relation")
                fact["fvs"] = self.calculate_fvs(
                    lv, query_time, relation=relation
                )
                fact["is_valid"] = fact["fvs"] >= threshold
            else:
                fact["fvs"] = 0.0
                fact["is_valid"] = False
        return facts

