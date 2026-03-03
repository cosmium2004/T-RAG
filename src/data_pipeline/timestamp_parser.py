"""
Module 1.2 — Timestamp Parser for T-RAG pipeline.
Converts raw date strings to ISO 8601 datetime objects.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Union

import pandas as pd
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)


class TimestampParser:
    """Parses and normalises heterogeneous date strings to ISO 8601."""

    # Known ICEWS date column format
    ICEWS_FORMAT = "%Y-%m-%d"

    def __init__(self, default_tz: timezone = timezone.utc):
        self.default_tz = default_tz
        self._parse_failures = 0
        self._parse_total = 0

    # ------------------------------------------------------------------
    # Single-value parsing
    # ------------------------------------------------------------------

    def parse(self, raw_date: Union[str, int, float, datetime]) -> Optional[datetime]:
        """
        Parse a single date value to a timezone-aware ``datetime``.

        Returns ``None`` if parsing fails (instead of raising).
        """
        self._parse_total += 1

        if isinstance(raw_date, datetime):
            return self._ensure_tz(raw_date)

        if pd.isna(raw_date):
            self._parse_failures += 1
            return None

        raw_str = str(raw_date).strip()
        if not raw_str:
            self._parse_failures += 1
            return None

        # Fast-path: ICEWS "YYYY-MM-DD"
        try:
            dt = datetime.strptime(raw_str[:10], self.ICEWS_FORMAT)
            return self._ensure_tz(dt)
        except ValueError:
            pass

        # Slow-path: dateutil fuzzy parser
        try:
            dt = dateutil_parser.parse(raw_str, fuzzy=True)
            return self._ensure_tz(dt)
        except (ValueError, OverflowError):
            logger.warning(f"Could not parse date: {raw_str!r}")
            self._parse_failures += 1
            return None

    # ------------------------------------------------------------------
    # Batch (DataFrame) processing
    # ------------------------------------------------------------------

    def parse_column(self, df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
        """
        Parse an entire DataFrame column in-place.

        Creates two new columns:
        - ``<column>_parsed``: ``datetime`` values
        - ``<column>_iso``: ISO 8601 string representation

        Rows with unparseable dates get ``None`` / ``NaT``.
        """
        self._parse_failures = 0
        self._parse_total = 0

        df[f"{column}_parsed"] = df[column].apply(self.parse)
        df[f"{column}_iso"] = df[f"{column}_parsed"].apply(
            lambda dt: dt.isoformat() if dt else None
        )

        success_rate = (
            (self._parse_total - self._parse_failures) / max(self._parse_total, 1) * 100
        )
        logger.info(
            f"Parsed {self._parse_total} dates — "
            f"{success_rate:.1f}% successful, "
            f"{self._parse_failures} failures"
        )

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_tz(self, dt: datetime) -> datetime:
        """Attach default timezone if the datetime is naïve."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.default_tz)
        return dt

    @property
    def success_rate(self) -> float:
        if self._parse_total == 0:
            return 0.0
        return (self._parse_total - self._parse_failures) / self._parse_total * 100
