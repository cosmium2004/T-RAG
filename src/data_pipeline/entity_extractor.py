"""
Module 1.3 — Entity Extractor for T-RAG pipeline.
Extracts (head, relation, tail, time) quadruples and normalises entities.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts and normalises temporal-knowledge quadruples.

    For ICEWS-style data the quadruples already exist in the raw file;
    this module cleans them and generates a textual representation
    suitable for embedding.
    """

    # Minimal entity name length after cleaning
    MIN_NAME_LEN = 1

    def __init__(self):
        self._extracted = 0
        self._dropped = 0

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_quadruples(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert a DataFrame to a list of fact dictionaries.

        Expected columns: ``head``, ``relation``, ``tail``, ``date_parsed``.
        An ``event_id`` column is used if present.

        Returns:
            List of dicts with keys:
            ``id``, ``head``, ``relation``, ``tail``,
            ``start_time``, ``end_time``, ``source``,
            ``confidence``, ``text``
        """
        facts: List[Dict[str, Any]] = []
        self._extracted = 0
        self._dropped = 0

        for idx, row in df.iterrows():
            head = self._clean_entity(row.get("head", ""))
            relation = self._clean_relation(row.get("relation", ""))
            tail = self._clean_entity(row.get("tail", ""))
            timestamp = row.get("date_parsed")

            if not head or not relation or not tail:
                self._dropped += 1
                continue

            fact: Dict[str, Any] = {
                "id": f"fact_{row.get('event_id', idx + 1)}",
                "head": head,
                "relation": relation,
                "tail": tail,
                "start_time": timestamp.isoformat() if timestamp else None,
                "end_time": None,               # open-ended by default
                "last_verified": timestamp.isoformat() if timestamp else None,
                "source": "ICEWS",
                "confidence": 0.90,
                "text": self._build_text(head, relation, tail),
            }
            facts.append(fact)
            self._extracted += 1

        logger.info(
            f"Extracted {self._extracted} quadruples, "
            f"dropped {self._dropped} invalid rows"
        )
        return facts

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_entity(raw: Any) -> Optional[str]:
        """Normalise an entity name: strip, title-case, collapse spaces."""
        if pd.isna(raw):
            return None
        name = str(raw).strip()
        name = re.sub(r"\s+", " ", name)
        if len(name) < EntityExtractor.MIN_NAME_LEN:
            return None
        return name.title()

    @staticmethod
    def _clean_relation(raw: Any) -> Optional[str]:
        """Normalise a relation label: lowercase, underscores → spaces."""
        if pd.isna(raw):
            return None
        rel = str(raw).strip().lower()
        rel = rel.replace("_", " ")
        rel = re.sub(r"\s+", " ", rel)
        if len(rel) < 1:
            return None
        return rel

    @staticmethod
    def _build_text(head: str, relation: str, tail: str) -> str:
        """Create a natural-language sentence from a triple."""
        return f"{head} {relation} {tail}"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        return {"extracted": self._extracted, "dropped": self._dropped}
