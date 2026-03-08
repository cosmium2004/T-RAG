"""
Module 1.8 — Quadruple Extractor for T-RAG pipeline.
Uses the LLM to extract (head, relation, tail, time) facts from free text.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.generator.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Extraction prompt ────────────────────────────────────────────────

_EXTRACTION_PROMPT = """Extract all factual relationships from this text as a JSON array.

For each fact, provide:
- "head": the subject entity (person, organization, country, etc.)
- "relation": the action or relationship (e.g., "visited", "signed agreement with", "criticized")
- "tail": the object entity
- "date": the date/time in ISO 8601 format (YYYY-MM-DD), or null if not mentioned
- "confidence": your confidence in this extraction from 0.0 to 1.0

Rules:
- Extract ONLY factual relationships, not opinions or speculation
- Each fact should be a simple (subject, action, object) triple
- Normalize entity names (e.g., "US" -> "United States")
- If a date is approximate (e.g., "early 2024"), use the best guess (e.g., "2024-02-01")
- Output ONLY the JSON array, no other text

Text:
{text}

JSON array:"""


class QuadrupleExtractor:
    """
    Extracts temporal knowledge quadruples from free text using an LLM.

    Sends structured prompts to the configured LLM and parses the
    JSON response into the same dict format used by the ICEWS pipeline.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: str = "ollama",
        model: Optional[str] = None,
    ):
        self.llm = llm_client or LLMClient(
            provider=provider,
            model=model,
            max_tokens=2000,
            temperature=0.1,  # Low temp for structured extraction
        )
        self._extracted = 0
        self._failed_chunks = 0

    def extract(self, chunk: str) -> List[Dict[str, Any]]:
        """
        Extract quadruples from a single text chunk.

        Args:
            chunk: Text to extract facts from.

        Returns:
            List of fact dicts matching entity_extractor.py output format.
        """
        if not chunk or not chunk.strip():
            return []

        prompt = _EXTRACTION_PROMPT.format(text=chunk)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise information extraction system. "
                    "Output only valid JSON arrays. No markdown, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self.llm.generate(messages)
            facts = self._parse_response(raw)
            self._extracted += len(facts)
            return facts
        except Exception as e:
            logger.warning(f"Extraction failed for chunk: {e}")
            self._failed_chunks += 1
            return []

    def extract_from_document(
        self,
        chunks: List[str],
        source_name: str = "document",
    ) -> List[Dict[str, Any]]:
        """
        Extract quadruples from all chunks of a document.

        Args:
            chunks: List of text chunks from TextChunker.
            source_name: Name of the source document for metadata.

        Returns:
            Deduplicated list of fact dicts.
        """
        all_facts: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            logger.info(
                f"Extracting from chunk {i + 1}/{len(chunks)} "
                f"({len(chunk)} chars)..."
            )
            facts = self.extract(chunk)

            # Tag each fact with source info
            for fact in facts:
                fact["source"] = source_name
                fact.setdefault("id", f"doc_{uuid.uuid4().hex[:8]}")

            all_facts.extend(facts)

        # Deduplicate by (head, relation, tail)
        deduped = self._deduplicate(all_facts)

        logger.info(
            f"Document extraction: {len(all_facts)} raw -> "
            f"{len(deduped)} deduplicated facts from {len(chunks)} chunks"
        )
        return deduped

    def _parse_response(self, raw: str) -> List[Dict[str, Any]]:
        """Parse LLM JSON response into fact dicts."""
        # Try to extract JSON array from the response
        text = raw.strip()

        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Find the JSON array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            logger.warning(f"No JSON array found in response: {text[:100]}...")
            return []

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []

        if not isinstance(items, list):
            return []

        # Convert to T-RAG fact format
        facts = []
        for item in items:
            if not isinstance(item, dict):
                continue

            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()

            if not head or not relation or not tail:
                continue

            # Normalize
            head = head.title()
            relation = relation.lower().replace("_", " ")
            tail = tail.title()

            date_str = item.get("date")
            iso_date = None
            if date_str and isinstance(date_str, str):
                try:
                    from dateutil import parser as dp
                    dt = dp.parse(date_str)
                    iso_date = dt.isoformat()
                except (ValueError, OverflowError):
                    iso_date = None

            confidence = float(item.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))

            fact = {
                "id": f"doc_{uuid.uuid4().hex[:8]}",
                "head": head,
                "relation": relation,
                "tail": tail,
                "start_time": iso_date,
                "end_time": None,
                "last_verified": iso_date or datetime.now(timezone.utc).isoformat(),
                "source": "document",
                "confidence": confidence,
                "text": f"{head} {relation} {tail}",
            }
            facts.append(fact)

        return facts

    @staticmethod
    def _deduplicate(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate facts based on (head, relation, tail) key."""
        seen = set()
        unique = []
        for fact in facts:
            key = (
                fact["head"].lower(),
                fact["relation"].lower(),
                fact["tail"].lower(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(fact)
        return unique

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "extracted": self._extracted,
            "failed_chunks": self._failed_chunks,
        }
