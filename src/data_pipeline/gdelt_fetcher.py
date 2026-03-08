"""
Module 1.9 — GDELT Fetcher for T-RAG pipeline.
Fetches recent geopolitical events from the GDELT 2.0 DOC API.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_pipeline.quadruple_extractor import QuadrupleExtractor

logger = logging.getLogger(__name__)

# ── GDELT 2.0 DOC API ───────────────────────────────────────────────
_GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTFetcher:
    """
    Fetches recent events from GDELT 2.0 DOC API.

    GDELT provides real-time access to news events worldwide.
    This fetcher queries the API and converts results into the
    same (head, relation, tail, time) format used by the ICEWS pipeline.
    """

    def __init__(self, cache_dir: str = "data/cache/gdelt"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetched = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _api_request(self, params: Dict[str, str]) -> Dict:
        """Make a request to the GDELT API with retry logic."""
        resp = requests.get(
            _GDELT_API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": "T-RAG/1.0"},
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_recent(
        self,
        query: str = "",
        days_back: int = 7,
        max_records: int = 250,
        source_country: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent events from GDELT.

        Args:
            query: Search query (e.g., "diplomatic relations", "North Korea").
                   Empty string returns all recent events.
            days_back: Number of days to look back (max 365).
            max_records: Maximum number of articles to retrieve (max 250).
            source_country: Optional 2-letter country code filter.

        Returns:
            List of fact dicts matching the T-RAG pipeline format.
        """
        # Check cache first
        cache_key = self._cache_key(query, days_back, max_records)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"GDELT cache hit: {len(cached)} facts for '{query}'")
            return cached

        # Build query parameters
        params = {
            "format": "json",
            "mode": "ArtList",
            "maxrecords": str(min(max_records, 250)),
            "timespan": f"{days_back}d",
            "sort": "DateDesc",
        }

        if query:
            params["query"] = query
        if source_country:
            params["sourcecountry"] = source_country

        try:
            data = self._api_request(params)
        except Exception as e:
            logger.error(f"GDELT API error: {e}")
            return []

        articles = data.get("articles", [])
        if not articles:
            logger.warning(f"GDELT returned no articles for query: '{query}'")
            return []

        facts = self._parse_articles(articles)
        self._fetched = len(facts)

        # Cache results
        self._save_cache(cache_key, facts)

        logger.info(
            f"GDELT: fetched {len(articles)} articles -> "
            f"{len(facts)} facts for '{query}'"
        )
        return facts

    def _parse_articles(
        self,
        articles: List[Dict[str, Any]],
        provider: str = "ollama",
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert GDELT articles to T-RAG fact format using LLM extraction.

        Groups article titles into batches and sends them through the
        QuadrupleExtractor for structured (head, relation, tail, time)
        extraction.
        """
        # Filter to English articles with titles
        valid_articles = []
        for article in articles:
            title = article.get("title", "").strip()
            language = article.get("language", "English")
            if title and language == "English":
                valid_articles.append(article)

        if not valid_articles:
            return []

        # Build text blocks from article titles (batch for efficiency)
        # Group ~20 titles per LLM call to reduce API overhead
        batch_size = 20
        all_facts: List[Dict[str, Any]] = []
        extractor = QuadrupleExtractor(
            provider=provider, model=model,
        )

        for i in range(0, len(valid_articles), batch_size):
            batch = valid_articles[i : i + batch_size]

            # Build a text block from titles with dates
            lines = []
            article_meta = {}  # map title -> article metadata
            for art in batch:
                title = art["title"].strip()
                date_str = art.get("seendate", "")
                iso_date = self._parse_gdelt_date(date_str)
                date_suffix = f" ({iso_date[:10]})" if iso_date else ""
                line = f"- {title}{date_suffix}"
                lines.append(line)
                article_meta[title] = art

            text_block = "News headlines:\n" + "\n".join(lines)

            # Extract facts via LLM
            chunk_facts = extractor.extract(text_block)

            # Tag each fact with GDELT source metadata
            for fact in chunk_facts:
                # Find the closest matching article for source info
                best_domain = "gdelt.org"
                best_url = ""
                for art in batch:
                    art_title = art.get("title", "").lower()
                    if (fact["head"].lower() in art_title
                            or fact["tail"].lower() in art_title):
                        best_domain = art.get("domain", "gdelt.org")
                        best_url = art.get("url", "")
                        break

                fact["source"] = f"GDELT ({best_domain})"
                fact["url"] = best_url
                id_str = f"{fact['head']}{fact['relation']}{fact['tail']}"
                id_hash = hashlib.md5(id_str.encode()).hexdigest()[:8]
                fact["id"] = f"gdelt_{id_hash}"

            all_facts.extend(chunk_facts)

        # Deduplicate
        return extractor._deduplicate(all_facts)

    @staticmethod
    def _parse_gdelt_date(date_str: str) -> Optional[str]:
        """Parse GDELT date format (YYYYMMDDTHHmmSS) to ISO 8601."""
        if not date_str:
            return None
        try:
            # GDELT uses format: 20240115T143000Z
            dt = datetime.strptime(
                date_str.replace("Z", ""), "%Y%m%dT%H%M%S"
            )
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            try:
                # Fallback: try standard ISO
                from dateutil import parser as dp
                dt = dp.parse(date_str)
                return dt.isoformat()
            except (ValueError, ImportError):
                return None

    # ── Caching ──────────────────────────────────────────────────────

    def _cache_key(
        self, query: str, days_back: int, max_records: int
    ) -> str:
        """Generate a cache key for a query."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw = f"{query}:{days_back}:{max_records}:{today}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[List[Dict]]:
        """Load cached results if they exist."""
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, facts: List[Dict]) -> None:
        """Save results to cache."""
        path = self.cache_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, default=str)
        logger.debug(f"Cached {len(facts)} facts to {path}")

    @property
    def stats(self) -> Dict[str, int]:
        return {"fetched": self._fetched}
