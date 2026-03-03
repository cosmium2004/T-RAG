"""
Module 2.3 — Bulk Importer for T-RAG TKG.

Loads processed facts into Neo4j as Entity nodes connected by
RELATES_TO relationships with temporal properties.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tkg.neo4j_client import Neo4jClient
from src.tkg.schema import GraphSchema

logger = logging.getLogger(__name__)


# ── Cypher templates ──────────────────────────────────────────────────

_MERGE_ENTITY = """
UNWIND $batch AS row
MERGE (e:Entity {name: row.name})
ON CREATE SET e.entity_type = row.entity_type,
              e.created_at  = datetime()
"""

_CREATE_FACT_REL = """
UNWIND $batch AS row
MATCH (h:Entity {name: row.head})
MATCH (t:Entity {name: row.tail})
MERGE (h)-[r:RELATES_TO {fact_id: row.fact_id}]->(t)
SET r.relation_type = row.relation,
    r.start_time    = CASE WHEN row.start_time IS NOT NULL
                           THEN datetime(row.start_time)
                           ELSE null END,
    r.end_time      = CASE WHEN row.end_time IS NOT NULL
                           THEN datetime(row.end_time)
                           ELSE null END,
    r.last_verified  = CASE WHEN row.last_verified IS NOT NULL
                            THEN datetime(row.last_verified)
                            ELSE null END,
    r.source         = row.source,
    r.confidence     = row.confidence,
    r.text           = row.text
"""


class BulkImporter:
    """
    Imports processed facts into Neo4j.

    Workflow:
        1. Apply schema (constraints + indexes)
        2. Create/merge Entity nodes
        3. Create RELATES_TO relationships with temporal metadata
    """

    def __init__(self, client: Optional[Neo4jClient] = None):
        self.client = client or Neo4jClient()
        self._entities_created = 0
        self._relations_created = 0

    # ── Public API ────────────────────────────────────────────────────

    def import_facts(
        self,
        facts: List[Dict[str, Any]],
        batch_size: int = 500,
        clear_first: bool = False,
    ) -> Dict[str, int]:
        """
        Load a list of fact dicts into Neo4j.

        Args:
            facts: Output from the data pipeline (``data/processed/facts.json``)
            batch_size: Rows per transaction batch
            clear_first: If True, delete all existing data before importing

        Returns:
            Dict with ``entities_created`` and ``relations_created`` counts.
        """
        self.client.connect()

        # Optionally clear
        if clear_first:
            logger.warning("Clearing all existing graph data…")
            with self.client._driver.session(database=self.client.database) as s:
                GraphSchema.clear_database(s)

        # 1. Apply schema
        logger.info("Applying graph schema…")
        with self.client._driver.session(database=self.client.database) as s:
            GraphSchema.apply_schema(s)

        # 2. Collect unique entities
        entities = self._collect_entities(facts)
        logger.info(f"Creating {len(entities)} entity nodes…")
        self._entities_created = self.client.write_batch(
            _MERGE_ENTITY, entities, batch_size
        )

        # 3. Create relationships
        rel_rows = self._prepare_relations(facts)
        logger.info(f"Creating {len(rel_rows)} relationships…")
        self._relations_created = self.client.write_batch(
            _CREATE_FACT_REL, rel_rows, batch_size
        )

        stats = {
            "entities_created": self._entities_created,
            "relations_created": self._relations_created,
        }
        logger.info(f"Import complete: {stats}")
        return stats

    def import_from_file(
        self,
        filepath: str = "data/processed/facts.json",
        batch_size: int = 500,
        clear_first: bool = False,
    ) -> Dict[str, int]:
        """Convenience: load facts from JSON file and import."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Facts file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            facts = json.load(f)

        logger.info(f"Loaded {len(facts)} facts from {path}")
        return self.import_facts(facts, batch_size, clear_first)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _collect_entities(facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract unique entities from facts."""
        seen = set()
        entities = []
        for fact in facts:
            for role in ("head", "tail"):
                name = fact.get(role)
                if name and name not in seen:
                    seen.add(name)
                    entities.append({
                        "name": name,
                        "entity_type": "Unknown",
                    })
        return entities

    @staticmethod
    def _prepare_relations(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map fact dicts to the shape expected by the Cypher template."""
        rows = []
        for fact in facts:
            rows.append({
                "fact_id":       fact["id"],
                "head":          fact["head"],
                "tail":          fact["tail"],
                "relation":      fact["relation"],
                "start_time":    fact.get("start_time"),
                "end_time":      fact.get("end_time"),
                "last_verified": fact.get("last_verified"),
                "source":        fact.get("source", "unknown"),
                "confidence":    fact.get("confidence", 0.5),
                "text":          fact.get("text", ""),
            })
        return rows

    # ── Stats ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "entities_created": self._entities_created,
            "relations_created": self._relations_created,
        }
