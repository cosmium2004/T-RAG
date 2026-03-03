"""
Module 2.2 — Graph Schema for T-RAG Temporal Knowledge Graph.

Defines the Neo4j schema (node types, relationship types, constraints,
indexes) and provides helpers to initialise a fresh database.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Cypher statements for schema setup ────────────────────────────────

CONSTRAINTS: List[str] = [
    # Unique entity names
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE",

    # Unique fact IDs
    "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS "
    "FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE",
]

INDEXES: List[str] = [
    # Entity type index for filtering
    "CREATE INDEX entity_type_idx IF NOT EXISTS "
    "FOR (e:Entity) ON (e.entity_type)",

    # Fact temporal indexes
    "CREATE INDEX fact_start_idx IF NOT EXISTS "
    "FOR (f:Fact) ON (f.start_time)",
    "CREATE INDEX fact_end_idx IF NOT EXISTS "
    "FOR (f:Fact) ON (f.end_time)",

    # Full-text search on entity names
    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name]",

    # Relationship range index for temporal queries
    "CREATE INDEX rel_temporal_idx IF NOT EXISTS "
    "FOR ()-[r:RELATES_TO]-() ON (r.start_time, r.end_time)",
]


class GraphSchema:
    """Manages the Neo4j schema for the T-RAG knowledge graph."""

    # Node labels
    ENTITY_LABEL = "Entity"
    FACT_LABEL = "Fact"

    # Relationship type
    RELATES_TO = "RELATES_TO"

    @staticmethod
    def get_setup_statements() -> List[str]:
        """Return all Cypher statements needed to initialise the schema."""
        return CONSTRAINTS + INDEXES

    @staticmethod
    def apply_schema(session) -> None:
        """
        Run all schema-setup statements against a Neo4j session.

        Args:
            session: An active ``neo4j.Session`` instance.
        """
        stmts = GraphSchema.get_setup_statements()
        for stmt in stmts:
            try:
                session.run(stmt)
                logger.debug(f"Applied: {stmt[:60]}…")
            except Exception as e:
                # Some statements may fail if the schema already exists
                logger.warning(f"Schema statement skipped: {e}")

        logger.info(f"Schema setup complete ({len(stmts)} statements)")

    @staticmethod
    def clear_database(session) -> None:
        """Delete all nodes and relationships (use with caution)."""
        session.run("MATCH (n) DETACH DELETE n")
        logger.warning("All nodes and relationships deleted")
