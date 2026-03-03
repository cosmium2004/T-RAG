"""
Module 2.1 — Neo4j Client for T-RAG.

Manages the connection to Neo4j with session pooling, health checks,
and convenience query helpers.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Thread-safe Neo4j driver wrapper.

    Usage::

        client = Neo4jClient()
        results = client.query("MATCH (n) RETURN n LIMIT 5")
        client.close()
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ):
        load_dotenv()
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        self._driver = None

    # ── Connection ────────────────────────────────────────────────────

    def connect(self):
        """Lazily create the Neo4j driver."""
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise

    def close(self):
        """Close the driver and release resources."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    # ── Query helpers ─────────────────────────────────────────────────

    def query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results as a list of dicts.
        """
        self.connect()
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def write(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a write transaction."""
        self.connect()
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, parameters or {})
            summary = result.consume()
            return summary

    def write_batch(
        self,
        cypher: str,
        batch: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        """
        Execute a parameterised Cypher statement in batches.

        The statement should use ``$batch`` as the parameter name
        for ``UNWIND $batch AS row ...``.

        Returns the total number of rows processed.
        """
        self.connect()
        total = 0
        for i in range(0, len(batch), batch_size):
            chunk = batch[i : i + batch_size]
            with self._driver.session(database=self.database) as session:
                session.run(cypher, {"batch": chunk})
            total += len(chunk)
            logger.debug(f"Batch {i // batch_size + 1}: {len(chunk)} rows")
        return total

    # ── Health ────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Return connection health info."""
        try:
            self.connect()
            info = self._driver.get_server_info()
            return {
                "status": "healthy",
                "server": str(info.address),
                "version": info.agent,
                "database": self.database,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
