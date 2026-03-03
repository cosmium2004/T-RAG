"""
Unit tests for TKG modules (neo4j_client, schema, bulk_importer).
Tests against the live Neo4j instance.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.tkg.neo4j_client import Neo4jClient
from src.tkg.schema import GraphSchema
from src.tkg.bulk_importer import BulkImporter


# ── Neo4jClient ──────────────────────────────────────────────────────

class TestNeo4jClient:
    def test_health_check(self):
        client = Neo4jClient()
        health = client.health_check()
        assert health["status"] == "healthy"
        assert "Neo4j" in health["version"]
        client.close()

    def test_query_returns_results(self):
        client = Neo4jClient()
        result = client.query("RETURN 1 AS value")
        assert len(result) == 1
        assert result[0]["value"] == 1
        client.close()

    def test_context_manager(self):
        with Neo4jClient() as client:
            result = client.query("RETURN 42 AS v")
            assert result[0]["v"] == 42

    def test_node_count(self):
        with Neo4jClient() as client:
            result = client.query("MATCH (n:Entity) RETURN count(n) AS cnt")
            assert result[0]["cnt"] > 10000  # Full dataset loaded

    def test_temporal_query(self):
        with Neo4jClient() as client:
            result = client.query(
                "MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity) "
                "WHERE r.start_time IS NOT NULL "
                "RETURN h.name, r.relation_type, t.name "
                "LIMIT 5"
            )
            assert len(result) == 5
            assert "h.name" in result[0]


# ── GraphSchema ──────────────────────────────────────────────────────

class TestGraphSchema:
    def test_get_setup_statements(self):
        stmts = GraphSchema.get_setup_statements()
        assert len(stmts) >= 5
        assert any("CONSTRAINT" in s for s in stmts)
        assert any("INDEX" in s for s in stmts)

    def test_apply_schema_idempotent(self):
        """Applying schema twice should not error."""
        with Neo4jClient() as client:
            with client._driver.session(database=client.database) as s:
                GraphSchema.apply_schema(s)
                GraphSchema.apply_schema(s)  # second time is fine


# ── BulkImporter ─────────────────────────────────────────────────────

class TestBulkImporter:
    def test_collect_entities(self):
        facts = [
            {"head": "USA", "tail": "China"},
            {"head": "USA", "tail": "Russia"},
            {"head": "France", "tail": "China"},
        ]
        entities = BulkImporter._collect_entities(facts)
        names = {e["name"] for e in entities}
        assert names == {"USA", "China", "Russia", "France"}

    def test_prepare_relations(self):
        facts = [
            {
                "id": "f1", "head": "A", "tail": "B",
                "relation": "r", "start_time": "2024-01-01",
                "end_time": None, "last_verified": "2024-01-01",
                "source": "S", "confidence": 0.9, "text": "A r B",
            }
        ]
        rows = BulkImporter._prepare_relations(facts)
        assert len(rows) == 1
        assert rows[0]["fact_id"] == "f1"
        assert rows[0]["head"] == "A"

    def test_import_small_set(self):
        """Import a small test set and verify in Neo4j."""
        client = Neo4jClient()
        importer = BulkImporter(client)

        test_facts = [
            {
                "id": "test_fact_001", "head": "TestEntityAlpha",
                "tail": "TestEntityBeta", "relation": "test_relation",
                "start_time": "2024-06-01T00:00:00+00:00",
                "end_time": None, "last_verified": "2024-06-01T00:00:00+00:00",
                "source": "unit_test", "confidence": 0.99,
                "text": "TestEntityAlpha test_relation TestEntityBeta",
            }
        ]
        stats = importer.import_facts(test_facts, batch_size=10, clear_first=False)
        assert stats["entities_created"] >= 2
        assert stats["relations_created"] >= 1

        # Verify in graph
        result = client.query(
            "MATCH (h:Entity {name: 'TestEntityAlpha'})-[r:RELATES_TO]->"
            "(t:Entity {name: 'TestEntityBeta'}) RETURN r.fact_id AS fid"
        )
        assert len(result) >= 1
        assert result[0]["fid"] == "test_fact_001"

        # Cleanup
        client.write(
            "MATCH (n:Entity) WHERE n.name STARTS WITH 'TestEntity' DETACH DELETE n"
        )
        client.close()
