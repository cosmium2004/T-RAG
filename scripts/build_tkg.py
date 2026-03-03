"""
CLI Script — Build the Temporal Knowledge Graph in Neo4j.

Usage:
    python scripts/build_tkg.py
    python scripts/build_tkg.py --facts data/processed/facts.json --clear
    python scripts/build_tkg.py --uri bolt://localhost:7687 --password mypass
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import setup_logging
from src.tkg.neo4j_client import Neo4jClient
from src.tkg.bulk_importer import BulkImporter

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="T-RAG TKG Builder")
    parser.add_argument("--facts", default="data/processed/facts.json",
                        help="Path to processed facts JSON")
    parser.add_argument("--uri", default=None,
                        help="Neo4j bolt URI (default: from .env)")
    parser.add_argument("--user", default=None,
                        help="Neo4j username (default: from .env)")
    parser.add_argument("--password", default=None,
                        help="Neo4j password (default: from .env)")
    parser.add_argument("--database", default="neo4j",
                        help="Neo4j database name")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for imports")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing data before import")
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("T-RAG TKG Builder")
    logger.info("=" * 60)

    # Connect to Neo4j
    client = Neo4jClient(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
    )

    # Health check
    health = client.health_check()
    if health["status"] != "healthy":
        logger.error(f"Neo4j is not healthy: {health}")
        print(f"\n❌ Cannot connect to Neo4j: {health.get('error', 'unknown')}")
        print("  Make sure Neo4j Desktop is running and a database is started.")
        sys.exit(1)

    logger.info(f"Neo4j: {health}")

    # Import facts
    t0 = time.time()
    importer = BulkImporter(client)

    try:
        stats = importer.import_from_file(
            filepath=args.facts,
            batch_size=args.batch_size,
            clear_first=args.clear,
        )

        elapsed = time.time() - t0

        # Verify counts
        node_count = client.query("MATCH (n:Entity) RETURN count(n) AS cnt")
        rel_count = client.query("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt")

        n_nodes = node_count[0]["cnt"] if node_count else 0
        n_rels = rel_count[0]["cnt"] if rel_count else 0

        # Sample temporal query
        sample = client.query("""
            MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.start_time IS NOT NULL
            RETURN h.name AS head, r.relation_type AS relation,
                   t.name AS tail,
                   toString(r.start_time) AS start_time
            ORDER BY r.start_time DESC
            LIMIT 3
        """)

        print(f"\n{'─' * 55}")
        print(f"  TKG Build Summary")
        print(f"{'─' * 55}")
        print(f"  Entity nodes:    {n_nodes}")
        print(f"  Relationships:   {n_rels}")
        print(f"  Time:            {elapsed:.1f}s")
        print(f"{'─' * 55}")

        if sample:
            print(f"\n  Sample facts (most recent):")
            for s in sample:
                print(f"    {s['head']} → {s['relation']} → {s['tail']}")
                print(f"      Valid from: {s['start_time']}")
        print()

    finally:
        client.close()


if __name__ == "__main__":
    main()
