"""
Ingest live data from external sources (GDELT) into T-RAG.

Usage:
    python scripts/ingest_live.py --source gdelt --query "diplomatic relations" --days 7
    python scripts/ingest_live.py --source gdelt --query "North Korea" --days 30 --max 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.gdelt_fetcher import GDELTFetcher
from src.data_pipeline.duplicate_resolver import DuplicateResolver
from src.data_pipeline.embedder import Embedder
from src.retriever.vector_search import VectorSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest live data into T-RAG knowledge base."
    )
    parser.add_argument(
        "--source", default="gdelt", choices=["gdelt"],
        help="Data source (default: gdelt)",
    )
    parser.add_argument(
        "--query", default="",
        help="Search query for events (e.g., 'diplomatic relations')",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days to look back (default: 7)",
    )
    parser.add_argument(
        "--max", type=int, default=250,
        help="Maximum articles to fetch (default: 250, max: 250)",
    )
    parser.add_argument(
        "--country", default=None,
        help="Source country filter (2-letter code, e.g., 'US')",
    )
    parser.add_argument(
        "--embeddings-dir", default="data/embeddings",
        help="FAISS index directory (default: data/embeddings)",
    )
    parser.add_argument(
        "--output-facts", default=None,
        help="Optional: save fetched facts to JSON file",
    )

    args = parser.parse_args()

    # ── Step 1: Fetch from source ──────────────────────────────────
    print(f"\n📡 Fetching from {args.source.upper()}...")
    print(f"   Query: '{args.query or '(all events)'}', Days: {args.days}")

    if args.source == "gdelt":
        fetcher = GDELTFetcher()
        facts = fetcher.fetch_recent(
            query=args.query,
            days_back=args.days,
            max_records=args.max,
            source_country=args.country,
        )
    else:
        print(f"❌ Unknown source: {args.source}")
        return

    print(f"   Fetched {len(facts)} facts")

    if not facts:
        print("\n⚠️  No facts retrieved. Try a broader query or longer time range.")
        return

    # ── Step 2: Deduplicate ─────────────────────────────────────────
    resolver = DuplicateResolver()
    facts = resolver.resolve(facts)
    print(f"   After deduplication: {len(facts)} facts")

    # ── Step 3: Embed ───────────────────────────────────────────────
    print("\n📐 Generating embeddings...")
    embedder = Embedder()
    embeddings = embedder.embed_facts(facts)
    fact_ids = [f["id"] for f in facts]
    print(f"   Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}d)")

    # ── Step 4: Update FAISS index ──────────────────────────────────
    print(f"\n📦 Updating FAISS index at {args.embeddings_dir}...")
    vs = VectorSearch()
    emb_dir = Path(args.embeddings_dir)

    if (emb_dir / "faiss_index.bin").exists():
        vs.load(args.embeddings_dir)
        print(f"   Loaded existing index: {vs.size} vectors")
    else:
        emb_dir.mkdir(parents=True, exist_ok=True)

    added = vs.append_to_index(embeddings, fact_ids, facts)
    vs.save(args.embeddings_dir)
    print(f"   Added {added} new vectors (total: {vs.size})")

    # ── Optional: save facts ────────────────────────────────────────
    if args.output_facts:
        with open(args.output_facts, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Saved facts to {args.output_facts}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("✅ Live ingestion complete!")
    print(f"   Source:     {args.source.upper()}")
    print(f"   Query:      {args.query or '(all events)'}")
    print(f"   Facts:      {len(facts)}")
    print(f"   Index size: {vs.size} vectors")
    print("=" * 50)


if __name__ == "__main__":
    main()
