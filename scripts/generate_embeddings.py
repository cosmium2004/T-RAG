"""
CLI Script — Generate embeddings for all facts and build FAISS index.

Usage:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --facts data/processed/facts_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import setup_logging
from src.data_pipeline.embedder import Embedder
from src.retriever.vector_search import VectorSearch

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="T-RAG Embedding Generator")
    parser.add_argument(
        "--facts", default="data/processed/facts_full.json",
        help="Path to processed facts JSON",
    )
    parser.add_argument(
        "--output", default="data/embeddings",
        help="Output directory for embeddings and FAISS index",
    )
    parser.add_argument(
        "--model", default="all-mpnet-base-v2",
        help="Sentence-BERT model name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("T-RAG Embedding Generator")
    logger.info("=" * 60)

    # Load facts
    with open(args.facts, encoding="utf-8") as f:
        facts = json.load(f)
    logger.info(f"Loaded {len(facts)} facts from {args.facts}")

    # Generate embeddings
    t0 = time.time()
    embedder = Embedder(model_name=args.model)
    embeddings = embedder.embed_facts(facts, batch_size=args.batch_size)
    embed_time = time.time() - t0

    # Save embeddings
    fact_ids = [f["id"] for f in facts]
    embedder.save_embeddings(embeddings, fact_ids, args.output)

    # Build and save FAISS index
    t1 = time.time()
    vs = VectorSearch(dim=embeddings.shape[1])
    vs.build_index(embeddings, fact_ids, facts)
    vs.save(args.output)
    index_time = time.time() - t1

    total_time = time.time() - t0

    print(f"\n{'─' * 55}")
    print(f"  Embedding Summary")
    print(f"{'─' * 55}")
    print(f"  Facts embedded:   {len(facts)}")
    print(f"  Vector dim:       {embeddings.shape[1]}")
    print(f"  Embed time:       {embed_time:.1f}s")
    print(f"  Index build time: {index_time:.1f}s")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Output dir:       {args.output}")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()
