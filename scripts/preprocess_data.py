"""
CLI Script — Run the entire data preprocessing pipeline.

Usage:
    python scripts/preprocess_data.py --dataset icews18 --limit 2000
    python scripts/preprocess_data.py --dataset icews18 --split test
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import setup_logging, load_config
from src.data_pipeline.fetcher import DataFetcher
from src.data_pipeline.timestamp_parser import TimestampParser
from src.data_pipeline.entity_extractor import EntityExtractor
from src.data_pipeline.duplicate_resolver import DuplicateResolver

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="T-RAG Data Preprocessing Pipeline")
    parser.add_argument("--dataset", default="icews18", choices=["icews18"],
                        help="Dataset to process (default: icews18)")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"],
                        help="Dataset split (default: train)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows (for testing)")
    parser.add_argument("--output", default="data/processed/facts.json",
                        help="Output file path")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download even if cached")
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("T-RAG Data Preprocessing Pipeline")
    logger.info("=" * 60)

    # --- Step 1: Fetch data ---
    logger.info(f"Step 1/4: Fetching {args.dataset}/{args.split}...")
    fetcher = DataFetcher(cache_dir="data/cache")
    df = fetcher.fetch_icews(
        dataset=args.dataset,
        split=args.split,
        force_download=args.force_download,
        limit=args.limit,
    )
    logger.info(f"  → Loaded {len(df)} raw records")

    # --- Step 2: Parse timestamps ---
    logger.info("Step 2/4: Parsing timestamps...")
    ts_parser = TimestampParser()
    df = ts_parser.parse_column(df, column="date")
    logger.info(f"  → Parse success rate: {ts_parser.success_rate:.1f}%")

    # --- Step 3: Extract quadruples ---
    logger.info("Step 3/4: Extracting quadruples...")
    extractor = EntityExtractor()
    facts = extractor.extract_quadruples(df)
    logger.info(f"  → Extracted {extractor.stats['extracted']} facts, "
                f"dropped {extractor.stats['dropped']}")

    # --- Step 4: Resolve duplicates ---
    logger.info("Step 4/4: Resolving duplicates...")
    resolver = DuplicateResolver()
    facts = resolver.resolve(facts)
    logger.info(f"  → Final count: {resolver.stats['output_count']} facts "
                f"({resolver.stats['duplicates_removed']} duplicates removed)")

    # --- Save output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"✓ Saved {len(facts)} facts to {output_path}")
    logger.info("=" * 60)

    # Summary
    print(f"\n{'─' * 50}")
    print(f"  Pipeline Summary")
    print(f"{'─' * 50}")
    print(f"  Dataset:       {args.dataset}/{args.split}")
    print(f"  Raw records:   {len(df)}")
    print(f"  Timestamps OK: {ts_parser.success_rate:.1f}%")
    print(f"  Facts out:     {len(facts)}")
    print(f"  Output:        {output_path}")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()
