"""
Learn per-relation decay rates (λ) from the dataset.

Usage:
    python scripts/learn_decay_rates.py
    python scripts/learn_decay_rates.py --facts data/processed/facts_full.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deprecation.decay import RelationDecayRates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Learn per-relation decay rates from dataset."
    )
    parser.add_argument(
        "--facts", default="data/processed/facts_full.json",
        help="Path to facts JSON file (default: data/processed/facts_full.json)",
    )
    parser.add_argument(
        "--output", default="data/cache/decay_rates.json",
        help="Output path for learned rates (default: data/cache/decay_rates.json)",
    )

    args = parser.parse_args()

    # Load facts
    print(f"\n📊 Loading facts from {args.facts}...")
    with open(args.facts, encoding="utf-8") as f:
        facts = json.load(f)
    print(f"   Loaded {len(facts)} facts")

    # Learn rates
    print("\n🧮 Learning per-relation decay rates...")
    learner = RelationDecayRates(rates_path=args.output)
    rates = learner.learn_rates(facts)

    # Display results
    print(f"\n📈 Learned rates for {len(rates)} relation types:\n")
    sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)
    print(f"   {'Relation':<40} {'λ':>10} {'Half-life (days)':>18}")
    print(f"   {'-' * 40} {'-' * 10} {'-' * 18}")
    for rel, lam in sorted_rates[:20]:
        half_life = 0.693 / lam if lam > 0 else float("inf")
        print(f"   {rel:<40} {lam:>10.6f} {half_life:>18.1f}")

    if len(sorted_rates) > 20:
        print(f"   ... and {len(sorted_rates) - 20} more relations")

    print(f"\n💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
