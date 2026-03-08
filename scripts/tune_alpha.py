"""
Tune the WRS alpha parameter using grid search.

Usage:
    python scripts/tune_alpha.py
    python scripts/tune_alpha.py --facts data/processed/facts_full.json --alphas 0.3,0.4,0.5,0.6,0.7,0.8
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deprecation.decay import DecayFunction
from src.retriever.wrs import WRSScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_alpha(
    facts: List[Dict[str, Any]],
    alpha: float,
    query_time: datetime,
) -> Dict[str, float]:
    """
    Evaluate a given alpha value by measuring ranking quality.

    Uses a proxy metric: how well does WRS ranking correlate with
    the expected ordering (high-similarity + high-FVS facts first)?
    """
    decay = DecayFunction()
    scorer = WRSScorer(alpha=alpha)

    # Score all facts
    scored_facts = decay.score_facts(list(facts), query_time=query_time)
    for fact in scored_facts:
        fact["wrs"] = scorer.score(fact)

    valid_facts = [f for f in scored_facts if f.get("fvs", 0) > 0]
    if not valid_facts:
        return {"alpha": alpha, "mrr": 0.0, "temporal_accuracy": 0.0}

    # Temporal accuracy: fraction of top-10 facts that are temporally valid
    ranked = sorted(valid_facts, key=lambda f: f["wrs"], reverse=True)
    top_10 = ranked[:10]
    temporal_acc = sum(1 for f in top_10 if f.get("is_valid", False)) / len(top_10)

    # Mean FVS of top-10 (higher = more temporally fresh results)
    mean_fvs = np.mean([f.get("fvs", 0.0) for f in top_10])

    # Mean similarity of top-10 (higher = more relevant results)
    mean_sim = np.mean([f.get("similarity", 0.0) for f in top_10])

    # Combined score: harmonic mean of temporal accuracy and mean FVS
    combined = 0.0
    if temporal_acc + mean_fvs > 0:
        combined = 2 * temporal_acc * mean_fvs / (temporal_acc + mean_fvs)

    return {
        "alpha": alpha,
        "temporal_accuracy": round(float(temporal_acc), 4),
        "mean_fvs": round(float(mean_fvs), 4),
        "mean_similarity": round(float(mean_sim), 4),
        "combined_score": round(float(combined), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune WRS alpha parameter via grid search."
    )
    parser.add_argument(
        "--facts", default="data/processed/facts_full.json",
        help="Path to facts JSON file",
    )
    parser.add_argument(
        "--alphas", default="0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated alpha values to test",
    )
    parser.add_argument(
        "--output", default="data/cache/optimal_alpha.json",
        help="Output path for results",
    )

    args = parser.parse_args()
    alpha_values = [float(a.strip()) for a in args.alphas.split(",")]

    # Load facts
    print(f"\n📊 Loading facts from {args.facts}...")
    with open(args.facts, encoding="utf-8") as f:
        facts = json.load(f)

    # Add dummy similarity scores if not present (for tuning purposes)
    for fact in facts:
        if "similarity" not in fact:
            fact["similarity"] = np.random.uniform(0.3, 0.9)

    print(f"   Loaded {len(facts)} facts")
    query_time = datetime.now(timezone.utc)

    # Grid search
    print(f"\n🔍 Testing alpha values: {alpha_values}")
    results = []
    for alpha in alpha_values:
        result = evaluate_alpha(facts, alpha, query_time)
        results.append(result)
        print(
            f"   α={alpha:.1f}  |  temporal_acc={result['temporal_accuracy']:.4f}  "
            f"mean_fvs={result['mean_fvs']:.4f}  "
            f"combined={result['combined_score']:.4f}"
        )

    # Find best alpha
    best = max(results, key=lambda r: r["combined_score"])
    print(f"\n🏆 Best alpha: {best['alpha']}")
    print(f"   Combined score: {best['combined_score']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "optimal_alpha": best["alpha"],
            "all_results": results,
        }, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
