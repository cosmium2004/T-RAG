"""
Benchmark runner for T-RAG evaluation.

Runs the full pipeline on the benchmark dataset, collecting metrics
for temporal accuracy, factual grounding, latency, and confidence.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --provider ollama --model llama3 --output results.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark_data import BENCHMARK_QUERIES
from src.evaluation.metrics import (
    calculate_mrr,
    calculate_temporal_accuracy,
    calculate_hits_at_k,
)
from src.retriever.vector_search import VectorSearch
from src.retriever.query_encoder import QueryEncoder
from src.deprecation.decay import DecayFunction
from src.retriever.temporal_filter import TemporalFilter
from src.retriever.wrs import WRSScorer
from src.retriever.context_assembler import ContextAssembler
from src.generator.prompt_builder import PromptBuilder
from src.generator.llm_client import LLMClient
from src.generator.post_processor import PostProcessor
from src.validator.consistency import ConsistencyValidator
from src.validator.confidence import ConfidenceScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_query(
    query_data: Dict[str, Any],
    vs: VectorSearch,
    encoder: QueryEncoder,
    llm: LLMClient,
    alpha: float = 0.5,
    lambda_val: float = 0.01,
    top_k: int = 5,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run the full T-RAG pipeline on a single benchmark query."""
    query = query_data["query"]
    query_date = query_data["query_date"]
    query_time = datetime.strptime(query_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    t0 = time.time()

    # Encode query
    query_vec = encoder.encode(query)

    # Search FAISS
    candidates = vs.search(query_vec, top_n=50)

    # Score temporal freshness
    decay = DecayFunction(default_lambda=lambda_val)
    decay.score_facts(candidates, query_time, threshold)

    # Filter deprecated facts
    tf = TemporalFilter(deprecation_threshold=threshold)
    valid = tf.filter(candidates, query_time)

    # Rank by WRS
    wrs = WRSScorer(alpha=alpha)
    ranked = wrs.rank(valid, top_k=top_k)

    # Assemble context
    ca = ContextAssembler()
    context = ca.format(ranked, query_time)

    # Generate answer
    pb = PromptBuilder()
    messages = pb.build_messages(query, context, query_time)
    raw_answer = llm.generate(messages)

    # Post-process
    pp = PostProcessor()
    answer = pp.process(raw_answer, source_facts=ranked)

    # Validate
    cv = ConsistencyValidator()
    validation = cv.validate(answer, query_time, ranked)
    cs = ConfidenceScorer()
    confidence = cs.score(validation, ranked)

    latency_ms = int((time.time() - t0) * 1000)

    # Evaluate against expected
    entity_recall = _entity_recall(answer, query_data.get("expected_entities", []))
    temporal_hit = _temporal_hit(answer, query_data.get("expected_temporal", ""))

    return {
        "id": query_data["id"],
        "query": query,
        "query_date": query_date,
        "category": query_data.get("category", ""),
        "difficulty": query_data.get("difficulty", ""),
        "answer": answer,
        "confidence": confidence["confidence"],
        "confidence_rating": confidence["rating"],
        "consistency_score": validation["consistency_score"],
        "contradictions": len(validation.get("contradictions", [])),
        "facts_used": len(ranked),
        "facts_available": len(candidates),
        "entity_recall": entity_recall,
        "temporal_hit": temporal_hit,
        "latency_ms": latency_ms,
        "context_conflicts": ca.conflicts,
    }


def _entity_recall(answer: str, expected_entities: List[str]) -> float:
    """What fraction of expected entities appear in the answer?"""
    if not expected_entities:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for e in expected_entities if e.lower() in answer_lower)
    return found / len(expected_entities)


def _temporal_hit(answer: str, expected_temporal: str) -> float:
    """Does the answer contain the expected temporal reference?"""
    if not expected_temporal:
        return 1.0
    return 1.0 if expected_temporal.lower() in answer.lower() else 0.0


def print_results(results: List[Dict[str, Any]]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 90)
    print("T-RAG BENCHMARK RESULTS")
    print("=" * 90)

    print(f"\n{'ID':<5} {'Category':<20} {'Diff':<7} {'Conf':>6} {'EntR':>6} "
          f"{'TempH':>6} {'Facts':>6} {'Ms':>6}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['id']:<5} {r['category']:<20} {r['difficulty']:<7} "
            f"{r['confidence']:>5.0%} {r['entity_recall']:>5.0%} "
            f"{r['temporal_hit']:>5.0%} {r['facts_used']:>6} "
            f"{r['latency_ms']:>5}ms"
        )

    print("-" * 90)

    # Aggregate metrics
    n = len(results)
    avg_conf = sum(r["confidence"] for r in results) / n
    avg_entity = sum(r["entity_recall"] for r in results) / n
    avg_temporal = sum(r["temporal_hit"] for r in results) / n
    avg_latency = sum(r["latency_ms"] for r in results) / n
    avg_facts = sum(r["facts_used"] for r in results) / n

    print(f"\n📊 AGGREGATE METRICS ({n} queries)")
    print(f"   Mean Confidence:      {avg_conf:.1%}")
    print(f"   Mean Entity Recall:   {avg_entity:.1%}")
    print(f"   Mean Temporal Hit:    {avg_temporal:.1%}")
    print(f"   Mean Latency:         {avg_latency:.0f}ms")
    print(f"   Mean Facts Used:      {avg_facts:.1f}")

    # By category
    categories = set(r["category"] for r in results)
    if len(categories) > 1:
        print(f"\n📊 BY CATEGORY:")
        for cat in sorted(categories):
            cat_results = [r for r in results if r["category"] == cat]
            cat_conf = sum(r["confidence"] for r in cat_results) / len(cat_results)
            cat_ent = sum(r["entity_recall"] for r in cat_results) / len(cat_results)
            print(f"   {cat:<25} conf={cat_conf:.0%}  entity_recall={cat_ent:.0%}")

    # By difficulty
    difficulties = set(r["difficulty"] for r in results)
    if len(difficulties) > 1:
        print(f"\n📊 BY DIFFICULTY:")
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r["difficulty"] == diff]
            if diff_results:
                diff_conf = sum(r["confidence"] for r in diff_results) / len(diff_results)
                print(f"   {diff:<10} conf={diff_conf:.0%}  (n={len(diff_results)})")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Run T-RAG benchmark evaluation."
    )
    parser.add_argument(
        "--provider", default="ollama",
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model name",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="WRS alpha (default: 0.5)",
    )
    parser.add_argument(
        "--lambda-val", type=float, default=0.01,
        help="Decay lambda (default: 0.01)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-k facts (default: 5)",
    )
    parser.add_argument(
        "--output", default="data/evaluation/benchmark_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--queries", default=None,
        help="Comma-separated query IDs to run (default: all)",
    )

    args = parser.parse_args()

    # Load FAISS index
    print("\n📦 Loading FAISS index...")
    vs = VectorSearch()
    try:
        vs.load("data/embeddings")
        print(f"   Index size: {vs.size} vectors")
    except Exception as e:
        print(f"   ❌ Failed to load index: {e}")
        print("   Run `python scripts/generate_embeddings.py` first.")
        return

    # Load encoder and LLM
    encoder = QueryEncoder()
    llm = LLMClient(provider=args.provider, model=args.model)

    # Select queries
    queries = BENCHMARK_QUERIES
    if args.queries:
        selected = set(args.queries.split(","))
        queries = [q for q in queries if q["id"] in selected]

    print(f"\n🚀 Running {len(queries)} benchmark queries ({args.provider})...\n")

    results = []
    for i, query_data in enumerate(queries, 1):
        print(f"   [{i}/{len(queries)}] {query_data['id']}: {query_data['query'][:60]}...")
        try:
            result = run_single_query(
                query_data, vs, encoder, llm,
                alpha=args.alpha,
                lambda_val=args.lambda_val,
                top_k=args.top_k,
            )
            results.append(result)
            print(f"           → conf={result['confidence']:.0%}, "
                  f"latency={result['latency_ms']}ms")
        except Exception as e:
            logger.error(f"   ❌ Query {query_data['id']} failed: {e}")
            results.append({
                "id": query_data["id"],
                "query": query_data["query"],
                "error": str(e),
                "confidence": 0.0,
                "entity_recall": 0.0,
                "temporal_hit": 0.0,
                "latency_ms": 0,
                "facts_used": 0,
                "category": query_data.get("category", ""),
                "difficulty": query_data.get("difficulty", ""),
            })

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": args.provider,
            "model": args.model,
            "alpha": args.alpha,
            "lambda": args.lambda_val,
            "top_k": args.top_k,
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
