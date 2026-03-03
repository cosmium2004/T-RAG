"""
Module 9 — Evaluation Metrics for T-RAG.
MRR, Temporal Accuracy, Hits@K.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def calculate_mrr(predictions: List[List[str]], ground_truths: List[str]) -> float:
    """
    Mean Reciprocal Rank.

    Args:
        predictions: List of ranked prediction lists
        ground_truths: List of correct answers

    Returns:
        MRR score in [0, 1]
    """
    rr_sum = 0.0
    for preds, gt in zip(predictions, ground_truths):
        gt_lower = gt.lower()
        for rank, pred in enumerate(preds, 1):
            if gt_lower in pred.lower():
                rr_sum += 1.0 / rank
                break
    return rr_sum / max(len(ground_truths), 1)


def calculate_temporal_accuracy(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Temporal Accuracy: fraction of predictions containing the correct
    temporal information.
    """
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if gt.lower() in pred.lower():
            correct += 1
    return correct / max(len(ground_truths), 1)


def calculate_hits_at_k(
    predictions: List[List[str]],
    ground_truths: List[str],
    k: int = 10,
) -> float:
    """
    Hits@K: fraction of queries where the correct answer appears
    in the top-K predictions.
    """
    hits = 0
    for preds, gt in zip(predictions, ground_truths):
        top_k = preds[:k]
        if any(gt.lower() in p.lower() for p in top_k):
            hits += 1
    return hits / max(len(ground_truths), 1)


def run_evaluation(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Convenience: compute all metrics from a list of result dicts."""
    predictions = [[r.get("answer", "")] for r in results]
    ground_truths = [r.get("ground_truth", "") for r in results]

    return {
        "mrr": round(calculate_mrr(predictions, ground_truths), 4),
        "temporal_accuracy": round(
            calculate_temporal_accuracy(
                [r.get("answer", "") for r in results], ground_truths
            ), 4
        ),
        "hits_at_1": round(calculate_hits_at_k(predictions, ground_truths, 1), 4),
        "hits_at_5": round(calculate_hits_at_k(predictions, ground_truths, 5), 4),
        "hits_at_10": round(calculate_hits_at_k(predictions, ground_truths, 10), 4),
    }
