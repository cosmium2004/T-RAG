"""
Tests for Phases 3-7 modules: deprecation, retrieval, generation, validation.
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Phase 5: Deprecation ────────────────────────────────────────────

from src.deprecation.decay import DecayFunction
from src.deprecation.classifier import DeprecationClassifier
from src.deprecation.update_tracker import UpdateTracker


class TestDecayFunction:
    def test_fresh_fact(self):
        d = DecayFunction(default_lambda=0.01)
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        fvs = d.calculate_fvs(now, now)
        assert fvs == 1.0

    def test_old_fact_decays(self):
        d = DecayFunction(default_lambda=0.01)
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        old = now - timedelta(days=100)
        fvs = d.calculate_fvs(old, now)
        assert 0.0 < fvs < 0.5

    def test_very_old_fact(self):
        d = DecayFunction(default_lambda=0.01)
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        old = now - timedelta(days=500)
        fvs = d.calculate_fvs(old, now)
        assert fvs < 0.01

    def test_is_valid(self):
        d = DecayFunction(default_lambda=0.01)
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        assert d.is_valid(now, now) is True
        assert d.is_valid(now - timedelta(days=500), now) is False

    def test_score_facts(self):
        d = DecayFunction()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [
            {"last_verified": (now - timedelta(days=1)).isoformat()},
            {"last_verified": None},
        ]
        scored = d.score_facts(facts, now)
        assert scored[0]["fvs"] > 0.99
        assert scored[1]["fvs"] == 0.0


class TestDeprecationClassifier:
    def test_fresh_fact_valid(self):
        c = DeprecationClassifier()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        fact = {
            "start_time": (now - timedelta(days=10)).isoformat(),
            "end_time": None,
            "last_verified": now.isoformat(),
        }
        result = c.classify(fact, now)
        assert result["deprecated"] is False

    def test_ended_fact_deprecated(self):
        c = DeprecationClassifier()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        fact = {
            "start_time": (now - timedelta(days=100)).isoformat(),
            "end_time": (now - timedelta(days=50)).isoformat(),
            "last_verified": (now - timedelta(days=50)).isoformat(),
        }
        result = c.classify(fact, now)
        assert result["deprecated"] is True

    def test_filter_valid(self):
        c = DeprecationClassifier()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [
            {"start_time": now.isoformat(), "end_time": None,
             "last_verified": now.isoformat()},
            {"start_time": None, "end_time": None, "last_verified": None},
        ]
        valid = c.filter_valid(facts, now)
        assert len(valid) == 1


class TestUpdateTracker:
    def test_log_and_retrieve(self, tmp_path):
        db = str(tmp_path / "test.db")
        tracker = UpdateTracker(db_path=db)
        tracker.log_verification("fact_1", source="test")
        history = tracker.get_history("fact_1")
        assert len(history) == 1
        assert history[0]["fact_id"] == "fact_1"

    def test_stats(self, tmp_path):
        db = str(tmp_path / "test2.db")
        tracker = UpdateTracker(db_path=db)
        tracker.log_verification("f1")
        tracker.log_verification("f2")
        tracker.log_verification("f1")
        stats = tracker.get_stats()
        assert stats["total_entries"] == 3
        assert stats["unique_facts_tracked"] == 2


# ── Phase 6: Retrieval ──────────────────────────────────────────────

from src.retriever.temporal_filter import TemporalFilter
from src.retriever.wrs import WRSScorer
from src.retriever.context_assembler import ContextAssembler


class TestTemporalFilter:
    def test_valid_fact_passes(self):
        tf = TemporalFilter()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [{"start_time": "2024-01-01T00:00:00+00:00", "end_time": None}]
        assert len(tf.filter(facts, now)) == 1

    def test_future_fact_rejected(self):
        tf = TemporalFilter()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [{"start_time": "2025-01-01T00:00:00+00:00", "end_time": None}]
        assert len(tf.filter(facts, now)) == 0

    def test_ended_fact_rejected(self):
        tf = TemporalFilter()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [{"start_time": "2023-01-01T00:00:00+00:00",
                  "end_time": "2023-12-31T00:00:00+00:00"}]
        assert len(tf.filter(facts, now)) == 0

    def test_low_fvs_rejected(self):
        tf = TemporalFilter(deprecation_threshold=0.5)
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [{"start_time": "2024-01-01T00:00:00+00:00", "fvs": 0.2}]
        assert len(tf.filter(facts, now)) == 0


class TestWRSScorer:
    def test_balanced_score(self):
        wrs = WRSScorer(alpha=0.5)
        fact = {"similarity": 0.8, "fvs": 0.6}
        score = wrs.score(fact)
        assert abs(score - 0.7) < 0.001

    def test_ranking(self):
        wrs = WRSScorer(alpha=0.5)
        facts = [
            {"similarity": 0.5, "fvs": 0.5, "id": "low"},
            {"similarity": 0.9, "fvs": 0.9, "id": "high"},
            {"similarity": 0.7, "fvs": 0.7, "id": "mid"},
        ]
        ranked = wrs.rank(facts, top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["id"] == "high"


class TestContextAssembler:
    def test_format(self):
        ca = ContextAssembler()
        facts = [
            {"start_time": "2024-01-01", "end_time": None,
             "text": "A did B", "source": "S", "confidence": 0.9,
             "fvs": 0.95, "wrs": 0.87},
        ]
        ctx = ca.format(facts)
        assert "A did B" in ctx
        assert "2024-01-01" in ctx


# ── Phase 7: Generation & Validation ────────────────────────────────

from src.generator.prompt_builder import PromptBuilder
from src.generator.llm_client import LLMClient
from src.generator.post_processor import PostProcessor
from src.validator.consistency import ConsistencyValidator
from src.validator.confidence import ConfidenceScorer


class TestPromptBuilder:
    def test_build_prompt(self):
        pb = PromptBuilder()
        prompt = pb.build_prompt("What happened?", "Context here")
        assert "What happened?" in prompt
        assert "Context here" in prompt

    def test_build_messages(self):
        pb = PromptBuilder()
        msgs = pb.build_messages("Q", "C")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestLLMClient:
    def test_local_fallback(self):
        llm = LLMClient(provider="local")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question: What?\n\nContext:\n1. Fact A\n\nInstructions:\nAnswer"},
        ]
        answer = llm.generate(msgs)
        assert len(answer) > 0

    def test_health_check(self):
        llm = LLMClient(provider="local")
        health = llm.health_check()
        assert health["provider"] == "local"


class TestPostProcessor:
    def test_remove_preamble(self):
        pp = PostProcessor()
        result = pp.process("Based on the context, North Korea did X.")
        assert not result.startswith("Based on the context")

    def test_whitespace(self):
        pp = PostProcessor()
        result = pp.process("Too   many    spaces\n\n\n\nextra lines")
        assert "   " not in result


class TestConsistencyValidator:
    def test_valid_response(self):
        cv = ConsistencyValidator()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [{"text": "North Korea make statement"}]
        result = cv.validate("North Korea made a statement.", now, facts)
        assert result["is_valid"] is True

    def test_no_temporal_claims(self):
        cv = ConsistencyValidator()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        result = cv.validate("Hello world.", now, [])
        assert result["claims_checked"] == 0


class TestConfidenceScorer:
    def test_high_confidence(self):
        cs = ConfidenceScorer()
        validation = {"consistency_score": 1.0}
        facts = [
            {"fvs": 0.95, "confidence": 0.9},
            {"fvs": 0.90, "confidence": 0.85},
            {"fvs": 0.92, "confidence": 0.88},
            {"fvs": 0.91, "confidence": 0.87},
            {"fvs": 0.93, "confidence": 0.89},
        ]
        result = cs.score(validation, facts)
        assert result["rating"] == "HIGH"

    def test_low_confidence(self):
        cs = ConfidenceScorer()
        validation = {"consistency_score": 0.2}
        facts = [{"fvs": 0.1, "confidence": 0.2}]
        result = cs.score(validation, facts)
        assert result["rating"] == "LOW"


# ── Phase 9: Evaluation ─────────────────────────────────────────────

from src.evaluation.metrics import calculate_mrr, calculate_hits_at_k, calculate_temporal_accuracy


class TestEvaluationMetrics:
    def test_mrr_perfect(self):
        preds = [["correct"]]
        gts = ["correct"]
        assert calculate_mrr(preds, gts) == 1.0

    def test_mrr_second(self):
        preds = [["wrong", "correct"]]
        gts = ["correct"]
        assert calculate_mrr(preds, gts) == 0.5

    def test_hits_at_1(self):
        preds = [["correct", "other"]]
        gts = ["correct"]
        assert calculate_hits_at_k(preds, gts, k=1) == 1.0

    def test_temporal_accuracy(self):
        preds = ["In 2024, X happened"]
        gts = ["2024"]
        assert calculate_temporal_accuracy(preds, gts) == 1.0
