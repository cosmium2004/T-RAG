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

    def test_per_relation_lambda(self):
        """Test that calculate_fvs uses per-relation λ when available."""
        from unittest.mock import MagicMock
        d = DecayFunction(default_lambda=0.01)
        # Mock relation rates
        d._relation_rates = MagicMock()
        d._relation_rates.get_lambda.return_value = 0.1  # 10x faster decay

        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        old = now - timedelta(days=30)

        fvs_default = d.calculate_fvs(old, now)  # no relation → default_lambda
        fvs_fast = d.calculate_fvs(old, now, relation="sanctions")

        # With 10x faster λ, the fact should decay much more
        assert fvs_fast < fvs_default

    def test_score_facts_with_start_time_fallback(self):
        """Test that score_facts falls back to start_time when last_verified is None."""
        d = DecayFunction()
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [
            {
                "last_verified": None,
                "start_time": (now - timedelta(days=5)).isoformat(),
                "relation": "sanctions",
            },
        ]
        scored = d.score_facts(facts, now)
        assert scored[0]["fvs"] > 0.0
        assert scored[0]["fvs"] < 1.0


class TestRelationDecayRates:
    def test_learn_rates_basic(self, tmp_path):
        from src.deprecation.decay import RelationDecayRates

        rates_path = str(tmp_path / "test_rates.json")
        learner = RelationDecayRates(rates_path=rates_path)

        # Create facts with known intervals
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        facts = [
            {"relation": "sanctions", "start_time": (now - timedelta(days=30)).isoformat()},
            {"relation": "sanctions", "start_time": (now - timedelta(days=20)).isoformat()},
            {"relation": "sanctions", "start_time": (now - timedelta(days=10)).isoformat()},
            {"relation": "sanctions", "start_time": now.isoformat()},
            {"relation": "visits", "start_time": (now - timedelta(days=100)).isoformat()},
            {"relation": "visits", "start_time": (now - timedelta(days=50)).isoformat()},
            {"relation": "visits", "start_time": now.isoformat()},
        ]
        rates = learner.learn_rates(facts)

        assert "sanctions" in rates
        assert "visits" in rates
        # Sanctions updates every ~10 days → higher λ
        # Visits updates every ~50 days → lower λ
        assert rates["sanctions"] > rates["visits"]

    def test_learn_rates_insufficient_data(self, tmp_path):
        from src.deprecation.decay import RelationDecayRates

        learner = RelationDecayRates(rates_path=str(tmp_path / "rates.json"))
        facts = [
            {"relation": "rare_event", "start_time": "2024-01-01"},
            {"relation": "rare_event", "start_time": "2024-06-01"},
        ]
        rates = learner.learn_rates(facts, default_lambda=0.05)
        # Only 2 data points → should use default
        assert rates["rare_event"] == 0.05

    def test_get_lambda_fallback(self, tmp_path):
        from src.deprecation.decay import RelationDecayRates

        learner = RelationDecayRates(rates_path=str(tmp_path / "rates.json"))
        learner.rates = {"sanctions": 0.1}
        assert learner.get_lambda("sanctions") == 0.1
        assert learner.get_lambda("unknown_relation", default=0.02) == 0.02

    def test_persistence(self, tmp_path):
        from src.deprecation.decay import RelationDecayRates

        rates_path = str(tmp_path / "persist.json")
        learner = RelationDecayRates(rates_path=rates_path)
        learner.rates = {"test": 0.05}
        learner._save()

        # Load in a new instance
        loaded = RelationDecayRates(rates_path=rates_path)
        assert loaded.rates["test"] == 0.05


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


class TestLLMClientOllama:
    def test_ollama_init(self):
        llm = LLMClient(provider="ollama")
        assert llm.provider == "ollama"
        assert llm.model == "llama3"  # default
        assert "11434" in llm.api_base
        assert llm.api_key == "ollama"

    def test_ollama_custom_model(self):
        llm = LLMClient(provider="ollama", model="phi3")
        assert llm.model == "phi3"

    def test_ollama_health_check(self):
        llm = LLMClient(provider="ollama")
        health = llm.health_check()
        assert health["provider"] == "ollama"
        assert health["model"] == "llama3"
        # Status will be "unreachable" if Ollama isn't running
        assert "status" in health

    def test_ollama_generate_mocked(self):
        """Verify Ollama calls OpenAI SDK with correct base_url."""
        from unittest.mock import patch, MagicMock

        llm = LLMClient(provider="ollama", model="llama3")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer from Ollama"

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            msgs = [{"role": "user", "content": "Hello"}]
            answer = llm.generate(msgs)

            # Verify correct base_url was passed
            mock_openai.assert_called_once_with(
                base_url=llm.api_base,
                api_key="ollama",
            )
            assert answer == "Test answer from Ollama"


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


class TestBenchmarkData:
    def test_benchmark_queries_exist(self):
        from src.evaluation.benchmark_data import BENCHMARK_QUERIES
        assert len(BENCHMARK_QUERIES) >= 10
        for q in BENCHMARK_QUERIES:
            assert "id" in q
            assert "query" in q
            assert "query_date" in q
            assert "category" in q

    def test_benchmark_categories(self):
        from src.evaluation.benchmark_data import BENCHMARK_QUERIES
        categories = set(q["category"] for q in BENCHMARK_QUERIES)
        assert len(categories) >= 3  # at least 3 categories


class TestUpgradedContextAssembler:
    def test_conflict_detection(self):
        from src.retriever.context_assembler import ContextAssembler
        ca = ContextAssembler()
        facts = [
            {"head": "USA", "tail": "China", "relation": "sanctions",
             "text": "USA sanctions China", "start_time": "2024-01-01"},
            {"head": "USA", "tail": "China", "relation": "cooperates with",
             "text": "USA cooperates with China", "start_time": "2024-03-01"},
        ]
        context = ca.format(facts)
        assert len(ca.conflicts) >= 1
        assert "conflict" in context.lower() or "⚠" in context

    def test_empty_facts(self):
        from src.retriever.context_assembler import ContextAssembler
        ca = ContextAssembler()
        context = ca.format([])
        assert "No relevant facts" in context

    def test_entity_summary(self):
        from src.retriever.context_assembler import ContextAssembler
        ca = ContextAssembler()
        facts = [
            {"head": "Russia", "tail": "Ukraine", "relation": "invades",
             "text": "Russia invades Ukraine", "start_time": "2022-02-24"},
        ]
        context = ca.format(facts)
        assert "Russia" in context
        assert "Ukraine" in context


class TestUpgradedPostProcessor:
    def test_temporal_normalization(self):
        from src.generator.post_processor import PostProcessor
        pp = PostProcessor()
        text = "The event on 2024-01-15T14:30:00+00:00 was significant."
        result = pp.process(text)
        assert "2024-01-15" in result
        assert "T14:30" not in result

    def test_staleness_warning(self):
        from src.generator.post_processor import PostProcessor
        pp = PostProcessor()
        stale_facts = [{"fvs": 0.1, "confidence": 0.3}]
        result = pp.process("Some answer text.", source_facts=stale_facts)
        assert "Caution" in result or "⚠" in result

    def test_no_warning_for_fresh(self):
        from src.generator.post_processor import PostProcessor
        pp = PostProcessor()
        fresh_facts = [
            {"fvs": 0.9, "confidence": 0.8},
            {"fvs": 0.85, "confidence": 0.75},
        ]
        result = pp.process("Some answer text.", source_facts=fresh_facts)
        assert "Caution" not in result
