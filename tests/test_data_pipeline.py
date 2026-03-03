"""
Unit tests for the T-RAG data pipeline.

Covers: DataFetcher, TimestampParser, EntityExtractor, DuplicateResolver.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_pipeline.fetcher import DataFetcher
from src.data_pipeline.timestamp_parser import TimestampParser
from src.data_pipeline.entity_extractor import EntityExtractor
from src.data_pipeline.duplicate_resolver import DuplicateResolver


# =====================================================================
# DataFetcher
# =====================================================================

class TestDataFetcher:
    @pytest.fixture
    def fetcher(self, tmp_path):
        return DataFetcher(cache_dir=str(tmp_path))

    def test_init_creates_cache_dir(self, fetcher, tmp_path):
        assert fetcher.cache_dir == tmp_path
        assert fetcher.cache_dir.exists()

    def test_load_from_file_success(self, fetcher, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("head\trelation\ttail\tdate\nUSA\tvisit\tChina\t2024-01-01\n")
        df = fetcher.load_from_file(str(csv), ["head", "relation", "tail", "date"])
        assert len(df) == 1
        assert df.iloc[0]["head"] == "USA"

    def test_load_from_file_missing_cols(self, fetcher, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("col_a\tcol_b\n1\t2\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            fetcher.load_from_file(str(csv), ["head", "relation"])

    def test_load_from_file_not_found(self, fetcher):
        with pytest.raises(FileNotFoundError):
            fetcher.load_from_file("nope.csv", ["col"])

    def test_fetch_icews_from_cache(self, fetcher, tmp_path):
        # New fetcher expects numeric IDs: head_id\trel_id\ttail_id\ttime_id\textra
        cache_file = tmp_path / "icews18_train.txt"
        cache_file.write_text("0\t0\t1\t0\t0\n2\t1\t3\t0\t0\n")
        # Also need entity / relation mapping files
        ent_file = tmp_path / "icews18_entity2id.txt"
        ent_file.write_text("USA\t0\nChina\t1\nRussia\t2\nNATO\t3\n")
        rel_file = tmp_path / "icews18_relation2id.txt"
        rel_file.write_text("visit\t0\ncriticize\t1\n")
        df = fetcher.fetch_icews(dataset="icews18", split="train")
        assert len(df) == 2
        assert "event_id" in df.columns
        assert df.iloc[0]["head"] == "USA"

    def test_fetch_icews_with_limit(self, fetcher, tmp_path):
        cache_file = tmp_path / "icews18_train.txt"
        lines = "0\t0\t1\t0\t0\n" * 50
        cache_file.write_text(lines)
        ent_file = tmp_path / "icews18_entity2id.txt"
        ent_file.write_text("USA\t0\nChina\t1\n")
        rel_file = tmp_path / "icews18_relation2id.txt"
        rel_file.write_text("visit\t0\n")
        df = fetcher.fetch_icews(dataset="icews18", split="train", limit=10)
        assert len(df) == 10

    def test_fetch_icews_unknown_dataset(self, fetcher):
        with pytest.raises(ValueError, match="Unknown dataset"):
            fetcher.fetch_icews(dataset="icews99", split="train")


# =====================================================================
# TimestampParser
# =====================================================================

class TestTimestampParser:
    @pytest.fixture
    def parser(self):
        return TimestampParser()

    def test_parse_icews_date(self, parser):
        dt = parser.parse("2014-01-15")
        assert dt is not None
        assert dt.year == 2014
        assert dt.month == 1
        assert dt.day == 15
        assert dt.tzinfo == timezone.utc

    def test_parse_iso_string(self, parser):
        dt = parser.parse("2025-06-15T14:30:00Z")
        assert dt is not None
        assert dt.year == 2025

    def test_parse_natural_language(self, parser):
        dt = parser.parse("January 20, 2025")
        assert dt is not None
        assert dt.month == 1
        assert dt.day == 20

    def test_parse_none_returns_none(self, parser):
        assert parser.parse(None) is None

    def test_parse_empty_string(self, parser):
        assert parser.parse("") is None

    def test_parse_garbage(self, parser):
        assert parser.parse("not_a_date_at_all") is None

    def test_parse_column(self, parser):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-06-15", "bad"]})
        result = parser.parse_column(df, "date")
        assert "date_parsed" in result.columns
        assert "date_iso" in result.columns
        assert result.iloc[0]["date_parsed"].year == 2024
        assert pd.isna(result.iloc[2]["date_parsed"])

    def test_success_rate(self, parser):
        parser.parse("2024-01-01")
        parser.parse("2024-02-01")
        parser.parse("garbage")
        assert 60 < parser.success_rate < 70  # 2/3 ≈ 66.7%


# =====================================================================
# EntityExtractor
# =====================================================================

class TestEntityExtractor:
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()

    def test_extract_quadruples_basic(self, extractor):
        df = pd.DataFrame({
            "event_id": [1, 2],
            "head": ["USA", "Russia"],
            "relation": ["make_statement", "criticize"],
            "tail": ["China", "NATO"],
            "date_parsed": [
                datetime(2014, 1, 15, tzinfo=timezone.utc),
                datetime(2014, 2, 10, tzinfo=timezone.utc),
            ],
        })
        facts = extractor.extract_quadruples(df)
        assert len(facts) == 2
        assert facts[0]["head"] == "Usa"  # title-cased
        assert facts[0]["relation"] == "make statement"  # underscore removed
        assert facts[0]["text"] == "Usa make statement China"

    def test_drops_empty_entities(self, extractor):
        df = pd.DataFrame({
            "event_id": [1],
            "head": [""],
            "relation": ["visit"],
            "tail": ["China"],
            "date_parsed": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        })
        facts = extractor.extract_quadruples(df)
        assert len(facts) == 0
        assert extractor.stats["dropped"] == 1

    def test_build_text(self):
        text = EntityExtractor._build_text("Usa", "visit", "China")
        assert text == "Usa visit China"


# =====================================================================
# DuplicateResolver
# =====================================================================

class TestDuplicateResolver:
    @pytest.fixture
    def resolver(self):
        return DuplicateResolver()

    def test_no_duplicates(self, resolver):
        facts = [
            {"head": "A", "relation": "r", "tail": "B", "confidence": 0.9,
             "start_time": "2024-01-01", "end_time": None, "source": "X"},
            {"head": "C", "relation": "s", "tail": "D", "confidence": 0.8,
             "start_time": "2024-02-01", "end_time": None, "source": "Y"},
        ]
        result = resolver.resolve(facts)
        assert len(result) == 2

    def test_exact_duplicates_merged(self, resolver):
        facts = [
            {"head": "USA", "relation": "visit", "tail": "China",
             "confidence": 0.9, "start_time": "2024-01-01",
             "end_time": None, "source": "ICEWS"},
            {"head": "usa", "relation": "visit", "tail": "china",
             "confidence": 0.8, "start_time": "2024-01-01",
             "end_time": None, "source": "Wikidata"},
        ]
        result = resolver.resolve(facts)
        assert len(result) == 1
        assert "ICEWS" in result[0]["source"]
        assert "Wikidata" in result[0]["source"]

    def test_time_range_merged(self, resolver):
        facts = [
            {"head": "A", "relation": "r", "tail": "B",
             "confidence": 0.9, "start_time": "2024-01-01",
             "end_time": "2024-06-01", "source": "S1"},
            {"head": "A", "relation": "r", "tail": "B",
             "confidence": 0.8, "start_time": "2024-03-01",
             "end_time": "2024-12-01", "source": "S2"},
        ]
        result = resolver.resolve(facts)
        assert len(result) == 1
        assert result[0]["start_time"] == "2024-01-01"
        assert result[0]["end_time"] == "2024-12-01"

    def test_stats(self, resolver):
        facts = [
            {"head": "A", "relation": "r", "tail": "B",
             "confidence": 0.9, "start_time": "2024-01-01",
             "end_time": None, "source": "X"},
        ] * 3
        resolver.resolve(facts)
        assert resolver.stats["total_input"] == 3
        assert resolver.stats["duplicates_removed"] == 2
        assert resolver.stats["output_count"] == 1
