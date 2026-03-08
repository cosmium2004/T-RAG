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


# =====================================================================
# DocumentLoader
# =====================================================================

from src.data_pipeline.document_loader import DocumentLoader


class TestDocumentLoader:
    def test_load_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world. This is a test document.", encoding="utf-8")
        loader = DocumentLoader()
        pages = loader.load_text(str(f))
        assert len(pages) == 1
        assert "Hello world" in pages[0]

    def test_load_text_not_found(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_text("nonexistent.txt")

    def test_load_autodetect_text(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Heading\nSome content.", encoding="utf-8")
        loader = DocumentLoader()
        pages = loader.load(str(f))
        assert len(pages) == 1
        assert "Heading" in pages[0]

    def test_load_url_mocked(self):
        from unittest.mock import patch, MagicMock

        loader = DocumentLoader()
        mock_resp = MagicMock()
        mock_resp.text = """
        <html><body>
            <p>North Korea launched a missile on January 15, 2024.</p>
            <p>The United Nations condemned the action immediately.</p>
        </body></html>
        """
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            pages = loader.load_url("https://example.com/article")
            assert len(pages) == 1
            assert "North Korea" in pages[0]


# =====================================================================
# TextChunker
# =====================================================================

from src.data_pipeline.text_chunker import TextChunker


class TestTextChunker:
    def test_short_text_no_split(self):
        chunker = TextChunker(chunk_size=1000)
        result = chunker.chunk("This is a short sentence.")
        assert len(result) == 1

    def test_long_text_splits(self):
        # Create text longer than chunk_size
        sentences = [f"Sentence number {i} is here." for i in range(50)]
        text = " ".join(sentences)
        chunker = TextChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_overlap_present(self):
        sentences = [f"Sentence {i} about topic {i}." for i in range(30)]
        text = " ".join(sentences)
        chunker = TextChunker(chunk_size=200, overlap=80)
        chunks = chunker.chunk(text)
        # With overlap, some content should appear in consecutive chunks
        if len(chunks) >= 2:
            # Last words of chunk[0] should appear in chunk[1]
            words_0 = set(chunks[0].split()[-5:])
            words_1 = set(chunks[1].split()[:10])
            assert len(words_0 & words_1) > 0

    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunk_pages(self):
        chunker = TextChunker(chunk_size=500)
        pages = ["Page one content here.", "Page two content here."]
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) >= 1


# =====================================================================
# QuadrupleExtractor
# =====================================================================

from src.data_pipeline.quadruple_extractor import QuadrupleExtractor


class TestQuadrupleExtractor:
    def test_parse_valid_json(self):
        extractor = QuadrupleExtractor(provider="local")
        raw = '''[
            {"head": "North Korea", "relation": "launched", "tail": "missile",
             "date": "2024-01-15", "confidence": 0.9},
            {"head": "United Nations", "relation": "condemned", "tail": "North Korea",
             "date": "2024-01-16", "confidence": 0.85}
        ]'''
        facts = extractor._parse_response(raw)
        assert len(facts) == 2
        assert facts[0]["head"] == "North Korea"
        assert facts[0]["relation"] == "launched"
        assert facts[0]["confidence"] == 0.9
        assert facts[0]["start_time"] is not None

    def test_parse_with_markdown_fences(self):
        extractor = QuadrupleExtractor(provider="local")
        raw = '''```json
        [{"head": "USA", "relation": "visited", "tail": "China",
          "date": "2024-03-01", "confidence": 0.8}]
        ```'''
        facts = extractor._parse_response(raw)
        assert len(facts) == 1
        assert facts[0]["head"] == "Usa"  # title-cased

    def test_parse_malformed_json(self):
        extractor = QuadrupleExtractor(provider="local")
        facts = extractor._parse_response("Not valid JSON at all")
        assert len(facts) == 0

    def test_parse_missing_fields(self):
        extractor = QuadrupleExtractor(provider="local")
        raw = '[{"head": "A", "relation": "", "tail": "B"}]'
        facts = extractor._parse_response(raw)
        assert len(facts) == 0  # empty relation should be dropped

    def test_deduplicate(self):
        facts = [
            {"head": "USA", "relation": "visit", "tail": "China", "id": "1"},
            {"head": "usa", "relation": "visit", "tail": "china", "id": "2"},
            {"head": "Russia", "relation": "criticize", "tail": "NATO", "id": "3"},
        ]
        deduped = QuadrupleExtractor._deduplicate(facts)
        assert len(deduped) == 2

    def test_extract_mocked(self):
        from unittest.mock import patch, MagicMock

        extractor = QuadrupleExtractor(provider="local")
        mock_response = '[{"head": "France", "relation": "signed treaty with", "tail": "Germany", "date": "2024-06-01", "confidence": 0.9}]'

        with patch.object(extractor.llm, "generate", return_value=mock_response):
            facts = extractor.extract("France signed a treaty with Germany in June 2024.")
            assert len(facts) == 1
            assert facts[0]["head"] == "France"


# =====================================================================
# GDELTFetcher
# =====================================================================

from src.data_pipeline.gdelt_fetcher import GDELTFetcher


class TestGDELTFetcher:
    def test_parse_gdelt_date(self):
        result = GDELTFetcher._parse_gdelt_date("20240115T143000Z")
        assert result is not None
        assert "2024-01-15" in result

    def test_parse_gdelt_date_empty(self):
        assert GDELTFetcher._parse_gdelt_date("") is None
        assert GDELTFetcher._parse_gdelt_date(None) is None

    def test_parse_articles_with_mocked_llm(self):
        from unittest.mock import patch, MagicMock

        fetcher = GDELTFetcher()
        articles = [
            {
                "title": "Russia sanctions Ukraine amid tensions",
                "url": "https://example.com/1",
                "domain": "example.com",
                "seendate": "20240301T120000Z",
                "language": "English",
            },
        ]

        # Mock the QuadrupleExtractor to return a known fact
        mock_facts = [{
            "head": "Russia",
            "relation": "sanctions",
            "tail": "Ukraine",
            "start_time": "2024-03-01T12:00:00+00:00",
            "confidence": 0.9,
            "source": "document",
            "text": "Russia sanctions Ukraine",
            "id": "doc_abc12345",
        }]

        with patch(
            "src.data_pipeline.gdelt_fetcher.QuadrupleExtractor"
        ) as MockExtractor:
            mock_instance = MagicMock()
            MockExtractor.return_value = mock_instance
            mock_instance.extract.return_value = mock_facts
            mock_instance._deduplicate.return_value = mock_facts

            facts = fetcher._parse_articles(articles)
            assert len(facts) >= 1
            assert facts[0]["source"].startswith("GDELT")
            assert facts[0]["id"].startswith("gdelt_")

    def test_caching(self, tmp_path):
        fetcher = GDELTFetcher(cache_dir=str(tmp_path))
        key = fetcher._cache_key("test", 7, 100)

        assert fetcher._load_cache(key) is None

        test_facts = [{"head": "A", "relation": "r", "tail": "B"}]
        fetcher._save_cache(key, test_facts)
        loaded = fetcher._load_cache(key)
        assert loaded == test_facts

    def test_fetch_recent_mocked(self):
        from unittest.mock import patch, MagicMock

        fetcher = GDELTFetcher(cache_dir="data/cache/gdelt_test")
        mock_response = {
            "articles": [
                {
                    "title": "NATO condemns Russia over missile strike",
                    "url": "https://example.com/article",
                    "domain": "example.com",
                    "seendate": "20240315T100000Z",
                    "language": "English",
                },
            ]
        }

        mock_facts = [{
            "head": "Nato",
            "relation": "condemns",
            "tail": "Russia",
            "start_time": "2024-03-15",
            "confidence": 0.85,
            "source": "document",
            "text": "NATO condemns Russia",
            "id": "doc_xyz",
        }]

        with patch.object(fetcher, "_api_request", return_value=mock_response):
            with patch.object(fetcher, "_load_cache", return_value=None):
                with patch.object(fetcher, "_save_cache"):
                    with patch(
                        "src.data_pipeline.gdelt_fetcher.QuadrupleExtractor"
                    ) as MockExtractor:
                        mock_inst = MagicMock()
                        MockExtractor.return_value = mock_inst
                        mock_inst.extract.return_value = mock_facts
                        mock_inst._deduplicate.return_value = mock_facts

                        facts = fetcher.fetch_recent(query="NATO", days_back=7)
                        assert isinstance(facts, list)
                        assert len(facts) >= 1

