"""
Module 1.1 — Data Fetcher for T-RAG pipeline.
Handles downloading and loading ICEWS datasets with ID resolution.

ICEWS datasets from RE-Net use numeric IDs for entities, relations,
and timestamps.  This module downloads the mapping files automatically
and resolves IDs to human-readable names.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ── RE-Net GitHub base URLs ──────────────────────────────────────────
_RENET_BASE = "https://github.com/INK-USC/RE-Net/raw/master/data"

DATASET_URLS: Dict[str, Dict[str, str]] = {
    "icews14": {
        "train": f"{_RENET_BASE}/ICEWS14/train.txt",
        "valid": f"{_RENET_BASE}/ICEWS14/valid.txt",
        "test":  f"{_RENET_BASE}/ICEWS14/test.txt",
        "entity2id": f"{_RENET_BASE}/ICEWS14/entity2id.txt",
        "relation2id": f"{_RENET_BASE}/ICEWS14/relation2id.txt",
        "stat": f"{_RENET_BASE}/ICEWS14/stat.txt",
    },
    "icews18": {
        "train": f"{_RENET_BASE}/ICEWS18/train.txt",
        "valid": f"{_RENET_BASE}/ICEWS18/valid.txt",
        "test":  f"{_RENET_BASE}/ICEWS18/test.txt",
        "entity2id": f"{_RENET_BASE}/ICEWS18/entity2id.txt",
        "relation2id": f"{_RENET_BASE}/ICEWS18/relation2id.txt",
        "stat": f"{_RENET_BASE}/ICEWS18/stat.txt",
    },
}

# ICEWS14 covers events from 2014-01-01 to 2014-12-31
# ICEWS18 covers events from 2018-01-01 to 2018-10-31
DATASET_EPOCH: Dict[str, datetime] = {
    "icews14": datetime(2014, 1, 1, tzinfo=timezone.utc),
    "icews18": datetime(2018, 1, 1, tzinfo=timezone.utc),
}


class DataFetcher:
    """Fetches and loads temporal knowledge datasets with ID resolution."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._entity_map: Dict[int, str] = {}
        self._relation_map: Dict[int, str] = {}
        logger.info(f"DataFetcher initialized with cache_dir: {cache_dir}")

    # ── Download ─────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _download_file(self, url: str, output_path: Path) -> None:
        """Download a file with progress bar and retry logic."""
        logger.info(f"Downloading {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                      desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        logger.info(f"Saved → {output_path}")

    def _ensure_cached(self, url: str, local_name: str,
                       force: bool = False) -> Path:
        """Download if not cached; return local path."""
        path = self.cache_dir / local_name
        if not path.exists() or force:
            self._download_file(url, path)
        return path

    # ── Mapping loaders ──────────────────────────────────────────────

    def _load_entity_map(self, dataset: str, force: bool = False) -> Dict[int, str]:
        """Load entity-name → id mapping, invert to id → name."""
        urls = DATASET_URLS.get(dataset)
        if not urls or "entity2id" not in urls:
            return {}
        path = self._ensure_cached(
            urls["entity2id"], f"{dataset}_entity2id.txt", force
        )
        mapping: Dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                name, eid = parts
                mapping[int(eid)] = name.strip()
        logger.info(f"Loaded {len(mapping)} entity mappings for {dataset}")
        return mapping

    def _load_relation_map(self, dataset: str,
                           force: bool = False) -> Dict[int, str]:
        """Load relation-name → id mapping, invert to id → name."""
        urls = DATASET_URLS.get(dataset)
        if not urls or "relation2id" not in urls:
            return {}
        path = self._ensure_cached(
            urls["relation2id"], f"{dataset}_relation2id.txt", force
        )
        mapping: Dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                name, rid = parts
                mapping[int(rid)] = name.strip()
        logger.info(f"Loaded {len(mapping)} relation mappings for {dataset}")
        return mapping

    # ── Main fetch ───────────────────────────────────────────────────

    def fetch_icews(
        self,
        dataset: str = "icews18",
        split: str = "train",
        force_download: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch an ICEWS dataset split with resolved entity/relation names.

        Returns DataFrame with columns:
            [event_id, head, relation, tail, date]
        where ``head``, ``relation``, ``tail`` are human-readable strings
        and ``date`` is an ISO-8601 date string.
        """
        urls = DATASET_URLS.get(dataset)
        if urls is None:
            raise ValueError(
                f"Unknown dataset: {dataset}. "
                f"Available: {list(DATASET_URLS)}"
            )

        # 1. Download raw data
        data_path = self._ensure_cached(
            urls[split], f"{dataset}_{split}.txt", force_download
        )

        # 2. Load mapping tables
        self._entity_map = self._load_entity_map(dataset, force_download)
        self._relation_map = self._load_relation_map(dataset, force_download)

        # 3. Parse the tab-separated file (columns: head_id, relation_id,
        #    tail_id, timestamp_id, ???)
        df = pd.read_csv(
            data_path, sep="\t", header=None, nrows=limit
        )
        # RE-Net format has 5 columns; take first 4
        df = df.iloc[:, :4]
        df.columns = ["head_id", "relation_id", "tail_id", "date_id"]

        # 4. Resolve IDs → names
        epoch = DATASET_EPOCH.get(dataset, datetime(2014, 1, 1, tzinfo=timezone.utc))
        df["head"] = df["head_id"].map(self._entity_map).fillna(
            df["head_id"].astype(str)
        )
        df["relation"] = df["relation_id"].map(self._relation_map).fillna(
            df["relation_id"].astype(str)
        )
        df["tail"] = df["tail_id"].map(self._entity_map).fillna(
            df["tail_id"].astype(str)
        )
        df["date"] = df["date_id"].apply(
            lambda d: (epoch + timedelta(days=int(d))).strftime("%Y-%m-%d")
        )

        df.insert(0, "event_id", range(1, len(df) + 1))
        df = df[["event_id", "head", "relation", "tail", "date"]]

        logger.info(
            f"Loaded {len(df)} records from {dataset}/{split} "
            f"({len(self._entity_map)} entities, "
            f"{len(self._relation_map)} relations)"
        )
        return df

    # ── Generic file loader ──────────────────────────────────────────

    def load_from_file(
        self,
        filepath: str,
        expected_cols: List[str],
        sep: str = "\t",
    ) -> pd.DataFrame:
        """Load and validate a local data file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath, sep=sep)
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df
