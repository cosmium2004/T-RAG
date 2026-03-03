"""
Module 1.5 — Embedding Generator for T-RAG pipeline.
Generates 768-dim sentence embeddings using Sentence-BERT.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates vector embeddings for fact text using Sentence-BERT.

    Uses ``all-mpnet-base-v2`` by default (768 dimensions).
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None
        self.dim: int = 768

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded — dim={self.dim}")

    # ── Single / batch encoding ──────────────────────────────────────

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts to a (N, dim) float32 array."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], show_progress=False)[0]

    # ── Fact-level helpers ───────────────────────────────────────────

    def embed_facts(
        self,
        facts: List[Dict[str, Any]],
        text_key: str = "text",
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of fact dicts.

        Args:
            facts: List of fact dicts (must have ``text_key`` field)
            text_key: Key to use for the text to embed
            batch_size: Encoding batch size

        Returns:
            (N, dim) float32 numpy array
        """
        texts = [f.get(text_key, "") for f in facts]
        logger.info(f"Embedding {len(texts)} facts (batch_size={batch_size})...")
        embeddings = self.encode(texts, batch_size=batch_size)
        logger.info(f"Generated embeddings: shape={embeddings.shape}")
        return embeddings

    # ── Persistence ──────────────────────────────────────────────────

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        fact_ids: List[str],
        output_dir: str = "data/embeddings",
    ) -> None:
        """Save embeddings and their ID mapping to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "embeddings.npy", embeddings)

        with open(out / "fact_ids.json", "w") as f:
            json.dump(fact_ids, f)

        logger.info(
            f"Saved {len(fact_ids)} embeddings to {out} "
            f"({embeddings.nbytes / 1024**2:.1f} MB)"
        )

    @staticmethod
    def load_embeddings(
        input_dir: str = "data/embeddings",
    ) -> tuple:
        """Load embeddings and ID mapping from disk."""
        inp = Path(input_dir)
        embeddings = np.load(inp / "embeddings.npy")
        with open(inp / "fact_ids.json") as f:
            fact_ids = json.load(f)
        logger.info(f"Loaded {len(fact_ids)} embeddings, shape={embeddings.shape}")
        return embeddings, fact_ids
