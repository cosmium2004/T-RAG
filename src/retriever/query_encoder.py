"""
Module 4.1 — Query Encoder for T-RAG retriever.
Converts natural-language queries to vector representations.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class QueryEncoder:
    """Encodes user queries to 768-dim vectors using Sentence-BERT."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None
        self.dim: int = 768

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading query encoder: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, query: str) -> np.ndarray:
        """Encode a query string to a 1-D float32 vector."""
        self._load_model()
        vec = self._model.encode([query], convert_to_numpy=True)
        return vec[0].astype("float32")
