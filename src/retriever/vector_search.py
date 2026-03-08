"""
Module 4.2 — FAISS Vector Search for T-RAG retriever.
Builds and queries a FAISS index for semantic similarity search.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorSearch:
    """
    FAISS-backed approximate nearest-neighbour search.

    Supports building an index from embeddings, saving/loading,
    and querying for top-N similar vectors.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        self._index = None
        self._fact_ids: List[str] = []
        self._facts_meta: Dict[str, Dict] = {}

    # ── Build ────────────────────────────────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        fact_ids: List[str],
        facts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Build a FAISS FlatIP (inner-product / cosine) index.

        Args:
            embeddings: (N, dim) float32 array
            fact_ids: Parallel list of fact IDs
            facts: Optional full fact dicts for metadata lookup
        """
        import faiss

        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim {embeddings.shape[1]} != expected {self.dim}"
        )

        # Normalise for cosine similarity
        faiss.normalize_L2(embeddings)

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(embeddings)
        self._fact_ids = list(fact_ids)

        if facts:
            self._facts_meta = {f["id"]: f for f in facts}

        logger.info(
            f"FAISS index built: {self._index.ntotal} vectors, dim={self.dim}"
        )

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_n: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for the top-N most similar facts.

        Args:
            query_vector: 1-D float32 array (dim,)
            top_n: Number of results to return

        Returns:
            List of dicts with ``fact_id``, ``similarity``, and
            any metadata from the stored facts.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        qv = query_vector.reshape(1, -1).astype("float32").copy()
        faiss.normalize_L2(qv)

        distances, indices = self._index.search(qv, top_n)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            fid = self._fact_ids[idx]
            entry = {
                "fact_id": fid,
                "similarity": float(dist),
            }
            if fid in self._facts_meta:
                entry.update(self._facts_meta[fid])
            results.append(entry)

        return results

    # ── Incremental update ───────────────────────────────────────────

    def append_to_index(
        self,
        new_embeddings: np.ndarray,
        new_fact_ids: List[str],
        new_facts: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add new vectors to the existing FAISS index.

        Args:
            new_embeddings: (M, dim) float32 array of new vectors.
            new_fact_ids: Parallel list of new fact IDs.
            new_facts: Optional full fact dicts for metadata.

        Returns:
            Number of new vectors added.
        """
        import faiss

        if self._index is None:
            # No existing index — build one from scratch
            self.build_index(new_embeddings, new_fact_ids, new_facts)
            return len(new_fact_ids)

        assert new_embeddings.shape[1] == self.dim, (
            f"Embedding dim {new_embeddings.shape[1]} != expected {self.dim}"
        )

        embeddings = new_embeddings.astype("float32").copy()
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._fact_ids.extend(new_fact_ids)

        if new_facts:
            for fact in new_facts:
                self._facts_meta[fact["id"]] = fact

        logger.info(
            f"Appended {len(new_fact_ids)} vectors to index "
            f"(total: {self._index.ntotal})"
        )
        return len(new_fact_ids)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, output_dir: str = "data/embeddings") -> None:
        """Save the FAISS index and metadata to disk."""
        import faiss

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(out / "faiss_index.bin"))
        with open(out / "fact_ids.json", "w") as f:
            json.dump(self._fact_ids, f)

        if self._facts_meta:
            with open(out / "facts_meta.json", "w", encoding="utf-8") as f:
                json.dump(self._facts_meta, f, ensure_ascii=False, default=str)

        logger.info(f"Saved FAISS index ({self._index.ntotal} vectors) to {out}")

    def load(self, input_dir: str = "data/embeddings") -> None:
        """Load FAISS index and metadata from disk."""
        import faiss

        inp = Path(input_dir)
        self._index = faiss.read_index(str(inp / "faiss_index.bin"))
        self.dim = self._index.d

        with open(inp / "fact_ids.json") as f:
            self._fact_ids = json.load(f)

        meta_path = inp / "facts_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self._facts_meta = json.load(f)

        logger.info(
            f"Loaded FAISS index: {self._index.ntotal} vectors, dim={self.dim}"
        )

    # ── Info ──────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
