"""
Module 4 — Time-Aware Retriever (main orchestrator).
Combines query encoding, vector search, temporal filtering,
WRS scoring, and context assembly into a single retrieval pipeline.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.deprecation.decay import DecayFunction
from src.retriever.temporal_filter import TemporalFilter
from src.retriever.wrs import WRSScorer
from src.retriever.context_assembler import ContextAssembler

logger = logging.getLogger(__name__)


class TimeAwareRetriever:
    """
    End-to-end time-aware retrieval pipeline.

    1. Encode query → vector
    2. FAISS search → top-N candidates
    3. Calculate FVS for each candidate
    4. Temporal filter → remove invalid facts
    5. WRS ranking → top-k
    6. Assemble context string
    """

    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        default_alpha: float = 0.5,
        default_top_k: int = 5,
        candidate_pool: int = 50,
        deprecation_threshold: float = 0.3,
        default_lambda: float = 0.01,
    ):
        self.embeddings_dir = embeddings_dir
        self.default_alpha = default_alpha
        self.default_top_k = default_top_k
        self.candidate_pool = candidate_pool

        self.decay = DecayFunction(default_lambda=default_lambda)
        self.temporal_filter = TemporalFilter(deprecation_threshold)
        self.wrs_scorer = WRSScorer(alpha=default_alpha)
        self.context_assembler = ContextAssembler()

        self._query_encoder = None
        self._vector_search = None

    def _ensure_loaded(self):
        """Lazy-load the encoder and FAISS index."""
        if self._vector_search is None:
            from src.retriever.query_encoder import QueryEncoder
            from src.retriever.vector_search import VectorSearch

            self._query_encoder = QueryEncoder()
            self._vector_search = VectorSearch()
            self._vector_search.load(self.embeddings_dir)
            logger.info(
                f"Retriever loaded: {self._vector_search.size} vectors"
            )

    def retrieve(
        self,
        query: str,
        query_time: Optional[datetime] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Full retrieval pipeline.

        Args:
            query: Natural-language query
            query_time: Point-in-time for temporal filtering
            top_k: Number of results to return
            alpha: Semantic vs temporal weight

        Returns:
            (context_string, list_of_ranked_facts)
        """
        self._ensure_loaded()

        if query_time is None:
            query_time = datetime.now(timezone.utc)
        if top_k is None:
            top_k = self.default_top_k
        if alpha is not None:
            self.wrs_scorer.alpha = alpha

        # 1. Encode query
        query_vec = self._query_encoder.encode(query)

        # 2. Vector search
        candidates = self._vector_search.search(query_vec, top_n=self.candidate_pool)

        # 3. Calculate FVS
        self.decay.score_facts(candidates, query_time)

        # 4. Temporal filter
        valid = self.temporal_filter.filter(candidates, query_time)

        # 5. WRS ranking
        ranked = self.wrs_scorer.rank(valid, top_k=top_k)

        # 6. Assemble context
        context = self.context_assembler.format(ranked, query_time)

        logger.info(
            f"Retrieved {len(ranked)} facts from {len(candidates)} candidates "
            f"for query: {query[:50]}..."
        )
        return context, ranked
