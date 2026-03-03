"""
Module 7 — Query Orchestrator for T-RAG.
End-to-end workflow: retrieve → generate → validate → respond.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.retriever.retriever import TimeAwareRetriever
from src.generator.prompt_builder import PromptBuilder
from src.generator.llm_client import LLMClient
from src.generator.post_processor import PostProcessor
from src.validator.consistency import ConsistencyValidator
from src.validator.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """
    Orchestrates the complete T-RAG query workflow.

    Pipeline: Retrieve → Generate → Validate → Respond
    """

    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        llm_provider: str = "local",
        llm_model: Optional[str] = None,
        default_alpha: float = 0.5,
        default_top_k: int = 5,
    ):
        self.retriever = TimeAwareRetriever(
            embeddings_dir=embeddings_dir,
            default_alpha=default_alpha,
            default_top_k=default_top_k,
        )
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient(
            provider=llm_provider,
            model=llm_model,
        )
        self.post_processor = PostProcessor()
        self.validator = ConsistencyValidator()
        self.confidence_scorer = ConfidenceScorer()

    async def process_query(
        self,
        query: str,
        query_time: Optional[datetime] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process a user query through the full T-RAG pipeline.

        Returns:
            Dict with keys: answer, confidence, sources, latency_ms,
            validation, metadata
        """
        t0 = time.time()

        if query_time is None:
            query_time = datetime.now(timezone.utc)

        # 1. Retrieve contexts
        context_str, source_facts = self.retriever.retrieve(
            query=query,
            query_time=query_time,
            top_k=top_k,
            alpha=alpha,
        )

        # 2. Generate answer
        messages = self.prompt_builder.build_messages(
            query, context_str, query_time
        )
        raw_answer = self.llm_client.generate(messages)
        answer = self.post_processor.process(raw_answer)

        # 3. Validate
        validation = self.validator.validate(answer, query_time, source_facts)
        confidence = self.confidence_scorer.score(validation, source_facts)

        latency_ms = int((time.time() - t0) * 1000)

        result = {
            "answer": answer,
            "confidence": confidence["confidence"],
            "confidence_rating": confidence["rating"],
            "sources": [
                {
                    "fact_id": f.get("fact_id", f.get("id", "")),
                    "text": f.get("text", ""),
                    "start_time": f.get("start_time"),
                    "end_time": f.get("end_time"),
                    "fvs": f.get("fvs", 0),
                    "wrs": f.get("wrs", 0),
                    "source": f.get("source", ""),
                }
                for f in source_facts
            ],
            "latency_ms": latency_ms,
            "validation": validation,
            "confidence_breakdown": confidence["breakdown"],
            "metadata": {
                "query": query,
                "query_time": query_time.isoformat(),
                "top_k": top_k,
                "alpha": alpha,
                "llm_provider": self.llm_client.provider,
                "candidates_found": len(source_facts),
            },
        }

        logger.info(
            f"Query processed in {latency_ms}ms — "
            f"confidence={confidence['confidence']:.2f} "
            f"({confidence['rating']})"
        )
        return result
