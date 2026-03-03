"""
Module 7 — FastAPI Application for T-RAG.
REST API with Swagger UI at /docs.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from src.api.orchestrator import QueryOrchestrator
from src.tkg.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="T-RAG API",
    description="Time-Aware Retrieval-Augmented Generation for LLMs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (lazy init) ───────────────────────────────────────────

_orchestrator: Optional[QueryOrchestrator] = None
_neo4j: Optional[Neo4jClient] = None


def get_orchestrator() -> QueryOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = QueryOrchestrator(
            embeddings_dir="data/embeddings",
            llm_provider=os.getenv("LLM_PROVIDER", "local"),
            llm_model=os.getenv("LLM_MODEL"),
        )
    return _orchestrator


def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j


# ── Request / Response models ────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    query_time: Optional[datetime] = Field(
        None, description="Point-in-time for temporal context (ISO 8601)"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of facts to retrieve")
    alpha: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Semantic vs temporal weight (0=temporal, 1=semantic)",
    )


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    confidence_rating: str
    sources: list
    latency_ms: int
    validation: dict
    confidence_breakdown: dict
    metadata: dict


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Submit a temporal query to the T-RAG system."""
    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.process_query(
            query=request.query,
            query_time=request.query_time or datetime.now(timezone.utc),
            top_k=request.top_k,
            alpha=request.alpha,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """System health check."""
    neo4j_health = get_neo4j().health_check()
    llm_health = get_orchestrator().llm_client.health_check()

    return {
        "status": "healthy" if neo4j_health["status"] == "healthy" else "degraded",
        "components": {
            "neo4j": neo4j_health,
            "llm": llm_health,
        },
    }


@app.get("/stats")
async def graph_stats():
    """Return graph statistics."""
    client = get_neo4j()
    nodes = client.query("MATCH (n:Entity) RETURN count(n) AS count")
    rels = client.query("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS count")
    return {
        "entity_nodes": nodes[0]["count"] if nodes else 0,
        "relationships": rels[0]["count"] if rels else 0,
    }
