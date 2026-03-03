# ⏳ T-RAG — Time-Aware Retrieval-Augmented Generation

> Enhancing LLMs with temporally-aware knowledge retrieval for factually accurate, time-sensitive responses.

T-RAG is a Retrieval-Augmented Generation system built for **temporal reasoning**. Unlike standard RAG, which treats all knowledge as equally valid, T-RAG understands that facts have lifespans — diplomatic relations shift, leaders change, and yesterday's truth can be today's misinformation.

It uses a **Temporal Knowledge Graph** (Neo4j), **FAISS vector search**, and a **Fact Validity Score (FVS)** to ensure LLMs generate answers grounded in temporally-correct information.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
- [How It Works](#-how-it-works)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [References](#-references)

---

## 🎯 Key Features

- **Temporal Knowledge Graph** — 21K+ entities and 219K+ relationships stored in Neo4j with temporal metadata (start/end times, verification dates)
- **Fact Validity Scoring (FVS)** — Exponential decay model (`FVS = e^(-λΔt)`) detects deprecated facts automatically
- **Weighted Relevance Score (WRS)** — Balances semantic similarity with temporal freshness: `WRS = α × Sim(q,d) + (1-α) × FVS`
- **Multi-Provider LLM Support** — OpenAI, Anthropic, or local template-based fallback (no API key required for testing)
- **Interactive Streamlit Demo** — Dark-themed UI with real-time pipeline visualization, confidence metrics, and graph exploration
- **FastAPI REST API** — Production-ready endpoints (`/query`, `/health`, `/stats`) with Pydantic validation

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│ Query Encoder│──▶│  FAISS Search │───▶│  50 Candidates   │
│ (SBERT 768d) │    │  (Cosine Sim) │    │                  │
└──────────────┘    └───────────────┘    └────────┬─────────┘
                                                  │
                                                  ▼
                                         ┌────────────────┐
                                         │  FVS Scoring   │
                                         │  exp(-λ × Δt)  │
                                         └────────┬───────┘
                                                  │
                                                  ▼
                                         ┌────────────────┐
                                         │Temporal Filter │
                                         │(start/end/FVS) │
                                         └────────┬───────┘
                                                  │
                                                  ▼
                    ┌──────────────┐      ┌────────────────┐
                    │  LLM Client  │◀─────│ WRS Ranking    │
                    │ (GPT/Claude/ │      │ α·Sim+(1-α)·FVS│
                    │   Local)     │      └────────────────┘
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐      ┌────────────────┐
                    │ Post-Process │────▶│  Validation    │
                    │              │      │ (Consistency + │
                    │              │      │  Confidence)   │
                    └──────────────┘      └────────┬───────┘
                                                   │
                                                   ▼
                                            Final Response
                                         (Answer + Sources +
                                          Confidence Score)
```

---

## 🛠️ Tech Stack

| Component             | Technology                                        |
|-----------------------|---------------------------------------------------|
| **Language**          | Python 3.11+                                      |
| **Knowledge Graph**   | Neo4j (Desktop or Server)                         |
| **Vector Search**     | FAISS (CPU)                                       |
| **Embeddings**        | Sentence-BERT (`all-mpnet-base-v2`, 768d)         |
| **LLM**               | OpenAI GPT-4 / Anthropic Claude / Local fallback  |
| **API**               | FastAPI + Pydantic + Uvicorn                      |
| **Demo UI**           | Streamlit                                         |
| **Database**          | SQLite (update tracking)                          |
| **Dataset**           | ICEWS18 (Integrated Crisis Early Warning System)  |

---

## 📁 Project Structure

```
T-RAG/
├── src/
│   ├── data_pipeline/       # Data ingestion & preprocessing
│   │   ├── fetcher.py           # ICEWS download + ID resolution
│   │   ├── timestamp_parser.py  # ISO 8601 normalization
│   │   ├── entity_extractor.py  # Quadruple extraction
│   │   ├── duplicate_resolver.py# Deduplication
│   │   └── embedder.py          # Sentence-BERT embeddings
│   ├── tkg/                 # Temporal Knowledge Graph
│   │   ├── neo4j_client.py      # Neo4j driver wrapper
│   │   ├── schema.py            # Graph constraints & indexes
│   │   └── bulk_importer.py     # Batch import to Neo4j
│   ├── retriever/           # Time-aware retrieval
│   │   ├── query_encoder.py     # Query → vector
│   │   ├── vector_search.py     # FAISS index
│   │   ├── temporal_filter.py   # Time validity checks
│   │   ├── wrs.py               # Weighted Relevance Score
│   │   ├── context_assembler.py # Prompt context builder
│   │   └── retriever.py         # Full pipeline orchestrator
│   ├── deprecation/         # Fact validity detection
│   │   ├── decay.py             # FVS exponential decay
│   │   ├── classifier.py        # Valid/deprecated classifier
│   │   └── update_tracker.py    # SQLite verification log
│   ├── generator/           # LLM response generation
│   │   ├── prompt_builder.py    # Time-aware prompts
│   │   ├── llm_client.py       # Multi-provider LLM client
│   │   └── post_processor.py    # Response cleanup
│   ├── validator/           # Response validation
│   │   ├── consistency.py       # Temporal consistency check
│   │   └── confidence.py        # Composite confidence scorer
│   ├── api/                 # REST API
│   │   ├── main.py              # FastAPI application
│   │   └── orchestrator.py      # Query orchestration
│   └── evaluation/          # Metrics
│       └── metrics.py           # MRR, TA, Hits@K
├── scripts/
│   ├── preprocess_data.py       # Data pipeline CLI
│   ├── build_tkg.py             # TKG builder CLI
│   └── generate_embeddings.py   # Embedding generator CLI
├── tests/                   # 63 unit tests
├── data/
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Processed facts (JSON)
│   ├── cache/                   # Downloaded ICEWS files
│   └── embeddings/              # FAISS index + vectors
├── config/
│   ├── config.yaml              # System configuration
│   └── neo4j_schema.cypher      # Neo4j schema reference
├── app.py                   # Streamlit demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/)
- **Neo4j Desktop** — [Download](https://neo4j.com/download/) (create a local database and start it)
- **Git** (optional)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-repo/T-RAG.git
cd T-RAG

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers faiss-cpu torch neo4j
pip install fastapi uvicorn streamlit
```

### 2. Configure Environment

```bash
# Copy example and edit with your credentials
cp .env.example .env
```

Edit `.env` with your Neo4j password and (optionally) LLM API keys:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Optional — leave blank to use the local fallback LLM
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

---

## 🗺️ Step-by-Step Execution Guide

Follow these steps **in order** to build the full T-RAG system from scratch. Each step lists the script you run and which source modules it uses under the hood.

### Step 1 — Data Pipeline: Fetch, Parse & Deduplicate

```bash
python scripts/preprocess_data.py --dataset icews18
```

| Order | Module invoked                                                      | What it does                                                                                                                                      |
|:-----:|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | [`fetcher.py`](src/data_pipeline/fetcher.py)                        | Downloads ICEWS18 from RE-Net GitHub, fetches `entity2id.txt` and `relation2id.txt`, resolves numeric IDs to human-readable entity/relation names |
| 2     | [`timestamp_parser.py`](src/data_pipeline/timestamp_parser.py)      | Converts day-index timestamps to ISO 8601 datetimes using dataset epoch offsets                                                                   |
| 3     | [`entity_extractor.py`](src/data_pipeline/entity_extractor.py)      | Builds `(head, relation, tail, time)` quadruples, normalizes names, generates embedding-ready text                                                |
| 4     | [`duplicate_resolver.py`](src/data_pipeline/duplicate_resolver.py)  | Groups by canonical key, merges overlapping time ranges, averages confidence scores                                                               |

**Output:** `data/processed/facts_full.json` — 219,576 clean, deduplicated facts

> 💡 Use `--limit 2000` for a quick test (produces ~1,800 facts in seconds).

---

### Step 2 — Build the Temporal Knowledge Graph in Neo4j

> **Prerequisite:** Open Neo4j Desktop → create a project → create & **start** a database.

```bash
python scripts/build_tkg.py --clear
```

| Order | Module invoked                                        | What it does                                                                                  |
|:-----:|-------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 1     | [`neo4j_client.py`](src/tkg/neo4j_client.py)          | Connects to Neo4j via Bolt (`bolt://localhost:7687`), verifies health                         |
| 2     | [`schema.py`](src/tkg/schema.py)                      | Applies uniqueness constraints, temporal indexes, and full-text search indexes                |
| 3     | [`bulk_importer.py`](src/tkg/bulk_importer.py)        | Batch-creates Entity nodes (`MERGE`) and `RELATES_TO` relationships with temporal properties  |

**Output:** Neo4j graph with **21,085 entity nodes** and **219,576 temporal relationships**

> Schema reference: [`config/neo4j_schema.cypher`](config/neo4j_schema.cypher) — can also be run directly in Neo4j Browser.

---

### Step 3 — Generate Embeddings & Build FAISS Index

```bash
python scripts/generate_embeddings.py
```

| Order | Module invoked                                       | What it does                                                                   |
|:-----:|------------------------------------------------------|--------------------------------------------------------------------------------|
| 1     | [`embedder.py`](src/data_pipeline/embedder.py)       | Loads Sentence-BERT (`all-mpnet-base-v2`, 768d), batch-encodes all fact texts  |
| 2     | [`vector_search.py`](src/retriever/vector_search.py) | Normalizes vectors, builds a FAISS `IndexFlatIP` (cosine similarity)           |

**Output:** `data/embeddings/` containing:
- `embeddings.npy` — raw vectors
- `faiss_index.bin` — searchable FAISS index
- `fact_ids.json` — ID → vector position mapping
- `facts_meta.json` — full fact metadata for result enrichment

> ⏱️ ~30 min on CPU for 30K facts. For a quick test: `--facts data/processed/facts.json` (~2 min for 1.8K facts).

---

### Step 4 — Launch the Application

With Steps 1–3 complete, the system is fully operational. Choose either interface:

#### Option A: Interactive Streamlit Demo

```bash
streamlit run app.py
```

The demo ([`app.py`](app.py)) loads these modules at runtime:

| Module                                                         | Role in the demo                                                   |
|----------------------------------------------------------------|--------------------------------------------------------------------|
| [`query_encoder.py`](src/retriever/query_encoder.py)           | Encodes user query → 768d vector                                   |
| [`vector_search.py`](src/retriever/vector_search.py)           | Searches FAISS index for top-50 candidates                         |
| [`decay.py`](src/deprecation/decay.py)                         | Computes FVS for each candidate                                    |
| [`temporal_filter.py`](src/retriever/temporal_filter.py)       | Removes facts outside the query time window                        |
| [`wrs.py`](src/retriever/wrs.py)                               | Ranks by `α×Sim + (1-α)×FVS`, returns top-k                        |
| [`context_assembler.py`](src/retriever/context_assembler.py)   | Formats ranked facts into numbered prompt context                  |
| [`prompt_builder.py`](src/generator/prompt_builder.py)         | Wraps context into a time-aware system+user prompt                 |
| [`llm_client.py`](src/generator/llm_client.py)                 | Calls OpenAI / Anthropic / local fallback                          |
| [`post_processor.py`](src/generator/post_processor.py)         | Cleans LLM response (removes preambles, fixes whitespace)          |
| [`consistency.py`](src/validator/consistency.py)               | Checks for future-date refs and unsupported claims                 |
| [`confidence.py`](src/validator/confidence.py)                 | Computes composite confidence (consistency + freshness + coverage) |
| [`neo4j_client.py`](src/tkg/neo4j_client.py)                   | Powers the "Explore" and "Graph" tabs                              |

#### Option B: FastAPI REST API

```bash
python -m uvicorn src.api.main:app --reload
# Swagger docs → http://localhost:8000/docs
```

The API ([`src/api/main.py`](src/api/main.py)) uses [`orchestrator.py`](src/api/orchestrator.py) which chains all the above modules into a single async `process_query()` call.

---

### Step 5 — Run Tests

```bash
# Run all 63 tests
python -m pytest tests/ -v
```

| Test file                                               | Modules tested                                                                                                                                            | Count |
|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|
| [`test_data_pipeline.py`](tests/test_data_pipeline.py)  | fetcher, timestamp_parser, entity_extractor, duplicate_resolver                                                                                           | 22    |
| [`test_tkg.py`](tests/test_tkg.py)                      | neo4j_client, schema, bulk_importer *(requires running Neo4j)*                                                                                            | 10    |
| [`test_modules.py`](tests/test_modules.py)              | decay, classifier, update_tracker, temporal_filter, wrs, context_assembler, prompt_builder, llm_client, post_processor, consistency, confidence, metrics  | 31    |

---

### Execution Summary

```
Step 1                 Step 2                    Step 3                     Step 4
──────────────         ──────────────           ──────────────              ──────────────
preprocess_data.py ──▶ build_tkg.py        ──▶ generate_embeddings.py ──▶ app.py
  │                      │                        |                          │
  ├─ fetcher.py          ├─ neo4j_client.py       │                          ├─ query_encoder
  ├─ timestamp_parser    ├─ schema.py             ├─ embedder.py             ├─ vector_search
  ├─ entity_extractor    └─ bulk_importer.py      └─ vector_search           ├─ decay / filter
  └─ duplicate_resolver                                                      ├─ wrs / context
                                                                             ├─ prompt_builder
  Output:                Output:                   Output:                   ├─ llm_client
  facts_full.json        Neo4j graph               FAISS index               └─ validator
  (219K facts)           (21K nodes, 219K rels)    (768d vectors)
                            
```

---

## 📊 How It Works

### The Problem
Standard RAG systems retrieve information without considering **when** it was true. This causes LLMs to present outdated or contradictory facts as current truth.

### The Solution
T-RAG introduces three temporal mechanisms:

1. **Fact Validity Score (FVS)** — Each fact has a freshness score that decays exponentially over time:
   ```
   FVS = exp(-λ × days_since_verification)
   ```
   Facts with low FVS are flagged as potentially deprecated.

2. **Weighted Relevance Score (WRS)** — Retrieved facts are ranked by both semantic relevance AND temporal freshness:
   ```
   WRS = α × Similarity + (1-α) × FVS
   ```
   The `α` parameter (0–1) lets users control the trade-off.

3. **Temporal Knowledge Graph** — Facts are stored in Neo4j with explicit time windows (`start_time`, `end_time`), enabling queries like *"What was true on February 1, 2018?"*

### Example Query

```
Query: "What diplomatic actions involved North Korea in early 2018?"
Time:  2018-03-01

Results (87% confidence, HIGH):
  1. South Korea investigate North Korea           [FVS: 0.90, WRS: 0.773]
  2. NK express intent for diplomatic cooperation   [FVS: 0.90, WRS: 0.763]
  3. Russia engage in diplomatic cooperation NK     [FVS: 0.90, WRS: 0.760]
```

---

## 🔧 API Reference

### `POST /query`
Submit a temporal query.

```json
{
  "query": "What happened with North Korea?",
  "query_time": "2018-06-01T00:00:00Z",
  "top_k": 5,
  "alpha": 0.5
}
```

**Response:**
```json
{
  "answer": "In early 2018, several diplomatic actions...",
  "confidence": 0.87,
  "confidence_rating": "HIGH",
  "sources": [...],
  "latency_ms": 1200,
  "validation": { "consistency_score": 0.75, "is_valid": true }
}
```

### `GET /health`
System health check (Neo4j + LLM status).

### `GET /stats`
Graph statistics (entity count, relationship count).

---

## 🧪 Testing

```bash
# Run all 63 tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_data_pipeline.py -v  # Data pipeline (22 tests)
python -m pytest tests/test_tkg.py -v            # TKG / Neo4j (10 tests)
python -m pytest tests/test_modules.py -v        # Core modules (31 tests)
```

---

## 📚 References

This project draws from research in temporal knowledge graph reasoning and retrieval-augmented generation. See the `References/` directory for the foundational papers.

---

## 📄 License

This project is developed for academic purposes.
