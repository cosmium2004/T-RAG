# ⏳ T-RAG — Time-Aware Retrieval-Augmented Generation

> Enhancing LLMs with temporally-aware knowledge retrieval for factually accurate, time-sensitive responses.

T-RAG is a Retrieval-Augmented Generation system built for **temporal reasoning**. Unlike standard RAG, which treats all knowledge as equally valid, T-RAG understands that facts have lifespans — diplomatic relations shift, leaders change, and yesterday's truth can be today's misinformation.

It uses a **Temporal Knowledge Graph** (Neo4j), **FAISS vector search**, and a **Fact Validity Score (FVS)** to ensure LLMs generate answers grounded in temporally-correct information.

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

### 3. Run the Data Pipeline

```bash
# Fetch ICEWS18 dataset, parse, extract, deduplicate
python scripts/preprocess_data.py --dataset icews18 --limit 2000

# For the full dataset (~373K rows, takes ~30s):
python scripts/preprocess_data.py --dataset icews18
```

### 4. Build the Knowledge Graph

Make sure Neo4j Desktop is running with a database started, then:

```bash
python scripts/build_tkg.py --clear
```

### 5. Generate Embeddings

```bash
# This downloads the SBERT model (~438 MB) on first run
python scripts/generate_embeddings.py
```

> ⏱️ This takes ~30 minutes on CPU for 30K facts. For faster results, use `--facts data/processed/facts.json` (1.8K facts, ~2 min).

### 6. Launch the Demo

```bash
# Interactive Streamlit UI
streamlit run app.py
```

Or start the REST API:

```bash
python -m uvicorn src.api.main:app --reload
# API docs at http://localhost:8000/docs
```

### 7. Run Tests

```bash
python -m pytest tests/ -v
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
