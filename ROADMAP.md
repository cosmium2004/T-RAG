# T-RAG Complete Implementation Roadmap
## Time-Aware Retrieval-Augmented Generation for LLMs

**Version:** 1.0.0  
**Target Duration:** 8-10 weeks (solo developer)  
**Difficulty:** Intermediate to Advanced

---

## Table of Contents

1.  [Executive Summary](#executive-summary)
2.  [System Overview](#system-overview)
3.  [Architecture](#architecture)
4.  [Module Breakdown](#module-breakdown)
5.  [Development Phases](#development-phases)
6.  [Week-by-Week Plan](#week-by-week-plan)
7.  [Implementation Details](#implementation-details)
8.  [Testing Strategy](#testing-strategy)
9.  [Deployment Guide](#deployment-guide)
10. [Troubleshooting](#troubleshooting)
11. [Success Metrics](#success-metrics)

---

## Executive Summary

### The Problem

Large Language Models (LLMs) suffer from **temporal drift** - they provide outdated information because their knowledge is frozen at training time. A model trained in 2023 might incorrectly say "Donald Trump is a former president" in 2026, when he's actually the current president.

### The Solution: T-RAG

T-RAG enhances LLMs with temporal awareness through:

1. **Temporal Knowledge Graph (TKG)**: Stores facts with validity timestamps
2. **Adaptive Deprecation Detection (ADD)**: Identifies outdated facts using freshness scores
3. **Time-Aware Retrieval**: Combines semantic similarity + temporal validity
4. **Validated Generation**: Produces accurate, citation-backed responses

### Target Performance

| Metric                     | Target          | Industry Baseline |
|----------------------------|-----------------|-------------------|
| Mean Reciprocal Rank (MRR) | ≥ 0.47          | 0.35              |
| Temporal Accuracy          | ≥ 90%           | 65%               |
| Query Latency              | ≤ 3 seconds     | 5-8 seconds       |
| Hits@10                    | ≥ 75%           | 60%               |

---

## System Overview

### Core Components
```
┌─────────────────────────────────────────────────────────────┐
│                      T-RAG SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │   MODULE 1   │  │   MODULE 2   │  │    MODULE 3     │    │
│  │   Data       │→ │   TKG        │→ │  Deprecation    │    │
│  │   Pipeline   │  │   Builder    │  │  Detection      │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
│                                              ↓              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │   MODULE 6   │← │   MODULE 5   │← │    MODULE 4     │    │
│  │   Response   │  │   LLM        │  │  Time-Aware     │    │
│  │   Validator  │  │   Generator  │  │  Retriever      │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           MODULE 7: API & Orchestration             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
User Query
    ↓
[Module 7] API receives query + timestamp
    ↓
[Module 4.1] Encode query to vector
    ↓
[Module 4.2] Search FAISS for top-50 similar facts
    ↓
[Module 3.1] Calculate freshness score (FVS) for each fact
    ↓
[Module 4.3] Filter: keep only temporally valid facts
    ↓
[Module 4.4] Rank by Weighted Relevance Score (WRS)
    ↓
[Module 4.5] Assemble top-k facts into context
    ↓
[Module 5.1-5.3] Generate answer with LLM
    ↓
[Module 6] Validate temporal consistency
    ↓
[Module 7] Return answer + sources + confidence
```

---

## Architecture

### Technology Stack

| Layer                | Technology             | Purpose                                   |
|----------------------|------------------------|-------------------------------------------|
| **Graph Database**   | Neo4j 5.13+            | Store temporal facts with relationships   |
| **Vector Store**     | FAISS                  | Semantic similarity search                |
| **Relational DB**    | PostgreSQL 15          | Metadata, logs, update tracking           |
| **Cache**            | Redis 7                | Query result caching                      |
| **API Framework**    | FastAPI                | REST API endpoints                        |
| **LLM Integration**  | OpenAI/Anthropic/Local | Answer generation                         |
| **Embeddings**       | Sentence-BERT          | Text vectorization                        |
| **NLP**              | spaCy                  | Entity extraction                         |
| **ML Framework**     | PyTorch                | Graph neural networks (optional)          |

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB SSD
- GPU: Not required (CPU-only mode available)

**Recommended:**
- CPU: 8+ cores
- RAM: 32GB
- Storage: 100GB NVMe SSD
- GPU: NVIDIA GTX 1050Ti or better (for embeddings)

---

## Module Breakdown

### Module 1: Data Pipeline

**Purpose:** Transform raw datasets into clean, timestamped facts.

**Units:**
- **1.1 Data Ingestion**: Download ICEWS18, Wikidata, GDELT, etc.
- **1.2 Timestamp Normalization**: Parse dates to ISO 8601
- **1.3 Entity Extraction**: Extract (head, relation, tail, time) quadruples
- **1.4 Duplicate Resolution**: Remove conflicting facts
- **1.5 Embedding Generation**: Create vector representations

**Key Outputs:**
```json
{
  "id": "fact_12345",
  "head": "Donald Trump",
  "relation": "holds_position",
  "tail": "President of USA",
  "start_time": "2025-01-20T00:00:00Z",
  "end_time": null,
  "last_verified": "2026-01-25T12:00:00Z",
  "source": "ICEWS18",
  "confidence": 0.95,
  "embedding": [0.123, -0.456, ...]
}
```

---

### Module 2: TKG Builder

**Purpose:** Construct queryable temporal knowledge graph.

**Units:**
- **2.1 Neo4j Connection**: Database client with pooling
- **2.2 Schema Design**: Node/edge types, constraints
- **2.3 Fact Ingestion**: Bulk insert with transactions
- **2.4 Temporal Indexing**: Optimize time-range queries
- **2.5 GNN Embeddings** (Optional): Graph neural network features

**Graph Schema:**
```cypher
// Nodes
(:Entity {name: String, type: String})
(:Event {id: String, timestamp: DateTime})

// Relationships
(:Entity)-[:RELATES_TO {type:       String,
                        start_time: DateTime,
                        end_time:   DateTime,
                        confidence: Float    }]->(:Entity)
```

---

### Module 3: Adaptive Deprecation Detection

**Purpose:** Identify and score facts by temporal freshness.

**Units:**
- **3.1 Half-Life Decay**: Calculate FVS = exp(-λ × Δt)
- **3.2 Decay Rate Learner**: ML model to predict λ
- **3.3 Deprecation Classifier**: Binary valid/deprecated
- **3.4 Update Tracker**: Log fact verifications
- **3.5 Refresh Scheduler**: Periodic recalculation

**FVS Formula:**
```
Fact Validity Score (FVS) = e^(-λ × days_since_verified)

Where:
- λ = decay rate (0.01 = slow, 0.1 = medium, 1.0 = fast)
- Δt = current_time - last_verified_time
- Higher FVS = fresher fact
```

**Example:**
```python
# Fact verified 100 days ago, λ = 0.01
FVS = exp(-0.01 × 100) = exp(-1) ≈ 0.368

# Threshold: 0.3
# Decision: 0.368 > 0.3 → VALID
```

---

### Module 4: Time-Aware Retriever

**Purpose:** Fetch semantically + temporally relevant facts.

**Units:**
- **4.1 Query Encoder**: Convert query to vector
- **4.2 Vector Search**: FAISS similarity search
- **4.3 Temporal Filter**: Remove invalid-at-query-time facts
- **4.4 WRS Scorer**: Combine semantic + temporal scores
- **4.5 Context Assembler**: Format for LLM prompt

**WRS Formula:**
```
Weighted Relevance Score (WRS) = α × Sim(q,d) + (1-α) × FVS

Where:
- Sim(q,d) = cosine similarity (query, document)
- α = semantic weight (default: 0.5)
- (1-α) = temporal weight
```

**Retrieval Workflow:**
1. Encode query → 768-dim vector
2. FAISS search → top-50 candidates
3. Calculate FVS for each candidate
4. Temporal filter → remove if start_time > query_time or end_time < query_time
5. Calculate WRS → rank by score
6. Select top-k (default k=5)
7. Format as context string

---

### Module 5: LLM Generator

**Purpose:** Generate natural language answers.

**Units:**
- **5.1 Prompt Builder**: Create time-aware prompts
- **5.2 LLM Client**: API calls with retry logic
- **5.3 Post-Processor**: Clean responses
- **5.4 Streaming Handler** (Optional): Real-time responses

**Prompt Template:**
```
You are a helpful assistant. Answer based ONLY on the provided context.
Always cite timestamps when stating facts.

Question: {user_query}

Context (valid as of {current_timestamp}):
1. [2025-01-20 to present] Donald Trump is President of USA (Source: ICEWS14)
2. [2021-01-20 to 2025-01-20] Joe Biden was President of USA (Source: Wikidata)

Instructions:
- Use only recent information (prioritize facts marked "to present")
- If conflicting facts exist, prefer the most recent
- State "I don't have current information" if context is insufficient

Answer:
```

---

### Module 6: Response Validator

**Purpose:** Verify temporal consistency and accuracy.

**Units:**
- **6.1 Consistency Checker**: Detect temporal contradictions
- **6.2 Citation Extractor**: Verify source timestamps
- **6.3 Confidence Scorer**: Calculate reliability
- **6.4 Explainability**: Generate reasoning trace

**Validation Checks:**
1. No future dates mentioned (beyond query time)
2. Tense consistency (past facts use past tense)
3. All claims backed by sources
4. No contradictory statements
5. Citations include timestamps

**Output:**
```json
{
  "consistency_score": 0.95,
  "contradictions": [],
  "citations_verified": true,
  "confidence": 0.92,
  "rejected_facts": [
    {
      "fact_id": "fact_67890",
      "reason": "Temporally invalid (ended 2025-01-20)",
      "fvs": 0.025
    }
  ]
}
```

---

### Module 7: API & Orchestration

**Purpose:** Expose T-RAG via REST API.

**Endpoints:**
```
POST   /query          # Main query endpoint
GET    /fact/{id}      # Retrieve specific fact
POST   /update         # Trigger fact refresh
GET    /health         # System health check
GET    /metrics        # Prometheus metrics
```

**Orchestration Workflow:**
```python
async def process_query(query, query_time, top_k, alpha):
    # 1. Check cache
    cache_key = hash(query + query_time)
    if cached := redis.get(cache_key):
        return cached
    
    # 2. Retrieve contexts (Module 4)
    contexts = retriever.get_contexts(query, query_time, top_k, alpha)
    
    # 3. Generate answer (Module 5)
    answer = generator.generate(query, contexts, query_time)
    
    # 4. Validate (Module 6)
    validation = validator.validate(answer, query_time, contexts)
    
    # 5. Assemble response
    response = {
        'answer': answer,
        'confidence': validation['confidence'],
        'sources': contexts,
        'latency_ms': elapsed_time
    }
    
    # 6. Cache and return
    redis.setex(cache_key, 3600, response)
    return response
```

---

## Development Phases

### Phase 1: Foundation (Weeks 1-2)
- ✅ Data ingestion from ICEWS18
- ✅ Timestamp normalization
- ✅ Entity/relation extraction
- ✅ Duplicate resolution
- ✅ Neo4j setup and schema

**Checkpoint:** 1000+ clean facts ready for graph loading

---

### Phase 2: Knowledge Graph (Weeks 3-4)
- ✅ Bulk import to Neo4j
- ✅ Temporal indexes
- ✅ Embedding generation with Sentence-BERT
- ✅ FAISS index creation

**Checkpoint:** Queryable TKG with vector search

---

### Phase 3: Deprecation Detection (Week 5)
- ✅ FVS calculation
- ✅ Decay rate learning (or fixed λ)
- ✅ Deprecation classification
- ✅ Update tracking
- ✅ Scheduled refresh

**Checkpoint:** System filters outdated facts

---

### Phase 4: Retrieval (Week 6)
- ✅ Query encoding
- ✅ Vector similarity search
- ✅ Temporal filtering
- ✅ WRS ranking
- ✅ Context assembly

**Checkpoint:** Retrieves temporally valid, relevant facts

---

### Phase 5: Generation & Validation (Week 7)
- ✅ Prompt engineering
- ✅ LLM API integration
- ✅ Response post-processing
- ✅ Consistency validation
- ✅ Confidence scoring

**Checkpoint:** Generates validated answers

---

### Phase 6: API & Integration (Week 8)
- ✅ FastAPI implementation
- ✅ Orchestration engine
- ✅ Caching layer
- ✅ Logging and metrics
- ✅ Error handling

**Checkpoint:** Functional API

---

### Phase 7: Evaluation & Optimization (Weeks 9-10)
- ✅ Benchmark evaluation (MRR, TA, Hits@K)
- ✅ Hyperparameter tuning (α, k, λ, threshold)
- ✅ Performance optimization
- ✅ Documentation

**Final Checkpoint:** Production-ready system

---

## Week-by-Week Plan

### Week 1: Data Pipeline Foundation

**Goals:**
- Set up development environment
- Implement data ingestion (Module 1.1)
- Implement timestamp parsing (Module 1.2)
- Implement entity extraction (Module 1.3)
- Implement duplicate resolution (Module 1.4)
- Process 1000+ facts successfully

**Deliverables:**
```
✓ requirements.txt with dependencies
✓ src/data_pipeline/fetcher.py
✓ src/data_pipeline/timestamp_parser.py
✓ src/data_pipeline/entity_extractor.py
✓ src/data_pipeline/duplicate_resolver.py
✓ tests/ with >80% coverage
✓ data/processed/facts.json (1000+ facts)
✓ scripts/preprocess_data.py
```

**Success Criteria:**
- Can download ICEWS18 dataset
- >95% timestamps parsed correctly
- All facts have required fields
- 10-20% duplicate reduction
- All unit tests passing

**Time Breakdown:**
- Environment setup: 0.5 hours
- Module 1.1 (Fetcher): 2 hours
- Module 1.2 (Parser): 2 hours
- Module 1.3 (Extractor): 3 hours
- Module 1.4 (Resolver): 2 hours
- Integration & testing: 2 hours
- Documentation: 1 hour
- **Total: ~12 hours**

---

### Week 2: Neo4j & Graph Construction

**Goals:**
- Install and configure Neo4j
- Design graph schema
- Implement bulk importer (Module 2.3)
- Create temporal indexes (Module 2.4)
- Load all facts into graph

**Deliverables:**
```
✓ Neo4j running with authentication
✓ src/tkg/neo4j_client.py
✓ src/tkg/schema.py
✓ src/tkg/bulk_importer.py
✓ config/neo4j_schema.cypher
✓ scripts/build_tkg.py
```

**Success Criteria:**
- All facts loaded into Neo4j
- Can query facts by time range
- Temporal queries execute <100ms
- Graph has proper indexes

**Example Queries:**
```cypher
// Find current president
MATCH (p:Entity)-[r:RELATES_TO {type: 'holds_position'}]->(pos:Entity)
WHERE pos.name = 'President of USA'
  AND r.start_time <= datetime()
  AND (r.end_time IS NULL OR r.end_time >= datetime())
RETURN p.name

// Find facts valid in specific time range
MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
WHERE r.start_time <= datetime('2024-01-15')
  AND (r.end_time IS NULL OR r.end_time >= datetime('2024-01-15'))
RETURN e1, r, e2
LIMIT 10
```

---

### Week 3: Embeddings & Vector Search

**Goals:**
- Generate embeddings with Sentence-BERT (Module 1.5)
- Build FAISS index (Module 4.2)
- Implement query encoding (Module 4.1)
- Test vector similarity search

**Deliverables:**
```
✓ src/data_pipeline/embedder.py
✓ src/retriever/query_encoder.py
✓ src/retriever/vector_search.py
✓ data/embeddings/faiss_index.bin
✓ Embedding metadata (fact_id mapping)
```

**Success Criteria:**
- All facts have 768-dim embeddings
- FAISS index built successfully
- Vector search returns top-50 in <100ms
- Semantic similarity makes sense (manual check)

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Generate embeddings
model = SentenceTransformer('all-mpnet-base-v2')
texts = [fact['text'] for fact in facts]
embeddings = model.encode(texts, show_progress_bar=True)

# Build FAISS index
dim = 768
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype('float32'))
faiss.write_index(index, 'data/embeddings/faiss_index.bin')

# Search
query = "Who is the president?"
query_vector = model.encode([query])[0]
distances, indices = index.search(query_vector.reshape(1, -1), k=50)
```

---

### Week 4: Complete TKG Setup

**Goals:**
- Integrate embeddings with Neo4j
- Optimize indexes for performance
- Add more data sources (optional)
- Performance testing

**Deliverables:**
```
✓ Optimized Neo4j indexes
✓ Performance benchmarks
✓ Documentation for querying TKG
```

---

### Week 5: Deprecation Detection

**Goals:**
- Implement half-life decay calculator (Module 3.1)
- Implement deprecation classifier (Module 3.3)
- Create update tracker (Module 3.4)
- Set up refresh scheduler (Module 3.5)

**Deliverables:**
```
✓ src/deprecation/decay.py
✓ src/deprecation/classifier.py
✓ src/deprecation/update_tracker.py
✓ PostgreSQL update_log table
✓ Celery task for periodic refresh
```

**Implementation:**
```python
class DecayFunction:
    def calculate_fvs(self, last_verified, current_time, lambda_val=0.01):
        delta_days = (current_time - last_verified).days
        fvs = np.exp(-lambda_val * delta_days)
        return fvs

# Example
fact_date = datetime(2025, 1, 20)
query_date = datetime(2026, 1, 26)
fvs = decay.calculate_fvs(fact_date, query_date)
# fvs ≈ 0.99 (very fresh)

is_valid = fvs >= 0.3  # Threshold
```

**Success Criteria:**
- FVS calculated for all facts
- Deprecation threshold tuned (start with 0.3)
- Outdated facts correctly identified
- Refresh runs successfully

---

### Week 6: Time-Aware Retrieval

**Goals:**
- Implement temporal filtering (Module 4.3)
- Implement WRS scorer (Module 4.4)
- Implement context assembler (Module 4.5)
- End-to-end retrieval pipeline

**Deliverables:**
```
✓ src/retriever/temporal_filter.py
✓ src/retriever/wrs.py
✓ src/retriever/context_assembler.py
✓ src/retriever/retriever.py (main class)
```

**Complete Retrieval Pipeline:**
```python
class TimeAwareRetriever:
    def retrieve(self, query, query_time, top_k=5, alpha=0.5):
        # 1. Encode query
        query_vec = self.encoder.encode(query)
        
        # 2. Vector search
        candidates = self.vector_search.search(query_vec, top_n=50)
        
        # 3. Calculate FVS
        for cand in candidates:
            cand['fvs'] = self.decay.calculate_fvs(
                cand['last_verified'], 
                query_time
            )
        
        # 4. Temporal filter
        valid = [c for c in candidates 
                 if self._is_temporally_valid(c, query_time)
                 and c['fvs'] >= 0.3]
        
        # 5. Calculate WRS
        for cand in valid:
            cand['wrs'] = alpha * cand['similarity'] + (1-alpha) * cand['fvs']
        
        # 6. Rank and select top-k
        ranked = sorted(valid, key=lambda x: x['wrs'], reverse=True)
        top_contexts = ranked[:top_k]
        
        # 7. Format as context string
        context = self.assembler.format(top_contexts, query_time)
        return context, top_contexts
```

**Success Criteria:**
- Retrieves relevant + temporally valid facts
- WRS ranking improves over pure semantic search
- Top-k facts fit in LLM token limit (~2000 tokens)

---

### Week 7: Generation & Validation

**Goals:**
- Implement prompt builder (Module 5.1)
- Integrate LLM API (Module 5.2)
- Implement consistency checker (Module 6.1)
- Implement confidence scorer (Module 6.3)

**Deliverables:**
```
✓ src/generator/prompt_builder.py
✓ src/generator/llm_client.py
✓ src/generator/post_processor.py
✓ src/validator/consistency.py
✓ src/validator/confidence.py
```

**Prompt Engineering:**
```python
class PromptBuilder:
    def build_prompt(self, query, contexts, query_time):
        context_str = "\n".join([
            f"{i+1}. [Valid: {c['start_time']} to {c['end_time'] or 'present'}] "
            f"{c['text']} (Source: {c['source']}, Confidence: {c['confidence']:.0%})"
            for i, c in enumerate(contexts)
        ])
        
        prompt = f"""You are a helpful assistant. Answer based ONLY on the provided context.
Always cite timestamps when stating facts.

Question: {query}

Context (valid as of {query_time}):
{context_str}

Instructions:
- Use only recent information (prioritize facts marked "to present")
- If conflicting facts exist, prefer the most recent
- State "I don't have current information" if context is insufficient
- Always include source citations

Answer:"""
        
        return prompt
```

**Validation Logic:**
```python
class ConsistencyValidator:
    def validate(self, response, query_time, source_facts):
        # Extract temporal claims
        claims = self.extract_temporal_claims(response)
        
        # Check contradictions
        contradictions = []
        for claim, timestamp in claims:
            if timestamp > query_time:
                contradictions.append(f"Future date: {timestamp}")
            
            # Check against sources
            if not self._verify_against_sources(claim, source_facts):
                contradictions.append(f"Unsupported claim: {claim}")
        
        consistency_score = 1.0 - (len(contradictions) / max(len(claims), 1))
        
        return {
            'consistency_score': consistency_score,
            'contradictions': contradictions,
            'is_valid': consistency_score > 0.8
        }
```

---

### Week 8: API & Orchestration

**Goals:**
- Build FastAPI application (Module 7.1)
- Implement orchestration engine (Module 7.2)
- Add caching with Redis (Module 7.3)
- Implement logging and metrics (Module 7.4)

**Deliverables:**
```
✓ src/api/main.py
✓ src/api/orchestrator.py
✓ src/api/metrics.py
✓ Docker setup (Dockerfile, docker-compose.yml)
```

**API Implementation:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="T-RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    query_time: datetime = None
    top_k: int = 5
    alpha: float = 0.5

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list
    latency_ms: int

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        result = await orchestrator.process_query(
            query=request.query,
            query_time=request.query_time or datetime.now(),
            top_k=request.top_k,
            alpha=request.alpha
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "neo4j": await check_neo4j(),
            "redis": await check_redis(),
            "faiss": check_faiss_loaded()
        }
    }
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  trag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - neo4j
      - postgres
      - redis
  
  neo4j:
    image: neo4j:5.13
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=trag
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  neo4j_data:
  postgres_data:
  redis_data:
```

---

### Week 9-10: Evaluation & Optimization

**Goals:**
- Implement evaluation metrics (MRR, TA, Hits@K)
- Run benchmark on ICEWS14 test set
- Hyperparameter tuning (grid search or Bayesian)
- Performance optimization
- Production deployment

**Evaluation Script:**
```python
# scripts/evaluate.py

async def run_evaluation():
    # Load test set
    with open('data/icews18_test.json') as f:
        test_data = json.load(f)
    
    predictions = []
    ground_truths = []
    
    for example in test_data:
        result = await orchestrator.process_query(
            query=example['query'],
            query_time=example['timestamp']
        )
        predictions.append(result['answer'])
        ground_truths.append(example['ground_truth'])
    
    # Calculate metrics
    mrr = calculate_mrr(predictions, ground_truths)
    ta = calculate_temporal_accuracy(predictions, ground_truths)
    hits_10 = calculate_hits_at_k(predictions, ground_truths, k=10)
    
    report = {
        'MRR': mrr,
        'Temporal_Accuracy': ta,
        'Hits@10': hits_10,
        'target_met': mrr >= 0.47 and ta >= 0.90
    }
    
    print(json.dumps(report, indent=2))
    return report
```

**Hyperparameter Tuning:**
```python
# scripts/tune_hyperparameters.py

from itertools import product

param_grid = {
    'alpha': [0.3, 0.5, 0.7],
    'top_k': [3, 5, 10],
    'deprecation_threshold': [0.2, 0.3, 0.4],
    'decay_lambda': [0.005, 0.01, 0.05]
}

best_score = 0
best_params = None

for params in product(*param_grid.values()):
    alpha, top_k, threshold, lambda_val = params
    
    # Run evaluation with these params
    score = evaluate_with_params(alpha, top_k, threshold, lambda_val)
    
    if score > best_score:
        best_score = score
        best_params = {
            'alpha': alpha,
            'top_k': top_k,
            'threshold': threshold,
            'lambda': lambda_val
        }

print(f"Best params: {best_params}")
print(f"Best score: {best_score}")
```

---

## Implementation Details

### Project Structure
```
t-rag/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app
│   │   ├── orchestrator.py         # Query workflow
│   │   └── metrics.py              # Prometheus metrics
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── fetcher.py              # Data ingestion
│   │   ├── timestamp_parser.py     # Date normalization
│   │   ├── entity_extractor.py     # NER + relations
│   │   ├── duplicate_resolver.py   # Deduplication
│   │   └── embedder.py             # Vector embeddings
│   ├── tkg/
│   │   ├── __init__.py
│   │   ├── neo4j_client.py         # Database client
│   │   ├── schema.py               # Graph schema
│   │   ├── bulk_importer.py        # Batch insertion
│   │   └── gnn_embedder.py         # Optional GNN
│   ├── deprecation/
│   │   ├── __init__.py
│   │   ├── decay.py                # FVS calculation
│   │   ├── learner.py              # Decay rate ML
│   │   ├── classifier.py           # Deprecation filter
│   │   └── update_tracker.py       # Verification log
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── query_encoder.py        # Query embedding
│   │   ├── vector_search.py        # FAISS search
│   │   ├── temporal_filter.py      # Time filtering
│   │   ├── wrs.py                  # WRS scoring
│   │   └── context_assembler.py    # Prompt formatting
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py       # Templates
│   │   ├── llm_client.py           # API calls
│   │   └── post_processor.py       # Response cleaning
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── consistency.py          # Temporal checks
│   │   ├── citation_extractor.py   # Source verification
│   │   ├── confidence.py           # Confidence scoring
│   │   └── explainability.py       # Report generation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # MRR, TA, Hits@K
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Structured logging
│       └── config.py               # Config management
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_tkg.py
│   ├── test_deprecation.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_validator.py
│   ├── test_integration.py
│   └── test_performance.py
├── scripts/
│   ├── setup_databases.sh
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   ├── build_tkg.py
│   ├── train_deprecation.py
│   ├── evaluate.py
│   ├── tune_hyperparameters.py
│   └── deploy.sh
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── cache/
├── config/
│   ├── config.yaml
│   ├── logging.yaml
│   └── neo4j_schema.cypher
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

### Key Files

#### requirements.txt
```txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0

# Database Clients
neo4j==5.14.0
psycopg2-binary==2.9.9
redis==5.0.1
sqlalchemy==2.0.23

# Graph & Embeddings
torch==2.1.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
transformers==4.35.2

# NLP & Data Processing
spacy==3.7.2
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2

# LLM APIs
openai==1.3.7
anthropic==0.7.1

# Utilities
python-dateutil==2.8.2
tenacity==8.2.3
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Hyperparameter Tuning
optuna==3.4.0

# Deployment
gunicorn==21.2.0
celery==5.3.4
```

#### config.yaml
```yaml
# System Configuration
system:
  name: "T-RAG"
  version: "1.0.0"
  environment: "development"

# API Settings
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30

# Database Configurations
databases:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "${NEO4J_PASSWORD}"
  
  postgres:
    uri: "postgresql://postgres:${POSTGRES_PASSWORD}@localhost:5432/trag"
  
  redis:
    url: "redis://localhost:6379/0"

# Vector Search
vector_search:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  embedding_dim: 768
  index_type: "IVFFlat"

# Deprecation Detection
deprecation:
  default_lambda: 0.01
  deprecation_threshold: 0.3
  update_frequency: "daily"

# Retrieval
retrieval:
  default_top_k: 5
  max_top_k: 20
  candidate_pool_size: 50
  default_alpha: 0.5

# LLM Generation
llm:
  provider: "openai"
  model: "gpt-4"
  max_tokens: 500
  temperature: 0.7
  api_key: "${OPENAI_API_KEY}"

# Caching
cache:
  enabled: true
  ttl: 3600
  max_size: 10000

# Logging
logging:
  level: "INFO"
  format: "json"
  file: "logs/trag.log"
```

#### .env.example
```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password

# PostgreSQL
POSTGRES_URI=postgresql://postgres:password@localhost:5432/trag
POSTGRES_PASSWORD=password

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Application
ENVIRONMENT=development
SECRET_KEY=your-secret-key
DEBUG=true
```

---

## Testing Strategy

### Test Pyramid
```
       /\
      /  \     E2E Tests (5%)
     /----\
    /      \   Integration Tests (15%)
   /--------\
  /          \ Unit Tests (80%)
 /____________\
```

### Unit Tests

**Example: test_decay.py**
```python
import pytest
from datetime import datetime, timedelta
from src.deprecation.decay import DecayFunction

class TestDecayFunction:
    def test_fresh_fact_high_fvs(self):
        """Recently verified facts should have high FVS"""
        decay = DecayFunction(default_lambda=0.01)
        
        last_verified = datetime.now() - timedelta(days=1)
        current = datetime.now()
        
        fvs = decay.calculate_fvs(last_verified, current)
        
        assert fvs > 0.99
    
    def test_old_fact_low_fvs(self):
        """Old facts should have low FVS"""
        decay = DecayFunction(default_lambda=0.01)
        
        last_verified = datetime.now() - timedelta(days=365)
        current = datetime.now()
        
        fvs = decay.calculate_fvs(last_verified, current)
        
        assert fvs < 0.01
    
    def test_custom_lambda(self):
        """Higher lambda should cause faster decay"""
        decay = DecayFunction()
        
        last_verified = datetime.now() - timedelta(days=100)
        current = datetime.now()
        
        fvs_slow = decay.calculate_fvs(last_verified, current, lambda_val=0.01)
        fvs_fast = decay.calculate_fvs(last_verified, current, lambda_val=0.1)
        
        assert fvs_fast < fvs_slow
```

### Integration Tests

**Example: test_integration.py**
```python
@pytest.mark.integration
class TestEndToEnd:
    @pytest.fixture
    async def orchestrator(self):
        # Initialize with test databases
        return QueryOrchestrator(...)
    
    async def test_simple_query(self, orchestrator):
        """Test complete query workflow"""
        result = await orchestrator.process_query(
            query="Who is the president of USA?",
            query_time=datetime(2026, 1, 26)
        )
        
        assert "Trump" in result['answer']
        assert result['confidence'] > 0.8
        assert len(result['sources']) > 0
        assert result['latency_ms'] < 3000
```

### Performance Tests
```python
@pytest.mark.performance
class TestPerformance:
    async def test_latency_under_threshold(self, orchestrator):
        """Ensure queries complete within 3 seconds"""
        queries = [
            "Who is the president?",
            "What is the GDP of USA?",
            "Who won the 2024 election?"
        ]
        
        for query in queries:
            start = time.time()
            result = await orchestrator.process_query(query)
            latency = time.time() - start
            
            assert latency < 3.0
```

---

## Deployment Guide

### Docker Deployment

**Build and Run:**
```bash
# Build image
docker build -t trag:latest .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f trag-api

# Health check
curl http://localhost:8000/health
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trag-api
  template:
    metadata:
      labels:
        app: trag-api
    spec:
      containers:
      - name: trag-api
        image: your-registry/trag:latest
        ports:
        - containerPort: 8000
        env:
        - name: NEO4J_URI
          valueFrom:
            secretKeyRef:
              name: trag-secrets
              key: neo4j-uri
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: trag-secrets
              key: openai-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Monitoring Setup

**Prometheus Configuration:**
```yaml
scrape_configs:
  - job_name: 'trag-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

**Grafana Dashboard:**
- Query throughput (queries/sec)
- Average latency
- Error rate
- Cache hit rate
- FVS distribution
- Top queries

---

## Troubleshooting

### Common Issues

#### Issue 1: Neo4j Connection Fails

**Symptoms:**
```
neo4j.exceptions.ServiceUnavailable: Failed to establish connection
```

**Solutions:**
1. Check Neo4j is running: `neo4j status`
2. Verify URI: `bolt://localhost:7687`
3. Test credentials in Neo4j Browser
4. Check firewall: `sudo ufw allow 7687`

#### Issue 2: FAISS Out of Memory

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. Use IVF index instead of Flat:
```python
quantizer = faiss.IndexFlatL2(768)
index = faiss.IndexIVFFlat(quantizer, 768, 100)
index.train(embeddings[:10000])
index.add(embeddings)
```

2. Process in batches:
```python
batch_size = 1000
for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i+batch_size]
    index.add(batch)
```

#### Issue 3: LLM API Rate Limiting

**Symptoms:**
```
openai.error.RateLimitError: Rate limit exceeded
```

**Solutions:**
1. Implement exponential backoff (already in Module 5.2)
2. Use token bucket rate limiter
3. Switch to cheaper model for testing (gpt-3.5-turbo)

#### Issue 4: Slow Temporal Queries

**Symptoms:**
- Neo4j queries take > 1 second

**Solutions:**
1. Create composite indexes:
```cypher
CREATE INDEX temporal_range FOR ()-[r:RELATES_TO]-() 
ON (r.start_time, r.end_time)
```

2. Use query parameters:
```python
query = "MATCH (n) WHERE n.start_time >= $start RETURN n"
session.run(query, start=start_time)
```

#### Issue 5: Inconsistent Timestamps

**Solutions:**
1. Implement conflict resolution in Module 1.4
2. Add source priority scoring
3. Store all versions with provenance

---

## Success Metrics

### Performance Benchmarks

| Metric                | MVP Target | Production Target | Achieved |
|-----------------------|------------|-------------------|----------|
| **MRR**               | ≥ 0.40     | ≥ 0.47            | ___      |
| **Temporal Accuracy** | ≥ 85%      | ≥ 90%             | ___      |
| **Hits@1**            | ≥ 35%      | ≥ 40%             | ___      |
| **Hits@5**            | ≥ 60%      | ≥ 70%             | ___      |
| **Hits@10**           | ≥ 70%      | ≥ 75%             | ___      |
| **Avg Latency**       | ≤ 5s       | ≤ 3s              | ___      |
| **P95 Latency**       | ≤ 8s       | ≤ 5s              | ___      |
| **Cache Hit Rate**    | ≥ 40%      | ≥ 60%             | ___      |
| **Uptime**            | ≥ 95%      | ≥ 99%             | ___      |

### Evaluation Metrics

**Mean Reciprocal Rank (MRR):**
```
MRR = (1/|Q|) × Σ(1/rank_i)

Where rank_i is the position of the first correct answer
```

**Temporal Accuracy (TA):**
```
TA = (Correct predictions with valid timestamps) / (Total predictions)
```

**Hits@K:**
```
Hits@K = (Queries with correct answer in top-K) / (Total queries)
```

### Quality Metrics

- **Fact Coverage**: % of queries with sufficient context (target: >90%)
- **Citation Accuracy**: % of responses with correct source attribution (target: >95%)
- **Temporal Consistency**: % of responses without contradictions (target: >98%)
- **Confidence Calibration**: Correlation between confidence score and accuracy (target: >0.8)

---

## Validation Checklist

### Module-Level Completion

- [ ] **Module 1: Data Pipeline**
  - [ ] Can ingest ICEWS14 and Wikidata
  - [ ] 95%+ timestamps normalized
  - [ ] 10k+ valid quadruples extracted
  - [ ] Embeddings generated for all facts
  
- [ ] **Module 2: TKG Builder**
  - [ ] Neo4j connected and schema created
  - [ ] All facts loaded into graph
  - [ ] Temporal queries < 100ms
  - [ ] Can retrieve facts by time range
  
- [ ] **Module 3: Deprecation Detection**
  - [ ] FVS calculated for all facts
  - [ ] Deprecation threshold tuned
  - [ ] Automated refresh works
  - [ ] Outdated facts filtered correctly
  
- [ ] **Module 4: Time-Aware Retriever**
  - [ ] Query encoding works
  - [ ] Vector search returns relevant candidates
  - [ ] Temporal filter removes invalid facts
  - [ ] WRS ranking improves retrieval
  - [ ] Context fits LLM token limit
  
- [ ] **Module 5: LLM Generator**
  - [ ] Prompts formatted correctly
  - [ ] LLM API calls succeed
  - [ ] Responses coherent and relevant
  - [ ] Post-processing handles edge cases
  
- [ ] **Module 6: Response Validator**
  - [ ] Temporal consistency detected
  - [ ] Citations extracted correctly
  - [ ] Confidence scores reasonable
  - [ ] Explainability reports generated
  
- [ ] **Module 7: API & Orchestration**
  - [ ] Health endpoint works
  - [ ] Query endpoint returns valid responses
  - [ ] End-to-end latency < 3s
  - [ ] Caching reduces repeat query time
  - [ ] Error handling covers edge cases

### System-Level Completion

- [ ] **Functional Requirements**
  - [ ] Answers temporal queries correctly
  - [ ] Filters outdated information
  - [ ] Provides source citations with timestamps
  - [ ] Handles ambiguous time references
  
- [ ] **Performance Requirements**
  - [ ] MRR ≥ 0.47 on test set
  - [ ] Temporal Accuracy ≥ 90%
  - [ ] Average latency ≤ 3000ms
  - [ ] Handles 10+ concurrent queries
  
- [ ] **Quality Requirements**
  - [ ] No hallucinated facts
  - [ ] Temporal contradictions flagged
  - [ ] Confidence scores calibrated
  - [ ] Explainability reports understandable
  
- [ ] **Deployment Requirements**
  - [ ] Docker container builds
  - [ ] Environment variables configured
  - [ ] Database migrations run
  - [ ] API documentation complete
  - [ ] Monitoring in place

---

## Quick Reference Commands

### Development
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python scripts/preprocess_data.py --source icews14
python scripts/build_tkg.py

# Start API
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/

# Type check
mypy src/
```

### Database Management
```bash
# Neo4j
neo4j start
neo4j status
neo4j stop

# PostgreSQL
psql -U postgres -d trag

# Redis
redis-cli
> KEYS *
> FLUSHDB
```

### Deployment
```bash
# Docker
docker-compose up -d
docker-compose logs -f
docker-compose down

# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the president?", "top_k": 5}'
```

---

## Resources

### Documentation
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence-BERT Models](https://www.sbert.net/docs/pretrained_models.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Datasets
- [ICEWS18](https://github.com/INK-USC/RE-Net)
- [Wikidata SPARQL](https://query.wikidata.org/)
- [Freebase](https://developers.google.com/freebase)
- [GDELT](https://gdeltproject.org/)

### Papers
- "Temporal Knowledge Graph Reasoning" (Various)
- "RE-Net: Recurrent Evolution Network for TKG" 
- "Retrieval-Augmented Generation for LLMs"
- For further details on project methodology and technical specifications, please refer to the [references/](./references/) directory.

---

## Final Notes

### Critical Success Factors

1. **Start Small**: Begin with 1000 facts, scale up gradually
2. **Test Continuously**: Write tests before moving to next module
3. **Version Control**: Commit after each functional unit
4. **Profile Early**: Find bottlenecks before they become problems
5. **Document As You Go**: Future you will be grateful

### When You Get Stuck

1. Check troubleshooting section for your issue
2. Review example walkthrough for expected behavior
3. Use print/logging statements to trace data flow
4. Simplify: Remove complexity until it works, then add back
5. Ask for help with error messages and what you've tried

### Support

- **GitHub Issues**: Report bugs and request features
- **Email**: support@yourdomain.com
- **Documentation**: Full API docs at `/docs` when running

---

## Week 1 Deep Dive

### Detailed Week 1 Implementation Prompt

**Objective:** Complete data pipeline foundation with production-ready code.

**Time Estimate:** 12-15 hours

**Prerequisites:**
- Python 3.10+ environment setup
- Git configured
- Terminal access

### Task Breakdown

#### Task 1: Environment Setup (0.5 hours)

**Create Project Structure:**
```bash
mkdir -p t-rag/{src/{data_pipeline,tkg,deprecation,retriever,generator,validator,api,evaluation,utils},tests,scripts,data/{raw,processed,cache,embeddings},config,docker,logs}
cd t-rag
git init
```

**Create requirements.txt:**
```txt
# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0

# Data Processing
pandas==2.1.3
numpy==1.26.2
python-dateutil==2.8.2

# NLP
spacy==3.7.2
sentence-transformers==2.2.2

# HTTP
requests==2.31.0
httpx==0.25.2

# Configuration
pyyaml==6.0.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.0
```

**Setup Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Create .env.example:**
```bash
# Data Sources
ICEWS_DATA_URL=https://github.com/INK-USC/RE-Net/raw/master/data/ICEWS18/train.txt

# Paths
DATA_RAW_DIR=data/raw
DATA_PROCESSED_DIR=data/processed
CACHE_DIR=data/cache

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/pipeline.log
```

**Create .gitignore:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.venv

# Data
data/raw/*
data/processed/*
data/cache/*
!data/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# Environment
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.coverage
htmlcov/
.pytest_cache/
```

#### Task 2: Module 1.1 - Data Fetcher (2 hours)

**File: src/data_pipeline/fetcher.py**
```python
"""
Data fetcher for T-RAG pipeline.
Handles downloading and loading datasets with caching.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
import requests
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and loads temporal knowledge datasets."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize DataFetcher.
        
        Args:
            cache_dir: Directory for caching downloaded files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataFetcher initialized with cache_dir: {cache_dir}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _download_file(self, url: str, output_path: Path) -> None:
        """
        Download file with progress bar and retry logic.
        
        Args:
            url: URL to download from
            output_path: Local path to save file
            
        Raises:
            requests.RequestException: If download fails after retries
        """
        logger.info(f"Downloading from {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {output_path}")
    
    def fetch_icews14(
        self,
        force_download: bool = False,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch ICEWS14 dataset.
        
        Args:
            force_download: Skip cache and re-download
            limit: Maximum number of rows to load (for testing)
            
        Returns:
            DataFrame with columns: [event_id, head, relation, tail, date]
            
        Raises:
            ValueError: If file format is invalid
        """
        url = "https://github.com/INK-USC/RE-Net/raw/master/data/ICEWS14/train.txt"
        cache_file = self.cache_dir / "icews14_train.txt"
        
        # Download if not cached or force_download
        if not cache_file.exists() or force_download:
            self._download_file(url, cache_file)
        else:
            logger.info(f"Using cached file: {cache_file}")
        
        # Load data
        try:
            df = pd.read_csv(
                cache_file,
                sep='\t',
                names=['head', 'relation', 'tail', 'date'],
                nrows=limit
            )
            
            # Add event_id
            df.insert(0, 'event_id', range(1, len(df) + 1))
            
            logger.info(f"Loaded {len(df)} records from ICEWS14")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse ICEWS14 file: {e}")
            raise ValueError(f"Invalid file format: {e}")
    
    def load_from_file(
        self,
        filepath: str,
        expected_cols: List[str],
        sep: str = '\t'
    ) -> pd.DataFrame:
        """
        Load data from local file with validation.
        
        Args:
            filepath: Path to data file
            expected_cols: Required column names
            sep: Column separator
            
        Returns:
            Validated DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading from {filepath}")
        
        df = pd.read_csv(filepath, sep=sep)
        
        # Validate columns
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
        return df
```

**File: tests/test_fetcher.py**
```python
"""Tests for data fetcher module."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.data_pipeline.fetcher import DataFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create DataFetcher with temporary cache directory."""
    return DataFetcher(cache_dir=str(tmp_path))


@pytest.fixture
def mock_response():
    """Mock successful HTTP response."""
    mock = Mock()
    mock.status_code = 200
    mock.headers = {'content-length': '1024'}
    mock.iter_content = lambda chunk_size: [b'test data'] * 10
    return mock


def test_fetcher_initialization(fetcher, tmp_path):
    """Test DataFetcher initialization."""
    assert fetcher.cache_dir == tmp_path
    assert fetcher.cache_dir.exists()


@patch('src.data_pipeline.fetcher.requests.get')
def test_download_file(mock_get, fetcher, tmp_path, mock_response):
    """Test file download."""
    mock_get.return_value = mock_response
    
    output_path = tmp_path / "test_file.txt"
    fetcher._download_file("http://example.com/data.txt", output_path)
    
    assert output_path.exists()
    mock_get.assert_called_once()


@patch('src.data_pipeline.fetcher.requests.get')
def test_fetch_icews14(mock_get, fetcher, tmp_path):
    """Test ICEWS14 dataset fetching."""
    # Create mock data
    mock_data = "USA\tmake_statement\tChina\t2014-01-15\n"
    mock_data += "Russia\tcriticize\tNATO\t2014-02-10\n"
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {'content-length': str(len(mock_data))}
    mock_response.iter_content = lambda chunk_size: [mock_data.encode()]
    mock_get.return_value = mock_response
    
    df = fetcher.fetch_icews14()
    
    assert isinstance(df, pd.DataFrame)
    assert 'event_id' in df.columns
    assert 'head' in df.columns
    assert 'relation' in df.columns
    assert 'tail' in df.columns
    assert 'date' in df.columns
    assert len(df) == 2


def test_fetch_icews14_with_limit(fetcher):
    """Test fetching with row limit."""
    # Create test data file
    test_file = fetcher.cache_dir / "icews14_train.txt"
    test_data = "USA\tmake_statement\tChina\t2014-01-15\n" * 100
    test_file.write_text(test_data)
    
    df = fetcher.fetch_icews14(limit=10)
    
    assert len(df) == 10


def test_fetch_icews14_cache_hit(fetcher):
    """Test cache hit scenario."""
    # Create cached file
    test_file = fetcher.cache_dir / "icews14_train.txt"
    test_data = "USA\tmake_statement\tChina\t2014-01-15\n"
    test_file.write_text(test_data)
    
    with patch('src.data_pipeline.fetcher.requests.get') as mock_get:
        df = fetcher.fetch_icews14()
        
        # Should not download if cached
        mock_get.assert_not_called()
        assert len(df) == 1


def test_load_from_file(fetcher, tmp_path):
    """Test loading from local file."""
    # Create test file
    test_file = tmp_path / "test_data.csv"
    test_data = pd.DataFrame({
        'head': ['USA', 'China'],
        'relation': ['make_statement', 'respond'],
        'tail': ['China', 'USA'],
        'date': ['2014-01-15', '2014-01-16']
    })
    test_data.to_csv(test_file, index=False)
    
    df = fetcher.load_from_file(
        str(test_file),
        expected_cols=['head', 'relation', 'tail', 'date'],
        sep=','
    )
    
    assert len(df) == 2
    assert list(df.columns) == ['head', 'relation', 'tail', 'date']


def test_load_from_file_missing_columns(fetcher, tmp_path):
    """Test error when columns are missing."""
    test_file = tmp_path / "test_data.csv"
    test_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    test_data.to_csv(test_file, index=False)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        fetcher.load_from_file(
            str(test_file),
            expected_cols=['head', 'relation', 'tail'],
            sep=','
        )


def test_load_from_file_not_found(fetcher):
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        fetcher.load_from_file(
            "nonexistent.csv",
            expected_cols=['col1'],
            sep=','
        )
```

**Run Tests:**
```bash
pytest tests/test_fetcher.py -v --cov=src/data_pipeline/fetcher
```

---

## Conclusion

This comprehensive roadmap provides:

✅ **Complete system architecture** with all 7 modules  
✅ **Week-by-week implementation plan** for solo developers  
✅ **Production-ready code examples** with full implementations  
✅ **Comprehensive testing strategy** at all levels  
✅ **Deployment guides** for Docker and Kubernetes  
✅ **Troubleshooting sections** for common issues  
✅ **Success metrics** and validation checklists  

**Total Estimated Time:** 8-10 weeks (solo developer)

**Start Point:** Week 1, Task 1 - Environment Setup

**End Goal:** Production-ready T-RAG system achieving:
- MRR ≥ 0.47
- Temporal Accuracy ≥ 90%
- Query Latency ≤ 3 seconds

**Next Step:** Begin with Week 1 implementation using the detailed prompt provided in this document.

Good luck building T-RAG! 🚀