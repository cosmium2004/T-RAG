"""
T-RAG Interactive Demo — Streamlit Application.

A rich, interactive web interface for the T-RAG system.
Run: streamlit run app.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tkg.neo4j_client import Neo4jClient
from src.retriever.vector_search import VectorSearch
from src.retriever.query_encoder import QueryEncoder
from src.deprecation.decay import DecayFunction
from src.retriever.temporal_filter import TemporalFilter
from src.retriever.wrs import WRSScorer
from src.retriever.context_assembler import ContextAssembler
from src.generator.prompt_builder import PromptBuilder
from src.generator.llm_client import LLMClient
from src.generator.post_processor import PostProcessor
from src.validator.consistency import ConsistencyValidator
from src.validator.confidence import ConfidenceScorer

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="T-RAG | Time-Aware RAG",
    page_icon="⏳",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}

.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #E0EAFC, #CFDEF3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: #B0C4DE;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.stat-card {
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: white;
}

.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #64FFDA;
}

.stat-card .stat-label {
    font-size: 0.8rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.confidence-high { color: #64FFDA; font-weight: 600; }
.confidence-medium { color: #FFD700; font-weight: 600; }
.confidence-low { color: #FF6B6B; font-weight: 600; }

.source-card {
    background: #0d1b2a;
    border: 1px solid #1b2838;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    color: #ccd6f6;
    font-size: 0.9rem;
}

.pipeline-step {
    display: inline-block;
    background: #112240;
    border: 1px solid #233554;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    margin: 0.2rem;
    font-size: 0.75rem;
    color: #8892b0;
}

.pipeline-step.active {
    border-color: #64FFDA;
    color: #64FFDA;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>⏳ T-RAG</h1>
    <p>Time-Aware Retrieval-Augmented Generation for LLMs</p>
</div>
""", unsafe_allow_html=True)


# ── Singleton loaders ────────────────────────────────────────────────

@st.cache_resource
def load_neo4j():
    try:
        client = Neo4jClient()
        client.connect()
        return client
    except Exception:
        return None


@st.cache_resource
def load_vector_search():
    try:
        vs = VectorSearch()
        vs.load("data/embeddings")
        return vs
    except Exception:
        return None


@st.cache_resource
def load_query_encoder():
    return QueryEncoder()


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    alpha = st.slider(
        "Semantic vs Temporal Weight (α)",
        0.0, 1.0, 0.5, 0.05,
        help="0 = pure temporal, 1 = pure semantic",
    )

    top_k = st.slider("Results (top-k)", 1, 20, 5)

    threshold = st.slider(
        "Deprecation Threshold",
        0.0, 1.0, 0.3, 0.05,
        help="Minimum FVS for a fact to be valid",
    )

    lambda_val = st.slider(
        "Decay Rate (λ)",
        0.001, 0.1, 0.01, 0.001,
        format="%.3f",
        help="Higher = faster fact deprecation",
    )

    llm_provider = st.selectbox(
        "LLM Provider",
        ["local", "openai", "anthropic"],
        help="'local' = template-based, no API key needed",
    )

    st.divider()

    # System status
    st.markdown("### 📊 System Status")

    neo4j_client = load_neo4j()
    vs = load_vector_search()

    if neo4j_client:
        try:
            health = neo4j_client.health_check()
            st.success(f"Neo4j: {health.get('version', 'connected')}")
            nodes = neo4j_client.query("MATCH (n:Entity) RETURN count(n) AS cnt")
            rels = neo4j_client.query(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt"
            )
            st.metric("Entity Nodes", f"{nodes[0]['cnt']:,}")
            st.metric("Relationships", f"{rels[0]['cnt']:,}")
        except Exception as e:
            st.error(f"Neo4j error: {e}")
    else:
        st.warning("Neo4j not connected")

    if vs:
        st.metric("FAISS Index Size", f"{vs.size:,}")
    else:
        st.warning("FAISS index not loaded")


# ── Main query area ──────────────────────────────────────────────────

tab_query, tab_explore, tab_graph = st.tabs([
    "🔍 Query", "📊 Explore Facts", "🌐 Graph View"
])


with tab_query:
    col_input, col_time = st.columns([3, 1])

    with col_input:
        query = st.text_input(
            "Ask a temporally-aware question",
            placeholder="e.g. What diplomatic actions involved North Korea in early 2018?",
        )

    with col_time:
        query_date = st.date_input(
            "Query time",
            value=datetime(2018, 6, 1),
        )

    if st.button("🚀 Submit Query", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a query.")
        elif not vs:
            st.error("FAISS index not built. Run `python scripts/generate_embeddings.py` first.")
        else:
            query_time = datetime.combine(query_date, datetime.min.time()).replace(
                tzinfo=timezone.utc
            )

            with st.spinner("Processing query through T-RAG pipeline..."):
                t0 = time.time()

                # Pipeline steps
                progress = st.progress(0, text="Encoding query...")
                encoder = load_query_encoder()
                query_vec = encoder.encode(query)
                progress.progress(20, text="Searching FAISS index...")

                candidates = vs.search(query_vec, top_n=50)
                progress.progress(40, text="Calculating FVS scores...")

                decay = DecayFunction(default_lambda=lambda_val)
                decay.score_facts(candidates, query_time, threshold)

                tf = TemporalFilter(deprecation_threshold=threshold)
                valid = tf.filter(candidates, query_time)
                progress.progress(55, text="Ranking by WRS...")

                wrs = WRSScorer(alpha=alpha)
                ranked = wrs.rank(valid, top_k=top_k)
                progress.progress(70, text="Assembling context...")

                ca = ContextAssembler()
                context = ca.format(ranked, query_time)
                progress.progress(80, text="Generating answer...")

                pb = PromptBuilder()
                messages = pb.build_messages(query, context, query_time)
                llm = LLMClient(provider=llm_provider)
                raw_answer = llm.generate(messages)
                pp = PostProcessor()
                answer = pp.process(raw_answer)
                progress.progress(90, text="Validating response...")

                cv = ConsistencyValidator()
                validation = cv.validate(answer, query_time, ranked)
                cs = ConfidenceScorer()
                confidence = cs.score(validation, ranked)

                latency = int((time.time() - t0) * 1000)
                progress.progress(100, text="Done!")
                time.sleep(0.3)
                progress.empty()

            # ── Results ──────────────────────────────────────────

            st.markdown("---")

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f'<div class="stat-card"><div class="stat-value">'
                f'{confidence["confidence"]:.0%}</div>'
                f'<div class="stat-label">Confidence</div></div>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div class="stat-card"><div class="stat-value">'
                f'{len(ranked)}</div>'
                f'<div class="stat-label">Facts Used</div></div>',
                unsafe_allow_html=True,
            )
            c3.markdown(
                f'<div class="stat-card"><div class="stat-value">'
                f'{latency}ms</div>'
                f'<div class="stat-label">Latency</div></div>',
                unsafe_allow_html=True,
            )
            rating = confidence["rating"]
            color_class = f"confidence-{rating.lower()}"
            c4.markdown(
                f'<div class="stat-card"><div class="stat-value '
                f'{color_class}">{rating}</div>'
                f'<div class="stat-label">Rating</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("### 💬 Answer")
            st.markdown(answer)

            # Confidence breakdown
            with st.expander("📊 Confidence Breakdown"):
                bd = confidence["breakdown"]
                cols = st.columns(4)
                cols[0].metric("Consistency", f"{bd['consistency']:.0%}")
                cols[1].metric("Freshness", f"{bd['temporal_freshness']:.0%}")
                cols[2].metric("Source Conf.", f"{bd['source_confidence']:.0%}")
                cols[3].metric("Coverage", f"{bd['fact_coverage']:.0%}")

            # Source facts
            st.markdown("### 📋 Source Facts")
            for i, fact in enumerate(ranked, 1):
                with st.container():
                    fc1, fc2, fc3 = st.columns([5, 1, 1])
                    fc1.markdown(
                        f"**{i}.** {fact.get('text', 'N/A')}"
                    )
                    fc2.metric("FVS", f"{fact.get('fvs', 0):.2f}")
                    fc3.metric("WRS", f"{fact.get('wrs', 0):.3f}")

            # Validation details
            if validation.get("contradictions"):
                with st.expander("⚠️ Validation Warnings"):
                    for c in validation["contradictions"]:
                        st.warning(c)


with tab_explore:
    st.markdown("### Browse Temporal Facts")

    search_term = st.text_input(
        "Search entities", placeholder="e.g. North Korea"
    )

    if search_term and neo4j_client:
        results = neo4j_client.query(
            "MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity) "
            "WHERE h.name CONTAINS $term OR t.name CONTAINS $term "
            "RETURN h.name AS head, r.relation_type AS relation, "
            "t.name AS tail, toString(r.start_time) AS start_time, "
            "r.confidence AS confidence "
            "ORDER BY r.start_time DESC LIMIT 20",
            {"term": search_term},
        )

        if results:
            st.dataframe(
                results,
                use_container_width=True,
                column_config={
                    "head": "Head Entity",
                    "relation": "Relation",
                    "tail": "Tail Entity",
                    "start_time": "Valid From",
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0,
                        max_value=1,
                        format="%.0%%",
                    ),
                },
            )
        else:
            st.info("No facts found matching that term.")


with tab_graph:
    st.markdown("### Knowledge Graph Visualization")
    st.info(
        "Open [Neo4j Browser](http://localhost:7474) for the full "
        "interactive graph. Below is a sample subgraph."
    )

    if neo4j_client:
        entity = st.text_input(
            "Center entity",
            value="North Korea",
            key="graph_entity",
        )

        if entity:
            neighbors = neo4j_client.query(
                "MATCH (h:Entity {name: $name})-[r:RELATES_TO]->(t:Entity) "
                "RETURN h.name AS source, r.relation_type AS relation, "
                "t.name AS target, toString(r.start_time) AS date "
                "LIMIT 15",
                {"name": entity},
            )

            incoming = neo4j_client.query(
                "MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity {name: $name}) "
                "RETURN h.name AS source, r.relation_type AS relation, "
                "t.name AS target, toString(r.start_time) AS date "
                "LIMIT 15",
                {"name": entity},
            )

            all_edges = neighbors + incoming

            if all_edges:
                st.dataframe(all_edges, use_container_width=True)
                st.caption(f"Showing {len(all_edges)} connections for '{entity}'")
            else:
                st.warning(f"No connections found for '{entity}'")
