"""
Benchmark dataset for T-RAG evaluation.

Contains temporal QA pairs for evaluating:
- Temporal accuracy (correct date references)
- Factual grounding (answers backed by source facts)
- Conflict resolution (handling contradictory information)
"""

BENCHMARK_QUERIES = [
    # ── Diplomatic events ────────────────────────────────────────
    {
        "id": "q01",
        "query": "What diplomatic events involved North Korea in early 2018?",
        "query_date": "2018-06-01",
        "expected_entities": ["North Korea", "Kim Jong-Un"],
        "expected_temporal": "2018",
        "category": "diplomatic",
        "difficulty": "easy",
    },
    {
        "id": "q02",
        "query": "Who met with Kim Jong-Un in 2018?",
        "query_date": "2018-12-31",
        "expected_entities": ["Kim Jong-Un"],
        "expected_temporal": "2018",
        "category": "diplomatic",
        "difficulty": "easy",
    },
    {
        "id": "q03",
        "query": "What was the status of US-North Korea relations in March 2018?",
        "query_date": "2018-04-01",
        "expected_entities": ["United States", "North Korea"],
        "expected_temporal": "March 2018",
        "category": "diplomatic",
        "difficulty": "medium",
    },
    # ── Military / Security ──────────────────────────────────────
    {
        "id": "q04",
        "query": "Were there any military threats in the Korean peninsula in 2017?",
        "query_date": "2018-01-01",
        "expected_entities": ["North Korea"],
        "expected_temporal": "2017",
        "category": "security",
        "difficulty": "medium",
    },
    {
        "id": "q05",
        "query": "What sanctions were imposed on North Korea?",
        "query_date": "2018-06-01",
        "expected_entities": ["North Korea"],
        "expected_temporal": "",
        "category": "sanctions",
        "difficulty": "easy",
    },
    # ── Conflict resolution ──────────────────────────────────────
    {
        "id": "q06",
        "query": "Did the relationship between US and North Korea improve or worsen in 2018?",
        "query_date": "2018-12-31",
        "expected_entities": ["United States", "North Korea"],
        "expected_temporal": "2018",
        "category": "conflict_resolution",
        "difficulty": "hard",
    },
    # ── Temporal reasoning ───────────────────────────────────────
    {
        "id": "q07",
        "query": "What happened after the Singapore summit in June 2018?",
        "query_date": "2018-12-31",
        "expected_entities": ["Singapore"],
        "expected_temporal": "June 2018",
        "category": "temporal_reasoning",
        "difficulty": "hard",
    },
    {
        "id": "q08",
        "query": "What was the most recent diplomatic event as of January 2018?",
        "query_date": "2018-01-31",
        "expected_entities": [],
        "expected_temporal": "January 2018",
        "category": "temporal_reasoning",
        "difficulty": "medium",
    },
    # ── Entity-specific ──────────────────────────────────────────
    {
        "id": "q09",
        "query": "What role did China play in Korean peninsula affairs?",
        "query_date": "2018-06-01",
        "expected_entities": ["China"],
        "expected_temporal": "",
        "category": "entity_specific",
        "difficulty": "medium",
    },
    {
        "id": "q10",
        "query": "What actions did the United Nations take regarding North Korea?",
        "query_date": "2018-06-01",
        "expected_entities": ["United Nations", "North Korea"],
        "expected_temporal": "",
        "category": "entity_specific",
        "difficulty": "easy",
    },
]
