// ============================================================
// T-RAG Neo4j Schema — Constraints & Indexes
// Run via Neo4j Browser or cypher-shell
// ============================================================

// --- Constraints ---
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT fact_id_unique IF NOT EXISTS
  FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE;

// --- Indexes ---
CREATE INDEX entity_type_idx IF NOT EXISTS
  FOR (e:Entity) ON (e.entity_type);

CREATE INDEX fact_start_idx IF NOT EXISTS
  FOR (f:Fact) ON (f.start_time);

CREATE INDEX fact_end_idx IF NOT EXISTS
  FOR (f:Fact) ON (f.end_time);

// Full-text search on entity names
CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS
  FOR (e:Entity) ON EACH [e.name];

// Composite relationship temporal index
CREATE INDEX rel_temporal_idx IF NOT EXISTS
  FOR ()-[r:RELATES_TO]-() ON (r.start_time, r.end_time);
