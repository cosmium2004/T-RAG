"""
Ingest a document into the T-RAG knowledge base.

Usage:
    python scripts/ingest_document.py --input report.pdf
    python scripts/ingest_document.py --input https://example.com/article
    python scripts/ingest_document.py --input notes.txt --provider ollama --model mistral
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.document_loader import DocumentLoader
from src.data_pipeline.text_chunker import TextChunker
from src.data_pipeline.quadruple_extractor import QuadrupleExtractor
from src.data_pipeline.duplicate_resolver import DuplicateResolver
from src.data_pipeline.embedder import Embedder
from src.retriever.vector_search import VectorSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a document into the T-RAG knowledge base."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to document (PDF, TXT) or URL",
    )
    parser.add_argument(
        "--provider", default="ollama",
        help="LLM provider for extraction (default: ollama)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: provider default)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Target chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--overlap", type=int, default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--embeddings-dir", default="data/embeddings",
        help="Directory for FAISS index and embeddings (default: data/embeddings)",
    )
    parser.add_argument(
        "--output-facts", default=None,
        help="Optional: save extracted facts to JSON file",
    )

    args = parser.parse_args()

    # ── Step 1: Load document ──────────────────────────────────────
    print(f"\n📄 Loading document: {args.input}")
    loader = DocumentLoader()
    pages = loader.load(args.input)
    print(f"   Loaded {len(pages)} page(s)")

    # ── Step 2: Chunk text ─────────────────────────────────────────
    print(f"\n✂️  Chunking text (size={args.chunk_size}, overlap={args.overlap})")
    chunker = TextChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    chunks = chunker.chunk_pages(pages)
    print(f"   Created {len(chunks)} chunks")

    # ── Step 3: Extract quadruples ──────────────────────────────────
    print(f"\n🔍 Extracting quadruples using {args.provider}...")
    extractor = QuadrupleExtractor(provider=args.provider, model=args.model)

    source_name = Path(args.input).stem if not args.input.startswith("http") else args.input
    facts = extractor.extract_from_document(chunks, source_name=source_name)
    print(f"   Extracted {len(facts)} unique facts")

    if not facts:
        print("\n⚠️  No facts extracted. Check the document content and LLM provider.")
        return

    # ── Step 4: Deduplicate ─────────────────────────────────────────
    resolver = DuplicateResolver()
    facts = resolver.resolve(facts)
    print(f"   After deduplication: {len(facts)} facts")

    # ── Step 5: Embed ───────────────────────────────────────────────
    print("\n📐 Generating embeddings...")
    embedder = Embedder()
    embeddings = embedder.embed_facts(facts)
    fact_ids = [f["id"] for f in facts]
    print(f"   Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}d)")

    # ── Step 6: Update FAISS index ──────────────────────────────────
    print(f"\n📦 Updating FAISS index at {args.embeddings_dir}...")
    vs = VectorSearch()
    emb_dir = Path(args.embeddings_dir)

    if (emb_dir / "faiss_index.bin").exists():
        vs.load(args.embeddings_dir)
        print(f"   Loaded existing index: {vs.size} vectors")
    else:
        emb_dir.mkdir(parents=True, exist_ok=True)

    added = vs.append_to_index(embeddings, fact_ids, facts)
    vs.save(args.embeddings_dir)
    print(f"   Added {added} new vectors (total: {vs.size})")

    # ── Optional: save facts ────────────────────────────────────────
    if args.output_facts:
        with open(args.output_facts, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Saved facts to {args.output_facts}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("✅ Ingestion complete!")
    print(f"   Source:     {args.input}")
    print(f"   Facts:      {len(facts)}")
    print(f"   Index size: {vs.size} vectors")
    print(f"   Provider:   {args.provider} ({extractor.llm.model})")
    print("=" * 50)


if __name__ == "__main__":
    main()
