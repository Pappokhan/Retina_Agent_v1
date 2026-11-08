#!/usr/bin/env python
"""Check (bm25) or build (faiss) the retrieval index over knowledge/*.md.

    python scripts/build_index.py
    RAG_BACKEND=faiss python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import rag  # noqa: E402


def main() -> int:
    chunks = rag.load_chunks()
    print(f"{len(chunks)} chunks from {len({c.metadata['source'] for c in chunks})} files in {config.KNOWLEDGE_DIR}")
    if config.RAG_BACKEND == "faiss":
        n = rag.rebuild_faiss_index()
        print(f"FAISS index written to {config.INDEX_DIR} ({n} chunks)")
    for q in ("test Dice and IoU", "Is FedAvg real federated averaging?", "what is the attention focus ratio"):
        docs = rag.get_retriever().invoke(q)
        print(f"\nQ: {q}\n   top sources: {[d.metadata['source'] for d in docs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
