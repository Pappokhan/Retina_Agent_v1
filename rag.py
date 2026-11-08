"""Retrieval-augmented generation over the project's own notes in knowledge/*.md.

Used everywhere a language model answers: the report, Ask, "explain this map", the variant
comparison and the chat agent (search_knowledge tool). Every answer that used notes returns
its sources, so the UI can show where a fact came from.

How retrieval works
  * Chunking follows the Markdown structure: each "## section" becomes one or more chunks that
    keep their file name and section title (so citations read "model_card.md > Reported test results").
  * Default backend: BM25 (pure Python, no download) with light stemming, stop-word removal and
    the section title counted twice, so a query matching a heading ranks that section first.
  * RAG_BACKEND=faiss switches to embedding search (needs langchain-huggingface,
    sentence-transformers and faiss-cpu).
  * The index is rebuilt automatically when a note is added, edited or removed, so you can edit
    knowledge/*.md while the server runs.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


@dataclass
class Hit:
    source: str        # file name, e.g. "xai.md"
    section: str       # section title, e.g. "Grad-CAM (HiResCAM)"
    text: str
    score: float

    @property
    def label(self) -> str:
        return f"{self.source} > {self.section}" if self.section else self.source


# --- chunking ---------------------------------------------------------------------
def load_chunks(knowledge_dir: Path | None = None) -> list[Document]:
    root = Path(knowledge_dir or config.KNOWLEDGE_DIR)
    by_header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"), ("##", "section")],
                                           strip_headers=True)
    by_size = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Document] = []
    for path in sorted(root.glob("*.md")):
        for part in by_header.split_text(path.read_text(encoding="utf-8")):
            title = part.metadata.get("title", "")
            section = part.metadata.get("section", "") or title
            for piece in by_size.split_text(part.page_content):
                if piece.strip():
                    chunks.append(Document(page_content=piece.strip(),
                                           metadata={"source": path.name, "title": title, "section": section}))
    return chunks


def _signature(root: Path) -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.glob("*.md")))


# --- BM25 -------------------------------------------------------------------------
_STOP = set("""a an and are as at be by can do does for from has have how i if in into is it its of on or
so that the their them then there these this to was what when where which who why will with you your
me my we our than about also not no""".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "ly", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower()) if w not in _STOP]


class BM25Index:
    """BM25 retriever with the same .invoke(query) interface as LangChain retrievers."""

    def __init__(self, chunks: list[Document], k: int):
        from rank_bm25 import BM25Okapi

        self.chunks, self.k = chunks, k
        corpus = [_tokens(f"{c.metadata.get('section', '')} {c.metadata.get('section', '')} "
                          f"{c.metadata.get('source', '').removesuffix('.md').replace('_', ' ')} {c.page_content}")
                  for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def scored(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        tokens = _tokens(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: k or self.k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def invoke(self, query: str) -> list[Document]:
        return [d for d, _ in self.scored(query)]


def _faiss(chunks: list[Document], k: int):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name=config.RAG_EMBEDDING_MODEL)
    index = Path(config.INDEX_DIR)
    if (index / "index.faiss").is_file():
        # Safe to unpickle: this index was written by rebuild_faiss_index() on this machine.
        db = FAISS.load_local(str(index), emb, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(chunks, emb)
        db.save_local(str(index))
    return db


def rebuild_faiss_index() -> int:
    import shutil

    shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
    chunks = load_chunks()
    Path(config.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    _faiss(chunks, config.RAG_K)
    clear_cache()
    return len(chunks)


# --- retriever cache (auto-refresh when notes change) ---------------------------------
_state: dict = {}
_lock = threading.Lock()


def get_retriever():
    root = Path(config.KNOWLEDGE_DIR)
    sig = (str(root), config.RAG_BACKEND, _signature(root))
    with _lock:
        if _state.get("sig") != sig:
            chunks = load_chunks(root)
            if not chunks:
                raise RuntimeError(f"No .md files found in {root}.")
            if config.RAG_BACKEND == "faiss":
                if _state.get("sig") is not None:      # notes changed: re-embed
                    import shutil
                    shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
                _state["retriever"] = _faiss(chunks, config.RAG_K)
            else:
                _state["retriever"] = BM25Index(chunks, config.RAG_K)
            _state["sig"] = sig
        return _state["retriever"]


def clear_cache() -> None:
    with _lock:
        _state.clear()


get_retriever.cache_clear = clear_cache      # old name, kept for callers and tests


# --- public API ---------------------------------------------------------------------
def retrieve(query: str, k: int | None = None) -> list[Hit]:
    """Top notes for a query, best first, without duplicate chunks."""
    k = k or config.RAG_K
    query = (query or "").strip()[:1000]
    if not query:
        return []
    r = get_retriever()
    if isinstance(r, BM25Index):
        pairs = r.scored(query, k)
    else:
        pairs = [(d, float(1 / (1 + s))) for d, s in r.similarity_search_with_score(query, k=k)]
    if pairs and isinstance(r, BM25Index):
        top = pairs[0][1]
        pairs = [(d, s) for d, s in pairs if s >= 0.35 * top]     # drop weak, off-topic tail matches
    seen, hits = set(), []
    for doc, score in pairs:
        key = (doc.metadata.get("source"), doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        hits.append(Hit(source=doc.metadata.get("source", "?"), section=doc.metadata.get("section", ""),
                        text=doc.page_content, score=round(score, 3)))
    return hits


def search(query: str, k: int | None = None) -> str:
    """Notes as plain text for the agent, each tagged [n] file > section."""
    hits = retrieve(query, k)
    if not hits:
        return "No matching notes found."
    return "\n\n".join(f"[{i}] {h.label}\n{h.text}" for i, h in enumerate(hits, 1))


def context_block(hits: list[Hit], max_chars: int = 4500) -> str:
    """Notes for a prompt, numbered so the model can cite them as [1], [2]..."""
    if not hits:
        return ""
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        entry = f"[{i}] ({h.label}) {h.text}"
        if used + len(entry) > max_chars:
            break
        parts.append(entry)
        used += len(entry)
    return ("Project notes (facts about this model, its training, metrics, explanations and limitations). "
            "Use them for any such fact and cite the note number like [2]. If the notes do not cover "
            "something, say so instead of guessing:\n" + "\n".join(parts))


def sources(hits: list[Hit]) -> list[dict]:
    """What the UI shows under an answer."""
    return [{"id": i, "source": h.source, "section": h.section,
             "snippet": (h.text[:220] + "…") if len(h.text) > 220 else h.text}
            for i, h in enumerate(hits, 1)]


# --- queries used by the explanation features ----------------------------------------------
_VARIANT_TERMS = {
    "best": "best validation checkpoint",
    "standard": "final epoch standard checkpoint test results",
    "fedavg": "FedAvg federated caveat sequential shard training",
    "fedprox": "FedProx federated caveat proximal anchor",
}
_XAI_TERMS = {
    "attention": "attention gate map psi focus ratio attention on vessels",
    "gradcam": "Grad-CAM HiResCAM gradient activation region evidence layer",
    "uncertainty": "uncertainty entropy bits undecided pixels vessel edges",
    "mc_dropout": "Monte-Carlo dropout passes standard deviation stable prediction",
}


def query_for(kind: str, variant: str = "best", extra: str = "") -> str:
    v = _VARIANT_TERMS.get(variant, "")
    if kind == "report":
        return (f"visual quality check accuracy attention gate focus ratio uncertainty entropy image quality "
                f"sharpness regions domain shift {v} {extra}")
    if kind == "xai":
        return f"{_XAI_TERMS.get(extra, extra)} explanations are not proof"
    if kind == "compare":
        return "compare variants agreement Dice disagreement FedAvg FedProx federated caveat best final epoch"
    return f"{extra} {v}"
