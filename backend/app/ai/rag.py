"""
Retrieval-Augmented Generation (RAG) pipeline for the OmniEdge copilot.

Indexes the markdown knowledge base under `app/ai/knowledge/` (and, if
present, the top-level `docs/ARCHITECTURE.md`) into chunks, embeds them,
and retrieves the top-k most relevant chunks for a natural-language
query so the agentic copilot can ground its answers in the actual
system documentation instead of hallucinating architecture details.

Two embedding backends, same interface:
  - `SentenceTransformerEmbedder` — real dense embeddings via
    `sentence-transformers`, used if installed.
  - `TfidfEmbedder`               — a dependency-free TF-IDF + cosine
    similarity fallback (NumPy only), used otherwise. Good enough for a
    knowledge base of this size; swap in the real embedder for a larger
    corpus or noisier queries.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("omniedge.ai.rag")

_DEFAULT_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
# rag.py lives at backend/app/ai/rag.py — the repo's docs/ dir is a
# sibling of backend/, i.e. three levels up from this file's directory.
_ARCHITECTURE_DOC = os.path.join(
    os.path.dirname(  # backend/app/ai -> backend/app
        os.path.dirname(  # backend/app -> backend
            os.path.dirname(os.path.abspath(__file__))  # backend/app/ai
        )
    ),
    "..",
    "docs",
    "ARCHITECTURE.md",
)
_ARCHITECTURE_DOC = os.path.normpath(_ARCHITECTURE_DOC)

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_SENTENCE_TRANSFORMERS = False
    logger.info(
        "sentence-transformers not found; RAG pipeline will use the "
        "dependency-free TF-IDF embedder. Install sentence-transformers "
        "for real dense-embedding retrieval."
    )


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


@dataclass
class Chunk:
    doc_id: str
    heading: str
    text: str
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.chunk_id = f"{self.doc_id}#{self.heading}".replace(" ", "-").lower()


def _split_into_chunks(doc_id: str, markdown_text: str) -> List[Chunk]:
    """Splits a markdown document into chunks along `##` headings."""
    sections = re.split(r"\n(?=##\s)", markdown_text)
    chunks: List[Chunk] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"##\s+(.*)", section)
        heading = heading_match.group(1).strip() if heading_match else doc_id
        body = section
        if len(body) < 20:
            continue
        chunks.append(Chunk(doc_id=doc_id, heading=heading, text=body))
    return chunks


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------


class TfidfEmbedder:
    """Dependency-free TF-IDF vectorizer + cosine similarity."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, documents: List[str]) -> None:
        doc_freq: Counter = Counter()
        tokenized_docs = [set(_tokenize(d)) for d in documents]
        for tokens in tokenized_docs:
            doc_freq.update(tokens)

        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq.keys()))}
        n_docs = len(documents)
        self.idf = np.zeros(len(self.vocab))
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        vec = np.zeros(len(self.vocab))
        tokens = _tokenize(text)
        if not tokens:
            return vec
        tf = Counter(tokens)
        max_count = max(tf.values())
        for term, count in tf.items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = (count / max_count) * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class SentenceTransformerEmbedder:
    """Real dense-embedding backend, used automatically if installed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)  # type: ignore

    def fit(self, documents: List[str]) -> None:
        pass  # no fitting step needed for a pretrained sentence embedder

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)  # type: ignore
        return np.array(vecs)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------
# The vector store / retriever itself
# --------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class KnowledgeBase:
    """
    In-memory RAG vector store. Loads every `.md` file from the
    knowledge-base directory (plus `docs/ARCHITECTURE.md` if present),
    chunks them by heading, embeds every chunk, and answers similarity
    queries. Swap this for a real vector DB (pgvector, Qdrant, Chroma,
    Pinecone...) for a corpus too large to hold in memory.
    """

    def __init__(self, kb_dir: str = _DEFAULT_KB_DIR, include_architecture_doc: bool = True) -> None:
        self.embedder = SentenceTransformerEmbedder() if _HAS_SENTENCE_TRANSFORMERS else TfidfEmbedder()
        self.backend_name = "sentence-transformers" if _HAS_SENTENCE_TRANSFORMERS else "tfidf"
        self.chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None
        self._load_and_index(kb_dir, include_architecture_doc)

    def _load_and_index(self, kb_dir: str, include_architecture_doc: bool) -> None:
        paths = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
        if include_architecture_doc and os.path.exists(_ARCHITECTURE_DOC):
            paths.append(_ARCHITECTURE_DOC)

        for path in paths:
            doc_id = os.path.splitext(os.path.basename(path))[0]
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.chunks.extend(_split_into_chunks(doc_id, text))

        if not self.chunks:
            logger.warning("No knowledge-base documents found under %s", kb_dir)
            return

        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        # Weight embeddings toward each chunk's heading (repeated) so a
        # query naming the situation ("node shows DOWN") ranks the chunk
        # titled that way above one that merely mentions it in passing.
        weighted_texts = [f"{c.heading} {c.heading} {c.text}" for c in self.chunks]
        self._vectors = self.embedder.embed_batch(weighted_texts)
        logger.info(
            "Indexed %d chunks from %d document(s) with the %s embedder",
            len(self.chunks),
            len(paths),
            self.backend_name,
        )

    def query(self, question: str, top_k: int = 3, min_score: float = 0.05) -> List[RetrievedChunk]:
        if self._vectors is None or not self.chunks:
            return []
        q_vec = self.embedder.embed(question)
        scores = [_cosine_sim(q_vec, self._vectors[i]) for i in range(len(self.chunks))]
        ranked = sorted(zip(self.chunks, scores), key=lambda cs: cs[1], reverse=True)
        return [RetrievedChunk(chunk=c, score=s) for c, s in ranked[:top_k] if s >= min_score]

    def answer_with_citations(self, question: str, top_k: int = 3) -> Tuple[str, List[RetrievedChunk]]:
        """
        Deterministic, dependency-free "answer": returns the single most
        relevant chunk's text (trimmed) as the grounded answer, plus the
        full retrieved set as citations. The agentic orchestrator can
        instead hand these chunks to a real LLM as context for a fluent
        answer — see `agentic_orchestrator.py`'s `_llm_answer_with_context`.
        """
        retrieved = self.query(question, top_k=top_k)
        if not retrieved:
            return (
                "I couldn't find anything in the OmniEdge knowledge base relevant to that question.",
                [],
            )
        top = retrieved[0].chunk.text
        snippet = top if len(top) <= 500 else top[:497] + "..."
        return snippet, retrieved
