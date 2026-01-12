from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder

from .index import IndexArtifacts, search
from .chunking import Chunk

@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float

def retrieve(
    artifacts: IndexArtifacts,
    *,
    query: str,
    embedder: SentenceTransformer,
    top_k: int,
    fetch_k: int,
    rerank: bool = False,
    reranker: CrossEncoder | None = None
) -> List[Retrieved]:
    # first-stage dense retrieval
    first = search(artifacts, query, model=embedder, top_k=fetch_k)
    candidates: List[Retrieved] = [Retrieved(chunk=artifacts.chunks[i], score=s) for i, s in first]

    if not rerank or reranker is None or not candidates:
        return candidates[:top_k]

    pairs = [(query, r.chunk.text) for r in candidates]
    rr_scores = reranker.predict(pairs)
    # sort by rerank score desc
    reranked = sorted(
        [Retrieved(chunk=r.chunk, score=float(rs)) for r, rs in zip(candidates, rr_scores)],
        key=lambda x: x.score,
        reverse=True,
    )
    return reranked[:top_k]

def build_context(retrieved: List[Retrieved], *, max_chars: int) -> str:
    parts = []
    total = 0
    for r in retrieved:
        header = f"[pdf_sha1={r.chunk.pdf_sha1} page_index={r.chunk.page_index}]\n"
        txt = r.chunk.text.strip()
        block = header + txt + "\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)
