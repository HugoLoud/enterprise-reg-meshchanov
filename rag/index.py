from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

from .pdf import iter_all_pages
from .chunking import chunk_page_text, Chunk

@dataclass
class IndexArtifacts:
    faiss_index: faiss.Index
    chunks: List[Chunk]
    embed_model_name: str

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def build_or_load_index(
    *,
    pdf_dir: Path,
    index_dir: Path,
    embed_model: str,
    chunk_chars: int,
    chunk_overlap: int
) -> IndexArtifacts:
    _ensure_dir(index_dir)
    idx_path = index_dir / "index.faiss"
    meta_path = index_dir / "chunks.jsonl"
    info_path = index_dir / "info.json"

    if idx_path.exists() and meta_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("embed_model") == embed_model and info.get("chunk_chars")==chunk_chars and info.get("chunk_overlap")==chunk_overlap:
            faiss_index = faiss.read_index(str(idx_path))
            chunks: List[Chunk] = []
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    chunks.append(Chunk(**obj))
            return IndexArtifacts(faiss_index=faiss_index, chunks=chunks, embed_model_name=embed_model)

    # rebuild
    model = SentenceTransformer(embed_model)

    chunks: List[Chunk] = []
    texts: List[str] = []

    for page in tqdm(list(iter_all_pages(pdf_dir)), desc="Extracting pages"):
        for ch in chunk_page_text(page.pdf_sha1, page.page_index, page.text, chunk_chars=chunk_chars, overlap=chunk_overlap):
            chunks.append(ch)
            texts.append(ch.text)

    # embeddings (normalized for cosine via inner product)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(emb)

    faiss.write_index(faiss_index, str(idx_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")

    info = {"embed_model": embed_model, "chunk_chars": chunk_chars, "chunk_overlap": chunk_overlap, "num_chunks": len(chunks)}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    return IndexArtifacts(faiss_index=faiss_index, chunks=chunks, embed_model_name=embed_model)

def search(
    artifacts: IndexArtifacts,
    query: str,
    *,
    model: SentenceTransformer,
    top_k: int = 8
) -> List[Tuple[int, float]]:
    q = model.encode([query], normalize_embeddings=True)
    D, I = artifacts.faiss_index.search(np.asarray(q, dtype="float32"), top_k)
    results = []
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx == -1:
            continue
        results.append((idx, float(score)))
    return results
