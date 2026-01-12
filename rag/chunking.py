from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

@dataclass(frozen=True)
class Chunk:
    pdf_sha1: str
    page_index: int
    chunk_index: int
    text: str

def chunk_page_text(pdf_sha1: str, page_index: int, text: str, *, chunk_chars: int, overlap: int) -> Iterator[Chunk]:
    if not text.strip():
        return
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk_text = text[start:end].strip()
        if chunk_text:
            yield Chunk(pdf_sha1=pdf_sha1, page_index=page_index, chunk_index=idx, text=chunk_text)
            idx += 1
        if end == n:
            break
        start = max(0, end - overlap)
