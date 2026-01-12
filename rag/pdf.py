from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import fitz  # PyMuPDF
from typing import Iterable, Iterator

@dataclass(frozen=True)
class Page:
    pdf_path: Path
    pdf_sha1: str
    page_index: int
    text: str

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_pdf_pages(pdf_path: Path) -> Iterator[Page]:
    # If filename looks like sha1, trust it to save time; otherwise compute.
    name = pdf_path.stem.lower()
    if len(name) == 40 and all(c in "0123456789abcdef" for c in name):
        pdf_sha1 = name
    else:
        pdf_sha1 = sha1_file(pdf_path)

    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            # text extraction: standard mode
            text = page.get_text("text") or ""
            # normalize whitespace slightly
            text = " ".join(text.replace("\u00a0", " ").split())
            yield Page(pdf_path=pdf_path, pdf_sha1=pdf_sha1, page_index=i, text=text)
    finally:
        doc.close()

def iter_all_pages(pdf_dir: Path) -> Iterable[Page]:
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        yield from iter_pdf_pages(pdf_path)
