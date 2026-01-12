from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    # data
    pdf_dir: Path = Path("data/pdfs")
    questions_path: Path = Path("data/questions/questions.json")

    # cache / outputs
    index_dir: Path = Path("indexes/faiss_v1")
    submissions_dir: Path = Path("submissions")

    # embeddings / retrieval
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "8"))
    fetch_k: int = int(os.getenv("FETCH_K", "25"))

    # rerank (optional)
    rerank: bool = os.getenv("RERANK", "false").lower() in {"1","true","yes"}
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # generator
    generator: str = os.getenv("GENERATOR", "heuristic")  # gemini | openai | heuristic

    # submission identity
    team_email: str = os.getenv("TEAM_EMAIL", "test@rag-tat.com")
    submission_name: str = os.getenv("SUBMISSION_NAME", "surname_v0")

    # OpenAI-compatible
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Gemini
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # behavior
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
