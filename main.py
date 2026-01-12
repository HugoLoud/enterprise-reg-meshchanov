from __future__ import annotations

import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.config import Settings
from rag.index import build_or_load_index
from rag.retrieval import retrieve, build_context
from rag.llm import GeminiClient, OpenAICompatibleClient, HeuristicLLM
from rag.answering import build_prompt, normalize_by_kind, default_heuristic_answer, pick_references
from rag.submission import Submission, Answer, SourceReference

def load_questions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # expected: list[{text, kind}]
    out = []
    for q in data:
        out.append({"text": q["text"], "kind": q.get("kind","name")})
    return out

def get_llm(settings: Settings):
    if settings.generator == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GENERATOR=gemini but GEMINI_API_KEY is not set")
        return GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if settings.generator == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("GENERATOR=openai but OPENAI_API_KEY is not set")
        return OpenAICompatibleClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.openai_model)
    return HeuristicLLM()

def main():
    settings = Settings()
    print("GENERATOR =", settings.generator)
    settings.submissions_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(settings.questions_path)

    artifacts = build_or_load_index(
        pdf_dir=settings.pdf_dir,
        index_dir=settings.index_dir,
        embed_model=settings.embed_model,
        chunk_chars=settings.chunk_chars,
        chunk_overlap=settings.chunk_overlap,
    )

    embedder = SentenceTransformer(settings.embed_model)
    reranker = CrossEncoder(settings.rerank_model) if settings.rerank else None
    llm = get_llm(settings)

    answers = []
    for q in questions:
        qtext = q["text"]
        kind = q["kind"]

        retrieved = retrieve(
            artifacts,
            query=qtext,
            embedder=embedder,
            top_k=settings.top_k,
            fetch_k=settings.fetch_k,
            rerank=settings.rerank,
            reranker=reranker,
        )
        context = build_context(retrieved, max_chars=settings.max_context_chars)
        if kind == "number":
            q = qtext.lower()
            keywords = [w for w in q.replace('"', '').replace("?", "").split() if len(w) >= 5]
            lines = []
            for chunk in retrieved[: min(len(retrieved), 12)]:
                txt = chunk["text"] if isinstance(chunk, dict) else getattr(chunk, "text", "")
                for sent in txt.split("."):
                    s = sent.strip()
                    if not s:
                        continue
                    sl = s.lower()
                    if any(k in sl for k in keywords):
                        lines.append(s)
                if len(" ".join(lines)) > settings.max_context_chars:
                    break
            if lines:
                context = ". ".join(lines)[: settings.max_context_chars]

        if settings.generator == "heuristic":
            value = default_heuristic_answer(kind, retrieved)
        else:
            prompt = build_prompt(qtext, kind, context)
            raw = llm.generate(prompt).text
            value = normalize_by_kind(kind, raw, qtext)
            

        refs = pick_references(retrieved, max_refs=2)
        # ===== FINAL SAFETY (ABSOLUTE MUST) =====
        if kind == "boolean":
            # по правилам: если нет упоминания -> False
            if value in ("N/A", "NA", None, ""):
                value = False
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("yes", "true", "1"):
                    value = True
                elif v in ("no", "false", "0"):
                    value = False
                else:
                    value = False

        if kind == "number":
            from rag.answering import sanitize_number
            value = sanitize_number(value, qtext)
        # =======================================

        ans = Answer(value=value, references=[SourceReference(**r) for r in refs], question_text=qtext, kind=kind)
        answers.append(ans)

    submission = Submission.model_validate({
        "email": settings.team_email,
        "submission_name": settings.submission_name,
        "answers": [a.model_dump() for a in answers],
    })

    out_path = settings.submissions_dir / f"submission_{settings.submission_name}.json"
    out_path.write_text(submission.model_dump_json(indent=2, by_alias=False), encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
