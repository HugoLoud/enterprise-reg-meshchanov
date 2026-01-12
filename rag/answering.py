from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Tuple

from .retrieval import Retrieved

BOOL_TRUE = {"true","yes","y","1"}
BOOL_FALSE = {"false","no","n","0"}

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_boolean(text: str) -> bool:
    t = _clean(text).lower()

    # явные true
    if any(x in t for x in ("yes", "true")):
        return True

    # явные false
    if any(x in t for x in ("no", "false")):
        return False

    # если написано N/A, NA, пусто — по правилам возвращаем False
    return False

def normalize_number(text: str) -> Any:
    t = _clean(text)
    if re.search(r"(?i)\bn\s*/\s*a\b|\bn/a\b|\bna\b", t):
        return "N/A"
    # find first number-like token
    m = re.search(r"[-+]?\d{1,3}(?:[\d,\s]\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", t)
    if not m:
        return "N/A"
    num = m.group(0)
    try:
        num_int = int(num.replace(",", "").replace(" ", ""))
        if 2000 <= num_int <= 2035:
            return "N/A"
    except Exception:
        pass
    num = num.replace(" ", "").replace(",", "")
    try:
        if "." in num:
            return float(num)
        return int(num)
    except Exception:
        try:
            return float(num)
        except Exception:
            return "N/A"
        
def sanitize_number(value: Any, question_text: str) -> Any:
    if value == "N/A":
        return "N/A"

    try:
        v = float(value)
    except Exception:
        return "N/A"

    qt = question_text.lower()

    # люди, патенты, объекты
    if any(k in qt for k in ["employee", "headcount", "staff", "patent", "store", "facility", "clinic", "hotel", "site"]):
        if v < 0 or v > 10_000_000:
            return "N/A"

    # проценты и маржи
    if any(k in qt for k in ["margin", "ratio", "%"]):
        if v < -1000 or v > 1000:
            return "N/A"

    # целые числа приводим к int
    if v.is_integer():
        return int(v)

    return v

def normalize_name(text: str) -> str:
    t = _clean(text)
    if not t or re.search(r"(?i)\bn/a\b|\bna\b", t):
        return "N/A"
    # keep first line/sentence
    t = t.split("\n")[0]
    t = re.split(r"[\.;]", t)[0]
    return t.strip().strip('"').strip()

def normalize_names(text: str) -> List[str] | str:
    t = _clean(text)
    if not t or re.search(r"(?i)\bn/a\b|\bna\b", t):
        return "N/A"
    # allow JSON-like list
    if t.startswith("[") and t.endswith("]"):
        # very light parsing
        items = re.findall(r"\"([^\"]+)\"", t)
        if items:
            return [i.strip() for i in items if i.strip()]
    # otherwise split by comma/and
    parts = re.split(r",|;|\band\b|\&", t, flags=re.IGNORECASE)
    names = [p.strip().strip('"') for p in parts if p.strip()]
    return names[:10] if names else "N/A"

def build_prompt(question_text: str, kind: str, context: str) -> str:
    # concise instruction; emphasize strict output format
    if kind == "number":
        fmt = "Return ONLY a plain number (no commas, no currency symbols). If unanswerable, return N/A."
    elif kind == "boolean":
        fmt = "Return ONLY yes or no (lowercase). If the context does not mention it, return no."
    elif kind == "name":
        fmt = "Return ONLY the single name as a string. If unanswerable, return N/A."
    elif kind == "names":
        fmt = "Return ONLY a comma-separated list of names, or N/A."
    else:
        fmt = "Return ONLY the answer."

    return f"""You will answer a question using ONLY the provided context from annual reports.

Question kind: {kind}
Question: {question_text}

Output rules: {fmt}
Do not add explanations.

Context:
{context}
"""

def normalize_by_kind(kind: str, raw: str, question_text: str):
    if kind == "number":
        num = normalize_number(raw)
        return sanitize_number(num, question_text)

    if kind == "boolean":
        return normalize_boolean(raw)

    if kind == "name":
        return normalize_name(raw)

    if kind == "names":
        return normalize_names(raw)

    return normalize_name(raw)

def default_heuristic_answer(kind: str, retrieved: List[Retrieved]) -> Any:
    if not retrieved:
        return "N/A"

    texts = [r.chunk.text for r in retrieved[:5]]

    if kind == "boolean":
        # если в тексте вообще есть что-то похожее на утверждение
        joined = " ".join(texts).lower()
        if any(k in joined for k in ["yes", "we did", "the company", "has", "completed", "acquired", "merged"]):
            return True
        return False

    if kind == "number":
        # ищем число в предложениях с ключевыми словами
        joined = " ".join(texts)
        sentences = re.split(r"[.\n]", joined)

        # ключевые слова из вопроса (грубая эвристика)
        # часто встречаются в числовых вопросах
        KEYWORDS = [
            "total", "number", "employees", "employee", "headcount",
            "revenue", "assets", "income", "profit", "margin",
            "facilities", "stores", "hotels", "clinics", "centers"
        ]

        candidates = []
        for s in sentences:
            sl = s.lower()
            if not any(k in sl for k in KEYWORDS):
                continue
            m = re.search(r"\b\d{2,9}\b", s.replace(",", ""))
            if m:
                try:
                    candidates.append(int(m.group(0)))
                except Exception:
                    pass

        if candidates:
            # часто правильное число не самое маленькое и не самое большое
            return candidates[0]

        return "N/A"

    if kind == "name":
        # попробуем взять первую сущность-подобную строку
        t = texts[0].strip()
        t = t.split("\n")[0]
        t = re.split(r"[.;]", t)[0]
        return normalize_name(t)

    if kind == "names":
        return normalize_names(" ".join(texts[:2]))

    return "N/A"

def pick_references(retrieved: List[Retrieved], max_refs: int = 2):
    refs = []
    used = set()
    for r in retrieved:
        key = (r.chunk.pdf_sha1, r.chunk.page_index)
        if key in used:
            continue
        used.add(key)
        refs.append({"pdf_sha1": r.chunk.pdf_sha1, "page_index": int(r.chunk.page_index)})
        if len(refs) >= max_refs:
            break
    return refs
