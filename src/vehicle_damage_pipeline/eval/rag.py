from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_knowledge_documents(root: str | Path) -> list[dict[str, str]]:
    root = Path(root)
    candidates = []
    for pattern in ["README.md", "docs/**/*.md", "reports/*.json", "reports/*.md", "runs/predict/demo/labels/*.txt"]:
        candidates.extend(root.glob(pattern))
    docs = []
    for path in sorted(set(candidates)):
        if path.is_file() and path.stat().st_size < 2_000_000:
            docs.append(
                {
                    "id": path.relative_to(root).as_posix(),
                    "text": path.read_text(encoding="utf-8", errors="ignore"),
                }
            )
    return docs


def tfidf_search(query: str, docs: list[dict[str, str]], top_k: int = 5) -> list[dict[str, object]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []
    doc_terms = [Counter(tokenize(doc["text"])) for doc in docs]
    doc_count = len(docs)
    df: Counter[str] = Counter()
    for terms in doc_terms:
        for term in terms:
            df[term] += 1
    scored = []
    for doc, terms in zip(docs, doc_terms):
        score = 0.0
        for term in query_terms:
            if term not in terms:
                continue
            idf = math.log((doc_count + 1) / (df[term] + 1)) + 1
            score += terms[term] * idf
        if score > 0:
            scored.append({"id": doc["id"], "score": round(score, 4), "text": doc["text"][:800]})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def evaluate_retrieval(docs: list[dict[str, str]]) -> dict[str, object]:
    checks = {
        "model": "YOLO11n-seg",
        "dataset": "CarDD",
        "box_map50": "0.644",
        "mask_map50": "0.638",
        "demo": "scratch",
        "artifacts": "best.pt",
    }
    results = {}
    hits = 0
    for name, query in checks.items():
        retrieved = tfidf_search(query, docs, top_k=5)
        found = any(query.lower() in item["text"].lower() for item in retrieved)
        results[name] = {"query": query, "hit": found, "retrieved": [item["id"] for item in retrieved]}
        hits += int(found)
    return {"recall_at_5": hits / len(checks), "checks": results}
