"""Tripletex developer docs search using hybrid BM25 + semantic."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import requests
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

DEVDOCS_INDEX_PATH = Path(__file__).parent / "devdocs_index.npz"
DEVDOCS_PATH = Path(__file__).parent / "developer_docs.json"
_devdocs_index: dict | None = None
_bm25_docs: BM25Okapi | None = None
_docs_list: list[dict] | None = None


def _load_devdocs_index() -> dict:
    global _devdocs_index
    if _devdocs_index:
        return _devdocs_index
    if not DEVDOCS_INDEX_PATH.exists():
        raise FileNotFoundError("Developer docs index not found.")
    data = np.load(DEVDOCS_INDEX_PATH, allow_pickle=False)
    _devdocs_index = {
        "embeddings": data["embeddings"],
        "titles": data["titles"].tolist(),
        "contents": data["contents"].tolist(),
    }
    return _devdocs_index


def _get_bm25_docs() -> tuple[BM25Okapi, list[dict]]:
    global _bm25_docs, _docs_list
    if _bm25_docs is not None and _docs_list is not None:
        return _bm25_docs, _docs_list
    index = _load_devdocs_index()
    docs = [{"title": t, "content": c} for t, c in zip(index["titles"], index["contents"])]
    corpus = [(d["title"] + " " + d["content"]).lower().split() for d in docs]
    _bm25_docs = BM25Okapi(corpus)
    _docs_list = docs
    return _bm25_docs, docs


@tool
def search_tripletex_docs(query: str) -> str:
    """Search official Tripletex developer docs for guides and troubleshooting.

    Args:
        query: What you need help with, e.g. "voucher posting customer required",
               "invoice VAT error", "department required for employee".

    Returns:
        Relevant documentation excerpts.
    """
    try:
        index = _load_devdocs_index()
    except FileNotFoundError:
        return "Developer docs index not available."

    bm25, docs = _get_bm25_docs()

    # BM25
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranking = list(np.argsort(bm25_scores)[::-1])

    # Semantic
    semantic_ranking = []
    try:
        embeddings = index["embeddings"]
        tei_url = os.getenv("TEI_URL", "http://localhost:8080")
        resp = requests.post(f"{tei_url}/embed", json={"inputs": [query[:2000]]}, timeout=10)
        resp.raise_for_status()
        query_emb = np.array(resp.json()[0])
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1, norms)
        similarities = np.dot(embeddings, query_emb) / norms
        semantic_ranking = list(np.argsort(similarities)[::-1])
    except Exception as e:
        logger.warning("Semantic search failed, BM25 only: %s", e)

    # RRF fusion
    k = 60
    rrf_scores = {}
    for rank, idx in enumerate(bm25_ranking[:20]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(semantic_ranking[:20]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:3]:
        if idx < len(docs):
            title = docs[idx]["title"]
            content = docs[idx]["content"]
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            results.append(f"## {title}\n\n{content}")

    if not results:
        return f"No relevant developer docs found for '{query}'."

    return "\n\n---\n\n".join(results)
