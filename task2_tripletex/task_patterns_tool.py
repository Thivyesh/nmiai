"""Task pattern lookup tool using hybrid BM25 + semantic search."""

import logging
import os
from pathlib import Path

import numpy as np
import requests
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

from task2_tripletex.task_patterns import TASK_PATTERNS

logger = logging.getLogger(__name__)

PATTERN_INDEX_PATH = Path(__file__).parent / "pattern_index.npz"
_pattern_index: dict | None = None
_bm25: BM25Okapi | None = None
_sections: list[tuple[str, str]] | None = None


def _build_pattern_sections() -> list[tuple[str, str]]:
    """Split TASK_PATTERNS into (title, content) sections."""
    global _sections
    if _sections is not None:
        return _sections
    sections = []
    current_title = ""
    current_lines = []
    for line in TASK_PATTERNS.split("\n"):
        if line.startswith("## "):
            if current_title and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line.replace("## ", "").strip()
            current_lines = [line]
        elif current_title:
            current_lines.append(line)
    if current_title and current_lines:
        sections.append((current_title, "\n".join(current_lines)))
    _sections = sections
    return sections


def _get_bm25() -> tuple[BM25Okapi, list[tuple[str, str]]]:
    """Get or build BM25 index over pattern sections."""
    global _bm25
    sections = _build_pattern_sections()
    if _bm25 is not None:
        return _bm25, sections
    # Tokenize: title + content as searchable text
    corpus = []
    for title, content in sections:
        text = f"{title} {content}".lower()
        corpus.append(text.split())
    _bm25 = BM25Okapi(corpus)
    return _bm25, sections


def _load_pattern_index() -> dict:
    global _pattern_index
    if _pattern_index:
        return _pattern_index
    if not PATTERN_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Pattern index not found. Run: uv run python -m task2_tripletex.build_index"
        )
    data = np.load(PATTERN_INDEX_PATH, allow_pickle=False)
    _pattern_index = {
        "embeddings": data["embeddings"],
        "titles": data["titles"].tolist(),
        "contents": data["contents"].tolist(),
    }
    return _pattern_index


def _hybrid_search(query: str, top_k: int = 3) -> list[tuple[str, str, float]]:
    """Hybrid search: BM25 keyword + semantic similarity, fused with RRF."""
    sections = _build_pattern_sections()

    # --- BM25 ---
    bm25, _ = _get_bm25()
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # --- Semantic ---
    semantic_ranking = []
    try:
        index = _load_pattern_index()
        embeddings = index["embeddings"]
        tei_url = os.getenv("TEI_URL", "http://localhost:8080")
        resp = requests.post(
            f"{tei_url}/embed", json={"inputs": [query[:2000]]}, timeout=10
        )
        resp.raise_for_status()
        query_emb = np.array(resp.json()[0])
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1, norms)
        similarities = np.dot(embeddings, query_emb) / norms

        # Map embedding indices back to section indices (dedup by title)
        section_best_sim = {}
        for idx in range(len(similarities)):
            title = index["titles"][idx]
            # Find matching section index
            for si, (st, _) in enumerate(sections):
                if st == title and si not in section_best_sim:
                    section_best_sim[si] = similarities[idx]
                    break
                elif st == title and similarities[idx] > section_best_sim.get(si, 0):
                    section_best_sim[si] = similarities[idx]

        # Sort by similarity
        semantic_ranking = sorted(section_best_sim.keys(), key=lambda x: section_best_sim[x], reverse=True)
    except Exception as e:
        logger.warning("Semantic search failed in hybrid, using BM25 only: %s", e)

    # --- Reciprocal Rank Fusion (RRF) ---
    k = 60  # RRF constant
    rrf_scores = {}
    for rank, idx in enumerate(bm25_ranking):
        if idx < len(sections):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(semantic_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sort by fused score
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:top_k]:
        title, content = sections[idx]
        results.append((title, content, score))

    return results


@tool
def lookup_task_pattern(task_description: str) -> str:
    """Look up the correct accounting workflow pattern for a task.

    CALL THIS FIRST. Returns workflow, risks, and field gotchas.

    Args:
        task_description: The task prompt or a brief description, e.g.
            "create employee with admin role", "opprett faktura",
            "nota de gastos de viaje", "kreditnota".

    Returns:
        Matching task patterns with workflow and scoring guidance.
    """
    sections = _build_pattern_sections()
    general = next((c for t, c in sections if "UNIVERSAL" in t), "")

    results = _hybrid_search(task_description, top_k=3)

    parts = [general] if general else []
    for title, content, score in results:
        if "UNIVERSAL" in title or "MULTILINGUAL" in title:
            continue
        parts.append(f"[score: {score:.4f}]\n{content}")

    if len(parts) > (1 if general else 0):
        return "\n\n---\n\n".join(parts)

    return general or "No matching task pattern found. Use lookup_api_docs instead."
