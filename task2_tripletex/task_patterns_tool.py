"""Task pattern lookup tool for the planner agent.

Provides accounting workflow patterns and scoring guidance.
Supports keyword search and semantic search via pre-built embeddings.
"""

import logging
import os
from pathlib import Path

import numpy as np
import requests
from langchain_core.tools import tool

from task2_tripletex.task_patterns import TASK_PATTERNS

logger = logging.getLogger(__name__)

PATTERN_INDEX_PATH = Path(__file__).parent / "pattern_index.npz"
_pattern_index: dict | None = None


def _load_pattern_index() -> dict:
    global _pattern_index
    if _pattern_index:
        return _pattern_index
    if not PATTERN_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Pattern index not found. Run: uv run python -m task2_tripletex.build_index"
        )
    data = np.load(PATTERN_INDEX_PATH, allow_pickle=False)
    _pattern_index = {
        "embeddings": data["embeddings"],
        "titles": data["titles"].tolist(),
        "contents": data["contents"].tolist(),
    }
    logger.info("Loaded pattern index: %d chunks", len(_pattern_index["titles"]))
    return _pattern_index


def _build_pattern_sections() -> list[tuple[str, str]]:
    """Split TASK_PATTERNS into (title, content) sections."""
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
    return sections


def _semantic_search_patterns(query: str, top_k: int = 3) -> list[tuple[str, str, float]]:
    """Search patterns by semantic similarity."""
    index = _load_pattern_index()
    embeddings = index["embeddings"]

    tei_url = os.getenv("TEI_URL", "http://localhost:8080")
    resp = requests.post(f"{tei_url}/embed", json={"inputs": [query[:2000]]}, timeout=10)
    resp.raise_for_status()
    query_emb = np.array(resp.json()[0])

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    norms = np.where(norms == 0, 1, norms)
    similarities = np.dot(embeddings, query_emb) / norms

    # Deduplicate by title (take best score per unique title)
    seen_titles = {}
    top_indices = np.argsort(similarities)[::-1]
    for idx in top_indices:
        title = index["titles"][idx]
        score = float(similarities[idx])
        if title not in seen_titles:
            seen_titles[title] = (title, index["contents"][idx], score)
        if len(seen_titles) >= top_k:
            break

    return list(seen_titles.values())


@tool
def lookup_task_pattern(task_description: str, semantic: bool = True) -> str:
    """Look up the correct accounting workflow pattern for a task.

    CALL THIS FIRST before planning. Returns:
    - What entities need to be created
    - What fields the competition checks
    - The exact API workflow steps
    - Common mistakes to avoid

    Args:
        task_description: The task prompt or a brief description, e.g.
            "create employee with admin role", "opprett faktura",
            "nota de gastos de viaje", "kreditnota".
        semantic: Use semantic search (default true). Better for multilingual
            prompts. Set false for keyword-only search.

    Returns:
        Matching task patterns with workflow and scoring guidance.
    """
    # Always include general rules
    sections = _build_pattern_sections()
    general = next((c for t, c in sections if "GENERAL" in t), "")

    if semantic:
        try:
            results = _semantic_search_patterns(task_description, top_k=3)
            if results:
                parts = [general] if general else []
                for title, content, score in results:
                    if score >= 0.3 and "GENERAL" not in title:
                        parts.append(f"[relevance: {score:.2f}]\n{content}")
                if len(parts) > (1 if general else 0):
                    return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.warning("Semantic pattern search failed: %s", e)

    # Fallback: keyword search
    search_lower = task_description.lower()
    scored = []
    for title, content in sections:
        if "GENERAL" in title or "MULTILINGUAL" in title:
            continue
        score = 0
        title_lower = title.lower()
        content_lower = content.lower()
        for word in search_lower.split():
            if len(word) < 3:
                continue
            if word in title_lower:
                score += 3
            if word in content_lower:
                score += 1
        if score > 0:
            scored.append((score, title, content))

    scored.sort(reverse=True)
    parts = [general] if general else []
    for score, title, content in scored[:2]:
        if score >= 2:
            parts.append(content)

    if len(parts) > (1 if general else 0):
        return "\n\n---\n\n".join(parts)

    return general or "No matching task pattern found. Use lookup_api_docs instead."
