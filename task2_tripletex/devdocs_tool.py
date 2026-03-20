"""Tripletex developer documentation search tool.

Searches the official Tripletex developer docs (developer.tripletex.no)
for workflow guides, FAQs, and best practices. Cached locally.
"""

import logging
import os
from pathlib import Path

import numpy as np
import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DEVDOCS_INDEX_PATH = Path(__file__).parent / "devdocs_index.npz"
_devdocs_index: dict | None = None


def _load_devdocs_index() -> dict:
    global _devdocs_index
    if _devdocs_index:
        return _devdocs_index
    if not DEVDOCS_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Developer docs index not found. Run: uv run python -m task2_tripletex.build_index"
        )
    data = np.load(DEVDOCS_INDEX_PATH, allow_pickle=False)
    _devdocs_index = {
        "embeddings": data["embeddings"],
        "titles": data["titles"].tolist(),
        "contents": data["contents"].tolist(),
    }
    return _devdocs_index


@tool
def search_tripletex_docs(query: str) -> str:
    """Search official Tripletex developer documentation for guides and troubleshooting.

    Use this when you encounter errors or need to understand HOW to use an API
    endpoint correctly. Contains official FAQs, workflow guides, and gotchas
    from developer.tripletex.no.

    Args:
        query: What you're looking for, e.g. "voucher posting customer required",
               "invoice VAT error", "order unit price excluding VAT",
               "department required for employee".

    Returns:
        Relevant documentation excerpts.
    """
    try:
        index = _load_devdocs_index()
    except FileNotFoundError:
        return "Developer docs index not available."

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

    top_indices = np.argsort(similarities)[-3:][::-1]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < 0.3:
            continue
        title = index["titles"][idx]
        content = index["contents"][idx]
        # Truncate long docs
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        results.append(f"## {title} [relevance: {score:.2f}]\n\n{content}")

    if not results:
        return f"No relevant developer docs found for '{query}'."

    return "\n\n---\n\n".join(results)
