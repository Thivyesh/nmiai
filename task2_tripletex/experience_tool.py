"""Search past task execution history using hybrid BM25 + semantic search."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import requests
from elasticsearch import Elasticsearch
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

ES_URL = "http://localhost:9200"
INDEX = "tripletex-experience"  # Enriched index with fixes
INDEX_FALLBACK = "tripletex-traces"  # Basic index
TEI_URL = os.getenv("TEI_URL", "http://localhost:8080")
TRACE_PATH = Path(__file__).parent / "trace_history.json"

# In-memory semantic index for traces
_trace_embeddings: np.ndarray | None = None
_trace_docs: list[dict] | None = None


def _build_semantic_index():
    """Build semantic embeddings for trace prompts + tool summaries."""
    global _trace_embeddings, _trace_docs
    if _trace_embeddings is not None:
        return

    if not TRACE_PATH.exists():
        return

    with open(TRACE_PATH) as f:
        traces = json.load(f)

    docs = []
    texts = []
    for t in traces:
        prompt = t.get("task_prompt", "")
        if not prompt:
            continue
        endpoints = " ".join(t.get("successful_endpoints", []))
        failed = " ".join(fe.get("endpoint", "") for fe in t.get("failed_endpoints", []))
        text = f"{prompt} {endpoints} {failed}"[:500]
        docs.append(t)
        texts.append(text)

    if not texts:
        return

    # Embed in batches
    all_embs = []
    for i in range(0, len(texts), 8):
        batch = [t[:2000] for t in texts[i:i + 8]]
        try:
            resp = requests.post(f"{TEI_URL}/embed", json={"inputs": batch}, timeout=10)
            resp.raise_for_status()
            all_embs.append(np.array(resp.json()))
        except Exception:
            return

    _trace_embeddings = np.vstack(all_embs)
    _trace_docs = docs
    logger.info("Built semantic index for %d traces", len(docs))


def _semantic_search(query: str, top_k: int = 5) -> list[tuple[dict, float]]:
    """Search traces by semantic similarity."""
    _build_semantic_index()
    if _trace_embeddings is None or _trace_docs is None:
        return []

    try:
        resp = requests.post(f"{TEI_URL}/embed", json={"inputs": [query[:2000]]}, timeout=10)
        resp.raise_for_status()
        query_emb = np.array(resp.json()[0])
    except Exception:
        return []

    norms = np.linalg.norm(_trace_embeddings, axis=1) * np.linalg.norm(query_emb)
    norms = np.where(norms == 0, 1, norms)
    similarities = np.dot(_trace_embeddings, query_emb) / norms

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [((_trace_docs[i], float(similarities[i]))) for i in top_indices]


def _bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """Search traces via Elasticsearch BM25 — uses enriched index."""
    try:
        es = Elasticsearch(ES_URL)
        # Try enriched index first, fallback to basic
        idx = INDEX if es.indices.exists(index=INDEX) else INDEX_FALLBACK
        if not es.indices.exists(index=idx):
            return []
    except Exception:
        return []

    results = es.search(
        index=idx,
        body={
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "task_prompt^2",
                        "successful_calls^2",
                        "failed_calls_with_fixes^3",
                        "correct_templates^2",
                        "tags^2",
                    ],
                    "type": "most_fields",
                    "fuzziness": "AUTO",
                }
            },
        },
    )
    return [hit["_source"] for hit in results.get("hits", {}).get("hits", [])]


def _format_trace(t: dict) -> str:
    """Format a trace for the agent — includes fixes for errors."""
    errors = t.get("total_errors", 0)
    prompt = t.get("task_prompt", "")[:150]

    entry = f"### Past task (errors: {errors})\n"
    entry += f"Task: {prompt}\n"

    # Enriched format (has fixes)
    successful = t.get("successful_calls", "")
    failed_with_fixes = t.get("failed_calls_with_fixes", "")
    correct_templates = t.get("correct_templates", "")

    if successful:
        lines = [l for l in successful.split("\n") if l.strip()][:5]
        entry += f"Worked: {'; '.join(lines)}\n"
    if failed_with_fixes:
        entry += f"Failures & Fixes:\n{failed_with_fixes[:400]}\n"
    if correct_templates:
        entry += f"Correct templates:\n{correct_templates[:300]}\n"

    # Fallback format (basic index)
    if not successful and not failed_with_fixes:
        tool_summary = t.get("tool_summary", "")
        success_lines = [l for l in tool_summary.split("\n") if " OK " in l and "tripletex_" in l]
        failed_lines = [l for l in tool_summary.split("\n") if " ERR " in l]
        if success_lines:
            entry += f"Worked: {'; '.join(success_lines[:5])}\n"
        if failed_lines:
            entry += f"Failed: {'; '.join(failed_lines[:3])}\n"

    return entry


@tool
def search_past_experience(task_description: str) -> str:
    """Search past task executions to learn what worked and what failed.

    Uses hybrid BM25 + semantic search. Works with any language.

    Args:
        task_description: Description of the task in any language.

    Returns:
        Past experiences with what worked (200 OK) and what failed (errors).
    """
    # Hybrid: BM25 + semantic, deduplicate by trace_id
    seen = set()
    results = []

    # BM25 from Elasticsearch
    for t in _bm25_search(task_description, top_k=5):
        tid = t.get("trace_id", "")
        if tid not in seen:
            seen.add(tid)
            results.append(t)

    # Semantic from TEI
    for t, score in _semantic_search(task_description, top_k=5):
        tid = t.get("trace_id", "")
        if tid not in seen and score > 0.3:
            seen.add(tid)
            results.append(t)

    if not results:
        return f"No past experience found for '{task_description}'."

    # Sort: fewer errors first, then by total tools (simpler = better)
    results.sort(key=lambda x: (x.get("total_errors", 99), x.get("total_tool_calls", 99)))

    output = [_format_trace(t) for t in results[:3]]
    return "\n---\n".join(output)
