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
                        "competition_notes^4",
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
    """Format a trace for the agent — structured flow showing what happened."""
    errors = t.get("total_errors", 0)
    prompt = t.get("task_prompt", "")[:150]

    entry = f"### Past task (errors: {errors})\n"
    entry += f"**Task:** {prompt}\n"

    # Competition notes first (most valuable)
    comp_notes = t.get("competition_notes", "")
    if comp_notes:
        entry += f"⚠️ **WARNING:** {comp_notes[:300]}\n"

    # Show the execution flow: what was called, with what payload, what happened
    tool_calls = t.get("tool_calls", [])
    failed_endpoints = t.get("failed_endpoints", [])
    failed_errors = {fe.get("endpoint", ""): fe.get("error", "")[:150] for fe in failed_endpoints}

    if tool_calls:
        entry += "**Execution flow:**\n"
        for tc in tool_calls[:10]:
            name = tc.get("name", "")
            args = tc.get("args", "")[:200]
            # Check if this was a failed call
            if name in failed_errors and any(name == fe.get("endpoint", "") for fe in failed_endpoints):
                entry += f"  ❌ {name}({args})\n"
            elif name.startswith("tripletex_"):
                entry += f"  ✓ {name}({args})\n"
            else:
                entry += f"  → {name}({args})\n"

    if failed_endpoints:
        entry += "**Errors encountered:**\n"
        for fe in failed_endpoints[:3]:
            entry += f"  - {fe.get('endpoint', '')}: {fe.get('error', '')[:200]}\n"

    # Enriched format (has fixes from enrich_traces.py)
    successful = t.get("successful_calls", "")
    failed_with_fixes = t.get("failed_calls_with_fixes", "")
    correct_templates = t.get("correct_templates", "")

    if failed_with_fixes:
        entry += f"**How to fix:**\n{failed_with_fixes[:400]}\n"
    if correct_templates:
        entry += f"**Correct payloads:**\n{correct_templates[:300]}\n"
    if successful and not tool_calls:
        lines = [l for l in successful.split("\n") if l.strip()][:5]
        entry += f"**Worked:** {'; '.join(lines)}\n"

    # Fallback format (basic index)
    if not successful and not failed_with_fixes and not tool_calls:
        tool_summary = t.get("tool_summary", "")
        success_lines = [l for l in tool_summary.split("\n") if " OK " in l and "tripletex_" in l]
        failed_lines = [l for l in tool_summary.split("\n") if " ERR " in l]
        if success_lines:
            entry += f"**Worked:** {'; '.join(success_lines[:5])}\n"
        if failed_lines:
            entry += f"**Failed:** {'; '.join(failed_lines[:3])}\n"

    return entry


@tool
def search_past_experience(task_description: str) -> str:
    """Search past task executions to learn from mistakes and copy successes.

    Returns similar past tasks showing:
    - What API calls worked (copy these approaches)
    - What failed and HOW TO FIX IT (avoid these mistakes)
    - Correct payload templates for failed endpoints

    Query in English for best results.

    Args:
        task_description: English description, e.g. "voucher posting",
            "salary payroll", "invoice payment", "travel expense".

    Returns:
        Past experiences with lessons learned, fixes, and correct templates.
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
