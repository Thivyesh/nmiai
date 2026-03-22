"""Focused tools for the researcher: workflow steps and payload templates."""

import json
import logging
import os

import numpy as np
import requests
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

from task2_tripletex.payload_templates import PAYLOAD_TEMPLATES
from task2_tripletex.task_patterns import TASK_PATTERNS

logger = logging.getLogger(__name__)

TEI_URL = os.getenv("TEI_URL", "http://localhost:8080")

# --- Task workflow search (keyword-based) ---

def _build_sections() -> dict[str, str]:
    """Parse task patterns into sections by title."""
    sections = {}
    current_title = ""
    current_lines = []
    for line in TASK_PATTERNS.split("\n"):
        if line.startswith("## "):
            if current_title:
                sections[current_title] = "\n".join(current_lines)
            current_title = line.replace("## ", "").strip()
            current_lines = [line]
        elif current_title:
            current_lines.append(line)
    if current_title:
        sections[current_title] = "\n".join(current_lines)
    return sections


_SECTIONS = _build_sections()


@tool
def get_task_workflow(task_description: str) -> str:
    """Get the workflow steps for a task type. Returns ONLY the ordered steps.

    Call this FIRST to understand what endpoints to call and in what order.

    Args:
        task_description: Brief English description, e.g. "create invoice",
            "travel expense", "voucher with dimensions", "salary payroll".

    Returns:
        The workflow steps and prerequisites for this task type.
    """
    search = task_description.lower()
    best_match = None
    best_score = 0

    for title, content in _SECTIONS.items():
        title_lower = title.lower()
        content_lower = content.lower()[:500]
        score = 0
        for word in search.split():
            if len(word) < 3:
                continue
            if word in title_lower:
                score += 3
            if word in content_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_match = (title, content)

    if not best_match or best_score < 2:
        return f"No workflow found for '{task_description}'. Available: " + ", ".join(
            t for t in _SECTIONS if "UNIVERSAL" not in t and "MULTILINGUAL" not in t
        )

    # Also include universal prerequisites
    universal = _SECTIONS.get("UNIVERSAL PREREQUISITES (check for EVERY task)", "")

    return f"{universal}\n\n---\n\n{best_match[1]}"


# --- Payload template search (hybrid BM25 + semantic) ---

# Endpoint path corrections (same as tripletex_get)
_PATH_CORRECTIONS = {
    "/account": "/ledger/account",
    "/voucherType": "/ledger/voucherType",
    "/voucher": "/ledger/voucher",
    "/vatType": "/ledger/vatType",
    "/paymentType": "/invoice/paymentType",
    "/occupationCode": "/employee/employment/occupationCode",
    "/employment": "/employee/employment",
    "/cost": "/travelExpense/cost",
    "/project/activity": "/project/projectActivity",
    "/activity": "/project/projectActivity",
}

# Build BM25 index over templates
_template_keys: list[str] = list(PAYLOAD_TEMPLATES.keys())
_template_corpus: list[list[str]] = []
for _k in _template_keys:
    _t = PAYLOAD_TEMPLATES[_k]
    _text = f"{_k} {_t['description']} {_t.get('notes', '')}".lower()
    _template_corpus.append(_text.split())
_template_bm25 = BM25Okapi(_template_corpus) if _template_corpus else None

# Semantic index (lazy-built)
_template_embeddings: np.ndarray | None = None


def _build_template_embeddings():
    """Build semantic embeddings for template search texts."""
    global _template_embeddings
    if _template_embeddings is not None:
        return

    texts = []
    for k in _template_keys:
        t = PAYLOAD_TEMPLATES[k]
        texts.append(f"{k} {t['description']} {t.get('notes', '')}"[:500])

    try:
        resp = requests.post(f"{TEI_URL}/embed", json={"inputs": texts}, timeout=5)
        resp.raise_for_status()
        _template_embeddings = np.array(resp.json())
        logger.info("Built template semantic index: %d templates", len(texts))
    except Exception:
        pass


def _hybrid_template_search(query: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Search templates using hybrid BM25 + semantic, return (key, score) pairs."""
    results = {}

    # BM25
    if _template_bm25:
        tokens = query.lower().split()
        scores = _template_bm25.get_scores(tokens)
        for i, score in enumerate(scores):
            if score > 0:
                results[_template_keys[i]] = score

    # Semantic
    _build_template_embeddings()
    if _template_embeddings is not None:
        try:
            resp = requests.post(f"{TEI_URL}/embed", json={"inputs": [query[:500]]}, timeout=5)
            resp.raise_for_status()
            query_emb = np.array(resp.json()[0])
            norms = np.linalg.norm(_template_embeddings, axis=1) * np.linalg.norm(query_emb)
            norms = np.where(norms == 0, 1, norms)
            sims = np.dot(_template_embeddings, query_emb) / norms
            for i, sim in enumerate(sims):
                if sim > 0.3:
                    key = _template_keys[i]
                    # RRF-style fusion: add semantic score (normalized)
                    results[key] = results.get(key, 0) + sim * 5
        except Exception:
            pass

    # Sort by score descending
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


@tool
def get_payload_template(endpoint: str) -> str:
    """Get the EXACT verified JSON payload template for an API endpoint.

    Call this for EACH endpoint you need to call. Copy the template and
    substitute the placeholder values with real data. Do NOT invent field names.

    Args:
        endpoint: The API endpoint, e.g. "POST /customer", "POST /ledger/voucher",
            "PUT /invoice/{id}/:payment", "POST /travelExpense/cost".

    Returns:
        The exact JSON template with placeholder values and field notes.
    """
    # Apply path corrections
    parts = endpoint.split(" ", 1)
    method = parts[0] if len(parts) > 1 else ""
    path = parts[-1]
    if path in _PATH_CORRECTIONS:
        path = _PATH_CORRECTIONS[path]
        endpoint = f"{method} {path}" if method else path

    # Try exact match first
    template = PAYLOAD_TEMPLATES.get(endpoint)
    if not template:
        # Case-insensitive exact
        for key in _template_keys:
            if key.lower() == endpoint.lower():
                template = PAYLOAD_TEMPLATES[key]
                endpoint = key
                break

    # Hybrid BM25 + semantic search
    if not template:
        ranked = _hybrid_template_search(endpoint)
        if ranked:
            best_key, best_score = ranked[0]
            if best_score > 0.5:
                template = PAYLOAD_TEMPLATES[best_key]
                endpoint = best_key

    if not template:
        # Fall back to API docs
        from task2_tripletex.api_docs_tool import lookup_api_docs
        docs = lookup_api_docs.invoke({"search": path})
        if docs and len(docs) > 50:
            return f"No verified template for '{endpoint}'. Schema from API docs:\n\n{docs}"
        available = "\n".join(f"  - {k}" for k in PAYLOAD_TEMPLATES)
        return f"No template for '{endpoint}'. Available templates:\n{available}"

    payload = template["payload"]
    if isinstance(payload, dict):
        payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        payload_str = str(payload)

    result = f"## {endpoint}\n{template['description']}\n\n"

    if "url_format" in template:
        result += f"URL: {template['url_format']}\n\n"

    result += f"Payload:\n```json\n{payload_str}\n```\n\n"
    result += f"Notes: {template['notes']}"

    return result
