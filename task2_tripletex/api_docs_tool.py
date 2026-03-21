"""Dynamic API documentation tool using hybrid BM25 + semantic search."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import requests
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_spec: dict | None = None
_index: dict | None = None
_bm25_api: BM25Okapi | None = None
_api_entries: list[tuple[str, str, str]] | None = None  # (path, method, summary)

SPEC_PATH = Path(__file__).parent / "openapi_cache.json"
INDEX_PATH = Path(__file__).parent / "search_index.npz"

# Well-known entity types — reference by {"id": N}, don't expand fields
_REFERENCE_ONLY = {
    "Customer", "Employee", "Department", "Project", "Product",
    "Currency", "VatType", "Account", "Contact", "Supplier",
    "Company", "Division", "Municipality", "Country",
    "CustomerCategory", "DiscountGroup", "ProductUnit",
    "Voucher", "Invoice", "Order", "Document",
    "TravelExpense", "TravelPaymentType", "TravelCostCategory",
    "OccupationCode", "Asset",
}


def _get_spec() -> dict:
    global _spec
    if _spec:
        return _spec
    if SPEC_PATH.exists():
        with open(SPEC_PATH) as f:
            _spec = json.load(f)
    else:
        resp = requests.get(
            "https://kkpqfuj-amager.tripletex.dev/v2/openapi.json", timeout=30
        )
        resp.raise_for_status()
        _spec = resp.json()
        with open(SPEC_PATH, "w") as f:
            json.dump(_spec, f)
    return _spec


def _load_index() -> dict:
    global _index
    if _index:
        return _index
    if not INDEX_PATH.exists():
        raise FileNotFoundError("Search index not found. Run: uv run python -m task2_tripletex.build_index")
    data = np.load(INDEX_PATH, allow_pickle=False)
    _index = {
        "embeddings": data["embeddings"],
        "paths": data["paths"].tolist(),
        "methods": data["methods"].tolist(),
        "summaries": data["summaries"].tolist(),
    }
    return _index


def _get_bm25_api() -> tuple[BM25Okapi, list[tuple[str, str, str]]]:
    """Build BM25 index over API endpoints."""
    global _bm25_api, _api_entries
    if _bm25_api is not None and _api_entries is not None:
        return _bm25_api, _api_entries
    spec = _get_spec()
    entries = []
    corpus = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method not in ("get", "post", "put", "delete"):
                continue
            summary = details.get("summary", "")
            text = f"{method} {path} {summary}".lower()
            entries.append((path, method, summary))
            corpus.append(text.split())
    _bm25_api = BM25Okapi(corpus)
    _api_entries = entries
    return _bm25_api, entries


def _resolve_ref(ref: str, spec: dict) -> dict:
    name = ref.split("/")[-1]
    return spec.get("components", {}).get("schemas", {}).get(name, {})


def _get_schema_fields(schema: dict, spec: dict, depth: int = 0) -> list[str]:
    if depth > 2:
        return []
    fields = []
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for name, prop in props.items():
        if name in ("id", "version", "changes", "url"):
            continue
        if prop.get("readOnly"):
            continue
        typ = prop.get("type", "")
        ref = prop.get("$ref", "")
        desc = prop.get("description", "")[:100]
        enum = prop.get("enum", [])
        if ref:
            ref_name = ref.split("/")[-1]
            if ref_name in _REFERENCE_ONLY:
                typ = f'object({ref_name}) — use {{"id": N}}'
            else:
                typ = f"object({ref_name})"
                if depth == 0:
                    sub_schema = _resolve_ref(ref, spec)
                    sub_fields = _get_schema_fields(sub_schema, spec, depth + 1)
                    if sub_fields:
                        typ += "\n" + "\n".join(f"    {sf}" for sf in sub_fields)
        if prop.get("items", {}).get("$ref"):
            item_name = prop["items"]["$ref"].split("/")[-1]
            if item_name in _REFERENCE_ONLY:
                typ = f'array[{item_name}] — use [{{"id": N}}]'
            else:
                typ = f"array[{item_name}]"
                if depth == 0:
                    item_schema = _resolve_ref(prop["items"]["$ref"], spec)
                    item_fields = _get_schema_fields(item_schema, spec, depth + 1)
                    if item_fields:
                        typ += "\n" + "\n".join(f"    {sf}" for sf in item_fields)
        req = " *REQUIRED*" if name in required else ""
        enum_str = f" enum={enum}" if enum else ""
        fields.append(f"  {name}: {typ}{req}{enum_str} — {desc}")
    return fields


def _format_endpoint(path: str, method: str, details: dict, spec: dict) -> str:
    summary = details.get("summary", "")
    lines = [f"### {method.upper()} {path}", f"{summary}", ""]
    params = details.get("parameters", [])
    query_params = [p for p in params if p.get("in") == "query"]
    if query_params:
        lines.append("Query parameters:")
        for p in query_params:
            req = " *REQUIRED*" if p.get("required") else ""
            schema = p.get("schema", {})
            enum = schema.get("enum", [])
            enum_str = f" enum={enum}" if enum else ""
            desc = p.get("description", "")[:80]
            lines.append(f"  {p['name']}: {schema.get('type', '?')}{req}{enum_str} — {desc}")
        lines.append("")
    body = details.get("requestBody", {})
    content = body.get("content", {})
    for ct, cd in content.items():
        schema_ref = cd.get("schema", {})
        if "$ref" in schema_ref:
            schema = _resolve_ref(schema_ref["$ref"], spec)
            schema_name = schema_ref["$ref"].split("/")[-1]
            fields = _get_schema_fields(schema, spec)
            if fields:
                lines.append(f"Request body ({schema_name}):")
                lines.extend(fields)
                lines.append("")
    return "\n".join(lines)


def _hybrid_search_api(query: str, top_k: int = 8) -> list[tuple[str, str, str, float]]:
    """Hybrid BM25 + semantic search over API endpoints."""
    spec = _get_spec()
    paths = spec.get("paths", {})

    # BM25
    bm25, entries = _get_bm25_api()
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranking = list(np.argsort(bm25_scores)[::-1])

    # Semantic
    semantic_ranking = []
    try:
        index = _load_index()
        tei_url = os.getenv("TEI_URL", "http://localhost:8080")
        resp = requests.post(f"{tei_url}/embed", json={"inputs": [query[:2000]]}, timeout=10)
        resp.raise_for_status()
        query_emb = np.array(resp.json()[0])
        embeddings = index["embeddings"]
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1, norms)
        similarities = np.dot(embeddings, query_emb) / norms
        semantic_ranking = list(np.argsort(similarities)[::-1])
    except Exception as e:
        logger.warning("Semantic search failed, BM25 only: %s", e)

    # RRF fusion
    k = 60
    rrf_scores = {}
    for rank, idx in enumerate(bm25_ranking[:50]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(semantic_ranking[:50]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:top_k]:
        if idx < len(entries):
            path, method, summary = entries[idx]
            results.append((path, method, summary, score))
    return results


@tool
def lookup_api_docs(search: str) -> str:
    """Look up Tripletex API documentation for specific endpoints.

    Uses hybrid BM25 + semantic search. Works with endpoint paths, keywords,
    and non-English terms.

    Args:
        search: What to search for. Examples: "travelExpense/cost", "employee",
                "invoice/:payment", "kontoadministrator", "nota de gastos".

    Returns:
        Formatted API documentation for matching endpoints.
    """
    spec = _get_spec()
    paths = spec.get("paths", {})

    results = _hybrid_search_api(search, top_k=8)

    if not results:
        return f"No endpoints found for '{search}'."

    matches = []
    for path, method, summary, score in results:
        details = paths.get(path, {}).get(method, {})
        if details:
            matches.append((path, method, details, score))

    if not matches:
        return f"No endpoints found for '{search}'."

    output = [f"# API docs: '{search}' ({len(matches)} results)\n"]
    for path, method, details, score in matches:
        output.append(_format_endpoint(path, method, details, spec))
        output.append("---\n")
    return "\n".join(output)


# Pre-download spec if not cached
if not SPEC_PATH.exists():
    try:
        _get_spec()
    except Exception:
        pass
