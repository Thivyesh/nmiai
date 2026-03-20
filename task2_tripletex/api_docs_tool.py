"""Dynamic API documentation tool for the planner agent.

Parses the Tripletex OpenAPI spec and provides endpoint documentation on demand.
Supports keyword search (default) and optional semantic search via pre-built embeddings.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_spec: dict | None = None
SPEC_PATH = Path(__file__).parent / "openapi_cache.json"
INDEX_PATH = Path(__file__).parent / "search_index.npz"

# Loaded once from disk
_index: dict | None = None


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
        raise FileNotFoundError(
            f"Search index not found at {INDEX_PATH}. Run: uv run python -m task2_tripletex.build_index"
        )
    data = np.load(INDEX_PATH, allow_pickle=False)
    _index = {
        "embeddings": data["embeddings"],
        "paths": data["paths"].tolist(),
        "methods": data["methods"].tolist(),
        "summaries": data["summaries"].tolist(),
    }
    logger.info("Loaded semantic index: %d endpoints", len(_index["paths"]))
    return _index


def _resolve_ref(ref: str, spec: dict) -> dict:
    name = ref.split("/")[-1]
    return spec.get("components", {}).get("schemas", {}).get(name, {})


def _get_schema_fields(schema: dict, spec: dict, depth: int = 0) -> list[str]:
    if depth > 1:
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
            typ = f"object({ref_name})"
            if depth == 0 and ref_name in (
                "TravelDetails", "InternationalId", "Address",
                "DeliveryAddress", "HolidayAllowanceEarned",
            ):
                sub_schema = _resolve_ref(ref, spec)
                sub_fields = _get_schema_fields(sub_schema, spec, depth + 1)
                if sub_fields:
                    typ += " fields: {" + ", ".join(
                        f.strip().split(":")[0] for f in sub_fields[:8]
                    ) + "}"
        if prop.get("items", {}).get("$ref"):
            item_name = prop["items"]["$ref"].split("/")[-1]
            typ = f"array[{item_name}]"
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
            lines.append(
                f"  {p['name']}: {schema.get('type', '?')}{req}{enum_str} — {desc}"
            )
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


def _semantic_search(
    query: str, top_k: int = 10
) -> list[tuple[str, str, str, float]]:
    """Search endpoints by cosine similarity against pre-built index."""
    index = _load_index()
    embeddings = index["embeddings"]

    # Embed query via TEI
    tei_url = os.getenv("TEI_URL", "http://localhost:8080")
    resp = requests.post(
        f"{tei_url}/embed", json={"inputs": [query]}, timeout=10
    )
    resp.raise_for_status()
    query_emb = np.array(resp.json()[0])

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    norms = np.where(norms == 0, 1, norms)
    similarities = np.dot(embeddings, query_emb) / norms

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        path = index["paths"][idx]
        method = index["methods"][idx]
        summary = index["summaries"][idx]
        results.append((path, method, summary, float(similarities[idx])))
    return results


@tool
def lookup_api_docs(search: str, semantic: bool = False) -> str:
    """Look up Tripletex API documentation for specific endpoints.

    Args:
        search: Resource path or keyword to search for.
                Examples: "travelExpense/cost", "employee", "invoice/:payment".
        semantic: Set true for meaning-based search. Use when searching with
                  Norwegian/Spanish/French terms or when keyword search fails.
                  Examples: "kontoadministrator", "kreditnota", "nota de gastos".

    Returns:
        Formatted API documentation for matching endpoints.
    """
    spec = _get_spec()
    paths = spec.get("paths", {})

    if semantic:
        try:
            results = _semantic_search(search, top_k=8)
            matches = []
            for path, method, summary, score in results:
                if score < 0.3:
                    continue
                details = paths.get(path, {}).get(method, {})
                if details:
                    matches.append((path, method, details, score))

            if not matches:
                return f"No relevant endpoints for '{search}'. Try keyword search."

            result = [f"# Semantic search: '{search}' ({len(matches)} results)\n"]
            for path, method, details, score in matches:
                result.append(f"[relevance: {score:.2f}]")
                result.append(_format_endpoint(path, method, details, spec))
                result.append("---\n")
            return "\n".join(result)
        except Exception as e:
            logger.warning("Semantic search failed: %s — falling back to keyword", e)

    # Keyword search
    search_lower = search.lower().strip("/")
    matches = []
    for path, methods in paths.items():
        path_lower = path.lower()
        if search_lower in path_lower:
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete"):
                    matches.append((path, method, details))
        elif any(
            search_lower in d.get("summary", "").lower() for d in methods.values()
        ):
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete"):
                    if search_lower in details.get("summary", "").lower():
                        matches.append((path, method, details))

    if not matches:
        return (
            f"No endpoints for '{search}'. "
            "Try semantic=true or broader terms like 'employee', 'customer', 'invoice'."
        )

    if len(matches) > 10:
        matches.sort(key=lambda x: len(x[0]))
        matches = matches[:10]

    result = [f"# API docs: '{search}' ({len(matches)} endpoints)\n"]
    for path, method, details in matches:
        result.append(_format_endpoint(path, method, details, spec))
        result.append("---\n")
    return "\n".join(result)


# Pre-download spec if not cached
if not SPEC_PATH.exists():
    try:
        _get_spec()
    except Exception:
        pass
