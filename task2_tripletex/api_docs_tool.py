"""Dynamic API documentation tool for the planner agent.

Parses the Tripletex OpenAPI spec and provides endpoint documentation on demand.
This replaces the static API_REFERENCE in the system prompt.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import requests
from langchain_core.tools import tool

# Cache the OpenAPI spec
_spec: dict | None = None
SPEC_PATH = Path(__file__).parent / "openapi_cache.json"


def _download_spec(base_url: str = "https://kkpqfuj-amager.tripletex.dev/v2") -> dict:
    """Download and cache the OpenAPI spec."""
    resp = requests.get(f"{base_url}/openapi.json", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get_spec() -> dict:
    """Get the OpenAPI spec, using cache if available."""
    global _spec
    if _spec:
        return _spec

    if SPEC_PATH.exists():
        with open(SPEC_PATH) as f:
            _spec = json.load(f)
    else:
        _spec = _download_spec()
        with open(SPEC_PATH, "w") as f:
            json.dump(_spec, f)

    return _spec


def _resolve_ref(ref: str, spec: dict) -> dict:
    """Resolve a $ref to its schema."""
    name = ref.split("/")[-1]
    return spec.get("components", {}).get("schemas", {}).get(name, {})


def _get_schema_fields(schema: dict, spec: dict, depth: int = 0) -> list[str]:
    """Extract writable fields from a schema."""
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
            # Show sub-fields for key references
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
    """Format a single endpoint's documentation."""
    summary = details.get("summary", "")
    lines = [f"### {method.upper()} {path}", f"{summary}", ""]

    # Query parameters
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

    # Request body
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


@tool
def lookup_api_docs(search: str) -> str:
    """Look up Tripletex API documentation for specific endpoints.

    Use this to find the correct fields, parameters, and schemas for API endpoints.
    Search by resource name (e.g., "employee", "travelExpense/cost", "invoice")
    or by keyword (e.g., "per diem", "payment", "credit note").

    Args:
        search: Resource path or keyword to search for, e.g. "travelExpense/cost",
                "employee", "invoice/:payment", "per diem", "department".

    Returns:
        Formatted API documentation for matching endpoints.
    """
    spec = _get_spec()
    paths = spec.get("paths", {})
    search_lower = search.lower().strip("/")

    # Find matching paths
    matches = []
    for path, methods in paths.items():
        path_lower = path.lower()
        # Direct path match
        if search_lower in path_lower:
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete"):
                    matches.append((path, method, details))
        # Keyword match in summary
        elif any(
            search_lower in details.get("summary", "").lower()
            for details in methods.values()
        ):
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete"):
                    if search_lower in details.get("summary", "").lower():
                        matches.append((path, method, details))

    if not matches:
        return f"No endpoints found matching '{search}'. Try broader terms like 'employee', 'customer', 'invoice', 'travelExpense'."

    # Limit to most relevant (max 10 endpoints)
    if len(matches) > 10:
        # Prioritize shorter paths (main endpoints over sub-resources)
        matches.sort(key=lambda x: len(x[0]))
        matches = matches[:10]

    result = [f"# API docs for '{search}' ({len(matches)} endpoints)\n"]
    for path, method, details in matches:
        result.append(_format_endpoint(path, method, details, spec))
        result.append("---\n")

    return "\n".join(result)


# Pre-download the spec on import if not cached
if not SPEC_PATH.exists():
    try:
        _get_spec()
    except Exception:
        pass  # Will download on first use
