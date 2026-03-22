"""Focused tools for the researcher: workflow steps and payload templates."""

import json

from langchain_core.tools import tool

from task2_tripletex.payload_templates import PAYLOAD_TEMPLATES
from task2_tripletex.task_patterns import TASK_PATTERNS


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
    # Try exact match first
    template = PAYLOAD_TEMPLATES.get(endpoint)

    # Try fuzzy match — prefer longest matching key to avoid "POST /project" matching "POST /project/projectActivity"
    if not template:
        endpoint_lower = endpoint.lower()
        best_key = None
        best_len = 0
        for key, val in PAYLOAD_TEMPLATES.items():
            key_lower = key.lower()
            if endpoint_lower == key_lower:
                best_key = key
                break
            if key_lower in endpoint_lower or endpoint_lower in key_lower:
                if len(key) > best_len:
                    best_key = key
                    best_len = len(key)
        if best_key:
            template = PAYLOAD_TEMPLATES[best_key]
            endpoint = best_key

    # Try partial match on the path part — also prefer longest
    if not template:
        best_key = None
        best_len = 0
        for key, val in PAYLOAD_TEMPLATES.items():
            path = key.split(" ", 1)[-1] if " " in key else key
            if path in endpoint or endpoint in path:
                if len(key) > best_len:
                    best_key = key
                    best_len = len(key)
        if best_key:
            template = PAYLOAD_TEMPLATES[best_key]
            endpoint = best_key

    if not template:
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
