"""Search past task execution history for learning from experience."""

import logging

from elasticsearch import Elasticsearch
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

ES_URL = "http://localhost:9200"
INDEX = "tripletex-traces"


@tool
def search_past_experience(task_description: str) -> str:
    """Search past task executions to learn what worked and what failed.

    Returns similar past tasks with their API calls, errors, and outcomes.
    Use this to avoid repeating mistakes and copy successful approaches.

    Args:
        task_description: English description of the current task, e.g.
            "create invoice with payment", "voucher posting", "travel expense".

    Returns:
        Past experiences with what worked (200 OK) and what failed (errors).
    """
    try:
        es = Elasticsearch(ES_URL)
        if not es.indices.exists(index=INDEX):
            return "No past experience available yet."
    except Exception:
        return "Experience search unavailable (Elasticsearch not running)."

    # Multi-match across task prompt and tool summary
    results = es.search(
        index=INDEX,
        body={
            "size": 5,
            "query": {
                "multi_match": {
                    "query": task_description,
                    "fields": ["task_prompt^3", "tool_summary", "failed_endpoints_text"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
            "sort": [
                {"total_errors": "asc"},  # Prefer low-error traces
                "_score",
            ],
        },
    )

    hits = results.get("hits", {}).get("hits", [])
    if not hits:
        return f"No past experience found for '{task_description}'."

    output = []
    for hit in hits[:3]:
        src = hit["_source"]
        errors = src.get("total_errors", 0)
        success_endpoints = src.get("successful_endpoints", [])
        failed_text = src.get("failed_endpoints_text", "")
        prompt = src.get("task_prompt", "")[:200]
        tool_summary = src.get("tool_summary", "")

        # Extract successful patterns
        success_lines = [
            line for line in tool_summary.split("\n")
            if " OK " in line and "tripletex_" in line
        ]
        # Extract failed patterns
        failed_lines = [
            line for line in tool_summary.split("\n")
            if " ERR " in line
        ]

        entry = f"### Past task (errors: {errors})\n"
        entry += f"Task: {prompt}\n"
        if success_lines:
            entry += f"Worked: {'; '.join(success_lines[:5])}\n"
        if failed_lines:
            entry += f"Failed: {'; '.join(failed_lines[:3])}\n"
        if failed_text:
            entry += f"Error details: {failed_text[:200]}\n"

        output.append(entry)

    return "\n---\n".join(output)
