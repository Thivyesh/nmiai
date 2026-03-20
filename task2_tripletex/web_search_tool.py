"""Web search tool as last-resort fallback for unknown tasks.

Only use when local knowledge sources (task patterns, API docs, developer docs)
don't have the answer.
"""

import logging

from duckduckgo_search import DDGS
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web for Tripletex API help. LAST RESORT — use only when
    lookup_task_pattern, lookup_api_docs, and search_tripletex_docs don't help.

    Automatically adds "Tripletex API" to the query.

    Args:
        query: What you need help with, e.g. "create accounting dimension value",
               "voucher posting with free dimension", "how to register salary".

    Returns:
        Top search results with titles and snippets.
    """
    full_query = f"Tripletex API v2 {query} site:developer.tripletex.no OR site:github.com/Tripletex OR site:tripletex.no/v2-docs"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(full_query, max_results=5))
        # Fallback to broader search if no results
        if not results:
            full_query = f"Tripletex API {query}"
            with DDGS() as ddgs:
                results = list(ddgs.text(full_query, max_results=5))
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return f"Web search failed: {e}"

    if not results:
        return f"No web results for '{full_query}'."

    output = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("body", "")
        url = r.get("href", "")
        output.append(f"**{title}**\n{snippet}\n{url}")

    return "\n\n---\n\n".join(output)
