"""MCP server exposing finn.no search to Claude.

Run over stdio (the transport Claude Desktop / Claude Code use):

    uv run finn-mcp

Tools:
    * search_finn   — search Torget (the general marketplace) for active ads
    * get_listing   — details for one ad + whether it is still active
    * raw_search    — advanced passthrough for other verticals / filters
"""

from __future__ import annotations

from typing import Any

# The high-level server class was renamed FastMCP -> MCPServer in mcp 2.0.
# Support both so this works whichever version the user has installed.
try:
    from mcp.server import MCPServer as _Server  # mcp >= 2.0
except ImportError:  # pragma: no cover - depends on installed version
    from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x

from .client import FinnClient, FinnError, Listing

mcp = _Server("finn")
_client = FinnClient()


def _render(listings: list[Listing]) -> str:
    if not listings:
        return "No active listings found. Try broadening the query or removing price filters."
    lines: list[str] = [f"Found {len(listings)} active listing(s):\n"]
    for i, l in enumerate(listings, 1):
        price = l.price_text or ("Price not stated" if l.price is None else f"{l.price} {l.currency}")
        bits = [f"{i}. {l.title}", f"   {price}"]
        if l.location:
            bits.append(f"   📍 {l.location}")
        if l.published:
            bits.append(f"   🕒 published {l.published}")
        if l.is_sold:
            bits.append("   ⚠️ marked SOLD")
        bits.append(f"   {l.url}")
        lines.append("\n".join(bits))
    return "\n\n".join(lines)


@mcp.tool()
async def search_finn(
    query: str,
    max_results: int = 20,
    sort: str = "relevance",
    min_price: int | None = None,
    max_price: int | None = None,
    published_since_days: int | None = None,
) -> str:
    """Search finn.no's Torget marketplace for currently ACTIVE listings.

    Queries finn.no's own live search index, so every result is an ad that is
    for sale right now — no sold/expired cache hits.

    Args:
        query: Free-text search, e.g. "elsykkel", "ikea skrivebord oslo".
        max_results: Max listings to return (paginates as needed).
        sort: One of relevance, newest, oldest, price_asc, price_desc.
        min_price: Minimum price in NOK (inclusive).
        max_price: Maximum price in NOK (inclusive).
        published_since_days: Only ads published within the last N days.

    Returns a human-readable list with title, price, location, publish time
    and a direct finn.no link for each listing.
    """
    try:
        listings = await _client.search(
            query,
            sort=sort,
            max_results=max_results,
            min_price=min_price,
            max_price=max_price,
            published_since_days=published_since_days,
        )
    except FinnError as exc:
        return f"finn.no search failed: {exc}"
    return _render(listings)


@mcp.tool()
async def get_listing(id_or_url: str) -> dict[str, Any]:
    """Get details for one finn.no ad and whether it is still active.

    Use this to confirm a specific listing is still for sale (not sold or
    removed). Accepts a finnkode ("123456789") or a full finn.no URL.

    Returns a dict with title, description, price, image, url, ``is_active``
    and ``is_sold``.
    """
    try:
        return await _client.get_listing(id_or_url)
    except FinnError as exc:
        return {"error": str(exc)}


@mcp.tool()
async def raw_search(params: dict[str, Any], max_results: int = 20) -> str:
    """Advanced: query finn.no's search-qf API with arbitrary parameters.

    For other verticals or filters not covered by ``search_finn``. Examples:

        Used cars:   {"searchkey": "SEARCH_ID_CAR_USED", "vertical": "car", "q": "tesla model 3"}
        Real estate: {"searchkey": "SEARCH_ID_REALESTATE_HOMES", "vertical": "realestate", "q": "leilighet oslo"}
        Torget + filter: {"searchkey": "SEARCH_ID_BAP_COMMON", "search_type": "SEARCH_ID_BAP_ALL", "vertical": "bap", "q": "kajakk", "sort": "PUBLISHED_DESC"}

    Only active listings are returned. Returns the same rendered list as
    ``search_finn``.
    """
    try:
        listings = await _client.raw_search(params, max_results=max_results)
    except FinnError as exc:
        return f"finn.no search failed: {exc}"
    return _render(listings)


def main() -> None:
    """Console entry point (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
