# finn-mcp — search active finn.no listings from Claude

An [MCP](https://modelcontextprotocol.io) server that lets Claude search
[finn.no](https://www.finn.no) — Norway's biggest classifieds site — for the
stuff you're after.

## The problem it solves

Ask a plain LLM to "find X on finn" and it reaches for web search, which
returns **cached Google/DuckDuckGo results**. Those are full of listings that
are already **sold or deactivated** — the ad you click is gone.

This server queries finn.no's **own live search index** (`/api/search-qf`, the
same backend that powers the finn.no website). That index only contains ads
that are **active right now**, so every result is something you can actually
still buy. There's also a `get_listing` tool to double-check a specific ad is
still up before you get excited.

## Tools

| Tool | What it does |
|------|--------------|
| `search_finn` | Search Torget (the general marketplace) for active ads. Filters: `sort` (relevance/newest/oldest/price_asc/price_desc), `min_price`, `max_price`, `published_since_days`, `max_results`. |
| `get_listing` | Given a finnkode or finn URL, return details and whether it's **still active / not sold**. |
| `raw_search` | Advanced passthrough to `search-qf` for other verticals (cars, real estate, jobs) or filters not covered above. |

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync            # installs deps, including `mcp` and `httpx`
```

Verify it runs (it will wait on stdin — Ctrl-C to exit):

```bash
uv run python -m finn_mcp
```

### Add to Claude Code

```bash
claude mcp add finn -- uv --directory /ABSOLUTE/PATH/TO/nmiai run python -m finn_mcp
```

### Add to Claude Desktop

Edit `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows: `%APPDATA%\Claude\claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "finn": {
      "command": "uv",
      "args": ["--directory", "/ABSOLUTE/PATH/TO/nmiai", "run", "python", "-m", "finn_mcp"]
    }
  }
}
```

Restart Claude Desktop. You should see the 🔌 tools indicator with `search_finn`,
`get_listing`, and `raw_search`.

## Example prompts

- "Find me an electric bike on finn under 15000 kr, newest first."
- "Search finn for an IKEA Kallax in Oslo and show the 5 cheapest."
- "Anything new on finn for 'kajakk' in the last 3 days?"
- "Is finnkode 412345678 still for sale?"
- "Use raw_search to find used Tesla Model 3 cars on finn."

## How it works

```
Claude ──stdio──> finn_mcp.server (MCPServer)
                       │
                       ▼
                 FinnClient  ──HTTPS──>  www.finn.no/api/search-qf   (live index → active ads)
                                          www.finn.no/.../item/<id>   (active/sold check)
```

- `finn_mcp/client.py` — async `httpx` client: query building, pagination,
  retries with backoff, and normalization of finn's `docs[]` into a stable
  `Listing` shape (id, title, url, price, location, published, image, sold flag).
- `finn_mcp/server.py` — registers the three tools and renders results as
  readable text with direct finn.no links.

### Notes / limits

- Uses finn.no's public web API without authentication — for personal use.
  Be considerate with request volume.
- Torget is the default vertical (general second-hand goods). Cars, real
  estate and jobs live under different `searchkey`/`vertical` values — reach
  them via `raw_search`.
- finn.no can change its internal API. If searches stop returning results,
  the endpoint/params in `finn_mcp/client.py` (`SEARCH_ENDPOINT`,
  `TORGET_PARAMS`) are the one place to adjust.
