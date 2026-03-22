# Tripletex AI Agent — NM i AI 2026

AI agent that solves accounting tasks in Tripletex via the Tripletex API v2.
Receives task prompts in 7 languages via HTTP, executes the required API calls autonomously.
Built for the Norwegian AI Championship (NM i AI 2026), March 19-22.

## Architecture

```
Request
  │
  ├─ Experience Checker (Elasticsearch, no LLM) ─────────────┐
  │                                                           │
  ├─ Prefetch (deterministic API calls) ──────────────────────┤
  │                                                           │
  ├─ PDF Extractor (GPT-4.1-mini) ──┐                        │
  │                                  ├── parallel ────────────┤
  └─ Schema Agent (GPT-4.1) ────────┘                        │
                                                              ▼
                                                     Executor (GPT-4.1)
                                                        5 tools only
                                                     tripletex_get/post/put/delete
                                                     + get_payload_template (fallback)
```

### Pipeline stages

| Stage | Model | Purpose | Time |
|-------|-------|---------|------|
| **Experience Checker** | None (ES lookup) | Finds past failures/warnings for this task type | <1s |
| **Prefetch** | None (deterministic) | Fetches reference IDs, auto-sets bank account on 1920 | ~2s |
| **PDF Extractor** | GPT-4.1-mini | Extracts structured data from PDF/image attachments | ~5s |
| **Schema Agent** | GPT-4.1 | Resolves endpoint templates and workflow for the task | ~10s |
| **Executor** | GPT-4.1 | Executes API calls using pre-resolved templates | ~15s |

PDF extraction and schema discovery run **in parallel** via `asyncio.gather`.

## Model findings

### Architecture evolution

| Architecture | Model(s) | Result | Why changed |
|---|---|---|---|
| Planner → Executor | Haiku + Sonnet | Rate limited (Tier 1, 50 RPM) | Burned through Anthropic limits |
| Researcher → Executor | Haiku + Opus | Over-researched (14+ calls, timeouts) | Researcher couldn't stop |
| Single agent | Opus | Good results but expensive + slow | Cost, switched to OpenAI |
| Single agent | GPT-4o | Researched but didn't execute | GPT-4o didn't follow through |
| Single agent + phased prompt | GPT-4.1 | Better execution, still missed templates | Needed schema prep |
| Schema agent + Executor | GPT-4.1 + GPT-4.1 | Current architecture, best results | Schema prep + focused execution |

### Model-specific findings

**Anthropic Claude (Haiku, Sonnet, Opus)**
- Opus: best at autonomous reasoning, figured out complex workflows without templates
- Sonnet: good balance but rate limited at Tier 1
- Haiku: fast researcher but shallow reasoning
- Uses `"type": "document"` for PDFs (Anthropic-specific format)
- Rate limits were the main blocker (50 RPM at Tier 1)

**OpenAI GPT-4.1**
- Needs explicit phased prompts ("PLAN → EXECUTE → CONTINUE")
- Won't execute POST/PUT unless the prompt says "You MUST execute"
- Works well as an executor when templates are pre-resolved
- Uses `"type": "file"` for PDFs (OpenAI format — NOT `"type": "document"`)
- Better availability and higher rate limits than Anthropic

**OpenAI GPT-4.1-mini**
- Good for PDF extraction and simple lookups
- Too weak for schema discovery — loops on ambiguous template matches
- Fast and cheap for deterministic tasks

**Google Gemini**
- Gemini Flash: doesn't support `"document"` type for PDFs
- Gemini Pro: decent but inconsistent on accounting tasks
- Abandoned due to format compatibility issues

### Key lesson: model-provider format differences

When switching providers, ALL code that sends content to the model must be audited:
- Anthropic: `"type": "document"` with `"source": {"type": "base64", ...}`
- OpenAI: `"type": "file"` with `"file": {"filename": "...", "file_data": "data:application/pdf;base64,..."}`
- This mismatch caused silent PDF extraction failures (0/10 scores) until caught.

## Tools and search

### Executor tools (5 total)

| Tool | Purpose |
|------|---------|
| `tripletex_get` | GET API data (find entities, look up IDs). Has auto-correction for common path mistakes |
| `tripletex_post` | POST to create entities |
| `tripletex_put` | PUT to update or trigger actions (payment, invoice, credit note) |
| `tripletex_delete` | DELETE entities |
| `get_payload_template` | Fallback — hybrid search for verified JSON templates + API docs auto-fallback |

### Schema agent tools (3 total)

| Tool | Purpose |
|------|---------|
| `get_task_workflow` | Hybrid BM25+semantic search over task patterns (workflow recipes) |
| `get_payload_template` | Returns verified JSON templates for endpoints |
| `lookup_api_docs` | Hybrid search over 800+ OpenAPI endpoints |

### Search tools (not on executor — used by schema agent or experience checker)

| Tool | Search method | Data source |
|------|---------------|-------------|
| `get_task_workflow` | Hybrid BM25 + semantic (TEI) | Task patterns (17+ task types) |
| `get_payload_template` | Hybrid BM25 + semantic (TEI) + path corrections + API docs fallback | 20+ verified templates |
| `lookup_api_docs` | Hybrid BM25 + semantic (TEI) | OpenAPI spec (800+ endpoints) |
| `search_past_experience` | Hybrid BM25 (Elasticsearch) + semantic (TEI) | Past execution traces |
| `lookup_task_pattern` | Hybrid BM25 + semantic (TEI) | Task patterns (detailed) |
| `search_tripletex_docs` | Hybrid BM25 + semantic (TEI) | Tripletex developer docs |
| `explain_accounting_concept` | Keyword match | 12 accounting concepts |
| `web_search` | DuckDuckGo | Internet (last resort) |

### Hybrid search implementation

All search tools use **Reciprocal Rank Fusion (RRF)** combining:
1. **BM25** (rank-bm25 library) — lexical keyword matching
2. **Semantic** (TEI with google/embeddinggemma-300m) — embedding similarity

This handles both exact endpoint names ("POST /ledger/voucher") and natural language queries ("how to book depreciation").

### Endpoint auto-correction

Common path mistakes are silently corrected:
```python
"/account"           → "/ledger/account"
"/voucher"           → "/ledger/voucher"
"/voucherType"       → "/ledger/voucherType"
"/vatType"           → "/ledger/vatType"
"/project/activity"  → "/project/projectActivity"
"/employment"        → "/employee/employment"
"/cost"              → "/travelExpense/cost"
```

## Observability

### Langfuse

All agent runs are traced via [Langfuse](https://langfuse.com/) (self-hosted at localhost:30001).

**What's traced:**
- Full message history (human → AI → tool calls → tool results)
- Token usage per call
- Trace duration
- Tool call arguments and responses

**How to use:**
- Dashboard: http://localhost:30001
- API: `GET /api/public/traces?limit=N` with Basic Auth (`LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY`)
- Traces are linked to the LangGraph agent via `langfuse-langchain` callback handler

**Example: fetch latest trace**
```bash
curl -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "http://localhost:30001/api/public/traces?limit=1" | python3 -m json.tool
```

### Elasticsearch

Stores enriched past execution traces for the experience system.

**Index:** `tripletex-experience`

**Fields:**
- `task_prompt` — original task text
- `tags` — classified task types (invoice, employee, voucher, etc.)
- `total_tool_calls`, `total_errors` — execution metrics
- `successful_calls`, `failed_calls_with_fixes` — what worked and what didn't
- `competition_notes` — manually enriched warnings (highest search weight, ^4)
- `lesson_learned` — auto-generated summary

**How experience flows through the pipeline:**
1. Experience checker extracts search terms from prompt (multilingual keyword mapping)
2. Searches Elasticsearch with terms like "incoming invoice supplier 403 permission"
3. Results sorted: competition warnings first → traces with errors → successful traces
4. Output passed to schema agent (as warnings on templates) and executor (as rules to follow)

## Key design decisions

### Why templates over free-form generation

GPT-4.1 constructs JSON from memory with wrong field names (`debitAmount` instead of `amountGross`, `phoneNumber` instead of `phoneNumberMobile`). Verified templates eliminate this class of errors entirely.

### Why 5 tools on the executor

With 10+ tools, GPT-4.1 wastes calls on research tools instead of executing. Reducing to 5 tools (get/post/put/delete + template fallback) forces it to execute. The schema agent handles all research beforehand.

### Why asyncio.Lock for concurrent requests

The Tripletex API client uses module-level state. Two concurrent requests would overwrite each other's credentials. An `asyncio.Lock` serializes requests. `contextvars` was attempted but didn't work with LangGraph's async internals.

### Why auto-set bank account

Invoice creation fails without a bank account on account 1920. Rather than relying on the LLM agent to figure this out (it often didn't), the deterministic prefetch step checks and sets it automatically.

### Why error recovery in prompt, not code

Handling 403/404/422 in the prompt ("if 403, use voucher fallback") lets the agent adapt in-context rather than requiring code changes for each new error pattern. This is more flexible and leverages the LLM's reasoning.

## File structure

```
task2_tripletex/
├── main.py                  # FastAPI entrypoint (POST /solve, POST /, GET /health)
├── agent.py                 # Main agent orchestration and pipeline
├── models.py                # SolveRequest, SolveResponse, FileAttachment dataclasses
├── tools.py                 # Tripletex API client + get/post/put/delete tools
├── schema_agent.py          # Schema discovery agent (resolves templates before executor)
├── experience_checker.py    # Pre-flight experience check (no LLM)
├── experience_tool.py       # Elasticsearch-based past experience search
├── pdf_extractor.py         # PDF/image data extraction (GPT-4.1-mini)
├── prefetch_agent.py        # Pre-fetch agent for sandbox reference data
├── workflow_tools.py        # get_task_workflow + get_payload_template (hybrid search)
├── payload_templates.py     # Verified JSON templates for 20+ endpoints
├── task_patterns.py         # Accounting workflow recipes per task type
├── task_patterns_tool.py    # Hybrid search over task patterns
├── api_docs_tool.py         # Hybrid search over OpenAPI spec (800+ endpoints)
├── accounting_concepts.py   # Maps accounting terms to Tripletex operations
├── devdocs_tool.py          # Tripletex developer docs search
├── web_search_tool.py       # DuckDuckGo web search (last resort)
├── build_index.py           # Builds semantic search indices
├── index_traces.py          # Indexes traces to Elasticsearch
├── enrich_traces.py         # Enriches traces with error fixes and lessons
└── test_agent.py            # Test suite with multi-language tasks
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for package management
- Elasticsearch (localhost:9200)
- TEI embeddings server (localhost:8080) — `google/embeddinggemma-300m`
- Langfuse (localhost:30001) for tracing

### Environment variables (.env)

```
OPENAI_API_KEY=...
TRIPLETEX_BASE_URL=https://api.tripletex.io/v2
TRIPLETEX_SESSION_TOKEN=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=http://localhost:30001
TEI_URL=http://localhost:8080
```

### Run

```bash
# Install dependencies
uv sync

# Build search indices (first time)
uv run python -m task2_tripletex.build_index

# Index past traces to Elasticsearch
uv run python -m task2_tripletex.index_traces
uv run python -m task2_tripletex.enrich_traces

# Start the server
uv run uvicorn task2_tripletex.main:app --host 0.0.0.0 --port 8000
```

### Infrastructure (Docker)

```bash
# Elasticsearch
docker run -d --name es -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.12.0

# TEI embeddings
docker run -d --name tei -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id google/embeddinggemma-300m

# Langfuse
docker compose -f langfuse-docker-compose.yml up -d
```

## Supported task types

| Task type | Difficulty | Endpoints | Status |
|-----------|-----------|-----------|--------|
| Customer | Tier 1 | POST /customer | Verified |
| Employee + admin | Tier 1 | POST /employee, POST /employee/employment, PUT /employee/entitlement | Verified |
| Product | Tier 1 | POST /product | Verified |
| Supplier | Tier 1 | POST /supplier | Verified |
| Order + Invoice | Tier 2 | POST /order, PUT /order/:invoice | Verified |
| Invoice payment | Tier 2 | PUT /invoice/:payment | Verified |
| Travel expense | Tier 2 | POST /travelExpense, POST /travelExpense/cost | Verified |
| Salary / payroll | Tier 2 | POST /salary/transaction | Verified |
| Project (simple) | Tier 2 | POST /project | Verified |
| Project (fixed price) | Tier 3 | POST /project, POST /project/orderline, PUT /order/:invoice | Verified |
| Time tracking | Tier 2 | POST /timesheet/entry, POST /project/projectActivity | Verified |
| Voucher / journal | Tier 2 | POST /ledger/voucher | Verified |
| Credit note | Tier 2 | PUT /invoice/:createCreditNote | Verified |
| Supplier / incoming invoice | Tier 2 | POST /incomingInvoice (fallback: voucher if 403) | Verified |
| Reminder fee | Tier 2 | POST /invoice/paymentReminder | Verified |
| Month-end closing | Tier 3 | POST /ledger/voucher (depreciation, periodization) | Verified |
| General ledger correction | Tier 3 | POST /ledger/voucher (correction entries) | Verified |

## Competition scores

Best scores are locked per task type — bad runs don't lower your score.

Notable scores observed:
- 7.5/10 on a tier 3 "find and correct 4 ledger errors" task
- 4/8 after adding phased prompt (up from 0/10)
- Various 0 scores traced back to: PDF format mismatch, missing templates, 403 on /incomingInvoice
