"""LangGraph-based Tripletex accounting agent with planner → executor architecture."""

import base64
import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.prebuilt import create_react_agent

from task2_tripletex.models import SolveRequest, SolveResponse
from task2_tripletex.tools import (
    EXECUTOR_TOOLS,
    PLANNER_TOOLS,
    TripletexClient,
    set_client,
)

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are an expert accounting task planner for Tripletex, a Norwegian accounting system.

## Your Tools (use in this priority order)
1. **lookup_task_pattern(task_description)** — CALL FIRST. Returns workflow, scoring criteria, and common mistakes for the task type.
2. **tripletex_get(endpoint, params)** — Read-only API access. Use to find real IDs (departments, accounts, employees, payment types).
3. **lookup_api_docs(search, semantic)** — Look up exact field names and schemas from the OpenAPI spec. Use semantic=True for non-English terms.
4. **search_tripletex_docs(query)** — Search official Tripletex developer FAQs and guides.
5. **web_search(query)** — Last resort web search when nothing else helps.

## Workflow

### Step 1: Look up the task pattern
Call lookup_task_pattern with the task description. This gives you:
- The exact workflow steps
- What fields the competition checks
- Common mistakes to avoid
Follow the pattern precisely.

### Step 2: Look up real IDs
Use tripletex_get to find IDs needed by the plan. The sandbox is FRESH — most entities must be created.
- GET /department?fields=id&count=1 (for employee tasks)
- GET /ledger/account?number=NNNN&fields=id (for voucher tasks — ALWAYS get account ID, never use number!)
- GET /travelExpense/paymentType (for travel expense tasks)
- GET /invoice/paymentType (for invoice payment tasks)

### Step 3: For unfamiliar tasks — DISCOVER
If no task pattern matches:
a) lookup_api_docs(search, semantic=True) to find relevant endpoints
b) lookup_api_docs for the POST schema to get exact field names
c) search_tripletex_docs for workflow guidance
d) tripletex_get to explore the endpoint
e) web_search as last resort

### Step 4: Output the plan
Include real IDs, exact field names, and exact payloads.

## CRITICAL RULES
- Every entity mentioned in the prompt must be CREATED as a separate record.
- Every field value in the prompt WILL be checked. Include ALL of them.
- Use EXACT values from the prompt. NEVER modify names, emails, amounts.
- ALWAYS use account {"id": N} not {"number": N} — look up the ID via GET first.
- For voucher postings: use amountGross + amountGrossCurrency (NEVER "amount").

## Output Format
TASK SUMMARY: <one line>

CONTEXT:
- <IDs found, prerequisites checked>

STEPS:
1. <METHOD> <endpoint>
   Payload: {<exact JSON with real IDs>}

NOTES:
- <any gotchas>
"""

EXECUTOR_SYSTEM_PROMPT = """\
You are a Tripletex API executor. Follow the plan step by step.

## Your Tools
- **tripletex_post/put/delete** — Execute API calls
- **tripletex_get** — Read API data (for error investigation)
- **lookup_api_docs(search, semantic)** — Look up exact field names and schemas
- **lookup_task_pattern(task_description)** — Look up accounting workflow guidance
- **search_tripletex_docs(query)** — Search official Tripletex FAQs
- **web_search(query)** — Last resort web search

## Execution
1. Follow the plan step by step, in order.
2. Use the EXACT endpoint, method, and payload from each step.
3. After POST/PUT, save the returned ID for subsequent steps.
4. Use EXACT values from the plan and original task. NEVER modify names, emails, amounts.
5. For query-param endpoints (entitlements, payment, credit note), put params in the URL, pass body="{}".

## Error Recovery (RESEARCH before retrying — max 2 retries per step)

When a step fails:
1. Read the error message.
2. Call **lookup_api_docs** for the failing endpoint to get the correct schema.
3. If that's not enough, call **search_tripletex_docs** with the error message.
4. Fix the specific issue and retry ONCE.
5. If still failing, call **lookup_task_pattern** for workflow guidance.
6. Last resort: **web_search** for the specific error.
7. After 2 failed retries, move to the next step.

## Common Error Fixes
- "Request mapping failed" → wrong field names. Look up the schema.
- "Feltet må fylles ut" → missing required field. Add it.
- "Object not found" / 404 → wrong ID or URL format. Check query params vs path.
- "Kunde mangler" → account requires customer ID in the posting.
- "Enhetspris må være uten mva" → use unitPriceExcludingVatCurrency and set isPrioritizeAmountsIncludingVat.
- Account references: ALWAYS use {"id": N}, never {"number": N}.
- Voucher amounts: use amountGross + amountGrossCurrency, NEVER "amount".

## Efficiency
- Every 4xx error hurts the score. Research before retrying.
- Do NOT add extra verification GET calls.
- Stop after completing all planned steps.
"""


class TripletexAgent:
    """Orchestrates the planner → executor pipeline for solving Tripletex tasks."""

    def __init__(self):
        self.planner_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0,
            max_retries=2,
            timeout=30.0,
        )
        self.executor_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
            max_retries=2,  # Don't retry more than 2x on rate limits
            timeout=60.0,  # 60s timeout per LLM call
        )
        # Planner: Haiku (fast, separate rate limit) + read-only tools
        self.planner = create_react_agent(
            model=self.planner_llm,
            tools=PLANNER_TOOLS,
            prompt=PLANNER_SYSTEM_PROMPT,
        )
        # Executor: ReAct agent with write tools only (POST, PUT, DELETE)
        self.executor = create_react_agent(
            model=self.executor_llm,
            tools=EXECUTOR_TOOLS,
            prompt=EXECUTOR_SYSTEM_PROMPT,
        )

    def _extract_file_content(self, request: SolveRequest) -> list:
        """Extract content from attached files as message parts."""
        parts = []
        for f in request.files:
            raw = base64.b64decode(f.content_base64)
            if f.mime_type.startswith("image/"):
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{f.mime_type};base64,{f.content_base64}",
                    },
                })
                parts.append({
                    "type": "text",
                    "text": f"[Above image: {f.filename}]",
                })
            elif f.mime_type == "application/pdf":
                parts.append({
                    "type": "text",
                    "text": f"[PDF file: {f.filename}, {len(raw)} bytes — base64 available]",
                })
            else:
                try:
                    text = raw.decode("utf-8")
                    parts.append({
                        "type": "text",
                        "text": f"### {f.filename}\n```\n{text[:8000]}\n```",
                    })
                except UnicodeDecodeError:
                    parts.append({
                        "type": "text",
                        "text": f"[Binary file: {f.filename} ({f.mime_type}, {len(raw)} bytes)]",
                    })
        return parts

    async def _plan(self, request: SolveRequest, config: dict) -> str:
        """Use the planner agent to research the API and create an execution plan."""
        content_parts = [{"type": "text", "text": f"## Task Prompt\n\n{request.prompt}"}]

        if request.files:
            content_parts.append({"type": "text", "text": "\n## Attached Files\n"})
            content_parts.extend(self._extract_file_content(request))

        result = await self.planner.ainvoke(
            {"messages": [HumanMessage(content=content_parts)]},
            config=config,
        )

        # Extract the final plan from the last AI message
        last_message = result["messages"][-1]
        plan = last_message.content
        logger.info("Plan:\n%s", plan)
        return plan

    async def solve(self, request: SolveRequest) -> SolveResponse:
        """Run the planner → executor pipeline to solve a Tripletex task."""
        client = TripletexClient(
            base_url=request.tripletex_credentials.base_url,
            session_token=request.tripletex_credentials.session_token,
        )
        set_client(client)

        config: dict = {}
        langfuse_handler = self._create_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        # Step 1: Plan (planner can do GET calls to research)
        # Cap planner at 20 iterations to avoid burning rate limits
        planner_config = {**config, "recursion_limit": 20}
        plan = await self._plan(request, planner_config)

        # Step 2: Execute
        # Cap executor at 15 iterations — if it can't finish in 15 tool calls, stop
        executor_config = {**config, "recursion_limit": 15}
        executor_message = HumanMessage(
            content=f"## Original Task\n{request.prompt}\n\n## Execution Plan\n{plan}"
        )
        await self.executor.ainvoke(
            {"messages": [executor_message]},
            config=executor_config,
        )

        return SolveResponse(status="completed")

    def _create_langfuse_handler(self) -> LangfuseCallbackHandler | None:
        """Create Langfuse callback handler if credentials are configured."""
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return LangfuseCallbackHandler()
        return None
