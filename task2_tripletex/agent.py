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

You have two tools:
1. **lookup_api_docs(search)** — Look up the exact API schema for any endpoint. ALWAYS use this before planning API calls to get correct field names and required fields.
2. **tripletex_get(endpoint, params)** — Read-only access to the live Tripletex API. Use to find existing entities and their IDs.

## Workflow
1. Parse the prompt to extract: task type, entity names, field values, relationships.
2. Use **lookup_api_docs** to find the correct endpoints and their exact field names/schemas.
3. Use **tripletex_get** to look up real IDs (departments, employees, customers, payment types, etc.)
4. Produce a concrete execution plan with REAL IDs and CORRECT field names.

## Key Patterns (use lookup_api_docs for exact schemas)
- Employees: POST /employee (requires department), entitlements via PUT /employee/entitlement/:grantEntitlementsByTemplate
- Customers: POST /customer
- Invoices: POST /order → POST /invoice (requires bank account on ledger account 1920)
- Payments: PUT /invoice/{id}/:payment (query params, not body)
- Travel expenses: POST /travelExpense → POST /travelExpense/cost for each expense line
- Per diem: POST /travelExpense/perDiemCompensation (requires travelDetails with dates)
- Projects: POST /project (requires customer and projectManager)
- Credit notes: PUT /invoice/{id}/:createCreditNote (query params)

## Output Format
TASK SUMMARY: <one line>

CONTEXT:
- <IDs discovered, prerequisites checked, schema findings>

STEPS:
1. <METHOD> <endpoint> — <why>
   Payload: {<exact JSON with real IDs and correct field names>}
   Expected: <what this returns>

NOTES:
- <edge cases, gotchas, fallback strategies>

## Rules
- ALWAYS look up the API docs before writing payloads. Field names must be exact.
- Use EXACT names, emails, amounts, dates from the prompt. NEVER modify them.
- Put real IDs in the plan, not variables (except for IDs returned by previous steps).
- Minimize planned execution calls. Do NOT plan verification GETs.
- For action endpoints (payment, credit note, entitlements), specify they use query params.
"""

EXECUTOR_SYSTEM_PROMPT = """\
You are a Tripletex API executor. You receive the original task and a researched plan.

## Execution Flow
1. Read the original task and the plan. Check that the plan addresses the task correctly.
2. Review the CONTEXT section — it contains IDs and findings from API research.
3. Execute each step in order.
4. After each step, check the response. If it succeeded, continue. If it failed, diagnose and fix.
5. When all steps are done, stop.

## Rules
- Use EXACT values from the plan. Do NOT modify names, emails, amounts, or dates.
- The plan contains real IDs from the planner's research. Trust them unless an error says otherwise.
- After POST/PUT, save the returned ID if subsequent steps reference it.
- For action endpoints (entitlements, payment, credit note), put query params in the endpoint URL.
- When plan says "Payload: None", pass body="{}".

## Error Handling
- If a step fails with a validation error, read the message carefully.
- Fix the specific issue (e.g., add a missing required field) and retry ONCE.
- If the error mentions a missing prerequisite (e.g., bank account needed), use tripletex_get to investigate, then adapt.
- NEVER fix errors by changing values from the original task (names, emails, amounts).
- If you cannot resolve an error after one retry, skip the step and continue.

## Efficiency
- Every 4xx error hurts the score. Be careful with payloads.
- Do NOT make extra verification GET calls after successful operations.
- Do NOT add steps beyond what the plan specifies.
"""


class TripletexAgent:
    """Orchestrates the planner → executor pipeline for solving Tripletex tasks."""

    def __init__(self):
        self.planner_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
        )
        self.executor_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
        )
        # Planner: ReAct agent with read-only GET tool
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
        plan = await self._plan(request, config)

        # Step 2: Execute — give the executor the plan AND original task for context
        # The executor needs the original prompt to verify the plan makes sense
        # and to recover intelligently if something goes wrong
        executor_message = HumanMessage(
            content=f"## Original Task\n{request.prompt}\n\n## Execution Plan\n{plan}"
        )
        await self.executor.ainvoke(
            {"messages": [executor_message]},
            config=config,
        )

        return SolveResponse(status="completed")

    def _create_langfuse_handler(self) -> LangfuseCallbackHandler | None:
        """Create Langfuse callback handler if credentials are configured."""
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return LangfuseCallbackHandler()
        return None
