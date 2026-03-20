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
1. **lookup_api_docs(search)** — Look up exact API schemas when unsure about field names.
2. **tripletex_get(endpoint, params)** — Read-only API access to find real IDs.

## Workflow
1. Parse the prompt to extract: task type, entity names, field values.
2. Use tripletex_get to find IDs (GET /department, GET /employee, GET /customer, etc.).
3. If unsure about field names for an endpoint, use lookup_api_docs.
4. Output a concrete plan with real IDs and correct field names.

## Common Endpoints & Verified Fields

POST /employee: {firstName, lastName, email, phoneNumberMobile, department: {"id": N}}
- department is REQUIRED. Get via GET /department?fields=id,name&count=1
- Phone field is "phoneNumberMobile" (NOT "phoneNumber")
PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=N&template=ALL_PRIVILEGES
- No body. Query params only. Grants admin (kontoadministrator) role.

POST /customer: {name, email, phoneNumber, isCustomer: true, organizationNumber: "string"}
- If the task includes an org number (org.nr), set organizationNumber on the customer.

POST /product: {name, number: "string", priceExcludingVatCurrency: N}
- If the task specifies product numbers, create products first, then reference them in order lines.

POST /order: {customer: {"id": N}, orderDate, deliveryDate, orderLines: [{product: {"id": N}, description, count, unitPriceExcludingVatCurrency}]}
- If products were created, include product: {"id": N} in each order line.
POST /invoice: {invoiceDate, invoiceDueDate, customer: {"id": N}, orders: [{"id": N}]}
PUT /invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=N&paidAmount=N (query params, NO body)
PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD (query params)

POST /travelExpense: {employee: {"id": N}, title, date, department: {"id": N}}
POST /travelExpense/cost: {travelExpense: {"id": N}, date, amountCurrencyIncVat: N, paymentType: {"id": N}, category: "text", isPaidByEmployee: true}
- paymentType REQUIRED — get via GET /travelExpense/paymentType
- Field is amountCurrencyIncVat NOT amount. Field is category NOT description.

POST /project: {name, startDate, customer: {"id": N}, projectManager: {"id": N}}
POST /department: {name, departmentNumber}
POST /contact: {firstName, lastName, email, customer: {"id": N}}

## Output Format
TASK SUMMARY: <one line>

CONTEXT:
- <IDs found, prerequisites checked>

STEPS:
1. <METHOD> <endpoint>
   Payload: {<exact JSON with real IDs>}

NOTES:
- <any gotchas>

## Rules
- Use EXACT values from the prompt. NEVER modify names, emails, amounts.
- Real IDs in the plan (except IDs returned by previous steps — use <from_step_N>).
- Minimize tool calls. Use the reference above when possible, only use lookup_api_docs for unfamiliar endpoints.
- For action endpoints (payment, credit note, entitlements), use query params not body.
"""

EXECUTOR_SYSTEM_PROMPT = """\
You are a Tripletex API executor. You MUST follow the plan exactly.

## CRITICAL: Follow the plan step by step
- Execute ONLY the steps listed in the plan, in the EXACT order given.
- Use the EXACT endpoint, method, and payload from each step.
- Do NOT skip steps. Do NOT add steps. Do NOT reorder steps.
- Do NOT try alternative approaches or shortcuts. The plan was researched and is correct.

## Payload Rules
- Copy field names EXACTLY from the plan. Common mistakes to avoid:
  - Employee phone: use "phoneNumberMobile" (NOT "phoneNumber")
  - Order lines: use "count" (NOT "quantity")
  - Travel costs: use "amountCurrencyIncVat" (NOT "amount")
  - Travel costs: use "category" (NOT "description")
- Use the EXACT values from the plan. Do NOT modify names, emails, amounts, or dates.
- After POST/PUT, save the returned ID for subsequent steps that reference it.

## URL Rules
- For entitlements: PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=N&template=X
  - The endpoint path is exactly "/employee/entitlement/:grantEntitlementsByTemplate"
  - employeeId goes in query params, NOT in the path
- For payment: PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=N
- For credit note: PUT /invoice/{id}/:createCreditNote?date=X
- Pass body="{}" for endpoints that use query params only.

## Error Handling
- If a step fails, read the error message. Fix the SPECIFIC issue and retry ONCE.
- If the error mentions wrong field names, use lookup_api_docs to find the correct schema.
- Do NOT try a completely different approach. The plan's endpoint and method are correct.
- If retry fails, move to the next step.
- NEVER modify values from the original task (names, emails, amounts).

## Efficiency
- Every 4xx error hurts the score. Follow the plan precisely to avoid errors.
- Do NOT make extra GET calls. Do NOT add verification steps.
- Stop after the last planned step.
"""


class TripletexAgent:
    """Orchestrates the planner → executor pipeline for solving Tripletex tasks."""

    def __init__(self):
        self.planner_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0,
        )
        self.executor_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
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
