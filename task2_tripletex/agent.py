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

You have three tools:
1. **lookup_task_pattern(task_description)** — Returns workflow patterns and scoring criteria for known task types.
2. **lookup_api_docs(search, semantic)** — Look up exact API schemas, field names, and parameters.
3. **tripletex_get(endpoint, params)** — Read-only API access to find real IDs and explore the API.

## Workflow

### Step 1: Understand the task
Parse the prompt (may be in Norwegian, English, Spanish, German, French, Portuguese, or Nynorsk).
Identify: what entities to create/modify/delete, what field values are given.

### Step 2: Look up the task pattern
Call lookup_task_pattern with the task description. If a pattern matches, follow it.

### Step 3: If NO pattern matches — DISCOVER the workflow yourself
This is critical. For unfamiliar tasks:
a) Use lookup_api_docs(search, semantic=True) to find relevant endpoints.
   Try the entity name: "employee", "voucher", "dimension", "salary", etc.
b) Use lookup_api_docs to read the POST schema — find ALL writable fields.
c) Use tripletex_get to explore: GET the endpoint with ?fields=* to see response structure.
d) Check for prerequisite entities: does this endpoint need a customer? employee? department?
e) Build the workflow step by step from what you discover.

### Step 4: Look up real IDs
Use tripletex_get to find: departments, employees, customers, payment types, VAT types, accounts.
The sandbox starts FRESH — most entities won't exist and must be created.

### Step 5: Verify field names
Before writing the plan, use lookup_api_docs for EVERY endpoint you plan to call.
Get the exact field names from the schema. Do NOT guess field names.

### Step 6: Output the plan
Follow the output format below with real IDs and exact field names.

## CRITICAL RULES
- Every entity mentioned in the prompt must be CREATED as a separate record.
- Every field value in the prompt WILL be checked. Include ALL of them.
- Use EXACT values from the prompt. NEVER modify names, emails, amounts.
- The sandbox is FRESH — no pre-existing data except the account owner.
- When unsure, ALWAYS look up the API docs rather than guessing.

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
You are a Tripletex API executor. You receive a plan and execute it.

## Execution
1. Follow the plan step by step, in order.
2. Use the EXACT endpoint, method, and payload from each step.
3. After POST/PUT, save the returned ID for subsequent steps.
4. Use EXACT values from the plan and original task. NEVER modify names, emails, amounts.

## Payload Rules
- Copy field names EXACTLY from the plan. Common corrections:
  - Employee phone: "phoneNumberMobile" (NOT "phoneNumber")
  - Order lines: "count" (NOT "quantity")
  - Travel costs: "amountCurrencyIncVat" (NOT "amount"), "category" (NOT "description")
- For query-param endpoints (entitlements, payment, credit note), put params in the URL and pass body="{}".

## Error Recovery
When a step fails:
1. Read the error message carefully.
2. If it says a field is missing or wrong:
   - Use lookup_api_docs to find the correct schema for that endpoint.
   - Fix the specific field and retry ONCE.
3. If it says a prerequisite is missing (e.g., "bank account needed"):
   - Use tripletex_get to investigate.
   - Create the prerequisite, then retry the original step.
4. If the plan seems incomplete or wrong for the task:
   - Use lookup_task_pattern to understand the expected workflow.
   - Adapt the remaining steps based on what you learn.
5. NEVER modify values from the original task (names, emails, amounts).
6. After one failed retry, move to the next step.

## Efficiency
- Every 4xx error hurts the score. Be precise with payloads.
- Do NOT add extra GET calls or verification steps beyond what's needed.
- Stop after completing all planned steps (or adapted steps).
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
