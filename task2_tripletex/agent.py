"""LangGraph-based Tripletex accounting agent with researcher → executor architecture."""

import asyncio
import base64
import logging
import os
import time

from anthropic import RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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

TOTAL_TIMEOUT = 270  # 4.5 min total budget
RESEARCHER_TIMEOUT = 60
EXECUTOR_TIMEOUT = 200

RESEARCHER_SYSTEM_PROMPT = """\
You are a research assistant for a Tripletex accounting agent. Your job is to investigate
the task and provide everything the executor needs to succeed on the FIRST try.

## Your Tools
1. **lookup_task_pattern(task_description)** — CALL FIRST. Returns the workflow, required fields, and common mistakes.
2. **tripletex_get(endpoint, params)** — Read the live API to find real IDs and check prerequisites.
3. **lookup_api_docs(search, semantic)** — Look up exact field names and schemas.
4. **search_tripletex_docs(query)** — Official Tripletex FAQs and guides.
5. **web_search(query)** — Last resort for unknown tasks.

## Your Workflow
1. Call lookup_task_pattern to understand the task type and workflow.
2. Identify the RISKS — what could cause the executor to fail:
   - Does it need a department ID? → GET /department
   - Does it involve invoices? → CHECK bank account: GET /ledger/account?number=1920&fields=id,version,bankAccountNumber
   - Does it need payment type IDs? → GET /travelExpense/paymentType or /invoice/paymentType
   - Does it reference existing entities? → GET them to find IDs
   - Does it involve products? → Do NOT set vatType (only default works)
   - Does it involve vouchers? → GET account IDs (never use account numbers)
3. Verify each risk using tripletex_get.
4. Output a RESEARCH BRIEF with your findings.

## Output Format
TASK TYPE: <e.g., "Create Invoice with Payment", "Create Employee as Admin">
TASK PATTERN: <key workflow steps from the pattern>

FINDINGS:
- Department ID: <N or "not needed">
- Bank account: <set/not set, account ID if needed>
- Payment type IDs: <if relevant>
- Existing entities found: <IDs>
- Prerequisites: <what needs to be created/set up first>

RISKS & WARNINGS:
- <specific things the executor must watch out for>
- <field name gotchas, required fields, etc.>

RECOMMENDED WORKFLOW:
1. <step with real IDs filled in>
2. <step>
...

## Rules
- Be FAST. Max 5 tool calls. Don't over-research.
- Focus on RISKS — things that would cause 4xx errors.
- The sandbox is FRESH — entities must be created, not searched for.
- Use EXACT values from the prompt. NEVER modify names, emails, amounts.
"""

EXECUTOR_SYSTEM_PROMPT = """\
You are a Tripletex API executor. You receive the original task and a research brief
with verified IDs, prerequisites, and warnings.

## Your Tools
- **tripletex_post/put/delete** — Execute API calls
- **tripletex_get** — Read API data (for error recovery)
- **lookup_api_docs(search, semantic)** — Look up exact field names
- **lookup_task_pattern(task_description)** — Workflow guidance
- **search_tripletex_docs(query)** — Official FAQs
- **web_search(query)** — Last resort

## How to Work
1. Read the research brief — it has verified IDs, prerequisites, and warnings.
2. Follow the RECOMMENDED WORKFLOW from the brief.
3. Execute each step using the tools.
4. Use EXACT values from the original task. NEVER modify names, emails, amounts.
5. For query-param endpoints (entitlements, payment, credit note), put params in the URL, pass body="{}".

## Error Recovery (max 1 retry per step)
1. Read the error message.
2. Call lookup_api_docs for the correct schema.
3. Fix the specific issue and retry ONCE.
4. If retry fails, SKIP and continue.

## Common Pitfalls (from the research brief warnings)
- Employee phone: "phoneNumberMobile" (NOT "phoneNumber")
- Order lines: "count" (NOT "quantity")
- Travel costs: "amountCurrencyIncVat" (NOT "amount"), "category" (NOT "description")
- Products: do NOT set vatType (only default id=6 works)
- Voucher accounts: ALWAYS {"id": N}, never {"number": N}
- Voucher amounts: amountGross + amountGrossCurrency, NEVER "amount"
- Invoice requires bank account on ledger account 1920

## Efficiency
- Every 4xx error hurts the score.
- Do NOT add extra verification GET calls.
- Stop after completing all steps.
"""


class TripletexAgent:
    """Orchestrates the researcher → executor pipeline for solving Tripletex tasks."""

    def __init__(self):
        # Researcher: Haiku (fast, focused on finding risks and IDs)
        self.researcher_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0,
            max_retries=2,
            timeout=30.0,
        )
        # Executor: Sonnet (smart, plans and executes with full context)
        self.executor_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
            max_retries=2,
            timeout=60.0,
        )
        # Fallback: Gemini Flash
        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_retries=2,
            timeout=60,
        )

        self.researcher = create_react_agent(
            model=self.researcher_llm,
            tools=PLANNER_TOOLS,
            prompt=RESEARCHER_SYSTEM_PROMPT,
        )
        self.executor = create_react_agent(
            model=self.executor_llm,
            tools=EXECUTOR_TOOLS,
            prompt=EXECUTOR_SYSTEM_PROMPT,
        )
        self.fallback_researcher = create_react_agent(
            model=self.fallback_llm,
            tools=PLANNER_TOOLS,
            prompt=RESEARCHER_SYSTEM_PROMPT,
        )
        self.fallback_executor = create_react_agent(
            model=self.fallback_llm,
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
                parts.append({"type": "text", "text": f"[Above image: {f.filename}]"})
            elif f.mime_type == "application/pdf":
                parts.append({
                    "type": "text",
                    "text": f"[PDF file: {f.filename}, {len(raw)} bytes — base64 available]",
                })
            else:
                try:
                    text = raw.decode("utf-8")
                    parts.append({"type": "text", "text": f"### {f.filename}\n```\n{text[:8000]}\n```"})
                except UnicodeDecodeError:
                    parts.append({
                        "type": "text",
                        "text": f"[Binary file: {f.filename} ({f.mime_type}, {len(raw)} bytes)]",
                    })
        return parts

    async def _run_with_fallback(self, primary, fallback, messages, config):
        """Run primary agent, fall back to Gemini on rate limit or server error."""
        try:
            return await primary.ainvoke(messages, config=config)
        except Exception as e:
            err = f"{type(e).__name__}: {e}".lower()
            if any(k in err for k in ["429", "rate", "503", "unavailable", "high demand", "overloaded", "servererror"]):
                logger.warning("Primary model unavailable (%s), falling back", type(e).__name__)
                try:
                    return await fallback.ainvoke(messages, config=config)
                except Exception as e2:
                    logger.warning("Fallback also failed: %s", e2)
                    raise e
            raise

    async def _research(self, request: SolveRequest, config: dict) -> str:
        """Use the researcher to investigate the task and gather context."""
        content_parts = [{"type": "text", "text": f"## Task Prompt\n\n{request.prompt}"}]
        if request.files:
            content_parts.append({"type": "text", "text": "\n## Attached Files\n"})
            content_parts.extend(self._extract_file_content(request))

        messages = {"messages": [HumanMessage(content=content_parts)]}
        research_config = {**config, "recursion_limit": 15}

        result = await asyncio.wait_for(
            self._run_with_fallback(
                self.researcher, self.fallback_researcher, messages, research_config
            ),
            timeout=RESEARCHER_TIMEOUT,
        )

        last_message = result["messages"][-1]
        brief = last_message.content
        if isinstance(brief, list):
            brief = "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in brief
            )
        logger.info("Research brief:\n%s", brief)
        return brief

    async def solve(self, request: SolveRequest) -> SolveResponse:
        """Run the researcher → executor pipeline."""
        start_time = time.time()

        client = TripletexClient(
            base_url=request.tripletex_credentials.base_url,
            session_token=request.tripletex_credentials.session_token,
        )
        set_client(client)

        config: dict = {}
        langfuse_handler = self._create_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        # Step 1: Research
        try:
            brief = await self._research(request, config)
        except asyncio.TimeoutError:
            logger.warning("Researcher timed out after %ds", RESEARCHER_TIMEOUT)
            brief = "Research timed out. Execute based on the task prompt alone."
        except Exception as e:
            logger.exception("Researcher failed: %s", e)
            brief = f"Research failed: {e}. Execute based on the task prompt alone."

        # Check remaining time
        elapsed = time.time() - start_time
        remaining = TOTAL_TIMEOUT - elapsed
        if remaining < 30:
            logger.warning("Only %ds left after research — skipping execution", remaining)
            return SolveResponse(status="completed")

        # Step 2: Execute
        executor_timeout = min(EXECUTOR_TIMEOUT, remaining - 10)
        executor_config = {**config, "recursion_limit": 40}
        executor_message = HumanMessage(
            content=f"## Original Task\n{request.prompt}\n\n## Research Brief\n{brief}"
        )

        try:
            await asyncio.wait_for(
                self._run_with_fallback(
                    self.executor, self.fallback_executor,
                    {"messages": [executor_message]}, executor_config
                ),
                timeout=executor_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Executor timed out after %ds — returning partial results", executor_timeout)
        except Exception as e:
            if "recursion" in str(e).lower():
                logger.warning("Executor hit recursion limit — returning partial results")
            else:
                logger.exception("Executor error — returning completed for partial scoring")

        total_time = time.time() - start_time
        logger.info("Task completed in %.1fs", total_time)
        return SolveResponse(status="completed")

    def _create_langfuse_handler(self) -> LangfuseCallbackHandler | None:
        """Create Langfuse callback handler if credentials are configured."""
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return LangfuseCallbackHandler()
        return None
