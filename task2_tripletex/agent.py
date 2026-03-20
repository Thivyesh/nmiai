"""LangGraph-based Tripletex accounting agent with planner → executor architecture."""

import asyncio
import base64
import logging
import os
import time

from anthropic import RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
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

# Total time budget: 4.5 minutes (leave 30s buffer before 5-min competition timeout)
TOTAL_TIMEOUT = 270
# Planner gets max 90 seconds
PLANNER_TIMEOUT = 90
# Executor gets the rest
EXECUTOR_TIMEOUT = 180

PLANNER_SYSTEM_PROMPT = """\
You are an expert accounting task planner for Tripletex, a Norwegian accounting system.

## Your Tools (use in this priority order)
1. **lookup_task_pattern(task_description)** — CALL FIRST. Returns workflow, scoring criteria, and common mistakes.
2. **tripletex_get(endpoint, params)** — Read-only API access to find real IDs.
3. **lookup_api_docs(search, semantic)** — Look up exact field names and schemas. Use semantic=True for non-English terms.
4. **search_tripletex_docs(query)** — Search official Tripletex developer FAQs.
5. **web_search(query)** — Last resort web search.

## Workflow
1. Call lookup_task_pattern with the task description to get the workflow pattern.
2. Use tripletex_get to find real IDs (departments, accounts, payment types).
3. If unfamiliar task: use lookup_api_docs and search_tripletex_docs to discover the workflow.
4. Output a concrete plan with real IDs and exact payloads.

## CRITICAL RULES
- Every entity mentioned in the prompt must be CREATED as a separate record.
- Every field value in the prompt WILL be checked. Include ALL of them.
- Use EXACT values from the prompt. NEVER modify names, emails, amounts.
- ALWAYS use account {"id": N} not {"number": N} — look up the ID first.
- For voucher postings: use amountGross + amountGrossCurrency (NEVER "amount").
- ONLY plan steps for endpoints you are CONFIDENT exist. Do NOT guess endpoints.
- Keep the plan to ESSENTIAL steps only. Fewer steps = fewer errors = higher score.

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
6. If a step references an endpoint you're not sure exists, SKIP it rather than guessing.

## Error Recovery (max 1 retry per step)
When a step fails:
1. Read the error message.
2. Call **lookup_api_docs** for the correct schema.
3. Fix the specific issue and retry ONCE.
4. If retry fails, SKIP and move to the next step. Do NOT spiral.

## Common Error Fixes
- "Request mapping failed" → wrong field names. Look up schema.
- "Feltet må fylles ut" → missing required field.
- "Object not found" / 404 → wrong ID or URL format.
- "Kunde mangler" → posting needs customer ID.
- Account references: ALWAYS {"id": N}, never {"number": N}.
- Voucher amounts: amountGross + amountGrossCurrency, NEVER "amount".

## Efficiency
- Every 4xx error hurts the score. Be precise.
- Do NOT add extra verification GET calls.
- Stop after completing all planned steps.
"""


class TripletexAgent:
    """Orchestrates the planner → executor pipeline for solving Tripletex tasks."""

    def __init__(self):
        # Planner: Gemini Flash (300 RPM, 1M TPM — no rate limit issues)
        self.planner_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_retries=2,
            timeout=30,
        )
        # Executor: Gemini Pro (strong reasoning + tool use, 300 RPM)
        self.executor_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            max_retries=2,
            timeout=60,
        )
        # Fallback: Ollama Qwen for rate-limited scenarios
        self.fallback_llm = ChatOllama(
            model="qwen2.5:32b",
            temperature=0,
            num_ctx=16384,
        )

        self.planner = create_react_agent(
            model=self.planner_llm,
            tools=PLANNER_TOOLS,
            prompt=PLANNER_SYSTEM_PROMPT,
        )
        self.executor = create_react_agent(
            model=self.executor_llm,
            tools=EXECUTOR_TOOLS,
            prompt=EXECUTOR_SYSTEM_PROMPT,
        )
        self.fallback_planner = create_react_agent(
            model=self.fallback_llm,
            tools=PLANNER_TOOLS,
            prompt=PLANNER_SYSTEM_PROMPT,
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

    async def _run_with_fallback(self, primary, fallback, messages, config):
        """Run primary agent, fall back to Ollama on rate limit."""
        try:
            return await primary.ainvoke(messages, config=config)
        except RateLimitError as e:
            logger.warning("Rate-limited, falling back to Ollama: %s", e)
            return await fallback.ainvoke(messages, config=config)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                logger.warning("Rate-limited (generic), falling back to Ollama: %s", e)
                return await fallback.ainvoke(messages, config=config)
            raise

    async def _plan(self, request: SolveRequest, config: dict) -> str:
        """Use the planner agent to research the API and create an execution plan."""
        content_parts = [{"type": "text", "text": f"## Task Prompt\n\n{request.prompt}"}]

        if request.files:
            content_parts.append({"type": "text", "text": "\n## Attached Files\n"})
            content_parts.extend(self._extract_file_content(request))

        messages = {"messages": [HumanMessage(content=content_parts)]}
        planner_config = {**config, "recursion_limit": 25}

        result = await asyncio.wait_for(
            self._run_with_fallback(
                self.planner, self.fallback_planner, messages, planner_config
            ),
            timeout=PLANNER_TIMEOUT,
        )

        last_message = result["messages"][-1]
        plan = last_message.content
        # Gemini returns content as list of dicts; extract text
        if isinstance(plan, list):
            plan = "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in plan
            )
        logger.info("Plan:\n%s", plan)
        return plan

    async def solve(self, request: SolveRequest) -> SolveResponse:
        """Run the planner → executor pipeline to solve a Tripletex task."""
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

        # Step 1: Plan (with timeout)
        try:
            plan = await self._plan(request, config)
        except asyncio.TimeoutError:
            logger.warning("Planner timed out after %ds", PLANNER_TIMEOUT)
            return SolveResponse(status="completed")
        except Exception as e:
            logger.exception("Planner failed: %s", e)
            return SolveResponse(status="completed")

        # Check remaining time
        elapsed = time.time() - start_time
        remaining = TOTAL_TIMEOUT - elapsed
        if remaining < 30:
            logger.warning("Only %ds left after planning — skipping execution", remaining)
            return SolveResponse(status="completed")

        # Step 2: Execute (with timeout based on remaining time)
        executor_timeout = min(EXECUTOR_TIMEOUT, remaining - 10)  # 10s buffer
        executor_config = {**config, "recursion_limit": 40}
        executor_message = HumanMessage(
            content=f"## Original Task\n{request.prompt}\n\n## Execution Plan\n{plan}"
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
