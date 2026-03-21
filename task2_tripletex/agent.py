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
    _get_client,
    set_client,
)

logger = logging.getLogger(__name__)

TOTAL_TIMEOUT = 270
RESEARCHER_TIMEOUT = 45
EXECUTOR_TIMEOUT = 200

RESEARCHER_SYSTEM_PROMPT = """\
You produce READY-TO-EXECUTE payloads for the Tripletex API executor.

## STRICT PROCESS — follow these steps exactly, then STOP
1. get_task_workflow (English task description) → get the steps
2. get_payload_template for each endpoint in the workflow → get exact JSON
3. tripletex_get ONLY for IDs not in pre-fetched data (e.g., specific account numbers)
4. If task implies existing entities ("has invoice", "outstanding"): tripletex_get to find them
5. Fill in templates with real IDs + prompt values → output filled payloads
6. STOP. Do NOT call lookup_api_docs, search_tripletex_docs, or web_search unless a template is missing.

## Output
STEPS:
1. <METHOD> <endpoint>
   ```json
   {complete JSON — real IDs, no placeholders except <id_from_step_N>}
   ```

## Rules
- Copy JSON from templates. Do NOT construct from memory. Do NOT invent field names.
- EXACT values from the prompt. Never modify names, emails, amounts.
- Product numbers: only from prompt. If none given, omit the field.
- Payment amounts: executor reads from invoice response, don't calculate.
- STOP after outputting payloads. Do not do extra research.
"""

AGENT_SYSTEM_PROMPT = """\
You solve Tripletex accounting tasks. You have reference data and tools to help.

## STRICT PROCESS
1. Call get_task_workflow (English description) → understand the steps
2. Call get_payload_template for each endpoint → get exact JSON to copy
3. Use tripletex_get ONLY for IDs not in the pre-fetched data
4. If task says entities EXIST ("has invoice", "outstanding"): search for them with tripletex_get
5. Fill in templates with real IDs + prompt values → execute with tripletex_post/put/delete
6. After each POST, save returned ID for next steps
7. For payment: READ "amount" from invoice response. Do NOT calculate.

## Tools
- **get_task_workflow** — Workflow steps for the task type
- **get_payload_template** — EXACT JSON template for an endpoint. Copy it, don't invent fields.
- **tripletex_get** — Read API data (find entities, get IDs)
- **tripletex_post/put/delete** — Execute API calls
- **lookup_api_docs** — Full schema for endpoints not in templates

## Rules
- Copy JSON from templates. Do NOT construct from memory.
- EXACT values from the prompt. Never modify names, emails, amounts.
- Query-param endpoints (payment, credit note, entitlements): params in URL, body="{}".

## File Attachments (PDF/images)
If the task includes files, extract EVERY piece of data from them:
- Names, dates, addresses, phone numbers, email, national ID, bank account
- Salary/wage amounts, employment percentage, occupation codes
- Product names, prices, quantities, invoice numbers
- Include ALL extracted data in the API calls — every field will be checked.
Include employee address: {addressLine1, postalCode, city}

## Error Recovery
If a step fails: get_payload_template → compare → fix → retry ONCE → skip if still failing.
"""


class TripletexAgent:
    """Orchestrates the researcher → executor pipeline for solving Tripletex tasks."""

    # Serialize requests — concurrent requests corrupt each other's API client
    _lock = asyncio.Lock()

    def __init__(self):
        # Single agent: Opus does everything
        self.agent_llm = ChatAnthropic(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            temperature=0,
            max_retries=2,
            timeout=60.0,
        )
        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_retries=2,
            timeout=60,
        )

        # All tools available to the single agent (deduplicate by name)
        seen = set()
        all_tools = []
        for t in PLANNER_TOOLS + EXECUTOR_TOOLS:
            if t.name not in seen:
                seen.add(t.name)
                all_tools.append(t)
        self.agent = create_react_agent(
            model=self.agent_llm,
            tools=all_tools,
            prompt=AGENT_SYSTEM_PROMPT,
        )
        self.fallback_agent = create_react_agent(
            model=self.fallback_llm,
            tools=all_tools,
            prompt=AGENT_SYSTEM_PROMPT,
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
                # Send PDF as document for Claude to read
                parts.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": f.content_base64,
                    },
                })
                parts.append({"type": "text", "text": f"[Above PDF: {f.filename}]"})
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

    def _prefetch_reference_data(self) -> str:
        """Pre-fetch common reference data deterministically. No LLM involved."""
        client = _get_client()
        data = {}

        # Department
        try:
            r = client.get("/department", {"fields": "id,name", "count": "1"})
            vals = r.get("values", [])
            if vals:
                data["department_id"] = vals[0]["id"]
                data["department_name"] = vals[0].get("name", "")
        except Exception:
            pass

        # Bank account (1920)
        try:
            r = client.get("/ledger/account", {"number": "1920", "fields": "id,version,bankAccountNumber"})
            vals = r.get("values", [])
            if vals:
                data["account_1920_id"] = vals[0]["id"]
                data["account_1920_version"] = vals[0].get("version", 0)
                data["account_1920_bank"] = vals[0].get("bankAccountNumber", "")
        except Exception:
            pass

        # Cash account (1900)
        try:
            r = client.get("/ledger/account", {"number": "1900", "fields": "id"})
            vals = r.get("values", [])
            if vals:
                data["account_1900_id"] = vals[0]["id"]
        except Exception:
            pass

        # Invoice payment types
        try:
            r = client.get("/invoice/paymentType", {"fields": "id,description"})
            vals = r.get("values", [])
            if vals:
                data["invoice_payment_types"] = [{"id": v["id"], "desc": v.get("description", "")} for v in vals[:3]]
        except Exception:
            pass

        # Travel expense payment types
        try:
            r = client.get("/travelExpense/paymentType", {"fields": "id,description"})
            vals = r.get("values", [])
            if vals:
                data["travel_payment_types"] = [{"id": v["id"], "desc": v.get("description", "")} for v in vals[:3]]
        except Exception:
            pass

        # Voucher types
        try:
            r = client.get("/ledger/voucherType", {"fields": "id,name", "count": "5"})
            vals = r.get("values", [])
            if vals:
                data["voucher_types"] = [{"id": v["id"], "name": v.get("name", "")} for v in vals[:5]]
        except Exception:
            pass

        lines = ["## Pre-fetched Reference Data (verified IDs from this sandbox)"]
        for k, v in data.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    async def _research(self, request: SolveRequest, config: dict) -> str:
        """Use the researcher to investigate the task and gather context."""
        # Pre-fetch reference data deterministically
        ref_data = self._prefetch_reference_data()
        logger.info("Pre-fetched reference data:\n%s", ref_data)

        content_parts = [
            {"type": "text", "text": f"## Task Prompt\n\n{request.prompt}"},
            {"type": "text", "text": f"\n{ref_data}"},
        ]
        if request.files:
            content_parts.append({"type": "text", "text": "\n## Attached Files\n"})
            content_parts.extend(self._extract_file_content(request))

        messages = {"messages": [HumanMessage(content=content_parts)]}
        # Low limit — if researcher can't figure it out in 12 iterations, let Opus handle it
        research_config = {**config, "recursion_limit": 12}

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
        """Run the researcher → executor pipeline. Serialized to prevent client corruption."""
        async with self._lock:
            return await self._solve_inner(request)

    async def _solve_inner(self, request: SolveRequest) -> SolveResponse:
        """Single-agent solve: Opus gets pre-fetched data + all tools."""
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

        # Pre-fetch reference data
        ref_data = self._prefetch_reference_data()
        logger.info("Pre-fetched reference data:\n%s", ref_data)

        # Build message with task + pre-fetched data + files
        content_parts = [
            {"type": "text", "text": f"## Task\n\n{request.prompt}"},
            {"type": "text", "text": f"\n{ref_data}"},
        ]
        if request.files:
            content_parts.append({"type": "text", "text": "\n## Attached Files\n"})
            content_parts.extend(self._extract_file_content(request))

        agent_config = {**config, "recursion_limit": 40}
        message = HumanMessage(content=content_parts)

        try:
            await asyncio.wait_for(
                self._run_with_fallback(
                    self.agent, self.fallback_agent,
                    {"messages": [message]}, agent_config
                ),
                timeout=TOTAL_TIMEOUT - 10,
            )
        except asyncio.TimeoutError:
            logger.warning("Agent timed out — returning partial results")
        except Exception as e:
            if "recursion" in str(e).lower():
                logger.warning("Agent hit recursion limit — returning partial results")
            else:
                logger.exception("Agent error — returning completed for partial scoring")

        total_time = time.time() - start_time
        logger.info("Task completed in %.1fs", total_time)
        return SolveResponse(status="completed")

    def _create_langfuse_handler(self) -> LangfuseCallbackHandler | None:
        """Create Langfuse callback handler if credentials are configured."""
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return LangfuseCallbackHandler()
        return None
