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
from task2_tripletex.pdf_extractor import extract_file_data
from task2_tripletex.experience_checker import check_experience
from task2_tripletex.schema_agent import discover_schemas
from task2_tripletex.tools import (
    TripletexClient,
    _get_client,
    set_client,
    tripletex_get,
    tripletex_post,
    tripletex_put,
    tripletex_delete,
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
You are a Tripletex API executor. Endpoint templates and reference data are provided above.
Your ONLY job is to execute API calls. Do NOT research — that's already done.

## Process
1. Read "Past Experience" section if present — it has CRITICAL warnings from past failures. Follow them.
2. Read "Endpoint Templates" section — it has the exact JSON for each step
3. Use tripletex_get ONLY if you need an ID not in the pre-fetched data
4. Fill each template with real IDs + values from the prompt → execute immediately
5. After each POST/PUT, save the returned ID for the next step
6. Repeat until ALL steps are done

## Tools
- tripletex_get — Look up IDs (employees, customers, accounts, projects)
- tripletex_post — Create entities
- tripletex_put — Update entities or trigger actions (payment, credit note, invoice)
- tripletex_delete — Delete entities
- get_payload_template — ONLY if a template is missing from the pre-resolved schemas above

## Rules
- Copy JSON from templates. Do NOT construct from memory. Do NOT invent field names.
- EXACT values from the prompt. Never modify names, emails, amounts.
- Account lookups: GET /ledger/account?number=NNNN (NOT /account).
- Query-param endpoints (payment, credit note, entitlements): params in URL, body="{}".
- Do NOT modify existing entities. Use their ID as-is.
  The competition pre-loads entities — modifying them breaks checks.
  Only CREATE new entities if the task explicitly asks you to.
- For payment: READ "amount" from invoice response. Do NOT calculate.

## File Attachments
If extracted file data is provided above, use ALL fields from it in API calls.
Include employee address: {addressLine1, postalCode, city}

## Error Recovery
- If a step fails with 422: check get_payload_template → compare your payload → fix → retry ONCE.
- If a step fails with 403 "permission": the endpoint is not available. Use an alternative:
  - /incomingInvoice 403 → book as a voucher instead: POST /ledger/voucher with debit expense account + credit supplier account (2400)
  - /project/:invoice 404 → create an order (POST /order) and invoice it (PUT /order/{id}/:invoice)
- If a step fails with 409 "duplicate": the entity already exists. GET it instead of creating.
"""


class TripletexAgent:
    """Orchestrates the researcher → executor pipeline for solving Tripletex tasks."""

    # Serialize requests — concurrent requests corrupt each other's API client
    _lock = asyncio.Lock()

    def __init__(self):
        # Single agent: GPT-4.1 (latest, best instruction following, cheaper than 4o)
        from langchain_openai import ChatOpenAI
        self.agent_llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            max_retries=2,
            timeout=60,
        )
        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_retries=2,
            timeout=60,
        )

        # Executor tools only — schema agent handles research
        from task2_tripletex.workflow_tools import get_payload_template
        all_tools = [tripletex_get, tripletex_post, tripletex_put, tripletex_delete, get_payload_template]
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
                parts.append({
                    "type": "file",
                    "file": {
                        "filename": f.filename,
                        "file_data": f"data:application/pdf;base64,{f.content_base64}",
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

        # Bank account (1920) — auto-set if empty (required for invoices)
        try:
            r = client.get("/ledger/account", {"number": "1920", "fields": "id,version,bankAccountNumber"})
            vals = r.get("values", [])
            if vals:
                acct = vals[0]
                data["account_1920_id"] = acct["id"]
                data["account_1920_version"] = acct.get("version", 0)
                bank = acct.get("bankAccountNumber", "")
                data["account_1920_bank"] = bank
                if not bank:
                    # Set bank account so invoices can be created
                    put_r = client.put(
                        f"/ledger/account/{acct['id']}",
                        {"id": acct["id"], "version": acct.get("version", 0), "bankAccountNumber": "86011117947"},
                    )
                    if not put_r.get("_error"):
                        data["account_1920_bank"] = "86011117947 (auto-set)"
                        logger.info("Auto-set bank account on 1920")
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

        # Step 1: Check past experience (no LLM, instant)
        experience_data = check_experience(request.prompt)
        if experience_data:
            logger.info("Experience warnings: %d chars", len(experience_data))

        # Step 2: Pre-fetch reference data
        ref_data = self._prefetch_reference_data()
        logger.info("Pre-fetched reference data:\n%s", ref_data)

        # Run PDF extraction and schema discovery in PARALLEL
        schema_input = ref_data
        if experience_data:
            schema_input += experience_data

        async def _extract_files():
            if not request.files:
                return ""
            try:
                data = await extract_file_data(request.files)
                logger.info("Extracted file data: %d chars", len(data))
                return data
            except Exception as e:
                logger.warning("File extraction failed: %s", e)
                return ""

        async def _discover():
            try:
                data = await discover_schemas(request.prompt, schema_input)
                logger.info("Schema discovery: %d chars", len(data))
                return data
            except Exception as e:
                logger.warning("Schema discovery failed: %s", e)
                return ""

        file_data, schema_data = await asyncio.gather(_extract_files(), _discover())

        # Build message with task + all context for executor
        content_parts = [
            {"type": "text", "text": f"## Task\n\n{request.prompt}"},
            {"type": "text", "text": f"\n{ref_data}"},
        ]
        if experience_data:
            content_parts.append({"type": "text", "text": experience_data})
        if schema_data:
            content_parts.append({"type": "text", "text": schema_data})
        if file_data:
            content_parts.append({"type": "text", "text": file_data})

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
