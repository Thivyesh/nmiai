"""LangGraph-based Tripletex accounting agent with planner → executor architecture."""

import base64
import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.prebuilt import create_react_agent

from task2_tripletex.api_reference import API_REFERENCE
from task2_tripletex.models import SolveRequest, SolveResponse
from task2_tripletex.tools import (
    ALL_TOOLS,
    PLANNER_TOOLS,
    TripletexClient,
    set_client,
)

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are an expert accounting task planner for Tripletex, a Norwegian accounting system.

You receive a task prompt (possibly in Norwegian or other languages) and have READ-ONLY
access to the Tripletex API via the tripletex_get tool. Use it to:
- Look up existing entities (customers, employees, departments, etc.)
- Discover required fields and valid values
- Find IDs needed for the execution plan

## Workflow
1. Parse the prompt to extract: task type, entity names, field values, relationships.
2. Use tripletex_get to look up any information you need:
   - Always GET /department first if the task involves employees (department is required).
   - GET /customer?name=X if the task references an existing customer.
   - GET /employee if you need an employee ID (e.g., for project manager).
   - GET /invoice/paymentType if the task involves payments.
   - GET /ledger/account?number=1920&fields=id,version,bankAccountNumber if creating invoices.
3. Produce a concrete execution plan with ACTUAL IDs (not placeholders).

## Output Format (after you've done your research)
When you have all the information, output your final plan in this format:

TASK SUMMARY: <one line>

STEPS:
1. <METHOD> <endpoint> — <why>
   Payload: {<exact JSON with real IDs>}
   Expected: <what this returns>

2. <METHOD> <endpoint> — <why>
   Payload: {<exact JSON>}

NOTES:
- <edge cases or special considerations>

## Rules
- Only use tripletex_get for research. Do NOT create, modify, or delete anything.
- Be precise with field names — they must match the API exactly.
- Include ALL required fields. Check the API reference for what's required.
- Minimize planned API calls — efficiency is scored. Do NOT plan verification GETs.
- Use the EXACT names, emails, amounts, dates from the prompt. NEVER modify them.
- Put real IDs in the plan (e.g., department_id=900006), not variables.
- For action endpoints (payment, credit note, entitlements), note that they use query params.
- The executor CANNOT do GET calls — all lookups must happen here in planning.

## API Reference
""" + API_REFERENCE

EXECUTOR_SYSTEM_PROMPT = """\
You are a Tripletex API executor. Follow the plan exactly, step by step.

## Rules
- Execute each step in order using the tools (tripletex_post, tripletex_put, tripletex_delete).
- The plan contains REAL IDs from the planner's research. Use them exactly.
- Use EXACT values from the plan. Do NOT modify names, emails, or amounts.
- After POST/PUT, note the returned ID if subsequent steps need it (the plan will say "use returned ID").
- If a call fails, read the error. Only retry if you can fix the specific issue. NEVER change values from the task.
- For action endpoints (entitlements, payment, credit note), put query params in the endpoint URL.
- When plan says "Payload: None", pass body="{}".
- Stop after the last step. No extra calls, no summaries, no verification.
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
        # Executor: ReAct agent with write tools (POST, PUT, DELETE)
        self.executor = create_react_agent(
            model=self.executor_llm,
            tools=ALL_TOOLS,
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

        # Step 2: Execute (only the plan, not the original prompt)
        executor_message = HumanMessage(content=f"Execute this plan:\n\n{plan}")
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
