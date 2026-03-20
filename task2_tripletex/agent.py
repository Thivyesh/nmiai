"""LangGraph-based Tripletex accounting agent with planner → executor architecture."""

import base64
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.prebuilt import create_react_agent

from task2_tripletex.api_reference import API_REFERENCE
from task2_tripletex.models import SolveRequest, SolveResponse
from task2_tripletex.tools import ALL_TOOLS, TripletexClient, set_client

PLANNER_SYSTEM_PROMPT = """\
You are an expert accounting task planner for Tripletex, a Norwegian accounting system.
You receive a task prompt (in Norwegian, English, or other languages) and must create
a precise, step-by-step execution plan.

## Your Job
1. Parse the prompt to extract: task type, entity names, field values, relationships.
2. Identify which Tripletex API endpoints to call, in what order.
3. Specify the exact HTTP method, endpoint, and payload for each step.
4. Note any dependencies between steps (e.g., "use customer_id from step 1").

## Output Format
Return a structured plan like this:

TASK SUMMARY: <one line describing what needs to be done>

STEPS:
1. <METHOD> <endpoint> — <why>
   Payload: {<exact JSON fields to send>}
   Save: <variable_name> = response.value.id

2. <METHOD> <endpoint> — <why>
   Payload: {<fields, referencing saved variables like ${customer_id}>}

NOTES:
- <any special considerations, edge cases, or verification steps>

## Rules
- Be precise with field names — they must match the API exactly.
- Include ALL required and relevant fields from the prompt.
- Minimize the number of API calls — efficiency is scored.
- Do NOT include unnecessary GET calls unless needed to find existing resources.
- For admin/kontoadministrator role, use userType="EXTENDED".
- Always use the exact names, emails, amounts from the prompt — do not modify them.

## API Reference
""" + API_REFERENCE

EXECUTOR_SYSTEM_PROMPT = """\
You are an expert Tripletex API executor. You receive a plan and execute it step by step
using the available tools.

## Guidelines
- Follow the plan precisely. Execute each step in order.
- Use the exact field names and values from the plan.
- After a POST/PUT, note the returned ID for use in subsequent steps.
- If a step fails, read the error message carefully and adjust the payload — do not blindly retry.
- Do NOT make API calls that are not in the plan unless a step fails and you need to adapt.
- Norwegian characters (æ, ø, å) work fine — send as-is.
- Every 4xx error hurts the efficiency score, so get it right the first time.
- When done, do not make any extra verification calls unless the plan says to.
"""


class TripletexAgent:
    """Orchestrates the planner → executor pipeline for solving Tripletex tasks."""

    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
        )
        self.executor = create_react_agent(
            model=self.llm,
            tools=ALL_TOOLS,
            prompt=EXECUTOR_SYSTEM_PROMPT,
        )

    def _extract_file_content(self, request: SolveRequest) -> str:
        """Extract text content from attached files."""
        parts = []
        for f in request.files:
            raw = base64.b64decode(f.content_base64)
            if f.mime_type.startswith("image/"):
                parts.append(
                    f"[Image file: {f.filename} ({f.mime_type}, {len(raw)} bytes)]"
                )
            else:
                try:
                    text = raw.decode("utf-8")
                    parts.append(f"### {f.filename}\n```\n{text[:5000]}\n```")
                except UnicodeDecodeError:
                    parts.append(
                        f"[Binary file: {f.filename} ({f.mime_type}, {len(raw)} bytes)]"
                    )
        return "\n".join(parts)

    async def _plan(self, request: SolveRequest, config: dict) -> str:
        """Use the LLM to create an execution plan from the task prompt."""
        content = f"## Task Prompt\n\n{request.prompt}"
        if request.files:
            content += f"\n\n## Attached Files\n\n{self._extract_file_content(request)}"

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]
        response = await self.llm.ainvoke(messages, config=config)
        return response.content

    async def solve(self, request: SolveRequest) -> SolveResponse:
        """Run the planner → executor pipeline to solve a Tripletex task."""
        # Initialize the Tripletex API client for this request
        client = TripletexClient(
            base_url=request.tripletex_credentials.base_url,
            session_token=request.tripletex_credentials.session_token,
        )
        set_client(client)

        # Configure Langfuse tracing
        config: dict = {}
        langfuse_handler = self._create_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        # Step 1: Plan
        plan = await self._plan(request, config)

        # Step 2: Execute
        executor_message = HumanMessage(
            content=f"## Original Task\n\n{request.prompt}\n\n## Execution Plan\n\n{plan}"
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
