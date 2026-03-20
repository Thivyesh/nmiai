"""LangGraph-based Tripletex accounting agent."""

import base64
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.prebuilt import create_react_agent

from task2_tripletex.models import SolveRequest, SolveResponse
from task2_tripletex.tools import ALL_TOOLS, TripletexClient, set_client

SYSTEM_PROMPT = """\
You are an expert accounting agent for Tripletex, a Norwegian accounting system.
You receive a task prompt (possibly in Norwegian, English, or other languages) and must
execute the requested accounting operations using the Tripletex API.

## Guidelines
- Parse the prompt carefully to extract entity names, field values, and relationships.
- Plan your API calls before executing. Minimize the number of calls and avoid errors.
- Some tasks require creating prerequisites first (e.g., create a customer before an invoice).
- Use GET with ?fields=* to discover available fields when unsure.
- Norwegian characters (æ, ø, å) work fine — send as-is.
- When the prompt includes file attachments, extract relevant data (names, amounts, dates) from them.
- After completing operations, verify critical results with a GET call.
- Do NOT make unnecessary API calls — efficiency is scored.

## Common API Endpoints
- /employee (GET, POST, PUT) — manage employees
- /customer (GET, POST, PUT) — manage customers
- /product (GET, POST) — manage products
- /invoice (GET, POST) — create and query invoices
- /order (GET, POST) — manage orders
- /travelExpense (GET, POST, PUT, DELETE) — travel expense reports
- /project (GET, POST) — manage projects
- /department (GET, POST) — manage departments
- /ledger/account (GET) — chart of accounts
- /ledger/voucher (GET, POST, DELETE) — manage vouchers

## API Tips
- List responses are wrapped: {"fullResultSize": N, "values": [...]}
- Use "fields" param to select specific fields: ?fields=id,firstName,lastName
- POST/PUT take JSON body
- DELETE uses ID in path: DELETE /employee/123
- Auth is handled automatically — just specify the endpoint and data.
"""


class TripletexAgent:
    """Orchestrates the LangGraph ReAct agent for solving Tripletex tasks."""

    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
        )
        self.agent = create_react_agent(
            model=self.llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )

    def _build_messages(self, request: SolveRequest) -> list:
        """Build the message list from the solve request."""
        content_parts = [f"## Task\n\n{request.prompt}"]

        if request.files:
            content_parts.append("\n## Attached Files\n")
            for f in request.files:
                raw = base64.b64decode(f.content_base64)
                if f.mime_type.startswith("image/"):
                    content_parts.append(
                        f"[Image file: {f.filename} ({f.mime_type}, {len(raw)} bytes) — "
                        f"base64 content available]"
                    )
                else:
                    try:
                        text = raw.decode("utf-8")
                        content_parts.append(
                            f"### {f.filename}\n```\n{text[:5000]}\n```"
                        )
                    except UnicodeDecodeError:
                        content_parts.append(
                            f"[Binary file: {f.filename} ({f.mime_type}, {len(raw)} bytes)]"
                        )

        return [HumanMessage(content="\n".join(content_parts))]

    def _create_langfuse_handler(self) -> LangfuseCallbackHandler | None:
        """Create Langfuse callback handler if credentials are configured."""
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return LangfuseCallbackHandler()
        return None

    async def solve(self, request: SolveRequest) -> SolveResponse:
        """Run the agent to solve a Tripletex accounting task."""
        # Initialize the Tripletex API client for this request
        client = TripletexClient(
            base_url=request.tripletex_credentials.base_url,
            session_token=request.tripletex_credentials.session_token,
        )
        set_client(client)

        messages = self._build_messages(request)

        # Configure Langfuse tracing callback
        config: dict = {}
        langfuse_handler = self._create_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        # Run the LangGraph ReAct agent
        await self.agent.ainvoke(
            {"messages": messages},
            config=config,
        )

        return SolveResponse(status="completed")
