"""Schema discovery agent: finds correct endpoints and payloads for unknown task types.

Runs between prefetch and main agent. Analyzes the task, identifies which
endpoints are needed, and for any that don't have templates, looks them up
in the API docs and returns ready-to-use payload structures.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from task2_tripletex.api_docs_tool import lookup_api_docs
from task2_tripletex.payload_templates import PAYLOAD_TEMPLATES
from task2_tripletex.workflow_tools import get_payload_template, get_task_workflow

logger = logging.getLogger(__name__)

AVAILABLE_TEMPLATES = "\n".join(f"- {k}" for k in PAYLOAD_TEMPLATES.keys())

SCHEMA_AGENT_PROMPT = f"""\
You are a Tripletex API schema expert. Your job is to prepare COMPLETE endpoint
documentation for the main agent so it can execute without searching.

## Available templates (already known)
{AVAILABLE_TEMPLATES}

## Your process
1. READ the "Past Experience" section if present — it contains critical warnings from past failures
2. Call get_task_workflow to understand what steps are needed
3. For EACH endpoint in the workflow:
   a. Call get_payload_template — if it returns a template, include it in output
   b. If NO template exists: call lookup_api_docs to find the endpoint schema
4. For any endpoint found via API docs, construct a template-like summary:
   - Method + path
   - Required fields with types
   - Example payload JSON
   - Any important notes (required fields, query params vs body, etc.)

## Output format
Return ALL endpoint templates the main agent will need, in execution order:

STEP 1: <METHOD> <endpoint>
```json
{{payload}}
```
Notes: <important details>

STEP 2: ...

## Rules
- If "Past Experience" has warnings, add them as WARNINGS in the relevant step's Notes
- Be thorough — find ALL endpoints including obscure ones
- For query-param endpoints (/:action), note that body should be "{{}}"
- Include field types and which are required vs optional
- Max 10 tool calls. Be efficient.
- Query all tools in English for best results.
"""


async def discover_schemas(prompt: str, ref_data: str, file_data: str = "") -> str:
    """Run schema discovery agent to find all needed endpoint templates.

    Returns structured template documentation for the main agent.
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_retries=2,
        timeout=25,
    )

    agent = create_react_agent(
        model=llm,
        tools=[get_task_workflow, get_payload_template, lookup_api_docs],
        prompt=SCHEMA_AGENT_PROMPT,
    )

    content = f"## Task\n\n{prompt}\n\n{ref_data}"
    if file_data:
        content += f"\n{file_data}"

    message = HumanMessage(content=content)

    try:
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [message]},
                config={"recursion_limit": 20},
            ),
            timeout=30,
        )
        last = result["messages"][-1]
        content = last.content
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in content
            )
        logger.info("Schema discovery result:\n%s", content[:500])
        return f"\n## Endpoint Templates (ready to use — copy these payloads)\n{content}"
    except asyncio.TimeoutError:
        logger.warning("Schema discovery timed out")
        return ""
    except Exception as e:
        logger.warning("Schema discovery failed: %s", e)
        return ""
