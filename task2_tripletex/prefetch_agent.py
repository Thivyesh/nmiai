"""Pre-fetch agent: uses GPT-4o-mini to understand the task and fetch relevant data.

Reads the prompt, identifies entities to search for, fetches them from the API,
and returns structured context for the main agent.
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from task2_tripletex.tools import tripletex_get
from task2_tripletex.api_docs_tool import lookup_api_docs

logger = logging.getLogger(__name__)

PREFETCH_PROMPT = """\
You gather data from the Tripletex API to help the main agent.
You have two tools:
- tripletex_get: read-only API access to fetch entities
- lookup_api_docs: search the API schema to find correct endpoints and field names

## Your job
Read the task prompt and fetch ALL entities that might already exist in the sandbox.
The competition pre-loads data — find it so the main agent doesn't have to search.

## What to fetch (do ALL that apply):
1. ALWAYS: GET /department?fields=id,name&count=1
2. ALWAYS: GET /ledger/account?number=1920&fields=id,version,bankAccountNumber
3. If task mentions ANY account numbers (e.g., 6590, 1500, 3400, 6010, 1720): GET /ledger/account?number=NNNN&fields=id,number,name for EACH one
4. If task mentions an employee name/email: GET /employee?email=X&fields=id,firstName,lastName,email
5. If employee found: GET /employee/employment?employeeId=N&fields=id,startDate,division
6. If task mentions a customer/supplier name/org: GET /customer?organizationNumber=X&fields=id,name,organizationNumber OR GET /supplier?organizationNumber=X
7. If task mentions invoices: GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&fields=id,invoiceNumber,amount,amountOutstanding,comment&count=10
8. If task involves salary: GET /salary/type?fields=id,number,name AND GET /division?fields=id,name&count=1
9. If task involves vouchers: GET /ledger/voucherType?fields=id,name&count=5
10. If task involves travel: GET /travelExpense/paymentType?fields=id,description
11. If task involves payments: GET /invoice/paymentType?fields=id,description
12. If task involves incoming/supplier invoice: GET /ledger/vatType?fields=id,number,name&count=10

## Output format
Return a structured summary:
EXISTING ENTITIES:
- employees: [list with id, name, email, has_employment]
- customers: [list with id, name, org_number]
- invoices: [list with id, number, amount, outstanding]
- division: id or "none"
- department_id: N

REFERENCE DATA:
- account_1920: {id, version, bankAccountNumber}
- salary_types: [list]
- voucher_types: [list]
- payment_types: [list]

## Rules
- Be FAST. Max 8 tool calls.
- If entity NOT FOUND, say so explicitly.
- Report what EXISTS and what DOESN'T.
"""


async def prefetch_for_task(prompt: str, files: list | None = None) -> str:
    """Run the pre-fetch agent to gather context for the main agent."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_retries=2,
        timeout=20,
    )

    agent = create_react_agent(
        model=llm,
        tools=[tripletex_get, lookup_api_docs],
        prompt=PREFETCH_PROMPT,
    )

    message = HumanMessage(content=f"## Task Prompt\n\n{prompt}")

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
        logger.info("Pre-fetch result:\n%s", content[:500])
        return f"\n## Pre-fetched Data (from sandbox)\nIMPORTANT: If entities exist below, use their IDs. Do NOT modify or recreate them.\n\n{content}"
    except asyncio.TimeoutError:
        logger.warning("Pre-fetch agent timed out")
        return ""
    except Exception as e:
        logger.warning("Pre-fetch agent failed: %s", e)
        return ""
