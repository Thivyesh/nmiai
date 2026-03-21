"""Index trace history into Elasticsearch for experience-based learning.

Run: uv run python -m task2_tripletex.index_traces
"""

import json
from pathlib import Path

from elasticsearch import Elasticsearch

TRACE_PATH = Path(__file__).parent / "trace_history.json"
ES_URL = "http://localhost:9200"
INDEX = "tripletex-traces"


def main():
    es = Elasticsearch(ES_URL)

    # Delete old index if exists
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)

    # Create index with mapping
    es.indices.create(
        index=INDEX,
        body={
            "mappings": {
                "properties": {
                    "trace_id": {"type": "keyword"},
                    "timestamp": {"type": "date", "format": "yyyy-MM-dd'T'HH:mm:ss||yyyy-MM-dd"},
                    "task_prompt": {"type": "text", "analyzer": "standard"},
                    "total_tool_calls": {"type": "integer"},
                    "total_errors": {"type": "integer"},
                    "successful_endpoints": {"type": "keyword"},
                    "failed_endpoints_text": {"type": "text"},
                    "tool_summary": {"type": "text"},
                }
            }
        },
    )

    with open(TRACE_PATH) as f:
        traces = json.load(f)

    indexed = 0
    for t in traces:
        # Skip traces without a task prompt
        if not t.get("task_prompt"):
            continue

        # Build searchable text from tool calls
        tool_lines = []
        for tc in t.get("tool_calls", []):
            status = "ERR" if tc["is_error"] else "OK"
            endpoint = tc.get("endpoint", "")
            tool_lines.append(f"{tc['name']} {endpoint} {status} {tc.get('error_msg','')}")

        # Add English tags based on endpoints used for better search
        endpoints_used = " ".join(tc.get("endpoint", "") for tc in t.get("tool_calls", []))
        tags = []
        if "/invoice" in endpoints_used: tags.append("invoice")
        if "/customer" in endpoints_used: tags.append("customer")
        if "/employee" in endpoints_used: tags.append("employee")
        if "/product" in endpoints_used: tags.append("product")
        if "/order" in endpoints_used: tags.append("order")
        if "/travelExpense" in endpoints_used: tags.append("travel expense")
        if "/voucher" in endpoints_used: tags.append("voucher journal entry")
        if "/payment" in endpoints_used or "/:payment" in endpoints_used: tags.append("payment")
        if "/supplier" in endpoints_used: tags.append("supplier")
        if "/salary" in endpoints_used: tags.append("salary payroll")
        if "/department" in endpoints_used: tags.append("department")
        if "/project" in endpoints_used: tags.append("project")
        if "Dimension" in endpoints_used or "dimension" in endpoints_used: tags.append("dimension")
        if "creditNote" in endpoints_used: tags.append("credit note")
        if "entitlement" in endpoints_used: tags.append("admin entitlements")

        failed_text = "; ".join(
            f"{fe['endpoint']}: {fe['error']}" for fe in t.get("failed_endpoints", [])
        )

        doc = {
            "trace_id": t["trace_id"],
            "timestamp": t["timestamp"],
            "task_prompt": t["task_prompt"] + " " + " ".join(tags),
            "total_tool_calls": t["total_tool_calls"],
            "total_errors": t["total_errors"],
            "successful_endpoints": t.get("successful_endpoints", []),
            "failed_endpoints_text": failed_text,
            "tool_summary": "\n".join(tool_lines) + "\n" + " ".join(tags),
        }

        es.index(index=INDEX, id=t["trace_id"], body=doc)
        indexed += 1

    es.indices.refresh(index=INDEX)
    print(f"Indexed {indexed} traces into '{INDEX}'")


if __name__ == "__main__":
    main()
