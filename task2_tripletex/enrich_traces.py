"""Enrich trace history with fixes from task patterns and payload templates.

For each failed API call in the traces, find the correct approach
from our verified patterns and templates.

Run: uv run python -m task2_tripletex.enrich_traces
"""

import json
from pathlib import Path

from elasticsearch import Elasticsearch

from task2_tripletex.payload_templates import PAYLOAD_TEMPLATES
from task2_tripletex.task_patterns import TASK_PATTERNS

TRACE_PATH = Path(__file__).parent / "trace_history.json"
ES_URL = "http://localhost:9200"
INDEX = "tripletex-experience"

# Common error → fix mappings from our verified testing
ERROR_FIXES = {
    "debitAmount": "Use 'amountGross' and 'amountGrossCurrency', NOT 'debitAmount' or 'creditAmount'",
    "creditAmount": "Use 'amountGross' (negative for credit) and 'amountGrossCurrency', NOT 'creditAmount'",
    "amount.*not.*valid": "Use 'amountGross' and 'amountGrossCurrency' for vouchers, 'amountCurrencyIncVat' for travel costs",
    "phoneNumber.*employee": "Use 'phoneNumberMobile' for employees, 'phoneNumber' for customers",
    "Brukertype": "Do NOT set userType on first attempt. If error, retry with userType: 'STANDARD'",
    "department.*fylles": "Department is REQUIRED for employees. GET /department first",
    "deliveryDate": "deliveryDate is REQUIRED on orders",
    "bankkontonummer": "Set bankAccountNumber on ledger account 1920 before creating invoices",
    "Produktnummeret.*bruk": "Product number already exists. GET /product?number=N to find its ID",
    "mva-kode": "Do NOT set vatType on products unless prompt specifies a VAT rate. Default works.",
    "account.*number": "ALWAYS use account {'id': N} from GET /ledger/account, NEVER {'number': N}",
    "quantity": "Use 'count' NOT 'quantity' for order lines",
    "description.*eksisterer": "Invoice uses 'comment' NOT 'description'",
    "Request mapping": "Wrong field names. Look up the exact schema with get_payload_template or lookup_api_docs",
    "Object not found": "Wrong ID or URL format. For entitlements: /employee/entitlement/:grantEntitlementsByTemplate?employeeId=N",
    "Kunde mangler": "Account has ledgerType CUSTOMER — include customer: {'id': N} in the voucher posting",
    "Leverandør mangler": "Account has ledgerType SUPPLIER — include supplier: {'id': N} in the voucher posting",
    "Arbeidsforholdet.*virksomhet": "Employment needs division. Create division first, then link it to employment",
    "Ugyldig år": "Year must be current (2026). Past years fail.",
    "proxy token": "Token expired or wrong sandbox. This was a concurrency issue — now serialized with asyncio.Lock",
}


def find_fix_for_error(error_msg: str) -> str:
    """Match error message to known fix."""
    error_lower = error_msg.lower()
    for pattern, fix in ERROR_FIXES.items():
        if pattern.lower() in error_lower:
            return fix
    return ""


def find_template_for_endpoint(endpoint: str) -> str:
    """Find the correct payload template for an endpoint."""
    for key, template in PAYLOAD_TEMPLATES.items():
        path = key.split(" ", 1)[-1] if " " in key else key
        if path in endpoint or endpoint in path:
            payload = template["payload"]
            if isinstance(payload, dict):
                return json.dumps(payload, indent=2)[:300]
            return str(payload)[:200]
    return ""


def main():
    with open(TRACE_PATH) as f:
        traces = json.load(f)

    es = Elasticsearch(ES_URL)

    # Delete old index
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)

    es.indices.create(
        index=INDEX,
        body={
            "mappings": {
                "properties": {
                    "trace_id": {"type": "keyword"},
                    "timestamp": {"type": "date", "format": "yyyy-MM-dd'T'HH:mm:ss||yyyy-MM-dd"},
                    "task_prompt": {"type": "text"},
                    "total_errors": {"type": "integer"},
                    "total_tool_calls": {"type": "integer"},
                    "successful_calls": {"type": "text"},
                    "failed_calls_with_fixes": {"type": "text"},
                    "correct_templates": {"type": "text"},
                    "tags": {"type": "keyword"},
                }
            }
        },
    )

    indexed = 0
    for t in traces:
        if not t.get("task_prompt"):
            continue

        # Build successful calls summary
        successful = []
        for tc in t.get("tool_calls", []):
            if not tc["is_error"] and tc["name"].startswith("tripletex_"):
                successful.append(f"{tc['name']} {tc.get('endpoint', '')} OK")

        # Build failed calls with fixes
        failed_with_fixes = []
        correct_templates = []
        for tc in t.get("tool_calls", []):
            if tc["is_error"]:
                error = tc.get("error_msg", "")
                endpoint = tc.get("endpoint", "")
                fix = find_fix_for_error(error)
                template = find_template_for_endpoint(endpoint)

                entry = f"{tc['name']} {endpoint} FAILED: {error}"
                if fix:
                    entry += f"\n  FIX: {fix}"
                if template:
                    entry += f"\n  CORRECT TEMPLATE: {template[:200]}"
                    correct_templates.append(f"{endpoint}: {template[:200]}")
                failed_with_fixes.append(entry)

        # Tags
        endpoints = " ".join(tc.get("endpoint", "") for tc in t.get("tool_calls", []))
        tags = []
        if "/invoice" in endpoints: tags.append("invoice")
        if "/customer" in endpoints: tags.append("customer")
        if "/employee" in endpoints: tags.append("employee")
        if "/product" in endpoints: tags.append("product")
        if "/order" in endpoints: tags.append("order")
        if "/travelExpense" in endpoints: tags.append("travel_expense")
        if "/voucher" in endpoints: tags.append("voucher")
        if "payment" in endpoints.lower(): tags.append("payment")
        if "/supplier" in endpoints: tags.append("supplier")
        if "/salary" in endpoints: tags.append("salary")
        if "/department" in endpoints: tags.append("department")
        if "/project" in endpoints: tags.append("project")
        if "dimension" in endpoints.lower(): tags.append("dimension")
        if "creditNote" in endpoints: tags.append("credit_note")
        if "entitlement" in endpoints: tags.append("entitlements")
        if "/division" in endpoints: tags.append("division")

        doc = {
            "trace_id": t["trace_id"],
            "timestamp": t["timestamp"],
            "task_prompt": t["task_prompt"],
            "total_errors": t["total_errors"],
            "total_tool_calls": t["total_tool_calls"],
            "successful_calls": "\n".join(successful),
            "failed_calls_with_fixes": "\n".join(failed_with_fixes),
            "correct_templates": "\n".join(correct_templates),
            "tags": tags,
        }

        es.index(index=INDEX, id=t["trace_id"], body=doc)
        indexed += 1

    es.indices.refresh(index=INDEX)
    print(f"Indexed {indexed} enriched traces into '{INDEX}'")

    # Show some stats
    errors_with_fixes = sum(1 for t in traces for tc in t.get("tool_calls", [])
                           if tc["is_error"] and find_fix_for_error(tc.get("error_msg", "")))
    total_errors = sum(1 for t in traces for tc in t.get("tool_calls", []) if tc["is_error"])
    print(f"Errors with fixes: {errors_with_fixes}/{total_errors}")


if __name__ == "__main__":
    main()
