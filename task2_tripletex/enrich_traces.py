"""Thorough trace enrichment: classify, map patterns, templates, fixes.

Run: uv run python -m task2_tripletex.enrich_traces
"""

import json
import re
from pathlib import Path

from elasticsearch import Elasticsearch

from task2_tripletex.accounting_concepts import CONCEPTS
from task2_tripletex.payload_templates import PAYLOAD_TEMPLATES
from task2_tripletex.task_patterns import TASK_PATTERNS

TRACE_PATH = Path(__file__).parent / "trace_history.json"
ES_URL = "http://localhost:9200"
INDEX = "tripletex-experience"

# Task type classification keywords
TASK_CLASSIFIERS = {
    "salary": ["lønn", "salary", "payroll", "gehalt", "salario", "nómina", "lønnskjøring"],
    "travel_expense": ["reiseregning", "travel expense", "nota de gastos", "reisekostenabrechnung", "note de frais"],
    "invoice": ["faktura", "invoice", "rechnung", "factura", "facture", "fatura"],
    "voucher": ["bilag", "voucher", "journal", "postering", "buchung", "lançamento"],
    "dimension": ["dimensjon", "dimension", "kostsenter", "dimensão"],
    "employee": ["ansatt", "employee", "mitarbeiter", "empleado", "employé", "funcionário", "tilsett"],
    "customer": ["kunde", "customer", "klient", "client", "cliente"],
    "supplier": ["leverandør", "supplier", "lieferant", "proveedor", "fournisseur", "fornecedor"],
    "product": ["produkt", "product", "produkt", "producto", "produit"],
    "project": ["prosjekt", "project", "projekt", "proyecto", "projet"],
    "department": ["avdeling", "department", "abteilung", "departamento"],
    "credit_note": ["kreditnota", "credit note", "nota de crédito", "gutschrift"],
    "payment": ["betaling", "payment", "zahlung", "pago", "paiement", "pagamento"],
    "payment_reversal": ["reverser", "tilbakeføring", "stornering", "zurückgebucht", "reverse", "annuler"],
    "incoming_invoice": ["leverandørfaktura", "incoming invoice", "inngående", "eingangsrechnung", "fatura do fornecedor"],
    "delete": ["slett", "delete", "eliminar", "löschen", "fjern"],
    "update": ["oppdater", "endre", "update", "actualizar", "ändern"],
    "month_end": ["månedsavslutning", "periodeslutt", "month-end", "periodiser", "avskrivning", "depreciation"],
    "reminder_fee": ["purregebyr", "purring", "reminder", "inkasso", "forfalt"],
    "currency_exchange": ["valutadifferanse", "agio", "disagio", "currency", "exchange rate", "valutagevinst"],
}

# Error → fix + template + concept mapping
ERROR_FIXES = {
    "debitAmount": {"fix": "Use 'amountGross' and 'amountGrossCurrency'", "template": "POST /ledger/voucher", "concept": "receipt booking"},
    "creditAmount": {"fix": "Use 'amountGross' (negative for credit)", "template": "POST /ledger/voucher", "concept": None},
    "Brukertype": {"fix": "Don't set userType first. Retry with userType: 'STANDARD'", "template": "POST /employee", "concept": None},
    "department.*fylles": {"fix": "Department REQUIRED. GET /department first", "template": "POST /employee", "concept": None},
    "deliveryDate": {"fix": "deliveryDate REQUIRED on orders", "template": "POST /order", "concept": None},
    "bankkontonummer": {"fix": "Set bankAccountNumber on ledger account 1920", "template": "PUT /ledger/account/{id}", "concept": None},
    "Produktnummeret.*bruk": {"fix": "Product number exists. GET /product?number=N", "template": "POST /product", "concept": None},
    "mva-kode": {"fix": "Don't set vatType on products unless specified", "template": "POST /product", "concept": "vat handling"},
    "quantity": {"fix": "Use 'count' NOT 'quantity'", "template": "POST /order", "concept": None},
    "description.*eksisterer": {"fix": "Invoice uses 'comment' NOT 'description'", "template": "POST /invoice", "concept": None},
    "Request mapping": {"fix": "Wrong field names. Use get_payload_template", "template": None, "concept": None},
    "Object not found": {"fix": "Wrong ID/URL. Entitlements: query params not path", "template": None, "concept": None},
    "Kunde mangler": {"fix": "Account needs customer ID in posting", "template": "POST /ledger/voucher", "concept": "reminder fee"},
    "Leverandør mangler": {"fix": "Account needs supplier ID in posting", "template": "POST /ledger/voucher", "concept": "supplier invoice"},
    "Arbeidsforholdet.*virksomhet": {"fix": "Employment needs division. Create division first", "template": "POST /employee/employment", "concept": None},
    "Ugyldig år": {"fix": "Year must be current (2026)", "template": "POST /salary/transaction", "concept": None},
    "proxy token": {"fix": "Concurrency issue — now serialized", "template": None, "concept": None},
    "Validering feilet": {"fix": "Validation error — check field names with get_payload_template", "template": None, "concept": None},
    "phoneNumber": {"fix": "Employee: phoneNumberMobile. Customer: phoneNumber", "template": "POST /employee", "concept": None},
    "municipalityDate": {"fix": "Division needs municipalityDate + municipality", "template": None, "concept": None},
    "amountExcludingVat": {"fix": "Not a valid invoice field. Use 'amount' or 'amountOutstanding'", "template": None, "concept": None},
    "amountPaid": {"fix": "Not a valid invoice field. Use 'amountOutstanding'", "template": None, "concept": None},
    "Enhetspris": {"fix": "Set isPrioritizeAmountsIncludingVat:false, use unitPriceExcludingVatCurrency", "template": "POST /order", "concept": "vat handling"},
    "externalId": {"fix": "Field format issue on incoming invoice", "template": None, "concept": "supplier invoice"},
}


def classify_task(prompt: str) -> list[str]:
    """Classify task type from prompt text."""
    prompt_lower = prompt.lower()
    types = []
    for task_type, keywords in TASK_CLASSIFIERS.items():
        if any(kw in prompt_lower for kw in keywords):
            types.append(task_type)
    return types or ["unknown"]


def get_pattern_section(task_types: list[str]) -> str:
    """Get relevant task pattern sections."""
    sections = []
    for line in TASK_PATTERNS.split("\n## "):
        line_lower = line.lower()
        for tt in task_types:
            if tt.replace("_", " ") in line_lower or tt in line_lower:
                # Extract just the workflow and gotchas
                lines = line.split("\n")
                relevant = [l for l in lines if l.strip() and not l.startswith("Keywords:")]
                sections.append("\n".join(relevant[:15]))
                break
    return "\n---\n".join(sections[:2]) if sections else ""


def find_fix(error_msg: str) -> dict:
    """Find fix for an error message."""
    for pattern, fix_info in ERROR_FIXES.items():
        if re.search(pattern, error_msg, re.IGNORECASE):
            result = {"fix": fix_info["fix"]}
            if fix_info.get("template") and fix_info["template"] in PAYLOAD_TEMPLATES:
                template = PAYLOAD_TEMPLATES[fix_info["template"]]
                payload = template["payload"]
                result["correct_template"] = json.dumps(payload, indent=2)[:300] if isinstance(payload, dict) else str(payload)[:200]
                result["template_notes"] = template.get("notes", "")[:150]
            if fix_info.get("concept") and fix_info["concept"] in CONCEPTS:
                concept = CONCEPTS[fix_info["concept"]]
                result["accounting_context"] = concept.get("explanation", "")[:150]
            return result
    return {"fix": "Unknown error. Use lookup_api_docs for the correct schema."}


def get_templates_for_endpoints(endpoints: list[str]) -> str:
    """Get payload templates for a list of endpoints."""
    templates = []
    for ep in endpoints:
        for key, tmpl in PAYLOAD_TEMPLATES.items():
            path = key.split(" ", 1)[-1] if " " in key else key
            if path in ep or ep in path:
                payload = tmpl["payload"]
                payload_str = json.dumps(payload, indent=2)[:250] if isinstance(payload, dict) else str(payload)[:150]
                templates.append(f"{key}: {payload_str}")
                break
    return "\n".join(templates[:3])


def build_lesson_learned(task_types, successful, failed_with_fixes, total_errors):
    """Build a lesson_learned summary."""
    lesson = f"Task type: {', '.join(task_types)}. "
    if total_errors == 0:
        lesson += "SUCCESS — 0 errors. "
        if successful:
            lesson += f"Approach: {'; '.join(successful[:3])}. "
    else:
        lesson += f"{total_errors} errors. "
        for f in failed_with_fixes[:3]:
            lesson += f"{f} "
    return lesson[:500]


def main():
    with open(TRACE_PATH) as f:
        traces = json.load(f)

    es = Elasticsearch(ES_URL)

    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)

    es.indices.create(index=INDEX, body={
        "mappings": {
            "properties": {
                "trace_id": {"type": "keyword"},
                "timestamp": {"type": "date", "format": "yyyy-MM-dd'T'HH:mm:ss||yyyy-MM-dd"},
                "task_prompt": {"type": "text"},
                "task_type": {"type": "keyword"},
                "total_errors": {"type": "integer"},
                "total_tool_calls": {"type": "integer"},
                "lesson_learned": {"type": "text"},
                "correct_workflow": {"type": "text"},
                "correct_templates": {"type": "text"},
                "error_fixes": {"type": "text"},
                "successful_calls": {"type": "text"},
                "tags": {"type": "keyword"},
            }
        }
    })

    indexed = 0
    fixes_found = 0

    for t in traces:
        prompt = t.get("task_prompt", "")
        if not prompt:
            continue

        task_types = classify_task(prompt)
        pattern_section = get_pattern_section(task_types)

        # Process tool calls
        successful = []
        failed_with_fixes = []
        error_fix_texts = []
        all_endpoints = []

        for tc in t.get("tool_calls", []):
            endpoint = tc.get("endpoint", "")
            if endpoint:
                all_endpoints.append(endpoint)

            if tc["is_error"]:
                error = tc.get("error_msg", "")
                fix_info = find_fix(error)
                fixes_found += 1
                entry = f"{tc['name']} {endpoint}: {fix_info['fix']}"
                if fix_info.get("correct_template"):
                    entry += f" TEMPLATE: {fix_info['correct_template'][:150]}"
                if fix_info.get("accounting_context"):
                    entry += f" CONTEXT: {fix_info['accounting_context']}"
                failed_with_fixes.append(entry)
                error_fix_texts.append(entry)
            elif tc["name"].startswith("tripletex_"):
                successful.append(f"{tc['name']} {endpoint} OK")

        templates = get_templates_for_endpoints(all_endpoints)
        lesson = build_lesson_learned(task_types, successful, failed_with_fixes, t["total_errors"])

        # Tags
        tags = list(set(task_types))
        if any("/voucher" in ep for ep in all_endpoints): tags.append("voucher")
        if any("/invoice" in ep for ep in all_endpoints): tags.append("invoice")
        if any("/employee" in ep for ep in all_endpoints): tags.append("employee")
        if any("/salary" in ep for ep in all_endpoints): tags.append("salary")
        if any("/travelExpense" in ep for ep in all_endpoints): tags.append("travel_expense")
        if any("/payment" in ep.lower() for ep in all_endpoints): tags.append("payment")
        if any("dimension" in ep.lower() for ep in all_endpoints): tags.append("dimension")

        doc = {
            "trace_id": t["trace_id"],
            "timestamp": t["timestamp"],
            "task_prompt": prompt,
            "task_type": task_types,
            "total_errors": t["total_errors"],
            "total_tool_calls": t["total_tool_calls"],
            "lesson_learned": lesson,
            "correct_workflow": pattern_section[:1000],
            "correct_templates": templates[:500],
            "error_fixes": "\n".join(error_fix_texts[:5]),
            "successful_calls": "\n".join(successful[:10]),
            "tags": list(set(tags)),
        }

        es.index(index=INDEX, id=t["trace_id"], body=doc)
        indexed += 1

    es.indices.refresh(index=INDEX)
    print(f"Indexed {indexed} enriched traces")
    print(f"Errors mapped to fixes: {fixes_found}")
    print(f"Task types found: {set(tt for t in traces for tt in classify_task(t.get('task_prompt','')))}")


if __name__ == "__main__":
    main()
