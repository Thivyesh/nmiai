"""Experience checker: runs first to find past mistakes for this task type.

Searches past execution history and returns warnings/fixes
so downstream agents don't repeat known errors.
"""

import logging

from task2_tripletex.experience_tool import search_past_experience

logger = logging.getLogger(__name__)


def check_experience(prompt: str) -> str:
    """Check past experience for this task type. No LLM needed — direct tool call.

    Returns warnings and fixes from past executions, or empty string.
    """
    try:
        # Translate common task patterns to English search terms
        search_terms = _extract_search_terms(prompt)
        if not search_terms:
            return ""

        result = search_past_experience.invoke({"task_description": search_terms})

        if not result or "No past experience" in result:
            return ""

        logger.info("Experience check found: %s", result[:200])
        return f"\n## Past Experience (avoid these mistakes)\n{result}"
    except Exception as e:
        logger.warning("Experience check failed: %s", e)
        return ""


def _extract_search_terms(prompt: str) -> str:
    """Extract English search terms from the task prompt."""
    prompt_lower = prompt.lower()

    # Map task keywords to English search terms
    mappings = [
        (["faktura", "invoice", "rechnung", "factura", "facture", "fatura"], "invoice"),
        (["ansatt", "employee", "angestellte", "empleado", "employé", "tilsette"], "employee"),
        (["kunde", "customer", "kund", "client", "cliente"], "customer"),
        (["reise", "travel", "viaje", "voyage"], "travel expense"),
        (["lønn", "salary", "payroll", "gehalt", "salario"], "salary payroll"),
        (["bilag", "voucher", "buchung", "journal"], "voucher"),
        (["avskrivning", "depreciation", "abschreibung"], "depreciation voucher"),
        (["periodiser", "periodization", "accrual"], "periodization voucher"),
        (["prosjekt", "project", "projekt", "proyecto", "projet"], "project"),
        (["fastpris", "fixed price", "milestone", "milepæl"], "fixed price project"),
        (["timer", "timesheet", "hours", "horas"], "timesheet hours"),
        (["leverandør", "supplier", "incoming", "proveedor", "fournisseur", "lieferant", "fornecedor", "eingangsrechnung", "leverandørfaktura"], "incoming invoice supplier 403 permission"),
        (["kreditnota", "credit note", "gutschrift"], "credit note"),
        (["betaling", "payment", "zahlung", "pago"], "payment"),
        (["purre", "reminder", "fee"], "reminder fee"),
        (["dimensjon", "dimension"], "dimension voucher"),
        (["produkt", "product", "produkt"], "product"),
        (["ordre", "order", "bestilling"], "order invoice"),
    ]

    terms = []
    for keywords, english in mappings:
        if any(kw in prompt_lower for kw in keywords):
            terms.append(english)

    return " ".join(terms[:3]) if terms else ""
