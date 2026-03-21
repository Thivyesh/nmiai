"""Accounting concept explainer — maps accounting terms to Tripletex API operations.

The agent knows the API but not accounting. This tool bridges the gap.
"""

from langchain_core.tools import tool

CONCEPTS = {
    # Month-end / Year-end
    "month-end closing": {
        "norwegian": "månedsavslutning, periodeslutt",
        "explanation": "Close the month by posting periodic journal entries.",
        "operations": [
            "Periodize prepaid costs: POST /ledger/voucher — debit expense account, credit prepaid account (1720)",
            "Book depreciation: POST /ledger/voucher — debit depreciation expense (6000-range), credit asset account (1200-range)",
            "Accrue expenses: POST /ledger/voucher — debit expense account, credit accrued liabilities (2900-range)",
            "Each operation is a separate voucher with balanced postings",
        ],
    },
    "year-end closing": {
        "norwegian": "årsavslutning, årsoppgjør",
        "explanation": "Close the fiscal year. Similar to month-end but includes profit/loss transfer.",
        "operations": [
            "All month-end closing operations",
            "Transfer profit/loss to equity: POST /ledger/voucher",
            "May use /yearEnd endpoints for specific year-end reports",
        ],
    },
    "depreciation": {
        "norwegian": "avskrivning",
        "explanation": "Reduce the book value of a fixed asset over time.",
        "operations": [
            "POST /ledger/voucher with two postings:",
            "  Debit: depreciation expense account (6000-6020 range) — positive amountGross",
            "  Credit: accumulated depreciation / asset account (1200 range) — negative amountGross",
            "Amount is typically: asset cost / useful life (annual) / 12 (monthly)",
        ],
    },
    "periodization": {
        "norwegian": "periodisering, forskuddsbetalt",
        "explanation": "Spread a prepaid cost across multiple periods.",
        "operations": [
            "POST /ledger/voucher with two postings:",
            "  Debit: expense account (relevant cost account) — positive amountGross",
            "  Credit: prepaid account (1720 or similar) — negative amountGross",
            "Monthly amount = total prepaid / number of months",
        ],
    },
    "accrual": {
        "norwegian": "påløpt kostnad, avsetning",
        "explanation": "Recognize an expense that hasn't been invoiced yet.",
        "operations": [
            "POST /ledger/voucher with two postings:",
            "  Debit: expense account — positive amountGross",
            "  Credit: accrued liabilities account (2900-range) — negative amountGross",
        ],
    },

    # Receipt / expense booking
    "receipt booking": {
        "norwegian": "kvittering, bilag, bokføring av kvittering",
        "explanation": "Book an expense from a receipt (hotel, transport, food, etc.).",
        "operations": [
            "Option 1 — Travel expense: POST /travelExpense + POST /travelExpense/cost",
            "Option 2 — Direct voucher: POST /ledger/voucher",
            "  Debit: expense account (6300=rent, 6800=office, 7140=travel, etc.) — positive amountGross",
            "  Credit: payment account (1900=cash, 2400=accounts payable) — negative amountGross",
            "VAT: if receipt has VAT, split into net amount + VAT amount with correct vatType",
        ],
    },
    "accommodation expense": {
        "norwegian": "overnatting, hotell",
        "explanation": "Book accommodation/hotel costs.",
        "operations": [
            "As travel expense cost: POST /travelExpense/cost with category='Overnatting/Hotel'",
            "Or as voucher: POST /ledger/voucher",
            "  Debit: travel cost account (7140) or accommodation account",
            "  Credit: bank/cash account",
            "VAT: hotel in Norway has reduced rate (12%)",
        ],
    },

    # Reminder / collection
    "reminder fee": {
        "norwegian": "purregebyr, purring, inkassogebyr",
        "explanation": "Charge a customer a fee for late payment.",
        "operations": [
            "1. Find the overdue invoice: GET /invoice with date filters, check amountOutstanding > 0",
            "2. Post the fee as voucher: POST /ledger/voucher",
            "   Debit: customer receivables (1500) — positive amountGross (include customer ID in posting)",
            "   Credit: reminder fee income (3400 or similar) — negative amountGross",
            "3. Create reminder invoice: POST /order + POST /invoice for the fee amount",
            "Important: posting to account 1500 requires customer: {'id': N} in the posting",
        ],
    },

    # Bank reconciliation
    "bank reconciliation": {
        "norwegian": "bankavstemming",
        "explanation": "Match bank statement transactions with bookkeeping.",
        "operations": [
            "POST /bank/statement/import — upload bank statement file",
            "GET /bank/statement/transaction — find unmatched transactions",
            "Match transactions with existing vouchers/invoices",
        ],
    },

    # VAT
    "vat handling": {
        "norwegian": "mva, merverdiavgift, mva-behandling",
        "explanation": "Norwegian VAT rates and how to apply them.",
        "rates": {
            "25%": "Standard rate — most goods and services",
            "15%": "Food and drink (matservering)",
            "12%": "Transport, cinema, accommodation",
            "0%": "Exempt — exports, some services",
        },
        "operations": [
            "Outgoing (sales): vatType with 'Utgående' in name",
            "Incoming (purchases): vatType with 'Fradrag inngående' or 'Inngående' in name",
            "On voucher postings: include vatType: {'id': N} to apply VAT",
            "GET /ledger/vatType to find correct IDs",
        ],
    },

    # Currency
    "currency exchange": {
        "norwegian": "valutadifferanse, agio, disagio, valutagevinst, valutatap",
        "explanation": "Handle exchange rate differences when receiving foreign currency payments.",
        "operations": [
            "1. Register payment at new rate: PUT /invoice/{id}/:payment with converted amount",
            "2. Post currency difference as voucher: POST /ledger/voucher",
            "   Agio (gain): Debit bank (1920), Credit currency gain account (8050 or 8060)",
            "   Disagio (loss): Debit currency loss account (8150 or 8160), Credit bank (1920)",
            "Amount = (new rate - old rate) × foreign amount",
        ],
    },

    # Supplier invoice
    "supplier invoice": {
        "norwegian": "leverandørfaktura, inngående faktura",
        "explanation": "Register an invoice received from a supplier.",
        "operations": [
            "POST /incomingInvoice?sendTo=ledger",
            "CRITICAL: uses FLAT IDs (vendorId, accountId, vatTypeId) NOT nested objects",
            "invoiceHeader: {vendorId, invoiceDate, dueDate, currencyId, invoiceAmount, invoiceNumber}",
            "orderLines: [{row, accountId, amountInclVat, vatTypeId, count, description}]",
        ],
    },
}


@tool
def explain_accounting_concept(concept: str) -> str:
    """Explain an accounting concept and how to implement it in Tripletex.

    Use when you encounter an unfamiliar accounting term or don't know
    which API operations to use for a task.

    Args:
        concept: The accounting concept or Norwegian term, e.g.
            "month-end closing", "depreciation", "purregebyr",
            "periodization", "VAT handling", "bank reconciliation".

    Returns:
        Explanation of the concept and the Tripletex API operations needed.
    """
    search = concept.lower()

    # Try exact match
    for key, val in CONCEPTS.items():
        if search in key or key in search:
            return _format_concept(key, val)
        norwegian = val.get("norwegian", "").lower()
        if any(term.strip() in search or search in term.strip() for term in norwegian.split(",")):
            return _format_concept(key, val)

    # Try partial match
    best_match = None
    best_score = 0
    for key, val in CONCEPTS.items():
        score = 0
        for word in search.split():
            if len(word) < 3:
                continue
            if word in key.lower():
                score += 3
            if word in val.get("norwegian", "").lower():
                score += 2
            if word in val.get("explanation", "").lower():
                score += 1
        if score > best_score:
            best_score = score
            best_match = (key, val)

    if best_match and best_score >= 2:
        return _format_concept(best_match[0], best_match[1])

    # List all available concepts
    available = "\n".join(f"- {k} ({v.get('norwegian', '')})" for k, v in CONCEPTS.items())
    return f"No concept found for '{concept}'. Available concepts:\n{available}"


def _format_concept(key: str, val: dict) -> str:
    result = f"## {key.title()}\n"
    result += f"Norwegian: {val.get('norwegian', '')}\n"
    result += f"\n{val.get('explanation', '')}\n"
    if "rates" in val:
        result += "\nRates:\n"
        for rate, desc in val["rates"].items():
            result += f"  {rate}: {desc}\n"
    result += "\nTripletex operations:\n"
    for op in val.get("operations", []):
        result += f"  {op}\n"
    return result
