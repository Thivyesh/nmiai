"""Test suite for the Tripletex agent.

Run: uv run python -m task2_tripletex.test_agent

Tests against the sandbox to verify the agent handles all known task types.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from task2_tripletex.agent import TripletexAgent
from task2_tripletex.models import SolveRequest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SANDBOX_BASE_URL = os.getenv("TRIPLETEX_BASE_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
SANDBOX_TOKEN = os.getenv("TRIPLETEX_SESSION_TOKEN", "")

# Test tasks covering all known competition task types
TEST_TASKS = [
    {
        "name": "T1: Simple customer (Norwegian)",
        "prompt": "Opprett en kunde med navn Testfirma AS, e-post info@testfirma.no, telefon 44556677.",
        "verify": lambda: _check_customer("Testfirma"),
        "tier": 1,
    },
    {
        "name": "T2: Employee + admin (Norwegian)",
        "prompt": "Opprett en ansatt med navn Test Testesen, e-post test@testesen.no. Han skal være kontoadministrator.",
        "verify": lambda: _check_employee_admin("Test"),
        "tier": 1,
    },
    {
        "name": "T3: Product (German)",
        "prompt": 'Erstellen Sie das Produkt "Testprodukt" mit der Produktnummer 9999. Der Preis beträgt 5000 NOK ohne MwSt.',
        "verify": lambda: _check_product("Testprodukt", "9999"),
        "tier": 1,
    },
    {
        "name": "T4: Travel expense (Spanish)",
        "prompt": 'Registre una nota de gastos de viaje para Test Testesen (test@testesen.no) por "Viaje de prueba". Gastos: transporte 1500 NOK y alojamiento 3000 NOK.',
        "verify": lambda: _check_travel_expense("Viaje de prueba"),
        "tier": 1,
    },
    {
        "name": "T5: Invoice + payment (French)",
        "prompt": "Créez une facture pour le client Testfirma AS. Date de facture 2026-04-01, échéance 2026-04-15. Une ligne: Services de test, 5 heures à 2000 NOK. Enregistrez ensuite un paiement complet à la date du 2026-04-10.",
        "verify": lambda: _check_invoice_paid("2026-04-01"),
        "tier": 2,
    },
    {
        "name": "T6: Project (Portuguese)",
        "prompt": 'Crie um projeto chamado "Projeto de teste" para o cliente Testfirma AS. Data de início: 2026-05-01.',
        "verify": lambda: _check_project("Projeto de teste"),
        "tier": 1,
    },
    {
        "name": "T7: Department (English)",
        "prompt": 'Create a department called "Engineering" with department number "ENG01".',
        "verify": lambda: _check_department("Engineering"),
        "tier": 1,
    },
]


def _api(endpoint, params=None):
    """Helper to call sandbox API."""
    resp = requests.get(
        f"{SANDBOX_BASE_URL}/{endpoint.lstrip('/')}",
        auth=("0", SANDBOX_TOKEN),
        params=params,
        timeout=10,
    )
    return resp.json() if resp.ok else {}


def _check_customer(name_contains):
    result = _api("/customer", {"fields": "id,name,email,phoneNumber", "count": "50"})
    for v in result.get("values", []):
        if name_contains.lower() in v.get("name", "").lower():
            has_email = bool(v.get("email"))
            has_phone = bool(v.get("phoneNumber"))
            return True, f"Found: {v['name']} email={v.get('email')} phone={v.get('phoneNumber')}", has_email and has_phone
    return False, f"Customer containing '{name_contains}' not found", False


def _check_employee_admin(first_name):
    result = _api("/employee", {"firstName": first_name, "fields": "id,firstName,lastName,email"})
    vals = result.get("values", [])
    if not vals:
        return False, f"Employee '{first_name}' not found", False
    emp = vals[-1]
    emp_id = emp["id"]
    ent = _api("/employee/entitlement", {"employeeId": str(emp_id), "fields": "name", "count": "5"})
    has_admin = any("ADMINISTRATOR" in e.get("name", "") for e in ent.get("values", []))
    return True, f"Found: {emp['firstName']} {emp['lastName']} admin={has_admin}", has_admin


def _check_product(name, number):
    result = _api("/product", {"name": name, "fields": "id,name,number,priceExcludingVatCurrency"})
    for v in result.get("values", []):
        if name.lower() in v.get("name", "").lower():
            correct_number = v.get("number") == number
            has_price = v.get("priceExcludingVatCurrency", 0) > 0
            return True, f"Found: {v['name']} num={v.get('number')} price={v.get('priceExcludingVatCurrency')}", correct_number and has_price
    return False, f"Product '{name}' not found", False


def _check_travel_expense(title_contains):
    result = _api("/travelExpense", {"fields": "id,title,amount", "count": "20"})
    for v in result.get("values", []):
        if title_contains.lower() in v.get("title", "").lower():
            has_costs = v.get("amount", 0) > 0
            return True, f"Found: '{v['title']}' amount={v.get('amount')}", has_costs
    return False, f"Travel expense containing '{title_contains}' not found", False


def _check_invoice_paid(invoice_date):
    result = _api("/invoice", {
        "invoiceDateFrom": invoice_date,
        "invoiceDateTo": "2026-12-31",
        "fields": "id,invoiceNumber,amount,amountOutstanding,invoiceDate",
        "count": "10",
    })
    for v in result.get("values", []):
        if v.get("invoiceDate") == invoice_date:
            is_paid = v.get("amountOutstanding", 1) == 0
            return True, f"Invoice #{v.get('invoiceNumber')} amount={v.get('amount')} outstanding={v.get('amountOutstanding')}", is_paid
    return False, f"Invoice with date {invoice_date} not found", False


def _check_project(name_contains):
    result = _api("/project", {"fields": "id,name,startDate,customer", "count": "20"})
    for v in result.get("values", []):
        if name_contains.lower() in v.get("name", "").lower():
            has_customer = bool(v.get("customer"))
            return True, f"Found: '{v['name']}' start={v.get('startDate')}", has_customer
    return False, f"Project containing '{name_contains}' not found", False


def _check_department(name_contains):
    result = _api("/department", {"fields": "id,name,departmentNumber", "count": "20"})
    for v in result.get("values", []):
        if name_contains.lower() in v.get("name", "").lower():
            has_number = bool(v.get("departmentNumber"))
            return True, f"Found: '{v['name']}' num={v.get('departmentNumber')}", has_number
    return False, f"Department containing '{name_contains}' not found", False


async def run_test(agent, task, idx, total):
    """Run a single test task."""
    name = task["name"]
    logger.info("=" * 60)
    logger.info(f"[{idx}/{total}] {name}")
    logger.info("=" * 60)

    request = SolveRequest.from_dict({
        "prompt": task["prompt"],
        "files": [],
        "tripletex_credentials": {
            "base_url": SANDBOX_BASE_URL,
            "session_token": SANDBOX_TOKEN,
        },
    })

    start = time.time()
    try:
        response = await agent.solve(request)
        elapsed = time.time() - start
        status = response.status
    except Exception as e:
        elapsed = time.time() - start
        status = f"ERROR: {e}"

    # Verify
    try:
        found, detail, full_pass = task["verify"]()
    except Exception as e:
        found, detail, full_pass = False, f"Verify error: {e}", False

    # Results
    icon = "✅" if full_pass else ("⚠️" if found else "❌")
    logger.info(f"{icon} {name} — {elapsed:.1f}s — {status}")
    logger.info(f"   {detail}")

    return {
        "name": name,
        "tier": task.get("tier", 0),
        "time": round(elapsed, 1),
        "status": status,
        "found": found,
        "full_pass": full_pass,
        "detail": detail,
    }


async def main():
    if not SANDBOX_TOKEN:
        logger.error("TRIPLETEX_SESSION_TOKEN not set in .env")
        return

    agent = TripletexAgent()
    results = []

    for idx, task in enumerate(TEST_TASKS, 1):
        result = await run_test(agent, task, idx, len(TEST_TASKS))
        results.append(result)
        # Brief pause between tests to avoid rate limits
        await asyncio.sleep(2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    passed = sum(1 for r in results if r["full_pass"])
    found = sum(1 for r in results if r["found"])
    total = len(results)
    total_time = sum(r["time"] for r in results)

    for r in results:
        icon = "✅" if r["full_pass"] else ("⚠️" if r["found"] else "❌")
        logger.info(f"  {icon} {r['name']:45} {r['time']:5.1f}s  {r['detail'][:50]}")

    logger.info(f"\nResults: {passed}/{total} full pass, {found}/{total} found, {total_time:.0f}s total")

    if passed == total:
        logger.info("🎉 All tests passed! Ready to submit.")
    elif found == total:
        logger.info("⚠️ All entities created but some checks incomplete.")
    else:
        logger.info("❌ Some tests failed. Review before submitting.")


if __name__ == "__main__":
    asyncio.run(main())
