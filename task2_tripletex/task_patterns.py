"""Verified task patterns for the Tripletex competition researcher.

Every fact here has been VERIFIED through sandbox testing or competition results.
Structured as risk checklists so the researcher finds what it needs fast.

Source of truth:
- Sandbox testing (March 20, 2026)
- Competition results: 7/7, 8/8, 6.5/8, 0/8, 6/13, 6/7, 7/7
"""

TASK_PATTERNS = """\
# RESEARCHER CHECKLIST — Tripletex Competition

## UNIVERSAL PREREQUISITES (check for EVERY task)
- Sandbox is FRESH — no customers, products, employees exist (only account owner)
- Every entity mentioned in the prompt must be CREATED as a separate record
- Every field value in the prompt WILL be checked field-by-field
- Use EXACT values from prompt — never modify names, emails, amounts

---

## INVOICE TASKS
Keywords: faktura, invoice, Rechnung, factura, facture, fatura

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Bank account on 1920 | GET /ledger/account?number=1920&fields=id,version,bankAccountNumber | Invoice creation FAILS without this. If empty, PUT to set bankAccountNumber="86011117947" |
| Customer exists? | Prompt names the customer | Must CREATE with name, email, organizationNumber, isCustomer: true |
| Products referenced? | Prompt mentions product names/numbers | Must CREATE each product. Do NOT set vatType (only default id=6 works) |
| Payment type needed? | Prompt asks for payment registration | GET /invoice/paymentType to find ID |

### Verified Workflow
1. PUT /ledger/account/{id} — set bank account (if empty)
2. POST /customer — with organizationNumber if given
3. POST /product — for each product (name, number, priceExcludingVatCurrency only)
4. POST /order — with customer, orderDate, deliveryDate (REQUIRED), orderLines with product refs
5. POST /invoice — with invoiceDate, invoiceDueDate, customer, orders: [{"id": N}]
6. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=N (query params, no body)

### Verified Field Gotchas
- Order lines: "count" NOT "quantity"
- Order: deliveryDate is REQUIRED
- Product: do NOT set vatType, account, or currency — causes validation errors
- Payment: ALL params are query params, not body
- Invoice needs orders array, cannot be created standalone

---

## EMPLOYEE TASKS
Keywords: ansatt, employee, Mitarbeiter, empleado, employé, empregado, funcionário

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Department ID | GET /department?fields=id&count=1 | Employee POST FAILS without department |
| Admin role? | Prompt says "kontoadministrator" / "admin" | Need entitlements endpoint after creation |
| Start date? | Prompt mentions start date / "data de início" / "startdato" | Need separate POST /employee/employment |

### Verified Workflow
1. POST /employee — {firstName, lastName, email, phoneNumberMobile, dateOfBirth, department: {"id": N}}
   - Do NOT include "employments" in the employee POST body — create employment separately
   - If sandbox requires userType, use "NO_ACCESS" as default
2. POST /employee/employment — {employee: {"id": N}, startDate: "YYYY-MM-DD", isMainEmployer: true}
   - Only if the prompt mentions a start date / employment start
   - This is a SEPARATE call, not embedded in the employee POST
3. PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=N&template=ALL_PRIVILEGES (if admin)

### Verified Field Gotchas
- Phone: "phoneNumberMobile" NOT "phoneNumber" (that's for customers!)
- userType: if POST fails with "Brukertype kan ikke være 0", retry with userType: "NO_ACCESS"
- Employment: create via separate POST /employee/employment, NOT as embedded array in employee POST
- dateOfBirth: include if mentioned in prompt (format YYYY-MM-DD)
- Entitlements: query params only, no body, path is exactly /employee/entitlement/:grantEntitlementsByTemplate
- PUT /employee requires dateOfBirth even if not changing it

---

## CUSTOMER TASKS
Keywords: kunde, customer, Kunde, cliente, client

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| None | — | Customer creation is straightforward |

### Verified Workflow
1. POST /customer — {name, email, phoneNumber, organizationNumber, isCustomer: true}

### Verified Field Gotchas
- Phone: "phoneNumber" for customers (NOT phoneNumberMobile — that's for employees!)
- Always set isCustomer: true
- Include organizationNumber if prompt mentions org.nr / org number
- Include postalAddress: {addressLine1, postalCode, city} if address given

---

## PRODUCT TASKS
Keywords: produkt, product, Produkt, producto, produit, produto

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Product number in use? | Only if resubmitting | GET /product?number=N — if exists, use that ID |

### Prerequisites (MUST check if VAT is mentioned)
| Check | How | Why |
|-------|-----|-----|
| VAT type ID | GET /ledger/vatType?fields=id,number,name,percentage | If prompt specifies a VAT rate, find the matching outgoing vatType |

### Verified Workflow
1. If prompt specifies VAT rate: GET /ledger/vatType to find correct ID
2. POST /product — {name, number, priceExcludingVatCurrency, vatType: {"id": N}}
   - Only set vatType if the prompt explicitly asks for a specific VAT rate
   - If no VAT mentioned, OMIT vatType (default id=6 = no VAT)

### Verified Field Gotchas
- Do NOT set account or currency — causes validation errors
- vatType: only set if prompt specifies a VAT rate. Some sandbox configs reject non-default vatTypes.
- If vatType fails, retry WITHOUT it — default (id=6, no VAT) always works
- Price: priceExcludingVatCurrency for "eks. mva" / "without VAT", priceIncludingVatCurrency for "inkl. mva" / "with VAT"

---

## TRAVEL EXPENSE TASKS
Keywords: reiseregning, travel expense, Reisekostenabrechnung, nota de gastos, note de frais

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Employee exists? | GET /employee?email=X or by name | Must CREATE if not found |
| Department ID | GET /department?fields=id&count=1 | Needed for employee and travel expense |
| Payment type | GET /travelExpense/paymentType | REQUIRED for each cost line |

### Verified Workflow
1. POST /employee (if needed) — with department
2. POST /travelExpense — {employee: {"id": N}, title, date: "YYYY-MM-DD", department: {"id": N}}
3. For regular expenses (flight, taxi, hotel): POST /travelExpense/cost for EACH line
4. For per diem/diett/dietas: POST /travelExpense/cost with EACH DAY as separate line (count × rate)
   - Create ONE cost line per day, OR one cost line with total amount
   - Always include "date" field on cost lines

### Verified Field Gotchas
- Cost amount: "amountCurrencyIncVat" NOT "amount"
- Cost description: "category" NOT "description"
- Cost date: ALWAYS include "date": "YYYY-MM-DD" on every cost line
- paymentType: REQUIRED on every cost line — {"id": N}
- isPaidByEmployee: true (usually)
- Per diem/diett: create as regular cost lines. Include the daily rate info in the category.
  Example: {category: "Diett dag 1", amountCurrencyIncVat: 800, date: "2026-01-15", ...}
  Create separate cost lines for each day if possible.

---

## PROJECT TASKS
Keywords: prosjekt, project, Projekt, proyecto, projet, projeto

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Customer exists? | GET /customer?name=X | Must CREATE if not found |
| Employee for manager | GET /employee?fields=id&count=1 | projectManager is expected |

### Verified Workflow
1. POST /customer (if needed)
2. POST /project — {name, startDate, customer: {"id": N}, projectManager: {"id": N}}

### Verified Field Gotchas
- projectManager usually required (use account owner ID if no specific manager named)

---

## CREDIT NOTE TASKS
Keywords: kreditnota, credit note, nota de crédito, Gutschrift

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Invoice to credit | GET /invoice with filters | Need the invoice ID |

### Verified Workflow
1. PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD (query params, no body)

### Verified Field Gotchas
- Date is REQUIRED as query param
- Use PUT not POST

---

## DEPARTMENT TASKS
Keywords: avdeling, department, Abteilung, departamento

### Prerequisites: None
### Verified Workflow
1. POST /department — {name, departmentNumber}

---

## SUPPLIER TASKS
Keywords: leverandør, supplier, Lieferant, proveedor, fournisseur, fornecedor

### Prerequisites: None
### Verified Workflow
1. POST /supplier — {name, organizationNumber, email}

### Verified Field Gotchas
- There is a separate /supplier endpoint (not /customer with isSupplier)

---

## VOUCHER / DIMENSION TASKS
Keywords: bilag, voucher, dimensjon, dimension, postering, Buchung, journal entry, lançamento contabilístico

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Account ID | GET /ledger/account?number=NNNN&fields=id | Postings MUST use account {"id": N}, NEVER {"number": N} |
| Balancing account | GET another account for the opposite posting | Postings must sum to zero |

### Verified Workflow
1. GET /ledger/account?number=NNNN&fields=id — get account IDs
2. POST /ledger/accountingDimensionName (if creating dimensions)
3. POST /ledger/accountingDimensionValue (for each value)
4. POST /ledger/voucher?sendToLedger=true — with postings

### Verified Field Gotchas
- Account: ALWAYS {"id": N} — NEVER {"number": N}
- Amounts: use amountGross AND amountGrossCurrency (both required, must be equal for NOK)
- Do NOT use "amount" — it does not work
- Include "date" and "row" on each posting
- Postings MUST balance (sum of amountGross = zero)
- freeAccountingDimension1/2/3 for linking to dimension values
- Some accounts (1920, 2400) are system-managed — use 1900 Kontanter for balancing

---

## DELETE TASKS
Keywords: slett, delete, eliminar, löschen, supprimer, fjern

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Entity to delete | GET with filters to find ID | Need the exact ID |

### Verified Workflow
1. GET the entity with filters → find ID
2. DELETE /entity/{id}

---

## UPDATE TASKS
Keywords: oppdater, endre, update, actualizar, ändern, modifier

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Current entity state | GET /entity/{id}?fields=* | Need version and current values |

### Verified Workflow
1. GET entity with ?fields=* → get id, version, current values
2. PUT /entity/{id} — include id, version, plus changed fields

### Verified Field Gotchas
- Employee PUT requires dateOfBirth even if not changing it
- Always include version field from GET response

---

## PAYMENT TASKS (register payment)
Keywords: betaling, payment, Zahlung, pago, paiement, pagamento, registrer betaling

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Invoice ID | GET /invoice with date filters (invoiceDateFrom, invoiceDateTo REQUIRED) | Need the exact invoice |
| Payment type | GET /invoice/paymentType | Need paymentTypeId |

### Verified Workflow
1. GET /invoice?invoiceDateFrom=YYYY-01-01&invoiceDateTo=YYYY-12-31&customerId=N — find the invoice
2. GET /invoice/paymentType — find paymentTypeId
3. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=N

### Verified Field Gotchas
- ALL params are query params, NOT body. Pass body="{}"
- GET /invoice REQUIRES invoiceDateFrom AND invoiceDateTo — will fail without them
- paidAmount must match invoice amount for full payment

---

## PAYMENT REVERSAL / CANCELLATION TASKS
Keywords: tilbakeføring, stornering, reversering, zurückgebucht, stornieren, reverse payment, cancel payment, annuler paiement, estornar pagamento

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Invoice ID | GET /invoice with date filters | Need the invoice that was paid |
| Payment type | GET /invoice/paymentType | Need paymentTypeId |
| Invoice amount | From GET /invoice response | Need exact amount for negative payment |

### Verified Workflow (negative payment method)
1. GET /invoice?invoiceDateFrom=YYYY-01-01&invoiceDateTo=YYYY-12-31&customerId=N — find the paid invoice
2. GET /invoice/paymentType — find paymentTypeId
3. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=-AMOUNT (NEGATIVE amount reverses payment)

### Alternative: Voucher reversal method
If the invoice has a voucher, you can reverse it:
1. GET /invoice/{id}?fields=voucher — get the voucher ID
2. PUT /ledger/voucher/{voucherId}/:reverse — reverses the voucher

### Verified Field Gotchas
- Use NEGATIVE paidAmount to reverse a payment (e.g., paidAmount=-55375)
- ALL params are query params, NOT body
- The invoice must already have a payment registered to reverse it
- After reversal, invoice amountOutstanding should equal the original amount

---

## INCOMING INVOICE / SUPPLIER INVOICE TASKS
Keywords: leverandørfaktura, supplier invoice, incoming invoice, fatura do fornecedor, facture fournisseur, Eingangsrechnung

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Supplier exists? | GET /supplier?name=X | Must CREATE with POST /supplier if not found |
| Account ID | GET /ledger/account?number=NNNN&fields=id | Need account ID for order lines |
| VAT type | GET /ledger/vatType?fields=id,number,name | Need vatTypeId for deductions |

### Verified Workflow
1. POST /supplier — {name, organizationNumber, email}
2. GET /ledger/account?number=NNNN&fields=id — get account ID for expense posting
3. POST /incomingInvoice?sendTo=ledger — register the invoice

### CRITICAL: Incoming invoice uses FLAT IDs, not nested objects!
The /incomingInvoice endpoint uses a DIFFERENT field format than other endpoints:
```
POST /incomingInvoice?sendTo=ledger
{
  "invoiceHeader": {
    "vendorId": 123,           ← flat integer, NOT {"id": 123}
    "invoiceDate": "YYYY-MM-DD",
    "dueDate": "YYYY-MM-DD",
    "currencyId": 1,           ← flat integer (1=NOK)
    "invoiceAmount": 35950,    ← total amount including VAT
    "description": "...",
    "invoiceNumber": "INV-2026-5787"
  },
  "orderLines": [
    {
      "row": 1,
      "description": "...",
      "accountId": 456,        ← flat integer, NOT {"id": 456}
      "amountInclVat": 35950,  ← amount including VAT
      "vatTypeId": 1,          ← flat integer (1 = 25% inbound VAT)
      "count": 1
    }
  ]
}
```

### Verified Field Gotchas
- ALL IDs are flat integers (vendorId, currencyId, accountId, vatTypeId) — NOT nested objects
- invoiceAmount is the total INCLUDING VAT
- orderLines use amountInclVat NOT amountCurrencyIncVat
- vendorId comes from POST /supplier response (value.id)
- currencyId: 1=NOK, 2=SEK, 5=EUR
- vatTypeId: 1=25% inbound, 11=15% inbound, 12=12% inbound

---

## MULTILINGUAL FIELD REFERENCE
| Prompt term | API field | Entity |
|-------------|-----------|--------|
| telefon/phone | phoneNumberMobile | Employee |
| telefon/phone | phoneNumber | Customer |
| org.nr/organization number | organizationNumber | Customer |
| antall/quantity/cantidad | count | Order line |
| pris/price | priceExcludingVatCurrency | Product |
| beløp/amount | amountCurrencyIncVat | Travel cost |
| kategori/category | category | Travel cost |
| fakturadato/invoice date | invoiceDate | Invoice |
| forfallsdato/due date | invoiceDueDate | Invoice |
"""
