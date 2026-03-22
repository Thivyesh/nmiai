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
- Sandbox starts FRESH but some tasks have PRE-LOADED data (customers, invoices, etc.)
- Read the prompt carefully to determine CREATE vs FIND:
  - "Create/opprett/crie" → CREATE the entity
  - "has an invoice / outstanding / existing" → FIND it first, it was pre-loaded
  - If FIND fails → fallback to CREATE
- The always pre-existing data: departments, ledger accounts, VAT types, payment types, salary types
- Every entity mentioned in the prompt must EXIST (created or found) as a separate record
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
   - ONLY set "number" if the prompt gives a specific product number
   - If no product number in prompt, OMIT the number field entirely
4. POST /order — with customer, orderDate, deliveryDate (REQUIRED), orderLines with product: {"id": N}
   - ALWAYS include product reference in order lines when products were created
5. POST /invoice — with invoiceDate, invoiceDueDate, customer, orders: [{"id": N}]
6. For payment: READ the "amount" field from the POST /invoice response
   PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=<amount_from_invoice>
   - Do NOT calculate the amount yourself — use the exact amount from the invoice response
7. If task says "send": PUT /invoice/{id}/:send?sendType=EMAIL (after invoice creation)

### Verified Field Gotchas
- Order lines: "count" NOT "quantity"
- Order: deliveryDate is REQUIRED
- Product number: only from prompt, never invented
- Product: do NOT set vatType, account, or currency — causes validation errors
- Payment amount: ALWAYS read from invoice response, never calculate VAT yourself
- Payment: ALL params are query params, not body
- Invoice needs orders array, cannot be created standalone
- "Enhetspris må være uten mva" error: set isPrioritizeAmountsIncludingVat: false on the order, and use unitPriceExcludingVatCurrency on order lines

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
   - Do NOT set userType unless the task requires admin access
   - If POST fails with "Brukertype kan ikke være 0", retry with userType: "STANDARD"
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
1. If prompt specifies VAT rate: GET /ledger/vatType to find correct outgoing ("utgående") vatType
   - 25% = look for "Utgående avgift, høy sats" (id varies per sandbox)
   - 15% = look for "Utgående avgift, middels sats" (food/drink rate)
   - 12% = look for "Utgående avgift, lav sats" (transport rate)
   - 0% = id=6 default (no VAT)
   - NOTE: Only OUTGOING (utgående) vatTypes work on products. INCOMING (inngående/fradrag) will fail.
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
2. POST /travelExpense — include travelDetails if trip dates are mentioned:
   {employee: {"id": N}, title, date: "YYYY-MM-DD", department: {"id": N},
    travelDetails: {isForeignTravel: false, isDayTrip: false,
    departureDate: "YYYY-MM-DD", returnDate: "YYYY-MM-DD",
    departureFrom: "City", destination: "City", purpose: "description"}}
3. For regular expenses (flight, taxi, hotel): POST /travelExpense/cost for EACH line
4. For per diem/diett/dietas: POST /travelExpense/cost with EACH DAY as separate line
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
Keywords: prosjekt, project, Projekt, proyecto, projet, projeto, fixed price, fastpris, milestone, milepæl

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Customer exists? | GET /customer?name=X or ?organizationNumber=N | Must CREATE if not found |
| Project manager | GET /employee?email=X | Must CREATE if not found. projectManager is required |

### Verified Workflow (simple project)
1. GET /employee?email=X — find project manager (CREATE if needed)
2. POST /customer (if needed) — with organizationNumber
3. POST /project — {name, startDate, customer: {"id": N}, projectManager: {"id": N}}

### Verified Workflow (fixed price project with invoice)
1. GET /employee?email=X — find project manager (CREATE if needed)
2. POST /customer (if needed) — with organizationNumber
3. POST /product — create product for the fixed-price line item
4. POST /project — {name, startDate, customer: {"id": N}, projectManager: {"id": N}, isFixedPrice: true}
5. POST /project/orderline — add line item with product, amount, and date (product REQUIRED for invoiceable lines)
6. PUT /project/{id}/:invoice — invoice the project (for milestone: set amount in orderline to milestone %)

### Verified Field Gotchas
- projectManager usually required (use account owner ID if no specific manager named)
- For fixed price: set isFixedPrice: true on POST /project
- For milestone invoice (e.g. "50%"): create orderline with 50% of total amount, then invoice
- Project orderline needs: project {"id": N}, description, unitPriceExcludingVatCurrency, count: 1

---

## TIME TRACKING / PROJECT HOURS TASKS
Keywords: timer, timeføring, hours, timesheet, registrer timer, Stunden, horas, heures, horas de projeto, project invoice, prosjektfaktura

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Employee exists? | GET /employee?email=X | Need employee ID for timesheet |
| Project exists? | GET /project?name=X | Need project ID |
| Activity exists? | GET /project/projectActivity?projectId=N | Need activity ID, CREATE if missing |

### Verified Workflow
1. GET /employee, GET /project, GET /customer — find existing entities
2. GET /project/projectActivity?projectId=N — find or create activity
3. If activity not found: POST /project/projectActivity — {project: {"id": N}, activity: {name: "X", activityType: "PROJECT_SPECIFIC_ACTIVITY"}}
4. POST /timesheet/entry — {project: {"id": N}, activity: {"id": N}, employee: {"id": N}, date: "YYYY-MM-DD", hours: N}
5. If task asks to invoice the project: PUT /project/{id}/:invoice?invoiceDate=YYYY-MM-DD&sendToCustomer=false (query params, body="{}")

### Verified Field Gotchas
- Activity lookup: GET /project/projectActivity?projectId=N (NOT /project/activity)
- Timesheet date is required
- For project invoice: must have hours logged first
- Hourly rates: POST /project/hourlyRates if rate is specified

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
Keywords: bilag, voucher, dimensjon, dimension, postering, Buchung, journal entry, lançamento contabilístico, avskrivning, depreciation, Abschreibung, depreciación, månedslukking, month-end, periodelukking, closing, bokfør, periodisering, accrual

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Account ID | GET /ledger/account?number=NNNN&fields=id | Postings MUST use account {"id": N}, NEVER {"number": N} |
| Balancing account | GET another account for the opposite posting | Postings must sum to zero |

### Verified Workflow
1. GET /ledger/account?number=NNNN&fields=id — get EVERY account ID you need
2. GET /ledger/voucherType?fields=id,name — get a valid voucherType ID (NEVER use id=1)
3. If task mentions dimensions/kostsenter:
   a. POST /ledger/accountingDimensionName — REQUIRED FIRST
      {"dimensionName": "Kostsenter", "description": "Cost center", "dimensionIndex": 1, "active": true}
   b. POST /ledger/accountingDimensionValue — for EACH value
      {"displayName": "IT", "dimensionIndex": 1, "active": true, "number": "IT", "showInVoucherRegistration": true}
4. POST /ledger/voucher?sendToLedger=true — EXACT format below:

### VOUCHER PAYLOAD — Copy this format EXACTLY
```
POST /ledger/voucher?sendToLedger=true
{
  "date": "YYYY-MM-DD",
  "description": "Description of the posting",
  "voucherType": {"id": <id_from_GET_voucherType>},
  "postings": [
    {
      "date": "YYYY-MM-DD",
      "row": 1,
      "account": {"id": <id_from_GET_account>},
      "amountGross": 1000,
      "amountGrossCurrency": 1000,
      "description": "Debit posting"
    },
    {
      "date": "YYYY-MM-DD",
      "row": 2,
      "account": {"id": <id_from_GET_another_account>},
      "amountGross": -1000,
      "amountGrossCurrency": -1000,
      "description": "Credit posting"
    }
  ]
}
```

### CRITICAL Voucher Rules — READ CAREFULLY
- Account: GET the ID first. NEVER use account number as ID.
  WRONG: {"id": 1920}  WRONG: {"id": 8050}
  RIGHT: GET /ledger/account?number=1920 → use returned id
- VoucherType: GET the ID first. NEVER hardcode id=1.
  GET /ledger/voucherType → use an ID from the response
- Amounts: ONLY use "amountGross" and "amountGrossCurrency"
  WRONG: debitAmount, creditAmount, amount
  RIGHT: amountGross (positive=debit, negative=credit)
- EVERY posting needs: date, row, account, amountGross, amountGrossCurrency
- Postings MUST balance (sum of amountGross = zero)
- "Kunde mangler" error: account has ledgerType CUSTOMER — include customer: {"id": N} in posting
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
| Invoice ID | GET /invoice with date filters | Need the exact invoice |
| Payment type | GET /invoice/paymentType | Need paymentTypeId |
| Bank account | GET /ledger/account?number=1920&fields=id,version,bankAccountNumber | Invoice needs bank account |

### Verified Workflow
NOTE: The invoice may ALREADY EXIST (pre-loaded). Try to FIND it first.
1. GET /customer?organizationNumber=N or ?name=X — find the customer
2. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&customerId=N&fields=id,invoiceNumber,amount,amountOutstanding,comment — find existing invoice
3. If invoice NOT found: create it (customer → order → invoice flow)
4. GET /invoice/paymentType — find paymentTypeId
5. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=<AMOUNT_FROM_INVOICE>
   - Use the "amount" value from the invoice, NOT a calculated value

### Verified Field Gotchas
- paidAmount: ALWAYS use the "amount" from the GET /invoice response, never calculate it
- ALL payment params are query params, NOT body. Pass body="{}"
- GET /invoice REQUIRES invoiceDateFrom AND invoiceDateTo — use wide range 2020-2030
- Invoice valid fields: id, invoiceNumber, amount, amountOutstanding, comment, invoiceDate, invoiceDueDate
- Invoice does NOT have: "description" (use "comment"), "amountPaid", "amountExcludingVat"
- Account path: ALWAYS /ledger/account — NOT /account or /bank

---

## PAYMENT REVERSAL / CANCELLATION TASKS
Keywords: tilbakeføring, stornering, reversering, zurückgebucht, stornieren, reverse payment, cancel payment, annuler paiement, estornar pagamento

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Customer ID | GET /customer?organizationNumber=N or ?name=X | Need customer to find invoice |
| Invoice ID | GET /invoice?invoiceDateFrom=YYYY-01-01&invoiceDateTo=YYYY-12-31&customerId=N&fields=id,amount,amountOutstanding,comment | Find the paid invoice |
| Payment type | GET /invoice/paymentType | Need paymentTypeId |
| Bank account | GET /ledger/account?number=1920&fields=id,version,bankAccountNumber | May need to set bank account |

### IMPORTANT: Try to FIND the invoice first, CREATE if not found
The task describes an existing paid invoice. Try to find it:
1. Find the customer
2. Search for the invoice
3. If found with payment → just reverse
4. If NOT found → create everything from scratch

### Verified Workflow
1. GET /customer?organizationNumber=N or ?name=X — find customer
2. If not found: POST /customer
3. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&customerId=N&fields=id,amount,amountOutstanding,comment — find invoice
4. If invoice found and amountOutstanding=0 (already paid):
   - GET /invoice/paymentType → paymentTypeId
   - PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=-<amount> — reverse
5. If invoice NOT found:
   a. Check/set bank account on ledger account 1920
   b. POST /order → POST /invoice — create invoice
   c. Read "amount" from invoice response
   d. GET /invoice/paymentType → paymentTypeId
   e. PUT /:payment with paidAmount=<amount_from_invoice> — register payment
   f. PUT /:payment with paidAmount=-<amount_from_invoice> — reverse payment
7. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=-AMOUNT — reverse with NEGATIVE amount

### Verified Field Gotchas
- Use NEGATIVE paidAmount to reverse (e.g., paidAmount=-53125)
- ALL payment params are query params, NOT body. Pass body="{}"
- Account path: /ledger/account NOT /account or /bank
- Invoice fields: use "comment" NOT "description", no "amountPaid" field
- GET /invoice REQUIRES invoiceDateFrom AND invoiceDateTo
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

## SALARY / PAYROLL TASKS
Keywords: lønn, salary, payroll, Gehalt, salario, salaire, lønnskjøring, kjør lønn

### Prerequisites (MUST check)
| Check | How | Why |
|-------|-----|-----|
| Employee exists? | GET /employee?email=X | Must exist with employment record |
| Department ID | GET /department?fields=id&count=1 | Needed for employee |
| Division exists? | GET /division?fields=id&count=1 | Employment MUST have division for salary |
| Municipality ID | GET /municipality?fields=id,name&count=5 | Needed if creating division |
| Company info | GET /token/session/>whoAmI?fields=companyId then GET /company/{id}?fields=organizationNumber | Division needs DIFFERENT org number. Or just use "999999999" as org number. |
| Salary types | GET /salary/type?fields=id,number,name | Need IDs for salary specifications |

### CRITICAL: Check if employee already exists FIRST
1. GET /employee?email=X — if employee found:
   - GET /employee/employment?employeeId=N — check if employment exists
   - If employment exists with division → SKIP steps 1-4, go straight to salary transaction
   - Do NOT create new employment/division for existing employees
   - Do NOT modify existing employee fields (dateOfBirth, userType, etc.)
2. If employee NOT found, create everything:

### Verified Workflow (only for NEW employees)
1. POST /employee — {firstName, lastName, email, department: {"id": N}, userType: "STANDARD", dateOfBirth: "YYYY-MM-DD"}
   - userType REQUIRED for salary employees — use "STANDARD"
   - dateOfBirth REQUIRED for employment creation
2. GET /division — check if division exists
3. If no division: Create one:
   a. GET /municipality?fields=id,name&count=1 → get any municipality ID
   b. POST /division:
      {name: "Hovedvirksomhet", startDate: "2026-01-01",
       organizationNumber: "999999999",
       municipality: {"id": N},
       municipalityDate: "2026-01-01"}
   - organizationNumber MUST be different from company's — use any 9-digit number
   - municipalityDate is REQUIRED (use same as startDate)
   - municipality is REQUIRED — get ID via GET /municipality
4. POST /employee/employment — {employee: {"id": N}, startDate: "YYYY-MM-DD", isMainEmployer: true, division: {"id": N}}
   - Division is REQUIRED — "Arbeidsforholdet er ikke knyttet mot en virksomhet" without it
5. GET /salary/type — find salary type IDs:
   - num=2000 "Fastlønn" (base salary)
   - num=2002 "Bonus"
   - num=2001 "Timelønn" (hourly)
6. POST /salary/transaction:
```
{
  "date": "YYYY-MM-DD",
  "year": 2026,
  "month": 3,
  "isHistorical": false,
  "payslips": [{
    "employee": {"id": N},
    "date": "YYYY-MM-DD",
    "year": 2026,
    "month": 3,
    "specifications": [
      {"salaryType": {"id": <fastlonn_id>}, "rate": 53350, "count": 1, "amount": 53350},
      {"salaryType": {"id": <bonus_id>}, "rate": 11050, "count": 1, "amount": 11050}
    ]
  }]
}
```

### Fallback: Manual voucher if salary API fails
1. GET /ledger/account?number=5000&fields=id (salary expense)
2. GET /ledger/account?number=2930&fields=id (salary payable)
3. POST /ledger/voucher?sendToLedger=true with balanced postings

### Verified Field Gotchas
- Division is REQUIRED on employment for salary — create one if none exists
- Division organizationNumber must be DIFFERENT from company org number
- userType: "STANDARD" required on employee for salary (not NO_ACCESS)
- dateOfBirth required on employee before employment can be created
- year MUST be current year (2026) — past years fail with "Ugyldig år"
- Specification requires: salaryType, rate, count (rate cannot be null)
- amount = rate × count
- Salary type IDs vary per sandbox — always GET /salary/type first

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
