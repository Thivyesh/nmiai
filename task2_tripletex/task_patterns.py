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
- Sandbox is ALWAYS FRESH — no customers, products, employees, suppliers exist
- NEVER search for existing entities — they DO NOT EXIST. Always CREATE them.
- The only pre-existing data: departments, ledger accounts, VAT types, payment types, salary types
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
2. If task mentions dimensions/kostsenter:
   a. POST /ledger/accountingDimensionName — REQUIRED FIRST, creates the dimension container
      {"dimensionName": "Kostsenter", "description": "Cost center", "dimensionIndex": 1, "active": true}
      This is NOT automatic — you MUST create it before creating values!
   b. POST /ledger/accountingDimensionValue — for EACH value
      {"displayName": "IT", "dimensionIndex": 1, "active": true, "number": "IT", "showInVoucherRegistration": true}
3. POST /ledger/voucher?sendToLedger=true — with postings

### Verified Field Gotchas
- Account: ALWAYS {"id": N} — NEVER {"number": N}
- Amounts: use amountGross AND amountGrossCurrency (both required, must be equal for NOK)
- Do NOT use "amount" — it does not work
- Include "date" and "row" on each posting
- "Kunde mangler"/"Leverandør mangler" error: the account has ledgerType CUSTOMER/SUPPLIER — include customer/supplier ID in the posting
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
| Invoice ID | GET /invoice with date filters | Need the exact invoice |
| Payment type | GET /invoice/paymentType | Need paymentTypeId |
| Bank account | GET /ledger/account?number=1920&fields=id,version,bankAccountNumber | Invoice needs bank account |

### Verified Workflow
1. GET /invoice?invoiceDateFrom=YYYY-01-01&invoiceDateTo=YYYY-12-31&customerId=N&fields=id,invoiceNumber,amount,amountOutstanding,comment
2. GET /invoice/paymentType — find paymentTypeId
3. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=N

### Verified Field Gotchas
- ALL payment params are query params, NOT body. Pass body="{}"
- GET /invoice REQUIRES invoiceDateFrom AND invoiceDateTo — will fail without them
- Invoice valid fields: id, invoiceNumber, amount, amountOutstanding, amountCurrency, comment, invoiceDate, invoiceDueDate, isCreditNote
- Invoice does NOT have: "description" (use "comment"), "amountPaid" (use "amountOutstanding"), "amountExcludingVat" fails in fields
- Account path: ALWAYS /ledger/account — NOT /account or /bank
- paidAmount must match invoice amount for full payment

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

### IMPORTANT: Fresh sandbox has NO invoices
On a fresh competition sandbox, the invoice doesn't exist yet. You must:
1. Create the customer
2. Check/set bank account on ledger account 1920
3. Create the invoice (order → invoice)
4. Register the payment
5. Then reverse the payment with negative amount

### Verified Workflow
1. POST /customer — create the customer
2. GET /ledger/account?number=1920&fields=id,version,bankAccountNumber — check bank account
3. If empty: PUT /ledger/account/{id} with bankAccountNumber
4. POST /order → POST /invoice — create the original invoice
5. GET /invoice/paymentType — find paymentTypeId
6. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=AMOUNT — register original payment
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

### Verified Workflow
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
