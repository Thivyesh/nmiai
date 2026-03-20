"""Curated Tripletex API reference for the planner node.

Extracted from the OpenAPI spec and validated by testing against the sandbox.
Covers all competition-relevant endpoints with gotchas discovered empirically.
"""

API_REFERENCE = """\
# Tripletex API v2 — Quick Reference (Verified)

## Authentication
All requests use Basic Auth: username="0", password=<session_token>.
All calls go through the provided base_url (proxy).

## Response Format
- Single entity: {"value": {...}}
- List: {"fullResultSize": N, "values": [...]}
- Use ?fields=id,name,... to select fields. Use ?fields=* for all.
- Pagination: ?from=0&count=100

## CRITICAL GOTCHAS (learned from testing)
1. **Employee department is REQUIRED** — POST /employee fails without department. Always GET /department first.
2. **userType is READ-ONLY** — Do NOT set on POST/PUT. Use entitlements endpoint instead.
3. **Invoice requires bank account on company** — Before creating invoices, ensure ledger account 1920 (Bankinnskudd) has a bankAccountNumber set.
4. **Invoice requires an Order** — Flow: Customer → Order (with orderLines) → Invoice (with orders).
5. **PUT for updates needs id + version** — Always include both from the GET response.
6. **Payment on invoice uses QUERY PARAMS** — PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=Y&paidAmount=Z (NOT body).
7. **Credit note needs date query param** — PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD
8. **Entitlements use query params** — PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=X&template=Y (no body).
9. **Employee PUT requires dateOfBirth** — When updating an employee, dateOfBirth must be included.
10. **Product vatType** — Products default to vatType id=6 (no VAT). Most other vatType IDs fail on products.
11. **Voucher postings need amountGross + amountGrossCurrency** — Both must be equal for NOK. Use row field. Some accounts (like 2400 Leverandørgjeld, 1920 Bankinnskudd) are system-managed and cannot be used in manual vouchers.
12. **Travel expense date** — POST /travelExpense ignores the date field and uses today's date.
13. **Order lines work without product** — You can create order lines with just description, count, and unitPriceExcludingVatCurrency.

---

## EMPLOYEE
### GET /employee
Query params: firstName, lastName, email, fields, from, count
### POST /employee
REQUIRED: department ({"id": N})
Key fields:
- firstName (string)
- lastName (string)
- email (string)
- dateOfBirth (string, YYYY-MM-DD) — required for PUT updates
- phoneNumberMobile (string)
- phoneNumberHome (string)
- phoneNumberWork (string)
- department ({"id": N}) — REQUIRED, get via GET /department
- nationalIdentityNumber (string)
- isContact (boolean) — true = external contact
- allowInformationRegistration (boolean)
- bankAccountNumber (string)
NOTE: userType is READ-ONLY. Do NOT set it. Use entitlements instead.
### PUT /employee/{id}
Same fields as POST. MUST include "id", "version", and "dateOfBirth" in body.

### EMPLOYEE ENTITLEMENTS (roles/permissions)
To assign roles (kontoadministrator, etc.), use entitlements:
### PUT /employee/entitlement/:grantEntitlementsByTemplate
QUERY PARAMS (not body):
- employeeId (integer) — the employee to grant entitlements to
- template (enum):
  - "ALL_PRIVILEGES" — kontoadministrator / full admin
  - "NONE_PRIVILEGES" — remove all privileges
  - "INVOICING_MANAGER" — invoicing access
  - "PERSONELL_MANAGER" — HR/personnel access
  - "ACCOUNTANT" — accountant access
  - "AUDITOR" — auditor access
  - "DEPARTMENT_LEADER" — department leader
No request body. This is a PUT with query params only.
### GET /employee/entitlement
Query: employeeId, fields, from, count

### EMPLOYEE EMPLOYMENT
### POST /employee/employment
Fields: employee ({"id": N}), startDate, endDate, employmentType ({"id": N})
Employment types: 1=Ordinært, 2=Maritimt, 3=Frilanser, 4=Pensjon

---

## CUSTOMER
### GET /customer
Query params: name, email, customerNumber, isCustomer, fields, from, count
### POST /customer
Key fields:
- name (string)
- email (string)
- phoneNumber (string)
- phoneNumberMobile (string)
- isCustomer (boolean) — set true for customers
- isSupplier (boolean)
- isPrivateIndividual (boolean)
- invoiceEmail (string)
- invoiceSendMethod (enum: "EMAIL" | "EHF" | "EFAKTURA" | "PAPER" | "MANUAL")
- language (enum: "NO" | "EN" | "SV" | "DA" | "FI" | "ES" | "DE" | "FR")
- invoicesDueIn (integer) — payment terms in days
- invoicesDueInType (enum: "DAYS" | "MONTHS" | "RECURRING_DAY_OF_MONTH")
- postalAddress ({"addressLine1": "", "postalCode": "", "city": ""})
- physicalAddress (same structure)
- department ({"id": N})
- currency ({"id": N}) — 1=NOK, 2=SEK, 3=DKK, 4=USD, 5=EUR, 6=GBP
### PUT /customer/{id}
Include "id" and "version".
### DELETE /customer/{id}

---

## PRODUCT
### GET /product
Query: name, number, fields, from, count
### POST /product
Key fields:
- name (string)
- number (string) — product number/SKU
- priceExcludingVatCurrency (number)
- priceIncludingVatCurrency (number)
- costExcludingVatCurrency (number)
- vatType ({"id": N}) — defaults to id=6. Most other IDs fail.
- isStockItem (boolean)
- isInactive (boolean)
- description (string)
- productUnit ({"id": N}) — stk=3924317, l=3924312, m=3924313, kg=3924316
- currency ({"id": N})
### PUT /product/{id}
Include "id" and "version".

---

## ORDER
### GET /order
Query: customerId, number, fields, from, count
### POST /order
Key fields:
- customer ({"id": N}) — REQUIRED
- orderDate (string, YYYY-MM-DD) — REQUIRED
- deliveryDate (string, YYYY-MM-DD) — REQUIRED
- orderLines (array):
  - description (string)
  - count (number) — quantity
  - unitPriceExcludingVatCurrency (number)
  - product ({"id": N}) — optional, can use description instead
  - vatType ({"id": N}) — optional
- department ({"id": N})
- project ({"id": N})
- reference (string)
- invoiceComment (string)
### POST /order/orderline
Fields: order ({"id": N}), description, count, unitPriceExcludingVatCurrency
### POST /order/orderline/list
Batch create order lines.

---

## INVOICE
### PREREQUISITE: Bank account must be set on the company
Before creating any invoice, ensure account 1920 has a bank account number:
1. GET /ledger/account?number=1920&fields=id,version,bankAccountNumber
2. If bankAccountNumber is empty: PUT /ledger/account/{id} with bankAccountNumber set

### GET /invoice
Query: invoiceDateFrom, invoiceDateTo, customerId, fields, from, count
### POST /invoice
Key fields:
- invoiceDate (string, YYYY-MM-DD)
- invoiceDueDate (string, YYYY-MM-DD)
- customer ({"id": N})
- orders (array: [{"id": N}]) — REQUIRED, link to existing order(s)
- comment (string)
- invoiceNumber (integer) — 0 or omit for auto
- currency ({"id": N})
Flow: POST /customer → POST /order (with orderLines) → POST /invoice

### PUT /invoice/{id}/:payment — Register payment
ALL PARAMS ARE QUERY PARAMS (not body):
- paymentDate (string, YYYY-MM-DD) — REQUIRED
- paymentTypeId (integer) — REQUIRED (get from GET /invoice/paymentType)
- paidAmount (number) — REQUIRED
Common paymentTypeIds (sandbox-specific, may vary):
  - "Kontant" and "Betalt til bank" — get actual IDs via GET /invoice/paymentType

### PUT /invoice/{id}/:createCreditNote — Create credit note
QUERY PARAMS:
- date (string, YYYY-MM-DD) — REQUIRED
- comment (string) — optional
- sendToCustomer (boolean) — optional

---

## TRAVEL EXPENSE
### GET /travelExpense
Query: employeeId, fields, from, count
### POST /travelExpense
Key fields:
- employee ({"id": N})
- title (string)
- date (string, YYYY-MM-DD) — note: may default to today
- project ({"id": N})
- department ({"id": N})
- isChargeable (boolean)
### PUT /travelExpense/{id}
### DELETE /travelExpense/{id} — returns HTTP 204
### PUT /travelExpense/:deliver — deliver for approval
### PUT /travelExpense/:approve — approve

### POST /travelExpense/cost — add cost line
Fields: travelExpense ({"id": N}), description, amount, date, vatType, currency, paymentType

---

## PROJECT
### GET /project
Query: name, number, projectManagerId, customerId, fields, from, count
### POST /project
Key fields:
- name (string)
- number (string) — auto if NULL
- startDate (string, YYYY-MM-DD)
- endDate (string, YYYY-MM-DD)
- projectManager ({"id": N}) — employee who manages the project
- customer ({"id": N})
- department ({"id": N})
- isInternal (boolean)
- isClosed (boolean)
- isFixedPrice (boolean)
- description (string)
- projectCategory ({"id": N})
### PUT /project/{id}
### DELETE /project

---

## DEPARTMENT
### GET /department
Query: name, departmentNumber, fields, from, count
### POST /department
Key fields:
- name (string)
- departmentNumber (string)
- departmentManager ({"id": N}) — employee
### PUT /department/{id}
### DELETE /department/{id}

---

## CONTACT
### GET /contact
Query: firstName, lastName, email, customerId, fields, from, count
### POST /contact
Key fields:
- firstName (string)
- lastName (string)
- email (string)
- phoneNumber (string)
- customer ({"id": N}) — link to customer
### PUT /contact/{id}

---

## LEDGER
### GET /ledger/account — chart of accounts
Query: number, fields, from, count
Key accounts: 1500=Kundefordringer, 1920=Bankinnskudd, 1900=Kontanter, 2400=Leverandørgjeld, 3000=Salgsinntekt, 6800=Kontorrekvisita
### PUT /ledger/account/{id} — update account (e.g., set bankAccountNumber)
### GET /ledger/vatType — VAT types
Common: id=3 (25% utgående), id=6 (ingen avgift), id=31 (15% middels), id=32 (12% lav)
### GET /ledger/voucherType — voucher types
Common: Utgående faktura, Leverandørfaktura, Betaling, Bankavstemming
### POST /ledger/voucher?sendToLedger=true — create voucher
Body: date, description, postings (array)
Each posting needs: date, row, account ({"id": N}), amountGross, amountGrossCurrency, description
NOTE: amountGross and amountGrossCurrency must be equal for NOK. Debits are positive, credits negative.
### DELETE /ledger/voucher/{id}

---

## COMPANY
### GET /company/{id}
### PUT /company — update company info (id + version required)
### GET /company/salesmodules — list active modules
### POST /company/salesmodules — activate a module

---

## REFERENCE DATA (sandbox values)
### Currencies: 1=NOK, 2=SEK, 3=DKK, 4=USD, 5=EUR, 6=GBP
### Product units: stk=3924317, l=3924312, m=3924313, km=3924314, g=3924315, kg=3924316
### Employment types: 1=Ordinært, 2=Maritimt, 3=Frilanser, 4=Pensjon

---

## COMMON TASK PATTERNS (competition)

### Create employee as administrator (kontoadministrator)
1. GET /department?fields=id&count=1 → get department_id
2. POST /employee with firstName, lastName, email, department={"id": department_id} → get employee_id
3. PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId={employee_id}&template=ALL_PRIVILEGES

### Create invoice for a customer
1. GET /ledger/account?number=1920&fields=id,version,bankAccountNumber → check bank account
2. (If empty) PUT /ledger/account/{id} with bankAccountNumber set
3. POST /customer (if not exists) → get customer_id
4. POST /order with customer, orderDate, deliveryDate, orderLines → get order_id
5. POST /invoice with invoiceDate, invoiceDueDate, customer, orders=[{id: order_id}]

### Register payment on invoice
1. GET /invoice/paymentType → get paymentTypeId
2. PUT /invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=N

### Create credit note
1. PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD

### Create project linked to customer
1. POST /customer (if not exists) → get customer_id
2. GET /employee → find project manager employee_id
3. POST /project with name, customer, projectManager, startDate

### Delete travel expense
1. GET /travelExpense with filters → find the expense
2. DELETE /travelExpense/{id}

### Update employee (e.g., add phone)
1. GET /employee/{id}?fields=* → get version, dateOfBirth, and current data
2. PUT /employee/{id} with id, version, dateOfBirth, plus updated fields

### Create department
1. POST /department with name and departmentNumber

### Enable department accounting module
1. POST /company/salesmodules to activate the module
2. POST /department with name and departmentNumber
"""
