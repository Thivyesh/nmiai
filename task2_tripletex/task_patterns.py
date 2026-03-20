"""Accounting task patterns for the Tripletex competition.

Each pattern describes:
- What the task asks for
- What fields the competition likely checks
- The exact API workflow to achieve full marks
- Common mistakes to avoid
- Multilingual field mapping (Norwegian/English/Spanish/German/French/Portuguese/Nynorsk)
"""

TASK_PATTERNS = """\
# Tripletex Competition — Task Patterns & Scoring Guide

## GENERAL RULES
1. Every entity mentioned in the prompt must be CREATED as a separate record.
   - Product with name/number → POST /product FIRST, then reference in order lines
   - Customer with name → POST /customer with ALL mentioned fields (org.nr, address, etc.)
   - Employee with name → POST /employee (or GET if the prompt says "existing"/"find")
2. Every field value mentioned in the prompt WILL be checked field-by-field.
3. The sandbox starts FRESH — no customers, products, or employees exist (only account owner).
4. Use EXACT values from the prompt. Never modify names, emails, amounts.
5. When the prompt mentions a field, ALWAYS include it in the API call — even if optional.

## MULTILINGUAL FIELD MAPPING (prompt term → API field)

### Employee
- navn/name/nombre/nom/Name → split into firstName + lastName
- e-post/email/correo/E-Mail → email
- telefon/phone/teléfono/Telefon → phoneNumberMobile (NOT phoneNumber!)
- fødselsdato/date of birth/fecha de nacimiento → dateOfBirth
- personnummer/national ID/Personnummer → nationalIdentityNumber
- bankkonto/bank account/Bankkonto → bankAccountNumber
- kontoadministrator/admin/administrador/Kontoadministrator → entitlements ALL_PRIVILEGES
- avdeling/department/departamento → department (REQUIRED)

### Customer
- navn/name/nombre/Name → name
- e-post/email → email
- telefon/phone/teléfono → phoneNumber (customer uses phoneNumber, NOT phoneNumberMobile!)
- org.nr/organisasjonsnummer/organization number/Org.-Nr. → organizationNumber
- adresse/address/dirección/Adresse → postalAddress: {addressLine1, postalCode, city}
- privatkunde/private individual → isPrivateIndividual: true

### Product
- produktnummer/product number/número de producto/Produktnummer → number
- pris/price/precio/Preis → priceExcludingVatCurrency (default: excl. VAT)
- pris inkl. mva/price incl. VAT → priceIncludingVatCurrency
- mva-kode/VAT code → vatType: {"id": N} (default id=6 for no VAT)
- enhet/unit → productUnit: {"id": N}
- beskrivelse/description → description

### Order/Invoice
- fakturadato/invoice date/fecha de factura/Rechnungsdatum → invoiceDate
- forfallsdato/due date/fecha de vencimiento/Fälligkeitsdatum → invoiceDueDate
- ordredato/order date → orderDate
- leveringsdato/delivery date → deliveryDate (REQUIRED on order!)
- antall/quantity/cantidad/Anzahl → count (NOT quantity!)
- pris per enhet/unit price/Stückpreis → unitPriceExcludingVatCurrency
- referanse/reference → reference

### Travel Expense
- reiseregning/travel expense/nota de gastos/Reisekostenabrechnung → POST /travelExpense
- diett/per diem/dietas/Tagesgeld → per diem compensation or cost line
- utgift/expense/gasto/Ausgabe → POST /travelExpense/cost
- beløp/amount/monto/Betrag → amountCurrencyIncVat (NOT amount!)
- kategori/category/categoría → category (NOT description!)

---

## TASK: Create Employee
Trigger: "opprett ansatt", "create employee", "erstellen Mitarbeiter", "crear empleado", "créer employé"
Checks: employee found, firstName, lastName, email, phoneNumberMobile, admin role
Workflow:
1. GET /department?fields=id&count=1 → department_id
2. POST /employee with: firstName, lastName, email, phoneNumberMobile, department: {"id": N}
   - Include ALL fields mentioned in prompt (phone, dateOfBirth, address, etc.)
3. If admin/kontoadministrator: PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=N&template=ALL_PRIVILEGES
Mistakes:
- "phoneNumber" → WRONG for employee. Use "phoneNumberMobile"
- Setting userType → READ-ONLY. Use entitlements endpoint
- Forgetting department → REQUIRED field

## TASK: Create Customer
Trigger: "opprett kunde", "create customer", "erstellen Kunde", "crear cliente", "créer client"
Checks: customer found, name, email, phoneNumber, organizationNumber, isCustomer
Workflow:
1. POST /customer with: name, email, phoneNumber, organizationNumber, isCustomer: true
   - Include ALL fields mentioned: address, language, invoiceSendMethod, etc.
   - If org.nr is mentioned, ALWAYS include organizationNumber
Mistakes:
- Forgetting organizationNumber when org.nr is in the prompt
- Forgetting isCustomer: true
- Note: customer uses "phoneNumber" (not phoneNumberMobile)

## TASK: Create Product
Trigger: "opprett produkt", "create product", "erstellen Produkt", "crear producto"
Checks: product found, name, number, priceExcludingVatCurrency, description
Workflow:
1. POST /product with: name, number, priceExcludingVatCurrency
   - If price is stated as "inkl. mva" (incl. VAT), use priceIncludingVatCurrency
   - If price is stated as "eks. mva" or just "pris" (excl. VAT), use priceExcludingVatCurrency
   - Do NOT set vatType — it defaults to id=6 which works. Other vatType IDs FAIL.
   - If product number already exists ("Produktnummeret er i bruk"), GET /product?number=N to find its ID.
Mistakes:
- Omitting product number
- Using wrong price field (incl vs excl VAT)
- Setting vatType on products (only default id=6 works)

## TASK: Create Invoice (multi-step)
Trigger: "opprett faktura", "create invoice", "erstellen Rechnung", "crear factura", "créer facture", "crie uma fatura"
Checks: customer (with all fields), products (if mentioned), order, invoice dates/amounts, payment
PREREQUISITE — MUST DO FIRST:
  GET /ledger/account?number=1920&fields=id,version,bankAccountNumber
  If bankAccountNumber is empty: PUT /ledger/account/{id} with {"id": N, "version": N, "number": 1920, "name": "Bankinnskudd", "bankAccountNumber": "86011117947"}
  WITHOUT THIS, INVOICE CREATION WILL FAIL.
Workflow:
1. Check bank account (see PREREQUISITE above) — the PLANNER must do this GET
2. POST /customer with ALL mentioned fields (name, email, org.nr, address, isCustomer: true) → customer_id
3. POST /product for EACH product mentioned (name, number, price) → product_ids
   - Do NOT set vatType on products — it defaults to id=6 which works. Other vatType IDs (3, 5, 31) FAIL on products.
   - If product number already exists, GET /product?number=NNNN to find its ID instead.
4. POST /order with: customer, orderDate, deliveryDate, orderLines (with product: {"id": N}, count, unitPriceExcludingVatCurrency) → order_id
5. POST /invoice with: invoiceDate, invoiceDueDate, customer, orders: [{"id": order_id}]
6. If payment: GET /invoice/paymentType → paymentTypeId
7. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=N
Mistakes:
- FORGETTING BANK ACCOUNT CHECK — #1 cause of invoice failures
- Setting vatType on products — only id=6 works, others cause validation errors
- Using description-only order lines instead of creating products first
- Creating invoice without order (must create order first)
- Forgetting deliveryDate on order (REQUIRED)

## TASK: Create Credit Note
Trigger: "kreditnota", "credit note", "nota de crédito", "Gutschrift"
Checks: credit note exists, correct date, linked to original invoice
Workflow:
1. GET /invoice (with date filters) → find invoice_id
2. PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD
Mistakes:
- Forgetting date query param (REQUIRED)
- Using POST instead of PUT

## TASK: Create Travel Expense
Trigger: "reiseregning", "travel expense", "nota de gastos", "Reisekostenabrechnung"
Checks: employee found, travel expense title, EACH cost line (category + amount), per diem
Workflow:
1. POST /employee or GET /employee → employee_id (create if not exists)
2. GET /department → department_id
3. POST /travelExpense with: employee, title, date, department → travelExpense_id
4. GET /travelExpense/paymentType → paymentType_id
5. POST /travelExpense/cost for EACH expense with: travelExpense, date, amountCurrencyIncVat, paymentType, category, isPaidByEmployee: true
6. If per diem/dietas mentioned: can use /travelExpense/cost as a cost line with the total amount
Mistakes:
- "amount" → WRONG. Use "amountCurrencyIncVat"
- "description" → WRONG. Use "category"
- Forgetting paymentType (REQUIRED)
- Trying perDiemCompensation (complex, needs travel details) — simpler to use /cost

## TASK: Create Project
Trigger: "opprett prosjekt", "create project", "erstellen Projekt", "crear proyecto"
Checks: project found, name, startDate, customer linked, projectManager
Workflow:
1. POST /customer (if new) → customer_id
2. GET /employee → projectManager employee_id
3. POST /project with: name, startDate, customer, projectManager, description
Mistakes:
- Forgetting projectManager

## TASK: Create Department
Trigger: "opprett avdeling", "create department"
Checks: department found, name, departmentNumber
Workflow:
1. POST /department with: name, departmentNumber

## TASK: Create Contact
Trigger: "opprett kontakt", "create contact", "kontaktperson"
Checks: contact found, firstName, lastName, email, phone, linked to customer
Workflow:
1. GET /customer or POST /customer → customer_id
2. POST /contact with: firstName, lastName, email, phoneNumberMobile, customer: {"id": N}

## TASK: Delete Entity
Trigger: "slett", "delete", "eliminar", "löschen", "fjern"
Checks: entity no longer exists
Workflow:
1. GET the entity with filters → find id
2. DELETE /entity/{id}

## TASK: Update Employee
Trigger: "oppdater", "endre", "update", "actualizar", "ändern"
Checks: field has new value
Workflow:
1. GET /employee/{id}?fields=* → get version, dateOfBirth, current data
2. PUT /employee/{id} with: id, version, dateOfBirth, plus changed fields
Mistakes:
- Forgetting dateOfBirth (REQUIRED on PUT even if unchanged)
- Forgetting version field

## TASK: Register Payment
Trigger: "registrer betaling", "register payment", "registrar pago", "Zahlung registrieren"
Checks: invoice amountOutstanding = 0, payment date correct
Workflow:
1. GET /invoice (find the invoice) → invoice_id, amount
2. GET /invoice/paymentType → paymentTypeId
3. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=N
Mistakes:
- Putting payment data in body (should be query params)
- Wrong paidAmount (must match invoice amount for full payment)

## TASK: Create Order (without invoice)
Trigger: "opprett ordre", "create order"
Checks: customer exists, order with correct lines, products (if referenced)
Workflow:
1. POST /customer (if new) → customer_id
2. POST /product for each product → product_ids
3. POST /order with: customer, orderDate, deliveryDate, orderLines

## TASK: Create Accounting Dimension and Voucher
Trigger: "dimensjon", "dimension", "bilag", "voucher", "journal entry", "postering", "Buchung", "dimensão contabilística", "lançar um lançamento"
Checks: dimension created, dimension values exist, voucher posted with correct account and dimension link
Workflow:
1. GET /ledger/account?number=NNNN&fields=id,number,name — MUST get account ID first (API requires ID, not number!)
   - Also get a second account for the balancing posting (e.g., account 1920 or another expense account)
2. POST /ledger/accountingDimensionName — create the dimension
   Payload: {"dimensionName": "Region", "description": "...", "dimensionIndex": 1, "active": true}
   - dimensionIndex: 1, 2, or 3 (for free dimensions 1-3)
3. POST /ledger/accountingDimensionValue — create first value, save returned ID
   Payload: {"displayName": "Vestlandet", "dimensionIndex": 1, "active": true, "number": "VEST", "showInVoucherRegistration": true}
4. POST /ledger/accountingDimensionValue — create second value, save returned ID
5. POST /ledger/voucher?sendToLedger=true — create the journal entry
   Payload: {
     "date": "YYYY-MM-DD",
     "description": "...",
     "postings": [
       {
         "date": "YYYY-MM-DD",
         "row": 1,
         "account": {"id": <account_id_from_step_1>},
         "description": "...",
         "amountGross": 47500,
         "amountGrossCurrency": 47500,
         "freeAccountingDimension1": {"id": <dimension_value_id_from_step_4>}
       },
       {
         "date": "YYYY-MM-DD",
         "row": 2,
         "account": {"id": <balancing_account_id>},
         "amountGross": -47500,
         "amountGrossCurrency": -47500
       }
     ]
   }
CRITICAL VOUCHER RULES:
- Account MUST be {"id": N} — NEVER {"number": N}. Always GET the account ID first!
- Use amountGross AND amountGrossCurrency (both required, must be equal for NOK)
- Do NOT use "amount" — it does not work. Use amountGross.
- Include "date" and "row" on each posting
- Postings MUST balance (sum of amountGross must be zero)
- Use freeAccountingDimension1 for dimensionIndex=1, freeAccountingDimension2 for index=2, etc.
- Some accounts (1920 Bankinnskudd, 2400 Leverandørgjeld) are system-managed — use 1900 Kontanter or other general accounts for balancing

## TASK: Enable Sales Module / Accounting Module
Trigger: "aktiver modul", "enable module", "aktivere", "sales module"
Checks: module activated
Workflow:
1. GET /company/salesmodules → check current modules
2. POST /company/salesmodules with module to activate

## TASK: Create Supplier / Vendor
Trigger: "opprett leverandør", "create supplier", "crear proveedor", "Lieferant erstellen"
Checks: supplier found, name, organizationNumber, isSupplier
Workflow:
1. POST /customer with: name, organizationNumber, isSupplier: true, isCustomer: false
   - Suppliers use the same /customer endpoint but with isSupplier: true
"""
