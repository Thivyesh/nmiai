"""Accounting task patterns for the Tripletex competition.

Each pattern describes:
- What the task asks for
- What fields the competition checks (verified or inferred)
- The exact API workflow to achieve full marks
- Common mistakes to avoid

The planner uses these patterns to produce correct plans.
"""

TASK_PATTERNS = """\
# Tripletex Competition — Task Patterns & Scoring Guide

## GENERAL RULES
1. Every entity mentioned in the prompt must be CREATED as a separate record, not just referenced by name.
   - If the prompt mentions a product by name and number → POST /product first
   - If the prompt mentions a customer by name → POST /customer (or GET if exists)
   - If the prompt mentions an employee → POST /employee (or GET if exists)
2. Every field value mentioned in the prompt will be checked field-by-field.
3. The competition verifies by querying the API after your agent completes.
4. Each sandbox starts FRESH — no pre-existing customers, products, or employees (except the account owner).

## TASK: Create Employee
Trigger: "opprett ansatt", "create employee", "erstellen Mitarbeiter", etc.
Checks: employee found, firstName, lastName, email, phone, admin role (if mentioned)
Workflow:
1. GET /department?fields=id&count=1 → department_id
2. POST /employee with: firstName, lastName, email, phoneNumberMobile, department
3. If admin/kontoadministrator: PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=N&template=ALL_PRIVILEGES
Common mistakes:
- Using "phoneNumber" instead of "phoneNumberMobile"
- Trying to set userType (it's read-only — use entitlements)
- Forgetting department (required)

## TASK: Create Customer
Trigger: "opprett kunde", "create customer", "créer client", etc.
Checks: customer found, name, email, phone, organizationNumber (if given), isCustomer=true
Workflow:
1. POST /customer with: name, email, phoneNumber, organizationNumber, isCustomer: true
Common mistakes:
- Forgetting organizationNumber when org.nr is in the prompt
- Forgetting isCustomer: true

## TASK: Create Product
Trigger: "opprett produkt", "create product", "erstellen Produkt", etc.
Checks: product found, name, number, priceExcludingVatCurrency
Workflow:
1. POST /product with: name, number, priceExcludingVatCurrency
Common mistakes:
- Omitting the product number

## TASK: Create Invoice (multi-step)
Trigger: "opprett faktura", "create invoice", "erstellen Rechnung", etc.
Checks: customer exists, products exist (if referenced), order exists, invoice with correct dates/amounts, payment (if requested)
Workflow:
1. GET /ledger/account?number=1920&fields=id,version,bankAccountNumber → check bank account
2. (If empty) PUT /ledger/account/{id} to set bankAccountNumber
3. POST /customer (if new) with: name, email, organizationNumber, isCustomer: true → customer_id
4. POST /product for EACH product mentioned with name and number → product_ids
5. POST /order with: customer, orderDate, deliveryDate, orderLines (with product references) → order_id
6. POST /invoice with: invoiceDate, invoiceDueDate, customer, orders → invoice_id
7. (If payment requested) GET /invoice/paymentType → paymentTypeId
8. PUT /invoice/{id}/:payment?paymentDate=X&paymentTypeId=N&paidAmount=N
Common mistakes:
- Skipping product creation and using description-only order lines
- Trying to POST invoice directly without creating an order first
- Forgetting deliveryDate on the order
- Forgetting to check/set bank account on ledger account 1920

## TASK: Create Credit Note
Trigger: "kreditnota", "credit note", "nota de crédito", etc.
Checks: credit note created, correct date, linked to original invoice
Workflow:
1. GET /invoice with filters to find the target invoice → invoice_id
2. PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD
Common mistakes:
- Forgetting the date query parameter (required)
- Using POST instead of PUT

## TASK: Create Travel Expense
Trigger: "reiseregning", "travel expense", "nota de gastos", "Reisekostenabrechnung", etc.
Checks: employee exists, travel expense found, title, individual cost lines with correct amounts
Workflow:
1. GET /employee or POST /employee → employee_id
2. GET /department → department_id
3. POST /travelExpense with: employee, title, date, department → travelExpense_id
4. GET /travelExpense/paymentType → paymentType_id (typically "Privat utlegg")
5. POST /travelExpense/cost for EACH expense line with: travelExpense, date, amountCurrencyIncVat, paymentType, category, isPaidByEmployee: true
Common mistakes:
- Using "amount" instead of "amountCurrencyIncVat"
- Using "description" instead of "category"
- Forgetting paymentType (required)
- Trying to set per diem as a single cost instead of using /travelExpense/perDiemCompensation

## TASK: Create Project
Trigger: "opprett prosjekt", "create project", "crear proyecto", etc.
Checks: project found, name, startDate, customer linked, projectManager linked
Workflow:
1. POST /customer (if new) → customer_id
2. GET /employee → find projectManager employee_id
3. POST /project with: name, startDate, customer, projectManager
Common mistakes:
- Forgetting projectManager (usually required)

## TASK: Create Department
Trigger: "opprett avdeling", "create department", etc.
Checks: department found, name, departmentNumber
Workflow:
1. POST /department with: name, departmentNumber

## TASK: Delete Travel Expense
Trigger: "slett reiseregning", "delete travel expense", etc.
Checks: travel expense no longer exists
Workflow:
1. GET /travelExpense with filters → find the expense id
2. DELETE /travelExpense/{id}

## TASK: Update Employee
Trigger: "oppdater ansatt", "update employee", "endre", etc.
Checks: employee has updated field values
Workflow:
1. GET /employee with filters → find employee, get id, version, and ALL current fields
2. PUT /employee/{id} with: id, version, dateOfBirth (required), plus updated fields
Common mistakes:
- Forgetting dateOfBirth on PUT (required even if not changing it)
- Forgetting version field
"""
