"""Verified payload templates for Tripletex API endpoints.

Each template is the EXACT JSON structure that works.
The researcher copies the template and substitutes values.
No field name guessing needed.
"""

PAYLOAD_TEMPLATES = {
    "POST /customer": {
        "description": "Create a customer",
        "payload": {
            "name": "<NAME>",
            "email": "<EMAIL>",
            "phoneNumber": "<PHONE>",
            "organizationNumber": "<ORG_NR>",
            "isCustomer": True,
        },
        "notes": "phoneNumber for customers (NOT phoneNumberMobile). Include organizationNumber only if given in prompt.",
    },
    "POST /employee": {
        "description": "Create an employee",
        "payload": {
            "firstName": "<FIRST>",
            "lastName": "<LAST>",
            "email": "<EMAIL>",
            "phoneNumberMobile": "<PHONE>",
            "dateOfBirth": "<YYYY-MM-DD>",
            "nationalIdentityNumber": "<NATIONAL_ID>",
            "bankAccountNumber": "<BANK_ACCOUNT>",
            "department": {"id": "<DEPARTMENT_ID>"},
            "address": {
                "addressLine1": "<STREET>",
                "postalCode": "<POSTAL_CODE>",
                "city": "<CITY>",
            },
        },
        "notes": "phoneNumberMobile (NOT phoneNumber). department is REQUIRED. Include ALL fields from the prompt/PDF: dateOfBirth, nationalIdentityNumber, bankAccountNumber, address. Omit fields not mentioned. Do NOT set userType unless POST fails with 'Brukertype' error — then retry with userType: 'STANDARD'.",
    },
    "POST /employee/employment": {
        "description": "Create employment record for employee",
        "payload": {
            "employee": {"id": "<EMPLOYEE_ID>"},
            "startDate": "<YYYY-MM-DD>",
            "isMainEmployer": True,
            "employmentDetails": [
                {
                    "date": "<START_DATE>",
                    "employmentType": "<ORDINARY|MARITIME|FREELANCE|PENSION>",
                    "employmentForm": "<PERMANENT|TEMPORARY>",
                    "remunerationType": "<MONTHLY_WAGE|HOURLY_WAGE|COMMISSION_PERCENTAGE|FEE>",
                    "workingHoursScheme": "<NOT_SHIFT|ROUND_THE_CLOCK|SHIFT_365|OFFSHORE_336|CONTINUOUS|OTHER_SHIFT>",
                    "percentageOfFullTimeEquivalent": 100.0,
                    "annualSalary": "<ANNUAL_SALARY>",
                    "occupationCode": {"id": "<OCCUPATION_CODE_ID>"},
                }
            ],
        },
        "notes": "Include employmentDetails with salary and occupation info if available from PDF/prompt. For salary tasks, must include division: {'id': N}. Look up occupationCode via GET /employee/employment/occupationCode.",
    },
    "PUT /employee/entitlement/:grantEntitlementsByTemplate": {
        "description": "Grant admin role to employee",
        "payload": "NONE — use query params only",
        "url_format": "/employee/entitlement/:grantEntitlementsByTemplate?employeeId=<ID>&template=ALL_PRIVILEGES",
        "notes": "Pass body='{}'. All params in URL.",
    },
    "POST /product": {
        "description": "Create a product",
        "payload": {
            "name": "<NAME>",
            "number": "<NUMBER_FROM_PROMPT_ONLY>",
            "priceExcludingVatCurrency": "<PRICE>",
        },
        "notes": "ONLY include 'number' if prompt gives a product number. Do NOT set vatType, account, or currency.",
    },
    "POST /order": {
        "description": "Create an order with order lines",
        "payload": {
            "customer": {"id": "<CUSTOMER_ID>"},
            "orderDate": "<YYYY-MM-DD>",
            "deliveryDate": "<YYYY-MM-DD>",
            "isPrioritizeAmountsIncludingVat": False,
            "orderLines": [
                {
                    "product": {"id": "<PRODUCT_ID>"},
                    "description": "<DESCRIPTION>",
                    "count": 1,
                    "unitPriceExcludingVatCurrency": "<PRICE>",
                }
            ],
        },
        "notes": "deliveryDate REQUIRED. Use 'count' NOT 'quantity'. Include product ref if product was created.",
    },
    "POST /invoice": {
        "description": "Create invoice from order",
        "payload": {
            "invoiceDate": "<YYYY-MM-DD>",
            "invoiceDueDate": "<YYYY-MM-DD>",
            "customer": {"id": "<CUSTOMER_ID>"},
            "orders": [{"id": "<ORDER_ID>"}],
        },
        "notes": "Bank account on ledger 1920 must be set first. Text field is 'comment' NOT 'description'.",
    },
    "PUT /invoice/{id}/:payment": {
        "description": "Register payment on invoice",
        "payload": "NONE — use query params only",
        "url_format": "/invoice/<INVOICE_ID>/:payment?paymentDate=<DATE>&paymentTypeId=<PAYMENT_TYPE_ID>&paidAmount=<AMOUNT_FROM_INVOICE>",
        "notes": "ALL params in URL, body='{}'. Use the 'amount' from the invoice response, NOT a calculated value.",
    },
    "PUT /invoice/{id}/:createCreditNote": {
        "description": "Create credit note for invoice",
        "payload": "NONE — use query params only",
        "url_format": "/invoice/<INVOICE_ID>/:createCreditNote?date=<YYYY-MM-DD>",
        "notes": "date is REQUIRED. Pass body='{}'.",
    },
    "POST /travelExpense": {
        "description": "Create travel expense",
        "payload": {
            "employee": {"id": "<EMPLOYEE_ID>"},
            "title": "<TITLE>",
            "date": "<YYYY-MM-DD>",
            "department": {"id": "<DEPARTMENT_ID>"},
        },
        "notes": "Can include travelDetails: {departureDate, returnDate, departureFrom, destination, purpose}.",
    },
    "POST /travelExpense/cost": {
        "description": "Add cost line to travel expense",
        "payload": {
            "travelExpense": {"id": "<TRAVEL_EXPENSE_ID>"},
            "date": "<YYYY-MM-DD>",
            "amountCurrencyIncVat": "<AMOUNT>",
            "paymentType": {"id": "<PAYMENT_TYPE_ID>"},
            "category": "<DESCRIPTION_TEXT>",
            "isPaidByEmployee": True,
        },
        "notes": "Use 'amountCurrencyIncVat' NOT 'amount'. Use 'category' NOT 'description'. paymentType REQUIRED.",
    },
    "POST /department": {
        "description": "Create department",
        "payload": {
            "name": "<NAME>",
            "departmentNumber": "<NUMBER>",
        },
        "notes": "Simple creation, no prerequisites.",
    },
    "POST /supplier": {
        "description": "Create supplier",
        "payload": {
            "name": "<NAME>",
            "organizationNumber": "<ORG_NR>",
            "email": "<EMAIL>",
        },
        "notes": "Separate endpoint from /customer.",
    },
    "POST /project": {
        "description": "Create project (set isFixedPrice for fixed-price projects)",
        "payload": {
            "name": "<NAME>",
            "startDate": "<YYYY-MM-DD>",
            "customer": {"id": "<CUSTOMER_ID>"},
            "projectManager": {"id": "<EMPLOYEE_ID>"},
            "isFixedPrice": False,
        },
        "notes": "projectManager required — find by email or create. Set isFixedPrice: true for fixed-price projects. Omit isFixedPrice or set false for hourly projects.",
    },
    "POST /project/orderline": {
        "description": "Add order line to project (for fixed-price amount or milestone)",
        "payload": {
            "project": {"id": "<PROJECT_ID>"},
            "product": {"id": "<PRODUCT_ID>"},
            "description": "<DESCRIPTION>",
            "count": 1,
            "unitPriceExcludingVatCurrency": "<AMOUNT>",
            "date": "<YYYY-MM-DD>",
            "isChargeable": True,
        },
        "notes": "isChargeable MUST be true for invoiceable lines. date and product are also REQUIRED. Create a product first (POST /product). For milestone billing: also create an order (POST /order) and invoice it (PUT /order/{id}/:invoice).",
    },
    "POST /contact": {
        "description": "Create contact on customer",
        "payload": {
            "firstName": "<FIRST>",
            "lastName": "<LAST>",
            "email": "<EMAIL>",
            "customer": {"id": "<CUSTOMER_ID>"},
        },
        "notes": "Links contact to a customer.",
    },
    "PUT /ledger/account/{id}": {
        "description": "Set bank account number",
        "payload": {
            "id": "<ACCOUNT_ID>",
            "version": "<VERSION>",
            "bankAccountNumber": "86011117947",
        },
        "notes": "Required before creating invoices. Get id and version from pre-fetched data.",
    },
    "POST /ledger/accountingDimensionName": {
        "description": "Create accounting dimension (REQUIRED before values)",
        "payload": {
            "dimensionName": "<NAME>",
            "description": "<DESCRIPTION>",
            "dimensionIndex": 1,
            "active": True,
        },
        "notes": "MUST be created BEFORE dimension values. Not automatic.",
    },
    "POST /ledger/accountingDimensionValue": {
        "description": "Create dimension value",
        "payload": {
            "displayName": "<VALUE_NAME>",
            "dimensionIndex": 1,
            "active": True,
            "number": "<SHORT_CODE>",
            "showInVoucherRegistration": True,
        },
        "notes": "dimensionIndex must match the dimension name created above.",
    },
    "POST /ledger/voucher": {
        "description": "Create journal entry / voucher",
        "url_format": "/ledger/voucher?sendToLedger=true",
        "payload": {
            "date": "<YYYY-MM-DD>",
            "description": "<DESCRIPTION>",
            "voucherType": {"id": "<VOUCHER_TYPE_ID>"},
            "postings": [
                {
                    "date": "<YYYY-MM-DD>",
                    "row": 1,
                    "account": {"id": "<ACCOUNT_ID_LOOKED_UP>"},
                    "amountGross": "<POSITIVE_FOR_DEBIT>",
                    "amountGrossCurrency": "<SAME_AS_AMOUNTGROSS>",
                    "description": "<POSTING_DESCRIPTION>",
                },
                {
                    "date": "<YYYY-MM-DD>",
                    "row": 2,
                    "account": {"id": "<BALANCING_ACCOUNT_ID>"},
                    "amountGross": "<NEGATIVE_FOR_CREDIT>",
                    "amountGrossCurrency": "<SAME_AS_AMOUNTGROSS>",
                    "description": "<POSTING_DESCRIPTION>",
                },
            ],
        },
        "notes": "Account: ALWAYS {'id': N} from GET, NEVER use account number. Amounts: ONLY 'amountGross' and 'amountGrossCurrency'. NEVER 'debitAmount', 'creditAmount', or 'amount'. voucherType: from pre-fetched data. Postings MUST balance (sum=0).",
    },
    "POST /salary/transaction": {
        "description": "Run payroll",
        "payload": {
            "date": "<YYYY-MM-DD>",
            "year": "<CURRENT_YEAR>",
            "month": "<CURRENT_MONTH>",
            "isHistorical": False,
            "payslips": [
                {
                    "employee": {"id": "<EMPLOYEE_ID>"},
                    "date": "<YYYY-MM-DD>",
                    "year": "<CURRENT_YEAR>",
                    "month": "<CURRENT_MONTH>",
                    "specifications": [
                        {
                            "salaryType": {"id": "<SALARY_TYPE_ID>"},
                            "rate": "<AMOUNT>",
                            "count": 1,
                            "amount": "<AMOUNT>",
                        }
                    ],
                }
            ],
        },
        "notes": "Year must be current (2026). Employee must have employment with division. Get salary type IDs from GET /salary/type.",
    },
    "POST /incomingInvoice": {
        "description": "Register incoming/supplier invoice",
        "url_format": "/incomingInvoice?sendTo=ledger",
        "payload": {
            "invoiceHeader": {
                "vendorId": "<SUPPLIER_ID_FLAT_INTEGER>",
                "invoiceDate": "<YYYY-MM-DD>",
                "dueDate": "<YYYY-MM-DD>",
                "currencyId": 1,
                "invoiceAmount": "<TOTAL_INCL_VAT>",
                "description": "<DESCRIPTION>",
                "invoiceNumber": "<INVOICE_NUMBER>",
            },
            "orderLines": [
                {
                    "externalId": "line1",
                    "row": 1,
                    "description": "<LINE_DESCRIPTION>",
                    "accountId": "<ACCOUNT_ID_FLAT_INTEGER>",
                    "amountInclVat": "<AMOUNT_INCL_VAT>",
                    "vatTypeId": 1,
                    "count": 1,
                }
            ],
        },
        "notes": "CRITICAL: ALL IDs are FLAT integers (vendorId, accountId, vatTypeId) NOT nested objects. externalId is REQUIRED on each orderLine. invoiceAmount is total INCLUDING VAT. vatTypeId: 1=25% inbound, 11=15%, 12=12%.",
    },
    "POST /timesheet/entry": {
        "description": "Register hours on a project activity",
        "payload": {
            "project": {"id": "<PROJECT_ID>"},
            "activity": {"id": "<ACTIVITY_ID>"},
            "employee": {"id": "<EMPLOYEE_ID>"},
            "date": "<YYYY-MM-DD>",
            "hours": "<HOURS>",
        },
        "notes": "Activity ID comes from GET /project/projectActivity?projectId=N or from creating one via POST /project/projectActivity. Date is required.",
    },
    "POST /project/projectActivity": {
        "description": "Add activity to a project",
        "payload": {
            "project": {"id": "<PROJECT_ID>"},
            "activity": {
                "name": "<ACTIVITY_NAME>",
                "activityType": "PROJECT_SPECIFIC_ACTIVITY",
            },
        },
        "notes": "Creates an activity on a project. Use GET /project/projectActivity?projectId=N to check if activity already exists.",
    },
    "PUT /order/{id}/:invoice": {
        "description": "Create invoice from order (use for project milestone billing too)",
        "payload": "NONE — use query params only",
        "url_format": "/order/<ORDER_ID>/:invoice?invoiceDate=<DATE>&sendToCustomer=false",
        "notes": "Invoices an order. For project milestones: create an order with the milestone amount first, then invoice it. There is NO /project/:invoice endpoint. Pass body='{}'.",
    },
    "POST /project/hourlyRates": {
        "description": "Set hourly rates for a project",
        "payload": {
            "project": {"id": "<PROJECT_ID>"},
            "startDate": "<YYYY-MM-DD>",
            "hourlyRateModel": "TYPE_PROJECT_SPECIFIC_HOURLY_RATES",
            "projectSpecificRates": [
                {
                    "hourlyRate": "<RATE>",
                    "employee": {"id": "<EMPLOYEE_ID>"},
                    "activity": {"id": "<ACTIVITY_ID>"},
                }
            ],
        },
        "notes": "Set before invoicing project. hourlyRateModel can be TYPE_PROJECT_SPECIFIC_HOURLY_RATES or TYPE_FIXED_HOURLY_RATE.",
    },
}
