"""PDF/image data extraction using Sonnet. Runs before the main agent.

Extracts ALL structured data from attached files so the agent
doesn't miss fields like phone, address, occupation code, etc.
"""

import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Extract ALL data from this document. Return a structured list of every field and value.

Extract these categories if present:
## Personal Information
- Full name (first name, last name)
- Date of birth
- National identity number (personnummer/fødselsnummer)
- Phone number(s)
- Email address
- Home address (street, postal code, city)
- Bank account number

## Employment Information
- Job title / position
- Occupation code (stillingskode/yrkeskode)
- Department
- Start date
- End date (if any)
- Employment type (permanent/temporary)
- Employment form (full-time/part-time)
- Working hours per week
- Percentage of full-time equivalent

## Salary Information
- Annual salary
- Monthly salary
- Hourly wage (if applicable)
- Bonus/commission details

## Invoice/Financial Information
- Invoice number
- Invoice date
- Due date
- Amount (with/without VAT)
- Currency
- Product/service description
- Quantities and unit prices
- VAT rate and amount
- Customer/supplier name and org number
- Payment terms

## Other
- Any other data fields, numbers, codes, or references found

Format: List each field with its exact value. Include EVERYTHING — every field will be checked.
"""


async def extract_file_data(files: list) -> str:
    """Extract structured data from PDF/image files using Sonnet.

    Returns extracted data as structured text, or empty string if no files.
    """
    if not files:
        return ""

    # Use GPT-4.1-mini for extraction — cheap, reliable
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_retries=2,
        timeout=30,
    )

    content_parts = [{"type": "text", "text": EXTRACTION_PROMPT}]

    for f in files:
        raw = base64.b64decode(f.content_base64)

        if f.mime_type.startswith("image/"):
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{f.mime_type};base64,{f.content_base64}",
                },
            })
            content_parts.append({"type": "text", "text": f"[Image: {f.filename}]"})

        elif f.mime_type == "application/pdf":
            content_parts.append({
                "type": "file",
                "file": {
                    "filename": f.filename,
                    "file_data": f"data:application/pdf;base64,{f.content_base64}",
                },
            })
            content_parts.append({"type": "text", "text": f"[PDF: {f.filename}]"})

        else:
            try:
                text = raw.decode("utf-8")
                content_parts.append({
                    "type": "text",
                    "text": f"### {f.filename}\n```\n{text[:8000]}\n```",
                })
            except UnicodeDecodeError:
                content_parts.append({
                    "type": "text",
                    "text": f"[Binary: {f.filename}, {len(raw)} bytes]",
                })

    try:
        response = await llm.ainvoke([HumanMessage(content=content_parts)])
        extracted = response.content
        logger.info("Extracted file data:\n%s", extracted[:500])
        return f"\n## Extracted File Data\n{extracted}"
    except Exception as e:
        logger.warning("PDF extraction failed: %s", e)
        return ""
