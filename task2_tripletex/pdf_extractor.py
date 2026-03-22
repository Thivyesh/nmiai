"""Robust PDF/image data extraction. Tries multiple approaches.

Extracts ALL structured data from attached files so the agent
doesn't miss fields like phone, address, occupation code, etc.
"""

import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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
- Working hours per week
- Percentage of full-time equivalent

## Salary Information
- Annual salary
- Monthly salary
- Hourly wage (if applicable)

## Invoice/Financial Information
- Invoice number, dates, amounts, VAT, products, quantities

## Other
- Any other data fields, numbers, codes, or references

Format: List each field with its exact value. Include EVERYTHING.
"""


async def extract_file_data(files: list) -> str:
    """Extract structured data from PDF/image files.

    Tries multiple approaches in order:
    1. Google Gemini (supports PDF natively)
    2. OpenAI GPT-4o (supports images, can read text files)
    3. Anthropic Claude (supports PDFs and images)
    4. Raw text extraction (for text-based files)

    Returns extracted data as structured text, or empty string if all fail.
    """
    if not files:
        return ""

    # Build content for each file
    file_descriptions = []
    text_content = []

    for f in files:
        raw = base64.b64decode(f.content_base64)
        file_descriptions.append(f"File: {f.filename} ({f.mime_type}, {len(raw)} bytes)")

        # Always try to extract text directly
        if not f.mime_type.startswith("image/") and f.mime_type != "application/pdf":
            try:
                text = raw.decode("utf-8")
                text_content.append(f"### {f.filename}\n{text[:10000]}")
            except UnicodeDecodeError:
                pass

    # If we have text content, we can use any LLM
    if text_content:
        text_result = await _extract_from_text("\n\n".join(text_content))
        if text_result:
            return f"\n## Extracted File Data\n{text_result}"

    # For PDFs and images, try each provider
    for method_name, method in [
        ("Gemini", _extract_with_gemini),
        ("OpenAI", _extract_with_openai),
        ("Anthropic", _extract_with_anthropic),
    ]:
        try:
            result = await method(files)
            if result:
                logger.info("PDF extraction succeeded with %s", method_name)
                return f"\n## Extracted File Data\n{result}"
        except Exception as e:
            logger.warning("PDF extraction with %s failed: %s", method_name, str(e)[:100])
            continue

    logger.warning("All PDF extraction methods failed for: %s", ", ".join(f.filename for f in files))
    return ""


async def _extract_from_text(text: str) -> str:
    """Extract data from text content using any available LLM."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, timeout=30)
        response = await llm.ainvoke([HumanMessage(content=f"{EXTRACTION_PROMPT}\n\n{text}")])
        return response.content
    except Exception:
        pass

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
        response = await llm.ainvoke([HumanMessage(content=f"{EXTRACTION_PROMPT}\n\n{text}")])
        return response.content
    except Exception:
        pass

    return ""


async def _extract_with_gemini(files: list) -> str:
    """Extract using Google Gemini (native PDF support)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    if not os.getenv("GOOGLE_API_KEY"):
        return ""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=1, timeout=30)

    content_parts = [{"type": "text", "text": EXTRACTION_PROMPT}]
    for f in files:
        if f.mime_type.startswith("image/"):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{f.mime_type};base64,{f.content_base64}"},
            })
        elif f.mime_type == "application/pdf":
            # Try multiple Gemini PDF formats
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:application/pdf;base64,{f.content_base64}"},
            })
        content_parts.append({"type": "text", "text": f"[File: {f.filename}]"})

    response = await llm.ainvoke([HumanMessage(content=content_parts)])
    return response.content if response.content else ""


async def _extract_with_openai(files: list) -> str:
    """Extract using OpenAI GPT-4o (image support, no native PDF)."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    if not os.getenv("OPENAI_API_KEY"):
        return ""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=1, timeout=30)

    content_parts = [{"type": "text", "text": EXTRACTION_PROMPT}]
    for f in files:
        if f.mime_type.startswith("image/"):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{f.mime_type};base64,{f.content_base64}"},
            })
        elif f.mime_type == "application/pdf":
            # OpenAI doesn't support PDF natively — try as data URL
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:application/pdf;base64,{f.content_base64}"},
            })
        else:
            try:
                raw = base64.b64decode(f.content_base64)
                text = raw.decode("utf-8")
                content_parts.append({"type": "text", "text": f"### {f.filename}\n{text[:8000]}"})
            except (UnicodeDecodeError, Exception):
                pass
        content_parts.append({"type": "text", "text": f"[File: {f.filename}]"})

    response = await llm.ainvoke([HumanMessage(content=content_parts)])
    return response.content if response.content else ""


async def _extract_with_anthropic(files: list) -> str:
    """Extract using Anthropic Claude (native PDF + image support)."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    if not os.getenv("ANTHROPIC_API_KEY"):
        return ""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0, max_retries=1, timeout=30)

    content_parts = [{"type": "text", "text": EXTRACTION_PROMPT}]
    for f in files:
        raw = base64.b64decode(f.content_base64)
        if f.mime_type.startswith("image/"):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{f.mime_type};base64,{f.content_base64}"},
            })
        elif f.mime_type == "application/pdf":
            content_parts.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": f.content_base64,
                },
            })
        else:
            try:
                text = raw.decode("utf-8")
                content_parts.append({"type": "text", "text": f"### {f.filename}\n{text[:8000]}"})
            except UnicodeDecodeError:
                pass
        content_parts.append({"type": "text", "text": f"[File: {f.filename}]"})

    response = await llm.ainvoke([HumanMessage(content=content_parts)])
    return response.content if response.content else ""
