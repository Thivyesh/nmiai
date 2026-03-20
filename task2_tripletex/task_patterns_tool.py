"""Task pattern lookup tool for the planner agent.

Provides accounting workflow patterns and scoring guidance so the planner
knows exactly what to create and what fields will be checked.
"""

import logging
import os
from pathlib import Path

import numpy as np
import requests
from langchain_core.tools import tool

from task2_tripletex.task_patterns import TASK_PATTERNS

logger = logging.getLogger(__name__)

# Pre-built index for semantic search over patterns
_pattern_index: dict | None = None
INDEX_PATH = Path(__file__).parent / "pattern_index.npz"


def _build_pattern_sections() -> list[tuple[str, str]]:
    """Split TASK_PATTERNS into (title, content) sections."""
    sections = []
    current_title = ""
    current_lines = []

    for line in TASK_PATTERNS.split("\n"):
        if line.startswith("## TASK:") or line.startswith("## GENERAL"):
            if current_title and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line.replace("## ", "").strip()
            current_lines = [line]
        elif current_title:
            current_lines.append(line)

    if current_title and current_lines:
        sections.append((current_title, "\n".join(current_lines)))

    return sections


@tool
def lookup_task_pattern(task_description: str) -> str:
    """Look up the correct accounting workflow pattern for a task.

    Use this FIRST before planning, to understand:
    - What entities need to be created
    - What fields the competition checks
    - The exact API workflow steps
    - Common mistakes to avoid

    Args:
        task_description: Brief description of the task type, e.g.
            "create employee", "create invoice with payment",
            "travel expense", "credit note", "create product".

    Returns:
        Detailed workflow pattern with scoring guidance.
    """
    sections = _build_pattern_sections()
    search_lower = task_description.lower()

    # Keyword matching against section titles and content
    scored = []
    for title, content in sections:
        title_lower = title.lower()
        content_lower = content.lower()

        score = 0
        # Check each search word
        for word in search_lower.split():
            if len(word) < 3:
                continue
            if word in title_lower:
                score += 3
            if word in content_lower:
                score += 1

        # Check for trigger phrases in content
        if "trigger:" in content_lower:
            trigger_line = [l for l in content_lower.split("\n") if "trigger:" in l]
            if trigger_line:
                triggers = trigger_line[0]
                for word in search_lower.split():
                    if len(word) >= 3 and word in triggers:
                        score += 5

        if score > 0:
            scored.append((score, title, content))

    scored.sort(reverse=True)

    if not scored:
        # Return general rules if no specific match
        general = [c for t, c in sections if "GENERAL" in t]
        return general[0] if general else "No matching task pattern found."

    # Return top matches (usually 1-2 relevant patterns)
    results = []
    for score, title, content in scored[:3]:
        if score >= 2:
            results.append(content)

    # Always prepend general rules
    general = [c for t, c in sections if "GENERAL" in t]
    if general:
        results.insert(0, general[0])

    return "\n\n".join(results)
