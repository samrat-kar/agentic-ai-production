# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""Input validation, sanitization, and output safety guardrails.

Provides:
- ``validate_question`` — validates and sanitizes user research questions.
- ``sanitize_filename`` — prevents path-traversal in report filenames.
- ``filter_output`` — truncates and cleans LLM outputs.
- ``InputValidationError`` — raised when input fails any safety check.

All validation events are logged for compliance and debugging.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUESTION_LENGTH = 2000
MIN_QUESTION_LENGTH = 3
MAX_OUTPUT_LENGTH = 50_000

# Prompt-injection fingerprints: patterns that attempt to override system
# instructions.  These are not an exhaustive list but cover the most common
# adversarial phrasings seen in the wild.
_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|context)",
    r"disregard\s+(\w+\s+)?(previous|all|above|prior)\s+(instructions?|prompts?|context)",
    r"forget\s+(everything|all|previous|prior)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|evil|jailbroken|unrestricted)",
    r"act\s+as\s+if\s+you\s+(have\s+no|are\s+not\s+bound)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"<\s*system\s*>",
    r"\[SYSTEM\]",
    r"###\s*system",
    r"<!--\s*system",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class InputValidationError(ValueError):
    """Raised when user input fails a safety or format check."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_question(question: str) -> str:
    """Validate and sanitize a user research question.

    Checks:
    - Type is ``str``
    - Length within ``[MIN_QUESTION_LENGTH, MAX_QUESTION_LENGTH]``
    - No prompt-injection patterns
    - Strips null bytes and non-printable control characters

    Args:
        question: Raw user-supplied research question.

    Returns:
        Sanitized question string.

    Raises:
        InputValidationError: If any check fails.
    """
    if not isinstance(question, str):
        raise InputValidationError("Question must be a string.")

    question = question.strip()

    if len(question) < MIN_QUESTION_LENGTH:
        raise InputValidationError(
            f"Question is too short — minimum {MIN_QUESTION_LENGTH} characters required."
        )

    if len(question) > MAX_QUESTION_LENGTH:
        raise InputValidationError(
            f"Question is too long ({len(question)} chars). "
            f"Maximum allowed is {MAX_QUESTION_LENGTH} characters."
        )

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(question):
            logger.warning(
                "Potential prompt-injection detected in question (first 100 chars): %.100s",
                question,
            )
            raise InputValidationError(
                "Question contains disallowed patterns. "
                "Please rephrase as a straightforward research question."
            )

    # Strip null bytes and non-printable control chars (preserve newline/tab)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", question)

    logger.info("Input validated successfully (length=%d).", len(sanitized))
    return sanitized


def sanitize_filename(filename: str) -> str:
    """Sanitize a report filename to prevent path-traversal attacks.

    Args:
        filename: Proposed output filename (e.g. ``report.md``).

    Returns:
        A safe filename string.
    """
    # Remove traversal sequences and path separators
    filename = filename.replace("..", "").replace("/", "_").replace("\\", "_")
    # Remove shell-special and filesystem-special characters
    filename = re.sub(r"[<>:|?*\x00-\x1f]", "_", filename)
    filename = filename.strip()
    if not filename:
        filename = "report.md"
    return filename


def filter_output(text: str, max_length: int = MAX_OUTPUT_LENGTH) -> str:
    """Truncate and clean LLM output for display and storage.

    Args:
        text: Raw LLM-generated text.
        max_length: Hard character cap before truncation.

    Returns:
        Cleaned, possibly truncated string.
    """
    if not text:
        return ""

    if len(text) > max_length:
        logger.warning(
            "LLM output truncated from %d to %d characters.", len(text), max_length
        )
        text = text[:max_length] + "\n\n*[Output truncated — exceeded safety length limit]*"

    return text
