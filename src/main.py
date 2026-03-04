# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""CLI entry point for the Multi-Agent RAG Research Assistant.

Parses a research question from command-line arguments and runs the
CrewAI sequential pipeline (Research → Analysis → Writing).  The final
report is saved to ``./outputs/report.md``.

Usage::

    python -m src.main "Your research question here"
"""

from __future__ import annotations

import logging
import sys

from .crew import build_crew

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI args and kick off the multi-agent crew.

    If no question is supplied via ``sys.argv``, a sensible default is used.
    The crew result is printed to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    question: str = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "Explain RAG and why chunk overlap helps."

    logger.info("Starting crew with question: %s", question)

    crew = build_crew(data_dir="data")
    result = crew.kickoff(inputs={"question": question})

    print("\n\n===== FINAL RESULT =====\n")
    print(result)

    logger.info("Crew finished successfully.")


if __name__ == "__main__":
    main()