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

    try:
        crew = build_crew(data_dir="data")
        result = crew.kickoff(inputs={"question": question})
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"\nConfiguration error: {exc}", file=sys.stderr)
        print("Check that OPENAI_API_KEY and TAVILY_API_KEY are set in your .env file.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.exception("Crew execution failed: %s", exc)
        print(f"\nError: {exc}", file=sys.stderr)
        print("Run `python -m src.health` to diagnose the issue.", file=sys.stderr)
        sys.exit(1)

    print("\n\n===== FINAL RESULT =====\n")
    print(result)

    logger.info("Crew finished successfully.")


if __name__ == "__main__":
    main()