# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `src/health.py` — dedicated health-check module (`python -m src.health`) with
  environment variable validation, dependency checks, data-directory inspection,
  and outputs-directory write-access verification. Supports `--json` flag for
  machine-readable output.
- Live **System Status** badge in the Streamlit sidebar showing OpenAI key,
  Tavily key, and document count at a glance.
- `Maintenance & Support` section in README covering versioning policy, support
  channels (GitHub Issues / Discussions), and maintenance status.

### Changed

- `src/main.py` — crew execution now catches `ValueError`, `KeyboardInterrupt`,
  and generic exceptions; prints actionable messages and exits with a non-zero
  code instead of crashing with a raw traceback.
- `demo.py` — added error handling around assistant initialisation and per-query
  execution; `EOFError`/`KeyboardInterrupt` in the interactive loop exit cleanly.
- `src/tools.py` — replaced silent `except Exception: pass` in `load_local_docs`
  with `logger.warning(...)` so skipped files are visible in the log.
- `src/crew.py` — `build_crew()` now raises `ValueError` with an explicit message
  when `OPENAI_API_KEY` or `TAVILY_API_KEY` is absent, surfacing misconfiguration
  before the crew pipeline starts.
- Streamlit sidebar **About** link updated to point to the correct GitHub URL and
  now also includes a direct link to the issue tracker.

## [1.0.0] - 2026-03-04

### Initial release

- Multi-agent RAG pipeline with CrewAI (Research, Analyst, Writer agents)
- TavilySearchTool for live web search
- LocalRAGSearchTool for semantic retrieval over local documents
- CalculatorTool for safe arithmetic evaluation
- SaveReportTool for exporting Markdown reports
- In-memory vector database with OpenAI embeddings and cosine similarity
- Interactive single-agent demo (`demo.py`)
- Sample knowledge-base documents (AI, biotech, quantum computing, etc.)
- Comprehensive documentation (README, instructions, contributing guide)
- Test suite with pytest
- Dockerfile for containerized execution
- Ruff linter configuration via pyproject.toml
