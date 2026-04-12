# Building a Production-Ready Multi-Agent RAG Assistant with CrewAI, Safety Guardrails, and Streamlit

A production-grade multi-agent research assistant that combines live web search with local document retrieval (RAG) to produce grounded, source-cited answers. Three specialised AI agents — Research Agent, Analyst Agent, and Writer Agent — collaborate in a sequential pipeline orchestrated by **CrewAI**. A **Streamlit web UI**, **safety guardrails**, **resilience utilities**, and a **63-test suite** make the system production-ready beyond a basic demo.

## Links

- Publication: [Building a Production-Ready Multi-Agent RAG Assistant with CrewAI, Safety Guardrails, and Streamlit](https://app.readytensor.ai/publications/building-a-production-ready-multi-agent-rag-assistant-with-crewai-safety-guardrails-and-streamlit-M8RsBzlrftZW)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Web UI (Streamlit)](#web-ui-streamlit)
- [Safety & Security](#safety--security)
- [Resilience & Monitoring](#resilience--monitoring)
- [Interface Specification](#interface-specification)
- [Code Examples](#code-examples)
- [Testing](#testing)
- [Deployment Guide](#deployment-guide)
- [Logging & Health Checks](#logging--health-checks)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Changelog](#changelog)
- [Contact](#contact)

---

## Project Overview

Plain chatbots often **hallucinate** or ignore your internal documents. This project reduces that risk by:

1. **Gathering evidence** from the web and your local files
2. **Validating and reconciling** findings (including calculations)
3. **Producing a structured report** you can submit/share

**Key features:**
- Grounded answers that cite **both** web sources (URLs) and local sources (your files)
- A clear multi-agent workflow (Research Agent → Analyst Agent → Writer Agent) orchestrated by **CrewAI**
- **Streamlit web UI** with Quick Mode and Full Research modes
- **Safety guardrails** — input validation, prompt-injection detection, output filtering
- **Resilience utilities** — exponential-backoff retry, timeout handling, iteration caps
- **63 automated tests** across unit, integration, and end-to-end levels
- Four integrated tools: web search, local semantic retrieval, safe math, and report writing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces                               │
│  streamlit_app.py (Web UI)    src/main.py (CLI)   demo.py (CLI) │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  src/safety.py            │  Input validation &
          │  validate_question()      │  prompt-injection detection
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  src/resilience.py        │  Retry + timeout +
          │  with_retry /             │  iteration caps
          │  run_with_timeout         │
          └─────────────┬─────────────┘
                        │
         ┌──────────────▼──────────────┐
         │        Agent Layer          │
         │  ┌───────────────────────┐  │
         │  │  src/crew.py          │  │  CrewAI sequential
         │  │  Research → Analyst   │  │  pipeline (3 agents)
         │  │  → Writer             │  │
         │  └───────────────────────┘  │
         │  ┌───────────────────────┐  │
         │  │  src/app.py           │  │  Single-agent RAGAssistant
         │  │  RAGAssistant         │  │  (Quick Mode / demo)
         │  └───────────────────────┘  │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │        Tool Layer           │
         │  LocalRAGSearchTool         │  Semantic search (./data)
         │  TavilySearchTool           │  Live web search
         │  CalculatorTool             │  Safe arithmetic
         │  SaveReportTool             │  Writes ./outputs/
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  src/vectordb.py            │  In-memory cosine-similarity
         │  VectorDB                   │  store (NumPy + OpenAI embeds)
         └─────────────────────────────┘
```

---

## Methodology

### Retrieval-Augmented Generation (RAG)

The system uses **RAG** to ground LLM answers in factual evidence rather than relying solely on the model's training data:

1. **Document Ingestion** — Files in `./data` (`.txt`, `.md`, `.csv`, `.json`) are loaded, split into chunks (500 chars, 50 char overlap) using `RecursiveCharacterTextSplitter`, and embedded via OpenAI's `text-embedding-3-small` model.
2. **Semantic Retrieval** — At query time, the user's question is embedded and compared against stored chunks using cosine similarity. The top-k most relevant chunks are returned as context.
3. **Augmented Generation** — The retrieved context is passed to the LLM along with the question, producing an answer grounded in actual documents.

### Multi-Agent Pipeline

The system uses CrewAI's sequential process to coordinate three agents:

```
User Question
     │
     ▼
┌─────────────────┐    TavilySearchTool
│ Research Agent   │◄── LocalRAGSearchTool
│ (evidence)       │
└────────┬────────┘
         │ research notes + sources
         ▼
┌─────────────────┐    CalculatorTool
│ Analyst Agent    │◄── LocalRAGSearchTool
│ (validation)     │
└────────┬────────┘
         │ analysis summary + outline
         ▼
┌─────────────────┐    SaveReportTool
│ Writer Agent     │──► ./outputs/report.md
│ (final report)   │
└─────────────────┘
```

**Coordination:** Each task's output is explicitly passed as context to the next task via CrewAI's `context` parameter, ensuring a deterministic Research → Analyze → Write flow.

### Tools

| Tool | Purpose | Used By |
|------|---------|---------|
| `TavilySearchTool` | Live web search for up-to-date information | Research Agent |
| `LocalRAGSearchTool` | Semantic search over local `./data` files using embeddings | Research Agent, Analyst Agent |
| `CalculatorTool` | Safe evaluation of arithmetic expressions (sandboxed `eval`) | Analyst Agent |
| `SaveReportTool` | Writes the final Markdown report to `./outputs/` | Writer Agent |

---

## Prerequisites

- **Python 3.12** ([download](https://www.python.org/downloads/))
- **OpenAI API key** — for GPT-4o-mini and text embeddings
- **Tavily API key** — for web search, free tier at [tavily.com](https://tavily.com)
- **Internet connection** for API calls
- **No GPU required** — all computation uses cloud APIs

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/samrat-kar/agentic-ai-production.git
cd agentic-ai-production
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Install dev dependencies (for testing & linting)

```bash
pip install -r requirements-dev.txt
```

---

## Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env   # then edit with your keys
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key for GPT-4o-mini and embeddings |
| `TAVILY_API_KEY` | **Yes** (Full Research) | — | Tavily API key for web search |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `CHROMA_COLLECTION_NAME` | No | `rag_documents` | Name for the in-memory vector collection |

### Knowledge Base

Place your documents in the `./data` directory. Supported formats: `.txt`, `.md`, `.csv`, `.json`.

The repository ships with sample files on AI, quantum computing, biotechnology, climate science, space exploration, and sustainable energy.

---

## Usage

### Web UI (recommended)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Choose **Quick Mode** or **Full Research** from the sidebar.

### Multi-Agent Crew (CLI)

```bash
python -m src.main "What are the latest advances in quantum computing?"
```

Output: verbose agent logs + `./outputs/report.md`.

### Demo / Interactive Mode

```bash
python demo.py
```

Runs three preset queries then enters an interactive CLI loop.

---

## Web UI (Streamlit)

![Home screen](https://raw.githubusercontent.com/samrat-kar/agentic-ai-production/main/screenshots/ui_home.png)

*Home screen — sidebar configuration and research question input*

![Results screen](https://raw.githubusercontent.com/samrat-kar/agentic-ai-production/main/screenshots/ui_results.png)

*Results screen — grounded answer with cited sources and elapsed time*

![History panel](https://raw.githubusercontent.com/samrat-kar/agentic-ai-production/main/screenshots/ui_history.png)

*Query history — timestamps, mode, and response times for all session queries*

### Features

| Feature | Description |
|---------|-------------|
| **Quick Mode** | Single RAGAssistant agent, local corpus only — ~5–10 seconds |
| **Full Research** | 3-agent CrewAI pipeline with live Tavily web search — ~30–60 seconds |
| **Sidebar config** | API keys, mode, timeout (30–300s), RAG top-k, data directory |
| **Input safety** | Questions validated and sanitised before reaching agents |
| **Progress spinner** | Real-time status while research runs |
| **Formatted output** | Answers rendered as Markdown; raw JSON via expander |
| **Query history** | Collapsible panel with timestamps and elapsed time |
| **Error messages** | Clear, actionable messages for every failure mode |

### Resetting the assistant cache

The RAGAssistant is cached in `st.session_state`. To reload documents after adding new files to `./data`, refresh the browser page.

---

## Safety & Security

All user inputs pass through `src/safety.py` before reaching any agent or tool.

### Input Validation (`validate_question`)

| Check | Detail |
|-------|--------|
| Type check | Must be a `str` |
| Length | Min 3 / max 2,000 characters |
| Prompt-injection detection | 10 regex patterns block override phrases (`"ignore previous instructions"`, `"DAN mode"`, `<system>` tags, etc.) |
| Control-character stripping | Null bytes and non-printable characters removed; newlines preserved |

### Filename Sanitisation (`sanitize_filename`)

- Removes `..` path-traversal sequences
- Replaces `/`, `\`, and shell-special characters with `_`
- Falls back to `report.md` if result is empty

### Output Filtering (`filter_output`)

- Truncates LLM responses exceeding 50,000 characters
- Appends a visible truncation notice — no silent data loss

### Calculator Sandboxing (`CalculatorTool`)

- Character allowlist: `0-9 + - * / ( ) . %` — anything else is rejected before `eval`
- `eval()` runs with `{"__builtins__": {}}` — no imports, no functions, no variables

---

## Resilience & Monitoring

`src/resilience.py` provides three independent building blocks.

### `with_retry` — Exponential Backoff Decorator

```python
from src.resilience import with_retry

@with_retry(max_retries=3, initial_wait=1.0, max_wait=30.0, backoff_factor=2.0,
            exceptions=(ConnectionError,))
def call_api():
    ...
```

- Wait schedule: 1s → 2s → 4s (capped at 30s)
- Each retry logged at `WARNING`; final failure at `ERROR`

### `run_with_timeout` — Wall-Clock Timeout

```python
from src.resilience import run_with_timeout

result = run_with_timeout(crew.kickoff, timeout_seconds=120, inputs={"question": q})
```

- Thread-based (`ThreadPoolExecutor`) — works on Windows and Unix
- Timeout configurable via the Streamlit sidebar (30–300 seconds)

### `IterationLimiter` — Loop Cap

```python
from src.resilience import IterationLimiter

limiter = IterationLimiter(max_iterations=50, label="agent_loop")
while condition:
    limiter.tick()   # raises RuntimeError after 50 iterations
```

- Prevents silent infinite loops in agent or tool cycles
- `.reset()` and `.remaining` for monitoring

---

## Interface Specification

### CLI (`src/main.py`)

```
python -m src.main [QUESTION]
```

| Input | Type | Description |
|-------|------|-------------|
| `QUESTION` | Positional string (optional) | Defaults to `"Explain RAG and why chunk overlap helps."` |

**Output:** verbose logs to `stdout`, report saved to `./outputs/report.md`, exit code `0`.

### RAGAssistant API (`src/app.py`)

```python
from src.app import RAGAssistant

assistant = RAGAssistant()
assistant.load_and_ingest("./data")
result = assistant.query_with_agent("Your question", n_results=3)
```

**Return schema:**

```json
{
  "question":       "string",
  "answer":         "string",
  "context_chunks": ["string"],
  "sources":        ["filename"],
  "mode":           "agent_tool_calling"
}
```

### `build_crew` API (`src/crew.py`)

```python
crew = build_crew(data_dir="data")
result = crew.kickoff(inputs={"question": "..."})
```

Input: `{"question": str}` — injected into all task prompts.
Output: `CrewOutput` (str-able) + saved `./outputs/report.md`.

### Safety API (`src/safety.py`)

| Function | Returns | Raises |
|----------|---------|--------|
| `validate_question(question)` | Sanitised `str` | `InputValidationError` |
| `sanitize_filename(filename)` | Safe `str` | — |
| `filter_output(text, max_length)` | Truncated `str` | — |

### Resilience API (`src/resilience.py`)

| Interface | Usage |
|-----------|-------|
| `@with_retry(max_retries, initial_wait, max_wait, backoff_factor, exceptions)` | Decorator |
| `run_with_timeout(func, timeout_seconds, *args, **kwargs)` | Returns result or raises `TimeoutError` |
| `IterationLimiter(max_iterations, label)` | `.tick()` / `.reset()` / `.remaining` / `.count` |

---

## Code Examples

### Run a research query from the command line

```bash
python -m src.main "Compare solar and wind energy efficiency"
```

### Use the RAGAssistant programmatically

```python
from src.app import RAGAssistant

assistant = RAGAssistant()
assistant.load_and_ingest("./data")
result = assistant.query_with_agent("What is feature engineering?")
print(result["answer"])
print(result["sources"])   # e.g. ['sample_documents.txt']
```

### Use safety and resilience utilities

```python
from src.safety import validate_question, InputValidationError
from src.resilience import with_retry, run_with_timeout

# Validate before passing to agents
try:
    question = validate_question(user_input)
except InputValidationError as e:
    print(f"Invalid input: {e}")

# Wrap a flaky API call with retry
@with_retry(max_retries=3, initial_wait=1.0, exceptions=(ConnectionError,))
def search_web(query):
    ...

# Run with timeout
result = run_with_timeout(assistant.query_with_agent, 60, question)
```

---

## Testing

### Run the full test suite

```bash
pytest tests/ -v
```

### Run with coverage report

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Test files — 63 tests across 7 files

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_vectordb.py` | 5 | VectorDB chunking, add, search, empty DB |
| `tests/test_tools.py` | 7 | CalculatorTool, SaveReportTool, LocalRAGSearchTool |
| `tests/test_main.py` | 2 | CLI argument parsing, default question |
| `tests/test_safety.py` | 22 | Input validation, injection detection, output filtering |
| `tests/test_resilience.py` | 18 | Retry, timeout, iteration limiter |
| `tests/test_app.py` | 11 | RAGAssistant init, load, ingest, query |
| `tests/test_integration.py` | 17 | Cross-module pipelines, end-to-end flows |

All external API calls are mocked — **no real API keys needed to run tests**.

---

## Deployment Guide

### Local development

```bash
# Clone
git clone https://github.com/samrat-kar/agentic-ai-production.git
cd agentic-ai-production

# Virtual environment
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1      # Windows
source .venv312/bin/activate          # macOS/Linux

# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure
cp .env.example .env
# Add OPENAI_API_KEY and TAVILY_API_KEY to .env

# Run web UI
streamlit run streamlit_app.py

# Run CLI
python -m src.main "Your research question"
```

### Docker

```bash
# Build
docker build -t rag-assistant .

# Run CLI
docker run --env-file .env rag-assistant "What is quantum entanglement?"

# Run web UI
docker run --env-file .env -p 8501:8501 rag-assistant \
  streamlit run streamlit_app.py --server.address 0.0.0.0
```

---

## Logging & Health Checks

### Log output

| Destination | Content |
|-------------|---------|
| `app.log` | Streamlit UI logs (file handler) |
| `stdout` | All runs (console handler) |

Format: `YYYY-MM-DD HH:MM:SS [LEVEL] module: message`

### Key logged events

| Event | Level | Module |
|-------|-------|--------|
| Input validated | INFO | `src.safety` |
| Injection detected | WARNING | `src.safety` |
| Output truncated | WARNING | `src.safety` |
| Retry attempt | WARNING | `src.resilience` |
| Final failure after retries | ERROR | `src.resilience` |
| Timeout exceeded | ERROR | `src.resilience` |
| Iteration limit exceeded | ERROR | `src.resilience` |
| Documents loaded | INFO | `src.app` / `src.tools` |
| Research complete | INFO | `streamlit_app` |

### Health check

```bash
python -c "
from src.tools import load_local_docs
docs = load_local_docs('data')
print(f'OK — loaded {len(docs)} documents from ./data')
"
```

Expected: `OK — loaded 7 documents from ./data`

---

## Troubleshooting & FAQ

### 1. `OPENAI_API_KEY is required`
**Fix:** `cp .env.example .env` then add `OPENAI_API_KEY=sk-...`

### 2. Tavily / web search fails
**Fix:** Add `TAVILY_API_KEY` to `.env` or switch to **Quick Mode** in the sidebar.

### 3. Streamlit shows "An unexpected error occurred"
**Fix:** Check `app.log` for the full traceback. Verify API keys and internet connection. If timeout, increase the slider.

### 4. `No relevant local context found` on every answer
**Fix:** Confirm `./data` has `.txt` / `.md` files:
```bash
python -c "from src.tools import load_local_docs; print(load_local_docs('data'))"
```

### 5. Tests fail with API key errors
Tests mock all external calls — no key needed. Run from the project root with the venv active:
```bash
pytest tests/ -v
```

### 6. `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install -r requirements.txt
```

### 7. Docker container exits immediately
```bash
docker run --env-file .env rag-assistant "Test question"
```

### 8. Reports not saved
`./outputs/` is created automatically. For Docker, mount a volume:
```bash
docker run --env-file .env -v $(pwd)/outputs:/app/outputs rag-assistant "Question"
```

---

## Repository Structure

```
agentic-ai-production/
├── .env.example              # Environment variable template
├── .gitignore
├── LICENSE                   # CC BY-NC-SA 4.0
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── Dockerfile
├── pyproject.toml            # Python config (linting, version)
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Dev/test dependencies
├── streamlit_app.py          # Streamlit web UI
├── demo.py                   # Interactive single-agent CLI demo
├── data/                     # Local knowledge base
│   ├── artificial_intelligence.txt
│   ├── biotechnology.txt
│   ├── climate_science.txt
│   ├── quantum_computing.txt
│   ├── sample_documents.txt
│   ├── space_exploration.txt
│   └── sustainable_energy.txt
├── screenshots/              # Streamlit UI screenshots
│   ├── ui_home.png
│   ├── ui_results.png
│   └── ui_history.png
├── outputs/                  # Generated reports (git-ignored)
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI entry point
│   ├── crew.py               # CrewAI agents, tasks, pipeline
│   ├── tools.py              # Custom tools (RAG, calculator, report saver)
│   ├── app.py                # RAGAssistant (single-agent, Quick Mode)
│   ├── vectordb.py           # In-memory vector store
│   ├── safety.py             # Input validation, output filtering
│   └── resilience.py         # Retry, timeout, iteration limiter
└── tests/
    ├── __init__.py
    ├── test_vectordb.py      # VectorDB unit tests
    ├── test_tools.py         # Tool unit tests
    ├── test_main.py          # CLI tests
    ├── test_safety.py        # Safety module tests (22 tests)
    ├── test_resilience.py    # Resilience module tests (18 tests)
    ├── test_app.py           # RAGAssistant tests (11 tests)
    └── test_integration.py   # Integration & E2E tests (17 tests)
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Licensed under [CC BY-NC-SA 4.0](LICENSE) — free for non-commercial use with attribution.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## Contact

**Maintainer:** Samrat Kar
**GitHub:** [samrat-kar77](https://github.com/samratkar77)
