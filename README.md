# Multi-Agent RAG Research Assistant (CrewAI)

A multi-agent research assistant that combines live web search with local document retrieval (RAG) to produce grounded, source-cited answers. Three specialized AI agents — Research Agent, Analyst Agent, and Writer Agent — collaborate in a sequential pipeline orchestrated by CrewAI, outputting a structured Markdown report.

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
- Four integrated tools: web search, local semantic retrieval, safe math, and report writing
- Also includes a simpler single-agent demo (`demo.py`) with interactive CLI mode

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
          │  with_retry / run_with_timeout │  iteration caps
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

Before setting up, ensure you have:

- **Python 3.12** installed ([download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (for cloning the repository)
- **OpenAI API key** — required for the LLM (GPT-4o-mini) and text embeddings
- **Tavily API key** — required for web search ([get one free](https://tavily.com))
- **Operating System:** Windows, macOS, or Linux
- **Internet connection** for API calls (web search + embeddings)
- **No GPU required** — all computation uses cloud APIs

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/samrat-kar/rag-research-assistant.git
cd rag-research-assistant
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

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env   # then edit with your keys
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key for GPT-4o-mini and embeddings |
| `TAVILY_API_KEY` | **Yes** | — | Tavily API key for web search (Research Agent) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI chat model to use |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model for the vector store |
| `CHROMA_COLLECTION_NAME` | No | `rag_documents` | Name for the in-memory vector collection |

### Knowledge Base

Place your documents in the `./data` directory. Supported formats: `.txt`, `.md`, `.csv`, `.json`.

The repository ships with sample files on AI, quantum computing, biotechnology, climate science, space exploration, and sustainable energy.

---

## Usage

### Multi-Agent Crew (main entry point)

Run the full 3-agent pipeline:

```bash
python -m src.main "What are the latest advances in quantum computing?"
```

If no question is provided, it defaults to: *"Explain RAG and why chunk overlap helps."*

**Output:**
- Verbose agent logs printed to console
- Final result printed after `===== FINAL RESULT =====`
- Markdown report saved to `./outputs/report.md`

### Demo / Interactive Mode

```bash
python demo.py
```

This runs three preset queries, then enters an interactive loop (type `quit` to exit).

---

## Code Examples

### Example 1: Run a research query from the command line

```bash
python -m src.main "Compare solar and wind energy efficiency"
```

**Expected output** (abbreviated):
```
===== FINAL RESULT =====

# Solar vs Wind Energy Efficiency Report
## Short Answer
Solar panels convert 15-22% of sunlight into electricity, while wind
turbines achieve 35-45% efficiency...
## Sources
- sustainable_energy.txt
- https://...
```

The full report is saved to `./outputs/report.md`.

### Example 2: Use the RAGAssistant class programmatically

```python
from src.app import RAGAssistant

assistant = RAGAssistant()
assistant.load_and_ingest("./data")

result = assistant.query_with_agent("What is feature engineering?")
print(result["answer"])    # LLM-generated answer
print(result["sources"])   # e.g. ['sample_documents.txt']
```

### Example 3: Interactive demo session

```bash
$ python demo.py
Loaded 7 documents from ./data
...
Example queries (tool-calling agent):

Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables
   systems to learn from data without explicit programming...
Sources: artificial_intelligence.txt, sample_documents.txt

Interactive mode (type 'quit' to exit)

You: What is CRISP-DM?
Assistant: CRISP-DM is a data science methodology with six phases:
   Business Understanding, Data Understanding, Data Preparation,
   Modeling, Evaluation, and Deployment...
Sources: sample_documents.txt
```

---

## Testing

### Run the test suite

```bash
pytest tests/ -v
```

### Run with coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### What's tested

- `tests/test_vectordb.py` — Vector database chunking, add/search operations
- `tests/test_tools.py` — Calculator tool, RAG search tool, save report tool
- `tests/test_main.py` — CLI entry point and argument parsing

---

## Repository Structure

```
rag-research-assistant/
├── .env.example              # Template for environment variables
├── .gitignore                # Git ignore rules
├── LICENSE                   # CC BY-NC-SA 4.0 license
├── README.md                 # This file — project documentation
├── CONTRIBUTING.md           # Contribution guidelines
├── CHANGELOG.md              # Version history
├── CODE_OF_CONDUCT.md        # Contributor code of conduct
├── Dockerfile                # Container build for reproducible runs
├── pyproject.toml            # Python project config (linting, Python version)
├── requirements.txt          # Production dependencies (pinned)
├── requirements-dev.txt      # Dev/test dependencies
├── demo.py                   # Interactive single-agent demo CLI
├── instructions.md           # Detailed setup & usage walkthrough
├── data/                     # Local knowledge base (RAG source documents)
│   ├── artificial_intelligence.txt
│   ├── biotechnology.txt
│   ├── climate_science.txt
│   ├── quantum_computing.txt
│   ├── sample_documents.txt
│   ├── space_exploration.txt
│   └── sustainable_energy.txt
├── outputs/                  # Generated reports (git-ignored)
│   └── report.md
├── src/                      # Application source code
│   ├── __init__.py
│   ├── main.py               # CLI entry point — parses question, runs crew
│   ├── crew.py               # CrewAI agents, tasks, and sequential workflow
│   ├── tools.py              # Custom tools: RAG search, calculator, report saver
│   ├── app.py                # RAGAssistant class (used by demo.py)
│   └── vectordb.py           # In-memory vector store with cosine-similarity search
└── tests/                    # Test suite
    ├── __init__.py
    ├── test_vectordb.py      # VectorDB unit tests
    ├── test_tools.py         # Tool unit tests
    └── test_main.py          # CLI integration tests
```

| Directory | Purpose |
|-----------|---------|
| `src/` | All application source code — agents, tools, vector DB, CLI |
| `data/` | Knowledge base documents ingested into the vector store at runtime |
| `outputs/` | Auto-generated reports from the Writer Agent |
| `tests/` | Automated test suite (pytest) |

---

## Web UI (Streamlit)

The project ships with a full Streamlit web application (`streamlit_app.py`) as the primary user-facing interface.

### Starting the UI

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501` by default.

### Features

| Feature | Description |
|---------|-------------|
| **Quick Mode** | Single RAGAssistant agent, local corpus only — fast results |
| **Full Research** | 3-agent CrewAI pipeline with live Tavily web search — thorough |
| **Sidebar config** | API keys (OpenAI, Tavily), mode, timeout, RAG top-k, data directory |
| **Input safety** | Questions are validated and sanitised before reaching agents |
| **Progress spinner** | Real-time status while research runs |
| **Formatted output** | Answers rendered as Markdown; raw JSON available via expander |
| **Query history** | Collapsible history panel with timestamps and elapsed time |
| **Error messages** | Clear, actionable messages for API key issues, timeouts, and validation failures |

### Resetting the assistant cache

The RAGAssistant is cached in `st.session_state`. To reload documents (e.g. after adding new files to `./data`), refresh the browser page.

---

## Safety & Security

All user inputs pass through `src/safety.py` before reaching any agent or tool.

### Input validation (`validate_question`)

| Check | Detail |
|-------|--------|
| Type check | Must be a `str` |
| Length | Min 3 / max 2 000 characters |
| Prompt-injection detection | Regex patterns block common override phrases (e.g. "ignore previous instructions", "DAN mode", `<system>` tags) |
| Control-character stripping | Null bytes and non-printable characters removed; newlines preserved |

### Filename sanitisation (`sanitize_filename`)

- Removes `..` path-traversal sequences
- Replaces `/`, `\`, and shell-special characters with `_`
- Ensures an empty filename falls back to `report.md`

### Output filtering (`filter_output`)

- Truncates LLM responses exceeding 50 000 characters
- Appends a visible truncation notice when the limit is hit

### Logging for compliance

Every validation event — success, rejection, and truncation — is written to `app.log` at `INFO` / `WARNING` level with the first 100 characters of the input for auditability without storing full user data.

---

## Resilience & Monitoring

`src/resilience.py` provides three independent building blocks.

### `with_retry` — exponential-backoff decorator

```python
from src.resilience import with_retry

@with_retry(max_retries=3, initial_wait=1.0, max_wait=30.0, backoff_factor=2.0,
            exceptions=(openai.RateLimitError, ConnectionError))
def call_api():
    ...
```

- Every failed attempt is logged at `WARNING` with attempt number and wait time.
- Final failure is logged at `ERROR`.
- Default: 3 retries, 1 s → 2 s → 4 s waits (capped at 30 s).

### `run_with_timeout` — wall-clock timeout

```python
from src.resilience import run_with_timeout

result = run_with_timeout(crew.kickoff, timeout_seconds=120, inputs={"question": q})
```

- Thread-based (`ThreadPoolExecutor`) — works on Windows and Unix.
- Raises `TimeoutError` with a clear message; logged at `ERROR`.

### `IterationLimiter` — loop cap

```python
from src.resilience import IterationLimiter

limiter = IterationLimiter(max_iterations=20, label="agent_loop")
while condition:
    limiter.tick()   # raises RuntimeError after 20 iterations
    ...
```

- Prevents silent infinite loops in agent or tool cycles.
- Supports `.reset()` and `.remaining` for monitoring.

---

## Interface Specification

### CLI interface (`src/main.py`)

```
python -m src.main [QUESTION]
```

| Input | Type | Description |
|-------|------|-------------|
| `QUESTION` | Positional string (optional) | Research question; defaults to `"Explain RAG and why chunk overlap helps."` |

**Output:**
- Verbose agent logs to `stdout`
- Final result printed after `===== FINAL RESULT =====`
- Report saved to `./outputs/report.md`
- Exit code `0` on success

### RAGAssistant API (`src/app.py`)

```python
from src.app import RAGAssistant

assistant = RAGAssistant()
assistant.load_and_ingest("./data")
result = assistant.query_with_agent("Your question", n_results=3)
```

**`query_with_agent` return schema:**

```json
{
  "question":      "string — original question",
  "answer":        "string — LLM-generated answer",
  "context_chunks": ["string", "..."],
  "sources":       ["string — filenames used"],
  "mode":          "agent_tool_calling"
}
```

### `build_crew` API (`src/crew.py`)

```python
from src.crew import build_crew

crew = build_crew(data_dir="data")
result = crew.kickoff(inputs={"question": "..."})
```

**`kickoff` input schema:**

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `question` | string | Yes | Research question injected into all task prompts |

**Output:** CrewAI `CrewOutput` object (str-able); report also saved to `./outputs/report.md`.

### Safety API (`src/safety.py`)

| Function | Signature | Returns | Raises |
|----------|-----------|---------|--------|
| `validate_question` | `(question: str) -> str` | Sanitised question | `InputValidationError` |
| `sanitize_filename` | `(filename: str) -> str` | Safe filename | — |
| `filter_output` | `(text: str, max_length: int) -> str` | Truncated text | — |

### Resilience API (`src/resilience.py`)

| Function / Class | Usage |
|-----------------|-------|
| `@with_retry(...)` | Decorator — wraps any callable with retry logic |
| `run_with_timeout(func, seconds, *args, **kwargs)` | Runs `func` with a hard timeout |
| `IterationLimiter(max_iterations, label)` | `.tick()` / `.reset()` / `.remaining` |

---

## Deployment Guide

### Local (development)

```bash
# 1. Clone & enter
git clone https://github.com/samrat-kar/rag-research-assistant.git
cd rag-research-assistant

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env and add OPENAI_API_KEY and TAVILY_API_KEY

# 5a. Run web UI
streamlit run streamlit_app.py

# 5b. Run CLI
python -m src.main "Your research question"
```

### Docker

```bash
# Build
docker build -t rag-assistant .

# Run CLI
docker run --env-file .env rag-assistant "What is quantum entanglement?"

# Run web UI (expose port 8501)
docker run --env-file .env -p 8501:8501 rag-assistant \
  streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Environment variables reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key for GPT-4o-mini and embeddings |
| `TAVILY_API_KEY` | **Yes** (Full Research) | — | Tavily API key for live web search |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `CHROMA_COLLECTION_NAME` | No | `rag_documents` | Name of the in-memory collection |

Copy `.env.example` to `.env` and fill in the required values.

---

## Logging & Health Checks

### Log output

The application writes structured logs to:
- **`app.log`** — Streamlit UI logs (file handler)
- **`stdout`** — all runs (console handler)

Log format: `YYYY-MM-DD HH:MM:SS [LEVEL] module: message`

Key logged events:

| Event | Level | Module |
|-------|-------|--------|
| Input validated | INFO | `src.safety` |
| Prompt-injection detected | WARNING | `src.safety` |
| Output truncated | WARNING | `src.safety` |
| Retry attempt | WARNING | `src.resilience` |
| Final failure after retries | ERROR | `src.resilience` |
| Timeout exceeded | ERROR | `src.resilience` |
| Iteration limit exceeded | ERROR | `src.resilience` |
| Documents loaded | INFO | `src.app` / `src.tools` |
| Embeddings generated | INFO | `src.vectordb` |
| Research complete | INFO | `streamlit_app` |

### Health check (CLI)

Verify the environment is set up correctly:

```bash
python -c "
from src.vectordb import VectorDB
from src.tools import load_local_docs
docs = load_local_docs('data')
print(f'OK — loaded {len(docs)} documents from ./data')
"
```

Expected output: `OK — loaded 7 documents from ./data`

---

## Troubleshooting & FAQ

### 1. `OPENAI_API_KEY is required` error

**Cause:** The `.env` file is missing or the key is not set.

**Fix:**
```bash
cp .env.example .env
# Open .env and add:  OPENAI_API_KEY=sk-...
```

---

### 2. `TavilySearchTool` / web search fails

**Cause:** `TAVILY_API_KEY` is missing or expired.

**Fix:** Add a valid Tavily key to `.env` or use **Quick Mode** (no web search required).

---

### 3. Streamlit app shows "An unexpected error occurred"

**Fix steps:**
1. Check `app.log` for the full traceback.
2. Verify both API keys are set in the sidebar or `.env`.
3. Check your internet connection (required for OpenAI and Tavily calls).
4. If the error is a timeout, increase the **Timeout** slider in the sidebar.

---

### 4. `No relevant local context found` in every answer

**Cause:** The `./data` directory is empty or the documents failed to ingest.

**Fix:**
```bash
ls data/          # confirm files exist
python -c "from src.tools import load_local_docs; print(load_local_docs('data'))"
```

If the list is empty, add `.txt` / `.md` files to `./data` and restart.

---

### 5. Tests fail with `OPENAI_API_KEY` errors

All tests mock the OpenAI client — no real key is needed.  Make sure you are running from the project root:

```bash
pytest tests/ -v
```

If you see import errors, ensure the virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements-dev.txt
```

---

### 6. `ModuleNotFoundError: No module named 'streamlit'`

```bash
pip install streamlit>=1.35.0
# or
pip install -r requirements.txt
```

---

### 7. Docker container exits immediately

Ensure the `.env` file is in the project root and passed to `docker run`:

```bash
docker run --env-file .env rag-assistant "Test question"
```

---

### 8. Reports are not saved

The `./outputs/` directory is created automatically.  If you see permission errors, ensure the process has write access to the working directory.  For Docker, mount a volume:

```bash
docker run --env-file .env -v $(pwd)/outputs:/app/outputs rag-assistant "Question"
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE) license.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## Contact

**Maintainer:** Samrat Kar
**Email:** samrat.kar@example.com
**GitHub:** [samrat-kar](https://github.com/samrat-kar)
