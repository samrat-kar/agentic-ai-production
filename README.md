# Multi-Agent RAG Research Assistant (CrewAI)

A multi-agent research assistant that combines live web search with local document retrieval (RAG) to produce grounded, source-cited answers. Three specialized AI agents — Research Agent, Analyst Agent, and Writer Agent — collaborate in a sequential pipeline orchestrated by CrewAI, outputting a structured Markdown report.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Examples](#code-examples)
- [Testing](#testing)
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
