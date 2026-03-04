# Multi-Agent RAG Research Assistant — Setup & Usage Instructions

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Running the Application](#running-the-application)
5. [Sample Inputs](#sample-inputs)
6. [Expected Outputs](#expected-outputs)
7. [Project Structure](#project-structure)

---

## Project Overview

This project implements a **Multi-Agent Retrieval-Augmented Generation (RAG) Research Assistant** using **CrewAI**. It uses three specialized AI agents that collaborate in a sequential pipeline to answer research questions:

| Agent | Role | Tools Used |
|-------|------|-----------|
| **Research Agent** | Gathers evidence from the web and a local knowledge base | TavilySearchTool, LocalRAGSearchTool |
| **Analyst Agent** | Validates claims, resolves conflicts, performs calculations | CalculatorTool, LocalRAGSearchTool |
| **Writer Agent** | Produces and saves a final structured markdown report | SaveReportTool |

The system also includes a simpler **demo mode** (`demo.py`) that runs a single tool-calling agent backed by the same local vector store.

---

## Prerequisites

- **Python 3.12** (required)
- **OpenAI API Key** — for the LLM (GPT-4o-mini) and text embeddings
- **Tavily API Key** — for web search functionality (used by the Research Agent)

---

## Setup Instructions

### Step 1 — Clone / Download the Repository

Place the project folder on your machine (e.g., `rt-aaidc-project1-template`).

### Step 2 — Create a Virtual Environment

Open a terminal in the project root and run:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
```

### Step 3 — Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root with the following content:

```
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Used for the GPT-4o-mini LLM and text-embedding-3-small embedding model |
| `TAVILY_API_KEY` | Used by the Tavily web search tool (Research Agent) |

### Step 5 — Add Knowledge Documents (Optional)

The `data/` folder comes pre-loaded with sample `.txt` files on topics such as AI, quantum computing, and biotechnology. You can add your own `.txt`, `.md`, `.csv`, or `.json` files to this folder to expand the local knowledge base.

---

## Running the Application

### Option A — Multi-Agent Crew (Main Entry Point)

This runs the full 3-agent pipeline (Research → Analysis → Writing) and saves a report.

```bash
python -m src.main "What are the latest advances in quantum computing?"
```

If no question is provided, it defaults to: *"Explain RAG and why chunk overlap helps."*

**Output locations:**
- Final result printed to the console
- Markdown report saved to `./outputs/report.md`

### Option B — Demo / Interactive Mode

This runs a simpler single-agent RAG assistant with predefined example queries, followed by an interactive prompt.

```bash
python demo.py
```

It will:
1. Ingest all documents from `./data`
2. Run three example questions automatically
3. Enter an interactive loop where you can type your own questions (type `quit` to exit)

---

## Sample Inputs

Below are example questions you can use with either entry point:

### For the Multi-Agent Crew (`src.main`)

```bash
python -m src.main "What are the latest advances in quantum computing?"
python -m src.main "How has CRISPR transformed gene editing in 2024?"
python -m src.main "Compare solar and wind energy efficiency"
python -m src.main "What are the main ethical concerns surrounding artificial intelligence?"
python -m src.main "Explain the CRISP-DM data science methodology"
```

### For the Demo Script (`demo.py`)

The demo automatically runs these three questions:

1. `"What is machine learning?"`
2. `"How does deep learning work?"`
3. `"What are key AI ethics concerns?"`

After that, you can type any question interactively, for example:

```
You: What is feature engineering?
You: Explain model evaluation techniques
You: quit
```

---

## Expected Outputs

### Multi-Agent Crew Output

When running `python -m src.main "What are the latest advances in quantum computing?"`, you will see:

1. **Console logs** showing each agent's verbose activity:
   - The Research Agent searches the web (Tavily) and the local knowledge base
   - The Analyst Agent reviews the research notes and validates claims
   - The Writer Agent composes and saves the final report

2. **Console final result** printed after `===== FINAL RESULT =====`

3. **Saved report** at `./outputs/report.md` containing structured markdown like:

```markdown
# [Topic] Report

## Short Answer
[A concise 1–2 sentence answer]

## Explanation
- Key finding 1
- Key finding 2
- Key finding 3

## Sources
- [Source 1]
- [Source 2]
```

A sample generated report is already included in `./outputs/report.md` for reference.

### Demo Script Output

When running `python demo.py`, expected console output looks like:

```
Loading OpenAI embedding model: text-embedding-3-small
Vector database initialized (in-memory): rag_documents
Using OpenAI model: gpt-4o-mini
RAG Assistant initialized successfully
Loaded 7 documents from ./data
Processing 7 documents...
  Document 1: split into X chunks
  ...
Documents added to vector database (XX chunks total)

Example queries (tool-calling agent):

Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables
   systems to learn and improve from experience without being explicitly
   programmed...
Sources: artificial_intelligence.txt, sample_documents.txt

Q: How does deep learning work?
A: Deep learning uses artificial neural networks with multiple layers
   to progressively extract higher-level features from raw input...
Sources: artificial_intelligence.txt

Q: What are key AI ethics concerns?
A: Key AI ethics concerns include bias and fairness, transparency,
   privacy, accountability, and the potential impact on employment...
Sources: artificial_intelligence.txt

Interactive mode (type 'quit' to exit)

You: _
```

---

## Project Structure

```
rt-aaidc-project1-template/
├── demo.py                  # Simple single-agent demo with interactive mode
├── requirements.txt         # Python dependencies
├── .env                     # API keys (you create this)
├── instructions.md          # This file
├── README.md                # Project overview
├── data/                    # Local knowledge base documents
│   ├── artificial_intelligence.txt
│   ├── biotechnology.txt
│   ├── climate_science.txt
│   ├── quantum_computing.txt
│   ├── sample_documents.txt
│   ├── space_exploration.txt
│   └── sustainable_energy.txt
├── outputs/                 # Generated reports
│   └── report.md            # Example output from a previous run
└── src/                     # Source code
    ├── main.py              # CLI entry point for the multi-agent crew
    ├── crew.py              # CrewAI agent/task/crew definitions
    ├── tools.py             # Custom tools (LocalRAGSearch, Calculator, SaveReport)
    ├── vectordb.py          # In-memory vector database with OpenAI embeddings
    └── app.py               # RAGAssistant class (used by demo.py)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY is required` | Make sure your `.env` file exists in the project root and contains a valid key |
| `ModuleNotFoundError` | Ensure the virtual environment is activated and `pip install -r requirements.txt` was run |
| `No relevant context found` | Add more documents to the `data/` folder or check that files are `.txt`, `.md`, `.csv`, or `.json` |
| Tavily search errors | Verify that `TAVILY_API_KEY` is set correctly in `.env` |
