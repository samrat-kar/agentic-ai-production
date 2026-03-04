# Multi-Agent RAG Research Assistant (CrewAI)

## Requirements Covered

### 1) Multi-Agent System (>= 3 agents)
- Research Agent: gathers evidence from web + local RAG
- Analyst Agent: validates, reconciles, calculates
- Writer Agent: produces final markdown report + saves to file

Orchestration: CrewAI `Process.sequential` (research -> analysis -> writing)

### 2) Tool Integration (>= 3 tools)
- TavilySearchTool (web search)
- LocalRAGSearchTool (custom tool; retrieves from local vector DB)
- CalculatorTool (custom)
- SaveReportTool (custom; writes output report)

## Setup

1. Create a Python 3.12 virtual environment and install dependencies:
   py -3.12 -m venv .venv312
   .\.venv312\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

2. Create `.env` from `.env.example` and fill:
   OPENAI_API_KEY=...
   TAVILY_API_KEY=...

3. Put any knowledge files in `./data` (.txt/.md/.json/.csv)

## Run
python -m src.main "Your research question here"

Outputs:
- Console prints final result
- Saved report at `./outputs/report.md`