# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""Custom CrewAI tools for the RAG research assistant.

Provides:
- ``LocalRAGSearchTool`` — semantic search over local knowledge-base files.
- ``CalculatorTool`` — safe arithmetic expression evaluator.
- ``SaveReportTool`` — writes the final report to ``./outputs/``.

Also includes helpers for loading documents and building the vector DB.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .vectordb import VectorDB

logger = logging.getLogger(__name__)


# -----------------------------
# Helper: build/load local RAG index
# -----------------------------
def load_local_docs(data_dir: str = "data") -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    p = Path(data_dir)
    if not p.exists():
        return docs

    for fp in sorted(p.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in (".txt", ".md", ".csv", ".json"):
            continue
        try:
            content = fp.read_text(encoding="utf-8")
            docs.append(
                {"content": content, "metadata": {"source": fp.name, "path": str(fp)}}
            )
        except Exception:
            pass
    return docs


def build_vectordb(data_dir: str = "data") -> VectorDB:
    vdb = VectorDB()
    docs = load_local_docs(data_dir)
    if docs:
        vdb.add_documents(docs)
    else:
        logger.warning("No docs found in %s. Add files to improve RAG answers.", data_dir)
    return vdb


# -----------------------------
# Tool 1: Local RAG retrieval tool
# -----------------------------
class RAGSearchInput(BaseModel):
    query: str = Field(..., description="User question or search query.")
    top_k: int = Field(4, description="How many chunks to retrieve.")


class LocalRAGSearchTool(BaseTool):
    name: str = "local_rag_search"
    description: str = (
        "Searches the local knowledge base (./data) using embeddings and returns top relevant chunks with sources."
    )
    args_schema: Type[BaseModel] = RAGSearchInput

    def __init__(self, vdb: VectorDB):
        super().__init__()
        self._vdb = vdb

    def _run(self, query: str, top_k: int = 4) -> str:
        res = self._vdb.search(query, n_results=top_k)
        chunks = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        if not chunks:
            return "No relevant local context found."

        lines = []
        for i, (c, m) in enumerate(zip(chunks, metas), start=1):
            src = m.get("source", "unknown")
            lines.append(f"[{i}] Source: {src}\n{c}")
        return "\n\n---\n\n".join(lines)


# -----------------------------
# Tool 2: Safe calculator tool (Python)
# -----------------------------
class CalcInput(BaseModel):
    expression: str = Field(..., description="Math expression like: (12*3)/4 + 10")


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluates basic math expressions safely (no imports, no variables, no functions)."
    args_schema: Type[BaseModel] = CalcInput

    def _run(self, expression: str) -> str:
        allowed = set("0123456789+-*/(). %")
        if any(ch not in allowed for ch in expression):
            return "Rejected: expression contains unsupported characters."

        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"


# -----------------------------
# Tool 3: Save final report to disk
# -----------------------------
class SaveReportInput(BaseModel):
    filename: str = Field(..., description="Output filename, e.g. report.md")
    content: str = Field(..., description="File content to write.")


class SaveReportTool(BaseTool):
    name: str = "save_report"
    description: str = "Writes content to an output file under ./outputs and returns the saved path."
    args_schema: Type[BaseModel] = SaveReportInput

    def _run(self, filename: str, content: str) -> str:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_name = filename.replace("..", "").replace("/", "_").replace("\\", "_")
        out_path = out_dir / safe_name
        out_path.write_text(content, encoding="utf-8")
        return f"Saved report to: {out_path}"