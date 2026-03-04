"""Unit tests for custom CrewAI tools."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Ensure env var exists before importing src modules
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class TestCalculatorTool:
    def test_simple_addition(self):
        from src.tools import CalculatorTool
        tool = CalculatorTool()
        assert tool._run("2 + 3") == "5"

    def test_complex_expression(self):
        from src.tools import CalculatorTool
        tool = CalculatorTool()
        assert tool._run("(12*3)/4 + 10") == "19.0"

    def test_rejects_unsafe_chars(self):
        from src.tools import CalculatorTool
        tool = CalculatorTool()
        result = tool._run("__import__('os')")
        assert "Rejected" in result or "Error" in result


class TestSaveReportTool:
    def test_saves_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.tools import SaveReportTool
        tool = SaveReportTool()
        result = tool._run(filename="test_report.md", content="# Hello")
        assert "Saved" in result
        assert (tmp_path / "outputs" / "test_report.md").read_text() == "# Hello"

    def test_sanitizes_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.tools import SaveReportTool
        tool = SaveReportTool()
        result = tool._run(filename="../evil.md", content="bad")
        # ".." should be stripped
        saved_path = tmp_path / "outputs" / "evil.md"
        assert saved_path.exists()


class TestLocalRAGSearchTool:
    @patch("src.vectordb.OpenAIEmbeddings")
    def test_returns_chunks(self, mock_emb_cls):
        instance = MagicMock()
        instance.embed_documents.side_effect = lambda texts: [
            [1.0, 0.0] for _ in texts
        ]
        instance.embed_query.return_value = [1.0, 0.0]
        mock_emb_cls.return_value = instance

        from src.vectordb import VectorDB
        from src.tools import LocalRAGSearchTool

        vdb = VectorDB(collection_name="test", embedding_model="mock")
        vdb.add_documents([{"content": "test content", "metadata": {"source": "t.txt"}}])

        tool = LocalRAGSearchTool(vdb=vdb)
        result = tool._run(query="test", top_k=1)
        assert "t.txt" in result
