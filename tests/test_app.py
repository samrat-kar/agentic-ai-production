"""Unit tests for src/app.py — RAGAssistant."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Provide required env var before importing src modules
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# RAGAssistant depends on langchain.agents; skip if not installed/compatible.
pytest.importorskip(
    "langchain.agents",
    reason="langchain.agents not available — run in .venv312",
)


@pytest.fixture()
def mock_openai():
    """Patch ChatOpenAI and OpenAIEmbeddings to avoid real API calls."""
    # Pre-import the module so the patch target is resolvable.
    import src.app  # noqa: F401
    with (
        patch("src.app.ChatOpenAI") as mock_llm_cls,
        patch("src.vectordb.OpenAIEmbeddings") as mock_emb_cls,
    ):
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # Mock embeddings
        mock_emb = MagicMock()
        mock_emb.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        mock_emb.embed_query.return_value = [1.0, 0.0]
        mock_emb_cls.return_value = mock_emb

        yield {"llm": mock_llm, "embeddings": mock_emb}


@pytest.fixture()
def mock_agent():
    """Patch create_agent so no LangChain agent is actually built."""
    with patch("src.app.create_agent") as mock_create:
        agent_instance = MagicMock()
        mock_create.return_value = agent_instance
        yield agent_instance


class TestRAGAssistantInit:
    def test_initialises_without_error(self, mock_openai, mock_agent):
        from src.app import RAGAssistant

        assistant = RAGAssistant()
        assert assistant is not None
        assert assistant.vector_db is not None

    def test_raises_without_api_key(self, mock_agent):
        """RAGAssistant should raise ValueError if OPENAI_API_KEY is missing."""
        env_backup = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                from src.app import RAGAssistant

                RAGAssistant()
        finally:
            if env_backup is not None:
                os.environ["OPENAI_API_KEY"] = env_backup
            else:
                os.environ["OPENAI_API_KEY"] = "test-key"


class TestLoadDocuments:
    def test_returns_empty_for_missing_dir(self, mock_openai, mock_agent, tmp_path):
        from src.app import RAGAssistant

        assistant = RAGAssistant()
        docs = assistant.load_documents(str(tmp_path / "nonexistent"))
        assert docs == []

    def test_loads_txt_files(self, mock_openai, mock_agent, tmp_path):
        # Create a sample .txt file
        (tmp_path / "doc.txt").write_text("Sample content", encoding="utf-8")
        from src.app import RAGAssistant

        assistant = RAGAssistant()
        docs = assistant.load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["content"] == "Sample content"
        assert docs[0]["metadata"]["source"] == "doc.txt"

    def test_ignores_unsupported_extensions(self, mock_openai, mock_agent, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
        from src.app import RAGAssistant

        assistant = RAGAssistant()
        docs = assistant.load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["metadata"]["source"] == "doc.txt"

    def test_loads_multiple_formats(self, mock_openai, mock_agent, tmp_path):
        (tmp_path / "a.txt").write_text("text", encoding="utf-8")
        (tmp_path / "b.md").write_text("markdown", encoding="utf-8")
        (tmp_path / "c.json").write_text('{"key": "value"}', encoding="utf-8")
        from src.app import RAGAssistant

        assistant = RAGAssistant()
        docs = assistant.load_documents(str(tmp_path))
        assert len(docs) == 3


class TestLoadAndIngest:
    def test_ingest_adds_documents(self, mock_openai, mock_agent, tmp_path):
        (tmp_path / "content.txt").write_text(
            "Machine learning is a subset of AI.", encoding="utf-8"
        )
        from src.app import RAGAssistant

        assistant = RAGAssistant()
        assistant.load_and_ingest(str(tmp_path))
        assert len(assistant.vector_db._documents) > 0

    def test_ingest_empty_dir_no_crash(self, mock_openai, mock_agent, tmp_path):
        from src.app import RAGAssistant

        assistant = RAGAssistant()
        assistant.load_and_ingest(str(tmp_path))  # Should not raise


class TestQueryWithAgent:
    def test_returns_expected_keys(self, mock_openai, mock_agent):
        # Configure mock agent to return a message-like result
        mock_message = MagicMock()
        mock_message.content = "Neural networks learn via backpropagation."
        mock_agent.invoke.return_value = {"messages": [mock_message]}

        from src.app import RAGAssistant

        assistant = RAGAssistant()
        result = assistant.query_with_agent("What is backpropagation?")

        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert "mode" in result
        assert result["mode"] == "agent_tool_calling"

    def test_answer_extracted_from_last_message(self, mock_openai, mock_agent):
        mock_message = MagicMock()
        mock_message.content = "The answer is 42."
        mock_agent.invoke.return_value = {"messages": [MagicMock(), mock_message]}

        from src.app import RAGAssistant

        assistant = RAGAssistant()
        result = assistant.query_with_agent("What is the answer?")
        assert result["answer"] == "The answer is 42."

    def test_handles_empty_messages(self, mock_openai, mock_agent):
        mock_agent.invoke.return_value = {"messages": []}

        from src.app import RAGAssistant

        assistant = RAGAssistant()
        result = assistant.query_with_agent("test?")
        assert result["answer"] == ""
