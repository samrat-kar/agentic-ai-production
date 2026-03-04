"""Unit tests for VectorDB."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest


@pytest.fixture()
def mock_embeddings():
    """Patch OpenAIEmbeddings so no API key is needed."""
    with patch("src.vectordb.OpenAIEmbeddings") as mock_cls:
        instance = MagicMock()
        # Return deterministic 4-d embeddings for documents
        instance.embed_documents.side_effect = lambda texts: [
            [float(i), 0.0, 0.0, 1.0] for i, _ in enumerate(texts)
        ]
        # Return a fixed query embedding
        instance.embed_query.return_value = [0.0, 0.0, 0.0, 1.0]
        mock_cls.return_value = instance
        yield instance


@pytest.fixture()
def vector_db(mock_embeddings):
    """Create a VectorDB instance with mocked embeddings."""
    import os
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from src.vectordb import VectorDB
    return VectorDB(collection_name="test", embedding_model="mock")


class TestChunkText:
    def test_short_text_single_chunk(self, vector_db):
        chunks = vector_db.chunk_text("Hello world", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_multiple_chunks(self, vector_db):
        text = "word " * 200  # ~1000 chars
        chunks = vector_db.chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1


class TestAddAndSearch:
    def test_add_documents(self, vector_db):
        docs = [
            {"content": "AI is great", "metadata": {"source": "a.txt"}},
            {"content": "ML overview", "metadata": {"source": "b.txt"}},
        ]
        vector_db.add_documents(docs)
        assert len(vector_db._documents) > 0

    def test_search_returns_results(self, vector_db):
        docs = [{"content": "Deep learning uses neural networks.", "metadata": {"source": "dl.txt"}}]
        vector_db.add_documents(docs)
        results = vector_db.search("neural networks", n_results=1)
        assert len(results["documents"][0]) == 1
        assert len(results["metadatas"][0]) == 1

    def test_search_empty_db(self, vector_db):
        results = vector_db.search("anything")
        assert results["documents"] == [[]]
