# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""Single-agent RAG assistant used by demo.py for interactive Q&A.

Provides the RAGAssistant class: a tool-calling LangChain agent backed by
a local in-memory vector store. Used as the demo/interactive entry point,
separate from the multi-agent CrewAI pipeline in crew.py.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import SecretStr
from .vectordb import VectorDB

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RAGAssistant:
    """Question-answering assistant powered by Retrieval-Augmented Generation."""

    def __init__(self):
        """Initialize the RAG assistant components (LLM, agent, vector store)."""
        # LLM used for final answer generation.
        self.llm = self._initialize_llm()
        self._agent_n_results = 3

    # Vector index used by the retriever step.
        self.vector_db = VectorDB()

    # Tool-calling agent setup.
        self.tools = self._build_tools()
        self.agent = self._initialize_agent()
        logger.info("RAG Assistant initialized successfully")

    # ------------------------------------------------------------------
    # Document loading & ingestion
    # ------------------------------------------------------------------

    def load_documents(self, data_path: str = "./data") -> List[Dict[str, Any]]:
        """
        Load documents from the data directory.

        Args:
            data_path: Path to folder containing .txt / .csv / .json files

        Returns:
            List of document dicts with 'content' and 'metadata'
        """
        results: List[Dict[str, Any]] = []
        dir_path = Path(data_path)

        if not dir_path.exists():
            logger.warning("Data directory not found: %s", data_path)
            return results

        for file_path in sorted(dir_path.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in (".txt", ".csv", ".json", ".md"):
                continue
            try:
                # Keep ingestion generic: all supported files are read as text content.
                content = file_path.read_text(encoding="utf-8")
                results.append({
                    "content": content,
                    "metadata": {
                        "source": file_path.name,
                        "path": str(file_path),
                    },
                })
            except Exception as exc:
                logger.warning("Could not read %s: %s", file_path.name, exc)

        logger.info("Loaded %d documents from %s", len(results), data_path)
        return results

    def load_and_ingest(self, data_path: str = "./data") -> None:
        """Load documents and ingest them into the vector store."""
        documents = self.load_documents(data_path)
        if documents:
            self.vector_db.add_documents(documents)
        else:
            logger.info("No documents to ingest.")

    # ------------------------------------------------------------------
    # RAG query
    # ------------------------------------------------------------------

    def query_with_agent(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Answer a question using a tool-calling agent.

        The agent can call the retrieval tool when needed before writing a response.
        """
        self._agent_n_results = n_results
        result = self.agent.invoke({
            "messages": [
                {"role": "user", "content": question},
            ]
        })

        answer = ""
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            last_message = messages[-1]
            answer = getattr(last_message, "content", "")
            if isinstance(answer, list):
                text_parts = []
                for part in answer:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                answer = "\n".join([p for p in text_parts if p]).strip()

        search_results = self.vector_db.search(question, n_results=n_results)
        chunks = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        sources = list({m.get("source", "unknown") for m in metadatas}) if metadatas else []

        return {
            "question": question,
            "answer": answer,
            "context_chunks": chunks,
            "sources": sources,
            "mode": "agent_tool_calling",
        }

    def _build_tools(self):
        """Create tools available to the agent."""

        @tool
        def retrieve_context(question: str) -> str:
            """Retrieve top relevant document chunks for a user question."""
            search_results = self.vector_db.search(question, n_results=self._agent_n_results)
            chunks = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]

            if not chunks:
                return "No relevant context found."

            lines = []
            for idx, (chunk, metadata) in enumerate(zip(chunks, metadatas), start=1):
                source = metadata.get("source", "unknown")
                lines.append(f"[{idx}] Source: {source}\n{chunk}")

            return "\n\n---\n\n".join(lines)

        return [retrieve_context]

    def _initialize_agent(self):
        """Create the tool-calling agent executor."""
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=(
                "You are a helpful assistant. Use tools when helpful to ground your answer in retrieved context. "
                "If context is insufficient, say so clearly instead of making up facts."
            ),
        )

    def _initialize_llm(self):
        """Initialize the chat model using environment-based OpenAI configuration."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required in .env")

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI model: %s", model_name)
        return ChatOpenAI(api_key=SecretStr(api_key), model=model_name, temperature=0.2)


def main():
    """Quick smoke test: ingest docs and answer one sample question."""
    assistant = RAGAssistant()
    assistant.load_and_ingest("./data")
    result = assistant.query_with_agent("What is machine learning?")
    print(result["answer"])


if __name__ == "__main__":
    main()
