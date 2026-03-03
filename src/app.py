"""Simple RAG assistant implementation for assignment submission."""
import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .vectordb import VectorDB

# Load environment variables
load_dotenv()


class RAGAssistant:
    """Question-answering assistant powered by Retrieval-Augmented Generation."""

    def __init__(self):
        """Initialize the RAG assistant components (LLM, prompt chain, vector store)."""
        # LLM used for final answer generation.
        self.llm = self._initialize_llm()

        # Prompt template for grounded answering.
        # The model is instructed to rely on retrieved context and avoid hallucinations.
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Use only the provided context to answer the question.
If the context does not contain enough information, say so honestly rather than making up an answer.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context provided."""
        )

    # End-to-end generation chain: prompt -> model -> plain string output.
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    # Vector index used by the retriever step.
        self.vector_db = VectorDB()
        print("RAG Assistant initialized successfully")

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
            print(f"⚠ Data directory not found: {data_path}")
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
                print(f"⚠ Could not read {file_path.name}: {exc}")

        print(f"Loaded {len(results)} documents from {data_path}")
        return results

    def load_and_ingest(self, data_path: str = "./data") -> None:
        """Load documents and ingest them into the vector store."""
        documents = self.load_documents(data_path)
        if documents:
            self.vector_db.add_documents(documents)
        else:
            print("No documents to ingest.")

    # ------------------------------------------------------------------
    # RAG query
    # ------------------------------------------------------------------

    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Answer a question using retrieved context (RAG).

        Args:
            question: The user's question
            n_results: Number of context chunks to retrieve

        Returns:
            Dict with answer text, retrieved chunks, and source file names
        """
        # 1. Retrieve relevant chunks
        search_results = self.vector_db.search(question, n_results=n_results)

        chunks = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]

        # 2. Build context string
        context = "\n\n---\n\n".join(chunks) if chunks else "No relevant context found."

        # 3. Generate answer via LLM chain
        answer = self.chain.invoke({"context": context, "question": question})

        # 4. Collect unique source file names for transparency/citations
        sources = list({m.get("source", "unknown") for m in metadatas}) if metadatas else []

        return {
            "question": question,
            "answer": answer,
            "context_chunks": chunks,
            "sources": sources,
        }

    def _initialize_llm(self):
        """Initialize the chat model using environment-based OpenAI configuration."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required in .env")

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.2)


def main():
    """Quick smoke test: ingest docs and answer one sample question."""
    assistant = RAGAssistant()
    assistant.load_and_ingest("./data")
    result = assistant.query("What is machine learning?")
    print(result["answer"])


if __name__ == "__main__":
    main()
