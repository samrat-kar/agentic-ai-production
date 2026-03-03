import os
from typing import List, Dict, Any
import numpy as np
from langchain_openai import OpenAIEmbeddings


class VectorDB:
    """
    A simple in-memory vector database wrapper using OpenAI embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Logical collection name (kept for compatibility)
            embedding_model: OpenAI embedding model name
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings. "
                "Please set OPENAI_API_KEY in your .env file."
            )

        # Initialize embedding client used during ingestion and retrieval.
        print(f"Loading OpenAI embedding model: {self.embedding_model_name}")
        self.embedding_model = OpenAIEmbeddings(
            model=self.embedding_model_name,
            api_key=api_key,
        )

        # In-memory index storage (documents + metadata + vectors).
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self._embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

        print(f"Vector database initialized (in-memory): {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into smaller chunks using RecursiveCharacterTextSplitter.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Recursive splitter preserves semantic boundaries better than naive fixed slicing.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents (dicts with 'content' and optional 'metadata')
        """
        print(f"Processing {len(documents)} documents...")

        all_chunks = []
        all_ids = []
        all_metadatas = []

        for doc_idx, doc in enumerate(documents):
            # Support either structured dict input or raw text input.
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}

            chunks = self.chunk_text(content)
            print(f"  Document {doc_idx + 1}: split into {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                # Store serializable metadata only.
                all_metadatas.append({
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    **{k: str(v) for k, v in metadata.items()},
                })

        if all_chunks:
            print(f"  Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = np.array(
                self.embedding_model.embed_documents(all_chunks),
                dtype=np.float32,
            )

            if self._embeddings.size == 0:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])

            self._documents.extend(all_chunks)
            self._metadatas.extend(all_metadatas)
            self._ids.extend(all_ids)

        print(f"Documents added to vector database ({len(all_chunks)} chunks total)")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        count = len(self._documents)
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        n_results = min(n_results, count)

        # Embed user query with same model space as document vectors.
        query_embedding = np.array(
            self.embedding_model.embed_query(query),
            dtype=np.float32,
        )

        # Cosine similarity
        doc_norms = np.linalg.norm(self._embeddings, axis=1) + 1e-12
        query_norm = np.linalg.norm(query_embedding) + 1e-12
        similarities = (self._embeddings @ query_embedding) / (doc_norms * query_norm)

        # Get top-k indices by similarity (descending)
        top_indices = np.argsort(similarities)[::-1][:n_results]

        top_docs = [self._documents[i] for i in top_indices]
        top_metas = [self._metadatas[i] for i in top_indices]
        top_ids = [self._ids[i] for i in top_indices]
        # Convert similarity to distance for compatibility with vector DB style output.
        top_distances = [float(1.0 - similarities[i]) for i in top_indices]

        return {
            "documents": [top_docs],
            "metadatas": [top_metas],
            "distances": [top_distances],
            "ids": [top_ids],
        }
