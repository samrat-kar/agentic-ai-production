# RAG-Based AI Assistant - AAIDC Project 1 Template

Welcome to your RAG (Retrieval-Augmented Generation) project! This repository provides a **template** that you need to complete. The framework is set up, but the core functionality is missing - that's your job to implement!

## 🎯 What You Need to Build

You will implement a complete RAG system that can:

- Load and chunk documents from the `data/` directory
- Create embeddings and store them in a vector database
- Search for relevant context based on user queries
- Generate responses using retrieved context and an LLM

## 🔧 Implementation Freedom

**Important:** This template uses specific packages (ChromaDB, LangChain, HuggingFace Transformers) and approaches, but **you are completely free to use whatever you prefer!**

### Alternative Options You Can Choose:

**Vector Databases:**
- FAISS (Facebook AI Similarity Search)
- Pinecone
- Weaviate
- Qdrant
- Or any other vector store you prefer

**LLM Frameworks:**
- Direct API calls (OpenAI, Anthropic, etc.)
- Ollama for local models
- Hugging Face Transformers
- LlamaIndex instead of LangChain

**Embedding Models:**
- OpenAI embeddings (ada-002)
- Cohere embeddings
- Any Hugging Face model
- Local embedding models

**Text Processing:**
- Custom chunking logic
- spaCy for advanced NLP
- NLTK for text processing
- Your own parsing methods

### Why This Template Exists

This template provides **one working approach** that's:
- ✅ Well-documented and beginner-friendly
- ✅ Uses popular, stable packages
- ✅ Demonstrates core RAG concepts clearly
- ✅ Gets you started quickly

**Feel free to replace any component with your preferred tools!** The learning objectives remain the same regardless of your technology choices.

## 📋 Prerequisites

Before starting, make sure you have:

- Python 3.8 or higher installed
- An API key from **one** of these providers:
  - [OpenAI](https://platform.openai.com/api-keys) (most popular)
  - [Groq](https://console.groq.com/keys) (free tier available)
  - [Google AI](https://aistudio.google.com/app/apikey) (competitive pricing)

## 🚀 Quick Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your API key:**

   ```bash
   # Create environment file (choose the method that works on your system)
   cp .env.example .env    # Linux/Mac
   copy .env.example .env  # Windows
   ```

   Edit `.env` and add your API key:

   ```
   OPENAI_API_KEY=your_key_here
   # OR
   GROQ_API_KEY=your_key_here  
   # OR
   GOOGLE_API_KEY=your_key_here
   ```
3. **Test the setup:**

   ```bash
   python src/app.py
   ```

## 📝 Implementation Steps

### Step 0: Prepare Your Documents

**Replace the sample documents with your own content**

The `data/` directory contains sample files on various topics. Replace these with documents relevant to your domain:

```
data/
├── your_topic_1.txt
├── your_topic_2.txt
└── your_topic_3.txt
```

Each file should contain text content you want your RAG system to search through.

---

### Step 1: Implement Text Chunking

**Location:** `src/vectordb.py`

```python
def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into smaller chunks for better retrieval.
  
    Args:
        text: Input text to chunk
        chunk_size: Approximate number of characters per chunk
  
    Returns:
        List of text chunks
    """
    # TODO: Your implementation here
```

**What you need to do:**

- Choose a chunking strategy (word-based, sentence-based, or use LangChain's text splitters)
- Split the input text into manageable chunks
- Return a list of text strings

**Hint:** You have multiple options - start simple with word-based splitting or explore LangChain's `RecursiveCharacterTextSplitter`.

---

### Step 2: Implement Document Ingestion

**Location:** `src/vectordb.py`

```python
def add_documents(self, documents: List[Dict[str, Any]]) -> None:
    """
    Process documents and add them to the vector database.
  
    Args:
        documents: List of documents with 'content' and optional 'metadata'
    """
    # TODO: Your implementation here
```

**What you need to do:**

- Loop through the documents list
- Extract content and metadata from each document
- Use your `chunk_text()` method to split documents
- Create embeddings using `self.embedding_model.encode()`
- Store everything in ChromaDB using `self.collection.add()`

**Key components:**

- Chunk each document's content
- Generate unique IDs for each chunk
- Create embeddings for all chunks
- Store in the vector database

---

### Step 3: Implement Similarity Search

**Location:** `src/vectordb.py`

```python
def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Find documents similar to the query.
  
    Args:
        query: Search query
        n_results: Number of results to return
  
    Returns:
        Dictionary with search results
    """
    # TODO: Your implementation here
```

**What you need to do:**

- Create an embedding for the query using `self.embedding_model.encode()`
- Search the ChromaDB collection using `self.collection.query()`
- Return results in the expected format with keys: `documents`, `metadatas`, `distances`, `ids`

---

### Step 4: Implement Document Loading in RAG Assistant

**Location:** `src/app.py`

```python
def add_documents(self, documents: List[Dict[str, Any]]) -> None:
    """
    Add documents to the RAG system's knowledge base.
  
    Args:
        documents: List of documents to add
    """
    # TODO: Your implementation here
```

**What you need to do:**

- Call `self.vector_db.add_documents()` with the provided documents
- Add appropriate logging/print statements

---

### Step 5: Implement RAG Query Pipeline

**Location:** `src/app.py`

```python
def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Answer questions using retrieved context.
  
    Args:
        question: User's question
        n_results: Number of context chunks to retrieve
  
    Returns:
        Dictionary with answer and context information
    """
    # TODO: Your implementation here
```

**What you need to do:**

- Use `self.vector_db.search()` to find relevant context
- Combine retrieved chunks into a context string
- Use `self.chain.invoke()` to generate a response
- Return a dictionary with the answer and metadata

**The RAG pipeline:**

1. Search for relevant chunks
2. Combine chunks into context
3. Generate response using LLM + context
4. Return structured results

---

## 🧪 Testing Your Implementation

### Test Individual Components

1. **Test chunking:**

   ```python
   from src.vectordb import VectorDB
   vdb = VectorDB()
   chunks = vdb.chunk_text("Your test text here...")
   print(f"Created {len(chunks)} chunks")
   ```
2. **Test document loading:**

   ```python
   documents = [{"content": "Test document", "metadata": {"title": "Test"}}]
   vdb.add_documents(documents)
   ```
3. **Test search:**

   ```python
   results = vdb.search("your test query")
   print(f"Found {len(results['documents'])} results")
   ```

### Test Full System

Once implemented, run:

```bash
python src/app.py
```

Try these example questions:

- "What is [topic from your documents]?"
- "Explain [concept from your documents]"
- "How does [process from your documents] work?"

---

## 📁 Project Structure

```
rt-aaidc-project1-template/
├── src/
│   ├── app.py           # Main RAG application (implement Steps 4-5)
│   └── vectordb.py      # Vector database wrapper (implement Steps 1-3)
├── data/               # Replace with your documents (Step 0)
│   ├── *.txt          # Your text files here
├── requirements.txt    # All dependencies included
├── .env.example       # Environment template
└── README.md          # This guide
```

---

## 🆘 Getting Help

### Common Issues

**Import Errors:**

```bash
pip install -r requirements.txt
```

**API Key Errors:**

- Check your `.env` file has the correct API key
- Verify the key is valid and has credits/quota

**Empty Search Results:**

- Make sure you've implemented and called `add_documents()`
- Check that your documents were actually added to the database

**Poor Responses:**

- Try different chunking strategies
- Adjust the number of retrieved results (`n_results`)
- Experiment with different embedding models

---

## 🎓 Learning Objectives

By completing this project, you will:

- ✅ Understand RAG architecture and data flow
- ✅ Implement text chunking strategies
- ✅ Work with vector databases and embeddings
- ✅ Build LLM-powered applications with LangChain
- ✅ Handle multiple API providers
- ✅ Create production-ready AI applications

---

## 🏁 Success Criteria

Your implementation is complete when:

1. ✅ You can load your own documents
2. ✅ The system chunks and embeds documents
3. ✅ Search returns relevant results
4. ✅ The RAG system generates contextual answers
5. ✅ You can ask questions and get meaningful responses

**Good luck building your RAG system! 🚀**
