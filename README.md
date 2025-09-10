# RAG-Based AI Assistant - AAIDC Project 1 Template

This repository contains a bare minimum working template for Project 1 in the Agentic AI Developer Certification (AAIDC) program. It implements a Retrieval-Augmented Generation (RAG) system using ChromaDB, HuggingFace embeddings, and OpenAI's language models.

## 🚀 Features

- **RAG-based AI Assistant**: Complete retrieval-augmented generation pipeline
- **Vector Database**: ChromaDB integration for document storage and similarity search  
- **Document Embedding**: HuggingFace Sentence Transformers for text embeddings
- **Simple Text Chunking**: Basic space-based text splitting for document processing
- **OpenAI Integration**: Uses OpenAI's GPT models for response generation
- **Interactive Interface**: Command-line interface for querying the assistant
- **Sample Data**: Pre-loaded with AI/ML related documents for demonstration

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for downloading models

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd rt-aaidc-project1-template
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env_example .env
```

Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application
```bash
cd src
python app.py
```

## 💻 Usage

### Basic Usage
1. Run the application: `python src/app.py`
2. The system will load sample documents automatically
3. Ask questions in the interactive prompt
4. Type 'quit' to exit

### Example Queries
- "What is artificial intelligence?"
- "Explain machine learning"
- "What are the stages of a data science workflow?"
- "Tell me about AI ethics"

### Adding Your Own Documents
You can modify the `load_sample_documents()` function in `src/app.py` to add your own documents:

```python
documents = [
    {
        "content": "Your document content here...",
        "metadata": {"title": "Document Title", "category": "your_category"}
    }
]
```

## 🏗️ Architecture

### Components

1. **VectorDB (`src/vectordb.py`)**
   - ChromaDB wrapper for vector storage
   - HuggingFace embeddings integration
   - Simple text chunking functionality
   - Document search and retrieval

2. **RAGAssistant (`src/app.py`)**
   - Main application class
   - OpenAI LLM integration
   - Query processing pipeline
   - Interactive interface

### Data Flow
1. **Document Ingestion**: Text documents are chunked and embedded
2. **Storage**: Embeddings stored in ChromaDB vector database  
3. **Query Processing**: User queries are embedded and matched against stored documents
4. **Context Retrieval**: Most similar document chunks are retrieved
5. **Response Generation**: OpenAI model generates responses using retrieved context

## 🔧 Configuration Options

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-3.5-turbo)
- `EMBEDDING_MODEL`: HuggingFace model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)
- `CHROMA_COLLECTION_NAME`: ChromaDB collection name (default: rag_documents)

### Customization
- **Chunk Size**: Modify `chunk_size` parameter in `VectorDB.chunk_text()`
- **Retrieval Count**: Adjust `n_results` in query methods
- **Temperature**: Change LLM temperature in `RAGAssistant.__init__()`

## 📁 Project Structure

```
rt-aaidc-project1-template/
├── src/
│   ├── app.py              # Main RAG assistant application
│   └── vectordb.py         # Vector database wrapper
├── data/
│   └── sample_documents.txt # Sample documents for testing
├── requirements.txt        # Python dependencies
├── .env_example           # Environment variables template
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🧪 Testing the System

### Verify Installation
```bash
cd src
python -c "from app import RAGAssistant; print('Installation successful!')"
```

### Test Basic Functionality
```bash
cd src
python app.py
# Ask: "What is machine learning?"
# Expected: Response based on loaded sample documents
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   - Error: `OPENAI_API_KEY environment variable is required`
   - Solution: Add your API key to the `.env` file

2. **Model Download Issues**
   - Error: Network errors when downloading HuggingFace models
   - Solution: Ensure stable internet connection, models will be cached locally

3. **ChromaDB Permissions**
   - Error: Unable to create `chroma_db` directory
   - Solution: Ensure write permissions in the project directory

### Performance Notes
- First run may be slower due to model downloads
- Embedding generation time depends on document size
- Consider using smaller embedding models for faster performance

## 📚 Key Learning Objectives

This template demonstrates:
- ✅ RAG system architecture and implementation
- ✅ Vector database integration with ChromaDB
- ✅ Document embedding using HuggingFace models
- ✅ Text chunking strategies for document processing
- ✅ LangChain integration for LLM workflows
- ✅ Environment configuration and security best practices

## 🔗 Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a template project for educational purposes. Feel free to fork and customize for your own RAG applications!
