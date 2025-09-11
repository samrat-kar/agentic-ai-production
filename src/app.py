import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant. Use the following retrieved context to answer the user's question.
If the context doesn't contain relevant information, say so and provide a general response based on your knowledge.

Context:
{context}

Question: {question}

Answer:
"""
        )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                print(f"Using OpenAI model: {model_name}")
                return ChatOpenAI(api_key=openai_key, model=model_name, temperature=0.0)
            except ImportError:
                print("Warning: langchain-openai not installed, skipping OpenAI")

        # Check for Groq API key
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
                print(f"Using Groq model: {model_name}")
                return ChatGroq(api_key=groq_key, model=model_name, temperature=0.0)
            except ImportError:
                print("Warning: langchain-groq not installed, skipping Groq")

        # Check for Google API key
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
                print(f"Using Google Gemini model: {model_name}")
                return ChatGoogleGenerativeAI(
                    google_api_key=google_key, model=model_name, temperature=0.0
                )
            except ImportError:
                print("Warning: langchain-google-genai not installed, skipping Google")

        return None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents with 'content' and optional 'metadata'
        """
        self.vector_db.add_documents(documents)

    def query(self, question: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            question: User's question
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        llm_answer = ""
        # TODO: Implement the RAG query pipeline
        # HINT: Use self.vector_db.search() to retrieve relevant context chunks
        # HINT: Combine the retrieved document chunks into a single context string
        # HINT: Use self.chain.invoke() with context and question to generate the response
        # HINT: Return a string answer from the LLM

        # Your implementation here
        return llm_answer


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    # TODO: Implement document loading
    # HINT: Read the documents from the data directory
    # HINT: Return a list of documents
    # HINT: Your implementation depends on the type of documents you are using (.txt, .pdf, etc.)

    # Your implementation here
    return results


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        # TODO: Uncomment the following lines once you implement the methods:
        # assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.query(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
