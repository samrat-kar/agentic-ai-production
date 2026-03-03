"""Minimal CLI for the RAG assignment."""
from src.app import RAGAssistant


def main():
    # Initialize assistant and build retrieval index from local documents.
    assistant = RAGAssistant()
    assistant.load_and_ingest("./data")

    example_questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are key AI ethics concerns?",
    ]

    # Run a few predefined questions to show baseline behavior.
    print("\nExample queries:\n")
    for q in example_questions:
        result = assistant.query(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}\n")

    # Simple CLI loop for manual testing.
    print("Interactive mode (type 'quit' to exit)")
    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        result = assistant.query(question)
        print(f"Assistant: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
