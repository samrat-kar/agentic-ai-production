"""Minimal CLI for the RAG assignment."""
import sys

from src.app import RAGAssistant


def main() -> None:
    try:
        # Initialize assistant and build retrieval index from local documents.
        assistant = RAGAssistant()
        assistant.load_and_ingest("./data")
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print("Set OPENAI_API_KEY in your .env file and try again.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to initialise assistant: {exc}", file=sys.stderr)
        print("Run `python -m src.health` to diagnose.", file=sys.stderr)
        sys.exit(1)

    example_questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are key AI ethics concerns?",
    ]

    # Run a few predefined questions to show tool-calling agent behavior.
    print("\nExample queries (tool-calling agent):\n")
    for q in example_questions:
        try:
            result = assistant.query_with_agent(q)
            print(f"Q: {q}")
            print(f"A: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}\n")
        except Exception as exc:
            print(f"Q: {q}")
            print(f"Error: {exc}\n", file=sys.stderr)

    # Simple CLI loop for manual testing.
    print("Interactive mode (type 'quit' to exit)")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        try:
            result = assistant.query_with_agent(question)
            print(f"Assistant: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
