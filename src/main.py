from __future__ import annotations

import sys
from .crew import build_crew


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "Explain RAG and why chunk overlap helps."

    crew = build_crew(data_dir="data")
    result = crew.kickoff(inputs={"question": question})
    print("\n\n===== FINAL RESULT =====\n")
    print(result)


if __name__ == "__main__":
    main()