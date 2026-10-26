
from rag.retriever import HybridRetriever
from rag.generator import generate_answer


def main():
    retriever = HybridRetriever()

    question = input("Ask PAIMANA AI a question: ")

    # Step 1: Find relevant project records
    results = retriever.search(question, top_k=3)

    print("\nRetrieved project records:")
    for item in results:
        project = item["project"]
        print(
            f"- {project['project_name']} "
            f"(Risk: {project['risk_status']})"
        )

    # Step 2: Generate an answer using Gemini
    print("\nPAIMANA AI is generating an answer...\n")

    answer = generate_answer(question, results)

    print("PAIMANA AI:")
    print(answer)


if __name__ == "__main__":
    main()
