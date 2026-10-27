
from rag.retriever import HybridRetriever

retriever = HybridRetriever()

questions = [
    "Which project has a schedule delay?",
    "Which project has a revised completion date?",
    "Which project has high expenditure compared with physical progress?",
]

for question in questions:
    print("\nQuestion:", question)
    print("-" * 50)

    results = retriever.search(question, top_k=3)

    for item in results:
        project = item["project"]

        print("Project:", project["project_name"])
        print("Risk:", project["risk_status"])
        print("Hybrid score:", round(item["score"], 3))
        print("BM25 score:", round(item["keyword_score"], 3))
        print("Semantic score:", round(item["semantic_score"], 3))
        print()
