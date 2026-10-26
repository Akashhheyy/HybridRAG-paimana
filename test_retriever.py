
from rag.retriever import HybridRetriever

retriever = HybridRetriever()

question = "Which project has a schedule delay?"
results = retriever.search(question, top_k=3)

print("\nQuestion:", question)
print("\nRelevant projects:\n")

for result in results:
    project = result["project"]
    print("Project:", project["project_name"])
    print("Risk:", project["risk_status"])
    print("Score:", round(result["score"], 3))
    print("-" * 40)
