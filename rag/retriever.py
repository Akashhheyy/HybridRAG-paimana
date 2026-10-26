
import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


# Locate the sample project data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "projects.json"


def load_projects():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def project_to_text(project):
    """Convert a project record into searchable text."""
    return " ".join(
        f"{key.replace('_', ' ')}: {value}"
        for key, value in project.items()
        if value is not None
    )


class HybridRetriever:
    def __init__(self):
        self.projects = load_projects()
        self.documents = [
            project_to_text(project) for project in self.projects
        ]

        # Keyword search
        tokenized_docs = [
            document.lower().split()
            for document in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Semantic search
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embeddings = self.model.encode(
            self.documents,
            convert_to_tensor=True
        )

    def search(self, question, top_k=3):
        # Retrieve by keywords
        tokens = question.lower().split()
        keyword_scores = self.bm25.get_scores(tokens)

        # Retrieve by meaning
        query_embedding = self.model.encode(
            question,
            convert_to_tensor=True
        )
        semantic_scores = util.cos_sim(
            query_embedding,
            self.embeddings
        )[0]

        # Normalize both score sets
        keyword_max = max(keyword_scores) if len(keyword_scores) else 0
        semantic_max = float(semantic_scores.max())

        results = []

        for index, project in enumerate(self.projects):
            keyword_score = (
                float(keyword_scores[index]) / keyword_max
                if keyword_max > 0 else 0
            )
            semantic_score = (
                float(semantic_scores[index]) / semantic_max
                if semantic_max > 0 else 0
            )

            # Combine keyword and semantic scores
            hybrid_score = (
                0.4 * keyword_score + 0.6 * semantic_score
            )

            results.append({
                "project": project,
                "score": hybrid_score
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )
        return results[:top_k]
