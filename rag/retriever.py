
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
        # Keyword search
        tokens = question.lower().split()
        keyword_scores = self.bm25.get_scores(tokens)

        # Semantic search
        query_embedding = self.model.encode(
            question,
            convert_to_tensor=True
        )
        semantic_scores = util.cos_sim(
            query_embedding,
            self.embeddings
        )[0]

        results = []

        for index, project in enumerate(self.projects):
            keyword_score = float(keyword_scores[index])
            semantic_score = float(semantic_scores[index])

            # Combine raw BM25 and semantic scores
            # Semantic cosine similarity is typically between -1 and 1.
            # BM25 is unbounded, so normalize keyword scores across projects.
            max_keyword = max(keyword_scores) if len(keyword_scores) else 0

            normalized_keyword = (
                keyword_score / max_keyword
                if max_keyword > 0 else 0
            )

            # Shift cosine similarity to a 0–1 range
            normalized_semantic = (
                max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))
            )

            hybrid_score = (
                0.4 * normalized_keyword
                + 0.6 * normalized_semantic
            )

            results.append({
                "project": project,
                "score": hybrid_score,
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]
