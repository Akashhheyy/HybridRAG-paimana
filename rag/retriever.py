
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
    """Convert project data into searchable text with useful synonyms."""
    text = " ".join(
        f"{key.replace('_', ' ')}: {value}"
        for key, value in project.items()
        if value is not None
    )

    planned = project.get("planned_completion")
    revised = project.get("revised_completion")

    extra_terms = []

    if revised:
        extra_terms.extend([
            "revised completion date",
            "schedule delay",
            "project delayed",
            "completion date extended",
            "schedule overrun",
        ])

        if planned and revised > planned:
            extra_terms.extend([
                "delayed beyond planned completion",
                "time overrun",
                "late completion",
            ])

    physical = project.get("physical_progress")
    expenditure = project.get("expenditure")

    if physical is not None and expenditure is not None:
        if expenditure > physical:
            extra_terms.extend([
                "expenditure exceeds physical progress",
                "financial physical progress gap",
                "high spending compared with progress",
            ])

    return text + " " + " ".join(extra_terms)


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
        import re

        # Tokenize the question consistently
        tokens = re.findall(r"\b\w+\b", question.lower())

        # 1. Keyword search using BM25
        keyword_scores = self.bm25.get_scores(tokens)

        keyword_ranking = sorted(
            range(len(self.projects)),
            key=lambda i: keyword_scores[i],
            reverse=True
        )

        # 2. Semantic search
        query_embedding = self.model.encode(
            question,
            convert_to_tensor=True
        )

        semantic_scores = util.cos_sim(
            query_embedding,
            self.embeddings
        )[0]

        semantic_ranking = sorted(
            range(len(self.projects)),
            key=lambda i: float(semantic_scores[i]),
            reverse=True
        )

        # 3. Combine rankings using Reciprocal Rank Fusion
        rrf_scores = {}
        k = 60

        for rank, index in enumerate(keyword_ranking):
            rrf_scores[index] = rrf_scores.get(index, 0) + (
                1 / (k + rank + 1)
            )

        for rank, index in enumerate(semantic_ranking):
            rrf_scores[index] = rrf_scores.get(index, 0) + (
                1 / (k + rank + 1)
            )

        # 4. Return ranked projects and diagnostic scores
        ranked_indices = sorted(
            rrf_scores,
            key=lambda i: rrf_scores[i],
            reverse=True
        )

        # Prioritize projects with an actual revised completion date
        # for schedule-related questions.
        question_lower = question.lower()

        schedule_query = any(term in question_lower for term in [
            "schedule delay",
            "schedule delayed",
            "revised completion date",
            "time overrun",
            "project delayed",
        ])

        if schedule_query:
            ranked_indices.sort(
                key=lambda i: (
                    self.projects[i].get("revised_completion") is not None,
                    rrf_scores[i],
                ),
                reverse=True,
            )

        results = []

        for index in ranked_indices[:top_k]:
            results.append({
                "project": self.projects[index],
                "score": rrf_scores[index],
                "keyword_score": float(keyword_scores[index]),
                "semantic_score": float(semantic_scores[index]),
            })

        return results
