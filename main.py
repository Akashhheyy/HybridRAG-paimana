
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.retriever import HybridRetriever
from rag.generator import generate_answer

app = FastAPI(
    title="PAIMANA Hybrid RAG API",
    description="AI assistant for infrastructure project risk analysis",
    version="1.0.0",
)

retriever = None


class ChatRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    global retriever
    retriever = HybridRetriever()


@app.get("/")
def home():
    return {
        "message": "PAIMANA Hybrid RAG API is running!",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        results = retriever.search(question, top_k=3)

        answer = generate_answer(question, results)

        sources = [
            {
                "project_id": item["project"]["project_id"],
                "project_name": item["project"]["project_name"],
                "risk_status": item["project"]["risk_status"],
            }
            for item in results
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to generate an answer: {type(error).__name__}",
        )
