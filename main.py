
from fastapi import FastAPI

app = FastAPI(
    title="PAIMANA Hybrid RAG API",
    description="AI assistant for infrastructure project risk analysis",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "PAIMANA Hybrid RAG API is running!",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
