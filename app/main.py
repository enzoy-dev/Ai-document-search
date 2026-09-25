from fastapi import FastAPI

app = FastAPI(
    title="AI Search API",
    description="Semantic document search with Python",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "AI Search API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }