from fastapi import FastAPI

from app.database import Base, engine
from app import models


Base.metadata.create_all(bind=engine)


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