from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from document_processor import process_document
from embedding_service import generate_embeddings, model
from vector_store import VectorStore

app = FastAPI(title="AI Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = VectorStore()


@app.get("/")
def root():
    return {"message": "AI Search API is running"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = process_document(str(file_path))

    embeddings = generate_embeddings(result["chunks"])

    vector_store.add(result["chunks"], embeddings)

    return {
        "filename": file.filename,
        "chunks": result["total_chunks"],
        "status": "indexed"
    }


@app.post("/search")
def search(question: str):
    query_embedding = model.encode(question)

    results = vector_store.search(query_embedding, k=3)

    return {
        "question": question,
        "results": results
    }