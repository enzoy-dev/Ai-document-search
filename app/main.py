from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from pathlib import Path
import os

from app.document_processor import process_document
from app.vector_store import VectorStore
from app.llm import generate_answer
from app.config import (
    SIMILARITY_THRESHOLD,
    TOP_K,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_DIR,
    API_KEY,
)
from app.embedding_service import generate_embeddings

app = FastAPI(
    title="AI Document Search",
    description="Semantic document search API with RAG",
    version="1.2.0",
)

vector_store = VectorStore()


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key


@app.get("/")
def root():
    return {
        "message": "AI Search API is running",
        "indexed_documents": len(vector_store.list_documents()),
        "total_chunks": vector_store.index.ntotal,
    }


@app.post("/documents/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    filename = os.path.basename(file.filename)

    if any(doc["filename"] == filename for doc in vector_store.list_documents()):
        raise HTTPException(status_code=409, detail="Document already exists.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed is {MAX_UPLOAD_SIZE_MB}MB.",
        )

    file_path = UPLOAD_DIR / filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        processed = process_document(str(file_path))
        chunks = processed["chunks"]  # lista de {"text", "page"}

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain readable text.",
            )

        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts)
        vector_store.add(chunks, embeddings, filename)

    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )

    return {
        "filename": filename,
        "total_chunks": len(chunks),
        "message": "Document processed and indexed successfully",
    }


@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents():
    return {"documents": vector_store.list_documents()}


@app.delete("/documents/{filename}", dependencies=[Depends(verify_api_key)])
def delete_document(filename: str):
    deleted = vector_store.delete_document(filename)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()

    return {
        "message": "Document deleted successfully",
        "filename": filename,
    }


@app.get("/search", dependencies=[Depends(verify_api_key)])
def search_documents(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet.",
        )

    query_embedding = generate_embeddings([question])[0]
    results = vector_store.search(query_embedding, k=TOP_K)

    relevant_results = [
        result for result in results
        if result["distance"] <= SIMILARITY_THRESHOLD
    ]

    if not relevant_results:
        return {
            "question": question,
            "answer": "No relevant information was found in the indexed documents.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[Página {r['page']}] {r['text']}" if r.get("page") else r["text"]
        for r in relevant_results
    )

    answer = generate_answer(question=question, context=context)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "filename": result["source"],
                "page": result.get("page"),
                "relevance": round(1 / (1 + result["distance"]), 3),
                "distance": round(result["distance"], 4),
            }
            for result in relevant_results
        ],
    }