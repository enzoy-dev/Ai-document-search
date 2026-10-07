from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import os

from document_processor import process_document
from vector_store import VectorStore
from app.llm import generate_answer

app = FastAPI(title="AI Document Search")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = VectorStore()


@app.get("/")
def root():
    return {"message": "AI Search API is running"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    filename = os.path.basename(file.filename)

    if any(
        document["filename"] == filename
        for document in vector_store.list_documents()
    ):
        raise HTTPException(
            status_code=409,
            detail="Document already exists."
        )

    file_path = UPLOAD_DIR / filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processed = process_document(str(file_path))

        chunks = processed["chunks"]

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain readable text."
            )

        from embedding_service import generate_embeddings

        embeddings = generate_embeddings(chunks)

        vector_store.add(
            chunks,
            embeddings,
            filename
        )

    except HTTPException:
        if file_path.exists():
            file_path.unlink()

        raise

    except Exception:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the document."
        )

    return {
        "filename": filename,
        "total_chunks": len(chunks),
        "message": "Document processed and indexed successfully"
    }

@app.get("/documents")
def list_documents():
    return {
        "documents": vector_store.list_documents()
    }

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    deleted = vector_store.delete_document(filename)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "message": "Document deleted successfully",
        "filename": filename
    }


@app.get("/search")
def search_documents(question: str):
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet."
        )
 
    from embedding_service import generate_embeddings

    query_embedding = generate_embeddings([question])[0]

    results = vector_store.search(
        query_embedding,
        k=3
    )

    relevant_results = [
        result
        for result in results
        if result["distance"] <= 0.95
    ]

    if not relevant_results:
        return {
            "question": question,
            "answer": "No relevant information was found in the indexed documents.",
            "sources": []
        }

    context = "\n\n".join(
        result["text"]
        for result in relevant_results
    )

    answer = generate_answer(
        question=question,
        context=context
    )

    return {
    "question": question,
    "answer": answer,
    "sources": [
        {
            "filename": result["source"],
            "relevance": round(
                1 - result["distance"],
                3
            )
        }
        for result in relevant_results
    ]
}