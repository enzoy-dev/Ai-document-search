from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import tempfile
import sqlite3
from secrets import compare_digest
from pypdf.errors import PdfReadError
from app.errors import ServiceUnavailableError
import os

from app.document_processor import process_document
from app.vector_store import VectorStore, DuplicateDocumentError, StorageError
from app.llm import generate_answer
from app.config import (
    SIMILARITY_THRESHOLD,
    TOP_K,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_DIR,
    API_KEY,
    GEMINI_API_KEY,
)
from app.embedding_service import generate_embeddings, get_model_identity
from app.retrieval import evidence_from_results, make_context, cited_sources, NO_ANSWER

app = FastAPI(
    title="AI Document Search",
    description="Semantic document search API with RAG",
    version="1.2.0",
)

vector_store = VectorStore()


@app.exception_handler(sqlite3.Error)
@app.exception_handler(StorageError)
def storage_failure(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Document storage is unavailable. Check disk space, permissions and model configuration."})


@app.exception_handler(ServiceUnavailableError)
def service_unavailable(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    if not API_KEY or API_KEY == "changeme":
        raise HTTPException(status_code=503, detail="API_KEY is not configured. Set a private API key to enable protected routes.")
    if not x_api_key or not compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key


@app.get("/")
def root():
    return {
        "message": "AI Search API is running",
        "indexed_documents": len(vector_store.list_documents()),
        "total_chunks": vector_store.index.ntotal,
        "api_key_configured": bool(API_KEY and API_KEY != "changeme"),
        "gemini_configured": bool(GEMINI_API_KEY),
    }


@app.post("/documents/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    filename = os.path.basename(file.filename or "")
    if filename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if any(doc["filename"] == filename for doc in vector_store.list_documents()):
        raise HTTPException(status_code=409, detail="Document already exists.")

    temporary = None
    try:
        # Read in bounded chunks. Nothing becomes visible before the SQLite commit.
        size = 0
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=".pdf", delete=False) as output:
            temporary = Path(output.name)
            while True:
                content = await file.read(64 * 1024)
                if not content:
                    break
                size += len(content)
                if size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed is {MAX_UPLOAD_SIZE_MB}MB.")
                await run_in_threadpool(output.write, content)
        chunks_count = await run_in_threadpool(index_upload, temporary, filename)
    except DuplicateDocumentError:
        raise HTTPException(status_code=409, detail="Document already exists.")
    except PdfReadError:
        raise HTTPException(status_code=400, detail="The PDF is invalid or corrupted.")
    except (HTTPException, ServiceUnavailableError, StorageError, sqlite3.Error):
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing document.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        await file.close()

    return {
        "filename": filename,
        "total_chunks": chunks_count,
        "message": "Document processed and indexed successfully",
    }


def index_upload(path, filename):
    processed = process_document(str(path))
    chunks = processed["chunks"]
    if not chunks:
        raise HTTPException(status_code=400, detail="The PDF does not contain readable text.")
    embeddings = generate_embeddings([chunk["text"] for chunk in chunks])
    vector_store.add(chunks, embeddings, filename, pdf_bytes=path.read_bytes(), encoder_identity=get_model_identity())
    return len(chunks)


@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents():
    return {"documents": vector_store.list_documents()}


@app.delete("/documents/{filename}", dependencies=[Depends(verify_api_key)])
def delete_document(filename: str):
    deleted = vector_store.delete_document(filename)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "message": "Document deleted successfully",
        "filename": filename,
    }


def retrieve(question):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet.")
    query_embedding = generate_embeddings([question])[0]
    return evidence_from_results(vector_store.search(query_embedding, k=TOP_K, encoder_identity=get_model_identity()), SIMILARITY_THRESHOLD)


@app.get("/retrieve", dependencies=[Depends(verify_api_key)])
def retrieve_documents(question: str = Query(..., min_length=1, max_length=2000)):
    return {"question": question, "sources": retrieve(question)}


@app.get("/search", dependencies=[Depends(verify_api_key)])
def search_documents(question: str = Query(..., min_length=1, max_length=2000)):
    sources = retrieve(question)
    if not sources:
        return {"question": question, "answer": "No relevant information was found in the indexed documents.", "sources": []}
    answer = generate_answer(question=question, context=make_context(sources))
    return {"question": question, "answer": answer, "sources": cited_sources(answer, sources)}
