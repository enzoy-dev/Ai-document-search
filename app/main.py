from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Header,
    Query,
    Request,
)
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import tempfile
import sqlite3
from secrets import compare_digest
from urllib.parse import quote
from time import perf_counter
import threading
from pypdf.errors import PdfReadError
from app.errors import ServiceUnavailableError
import os

from app.document_processor import process_document
from app.vector_store import VectorStore, DuplicateDocumentError, StorageError
from app.llm import generate_answer_draft as generate_answer
from app.config import (
    SIMILARITY_THRESHOLD,
    TOP_K,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_DIR,
    API_KEY,
    GEMINI_API_KEY,
    DATA_DIR,
    DEMO_MODE,
    DEMO_ALLOW_GENERATION,
    DEMO_REQUESTS_PER_MINUTE,
    DEMO_GENERATIONS_PER_DAY,
)
from app.embedding_service import generate_embeddings, get_model_identity
from app.retrieval import (
    evidence_from_results,
    make_context,
    cited_sources,
    NO_ANSWER,
    NO_ANSWER_EN,
)
from app.demo import (
    DemoBudget,
    seed_examples,
    EXAMPLES,
    EXAMPLE_NAMES,
    EXAMPLE_QUESTIONS,
)

app = FastAPI(
    title="AI Document Search",
    description="Semantic document search API with RAG",
    version="1.3.0",
)

vector_store = VectorStore()
demo_store = VectorStore(directory=DATA_DIR / "demo") if DEMO_MODE else None
demo_budget = (
    DemoBudget(
        DATA_DIR / "demo-usage.sqlite3",
        DEMO_REQUESTS_PER_MINUTE,
        DEMO_GENERATIONS_PER_DAY,
    )
    if DEMO_MODE
    else None
)
_demo_store_lock = threading.Lock()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def browser_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if not request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    if request.url.path in {"/", "/ui"} and "text/html" in response.headers.get(
        "content-type", ""
    ):
        ancestors = "'self' https://huggingface.co" if DEMO_MODE else "'none'"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "frame-src blob:; object-src 'none'; base-uri 'none'; "
            f"frame-ancestors {ancestors}"
        )
    return response


@app.exception_handler(sqlite3.Error)
@app.exception_handler(StorageError)
def storage_failure(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Document storage is unavailable. Check disk space, permissions and model configuration."
        },
    )


@app.exception_handler(ServiceUnavailableError)
def service_unavailable(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    if DEMO_MODE:
        raise HTTPException(
            status_code=403,
            detail="A demonstração pública permite apenas os documentos de exemplo.",
        )
    if not API_KEY or API_KEY == "changeme":
        raise HTTPException(
            status_code=503,
            detail="API_KEY is not configured. Set a private API key to enable protected routes.",
        )
    if not x_api_key or not compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key


@app.get("/health")
def health():
    collection = demo_store if DEMO_MODE else vector_store
    return {
        "message": "AI Search API is running",
        "indexed_documents": len(collection.list_documents())
        if collection is not None
        else 0,
        "total_chunks": collection.index.ntotal if collection is not None else 0,
        "api_key_configured": bool(API_KEY and API_KEY != "changeme"),
        "gemini_configured": bool(GEMINI_API_KEY),
        "demo_mode": DEMO_MODE,
    }


@app.get("/", include_in_schema=False)
def root(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return FileResponse(STATIC_DIR / "index.html")
    return health()


@app.get("/ui", include_in_schema=False)
def workspace():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ui-config", include_in_schema=False)
def ui_config():
    return {
        "demo_mode": DEMO_MODE,
        "generation_enabled": bool(
            GEMINI_API_KEY and (not DEMO_MODE or DEMO_ALLOW_GENERATION)
        ),
        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
        "example_questions": list(EXAMPLE_QUESTIONS) if DEMO_MODE else [],
        "requests_per_minute": DEMO_REQUESTS_PER_MINUTE if DEMO_MODE else None,
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
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_DIR, suffix=".pdf", delete=False
        ) as output:
            temporary = Path(output.name)
            while True:
                content = await file.read(64 * 1024)
                if not content:
                    break
                size += len(content)
                if size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed is {MAX_UPLOAD_SIZE_MB}MB.",
                    )
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
        raise HTTPException(
            status_code=400, detail="The PDF does not contain readable text."
        )
    embeddings = generate_embeddings([chunk["text"] for chunk in chunks])
    vector_store.add(
        chunks,
        embeddings,
        filename,
        pdf_bytes=path.read_bytes(),
        encoder_identity=get_model_identity(),
    )
    return len(chunks)


@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents():
    return {"documents": vector_store.list_documents()}


def pdf_response(content, filename):
    if content is None:
        raise HTTPException(
            404,
            "PDF original indisponível. Reenvie o documento se ele veio de uma migração sem o original.",
        )
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename, safe='')}",
        },
    )


@app.get("/documents/{filename}/pdf", dependencies=[Depends(verify_api_key)])
def get_document_pdf(filename: str):
    return pdf_response(vector_store.get_pdf(filename), filename)


@app.delete("/documents/{filename}", dependencies=[Depends(verify_api_key)])
def delete_document(filename: str):
    deleted = vector_store.delete_document(filename)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "message": "Document deleted successfully",
        "filename": filename,
    }


def retrieve(question, store=None):
    store = vector_store if store is None else store
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if store.index.ntotal == 0:
        raise HTTPException(
            status_code=400, detail="No documents have been indexed yet."
        )
    query_embedding = generate_embeddings([question])[0]
    return evidence_from_results(
        store.search(query_embedding, k=TOP_K, encoder_identity=get_model_identity()),
        SIMILARITY_THRESHOLD,
    )


@app.get("/retrieve", dependencies=[Depends(verify_api_key)])
def retrieve_documents(question: str = Query(..., min_length=1, max_length=2000)):
    return {"question": question, "sources": retrieve(question)}


@app.get("/search", dependencies=[Depends(verify_api_key)])
def search_documents(question: str = Query(..., min_length=1, max_length=2000)):
    return answer_question(question)


def answer_question(question, store=None):
    started = perf_counter()
    sources = retrieve(question, store)
    retrieved = perf_counter()
    if not sources:
        return {
            "question": question,
            "status": "not_found",
            "answer": NO_ANSWER,
            "sources": [],
            "timings_ms": {
                "retrieval": round((retrieved - started) * 1000, 2),
                "generation": 0,
            },
        }
    draft = generate_answer(question=question, context=make_context(sources))
    if draft.status == "answered":
        answer = draft.answer
        selected = cited_sources(answer, sources)
        if not selected:
            raise ServiceUnavailableError(
                "The generated answer claims evidence but contains no valid citations."
            )
    else:
        selected = []
        answer = (
            (NO_ANSWER_EN if draft.language == "en" else NO_ANSWER)
            if draft.status == "not_found"
            else draft.answer
        )
    return {
        "question": question,
        "status": draft.status,
        "answer": answer,
        "sources": selected,
        "timings_ms": {
            "retrieval": round((retrieved - started) * 1000, 2),
            "generation": round((perf_counter() - retrieved) * 1000, 2),
        },
    }


def require_demo():
    if not DEMO_MODE:
        raise HTTPException(404, "Demonstração desativada.")


def demo_collection():
    global demo_store
    with _demo_store_lock:
        if demo_store is None:
            demo_store = VectorStore(directory=DATA_DIR / "demo")
    seed_examples(demo_store)
    return demo_store


@app.get("/demo/documents", dependencies=[Depends(require_demo)])
def demo_documents():
    return {"documents": [{"filename": name, "sample": True} for name in EXAMPLE_NAMES]}


@app.get("/demo/documents/{filename}/pdf", dependencies=[Depends(require_demo)])
def demo_pdf(filename: str):
    if filename not in EXAMPLE_NAMES:
        raise HTTPException(404, "Documento de exemplo não encontrado.")
    return pdf_response((EXAMPLES / filename).read_bytes(), filename)


@app.get("/demo/retrieve", dependencies=[Depends(require_demo)])
def demo_retrieve(question: str = Query(..., min_length=1, max_length=2000)):
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    demo_budget.reserve()
    return {"question": question, "sources": retrieve(question, demo_collection())}


@app.get("/demo/search", dependencies=[Depends(require_demo)])
def demo_search(question: str = Query(..., min_length=1, max_length=2000)):
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if not DEMO_ALLOW_GENERATION or not GEMINI_API_KEY:
        raise HTTPException(
            403,
            "Respostas geradas estão desativadas nesta demonstração. Use a busca de trechos.",
        )
    demo_budget.reserve(generation=True)
    return answer_question(question, demo_collection())
