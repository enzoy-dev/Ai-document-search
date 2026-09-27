from fastapi import Depends, FastAPI, File, UploadFile
from sqlalchemy.orm import Session

from app import models
from app.database import Base, SessionLocal, engine
from app.pdf import extract_text_from_pdf
from app.schemas import DocumentCreate, DocumentResponse


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AI Search API",
    description="Semantic document search with Python",
    version="1.0.0"
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


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


@app.post("/documents", response_model=DocumentResponse)
def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db)
):
    new_document = models.Document(
        filename=document.filename
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    return new_document


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    return db.query(models.Document).all()


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    extracted_text = extract_text_from_pdf(file_path)

    print(extracted_text)

    new_document = models.Document(
        filename=file.filename
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    return new_document