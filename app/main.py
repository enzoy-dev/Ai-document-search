from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from app import models
from app.database import Base, SessionLocal, engine
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