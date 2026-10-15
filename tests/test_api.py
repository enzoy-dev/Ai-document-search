
import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient

from app.config import API_KEY

HEADERS = {"X-API-Key": API_KEY}


# Simula o armazenamento vetorial antes de importar a API.
vector_store_module = types.ModuleType("app.vector_store")


class FakeIndex:
    def __init__(self):
        self.ntotal = 0


class FakeVectorStore:
    def __init__(self):
        self.index = FakeIndex()
        self.documents = [
            {
                "filename": "EnzoSousaDosSantos_agenda4_TI_I.pdf",
                "chunks": 2,
            }
        ]
        self.search_results = []

    def list_documents(self):
        return self.documents

    def delete_document(self, filename):
        for document in self.documents:
            if document["filename"] == filename:
                self.documents.remove(document)
                return True
        return False

    def search(self, query_embedding, k=3):
        return self.search_results

    def add(self, chunks, embeddings, source):
        self.documents.append(
            {
                "filename": source,
                "chunks": len(chunks),
            }
        )
        self.index.ntotal += len(chunks)


vector_store_module.VectorStore = FakeVectorStore
sys.modules["app.vector_store"] = vector_store_module

# Importa o módulo da API, não apenas o objeto FastAPI.
main_module = importlib.import_module("app.main")
app = main_module.app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_vector_store():
    main_module.vector_store = FakeVectorStore()


def install_fake_embeddings(monkeypatch):
    module = types.ModuleType("app.embedding_service")

    module.generate_embeddings = lambda texts: [
        [0.1] * 384 for _ in texts
    ]

    monkeypatch.setitem(
        sys.modules,
        "app.embedding_service",
        module
    )


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "AI Search API is running"
    assert "indexed_documents" in data
    assert "total_chunks" in data


def test_list_documents():
    response = client.get("/documents", headers=HEADERS)

    assert response.status_code == 200
    assert isinstance(response.json()["documents"], list)


def test_upload_non_pdf():
    response = client.post(
        "/documents/upload",
        headers=HEADERS,
        files={
            "file": (
                "teste.txt",
                b"arquivo de teste",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }


def test_duplicate_document():
    response = client.post(
        "/documents/upload",
        headers=HEADERS,
        files={
            "file": (
                "EnzoSousaDosSantos_agenda4_TI_I.pdf",
                b"arquivo de teste",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Document already exists."
    }


def test_delete_nonexistent_document():
    response = client.delete(
        "/documents/documento-que-nao-existe.pdf",
        headers=HEADERS
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found."
    }


def test_search_without_documents():
    main_module.vector_store.index.ntotal = 0

    response = client.get(
        "/search",
        params={"question": "Qual é o objetivo?"},
        headers=HEADERS
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "No documents have been indexed yet."
    }


def test_search_without_relevant_results(monkeypatch):
    install_fake_embeddings(monkeypatch)

    store = main_module.vector_store
    store.index.ntotal = 1
    store.search_results = [
        {
            "text": "Conteúdo sem relação com a pergunta.",
            "source": "documento.pdf",
            "distance": 1.5,
        }
    ]

    response = client.get(
        "/search",
        params={"question": "Receita de bolo?"},
        headers=HEADERS
    )

    assert response.status_code == 200

    data = response.json()
    assert data["sources"] == []
    assert data["answer"] == (
        "No relevant information was found in the indexed documents."
    )


def test_search_with_relevant_results(monkeypatch):
    install_fake_embeddings(monkeypatch)

    store = main_module.vector_store
    store.index.ntotal = 1
    store.search_results = [
        {
            "text": "O sistema pesquisa documentos PDF.",
            "source": "documento.pdf",
            "distance": 0.5,
        }
    ]

    monkeypatch.setattr(
        main_module,
        "generate_answer",
        lambda question, context: "O sistema pesquisa PDFs.",
    )

    response = client.get(
        "/search",
        params={"question": "O que o sistema pesquisa?"},
        headers=HEADERS
    )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "O sistema pesquisa PDFs."
    assert data["sources"] == [
    {
        "filename": "documento.pdf",
        "page": None,          # nos fakes não tem página
        "relevance": 0.667,
        "distance": 0.5,
    }
]


def test_upload_pdf_without_readable_text(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "process_document",
        lambda path: {
            "text": "",
            "chunks": [],
            "total_chunks": 0,
        },
    )

    response = client.post(
        "/documents/upload",
        headers=HEADERS,
        files={
            "file": (
                "pdf-vazio-teste.pdf",
                b"conteudo ficticio",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The PDF does not contain readable text."
    }