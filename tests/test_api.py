from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Search API is running"
    }


def test_list_documents():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_upload_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "teste.txt",
                b"arquivo de teste",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }


def test_duplicate_document():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "EnzoSousaDosSantos_agenda4_TI_I.pdf",
                b"arquivo de teste",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Document already exists."
    }


def test_delete_nonexistent_document():
    response = client.delete(
        "/documents/documento-que-nao-existe.pdf"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found."
    }