from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from app.demo import DemoBudget, EXAMPLE_NAMES
from app.llm import AnswerDraft
from app.vector_store import VectorStore
from tests.test_api import HEADERS, upload


def test_workspace_and_configuration_are_available_without_keys(api, monkeypatch):
    client, main = api
    monkeypatch.setattr(main, "API_KEY", "")
    for path in ["/ui", "/"]:
        response = client.get(path, headers={"Accept": "text/html"})
        assert response.status_code == 200
        assert "Faça uma pergunta" in response.text
        assert "script-src 'self'" in response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/health").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/ui-config").json()["demo_mode"] is False
    assert "unit-test-api-key" not in client.get("/ui-config").text


def test_pdf_download_is_authenticated_and_preserves_source(
    api, monkeypatch, sample_pdf, fake_embeddings
):
    client, main = api
    monkeypatch.setattr(main, "generate_embeddings", fake_embeddings)
    assert upload(client, sample_pdf, "ação com espaço.pdf").status_code == 200
    path = "/documents/ação com espaço.pdf/pdf"
    assert client.get(path).status_code == 401
    response = client.get(path, headers=HEADERS)
    assert response.content == sample_pdf.read_bytes()
    assert response.headers["content-type"] == "application/pdf"
    assert (
        "filename*=UTF-8''a%C3%A7%C3%A3o%20com%20espa%C3%A7o.pdf"
        in response.headers["content-disposition"]
    )
    assert response.headers["cache-control"] == "no-store"
    client.delete("/documents/ação com espaço.pdf", headers=HEADERS)
    assert client.get(path, headers=HEADERS).status_code == 404


@pytest.mark.parametrize("status", ["not_found", "needs_clarification"])
def test_abstention_with_related_chunks_never_claims_evidence(
    api, monkeypatch, sample_pdf, fake_embeddings, status
):
    client, main = api
    monkeypatch.setattr(main, "generate_embeddings", fake_embeddings)
    assert upload(client, sample_pdf).status_code == 200
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda **kwargs: AnswerDraft(
            status=status, answer="Which project do you mean?", language="en"
        ),
    )
    response = client.get(
        "/search", params={"question": "What is its cost?"}, headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json()["status"] == status
    assert response.json()["sources"] == []
    if status == "not_found":
        assert "couldn't find" in response.json()["answer"]
    assert response.json()["timings_ms"]["generation"] >= 0


@pytest.fixture
def demo(api, monkeypatch, tmp_path, fake_embeddings):
    client, main = api
    monkeypatch.setattr(main, "DEMO_MODE", True)
    monkeypatch.setattr(main, "DEMO_ALLOW_GENERATION", False)
    monkeypatch.setattr(main, "demo_store", VectorStore(directory=tmp_path / "demo"))
    monkeypatch.setattr(
        main,
        "demo_budget",
        DemoBudget(
            tmp_path / "usage.sqlite3", requests_per_minute=2, generations_per_day=1
        ),
    )
    monkeypatch.setattr("app.demo.generate_embeddings", fake_embeddings)
    monkeypatch.setattr("app.demo.get_model_identity", lambda: "offline-fixture-v1")
    monkeypatch.setattr(main, "generate_embeddings", fake_embeddings)
    return client, main


def test_public_demo_is_read_only_and_cannot_read_private_documents(
    demo, sample_pdf, fake_embeddings
):
    client, main = demo
    main.vector_store.add(
        [{"text": "Private information", "page": 1}],
        fake_embeddings(["private"]),
        "private.pdf",
        pdf_bytes=sample_pdf.read_bytes(),
        encoder_identity="offline-fixture-v1",
    )
    for path in [
        "/documents",
        "/documents/private.pdf/pdf",
        "/retrieve?question=private",
        "/search?question=private",
    ]:
        assert client.get(path, headers=HEADERS).status_code == 403
    assert upload(client, sample_pdf).status_code == 403
    assert client.delete("/documents/private.pdf", headers=HEADERS).status_code == 403
    assert {
        d["filename"] for d in client.get("/demo/documents").json()["documents"]
    } == set(EXAMPLE_NAMES)
    assert client.get("/demo/documents/private.pdf/pdf").status_code == 404
    assert client.get("/demo/documents/energia.pdf/pdf").content.startswith(b"%PDF")
    assert client.get("/ui-config").json()["generation_enabled"] is False
    assert client.get("/health").json()["indexed_documents"] == 0
    assert (
        "frame-ancestors 'self' https://huggingface.co"
        in client.get("/ui").headers["content-security-policy"]
    )


def test_demo_retrieval_limits_and_generation_disabled(demo, monkeypatch):
    client, main = demo
    generate = Mock(side_effect=AssertionError("Generation must remain disabled"))
    monkeypatch.setattr(main, "generate_answer", generate)
    assert client.get("/demo/search?question=Aurora").status_code == 403
    for _ in range(2):
        response = client.get("/demo/retrieve?question=Aurora")
        assert response.status_code == 200
        assert all(s["filename"] in EXAMPLE_NAMES for s in response.json()["sources"])
    response = client.get("/demo/retrieve?question=Aurora")
    assert response.status_code == 429 and int(response.headers["retry-after"]) > 0
    generate.assert_not_called()


def test_demo_generation_budget_includes_failed_attempts(demo, monkeypatch):
    from app.errors import ServiceUnavailableError

    client, main = demo
    monkeypatch.setattr(main, "DEMO_ALLOW_GENERATION", True)
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-key")
    generate = Mock(side_effect=ServiceUnavailableError("Provider unavailable"))
    monkeypatch.setattr(main, "generate_answer", generate)
    assert client.get("/demo/search?question=Aurora").status_code == 503
    assert client.get("/demo/search?question=Aurora").status_code == 429
    assert generate.call_count == 1


def test_budget_is_persistent_atomic_and_resets_windows(tmp_path):
    path = tmp_path / "budget.sqlite3"
    budget = DemoBudget(
        path, requests_per_minute=100, generations_per_day=3, clock=lambda: 86400
    )

    def attempt(_):
        try:
            budget.reserve(generation=True)
            return True
        except HTTPException:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(12))) == 3
    with pytest.raises(HTTPException):
        DemoBudget(path, 100, 3, clock=lambda: 86460).reserve(generation=True)
    DemoBudget(path, 100, 3, clock=lambda: 172800).reserve(generation=True)


def test_demo_endpoints_are_disabled_in_private_mode(api):
    client, _ = api
    assert client.get("/demo/documents").status_code == 404
    assert client.get("/demo/retrieve?question=Aurora").status_code == 404
