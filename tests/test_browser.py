"""Browser interaction checks; backend integration is covered separately."""

from pathlib import Path
import os
from urllib.parse import urlsplit

import pytest

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser
STATIC = Path(__file__).resolve().parents[1] / "app/static"


@pytest.fixture
def browser_page(sample_pdf):
    state = {
        "demo": True,
        "generation": True,
        "error": None,
        "status": "answered",
        "uploaded": False,
        "headers": [],
        "sources": [
            {
                "citation_id": "S1",
                "filename": "energia.pdf",
                "page": 1,
                "text": "O projeto Aurora funciona em Campinas.",
                "distance": 0.5,
                "relevance": 0.667,
            }
        ],
    }
    errors = []
    with playwright.sync_playwright() as driver:
        executable = os.getenv("CHROMIUM_EXECUTABLE")
        browser = driver.chromium.launch(
            executable_path=executable or None, headless=True
        )
        page = browser.new_page(viewport={"width": 1280, "height": 960})
        page.on("pageerror", lambda error: errors.append(str(error)))

        def handle(route):
            request = route.request
            path = urlsplit(request.url).path
            if path in {"/ui", "/"}:
                route.fulfill(
                    content_type="text/html", body=(STATIC / "index.html").read_text()
                )
            elif path.startswith("/static/"):
                file = STATIC / Path(path).name
                route.fulfill(
                    content_type="text/javascript"
                    if file.suffix == ".js"
                    else "text/css",
                    body=file.read_text(),
                )
            elif path == "/ui-config":
                route.fulfill(
                    json={
                        "demo_mode": state["demo"],
                        "generation_enabled": state["generation"],
                        "max_upload_size_mb": 15,
                        "requests_per_minute": 20,
                        "example_questions": [
                            "Em que cidade funciona o projeto Aurora?"
                        ],
                    }
                )
            elif path in {"/documents", "/demo/documents"}:
                state["headers"].append(request.headers)
                route.fulfill(
                    json={
                        "documents": [{"filename": "energia.pdf"}]
                        if state["demo"] or state["uploaded"]
                        else []
                    }
                )
            elif path.endswith("/pdf"):
                route.fulfill(
                    content_type="application/pdf", body=sample_pdf.read_bytes()
                )
            elif path == "/documents/upload":
                state["uploaded"] = True
                state["headers"].append(request.headers)
                route.fulfill(json={"filename": "energia.pdf", "total_chunks": 2})
            elif path.endswith("/retrieve") or path.endswith("/search"):
                state["headers"].append(request.headers)
                if state["error"]:
                    route.fulfill(status=state["error"], json={"detail": "Unavailable"})
                else:
                    route.fulfill(
                        json={
                            "question": "Aurora?",
                            "status": state["status"],
                            "answer": "Campinas. [S1]"
                            if state["status"] == "answered"
                            else "A qual projeto você se refere?",
                            "sources": state["sources"]
                            if state["status"] == "answered"
                            else [],
                        }
                    )
            else:
                route.fulfill(status=404, body="Not found")

        page.route("**/*", handle)
        yield page, state
        browser.close()
    assert not errors, errors


def ask(page):
    page.get_by_label("O que você quer descobrir?").fill(
        "Em que cidade funciona o projeto Aurora?"
    )
    page.get_by_role("button", name="Pesquisar").click()


def test_demo_search_sources_pdf_and_mobile_layout(browser_page):
    page, state = browser_page
    page.goto("http://workspace.test/ui")
    playwright.expect(page.locator("#connection-form")).to_be_hidden()
    page.locator(".example-button").click()
    page.get_by_role("button", name="Pesquisar").click()
    playwright.expect(page.locator("#result-title")).to_have_text("Trechos encontrados")
    playwright.expect(page.locator(".source-text")).to_contain_text("Campinas")
    page.get_by_role("button", name="Conferir no PDF").click()
    playwright.expect(page.locator("#pdf-panel")).to_be_visible()
    playwright.expect(page.locator("#pdf-viewer")).to_have_attribute(
        "src", __import__("re").compile(r"blob:.*#page=1")
    )
    page.get_by_role("button", name="Fechar documento").click()
    playwright.expect(page.locator("#pdf-panel")).to_be_hidden()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert all("x-api-key" not in headers for headers in state["headers"])


def test_private_upload_authentication_and_no_persisted_key(browser_page, sample_pdf):
    page, state = browser_page
    state["demo"] = False
    page.goto("http://workspace.test/ui")
    ask(page)
    playwright.expect(page.get_by_role("alert")).to_contain_text("Conecte sua coleção")
    page.get_by_label("Chave de acesso").fill("private-browser-key")
    page.get_by_role("button", name="Conectar coleção").click()
    playwright.expect(page.locator("#upload-form")).to_be_visible()
    page.locator("#pdf-file").set_input_files(sample_pdf)
    page.get_by_role("button", name="Enviar e processar").click()
    playwright.expect(
        page.get_by_role("button", name="Abrir energia.pdf")
    ).to_be_visible()
    assert all(
        headers.get("x-api-key") == "private-browser-key"
        for headers in state["headers"]
    )
    assert page.evaluate("localStorage.length + sessionStorage.length") == 0
    assert page.locator("#api-key").input_value() == ""


def test_clarification_and_untrusted_text_are_displayed_as_text(browser_page):
    page, state = browser_page
    state["sources"][0]["text"] = '<img src=x onerror="alert(1)">'
    page.goto("http://workspace.test/ui")
    ask(page)
    playwright.expect(page.locator(".source-text")).to_have_text(
        state["sources"][0]["text"]
    )
    playwright.expect(page.locator("#sources img")).to_have_count(0)
    page.get_by_label("Gerar resposta").check()
    state["status"] = "needs_clarification"
    ask(page)
    playwright.expect(page.locator("#result-title")).to_have_text(
        "Preciso de mais um detalhe"
    )
    playwright.expect(page.locator(".source-card")).to_have_count(0)


def test_rate_limit_message_and_recovery(browser_page):
    page, state = browser_page
    state["error"] = 429
    page.goto("http://workspace.test/ui")
    ask(page)
    playwright.expect(page.get_by_role("alert")).to_contain_text("limite de uso")
    playwright.expect(page.locator("#ask-button")).to_be_enabled()
    state["error"] = None
    ask(page)
    playwright.expect(page.locator("#results")).to_be_visible()
    playwright.expect(page.get_by_role("alert")).to_be_hidden()
