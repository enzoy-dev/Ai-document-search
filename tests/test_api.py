from unittest.mock import Mock

import pytest
from pypdf import PdfWriter

from app.errors import ServiceUnavailableError
from app.llm import AnswerDraft

HEADERS = {'X-API-Key': 'unit-test-api-key'}


def upload(client, sample_pdf, name='sample.pdf'):
    return client.post('/documents/upload', headers=HEADERS,
                       files={'file': (name, sample_pdf.read_bytes(), 'application/pdf')})


def test_health_and_docs_without_keys(api, monkeypatch):
    client, main = api
    monkeypatch.setattr(main, 'API_KEY', '')
    monkeypatch.setattr(main, 'GEMINI_API_KEY', '')
    result = client.get('/')
    assert result.status_code == 200
    assert result.json()['message'] == 'AI Search API is running'
    assert result.json()['gemini_configured'] is False
    assert result.json()['api_key_configured'] is False
    assert client.get('/docs').status_code == 200
    assert client.get('/openapi.json').status_code == 200
    assert client.get('/documents', headers=HEADERS).status_code == 503


@pytest.mark.parametrize('headers', [{}, {'X-API-Key': 'wrong'}])
def test_authentication(api, headers):
    client, _ = api
    assert client.get('/documents', headers=headers).status_code == 401


def test_default_key_is_rejected(api, monkeypatch):
    client, main = api
    monkeypatch.setattr(main, 'API_KEY', 'changeme')
    assert client.get('/documents', headers={'X-API-Key': 'changeme'}).status_code == 503


def test_upload_search_persist_and_delete(api, monkeypatch, sample_pdf, fake_embeddings):
    from app.vector_store import VectorStore
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    answer = Mock(return_value=AnswerDraft(status='answered', answer='The project researches renewable energy. [S1] [S2]'))
    monkeypatch.setattr(main, 'generate_answer', answer)
    assert upload(client, sample_pdf).status_code == 200
    assert client.get('/documents', headers=HEADERS).json()['documents'] == [{'filename': 'sample.pdf', 'chunks': 2}]
    assert VectorStore().index.ntotal == 2  # Reload actual FAISS persistence.
    assert upload(client, sample_pdf).status_code == 409
    result = client.get('/search', headers=HEADERS, params={'question': 'What does Aurora research?'})
    assert result.status_code == 200
    assert 'renewable energy' in result.json()['answer']
    assert {s['page'] for s in result.json()['sources']} == {1, 2}
    assert all(s['filename'] == 'sample.pdf' for s in result.json()['sources'])
    assert 'renewable energy' in answer.call_args.kwargs['context']
    assert client.delete('/documents/sample.pdf', headers=HEADERS).status_code == 200
    assert not (main.UPLOAD_DIR / 'sample.pdf').exists()
    assert VectorStore().index.ntotal == 0
    assert client.delete('/documents/sample.pdf', headers=HEADERS).status_code == 404


def test_empty_search(api):
    client, _ = api
    assert client.get('/search', headers=HEADERS, params={'question': ' '}).status_code == 400
    assert client.get('/search', headers=HEADERS, params={'question': 'Question'}).status_code == 400


def test_no_relevant_results_do_not_call_gemini(api, monkeypatch, sample_pdf, fake_embeddings):
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    assert upload(client, sample_pdf).status_code == 200
    monkeypatch.setattr(main, 'generate_embeddings', lambda texts: [[-1.0] + [0.0] * 383])
    generate = Mock(side_effect=AssertionError('Gemini must not be called'))
    monkeypatch.setattr(main, 'generate_answer', generate)
    result = client.get('/search', headers=HEADERS, params={'question': 'unrelated'})
    assert result.status_code == 200 and result.json()['sources'] == []
    generate.assert_not_called()


def test_missing_gemini_returns_503_only_when_needed(api, monkeypatch, sample_pdf, fake_embeddings):
    from app import llm
    client, main = api
    monkeypatch.setattr(llm, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    assert upload(client, sample_pdf).status_code == 200
    result = client.get('/search', headers=HEADERS, params={'question': 'Aurora?'})
    assert result.status_code == 503 and 'GEMINI_API_KEY' in result.json()['detail']
    assert main.vector_store.index.ntotal == 2


def test_embedding_failure_cleans_upload(api, monkeypatch, sample_pdf):
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', Mock(side_effect=ServiceUnavailableError('Model unavailable')))
    result = upload(client, sample_pdf)
    assert result.status_code == 503
    assert not (main.UPLOAD_DIR / 'sample.pdf').exists()
    assert main.vector_store.index.ntotal == 0


def test_corrupt_and_non_pdf_upload(api):
    client, main = api
    for content_type, content, expected in [('text/plain', b'text', 400), ('application/pdf', b'broken pdf', 400)]:
        result = client.post('/documents/upload', headers=HEADERS, files={'file': ('broken.pdf', content, content_type)})
        assert result.status_code == expected
    assert not (main.UPLOAD_DIR / 'broken.pdf').exists()
    assert main.vector_store.index.ntotal == 0


def test_blank_pdf(api, tmp_path):
    client, _ = api
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    path = tmp_path / 'blank.pdf'
    writer.write(path)
    result = upload(client, path)
    assert result.status_code == 400 and 'readable text' in result.json()['detail']


def test_upload_size_limit(api, monkeypatch, sample_pdf):
    client, main = api
    monkeypatch.setattr(main, 'MAX_UPLOAD_SIZE_MB', 0.00001)
    assert upload(client, sample_pdf).status_code == 413
    assert not (main.UPLOAD_DIR / 'sample.pdf').exists()


def test_retrieval_works_without_gemini(api, monkeypatch, sample_pdf, fake_embeddings):
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    assert upload(client, sample_pdf).status_code == 200
    monkeypatch.setattr(main, 'generate_answer', Mock(side_effect=AssertionError('Must not call Gemini')))
    result = client.get('/retrieve', headers=HEADERS, params={'question': 'Aurora?'})
    assert result.status_code == 200
    sources = result.json()['sources']
    assert [s['citation_id'] for s in sources] == ['S1', 'S2']
    assert 'renewable energy' in sources[0]['text']


@pytest.mark.parametrize('answer', ['Unsupported statement.', 'Claim [S999].'])
def test_unverifiable_answer_is_not_returned(api, monkeypatch, sample_pdf, fake_embeddings, answer):
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    assert upload(client, sample_pdf).status_code == 200
    monkeypatch.setattr(main, 'generate_answer', lambda **kwargs: AnswerDraft(status='answered', answer=answer))
    result = client.get('/search', headers=HEADERS, params={'question': 'Aurora?'})
    assert result.status_code == 503 and 'citations' in result.json()['detail']


def test_sources_only_include_cited_evidence(api, monkeypatch, sample_pdf, fake_embeddings):
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    assert upload(client, sample_pdf).status_code == 200
    monkeypatch.setattr(main, 'generate_answer', lambda **kwargs: AnswerDraft(status='answered', answer='Renewable energy. [S1]'))
    result = client.get('/search', headers=HEADERS, params={'question': 'Aurora?'})
    assert result.status_code == 200
    assert len(result.json()['sources']) == 1
    assert result.json()['sources'][0]['citation_id'] == 'S1'


def test_upload_processing_runs_off_event_loop(api, monkeypatch, sample_pdf, fake_embeddings):
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    original = main.process_document
    def check(path):
        import asyncio
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return original(path)
    monkeypatch.setattr(main, 'process_document', check)
    assert upload(client, sample_pdf).status_code == 200
    assert list(main.UPLOAD_DIR.iterdir()) == []
    assert main.vector_store.get_pdf('sample.pdf') == sample_pdf.read_bytes()


def test_concurrent_api_uploads_do_not_overwrite_or_delete_each_other(api, monkeypatch, sample_pdf, fake_embeddings):
    from concurrent.futures import ThreadPoolExecutor
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(lambda _: upload(client, sample_pdf).status_code, range(2)))
    assert sorted(codes) == [200, 409]
    assert main.vector_store.index.ntotal == 2
    assert main.vector_store.get_pdf('sample.pdf') == sample_pdf.read_bytes()
    assert list(main.UPLOAD_DIR.iterdir()) == []


def test_upload_reads_in_bounded_chunks(api, monkeypatch):
    import io
    from starlette.datastructures import UploadFile, Headers
    from fastapi import HTTPException
    import asyncio
    client, main = api
    monkeypatch.setattr(main, 'MAX_UPLOAD_SIZE_MB', 0.1)
    requested = []
    class BoundedUpload(UploadFile):
        async def read(self, size=-1):
            requested.append(size)
            assert 0 < size <= 64 * 1024
            return await super().read(size)
    file = BoundedUpload(io.BytesIO(b'x' * 200000), filename='large.pdf',
                         headers=Headers({'content-type': 'application/pdf'}))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.upload_document(file))
    assert exc.value.status_code == 413
    assert len(requested) == 2
    assert list(main.UPLOAD_DIR.iterdir()) == []


def test_model_refusal_does_not_claim_sources(api, monkeypatch, sample_pdf, fake_embeddings):
    from app.retrieval import NO_ANSWER
    client, main = api
    monkeypatch.setattr(main, 'generate_embeddings', fake_embeddings)
    assert upload(client, sample_pdf).status_code == 200
    monkeypatch.setattr(main, 'generate_answer', lambda **kwargs: AnswerDraft(status='not_found', answer=NO_ANSWER))
    result = client.get('/search', headers=HEADERS, params={'question': 'Unknown fact?'})
    assert result.status_code == 200 and result.json()['sources'] == []
