from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import llm
from app.errors import ServiceUnavailableError


def test_answer_uses_context_and_configured_model(monkeypatch):
    generate = Mock(return_value=SimpleNamespace(text='  Solar energy.  '))
    monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    assert llm.generate_answer('What energy?', 'Solar panels generate electricity.') == 'Solar energy.'
    params = generate.call_args.kwargs
    assert params['model'] == llm.GEMINI_MODEL
    assert 'What energy?' in params['contents']
    assert 'Solar panels generate electricity.' in params['contents']


def test_missing_key_is_lazy_and_explicit(monkeypatch):
    monkeypatch.setattr(llm, 'GEMINI_API_KEY', '')
    with pytest.raises(ServiceUnavailableError, match='GEMINI_API_KEY'):
        llm.generate_answer('Question', 'Context')


@pytest.mark.parametrize('text', [None, '', '   '])
def test_empty_answer(monkeypatch, text):
    generate = Mock(return_value=SimpleNamespace(text=text))
    monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    with pytest.raises(ServiceUnavailableError, match='empty answer'):
        llm.generate_answer('Question', 'Context')


def test_upstream_failure_does_not_expose_details(monkeypatch):
    generate = Mock(side_effect=RuntimeError('secret provider diagnostics'))
    monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    with pytest.raises(ServiceUnavailableError, match='Gemini request failed') as exc:
        llm.generate_answer('Question', 'Context')
    assert 'secret provider diagnostics' not in str(exc.value)


def test_client_is_cached(monkeypatch):
    constructor = Mock(return_value=object())
    monkeypatch.setattr(llm.genai, 'Client', constructor)
    monkeypatch.setattr(llm, 'GEMINI_API_KEY', 'unit-test-not-a-real-key')
    monkeypatch.setattr(llm, '_client', None)
    assert llm.get_client() is llm.get_client()
    assert constructor.call_count == 1
