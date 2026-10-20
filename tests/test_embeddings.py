from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app import embedding_service as service
from app.errors import ServiceUnavailableError


def test_generate_embeddings(monkeypatch):
    encode = Mock(return_value=np.ones((2, 384), dtype='float32'))
    monkeypatch.setattr(service, 'get_model', lambda: SimpleNamespace(encode=encode))
    vectors = service.generate_embeddings(['solar energy', 'headquarters'])
    assert len(vectors) == 2 and len(vectors[0]) == 384
    encode.assert_called_once_with(['solar energy', 'headquarters'], show_progress_bar=False)


def test_empty_input_does_not_load_model(monkeypatch):
    get_model = Mock(side_effect=AssertionError('Model must not load'))
    monkeypatch.setattr(service, 'get_model', get_model)
    assert service.generate_embeddings([]) == []
    get_model.assert_not_called()


def test_model_load_error_is_actionable(monkeypatch):
    import sys
    constructor = Mock(side_effect=OSError('private path'))
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *args, **kwargs: '/cache/snapshots/test-revision/modules.json')
    monkeypatch.setattr(service, '_model', None)
    with pytest.raises(ServiceUnavailableError, match='model cache') as exc:
        service.get_model()
    assert 'private path' not in str(exc.value)


def test_encoding_error_is_actionable(monkeypatch):
    monkeypatch.setattr(service, 'get_model', lambda: SimpleNamespace(encode=Mock(side_effect=RuntimeError('internal'))))
    with pytest.raises(ServiceUnavailableError, match='Embedding generation failed'):
        service.generate_embeddings(['text'])
