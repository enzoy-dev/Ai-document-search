import os
from pathlib import Path
import tempfile

import numpy as np
import pytest

# Isolate even the store initialized when app.main is imported during collection.
_sandbox = tempfile.TemporaryDirectory(prefix='ai-search-tests-')
os.environ['DATA_DIR'] = str(Path(_sandbox.name) / 'data')
os.environ['UPLOAD_DIR'] = str(Path(_sandbox.name) / 'uploads')


@pytest.fixture
def sample_pdf():
    return Path(__file__).parent / 'fixtures' / 'sample.pdf'


@pytest.fixture
def fake_embeddings():
    # Deterministic vectors for integration wiring, not a semantic quality test.
    def encode(texts):
        return np.tile(np.array([[1.0] + [0.0] * 383], dtype='float32'), (len(texts), 1)).tolist()
    return encode


@pytest.fixture
def api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import main, vector_store
    uploads = tmp_path / 'uploads'
    data = tmp_path / 'data'
    uploads.mkdir()
    data.mkdir()
    monkeypatch.setattr(main, 'UPLOAD_DIR', uploads)
    monkeypatch.setattr(vector_store, 'DATA_DIR', data)
    monkeypatch.setattr(main, 'vector_store', vector_store.VectorStore())
    monkeypatch.setattr(main, 'API_KEY', 'unit-test-api-key')
    monkeypatch.setattr(main, 'get_model_identity', lambda: 'offline-fixture-v1')
    with TestClient(main.app) as client:
        yield client, main


@pytest.fixture(autouse=True)
def forbid_network_in_default_tests(request, monkeypatch):
    if request.node.get_closest_marker('live'):
        return
    import socket
    original_connect = socket.socket.connect
    def connect(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            raise AssertionError('Network access is forbidden in offline tests; use the live marker')
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', connect)
