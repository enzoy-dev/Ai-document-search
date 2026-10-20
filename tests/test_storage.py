from concurrent.futures import ThreadPoolExecutor
import sqlite3

import faiss
import numpy as np
import pytest

from app.vector_store import VectorStore, DuplicateDocumentError, ModelMismatchError, StorageError
from app.migrate_legacy import migrate

A = [1.0] + [0.0] * 383
B = [0.0, 1.0] + [0.0] * 382
CHUNK = {'text': 'A factual statement.', 'page': 1}


def test_transaction_rollback_does_not_mutate_memory_or_disk(tmp_path, monkeypatch):
    store = VectorStore(directory=tmp_path)
    store.add([CHUNK], [A], 'first.pdf', b'original pdf')
    original_insert = store._insert
    def fail(*args, **kwargs):
        original_insert(*args, **kwargs)
        raise OSError('Simulated failure after all inserts')
    monkeypatch.setattr(store, '_insert', fail)
    with pytest.raises(OSError):
        store.add([CHUNK], [B], 'second.pdf', b'new pdf')
    assert store.index.ntotal == 1
    assert VectorStore(directory=tmp_path).index.ntotal == 1
    assert store.get_pdf('second.pdf') is None
    assert store.get_pdf('first.pdf') == b'original pdf'


def test_failed_delete_rolls_back(tmp_path):
    store = VectorStore(directory=tmp_path)
    store.add([CHUNK], [A], 'first.pdf', b'pdf')
    with store._connection() as connection:
        connection.execute("CREATE TRIGGER fail_delete AFTER DELETE ON chunks BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.delete_document('first.pdf')
    assert store.index.ntotal == 1 and store.get_pdf('first.pdf') == b'pdf'


def test_multiple_instances_refresh_after_add_and_delete(tmp_path):
    first, second = VectorStore(directory=tmp_path), VectorStore(directory=tmp_path)
    assert first.index.ntotal == 0
    second.add([CHUNK], [A], 'first.pdf')
    assert first.search(A)[0]['source'] == 'first.pdf'
    second.delete_document('first.pdf')
    assert first.index.ntotal == 0


def test_simultaneous_duplicate_uploads_commit_only_once(tmp_path):
    def add(_):
        store = VectorStore(directory=tmp_path)
        try:
            store.add([CHUNK], [A], 'same.pdf', b'pdf')
            return 'created'
        except DuplicateDocumentError:
            return 'duplicate'
    VectorStore(directory=tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(add, range(4)))
    assert outcomes.count('created') == 1 and outcomes.count('duplicate') == 3
    assert VectorStore(directory=tmp_path).index.ntotal == 1


def test_simultaneous_distinct_documents_are_preserved(tmp_path):
    VectorStore(directory=tmp_path)
    def add(number):
        VectorStore(directory=tmp_path).add([CHUNK], [A], f'{number}.pdf')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add, range(8)))
    assert len(VectorStore(directory=tmp_path).list_documents()) == 8


@pytest.mark.parametrize('settings', [{'model': 'another-model'}, {'revision': 'different-revision'}, {'dimension': 3}])
def test_changed_embedding_identity_is_rejected(tmp_path, settings):
    VectorStore(directory=tmp_path).add([CHUNK], [A], 'sample.pdf')
    with pytest.raises(ModelMismatchError, match='reindex'):
        VectorStore(directory=tmp_path, **settings)


@pytest.mark.parametrize('vectors', [[[1.0]], [[float('nan')] * 384], [A, B]])
def test_bad_vectors_do_not_create_documents(tmp_path, vectors):
    store = VectorStore(directory=tmp_path)
    with pytest.raises(ValueError):
        store.add([CHUNK], vectors, 'sample.pdf')
    assert store.list_documents() == []


def make_legacy(directory):
    index = faiss.IndexFlatL2(384)
    index.add(np.asarray([A], dtype='float32'))
    faiss.write_index(index, str(directory / 'faiss.index'))
    np.save(directory / 'chunks.npy', np.array([{**CHUNK, 'source': 'sample.pdf'}], dtype=object))
    np.save(directory / 'embeddings.npy', np.asarray([A], dtype='float32'))


def test_legacy_is_never_silently_replaced(tmp_path):
    (tmp_path / 'faiss.index').write_bytes(b'partial index')
    with pytest.raises(StorageError, match='Legacy'):
        VectorStore(directory=tmp_path)
    with pytest.raises(ValueError, match='incomplete'):
        migrate(tmp_path, tmp_path, 'old-model', revision='known-legacy-revision', trust_pickle=True)
    assert not (tmp_path / 'documents.sqlite3').exists()


def test_legacy_migration_preserves_originals_and_can_be_searched(tmp_path):
    make_legacy(tmp_path)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert migrate(tmp_path, tmp_path, 'old-model', revision='known-legacy-revision', trust_pickle=True) == 1
    store = VectorStore(directory=tmp_path, model='old-model', revision='known-legacy-revision')
    assert store.search(A)[0]['source'] == 'sample.pdf'
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())
    with pytest.raises(ValueError, match='already exists'):
        migrate(tmp_path, tmp_path, 'old-model', revision='known-legacy-revision', trust_pickle=True)


def test_corrupt_legacy_migration_never_publishes_empty_database(tmp_path):
    make_legacy(tmp_path)
    np.save(tmp_path / 'embeddings.npy', np.asarray([B], dtype='float32'))
    with pytest.raises(ValueError, match='differ'):
        migrate(tmp_path, tmp_path, 'old-model', revision='known-legacy-revision', trust_pickle=True)
    assert not (tmp_path / 'documents.sqlite3').exists()


def test_pickle_requires_explicit_trust(tmp_path):
    make_legacy(tmp_path)
    with pytest.raises(ValueError, match='trusted'):
        migrate(tmp_path, tmp_path, 'old-model')
    assert not (tmp_path / 'documents.sqlite3').exists()


def test_process_crash_before_commit_preserves_previous_snapshot(tmp_path):
    import subprocess
    import sys
    store = VectorStore(directory=tmp_path)
    store.add([CHUNK], [A], 'original.pdf', b'original')
    code = '''
import os, sqlite3, sys
connection=sqlite3.connect(sys.argv[1])
connection.execute('BEGIN IMMEDIATE')
connection.execute("INSERT INTO documents(filename,pdf) VALUES ('uncommitted.pdf',X'00')")
connection.execute('UPDATE metadata SET revision=revision+1 WHERE id=1')
os._exit(9)
'''
    result = subprocess.run([sys.executable, '-c', code, str(store.path)], timeout=10)
    assert result.returncode == 9
    restarted = VectorStore(directory=tmp_path)
    assert restarted.list_documents() == [{'filename': 'original.pdf', 'chunks': 1}]
    assert restarted.get_pdf('uncommitted.pdf') is None
    assert restarted.search(A)[0]['source'] == 'original.pdf'


def test_loaded_weight_identity_is_checked_even_if_model_name_is_unchanged(tmp_path):
    store = VectorStore(directory=tmp_path)
    store.add([CHUNK], [A], 'first.pdf', encoder_identity='weight-revision-a')
    assert store.search(A, encoder_identity='weight-revision-a')
    with pytest.raises(ModelMismatchError, match='weights differ'):
        store.search(A, encoder_identity='weight-revision-b')
    with pytest.raises(ModelMismatchError):
        store.add([CHUNK], [B], 'second.pdf', encoder_identity='weight-revision-b')
    assert store.index.ntotal == 1


def test_unnormalized_vectors_use_consistent_cosine_distance(tmp_path):
    store = VectorStore(directory=tmp_path)
    store.add([CHUNK], [[value * 10 for value in A]], 'sample.pdf')
    assert store.search([value * 5 for value in A])[0]['distance'] == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError, match='nonzero'):
        store.add([CHUNK], [[0.0] * 384], 'zero.pdf')


def test_stale_reader_keeps_coherent_snapshot_during_concurrent_deletes(tmp_path):
    first = VectorStore(directory=tmp_path)
    first.add([CHUNK], [A], 'original.pdf')
    snapshot = first.snapshot()
    VectorStore(directory=tmp_path).delete_document('original.pdf')
    assert snapshot.index.ntotal == len(snapshot.chunks) == 1
    assert first.search(A) == []
