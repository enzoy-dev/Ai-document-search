"""SQLite is authoritative; FAISS is a disposable, revisioned search cache."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from threading import RLock

import faiss
import numpy as np

from app.config import DATA_DIR, EMBEDDING_MODEL, EMBEDDING_REVISION


class StorageError(RuntimeError):
    pass


class DuplicateDocumentError(StorageError):
    pass


class ModelMismatchError(StorageError):
    pass


@dataclass(frozen=True)
class Snapshot:
    revision: int
    index: object
    chunks: tuple


class VectorStore:
    def __init__(self, dimension=384, directory=None, model=None, revision=None, *, allow_legacy=False):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer")
        self.directory = Path(directory if directory is not None else DATA_DIR)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'documents.sqlite3'
        self.dimension = dimension
        self.model = model if model is not None else EMBEDDING_MODEL
        self.model_revision = revision if revision is not None else EMBEDDING_REVISION
        self._lock = RLock()
        self._cache = None
        if not self.path.exists() and not allow_legacy and any(
            (self.directory / name).exists() for name in ('faiss.index', 'chunks.npy', 'embeddings.npy')
        ):
            raise StorageError('Legacy index detected. Back up the data and run python -m app.migrate_legacy; no files were overwritten.')
        with self._connection() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('BEGIN IMMEDIATE')
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise StorageError('Unsupported storage version. Restore a compatible database backup.')
            connection.execute('''CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY CHECK(id=1), model TEXT NOT NULL, model_revision TEXT NOT NULL,
                dimension INTEGER NOT NULL, metric TEXT NOT NULL, revision INTEGER NOT NULL, encoder_identity TEXT NOT NULL DEFAULT '')''')
            connection.execute('''CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, filename TEXT NOT NULL UNIQUE, pdf BLOB)''')
            connection.execute('''CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL, page INTEGER, embedding BLOB NOT NULL)''')
            connection.execute('CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id)')
            connection.execute('INSERT OR IGNORE INTO metadata(id, model, model_revision, dimension, metric, revision) VALUES (1, ?, ?, ?, ?, 0)',
                               (self.model, self.model_revision, self.dimension, 'normalized_squared_l2'))
            connection.execute('PRAGMA user_version=1')
            self._validate_identity(connection)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _validate_identity(self, connection):
        identity = connection.execute('SELECT model, model_revision, dimension, metric FROM metadata WHERE id=1').fetchone()
        if identity != (self.model, self.model_revision, self.dimension, 'normalized_squared_l2'):
            raise ModelMismatchError('Embedding model/revision/dimension changed. Use the original configuration or reindex PDFs into a new DATA_DIR.')

    def _check_encoder(self, connection, identity, *, creating=False):
        if identity is None:
            return
        stored = connection.execute('SELECT encoder_identity FROM metadata WHERE id=1').fetchone()[0]
        if not stored:
            if connection.execute('SELECT count(*) FROM chunks').fetchone()[0]:
                raise ModelMismatchError('Stored embeddings have no verified model revision. Reindex the original PDFs into a new DATA_DIR.')
            if creating:
                connection.execute('UPDATE metadata SET encoder_identity=? WHERE id=1', (identity,))
        elif stored != identity:
            raise ModelMismatchError('Loaded embedding weights differ from the index. Restore the original model revision or reindex the PDFs.')

    def _vectors(self, embeddings):
        vectors = np.asarray(embeddings, dtype='float32')
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension or not np.isfinite(vectors).all():
            raise ValueError(f'Embeddings must contain finite vectors of dimension {self.dimension}')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all():
            raise ValueError('Embedding norms must be finite')
        if np.any(norms <= np.finfo('float32').tiny):
            raise ValueError('Embeddings must have nonzero norm')
        return vectors / norms

    def _insert(self, connection, chunks, embeddings, source, pdf_bytes):
        vectors = self._vectors(embeddings)
        if not chunks or len(chunks) != len(vectors):
            raise ValueError('Each nonempty document chunk must have exactly one embedding')
        if not source or Path(source).name != source or source in {'.', '..'}:
            raise ValueError('Invalid document filename')
        if any(not isinstance(chunk.get('text'), str) or not chunk['text'].strip()
               or (chunk.get('page') is not None and (type(chunk['page']) is not int or chunk['page'] <= 0))
               for chunk in chunks):
            raise ValueError('Chunks require readable text and positive page numbers')
        try:
            doc_id = connection.execute('INSERT INTO documents(filename, pdf) VALUES (?, ?)',
                                        (source, pdf_bytes)).lastrowid
        except sqlite3.IntegrityError as exc:
            raise DuplicateDocumentError('Document already exists.') from exc
        connection.executemany('INSERT INTO chunks(document_id, text, page, embedding) VALUES (?, ?, ?, ?)',
                               [(doc_id, chunk['text'], chunk.get('page'), vector.astype('<f4').tobytes())
                                for chunk, vector in zip(chunks, vectors)])

    def add(self, chunks, embeddings, source, pdf_bytes=None, encoder_identity=None):
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._validate_identity(connection)
            self._check_encoder(connection, encoder_identity, creating=True)
            self._insert(connection, chunks, embeddings, source, pdf_bytes)
            connection.execute('UPDATE metadata SET revision=revision+1 WHERE id=1')
        # No shared state is mutated before commit. Other instances observe the revision.

    def snapshot(self, encoder_identity=None):
        with self._lock, self._connection() as connection:
            connection.execute('BEGIN')
            self._validate_identity(connection)
            self._check_encoder(connection, encoder_identity)
            revision = connection.execute('SELECT revision FROM metadata WHERE id=1').fetchone()[0]
            if self._cache is not None and self._cache.revision == revision:
                return self._cache
            rows = connection.execute('''SELECT c.text, c.page, c.embedding, d.filename
                FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.id''').fetchall()
            index = faiss.IndexFlatL2(self.dimension)
            if rows:
                vectors = np.stack([np.frombuffer(row[2], dtype='<f4') for row in rows])
                index.add(self._vectors(vectors))
            chunks = tuple({'text': row[0], 'page': row[1], 'source': row[3]} for row in rows)
            self._cache = Snapshot(revision, index, chunks)
            return self._cache

    @property
    def index(self):
        return self.snapshot().index

    def search(self, query_embedding, k=4, encoder_identity=None):
        if k <= 0:
            raise ValueError('k must be positive')
        vector = self._vectors([query_embedding])
        snapshot = self.snapshot(encoder_identity)
        if not snapshot.chunks:
            return []
        distances, indices = snapshot.index.search(vector, min(k, len(snapshot.chunks)))
        return [{**snapshot.chunks[int(idx)], 'distance': max(0.0, float(distance))}
                for idx, distance in zip(indices[0], distances[0])]

    def list_documents(self):
        with self._connection() as connection:
            return [{'filename': name, 'chunks': count} for name, count in connection.execute('''
                SELECT d.filename, COUNT(c.id) FROM documents d JOIN chunks c ON c.document_id=d.id
                GROUP BY d.id ORDER BY d.id''')]

    def delete_document(self, filename):
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            deleted = connection.execute('DELETE FROM documents WHERE filename=?', (filename,)).rowcount
            if deleted:
                connection.execute('UPDATE metadata SET revision=revision+1 WHERE id=1')
            return bool(deleted)

    def get_pdf(self, filename):
        with self._connection() as connection:
            row = connection.execute('SELECT pdf FROM documents WHERE filename=?', (filename,)).fetchone()
            return row[0] if row else None
