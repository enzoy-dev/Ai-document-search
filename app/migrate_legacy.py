"""Explicit, validated migration of the old three-file index. Originals stay intact."""
import argparse
import os
from pathlib import Path
import tempfile

import faiss
import numpy as np

from app.config import DATA_DIR, UPLOAD_DIR
from app.vector_store import VectorStore


def migrate(directory, uploads, model, revision='', *, trust_pickle=False):
    directory, uploads = Path(directory), Path(uploads)
    if not trust_pickle:
        raise ValueError('Legacy metadata uses pickle. Only migrate your own trusted files with --trust-legacy-pickle.')
    target = directory / 'documents.sqlite3'
    if target.exists():
        raise ValueError('SQLite database already exists; migration will not overwrite it')
    names = ('faiss.index', 'chunks.npy', 'embeddings.npy')
    if not all((directory / name).is_file() for name in names):
        raise ValueError('Legacy index is incomplete; restore all three files from a consistent backup')
    if not revision:
        raise ValueError('Original model revision is required. If unknown, reindex the original PDFs into a new DATA_DIR.')
    chunks = np.load(directory / 'chunks.npy', allow_pickle=True).tolist()
    embeddings = np.load(directory / 'embeddings.npy', allow_pickle=False)
    old_index = faiss.read_index(str(directory / 'faiss.index'))
    if not isinstance(chunks, list) or embeddings.ndim != 2 or len(chunks) != len(embeddings) or old_index.ntotal != len(chunks):
        raise ValueError('Legacy vector and metadata counts do not match')
    if old_index.d != embeddings.shape[1] or not np.isfinite(embeddings).all():
        raise ValueError('Legacy embeddings are invalid')
    if len(chunks) and not np.allclose(old_index.reconstruct_n(0, len(chunks)), embeddings):
        raise ValueError('Legacy FAISS vectors differ from stored embeddings')
    groups = {}
    for chunk, vector in zip(chunks, embeddings):
        if not isinstance(chunk, dict) or not isinstance(chunk.get('source'), str):
            raise ValueError('Invalid legacy chunk metadata')
        name = chunk['source']
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Invalid legacy filename')
        group = groups.setdefault(name, ([], []))
        group[0].append(chunk)
        group[1].append(vector)
    # Build privately; a failed import never leaves an apparently empty live database.
    with tempfile.TemporaryDirectory(prefix='.migration-', dir=directory) as temporary:
        store = VectorStore(dimension=embeddings.shape[1], directory=temporary, model=model, revision=revision)
        with store._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            for name, (texts, vectors) in groups.items():
                pdf = uploads / name
                store._insert(connection, texts, vectors, name, pdf.read_bytes() if pdf.is_file() else None)
            connection.execute('UPDATE metadata SET revision=revision+1, encoder_identity=? WHERE id=1', (revision,))
        with store._connection() as connection:
            connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        with store.path.open('rb') as database:
            os.fsync(database.fileno())
        # Exclusive publish: do not replace a database created concurrently.
        os.link(store.path, target)
    return len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--uploads-dir', type=Path, default=UPLOAD_DIR)
    parser.add_argument('--legacy-model', required=True, help='Actual model used to create the legacy embeddings')
    parser.add_argument('--legacy-revision', required=True, help='Actual original weight commit/hash; if unknown, reindex instead')
    parser.add_argument('--trust-legacy-pickle', action='store_true')
    args = parser.parse_args()
    try:
        count = migrate(args.data_dir, args.uploads_dir, args.legacy_model, args.legacy_revision,
                        trust_pickle=args.trust_legacy_pickle)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Migration failed: {exc}\n')
    print(f'Migrated {count} documents. Original files were preserved.')


if __name__ == '__main__':
    main()
