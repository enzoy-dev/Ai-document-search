"""Read-only sample collection and persistent, instance-wide usage budgets."""

from pathlib import Path
import sqlite3
import threading
import time

from fastapi import HTTPException

from app.document_processor import process_document
from app.embedding_service import generate_embeddings, get_model_identity

EXAMPLES = Path(__file__).resolve().parents[1] / "evaluations" / "documents"
EXAMPLE_NAMES = ("energia.pdf", "biblioteca.pdf", "transporte.pdf", "museu.pdf")
EXAMPLE_QUESTIONS = (
    "Em que cidade funciona o projeto Aurora?",
    "Qual é o prazo para devolver os livros?",
    "Qual foi o custo total do projeto Aurora?",
)
_seed_lock = threading.Lock()


def seed_examples(store):
    with _seed_lock:
        existing = {row["filename"] for row in store.list_documents()}
        for name in EXAMPLE_NAMES:
            if name in existing:
                continue
            path = EXAMPLES / name
            chunks = process_document(str(path))["chunks"]
            vectors = generate_embeddings([chunk["text"] for chunk in chunks])
            store.add(
                chunks,
                vectors,
                name,
                pdf_bytes=path.read_bytes(),
                encoder_identity=get_model_identity(),
            )


class DemoBudget:
    def __init__(
        self, path, requests_per_minute=20, generations_per_day=20, clock=time.time
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.requests_per_minute = requests_per_minute
        self.generations_per_day = generations_per_day
        self.clock = clock
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS usage (kind TEXT, window INTEGER, count INTEGER, PRIMARY KEY(kind, window))"
            )

    def reserve(self, generation=False):
        now = int(self.clock())
        limits = [("request", now // 60, self.requests_per_minute, 60 - now % 60)]
        if generation:
            limits.append(
                (
                    "generation",
                    now // 86400,
                    self.generations_per_day,
                    86400 - now % 86400,
                )
            )
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for kind, window, maximum, retry in limits:
                row = connection.execute(
                    "SELECT count FROM usage WHERE kind=? AND window=?", (kind, window)
                ).fetchone()
                if row and row[0] >= maximum:
                    raise HTTPException(
                        429,
                        "A demonstração atingiu seu limite de uso. Tente novamente mais tarde.",
                        headers={"Retry-After": str(retry)},
                    )
            for kind, window, _, _ in limits:
                connection.execute(
                    "INSERT INTO usage VALUES (?, ?, 1) ON CONFLICT(kind, window) DO UPDATE SET count=count+1",
                    (kind, window),
                )
            connection.execute(
                "DELETE FROM usage WHERE (kind='request' AND window < ?) OR (kind='generation' AND window < ?)",
                (now // 60, now // 86400),
            )
