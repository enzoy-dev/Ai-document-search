import faiss
import numpy as np
from pathlib import Path
from app.config import DATA_DIR


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.index_path = DATA_DIR / "faiss.index"
        self.chunks_path = DATA_DIR / "chunks.npy"
        self.embeddings_path = DATA_DIR / "embeddings.npy"

        if (
            self.index_path.exists()
            and self.chunks_path.exists()
            and self.embeddings_path.exists()
        ):
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = np.load(self.chunks_path, allow_pickle=True).tolist()
            self.embeddings = np.load(self.embeddings_path)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            self.embeddings = np.empty((0, dimension), dtype="float32")

    def add(self, chunks: list[dict], embeddings: list[list[float]], source: str):
        """
        chunks agora é lista de {"text": str, "page": int}
        """
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.embeddings = np.vstack([self.embeddings, vectors])

        for chunk in chunks:
            self.chunks.append({
                "text": chunk["text"],
                "source": source,
                "page": chunk.get("page"),
            })

        self.save()

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.chunks_path, np.array(self.chunks, dtype=object))
        np.save(self.embeddings_path, self.embeddings)

    def search(self, query_embedding, k: int = 4):
        vector = np.array([query_embedding]).astype("float32")
        k = min(k, self.index.ntotal)

        if k == 0:
            return []

        distances, indices = self.index.search(vector, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk.get("page"),
                "distance": float(distance),
            })
        return results

    def list_documents(self):
        documents = {}
        for chunk in self.chunks:
            source = chunk["source"]
            documents[source] = documents.get(source, 0) + 1

        return [
            {"filename": filename, "chunks": count}
            for filename, count in documents.items()
        ]

    def delete_document(self, filename: str) -> bool:
        indexes_to_keep = [
            i for i, chunk in enumerate(self.chunks)
            if chunk["source"] != filename
        ]

        if len(indexes_to_keep) == len(self.chunks):
            return False

        self.chunks = [self.chunks[i] for i in indexes_to_keep]
        self.embeddings = self.embeddings[indexes_to_keep]

        dimension = self.embeddings.shape[1] if len(self.embeddings) > 0 else 384
        self.index = faiss.IndexFlatL2(dimension)

        if len(self.embeddings) > 0:
            self.index.add(self.embeddings)

        self.save()
        return True