import faiss
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(
        self,
        dimension: int = 384,
        index_path: str = "data/faiss.index"
    ):
        self.index_path = Path(index_path)
        self.chunks_path = Path("data/chunks.npy")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.chunks_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = np.load(
                self.chunks_path,
                allow_pickle=True
            ).tolist()
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []

    def add(self, chunks: list[str], embeddings):
        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)
        self.chunks.extend(chunks)

        self.save()

    def save(self):
        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        np.save(
            self.chunks_path,
            np.array(self.chunks, dtype=object)
        )

    def search(self, query_embedding, k: int = 3):
        vector = np.array(
            [query_embedding]
        ).astype("float32")

        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(
            vector,
            k
        )

        results = []

        for idx, distance in zip(
            indices[0],
            distances[0]
        ):
            if idx == -1:
                continue

            results.append({
                "text": self.chunks[idx],
                "distance": float(distance)
            })

        return results