import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int = 384):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add(self, chunks: list[str], embeddings):
        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding, k: int = 3):
        vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(vector, k)

        results = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            results.append({
                "text": self.chunks[idx],
                "distance": float(distance)
            })

        return results