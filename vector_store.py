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
        self.embeddings_path = Path("data/embeddings.npy")

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if (
            self.index_path.exists()
            and self.chunks_path.exists()
            and self.embeddings_path.exists()
        ):
            self.index = faiss.read_index(
                str(self.index_path)
            )

            self.chunks = np.load(
                self.chunks_path,
                allow_pickle=True
            ).tolist()

            self.embeddings = np.load(
                self.embeddings_path
            )

        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            self.embeddings = np.empty(
                (0, dimension),
                dtype="float32"
            )

    def add(
        self,
        chunks: list[str],
        embeddings: list[list[float]]                                                           ,
        source: str
    ):
        vectors = np.array(
            embeddings
        ).astype("float32")

        self.index.add(vectors)

        self.embeddings = np.vstack(
            [self.embeddings, vectors]
        )

        for chunk in chunks:
            self.chunks.append({
                "text": chunk,
                "source": source
            })

        self.save()

    def save(self):
        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        np.save(
            self.chunks_path,
            np.array(
                self.chunks,
                dtype=object
            )
        )

        np.save(
            self.embeddings_path,
            self.embeddings
        )

    def search(
        self,
        query_embedding,
        k: int = 3
    ):
        vector = np.array(
            [query_embedding]
        ).astype("float32")

        k = min(
            k,
            self.index.ntotal
        )

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

            chunk = self.chunks[idx]

            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "distance": float(distance)
            })

        return results

    def list_documents(self):
        documents = {}

        for chunk in self.chunks:
            source = chunk["source"]

            if source not in documents:
                documents[source] = 0

            documents[source] += 1

        return [
            {
                "filename": filename,
                "chunks": chunks
            }
            for filename, chunks in documents.items()
        ]

    def delete_document(self, filename: str):
        indexes_to_keep = [
            i
            for i, chunk in enumerate(self.chunks)
            if chunk["source"] != filename
        ]

        if len(indexes_to_keep) == len(self.chunks):
            return False

        self.chunks = [
            self.chunks[i]
            for i in indexes_to_keep
        ]

        self.embeddings = self.embeddings[
            indexes_to_keep
        ]

        self.index = faiss.IndexFlatL2(
            self.embeddings.shape[1]
        )

        if len(self.embeddings) > 0:
            self.index.add(
                self.embeddings
            )

        self.save()

        return True