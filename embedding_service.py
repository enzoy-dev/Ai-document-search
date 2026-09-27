from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks: list[str]):
    """
    Converts text chunks into numerical vectors.
    """

    embeddings = model.encode(chunks)

    return embeddings