from app.config import EMBEDDING_MODEL
from app.errors import ServiceUnavailableError

_model = None


def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            raise ServiceUnavailableError(
                "Embedding model could not be loaded. Check EMBEDDING_MODEL, model cache and network access."
            ) from exc
    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        return get_model().encode(texts, show_progress_bar=False).tolist()
    except ServiceUnavailableError:
        raise
    except Exception as exc:
        raise ServiceUnavailableError("Embedding generation failed. Check the model configuration.") from exc
