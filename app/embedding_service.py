import hashlib
from pathlib import Path
from threading import RLock

from app.config import EMBEDDING_MODEL, EMBEDDING_REVISION
from app.errors import ServiceUnavailableError

_model = None
_model_identity = None
_model_lock = RLock()


def get_model():
    global _model, _model_identity
    with _model_lock:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer
                local = Path(EMBEDDING_MODEL)
                if local.is_dir():
                    digest = hashlib.sha256()
                    files = sorted(p for p in local.rglob('*') if p.is_file())
                    for path in files:
                        digest.update(str(path.relative_to(local)).encode())
                        with path.open('rb') as source:
                            for block in iter(lambda: source.read(1024 * 1024), b''):
                                digest.update(block)
                    identity = 'local-sha256:' + digest.hexdigest()
                    model = SentenceTransformer(EMBEDDING_MODEL)
                else:
                    from huggingface_hub import hf_hub_download
                    # Resolve the ref first, then pin all loaded weights to that exact snapshot.
                    config = Path(hf_hub_download(EMBEDDING_MODEL, 'modules.json', revision=EMBEDDING_REVISION or 'main'))
                    identity = config.parent.name
                    model = SentenceTransformer(EMBEDDING_MODEL, revision=identity)
                _model_identity = identity
                _model = model
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


def get_model_identity():
    get_model()
    return _model_identity
