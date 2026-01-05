from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL

_model = None


def get_embedder():
    """
    Returns a singleton embedding model
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    """
    Backward-compatible helper
    """
    return get_embedder().encode(texts, show_progress_bar=False)
