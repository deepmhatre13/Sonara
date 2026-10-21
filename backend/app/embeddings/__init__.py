from app.embeddings.interface import EmbeddingProvider
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

_embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Dependency injection / provider for dense embeddings."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = SentenceTransformerEmbeddingProvider()
    return _embedding_provider


__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "get_embedding_provider",
]

