from app.retrieval.rerank.cross_encoder import SentenceTransformerCrossEncoderReranker
from app.retrieval.rerank.interface import RankedCandidate, Reranker

_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    """Dependency injection / provider for cross-encoder reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = SentenceTransformerCrossEncoderReranker()
    return _reranker_instance


__all__ = [
    "Reranker",
    "RankedCandidate",
    "SentenceTransformerCrossEncoderReranker",
    "get_reranker",
]

