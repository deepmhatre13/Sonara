import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.retrieval.rerank.interface import RankedCandidate, Reranker

logger = logging.getLogger("sonara.retrieval.rerank")


class SentenceTransformerCrossEncoderReranker(Reranker):
    """
    Cross-Encoder reranking adapter using SentenceTransformers CrossEncoder.
    Defaults to BAAI/bge-reranker-base.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                logger.error("CrossEncoder is not available in sentence_transformers.")
                raise ProviderUnavailableError(
                    provider_name="cross-encoder",
                    detail="CrossEncoder library is required for reranking.",
                ) from e

            logger.info(f"Loading CrossEncoder model '{self._model_name}'...")
            self._model = CrossEncoder(self._model_name)
            logger.info(f"Loaded CrossEncoder '{self._model_name}'.")
        return self._model

    def _rerank_sync(
        self, query: str, candidates: list[dict[str, str]], top_k: int = 10
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        model = self._get_model()
        sentence_pairs = [[query, c["text"]] for c in candidates]
        scores = model.predict(sentence_pairs)

        ranked: list[RankedCandidate] = []
        for idx, c in enumerate(candidates):
            score_val = float(scores[idx]) if hasattr(scores, "__getitem__") else float(scores)
            ranked.append(
                RankedCandidate(
                    chunk_id=c["chunk_id"],
                    score=score_val,
                    text=c["text"],
                )
            )

        # Sort by score descending
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_k]

    async def rerank(
        self, query: str, candidates: list[dict[str, str]], top_k: int = 10
    ) -> list[RankedCandidate]:
        return await asyncio.to_thread(self._rerank_sync, query, candidates, top_k)

