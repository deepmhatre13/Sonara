import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.embeddings.interface import EmbeddingProvider

logger = logging.getLogger("sonara.embeddings.sentence_transformer")


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Production-grade dense embedding provider using Sentence Transformers.
    Defaults to BAAI/bge-small-en-v1.5 with output dimension 384.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
    ):
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                logger.error("sentence-transformers is not installed.")
                raise ProviderUnavailableError(
                    provider_name="sentence-transformers",
                    detail="sentence-transformers is required for dense vector embedding generation.",
                ) from e

            logger.info(f"Loading SentenceTransformer model '{self._model_name}'...")
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded SentenceTransformer '{self._model_name}'.")
        return self._model

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()
        # SentenceTransformer encode returns numpy ndarray or torch tensor
        # normalize_embeddings=True calculates unit norm vectors for cosine similarity via dot product
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        results: list[list[float]] = []
        for emb in embeddings:
            vec = [float(x) for x in emb]
            if len(vec) != self._dimension:
                raise ValueError(
                    f"Generated embedding dimension {len(vec)} does not match configured dimension {self._dimension}."
                )
            results.append(vec)
        return results

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_texts_sync, texts)

    async def embed_query(self, query: str) -> list[float]:
        # BGE models recommend query instruction if applicable, but standard encode works seamlessly
        results = await self.embed_texts([query])
        return results[0]

