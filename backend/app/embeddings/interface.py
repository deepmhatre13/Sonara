from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Abstract interface for dense embedding generation.
    Embeddings are strictly versioned by model name and dimension.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model name identifier."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns vector embedding dimension."""
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a batch of text chunks."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generates embedding for a single search query."""
        pass

