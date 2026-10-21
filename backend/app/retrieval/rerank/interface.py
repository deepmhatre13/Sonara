from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class RankedCandidate(BaseModel):
    chunk_id: str
    score: float = Field(..., description="Calibrated cross-encoder score")
    text: str


class Reranker(ABC):
    """Abstract interface for Cross-Encoder reranking."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    async def rerank(
        self, query: str, candidates: list[dict[str, str]], top_k: int = 10
    ) -> list[RankedCandidate]:
        """Reranks candidate documents for a specific query."""
        pass

