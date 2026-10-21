import uuid
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMode(str, Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language or keyword search query")
    mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="Retrieval mode: keyword, semantic, or hybrid")
    recording_id: uuid.UUID | None = Field(default=None, description="Optional filter by recording ID")
    speaker_filter: str | None = Field(default=None, max_length=64, description="Optional filter by anonymous speaker label")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top evidence chunks to return")
    enable_reranking: bool = Field(default=False, description="Whether to apply cross-encoder reranking on top candidates")


class RetrievedChunkEvidence(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    recording_id: uuid.UUID
    text: str
    start_ms: int = Field(..., ge=0, description="Chunk start offset in milliseconds")
    end_ms: int = Field(..., ge=0, description="Chunk end offset in milliseconds")
    token_count: int
    source_segment_ids: list[str]
    speaker_labels: list[str]
    transcript_version: int
    chunking_version: str
    index_version: int

    # Retrieval diagnostic scores
    semantic_rank: int | None = None
    semantic_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: RetrievalMode
    total_results: int
    results: list[RetrievedChunkEvidence]
