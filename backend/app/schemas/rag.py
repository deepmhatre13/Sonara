import uuid
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.search import RetrievalMode


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Question to answer using indexed audio evidence")
    recording_id: uuid.UUID | None = Field(default=None, description="Optional scope filter to a single recording")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of evidence chunks to retrieve and inject")
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="Retrieval strategy: hybrid, semantic, or keyword")
    enable_reranking: bool = Field(default=True, description="Whether to apply cross-encoder reranking on retrieved evidence")


class ValidatedCitation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    evidence_id: str = Field(..., description="Evidence identifier in prompt (e.g. [E1])")
    chunk_id: uuid.UUID
    recording_id: uuid.UUID
    start_ms: int = Field(..., ge=0, description="Authoritative start offset in milliseconds")
    end_ms: int = Field(..., ge=0, description="Authoritative end offset in milliseconds")
    speaker_labels: list[str] = Field(default_factory=list)
    exact_text: str = Field(..., description="Authoritative source chunk transcript excerpt")


class AskResponse(BaseModel):
    question: str
    status: AnswerStatus
    answer: str
    citations: list[ValidatedCitation] = Field(default_factory=list)
    abstention_reason: str | None = None
    provider_name: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None


class StructuredAnswerPayload(BaseModel):
    """
    Internal Pydantic schema requested from LLM providers.
    """
    answer: str = Field(..., description="Factual answer strictly grounded in supplied evidence. If insufficient, state clearly.")
    citations: list[str] = Field(default_factory=list, description="List of evidence labels cited, e.g. ['[E1]', '[E2]']")
    insufficient_evidence: bool = Field(default=False, description="True if evidence does not contain facts needed to answer question")
    abstention_reason: str | None = Field(default=None, description="Explanation when insufficient_evidence is True")

