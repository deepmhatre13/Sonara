import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.exceptions import EntityNotFoundError, SonaraException
from app.database.session import get_db_session
from app.models.core import Recording, RecordingStatus, User
from app.retrieval.hybrid import retrieve_hybrid
from app.retrieval.keyword import retrieve_keyword
from app.retrieval.semantic import retrieve_semantic
from app.schemas.search import (
    RetrievalMode,
    RetrievedChunkEvidence,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger("sonara.api.search")

router = APIRouter(prefix="/search", tags=["Hybrid Retrieval & Search"])


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search audio knowledge base",
    description="Executes keyword, semantic, or hybrid search using Reciprocal Rank Fusion and optional cross-encoder reranking.",
)
async def search_recordings(
    payload: SearchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> SearchResponse:
    # 1. Verify recording filter ownership if recording_id is provided
    if payload.recording_id is not None:
        rec = await db.get(Recording, payload.recording_id)
        if not rec:
            raise EntityNotFoundError("Recording", str(payload.recording_id))
        if rec.user_id != current_user.id and not current_user.is_superuser:
            raise SonaraException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to search this recording.",
                code="RECORDING_ACCESS_DENIED",
            )
        if rec.status == RecordingStatus.DELETING:
            raise SonaraException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Recording is currently being deleted and cannot be searched.",
                code="RECORDING_DELETING",
            )


    results: list[RetrievedChunkEvidence] = []

    # 2. Execute requested retrieval strategy
    if payload.mode == RetrievalMode.KEYWORD:
        kw_candidates = await retrieve_keyword(
            db=db,
            query=payload.query,
            user_id=current_user.id,
            recording_id=payload.recording_id,
            speaker_filter=payload.speaker_filter,
            top_k=payload.top_k,
        )
        for c in kw_candidates:
            results.append(
                RetrievedChunkEvidence(
                    chunk_id=c.chunk.id,
                    recording_id=c.chunk.recording_id,
                    text=c.chunk.text,
                    start_ms=c.chunk.start_ms,
                    end_ms=c.chunk.end_ms,
                    token_count=c.chunk.token_count,
                    source_segment_ids=c.chunk.source_segment_ids,
                    speaker_labels=c.chunk.speaker_labels,
                    transcript_version=c.chunk.transcript_version,
                    chunking_version=c.chunking_version,
                    index_version=c.chunk.index_version,
                    keyword_rank=c.rank,
                    keyword_score=c.score,
                )
            )

    elif payload.mode == RetrievalMode.SEMANTIC:
        sem_candidates = await retrieve_semantic(
            db=db,
            query=payload.query,
            user_id=current_user.id,
            recording_id=payload.recording_id,
            speaker_filter=payload.speaker_filter,
            top_k=payload.top_k,
        )
        for c in sem_candidates:
            results.append(
                RetrievedChunkEvidence(
                    chunk_id=c.chunk.id,
                    recording_id=c.chunk.recording_id,
                    text=c.chunk.text,
                    start_ms=c.chunk.start_ms,
                    end_ms=c.chunk.end_ms,
                    token_count=c.chunk.token_count,
                    source_segment_ids=c.chunk.source_segment_ids,
                    speaker_labels=c.chunk.speaker_labels,
                    transcript_version=c.chunk.transcript_version,
                    chunking_version=c.chunking_version,
                    index_version=c.chunk.index_version,
                    semantic_rank=c.rank,
                    semantic_score=c.score,
                )
            )

    else:  # HYBRID
        hybrid_evidence = await retrieve_hybrid(
            db=db,
            query=payload.query,
            user_id=current_user.id,
            recording_id=payload.recording_id,
            speaker_filter=payload.speaker_filter,
            top_k=payload.top_k,
            enable_reranking=payload.enable_reranking,
        )
        for h in hybrid_evidence:
            results.append(
                RetrievedChunkEvidence(
                    chunk_id=h.chunk_id,
                    recording_id=h.recording_id,
                    text=h.text,
                    start_ms=h.start_ms,
                    end_ms=h.end_ms,
                    token_count=h.token_count,
                    source_segment_ids=h.source_segment_ids,
                    speaker_labels=h.speaker_labels,
                    transcript_version=h.transcript_version,
                    chunking_version=h.chunking_version,
                    index_version=h.index_version,
                    semantic_rank=h.semantic_rank,
                    semantic_score=h.semantic_score,
                    keyword_rank=h.keyword_rank,
                    keyword_score=h.keyword_score,
                    rrf_score=h.rrf_score,
                    reranker_score=h.reranker_score,
                )
            )

    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        total_results=len(results),
        results=results,
    )

