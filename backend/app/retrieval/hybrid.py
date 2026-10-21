import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.core import TranscriptChunk
from app.retrieval.keyword import retrieve_keyword
from app.retrieval.rerank import get_reranker
from app.retrieval.rrf import RankedItem, reciprocal_rank_fusion
from app.retrieval.semantic import retrieve_semantic


@dataclass
class RetrievedEvidence:
    chunk_id: uuid.UUID
    recording_id: uuid.UUID
    text: str
    start_ms: int
    end_ms: int
    token_count: int
    source_segment_ids: list[str]
    speaker_labels: list[str]
    transcript_version: int
    chunking_version: str
    index_version: int
    semantic_rank: int | None = None
    semantic_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


async def retrieve_hybrid(
    db: AsyncSession,
    query: str,
    user_id: uuid.UUID,
    recording_id: uuid.UUID | None = None,
    speaker_filter: str | None = None,
    top_k: int = 10,
    enable_reranking: bool = False,
    rrf_k: int = 60,
) -> list[RetrievedEvidence]:
    """
    Executes hybrid retrieval combining independent semantic and lexical search via RRF.
    Optionally applies cross-encoder reranking on top candidates.
    """
    # 1. Fetch independent candidates (top 25 from each retriever)
    candidate_depth = max(top_k * 2, 25)
    semantic_results = await retrieve_semantic(
        db=db,
        query=query,
        user_id=user_id,
        recording_id=recording_id,
        speaker_filter=speaker_filter,
        top_k=candidate_depth,
    )
    keyword_results = await retrieve_keyword(
        db=db,
        query=query,
        user_id=user_id,
        recording_id=recording_id,
        speaker_filter=speaker_filter,
        top_k=candidate_depth,
    )

    # 2. Build candidate map to avoid redundant chunk queries
    chunk_map: dict[str, TranscriptChunk] = {}
    sem_items: list[RankedItem] = []
    for c in semantic_results:
        cid = str(c.chunk_id)
        chunk_map[cid] = c.chunk
        sem_items.append(RankedItem(id=cid, rank=c.rank, score=c.score, source="semantic"))

    kw_items: list[RankedItem] = []
    for c in keyword_results:
        cid = str(c.chunk_id)
        chunk_map[cid] = c.chunk
        kw_items.append(RankedItem(id=cid, rank=c.rank, score=c.score, source="keyword"))

    # 3. Fuse rankings with Reciprocal Rank Fusion
    ranked_lists = {"semantic": sem_items, "keyword": kw_items}
    fused = reciprocal_rank_fusion(ranked_lists=ranked_lists, k=rrf_k)

    # 4. Assemble candidate evidence objects
    top_fused = fused[: max(top_k * 2, 15)]
    evidence_list: list[RetrievedEvidence] = []

    for f in top_fused:
        chunk = chunk_map[f.id]
        evidence_list.append(
            RetrievedEvidence(
                chunk_id=chunk.id,
                recording_id=chunk.recording_id,
                text=chunk.text,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                token_count=chunk.token_count,
                source_segment_ids=chunk.source_segment_ids,
                speaker_labels=chunk.speaker_labels,
                transcript_version=chunk.transcript_version,
                chunking_version=chunk.chunking_version,
                index_version=chunk.index_version,
                semantic_rank=f.source_ranks.get("semantic"),
                semantic_score=f.source_scores.get("semantic"),
                keyword_rank=f.source_ranks.get("keyword"),
                keyword_score=f.source_scores.get("keyword"),
                rrf_score=f.rrf_score,
            )
        )

    # 5. Optional Cross-Encoder Reranking
    if enable_reranking and evidence_list and settings.ENABLE_RERANKER:
        try:
            reranker = get_reranker()
            rerank_payload = [{"chunk_id": str(e.chunk_id), "text": e.text} for e in evidence_list]
            reranked_candidates = await reranker.rerank(
                query=query, candidates=rerank_payload, top_k=top_k
            )

            reranked_evidence = []
            for rc in reranked_candidates:
                matching = next((e for e in evidence_list if str(e.chunk_id) == rc.chunk_id), None)
                if matching:
                    matching.reranker_score = rc.score
                    reranked_evidence.append(matching)
            return reranked_evidence
        except Exception:
            # Fall back to RRF ordering if reranker model is not initialized/available
            pass

    return evidence_list[:top_k]
