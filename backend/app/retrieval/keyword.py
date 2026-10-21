import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.core import Recording, TranscriptChunk


@dataclass
class KeywordCandidate:
    chunk_id: uuid.UUID
    recording_id: uuid.UUID
    chunk: TranscriptChunk
    rank: int
    score: float


async def retrieve_keyword(
    db: AsyncSession,
    query: str,
    user_id: uuid.UUID,
    recording_id: uuid.UUID | None = None,
    speaker_filter: str | None = None,
    top_k: int = 20,
) -> list[KeywordCandidate]:
    """
    Performs PostgreSQL full-text search against transcript_chunks using to_tsquery / plainto_tsquery.
    Enforces user ownership and optional recording/speaker filters strictly in database query.
    """
    if not query.strip():
        return []

    # Use plainto_tsquery for safe, user-friendly lexical matching without syntax error vulnerabilities
    ts_query = func.plainto_tsquery("english", query)
    rank_col = func.ts_rank(TranscriptChunk.tsv, ts_query).label("rank_score")

    stmt = (
        select(TranscriptChunk, rank_col)
        .join(Recording, Recording.id == TranscriptChunk.recording_id)
        .where(
            Recording.user_id == user_id,
            TranscriptChunk.is_active.is_(True),
            TranscriptChunk.tsv.op("@@")(ts_query),
        )
    )

    if recording_id is not None:
        stmt = stmt.where(TranscriptChunk.recording_id == recording_id)

    if speaker_filter is not None:
        stmt = stmt.where(TranscriptChunk.speaker_labels.contains([speaker_filter]))

    stmt = stmt.order_by(rank_col.desc(), TranscriptChunk.start_ms.asc()).limit(top_k)

    result = await db.execute(stmt)
    rows = result.all()

    candidates: list[KeywordCandidate] = []
    for idx, (chunk, score) in enumerate(rows, start=1):
        candidates.append(
            KeywordCandidate(
                chunk_id=chunk.id,
                recording_id=chunk.recording_id,
                chunk=chunk,
                rank=idx,
                score=float(score or 0.0),
            )
        )
    return candidates
