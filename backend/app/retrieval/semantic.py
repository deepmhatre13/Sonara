import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import get_embedding_provider
from app.models.core import Recording, TranscriptChunk


@dataclass
class SemanticCandidate:
    chunk_id: uuid.UUID
    recording_id: uuid.UUID
    chunk: TranscriptChunk
    rank: int
    score: float


async def retrieve_semantic(
    db: AsyncSession,
    query: str,
    user_id: uuid.UUID,
    recording_id: uuid.UUID | None = None,
    speaker_filter: str | None = None,
    top_k: int = 20,
) -> list[SemanticCandidate]:
    """
    Performs pgvector dense vector cosine distance search against transcript_chunks.
    Applies user ownership filters strictly at the database level before candidate selection.
    """
    if not query.strip():
        return []

    embedder = get_embedding_provider()
    query_vector = await embedder.embed_query(query)

    # Cosine distance operator in pgvector is <=>
    # Cosine similarity = 1 - cosine_distance
    distance_expr = TranscriptChunk.embedding.cosine_distance(query_vector)

    stmt = (
        select(TranscriptChunk, distance_expr.label("distance"))
        .join(Recording, Recording.id == TranscriptChunk.recording_id)
        .where(
            Recording.user_id == user_id,
            TranscriptChunk.is_active.is_(True),
            TranscriptChunk.embedding.is_not(None),
        )
    )

    if recording_id is not None:
        stmt = stmt.where(TranscriptChunk.recording_id == recording_id)

    if speaker_filter is not None:
        stmt = stmt.where(TranscriptChunk.speaker_labels.contains([speaker_filter]))

    stmt = stmt.order_by(distance_expr.asc(), TranscriptChunk.start_ms.asc()).limit(top_k)

    result = await db.execute(stmt)
    rows = result.all()

    candidates: list[SemanticCandidate] = []
    for idx, (chunk, dist) in enumerate(rows, start=1):
        # Convert distance to similarity score
        similarity = max(0.0, 1.0 - float(dist or 0.0))
        candidates.append(
            SemanticCandidate(
                chunk_id=chunk.id,
                recording_id=chunk.recording_id,
                chunk=chunk,
                rank=idx,
                score=similarity,
            )
        )
    return candidates
