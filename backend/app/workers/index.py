import asyncio
import logging
import uuid
from typing import Any

from sqlalchemy import delete, select

from app.chunking import generate_chunks
from app.core.celery_app import celery_app
from app.database.session import SyncSessionLocal
from app.embeddings import get_embedding_provider
from app.models.core import (
    JobStatus,
    ProcessingJob,
    Recording,
    RecordingStatus,
    SpeakerTurn,
    TranscriptChunk,
    TranscriptSegment,
)

logger = logging.getLogger("sonara.tasks.index")


@celery_app.task(
    bind=True,
    name="sonara.index.build_transcript_index",
    queue="index",
    max_retries=3,
    default_retry_delay=15,
)
def build_transcript_index_task(
    self,
    recording_id: str,
    job_id: str,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """
    Asynchronous Celery task that coordinates:
    1. Deterministic token-aware chunking over persisted transcript segments and speaker turns.
    2. Dense vector embedding generation via SentenceTransformer in bounded batches.
    3. Transactional persistence into PostgreSQL with pgvector and TSVECTOR full-text representations.
    4. Safe atomic activation of index version.
    """
    rec_uuid = uuid.UUID(recording_id)
    job_uuid = uuid.UUID(job_id)
    embedder = get_embedding_provider()

    logger.info(
        f"[Task {self.request.id}] Building semantic & keyword index for recording {recording_id} (Job: {job_id})"
    )

    try:
        # 1. Acquire job and transition recording state to INDEXING
        with SyncSessionLocal() as session:
            job = session.get(ProcessingJob, job_uuid)
            rec = session.get(Recording, rec_uuid)
            if not job or not rec:
                logger.error(f"Job {job_id} or Recording {recording_id} not found.")
                return {"status": "error", "message": "Record not found"}

            job.status = JobStatus.PROCESSING
            job.celery_task_id = self.request.id
            rec.status = RecordingStatus.INDEXING
            session.commit()

            # 2. Fetch transcript segments and speaker turns
            seg_stmt = (
                select(TranscriptSegment)
                .where(TranscriptSegment.recording_id == rec_uuid)
                .order_by(TranscriptSegment.sequence_index.asc())
            )
            segments = list(session.execute(seg_stmt).scalars().all())

            turn_stmt = (
                select(SpeakerTurn)
                .where(SpeakerTurn.recording_id == rec_uuid)
                .order_by(SpeakerTurn.start_ms.asc())
            )
            turns = list(session.execute(turn_stmt).scalars().all())

        if not segments:
            logger.warning(f"Recording {recording_id} has no transcript segments to index.")
            with SyncSessionLocal() as session:
                rec = session.get(Recording, rec_uuid)
                job = session.get(ProcessingJob, job_uuid)
                if rec:
                    rec.status = RecordingStatus.SEARCHABLE
                if job:
                    job.status = JobStatus.COMPLETED
                session.commit()
            return {"status": "success", "chunks_indexed": 0}

        # 3. Generate deterministic chunks with provenance
        chunks = generate_chunks(segments=segments, speaker_turns=turns)
        chunk_texts = [c.text for c in chunks]

        # 4. Generate dense embeddings using provider
        loop = asyncio.new_event_loop()
        try:
            embeddings = loop.run_until_complete(embedder.embed_texts(chunk_texts))
        finally:
            loop.close()

        # 5. Transactionally persist chunks and activate index version
        with SyncSessionLocal() as session:
            # Deactivate or delete old chunks for this recording for idempotent replacement
            session.execute(
                delete(TranscriptChunk).where(TranscriptChunk.recording_id == rec_uuid)
            )

            db_chunks = []
            for idx, c in enumerate(chunks):
                vec = embeddings[idx] if idx < len(embeddings) else None
                chunk_obj = TranscriptChunk(
                    id=uuid.uuid4(),
                    recording_id=rec_uuid,
                    chunk_index=c.chunk_index,
                    start_ms=c.start_ms,
                    end_ms=c.end_ms,
                    text=c.text,
                    token_count=c.token_count,
                    source_segment_ids=c.source_segment_ids,
                    speaker_labels=c.speaker_labels,
                    embedding=vec,
                    embedding_model=embedder.model_name,
                    embedding_dimension=embedder.dimension,
                    transcript_version=1,
                    chunking_version=c.chunking_version,
                    index_version=1,
                    is_active=True,
                )
                db_chunks.append(chunk_obj)

            session.add_all(db_chunks)

            # Update recording state to SEARCHABLE
            rec = session.get(Recording, rec_uuid)
            job = session.get(ProcessingJob, job_uuid)
            if rec:
                rec.status = RecordingStatus.SEARCHABLE
                rec.error_message = None
            if job:
                job.status = JobStatus.COMPLETED
                job.error_message = None

            session.commit()

        logger.info(
            f"[Task {self.request.id}] Successfully built index for recording {recording_id} ({len(chunks)} chunks)."
        )
        return {
            "status": "success",
            "recording_id": recording_id,
            "chunks_count": len(chunks),
        }

    except Exception as exc:
        logger.error(f"Indexing failed for recording {recording_id}: {exc}", exc_info=True)
        if self.request.retries < self.max_retries:
            with SyncSessionLocal() as session:
                job = session.get(ProcessingJob, job_uuid)
                if job:
                    job.retry_count = self.request.retries + 1
                    job.error_message = str(exc)
                    session.commit()
            raise self.retry(exc=exc) from exc
        else:
            with SyncSessionLocal() as session:
                rec = session.get(Recording, rec_uuid)
                job = session.get(ProcessingJob, job_uuid)
                if rec:
                    rec.status = RecordingStatus.FAILED
                    rec.error_message = f"Indexing failed: {exc}"
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(exc)
                session.commit()
            return {"status": "failed", "error": str(exc)}

