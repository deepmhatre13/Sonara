import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from typing import Any

from app.core.celery_app import celery_app
from app.database.session import SyncSessionLocal
from app.diarization import get_diarization_provider
from app.models.core import (
    JobStatus,
    ProcessingJob,
    Recording,
    RecordingStatus,
)
from app.models.core import (
    SpeakerTurn as DbSpeakerTurn,
)
from app.models.core import (
    TranscriptSegment as DbTranscriptSegment,
)
from app.storage import get_audio_storage
from app.transcription import get_transcription_provider

logger = logging.getLogger("sonara.tasks.speech")


@celery_app.task(
    bind=True,
    name="sonara.speech.transcribe_and_diarize",
    queue="asr",
    max_retries=3,
    default_retry_delay=15,
)
def transcribe_and_diarize_task(
    self,
    recording_id: str,
    job_id: str,
    language: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """
    Asynchronous Celery worker task that executes speech transcription and diarization
    using normalized 16kHz mono audio. Persists transcript segments and speaker turns transactionally.
    """
    rec_uuid = uuid.UUID(recording_id)
    job_uuid = uuid.UUID(job_id)
    storage = get_audio_storage()
    transcription_provider = get_transcription_provider()
    diarization_provider = get_diarization_provider()
    temp_work_dir = tempfile.mkdtemp(prefix=f"sonara_speech_{recording_id}_")

    logger.info(
        f"[Task {self.request.id}] Starting speech pipeline for recording {recording_id} (Job: {job_id})"
    )

    try:
        # 1. Acquire job and transition recording to TRANSCRIBING state
        with SyncSessionLocal() as session:
            job = session.get(ProcessingJob, job_uuid)
            rec = session.get(Recording, rec_uuid)
            if not job or not rec:
                logger.error(f"Job {job_id} or Recording {recording_id} not found.")
                return {"status": "error", "message": "Record not found"}

            if not rec.normalized_s3_key:
                logger.error(f"Recording {recording_id} has no normalized audio artifact.")
                rec.status = RecordingStatus.FAILED
                rec.error_message = "No normalized audio artifact available for transcription."
                job.status = JobStatus.FAILED
                job.error_message = rec.error_message
                session.commit()
                return {"status": "failed", "error": rec.error_message}

            job.status = JobStatus.PROCESSING
            job.celery_task_id = self.request.id
            rec.status = RecordingStatus.TRANSCRIBING
            session.commit()

        # 2. Download canonical normalized audio to worker scratch directory
        local_audio_path = os.path.join(temp_work_dir, "normalized.wav")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                storage.download_file_to_path(rec.normalized_s3_key, local_audio_path)
            )
        finally:
            loop.close()

        # 3. Execute transcription
        loop = asyncio.new_event_loop()
        try:
            transcription_result = loop.run_until_complete(
                transcription_provider.transcribe(local_audio_path, language=language)
            )
        finally:
            loop.close()

        # 4. Atomically persist transcript segments (clear previous version if reprocessed)
        with SyncSessionLocal() as session:
            # Delete existing segments for idempotent reprocessing
            from sqlalchemy import delete

            session.execute(
                delete(DbTranscriptSegment).where(DbTranscriptSegment.recording_id == rec_uuid)
            )

            db_segments = [
                DbTranscriptSegment(
                    id=uuid.uuid4(),
                    recording_id=rec_uuid,
                    sequence_index=idx,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=seg.text,
                    language=transcription_result.language,
                    confidence=seg.confidence,
                    model_identifier=transcription_result.model_identifier,
                    transcript_version=1,
                )
                for idx, seg in enumerate(transcription_result.segments)
            ]
            session.add_all(db_segments)

            # Update recording state to DIARIZING
            rec = session.get(Recording, rec_uuid)
            if rec:
                rec.status = RecordingStatus.DIARIZING
            session.commit()

        # 5. Execute speaker diarization (if provider available)
        diarization_performed = False
        if diarization_provider.is_available:
            try:
                loop = asyncio.new_event_loop()
                try:
                    diarization_result = loop.run_until_complete(
                        diarization_provider.diarize(local_audio_path)
                    )
                finally:
                    loop.close()

                with SyncSessionLocal() as session:
                    session.execute(
                        delete(DbSpeakerTurn).where(DbSpeakerTurn.recording_id == rec_uuid)
                    )
                    db_turns = [
                        DbSpeakerTurn(
                            id=uuid.uuid4(),
                            recording_id=rec_uuid,
                            speaker_label=turn.speaker_label,
                            start_ms=turn.start_ms,
                            end_ms=turn.end_ms,
                            confidence=turn.confidence,
                            diarization_model=diarization_result.model_identifier,
                            turn_version=1,
                        )
                        for turn in diarization_result.turns
                    ]
                    session.add_all(db_turns)
                    session.commit()
                diarization_performed = True
            except Exception as diarize_err:
                logger.warning(
                    f"Diarization skipped or encountered non-fatal error: {diarize_err}"
                )

        # 6. Finalize status to SPEECH_COMPLETE and trigger indexing pipeline
        with SyncSessionLocal() as session:
            rec = session.get(Recording, rec_uuid)
            job = session.get(ProcessingJob, job_uuid)
            if rec:
                rec.status = RecordingStatus.SPEECH_COMPLETE
                rec.error_message = None
            if job:
                job.status = JobStatus.COMPLETED
                job.error_message = None

            # Enqueue index task on 'index' queue
            index_job_id = uuid.uuid4()
            index_job = ProcessingJob(
                id=index_job_id,
                recording_id=rec_uuid,
                task_name="sonara.index.build_transcript_index",
                status=JobStatus.PENDING,
            )
            session.add(index_job)
            session.commit()

        # Dispatch indexing task
        try:
            from app.workers.index import build_transcript_index_task

            idx_celery = build_transcript_index_task.apply_async(
                kwargs={"recording_id": str(rec_uuid), "job_id": str(index_job_id)},
                queue="index",
            )
            with SyncSessionLocal() as session:
                ij = session.get(ProcessingJob, index_job_id)
                if ij:
                    ij.celery_task_id = idx_celery.id
                    session.commit()
        except Exception as dispatch_err:
            logger.error(f"Failed to dispatch indexing task for recording {recording_id}: {dispatch_err}")

        logger.info(
            f"[Task {self.request.id}] Successfully completed speech pipeline for recording {recording_id} "
            f"({len(transcription_result.segments)} segments, diarization={diarization_performed})."
        )
        return {
            "status": "success",
            "recording_id": recording_id,
            "segments_count": len(transcription_result.segments),
            "diarization_performed": diarization_performed,
        }

    except Exception as exc:
        logger.error(
            f"Speech processing failed for recording {recording_id}: {exc}", exc_info=True
        )
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
                    rec.error_message = f"Speech processing failed: {exc}"
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(exc)
                session.commit()
            return {"status": "failed", "error": str(exc)}

    finally:
        shutil.rmtree(temp_work_dir, ignore_errors=True)
