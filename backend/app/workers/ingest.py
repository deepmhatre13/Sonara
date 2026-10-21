import logging
import os
import shutil
import tempfile
import uuid
from typing import Any

from app.audio.pipeline import normalize_audio, probe_audio
from app.core.celery_app import celery_app
from app.core.exceptions import AudioValidationError
from app.database.session import SyncSessionLocal
from app.models.core import JobStatus, ProcessingJob, Recording, RecordingStatus
from app.storage import get_audio_storage

logger = logging.getLogger("sonara.tasks.ingest")


@celery_app.task(
    bind=True,
    name="sonara.audio.ingest_pipeline",
    queue="ingest",
    max_retries=3,
    default_retry_delay=10,
)
def ingest_pipeline_task(
    self,
    recording_id: str,
    job_id: str,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """
    Asynchronous Celery worker task that executes media validation, audio probing,
    and normalization. Updates PostgreSQL authoritative state throughout the lifecycle.
    """
    rec_uuid = uuid.UUID(recording_id)
    job_uuid = uuid.UUID(job_id)
    storage = get_audio_storage()
    temp_work_dir = tempfile.mkdtemp(prefix=f"sonara_ingest_{recording_id}_")

    logger.info(
        f"[Task {self.request.id}] Starting ingest pipeline for recording {recording_id} (Job: {job_id})"
    )

    try:
        # 1. Acquire job and transition to PROCESSING state
        with SyncSessionLocal() as session:
            job = session.get(ProcessingJob, job_uuid)
            rec = session.get(Recording, rec_uuid)
            if not job or not rec:
                logger.error(f"Job {job_id} or Recording {recording_id} not found in database.")
                return {"status": "error", "message": "Record not found"}

            if rec.status == RecordingStatus.INGESTION_COMPLETE:
                logger.info(f"Recording {recording_id} already marked INGESTION_COMPLETE. Idempotent exit.")
                return {"status": "success", "message": "Already completed"}

            job.status = JobStatus.PROCESSING
            job.celery_task_id = self.request.id
            rec.status = RecordingStatus.VALIDATING
            session.commit()

        # 2. Download original audio from S3/MinIO to worker temporary directory
        local_raw_path = os.path.join(temp_work_dir, "source_audio")
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(storage.download_file_to_path(rec.audio_s3_key, local_raw_path))
        finally:
            loop.close()

        # 3. Probe and validate media streams
        raw_meta = probe_audio(local_raw_path)
        logger.info(
            f"Probed recording {recording_id}: duration={raw_meta.duration_ms}ms, "
            f"rate={raw_meta.sample_rate}Hz, channels={raw_meta.channels}"
        )

        # Update recording status to NORMALIZING
        with SyncSessionLocal() as session:
            rec = session.get(Recording, rec_uuid)
            if rec:
                rec.status = RecordingStatus.NORMALIZING
                session.commit()

        # 4. Normalize audio into standardized 16kHz mono WAV
        normalized_result = normalize_audio(
            source_path=local_raw_path,
            output_dir=temp_work_dir,
            target_sample_rate=16000,
            target_channels=1,
        )

        # 5. Upload normalized artifact to storage
        normalized_s3_key = f"processed/{recording_id}/{normalized_result.pipeline_version}/audio.wav"
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                storage.upload_file_from_path(
                    normalized_result.output_path,
                    normalized_s3_key,
                    content_type="audio/wav",
                )
            )
        finally:
            loop.close()

        # 6. Finalize database transaction with verified metadata
        with SyncSessionLocal() as session:
            rec = session.get(Recording, rec_uuid)
            job = session.get(ProcessingJob, job_uuid)
            if rec:
                rec.status = RecordingStatus.INGESTION_COMPLETE
                rec.normalized_s3_key = normalized_s3_key
                rec.duration_ms = normalized_result.metadata.duration_ms
                rec.sample_rate = normalized_result.metadata.sample_rate
                rec.channels = normalized_result.metadata.channels
                rec.codec = normalized_result.metadata.codec
                rec.checksum_sha256 = normalized_result.metadata.checksum_sha256
                rec.normalization_profile = normalized_result.profile
                rec.pipeline_version = normalized_result.pipeline_version
                rec.error_message = None

            if job:
                job.status = JobStatus.COMPLETED
            # Enqueue speech pipeline (transcription & diarization)
            speech_job_id = uuid.uuid4()
            speech_job = ProcessingJob(
                id=speech_job_id,
                recording_id=rec_uuid,
                task_name="sonara.speech.transcribe_and_diarize",
                status=JobStatus.PENDING,
            )
            session.add(speech_job)
            session.commit()

        # Dispatch speech task on 'asr' queue
        try:
            from app.workers.speech import transcribe_and_diarize_task

            speech_celery = transcribe_and_diarize_task.apply_async(
                kwargs={"recording_id": str(rec_uuid), "job_id": str(speech_job_id)},
                queue="asr",
            )
            with SyncSessionLocal() as session:
                sj = session.get(ProcessingJob, speech_job_id)
                if sj:
                    sj.celery_task_id = speech_celery.id
                    session.commit()
        except Exception as dispatch_err:
            logger.error(f"Failed to dispatch speech task for recording {recording_id}: {dispatch_err}")

        logger.info(
            f"[Task {self.request.id}] Successfully completed ingest pipeline for recording {recording_id}."
        )
        return {
            "status": "success",
            "recording_id": recording_id,
            "duration_ms": normalized_result.metadata.duration_ms,
        }

    except AudioValidationError as e:
        logger.warning(f"Audio validation failed for recording {recording_id}: {e.detail}")
        with SyncSessionLocal() as session:
            rec = session.get(Recording, rec_uuid)
            job = session.get(ProcessingJob, job_uuid)
            if rec:
                rec.status = RecordingStatus.FAILED
                rec.error_message = e.detail
            if job:
                job.status = JobStatus.FAILED
                job.error_message = e.detail
            session.commit()
        return {"status": "failed", "error": e.detail}

    except Exception as exc:
        logger.error(f"Unexpected error processing recording {recording_id}: {exc}", exc_info=True)
        # Bounded retry on transient system/IO failures
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
                    rec.error_message = "Processing failed after maximum retries."
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(exc)
                session.commit()
            return {"status": "failed", "error": str(exc)}

    finally:
        # Ensure temporary work directory is safely cleaned up
        shutil.rmtree(temp_work_dir, ignore_errors=True)

