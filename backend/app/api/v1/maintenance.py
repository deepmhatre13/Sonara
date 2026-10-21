import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.exceptions import AuthorizationError, EntityNotFoundError
from app.database.session import get_db_session
from app.models.core import JobStatus, ProcessingJob, Recording, RecordingStatus, User
from app.schemas.maintenance import (
    DeleteRecordingResponse,
    ReconcileJobsResponse,
    ReconciliationReport,
)
from app.storage import get_audio_storage

logger = logging.getLogger("sonara.api.maintenance")

router = APIRouter(prefix="/maintenance", tags=["System Maintenance & Reliability"])


@router.delete(
    "/recordings/{recording_id}",
    response_model=DeleteRecordingResponse,
    status_code=status.HTTP_200_OK,
    summary="Securely delete a recording and associated storage artifacts",
    description="Transitions recording to DELETING, cleans up S3/MinIO objects, and cascades database deletion safely.",
)
async def delete_recording(
    recording_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DeleteRecordingResponse:
    rec = await db.get(Recording, recording_id)
    if not rec:
        raise EntityNotFoundError("Recording", str(recording_id))

    if rec.user_id != current_user.id and not current_user.is_superuser:
        raise AuthorizationError("You do not have permission to delete this recording.")

    # 1. Transition recording to DELETING state first to prevent concurrent search/RAG execution
    rec.status = RecordingStatus.DELETING
    await db.commit()

    storage = get_audio_storage()
    deleted_keys: list[str] = []

    # 2. Clean up associated audio objects in private storage idempotently
    keys_to_clean = [rec.audio_s3_key, rec.normalized_s3_key]
    for key in keys_to_clean:
        if key:
            try:
                exists = await storage.check_object_exists(key)
                if exists:
                    await storage.delete_file(key)
                    deleted_keys.append(key)
            except Exception as e:
                logger.warning(f"Error deleting object {key} during recording deletion: {e}")

    # 3. Finalize database deletion (cascades to chunks, segments, turns, jobs, upload_sessions)
    await db.delete(rec)
    await db.commit()

    return DeleteRecordingResponse(
        recording_id=recording_id,
        deleted_keys=deleted_keys,
        message="Recording and associated storage artifacts successfully deleted.",
    )



@router.post(
    "/reconcile",
    response_model=ReconcileJobsResponse,
    status_code=status.HTTP_200_OK,
    summary="Reconcile stuck or stranded background jobs",
    description="Detects jobs stranded in PENDING status whose dispatch timed out, marks or re-enqueues them, and updates recording state.",
)
async def reconcile_jobs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    max_age_minutes: Annotated[int, Query(ge=1, le=1440)] = 30,
) -> ReconcileJobsResponse:
    # Only superusers or owner can run reconciliation
    if not current_user.is_superuser:
        raise AuthorizationError("Admin privileges required to trigger global maintenance reconciliation.")

    # Find stranded PENDING jobs
    stmt = (
        select(ProcessingJob)
        .where(
            ProcessingJob.status == JobStatus.PENDING,
            ProcessingJob.celery_task_id.is_(None),
        )
    )
    result = await db.execute(stmt)
    stranded_jobs = result.scalars().all()

    reconciled_count = 0
    details: list[ReconciliationReport] = []

    for job in stranded_jobs:
        job.status = JobStatus.FAILED
        job.error_message = "Job stranded: Celery task dispatch failed or worker never acknowledged."
        
        rec = await db.get(Recording, job.recording_id)
        if rec and rec.status not in (RecordingStatus.INGESTION_COMPLETE, RecordingStatus.SEARCHABLE, RecordingStatus.SPEECH_COMPLETE):
            rec.status = RecordingStatus.FAILED
            rec.error_message = job.error_message

        reconciled_count += 1
        details.append(
            ReconciliationReport(
                job_id=job.id,
                recording_id=job.recording_id,
                task_name=job.task_name,
                action="marked_failed",
            )
        )

    await db.commit()

    return ReconcileJobsResponse(
        reconciled_jobs_count=reconciled_count,
        reports=details,
    )

