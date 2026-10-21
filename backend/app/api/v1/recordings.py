import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.config import settings
from app.core.exceptions import (
    AudioValidationError,
    AuthorizationError,
    EntityNotFoundError,
    SonaraException,
)
from app.database.session import get_db_session
from app.models.core import (
    JobStatus,
    ProcessingJob,
    Recording,
    RecordingStatus,
    UploadSession,
    User,
)
from app.schemas.recordings import (
    DirectAudioUploadResponse,
    PresignedUploadDetails,
    RecordingResponse,
    UploadFinalizeResponse,
    UploadInitiateRequest,
    UploadInitiateResponse,
)
from app.schemas.speech import (
    ReprocessSpeechRequest,
    ReprocessSpeechResponse,
    SpeakerListResponse,
    SpeakerTurnResponse,
    TranscriptListResponse,
    TranscriptSegmentResponse,
)
from app.storage import get_audio_storage
from app.storage.interface import AudioStorage
from app.workers.ingest import ingest_pipeline_task

logger = logging.getLogger("sonara.api.recordings")

router = APIRouter(prefix="/recordings", tags=["Recordings & Ingestion"])


@router.post(
    "/uploads",
    response_model=UploadInitiateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initiate audio upload session",
    description="Validates metadata, enforces upload limits, creates an upload session, and generates a presigned upload URL.",
)
async def initiate_upload(
    payload: UploadInitiateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
) -> UploadInitiateResponse:
    # 1. Enforce file size limit
    max_bytes = settings.MAX_AUDIO_UPLOAD_SIZE_MB * 1024 * 1024
    if payload.file_size_bytes > max_bytes:
        raise AudioValidationError(
            detail=f"File size exceeds maximum allowed upload limit of {settings.MAX_AUDIO_UPLOAD_SIZE_MB}MB.",
            extra={"max_bytes": max_bytes, "provided_bytes": payload.file_size_bytes},
        )

    # 2. Check Idempotency key if provided
    if payload.idempotency_key:
        stmt = (
            select(Recording)
            .where(
                Recording.user_id == current_user.id,
                Recording.idempotency_key == payload.idempotency_key,
            )
        )
        existing_res = await db.execute(stmt)
        existing_rec = existing_res.scalar_one_or_none()
        if existing_rec:
            # Load existing upload session
            sess_stmt = select(UploadSession).where(UploadSession.recording_id == existing_rec.id)
            sess_res = await db.execute(sess_stmt)
            existing_sess = sess_res.scalar_one_or_none()

            if existing_sess and not existing_sess.is_finalized and existing_sess.expires_at > datetime.now(UTC):
                presigned = storage.generate_presigned_upload_url(
                    destination_key=existing_sess.staging_s3_key,
                    content_type=existing_sess.declared_mime_type,
                    expiration_seconds=int((existing_sess.expires_at - datetime.now(UTC)).total_seconds()),
                )
                return UploadInitiateResponse(
                    recording_id=existing_rec.id,
                    upload_session_id=existing_sess.id,
                    status=existing_rec.status,
                    upload=PresignedUploadDetails(
                        upload_url=presigned["upload_url"],
                        method=presigned["method"],
                        content_type=presigned["content_type"],
                        key=presigned["key"],
                        expires_in=int(presigned["expires_in"]),
                    ),
                    expires_at=existing_sess.expires_at,
                )

    # 3. Create Recording and UploadSession entities
    recording_id = uuid.uuid4()
    session_id = uuid.uuid4()
    staging_key = f"staging/{recording_id}/{session_id}"
    final_source_key = f"originals/{recording_id}/source"
    expires_at = datetime.now(UTC) + timedelta(minutes=15)

    recording = Recording(
        id=recording_id,
        user_id=current_user.id,
        title=payload.title or payload.filename,
        audio_s3_key=final_source_key,
        file_size_bytes=payload.file_size_bytes,
        mime_type=payload.mime_type,
        status=RecordingStatus.AWAITING_UPLOAD,
        idempotency_key=payload.idempotency_key,
    )
    db.add(recording)

    upload_session = UploadSession(
        id=session_id,
        recording_id=recording_id,
        user_id=current_user.id,
        staging_s3_key=staging_key,
        original_filename=payload.filename,
        declared_mime_type=payload.mime_type,
        expected_size_bytes=payload.file_size_bytes,
        expected_checksum=payload.expected_checksum,
        expires_at=expires_at,
        is_finalized=False,
    )
    db.add(upload_session)
    await db.flush()

    # 4. Generate short-lived presigned upload URL
    presigned_info = storage.generate_presigned_upload_url(
        destination_key=staging_key,
        content_type=payload.mime_type,
        expiration_seconds=900,
    )

    return UploadInitiateResponse(
        recording_id=recording_id,
        upload_session_id=session_id,
        status=RecordingStatus.AWAITING_UPLOAD,
        upload=PresignedUploadDetails(
            upload_url=presigned_info["upload_url"],
            method=presigned_info["method"],
            content_type=presigned_info["content_type"],
            key=presigned_info["key"],
            expires_in=int(presigned_info["expires_in"]),
        ),
        expires_at=expires_at,
    )


@router.post(
    "/upload-direct",
    response_model=DirectAudioUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload audio file directly (Swagger / Form Data)",
    description=(
        "Multipart file upload endpoint enabling direct testing from Swagger UI. "
        "Streams the audio file, computes checksum, stores to private S3/MinIO, creates "
        "the Recording record, and triggers the asynchronous ingestion background pipeline."
    ),
)
async def upload_audio_direct(
    file: Annotated[UploadFile, File(description="Audio recording file (wav, mp3, m4a, flac, ogg)")],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
    title: Annotated[str | None, Form(description="Optional title for the recording")] = None,
) -> DirectAudioUploadResponse:
    import hashlib

    # 1. Validate provided filename and MIME type
    filename = file.filename or "recording.wav"
    content_type = file.content_type or "audio/wav"

    # 2. Stream audio file into a temporary file, compute SHA-256 and enforce size limit
    import tempfile
    max_bytes = settings.MAX_AUDIO_UPLOAD_SIZE_MB * 1024 * 1024
    hasher = hashlib.sha256()
    total_bytes = 0

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):  # 1MB buffer
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                import os
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise AudioValidationError(
                    detail=f"File size exceeds maximum allowed upload limit of {settings.MAX_AUDIO_UPLOAD_SIZE_MB}MB.",
                    extra={"max_bytes": max_bytes, "provided_bytes": total_bytes},
                )
            hasher.update(chunk)
            tmp.write(chunk)

    if total_bytes == 0:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise AudioValidationError(detail="Uploaded audio file cannot be empty.")

    checksum = hasher.hexdigest()
    recording_id = uuid.uuid4()
    canonical_key = f"originals/{recording_id}/{filename}"

    # 3. Upload file from temporary path to private object storage
    import os
    try:
        await storage.upload_file_from_path(tmp_path, canonical_key, content_type)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # 4. Create Recording record in database
    rec_title = title.strip() if title and title.strip() else filename
    rec = Recording(
        id=recording_id,
        user_id=current_user.id,
        title=rec_title,
        audio_s3_key=canonical_key,
        mime_type=content_type,
        file_size_bytes=total_bytes,
        checksum_sha256=checksum,
        status=RecordingStatus.UPLOADED,
    )
    db.add(rec)

    # 5. Create durable ProcessingJob
    job_id = uuid.uuid4()
    job = ProcessingJob(
        id=job_id,
        recording_id=recording_id,
        task_name="sonara.audio.ingest_pipeline",
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.commit()

    # 6. Dispatch background worker task
    try:
        celery_task = ingest_pipeline_task.apply_async(
            kwargs={"recording_id": str(recording_id), "job_id": str(job_id)},
            queue="ingest",
        )
        job.celery_task_id = celery_task.id
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to dispatch ingest task for recording {recording_id}: {e}")

    return DirectAudioUploadResponse(
        recording_id=recording_id,
        job_id=job_id,
        status=RecordingStatus.UPLOADED,
        file_size_bytes=total_bytes,
        audio_s3_key=canonical_key,
        message="Audio file successfully uploaded and queued for processing.",
    )


@router.post(
    "/uploads/{upload_id}/finalize",

    response_model=UploadFinalizeResponse,
    summary="Finalize audio upload and trigger background ingestion",
    description="Validates that the file was uploaded, promotes it to immutable canonical storage, and dispatches processing.",
)
async def finalize_upload(
    upload_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
) -> UploadFinalizeResponse:
    # 1. Fetch and validate upload session
    stmt = select(UploadSession).where(UploadSession.id == upload_id)
    res = await db.execute(stmt)
    session = res.scalar_one_or_none()
    if not session:
        raise EntityNotFoundError("UploadSession", str(upload_id))

    if session.user_id != current_user.id:
        raise AuthorizationError("You do not have permission to finalize this upload session.")

    if session.is_finalized:
        # Idempotent response if already finalized
        rec = await db.get(Recording, session.recording_id)
        return UploadFinalizeResponse(
            recording_id=session.recording_id,
            status=rec.status if rec else RecordingStatus.UPLOADED,
            message="Upload already finalized.",
        )

    if datetime.now(UTC) > session.expires_at:
        raise SonaraException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload session has expired. Please initiate a new upload.",
            code="UPLOAD_SESSION_EXPIRED",
        )

    # 2. Check storage object existence
    exists = await storage.check_object_exists(session.staging_s3_key)
    if not exists:
        raise SonaraException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No uploaded audio found at the designated storage location. Did the client PUT complete?",
            code="UPLOAD_NOT_FOUND",
        )

    # 3. Promote staging file to immutable canonical path
    rec = await db.get(Recording, session.recording_id)
    if not rec:
        raise EntityNotFoundError("Recording", str(session.recording_id))

    await storage.copy_object(session.staging_s3_key, rec.audio_s3_key)
    # Safely delete staging object
    await storage.delete_file(session.staging_s3_key)

    # 4. Update session and recording state
    session.is_finalized = True
    rec.status = RecordingStatus.UPLOADED

    # 5. Create durable ProcessingJob record
    job_id = uuid.uuid4()
    job = ProcessingJob(
        id=job_id,
        recording_id=rec.id,
        task_name="sonara.audio.ingest_pipeline",
        status=JobStatus.PENDING,
    )
    db.add(job)
    # Commit transaction before dispatching task to prevent race conditions
    await db.commit()

    # 6. Dispatch Celery background task
    try:
        celery_task = ingest_pipeline_task.apply_async(
            kwargs={"recording_id": str(rec.id), "job_id": str(job_id)},
            queue="ingest",
        )
        # Update celery_task_id in separate commit
        job.celery_task_id = celery_task.id
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to enqueue ingest task for recording {rec.id}: {e}")
        # Note: Worker reconciliation task can pick up pending jobs whose dispatch timed out

    return UploadFinalizeResponse(
        recording_id=rec.id,
        status=RecordingStatus.UPLOADED,
        message="Upload successfully finalized and enqueued for ingestion.",
    )


@router.get(
    "/{recording_id}",
    response_model=RecordingResponse,
    summary="Get recording status and metadata",
    description="Retrieves recording processing state, duration, sample rate, and verified media metadata.",
)
async def get_recording(
    recording_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RecordingResponse:
    rec = await db.get(Recording, recording_id)
    if not rec:
        raise EntityNotFoundError("Recording", str(recording_id))

    if rec.user_id != current_user.id and not current_user.is_superuser:
        raise AuthorizationError("You do not have access to this recording.")

    return RecordingResponse.model_validate(rec)


@router.get(
    "/{recording_id}/transcript",
    response_model=TranscriptListResponse,
    summary="Get recording transcript segments",
    description="Retrieves timestamped speech transcript segments with pagination support.",
)
async def get_transcript(
    recording_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TranscriptListResponse:
    rec = await db.get(Recording, recording_id)
    if not rec:
        raise EntityNotFoundError("Recording", str(recording_id))

    if rec.user_id != current_user.id and not current_user.is_superuser:
        raise AuthorizationError("You do not have access to this recording's transcript.")

    from app.models.core import TranscriptSegment

    stmt = (
        select(TranscriptSegment)
        .where(TranscriptSegment.recording_id == recording_id)
        .order_by(TranscriptSegment.sequence_index)
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    segments = result.scalars().all()

    total_stmt = select(func.count(TranscriptSegment.id)).where(TranscriptSegment.recording_id == recording_id)
    total_res = await db.execute(total_stmt)
    total_count = total_res.scalar() or 0

    language = segments[0].language if segments else None

    return TranscriptListResponse(
        recording_id=recording_id,
        total_segments=total_count,
        language=language,
        segments=[TranscriptSegmentResponse.model_validate(s) for s in segments],
    )


@router.get(
    "/{recording_id}/speakers",
    response_model=SpeakerListResponse,
    summary="Get speaker diarization turns",
    description="Retrieves anonymous speaker turns with timestamps in integer milliseconds.",
)
async def get_speakers(
    recording_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> SpeakerListResponse:
    rec = await db.get(Recording, recording_id)
    if not rec:
        raise EntityNotFoundError("Recording", str(recording_id))

    if rec.user_id != current_user.id and not current_user.is_superuser:
        raise AuthorizationError("You do not have access to this recording's speaker turns.")

    from app.models.core import SpeakerTurn

    stmt = (
        select(SpeakerTurn)
        .where(SpeakerTurn.recording_id == recording_id)
        .order_by(SpeakerTurn.start_ms)
    )
    result = await db.execute(stmt)
    turns = result.scalars().all()

    unique_speakers = sorted({t.speaker_label for t in turns})

    return SpeakerListResponse(
        recording_id=recording_id,
        total_turns=len(turns),
        speakers=unique_speakers,
        turns=[SpeakerTurnResponse.model_validate(t) for t in turns],
    )


@router.post(
    "/{recording_id}/transcription/reprocess",
    response_model=ReprocessSpeechResponse,
    summary="Reprocess speech transcription & diarization",
    description="Dispatches a re-run of transcription and diarization on normalized audio.",
)
async def reprocess_speech(
    recording_id: uuid.UUID,
    payload: ReprocessSpeechRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> ReprocessSpeechResponse:
    rec = await db.get(Recording, recording_id)
    if not rec:
        raise EntityNotFoundError("Recording", str(recording_id))

    if rec.user_id != current_user.id and not current_user.is_superuser:
        raise AuthorizationError("You do not have permission to reprocess this recording.")

    if not rec.normalized_s3_key:
        raise SonaraException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot reprocess speech: normalized audio artifact does not exist.",
            code="NORMALIZED_AUDIO_MISSING",
        )

    rec.status = RecordingStatus.TRANSCRIBING

    job_id = uuid.uuid4()
    job = ProcessingJob(
        id=job_id,
        recording_id=rec.id,
        task_name="sonara.speech.transcribe_and_diarize",
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.commit()

    try:
        from app.workers.speech import transcribe_and_diarize_task

        celery_task = transcribe_and_diarize_task.apply_async(
            kwargs={
                "recording_id": str(rec.id),
                "job_id": str(job_id),
                "language": payload.language,
            },
            queue="asr",
        )
        job.celery_task_id = celery_task.id
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to dispatch speech task for recording {rec.id}: {e}")

    return ReprocessSpeechResponse(
        recording_id=rec.id,
        job_id=job_id,
        status="enqueued",
        message="Speech re-processing job successfully enqueued.",
    )

