import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.database.session import get_db_session
from app.main import app
from app.models.core import JobStatus, ProcessingJob, Recording, RecordingStatus, User


@pytest.fixture
def test_user():
    return User(
        id=uuid.uuid4(),
        email="owner@example.com",
        hashed_password="hashed_pw",
        is_active=True,
        is_superuser=False,
    )


@pytest.fixture
def admin_user():
    return User(
        id=uuid.uuid4(),
        email="admin@example.com",
        hashed_password="admin_pw",
        is_active=True,
        is_superuser=True,
    )


@pytest.mark.asyncio
async def test_delete_recording_success(test_user):
    rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_storage = AsyncMock()

    mock_db.get.return_value = Recording(
        id=rec_id,
        user_id=test_user.id,
        title="To be deleted",
        audio_s3_key="originals/audio.wav",
        normalized_s3_key="processed/audio.wav",
        mime_type="audio/wav",
    )
    mock_storage.check_object_exists.return_value = True
    mock_storage.delete_file.return_value = True

    app.dependency_overrides[get_current_user] = lambda: test_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    with patch("app.api.v1.maintenance.get_audio_storage", return_value=mock_storage):
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.delete(f"/api/v1/maintenance/recordings/{rec_id}")
                assert resp.status_code == 200
                data = resp.json()
                assert data["recording_id"] == str(rec_id)
                assert len(data["deleted_keys"]) == 2
                assert "originals/audio.wav" in data["deleted_keys"]
                assert "processed/audio.wav" in data["deleted_keys"]
                assert mock_db.delete.called
                assert mock_db.commit.called
        finally:
            app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_reconcile_jobs_requires_admin(test_user):
    mock_db = AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: test_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post("/api/v1/maintenance/reconcile")
            assert resp.status_code == 403
            assert "Admin privileges required" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_reconcile_stranded_pending_jobs(admin_user):
    mock_db = AsyncMock()
    job_id = uuid.uuid4()
    rec_id = uuid.uuid4()

    stranded_job = ProcessingJob(
        id=job_id,
        recording_id=rec_id,
        task_name="sonara.audio.ingest_pipeline",
        status=JobStatus.PENDING,
        celery_task_id=None,  # Dispatch failed
    )
    stranded_rec = Recording(
        id=rec_id,
        user_id=admin_user.id,
        title="Unprocessed upload",
        status=RecordingStatus.UPLOADED,
        audio_s3_key="raw/key.wav",
        mime_type="audio/wav",
    )

    from unittest.mock import MagicMock

    mock_exec_result = MagicMock()
    mock_exec_result.scalars.return_value.all.return_value = [stranded_job]

    mock_db.execute.return_value = mock_exec_result
    mock_db.get.return_value = stranded_rec


    app.dependency_overrides[get_current_user] = lambda: admin_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post("/api/v1/maintenance/reconcile")
            assert resp.status_code == 200
            data = resp.json()
            assert data["reconciled_jobs_count"] == 1
            assert stranded_job.status == JobStatus.FAILED
            assert stranded_rec.status == RecordingStatus.FAILED
            assert "stranded" in stranded_job.error_message
    finally:
        app.dependency_overrides.clear()
