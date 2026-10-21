import io
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.database.session import get_db_session
from app.main import app
from app.models.core import RecordingStatus, User


@pytest.fixture
def swagger_user():
    return User(
        id=uuid.uuid4(),
        email="swagger_tester@example.com",
        hashed_password="pw",
        is_active=True,
    )


@pytest.mark.asyncio
async def test_upload_audio_direct_unauthenticated():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/recordings/upload-direct",
            files={"file": ("test.wav", io.BytesIO(b"RIFF dummy wav header bytes"), "audio/wav")},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_upload_audio_direct_empty_file_rejected(swagger_user):
    mock_db = AsyncMock()

    app.dependency_overrides[get_current_user] = lambda: swagger_user
    app.dependency_overrides[get_db_session] = lambda: mock_db


    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/recordings/upload-direct",
                files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
            )
            assert resp.status_code == 400
            assert "cannot be empty" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_audio_direct_success(swagger_user):
    from app.storage import get_audio_storage

    mock_db = AsyncMock()
    mock_storage = AsyncMock()
    mock_storage.upload_file_from_path.return_value = "originals/some-id/sample.wav"

    app.dependency_overrides[get_current_user] = lambda: swagger_user
    app.dependency_overrides[get_db_session] = lambda: mock_db
    app.dependency_overrides[get_audio_storage] = lambda: mock_storage

    sample_audio_bytes = b"RIFF" + b"\x00" * 36 + b"WAVEfmt " + b"\x10\x00\x00\x00" + b"\x01\x00\x01\x00"

    with patch("app.api.v1.recordings.ingest_pipeline_task.apply_async") as mock_celery:
        mock_celery.return_value = AsyncMock(id="celery-job-123")
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/api/v1/recordings/upload-direct",
                    files={"file": ("sample.wav", io.BytesIO(sample_audio_bytes), "audio/wav")},
                    data={"title": "Team Daily Standup Audio"},
                )
                assert resp.status_code == 201
                data = resp.json()
                assert data["status"] == RecordingStatus.UPLOADED
                assert data["recording_id"] is not None
                assert data["job_id"] is not None
                assert "sample.wav" in data["audio_s3_key"]
                assert data["file_size_bytes"] == len(sample_audio_bytes)
                assert mock_db.commit.called
                assert mock_storage.upload_file_from_path.called
                assert mock_celery.called
        finally:
            app.dependency_overrides.clear()


