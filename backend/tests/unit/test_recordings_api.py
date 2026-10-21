import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.main import app
from app.models.core import User
from app.storage import get_audio_storage
from app.storage.interface import AudioStorage


@pytest.fixture
def mock_user():
    return User(
        id=uuid.uuid4(),
        email="audio_tester@example.com",
        hashed_password="pw",
        is_active=True,
    )


@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=AudioStorage)
    storage.generate_presigned_upload_url.return_value = {
        "upload_url": "https://storage.example.com/upload?signed=true",
        "method": "PUT",
        "content_type": "audio/wav",
        "key": "staging/rec-id/sess-id",
        "expires_in": "900",
    }
    storage.check_object_exists = AsyncMock(return_value=True)
    storage.copy_object = AsyncMock()
    storage.delete_file = AsyncMock(return_value=True)
    return storage


@pytest.mark.asyncio
async def test_initiate_upload_unauthorized():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/recordings/uploads",
            json={
                "filename": "interview.wav",
                "mime_type": "audio/wav",
                "file_size_bytes": 1048576,
            },
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_initiate_upload_success(mock_user, mock_storage):
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_audio_storage] = lambda: mock_storage

    # Also mock database session flush to avoid real DB dependency in unit tests
    mock_db = AsyncMock()
    mock_db.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    mock_db.flush = AsyncMock()
    mock_db.add = MagicMock()

    from app.database.session import get_db_session

    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/recordings/uploads",
                json={
                    "filename": "meeting.wav",
                    "mime_type": "audio/wav",
                    "file_size_bytes": 5242880,
                    "title": "Quarterly All-Hands Meeting",
                },
            )
            assert resp.status_code == 201
            data = resp.json()
            assert "recording_id" in data
            assert "upload_session_id" in data
            assert data["status"] == "awaiting_upload"
            assert "upload_url" in data["upload"]
            assert data["upload"]["method"] == "PUT"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_initiate_upload_size_limit_exceeded(mock_user):
    app.dependency_overrides[get_current_user] = lambda: mock_user
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/recordings/uploads",
                json={
                    "filename": "huge_audio.wav",
                    "mime_type": "audio/wav",
                    "file_size_bytes": 500 * 1024 * 1024,  # 500MB > 250MB limit
                },
            )
            assert resp.status_code == 400
            data = resp.json()
            assert data["error"]["code"] == "INVALID_AUDIO_FORMAT"
            assert "exceeds maximum allowed upload limit" in data["error"]["message"]
    finally:
        app.dependency_overrides.clear()

