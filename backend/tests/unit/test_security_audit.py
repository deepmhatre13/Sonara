import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.database.session import get_db_session
from app.main import app
from app.models.core import Recording, RecordingStatus, User


@pytest.fixture
def user_alice():
    return User(
        id=uuid.uuid4(),
        email="alice@example.com",
        hashed_password="pw",
        is_active=True,
        is_superuser=False,
    )


@pytest.fixture
def user_bob():
    return User(
        id=uuid.uuid4(),
        email="bob@example.com",
        hashed_password="pw",
        is_active=True,
        is_superuser=False,
    )


@pytest.mark.asyncio
async def test_cross_user_recording_metadata_isolation(user_alice, user_bob):
    """
    User Bob must NOT be able to view Alice's recording metadata.
    """
    alice_rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=alice_rec_id,
        user_id=user_alice.id,
        title="Alice confidential call",
        audio_s3_key="originals/alice/audio.wav",
        mime_type="audio/wav",
        status=RecordingStatus.SEARCHABLE,
    )

    app.dependency_overrides[get_current_user] = lambda: user_bob
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/v1/recordings/{alice_rec_id}")
            assert resp.status_code == 403
            assert "PERMISSION_DENIED" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cross_user_transcript_isolation(user_alice, user_bob):
    """
    User Bob must NOT be able to access Alice's recording transcripts.
    """
    alice_rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=alice_rec_id,
        user_id=user_alice.id,
        title="Alice private discussion",
        audio_s3_key="originals/alice/audio.wav",
        mime_type="audio/wav",
        status=RecordingStatus.SEARCHABLE,
    )

    app.dependency_overrides[get_current_user] = lambda: user_bob
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/v1/recordings/{alice_rec_id}/transcript")
            assert resp.status_code == 403
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cross_user_speaker_isolation(user_alice, user_bob):
    """
    User Bob must NOT be able to access Alice's speaker diarization turns.
    """
    alice_rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=alice_rec_id,
        user_id=user_alice.id,
        title="Alice secret recording",
        audio_s3_key="originals/alice/audio.wav",
        mime_type="audio/wav",
    )

    app.dependency_overrides[get_current_user] = lambda: user_bob
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/v1/recordings/{alice_rec_id}/speakers")
            assert resp.status_code == 403
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cross_user_search_isolation(user_alice, user_bob):
    """
    User Bob cannot search Alice's recording ID filter.
    """
    alice_rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=alice_rec_id,
        user_id=user_alice.id,
        title="Alice private recording",
        audio_s3_key="originals/alice/audio.wav",
        mime_type="audio/wav",
    )

    app.dependency_overrides[get_current_user] = lambda: user_bob
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/search",
                json={
                    "query": "financial figures",
                    "mode": "hybrid",
                    "recording_id": str(alice_rec_id),
                },
            )
            assert resp.status_code == 403
            assert "RECORDING_ACCESS_DENIED" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cross_user_rag_isolation(user_alice, user_bob):
    """
    User Bob cannot query Alice's recording via /api/v1/ask.
    """
    alice_rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=alice_rec_id,
        user_id=user_alice.id,
        title="Alice executive session",
        audio_s3_key="originals/alice/audio.wav",
        mime_type="audio/wav",
    )

    app.dependency_overrides[get_current_user] = lambda: user_bob
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/ask",
                json={
                    "question": "What was the secret key?",
                    "recording_id": str(alice_rec_id),
                },
            )
            assert resp.status_code == 403
            assert "RECORDING_ACCESS_DENIED" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cross_user_deletion_isolation(user_alice, user_bob):
    """
    User Bob cannot delete Alice's recording.
    """
    alice_rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=alice_rec_id,
        user_id=user_alice.id,
        title="Alice asset",
        audio_s3_key="originals/alice/audio.wav",
        mime_type="audio/wav",
    )

    app.dependency_overrides[get_current_user] = lambda: user_bob
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.delete(f"/api/v1/maintenance/recordings/{alice_rec_id}")
            assert resp.status_code == 403
            assert "PERMISSION_DENIED" in resp.text
    finally:
        app.dependency_overrides.clear()

