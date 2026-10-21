import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.database.session import get_db_session
from app.main import app
from app.models.core import Recording, RecordingStatus, SpeakerTurn, TranscriptSegment, User


@pytest.fixture
def test_user():
    return User(
        id=uuid.uuid4(),
        email="speech_tester@example.com",
        hashed_password="pw",
        is_active=True,
    )


@pytest.mark.asyncio
async def test_get_transcript_endpoint(test_user):
    rec_id = uuid.uuid4()
    rec = Recording(
        id=rec_id,
        user_id=test_user.id,
        title="Test Speech",
        audio_s3_key="originals/test/source",
        file_size_bytes=1000,
        mime_type="audio/wav",
        status=RecordingStatus.SPEECH_COMPLETE,
    )

    seg1 = TranscriptSegment(
        id=uuid.uuid4(),
        recording_id=rec_id,
        sequence_index=0,
        start_ms=0,
        end_ms=1200,
        text="Welcome to the Sonara demonstration.",
        language="en",
        confidence=0.97,
        model_identifier="faster-whisper-base",
    )

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=rec)

    # Mock execute for scalar list and total count
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [seg1]
    mock_total = MagicMock()
    mock_total.scalar.return_value = 1

    mock_db.execute = AsyncMock(side_effect=[MagicMock(scalars=MagicMock(return_value=mock_scalars)), mock_total])

    app.dependency_overrides[get_current_user] = lambda: test_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/v1/recordings/{rec_id}/transcript")
            assert resp.status_code == 200
            data = resp.json()
            assert data["recording_id"] == str(rec_id)
            assert data["total_segments"] == 1
            assert len(data["segments"]) == 1
            assert data["segments"][0]["text"] == "Welcome to the Sonara demonstration."
            assert data["segments"][0]["start_ms"] == 0
            assert data["segments"][0]["end_ms"] == 1200
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_speakers_endpoint(test_user):
    rec_id = uuid.uuid4()
    rec = Recording(
        id=rec_id,
        user_id=test_user.id,
        title="Test Speech",
        audio_s3_key="originals/test/source",
        file_size_bytes=1000,
        mime_type="audio/wav",
        status=RecordingStatus.SPEECH_COMPLETE,
    )

    turn1 = SpeakerTurn(
        id=uuid.uuid4(),
        recording_id=rec_id,
        speaker_label="SPEAKER_00",
        start_ms=0,
        end_ms=3000,
        confidence=0.94,
        diarization_model="pyannote/speaker-diarization-3.1",
    )

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=rec)
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [turn1]
    mock_db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=mock_scalars)))

    app.dependency_overrides[get_current_user] = lambda: test_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/v1/recordings/{rec_id}/speakers")
            assert resp.status_code == 200
            data = resp.json()
            assert data["recording_id"] == str(rec_id)
            assert data["total_turns"] == 1
            assert data["speakers"] == ["SPEAKER_00"]
            assert data["turns"][0]["speaker_label"] == "SPEAKER_00"
    finally:
        app.dependency_overrides.clear()
