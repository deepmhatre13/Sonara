import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.database.session import get_db_session
from app.main import app
from app.models.core import Recording, User
from app.retrieval.hybrid import RetrievedEvidence


@pytest.fixture
def rag_user():
    return User(
        id=uuid.uuid4(),
        email="raguser@example.com",
        hashed_password="hashed_pw",
        is_active=True,
    )


@pytest.mark.asyncio
async def test_ask_unauthenticated():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/ask",
            json={"question": "What was discussed in Q3?"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_ask_recording_forbidden(rag_user):
    rec_id = uuid.uuid4()
    mock_db = AsyncMock()
    # Mock finding a recording owned by someone else
    mock_db.get.return_value = Recording(
        id=rec_id,
        user_id=uuid.uuid4(),  # Different owner
        title="confidential",
        audio_s3_key="raw/rec.wav",
        mime_type="audio/wav",
    )


    app.dependency_overrides[get_current_user] = lambda: rag_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/ask",
                json={"question": "What is the secret?", "recording_id": str(rec_id)},
            )
            assert resp.status_code == 403
            assert "RECORDING_ACCESS_DENIED" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ask_success(rag_user):
    chunk_id = uuid.uuid4()
    rec_id = uuid.uuid4()

    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=rec_id,
        user_id=rag_user.id,
        title="meeting",
        audio_s3_key="raw/rec.wav",
        mime_type="audio/wav",
    )


    app.dependency_overrides[get_current_user] = lambda: rag_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    mock_evidence = [
        RetrievedEvidence(
            chunk_id=chunk_id,
            recording_id=rec_id,
            text="The project deadline was officially confirmed for December 15th.",
            start_ms=12000,
            end_ms=16000,
            token_count=12,
            source_segment_ids=["seg-1"],
            speaker_labels=["SPEAKER_00"],
            transcript_version=1,
            chunking_version="v1.0",
            index_version=1,
            rrf_score=0.05,
        )
    ]

    from app.llm.interface import LLMResponse

    mock_gemini = AsyncMock()
    mock_gemini.is_configured = True
    mock_gemini.provider_name = "gemini"
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_gemini.generate_response.return_value = LLMResponse(
        content='{"answer": "The project deadline is December 15th.", "citations": ["[E1]"], "insufficient_evidence": false}',
        provider_name="gemini",
        model_name="gemini-2.5-flash",
    )


    with (
        patch("app.rag.service.retrieve_hybrid", new=AsyncMock(return_value=mock_evidence)),
        patch("app.rag.service.get_gemini_provider", return_value=mock_gemini),
    ):
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/api/v1/ask",
                    json={
                        "question": "When is the project deadline?",
                        "recording_id": str(rec_id),
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "answered"
                assert "December 15th" in data["answer"]
                assert len(data["citations"]) == 1
                assert data["citations"][0]["evidence_id"] == "[E1]"
                assert data["citations"][0]["chunk_id"] == str(chunk_id)
                assert data["citations"][0]["start_ms"] == 12000
                assert data["citations"][0]["end_ms"] == 16000
                assert data["model_name"] == "gemini-2.5-flash"
                assert data["provider_name"] == "gemini"
        finally:
            app.dependency_overrides.clear()


