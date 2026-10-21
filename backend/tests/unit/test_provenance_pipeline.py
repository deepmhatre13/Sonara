import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.models.core import Recording, RecordingStatus, User
from app.rag.service import RAGService
from app.retrieval.hybrid import RetrievedEvidence
from app.schemas.rag import AnswerStatus, AskRequest


@pytest.fixture
def e2e_user():
    return User(
        id=uuid.uuid4(),
        email="auditor@example.com",
        hashed_password="hashed_pw",
        is_active=True,
    )


@pytest.mark.asyncio
async def test_provenance_and_citation_invariant_pipeline(e2e_user):
    """
    Validates end-to-end evidence selection, prompt generation, citation validation,
    and server-authoritative timestamp linking against simulated persisted recording evidence.
    """
    rec_id = uuid.uuid4()
    chunk_1 = uuid.uuid4()
    chunk_2 = uuid.uuid4()

    mock_db = AsyncMock()
    mock_db.get.return_value = Recording(
        id=rec_id,
        user_id=e2e_user.id,
        title="Quarterly Review Audio",
        audio_s3_key="originals/audio.wav",
        normalized_s3_key="processed/audio.wav",
        mime_type="audio/wav",
        status=RecordingStatus.SEARCHABLE,
    )

    persisted_evidence = [
        RetrievedEvidence(
            chunk_id=chunk_1,
            recording_id=rec_id,
            text="In fiscal year 2026, total customer expansion increased by 42 percent.",
            start_ms=15000,
            end_ms=21000,
            token_count=13,
            source_segment_ids=["seg-1", "seg-2"],
            speaker_labels=["SPEAKER_01"],
            transcript_version=1,
            chunking_version="v1.0",
            index_version=1,
            rrf_score=0.045,
        ),
        RetrievedEvidence(
            chunk_id=chunk_2,
            recording_id=rec_id,
            text="The team finalized engineering milestones ahead of schedule in November.",
            start_ms=22500,
            end_ms=28000,
            token_count=11,
            source_segment_ids=["seg-3"],
            speaker_labels=["SPEAKER_02"],
            transcript_version=1,
            chunking_version="v1.0",
            index_version=1,
            rrf_score=0.038,
        ),
    ]

    from app.llm.interface import LLMResponse

    # LLM outputs answer citing [E1]
    mock_gemini = AsyncMock()
    mock_gemini.is_configured = True
    mock_gemini.provider_name = "gemini"
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_gemini.generate_response.return_value = LLMResponse(
        content='{"answer": "Customer expansion increased by 42% in FY2026.", "citations": ["[E1]"], "insufficient_evidence": false}',
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        prompt_tokens=450,
        completion_tokens=35,
    )

    rag_service = RAGService(db=mock_db)

    with (
        patch("app.rag.service.retrieve_hybrid", new=AsyncMock(return_value=persisted_evidence)),
        patch("app.rag.service.get_gemini_provider", return_value=mock_gemini),
    ):
        req = AskRequest(
            question="What was the customer expansion rate in 2026?",
            recording_id=rec_id,
        )
        response = await rag_service.answer_question(request=req, user_id=e2e_user.id)

        # Invariant 1: Validated status
        assert response.status == AnswerStatus.ANSWERED
        assert "42%" in response.answer

        # Invariant 2: Exactly 1 citation matched to authoritative database chunk
        assert len(response.citations) == 1
        cit = response.citations[0]
        assert cit.evidence_id == "[E1]"
        assert cit.chunk_id == chunk_1
        assert cit.recording_id == rec_id

        # Invariant 3: Server-authoritative integer millisecond timestamps preserved
        assert cit.start_ms == 15000
        assert cit.end_ms == 21000
        assert cit.speaker_labels == ["SPEAKER_01"]

        # Invariant 4: Token counts and request provenance populated
        assert response.prompt_tokens == 450
        assert response.completion_tokens == 35

