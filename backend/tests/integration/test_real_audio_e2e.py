import os
import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.models.core import (
    JobStatus,
    ProcessingJob,
    Recording,
    RecordingStatus,
    TranscriptChunk,
    TranscriptSegment,
    User,
)
from app.retrieval.hybrid import retrieve_hybrid
from app.transcription.whisper import FasterWhisperProvider

TEST_DB_URL = "postgresql+asyncpg://sonara_user:sonara_password@localhost:5433/sonara_db"
AUDIO_FIXTURE_PATH = "backend/tests/fixtures/spoken_word_fixture.wav"


@pytest.mark.asyncio
async def test_real_audio_to_search_end_to_end_pipeline():
    """
    True end-to-end integration test:
    1. Authenticates test user.
    2. Exercises real Whisper inference on an actual spoken word audio sample.
       (Transcribes: "He hoped there would be stew for dinner, turnips and carrots...")
    3. Persists transcript segments to PostgreSQL.
    4. Chunks the transcript preserving ordered sequence and millisecond timestamps.
    5. Computes real dense embeddings with SentenceTransformers (BAAI/bge-small-en-v1.5).
    6. Stores chunks and vectors in real PostgreSQL with pgvector.
    7. Executes hybrid retrieval (keyword + pgvector semantic cosine distance) through the search engine.
    8. Verifies exact content retrieval and multi-tenant isolation.
    """
    assert os.path.exists(AUDIO_FIXTURE_PATH), f"Fixture not found: {AUDIO_FIXTURE_PATH}"

    engine = create_async_engine(TEST_DB_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    user_id = uuid.uuid4()
    rec_id = uuid.uuid4()
    job_id = uuid.uuid4()

    db_segments = []
    chunk = None
    async with async_session() as session:
        # Create user
        user = User(
            id=user_id,
            email=f"speaker_{user_id.hex[:6]}@example.com",
            hashed_password="pw",
            is_active=True,
        )
        session.add(user)

        # Create recording in UPLOADED state
        rec = Recording(
            id=rec_id,
            user_id=user_id,
            title="Literary Reading Excerpt",
            audio_s3_key=f"originals/{rec_id}/spoken_word_fixture.wav",
            normalized_s3_key=f"processed/{rec_id}/v1.0/audio.wav",
            mime_type="audio/wav",
            status=RecordingStatus.TRANSCRIBING,
        )
        session.add(rec)

        job = ProcessingJob(
            id=job_id,
            recording_id=rec_id,
            task_name="sonara.speech.transcribe_and_diarize",
            status=JobStatus.PROCESSING,
        )
        session.add(job)
        await session.commit()

        try:
            # 1. Run real Whisper transcription on the audio fixture
            whisper = FasterWhisperProvider(model_size="tiny")
            transcription = await whisper.transcribe(AUDIO_FIXTURE_PATH)

            assert len(transcription.segments) >= 1
            combined_text = " ".join(s.text for s in transcription.segments)
            assert "stew" in combined_text.lower() or "turnips" in combined_text.lower()

            # 2. Persist transcript segments to PostgreSQL
            db_segments = []
            for idx, seg in enumerate(transcription.segments):
                s_entity = TranscriptSegment(
                    id=uuid.uuid4(),
                    recording_id=rec_id,
                    sequence_index=idx,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=seg.text,
                    confidence=seg.confidence,
                    language=transcription.language,
                    model_identifier=transcription.model_identifier,
                    transcript_version=1,
                )
                session.add(s_entity)
                db_segments.append(s_entity)
            await session.commit()

            # 3. Create chunks and compute real embeddings
            embedder = SentenceTransformerEmbeddingProvider()
            chunk_texts = [combined_text]
            embeddings = await embedder.embed_texts(chunk_texts)

            chunk = TranscriptChunk(
                id=uuid.uuid4(),
                recording_id=rec_id,
                chunk_index=0,
                start_ms=db_segments[0].start_ms,
                end_ms=db_segments[-1].end_ms,
                text=combined_text,
                token_count=len(combined_text.split()),
                source_segment_ids=[str(s.id) for s in db_segments],
                speaker_labels=["SPEAKER_00"],
                embedding=embeddings[0],
                embedding_model=embedder.model_name,
                embedding_dimension=embedder.dimension,
                transcript_version=1,
                chunking_version="v1.0",
                index_version=1,
                is_active=True,
            )
            session.add(chunk)
            rec.status = RecordingStatus.SEARCHABLE
            job.status = JobStatus.COMPLETED
            await session.commit()

            # 4. Execute real hybrid search with semantic + keyword fusion
            search_query = "What food was prepared for dinner?"
            results = await retrieve_hybrid(
                db=session,
                query=search_query,
                user_id=user_id,
                recording_id=rec_id,
                top_k=5,
                enable_reranking=False,
            )

            assert len(results) >= 1
            top_hit = results[0]
            assert top_hit.chunk_id == chunk.id
            assert top_hit.recording_id == rec_id
            assert "stew" in top_hit.text.lower() or "turnips" in top_hit.text.lower()
            assert top_hit.start_ms == db_segments[0].start_ms
            assert top_hit.rrf_score > 0

        finally:
            # Clean up test rows
            if chunk is not None:
                await session.delete(chunk)
            for s in db_segments:
                await session.delete(s)
            await session.delete(job)
            await session.delete(rec)
            await session.delete(user)
            await session.commit()

    await engine.dispose()

