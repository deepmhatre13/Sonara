import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.models.core import Recording, RecordingStatus, TranscriptChunk, User
from app.retrieval.hybrid import retrieve_hybrid
from app.retrieval.keyword import retrieve_keyword
from app.retrieval.semantic import retrieve_semantic

TEST_DB_URL = "postgresql+asyncpg://sonara_user:sonara_password@localhost:5433/sonara_db"


@pytest.mark.asyncio
async def test_real_pgvector_and_fulltext_integration():
    """
    Live PostgreSQL + pgvector integration test.
    Tests:
    1. Real SentenceTransformer inference producing 384-dim embeddings.
    2. Real pgvector persistence into table transcript_chunks.
    3. Cosine distance semantic similarity search via pgvector operators.
    4. GIN indexed TSVECTOR full-text search.
    5. Hybrid retrieval and multi-tenant user ownership filtering in actual SQL.
    """
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    user_a_id = uuid.uuid4()
    user_b_id = uuid.uuid4()
    rec_a_id = uuid.uuid4()
    rec_b_id = uuid.uuid4()

    embedder = SentenceTransformerEmbeddingProvider()

    test_sentences = [
        "Annual enterprise recurring revenue expanded to eighty million dollars.",
        "Engineering teams deployed critical zero-downtime database migrations.",
        "The marketing campaign focused primarily on audio intelligence capabilities.",
    ]

    embeddings = await embedder.embed_texts(test_sentences)
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 384

    async with async_session() as session:
        # Create users
        user_a = User(
            id=user_a_id,
            email=f"alice_{user_a_id.hex[:6]}@example.com",
            hashed_password="pw",
            is_active=True,
        )
        user_b = User(
            id=user_b_id,
            email=f"bob_{user_b_id.hex[:6]}@example.com",
            hashed_password="pw",
            is_active=True,
        )
        session.add_all([user_a, user_b])

        # Create recordings
        rec_a = Recording(
            id=rec_a_id,
            user_id=user_a_id,
            title="Alice Financial Briefing",
            audio_s3_key="originals/alice.wav",
            mime_type="audio/wav",
            status=RecordingStatus.SEARCHABLE,
        )
        rec_b = Recording(
            id=rec_b_id,
            user_id=user_b_id,
            title="Bob Engineering Briefing",
            audio_s3_key="originals/bob.wav",
            mime_type="audio/wav",
            status=RecordingStatus.SEARCHABLE,
        )
        session.add_all([rec_a, rec_b])

        # Create chunks
        chunk_a1 = TranscriptChunk(
            id=uuid.uuid4(),
            recording_id=rec_a_id,
            chunk_index=0,
            text=test_sentences[0],
            embedding=embeddings[0],
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dimension=384,
            start_ms=1000,
            end_ms=6000,
            token_count=10,
            source_segment_ids=[],
            speaker_labels=["SPEAKER_00"],
            transcript_version=1,
            chunking_version="v1.0",
            index_version=1,
            is_active=True,
        )
        chunk_b1 = TranscriptChunk(
            id=uuid.uuid4(),
            recording_id=rec_b_id,
            chunk_index=0,
            text=test_sentences[1],
            embedding=embeddings[1],
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dimension=384,
            start_ms=2000,
            end_ms=7000,
            token_count=8,
            source_segment_ids=[],
            speaker_labels=["SPEAKER_01"],
            transcript_version=1,
            chunking_version="v1.0",
            index_version=1,
            is_active=True,
        )

        session.add_all([chunk_a1, chunk_b1])
        await session.commit()

        try:
            # 1. Semantic retrieval with real pgvector cosine distance
            query = "What is the annual revenue growth?"
            semantic_candidates = await retrieve_semantic(
                db=session,
                query=query,
                user_id=user_a_id,
                recording_id=None,
                top_k=5,
            )
            assert len(semantic_candidates) == 1
            assert semantic_candidates[0].chunk.id == chunk_a1.id
            assert "eighty million dollars" in semantic_candidates[0].chunk.text

            # 2. Multi-tenant ownership filter: Bob cannot retrieve Alice's chunk
            bob_semantic = await retrieve_semantic(
                db=session,
                query=query,
                user_id=user_b_id,
                recording_id=None,
                top_k=5,
            )
            # Bob's query should only match Bob's chunk or none if below threshold
            bob_chunk_ids = {c.chunk.id for c in bob_semantic}
            assert chunk_a1.id not in bob_chunk_ids

            # 3. PostgreSQL full-text search with TSVECTOR and plainto_tsquery
            kw_candidates = await retrieve_keyword(
                db=session,
                query="recurring revenue",
                user_id=user_a_id,
                recording_id=None,
                top_k=5,
            )
            assert len(kw_candidates) == 1
            assert kw_candidates[0].chunk.id == chunk_a1.id

            # 4. Real hybrid retrieval with RRF fusion
            hybrid_results = await retrieve_hybrid(
                db=session,
                query="annual recurring revenue",
                user_id=user_a_id,
                recording_id=None,
                top_k=5,
                enable_reranking=False,
            )
            assert len(hybrid_results) == 1
            assert hybrid_results[0].chunk_id == chunk_a1.id
            assert hybrid_results[0].rrf_score > 0
            assert hybrid_results[0].start_ms == 1000
            assert hybrid_results[0].end_ms == 6000

        finally:
            # Clean up test rows
            await session.delete(chunk_a1)
            await session.delete(chunk_b1)
            await session.delete(rec_a)
            await session.delete(rec_b)
            await session.delete(user_a)
            await session.delete(user_b)
            await session.commit()

    await engine.dispose()

