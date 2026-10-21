import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.database.session import get_db_session
from app.main import app
from app.models.core import User


@pytest.fixture
def search_user():
    return User(
        id=uuid.uuid4(),
        email="searcher@example.com",
        hashed_password="pw",
        is_active=True,
    )


@pytest.mark.asyncio
async def test_search_unauthenticated():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/search",
            json={"query": "financial results", "mode": "hybrid"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_search_hybrid_success(search_user):
    rec_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    mock_db = AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: search_user
    app.dependency_overrides[get_db_session] = lambda: mock_db

    from app.retrieval.hybrid import RetrievedEvidence

    mock_evidence = [
        RetrievedEvidence(
            chunk_id=chunk_id,
            recording_id=rec_id,
            text="Our Q3 revenue increased by 25 percent.",
            start_ms=1000,
            end_ms=4000,
            token_count=10,
            source_segment_ids=["seg-1"],
            speaker_labels=["SPEAKER_00"],
            transcript_version=1,
            chunking_version="v1.0",
            index_version=1,
            semantic_rank=1,
            semantic_score=0.92,
            keyword_rank=1,
            keyword_score=1.5,
            rrf_score=0.032,
        )
    ]

    with patch("app.api.v1.search.retrieve_hybrid", new=AsyncMock(return_value=mock_evidence)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/search",
                json={
                    "query": "Q3 revenue increase",
                    "mode": "hybrid",
                    "top_k": 5,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["query"] == "Q3 revenue increase"
            assert data["mode"] == "hybrid"
            assert data["total_results"] == 1
            res = data["results"][0]
            assert res["chunk_id"] == str(chunk_id)
            assert res["start_ms"] == 1000
            assert res["end_ms"] == 4000
            assert "Q3 revenue" in res["text"]
            assert res["speaker_labels"] == ["SPEAKER_00"]
            assert res["rrf_score"] == 0.032

    app.dependency_overrides.clear()
