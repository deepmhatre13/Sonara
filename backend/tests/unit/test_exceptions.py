import pytest
from fastapi import APIRouter
from httpx import ASGITransport, AsyncClient

from app.core.exceptions import AudioValidationError, EntityNotFoundError
from app.main import app

err_router = APIRouter()


@err_router.get("/test-not-found")
async def trigger_not_found():
    raise EntityNotFoundError(entity_name="Recording", entity_id="rec-abc-123")


@err_router.get("/test-audio-validation")
async def trigger_audio_error():
    raise AudioValidationError(
        detail="Audio sample rate too low",
        extra={"sample_rate": 8000, "min_required": 16000},
    )


@pytest.fixture(autouse=True)
def mount_test_router():
    app.include_router(err_router, prefix="/api/v1/test-errors")


@pytest.mark.asyncio
async def test_entity_not_found_handling():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/test-errors/test-not-found")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "ENTITY_NOT_FOUND"
        assert "Recording" in data["error"]["message"]
        assert data["error"]["details"]["entity_id"] == "rec-abc-123"
        assert "request_id" in data


@pytest.mark.asyncio
async def test_audio_validation_handling():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/test-errors/test-audio-validation")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_AUDIO_FORMAT"
        assert data["error"]["details"]["sample_rate"] == 8000
        assert "request_id" in data
