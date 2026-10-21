import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_live_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time-Ms" in response.headers


@pytest.mark.asyncio
async def test_request_id_propagation():
    transport = ASGITransport(app=app)
    custom_id = "test-request-id-12345"
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/health/live", headers={"X-Request-ID": custom_id})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id


@pytest.mark.asyncio
async def test_readiness_endpoint_structure():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/health/ready")
        # In unit test environment without real postgres/redis running, readiness returns 503
        # but the JSON response structure must be valid without leaking secrets.
        data = response.json()
        assert "status" in data
        assert "dependencies" in data
        assert "database" in data["dependencies"]
        assert "redis" in data["dependencies"]
        assert data["version"] == "0.1.0"

