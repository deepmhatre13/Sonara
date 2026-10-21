import time
from datetime import UTC, datetime

from fastapi import APIRouter, Response, status
from redis import Redis
from sqlalchemy import text

from app.core.config import settings
from app.database.session import AsyncSessionLocal
from app.schemas.health import (
    HealthStatus,
    ReadinessDependency,
    ReadinessResponse,
)

router = APIRouter(prefix="/health", tags=["Health & Readiness"])


@router.get(
    "/live",
    response_model=HealthStatus,
    summary="Liveness probe",
    description="Returns liveness status. Never depends on the database or external systems.",
)
async def get_liveness() -> HealthStatus:
    return HealthStatus(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Verifies essential dependencies (PostgreSQL, Redis). Returns 503 if any required service is down.",
)
async def get_readiness(response: Response) -> ReadinessResponse:
    dependencies: dict[str, ReadinessDependency] = {}
    is_ready = True

    # 1. Check PostgreSQL Database connection
    db_start = time.perf_counter()
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_latency = (time.perf_counter() - db_start) * 1000
        dependencies["database"] = ReadinessDependency(
            status="reachable",
            latency_ms=round(db_latency, 2),
        )
    except Exception:
        is_ready = False
        db_latency = (time.perf_counter() - db_start) * 1000
        dependencies["database"] = ReadinessDependency(
            status="unreachable",
            latency_ms=round(db_latency, 2),
            error="Database connection failed",
        )

    # 2. Check Redis connection
    redis_start = time.perf_counter()
    try:
        redis_client = Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        redis_client.ping()
        redis_latency = (time.perf_counter() - redis_start) * 1000
        dependencies["redis"] = ReadinessDependency(
            status="reachable",
            latency_ms=round(redis_latency, 2),
        )
    except Exception:
        is_ready = False
        redis_latency = (time.perf_counter() - redis_start) * 1000
        dependencies["redis"] = ReadinessDependency(
            status="unreachable",
            latency_ms=round(redis_latency, 2),
            error="Redis connection failed",
        )

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        dependencies=dependencies,
        version="0.1.0",
        timestamp=datetime.now(UTC).isoformat(),
    )

