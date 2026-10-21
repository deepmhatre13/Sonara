from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: dict[str, Any] | None = Field(default=None, description="Optional contextual error metadata")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: ErrorDetail
    request_id: str | None = Field(default=None, description="Request correlation identifier")


class HealthStatus(BaseModel):
    status: str = Field(..., description="Health status (e.g., healthy, degraded, unhealthy)")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current server ISO timestamp")


class ReadinessDependency(BaseModel):
    status: str = Field(..., description="Dependency status (e.g., reachable, unreachable, error)")
    latency_ms: float | None = Field(default=None, description="Check roundtrip latency in milliseconds")
    error: str | None = Field(default=None, description="Error reason if degraded or unreachable")


class ReadinessResponse(BaseModel):
    status: str = Field(..., description="Overall readiness status")
    dependencies: dict[str, ReadinessDependency]
    version: str
    timestamp: str

