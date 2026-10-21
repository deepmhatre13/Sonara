import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.core import RecordingStatus


class UploadInitiateRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255, description="Original audio file name")
    mime_type: str = Field(..., min_length=3, max_length=64, description="Declared MIME type, e.g., audio/wav")
    file_size_bytes: int = Field(..., gt=0, description="Expected file size in bytes")
    expected_checksum: str | None = Field(default=None, max_length=64, description="Optional SHA-256 checksum")
    title: str | None = Field(default=None, max_length=255, description="Optional human-readable title")
    idempotency_key: str | None = Field(default=None, max_length=128, description="Optional client idempotency key")


class PresignedUploadDetails(BaseModel):
    upload_url: str
    method: str
    content_type: str
    key: str
    expires_in: int


class UploadInitiateResponse(BaseModel):
    recording_id: uuid.UUID
    upload_session_id: uuid.UUID
    status: RecordingStatus
    upload: PresignedUploadDetails
    expires_at: datetime


class UploadFinalizeResponse(BaseModel):
    recording_id: uuid.UUID
    status: RecordingStatus
    message: str


class DirectAudioUploadResponse(BaseModel):
    recording_id: uuid.UUID
    job_id: uuid.UUID
    status: RecordingStatus
    file_size_bytes: int
    audio_s3_key: str
    message: str


class RecordingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: RecordingStatus
    mime_type: str
    file_size_bytes: int
    duration_ms: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None
    checksum_sha256: str | None = None
    normalization_profile: str | None = None
    pipeline_version: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

