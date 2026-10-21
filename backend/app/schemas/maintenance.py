import uuid

from pydantic import BaseModel, Field


class DeleteRecordingResponse(BaseModel):
    recording_id: uuid.UUID
    deleted_keys: list[str] = Field(default_factory=list)
    message: str


class ReconciliationReport(BaseModel):
    job_id: uuid.UUID
    recording_id: uuid.UUID
    task_name: str
    action: str


class ReconcileJobsResponse(BaseModel):
    reconciled_jobs_count: int
    reports: list[ReconciliationReport] = Field(default_factory=list)


class CleanupOrphanedUploadsResponse(BaseModel):
    cleaned_sessions_count: int
    cleaned_storage_keys: list[str] = Field(default_factory=list)

