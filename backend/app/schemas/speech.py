import uuid

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    sequence_index: int
    start_ms: int = Field(..., ge=0, description="Start time in milliseconds")
    end_ms: int = Field(..., ge=0, description="End time in milliseconds")
    text: str
    language: str | None = None
    confidence: float | None = None
    model_identifier: str
    transcript_version: int


class TranscriptListResponse(BaseModel):
    recording_id: uuid.UUID
    total_segments: int
    language: str | None = None
    segments: list[TranscriptSegmentResponse]


class SpeakerTurnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    speaker_label: str
    start_ms: int = Field(..., ge=0, description="Start time in milliseconds")
    end_ms: int = Field(..., ge=0, description="End time in milliseconds")
    confidence: float | None = None
    diarization_model: str
    turn_version: int


class SpeakerListResponse(BaseModel):
    recording_id: uuid.UUID
    total_turns: int
    speakers: list[str]
    turns: list[SpeakerTurnResponse]


class ReprocessSpeechRequest(BaseModel):
    language: str | None = Field(default=None, max_length=16, description="Optional language code override, e.g. 'en'")
    enable_diarization: bool = Field(default=True, description="Whether to execute speaker diarization")


class ReprocessSpeechResponse(BaseModel):
    recording_id: uuid.UUID
    job_id: uuid.UUID
    status: str
    message: str
