from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class TranscriptSegmentResult(BaseModel):
    """
    Standardized transcription segment.
    All timestamps are strictly integer milliseconds.
    """

    start_ms: int = Field(..., ge=0, description="Segment start in milliseconds")
    end_ms: int = Field(..., ge=0, description="Segment end in milliseconds")
    text: str = Field(..., description="Transcribed text content")
    confidence: float | None = Field(default=None, description="Model confidence score (0.0 to 1.0)")


class TranscriptionResult(BaseModel):
    language: str
    duration_ms: int
    model_identifier: str
    segments: list[TranscriptSegmentResult]


class TranscriptionProvider(ABC):
    """Abstract interface for speech transcription providers (e.g. faster-whisper)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for the transcription model."""
        pass

    @abstractmethod
    async def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio from a local file path."""
        pass
