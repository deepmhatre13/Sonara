from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class SpeakerTurn(BaseModel):
    """
    Standardized speaker diarization turn.
    Timestamps are integer milliseconds. Overlapping turns are representable.
    Speaker labels are anonymous and scoped strictly to a recording.
    """

    start_ms: int = Field(..., ge=0, description="Speaker turn start in milliseconds")
    end_ms: int = Field(..., ge=0, description="Speaker turn end in milliseconds")
    speaker_label: str = Field(..., description="Anonymous speaker identifier, e.g., 'SPEAKER_00'")
    confidence: float | None = Field(default=None, description="Optional confidence score (0.0 to 1.0)")


class DiarizationResult(BaseModel):
    num_speakers: int
    model_identifier: str
    turns: list[SpeakerTurn]


class DiarizationProvider(ABC):
    """Abstract interface for speaker diarization providers (e.g. pyannote.audio)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for the diarization model."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Indicates if the provider has all dependencies, models, and tokens to operate."""
        pass

    @abstractmethod
    async def diarize(self, audio_path: str, num_speakers: int | None = None) -> DiarizationResult:
        """Perform speaker diarization on audio file."""
        pass
