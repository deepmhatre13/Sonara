from app.diarization.interface import (
    DiarizationProvider,
    DiarizationResult,
    SpeakerTurn,
)
from app.diarization.pyannote import PyannoteDiarizationProvider

_diarization_provider: DiarizationProvider | None = None


def get_diarization_provider() -> DiarizationProvider:
    """Dependency injection / provider for speaker diarization."""
    global _diarization_provider
    if _diarization_provider is None:
        _diarization_provider = PyannoteDiarizationProvider()
    return _diarization_provider


__all__ = [
    "DiarizationProvider",
    "DiarizationResult",
    "SpeakerTurn",
    "PyannoteDiarizationProvider",
    "get_diarization_provider",
]

