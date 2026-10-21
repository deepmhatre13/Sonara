from app.transcription.interface import (
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptSegmentResult,
)
from app.transcription.whisper import FasterWhisperProvider

_transcription_provider: TranscriptionProvider | None = None


def get_transcription_provider() -> TranscriptionProvider:
    """Dependency injection / provider for speech transcription."""
    global _transcription_provider
    if _transcription_provider is None:
        _transcription_provider = FasterWhisperProvider()
    return _transcription_provider


__all__ = [
    "TranscriptionProvider",
    "TranscriptionResult",
    "TranscriptSegmentResult",
    "FasterWhisperProvider",
    "get_transcription_provider",
]

