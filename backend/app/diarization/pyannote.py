import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.diarization.interface import (
    DiarizationProvider,
    DiarizationResult,
    SpeakerTurn,
)

logger = logging.getLogger("sonara.diarization.pyannote")


class PyannoteDiarizationProvider(DiarizationProvider):
    """
    Production-grade speaker diarization provider using pyannote.audio.
    Applies anonymous labels (SPEAKER_00, SPEAKER_01) and integer millisecond timestamps.
    """

    def __init__(
        self,
        model_name: str | None = None,
        use_auth_token: str | None = None,
        device: str | None = None,
    ):
        self._model_name = model_name or settings.DIARIZATION_MODEL_NAME
        self.auth_token = use_auth_token or settings.HUGGINGFACE_TOKEN
        self.device = device or settings.WHISPER_DEVICE
        self._pipeline = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        """
        Verify if pyannote.audio and required dependencies/tokens are available.
        """
        if not settings.ENABLE_DIARIZATION:
            return False
        try:
            import pyannote.audio  # noqa: F401
            import torch  # noqa: F401
            # Diarization 3.1 gated model requires Hugging Face authentication
            if not self.auth_token:
                logger.info("pyannote.audio token (HUGGINGFACE_TOKEN) is not set.")
                return False
            return True
        except ImportError:
            return False

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            if not self.is_available:
                raise ProviderUnavailableError(
                    provider_name="pyannote.audio",
                    detail="Diarization is disabled, uninstalled, or missing HUGGINGFACE_TOKEN credentials.",
                )

            try:
                import torch
                from pyannote.audio import Pipeline
            except ImportError as e:
                raise ProviderUnavailableError(
                    provider_name="pyannote.audio",
                    detail="pyannote.audio or torch is not installed.",
                ) from e

            logger.info(f"Loading pyannote diarization pipeline '{self._model_name}'...")
            self._pipeline = Pipeline.from_pretrained(
                self._model_name,
                use_auth_token=self.auth_token,
            )
            if self.device != "cpu" and torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))
            logger.info("pyannote diarization pipeline successfully loaded.")
        return self._pipeline

    def _diarize_sync(self, audio_path: str, num_speakers: int | None = None) -> DiarizationResult:
        pipeline = self._get_pipeline()

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        diarization = pipeline(audio_path, **kwargs)

        turns: list[SpeakerTurn] = []
        speaker_mapping: dict[str, str] = {}
        speaker_count = 0

        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            # Enforce anonymous labels scoped strictly to this recording
            if speaker_id not in speaker_mapping:
                speaker_mapping[speaker_id] = f"SPEAKER_{speaker_count:02d}"
                speaker_count += 1

            start_ms = int(round(turn.start * 1000))
            end_ms = int(round(turn.end * 1000))

            if start_ms < 0:
                start_ms = 0
            if end_ms <= start_ms:
                end_ms = start_ms + 10

            turns.append(
                SpeakerTurn(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_label=speaker_mapping[speaker_id],
                )
            )

        return DiarizationResult(
            num_speakers=len(speaker_mapping),
            model_identifier=self.model_name,
            turns=turns,
        )

    async def diarize(self, audio_path: str, num_speakers: int | None = None) -> DiarizationResult:
        """Run diarization in a separate worker thread."""
        return await asyncio.to_thread(self._diarize_sync, audio_path, num_speakers)

