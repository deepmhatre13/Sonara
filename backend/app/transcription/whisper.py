import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.transcription.interface import (
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptSegmentResult,
)

logger = logging.getLogger("sonara.transcription.whisper")


class FasterWhisperProvider(TranscriptionProvider):
    """
    Production-grade speech transcription provider using faster-whisper.
    Applies lazy singleton model loading to prevent redundant memory allocation.
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        beam_size: int | None = None,
        vad_filter: bool | None = None,
    ):
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self.beam_size = beam_size or settings.WHISPER_BEAM_SIZE
        self.vad_filter = vad_filter if vad_filter is not None else settings.WHISPER_VAD_FILTER
        self._model = None

    @property
    def model_name(self) -> str:
        return f"faster-whisper-{self.model_size}"

    def _get_model(self) -> Any:
        """Lazily load the faster-whisper model, with HuggingFace pipeline fallback."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(
                    f"Loading faster-whisper model '{self.model_size}' on device '{self.device}' with compute_type '{self.compute_type}'..."
                )
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info(f"Model '{self.model_name}' successfully loaded.")
            except ImportError:
                # Fallback to HuggingFace transformers Whisper pipeline
                try:
                    from transformers import pipeline
                    hf_model = f"openai/whisper-{self.model_size}.en" if self.model_size in ("tiny", "base", "small") else f"openai/whisper-{self.model_size}"
                    logger.info(f"faster-whisper not installed; falling back to HuggingFace pipeline '{hf_model}'...")
                    self._model = pipeline("automatic-speech-recognition", model=hf_model, return_timestamps=True)
                    self._is_hf = True
                    logger.info(f"HuggingFace pipeline '{hf_model}' successfully loaded.")
                except Exception as e:
                    logger.error(f"Failed to load speech recognition pipeline: {e}")
                    raise ProviderUnavailableError(
                        provider_name="whisper",
                        detail=f"Neither faster-whisper nor transformers pipeline could be loaded: {e}",
                    ) from e
        return self._model

    def _transcribe_sync(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        model = self._get_model()

        # Handle HuggingFace transformers pipeline fallback
        if getattr(self, "_is_hf", False):
            # Pass pre-decoded numpy array to bypass transformers internal ffmpeg dependency
            try:
                import soundfile as sf
                audio_array, sr = sf.read(audio_path)
                audio_input = {"array": audio_array, "sampling_rate": sr}
            except Exception:
                audio_input = audio_path

            result = model(audio_input)
            raw_chunks = result.get("chunks", [])

            segments: list[TranscriptSegmentResult] = []
            for chunk in raw_chunks:
                ts = chunk.get("timestamp", (0.0, 1.0))
                s_sec = ts[0] if ts[0] is not None else 0.0
                e_sec = ts[1] if ts[1] is not None else s_sec + 1.0
                s_ms = int(round(s_sec * 1000))
                e_ms = int(round(e_sec * 1000))
                if e_ms <= s_ms:
                    e_ms = s_ms + 10
                txt = chunk.get("text", "").strip()
                if txt:
                    segments.append(
                        TranscriptSegmentResult(
                            start_ms=s_ms,
                            end_ms=e_ms,
                            text=txt,
                            confidence=0.95,
                        )
                    )
            if not segments and result.get("text", "").strip():
                segments.append(
                    TranscriptSegmentResult(
                        start_ms=0,
                        end_ms=3000,
                        text=result["text"].strip(),
                        confidence=0.95,
                    )
                )
            return TranscriptionResult(
                language=language or "en",
                duration_ms=segments[-1].end_ms if segments else 0,
                model_identifier=self.model_name,
                segments=segments,
            )

        segments_generator, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if self.vad_filter else None,
        )


        segments: list[TranscriptSegmentResult] = []
        for seg in segments_generator:
            # Strictly convert float seconds to integer milliseconds
            start_ms = int(round(seg.start * 1000))
            end_ms = int(round(seg.end * 1000))

            # Validate timestamp integrity invariants
            if start_ms < 0:
                start_ms = 0
            if end_ms <= start_ms:
                end_ms = start_ms + 10  # Ensure non-zero positive segment duration

            text = seg.text.strip()
            if text:
                # Calculate average probability if available
                confidence = None
                if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                    import math
                    confidence = round(min(1.0, max(0.0, math.exp(seg.avg_logprob))), 4)

                segments.append(
                    TranscriptSegmentResult(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=text,
                        confidence=confidence,
                    )
                )

        detected_lang = info.language if info and hasattr(info, "language") else "en"
        duration_ms = int(round(info.duration * 1000)) if info and hasattr(info, "duration") else 0

        return TranscriptionResult(
            language=detected_lang,
            duration_ms=duration_ms,
            model_identifier=self.model_name,
            segments=segments,
        )

    async def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        """Run speech recognition in a separate worker thread."""
        return await asyncio.to_thread(self._transcribe_sync, audio_path, language)
