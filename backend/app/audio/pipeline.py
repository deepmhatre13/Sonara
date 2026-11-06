import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import AudioValidationError

logger = logging.getLogger("sonara.audio.pipeline")

# Configurable Allowlist of Audio Formats / Containers / Codecs
ALLOWED_FORMAT_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".wma"}
ALLOWED_CODECS = {
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "mp3",
    "aac",
    "flac",
    "vorbis",
    "opus",
    "alac",
    "wmav2",
}

MAX_AUDIO_DURATION_MS = 14400 * 1000  # 4 hours maximum per recording
MIN_AUDIO_DURATION_MS = 250  # 250 milliseconds minimum


@dataclass
class AudioMetadata:
    duration_ms: int
    sample_rate: int
    channels: int
    codec: str
    format_name: str
    file_size_bytes: int
    checksum_sha256: str


@dataclass
class NormalizationResult:
    output_path: str
    metadata: AudioMetadata
    profile: str
    pipeline_version: str


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a local file in 64KB chunks."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def probe_audio(file_path: str, timeout_seconds: int = 15) -> AudioMetadata:
    """
    Inspects media streams and container metadata safely using ffprobe.
    Validates streams, channels, codec, and duration.
    """
    if not os.path.exists(file_path):
        raise AudioValidationError(detail=f"Audio file '{file_path}' does not exist on disk.")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise AudioValidationError(detail="Uploaded audio file is empty (0 bytes).")

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        # Fallback check common Windows / container paths if not in PATH
        for fallback in [r"C:\ffmpeg\bin\ffprobe.exe", "/usr/bin/ffprobe", "/usr/local/bin/ffprobe"]:
            if os.path.exists(fallback):
                ffprobe_bin = fallback
                break

    if not ffprobe_bin:
        # Fallback to soundfile / audioread for audio files when ffprobe is not installed on bare-metal
        try:
            import soundfile as sf
            info = sf.info(file_path)
            dur_ms = int(round(info.duration * 1000))
            if dur_ms < MIN_AUDIO_DURATION_MS:
                raise AudioValidationError(detail=f"Audio duration ({dur_ms}ms) is shorter than minimum required ({MIN_AUDIO_DURATION_MS}ms).")
            if dur_ms > MAX_AUDIO_DURATION_MS:
                raise AudioValidationError(detail=f"Audio duration ({dur_ms // 1000}s) exceeds maximum allowed ({MAX_AUDIO_DURATION_MS // 1000}s).")
            return AudioMetadata(
                duration_ms=dur_ms,
                sample_rate=info.samplerate,
                channels=info.channels,
                codec=info.subtype.lower() if info.subtype else "pcm",
                format_name=info.format.lower() if info.format else "wav",
                file_size_bytes=file_size,
                checksum_sha256=compute_sha256(file_path),
            )
        except AudioValidationError:
            raise
        except Exception as sf_err:
            logger.warning(f"soundfile probe fallback failed: {sf_err}")
            raise RuntimeError(f"Audio probe failed (no ffprobe and soundfile error: {sf_err})") from sf_err


    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",  # Select first audio stream
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration:format=format_name,duration,size",
        "-of",
        "json",
        file_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioValidationError(detail="Media probe timed out while inspecting audio streams.") from e
    except Exception as e:
        raise AudioValidationError(detail=f"Failed to execute ffprobe: {e}") from e

    if result.returncode != 0:
        err = result.stderr.strip()
        logger.warning(f"ffprobe rejected media file {file_path}: {err}")
        raise AudioValidationError(
            detail="File is not a valid or readable audio recording.",
            extra={"probe_error": err[:200]},
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AudioValidationError(detail="Failed to parse media probe metadata.") from e

    streams = data.get("streams", [])
    if not streams:
        raise AudioValidationError(detail="No valid audio stream detected in uploaded file.")

    audio_stream = streams[0]
    format_info = data.get("format", {})

    codec = audio_stream.get("codec_name", "").lower()
    if not codec or (ALLOWED_CODECS and codec not in ALLOWED_CODECS):
        raise AudioValidationError(
            detail=f"Unsupported audio codec '{codec}'. Supported codecs: {', '.join(sorted(ALLOWED_CODECS))}",
            extra={"codec": codec},
        )

    # Resolve duration (stream duration takes precedence, fallback to format duration)
    raw_duration = audio_stream.get("duration") or format_info.get("duration")
    if not raw_duration:
        raise AudioValidationError(detail="Could not determine audio duration.")

    try:
        duration_float = float(raw_duration)
        duration_ms = int(round(duration_float * 1000))
    except (ValueError, TypeError) as e:
        raise AudioValidationError(detail=f"Invalid duration format: {raw_duration}") from e

    if duration_ms < MIN_AUDIO_DURATION_MS:
        raise AudioValidationError(
            detail=f"Audio duration ({duration_ms}ms) is shorter than minimum required ({MIN_AUDIO_DURATION_MS}ms)."
        )

    if duration_ms > MAX_AUDIO_DURATION_MS:
        raise AudioValidationError(
            detail=f"Audio duration ({duration_ms // 1000}s) exceeds maximum allowed ({MAX_AUDIO_DURATION_MS // 1000}s)."
        )

    sample_rate = int(audio_stream.get("sample_rate", 0))
    channels = int(audio_stream.get("channels", 0))
    format_name = format_info.get("format_name", "unknown")
    checksum = compute_sha256(file_path)

    return AudioMetadata(
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
        codec=codec,
        format_name=format_name,
        file_size_bytes=file_size,
        checksum_sha256=checksum,
    )


def normalize_audio(
    source_path: str,
    output_dir: str,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    timeout_seconds: int = 300,
) -> NormalizationResult:
    """
    Converts and normalizes audio into high-fidelity 16kHz Mono 16-bit PCM WAV.
    Preserves original timeline alignment for downstream speech recognition (whisper) and diarization.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "normalized.wav")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        for fallback in [r"C:\ffmpeg\bin\ffmpeg.exe", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if os.path.exists(fallback):
                ffmpeg_bin = fallback
                break

    if not ffmpeg_bin:
        # Fallback to soundfile / librosa resampling when ffmpeg is not installed on bare-metal
        try:
            import numpy as np
            import soundfile as sf
            data, sr = sf.read(source_path)
            # Convert to mono if stereo/multichannel
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            # Resample if needed
            if sr != target_sample_rate:
                try:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sample_rate)
                except Exception:
                    pass
            sf.write(output_path, data, target_sample_rate, subtype="PCM_16")
            meta = probe_audio(output_path)
            return NormalizationResult(
                output_path=output_path,
                metadata=meta,
                profile="soundfile_pcm_s16le_16khz_mono",
                pipeline_version="v1.0",
            )
        except Exception as norm_err:
            logger.warning(f"soundfile normalization fallback failed: {norm_err}")
            raise RuntimeError(f"Audio normalization failed (no ffmpeg and soundfile error: {norm_err})") from norm_err


    # Safe subprocess execution with strict audio filters
    # -vn: Drop video if container had video streams
    # -ac: Downmix to mono (1 channel)
    # -ar: Resample to 16000 Hz
    # -acodec: pcm_s16le (16-bit PCM uncompressed)
    cmd = [
        ffmpeg_bin,
        "-y",  # Overwrite output
        "-v",
        "error",
        "-i",
        source_path,
        "-vn",
        "-ac",
        str(target_channels),
        "-ar",
        str(target_sample_rate),
        "-acodec",
        "pcm_s16le",
        output_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioValidationError(detail="Media normalization timed out.") from e

    if proc.returncode != 0:
        err = proc.stderr.strip()
        logger.error(f"FFmpeg normalization failed for {source_path}: {err}")
        raise AudioValidationError(
            detail="Failed to normalize audio file.",
            extra={"ffmpeg_error": err[:200]},
        )

    # Post-conversion validation
    normalized_meta = probe_audio(output_path, timeout_seconds=15)

    return NormalizationResult(
        output_path=output_path,
        metadata=normalized_meta,
        profile="16khz_mono_pcm16",
        pipeline_version="v1.0",
    )

