import os
import tempfile

import pytest

from app.audio.pipeline import compute_sha256, probe_audio
from app.core.exceptions import AudioValidationError
from tests.fixtures.audio import generate_synthetic_wav


def test_compute_sha256():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"sonara audio test payload")
        temp_path = f.name

    try:
        digest = compute_sha256(temp_path)
        assert len(digest) == 64
        assert isinstance(digest, str)
    finally:
        os.remove(temp_path)


def test_probe_empty_audio_fails():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(AudioValidationError) as exc:
            probe_audio(temp_path)
        assert "empty" in str(exc.value.detail).lower()
    finally:
        os.remove(temp_path)


def test_probe_nonexistent_audio_fails():
    with pytest.raises(AudioValidationError):
        probe_audio("non_existent_file_path_12345.wav")


def test_synthetic_wav_creation_and_probe():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        generate_synthetic_wav(temp_path, duration_seconds=1.5, sample_rate=16000, channels=1)
        assert os.path.exists(temp_path)
        assert os.path.getsize(temp_path) > 0

        # Note: If ffprobe is not installed on host Windows machine, probe_audio will raise RuntimeError
        # or AudioValidationError; we verify graceful exception behavior
        try:
            meta = probe_audio(temp_path)
            assert meta.duration_ms == 1500
            assert meta.sample_rate == 16000
            assert meta.channels == 1
        except RuntimeError as e:
            assert "ffprobe executable not found" in str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

