import math
import struct
import wave


def generate_synthetic_wav(
    filepath: str,
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    frequency: float = 440.0,
) -> str:
    """
    Generates a clean synthetic PCM WAV file (pure sine wave) using Python's standard wave library.
    No external dependencies or large files needed.
    """
    num_samples = int(sample_rate * duration_seconds)
    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            # Generate sine wave sample
            sample_val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack("<h", sample_val)
            if channels == 2:
                data += data
            wav_file.writeframesraw(data)

    return filepath

