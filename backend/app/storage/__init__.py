from app.storage.interface import AudioStorage
from app.storage.s3 import S3AudioStorage

_storage_instance: AudioStorage | None = None


def get_audio_storage() -> AudioStorage:
    """Dependency injection / provider for AudioStorage."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = S3AudioStorage()
    return _storage_instance

