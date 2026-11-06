from app.core.config import settings
from app.storage.interface import AudioStorage
from app.storage.local import LocalAudioStorage
from app.storage.s3 import S3AudioStorage

_storage_instance: AudioStorage | None = None


def get_audio_storage() -> AudioStorage:
    """Dependency injection / provider for AudioStorage based on configuration."""
    global _storage_instance
    if _storage_instance is None:
        if settings.STORAGE_BACKEND.lower() == "s3":
            _storage_instance = S3AudioStorage()
        else:
            _storage_instance = LocalAudioStorage(base_dir=settings.LOCAL_STORAGE_PATH)
    return _storage_instance
