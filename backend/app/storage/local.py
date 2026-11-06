import asyncio
import hashlib
import os
import shutil
from pathlib import Path
from typing import BinaryIO

from app.core.exceptions import AudioValidationError, EntityNotFoundError
from app.storage.interface import AudioStorage


class LocalAudioStorage(AudioStorage):
    """
    Windows-native local filesystem audio storage backend.
    Stores audio files and normalized artifacts under a configured root directory.
    Provides strict path traversal protection and atomic writes.
    """

    def __init__(self, base_dir: str = r"D:\Sonara\data\objects"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, key: str) -> Path:
        """
        Validates and resolves the storage key to a secure local path inside self.base_dir.
        Guarantees protection against directory traversal attacks.
        """
        clean_key = key.replace("\\", "/").strip("/ ")
        if not clean_key or ".." in clean_key.split("/"):
            raise AudioValidationError(detail="Invalid storage key: directory traversal prohibited.")

        target_path = (self.base_dir / clean_key).resolve()
        try:
            target_path.relative_to(self.base_dir)
        except ValueError as err:
            raise AudioValidationError(detail="Path traversal outside storage directory detected.") from err

        return target_path

    async def upload_file(self, file_obj: BinaryIO, destination_key: str, content_type: str) -> str:
        target_path = self._resolve_safe_path(destination_key)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp_{os.getpid()}")
        try:
            with open(temp_path, "wb") as f_out:
                shutil.copyfileobj(file_obj, f_out)
            # Atomic replace
            os.replace(temp_path, target_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return destination_key

    async def download_file(self, source_key: str) -> bytes:
        target_path = self._resolve_safe_path(source_key)
        if not target_path.is_file():
            raise EntityNotFoundError("StorageObject", source_key)

        def _read() -> bytes:
            with open(target_path, "rb") as f:
                return f.read()

        return await asyncio.to_thread(_read)

    async def download_file_to_path(self, source_key: str, local_destination_path: str) -> None:
        target_path = self._resolve_safe_path(source_key)
        if not target_path.is_file():
            raise EntityNotFoundError("StorageObject", source_key)

        dest = Path(local_destination_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _copy() -> None:
            shutil.copyfile(target_path, dest)

        await asyncio.to_thread(_copy)

    async def upload_file_from_path(self, local_source_path: str, destination_key: str, content_type: str) -> str:
        src = Path(local_source_path).resolve()
        if not src.is_file():
            raise EntityNotFoundError("LocalFile", local_source_path)

        target_path = self._resolve_safe_path(destination_key)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp_{os.getpid()}")
        try:
            shutil.copyfile(src, temp_path)
            os.replace(temp_path, target_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return destination_key

    async def delete_file(self, source_key: str) -> bool:
        try:
            target_path = self._resolve_safe_path(source_key)
            if target_path.is_file():
                target_path.unlink()
                return True
            return False
        except Exception:
            return False

    def generate_presigned_url(self, source_key: str, expiration_seconds: int = 3600) -> str:
        # For local native operation, return a direct API stream/playback path
        clean_key = source_key.replace("\\", "/").strip("/ ")
        return f"/api/v1/recordings/media/{clean_key}"

    def generate_presigned_upload_url(
        self,
        destination_key: str,
        content_type: str,
        expiration_seconds: int = 900,
        max_size_bytes: int | None = None,
    ) -> dict[str, str]:
        clean_key = destination_key.replace("\\", "/").strip("/ ")
        return {
            "upload_url": f"/api/v1/recordings/uploads/direct?key={clean_key}",
            "method": "POST",
            "content_type": content_type,
            "key": destination_key,
            "expires_in": str(expiration_seconds),
        }

    async def check_object_exists(self, key: str) -> bool:
        try:
            target_path = self._resolve_safe_path(key)
            return target_path.is_file()
        except Exception:
            return False

    async def get_object_metadata(self, key: str) -> dict[str, str | int]:
        target_path = self._resolve_safe_path(key)
        if not target_path.is_file():
            raise EntityNotFoundError("StorageObject", key)

        stat = target_path.stat()
        hasher = hashlib.sha256()
        with open(target_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)

        ext = target_path.suffix.lower().lstrip(".")
        content_type = f"audio/{ext}" if ext in ["wav", "mp3", "ogg", "flac", "m4a"] else "application/octet-stream"

        return {
            "content_length": stat.st_size,
            "content_type": content_type,
            "etag": hasher.hexdigest(),
        }

    async def copy_object(self, source_key: str, destination_key: str) -> None:
        src = self._resolve_safe_path(source_key)
        if not src.is_file():
            raise EntityNotFoundError("StorageObject", source_key)

        dest = self._resolve_safe_path(destination_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
