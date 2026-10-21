from abc import ABC, abstractmethod
from typing import BinaryIO


class AudioStorage(ABC):
    """Abstract interface for audio file storage (S3, MinIO, or local filesystem)."""

    @abstractmethod
    async def upload_file(self, file_obj: BinaryIO, destination_key: str, content_type: str) -> str:
        """Uploads an audio file object and returns the storage reference key."""
        pass

    @abstractmethod
    async def download_file(self, source_key: str) -> bytes:
        """Downloads audio file bytes for the specified storage key."""
        pass

    @abstractmethod
    async def download_file_to_path(self, source_key: str, local_destination_path: str) -> None:
        """Downloads audio file directly to a local filesystem path for worker processing."""
        pass

    @abstractmethod
    async def upload_file_from_path(self, local_source_path: str, destination_key: str, content_type: str) -> str:
        """Uploads a local file to storage and returns the storage key."""
        pass

    @abstractmethod
    async def delete_file(self, source_key: str) -> bool:
        """Deletes an audio file from storage."""
        pass

    @abstractmethod
    def generate_presigned_url(self, source_key: str, expiration_seconds: int = 3600) -> str:
        """Generates a secure temporary presigned playback / download URL."""
        pass

    @abstractmethod
    def generate_presigned_upload_url(
        self,
        destination_key: str,
        content_type: str,
        expiration_seconds: int = 900,
        max_size_bytes: int | None = None,
    ) -> dict[str, str]:
        """
        Generates a secure temporary presigned upload URL or POST policy.
        Returns a dictionary with 'upload_url', 'method', and any required fields/headers.
        """
        pass

    @abstractmethod
    async def check_object_exists(self, key: str) -> bool:
        """Checks if an object exists in storage."""
        pass

    @abstractmethod
    async def get_object_metadata(self, key: str) -> dict[str, str | int]:
        """Retrieves object metadata such as content_length, content_type, and etag."""
        pass

    @abstractmethod
    async def copy_object(self, source_key: str, destination_key: str) -> None:
        """Copies an object within the storage bucket."""
        pass
