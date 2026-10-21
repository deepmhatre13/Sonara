import asyncio
import logging
from typing import BinaryIO

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.exceptions import SonaraException
from app.storage.interface import AudioStorage

logger = logging.getLogger("sonara.storage.s3")


class S3AudioStorage(AudioStorage):
    """
    Production-grade S3-compatible object storage adapter.
    Works seamlessly with MinIO and AWS S3.
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket_name: str | None = None,
        region_name: str | None = None,
        use_ssl: bool | None = None,
    ):
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.endpoint_url = endpoint_url or settings.S3_ENDPOINT_URL
        self.access_key = access_key or settings.S3_ACCESS_KEY
        self.secret_key = secret_key or settings.S3_SECRET_KEY
        self.region_name = region_name or settings.S3_REGION
        self.use_ssl = use_ssl if use_ssl is not None else settings.S3_USE_SSL

        # Configure boto3 S3 client with signature version s3v4 and path-style addressing for MinIO compatibility
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url if self.endpoint_url else None,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name,
            use_ssl=self.use_ssl,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def _ensure_bucket_exists_sync(self) -> None:
        """Create the storage bucket if it does not already exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchBucket"):
                try:
                    kwargs = {"Bucket": self.bucket_name}
                    # LocationConstraint is required outside us-east-1 on AWS S3
                    if self.region_name and self.region_name != "us-east-1" and not self.endpoint_url:
                        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region_name}
                    self.client.create_bucket(**kwargs)
                    logger.info(f"Created object storage bucket: {self.bucket_name}")
                except ClientError as create_err:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {create_err}")
                    raise
            else:
                logger.error(f"Error checking bucket {self.bucket_name}: {e}")
                raise

    async def ensure_bucket_exists(self) -> None:
        await asyncio.to_thread(self._ensure_bucket_exists_sync)

    def generate_presigned_upload_url(
        self,
        destination_key: str,
        content_type: str,
        expiration_seconds: int = 900,
        max_size_bytes: int | None = None,
    ) -> dict[str, str]:
        """
        Generate a presigned PUT URL for direct client-to-storage upload.
        """
        params = {
            "Bucket": self.bucket_name,
            "Key": destination_key,
            "ContentType": content_type,
        }
        try:
            presigned_url = self.client.generate_presigned_url(
                ClientMethod="put_object",
                Params=params,
                ExpiresIn=expiration_seconds,
                HttpMethod="PUT",
            )
            return {
                "upload_url": presigned_url,
                "method": "PUT",
                "content_type": content_type,
                "key": destination_key,
                "expires_in": str(expiration_seconds),
            }
        except ClientError as e:
            logger.error(f"Failed to generate presigned upload URL: {e}")
            raise SonaraException(
                status_code=500,
                detail="Unable to generate secure upload credentials.",
                code="STORAGE_PRESIGN_ERROR",
            ) from e

    def generate_presigned_url(self, source_key: str, expiration_seconds: int = 3600) -> str:
        """
        Generate a presigned GET URL for temporary private audio playback.
        """
        try:
            return self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket_name, "Key": source_key},
                ExpiresIn=expiration_seconds,
            )
        except ClientError as e:
            logger.error(f"Failed to generate presigned playback URL: {e}")
            raise SonaraException(
                status_code=500,
                detail="Unable to generate audio playback link.",
                code="STORAGE_PLAYBACK_URL_ERROR",
            ) from e

    def _upload_file_sync(self, file_obj: BinaryIO, destination_key: str, content_type: str) -> str:
        self.client.upload_fileobj(
            file_obj,
            self.bucket_name,
            destination_key,
            ExtraArgs={"ContentType": content_type},
        )
        return destination_key

    async def upload_file(self, file_obj: BinaryIO, destination_key: str, content_type: str) -> str:
        return await asyncio.to_thread(self._upload_file_sync, file_obj, destination_key, content_type)

    def _upload_file_from_path_sync(self, local_source_path: str, destination_key: str, content_type: str) -> str:
        self.client.upload_file(
            local_source_path,
            self.bucket_name,
            destination_key,
            ExtraArgs={"ContentType": content_type},
        )
        return destination_key

    async def upload_file_from_path(self, local_source_path: str, destination_key: str, content_type: str) -> str:
        return await asyncio.to_thread(self._upload_file_from_path_sync, local_source_path, destination_key, content_type)

    def _download_file_sync(self, source_key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket_name, Key=source_key)
        return response["Body"].read()

    async def download_file(self, source_key: str) -> bytes:
        return await asyncio.to_thread(self._download_file_sync, source_key)

    def _download_file_to_path_sync(self, source_key: str, local_destination_path: str) -> None:
        self.client.download_file(self.bucket_name, source_key, local_destination_path)

    async def download_file_to_path(self, source_key: str, local_destination_path: str) -> None:
        await asyncio.to_thread(self._download_file_to_path_sync, source_key, local_destination_path)

    def _delete_file_sync(self, source_key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=source_key)
            return True
        except ClientError as e:
            logger.warning(f"Failed to delete object {source_key}: {e}")
            return False

    async def delete_file(self, source_key: str) -> bool:
        return await asyncio.to_thread(self._delete_file_sync, source_key)

    def _check_object_exists_sync(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise

    async def check_object_exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._check_object_exists_sync, key)

    def _get_object_metadata_sync(self, key: str) -> dict[str, str | int]:
        try:
            res = self.client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                "content_length": int(res.get("ContentLength", 0)),
                "content_type": str(res.get("ContentType", "application/octet-stream")),
                "etag": str(res.get("ETag", "")).strip('"'),
            }
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey"):
                raise SonaraException(
                    status_code=404,
                    detail=f"Storage object '{key}' not found.",
                    code="OBJECT_NOT_FOUND",
                ) from e
            raise

    async def get_object_metadata(self, key: str) -> dict[str, str | int]:
        return await asyncio.to_thread(self._get_object_metadata_sync, key)

    def _copy_object_sync(self, source_key: str, destination_key: str) -> None:
        copy_source = {"Bucket": self.bucket_name, "Key": source_key}
        self.client.copy_object(
            CopySource=copy_source,
            Bucket=self.bucket_name,
            Key=destination_key,
        )

    async def copy_object(self, source_key: str, destination_key: str) -> None:
        await asyncio.to_thread(self._copy_object_sync, source_key, destination_key)

