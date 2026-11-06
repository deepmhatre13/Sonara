import io

import pytest

from app.core.exceptions import AudioValidationError, EntityNotFoundError
from app.storage.local import LocalAudioStorage


@pytest.fixture
def temp_local_storage(tmp_path):
    storage = LocalAudioStorage(base_dir=str(tmp_path / "objects"))
    return storage


@pytest.mark.asyncio
async def test_local_storage_upload_download(temp_local_storage):
    data = b"Windows-Native Sonara Audio Storage Content"
    key = "recordings/123/audio.wav"

    # Upload
    uploaded_key = await temp_local_storage.upload_file(io.BytesIO(data), key, "audio/wav")
    assert uploaded_key == key
    assert await temp_local_storage.check_object_exists(key) is True

    # Download
    downloaded = await temp_local_storage.download_file(key)
    assert downloaded == data

    # Metadata
    meta = await temp_local_storage.get_object_metadata(key)
    assert meta["content_length"] == len(data)
    assert meta["content_type"] == "audio/wav"

    # Delete
    deleted = await temp_local_storage.delete_file(key)
    assert deleted is True
    assert await temp_local_storage.check_object_exists(key) is False


@pytest.mark.asyncio
async def test_local_storage_path_traversal_protection(temp_local_storage):
    # Path traversal attempts must raise AudioValidationError
    with pytest.raises(AudioValidationError):
        await temp_local_storage.upload_file(io.BytesIO(b"evil"), "../outside.wav", "audio/wav")

    with pytest.raises(AudioValidationError):
        await temp_local_storage.download_file("../../windows/system32/cmd.exe")


@pytest.mark.asyncio
async def test_local_storage_missing_object_raises_entity_not_found(temp_local_storage):
    with pytest.raises(EntityNotFoundError):
        await temp_local_storage.download_file("nonexistent/key.wav")

