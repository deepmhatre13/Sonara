import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_default_values():
    settings = Settings()
    assert settings.PROJECT_NAME == "Sonara — Audio Intelligence & Knowledge Retrieval"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.EMBEDDING_DIMENSION == 384
    assert settings.MAX_AUDIO_UPLOAD_SIZE_MB == 250
    assert "postgresql+asyncpg://" in settings.DATABASE_URL
    assert "postgresql+psycopg2://" in settings.SYNC_DATABASE_URL


def test_settings_custom_database_url():
    custom_url = "postgresql+asyncpg://custom_user:secret@dbhost:5433/custom_db"
    settings = Settings(DATABASE_URL=custom_url)
    assert settings.DATABASE_URL == custom_url


def test_settings_upload_size_validation():
    with pytest.raises(ValidationError):
        Settings(MAX_AUDIO_UPLOAD_SIZE_MB=5000)  # Max allowed is 2000

