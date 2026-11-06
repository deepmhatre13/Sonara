from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Core Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    PROJECT_NAME: str = "Sonara — Audio Intelligence & Knowledge Retrieval"
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "insecure_dev_secret_key_needs_override_in_production_min32chars"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "sonara_user"
    POSTGRES_PASSWORD: str = "sonara_password"
    POSTGRES_DB: str = "sonara_db"
    DATABASE_URL: str | None = None
    SYNC_DATABASE_URL: str | None = None

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_async_db_url(cls, v: str | None, info: Any) -> str:
        if isinstance(v, str) and v.strip():
            return v
        data = info.data
        user = data.get("POSTGRES_USER", "sonara_user")
        pwd = data.get("POSTGRES_PASSWORD", "sonara_password")
        srv = data.get("POSTGRES_SERVER", "localhost")
        port = data.get("POSTGRES_PORT", 5432)
        db = data.get("POSTGRES_DB", "sonara_db")
        return f"postgresql+asyncpg://{user}:{pwd}@{srv}:{port}/{db}"

    @field_validator("SYNC_DATABASE_URL", mode="before")
    @classmethod
    def assemble_sync_db_url(cls, v: str | None, info: Any) -> str:
        if isinstance(v, str) and v.strip():
            return v
        data = info.data
        user = data.get("POSTGRES_USER", "sonara_user")
        pwd = data.get("POSTGRES_PASSWORD", "sonara_password")
        srv = data.get("POSTGRES_SERVER", "localhost")
        port = data.get("POSTGRES_PORT", 5432)
        db = data.get("POSTGRES_DB", "sonara_db")
        return f"postgresql+psycopg2://{user}:{pwd}@{srv}:{port}/{db}"

    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Object Storage Backend ('local' or 's3')
    STORAGE_BACKEND: str = "local"
    LOCAL_STORAGE_PATH: str = r"D:\Sonara\data\objects"
    MODEL_CACHE_DIR: str = r"D:\Sonara\data\models"

    # Object Storage (S3 / MinIO)
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "sonara-recordings"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False

    # Authentication & Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_ALGORITHM: str = "HS256"

    # AI / Provider Settings (Optional at startup)
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TIMEOUT_SECONDS: int = 30
    LLM_MAX_OUTPUT_TOKENS: int = 1024
    MAX_EVIDENCE_ITEMS: int = 5
    MAX_CONTEXT_TOKENS: int = 3000

    # Audio & Retrieval Parameters
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    ENABLE_RERANKER: bool = True
    RRF_K: int = 60
    CHUNK_MAX_TOKENS: int = 250
    CHUNK_OVERLAP_TOKENS: int = 40
    MAX_AUDIO_UPLOAD_SIZE_MB: int = Field(default=250, ge=1, le=2000)

    # Speech Intelligence Parameters (faster-whisper & pyannote.audio)
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_VAD_FILTER: bool = True
    HUGGINGFACE_TOKEN: str | None = None
    DIARIZATION_MODEL_NAME: str = "pyannote/speaker-diarization-3.1"
    ENABLE_DIARIZATION: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"


settings = Settings()

