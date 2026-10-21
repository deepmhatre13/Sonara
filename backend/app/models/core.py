import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.database.types import FullTextSearchType, PgVector


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecordingStatus(str, enum.Enum):
    AWAITING_UPLOAD = "awaiting_upload"
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    NORMALIZING = "normalizing"
    INGESTION_COMPLETE = "ingestion_complete"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    SPEECH_COMPLETE = "speech_complete"
    INDEXING = "indexing"
    SEARCHABLE = "searchable"
    DELETING = "deleting"
    FAILED = "failed"
    CANCELLED = "cancelled"



class User(Base):
    """
    User entity representing registered system users and resource owners.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(default=False, nullable=False)

    recordings: Mapped[list["Recording"]] = relationship(
        "Recording", back_populates="owner", cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "is_active" not in kwargs:
            self.is_active = True
        if "is_superuser" not in kwargs:
            self.is_superuser = False


class Recording(Base):
    """
    Recording entity representing ingested audio files and media metadata.
    All media timestamps in derived tables use integer milliseconds.
    """

    __tablename__ = "recordings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    audio_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    normalized_s3_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    status: Mapped[RecordingStatus] = mapped_column(
        Enum(RecordingStatus, name="recording_status_enum", native_enum=False),
        default=RecordingStatus.AWAITING_UPLOAD,
        nullable=False,
        index=True,
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    normalization_profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    owner: Mapped["User"] = relationship("User", back_populates="recordings")
    jobs: Mapped[list["ProcessingJob"]] = relationship(
        "ProcessingJob", back_populates="recording", cascade="all, delete-orphan"
    )
    upload_session: Mapped["UploadSession | None"] = relationship(
        "UploadSession", back_populates="recording", uselist=False, cascade="all, delete-orphan"
    )
    transcript_segments: Mapped[list["TranscriptSegment"]] = relationship(
        "TranscriptSegment",
        back_populates="recording",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.sequence_index",
    )
    speaker_turns: Mapped[list["SpeakerTurn"]] = relationship(
        "SpeakerTurn",
        back_populates="recording",
        cascade="all, delete-orphan",
        order_by="SpeakerTurn.start_ms",
    )
    chunks: Mapped[list["TranscriptChunk"]] = relationship(
        "TranscriptChunk",
        back_populates="recording",
        cascade="all, delete-orphan",
        order_by="TranscriptChunk.chunk_index",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "status" not in kwargs:
            self.status = RecordingStatus.AWAITING_UPLOAD
        if "file_size_bytes" not in kwargs:
            self.file_size_bytes = 0


class UploadSession(Base):
    """
    Tracks an upload authorization lifecycle and state machine.
    """

    __tablename__ = "upload_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    staging_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    declared_mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expected_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_finalized: Mapped[bool] = mapped_column(default=False, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="upload_session")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "is_finalized" not in kwargs:
            self.is_finalized = False


class TranscriptSegment(Base):
    """
    Persisted timestamped speech transcript segment.
    All timestamps strictly use integer milliseconds.
    """

    __tablename__ = "transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    model_identifier: Mapped[str] = mapped_column(String(128), nullable=False)
    transcript_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="transcript_segments")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "transcript_version" not in kwargs:
            self.transcript_version = 1


class SpeakerTurn(Base):
    """
    Persisted anonymous speaker diarization turn.
    Timestamps use integer milliseconds. Overlapping turns are representable.
    Speaker labels are strictly anonymous and scoped to the recording.
    """

    __tablename__ = "speaker_turns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    speaker_label: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    diarization_model: Mapped[str] = mapped_column(String(128), nullable=False)
    turn_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="speaker_turns")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "turn_version" not in kwargs:
            self.turn_version = 1


class TranscriptChunk(Base):
    """
    Persisted deterministic retrieval unit.
    Contains dense vector embeddings and PostgreSQL TSVECTOR full-text search representation.
    Preserves exact provenance to source transcript segments and speaker turns.
    """

    __tablename__ = "transcript_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    source_segment_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    speaker_labels: Mapped[list[str]] = mapped_column(JSONB, nullable=False)

    # Dense vector representation (BAAI/bge-small-en-v1.5 dimension: 384)
    embedding: Mapped[list[float] | None] = mapped_column(PgVector(384), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False, default=384)

    # Lexical full-text search tsvector column
    tsv: Mapped[str | None] = mapped_column(FullTextSearchType, nullable=True)

    transcript_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    chunking_version: Mapped[str] = mapped_column(String(32), default="v1.0", nullable=False)
    index_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False, index=True)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="chunks")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "transcript_version" not in kwargs:
            self.transcript_version = 1
        if "chunking_version" not in kwargs:
            self.chunking_version = "v1.0"
        if "index_version" not in kwargs:
            self.index_version = 1
        if "is_active" not in kwargs:
            self.is_active = True
        if "embedding_dimension" not in kwargs:
            self.embedding_dimension = 384


class ProcessingJob(Base):
    """
    Durable processing state tracking background pipeline execution.
    PostgreSQL is the authoritative store for processing state.
    """

    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status_enum", native_enum=False),
        default=JobStatus.PENDING,
        nullable=False,
        index=True,
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="jobs")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "status" not in kwargs:
            self.status = JobStatus.PENDING
        if "retry_count" not in kwargs:
            self.retry_count = 0
