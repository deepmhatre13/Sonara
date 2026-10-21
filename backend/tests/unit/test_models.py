import uuid

from app.models.core import JobStatus, ProcessingJob, Recording, User


def test_user_model_instantiation():
    user = User(
        id=uuid.uuid4(),
        email="researcher@example.com",
        hashed_password="hashed_pw_secret",
        is_active=True,
    )
    assert user.email == "researcher@example.com"
    assert user.is_active is True
    assert user.is_superuser is False


def test_recording_and_processing_job_models():
    user_id = uuid.uuid4()
    recording_id = uuid.uuid4()

    recording = Recording(
        id=recording_id,
        user_id=user_id,
        title="Quarterly Review Audio",
        audio_s3_key="recordings/2026/01/sample.wav",
        file_size_bytes=10485760,
        mime_type="audio/wav",
        duration_ms=60000,
        sample_rate=16000,
        channels=1,
    )
    assert recording.title == "Quarterly Review Audio"
    assert recording.duration_ms == 60000

    job = ProcessingJob(
        id=uuid.uuid4(),
        recording_id=recording_id,
        task_name="asr.transcription",
        status=JobStatus.PENDING,
    )
    assert job.status == JobStatus.PENDING
    assert job.retry_count == 0

