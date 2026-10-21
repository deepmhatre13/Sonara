
from app.diarization.interface import SpeakerTurn
from app.transcription.interface import (
    TranscriptSegmentResult,
)


def test_transcript_segment_result_validation():
    seg = TranscriptSegmentResult(
        start_ms=0,
        end_ms=1500,
        text="Hello world",
        confidence=0.98,
    )
    assert seg.start_ms == 0
    assert seg.end_ms == 1500
    assert seg.text == "Hello world"
    assert seg.confidence == 0.98


def test_speaker_turn_validation():
    turn = SpeakerTurn(
        start_ms=500,
        end_ms=2500,
        speaker_label="SPEAKER_00",
        confidence=0.95,
    )
    assert turn.speaker_label == "SPEAKER_00"
    assert turn.start_ms == 500
    assert turn.end_ms == 2500


def test_transcribe_task_registration():
    from app.core.celery_app import celery_app
    assert "sonara.speech.transcribe_and_diarize" in celery_app.tasks
    task = celery_app.tasks["sonara.speech.transcribe_and_diarize"]
    assert task.queue == "asr"
