import uuid

from app.chunking.chunker import estimate_tokens, generate_chunks
from app.models.core import SpeakerTurn, TranscriptSegment


def test_estimate_tokens():
    text = "Sonara audio intelligence and knowledge retrieval system."
    tokens = estimate_tokens(text)
    assert tokens >= 7
    assert estimate_tokens("") == 1


def test_deterministic_chunking():
    rec_id = uuid.uuid4()
    segments = [
        TranscriptSegment(
            id=uuid.uuid4(),
            recording_id=rec_id,
            sequence_index=0,
            start_ms=0,
            end_ms=2000,
            text="First sentence of the speech.",
            model_identifier="test",
        ),
        TranscriptSegment(
            id=uuid.uuid4(),
            recording_id=rec_id,
            sequence_index=1,
            start_ms=2100,
            end_ms=4500,
            text="Second sentence detailing financial results.",
            model_identifier="test",
        ),
        TranscriptSegment(
            id=uuid.uuid4(),
            recording_id=rec_id,
            sequence_index=2,
            start_ms=4600,
            end_ms=6000,
            text="Third sentence concluding remarks.",
            model_identifier="test",
        ),
    ]

    speaker_turns = [
        SpeakerTurn(
            id=uuid.uuid4(),
            recording_id=rec_id,
            speaker_label="SPEAKER_00",
            start_ms=0,
            end_ms=3000,
            diarization_model="test",
        ),
        SpeakerTurn(
            id=uuid.uuid4(),
            recording_id=rec_id,
            speaker_label="SPEAKER_01",
            start_ms=3000,
            end_ms=6000,
            diarization_model="test",
        ),
    ]

    # Large max_tokens to group all into one chunk
    chunks = generate_chunks(segments=segments, speaker_turns=speaker_turns, max_tokens=100)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_index == 0
    assert c.start_ms == 0
    assert c.end_ms == 6000
    assert len(c.source_segment_ids) == 3
    # Provenance contains both speakers across timeline
    assert "SPEAKER_00" in c.speaker_labels
    assert "SPEAKER_01" in c.speaker_labels


def test_chunking_with_overlap():
    rec_id = uuid.uuid4()
    segments = [
        TranscriptSegment(
            id=uuid.uuid4(),
            recording_id=rec_id,
            sequence_index=i,
            start_ms=i * 2000,
            end_ms=(i + 1) * 2000,
            text=f"This is segment number {i} with several words for token boundary testing.",
            model_identifier="test",
        )
        for i in range(10)
    ]

    # Force small chunks to trigger overlap splitting
    chunks = generate_chunks(segments=segments, max_tokens=30, overlap_tokens=15)
    assert len(chunks) > 1
    for c in chunks:
        assert c.start_ms >= 0
        assert c.end_ms > c.start_ms
        assert len(c.source_segment_ids) > 0
