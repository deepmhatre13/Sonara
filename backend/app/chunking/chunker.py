import re
from dataclasses import dataclass

from app.core.config import settings
from app.models.core import SpeakerTurn, TranscriptSegment


@dataclass(frozen=True)
class ChunkProvenance:
    chunk_index: int
    start_ms: int
    end_ms: int
    text: str
    token_count: int
    source_segment_ids: list[str]
    speaker_labels: list[str]
    chunking_version: str = "v1.0"


def estimate_tokens(text: str) -> int:
    """
    Estimates token count conservatively using whitespace and punctuation word-splitting.
    Avoids requiring a heavy LLM tokenizer while remaining token-length aware (~1.3 tokens per word).
    """
    words = re.findall(r"\b\w+\b|[^\w\s]", text)
    return max(1, int(len(words) * 1.3))


def generate_chunks(
    segments: list[TranscriptSegment],
    speaker_turns: list[SpeakerTurn] | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    chunking_version: str = "v1.0",
) -> list[ChunkProvenance]:
    """
    Produces deterministic, token-aware transcript chunks preserving segment provenance and speaker attribution.
    Groups adjacent segments by timeline and sequence index, respecting sentence/segment boundaries.
    """
    if not segments:
        return []

    max_tok = max_tokens or settings.CHUNK_MAX_TOKENS
    overlap_tok = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS

    # Sort segments strictly by sequence_index and start_ms
    sorted_segments = sorted(segments, key=lambda s: (s.sequence_index, s.start_ms))

    # Helper function to find overlapping speaker labels for a time interval [start_ms, end_ms]
    def get_speakers_for_interval(start_ms: int, end_ms: int) -> list[str]:
        if not speaker_turns:
            return []
        speakers = set()
        for turn in speaker_turns:
            # Overlap check: max(start1, start2) < min(end1, end2)
            if max(start_ms, turn.start_ms) < min(end_ms, turn.end_ms):
                speakers.add(turn.speaker_label)
        return sorted(speakers)

    chunks: list[ChunkProvenance] = []
    current_segments: list[TranscriptSegment] = []
    current_tokens = 0
    chunk_idx = 0

    i = 0
    while i < len(sorted_segments):
        seg = sorted_segments[i]
        seg_tok = estimate_tokens(seg.text)

        # If adding this segment exceeds max_tokens and we already have segments, finalize current chunk
        if current_segments and (current_tokens + seg_tok > max_tok):
            chunk_text = " ".join(s.text.strip() for s in current_segments)
            c_start = current_segments[0].start_ms
            c_end = current_segments[-1].end_ms
            c_speakers = get_speakers_for_interval(c_start, c_end)

            chunks.append(
                ChunkProvenance(
                    chunk_index=chunk_idx,
                    start_ms=c_start,
                    end_ms=c_end,
                    text=chunk_text,
                    token_count=current_tokens,
                    source_segment_ids=[str(s.id) for s in current_segments],
                    speaker_labels=c_speakers,
                    chunking_version=chunking_version,
                )
            )
            chunk_idx += 1

            # Handle overlap: retain trailing segments whose token count is <= overlap_tok
            overlap_accum = 0
            new_current = []
            for rev_seg in reversed(current_segments):
                stok = estimate_tokens(rev_seg.text)
                if overlap_accum + stok <= overlap_tok:
                    overlap_accum += stok
                    new_current.insert(0, rev_seg)
                else:
                    break

            current_segments = new_current
            current_tokens = sum(estimate_tokens(s.text) for s in current_segments)

        current_segments.append(seg)
        current_tokens += seg_tok
        i += 1

    # Finalize any remaining segments
    if current_segments:
        chunk_text = " ".join(s.text.strip() for s in current_segments)
        c_start = current_segments[0].start_ms
        c_end = current_segments[-1].end_ms
        c_speakers = get_speakers_for_interval(c_start, c_end)

        chunks.append(
            ChunkProvenance(
                chunk_index=chunk_idx,
                start_ms=c_start,
                end_ms=c_end,
                text=chunk_text,
                token_count=current_tokens,
                source_segment_ids=[str(s.id) for s in current_segments],
                speaker_labels=c_speakers,
                chunking_version=chunking_version,
            )
        )

    return chunks
