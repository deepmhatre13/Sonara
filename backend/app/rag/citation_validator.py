import re

from app.rag.evidence_selector import SelectedEvidenceItem
from app.schemas.rag import ValidatedCitation


def validate_citations(
    cited_labels: list[str],
    evidence_items: list[SelectedEvidenceItem],
    answer_text: str = "",
) -> list[ValidatedCitation]:
    """
    Validates model-cited evidence labels against the authoritative evidence set.
    Extracts explicit labels from both the citations list and text brackets (e.g. [E1]).
    Resolves authoritative timestamps, chunk IDs, and speaker labels from persisted evidence.
    Rejects any fabricated or out-of-scope identifiers.
    """
    # Build lookup map: "[E1]" -> SelectedEvidenceItem
    evidence_map = {item.label: item for item in evidence_items}

    # Normalize cited labels from structured output and inline text brackets
    found_labels = set()
    for lbl in cited_labels:
        clean = lbl.strip().upper()
        if not clean.startswith("["):
            clean = f"[{clean}]"
        found_labels.add(clean)

    # Also detect inline [E#] citations within the answer text
    inline_matches = re.findall(r"\[E\d+\]", answer_text, re.IGNORECASE)
    for m in inline_matches:
        found_labels.add(m.upper())

    validated: list[ValidatedCitation] = []
    seen_chunks = set()

    # Sort labels deterministically ([E1], [E2], ...)
    sorted_labels = sorted(found_labels)

    for label in sorted_labels:
        if label in evidence_map:
            item = evidence_map[label]
            chunk_id = item.evidence.chunk_id
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                validated.append(
                    ValidatedCitation(
                        evidence_id=label,
                        chunk_id=chunk_id,
                        recording_id=item.evidence.recording_id,
                        start_ms=item.evidence.start_ms,
                        end_ms=item.evidence.end_ms,
                        speaker_labels=item.evidence.speaker_labels,
                        exact_text=item.evidence.text,
                    )
                )

    return validated

