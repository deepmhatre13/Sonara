from dataclasses import dataclass

from app.retrieval.hybrid import RetrievedEvidence


@dataclass(frozen=True)
class SelectedEvidenceItem:
    label: str  # e.g. "[E1]"
    evidence: RetrievedEvidence


def select_evidence(
    candidates: list[RetrievedEvidence],
    max_items: int = 5,
    max_context_tokens: int = 3000,
) -> list[SelectedEvidenceItem]:
    """
    Selects, bounds, and numbers evidence chunks into deterministic identifiers [E1], [E2], ...
    Enforces maximum item budget and token limits while preserving source provenance.
    """
    if not candidates:
        return []

    selected: list[SelectedEvidenceItem] = []
    accumulated_tokens = 0

    for _idx, cand in enumerate(candidates):
        if len(selected) >= max_items:
            break

        cand_tok = cand.token_count or max(1, len(cand.text.split()))
        if accumulated_tokens + cand_tok > max_context_tokens and selected:
            # Context token budget reached
            break

        label = f"[E{len(selected) + 1}]"
        selected.append(SelectedEvidenceItem(label=label, evidence=cand))
        accumulated_tokens += cand_tok

    return selected

