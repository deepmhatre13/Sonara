from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItem:
    id: str
    rank: int
    score: float | None = None
    source: str = ""


@dataclass(frozen=True)
class FusedResult:
    id: str
    rrf_score: float
    source_ranks: dict[str, int]
    source_scores: dict[str, float | None]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RankedItem]],
    k: int = 60,
) -> list[FusedResult]:
    """
    Computes Reciprocal Rank Fusion (RRF) across multiple independent retriever rankings.
    Formula: RRF_score(d) = Σᵣ 1 / (k + rankᵣ(d))
    where rankᵣ(d) is 1-based index in retriever r.
    Preserves provenance of source ranks and diagnostic scores.
    """
    if k <= 0:
        raise ValueError("RRF constant k must be a positive integer.")

    scores: dict[str, float] = defaultdict(float)
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    source_scores: dict[str, dict[str, float | None]] = defaultdict(dict)

    for source_name, items in ranked_lists.items():
        for item in items:
            # item.rank is 1-based rank
            rank_val = item.rank if item.rank >= 1 else 1
            scores[item.id] += 1.0 / (k + rank_val)
            source_ranks[item.id][source_name] = rank_val
            source_scores[item.id][source_name] = item.score

    # Sort deterministically by RRF score descending, tie-breaking by ID
    sorted_ids = sorted(scores.keys(), key=lambda doc_id: (-scores[doc_id], str(doc_id)))

    fused_results = [
        FusedResult(
            id=doc_id,
            rrf_score=scores[doc_id],
            source_ranks=source_ranks[doc_id],
            source_scores=source_scores[doc_id],
        )
        for doc_id in sorted_ids
    ]

    return fused_results

