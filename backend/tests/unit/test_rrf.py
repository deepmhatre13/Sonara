import pytest

from app.retrieval.rrf import RankedItem, reciprocal_rank_fusion


def test_rrf_scoring_and_deduplication():
    # Retriever 1: docA at rank 1, docB at rank 2
    # Retriever 2: docB at rank 1, docC at rank 2
    list_sem = [
        RankedItem(id="docA", rank=1, score=0.95),
        RankedItem(id="docB", rank=2, score=0.85),
    ]
    list_kw = [
        RankedItem(id="docB", rank=1, score=12.5),
        RankedItem(id="docC", rank=2, score=8.1),
    ]

    fused = reciprocal_rank_fusion({"semantic": list_sem, "keyword": list_kw}, k=60)
    assert len(fused) == 3

    # docB appeared in both lists (rank 2 and rank 1): 1/(60+2) + 1/(60+1)
    # docA appeared once (rank 1): 1/(60+1)
    # docC appeared once (rank 2): 1/(60+2)
    # Therefore docB must be ranked 1st
    assert fused[0].id == "docB"
    assert fused[0].source_ranks["semantic"] == 2
    assert fused[0].source_ranks["keyword"] == 1

    expected_score_b = (1.0 / 62) + (1.0 / 61)
    assert pytest.approx(fused[0].rrf_score, 1e-6) == expected_score_b


def test_rrf_deterministic_tie_breaking():
    # Two documents with exact same ranks across lists
    list1 = [RankedItem(id="docZ", rank=1), RankedItem(id="docA", rank=2)]
    list2 = [RankedItem(id="docA", rank=1), RankedItem(id="docZ", rank=2)]

    fused = reciprocal_rank_fusion({"r1": list1, "r2": list2}, k=60)
    assert len(fused) == 2
    assert fused[0].rrf_score == fused[1].rrf_score
    # Tie broken alphabetically: docA before docZ
    assert fused[0].id == "docA"
    assert fused[1].id == "docZ"


def test_rrf_invalid_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({}, k=0)
