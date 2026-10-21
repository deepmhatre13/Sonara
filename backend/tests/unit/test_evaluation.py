from app.evaluation.retrieval import calculate_retrieval_metrics


def test_calculate_retrieval_metrics():
    # Top 5 retrieved items
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc2", "doc5"}  # 2 relevant docs total

    metrics = calculate_retrieval_metrics(retrieved_ids=retrieved, relevant_ids=relevant, k=5)

    # Both relevant docs are in top 5 -> Recall@5 = 2/2 = 1.0
    assert metrics.recall_at_k == 1.0
    # 2 hits out of 5 -> Precision@5 = 2/5 = 0.4
    assert metrics.precision_at_k == 0.4
    # First relevant doc is doc2 at rank 2 -> MRR = 1/2 = 0.5
    assert metrics.mrr == 0.5
    # nDCG should be positive and <= 1.0
    assert 0.0 < metrics.ndcg_at_k <= 1.0


def test_metrics_empty_relevant():
    metrics = calculate_retrieval_metrics(retrieved_ids=["doc1"], relevant_ids=set(), k=5)
    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg_at_k == 0.0
