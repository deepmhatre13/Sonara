import math
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float


def calculate_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    graded_relevance: dict[str, float] | None = None,
    k: int = 10,
) -> EvaluationMetrics:
    """
    Computes standard Information Retrieval metrics:
    - Recall@K: Proportion of relevant documents retrieved in top K.
    - Precision@K: Proportion of top K retrieved documents that are relevant.
    - MRR (Mean Reciprocal Rank): Reciprocal rank of the first relevant document.
    - nDCG@K (normalized Discounted Cumulative Gain): Considers graded relevance position.
    """
    if not relevant_ids:
        return EvaluationMetrics(recall_at_k=0.0, precision_at_k=0.0, mrr=0.0, ndcg_at_k=0.0)

    top_k_retrieved = retrieved_ids[:k]

    # Hits in top K
    hits = [doc_id for doc_id in top_k_retrieved if doc_id in relevant_ids]
    recall = len(hits) / len(relevant_ids)
    precision = len(hits) / k if k > 0 else 0.0

    # MRR calculation
    mrr = 0.0
    for idx, doc_id in enumerate(top_k_retrieved, start=1):
        if doc_id in relevant_ids:
            mrr = 1.0 / idx
            break

    # nDCG calculation
    rel_map = graded_relevance or dict.fromkeys(relevant_ids, 1.0)
    dcg = 0.0
    for idx, doc_id in enumerate(top_k_retrieved, start=1):
        rel = rel_map.get(doc_id, 0.0)
        dcg += rel / math.log2(idx + 1)

    # Ideal DCG (sorted true relevance descending)
    ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(ideal_rels, start=1))

    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

    return EvaluationMetrics(
        recall_at_k=round(recall, 4),
        precision_at_k=round(precision, 4),
        mrr=round(mrr, 4),
        ndcg_at_k=round(ndcg, 4),
    )
