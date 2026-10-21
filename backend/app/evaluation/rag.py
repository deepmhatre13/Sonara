from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.rag import AnswerStatus, ValidatedCitation


@dataclass
class RAGEvaluationMetrics:
    citation_precision: float
    citation_recall: float
    citation_f1: float
    groundedness_score: float
    abstention_accuracy: float
    has_hallucinated_citations: bool


def evaluate_rag_answer(
    answer: str,
    status: AnswerStatus,
    predicted_citations: Sequence[ValidatedCitation],
    gold_chunk_ids: set[str],
    is_answerable: bool,
    forbidden_tokens: Sequence[str] | None = None,
) -> RAGEvaluationMetrics:
    """
    Evaluates RAG generation output across:
    1. Abstention accuracy: Did the model abstain when it was unanswerable, and answer when it was answerable?
    2. Citation Precision/Recall/F1: Model citations vs gold relevant chunk IDs.
    3. Groundedness Score: Heuristic checking citation coverage and absence of forbidden/hallucinated tokens.
    4. Hallucination check: Whether predicted citations cite non-gold chunks or ungrounded claims.
    """
    # 1. Abstention accuracy
    is_abstained = status in (
        AnswerStatus.INSUFFICIENT_EVIDENCE,
        AnswerStatus.REFUSAL,
    )
    if not is_answerable:
        abstention_acc = 1.0 if is_abstained else 0.0
    else:
        abstention_acc = 1.0 if not is_abstained else 0.0

    # 2. Citation metrics
    predicted_ids = {str(c.chunk_id) for c in predicted_citations}
    
    if not predicted_ids and not gold_chunk_ids:
        cit_precision = 1.0
        cit_recall = 1.0
        cit_f1 = 1.0
    elif not predicted_ids or not gold_chunk_ids:
        cit_precision = 0.0 if predicted_ids else 1.0
        cit_recall = 0.0 if gold_chunk_ids else 1.0
        cit_f1 = 0.0
    else:
        true_positives = len(predicted_ids & gold_chunk_ids)
        cit_precision = true_positives / len(predicted_ids)
        cit_recall = true_positives / len(gold_chunk_ids)
        if cit_precision + cit_recall > 0:
            cit_f1 = 2 * (cit_precision * cit_recall) / (cit_precision + cit_recall)
        else:
            cit_f1 = 0.0

    # 3. Hallucination check
    has_hallucinated = False
    if gold_chunk_ids and any(pid not in gold_chunk_ids for pid in predicted_ids):
        has_hallucinated = True

    # 4. Groundedness score (0.0 to 1.0)
    # Penalize if answer claims info but has 0 citations, or includes forbidden hallucination tokens
    if is_abstained:
        groundedness = 1.0 if not is_answerable else 0.0
    else:
        score = 0.5 + (0.5 * cit_precision)
        if forbidden_tokens:
            for token in forbidden_tokens:
                if token.lower() in answer.lower():
                    score = max(0.0, score - 0.5)
        if not predicted_citations and is_answerable:
            score = 0.0
        groundedness = score

    return RAGEvaluationMetrics(
        citation_precision=round(cit_precision, 4),
        citation_recall=round(cit_recall, 4),
        citation_f1=round(cit_f1, 4),
        groundedness_score=round(groundedness, 4),
        abstention_accuracy=round(abstention_acc, 4),
        has_hallucinated_citations=has_hallucinated,
    )

