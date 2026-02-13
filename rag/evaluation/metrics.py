"""
Retrieval evaluation metrics: Recall@K, MRR, Precision@K.
"""

from typing import List, Set


def recall_at_k(gold_chunk_ids: List[str], retrieved_chunk_ids: List[str], k: int) -> float:
    """
    Recall@K: 1 if any gold chunk appears in the top-K retrieved, else 0.
    """
    if not gold_chunk_ids:
        return 0.0
    gold = set(gold_chunk_ids)
    top_k = retrieved_chunk_ids[:k]
    return 1.0 if gold & set(top_k) else 0.0


def precision_at_k(gold_chunk_ids: List[str], retrieved_chunk_ids: List[str], k: int) -> float:
    """
    Precision@K: (number of gold in top-K) / k.
    """
    if k <= 0:
        return 0.0
    gold = set(gold_chunk_ids)
    top_k = retrieved_chunk_ids[:k]
    hits = len(gold & set(top_k))
    return hits / k


def reciprocal_rank(gold_chunk_ids: List[str], retrieved_chunk_ids: List[str]) -> float:
    """
    Reciprocal rank: 1/rank of first gold chunk in retrieved list, or 0 if none.
    """
    if not gold_chunk_ids:
        return 0.0
    gold = set(gold_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    results: List[tuple],
    k_values: List[int] = None,
) -> dict:
    """
    results: list of (gold_chunk_ids, retrieved_chunk_ids) per question.
    Returns dict with Recall@k, Precision@k, MRR.
    """
    if not results:
        return {"n_questions": 0}
    k_values = k_values or [1, 5, 10]
    n = len(results)
    out = {"n_questions": n}

    for k in k_values:
        recall_sum = sum(recall_at_k(gold, retrieved, k) for gold, retrieved in results)
        prec_sum = sum(precision_at_k(gold, retrieved, k) for gold, retrieved in results)
        out[f"recall@{k}"] = recall_sum / n
        out[f"precision@{k}"] = prec_sum / n

    mrr_sum = sum(reciprocal_rank(gold, retrieved) for gold, retrieved in results)
    out["mrr"] = mrr_sum / n
    return out
