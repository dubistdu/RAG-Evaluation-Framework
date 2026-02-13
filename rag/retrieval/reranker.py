"""
Rerank candidate chunks with a cross-encoder (query + passage scored together).
Uses sentence-transformers; install with: pip install sentence-transformers
"""

from typing import List, Tuple


class Reranker:
    """Cross-encoder reranker; loads model once and reranks (query, candidates) to top_k chunk_ids."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for reranking. pip install sentence-transformers"
            ) from e
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> List[str]:
        """
        Rerank (chunk_id, chunk_text) pairs by relevance to the query.
        Returns ordered list of chunk_ids (best first), length at most top_k.
        """
        if not candidates:
            return []
        pairs = [(query, text) for _cid, text in candidates]
        scores = self._model.predict(pairs)
        indexed = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in indexed[:top_k]]
