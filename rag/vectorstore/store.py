"""
Vectorstore: persist and query chunk embeddings (no external DB).
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
METADATA_FILE = "metadata.json"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Single query to single vector."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _cosine_similarities(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Query (d,) vs matrix (n, d) -> (n,) similarities."""
    query = np.asarray(query, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    dots = np.dot(vectors, query)
    norms_q = np.linalg.norm(query)
    norms_v = np.linalg.norm(vectors, axis=1)
    norms_v = np.where(norms_v == 0, 1e-10, norms_v)
    return (dots / (norms_q * norms_v)).astype(np.float64)


class VectorStore:
    """
    In-memory vector store: chunk_id, text, vector.
    Persist to directory: vectors.npy + metadata.json.
    """

    def __init__(self) -> None:
        self._chunk_ids: List[str] = []
        self._texts: List[str] = []
        self._vectors: Optional[np.ndarray] = None  # (n, d)

    def add(self, chunk_id: str, text: str, embedding: List[float]) -> None:
        """Add one chunk."""
        self._chunk_ids.append(chunk_id)
        self._texts.append(text)
        arr = np.array(embedding, dtype=np.float64)
        if self._vectors is None:
            self._vectors = arr.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, arr.reshape(1, -1)])

    def add_many(self, chunk_ids: List[str], texts: List[str], embeddings: List[List[float]]) -> None:
        """Add multiple chunks."""
        if not (len(chunk_ids) == len(texts) == len(embeddings)):
            raise ValueError("chunk_ids, texts, embeddings must have same length")
        for cid, text, emb in zip(chunk_ids, texts, embeddings):
            self.add(cid, text, emb)

    def query(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Return top-k (chunk_id, text, score) by cosine similarity.
        """
        if self._vectors is None or len(self._chunk_ids) == 0:
            return []
        k = min(k, len(self._chunk_ids))
        sims = _cosine_similarities(query_embedding, self._vectors)
        top_indices = np.argsort(sims)[::-1][:k]
        out: List[Tuple[str, str, float]] = []
        for idx in top_indices:
            score = float(sims[idx])
            if min_score is not None and score < min_score:
                continue
            out.append((self._chunk_ids[idx], self._texts[idx], score))
        return out

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def save(self, directory: Path) -> Path:
        """Write vectors.npy and metadata.json to directory."""
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(directory / VECTORS_FILE, self._vectors)
        metadata = [
            {"chunk_id": cid, "text": text}
            for cid, text in zip(self._chunk_ids, self._texts)
        ]
        with open(directory / METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Saved vectorstore to %s (%s chunks)", directory, len(self._chunk_ids))
        return directory

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Load from directory (vectors.npy + metadata.json)."""
        directory = Path(directory).resolve()
        store = cls()
        vectors_path = directory / VECTORS_FILE
        meta_path = directory / METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        store._chunk_ids = [m["chunk_id"] for m in metadata]
        store._texts = [m["text"] for m in metadata]
        if vectors_path.exists():
            store._vectors = np.load(vectors_path)
        else:
            store._vectors = None
        logger.info("Loaded vectorstore from %s (%s chunks)", directory, len(store._chunk_ids))
        return store
