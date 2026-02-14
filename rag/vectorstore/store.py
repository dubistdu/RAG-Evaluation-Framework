"""
Vectorstore: FAISS index + metadata for chunk embeddings.
Persist to directory: index.faiss + metadata.json.
Backward compatible: can load legacy vectors.npy + metadata.json (builds FAISS in memory).
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
VECTORS_FILE = "vectors.npy"  # legacy


def _normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize vectors (row-wise). For cosine similarity with FAISS IndexFlatIP."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return (x / norms).astype(np.float32)


class VectorStore:
    """
    FAISS-backed vector store: chunk_id, text, vector.
    Uses IndexFlatIP with normalized vectors (inner product = cosine similarity).
    Persist: index.faiss + metadata.json.
    """

    def __init__(self) -> None:
        self._chunk_ids: List[str] = []
        self._texts: List[str] = []
        self._index: Any = None  # faiss.IndexFlatIP
        self._dim: Optional[int] = None

    def add(self, chunk_id: str, text: str, embedding: List[float]) -> None:
        """Add one chunk."""
        self._chunk_ids.append(chunk_id)
        self._texts.append(text)
        arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if self._dim is None:
            self._dim = arr.shape[1]
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self._dim)
            except ImportError as e:
                raise ImportError("faiss-cpu required. pip install faiss-cpu") from e
        arr_norm = _normalize(arr)
        self._index.add(arr_norm)

    def add_many(self, chunk_ids: List[str], texts: List[str], embeddings: List[List[float]]) -> None:
        """Add multiple chunks."""
        if not (len(chunk_ids) == len(texts) == len(embeddings)):
            raise ValueError("chunk_ids, texts, embeddings must have same length")
        if not chunk_ids:
            return
        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)
        matrix = np.array(embeddings, dtype=np.float32)
        if self._dim is None:
            self._dim = matrix.shape[1]
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self._dim)
            except ImportError as e:
                raise ImportError("faiss-cpu required. pip install faiss-cpu") from e
        matrix_norm = _normalize(matrix)
        self._index.add(matrix_norm)

    def query(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Return top-k (chunk_id, text, score) by cosine similarity (inner product on normalized vectors).
        """
        if self._index is None or len(self._chunk_ids) == 0:
            return []
        k = min(k, len(self._chunk_ids))
        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        q_norm = _normalize(q)
        scores, indices = self._index.search(q_norm, k)
        out: List[Tuple[str, str, float]] = []
        for i in range(indices.shape[1]):
            idx = int(indices[0, i])
            if idx < 0:
                continue
            score = float(scores[0, i])
            if min_score is not None and score < min_score:
                continue
            out.append((self._chunk_ids[idx], self._texts[idx], score))
        return out

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def save(self, directory: Path) -> Path:
        """Write index.faiss and metadata.json to directory."""
        import faiss
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / INDEX_FILE))
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
        """Load from directory. Prefers index.faiss + metadata.json; falls back to vectors.npy + metadata.json (legacy)."""
        import faiss
        directory = Path(directory).resolve()
        store = cls()
        index_path = directory / INDEX_FILE
        meta_path = directory / METADATA_FILE
        vectors_path = directory / VECTORS_FILE

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        store._chunk_ids = [m["chunk_id"] for m in metadata]
        store._texts = [m["text"] for m in metadata]

        if index_path.exists():
            store._index = faiss.read_index(str(index_path))
            store._dim = store._index.d
        elif vectors_path.exists():
            vectors = np.load(vectors_path).astype(np.float32)
            store._dim = vectors.shape[1]
            store._index = faiss.IndexFlatIP(store._dim)
            store._index.add(_normalize(vectors))
            logger.info("Loaded legacy vectors.npy and built FAISS index in memory")
        else:
            raise FileNotFoundError(f"No index found in {directory} (expected {INDEX_FILE} or {VECTORS_FILE})")

        logger.info("Loaded vectorstore from %s (%s chunks)", directory, len(store._chunk_ids))
        return store
