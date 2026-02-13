"""
Synthetic question generation for retrieval evaluation.
Generates grounded questions from chunks; no answer generation.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, List, Optional, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

QUESTION_TYPES = ("factual", "analytical", "multi_hop", "definition")
DIFFICULTY_LEVELS = ("easy", "medium", "hard")


class SyntheticQuestion(BaseModel):
    """One synthetic QA entry for retrieval evaluation (question only, no answer)."""

    question_id: str = Field(..., description="Unique id")
    question: str = Field(..., description="Question text")
    gold_chunk_ids: List[str] = Field(..., description="Chunk IDs that contain the answer")
    question_type: str = Field(..., description="factual | analytical | multi_hop | definition")
    difficulty: str = Field(..., description="easy | medium | hard")
    source_config: str = Field(..., description="Chunk config name this was generated from")


class QAConfig(BaseModel):
    """Configuration for synthetic QA generation."""

    questions_per_chunk_min: int = Field(2, ge=1, le=10, description="Min questions per chunk")
    questions_per_chunk_max: int = Field(3, ge=1, le=10, description="Max questions per chunk")
    batch_size: int = Field(5, ge=1, le=20, description="Chunks per LLM call (reduces API calls)")
    max_chunks: Optional[int] = Field(None, ge=1, description="Cap number of chunks to process (None = all)")
    multi_hop_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of questions that are multi-hop")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Min cosine sim for related chunk (multi-hop)")
    max_related_chunks: int = Field(2, ge=1, le=5, description="Max chunks for multi-hop gold set")


# ---------------------------------------------------------------------------
# LLM interface (no hardcoded API calls inside generator)
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Abstract LLM client: generate(prompt) -> response text."""

    def generate(self, prompt: str) -> str:
        ...


class OpenAILLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._client: Any = None
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError as e:
                raise ImportError("openai package required for OpenAILLMClient. pip install openai") from e
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()


class OpenRouterLLMClient:
    """OpenRouter LLM client (same interface)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError as e:
                raise ImportError("openai package required for OpenRouterLLMClient") from e
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Embedding interface (for multi-hop related chunks)
# ---------------------------------------------------------------------------


class EmbeddingClient(Protocol):
    """Abstract embedding client: embed(texts) -> list of vectors."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class OpenAIEmbeddingClient:
    """OpenAI embedding client."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._client: Any = None
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError as e:
                raise ImportError("openai package required for OpenAIEmbeddingClient") from e
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        order = sorted(resp.data, key=lambda d: d.index)
        return [order[i].embedding for i in range(len(order))]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def _build_single_chunk_prompt(chunk_text: str, num_questions: int = 3) -> str:
    """Structured prompt: generate diverse questions grounded only in the chunk."""
    return f"""You are generating evaluation questions for a retrieval system. Given the following document excerpt, generate exactly {num_questions} questions that can be answered using ONLY information from this excerpt.

RULES:
- Each question must be answerable from this text alone. Do not assume external knowledge.
- Vary question types: factual (What is X?), definition (Define X), analytical (Why does X happen?).
- Questions should be clear and specific, not vague.
- Output exactly one question per line. No numbering, no prefixes. One question per line.

DOCUMENT EXCERPT:
---
{chunk_text[:8000]}
---

Output {num_questions} questions, one per line:"""


def _build_batch_chunk_prompt(chunk_texts: List[str], questions_per_chunk: int = 2) -> str:
    """One prompt for multiple chunks: generate N questions per chunk with clear section labels."""
    parts = []
    for i, text in enumerate(chunk_texts, start=1):
        excerpt = (text or "")[:4000]
        parts.append(f"[CHUNK {i}]\n{excerpt}\n")
    combined = "\n".join(parts)
    return f"""You are generating evaluation questions for a retrieval system. Below are {len(chunk_texts)} document excerpts labeled [CHUNK 1], [CHUNK 2], etc.

For EACH chunk, generate exactly {questions_per_chunk} questions that can be answered using ONLY that chunk. Do not mix information across chunks.

RULES:
- Each question must be answerable from its chunk alone. Vary types: factual, definition, analytical.
- Output format: after each chunk label, write exactly {questions_per_chunk} questions, one per line. Then the next [CHUNK N] and its questions.

Example format:
[CHUNK 1]
What is X?
When did Y happen?
[CHUNK 2]
Define Z.
Why does W occur?

---
{combined}
---

Output questions for each chunk in the same [CHUNK N] order, one question per line under each label:"""


def _parse_batch_response(response: str, num_chunks: int) -> List[List[str]]:
    """Parse batch response into questions per chunk: result[chunk_idx] = [q1, q2, ...]."""
    result: List[List[str]] = [[] for _ in range(num_chunks)]
    current = -1
    for line in response.strip().splitlines():
        line = line.strip()
        m = re.match(r"\[CHUNK\s+(\d+)\]", line, re.IGNORECASE)
        if m:
            current = int(m.group(1)) - 1
            if 0 <= current < num_chunks:
                continue
        if current >= 0 and current < num_chunks and line:
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            if len(line) > 10 and (line.endswith("?") or "." in line or "?" in line):
                if not line.endswith("?"):
                    line = line + "?" if not line.endswith(".") else line
                result[current].append(line)
    return result


def _build_multi_hop_prompt(chunk_texts: List[str], num_questions: int = 1) -> str:
    """Prompt to generate a question that requires multiple chunks to answer."""
    combined = "\n\n---\n\n".join(chunk_texts[:3])  # up to 3 chunks
    combined = combined[:12000]
    return f"""You are generating a multi-hop evaluation question for a retrieval system. The question must require information from ALL of the following excerpts together to answer.

RULES:
- The question must not be answerable from any single excerpt alone.
- Use relationships between the excerpts (e.g. "How does X in excerpt 1 relate to Y in excerpt 2?").
- Output exactly one question. No numbering or prefix.

EXCERPTS:
---
{combined}
---

Output exactly one multi-hop question:"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_questions_from_response(response: str) -> List[str]:
    """Extract one question per line; strip numbering and empty lines."""
    questions: List[str] = []
    for line in response.strip().splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if line and line.endswith("?"):
            questions.append(line)
        elif line and len(line) > 15:
            if not line.endswith("?"):
                line = line + "?" if not line.endswith(".") else line
            questions.append(line)
    return questions[:10]


# ---------------------------------------------------------------------------
# Safety filters
# ---------------------------------------------------------------------------


def _is_vague(question: str) -> bool:
    """Reject very short or generic questions."""
    q = question.strip()
    if len(q) < 15:
        return True
    vague_starts = ("what is the", "tell me", "explain", "something", "anything", "everything")
    if q.lower().startswith(vague_starts) and len(q) < 40:
        return True
    return False


def _normalize_for_dedup(question: str) -> str:
    """Normalize question for duplicate detection."""
    return re.sub(r"\s+", " ", question.strip().lower())


def _is_duplicate(question: str, seen_normalized: set[str]) -> bool:
    """Check if question is duplicate of any already seen."""
    return _normalize_for_dedup(question) in seen_normalized


# ---------------------------------------------------------------------------
# Difficulty assignment
# ---------------------------------------------------------------------------


def _assign_difficulty(question: str, question_type: str) -> str:
    """Short direct = easy, explanation = medium, multi-hop = hard."""
    if question_type == "multi_hop":
        return "hard"
    q = question.strip().lower()
    if len(q) < 60 and ("what is" in q or "define" in q or "when did" in q or "how many" in q):
        return "easy"
    if "why" in q or "how does" in q or "explain" in q or "relationship" in q:
        return "medium"
    return "medium"


def _infer_question_type(question: str) -> str:
    """Heuristic: factual | definition | analytical | multi_hop."""
    q = question.lower()
    if "define" in q or "definition" in q or "what is" in q and "mean" in q:
        return "definition"
    if "why" in q or "how does" in q or "explain" in q or "relationship" in q:
        return "analytical"
    if "what is" in q or "when" in q or "where" in q or "how many" in q:
        return "factual"
    return "factual"


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------


def load_chunks_for_config(chunks_dir: Path, config_name: str) -> List[dict]:
    """Load all chunk JSON files under data/chunks/<config_name>/ into one list."""
    config_dir = chunks_dir / config_name
    if not config_dir.is_dir():
        logger.warning("Chunk config dir not found: %s", config_dir)
        return []
    chunks: List[dict] = []
    for path in sorted(config_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                chunks.extend(data)
            else:
                chunks.append(data)
        except Exception as e:
            logger.exception("Failed to load %s: %s", path, e)
    logger.info("Loaded %s chunks for config %s", len(chunks), config_name)
    return chunks


# ---------------------------------------------------------------------------
# Single-chunk question generation
# ---------------------------------------------------------------------------


def _generate_single_chunk_questions(
    chunk: dict,
    config_name: str,
    qa_config: QAConfig,
    llm: LLMClient,
    seen_normalized: set[str],
) -> List[SyntheticQuestion]:
    """Generate 2–3 questions for one chunk; apply safety filters."""
    import random
    text = chunk.get("text") or ""
    chunk_id = chunk.get("chunk_id") or ""
    if not text or not chunk_id:
        return []
    num = random.randint(qa_config.questions_per_chunk_min, qa_config.questions_per_chunk_max)
    prompt = _build_single_chunk_prompt(text, num_questions=num)
    try:
        response = llm.generate(prompt)
    except Exception as e:
        logger.exception("LLM generate failed for chunk %s: %s", chunk_id, e)
        return []
    questions = _parse_questions_from_response(response)
    results: List[SyntheticQuestion] = []
    for q in questions:
        if _is_vague(q):
            continue
        if _is_duplicate(q, seen_normalized):
            continue
        seen_normalized.add(_normalize_for_dedup(q))
        q_type = _infer_question_type(q)
        difficulty = _assign_difficulty(q, q_type)
        results.append(
            SyntheticQuestion(
                question_id=f"q_{uuid.uuid4().hex[:12]}",
                question=q,
                gold_chunk_ids=[chunk_id],
                question_type=q_type,
                difficulty=difficulty,
                source_config=config_name,
            )
        )
    return results


def _generate_batch_questions(
    batch_chunks: List[dict],
    config_name: str,
    qa_config: QAConfig,
    llm: LLMClient,
    seen_normalized: set[str],
) -> List[SyntheticQuestion]:
    """Generate questions for a batch of chunks in one LLM call."""
    import random
    if not batch_chunks:
        return []
    texts = [c.get("text") or "" for c in batch_chunks]
    ids = [c.get("chunk_id") or "" for c in batch_chunks]
    if not all(ids):
        return []
    n_per = random.randint(qa_config.questions_per_chunk_min, qa_config.questions_per_chunk_max)
    prompt = _build_batch_chunk_prompt(texts, questions_per_chunk=n_per)
    try:
        response = llm.generate(prompt)
    except Exception as e:
        logger.exception("LLM batch generate failed for %s chunks: %s", len(batch_chunks), e)
        return []
    per_chunk = _parse_batch_response(response, len(batch_chunks))
    results: List[SyntheticQuestion] = []
    for chunk_idx, questions in enumerate(per_chunk):
        chunk_id = ids[chunk_idx]
        for q in questions:
            if _is_vague(q):
                continue
            if _is_duplicate(q, seen_normalized):
                continue
            seen_normalized.add(_normalize_for_dedup(q))
            q_type = _infer_question_type(q)
            difficulty = _assign_difficulty(q, q_type)
            results.append(
                SyntheticQuestion(
                    question_id=f"q_{uuid.uuid4().hex[:12]}",
                    question=q,
                    gold_chunk_ids=[chunk_id],
                    question_type=q_type,
                    difficulty=difficulty,
                    source_config=config_name,
                )
            )
    return results


# ---------------------------------------------------------------------------
# Multi-hop question generation
# ---------------------------------------------------------------------------


def _find_related_chunks(
    chunk: dict,
    all_chunks: List[dict],
    embedding_client: EmbeddingClient,
    qa_config: QAConfig,
) -> List[dict]:
    """Find chunks semantically related to this one (cosine similarity)."""
    chunk_id = chunk.get("chunk_id")
    text = chunk.get("text") or ""
    if not chunk_id or not text:
        return []
    texts = [c.get("text") or "" for c in all_chunks]
    ids = [c.get("chunk_id") or "" for c in all_chunks]
    try:
        vectors = embedding_client.embed(texts)
    except Exception as e:
        logger.warning("Embedding failed for related-chunk search: %s", e)
        return []
    if not vectors or len(vectors) != len(ids):
        return []
    try:
        idx = ids.index(chunk_id)
    except ValueError:
        return []
    query_vec = vectors[idx]
    sims = [(i, _cosine_similarity(query_vec, vectors[i])) for i in range(len(vectors)) if i != idx]
    sims.sort(key=lambda x: -x[1])
    related: List[dict] = []
    for i, sim in sims[: qa_config.max_related_chunks]:
        if sim >= qa_config.similarity_threshold:
            related.append(all_chunks[i])
    return related


def _generate_multi_hop_question(
    chunk: dict,
    related: List[dict],
    config_name: str,
    llm: LLMClient,
    seen_normalized: set[str],
) -> Optional[SyntheticQuestion]:
    """Generate one multi-hop question spanning chunk + related chunks."""
    chunk_texts = [chunk.get("text") or ""] + [r.get("text") or "" for r in related]
    chunk_ids = [chunk.get("chunk_id") or ""] + [r.get("chunk_id") or "" for r in related]
    chunk_ids = [c for c in chunk_ids if c]
    if len(chunk_ids) < 2:
        return None
    prompt = _build_multi_hop_prompt(chunk_texts, num_questions=1)
    try:
        response = llm.generate(prompt)
    except Exception as e:
        logger.warning("LLM multi-hop generate failed: %s", e)
        return None
    questions = _parse_questions_from_response(response)
    if not questions:
        return None
    q = questions[0]
    if _is_vague(q) or _is_duplicate(q, seen_normalized):
        return None
    seen_normalized.add(_normalize_for_dedup(q))
    return SyntheticQuestion(
        question_id=f"q_{uuid.uuid4().hex[:12]}",
        question=q,
        gold_chunk_ids=chunk_ids,
        question_type="multi_hop",
        difficulty="hard",
        source_config=config_name,
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_questions_for_config(
    config_name: str,
    chunks_dir: Path,
    output_dir: Path,
    qa_config: QAConfig,
    llm: LLMClient,
    embedding_client: Optional[EmbeddingClient] = None,
) -> Path:
    """
    Load chunks for config, generate single- and multi-hop questions, save to JSON.
    Returns path to saved QA file.
    """
    chunks = load_chunks_for_config(chunks_dir, config_name)
    if not chunks:
        logger.warning("No chunks for config %s; skipping QA generation", config_name)
        output_path = output_dir / f"{config_name}_qa.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return output_path

    if qa_config.max_chunks is not None:
        chunks = chunks[: qa_config.max_chunks]
        logger.info("Capped to first %s chunks", len(chunks))

    seen_normalized: set[str] = set()
    all_questions: List[SyntheticQuestion] = []
    batch_size = qa_config.batch_size
    multi_hop_every_n = max(1, int(1.0 / qa_config.multi_hop_ratio)) if qa_config.multi_hop_ratio > 0 else 9999

    # Batched: one LLM call per batch of chunks (fewer API calls)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_questions = _generate_batch_questions(batch, config_name, qa_config, llm, seen_normalized)
        all_questions.extend(batch_questions)
        logger.info("Processed chunks %s-%s (%s questions so far)", start + 1, start + len(batch), len(all_questions))

    # Optional multi-hop: sample some chunks, one LLM call each
    if embedding_client:
        for i in range(0, len(chunks), multi_hop_every_n):
            chunk = chunks[i]
            related = _find_related_chunks(chunk, chunks, embedding_client, qa_config)
            if related:
                multi = _generate_multi_hop_question(chunk, related, config_name, llm, seen_normalized)
                if multi:
                    all_questions.append(multi)

    output_path = output_dir / f"{config_name}_qa.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [q.model_dump() for q in all_questions]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s questions to %s", len(all_questions), output_path)
    return output_path
