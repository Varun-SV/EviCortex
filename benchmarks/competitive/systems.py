"""Memory-system adapters compared in Phase D.

Every adapter stores exactly ``Unit.rendered`` for each turn, in the same order,
and must return a context whose ``estimate_tokens`` is within the requested
budget. The two controls that ignore the budget on purpose (full history and
oracle evidence) declare ``budget_exempt = True`` and are reported separately.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

from evicortex import Embedder, EviCortex
from evicortex.budget import estimate_tokens
from evicortex.plugins import embed_document_batch, embed_query_text

from ..baseline import FixedChunkFTS5Baseline, fixed_chunks
from ..hybrid_baseline import DiagnosticHybridRAGBaseline
from .types import Conversation, Question, Retrieved, Unit


class CachingEmbedder:
    """Share one encoder across systems and never embed the same text twice.

    The cache is keyed by role and exact text, so two systems that happen to embed
    the same string reuse the vector while each system still decides *what* it
    embeds. Identity, dimensions, and normalization are the wrapped provider's.
    """

    def __init__(self, wrapped: Embedder) -> None:
        if not isinstance(wrapped, Embedder):
            raise TypeError("wrapped must implement the Embedder protocol")
        self._wrapped = wrapped
        self._documents: dict[str, tuple[float, ...]] = {}
        self._queries: dict[str, tuple[float, ...]] = {}
        self._lock = threading.Lock()
        self.document_texts_encoded = 0
        self.query_texts_encoded = 0
        self.encode_seconds = 0.0

    @property
    def identity(self) -> str:
        return self._wrapped.identity

    @property
    def dimensions(self) -> int:
        return self._wrapped.dimensions

    @property
    def normalized(self) -> bool:
        return self._wrapped.normalized

    @property
    def is_semantic(self) -> bool:
        return getattr(self._wrapped, "is_semantic", True) is True

    @property
    def diagnostic_only(self) -> bool:
        return getattr(self._wrapped, "diagnostic_only", False) is True

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        frozen = tuple(texts)
        keys = [self._key(text) for text in frozen]
        with self._lock:
            missing: dict[str, str] = {}
            for key, text in zip(keys, frozen, strict=True):
                if key not in self._documents:
                    missing.setdefault(key, text)
            if missing:
                started = time.perf_counter()
                vectors = embed_document_batch(self._wrapped, tuple(missing.values()))
                self.encode_seconds += time.perf_counter() - started
                self.document_texts_encoded += len(missing)
                for key, vector in zip(missing, vectors, strict=True):
                    self._documents[key] = tuple(float(value) for value in vector)
            return tuple(self._documents[key] for key in keys)

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        key = self._key(text)
        with self._lock:
            if key not in self._queries:
                started = time.perf_counter()
                vector = embed_query_text(self._wrapped, text)
                self.encode_seconds += time.perf_counter() - started
                self.query_texts_encoded += 1
                self._queries[key] = tuple(float(value) for value in vector)
            return self._queries[key]


def _visible(units: Sequence[Unit], question: Question) -> list[Unit]:
    if question.question_date is None:
        return list(units)
    return [unit for unit in units if unit.occurred_at <= question.question_date]


class _Base:
    name = "base"
    label = ""
    budget_exempt = False
    uses_embedder = False

    def __init__(self) -> None:
        self._ingest_seconds = 0.0
        self._units: dict[str, tuple[Unit, ...]] = {}

    def ingest(self, conversation: Conversation) -> None:
        started = time.perf_counter()
        self._units[conversation.namespace] = conversation.units
        self._ingest(conversation)
        self._ingest_seconds += time.perf_counter() - started

    def _ingest(self, conversation: Conversation) -> None:
        return None

    def ingestion_cost(self) -> dict[str, Any]:
        return {
            "wall_seconds": round(self._ingest_seconds, 3),
            "llm_calls": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_cost_usd": 0.0,
        }

    def close(self) -> None:
        return None


class NoMemory(_Base):
    name = "no_memory"
    label = "No memory (reader sees only the question)"

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        return Retrieved(context="", unit_ids=())


class FullHistory(_Base):
    name = "full_history"
    label = "Full history (every visible turn, no budget)"
    budget_exempt = True

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        visible = _visible(self._units[question.namespace], question)
        return Retrieved(
            context="\n".join(unit.rendered for unit in visible),
            unit_ids=tuple(unit.unit_id for unit in visible),
        )


class OracleEvidence(_Base):
    """Gold evidence only: separates retrieval failure from reader failure."""

    name = "oracle"
    label = "Oracle evidence (gold turns or gold sessions, no budget)"
    budget_exempt = True

    def __init__(self, *, by_session: bool) -> None:
        super().__init__()
        self._by_session = by_session

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        units = self._units[question.namespace]
        if self._by_session:
            sessions = set(question.gold_session_ids)
            chosen = [unit for unit in _visible(units, question) if unit.session_id in sessions]
        else:
            wanted = set(question.gold_unit_ids)
            chosen = [unit for unit in units if unit.unit_id in wanted]
        return Retrieved(
            context="\n".join(unit.rendered for unit in chosen),
            unit_ids=tuple(unit.unit_id for unit in chosen),
        )


class BM25Chunks(_Base):
    name = "bm25_chunks"
    label = "Fixed-chunk BM25 (SQLite FTS5)"

    def __init__(self, path: str | Path, *, chunk_chars: int = 600, overlap_chars: int = 80) -> None:
        super().__init__()
        self._store = FixedChunkFTS5Baseline(
            path, chunk_chars=chunk_chars, overlap_chars=overlap_chars, candidate_limit=200
        )
        self.config = {"chunk_chars": chunk_chars, "overlap_chars": overlap_chars}

    def _ingest(self, conversation: Conversation) -> None:
        for unit in conversation.units:
            # Source IDs are a global primary key in the baseline; dataset turn
            # IDs repeat across conversations.
            self._store.remember(
                namespace=conversation.namespace,
                content=unit.rendered,
                source_id=f"{conversation.namespace}|{unit.unit_id}",
                occurred_at=unit.occurred_at,
            )

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        result = self._store.recall(
            namespace=question.namespace,
            query=question.text,
            budget_tokens=budget_tokens,
            as_of=question.question_date,
        )
        return Retrieved(
            context=result.to_prompt() if result.evidence else "",
            unit_ids=tuple(
                dict.fromkeys(item.source_id.split("|", 1)[1] for item in result.evidence)
            ),
        )

    def close(self) -> None:
        self._store.close()


class DenseRAG(_Base):
    """Plain dense retrieval: embed fixed chunks, rank by cosine, pack greedily."""

    name = "dense_rag"
    label = "Dense RAG (same encoder, cosine top-k over fixed chunks)"
    uses_embedder = True

    def __init__(self, embedder: Embedder, *, chunk_chars: int = 600, overlap_chars: int = 80) -> None:
        super().__init__()
        import numpy

        self._numpy = numpy
        self._embedder = embedder
        self._chunk_chars = chunk_chars
        self._overlap_chars = overlap_chars
        self._index: dict[str, tuple[Any, list[tuple[Unit, str]]]] = {}
        self.config = {"chunk_chars": chunk_chars, "overlap_chars": overlap_chars}

    def _ingest(self, conversation: Conversation) -> None:
        rows: list[tuple[Unit, str]] = []
        for unit in conversation.units:
            for piece in fixed_chunks(
                unit.rendered, size=self._chunk_chars, overlap=self._overlap_chars
            ):
                rows.append((unit, piece))
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(rows), 64):
            batch = rows[start : start + 64]
            vectors.extend(embed_document_batch(self._embedder, [piece for _, piece in batch]))
        matrix = self._numpy.asarray(vectors, dtype=self._numpy.float32)
        self._index[conversation.namespace] = (matrix, rows)

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        matrix, rows = self._index[question.namespace]
        query = self._numpy.asarray(
            embed_query_text(self._embedder, question.text), dtype=self._numpy.float32
        )
        scores = matrix @ query
        order = self._numpy.argsort(-scores, kind="stable")
        selected: list[dict[str, str]] = []
        unit_ids: list[str] = []
        for position in order[:400]:
            unit, piece = rows[int(position)]
            if question.question_date is not None and unit.occurred_at > question.question_date:
                continue
            candidate = [*selected, {"source": unit.unit_id, "text": piece}]
            if estimate_tokens(_dump({"evidence": candidate})) > budget_tokens:
                continue
            selected = candidate
            unit_ids.append(unit.unit_id)
        return Retrieved(
            context=_dump({"evidence": selected}) if selected else "",
            unit_ids=tuple(dict.fromkeys(unit_ids)),
        )


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class _FastCosine:
    """Drop-in for ``evicortex.semantic.cosine_similarity`` inside the comparator.

    The comparator's maximal-marginal-relevance step calls a validating,
    pure-Python cosine a cubic number of times in the candidate count, which
    takes minutes per query once the candidate pool is large enough to fill a
    4,000-token budget. This computes the same quantity in float64 with NumPy
    and memoizes per vector. It changes speed only: ``tests/test_competitive.py``
    checks that rankings match the reference implementation.
    """

    def __init__(self) -> None:
        import numpy

        self._numpy = numpy
        # Keyed by object identity. Each entry keeps a reference to its vector, so
        # an id cannot be recycled for a different vector while it is cached.
        self._vectors: dict[int, tuple[Any, Any, float]] = {}
        self._pairs: dict[tuple[int, int], float] = {}

    def _prepared(self, vector: Sequence[float]) -> tuple[Any, float]:
        hit = self._vectors.get(id(vector))
        if hit is None or hit[0] is not vector:
            array = self._numpy.asarray(vector, dtype=self._numpy.float64)
            hit = (vector, array, float(self._numpy.linalg.norm(array)))
            self._vectors[id(vector)] = hit
        return hit[1], hit[2]

    def __call__(
        self, left: Sequence[float], right: Sequence[float], *, normalized: bool = False
    ) -> float:
        del normalized  # the definition is identical for unit and non-unit vectors
        key = (id(left), id(right))
        score = self._pairs.get(key)
        if score is not None:
            return score
        if len(self._vectors) > 4_000 or len(self._pairs) > 2_000_000:
            self._vectors.clear()
            self._pairs.clear()
        left_array, left_norm = self._prepared(left)
        right_array, right_norm = self._prepared(right)
        if left_norm == 0.0 or right_norm == 0.0:
            score = 0.0
        else:
            score = float(left_array @ right_array) / (left_norm * right_norm)
            score = max(-1.0, min(1.0, score))
        self._pairs[key] = score
        return score


class HybridRAG(_Base):
    name = "hybrid_rag"
    label = "Hybrid RAG (BM25 + same encoder, RRF, MMR, neighbor expansion)"
    uses_embedder = True

    def __init__(self, path: str | Path, embedder: Embedder, *, fast_cosine: bool = True) -> None:
        super().__init__()
        if fast_cosine:
            from .. import hybrid_baseline

            hybrid_baseline.cosine_similarity = _FastCosine()
        self._store = DiagnosticHybridRAGBaseline.open(
            path,
            embedder=embedder,
            channel_limit=120,
            candidate_limit=120,
        )
        if fast_cosine:
            self._memoize_diversify()
        self.config = {
            "chunk_chars": list(self._store.chunk_chars),
            "chunk_overlaps": list(self._store.chunk_overlaps),
            "channel_limit": 120,
            "candidate_limit": 120,
            "cosine": "numpy float64 (speed only)" if fast_cosine else "reference",
        }

    def _memoize_diversify(self) -> None:
        """Reuse the diversified ranking when only the packing budget changes."""

        from ..hybrid_baseline import _candidate_key

        reference = self._store._diversify
        last: dict[str, Any] = {"key": None, "value": None}

        def diversify(candidates: Sequence[Any]) -> Any:
            key = tuple((_candidate_key(item), item.fused_score) for item in candidates)
            if key != last["key"]:
                last["value"] = reference(candidates)
                last["key"] = key
            return last["value"]

        self._store._diversify = diversify  # type: ignore[method-assign]

    def _ingest(self, conversation: Conversation) -> None:
        for unit in conversation.units:
            # The comparator requires globally unique source IDs, while dataset
            # turn IDs repeat across conversations.
            self._store.remember(
                namespace=conversation.namespace,
                content=unit.rendered,
                source_id=f"{conversation.namespace}|{unit.unit_id}",
                occurred_at=unit.occurred_at,
                recorded_at=unit.occurred_at,
                session_id=unit.session_id,
                speaker=unit.speaker,
            )

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        result = self._store.recall(
            namespace=question.namespace,
            query=question.text,
            budget_tokens=budget_tokens,
            as_of=question.question_date,
        )
        return Retrieved(
            context=result.to_prompt() if result.evidence else "",
            unit_ids=tuple(
                dict.fromkeys(item.source_id.split("|", 1)[1] for item in result.evidence)
            ),
        )

    def close(self) -> None:
        self._store.close()


class EviCortexSystem(_Base):
    """EviCortex through its public API only: ``remember`` then ``recall``.

    With ``claim_source=None`` no LLM is used at ingestion and the capsule holds
    exact-evidence episodes. With a claim source, LLM-extracted temporal claims
    are attached to the turn that states them (see ``claims.py``).
    """

    uses_embedder = True

    def __init__(
        self,
        path: str | Path,
        embedder: Embedder,
        *,
        claim_source: Any | None = None,
        candidate_limit: int = 120,
        semantic_scan_cap: int = 16384,
    ) -> None:
        super().__init__()
        self._memory = EviCortex.open(
            path,
            embedder=embedder,
            auto_index=False,
            semantic_scan_cap=semantic_scan_cap,
        )
        self._claim_source = claim_source
        self._candidate_limit = candidate_limit
        self._unit_of: dict[str, str] = {}
        self.name = "evicortex" if claim_source is None else "evicortex_claims"
        self.label = (
            "EviCortex (evidence ledger + hybrid recall; no LLM at ingestion)"
            if claim_source is None
            else "EviCortex + LLM-extracted temporal claims"
        )
        self.config = {
            "candidate_limit": candidate_limit,
            "semantic_scan_cap": semantic_scan_cap,
            "claims": claim_source is not None,
        }
        self.claim_stats: dict[str, int] = {"claims_written": 0, "claims_rejected": 0}

    def _ingest(self, conversation: Conversation) -> None:
        claims_by_unit = (
            self._claim_source.claims_for(conversation, self.claim_stats)
            if self._claim_source is not None
            else {}
        )
        writer = (
            self._claim_source.writer() if self._claim_source is not None else None
        )
        for unit in conversation.units:
            claims = ()
            if writer is not None:
                claims = writer.prepare(unit, claims_by_unit.get(unit.unit_id, ()))
            record = self._write(conversation.namespace, unit, claims)
            if writer is not None:
                writer.committed(record)
            self._unit_of[str(record.id)] = unit.unit_id
        self._memory.sync_semantic(conversation.namespace, batch_size=64)

    def _write(self, namespace: str, unit: Unit, claims: Sequence[Any]):
        arguments = dict(
            namespace=namespace,
            content=unit.rendered,
            source=unit.unit_id,
            occurred_at=unit.occurred_at,
            recorded_at=unit.occurred_at,
            metadata={"session_id": unit.session_id, "speaker": unit.speaker},
            idempotency_key=f"{namespace}:{unit.unit_id}",
        )
        if not claims:
            return self._memory.remember(**arguments)
        try:
            record = self._memory.remember(**arguments, claims=claims)
            self.claim_stats["claims_written"] += len(claims)
            return record
        except Exception:  # noqa: BLE001 - a rejected claim must not lose evidence
            self.claim_stats["claims_rejected"] += len(claims)
            return self._memory.remember(**arguments)

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        capsule = self._memory.recall(
            namespace=question.namespace,
            query=question.text,
            budget_tokens=budget_tokens,
            as_of=question.question_date,
            limit=self._candidate_limit,
        )
        ordered: list[str] = []
        for fact in capsule.facts:
            ordered.extend(fact.evidence_ids)
        for episode in capsule.episodes:
            ordered.append(episode.memory_id)
        unit_ids = tuple(
            dict.fromkeys(self._unit_of[item] for item in ordered if item in self._unit_of)
        )
        has_content = bool(capsule.facts or capsule.episodes or capsule.conflicts)
        return Retrieved(
            context=capsule.to_prompt() if has_content else "",
            unit_ids=unit_ids,
            extra={
                "facts": len(capsule.facts),
                "episodes": len(capsule.episodes),
                "conflicts": len(capsule.conflicts),
            },
        )

    def ingestion_cost(self) -> dict[str, Any]:
        cost = super().ingestion_cost()
        if self._claim_source is not None:
            cost.update(self._claim_source.cost())
            cost.update(self.claim_stats)
        return cost

    def close(self) -> None:
        self._memory.close()
