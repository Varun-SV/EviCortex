"""Public facade for the deterministic EviCortex storage and retrieval kernel."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .budget import DEFAULT_BUDGET_TOKENS
from .errors import (
    SemanticNotConfiguredError,
    SemanticProjectionError,
    StorageClosedError,
)
from .models import (
    ClaimInput,
    EmbeddingSyncStatus,
    EmbeddingUpsertResult,
    ExtractionRecord,
    IntegrityReport,
    MemoryRecord,
    StoreStats,
)
from .plugins import (
    ClaimExtractor,
    Embedder,
    EmbeddingValidationError,
    EvidenceContext,
    Reranker,
    embed_document_batch,
    validate_claim_extraction,
    validate_embedding_batch,
)
from .retrieval import DeterministicRetriever, MemoryCapsule
from .semantic import (
    EmbeddingInput,
    EmbeddingProfile,
    claim_embedding_input,
    memory_embedding_input,
)
from .store import (
    SQLiteMemoryStore,
    _canonical_metadata as _store_canonical_metadata,
    _normalize_timestamp as _store_normalize_timestamp,
    _required_label as _store_required_label,
)


@dataclass(frozen=True, slots=True)
class SemanticIndexStatus:
    """Projection completeness for one namespace and configured profile."""

    profile: EmbeddingProfile
    namespace: str
    memories: EmbeddingSyncStatus
    claims: EmbeddingSyncStatus

    @property
    def complete(self) -> bool:
        return self.memories.complete and self.claims.complete

    @property
    def registered(self) -> bool:
        has_sources = bool(
            self.memories.total_sources + self.claims.total_sources
        )
        return (
            has_sources
            and self.memories.registered
            and self.claims.registered
        )

    @property
    def missing_count(self) -> int:
        return len(self.memories.missing_ids) + len(self.claims.missing_ids)

    @property
    def stale_count(self) -> int:
        return len(self.memories.stale_ids) + len(self.claims.stale_ids)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _semantic_score(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("minimum_semantic_score must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("minimum_semantic_score must be a finite number") from error
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("minimum_semantic_score must be between -1 and 1")
    return score


def _same_immutable_memory(left: MemoryRecord, right: MemoryRecord) -> bool:
    """Compare persisted evidence identity, excluding mutable claim projections."""

    return (
        left.id == right.id
        and left.namespace == right.namespace
        and left.kind == right.kind
        and left.content == right.content
        and left.content_hash == right.content_hash
        and left.source == right.source
        and left.occurred_at == right.occurred_at
        and left.recorded_at == right.recorded_at
        and left.metadata == right.metadata
        and left.idempotency_key == right.idempotency_key
    )


def _snapshot_extractor_identity(extractor: ClaimExtractor) -> str:
    """Validate and copy the stable identity at a provider boundary."""

    if not isinstance(extractor, ClaimExtractor):
        raise TypeError("extractor does not implement the ClaimExtractor protocol")
    identity = extractor.identity
    if not isinstance(identity, str):
        raise TypeError("extractor identity must be a string")
    if not identity or identity != identity.strip():
        raise ValueError(
            "extractor identity must be non-empty without edge whitespace"
        )
    return identity


class EviCortex:
    """A namespace-safe facade over one SQLite memory store.

    Construct instances through :meth:`open` so the backing resource and context-manager
    lifecycle remain explicit.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        embedder: Embedder | None = None,
        extractor: ClaimExtractor | None = None,
        auto_index: bool = True,
        semantic_scan_cap: int = 1024,
        minimum_semantic_score: float = 0.05,
        reranker: Reranker | None = None,
        rerank_limit: int = 16,
        reranker_weight: float = 1.25,
    ) -> None:
        if not isinstance(auto_index, bool):
            raise TypeError("auto_index must be a boolean")
        self._store = store
        self._embedder = embedder
        self._embedding_profile = (
            None if embedder is None else EmbeddingProfile.from_embedder(embedder)
        )
        self._extractor = extractor
        self._extractor_identity = (
            None
            if extractor is None
            else _snapshot_extractor_identity(extractor)
        )
        self._auto_index = auto_index
        self._semantic_scan_cap = _positive_integer(
            semantic_scan_cap,
            "semantic_scan_cap",
        )
        self._minimum_semantic_score = _semantic_score(minimum_semantic_score)
        self._retriever = DeterministicRetriever(
            store,
            embedder=embedder,
            embedding_profile=self._embedding_profile,
            semantic_scan_cap=self._semantic_scan_cap,
            minimum_semantic_score=self._minimum_semantic_score,
            reranker=reranker,
            rerank_limit=rerank_limit,
            reranker_weight=reranker_weight,
        )
        self._reranker_identity = self._retriever.reranker_identity
        self._closed = False

    @classmethod
    def open(
        cls,
        path: str | Path = ":memory:",
        *,
        embedder: Embedder | None = None,
        extractor: ClaimExtractor | None = None,
        auto_index: bool = True,
        semantic_scan_cap: int = 1024,
        minimum_semantic_score: float = 0.05,
        reranker: Reranker | None = None,
        rerank_limit: int = 16,
        reranker_weight: float = 1.25,
    ) -> Self:
        """Open a local memory database."""

        store = SQLiteMemoryStore(path)
        try:
            return cls(
                store,
                embedder=embedder,
                extractor=extractor,
                auto_index=auto_index,
                semantic_scan_cap=semantic_scan_cap,
                minimum_semantic_score=minimum_semantic_score,
                reranker=reranker,
                rerank_limit=rerank_limit,
                reranker_weight=reranker_weight,
            )
        except BaseException:
            store.close()
            raise

    @property
    def embedding_profile(self) -> EmbeddingProfile | None:
        """Immutable compatibility snapshot captured when this facade opened."""

        return self._embedding_profile

    @property
    def extractor_identity(self) -> str | None:
        """Immutable identity snapshot for the configured claim extractor."""

        return self._extractor_identity

    @property
    def reranker_identity(self) -> str | None:
        """Immutable identity snapshot for the configured retrieval reranker."""

        return self._reranker_identity

    def remember(
        self,
        *,
        namespace: str,
        content: str,
        kind: str = "episode",
        source: str = "",
        occurred_at: datetime | str | None = None,
        recorded_at: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        claims: Iterable[ClaimInput] = (),
    ) -> MemoryRecord:
        """Store immutable evidence and any explicit, evidence-linked claims."""

        self._ensure_open()
        record = self._store.remember(
            namespace=namespace,
            content=content,
            kind=kind,
            source=source,
            occurred_at=occurred_at,
            recorded_at=recorded_at,
            metadata=metadata,
            idempotency_key=idempotency_key,
            claims=tuple(claims),
        )
        if self._embedder is not None and self._auto_index:
            self._index_persisted_semantic(
                record,
                evidence_committed=True,
            )
        return record

    def remember_extracted(
        self,
        *,
        namespace: str,
        content: str,
        kind: str = "episode",
        source: str = "",
        occurred_at: datetime | str | None = None,
        recorded_at: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        claim_ids_by_source: Mapping[str, str] | None = None,
        extractor: ClaimExtractor | None = None,
    ) -> MemoryRecord:
        """Extract claims from exact evidence, then persist one atomic audit run.

        A per-call ``extractor`` overrides the provider configured at :meth:`open`.
        Provider execution and validation finish before the store is asked to write
        evidence, so malformed output cannot leave a partial memory behind.
        """

        self._ensure_open()
        selected = self._extractor if extractor is None else extractor
        if selected is None:
            raise ValueError("claim extractor is not configured")
        identity = (
            self._extractor_identity
            if extractor is None
            else _snapshot_extractor_identity(selected)
        )
        assert identity is not None
        if _snapshot_extractor_identity(selected) != identity:
            raise ValueError("extractor identity changed after it was configured")

        # Mirror the store's canonical input preparation before invoking an
        # untrusted provider.  Passing the same canonical values to the store keeps
        # the extraction context and durable evidence request byte-for-byte aligned.
        normalized_namespace = _store_required_label(namespace, "namespace")
        normalized_kind = _store_required_label(kind, "kind")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")
        content.encode("utf-8")
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        normalized_occurred = _store_normalize_timestamp(occurred_at)
        normalized_recorded = _store_normalize_timestamp(recorded_at)
        normalized_metadata = json.loads(_store_canonical_metadata(metadata))
        normalized_key = (
            None
            if idempotency_key is None
            else _store_required_label(idempotency_key, "idempotency_key")
        )
        evidence = EvidenceContext.from_content(
            namespace=normalized_namespace,
            content=content,
            source=source,
            occurred_at=normalized_occurred,
            metadata=normalized_metadata,
            claim_ids_by_source=claim_ids_by_source,
        )
        extraction = selected.extract(evidence)
        if _snapshot_extractor_identity(selected) != identity:
            raise ValueError("extractor identity changed during extraction")
        validated = validate_claim_extraction(
            evidence,
            extraction,
            expected_identity=identity,
        )
        record = self._store.remember(
            namespace=normalized_namespace,
            content=content,
            kind=normalized_kind,
            source=source,
            occurred_at=normalized_occurred,
            recorded_at=normalized_recorded,
            metadata=normalized_metadata,
            idempotency_key=normalized_key,
            extraction=validated,
            extraction_identity=identity,
        )
        if self._embedder is not None and self._auto_index:
            self._index_persisted_semantic(
                record,
                evidence_committed=True,
            )
        return record

    def index_semantic(self, record: MemoryRecord) -> EmbeddingUpsertResult:
        """Build or repair semantic projections for one persisted evidence record."""

        self._ensure_open()
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")
        self._require_semantic_profile()
        try:
            persisted = self._store.get_memory(record.namespace, record.id)
        except Exception as error:
            raise SemanticProjectionError(
                "semantic projection target could not be validated in this store",
                memory_id=record.id,
                evidence_committed=False,
            ) from error
        if persisted is None or not _same_immutable_memory(record, persisted):
            raise SemanticProjectionError(
                "semantic projection target is absent or does not match this store",
                memory_id=record.id,
                evidence_committed=False,
            )
        return self._index_persisted_semantic(
            persisted,
            evidence_committed=True,
        )

    def _index_persisted_semantic(
        self,
        record: MemoryRecord,
        *,
        evidence_committed: bool,
    ) -> EmbeddingUpsertResult:
        profile = self._require_semantic_profile()
        try:
            inputs = (
                memory_embedding_input(record),
                *(
                    claim_embedding_input(claim)
                    for claim in sorted(record.claims, key=lambda item: item.id)
                ),
            )
            if self._store.embeddings_current(profile, inputs):
                return EmbeddingUpsertResult(
                    profile=profile,
                    written=0,
                    memory_embeddings=0,
                    claim_embeddings=0,
                    namespaces=(record.namespace,),
                )
            vectors = self._embed_inputs(inputs)
            return self._store.upsert_embeddings(profile, inputs, vectors)
        except SemanticProjectionError:
            raise
        except Exception as error:
            raise SemanticProjectionError(
                f"semantic projection failed for memory {record.id}",
                memory_id=record.id,
                evidence_committed=evidence_committed,
            ) from error

    def semantic_status(self, namespace: str) -> SemanticIndexStatus:
        """Return missing and stale projection state for one namespace."""

        self._ensure_open()
        profile = self._require_semantic_profile()
        memories = self._store.embedding_sync_status(profile, namespace, "memory")
        claims = self._store.embedding_sync_status(profile, namespace, "claim")
        return SemanticIndexStatus(
            profile=profile,
            namespace=memories.namespace,
            memories=memories,
            claims=claims,
        )

    def sync_semantic(
        self,
        namespace: str,
        batch_size: int = 64,
    ) -> SemanticIndexStatus:
        """Embed every missing or stale projection in deterministic batches."""

        self._ensure_open()
        size = _positive_integer(batch_size, "batch_size")
        status = self.semantic_status(namespace)
        targets = [
            *(('memory', target_id) for target_id in (
                *status.memories.missing_ids,
                *status.memories.stale_ids,
            )),
            *(('claim', target_id) for target_id in (
                *status.claims.missing_ids,
                *status.claims.stale_ids,
            )),
        ]
        for start in range(0, len(targets), size):
            inputs: list[EmbeddingInput] = []
            evidence_ids: list[str] = []
            for target_kind, target_id in targets[start : start + size]:
                if target_kind == "memory":
                    record = self._store.get_memory(status.namespace, target_id)
                    if record is None:
                        continue
                    inputs.append(memory_embedding_input(record))
                    evidence_ids.append(record.id)
                else:
                    claim = self._store.get_claim(status.namespace, target_id)
                    if claim is None:
                        continue
                    inputs.append(claim_embedding_input(claim))
                    evidence_ids.append(claim.memory_id)
            if not inputs:
                continue
            try:
                vectors = self._embed_inputs(inputs)
                self._store.upsert_embeddings(status.profile, inputs, vectors)
            except Exception as error:
                memory_id = min(evidence_ids) if evidence_ids else "unknown"
                raise SemanticProjectionError(
                    "semantic synchronization failed for committed evidence",
                    memory_id=memory_id,
                    evidence_committed=True,
                ) from error
        return self.semantic_status(status.namespace)

    def _require_semantic_profile(self) -> EmbeddingProfile:
        if self._embedder is None or self._embedding_profile is None:
            raise SemanticNotConfiguredError("semantic embedder is not configured")
        return self._embedding_profile

    def _embed_inputs(
        self,
        inputs: Sequence[EmbeddingInput],
    ) -> tuple[tuple[float, ...], ...]:
        profile = self._require_semantic_profile()
        assert self._embedder is not None
        if not profile.is_compatible(self._embedder):
            raise EmbeddingValidationError(
                "embedder compatibility fields changed after EviCortex.open"
            )
        raw_vectors = tuple(
            embed_document_batch(
                self._embedder,
                tuple(item.text for item in inputs),
            )
        )
        if not profile.is_compatible(self._embedder):
            raise EmbeddingValidationError(
                "embedder compatibility fields changed during embedding"
            )
        return validate_embedding_batch(
            raw_vectors,
            expected_count=len(inputs),
            dimensions=profile.dimensions,
            normalized=profile.normalized,
            allow_zero=True,
        )

    def recall(
        self,
        *,
        namespace: str,
        query: str,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        as_of: datetime | str | None = None,
        limit: int = 20,
    ) -> MemoryCapsule:
        """Retrieve and compile a deterministic capsule under a hard token budget."""

        self._ensure_open()
        return self._retriever.recall(
            namespace=namespace,
            query=query,
            budget_tokens=budget_tokens,
            as_of=as_of,
            limit=limit,
        )

    def expand(self, *, namespace: str, memory_id: str) -> MemoryRecord:
        """Return the exact immutable evidence record referenced by a capsule."""

        self._ensure_open()
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        record = self._store.get_memory(namespace.strip(), memory_id.strip())
        if record is None or record.namespace != namespace.strip():
            # The same error is used for a missing ID and a namespace mismatch so the
            # facade does not disclose whether another namespace owns the evidence.
            raise KeyError(f"memory evidence not found: {memory_id.strip()}")
        return record

    def get_extraction(
        self,
        namespace: str,
        memory_id: str,
    ) -> ExtractionRecord:
        """Return one durable extraction run without cross-namespace disclosure."""

        self._ensure_open()
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        normalized_namespace = namespace.strip()
        normalized_memory_id = memory_id.strip()
        extraction = self._store.get_extraction(
            normalized_namespace,
            normalized_memory_id,
        )
        if (
            extraction is None
            or extraction.namespace != normalized_namespace
            or extraction.memory_id != normalized_memory_id
        ):
            raise KeyError(
                f"memory extraction not found: {normalized_memory_id}"
            )
        return extraction

    def verify_integrity(self, namespace: str | None = None) -> IntegrityReport:
        """Verify evidence digests and derived index consistency."""

        self._ensure_open()
        return self._store.verify_integrity(namespace=namespace)

    def rebuild_indexes(self) -> None:
        """Rebuild deterministic local claim and FTS projections only.

        Provider-derived semantic vectors are preserved and are never generated or
        repaired by this store-wide operation.  With an embedder configured, call
        :meth:`sync_semantic` for each authorized namespace that should be repaired.
        """

        self._ensure_open()
        self._store.rebuild_indexes()

    def purge_memory(self, *, namespace: str, memory_id: str) -> bool:
        """Hard-delete one authorized evidence record and its derived projections."""

        self._ensure_open()
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        return self._store.purge_memory(namespace.strip(), memory_id.strip())

    def stats(self, namespace: str | None = None) -> StoreStats:
        """Return row-count diagnostics for the whole store or one namespace."""

        self._ensure_open()
        return self._store.stats(namespace=namespace)

    def close(self) -> None:
        """Close the backing store; repeated calls are harmless."""

        if self._closed:
            return
        self._store.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageClosedError("EviCortex is closed")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["EviCortex", "SemanticIndexStatus"]
