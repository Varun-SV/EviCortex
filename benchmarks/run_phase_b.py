"""Standalone Phase B diagnostic comparison runner.

This runner compares EviCortex's semantic/temporal wiring with the isolated
``DiagnosticHybridRAGBaseline``.  Both systems use separate counting wrappers
around the same deterministic ``HashingEmbedder`` configuration.  Feature
hashing is non-semantic, and the comparator is not a validated strong RAG
baseline; this module therefore supports diagnostics, not superiority claims.

Example::

    python -m benchmarks.run_phase_b --events 100 --seed 7 \
        --budget-tokens 512 --json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import tempfile
import threading
import time
from typing import Any, TypeVar

from evicortex import (
    CanonicalClaimExtractor,
    ClaimExtraction,
    ClaimExtractor,
    Embedder,
    EvidenceContext,
    HashingEmbedder,
    EviCortex,
    embed_document_batch,
    embed_query_text,
)
from evicortex.budget import estimate_tokens
from evicortex.semantic import EmbeddingProfile

from .hybrid_baseline import (
    DIAGNOSTIC_HYBRID_LABEL,
    DiagnosticHybridRAGBaseline,
)
from .run import (
    ABSTENTION_ACCURACY_SEMANTICS,
    _aggregate,
    _code_revision,
    _comparison,
    _compacted_database_bytes,
    _percentile,
    _query_observation,
    _suite_digest,
    SUITE_DIGEST_SEMANTICS,
)
from .synthetic import QueryCase, SyntheticEvent, SyntheticSuite, generate_suite


EVICORTEX_PHASE_B_LABEL = (
    "EviCortex durable-extraction semantic+temporal diagnostic candidate using "
    "HashingEmbedder; feature hashing is non-semantic"
)
PHASE_B_SCOPE = (
    "Diagnostic wiring comparison only: HashingEmbedder is non-semantic and the "
    "hybrid comparator is not a validated strong RAG baseline. No superiority "
    "claim is made."
)
DEFAULT_EMBEDDING_DIMENSIONS = 256
DEFAULT_SEMANTIC_SCAN_CAP = 2_048
_DETAIL_NUMBER = re.compile(r"(?:^|;)(scanned|scan_cap)=(\d+)(?=;|$)")
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class EmbeddingCounterSnapshot:
    """Cumulative provider work observed by one isolated wrapper."""

    batch_calls: int = 0
    text_count: int = 0
    text_utf8_bytes: int = 0

    def minus(self, earlier: "EmbeddingCounterSnapshot") -> "EmbeddingCounterSnapshot":
        return EmbeddingCounterSnapshot(
            self.batch_calls - earlier.batch_calls,
            self.text_count - earlier.text_count,
            self.text_utf8_bytes - earlier.text_utf8_bytes,
        )

    def plus(self, other: "EmbeddingCounterSnapshot") -> "EmbeddingCounterSnapshot":
        return EmbeddingCounterSnapshot(
            self.batch_calls + other.batch_calls,
            self.text_count + other.text_count,
            self.text_utf8_bytes + other.text_utf8_bytes,
        )


class CountingEmbedder:
    """Count symmetric or role-aware calls without changing provider identity."""

    def __init__(self, wrapped: Embedder, *, counter_name: str) -> None:
        if not isinstance(wrapped, Embedder):
            raise TypeError("wrapped must implement the Embedder protocol")
        if not isinstance(counter_name, str) or not counter_name.strip():
            raise ValueError("counter_name must be a non-empty string")
        self._wrapped = wrapped
        self.counter_name = counter_name.strip()
        self._batch_calls = 0
        self._text_count = 0
        self._text_utf8_bytes = 0
        self._lock = threading.Lock()

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

    def _record(self, texts: tuple[str, ...]) -> None:
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"text {index} must be a string")
        with self._lock:
            self._batch_calls += 1
            self._text_count += len(texts)
            self._text_utf8_bytes += sum(
                len(text.encode("utf-8")) for text in texts
            )

    def embed_batch(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        frozen = tuple(texts)
        self._record(frozen)
        return self._wrapped.embed_batch(frozen)

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        frozen = tuple(texts)
        self._record(frozen)
        return tuple(embed_document_batch(self._wrapped, frozen))

    def embed_query(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        self._record((text,))
        return tuple(embed_query_text(self._wrapped, text))

    def snapshot(self) -> EmbeddingCounterSnapshot:
        with self._lock:
            return EmbeddingCounterSnapshot(
                self._batch_calls,
                self._text_count,
                self._text_utf8_bytes,
            )


@dataclass(frozen=True, slots=True)
class ExtractionObservation:
    """One successful extractor invocation observed at the public boundary."""

    extraction: ClaimExtraction
    elapsed_ns: int


class TimingClaimExtractor:
    """Time one delegate while leaving validation ownership with the facade."""

    def __init__(self, wrapped: ClaimExtractor) -> None:
        if not isinstance(wrapped, ClaimExtractor):
            raise TypeError("wrapped must implement ClaimExtractor")
        self._wrapped = wrapped
        self._identity = wrapped.identity
        self._invocations = 0
        self._latest: ExtractionObservation | None = None
        self._lock = threading.Lock()

    @property
    def identity(self) -> str:
        return self._identity

    def extract(self, evidence: EvidenceContext) -> ClaimExtraction:
        started = time.perf_counter_ns()
        extraction = self._wrapped.extract(evidence)
        elapsed_ns = time.perf_counter_ns() - started
        with self._lock:
            self._invocations += 1
            self._latest = ExtractionObservation(extraction, elapsed_ns)
        return extraction

    def invocation_count(self) -> int:
        with self._lock:
            return self._invocations

    def latest(self) -> ExtractionObservation:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("extractor has not completed an invocation")
            return self._latest


def _empty_usage() -> dict[str, EmbeddingCounterSnapshot]:
    return {
        "ingestion": EmbeddingCounterSnapshot(),
        "warmup": EmbeddingCounterSnapshot(),
        "measured_query": EmbeddingCounterSnapshot(),
    }


def _counted_call(
    counter: CountingEmbedder,
    usage: dict[str, EmbeddingCounterSnapshot],
    phase: str,
    call: Callable[[], _T],
) -> _T:
    before = counter.snapshot()
    try:
        return call()
    finally:
        delta = counter.snapshot().minus(before)
        usage[phase] = usage[phase].plus(delta)


def _warm_and_time(
    call: Callable[[], _T],
    *,
    counter: CountingEmbedder,
    usage: dict[str, EmbeddingCounterSnapshot],
) -> tuple[_T | None, float, str | None, str | None]:
    warmup_error: str | None = None
    try:
        _counted_call(counter, usage, "warmup", call)
    except Exception as exc:
        warmup_error = f"{type(exc).__name__}: {exc}"

    started = time.perf_counter_ns()
    try:
        result = _counted_call(counter, usage, "measured_query", call)
        error = None
    except Exception as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return result, elapsed_ms, error, warmup_error


def _usage_report(
    counter: CountingEmbedder,
    usage: dict[str, EmbeddingCounterSnapshot],
) -> dict[str, Any]:
    accounted = EmbeddingCounterSnapshot()
    for snapshot in usage.values():
        accounted = accounted.plus(snapshot)
    total = counter.snapshot()
    return {
        "counter_name": counter.counter_name,
        "by_phase": {name: asdict(snapshot) for name, snapshot in usage.items()},
        "total": asdict(total),
        "unaccounted": asdict(total.minus(accounted)),
    }


def _event_level_temporal_metadata(
    event: SyntheticEvent,
    extraction: ClaimExtraction,
    claim_source_by_id: dict[str, str],
) -> tuple[str, str | None, str | None]:
    """Translate one extracted claim into the hybrid event-level projection."""

    if len(extraction.claims) > 1:
        raise RuntimeError(
            "the diagnostic hybrid event projection cannot fairly encode multiple "
            f"claim timelines from source {event.source_id}"
        )
    if not extraction.claims:
        return event.occurred_at, None, None
    claim = extraction.claims[0]
    supersedes_source_id = None
    if claim.supersedes is not None:
        supersedes_source_id = claim_source_by_id.get(claim.supersedes)
        if supersedes_source_id is None:
            raise RuntimeError(
                "extracted predecessor claim ID has no source resolution: "
                f"{claim.supersedes}"
            )
    valid_from = (
        event.occurred_at if claim.valid_from is None else str(claim.valid_from)
    )
    valid_to = None if claim.valid_to is None else str(claim.valid_to)
    return valid_from, valid_to, supersedes_source_id


def _phase_b_observation(
    case: QueryCase,
    *,
    source_ids: Sequence[str],
    context_tokens: int,
    latency_ms: float,
    budget_tokens: int,
    error: str | None,
    warmup_error: str | None,
) -> dict[str, Any]:
    observation = _query_observation(
        case,
        source_ids=source_ids,
        estimated_context_tokens=context_tokens,
        retrieval_latency_ms=latency_ms,
        budget_tokens=budget_tokens,
        error=error,
    )
    # A failed retrieval is never credited as a correct abstention.
    if error is not None:
        observation["abstained"] = False
    observation["warmup_error"] = warmup_error
    return observation


def _candidate_diagnostics(capsule: Any | None) -> list[dict[str, str]]:
    if capsule is None:
        return []
    return [diagnostic.as_dict() for diagnostic in capsule.diagnostics]


def _aggregate_candidate_diagnostics(
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    codes: Counter[str] = Counter()
    truncation_queries = 0
    reported_scanned = 0
    reported_caps: set[int] = set()
    for observation in observations:
        query_truncated = False
        for diagnostic in observation.get("retrieval_diagnostics", ()):
            code = str(diagnostic.get("code", "unknown"))
            codes[code] += 1
            if code.endswith("_scan_truncated"):
                query_truncated = True
            detail = str(diagnostic.get("detail", ""))
            for key, value in _DETAIL_NUMBER.findall(detail):
                if key == "scanned":
                    reported_scanned += int(value)
                elif key == "scan_cap":
                    reported_caps.add(int(value))
        truncation_queries += query_truncated
    return {
        "diagnostic_codes": dict(sorted(codes.items())),
        "queries_with_scan_truncation": truncation_queries,
        "reported_scanned_total_on_truncation": reported_scanned,
        "reported_scan_caps": sorted(reported_caps),
        "normal_scan_counts_exposed_by_capsule": False,
    }


def _aggregate_hybrid_diagnostics(
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    successful = [
        observation["retrieval_diagnostics"]
        for observation in observations
        if observation.get("retrieval_diagnostics") is not None
    ]
    semantic_scored = [
        int(item["semantic_candidates_scored"]) for item in successful
    ]
    lexical_returned = [
        int(item["lexical_candidates_returned"]) for item in successful
    ]
    return {
        "successful_queries_with_diagnostics": len(successful),
        "query_embedding_calls_total": sum(
            int(item["query_embedding_calls"]) for item in successful
        ),
        "semantic_candidates_scored_total": sum(semantic_scored),
        "semantic_candidates_scored_max": max(semantic_scored, default=0),
        "lexical_candidates_returned_total": sum(lexical_returned),
        "scan_overflow_query_errors": sum(
            "max_exact_candidates" in str(observation.get("error") or "")
            for observation in observations
        ),
    }


def _semantic_status_report(status: Any) -> dict[str, Any]:
    return {
        "complete": status.complete,
        "registered": status.registered,
        "missing_count": status.missing_count,
        "stale_count": status.stale_count,
        "memories": {
            "total_sources": status.memories.total_sources,
            "projected_sources": status.memories.projected_sources,
        },
        "claims": {
            "total_sources": status.claims.total_sources,
            "projected_sources": status.claims.projected_sources,
        },
    }


def run_phase_b(
    *,
    events: int = 100,
    seed: int = 7,
    budget_tokens: int = 512,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    semantic_scan_cap: int = DEFAULT_SEMANTIC_SCAN_CAP,
    minimum_semantic_score: float = 0.05,
    candidate_limit: int = 40,
    hybrid_channel_limit: int = 80,
    hybrid_chunk_chars: tuple[int, int] = (192, 576),
    hybrid_chunk_overlaps: tuple[int, int] = (32, 96),
) -> dict[str, Any]:
    """Run Phase B systems over identical raw events and streaming snapshots."""

    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    if semantic_scan_cap <= 0:
        raise ValueError("semantic_scan_cap must be positive")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    suite = generate_suite(event_count=events, seed=seed)
    extractor = CanonicalClaimExtractor()
    timed_extractor = TimingClaimExtractor(extractor)
    queries_by_sequence: dict[int, list[QueryCase]] = defaultdict(list)
    for case in suite.queries:
        queries_by_sequence[case.after_sequence].append(case)

    candidate_counter = CountingEmbedder(
        HashingEmbedder(dimensions=embedding_dimensions),
        counter_name="evicortex",
    )
    hybrid_counter = CountingEmbedder(
        HashingEmbedder(dimensions=embedding_dimensions),
        counter_name="hybrid_baseline",
    )
    candidate_profile = EmbeddingProfile.from_embedder(candidate_counter)
    hybrid_profile = EmbeddingProfile.from_embedder(hybrid_counter)
    if candidate_profile != hybrid_profile:
        raise RuntimeError("candidate and hybrid embedding profiles differ")

    candidate_usage = _empty_usage()
    hybrid_usage = _empty_usage()
    extraction_latencies: list[float] = []
    extraction_claims = 0
    extraction_abstentions = 0
    extraction_provenance = 0
    extraction_supersessions = 0
    extraction_diagnostic_codes: Counter[str] = Counter()
    extraction_diagnostic_events: list[dict[str, Any]] = []
    durable_extraction_runs = 0
    durable_extraction_derivations = 0
    durable_extraction_diagnostics = 0
    durable_extraction_abstentions = 0
    durable_claim_id_mappings_verified = 0
    candidate_ingestion_ns = 0
    hybrid_ingestion_ns = 0
    candidate_observations: list[dict[str, Any]] = []
    hybrid_observations: list[dict[str, Any]] = []
    record_source_by_id: dict[str, str] = {}
    claim_ids_by_namespace_source: dict[str, dict[str, str]] = defaultdict(dict)
    claim_source_by_id: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="evicortex-phase-b-") as directory:
        root = Path(directory)
        candidate_path = root / "evicortex-phase-b.db"
        hybrid_path = root / "hybrid-phase-b.db"
        candidate = EviCortex.open(
            candidate_path,
            embedder=candidate_counter,
            semantic_scan_cap=semantic_scan_cap,
            minimum_semantic_score=minimum_semantic_score,
        )
        hybrid = DiagnosticHybridRAGBaseline.open(
            hybrid_path,
            embedder=hybrid_counter,
            chunk_chars=hybrid_chunk_chars,
            chunk_overlaps=hybrid_chunk_overlaps,
            channel_limit=hybrid_channel_limit,
            candidate_limit=candidate_limit,
            max_exact_candidates=semantic_scan_cap,
            minimum_semantic_score=minimum_semantic_score,
        )
        try:
            for event in suite.events:
                observations_before = timed_extractor.invocation_count()
                started = time.perf_counter_ns()
                record = _counted_call(
                    candidate_counter,
                    candidate_usage,
                    "ingestion",
                    lambda event=event: candidate.remember_extracted(
                        namespace=event.namespace,
                        content=event.content,
                        source=event.source_id,
                        occurred_at=event.occurred_at,
                        recorded_at=event.occurred_at,
                        claim_ids_by_source=(
                            claim_ids_by_namespace_source[event.namespace]
                        ),
                        extractor=timed_extractor,
                    ),
                )
                candidate_ingestion_ns += time.perf_counter_ns() - started
                if timed_extractor.invocation_count() != observations_before + 1:
                    raise RuntimeError(
                        "remember_extracted must invoke the extractor exactly once"
                    )
                extraction_observation = timed_extractor.latest()
                extraction = extraction_observation.extraction
                extraction_latencies.append(
                    extraction_observation.elapsed_ns / 1_000_000
                )
                extraction_claims += len(extraction.claims)
                extraction_provenance += len(extraction.provenance)
                extraction_abstentions += extraction.abstained
                extraction_supersessions += sum(
                    claim.supersedes is not None for claim in extraction.claims
                )
                if extraction.diagnostics:
                    event_diagnostics = []
                    for diagnostic in extraction.diagnostics:
                        extraction_diagnostic_codes[diagnostic.code] += 1
                        event_diagnostics.append(asdict(diagnostic))
                    extraction_diagnostic_events.append(
                        {
                            "source_id": event.source_id,
                            "diagnostics": event_diagnostics,
                        }
                    )

                durable = candidate.get_extraction(
                    namespace=event.namespace,
                    memory_id=str(record.id),
                )
                if len(record.claims) != len(extraction.claims):
                    raise RuntimeError("persisted claim count differs from extraction")
                if len(record.claims) > 1:
                    raise RuntimeError(
                        "synthetic source emitted multiple claims; explicit source-to-"
                        "claim supersession resolution would be ambiguous"
                    )
                expected_claim_ids = tuple(str(claim.id) for claim in record.claims)
                if durable.claim_ids != expected_claim_ids:
                    raise RuntimeError(
                        "durable derivations do not map to persisted claim IDs"
                    )
                if durable.extractor_identity != timed_extractor.identity:
                    raise RuntimeError("durable extractor identity differs from provider")
                if (
                    durable.memory_id != str(record.id)
                    or durable.namespace != event.namespace
                    or durable.recorded_at != record.recorded_at
                ):
                    raise RuntimeError("durable extraction run identity differs")
                expected_content_sha256 = hashlib.sha256(
                    event.content.encode("utf-8")
                ).hexdigest()
                if durable.content_sha256 != expected_content_sha256:
                    raise RuntimeError("durable extraction content digest differs")
                if durable.abstained != extraction.abstained:
                    raise RuntimeError("durable extraction abstention differs")
                if len(durable.derivations) != len(extraction.provenance):
                    raise RuntimeError("durable derivation count differs from extraction")
                for derivation, provenance in zip(
                    durable.derivations,
                    extraction.provenance,
                    strict=True,
                ):
                    durable_fields = (
                        derivation.claim_index,
                        derivation.start,
                        derivation.end,
                        derivation.canonical_text,
                        derivation.content_sha256,
                        derivation.extractor_identity,
                    )
                    emitted_fields = (
                        provenance.claim_index,
                        provenance.start,
                        provenance.end,
                        provenance.canonical_text,
                        provenance.content_sha256,
                        provenance.extractor_identity,
                    )
                    if durable_fields != emitted_fields:
                        raise RuntimeError("durable derivation differs from extraction")
                durable_diagnostic_fields = tuple(
                    (
                        diagnostic.diagnostic_index,
                        diagnostic.code,
                        diagnostic.message,
                        diagnostic.start,
                        diagnostic.end,
                    )
                    for diagnostic in durable.diagnostics
                )
                emitted_diagnostic_fields = tuple(
                    (
                        index,
                        diagnostic.code,
                        diagnostic.message,
                        diagnostic.start,
                        diagnostic.end,
                    )
                    for index, diagnostic in enumerate(extraction.diagnostics)
                )
                if durable_diagnostic_fields != emitted_diagnostic_fields:
                    raise RuntimeError("durable diagnostics differ from extraction")
                durable_extraction_runs += 1
                durable_extraction_derivations += len(durable.derivations)
                durable_extraction_diagnostics += len(durable.diagnostics)
                durable_extraction_abstentions += durable.abstained
                durable_claim_id_mappings_verified += len(durable.claim_ids)

                record_source_by_id[str(record.id)] = event.source_id
                if record.claims:
                    claim_id = str(record.claims[0].id)
                    claim_ids_by_namespace_source[event.namespace][event.source_id] = (
                        claim_id
                    )
                    claim_source_by_id[claim_id] = event.source_id

                valid_from, valid_to, supersedes_source_id = (
                    _event_level_temporal_metadata(
                        event,
                        extraction,
                        claim_source_by_id,
                    )
                )

                started = time.perf_counter_ns()
                _counted_call(
                    hybrid_counter,
                    hybrid_usage,
                    "ingestion",
                    lambda event=event: hybrid.remember(
                        namespace=event.namespace,
                        content=event.content,
                        source_id=event.source_id,
                        occurred_at=event.occurred_at,
                        recorded_at=event.occurred_at,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        supersedes_source_id=supersedes_source_id,
                    ),
                )
                # Both system ingestion metrics include the one shared extraction
                # cost. The provider is physically invoked once and its validated
                # result is reused for equivalent hybrid temporal metadata.
                hybrid_ingestion_ns += (
                    time.perf_counter_ns() - started
                    + extraction_observation.elapsed_ns
                )

                for case in queries_by_sequence.get(event.sequence, ()):
                    hybrid_result, latency, error, warmup_error = _warm_and_time(
                        lambda case=case: hybrid.recall(
                            namespace=case.namespace,
                            query=case.query,
                            budget_tokens=budget_tokens,
                            as_of=case.as_of,
                        ),
                        counter=hybrid_counter,
                        usage=hybrid_usage,
                    )
                    hybrid_observation = _phase_b_observation(
                        case,
                        source_ids=(
                            hybrid_result.evidence_ids
                            if hybrid_result is not None
                            else ()
                        ),
                        context_tokens=(
                            hybrid_result.estimated_tokens
                            if hybrid_result is not None
                            else 0
                        ),
                        latency_ms=latency,
                        budget_tokens=budget_tokens,
                        error=error,
                        warmup_error=warmup_error,
                    )
                    hybrid_observation["retrieval_diagnostics"] = (
                        asdict(hybrid_result.diagnostics)
                        if hybrid_result is not None
                        else None
                    )
                    hybrid_observations.append(hybrid_observation)

                    capsule, latency, error, warmup_error = _warm_and_time(
                        lambda case=case: candidate.recall(
                            namespace=case.namespace,
                            query=case.query,
                            budget_tokens=budget_tokens,
                            as_of=case.as_of,
                            limit=candidate_limit,
                        ),
                        counter=candidate_counter,
                        usage=candidate_usage,
                    )
                    internal_ids = (
                        tuple(str(item) for item in capsule.evidence_ids)
                        if capsule is not None
                        else ()
                    )
                    source_ids = tuple(
                        record_source_by_id.get(item, f"unmapped:{item}")
                        for item in internal_ids
                    )
                    candidate_observation = _phase_b_observation(
                        case,
                        source_ids=source_ids,
                        context_tokens=(
                            estimate_tokens(capsule.to_prompt())
                            if capsule is not None
                            else 0
                        ),
                        latency_ms=latency,
                        budget_tokens=budget_tokens,
                        error=error,
                        warmup_error=warmup_error,
                    )
                    candidate_observation["retrieval_diagnostics"] = (
                        _candidate_diagnostics(capsule)
                    )
                    candidate_observations.append(candidate_observation)

            candidate_store_stats = asdict(candidate.stats())
            hybrid_index_stats = asdict(hybrid.stats())
            semantic_statuses = {
                namespace: _semantic_status_report(
                    candidate.semantic_status(namespace)
                )
                for namespace in sorted({event.namespace for event in suite.events})
            }
        finally:
            candidate.close()
            hybrid.close()

        # Normalize random insertion/B-tree layout only after clean close. All
        # persistent extraction tables remain in scope; VACUUM time is excluded
        # from ingestion and applied identically to both systems.
        candidate_bytes = _compacted_database_bytes(candidate_path)
        hybrid_bytes = _compacted_database_bytes(hybrid_path)

    extraction_persistence = {
        "mode": "durable",
        "api": "EviCortex.remember_extracted/get_extraction",
        "extraction_runs": durable_extraction_runs,
        "claim_derivation_rows": durable_extraction_derivations,
        "diagnostic_rows": durable_extraction_diagnostics,
        "abstained_runs": durable_extraction_abstentions,
        "get_extraction_records_verified": durable_extraction_runs,
        "claim_id_mappings_verified": durable_claim_id_mappings_verified,
        "included_in_evicortex_database_bytes": True,
    }
    candidate_result = {
        "label": EVICORTEX_PHASE_B_LABEL,
        **_aggregate(
            candidate_observations,
            ingestion_time_ms=candidate_ingestion_ns / 1_000_000,
            database_bytes=candidate_bytes,
            source_bytes=suite.source_bytes,
        ),
        "embedding_usage": _usage_report(candidate_counter, candidate_usage),
        "index_stats": {
            "store": candidate_store_stats,
            "semantic_by_namespace": semantic_statuses,
            "durable_extraction": extraction_persistence,
        },
        "semantic_scan_diagnostics": _aggregate_candidate_diagnostics(
            candidate_observations
        ),
    }
    hybrid_result = {
        "label": DIAGNOSTIC_HYBRID_LABEL,
        **_aggregate(
            hybrid_observations,
            ingestion_time_ms=hybrid_ingestion_ns / 1_000_000,
            database_bytes=hybrid_bytes,
            source_bytes=suite.source_bytes,
        ),
        "embedding_usage": _usage_report(hybrid_counter, hybrid_usage),
        "index_stats": hybrid_index_stats,
        "semantic_scan_diagnostics": _aggregate_hybrid_diagnostics(
            hybrid_observations
        ),
    }

    comparison = _comparison(
        {"baseline": hybrid_result, "evicortex": candidate_result}
    )
    assert comparison is not None
    comparison["baseline_key"] = "hybrid_baseline"
    comparison["interpretation"] = (
        "Arithmetic diagnostic differences only. Positive quality deltas favor the "
        "candidate, negative stale-leakage deltas favor the candidate, and resource "
        "ratios below one use fewer measured resources. These are not superiority "
        "claims because hashing is non-semantic and the comparator is not strong RAG. "
        "Resource ratios are suppressed when either system has measured query errors."
    )

    extraction_report = {
        "scope": (
            "EviCortex.remember_extracted owns one validated extractor invocation "
            "per event and durably persists its run, claim derivations, and "
            "diagnostics. The same in-memory validated output is observed only for "
            "timing/reporting and equivalent hybrid temporal metadata."
        ),
        "extractor_identity": extractor.identity,
        "extractor_invocations": timed_extractor.invocation_count(),
        "invocation_policy": "exactly once per event through remember_extracted",
        "events_processed": len(suite.events),
        "claims_emitted": extraction_claims,
        "provenance_rows": durable_extraction_derivations,
        "provenance_row_semantics": (
            "Durable claim-derivation rows; verified equal to emitted provenance."
        ),
        "abstentions": extraction_abstentions,
        "supersession_references_resolved": extraction_supersessions,
        "diagnostics_total": sum(extraction_diagnostic_codes.values()),
        "diagnostics_by_code": dict(sorted(extraction_diagnostic_codes.items())),
        "diagnostic_events": extraction_diagnostic_events,
        "elapsed_ms_total": sum(extraction_latencies),
        "elapsed_ms_p50": _percentile(extraction_latencies, 0.50),
        "elapsed_ms_p95": _percentile(extraction_latencies, 0.95),
        "persistence": extraction_persistence,
        "transient_observation": {
            "lifetime": (
                "in memory during the run; only aggregate counts and diagnostic "
                "details are retained in this report"
            ),
            "claims": extraction_claims,
            "provenance_rows": extraction_provenance,
            "diagnostics": sum(extraction_diagnostic_codes.values()),
            "purpose": "hybrid metadata parity and extractor timing",
        },
    }

    return {
        "schema_version": 2,
        "benchmark": "evicortex-phase-b-diagnostic-streaming",
        "claim_scope": PHASE_B_SCOPE,
        "config": {
            "events": events,
            "seed": seed,
            "budget_tokens": budget_tokens,
            "candidate_limit": candidate_limit,
            "embedding": {
                "identity": candidate_profile.identity,
                "dimensions": candidate_profile.dimensions,
                "normalized": candidate_profile.normalized,
                "profile_sha256": candidate_profile.profile_sha256,
                "is_semantic": False,
                "diagnostic_only": True,
            },
            "semantic_scan_cap": semantic_scan_cap,
            "hybrid_max_exact_candidates": semantic_scan_cap,
            "minimum_semantic_score": minimum_semantic_score,
            "hybrid_channel_limit": hybrid_channel_limit,
            "hybrid_chunk_chars": list(hybrid_chunk_chars),
            "hybrid_chunk_overlaps": list(hybrid_chunk_overlaps),
            "cache_condition": "one unmeasured warmup per query",
            "database_size_measurement": (
                "compacted durable bytes after clean close and symmetric SQLite "
                "VACUUM; compaction time excluded from ingestion"
            ),
            "ingestion_timing": (
                "Both systems include one canonical extraction invocation; the "
                "validated result is physically computed once and shared. "
                "EviCortex additionally includes durable provenance persistence."
            ),
            "token_estimator": "evicortex.budget.estimate_tokens (UTF-8 bytes/3)",
            "reader_model": None,
            "code_revision": _code_revision(),
            "python_version": platform.python_version(),
            "sqlite_version": sqlite3.sqlite_version,
        },
        "scan_cap_policy": {
            "configured_cap": semantic_scan_cap,
            "evicortex_scope": "per semantic target channel: memory and claim",
            "hybrid_scope": (
                "per exact lexical corpus/match scan and per exact-cosine chunk "
                "granularity"
            ),
            "representation_warning": (
                "The same numeric cap bounds different projection rows and is not an "
                "equal count of source events. EviCortex reports and discards a "
                "truncated semantic channel; the hybrid applies this safety cap to "
                "all exact scans and fails loudly on overflow."
            ),
        },
        "dataset": {
            "events": len(suite.events),
            "queries": len(suite.queries),
            "source_bytes": suite.source_bytes,
            "sha256": _suite_digest(suite),
            "digest_semantics": SUITE_DIGEST_SEMANTICS,
            "tasks": sorted({case.task for case in suite.queries}),
        },
        "metric_semantics": {
            "abstention_accuracy": ABSTENTION_ACCURACY_SEMANTICS,
        },
        "extraction": extraction_report,
        "results": {
            "hybrid_baseline": hybrid_result,
            "evicortex": candidate_result,
        },
        "comparison": comparison,
        "limitations": [
            "HashingEmbedder performs deterministic feature hashing, not semantic inference.",
            "The hybrid comparator is diagnostic and is not a validated strong RAG baseline.",
            "Supported-answer metrics measure gold evidence sufficiency without a reader model.",
            "One extractor result is shared to avoid double extraction and timing variance.",
            "A single synthetic seed is a smoke workload and supports no statistical claim.",
            (
                "Literal 'not X' exclusions can reject truthful evidence that itself "
                "contains the same negated phrase; a structured constraint parser is "
                "still required."
            ),
            (
                "Retrieval-level abstention is uncalibrated; this run observed "
                f"hybrid={hybrid_result['metrics']['abstention_accuracy']:.3f} and "
                f"evicortex={candidate_result['metrics']['abstention_accuracy']:.3f} "
                f"over {sum(case.task == 'abstention' for case in suite.queries)} "
                "unsupported probes, including partial-OOD cases."
            ),
        ],
    }


def _parse_pair(value: str) -> tuple[int, int]:
    pieces = tuple(piece.strip() for piece in value.split(","))
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated integers")
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be integers") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument(
        "--embedding-dimensions", type=int, default=DEFAULT_EMBEDDING_DIMENSIONS
    )
    parser.add_argument(
        "--semantic-scan-cap", type=int, default=DEFAULT_SEMANTIC_SCAN_CAP
    )
    parser.add_argument("--minimum-semantic-score", type=float, default=0.05)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--hybrid-channel-limit", type=int, default=80)
    parser.add_argument("--hybrid-chunk-chars", type=_parse_pair, default=(192, 576))
    parser.add_argument("--hybrid-chunk-overlaps", type=_parse_pair, default=(32, 96))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = run_phase_b(
            events=args.events,
            seed=args.seed,
            budget_tokens=args.budget_tokens,
            embedding_dimensions=args.embedding_dimensions,
            semantic_scan_cap=args.semantic_scan_cap,
            minimum_semantic_score=args.minimum_semantic_score,
            candidate_limit=args.candidate_limit,
            hybrid_channel_limit=args.hybrid_channel_limit,
            hybrid_chunk_chars=args.hybrid_chunk_chars,
            hybrid_chunk_overlaps=args.hybrid_chunk_overlaps,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    print(report["claim_scope"])
    extraction = report["extraction"]
    print(
        "extraction: "
        f"claims={extraction['claims_emitted']} "
        f"abstentions={extraction['abstentions']} "
        f"diagnostics={extraction['diagnostics_total']} "
        f"elapsed_ms={extraction['elapsed_ms_total']:.3f}"
    )
    for name, result in report["results"].items():
        metrics = result["metrics"]
        embedding = result["embedding_usage"]["total"]
        print(
            f"{name}: evidence_recall={metrics['evidence_recall_at_budget']:.3f} "
            f"supported={metrics['supported_answer_rate']:.3f} "
            f"tokens={metrics['estimated_context_tokens_total']} "
            f"p50_ms={metrics['retrieval_latency_ms_p50']:.3f} "
            f"p95_ms={metrics['retrieval_latency_ms_p95']:.3f} "
            f"p99_ms={metrics['retrieval_latency_ms_p99']:.3f} "
            f"embedding_batches={embedding['batch_calls']} "
            f"embedding_texts={embedding['text_count']}"
        )
    for limitation in report["limitations"]:
        print(f"warning: {limitation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
