"""Offline Phase C retrieval evaluation on a local LongMemEval subset.

This runner measures session retrieval after prompt packing for EviCortex and
the metadata-aware diagnostic hybrid comparator.  It does not use a reader
model, download a dataset/model, or make an answer-quality/superiority claim.

Example with an already-cached local Transformers model::

    python -m benchmarks.run_phase_c \
        --dataset C:/data/longmemeval_s_cleaned.json.gz \
        --model C:/models/bge-small-en-v1.5 \
        --revision 7382f1122c10708a1faa0bbe548674a14b1ffe7e \
        --pooling cls \
        --query-prefix "Represent this sentence for searching relevant passages: " \
        --document-prefix "" \
        --budget-tokens 512 --json

The CLI always constructs ``TransformersEmbedder`` with ``local_files_only``.
Programmatic callers can inject any real semantic ``Embedder`` implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import tempfile
import time
from typing import Any

from evicortex import Embedder, EmbeddingProfile, EviCortex
from evicortex.budget import estimate_tokens

from .hybrid_baseline import (
    DIAGNOSTIC_HYBRID_LABEL,
    DiagnosticHybridRAGBaseline,
)
from .longmemeval import (
    LONGMEMEVAL_LICENSE,
    LONGMEMEVAL_SOURCE,
    LongMemEvalCase,
    LongMemEvalDataset,
    LongMemEvalSession,
    load_longmemeval,
    official_longmemeval_variant,
    parse_longmemeval_timestamp,
    session_retrieval_metrics,
)
from .run import _code_revision, _compacted_database_bytes, _percentile
from .run_phase_b import CountingEmbedder


PHASE_C_SCOPE = (
    "Reader-free session-retrieval evaluation on a caller-supplied local "
    "LongMemEval subset. Metrics describe retrieved evidence under a hard prompt "
    "budget; they do not measure answer accuracy or establish superiority."
)
EVICORTEX_PHASE_C_LABEL = "EviCortex semantic retrieval candidate; reader-free"
ABSTENTION_POLICY = (
    "Official-style retrieval metrics skip question IDs ending in '_abs' because "
    "they have no gold answer session. A separate retrieval-only no-evidence rate "
    "is reported for those questions; it is not reader refusal accuracy."
)
EVIDENCE_UNIT_POLICY = (
    "One source row per conversation turn. Session date, zero-based turn index, "
    "role, and content are rendered; has_answer and answer labels are never indexed."
)


@dataclass(frozen=True, slots=True)
class _EvidenceUnit:
    unit_id: str
    session_id: str
    session_date: str
    occurred_at: str
    turn_index: int
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class _DatasetInput:
    dataset: LongMemEvalDataset
    compression: str
    source_file_sha256: str
    source_file_bytes: int


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _finite_score(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a number") from error
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between -1 and 1")
    return result


def _normalize_cutoffs(cutoffs: Iterable[int]) -> tuple[int, ...]:
    if isinstance(cutoffs, (str, bytes)):
        raise TypeError("cutoffs must be an iterable of integers")
    output: list[int] = []
    for value in cutoffs:
        cutoff = _positive_integer(value, "cutoff")
        if cutoff not in output:
            output.append(cutoff)
    if not output:
        raise ValueError("at least one cutoff is required")
    return tuple(sorted(output))


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("cutoffs must be comma-separated integers") from error
    try:
        return _normalize_cutoffs(parts)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _report_path(path: Path) -> str:
    """Prefer a stable workspace-relative path in persisted reports."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load_local_dataset(path: str | Path, *, limit: int | None) -> _DatasetInput:
    """Load JSON or JSON.GZ without network access and preserve both digests."""

    source = Path(path).expanduser().resolve()
    source_bytes = source.stat().st_size
    source_digest = _file_sha256(source)
    if source.suffix.casefold() != ".gz":
        dataset = load_longmemeval(source, limit=limit)
        return _DatasetInput(dataset, "none", source_digest, source_bytes)

    with tempfile.TemporaryDirectory(prefix="evicortex-longmemeval-") as directory:
        logical_path = Path(directory) / "longmemeval.json"
        try:
            with gzip.open(source, "rb") as compressed, logical_path.open("wb") as plain:
                shutil.copyfileobj(compressed, plain, length=1024 * 1024)
        except (gzip.BadGzipFile, EOFError, OSError) as error:
            raise ValueError(f"invalid gzip LongMemEval input: {source}") from error
        loaded = load_longmemeval(logical_path, limit=limit)
        dataset = LongMemEvalDataset(
            path=source,
            sha256=loaded.sha256,
            source_bytes=loaded.source_bytes,
            cases=loaded.cases,
        )
    return _DatasetInput(dataset, "gzip", source_digest, source_bytes)


def _benchmark_timestamp(value: str, *, field: str) -> str:
    """Normalize an official date with the shared locale-independent parser."""

    try:
        parsed = parse_longmemeval_timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _session_units(session: LongMemEvalSession) -> tuple[_EvidenceUnit, ...]:
    occurred_at = _benchmark_timestamp(
        session.date,
        field=f"session {session.session_id!r} date",
    )
    units: list[_EvidenceUnit] = []
    for index, turn in enumerate(session.turns):
        unit_id = f"{session.session_id}::turn:{index:06d}"
        content = (
            f"Session date: {session.date}\n"
            f"Turn index: {index}\n"
            f"{turn.role}: {turn.content}"
        )
        units.append(
            _EvidenceUnit(
                unit_id=unit_id,
                session_id=session.session_id,
                session_date=session.date,
                occurred_at=occurred_at,
                turn_index=index,
                role=turn.role,
                content=content,
            )
        )
    if not units:
        # Retain an empty released session as auditable evidence instead of silently
        # dropping its session ID. This does not add an answer annotation.
        units.append(
            _EvidenceUnit(
                unit_id=f"{session.session_id}::empty",
                session_id=session.session_id,
                session_date=session.date,
                occurred_at=occurred_at,
                turn_index=0,
                role="empty",
                content=f"Session date: {session.date}\nSession contains no turns.",
            )
        )
    return tuple(units)


def _case_units(case: LongMemEvalCase) -> tuple[_EvidenceUnit, ...]:
    return tuple(unit for session in case.sessions for unit in _session_units(session))


def _ranked_unique_sessions(
    evidence_ids: Sequence[str],
    evidence_to_session: Mapping[str, str],
) -> tuple[str, ...]:
    sessions: list[str] = []
    seen: set[str] = set()
    for evidence_id in evidence_ids:
        session_id = evidence_to_session.get(evidence_id)
        if session_id is None:
            raise RuntimeError(f"retrieval returned unmapped evidence id {evidence_id!r}")
        if session_id not in seen:
            seen.add(session_id)
            sessions.append(session_id)
    return tuple(sessions)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_observations(
    observations: Sequence[dict[str, Any]],
    *,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    answerable = [row for row in observations if not row["is_abstention"]]
    abstentions = [row for row in observations if row["is_abstention"]]
    metric_names = tuple(
        f"{prefix}@{cutoff}"
        for cutoff in cutoffs
        for prefix in (
            "recall_fraction",
            "recall_all",
            "recall_any",
            "ndcg_any",
        )
    )
    metrics = {
        name: _mean([float(row["metrics"][name]) for row in answerable])
        for name in metric_names
    }
    context_tokens = [int(row["estimated_context_tokens"]) for row in observations]
    retrieved_counts = [len(row["retrieved_session_ids"]) for row in observations]
    no_evidence = [not row["retrieved_session_ids"] for row in abstentions]
    return {
        "queries": len(observations),
        "answerable_queries_scored": len(answerable),
        "abstention_queries_skipped_by_official_metrics": len(abstentions),
        "session_retrieval": metrics,
        "retrieval_only_abstention_no_evidence_rate": (
            sum(no_evidence) / len(no_evidence) if no_evidence else None
        ),
        "budget_compliance_rate": (
            sum(bool(row["budget_compliant"]) for row in observations)
            / len(observations)
            if observations
            else None
        ),
        "context_tokens_mean": _mean(context_tokens),
        "context_tokens_p95": _percentile(context_tokens, 0.95),
        "context_tokens_max": max(context_tokens, default=0),
        "retrieved_sessions_mean": _mean(retrieved_counts),
        "retrieval_latency_ms_p50": _percentile(
            [float(row["retrieval_latency_ms"]) for row in observations], 0.50
        ),
        "retrieval_latency_ms_p95": _percentile(
            [float(row["retrieval_latency_ms"]) for row in observations], 0.95
        ),
    }


def _system_report(
    observations: Sequence[dict[str, Any]],
    *,
    cutoffs: Sequence[int],
    label: str,
    ingestion_ms: float,
    database_bytes: int,
    embedding_calls: Mapping[str, int],
    index_stats: Mapping[str, Any],
) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_type[str(observation["question_type"])].append(observation)
    return {
        "label": label,
        "metrics": _aggregate_observations(observations, cutoffs=cutoffs),
        "metrics_by_question_type": {
            question_type: _aggregate_observations(rows, cutoffs=cutoffs)
            for question_type, rows in sorted(by_type.items())
        },
        "queries": list(observations),
        "ingestion_time_ms": ingestion_ms,
        "database_bytes_after_compaction": database_bytes,
        "embedding_usage": dict(embedding_calls),
        "index_stats": dict(index_stats),
    }


def _json_mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    candidate = dict(value)
    try:
        encoded = json.dumps(candidate, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError):
        return None
    decoded = json.loads(encoded)
    return decoded if isinstance(decoded, dict) else None


def _embedder_metadata(embedder: Embedder) -> dict[str, Any]:
    profile = EmbeddingProfile.from_embedder(embedder)
    report: dict[str, Any] = {
        "implementation": f"{type(embedder).__module__}.{type(embedder).__qualname__}",
        "identity": profile.identity,
        "dimensions": profile.dimensions,
        "normalized": profile.normalized,
        "profile_sha256": profile.profile_sha256,
        "is_semantic": getattr(embedder, "is_semantic", None) is True,
        "diagnostic_only": getattr(embedder, "diagnostic_only", False) is True,
    }
    identity_fields = _json_mapping(getattr(embedder, "identity_fields", None))
    runtime_metadata = _json_mapping(getattr(embedder, "runtime_metadata", None))
    if identity_fields is not None:
        report["identity_fields"] = identity_fields
    if runtime_metadata is not None:
        report["runtime_metadata"] = runtime_metadata
    return report


def _observation(
    case: LongMemEvalCase,
    *,
    ranked_session_ids: Sequence[str],
    context_tokens: int,
    budget_tokens: int,
    latency_ms: float,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    if context_tokens > budget_tokens:
        raise AssertionError(
            f"hard prompt budget exceeded for {case.question_id}: "
            f"{context_tokens} > {budget_tokens}"
        )
    metrics = session_retrieval_metrics(
        case,
        ranked_session_ids,
        cutoffs=cutoffs,
    )
    return {
        "question_id": case.question_id,
        "question_type": case.question_type,
        "is_abstention": case.is_abstention,
        "gold_session_ids": list(case.answer_session_ids),
        "retrieved_session_ids": list(ranked_session_ids),
        "metrics": metrics,
        "estimated_context_tokens": context_tokens,
        "budget_compliant": True,
        "retrieval_latency_ms": latency_ms,
    }


def run_longmemeval_benchmark(
    dataset_path: str | Path,
    *,
    embedder: Embedder,
    dataset_limit: int | None = None,
    budget_tokens: int = 512,
    cutoffs: Iterable[int] = (5, 10),
    candidate_limit: int = 40,
    semantic_scan_cap: int = 4096,
    minimum_semantic_score: float = 0.05,
    hybrid_chunk_chars: tuple[int, int] = (192, 576),
    hybrid_chunk_overlaps: tuple[int, int] = (32, 96),
    hybrid_channel_limit: int = 80,
    hybrid_candidate_limit: int = 40,
    hybrid_max_exact_candidates: int = 100_000,
) -> dict[str, Any]:
    """Run a local, reader-free, hard-budget session retrieval comparison."""

    if not isinstance(embedder, Embedder):
        raise TypeError("embedder must implement the Embedder protocol")
    if getattr(embedder, "is_semantic", None) is not True:
        raise ValueError("Phase C requires an embedder declaring is_semantic=True")
    if getattr(embedder, "diagnostic_only", False) is True:
        raise ValueError("Phase C rejects diagnostic-only embedders")
    budget = _positive_integer(budget_tokens, "budget_tokens")
    normalized_cutoffs = _normalize_cutoffs(cutoffs)
    candidate_count = _positive_integer(candidate_limit, "candidate_limit")
    scan_cap = _positive_integer(semantic_scan_cap, "semantic_scan_cap")
    hybrid_channel_count = _positive_integer(
        hybrid_channel_limit, "hybrid_channel_limit"
    )
    hybrid_candidate_count = _positive_integer(
        hybrid_candidate_limit, "hybrid_candidate_limit"
    )
    hybrid_exact_cap = _positive_integer(
        hybrid_max_exact_candidates, "hybrid_max_exact_candidates"
    )
    if candidate_count < max(normalized_cutoffs):
        raise ValueError("candidate_limit must be at least the largest cutoff")
    if hybrid_candidate_count < max(normalized_cutoffs):
        raise ValueError("hybrid_candidate_limit must be at least the largest cutoff")
    if hybrid_channel_count < max(normalized_cutoffs):
        raise ValueError("hybrid_channel_limit must be at least the largest cutoff")
    minimum_score = _finite_score(
        minimum_semantic_score,
        "minimum_semantic_score",
    )
    loaded = _load_local_dataset(dataset_path, limit=dataset_limit)
    dataset = loaded.dataset
    if not dataset.cases:
        raise ValueError("the selected LongMemEval subset contains no cases")

    units_by_question = {
        case.question_id: _case_units(case) for case in dataset.cases
    }
    oversized = {
        question_id: len(units)
        for question_id, units in units_by_question.items()
        if len(units) > scan_cap
    }
    if oversized:
        first_id = sorted(oversized)[0]
        raise ValueError(
            "semantic_scan_cap is smaller than the evidence units in namespace "
            f"{first_id!r}: {oversized[first_id]} > {scan_cap}; refusing a "
            "silently incomplete semantic channel"
        )

    embedder_before = EmbeddingProfile.from_embedder(embedder)
    candidate_embedder = CountingEmbedder(embedder, counter_name="evicortex")
    hybrid_embedder = CountingEmbedder(embedder, counter_name="hybrid")
    candidate_observations: list[dict[str, Any]] = []
    hybrid_observations: list[dict[str, Any]] = []
    candidate_memory_to_session: dict[str, str] = {}
    hybrid_source_to_session: dict[str, str] = {}
    evidence_units = 0
    ingestion_candidate_ns = 0
    ingestion_hybrid_ns = 0

    with tempfile.TemporaryDirectory(prefix="evicortex-phase-c-") as directory:
        root = Path(directory)
        candidate_path = root / "evicortex.sqlite3"
        hybrid_path = root / "hybrid.sqlite3"
        candidate = EviCortex.open(
            candidate_path,
            embedder=candidate_embedder,
            semantic_scan_cap=scan_cap,
            minimum_semantic_score=minimum_score,
        )
        hybrid: DiagnosticHybridRAGBaseline | None = None
        try:
            hybrid = DiagnosticHybridRAGBaseline.open(
                hybrid_path,
                embedder=hybrid_embedder,
                chunk_chars=hybrid_chunk_chars,
                chunk_overlaps=hybrid_chunk_overlaps,
                channel_limit=hybrid_channel_count,
                candidate_limit=hybrid_candidate_count,
                max_exact_candidates=hybrid_exact_cap,
                minimum_semantic_score=minimum_score,
            )
            for case in dataset.cases:
                for unit in units_by_question[case.question_id]:
                    evidence_units += 1
                    started = time.perf_counter_ns()
                    record = candidate.remember(
                        namespace=case.namespace,
                        content=unit.content,
                        source=unit.unit_id,
                        occurred_at=unit.occurred_at,
                        recorded_at=unit.occurred_at,
                        metadata={
                            "longmemeval_session_id": unit.session_id,
                            "session_date": unit.session_date,
                            "turn_index": unit.turn_index,
                            "role": unit.role,
                        },
                        idempotency_key=unit.unit_id,
                    )
                    ingestion_candidate_ns += time.perf_counter_ns() - started
                    candidate_memory_to_session[str(record.id)] = unit.session_id

                    started = time.perf_counter_ns()
                    hybrid.remember(
                        namespace=case.namespace,
                        content=unit.content,
                        source_id=unit.unit_id,
                        occurred_at=unit.occurred_at,
                        recorded_at=unit.occurred_at,
                        session_id=unit.session_id,
                        speaker=unit.role,
                        metadata={
                            "session_date": unit.session_date,
                            "turn_index": unit.turn_index,
                        },
                    )
                    ingestion_hybrid_ns += time.perf_counter_ns() - started
                    hybrid_source_to_session[unit.unit_id] = unit.session_id

            for case in dataset.cases:
                as_of = _benchmark_timestamp(
                    case.question_date,
                    field=f"question {case.question_id!r} date",
                )

                started = time.perf_counter_ns()
                capsule = candidate.recall(
                    namespace=case.namespace,
                    query=case.question,
                    budget_tokens=budget,
                    as_of=as_of,
                    limit=candidate_count,
                )
                candidate_latency = (time.perf_counter_ns() - started) / 1_000_000
                if capsule.facts:
                    raise AssertionError("Phase C ingested no claims but recall returned facts")
                candidate_ids = tuple(episode.memory_id for episode in capsule.episodes)
                candidate_sessions = _ranked_unique_sessions(
                    candidate_ids,
                    candidate_memory_to_session,
                )
                candidate_observations.append(
                    _observation(
                        case,
                        ranked_session_ids=candidate_sessions,
                        context_tokens=estimate_tokens(capsule.to_prompt()),
                        budget_tokens=budget,
                        latency_ms=candidate_latency,
                        cutoffs=normalized_cutoffs,
                    )
                )

                started = time.perf_counter_ns()
                hybrid_recall = hybrid.recall(
                    namespace=case.namespace,
                    query=case.question,
                    budget_tokens=budget,
                    as_of=as_of,
                )
                hybrid_latency = (time.perf_counter_ns() - started) / 1_000_000
                hybrid_sessions = _ranked_unique_sessions(
                    tuple(item.source_id for item in hybrid_recall.evidence),
                    hybrid_source_to_session,
                )
                hybrid_observations.append(
                    _observation(
                        case,
                        ranked_session_ids=hybrid_sessions,
                        context_tokens=estimate_tokens(hybrid_recall.to_prompt()),
                        budget_tokens=budget,
                        latency_ms=hybrid_latency,
                        cutoffs=normalized_cutoffs,
                    )
                )

            candidate_stats = asdict(candidate.stats())
            hybrid_stats = asdict(hybrid.stats())
        finally:
            candidate.close()
            if hybrid is not None:
                hybrid.close()

        candidate_bytes = _compacted_database_bytes(candidate_path)
        hybrid_bytes = _compacted_database_bytes(hybrid_path)

    embedder_after = EmbeddingProfile.from_embedder(embedder)
    if embedder_after != embedder_before:
        raise ValueError("embedder compatibility identity changed during the benchmark")

    candidate_usage = asdict(candidate_embedder.snapshot())
    hybrid_usage = asdict(hybrid_embedder.snapshot())
    question_type_counts = dict(
        sorted(Counter(case.question_type for case in dataset.cases).items())
    )
    abstention_count = sum(case.is_abstention for case in dataset.cases)
    official_variant = official_longmemeval_variant(dataset.sha256)
    return {
        "schema_version": 1,
        "benchmark": "evicortex-phase-c-longmemeval-session-retrieval",
        "claim_scope": PHASE_C_SCOPE,
        "config": {
            "budget_tokens": budget,
            "cutoffs": list(normalized_cutoffs),
            "candidate_limit": candidate_count,
            "semantic_scan_cap": scan_cap,
            "minimum_semantic_score": minimum_score,
            "hybrid_chunk_chars": list(hybrid_chunk_chars),
            "hybrid_chunk_overlaps": list(hybrid_chunk_overlaps),
            "hybrid_channel_limit": hybrid_channel_count,
            "hybrid_candidate_limit": hybrid_candidate_count,
            "hybrid_max_exact_candidates": hybrid_exact_cap,
            "token_estimator": "evicortex.budget.estimate_tokens (UTF-8 bytes/3)",
            "query_rendering": "raw LongMemEval question; question_date is as_of only",
            "evidence_unit_policy": EVIDENCE_UNIT_POLICY,
            "representation_note": (
                "Both systems receive the same rendered turn sources. EviCortex "
                "embeds its canonical memory envelope; the hybrid embeds its "
                "configured child chunks. Those system-specific representations "
                "are intentionally measured, not asserted to be identical."
            ),
            "date_timezone_assumption": (
                "Naive released LongMemEval dates are interpreted as UTC for "
                "deterministic ordering; original date text remains in evidence."
            ),
            "reader_model": None,
            "network_access": (
                "The runner performs none and the CLI forces local_files_only; a "
                "programmatically injected provider remains caller-controlled."
            ),
            "embedder_sharing": (
                "Both systems use separate counting wrappers over the same injected "
                "provider instance; quality configuration is identical."
            ),
            "latency_policy": (
                "Single cold-order pass, EviCortex then hybrid per query; latency "
                "is exploratory and not a statistically controlled comparison."
            ),
        },
        "dataset": {
            "input_kind": (
                "byte-verified official LongMemEval cleaned artifact"
                if official_variant is not None
                else "caller-supplied LongMemEval-schema data"
            ),
            "official_artifact_verified": official_variant is not None,
            "declared_source": (
                LONGMEMEVAL_SOURCE if official_variant is not None else None
            ),
            "declared_variant": official_variant,
            "declared_license": (
                LONGMEMEVAL_LICENSE if official_variant is not None else None
            ),
            "local_path": _report_path(dataset.path),
            "compression": loaded.compression,
            "source_file_sha256": loaded.source_file_sha256,
            "source_file_bytes": loaded.source_file_bytes,
            "logical_json_sha256": dataset.sha256,
            "logical_json_bytes": dataset.source_bytes,
            "selected_cases": len(dataset.cases),
            "answerable_cases": len(dataset.cases) - abstention_count,
            "abstention_cases": abstention_count,
            "question_types": question_type_counts,
            "evidence_units": evidence_units,
            "dataset_limit": dataset_limit,
        },
        "embedding": _embedder_metadata(embedder),
        "evaluation_policy": {
            "gold_unit": "LongMemEval answer_session_ids",
            "rank_unit": "first occurrence of each session in packed evidence order",
            "metrics_stage": "after hard-budget prompt packing",
            "ndcg_naming": (
                "ndcg_any@k reproduces the released LongMemEval eval_utils.py "
                "binary-relevance DCG formula, including equal rank-1/rank-2 weight."
            ),
            "abstention": ABSTENTION_POLICY,
            "answer_and_has_answer_ingested": False,
            "superiority_claim": False,
        },
        "systems": {
            "evicortex": _system_report(
                candidate_observations,
                cutoffs=normalized_cutoffs,
                label=EVICORTEX_PHASE_C_LABEL,
                ingestion_ms=ingestion_candidate_ns / 1_000_000,
                database_bytes=candidate_bytes,
                embedding_calls=candidate_usage,
                index_stats=candidate_stats,
            ),
            "hybrid_baseline": _system_report(
                hybrid_observations,
                cutoffs=normalized_cutoffs,
                label=DIAGNOSTIC_HYBRID_LABEL,
                ingestion_ms=ingestion_hybrid_ns / 1_000_000,
                database_bytes=hybrid_bytes,
                embedding_calls=hybrid_usage,
                index_stats=hybrid_stats,
            ),
        },
        "reproducibility": {
            "code_revision": _code_revision(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sqlite_version": sqlite3.sqlite_version,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--model", required=True, help="local path or cached model ID")
    parser.add_argument("--revision", required=True, help="pinned local model revision")
    parser.add_argument(
        "--code-revision",
        help="separate 40-character commit pin required with --trust-remote-code",
    )
    parser.add_argument(
        "--local-artifact-sha256",
        help="required content pin when --model is an existing filesystem path",
    )
    parser.add_argument("--pooling", choices=("cls", "mean", "last_token"), required=True)
    parser.add_argument("--query-prefix", required=True)
    parser.add_argument("--document-prefix", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overflow-policy", choices=("error", "truncate"), default="error")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--cutoffs", type=_parse_cutoffs, default=(5, 10))
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--semantic-scan-cap", type=int, default=4096)
    parser.add_argument("--minimum-semantic-score", type=float, default=0.05)
    parser.add_argument("--hybrid-channel-limit", type=int, default=80)
    parser.add_argument("--hybrid-candidate-limit", type=int, default=40)
    parser.add_argument("--hybrid-max-exact-candidates", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Deliberately lazy: importing the benchmark remains stdlib-only apart from
    # EviCortex, and the adapter itself forces local_files_only=True.
    from evicortex.providers import TransformersEmbedder

    embedder = TransformersEmbedder(
        args.model,
        revision=args.revision,
        code_revision=args.code_revision,
        pooling=args.pooling,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        max_length=args.max_length,
        overflow_policy=args.overflow_policy,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        local_artifact_sha256=args.local_artifact_sha256,
    )
    report = run_longmemeval_benchmark(
        args.dataset,
        embedder=embedder,
        dataset_limit=args.dataset_limit,
        budget_tokens=args.budget_tokens,
        cutoffs=args.cutoffs,
        candidate_limit=args.candidate_limit,
        semantic_scan_cap=args.semantic_scan_cap,
        minimum_semantic_score=args.minimum_semantic_score,
        hybrid_channel_limit=args.hybrid_channel_limit,
        hybrid_candidate_limit=args.hybrid_candidate_limit,
        hybrid_max_exact_candidates=args.hybrid_max_exact_candidates,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        for key, system in report["systems"].items():
            metrics = system["metrics"]
            recall = metrics["session_retrieval"].get(
                f"recall_all@{max(args.cutoffs)}"
            )
            print(
                f"{key}: recall_all@{max(args.cutoffs)}={recall} "
                f"context_tokens_mean={metrics['context_tokens_mean']}"
            )
        print("Reader-free retrieval metrics only; no superiority claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABSTENTION_POLICY",
    "EVIDENCE_UNIT_POLICY",
    "EVICORTEX_PHASE_C_LABEL",
    "PHASE_C_SCOPE",
    "run_longmemeval_benchmark",
]
