"""Run the deterministic Prototype A EviCortex/lexical-baseline comparison.

Example::

    python -m benchmarks.run --events 1000 --seed 7 --budget-tokens 512 --json

The reported ``supported_answer_rate`` is an evidence-sufficiency proxy: no LLM
reader is used in Prototype A.  The baseline is intentionally labelled as a
mechanical lexical comparator, not as representative of every RAG system.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import tempfile
import time
from typing import Any

from .baseline import FixedChunkFTS5Baseline, estimate_tokens
from .synthetic import ClaimSpec, QueryCase, SyntheticSuite, generate_suite


BASELINE_LABEL = "mechanical fixed-chunk SQLite FTS5 lexical RAG baseline"
CANDIDATE_LABEL = "EviCortex deterministic Prototype A kernel"
SUITE_DIGEST_SEMANTICS = "evicortex.synthetic-suite.complete-canonical-json.v2"
SOURCE_DIGEST_SEMANTICS = (
    "evicortex.runtime-source-tree.path-and-content-framed-sha256.v1"
)
ABSTENTION_ACCURACY_SEMANTICS = (
    "Retrieval-level no-evidence abstention: fraction of annotated unanswerable "
    "probes whose packed context contains zero evidence IDs. This does not measure "
    "reader-model refusal or calibrated answer uncertainty."
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _database_bytes(path: Path) -> int:
    """Measure SQLite files after every owning connection is closed."""

    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


def _compacted_database_bytes(path: Path) -> int:
    """Normalize durable page layout with VACUUM, then measure closed files."""

    if not path.exists():
        return 0
    connection = sqlite3.connect(path)
    try:
        connection.execute("VACUUM")
    finally:
        connection.close()
    return _database_bytes(path)


def _code_revision() -> dict[str, Any]:
    """Identify both the Git base and exact runtime source, including dirty files."""

    root = Path(__file__).resolve().parents[1]
    candidates = [root / "pyproject.toml"]
    for directory in (root / "benchmarks", root / "src" / "evicortex"):
        if directory.is_dir():
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and (path.suffix == ".py" or path.name == "py.typed")
            )
    files = tuple(sorted({path.resolve() for path in candidates if path.is_file()}))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    git_commit: str | None = None
    git_dirty: bool | None = None
    try:
        commit = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            candidate = commit.stdout.strip()
            if candidate:
                git_commit = candidate
        status = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            git_dirty = bool(status.stdout)
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runtime_source_sha256": digest.hexdigest(),
        "runtime_source_files": len(files),
        "source_digest_semantics": SOURCE_DIGEST_SEMANTICS,
        "source_digest_scope": [
            "pyproject.toml",
            "benchmarks/**/*.py",
            "src/evicortex/**/*.py",
            "src/evicortex/py.typed",
        ],
    }


def _suite_digest(suite: SyntheticSuite) -> str:
    """Hash every suite field that can define a workload or metric."""

    payload = {
        "digest_semantics": SUITE_DIGEST_SEMANTICS,
        "suite": asdict(suite),
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _timed_recall(call: Callable[[], Any]) -> tuple[Any | None, float, str | None]:
    """Warm a deterministic query once, then time the identical query."""

    try:
        call()
    except Exception:
        # Repeat inside the measured region so failures appear in the report.
        pass
    started = time.perf_counter_ns()
    try:
        value = call()
        error = None
    except Exception as exc:  # surfaced per-query rather than corrupting metrics
        value = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return value, elapsed_ms, error


def _query_observation(
    case: QueryCase,
    *,
    source_ids: Iterable[str],
    estimated_context_tokens: int,
    retrieval_latency_ms: float,
    budget_tokens: int,
    error: str | None = None,
) -> dict[str, Any]:
    # Source IDs, unlike EviCortex's internal UUIDs, are stable across fresh runs.
    # Metrics are set-based, so canonicalize their report order as well.
    retrieved = tuple(sorted(set(source_ids)))
    retrieved_set = set(retrieved)
    gold = set(case.gold_evidence_ids)
    forbidden = set(case.forbidden_evidence_ids)
    if gold:
        recall = len(gold & retrieved_set) / len(gold)
        supported: bool | None = gold <= retrieved_set
    else:
        recall = None
        supported = None
    # Retrieval failure is not a successful no-evidence decision.
    abstained = not retrieved and error is None
    forbidden_leakage = bool(forbidden & retrieved_set)
    return {
        "query_id": case.query_id,
        "task": case.task,
        "namespace": case.namespace,
        "query": case.query,
        "after_sequence": case.after_sequence,
        "as_of": case.as_of,
        "expected_answerable": case.expected_answerable,
        "expected_values": list(case.expected_values),
        "gold_evidence_ids": list(case.gold_evidence_ids),
        "forbidden_evidence_ids": list(case.forbidden_evidence_ids),
        "retrieved_source_ids": list(retrieved),
        "gold_evidence_count": len(gold),
        "retrieved_evidence_count": len(retrieved),
        "evidence_recall": recall,
        "has_sufficient_gold_evidence": supported,
        "abstained": abstained,
        "forbidden_evidence_leakage": forbidden_leakage,
        "estimated_context_tokens": estimated_context_tokens,
        # A failed retrieval produced no usable context.  Treating its synthetic
        # zero-token payload as compliant would reward an execution failure.
        "budget_compliant": error is None
        and estimated_context_tokens <= budget_tokens,
        "retrieval_latency_ms": retrieval_latency_ms,
        "error": error,
    }


def _mean_bools(values: Iterable[bool]) -> float | None:
    collected = list(values)
    return (sum(collected) / len(collected)) if collected else None


def _mean_numbers(values: Iterable[float | int | None]) -> float | None:
    collected = [float(value) for value in values if value is not None]
    return statistics.fmean(collected) if collected else None


def _aggregate(
    observations: list[dict[str, Any]],
    *,
    ingestion_time_ms: float,
    database_bytes: int,
    source_bytes: int,
) -> dict[str, Any]:
    answerable = [item for item in observations if item["expected_answerable"]]
    current = [item for item in observations if item["task"] == "current"]
    historical = [item for item in observations if item["task"] == "historical"]
    stale_cases = [item for item in current if item["forbidden_evidence_ids"]]
    abstention = [item for item in observations if item["task"] == "abstention"]
    namespace = [item for item in observations if item["task"] == "namespace_isolation"]
    latencies = [float(item["retrieval_latency_ms"]) for item in observations]
    context_tokens = [int(item["estimated_context_tokens"]) for item in observations]

    by_task: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[str(item["task"])].append(item)
    for task in sorted(grouped):
        rows = grouped[task]
        task_answerable = [row for row in rows if row["expected_answerable"]]
        by_task[task] = {
            "queries": len(rows),
            "evidence_recall_at_budget": _mean_numbers(
                row["evidence_recall"] for row in task_answerable
            ),
            "supported_answer_rate": _mean_bools(
                bool(row["has_sufficient_gold_evidence"])
                for row in task_answerable
            ),
            "forbidden_evidence_leakage_rate": _mean_bools(
                bool(row["forbidden_evidence_leakage"])
                for row in rows
                if row["forbidden_evidence_ids"]
            ),
            "mean_estimated_context_tokens": _mean_numbers(
                row["estimated_context_tokens"] for row in rows
            ),
        }

    current_accuracy = _mean_bools(
        bool(row["has_sufficient_gold_evidence"])
        and not bool(row["forbidden_evidence_leakage"])
        for row in current
    )
    historical_accuracy = _mean_bools(
        bool(row["has_sufficient_gold_evidence"])
        and not bool(row["forbidden_evidence_leakage"])
        for row in historical
    )
    return {
        "metrics": {
            "evidence_recall_at_budget": _mean_numbers(
                row["evidence_recall"] for row in answerable
            ),
            "supported_answer_rate": _mean_bools(
                bool(row["has_sufficient_gold_evidence"])
                for row in answerable
            ),
            "current_accuracy": current_accuracy,
            "historical_as_of_accuracy": historical_accuracy,
            "stale_leakage_rate": _mean_bools(
                bool(row["forbidden_evidence_leakage"])
                for row in stale_cases
            ),
            "abstention_accuracy": _mean_bools(
                bool(row["abstained"]) for row in abstention
            ),
            "namespace_isolation_accuracy": _mean_bools(
                bool(row["has_sufficient_gold_evidence"])
                and not bool(row["forbidden_evidence_leakage"])
                for row in namespace
            ),
            "budget_compliance_rate": _mean_bools(
                bool(row["budget_compliant"]) for row in observations
            ),
            "estimated_context_tokens_total": sum(context_tokens),
            "estimated_context_tokens_mean": _mean_numbers(context_tokens),
            "retrieval_latency_ms_p50": _percentile(latencies, 0.50),
            "retrieval_latency_ms_p95": _percentile(latencies, 0.95),
            "retrieval_latency_ms_p99": _percentile(latencies, 0.99),
            "ingestion_time_ms": ingestion_time_ms,
            "database_bytes": database_bytes,
            "storage_overhead_ratio": (
                database_bytes / source_bytes if source_bytes else None
            ),
            "query_errors": sum(item["error"] is not None for item in observations),
        },
        "by_task": by_task,
        "queries": observations,
    }


def _build_claim_input(
    specification: ClaimSpec,
    *,
    claim_id_by_source: dict[str, str],
    claim_input_type: type[Any],
) -> Any:
    supersedes = None
    if specification.supersedes_source_id is not None:
        supersedes = claim_id_by_source.get(specification.supersedes_source_id)
        if supersedes is None:
            raise RuntimeError(
                "superseded source was not ingested before its replacement: "
                f"{specification.supersedes_source_id}"
            )
    return claim_input_type(
        subject=specification.subject,
        predicate=specification.predicate,
        value=specification.value,
        valid_from=specification.valid_from,
        valid_to=specification.valid_to,
        confidence=specification.confidence,
        authority=specification.authority,
        supersedes=supersedes,
    )


def _comparison(results: dict[str, Any]) -> dict[str, Any] | None:
    if "evicortex" not in results:
        return None
    baseline = results["baseline"]["metrics"]
    candidate = results["evicortex"]["metrics"]
    baseline_query_errors = int(baseline.get("query_errors") or 0)
    candidate_query_errors = int(candidate.get("query_errors") or 0)
    resource_ratios_comparable = not (
        baseline_query_errors or candidate_query_errors
    )

    def delta(name: str) -> float | None:
        left = candidate.get(name)
        right = baseline.get(name)
        if left is None or right is None:
            return None
        return float(left) - float(right)

    def ratio(name: str) -> float | None:
        if not resource_ratios_comparable:
            return None
        numerator = candidate.get(name)
        denominator = baseline.get(name)
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator) / float(denominator)

    return {
        "candidate_minus_baseline": {
            "evidence_recall_at_budget": delta("evidence_recall_at_budget"),
            "supported_answer_rate": delta("supported_answer_rate"),
            "current_accuracy": delta("current_accuracy"),
            "historical_as_of_accuracy": delta("historical_as_of_accuracy"),
            "stale_leakage_rate": delta("stale_leakage_rate"),
        },
        "candidate_over_baseline": {
            "estimated_context_tokens_total": ratio(
                "estimated_context_tokens_total"
            ),
            "retrieval_latency_ms_p50": ratio("retrieval_latency_ms_p50"),
            "retrieval_latency_ms_p95": ratio("retrieval_latency_ms_p95"),
            "retrieval_latency_ms_p99": ratio("retrieval_latency_ms_p99"),
            "ingestion_time_ms": ratio("ingestion_time_ms"),
            "database_bytes": ratio("database_bytes"),
        },
        "resource_ratios_comparable": resource_ratios_comparable,
        "resource_ratios_exclusion_reason": (
            None
            if resource_ratios_comparable
            else (
                "Resource ratios are suppressed because measured query errors make "
                "cross-system resource interpretation unsafe "
                f"(candidate_errors={candidate_query_errors};"
                f"baseline_errors={baseline_query_errors})."
            )
        ),
        "interpretation": (
            "Positive quality deltas favor EviCortex; negative stale-leakage deltas "
            "favor EviCortex; ratios below 1 use fewer resources. This compares only "
            "with the named mechanical lexical baseline."
        ),
    }


def run_benchmark(
    *,
    events: int,
    seed: int,
    budget_tokens: int,
    chunk_chars: int = 320,
    chunk_overlap: int = 48,
    candidate_limit: int = 40,
    include_evicortex: bool = True,
) -> dict[str, Any]:
    """Run both systems over identical canonical events and streaming snapshots."""

    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    suite = generate_suite(event_count=events, seed=seed)
    queries_by_sequence: dict[int, list[QueryCase]] = defaultdict(list)
    for case in suite.queries:
        queries_by_sequence[case.after_sequence].append(case)

    with tempfile.TemporaryDirectory(prefix="evicortex-benchmark-") as directory:
        root = Path(directory)
        baseline_path = root / "baseline.db"
        baseline = FixedChunkFTS5Baseline(
            baseline_path,
            chunk_chars=chunk_chars,
            overlap_chars=chunk_overlap,
            candidate_limit=candidate_limit,
        )

        candidate = None
        claim_input_type: type[Any] | None = None
        candidate_path = root / "evicortex.db"
        if include_evicortex:
            try:
                from evicortex import ClaimInput, EviCortex
            except ImportError as exc:  # a clear failure for the comparison CLI
                baseline.close()
                raise RuntimeError(
                    "EviCortex/ClaimInput public interfaces are unavailable"
                ) from exc
            candidate = EviCortex.open(candidate_path)
            claim_input_type = ClaimInput

        baseline_ingestion_ns = 0
        candidate_ingestion_ns = 0
        baseline_observations: list[dict[str, Any]] = []
        candidate_observations: list[dict[str, Any]] = []
        record_id_to_source: dict[str, str] = {}
        claim_id_by_source: dict[str, str] = {}

        try:
            for event in suite.events:
                started = time.perf_counter_ns()
                baseline.remember(
                    namespace=event.namespace,
                    content=event.content,
                    source_id=event.source_id,
                    occurred_at=event.occurred_at,
                )
                baseline_ingestion_ns += time.perf_counter_ns() - started

                if candidate is not None and claim_input_type is not None:
                    claims = tuple(
                        _build_claim_input(
                            claim,
                            claim_id_by_source=claim_id_by_source,
                            claim_input_type=claim_input_type,
                        )
                        for claim in event.claims
                    )
                    started = time.perf_counter_ns()
                    record = candidate.remember(
                        namespace=event.namespace,
                        content=event.content,
                        source=event.source_id,
                        occurred_at=event.occurred_at,
                        recorded_at=event.occurred_at,
                        claims=claims,
                    )
                    candidate_ingestion_ns += time.perf_counter_ns() - started
                    record_id_to_source[str(record.id)] = event.source_id
                    for stored_claim in getattr(record, "claims", ()):
                        claim_id_by_source[event.source_id] = str(stored_claim.id)

                for case in queries_by_sequence.get(event.sequence, ()):
                    baseline_result, latency, error = _timed_recall(
                        lambda case=case: baseline.recall(
                            namespace=case.namespace,
                            query=case.query,
                            budget_tokens=budget_tokens,
                            as_of=case.as_of,
                        )
                    )
                    baseline_observations.append(
                        _query_observation(
                            case,
                            source_ids=(
                                baseline_result.evidence_ids
                                if baseline_result is not None
                                else ()
                            ),
                            estimated_context_tokens=(
                                baseline_result.estimated_tokens
                                if baseline_result is not None
                                else 0
                            ),
                            retrieval_latency_ms=latency,
                            budget_tokens=budget_tokens,
                            error=error,
                        )
                    )

                    if candidate is not None:
                        capsule, latency, error = _timed_recall(
                            lambda case=case: candidate.recall(
                                namespace=case.namespace,
                                query=case.query,
                                budget_tokens=budget_tokens,
                                as_of=case.as_of,
                                limit=candidate_limit,
                            )
                        )
                        internal_ids = (
                            tuple(str(item) for item in capsule.evidence_ids)
                            if capsule is not None
                            else ()
                        )
                        source_ids = tuple(
                            record_id_to_source.get(item, f"unmapped:{item}")
                            for item in internal_ids
                        )
                        prompt_tokens = (
                            estimate_tokens(capsule.to_prompt())
                            if capsule is not None
                            else 0
                        )
                        candidate_observations.append(
                            _query_observation(
                                case,
                                source_ids=source_ids,
                                estimated_context_tokens=prompt_tokens,
                                retrieval_latency_ms=latency,
                                budget_tokens=budget_tokens,
                                error=error,
                            )
                        )

        finally:
            baseline.close()
            if candidate is not None:
                candidate.close()

        # Normalize random insertion/B-tree layout only after clean close. VACUUM
        # time is deliberately outside ingestion timing and applied symmetrically.
        baseline_bytes = _compacted_database_bytes(baseline_path)
        candidate_bytes = (
            _compacted_database_bytes(candidate_path)
            if candidate is not None
            else 0
        )

    system_results: dict[str, Any] = {
        "baseline": {
            "label": BASELINE_LABEL,
            **_aggregate(
                baseline_observations,
                ingestion_time_ms=baseline_ingestion_ns / 1_000_000,
                database_bytes=baseline_bytes,
                source_bytes=suite.source_bytes,
            ),
        }
    }
    if include_evicortex:
        system_results["evicortex"] = {
            "label": CANDIDATE_LABEL,
            **_aggregate(
                candidate_observations,
                ingestion_time_ms=candidate_ingestion_ns / 1_000_000,
                database_bytes=candidate_bytes,
                source_bytes=suite.source_bytes,
            ),
        }

    report: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "evicortex-prototype-a-synthetic-streaming",
        "claim_scope": (
            "Deterministic comparison with a mechanical fixed-chunk lexical FTS5 "
            "baseline; not a claim against all RAG systems."
        ),
        "config": {
            "events": events,
            "seed": seed,
            "budget_tokens": budget_tokens,
            "chunk_chars": chunk_chars,
            "chunk_overlap": chunk_overlap,
            "candidate_limit": candidate_limit,
            "token_estimator": "evicortex.budget.estimate_tokens (UTF-8 bytes/3)",
            "reader_model": None,
            "cache_condition": "one unmeasured warmup per query",
            "database_size_measurement": (
                "compacted durable bytes after clean close and symmetric SQLite "
                "VACUUM; compaction time excluded from ingestion"
            ),
            "code_revision": _code_revision(),
            "python_version": platform.python_version(),
            "sqlite_version": sqlite3.sqlite_version,
        },
        "dataset": {
            "source_bytes": suite.source_bytes,
            "sha256": _suite_digest(suite),
            "digest_semantics": SUITE_DIGEST_SEMANTICS,
            "events": len(suite.events),
            "queries": len(suite.queries),
            "tasks": sorted({case.task for case in suite.queries}),
        },
        "metric_semantics": {
            "supported_answer_rate": (
                "Fraction of answerable queries whose packed context contains every "
                "gold source ID; no reader model is used."
            ),
            "current_accuracy": (
                "Current probes with all gold source IDs and no annotated stale "
                "source IDs in the packed context."
            ),
            "historical_as_of_accuracy": (
                "Historical probes with all gold source IDs and no annotated future "
                "source IDs in the packed context."
            ),
            "stale_leakage_rate": (
                "Current post-update probes whose packed context contains an "
                "annotated superseded source."
            ),
            "abstention_accuracy": ABSTENTION_ACCURACY_SEMANTICS,
        },
        "results": system_results,
    }
    comparison = _comparison(system_results)
    if comparison is not None:
        report["comparison"] = comparison
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--chunk-chars", type=int, default=320)
    parser.add_argument("--chunk-overlap", type=int, default=48)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete machine-readable report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_benchmark(
            events=args.events,
            seed=args.seed,
            budget_tokens=args.budget_tokens,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            candidate_limit=args.candidate_limit,
        )
    except (RuntimeError, ValueError) as exc:
        _parser().error(str(exc))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["claim_scope"])
        for name, result in report["results"].items():
            metrics = result["metrics"]
            print(
                f"{name}: evidence_recall={metrics['evidence_recall_at_budget']:.3f} "
                f"supported={metrics['supported_answer_rate']:.3f} "
                f"tokens={metrics['estimated_context_tokens_total']} "
                f"p50_ms={metrics['retrieval_latency_ms_p50']:.3f} "
                f"p95_ms={metrics['retrieval_latency_ms_p95']:.3f} "
                f"p99_ms={metrics['retrieval_latency_ms_p99']:.3f} "
                f"db_bytes={metrics['database_bytes']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
