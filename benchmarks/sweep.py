"""Aggregate Prototype A results across independently seeded synthetic streams.

The bootstrap intervals in this module are descriptive smoke-benchmark intervals.
They use the synthetic stream seed as the paired resampling unit and must not be
interpreted as a statistical-significance test or as evidence about all RAG systems.

Example::

    python -m benchmarks.sweep --seeds 7,11,19 --events 100 \
        --budget-tokens 512 --bootstrap-samples 2000 --json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
import random
import statistics
from typing import Any

from .run import run_benchmark


DEFAULT_SEEDS = (7, 11, 19)
DEFAULT_BOOTSTRAP_SEED = 20_260_721

QUALITY_METRICS: dict[str, str] = {
    "evidence_recall_at_budget": "higher_is_better",
    "supported_answer_rate": "higher_is_better",
    "current_accuracy": "higher_is_better",
    "historical_as_of_accuracy": "higher_is_better",
    "stale_leakage_rate": "lower_is_better",
    "abstention_accuracy": "higher_is_better",
    "namespace_isolation_accuracy": "higher_is_better",
}
RESOURCE_METRICS: dict[str, str] = {
    "estimated_context_tokens_total": "lower_is_better",
    "retrieval_latency_ms_p50": "lower_is_better",
    "retrieval_latency_ms_p95": "lower_is_better",
    "retrieval_latency_ms_p99": "lower_is_better",
    "ingestion_time_ms": "lower_is_better",
    "database_bytes": "lower_is_better",
}


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _metric_seed(bootstrap_seed: int, label: str) -> int:
    """Derive a stable RNG seed so adding a metric does not perturb other CIs."""

    digest = hashlib.sha256(f"{bootstrap_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _interval_record(
    *,
    estimate: float | None,
    distribution: Sequence[float],
    independent_streams: int,
    direction: str,
) -> dict[str, Any]:
    if estimate is None or not distribution:
        interval: list[float | None] = [None, None]
    else:
        interval = [_quantile(distribution, 0.025), _quantile(distribution, 0.975)]
    return {
        "estimate": estimate,
        "ci95": interval,
        "independent_streams": independent_streams,
        "direction": direction,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    label: str,
    direction: str,
) -> dict[str, Any]:
    """Return a deterministic percentile-bootstrap interval for a stream mean."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    clean = tuple(float(value) for value in values if _is_number(value))
    if not clean:
        return _interval_record(
            estimate=None,
            distribution=(),
            independent_streams=0,
            direction=direction,
        )
    estimate = statistics.fmean(clean)
    rng = random.Random(_metric_seed(seed, label))
    distribution = [
        statistics.fmean(clean[rng.randrange(len(clean))] for _ in clean)
        for _ in range(samples)
    ]
    return _interval_record(
        estimate=estimate,
        distribution=distribution,
        independent_streams=len(clean),
        direction=direction,
    )


def bootstrap_paired_delta_ci(
    pairs: Sequence[tuple[float, float]],
    *,
    samples: int,
    seed: int,
    label: str,
    direction: str,
) -> dict[str, Any]:
    """Bootstrap the mean candidate-minus-baseline delta across streams."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    clean = tuple(
        (float(candidate), float(baseline))
        for candidate, baseline in pairs
        if _is_number(candidate) and _is_number(baseline)
    )
    if not clean:
        return _interval_record(
            estimate=None,
            distribution=(),
            independent_streams=0,
            direction=direction,
        )

    def statistic(selected: Sequence[tuple[float, float]]) -> float:
        return statistics.fmean(candidate - baseline for candidate, baseline in selected)

    estimate = statistic(clean)
    rng = random.Random(_metric_seed(seed, label))
    distribution = [
        statistic(tuple(clean[rng.randrange(len(clean))] for _ in clean))
        for _ in range(samples)
    ]
    result = _interval_record(
        estimate=estimate,
        distribution=distribution,
        independent_streams=len(clean),
        direction=direction,
    )
    result["operation"] = "candidate_minus_baseline"
    return result


def bootstrap_paired_ratio_ci(
    pairs: Sequence[tuple[float, float]],
    *,
    samples: int,
    seed: int,
    label: str,
    direction: str,
) -> dict[str, Any]:
    """Bootstrap an aggregate candidate-over-baseline resource ratio."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    clean = tuple(
        (float(candidate), float(baseline))
        for candidate, baseline in pairs
        if _is_number(candidate)
        and _is_number(baseline)
        and float(baseline) > 0.0
    )
    if not clean:
        result = _interval_record(
            estimate=None,
            distribution=(),
            independent_streams=0,
            direction=direction,
        )
        result["operation"] = "candidate_over_baseline"
        return result

    def statistic(selected: Sequence[tuple[float, float]]) -> float:
        denominator = sum(baseline for _, baseline in selected)
        if denominator <= 0.0:
            raise ValueError("baseline resource total must be positive")
        return sum(candidate for candidate, _ in selected) / denominator

    estimate = statistic(clean)
    rng = random.Random(_metric_seed(seed, label))
    distribution = [
        statistic(tuple(clean[rng.randrange(len(clean))] for _ in clean))
        for _ in range(samples)
    ]
    result = _interval_record(
        estimate=estimate,
        distribution=distribution,
        independent_streams=len(clean),
        direction=direction,
    )
    result["operation"] = "candidate_over_baseline"
    return result


def _system_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "quality": {name: metrics.get(name) for name in QUALITY_METRICS},
        "resources": {name: metrics.get(name) for name in RESOURCE_METRICS},
        "query_errors": metrics.get("query_errors"),
        "budget_compliance_rate": metrics.get("budget_compliance_rate"),
    }


def _per_seed_summary(report: dict[str, Any]) -> dict[str, Any]:
    systems = {
        name: _system_summary(result)
        for name, result in report["results"].items()
    }
    summary: dict[str, Any] = {
        "seed": report["config"]["seed"],
        "dataset_sha256": report["dataset"]["sha256"],
        "source_bytes": report["dataset"]["source_bytes"],
        "systems": systems,
    }
    comparison = report.get("comparison")
    if comparison is not None:
        summary["comparison"] = {
            "quality_deltas": comparison["candidate_minus_baseline"],
            "resource_ratios": comparison["candidate_over_baseline"],
        }
    return summary


def _system_aggregate(
    reports: Sequence[dict[str, Any]],
    *,
    system: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics = [report["results"][system]["metrics"] for report in reports]
    quality = {
        name: bootstrap_mean_ci(
            [item[name] for item in metrics if _is_number(item.get(name))],
            samples=samples,
            seed=seed,
            label=f"system:{system}:quality:{name}",
            direction=direction,
        )
        for name, direction in QUALITY_METRICS.items()
    }
    resources = {
        name: bootstrap_mean_ci(
            [item[name] for item in metrics if _is_number(item.get(name))],
            samples=samples,
            seed=seed,
            label=f"system:{system}:resource:{name}",
            direction=direction,
        )
        for name, direction in RESOURCE_METRICS.items()
    }
    return {"quality": quality, "resources": resources}


def _paired_values(
    reports: Sequence[dict[str, Any]], metric: str
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for report in reports:
        baseline = report["results"]["baseline"]["metrics"].get(metric)
        candidate = report["results"]["evicortex"]["metrics"].get(metric)
        if _is_number(baseline) and _is_number(candidate):
            pairs.append((float(candidate), float(baseline)))
    return pairs


def _comparison_aggregate(
    reports: Sequence[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    quality = {
        name: bootstrap_paired_delta_ci(
            _paired_values(reports, name),
            samples=samples,
            seed=seed,
            label=f"comparison:quality:{name}",
            direction=(
                "positive_favors_candidate"
                if direction == "higher_is_better"
                else "negative_favors_candidate"
            ),
        )
        for name, direction in QUALITY_METRICS.items()
    }
    resources = {
        name: bootstrap_paired_ratio_ci(
            _paired_values(reports, name),
            samples=samples,
            seed=seed,
            label=f"comparison:resource:{name}",
            direction="below_one_favors_candidate",
        )
        for name in RESOURCE_METRICS
    }
    return {
        "quality_deltas": quality,
        "resource_ratios": resources,
        "interpretation": (
            "Intervals are descriptive variation across seeded synthetic streams. "
            "They are not statistical-significance tests."
        ),
    }


def run_sweep(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    events: int = 64,
    budget_tokens: int = 512,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    chunk_chars: int = 320,
    chunk_overlap: int = 48,
    candidate_limit: int = 40,
    include_evicortex: bool = True,
) -> dict[str, Any]:
    """Run one independent synthetic stream per seed and aggregate paired results."""

    canonical_seeds: list[int] = []
    for value in seeds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("every seed must be an integer")
        canonical_seeds.append(value)
    if not canonical_seeds:
        raise ValueError("at least one stream seed is required")
    if len(set(canonical_seeds)) != len(canonical_seeds):
        raise ValueError("stream seeds must be unique")
    canonical_seeds.sort()
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    reports = [
        run_benchmark(
            events=events,
            seed=stream_seed,
            budget_tokens=budget_tokens,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            candidate_limit=candidate_limit,
            include_evicortex=include_evicortex,
        )
        for stream_seed in canonical_seeds
    ]
    systems = {
        "baseline": _system_aggregate(
            reports,
            system="baseline",
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    }
    if include_evicortex:
        systems["evicortex"] = _system_aggregate(
            reports,
            system="evicortex",
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )

    independent_streams = len(canonical_seeds)
    limitations = [
        (
            "Seeded streams vary generated values and distractors but share the same "
            "synthetic templates; they are not independent real-world populations."
        ),
        (
            "The comparator is a mechanical fixed-chunk lexical FTS5 baseline, not "
            "the broad class of RAG systems."
        ),
        (
            "Bootstrap intervals describe variation among these streams and must not "
            "be read as statistical significance."
        ),
    ]
    small_n = independent_streams < 10
    if small_n:
        limitations.insert(
            0,
            (
                f"Only {independent_streams} independent synthetic streams were run; "
                "the 95% intervals are unstable small-N smoke estimates."
            ),
        )

    aggregate: dict[str, Any] = {
        "independent_streams": independent_streams,
        "systems": systems,
    }
    if include_evicortex:
        aggregate["comparison"] = _comparison_aggregate(
            reports,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )

    return {
        "schema_version": 1,
        "benchmark": "evicortex-prototype-a-multi-seed-sweep",
        "claim_scope": (
            "Descriptive synthetic-stream smoke comparison with a mechanical lexical "
            "baseline; no statistical-significance or general RAG claim."
        ),
        "config": {
            "seeds": canonical_seeds,
            "events_per_stream": events,
            "budget_tokens": budget_tokens,
            "chunk_chars": chunk_chars,
            "chunk_overlap": chunk_overlap,
            "candidate_limit": candidate_limit,
            "include_evicortex": include_evicortex,
        },
        "resampling": {
            "method": "paired nonparametric percentile bootstrap",
            "unit": "synthetic stream seed",
            "confidence_level": 0.95,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "small_n": small_n,
        },
        "limitations": limitations,
        "per_seed": [_per_seed_summary(report) for report in reports],
        "aggregate": aggregate,
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("provide at least one comma-separated seed")
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--chunk-chars", type=int, default=320)
    parser.add_argument("--chunk-overlap", type=int, default=48)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="aggregate the lexical baseline without opening EviCortex",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _format_interval(record: dict[str, Any]) -> str:
    estimate = record["estimate"]
    low, high = record["ci95"]
    if estimate is None or low is None or high is None:
        return "unavailable"
    return f"{estimate:.4f} [95% bootstrap CI {low:.4f}, {high:.4f}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = run_sweep(
            seeds=args.seeds,
            events=args.events,
            budget_tokens=args.budget_tokens,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            candidate_limit=args.candidate_limit,
            include_evicortex=not args.baseline_only,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    print(report["claim_scope"])
    for limitation in report["limitations"]:
        print(f"warning: {limitation}")
    for item in report["per_seed"]:
        systems = ", ".join(
            f"{name} supported={summary['quality']['supported_answer_rate']:.3f}"
            for name, summary in item["systems"].items()
        )
        print(f"seed={item['seed']}: {systems}")
    comparison = report["aggregate"].get("comparison")
    if comparison is not None:
        print("quality deltas (candidate - baseline):")
        for name, record in comparison["quality_deltas"].items():
            print(f"  {name}: {_format_interval(record)}")
        print("resource ratios (candidate / baseline):")
        for name, record in comparison["resource_ratios"].items():
            print(f"  {name}: {_format_interval(record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
