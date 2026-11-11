"""Stage 5 of Phase D: aggregate retrieval, answer, cost, and audit results.

    python -m benchmarks.competitive.report --run runs/locomo --run runs/longmemeval \
        --out docs/results/phase-d

Writes ``summary.json`` (every number in the write-up), ``judgments.jsonl``
(per-answer verdicts without dataset text), and figures.

Uncertainty is a percentile bootstrap over independent units: whole
conversations for LoCoMo (questions inside one conversation share a haystack)
and questions for LongMemEval (each has its own haystack). Differences against
EviCortex are paired on the same resampled units.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import random
import statistics
from typing import Any

from .evaluate import _read_jsonl

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261110
REFERENCE_SYSTEM = "evicortex"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _cluster_bootstrap(
    clusters: dict[str, list[Any]],
    statistic: Callable[[list[Any]], float | None],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    names = sorted(clusters)
    if len(names) < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        drawn: list[Any] = []
        for _ in names:
            drawn.extend(clusters[rng.choice(names)])
        value = statistic(drawn)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return (round(low, 4), round(high, 4))


def _estimate(
    rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], float | None]
) -> dict[str, Any]:
    usable = [(row["namespace"], number) for row in rows if (number := value(row)) is not None]
    clusters: dict[str, list[float]] = defaultdict(list)
    for namespace, number in usable:
        clusters[namespace].append(number)
    numbers = [number for _, number in usable]
    mean = _mean(numbers)
    return {
        "n": len(numbers),
        "mean": None if mean is None else round(mean, 4),
        "ci95": _cluster_bootstrap(clusters, lambda drawn: _mean(drawn)),
    }


def _paired_difference(
    rows: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    value: Callable[[dict[str, Any]], float | None],
) -> dict[str, Any] | None:
    """Mean of (system - reference) over questions both systems answered."""

    reference_by_question = {row["question_id"]: value(row) for row in reference}
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        left = value(row)
        right = reference_by_question.get(row["question_id"])
        if left is None or right is None:
            continue
        clusters[row["namespace"]].append(left - right)
    differences = [item for values in clusters.values() for item in values]
    if not differences:
        return None
    return {
        "n": len(differences),
        "mean": round(sum(differences) / len(differences), 4),
        "ci95": _cluster_bootstrap(clusters, lambda drawn: _mean(drawn)),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 3)


def _correct(row: dict[str, Any]) -> float:
    return 1.0 if row["verdict"] == "correct" else 0.0


def summarize_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metas = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run.glob("retrieval_meta*.json"))]
    if not metas:
        raise FileNotFoundError(f"no retrieval_meta*.json under {run}")
    retrieval: list[dict[str, Any]] = []
    for path in sorted(run.glob("retrieval*.jsonl")):
        for row in _read_jsonl(path):
            row.pop("context", None)
            retrieval.append(row)
    judgments = _read_jsonl(run / "judgments.jsonl") if (run / "judgments.jsonl").exists() else []

    systems: dict[str, dict[str, Any]] = {}
    for meta in metas:
        for name, info in meta["systems"].items():
            entry = systems.setdefault(
                name,
                {"label": info["label"], "budget_exempt": info["budget_exempt"],
                 "config": info["config"], "ingestion": {}},
            )
            for key, number in info["ingestion"].items():
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    continue
                entry["ingestion"][key] = round(entry["ingestion"].get(key, 0) + number, 4)

    def by_cell(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, Any], list[dict[str, Any]]]:
        cells: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            cells[(row["system"], row["budget"])].append(row)
        return cells

    retrieval_cells = by_cell(retrieval)
    retrieval_summary: list[dict[str, Any]] = []
    for (system, budget), rows in sorted(retrieval_cells.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        categories = sorted({row["category"] for row in rows})
        retrieval_summary.append(
            {
                "system": system,
                "budget": budget,
                "questions": len(rows),
                "unit_recall": _estimate(rows, lambda row: row["unit_recall"]),
                "all_gold_units": _estimate(
                    rows, lambda row: None if row["all_gold_units"] is None else float(row["all_gold_units"])
                ),
                "session_recall": _estimate(rows, lambda row: row["session_recall"]),
                "mean_context_tokens": round(statistics.fmean(row["context_tokens"] for row in rows), 1),
                "latency_ms_p50": _percentile([row["latency_ms"] for row in rows], 0.5),
                "latency_ms_p95": _percentile([row["latency_ms"] for row in rows], 0.95),
                "unit_recall_by_category": {
                    category: _estimate(
                        [row for row in rows if row["category"] == category],
                        lambda row: row["unit_recall"],
                    )["mean"]
                    for category in categories
                },
            }
        )

    answer_cells = by_cell(judgments)
    answer_summary: list[dict[str, Any]] = []
    for (system, budget), rows in sorted(answer_cells.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        answerable = [row for row in rows if not row["is_abstention"]]
        unanswerable = [row for row in rows if row["is_abstention"]]
        entry: dict[str, Any] = {
            "system": system,
            "budget": budget,
            "accuracy": _estimate(rows, _correct),
            "answerable_accuracy": _estimate(answerable, _correct),
            "unanswerable_accuracy": _estimate(unanswerable, _correct),
            "abstained_on_answerable": _estimate(answerable, lambda row: float(row["abstained"])),
            "accuracy_by_category": {
                category: _estimate([row for row in rows if row["category"] == category], _correct)
                for category in sorted({row["category"] for row in rows})
            },
            "mean_context_tokens": round(statistics.fmean(row["context_tokens"] for row in rows), 1),
            "mean_reader_input_tokens": round(
                statistics.fmean(row["reader_input_tokens"] for row in rows), 1
            ),
            "reader_cost_usd": round(sum(row["reader_cost_usd"] or 0.0 for row in rows), 4),
        }
        if system != REFERENCE_SYSTEM:
            # Budget-exempt controls are compared with EviCortex at every budget.
            reference_budgets = (
                sorted({key[1] for key in answer_cells if key[0] == REFERENCE_SYSTEM})
                if budget is None
                else [budget]
            )
            entry["accuracy_minus_evicortex"] = {
                str(reference_budget): _paired_difference(
                    rows, answer_cells.get((REFERENCE_SYSTEM, reference_budget), []), _correct
                )
                for reference_budget in reference_budgets
            }
        answer_summary.append(entry)

    audit_path = run / "audit.json"
    sample_path = run / "sample.json"
    first = metas[0]
    dataset = dict(first["dataset"])
    for key in ("units", "conversations", "selected_questions", "questions"):
        if key in dataset and len(metas) > 1 and "shard" in first:
            dataset.pop(key)
    summary = {
        "dataset": dataset,
        "units_ingested": None,
        "budgets": first["budgets"],
        "encoder": {
            key: value
            for key, value in (first.get("encoder") or {}).items()
            if key not in ("document_texts_encoded", "query_texts_encoded", "encode_seconds")
        },
        "token_estimator": first["token_estimator"],
        "systems": systems,
        "retrieval": {
            "questions": len({row["question_id"] for row in retrieval}),
            "independent_units": len({row["namespace"] for row in retrieval}),
            "cells": retrieval_summary,
        },
        "answers": {
            "questions": len({row["question_id"] for row in judgments}),
            "independent_units": len({row["namespace"] for row in judgments}),
            "sample": (
                {key: value for key, value in json.loads(sample_path.read_text(encoding="utf-8")).items()
                 if key != "question_ids"}
                if sample_path.exists()
                else None
            ),
            "category_counts": {
                category: len({row["question_id"] for row in judgments if row["category"] == category})
                for category in sorted({row["category"] for row in judgments})
            },
            "reader": judgments[0]["reader"] if judgments else None,
            "judge": next(
                (row["graded_by"] for row in judgments if not row["graded_by"].startswith("rule:")),
                None,
            ),
            "cells": answer_summary,
        },
        "judge_audit": json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None,
        "code_revision": first["reproducibility"],
        "latency_note": first["latency_note"],
    }
    compact = [
        {
            "dataset": row["dataset"],
            "question_id": row["question_id"],
            "category": row["category"],
            "is_abstention": row["is_abstention"],
            "system": row["system"],
            "budget": row["budget"],
            "context_tokens": row["context_tokens"],
            "reader_input_tokens": row["reader_input_tokens"],
            "abstained": row["abstained"],
            "verdict": row["verdict"],
        }
        for row in judgments
    ]
    return summary, compact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args(argv)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Any] = {}
    compact: list[dict[str, Any]] = []
    for run in arguments.run:
        summary, rows = summarize_run(Path(run))
        runs[summary["dataset"]["name"]] = summary
        compact.extend(rows)
    payload = {
        "schema_version": 1,
        "benchmark": "evicortex-phase-d-competitive",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile, resampling conversations (LoCoMo) or questions (LongMemEval)",
        },
        "runs": runs,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    with (out / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for row in compact:
            handle.write(json.dumps(row) + "\n")
    if not arguments.no_figures:
        from .figures import write_figures

        write_figures(payload, out)
    print(f"wrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
