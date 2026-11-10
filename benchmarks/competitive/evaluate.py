"""Stages 2-4 of Phase D: shared reader, blinded judge, and judge audit.

    python -m benchmarks.competitive.evaluate read  --run runs/locomo --reader claude-cli:claude-haiku-5-5
    python -m benchmarks.competitive.evaluate judge --run runs/locomo --judge  claude-cli:claude-sonnet-5-5
    python -m benchmarks.competitive.evaluate audit --run runs/locomo --auditor claude-cli:claude-opus-5-5

Any spec accepted by ``llm.make_client`` works, so the same protocol runs with
OpenAI, a local Ollama or vLLM server, or any other Chat Completions endpoint.

Every system's context goes through the same reader prompt. The judge sees only
the question, the gold answer, and the candidate answer - never the system name,
the budget, or the context - and items from all systems are shuffled together.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import threading
from typing import Any

from .llm import CachedLLM, make_client

READER_SYSTEM = (
    "You answer a question about someone's past conversations using only the memory "
    "context you are given. The context may be raw conversation turns, JSON evidence "
    "records, or extracted memory statements, and most items carry a date. Resolve "
    "relative time expressions such as \"yesterday\" or \"last week\" against the date "
    "of the item that contains them. Reply with the shortest answer that fully answers "
    "the question - a phrase or a single sentence - and no explanation. If the context "
    "does not contain enough information to answer, reply exactly: I don't know."
)

JUDGE_SYSTEM = (
    "You grade answers to questions about past conversations. For each item you get a "
    "question, the gold answer, and a candidate answer. Grade strictly on content:\n"
    "- CORRECT: the candidate states the same fact as the gold answer. Paraphrases, "
    "extra harmless detail, and different formats for the same date, time, number, or "
    "name are fine. If the gold answer is approximate or relative (for example \"the "
    "week before 9 June 2023\"), a candidate naming the same period or a date inside "
    "it is correct.\n"
    "- INCORRECT: the candidate contradicts the gold answer, misses a fact the gold "
    "answer requires (including items of a gold list), gives a different date or "
    "value, or says it does not know.\n"
    "- UNANSWERABLE items: when the gold answer says the conversation does not contain "
    "the information, the candidate is CORRECT only if it declines to answer or says "
    "the information is not available, and INCORRECT if it offers a specific answer.\n"
    "Reply with only a JSON array, one object per item in the same order: "
    "[{\"id\": <item id>, \"verdict\": \"correct\" | \"incorrect\"}, ...]"
)

LOCOMO_QUOTA = {
    "multi-hop": 6,
    "temporal": 6,
    "open-domain": 3,
    "single-hop": 10,
    "adversarial": 5,
}
_DONT_KNOW = re.compile(r"^\W*i\s+(don'?t|do\s+not)\s+know\W*$", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_retrieval_rows(run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run.glob("retrieval*.jsonl")):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"no retrieval*.jsonl under {run}")
    return rows


def select_questions(rows: Sequence[dict[str, Any]], *, seed: int) -> list[str]:
    """Seeded stratified answer-level sample.

    LoCoMo: a fixed quota per category within each conversation, so every
    conversation contributes equally. LongMemEval: every question in the run.
    """

    questions: dict[str, dict[str, Any]] = {}
    for row in rows:
        questions.setdefault(row["question_id"], row)
    if all(row["dataset"] == "longmemeval" for row in questions.values()):
        return sorted(questions)
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for question_id, row in questions.items():
        grouped[(row["namespace"], row["category"])].append(question_id)
    selected: list[str] = []
    for (_, category), ids in sorted(grouped.items()):
        pool = sorted(ids)
        rng.shuffle(pool)
        selected.extend(pool[: LOCOMO_QUOTA.get(category, 0)])
    return sorted(selected)


def reader_prompt(row: dict[str, Any]) -> str:
    context = row["context"] if row["context"] else "(no memory context is available)"
    date = (
        f"Date the question is asked: {row['question_date_text']}\n"
        if row.get("question_date_text")
        else ""
    )
    return f"Memory context:\n{context}\n\n{date}Question: {row['question']}\nAnswer:"


def _answer_key(row: dict[str, Any]) -> str:
    return f"{row['question_id']}|{row['system']}|{row['budget']}"


def stage_read(arguments: argparse.Namespace) -> int:
    run = Path(arguments.run)
    rows = load_retrieval_rows(run)
    budgets = {int(value) for value in arguments.budgets.split(",")}
    systems = (
        {name.strip() for name in arguments.systems.split(",")} if arguments.systems else None
    )
    sample_path = run / "sample.json"
    if sample_path.exists():
        selected = set(json.loads(sample_path.read_text(encoding="utf-8"))["question_ids"])
    else:
        ids = select_questions(rows, seed=arguments.seed)
        sample_path.write_text(
            json.dumps({"seed": arguments.seed, "quota": LOCOMO_QUOTA, "question_ids": ids}, indent=1)
            + "\n",
            encoding="utf-8",
        )
        selected = set(ids)
    work = [
        row
        for row in rows
        if row["question_id"] in selected
        and (row["budget"] is None or row["budget"] in budgets)
        and (systems is None or row["system"] in systems)
    ]
    reader = CachedLLM(make_client(arguments.reader), run / "reader_cache.jsonl")
    lock = threading.Lock()
    done = 0

    def answer(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        response = reader.complete(
            system=READER_SYSTEM, prompt=reader_prompt(row), max_tokens=200
        )
        with lock:
            done += 1
            if done % 100 == 0:
                print(f"read {done}/{len(work)}", flush=True)
        return {
            "key": _answer_key(row),
            "dataset": row["dataset"],
            "question_id": row["question_id"],
            "namespace": row["namespace"],
            "category": row["category"],
            "is_abstention": row["is_abstention"],
            "system": row["system"],
            "budget": row["budget"],
            "context_tokens": row["context_tokens"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "answer": response.text.strip(),
            "reader": reader.identity,
            "reader_input_tokens": response.input_tokens,
            "reader_output_tokens": response.output_tokens,
            "reader_cost_usd": response.cost_usd,
        }

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        answers = list(pool.map(answer, work))
    with (run / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for item in sorted(answers, key=lambda item: item["key"]):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(answers)} answers; reader usage {reader.usage()}")
    return 0


def _parse_verdicts(text: str, expected: Sequence[int]) -> dict[int, str] | None:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    verdicts: dict[int, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            return None
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in ("correct", "incorrect"):
            return None
        try:
            verdicts[int(item.get("id"))] = verdict
        except (TypeError, ValueError):
            return None
    return verdicts if set(verdicts) == set(expected) else None


def _judge_batch(
    judge: CachedLLM, batch: Sequence[tuple[int, dict[str, Any]]]
) -> dict[int, str]:
    body = "\n\n".join(
        f"Item {number}\nQuestion: {item['question']}\nGold answer: {item['gold_answer']}\n"
        f"Candidate answer: {item['answer'] or '(empty)'}"
        for number, item in batch
    )
    expected = [number for number, _ in batch]
    for attempt in range(3):
        suffix = "" if attempt == 0 else f"\n\n(Formatting retry {attempt}: JSON array only.)"
        response = judge.complete(system=JUDGE_SYSTEM, prompt=body + suffix, max_tokens=1500)
        verdicts = _parse_verdicts(response.text, expected)
        if verdicts is not None:
            return verdicts
    if len(batch) == 1:
        raise RuntimeError(f"judge returned no parseable verdict for item {expected[0]}")
    merged: dict[int, str] = {}
    for entry in batch:
        merged.update(_judge_batch(judge, [entry]))
    return merged


def _run_judge(
    judge: CachedLLM,
    items: Sequence[dict[str, Any]],
    *,
    seed: int,
    batch_size: int,
    workers: int,
) -> dict[str, str]:
    """Blind, shuffled, batched grading. Returns verdict by answer key."""

    order = list(range(len(items)))
    # Shuffle on content digests so batch composition does not depend on system order.
    order.sort(key=lambda index: hashlib.sha256(f"{seed}|{items[index]['key']}".encode()).hexdigest())
    batches = [
        [(position, items[index]) for position, index in enumerate(order[start : start + batch_size], start)]
        for start in range(0, len(order), batch_size)
    ]
    verdict_by_position: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, result in enumerate(pool.map(lambda batch: _judge_batch(judge, batch), batches), 1):
            verdict_by_position.update(result)
            if count % 20 == 0:
                print(f"judged {count}/{len(batches)} batches", flush=True)
    return {
        items[index]["key"]: verdict_by_position[position] for position, index in enumerate(order)
    }


def stage_judge(arguments: argparse.Namespace) -> int:
    run = Path(arguments.run)
    answers = _read_jsonl(run / "answers.jsonl")
    judge = CachedLLM(make_client(arguments.judge), run / "judge_cache.jsonl")
    rule_based: dict[str, str] = {}
    pending: list[dict[str, Any]] = []
    for item in answers:
        # An exact "I don't know." is graded by rule: wrong when an answer exists,
        # right when the question is unanswerable. Everything else goes to the judge.
        if _DONT_KNOW.match(item["answer"]) or not item["answer"]:
            rule_based[item["key"]] = "correct" if item["is_abstention"] else "incorrect"
        else:
            pending.append(item)
    verdicts = _run_judge(
        judge, pending, seed=arguments.seed, batch_size=arguments.batch_size, workers=arguments.workers
    )
    with (run / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for item in answers:
            key = item["key"]
            abstained = key in rule_based
            row = {
                **item,
                "abstained": abstained,
                "verdict": rule_based[key] if abstained else verdicts[key],
                "graded_by": "rule:exact-dont-know" if abstained else judge.identity,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"graded {len(answers)} answers ({len(rule_based)} by rule, {len(pending)} by judge); "
        f"judge usage {judge.usage()}"
    )
    return 0


def cohen_kappa(pairs: Iterable[tuple[str, str]]) -> float | None:
    pairs = list(pairs)
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_rate = sum(left == "correct" for left, _ in pairs) / total
    right_rate = sum(right == "correct" for _, right in pairs) / total
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def stage_audit(arguments: argparse.Namespace) -> int:
    """Re-grade a seeded sample with an independent, stronger model."""

    run = Path(arguments.run)
    judged = [row for row in _read_jsonl(run / "judgments.jsonl") if not row["abstained"]]
    rng = random.Random(arguments.seed)
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged:
        by_system[row["system"]].append(row)
    per_system = max(1, arguments.sample // max(1, len(by_system)))
    sample: list[dict[str, Any]] = []
    for system in sorted(by_system):
        pool = sorted(by_system[system], key=lambda row: row["key"])
        rng.shuffle(pool)
        sample.extend(pool[:per_system])
    auditor = CachedLLM(make_client(arguments.auditor), run / "audit_cache.jsonl")
    verdicts = _run_judge(
        auditor, sample, seed=arguments.seed + 1, batch_size=arguments.batch_size, workers=arguments.workers
    )
    pairs = [(row["verdict"], verdicts[row["key"]]) for row in sample]
    disagreements = [
        {
            "key": row["key"],
            "system": row["system"],
            "category": row["category"],
            "judge": row["verdict"],
            "auditor": verdicts[row["key"]],
        }
        for row in sample
        if row["verdict"] != verdicts[row["key"]]
    ]
    by_system_agreement = {
        system: {
            "n": len(rows),
            "agreement": sum(row["verdict"] == verdicts[row["key"]] for row in rows) / len(rows),
            "judge_correct_rate": sum(row["verdict"] == "correct" for row in rows) / len(rows),
            "auditor_correct_rate": sum(verdicts[row["key"]] == "correct" for row in rows) / len(rows),
        }
        for system in sorted(by_system)
        if (rows := [row for row in sample if row["system"] == system])
    }
    report = {
        "judge": judged[0]["graded_by"] if judged else None,
        "auditor": auditor.identity,
        "sampled_items": len(sample),
        "sampling": f"seeded, up to {per_system} judge-graded answers per system",
        "agreement": sum(left == right for left, right in pairs) / len(pairs),
        "cohen_kappa": cohen_kappa(pairs),
        "judge_correct_rate": sum(left == "correct" for left, _ in pairs) / len(pairs),
        "auditor_correct_rate": sum(right == "correct" for _, right in pairs) / len(pairs),
        "by_system": by_system_agreement,
        "disagreements": disagreements,
        "auditor_usage": auditor.usage(),
        "note": (
            "Agreement between two LLM graders. This is a consistency check, not a "
            "substitute for the human-reviewed calibration the benchmark contract requires."
        ),
    }
    (run / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("sampled_items", "agreement", "cohen_kappa")}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    stages = parser.add_subparsers(dest="stage", required=True)

    read = stages.add_parser("read")
    read.add_argument("--run", required=True)
    read.add_argument("--reader", default="claude-cli:claude-haiku-5-5")
    read.add_argument("--budgets", default="1000,4000")
    read.add_argument("--systems", default=None)
    read.add_argument("--seed", type=int, default=20261110)
    read.add_argument("--workers", type=int, default=10)
    read.set_defaults(handler=stage_read)

    judge = stages.add_parser("judge")
    judge.add_argument("--run", required=True)
    judge.add_argument("--judge", default="claude-cli:claude-sonnet-5-5")
    judge.add_argument("--seed", type=int, default=20261110)
    judge.add_argument("--batch-size", type=int, default=10)
    judge.add_argument("--workers", type=int, default=8)
    judge.set_defaults(handler=stage_judge)

    audit = stages.add_parser("audit")
    audit.add_argument("--run", required=True)
    audit.add_argument("--auditor", default="claude-cli:claude-opus-5-5")
    audit.add_argument("--sample", type=int, default=160)
    audit.add_argument("--seed", type=int, default=20261110)
    audit.add_argument("--batch-size", type=int, default=10)
    audit.add_argument("--workers", type=int, default=6)
    audit.set_defaults(handler=stage_audit)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
