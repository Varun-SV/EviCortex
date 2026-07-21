"""Schema and metric tests for the external LongMemEval adapter."""

from __future__ import annotations

from copy import deepcopy
from datetime import timezone
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.longmemeval import (
    LONGMEMEVAL_ARTIFACT_SHA256,
    load_longmemeval,
    official_longmemeval_variant,
    parse_longmemeval_timestamp,
    session_retrieval_metrics,
)


def _row(*, question_id: str = "q1") -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "knowledge-update",
        "question": "Which theme does Ada now prefer?",
        "answer": "dark",
        "question_date": "2026/03/01 (Sun) 12:00",
        "haystack_session_ids": ["old", "new", "noise"],
        "haystack_dates": [
            "2026/01/01 (Thu) 10:00",
            "2026/02/01 (Sun) 10:00",
            "2026/02/10 (Tue) 10:00",
        ],
        "haystack_sessions": [
            [{"role": "user", "content": "I prefer light mode."}],
            [
                {
                    "role": "user",
                    "content": "I switched to dark mode.",
                    "has_answer": True,
                }
            ],
            [{"role": "assistant", "content": "Bananas are yellow."}],
        ],
        "answer_session_ids": ["new"],
    }


class LongMemEvalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "longmemeval_s_cleaned.json"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write(self, rows: object) -> None:
        self.path.write_text(
            json.dumps(rows, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_loads_exact_schema_renders_sessions_and_hashes_source(self) -> None:
        rows = [_row(), _row(question_id="q2_abs")]
        rows[1]["answer_session_ids"] = []  # type: ignore[index]
        for session in rows[1]["haystack_sessions"]:  # type: ignore[index]
            for turn in session:
                turn.pop("has_answer", None)
        self._write(rows)

        dataset = load_longmemeval(self.path)
        limited = load_longmemeval(self.path, limit=1)

        self.assertEqual(len(dataset.cases), 2)
        self.assertEqual(len(limited.cases), 1)
        self.assertEqual(dataset.source_bytes, self.path.stat().st_size)
        self.assertEqual(len(dataset.sha256), 64)
        first = dataset.cases[0]
        self.assertEqual(first.namespace, "longmemeval/q1")
        self.assertFalse(first.is_abstention)
        self.assertTrue(first.sessions[1].has_answer)
        self.assertEqual(
            first.sessions[1].render(),
            "Session date: 2026/02/01 (Sun) 10:00\nuser: I switched to dark mode.",
        )
        self.assertTrue(dataset.cases[1].is_abstention)
        self.assertEqual(len(dataset.answerable_cases), 1)

    def test_timestamp_parser_is_locale_independent_and_checks_weekday(self) -> None:
        parsed = parse_longmemeval_timestamp("2023/04/10 (Mon) 23:07")
        self.assertEqual(parsed.tzinfo, timezone.utc)
        self.assertEqual(parsed.isoformat(), "2023-04-10T23:07:00+00:00")
        for invalid in (
            "2023-04-10T23:07:00Z",
            "2023/04/10 (Tue) 23:07",
            "2023/02/30 (Thu) 23:07",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_longmemeval_timestamp(invalid)

    def test_official_artifacts_require_an_exact_logical_digest(self) -> None:
        official_digest = LONGMEMEVAL_ARTIFACT_SHA256[
            "longmemeval_s_cleaned.json"
        ]
        self.assertEqual(
            official_longmemeval_variant(official_digest),
            "longmemeval_s_cleaned.json",
        )
        self.assertIsNone(official_longmemeval_variant("0" * 64))

    def test_session_metrics_distinguish_any_all_fraction_and_ndcg(self) -> None:
        row = _row()
        row["answer_session_ids"] = ["old", "new"]
        row["haystack_sessions"][0][0]["has_answer"] = True  # type: ignore[index]
        self._write([row])
        case = load_longmemeval(self.path).cases[0]

        metrics = session_retrieval_metrics(
            case,
            ("noise", "new", "new", "old"),
            cutoffs=(1, 2, 3),
        )
        self.assertEqual(metrics["recall_any@1"], 0.0)
        self.assertEqual(metrics["recall_fraction@2"], 0.5)
        self.assertEqual(metrics["recall_all@2"], 0.0)
        self.assertEqual(metrics["recall_all@3"], 1.0)
        self.assertGreater(metrics["ndcg_any@3"], metrics["ndcg_any@2"])

        # The released LongMemEval evaluator intentionally gives ranks one and
        # two equal DCG weight; retain that behavior for leaderboard parity.
        second_rank_hit = session_retrieval_metrics(
            case,
            ("noise", "old"),
            cutoffs=(2,),
        )
        self.assertEqual(second_rank_hit["ndcg_any@2"], 0.5)

        one_gold = deepcopy(row)
        one_gold["answer_session_ids"] = ["old"]
        one_gold["haystack_sessions"][1][0].pop(  # type: ignore[index]
            "has_answer"
        )
        self._write([one_gold])
        one_gold_case = load_longmemeval(self.path).cases[0]
        official_edge = session_retrieval_metrics(
            one_gold_case,
            ("noise", "old"),
            cutoffs=(2,),
        )
        self.assertEqual(official_edge["ndcg_any@2"], 1.0)

        abstention = deepcopy(row)
        abstention["question_id"] = "q_abs"
        abstention["answer_session_ids"] = []
        for session in abstention["haystack_sessions"]:  # type: ignore[index]
            for turn in session:
                turn.pop("has_answer", None)
        self._write([abstention])
        skipped = session_retrieval_metrics(
            load_longmemeval(self.path).cases[0],
            (),
        )
        self.assertEqual(skipped, {"skipped_abstention": True})

    def test_rejects_misalignment_unknown_gold_and_malformed_turns(self) -> None:
        invalid_rows: list[dict[str, object]] = []
        misaligned = _row()
        misaligned["haystack_dates"] = ["one"]
        invalid_rows.append(misaligned)
        unknown = _row()
        unknown["answer_session_ids"] = ["missing"]
        invalid_rows.append(unknown)
        malformed = _row()
        malformed["haystack_sessions"] = [
            [{"role": "system", "content": "bad role"}],
            [],
            [],
        ]
        invalid_rows.append(malformed)
        missing_gold = _row()
        missing_gold["answer_session_ids"] = []
        invalid_rows.append(missing_gold)
        abstention_with_gold = _row(question_id="bad_abs")
        invalid_rows.append(abstention_with_gold)
        label_mismatch = _row()
        label_mismatch["haystack_sessions"][1][0].pop(  # type: ignore[index]
            "has_answer"
        )
        invalid_rows.append(label_mismatch)

        for row in invalid_rows:
            with self.subTest(row=row):
                self._write([row])
                with self.assertRaises(ValueError):
                    load_longmemeval(self.path)


if __name__ == "__main__":
    unittest.main()
