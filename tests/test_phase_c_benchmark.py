"""Tests for the offline LongMemEval Phase C retrieval runner."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

from benchmarks.run_phase_c import run_longmemeval_benchmark
from evicortex import HashingEmbedder


class _SemanticFixtureEmbedder:
    identity = "tests.semantic-fixture.v1"
    dimensions = 3
    normalized = True
    is_semantic = True
    diagnostic_only = False

    def __init__(self) -> None:
        self.seen_documents: list[str] = []
        self.seen_queries: list[str] = []

    @staticmethod
    def _vector(text: str) -> tuple[float, float, float]:
        normalized = text.casefold()
        if "launch code" in normalized or "orbit-7" in normalized:
            return (1.0, 0.0, 0.0)
        if "banana" in normalized:
            return (0.0, 1.0, 0.0)
        if "weather" in normalized:
            return (0.0, 0.0, 1.0)
        value = 1.0 / math.sqrt(3.0)
        return (value, value, value)

    def embed_documents(
        self, texts: list[str] | tuple[str, ...]
    ) -> tuple[tuple[float, float, float], ...]:
        frozen = tuple(texts)
        self.seen_documents.extend(frozen)
        return tuple(self._vector(text) for text in frozen)

    def embed_query(self, text: str) -> tuple[float, float, float]:
        self.seen_queries.append(text)
        return self._vector(text)

    def embed_batch(
        self, texts: list[str] | tuple[str, ...]
    ) -> tuple[tuple[float, float, float], ...]:
        return self.embed_documents(texts)


def _case(*, question_id: str, abstention: bool = False) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": (
            "What is tomorrow's weather?"
            if abstention
            else "What is my launch code?"
        ),
        # This sentinel must never reach an embedder; retrieval uses only locations.
        "answer": "ANSWER_FIELD_SENTINEL",
        "question_date": "2026/03/01 (Sun) 12:00",
        "haystack_session_ids": ["gold", "noise"],
        "haystack_dates": [
            "2026/01/01 (Thu) 10:00",
            "2026/02/01 (Sun) 10:00",
        ],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "My launch code is ORBIT-7.",
                    "has_answer": not abstention,
                }
            ],
            [{"role": "assistant", "content": "A banana is yellow."}],
        ],
        "answer_session_ids": [] if abstention else ["gold"],
    }


class PhaseCLongMemEvalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.rows = [
            _case(question_id="q-answerable"),
            _case(question_id="q-abstention_abs", abstention=True),
        ]
        self.json_bytes = json.dumps(
            self.rows,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        self.path = self.root / "longmemeval_subset.json"
        self.path.write_bytes(self.json_bytes)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _run(self, path: Path, embedder: _SemanticFixtureEmbedder) -> dict[str, object]:
        return run_longmemeval_benchmark(
            path,
            embedder=embedder,
            budget_tokens=256,
            cutoffs=(1, 2),
            candidate_limit=4,
            semantic_scan_cap=8,
            hybrid_channel_limit=4,
            hybrid_candidate_limit=4,
            hybrid_max_exact_candidates=32,
        )

    def test_scores_both_systems_after_budget_and_separates_abstentions(self) -> None:
        embedder = _SemanticFixtureEmbedder()
        report = self._run(self.path, embedder)

        self.assertEqual(report["schema_version"], 1)
        self.assertFalse(report["evaluation_policy"]["superiority_claim"])
        self.assertFalse(
            report["evaluation_policy"]["answer_and_has_answer_ingested"]
        )
        self.assertEqual(report["dataset"]["selected_cases"], 2)
        self.assertEqual(report["dataset"]["answerable_cases"], 1)
        self.assertEqual(report["dataset"]["abstention_cases"], 1)
        self.assertEqual(report["dataset"]["evidence_units"], 4)
        self.assertFalse(report["dataset"]["official_artifact_verified"])
        self.assertIsNone(report["dataset"]["declared_source"])
        self.assertEqual(
            report["dataset"]["input_kind"],
            "caller-supplied LongMemEval-schema data",
        )

        for system_name in ("evicortex", "hybrid_baseline"):
            with self.subTest(system=system_name):
                system = report["systems"][system_name]
                metrics = system["metrics"]
                self.assertEqual(metrics["answerable_queries_scored"], 1)
                self.assertEqual(
                    metrics["abstention_queries_skipped_by_official_metrics"],
                    1,
                )
                self.assertEqual(metrics["session_retrieval"]["recall_all@1"], 1.0)
                self.assertEqual(metrics["session_retrieval"]["ndcg_any@1"], 1.0)
                self.assertEqual(
                    metrics["retrieval_only_abstention_no_evidence_rate"],
                    1.0,
                )
                self.assertEqual(metrics["budget_compliance_rate"], 1.0)
                self.assertLessEqual(metrics["context_tokens_max"], 256)
                abstention = next(
                    row for row in system["queries"] if row["is_abstention"]
                )
                self.assertEqual(
                    abstention["metrics"], {"skipped_abstention": True}
                )

        embedded = "\n".join(embedder.seen_documents)
        self.assertNotIn("ANSWER_FIELD_SENTINEL", embedded)
        self.assertNotIn("has_answer", embedded)
        self.assertIn("Session date: 2026/01/01 (Thu) 10:00", embedded)
        self.assertIn("What is my launch code?", embedder.seen_queries)

    def test_json_gzip_preserves_logical_digest_and_reports_compression(self) -> None:
        gzip_path = self.root / "longmemeval_subset.json.gz"
        with gzip_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(self.json_bytes)

        plain = self._run(self.path, _SemanticFixtureEmbedder())
        compressed = self._run(gzip_path, _SemanticFixtureEmbedder())

        self.assertEqual(plain["dataset"]["compression"], "none")
        self.assertEqual(compressed["dataset"]["compression"], "gzip")
        self.assertEqual(
            plain["dataset"]["logical_json_sha256"],
            compressed["dataset"]["logical_json_sha256"],
        )
        self.assertEqual(
            plain["dataset"]["logical_json_bytes"],
            compressed["dataset"]["logical_json_bytes"],
        )
        self.assertNotEqual(
            plain["dataset"]["source_file_sha256"],
            compressed["dataset"]["source_file_sha256"],
        )

    def test_rejects_nonsemantic_embedders_and_truncated_scan_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "is_semantic"):
            run_longmemeval_benchmark(self.path, embedder=HashingEmbedder())

        with self.assertRaisesRegex(ValueError, "silently incomplete"):
            run_longmemeval_benchmark(
                self.path,
                embedder=_SemanticFixtureEmbedder(),
                cutoffs=(1,),
                candidate_limit=1,
                hybrid_channel_limit=1,
                hybrid_candidate_limit=1,
                semantic_scan_cap=1,
            )


if __name__ == "__main__":
    unittest.main()
