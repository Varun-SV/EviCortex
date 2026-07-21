"""Fast determinism, fairness, and smoke tests for Prototype A benchmarks."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.baseline import FixedChunkFTS5Baseline, estimate_tokens
from benchmarks.run import (
    BASELINE_LABEL,
    SOURCE_DIGEST_SEMANTICS,
    SUITE_DIGEST_SEMANTICS,
    _percentile,
    _suite_digest,
    run_benchmark,
)
from benchmarks.sweep import (
    bootstrap_paired_delta_ci,
    bootstrap_paired_ratio_ci,
    run_sweep,
)
from benchmarks.synthetic import generate_suite


class SyntheticSuiteTests(unittest.TestCase):
    def test_seeded_suite_is_deterministic_and_covers_contract_tasks(self) -> None:
        first = generate_suite(event_count=32, seed=19)
        second = generate_suite(event_count=32, seed=19)
        different = generate_suite(event_count=32, seed=20)

        self.assertEqual(first, second)
        self.assertNotEqual(first.events, different.events)
        self.assertEqual(len(first.events), 32)
        self.assertEqual(len(first.queries), 12)
        self.assertTrue(
            {
                "direct",
                "current",
                "historical",
                "temporal",
                "multi_hop",
                "contradiction",
                "abstention",
                "distractor_growth",
                "namespace_isolation",
            }
            <= {case.task for case in first.queries}
        )
        source_ids = {event.source_id for event in first.events}
        for case in first.queries:
            self.assertTrue(set(case.gold_evidence_ids) <= source_ids)
            self.assertTrue(set(case.forbidden_evidence_ids) <= source_ids)

    def test_abstention_suite_covers_double_and_partial_ood_queries(self) -> None:
        suite = generate_suite(event_count=12, seed=19)
        cases = {
            case.query_id: case
            for case in suite.queries
            if case.task == "abstention"
        }
        self.assertEqual(
            set(cases),
            {
                "abstention-unknown",
                "abstention-known-entity-unseen-predicate",
                "abstention-unknown-entity-seen-predicate",
            },
        )
        for case in cases.values():
            self.assertEqual(case.gold_evidence_ids, ())
            self.assertFalse(case.expected_answerable)

        known_subject = suite.events[0].claims[0].subject.split(":", 1)[1]
        known_unseen = cases["abstention-known-entity-unseen-predicate"]
        self.assertIn(known_subject, known_unseen.query.casefold())
        self.assertIn("xylophonic_allergy", known_unseen.query)
        self.assertTrue(
            all(
                claim.predicate != "xylophonic_allergy"
                for event in suite.events
                for claim in event.claims
            )
        )

        unknown_seen = cases["abstention-unknown-entity-seen-predicate"]
        self.assertIn("Zephyr-999", unknown_seen.query)
        self.assertIn("home_city", unknown_seen.query)
        self.assertTrue(
            any(
                claim.predicate == "home_city"
                for event in suite.events
                for claim in event.claims
            )
        )
        self.assertTrue(
            all("Zephyr-999" not in event.content for event in suite.events)
        )

    def test_structured_claims_are_present_verbatim_in_baseline_text(self) -> None:
        suite = generate_suite(event_count=16, seed=3)
        for event in suite.events:
            for claim in event.claims:
                self.assertIn(f"subject={claim.subject}", event.content)
                self.assertIn(f"predicate={claim.predicate}", event.content)
                self.assertIn(f"value={claim.value}", event.content)

    def test_suite_digest_covers_every_suite_event_claim_and_query_field(self) -> None:
        suite = generate_suite(event_count=12, seed=13)
        original = _suite_digest(suite)
        self.assertEqual(original, _suite_digest(suite))

        event = suite.events[0]
        claim = event.claims[0]
        query = suite.queries[0]

        suite_mutations = {
            "seed": suite.seed + 1,
            "requested_event_count": suite.requested_event_count + 1,
        }
        event_mutations = {
            "sequence": event.sequence + 1,
            "source_id": f"{event.source_id}/changed",
            "namespace": f"{event.namespace}/changed",
            "occurred_at": f"{event.occurred_at}/changed",
            "content": f"{event.content} changed",
        }
        claim_mutations = {
            "subject": f"{claim.subject}:changed",
            "predicate": f"{claim.predicate}:changed",
            "value": f"{claim.value}:changed",
            "valid_from": "2027-01-01T00:00:00Z",
            "valid_to": "2027-02-01T00:00:00Z",
            "confidence": claim.confidence - 0.01,
            "authority": f"{claim.authority}:changed",
            "supersedes_source_id": "synthetic/changed/predecessor",
        }
        query_mutations = {
            "query_id": f"{query.query_id}-changed",
            "task": f"{query.task}-changed",
            "after_sequence": query.after_sequence + 1,
            "namespace": f"{query.namespace}/changed",
            "query": f"{query.query} changed",
            "gold_evidence_ids": (*query.gold_evidence_ids, "synthetic/changed/gold"),
            "forbidden_evidence_ids": (
                *query.forbidden_evidence_ids,
                "synthetic/changed/forbidden",
            ),
            "as_of": "2027-03-01T00:00:00Z",
            "expected_answerable": not query.expected_answerable,
            "expected_values": (*query.expected_values, "changed"),
        }

        for field, value in suite_mutations.items():
            with self.subTest(scope="suite", field=field):
                self.assertNotEqual(original, _suite_digest(replace(suite, **{field: value})))
        for field, value in event_mutations.items():
            with self.subTest(scope="event", field=field):
                changed = replace(event, **{field: value})
                mutated = replace(suite, events=(changed, *suite.events[1:]))
                self.assertNotEqual(original, _suite_digest(mutated))
        for field, value in claim_mutations.items():
            with self.subTest(scope="claim", field=field):
                changed_claim = replace(claim, **{field: value})
                changed_event = replace(
                    event, claims=(changed_claim, *event.claims[1:])
                )
                mutated = replace(suite, events=(changed_event, *suite.events[1:]))
                self.assertNotEqual(original, _suite_digest(mutated))
        for field, value in query_mutations.items():
            with self.subTest(scope="query", field=field):
                changed = replace(query, **{field: value})
                mutated = replace(suite, queries=(changed, *suite.queries[1:]))
                self.assertNotEqual(original, _suite_digest(mutated))


class LexicalBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.suite = generate_suite(event_count=24, seed=11)
        self.baseline = FixedChunkFTS5Baseline(
            Path(self.temporary_directory.name) / "baseline.db",
            chunk_chars=180,
            overlap_chars=24,
            candidate_limit=40,
        )
        for event in self.suite.events:
            self.baseline.remember(
                namespace=event.namespace,
                content=event.content,
                source_id=event.source_id,
                occurred_at=event.occurred_at,
            )

    def tearDown(self) -> None:
        self.baseline.close()
        self.temporary_directory.cleanup()

    def test_recall_is_deterministic_and_serialized_context_obeys_budget(self) -> None:
        case = next(
            item for item in self.suite.queries if item.query_id == "direct-home-city"
        )
        first = self.baseline.recall(
            namespace=case.namespace,
            query=case.query,
            budget_tokens=256,
        )
        second = self.baseline.recall(
            namespace=case.namespace,
            query=case.query,
            budget_tokens=256,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.estimated_tokens, estimate_tokens(first.to_prompt()))
        self.assertLessEqual(first.estimated_tokens, first.budget_tokens)
        self.assertIn(case.gold_evidence_ids[0], first.evidence_ids)

    def test_namespace_and_as_of_filters_apply_before_ranking(self) -> None:
        namespace_case = next(
            item
            for item in self.suite.queries
            if item.query_id == "namespace-recovery-code"
        )
        result = self.baseline.recall(
            namespace=namespace_case.namespace,
            query=namespace_case.query,
            budget_tokens=512,
        )
        self.assertFalse(set(namespace_case.forbidden_evidence_ids) & set(result.evidence_ids))

        history_case = next(
            item
            for item in self.suite.queries
            if item.query_id == "historical-before-update"
        )
        historical = self.baseline.recall(
            namespace=history_case.namespace,
            query=history_case.query,
            budget_tokens=512,
            as_of=history_case.as_of,
        )
        self.assertIn(history_case.gold_evidence_ids[0], historical.evidence_ids)
        self.assertFalse(set(history_case.forbidden_evidence_ids) & set(historical.evidence_ids))


class BenchmarkRunnerTests(unittest.TestCase):
    @staticmethod
    def _stable_outcomes(
        report: dict[str, object], system: str = "baseline"
    ) -> tuple[object, ...]:
        result = report["results"][system]  # type: ignore[index]
        queries = result["queries"]  # type: ignore[index]
        return tuple(
            (
                item["query_id"],
                tuple(item["retrieved_source_ids"]),
                item["evidence_recall"],
                item["has_sufficient_gold_evidence"],
                item["estimated_context_tokens"],
                item["budget_compliant"],
                item["error"],
            )
            for item in queries
        )

    def test_baseline_only_runner_smoke_and_deterministic_outcomes(self) -> None:
        self.assertAlmostEqual(_percentile(tuple(range(12)), 0.99), 10.89)
        first = run_benchmark(
            events=24,
            seed=23,
            budget_tokens=384,
            include_evicortex=False,
        )
        second = run_benchmark(
            events=24,
            seed=23,
            budget_tokens=384,
            include_evicortex=False,
        )

        self.assertEqual(first["dataset"], second["dataset"])
        self.assertEqual(
            first["results"]["baseline"]["metrics"]["database_bytes"],
            second["results"]["baseline"]["metrics"]["database_bytes"],
        )
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(
            first["dataset"]["digest_semantics"], SUITE_DIGEST_SEMANTICS
        )
        revision = first["config"]["code_revision"]
        self.assertEqual(
            revision["source_digest_semantics"], SOURCE_DIGEST_SEMANTICS
        )
        self.assertEqual(len(revision["runtime_source_sha256"]), 64)
        self.assertGreater(revision["runtime_source_files"], 0)
        self.assertIn(revision["git_dirty"], (True, False, None))
        self.assertEqual(self._stable_outcomes(first), self._stable_outcomes(second))
        self.assertEqual(first["results"]["baseline"]["label"], BASELINE_LABEL)
        metrics = first["results"]["baseline"]["metrics"]
        self.assertEqual(metrics["budget_compliance_rate"], 1.0)
        self.assertGreater(metrics["database_bytes"], 0)
        self.assertEqual(metrics["query_errors"], 0)
        self.assertGreaterEqual(
            metrics["retrieval_latency_ms_p99"],
            metrics["retrieval_latency_ms_p95"],
        )
        abstention_rows = [
            query
            for query in first["results"]["baseline"]["queries"]
            if query["task"] == "abstention"
        ]
        self.assertEqual(len(abstention_rows), 3)
        self.assertEqual(
            first["results"]["baseline"]["by_task"]["abstention"]["queries"],
            3,
        )
        self.assertEqual(
            metrics["abstention_accuracy"],
            sum(query["abstained"] for query in abstention_rows) / 3,
        )
        self.assertIn(
            "retrieval-level no-evidence",
            first["metric_semantics"]["abstention_accuracy"].casefold(),
        )

    def test_failed_phase_a_queries_are_never_budget_compliant(self) -> None:
        with patch(
            "benchmarks.run.FixedChunkFTS5Baseline.recall",
            side_effect=RuntimeError("forced retrieval failure"),
        ):
            report = run_benchmark(
                events=12,
                seed=7,
                budget_tokens=512,
                include_evicortex=False,
            )

        result = report["results"]["baseline"]
        self.assertEqual(result["metrics"]["query_errors"], len(result["queries"]))
        self.assertEqual(result["metrics"]["budget_compliance_rate"], 0.0)
        self.assertTrue(all(query["error"] for query in result["queries"]))
        self.assertTrue(
            all(not query["budget_compliant"] for query in result["queries"])
        )
        self.assertEqual(result["metrics"]["abstention_accuracy"], 0.0)

    @unittest.skipUnless(
        importlib.util.find_spec("evicortex") is not None,
        "public EviCortex package is not present yet",
    )
    def test_comparison_runner_uses_original_source_ids(self) -> None:
        report = run_benchmark(
            events=16,
            seed=29,
            budget_tokens=512,
        )
        repeated = run_benchmark(
            events=16,
            seed=29,
            budget_tokens=512,
        )
        self.assertIn("evicortex", report["results"])
        self.assertEqual(
            self._stable_outcomes(report, "evicortex"),
            self._stable_outcomes(repeated, "evicortex"),
        )
        for system in ("baseline", "evicortex"):
            self.assertEqual(
                report["results"][system]["metrics"]["database_bytes"],
                repeated["results"][system]["metrics"]["database_bytes"],
            )
        candidate = report["results"]["evicortex"]
        self.assertEqual(candidate["metrics"]["budget_compliance_rate"], 1.0)
        self.assertEqual(candidate["metrics"]["query_errors"], 0)
        for item in candidate["queries"]:
            self.assertFalse(
                any(source.startswith("unmapped:") for source in item["retrieved_source_ids"])
            )
        isolation = next(
            item
            for item in candidate["queries"]
            if item["query_id"] == "namespace-recovery-code"
        )
        self.assertFalse(
            set(isolation["forbidden_evidence_ids"])
            & set(isolation["retrieved_source_ids"])
        )


class BenchmarkSweepTests(unittest.TestCase):
    def test_seeded_bootstrap_is_deterministic_and_paired(self) -> None:
        pairs = ((0.8, 0.5), (0.7, 0.6), (0.9, 0.7))
        first = bootstrap_paired_delta_ci(
            pairs,
            samples=128,
            seed=41,
            label="quality:test",
            direction="positive_favors_candidate",
        )
        second = bootstrap_paired_delta_ci(
            pairs,
            samples=128,
            seed=41,
            label="quality:test",
            direction="positive_favors_candidate",
        )
        ratio = bootstrap_paired_ratio_ci(
            ((8.0, 10.0), (6.0, 10.0)),
            samples=128,
            seed=41,
            label="resource:test",
            direction="below_one_favors_candidate",
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.2)
        self.assertEqual(first["operation"], "candidate_minus_baseline")
        self.assertAlmostEqual(ratio["estimate"], 0.7)
        self.assertEqual(ratio["operation"], "candidate_over_baseline")

    def test_baseline_only_sweep_has_stable_quality_and_small_n_warning(self) -> None:
        first = run_sweep(
            seeds=(5, 3),
            events=12,
            budget_tokens=512,
            bootstrap_samples=64,
            bootstrap_seed=101,
            include_evicortex=False,
        )
        second = run_sweep(
            seeds=(3, 5),
            events=12,
            budget_tokens=512,
            bootstrap_samples=64,
            bootstrap_seed=101,
            include_evicortex=False,
        )

        self.assertEqual(first["config"]["seeds"], [3, 5])
        self.assertEqual(
            [item["dataset_sha256"] for item in first["per_seed"]],
            [item["dataset_sha256"] for item in second["per_seed"]],
        )
        self.assertEqual(
            first["aggregate"]["systems"]["baseline"]["quality"],
            second["aggregate"]["systems"]["baseline"]["quality"],
        )
        self.assertTrue(first["resampling"]["small_n"])
        self.assertNotIn("comparison", first["aggregate"])
        self.assertTrue(
            any(
                "statistical significance" in item and "not" in item
                for item in first["limitations"]
            )
        )

    @unittest.skipUnless(
        importlib.util.find_spec("evicortex") is not None,
        "public EviCortex package is not present yet",
    )
    def test_candidate_sweep_reports_deltas_ratios_and_per_seed_summaries(self) -> None:
        report = run_sweep(
            seeds=(3, 5),
            events=12,
            budget_tokens=512,
            bootstrap_samples=64,
            bootstrap_seed=103,
        )

        self.assertEqual(len(report["per_seed"]), 2)
        self.assertTrue(
            all(set(item["systems"]) == {"baseline", "evicortex"} for item in report["per_seed"])
        )
        comparison = report["aggregate"]["comparison"]
        quality = comparison["quality_deltas"]["supported_answer_rate"]
        resources = comparison["resource_ratios"]["database_bytes"]
        self.assertIn(
            "retrieval_latency_ms_p99", comparison["resource_ratios"]
        )
        self.assertIn(
            "retrieval_latency_ms_p50", comparison["resource_ratios"]
        )
        self.assertEqual(quality["operation"], "candidate_minus_baseline")
        self.assertEqual(resources["operation"], "candidate_over_baseline")
        self.assertEqual(quality["independent_streams"], 2)
        self.assertEqual(len(quality["ci95"]), 2)
        self.assertNotIn("significant", comparison["interpretation"].casefold())


if __name__ == "__main__":
    unittest.main()
