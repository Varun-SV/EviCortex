"""Focused smoke, determinism, and fairness tests for the Phase B runner."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from unittest.mock import patch
import unittest

from benchmarks.run_phase_b import (
    CountingEmbedder,
    EVICORTEX_PHASE_B_LABEL,
    main,
    run_phase_b,
)
from benchmarks.synthetic import generate_suite
from evicortex import HashingEmbedder
from evicortex.semantic import EmbeddingProfile


class CountingEmbedderTests(unittest.TestCase):
    def test_separate_wrappers_share_profile_but_not_counters(self) -> None:
        candidate = CountingEmbedder(
            HashingEmbedder(dimensions=32), counter_name="candidate"
        )
        hybrid = CountingEmbedder(
            HashingEmbedder(dimensions=32), counter_name="hybrid"
        )

        self.assertEqual(
            EmbeddingProfile.from_embedder(candidate),
            EmbeddingProfile.from_embedder(hybrid),
        )
        candidate.embed_batch(("alpha", "beta"))
        self.assertEqual(candidate.snapshot().batch_calls, 1)
        self.assertEqual(candidate.snapshot().text_count, 2)
        self.assertEqual(hybrid.snapshot().batch_calls, 0)
        self.assertFalse(candidate.is_semantic)
        self.assertTrue(candidate.diagnostic_only)

    def test_capability_flags_are_proxied_from_the_wrapped_provider(self) -> None:
        class CapabilityHashingEmbedder(HashingEmbedder):
            @property
            def is_semantic(self) -> bool:
                return True

            @property
            def diagnostic_only(self) -> bool:
                return False

        counter = CountingEmbedder(
            CapabilityHashingEmbedder(dimensions=16),
            counter_name="capability-proxy",
        )

        self.assertTrue(counter.is_semantic)
        self.assertFalse(counter.diagnostic_only)

    def test_role_aware_calls_are_counted_without_symmetric_fallback(self) -> None:
        class RoleAwareProvider:
            identity = "tests/counting-role-aware-v1"
            dimensions = 2
            normalized = True

            def embed_batch(
                self, texts: Sequence[str]
            ) -> tuple[tuple[float, float], ...]:
                raise AssertionError("legacy path used")

            def embed_documents(
                self, texts: Sequence[str]
            ) -> tuple[tuple[float, float], ...]:
                return tuple((1.0, 0.0) for _ in texts)

            def embed_query(self, text: str) -> tuple[float, float]:
                return (0.0, 1.0)

        counter = CountingEmbedder(
            RoleAwareProvider(),
            counter_name="role-aware",
        )
        self.assertEqual(
            counter.embed_documents(("one", "two")),
            ((1.0, 0.0), (1.0, 0.0)),
        )
        self.assertEqual(counter.embed_query("query"), (0.0, 1.0))
        self.assertEqual(counter.snapshot().batch_calls, 2)
        self.assertEqual(counter.snapshot().text_count, 3)


class PhaseBDiagnosticRunnerTests(unittest.TestCase):
    def test_raw_content_extraction_and_equivalent_hybrid_metadata(self) -> None:
        suite = generate_suite(event_count=12, seed=17)
        # If the runner improperly reads SyntheticEvent.claims, this stripped suite
        # cannot create claims or resolve the update.  Canonical raw content remains.
        stripped = replace(
            suite,
            events=tuple(replace(event, claims=()) for event in suite.events),
        )
        with patch("benchmarks.run_phase_b.generate_suite", return_value=stripped):
            report = run_phase_b(
                events=12,
                seed=17,
                budget_tokens=512,
                embedding_dimensions=64,
                semantic_scan_cap=128,
            )

        extraction = report["extraction"]
        self.assertEqual(extraction["claims_emitted"], 12)
        self.assertEqual(extraction["provenance_rows"], 12)
        self.assertEqual(extraction["abstentions"], 0)
        self.assertEqual(extraction["diagnostics_total"], 0)
        self.assertEqual(extraction["supersession_references_resolved"], 1)
        self.assertEqual(extraction["extractor_invocations"], 12)
        self.assertEqual(
            extraction["invocation_policy"],
            "exactly once per event through remember_extracted",
        )
        persistence = extraction["persistence"]
        self.assertEqual(persistence["mode"], "durable")
        self.assertEqual(persistence["extraction_runs"], 12)
        self.assertEqual(persistence["claim_derivation_rows"], 12)
        self.assertEqual(persistence["diagnostic_rows"], 0)
        self.assertEqual(persistence["get_extraction_records_verified"], 12)
        self.assertEqual(persistence["claim_id_mappings_verified"], 12)
        self.assertTrue(persistence["included_in_evicortex_database_bytes"])
        self.assertEqual(extraction["transient_observation"]["provenance_rows"], 12)

        candidate = report["results"]["evicortex"]
        hybrid = report["results"]["hybrid_baseline"]
        query_count = len(stripped.queries)
        self.assertEqual(query_count, 12)
        self.assertEqual(candidate["index_stats"]["store"]["claims"], 12)
        self.assertEqual(candidate["index_stats"]["durable_extraction"], persistence)
        self.assertEqual(hybrid["index_stats"]["sources"], 12)
        self.assertEqual(
            candidate["embedding_usage"]["by_phase"]["ingestion"]["text_count"],
            24,
        )
        self.assertEqual(
            hybrid["embedding_usage"]["by_phase"]["ingestion"]["text_count"],
            hybrid["index_stats"]["vectors"],
        )
        for result in (candidate, hybrid):
            self.assertEqual(result["metrics"]["query_errors"], 0)
            self.assertEqual(result["metrics"]["budget_compliance_rate"], 1.0)
            self.assertGreaterEqual(
                result["metrics"]["retrieval_latency_ms_p99"],
                result["metrics"]["retrieval_latency_ms_p95"],
            )
            abstention_queries = [
                query for query in result["queries"] if query["task"] == "abstention"
            ]
            self.assertEqual(len(abstention_queries), 3)
            self.assertEqual(result["by_task"]["abstention"]["queries"], 3)
            self.assertEqual(
                result["metrics"]["abstention_accuracy"],
                sum(query["abstained"] for query in abstention_queries) / 3,
            )
            self.assertEqual(
                result["embedding_usage"]["by_phase"]["warmup"]["text_count"],
                query_count,
            )
            self.assertEqual(
                result["embedding_usage"]["by_phase"]["measured_query"][
                    "text_count"
                ],
                query_count,
            )
            self.assertEqual(
                result["embedding_usage"]["unaccounted"],
                {"batch_calls": 0, "text_count": 0, "text_utf8_bytes": 0},
            )
            for query in result["queries"]:
                self.assertFalse(
                    any(
                        source.startswith("unmapped:")
                        for source in query["retrieved_source_ids"]
                    )
                )
        self.assertEqual(candidate["metrics"]["current_accuracy"], 1.0)
        self.assertEqual(hybrid["metrics"]["current_accuracy"], 1.0)
        self.assertIn(
            "retrieval-level no-evidence",
            report["metric_semantics"]["abstention_accuracy"].casefold(),
        )

    def test_durable_abstention_diagnostic_counts_are_verified(self) -> None:
        suite = generate_suite(event_count=12, seed=37)
        abstaining_event = replace(
            suite.events[0],
            content="No canonical claim is present in this evidence.",
            claims=(),
        )
        patched_suite = replace(
            suite,
            events=(abstaining_event, *suite.events[1:]),
        )
        with patch("benchmarks.run_phase_b.generate_suite", return_value=patched_suite):
            report = run_phase_b(
                events=12,
                seed=37,
                budget_tokens=512,
                embedding_dimensions=32,
                semantic_scan_cap=128,
            )

        extraction = report["extraction"]
        persistence = extraction["persistence"]
        self.assertEqual(extraction["extractor_invocations"], 12)
        self.assertEqual(extraction["claims_emitted"], 11)
        self.assertEqual(extraction["provenance_rows"], 11)
        self.assertEqual(extraction["abstentions"], 1)
        self.assertEqual(extraction["diagnostics_total"], 1)
        self.assertEqual(extraction["diagnostics_by_code"], {"no_canonical_claims": 1})
        self.assertEqual(persistence["extraction_runs"], 12)
        self.assertEqual(persistence["claim_derivation_rows"], 11)
        self.assertEqual(persistence["diagnostic_rows"], 1)
        self.assertEqual(persistence["abstained_runs"], 1)
        self.assertEqual(persistence["claim_id_mappings_verified"], 11)
        self.assertGreater(report["results"]["evicortex"]["metrics"]["database_bytes"], 0)

    @staticmethod
    def _stable_projection(report: dict[str, object]) -> dict[str, object]:
        extraction = report["extraction"]  # type: ignore[index]
        systems: dict[str, object] = {}
        for name, result in report["results"].items():  # type: ignore[index]
            metrics = result["metrics"]
            systems[name] = {
                "quality": {
                    key: metrics[key]
                    for key in (
                        "evidence_recall_at_budget",
                        "supported_answer_rate",
                        "current_accuracy",
                        "historical_as_of_accuracy",
                        "stale_leakage_rate",
                        "abstention_accuracy",
                        "namespace_isolation_accuracy",
                        "budget_compliance_rate",
                        "estimated_context_tokens_total",
                        "query_errors",
                    )
                },
                "embedding_usage": result["embedding_usage"],
                "index_stats": result["index_stats"],
                "scan": result["semantic_scan_diagnostics"],
                "queries": tuple(
                    (
                        query["query_id"],
                        tuple(query["retrieved_source_ids"]),
                        query["evidence_recall"],
                        query["has_sufficient_gold_evidence"],
                        query["estimated_context_tokens"],
                        query["error"],
                        query["warmup_error"],
                        query["retrieval_diagnostics"],
                    )
                    for query in result["queries"]
                ),
            }
        return {
            "dataset": report["dataset"],
            "extraction": {
                key: extraction[key]
                for key in (
                    "extractor_identity",
                    "extractor_invocations",
                    "invocation_policy",
                    "events_processed",
                    "claims_emitted",
                    "provenance_rows",
                    "abstentions",
                    "supersession_references_resolved",
                    "diagnostics_total",
                    "diagnostics_by_code",
                    "diagnostic_events",
                    "persistence",
                    "transient_observation",
                )
            },
            "systems": systems,
        }

    def test_repeated_seed_has_deterministic_non_timing_results(self) -> None:
        first = run_phase_b(
            events=12,
            seed=23,
            budget_tokens=512,
            embedding_dimensions=32,
            semantic_scan_cap=128,
        )
        second = run_phase_b(
            events=12,
            seed=23,
            budget_tokens=512,
            embedding_dimensions=32,
            semantic_scan_cap=128,
        )
        self.assertEqual(
            self._stable_projection(first),
            self._stable_projection(second),
        )
        for system in ("hybrid_baseline", "evicortex"):
            self.assertEqual(
                first["results"][system]["metrics"]["database_bytes"],
                second["results"][system]["metrics"]["database_bytes"],
            )

    def test_scan_cap_policy_reports_both_failure_modes(self) -> None:
        report = run_phase_b(
            events=12,
            seed=7,
            budget_tokens=512,
            embedding_dimensions=32,
            semantic_scan_cap=1,
        )
        hybrid = report["results"]["hybrid_baseline"]
        candidate = report["results"]["evicortex"]

        self.assertEqual(report["config"]["semantic_scan_cap"], 1)
        self.assertEqual(report["config"]["hybrid_max_exact_candidates"], 1)
        self.assertIn("lexical", report["scan_cap_policy"]["hybrid_scope"])
        self.assertGreater(hybrid["metrics"]["query_errors"], 0)
        failed_hybrid_queries = [
            query for query in hybrid["queries"] if query["error"] is not None
        ]
        self.assertTrue(failed_hybrid_queries)
        self.assertTrue(
            all(not query["budget_compliant"] for query in failed_hybrid_queries)
        )
        self.assertEqual(
            hybrid["metrics"]["budget_compliance_rate"],
            sum(query["budget_compliant"] for query in hybrid["queries"])
            / len(hybrid["queries"]),
        )
        self.assertLess(hybrid["metrics"]["budget_compliance_rate"], 1.0)
        comparison = report["comparison"]
        self.assertFalse(comparison["resource_ratios_comparable"])
        self.assertIn(
            "query errors",
            comparison["resource_ratios_exclusion_reason"].casefold(),
        )
        self.assertTrue(
            all(
                value is None
                for value in comparison["candidate_over_baseline"].values()
            )
        )
        self.assertGreater(
            hybrid["semantic_scan_diagnostics"]["scan_overflow_query_errors"], 0
        )
        self.assertEqual(candidate["metrics"]["query_errors"], 0)
        self.assertGreater(
            candidate["semantic_scan_diagnostics"][
                "queries_with_scan_truncation"
            ],
            0,
        )
        self.assertEqual(
            candidate["semantic_scan_diagnostics"]["reported_scan_caps"], [1]
        )

    def test_json_cli_is_machine_readable_and_labels_limitations(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                (
                    "--events",
                    "12",
                    "--seed",
                    "31",
                    "--budget-tokens",
                    "512",
                    "--embedding-dimensions",
                    "32",
                    "--semantic-scan-cap",
                    "128",
                    "--json",
                )
            )
        report = json.loads(output.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(report["schema_version"], 2)
        self.assertIn("complete-canonical-json.v2", report["dataset"]["digest_semantics"])
        revision = report["config"]["code_revision"]
        self.assertEqual(len(revision["runtime_source_sha256"]), 64)
        self.assertGreater(revision["runtime_source_files"], 0)
        self.assertIn("non-semantic", report["claim_scope"])
        self.assertIn("non-semantic", report["results"]["evicortex"]["label"])
        self.assertEqual(
            report["results"]["evicortex"]["label"], EVICORTEX_PHASE_B_LABEL
        )
        hybrid_label = report["results"]["hybrid_baseline"]["label"].casefold()
        self.assertIn("not", hybrid_label)
        self.assertIn("strong", hybrid_label)
        self.assertIn("no superiority", report["claim_scope"].casefold())
        limitations = " ".join(report["limitations"]).casefold()
        self.assertIn("literal 'not x'", limitations)
        self.assertIn("partial-ood", limitations)
        self.assertIn("0.333", limitations)


if __name__ == "__main__":
    unittest.main()
