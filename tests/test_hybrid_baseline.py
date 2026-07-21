"""Focused tests for the isolated diagnostic hybrid comparator."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from benchmarks.hybrid_baseline import (
    DIAGNOSTIC_HYBRID_LABEL,
    DiagnosticHybridRAGBaseline,
    _chunk_spans,
)
from evicortex import ClaimInput, EviCortex
from evicortex.budget import BudgetTooSmallError, estimate_tokens
from evicortex.plugins import HashingEmbedder


class _MeaningEmbedder:
    """Tiny protocol-conforming test double with one deliberate synonym group."""

    identity = "tests/meaning-embedder-v1"
    dimensions = 2
    normalized = True

    def embed_batch(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, float], ...]:
        vectors: list[tuple[float, float]] = []
        for value in texts:
            text = value.casefold()
            if any(term in text for term in ("automobile", "car", "vehicle")):
                vectors.append((1.0, 0.0))
            elif any(term in text for term in ("banana", "fruit")):
                vectors.append((0.0, 1.0))
            else:
                root = math.sqrt(0.5)
                vectors.append((root, root))
        return tuple(vectors)


class _RoleAwareMeaningEmbedder:
    """Asymmetric test provider that rejects symmetric fallback calls."""

    identity = "tests/role-aware-meaning-v1"
    dimensions = 2
    normalized = True

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_batch(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, float], ...]:
        raise AssertionError("legacy symmetric embedding path was used")

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, float], ...]:
        frozen = tuple(texts)
        self.document_calls.append(frozen)
        return tuple((1.0, 0.0) for _ in frozen)

    def embed_query(self, text: str) -> tuple[float, float]:
        self.query_calls.append(text)
        return (1.0, 0.0)


class DiagnosticHybridBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "hybrid.db"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _open(
        self,
        **kwargs: object,
    ) -> DiagnosticHybridRAGBaseline:
        defaults: dict[str, object] = {
            "chunk_chars": (64, 160),
            "chunk_overlaps": (8, 24),
            "channel_limit": 20,
            "candidate_limit": 10,
            "max_expansion_chars": 240,
        }
        defaults.update(kwargs)
        return DiagnosticHybridRAGBaseline.open(self.path, **defaults)

    def test_label_is_explicitly_diagnostic(self) -> None:
        self.assertIn("diagnostic", DIAGNOSTIC_HYBRID_LABEL.casefold())
        self.assertIn("not", DIAGNOSTIC_HYBRID_LABEL.casefold())
        self.assertTrue(DiagnosticHybridRAGBaseline.diagnostic_only)

    def test_role_aware_provider_separates_index_and_query_encoding(self) -> None:
        embedder = _RoleAwareMeaningEmbedder()
        with self._open(embedder=embedder) as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="semantic/source",
                content="Opaque document text.",
                occurred_at="2026-01-01T00:00:00Z",
            )
            result = baseline.recall(
                namespace="tenant-a",
                query="semantic paraphrase",
                budget_tokens=512,
            )

        self.assertEqual(result.evidence_ids, ("semantic/source",))
        self.assertEqual(len(embedder.document_calls), 1)
        self.assertEqual(embedder.query_calls, ["semantic paraphrase"])

    def test_boundary_snapped_chunking_is_deterministic_and_loses_no_text(self) -> None:
        text = (
            "Alpha sentence ends here. "
            "Beta sentence follows with more words.\n\n"
            "Gamma paragraph remains fully covered."
        )
        without_overlap = _chunk_spans(text, size=38, overlap=0)
        repeated = _chunk_spans(text, size=38, overlap=0)
        with_overlap = _chunk_spans(text, size=38, overlap=7)

        self.assertEqual(without_overlap, repeated)
        self.assertEqual("".join(text[start:end] for start, end in without_overlap), text)
        self.assertEqual(without_overlap[0][1], text.index("Beta"))
        self.assertNotEqual(without_overlap[0][1], 38)
        coverage = [0] * len(text)
        for start, end in with_overlap:
            self.assertLess(start, end)
            for index in range(start, end):
                coverage[index] += 1
        self.assertTrue(all(count >= 1 for count in coverage))

    def test_two_granularities_fuse_and_deduplicate_one_short_source(self) -> None:
        with self._open() as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="source/target",
                occurred_at="2026-01-01T00:00:00Z",
                content="The Quartz device has serial number QZ-4401.",
            )

            stats = baseline.stats()
            result = baseline.recall(
                namespace="tenant-a",
                query="Quartz serial number",
                budget_tokens=512,
            )

        self.assertEqual(stats.sources, 1)
        self.assertEqual(stats.chunks, 2)
        self.assertEqual(stats.vectors, 2)
        self.assertEqual(
            dict(stats.chunks_by_granularity),
            {"large": 1, "small": 1},
        )
        self.assertEqual(result.evidence_ids, ("source/target",))
        self.assertEqual(len(result.evidence), 1)
        channels = {reason.channel for reason in result.evidence[0].reasons}
        self.assertEqual(
            channels,
            {
                "bm25:large",
                "bm25:small",
                "cosine:large",
                "cosine:small",
            },
        )

    def test_namespace_and_as_of_filters_precede_both_channels(self) -> None:
        with self._open() as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="alpha/old",
                occurred_at="2026-01-01T00:00:00Z",
                content="Atlas launch code is ALPHA-17.",
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="alpha/future",
                occurred_at="2026-03-01T00:00:00Z",
                content="Atlas launch code is FUTURE-88.",
            )
            baseline.remember(
                namespace="tenant-b",
                source_id="beta/secret",
                occurred_at="2026-01-02T00:00:00Z",
                content="Atlas launch code is OTHER-99.",
            )

            result = baseline.recall(
                namespace="tenant-a",
                query="Atlas launch code",
                budget_tokens=512,
                as_of="2026-01-15T00:00:00Z",
            )

        self.assertEqual(result.evidence_ids, ("alpha/old",))
        # One eligible short source at each of the two granularities was scored.
        self.assertEqual(result.diagnostics.semantic_candidates_scored, 2)
        self.assertNotIn("alpha/future", result.evidence_ids)
        self.assertNotIn("beta/secret", result.evidence_ids)

    def test_bm25_ranking_is_invariant_to_other_namespace_term_statistics(self) -> None:
        with self._open() as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="alpha/one",
                occurred_at="2026-01-01T00:00:00Z",
                content="Amber common signal appears in the first record.",
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="alpha/two",
                occurred_at="2026-01-02T00:00:00Z",
                content="Amber appears twice: amber.",
            )
            before = baseline.recall(
                namespace="tenant-a",
                query="amber common",
                budget_tokens=512,
            )

            # These rows radically change global FTS document frequencies.  They
            # must not alter either ranks or raw BM25 scores in tenant-a.
            for index in range(24):
                baseline.remember(
                    namespace="tenant-b",
                    source_id=f"beta/{index:03d}",
                    occurred_at=f"2026-02-{index + 1:02d}T00:00:00Z",
                    content=(
                        "Amber appears in this unrelated namespace."
                        if index < 23
                        else "Common appears in this unrelated namespace."
                    ),
                )
            after = baseline.recall(
                namespace="tenant-a",
                query="amber common",
                budget_tokens=512,
            )

        self.assertEqual(before, after)
        before_bm25 = tuple(
            (
                item.source_id,
                reason.channel,
                reason.rank,
                reason.raw_score,
            )
            for item in before.evidence
            for reason in item.reasons
            if reason.channel.startswith("bm25:")
        )
        self.assertTrue(before_bm25)

    def test_append_only_supersession_resolves_current_and_historical_views(self) -> None:
        with self._open() as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="theme/light",
                occurred_at="2026-01-01T00:00:00Z",
                content="Ada prefers the light interface theme.",
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="theme/dark",
                occurred_at="2026-02-01T00:00:00Z",
                valid_from="2026-02-01T00:00:00Z",
                supersedes_source_id="theme/light",
                content="Ada now prefers the dark interface theme.",
            )

            current = baseline.recall(
                namespace="tenant-a",
                query="Ada interface theme",
                budget_tokens=512,
            )
            historical = baseline.recall(
                namespace="tenant-a",
                query="Ada interface theme",
                budget_tokens=512,
                as_of="2026-01-15T00:00:00Z",
            )
            stats = baseline.stats()

        self.assertEqual(current.evidence_ids, ("theme/dark",))
        self.assertEqual(historical.evidence_ids, ("theme/light",))
        # Supersession is a projection filter, not destructive replacement.
        self.assertEqual(stats.sources, 2)

    def test_finite_retroactive_successor_is_old_then_new_then_unknown(self) -> None:
        with self._open() as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="policy/original",
                occurred_at="2026-01-01T00:00:00Z",
                valid_from="2026-01-01T00:00:00Z",
                content="The travel policy allowance is 100 credits.",
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="policy/retroactive-revision",
                # The later observation claims an earlier effective date.
                occurred_at="2026-03-01T00:00:00Z",
                recorded_at="2026-03-02T00:00:00Z",
                valid_from="2026-01-10T00:00:00Z",
                valid_to="2026-05-01T00:00:00Z",
                supersedes_source_id="policy/original",
                content="The revised travel policy allowance is 120 credits.",
            )

            before_successor_is_visible = baseline.recall(
                namespace="tenant-a",
                query="travel policy allowance credits",
                budget_tokens=512,
                as_of="2026-02-01T00:00:00Z",
            )
            while_successor_is_visible = baseline.recall(
                namespace="tenant-a",
                query="travel policy allowance credits",
                budget_tokens=512,
                as_of="2026-03-15T00:00:00Z",
            )
            after_successor_expires = baseline.recall(
                namespace="tenant-a",
                query="travel policy allowance credits",
                budget_tokens=512,
                as_of="2026-06-01T00:00:00Z",
            )

        self.assertEqual(
            before_successor_is_visible.evidence_ids,
            ("policy/original",),
        )
        self.assertEqual(
            while_successor_is_visible.evidence_ids,
            ("policy/retroactive-revision",),
        )
        self.assertEqual(after_successor_expires.evidence_ids, ())

    def test_future_effective_successor_does_not_close_predecessor_early(self) -> None:
        with self._open() as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="schedule/original",
                occurred_at="2026-01-01T00:00:00Z",
                valid_from="2026-01-01T00:00:00Z",
                content="The support schedule uses weekday coverage.",
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="schedule/summer",
                occurred_at="2026-02-01T00:00:00Z",
                valid_from="2026-04-01T00:00:00Z",
                valid_to="2026-05-01T00:00:00Z",
                supersedes_source_id="schedule/original",
                content="The support schedule uses summer coverage.",
            )

            before_effective_start = baseline.recall(
                namespace="tenant-a",
                query="support schedule coverage",
                budget_tokens=512,
                as_of="2026-03-01T00:00:00Z",
            )
            during_successor = baseline.recall(
                namespace="tenant-a",
                query="support schedule coverage",
                budget_tokens=512,
                as_of="2026-04-15T00:00:00Z",
            )
            after_successor = baseline.recall(
                namespace="tenant-a",
                query="support schedule coverage",
                budget_tokens=512,
                as_of="2026-06-01T00:00:00Z",
            )

        # This non-retroactive case is directly representable by EviCortex's
        # claim-level valid-time model, so assert the comparator states against it.
        core_path = Path(self.temporary_directory.name) / "core-temporal.db"
        with EviCortex.open(core_path) as memory:
            original = memory.remember(
                namespace="tenant-a",
                source="schedule/original",
                occurred_at="2026-01-01T00:00:00Z",
                content="The support schedule uses weekday coverage.",
                claims=(
                    ClaimInput(
                        "support:schedule",
                        "coverage",
                        "weekday",
                        valid_from="2026-01-01T00:00:00Z",
                    ),
                ),
            )
            memory.remember(
                namespace="tenant-a",
                source="schedule/summer",
                occurred_at="2026-02-01T00:00:00Z",
                content="The support schedule uses summer coverage.",
                claims=(
                    ClaimInput(
                        "support:schedule",
                        "coverage",
                        "summer",
                        valid_from="2026-04-01T00:00:00Z",
                        valid_to="2026-05-01T00:00:00Z",
                        supersedes=original.claims[0].id,
                    ),
                ),
            )
            core_values = tuple(
                tuple(fact.value for fact in memory.recall(
                    namespace="tenant-a",
                    query="support schedule coverage",
                    budget_tokens=512,
                    as_of=cutoff,
                ).facts)
                for cutoff in (
                    "2026-03-01T00:00:00Z",
                    "2026-04-15T00:00:00Z",
                    "2026-06-01T00:00:00Z",
                )
            )

        self.assertEqual(
            before_effective_start.evidence_ids,
            ("schedule/original",),
        )
        self.assertEqual(during_successor.evidence_ids, ("schedule/summer",))
        self.assertEqual(after_successor.evidence_ids, ())
        self.assertEqual(core_values, (("weekday",), ("summer",), ()))

    def test_exact_cosine_channel_can_retrieve_without_a_lexical_match(self) -> None:
        with self._open(embedder=_MeaningEmbedder()) as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="garage",
                occurred_at="2026-01-01T00:00:00Z",
                content="A red car is parked in the garage.",
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="kitchen",
                occurred_at="2026-01-01T00:01:00Z",
                content="A ripe banana rests in the kitchen.",
            )

            result = baseline.recall(
                namespace="tenant-a",
                query="Where is the automobile?",
                budget_tokens=512,
            )

        self.assertEqual(result.evidence_ids[0], "garage")
        first_channels = {
            reason.channel for reason in result.evidence[0].reasons
        }
        self.assertTrue(first_channels)
        self.assertTrue(
            all(channel.startswith("cosine:") for channel in first_channels)
        )
        self.assertEqual(
            result.diagnostics.diagnostic_semantic_candidates_filtered,
            0,
        )

    def test_hash_collision_cannot_answer_an_unknown_query(self) -> None:
        embedder = HashingEmbedder(dimensions=1)
        query_token = "zephyrunknown"
        query_vector = embedder.embed_batch((query_token,))[0]
        collision_token = next(
            token
            for index in range(100)
            if (token := f"unrelatedtoken{index}") != query_token
            and embedder.embed_batch((token,))[0] == query_vector
        )

        with self._open(embedder=embedder) as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="collision/source",
                occurred_at="2026-01-01T00:00:00Z",
                content=collision_token,
            )
            unknown = baseline.recall(
                namespace="tenant-a",
                query=query_token,
                budget_tokens=512,
            )
            lexical = baseline.recall(
                namespace="tenant-a",
                query=collision_token,
                budget_tokens=512,
            )

        self.assertEqual(unknown.evidence_ids, ())
        self.assertEqual(unknown.diagnostics.semantic_candidates_scored, 2)
        self.assertEqual(
            unknown.diagnostics.diagnostic_semantic_candidates_filtered,
            2,
        )
        self.assertEqual(lexical.evidence_ids, ("collision/source",))
        self.assertTrue(
            any(
                reason.channel.startswith("bm25:")
                for reason in lexical.evidence[0].reasons
            )
        )

    def test_neighbor_expansion_diversity_and_prompt_budget_are_deterministic(self) -> None:
        with self._open(
            chunk_chars=(40, 88),
            chunk_overlaps=(4, 12),
            max_expansion_chars=150,
            candidate_limit=6,
        ) as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="source/one",
                occurred_at="2026-01-01T00:00:00Z",
                content=(
                    "preface alpha alpha alpha. "
                    "neighbor-before context. TARGET-NEEDLE is here. "
                    "neighbor-after context. alpha alpha epilogue."
                ),
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="source/two",
                occurred_at="2026-01-02T00:00:00Z",
                content="A second TARGET-NEEDLE record provides independent evidence.",
            )

            roomy = baseline.recall(
                namespace="tenant-a",
                query="TARGET-NEEDLE",
                budget_tokens=512,
            )
            first = baseline.recall(
                namespace="tenant-a",
                query="TARGET-NEEDLE",
                budget_tokens=96,
            )
            second = baseline.recall(
                namespace="tenant-a",
                query="TARGET-NEEDLE",
                budget_tokens=96,
            )

        self.assertEqual(first, second)
        self.assertEqual(first.estimated_tokens, estimate_tokens(first.to_prompt()))
        self.assertLessEqual(first.estimated_tokens, 96)
        self.assertTrue(roomy.evidence)
        self.assertTrue(any(item.expanded for item in roomy.evidence))
        self.assertEqual(
            set(roomy.evidence_ids),
            {"source/one", "source/two"},
        )
        for index, left in enumerate(roomy.evidence):
            for right in roomy.evidence[index + 1 :]:
                if left.source_id != right.source_id:
                    continue
                overlap = max(
                    0,
                    min(left.end_char, right.end_char)
                    - max(left.start_char, right.start_char),
                )
                shorter = min(
                    left.end_char - left.start_char,
                    right.end_char - right.start_char,
                )
                self.assertLess(overlap / shorter, 0.5)

    def test_too_small_budget_and_exact_scan_overflow_fail_loudly(self) -> None:
        with self._open(max_exact_candidates=1) as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="one",
                occurred_at="2026-01-01T00:00:00Z",
                content="Quartz one.",
            )
            baseline.remember(
                namespace="tenant-a",
                source_id="two",
                occurred_at="2026-01-02T00:00:00Z",
                content="Quartz two.",
            )

            with self.assertRaises(BudgetTooSmallError):
                baseline.recall(
                    namespace="missing",
                    query="Quartz",
                    budget_tokens=1,
                )
            with self.assertRaisesRegex(
                RuntimeError, "max_exact_candidates"
            ):
                baseline.recall(
                    namespace="tenant-a",
                    query="Quartz",
                    budget_tokens=512,
                )

    def test_reopening_rejects_an_incompatible_embedder_index(self) -> None:
        with self._open(embedder=HashingEmbedder(dimensions=32)) as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="source",
                occurred_at="2026-01-01T00:00:00Z",
                content="Persistent vector profile.",
            )

        with self.assertRaisesRegex(ValueError, "incompatible"):
            self._open(embedder=HashingEmbedder(dimensions=64))

    def test_populated_index_rejects_missing_compatibility_config(self) -> None:
        with self._open(embedder=HashingEmbedder(dimensions=32)) as baseline:
            baseline.remember(
                namespace="tenant-a",
                source_id="source",
                occurred_at="2026-01-01T00:00:00Z",
                content="Compatibility metadata must fail closed.",
            )

        connection = sqlite3.connect(self.path)
        try:
            with connection:
                connection.execute(
                    "DELETE FROM hybrid_index_config "
                    "WHERE key = 'embedding_profile_sha256'"
                )
        finally:
            connection.close()

        with self.assertRaisesRegex(
            ValueError, "missing compatibility config"
        ):
            self._open(embedder=HashingEmbedder(dimensions=32))


if __name__ == "__main__":
    unittest.main()
