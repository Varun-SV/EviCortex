"""Contract tests for the public EviCortex facade and context compiler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evicortex import (
    CanonicalClaimExtractor,
    ClaimExtraction,
    ClaimInput,
    ClaimProvenance,
    EmbeddingValidationError,
    EvidenceContext,
    ExtractionDiagnostic,
    ExtractionRecord,
    HashingEmbedder,
    EviCortex,
    SemanticProjectionError,
    memory_embedding_input,
)
from evicortex.budget import BudgetTooSmallError, estimate_tokens
from evicortex.errors import IdempotencyConflictError
from evicortex.models import ClaimHit, ClaimRecord, MemoryRecord
from evicortex.retrieval import DeterministicRetriever


class _ControlledEmbedder:
    """Protocol-conforming synonym embedder with controllable failures."""

    identity = "tests/controlled-semantic-v1"
    normalized = True

    def __init__(
        self,
        *,
        failures: int = 0,
        wrong_dimensions: bool = False,
    ) -> None:
        self.dimensions = 2
        self.failures_remaining = failures
        self.wrong_dimensions = wrong_dimensions
        self.calls: list[tuple[str, ...]] = []

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        frozen = tuple(texts)
        self.calls.append(frozen)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("controlled embedding outage")
        if self.wrong_dimensions:
            return tuple((1.0,) for _ in frozen)
        vectors: list[tuple[float, float]] = []
        for text in frozen:
            normalized = text.casefold()
            if normalized == "zero semantic query":
                vectors.append((0.0, 0.0))
            elif any(
                term in normalized
                for term in ("automobile", "car", "vehicle")
            ):
                vectors.append((1.0, 0.0))
            else:
                vectors.append((0.0, 1.0))
        return tuple(vectors)


class _TiedEmbedder:
    """Real-provider test double whose non-zero vectors all tie exactly."""

    identity = "tests/tied-semantic-v1"
    dimensions = 1
    normalized = True

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0,) for _ in texts)


class _RoleAwareTestEmbedder:
    """Asymmetric provider that fails if legacy batch dispatch is used."""

    identity = "tests/role-aware-semantic-v1"
    dimensions = 2
    normalized = True

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, float], ...]:
        raise AssertionError("legacy symmetric embedding path was used")

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, float], ...]:
        frozen = tuple(texts)
        self.document_calls.append(frozen)
        return tuple((1.0, 0.0) for _ in frozen)

    def embed_query(self, text: str) -> tuple[float, float]:
        self.query_calls.append(text)
        return (1.0, 0.0)


class _IdentityCanonicalExtractor:
    """Canonical parser wrapper with a caller-controlled stable identity."""

    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.calls: list[EvidenceContext] = []

    def extract(self, evidence: EvidenceContext) -> ClaimExtraction:
        self.calls.append(evidence)
        canonical = CanonicalClaimExtractor().extract(evidence)
        return ClaimExtraction(
            claims=canonical.claims,
            provenance=tuple(
                replace(item, extractor_identity=self.identity)
                for item in canonical.provenance
            ),
            diagnostics=canonical.diagnostics,
        )


class _MalformedExtractor:
    identity = "tests/malformed-extractor-v1"

    def extract(self, evidence: EvidenceContext) -> ClaimExtraction:
        canonical = CanonicalClaimExtractor().extract(evidence)
        return ClaimExtraction(
            claims=canonical.claims,
            provenance=tuple(
                replace(
                    item,
                    canonical_text=item.canonical_text + " forged",
                    extractor_identity=self.identity,
                )
                for item in canonical.provenance
            ),
            diagnostics=canonical.diagnostics,
        )


class _DiagnosticExtractor:
    identity = "tests/diagnostic-extractor-v1"

    def __init__(self) -> None:
        self.message = "initial output"

    def extract(self, evidence: EvidenceContext) -> ClaimExtraction:
        del evidence
        return ClaimExtraction(
            diagnostics=(
                ExtractionDiagnostic("controlled_abstention", self.message),
            )
        )


class EviCortexEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        database = Path(self._temporary_directory.name) / "memory.db"
        self.memory = EviCortex.open(database)

    def tearDown(self) -> None:
        self.memory.close()
        self._temporary_directory.cleanup()

    def test_hard_budget_and_clear_impossibly_small_error(self) -> None:
        self.memory.remember(
            namespace="budget/user",
            content="budgetneedle " + ("detailed evidence " * 200),
            source="session/1",
        )

        for budget in (16, 32, 64, 96):
            capsule = self.memory.recall(
                namespace="budget/user",
                query="budgetneedle",
                budget_tokens=budget,
            )
            self.assertLessEqual(estimate_tokens(capsule.to_prompt()), budget)
            self.assertEqual(
                capsule.estimated_tokens,
                estimate_tokens(capsule.to_prompt()),
            )

        with self.assertRaises(BudgetTooSmallError) as raised:
            self.memory.recall(
                namespace="budget/user",
                query="budgetneedle",
                budget_tokens=1,
            )
        self.assertGreater(raised.exception.minimum_tokens, 1)
        self.assertIn("at least", str(raised.exception))

    def test_recall_is_deterministic_and_every_candidate_has_a_trace(self) -> None:
        self.memory.remember(
            namespace="determinism/user",
            content="The preferred interface theme is dark.",
            source="session/1",
            occurred_at="2026-04-18T09:30:00Z",
            claims=(
                ClaimInput(
                    subject="user:determinism",
                    predicate="prefers_theme",
                    value="dark",
                    valid_from="2026-04-18T09:30:00Z",
                ),
            ),
        )

        first = self.memory.recall(
            namespace="determinism/user",
            query="preferred theme",
            budget_tokens=1024,
        )
        second = self.memory.recall(
            namespace="determinism/user",
            query="preferred theme",
            budget_tokens=1024,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.to_prompt(), second.to_prompt())
        self.assertTrue(first.trace)
        self.assertTrue(any(item.selected for item in first.trace))
        for item in first.trace:
            self.assertTrue(item.reasons)
            self.assertTrue(all(reason.channel for reason in item.reasons))

    def test_namespace_isolation_applies_before_relevance(self) -> None:
        alpha = self.memory.remember(
            namespace="tenant/alpha",
            content="The favorite color for Alpha is amber.",
            claims=(
                ClaimInput("user:alpha", "favorite_color", "amber"),
            ),
        )
        beta = self.memory.remember(
            namespace="tenant/beta",
            content="The favorite color for Beta is cobalt.",
            claims=(
                ClaimInput("user:beta", "favorite_color", "cobalt"),
            ),
        )

        capsule = self.memory.recall(
            namespace="tenant/alpha",
            query="favorite color",
            budget_tokens=1024,
        )
        self.assertIn(alpha.id, capsule.evidence_ids)
        self.assertNotIn(beta.id, capsule.evidence_ids)
        self.assertNotIn("cobalt", capsule.to_prompt())

        with self.assertRaises(KeyError):
            self.memory.expand(namespace="tenant/alpha", memory_id=beta.id)

    def test_current_and_historical_claim_resolution(self) -> None:
        old = self.memory.remember(
            namespace="timeline/user",
            content="The user preferred the light theme.",
            occurred_at="2026-01-02T08:00:00Z",
            claims=(
                ClaimInput(
                    "user:timeline",
                    "prefers_theme",
                    "light",
                    valid_from="2026-01-02T08:00:00Z",
                ),
            ),
        )
        old_claim_id = old.claims[0].id
        new = self.memory.remember(
            namespace="timeline/user",
            content="The user switched to the dark theme.",
            occurred_at="2026-04-18T09:30:00Z",
            claims=(
                ClaimInput(
                    "user:timeline",
                    "prefers_theme",
                    "dark",
                    valid_from="2026-04-18T09:30:00Z",
                    supersedes=old_claim_id,
                ),
            ),
        )

        current = self.memory.recall(
            namespace="timeline/user",
            query="theme",
            budget_tokens=1024,
        )
        self.assertIn("dark", {fact.value for fact in current.facts})
        self.assertNotIn("light", {fact.value for fact in current.facts})
        self.assertIn(new.id, current.evidence_ids)
        self.assertNotIn(old.id, current.evidence_ids)
        self.assertFalse(any(item.memory_id == old.id for item in current.episodes))

        historical = self.memory.recall(
            namespace="timeline/user",
            query="theme",
            as_of="2026-02-01T00:00:00Z",
            budget_tokens=1024,
        )
        self.assertIn("light", {fact.value for fact in historical.facts})
        self.assertNotIn("dark", {fact.value for fact in historical.facts})
        self.assertIn(old.id, historical.evidence_ids)
        self.assertNotIn(new.id, historical.evidence_ids)
        self.assertFalse(
            any(item.memory_id == new.id for item in historical.episodes)
        )

    def test_future_successor_has_no_current_gap_and_finite_end_does_not_reopen(
        self,
    ) -> None:
        now = datetime.now(timezone.utc)
        predecessor_start = now - timedelta(days=2)
        boundary = now + timedelta(days=1)
        successor_end = now + timedelta(days=2)
        predecessor = self.memory.remember(
            namespace="future-successor/user",
            content="The support schedule uses weekday coverage.",
            occurred_at=predecessor_start,
            claims=(
                ClaimInput(
                    "support:schedule",
                    "coverage",
                    "weekday",
                    valid_from=predecessor_start,
                ),
            ),
        )
        successor = self.memory.remember(
            namespace="future-successor/user",
            content="The support schedule uses summer coverage.",
            occurred_at=now,
            claims=(
                ClaimInput(
                    "support:schedule",
                    "coverage",
                    "summer",
                    valid_from=boundary,
                    valid_to=successor_end,
                    supersedes=predecessor.claims[0].id,
                ),
            ),
        )

        def recall(at: datetime | None):
            return self.memory.recall(
                namespace="future-successor/user",
                query="support schedule coverage",
                budget_tokens=1024,
                as_of=at,
            )

        current = recall(None)
        just_before = recall(boundary - timedelta(microseconds=1))
        at_boundary = recall(boundary)
        during_successor = recall(boundary + timedelta(microseconds=1))
        at_successor_end = recall(successor_end)
        after_successor = recall(successor_end + timedelta(days=1))

        for capsule in (current, just_before):
            self.assertEqual({fact.value for fact in capsule.facts}, {"weekday"})
            self.assertEqual(capsule.facts[0].status, "active")
            self.assertEqual(capsule.evidence_ids, (predecessor.id,))
            serialized = json.loads(capsule.to_prompt())
            self.assertNotIn("status", serialized["facts"][0])
        for capsule in (at_boundary, during_successor):
            self.assertEqual({fact.value for fact in capsule.facts}, {"summer"})
            self.assertEqual(capsule.evidence_ids, (successor.id,))
        for capsule in (at_successor_end, after_successor):
            self.assertEqual(capsule.facts, ())
            self.assertEqual(capsule.episodes, ())
            self.assertEqual(capsule.evidence_ids, ())

    def test_multi_claim_episode_cannot_leak_future_value_as_raw_text(self) -> None:
        future_value = "FUTURE-LEAK-991"
        record = self.memory.remember(
            namespace="mixed-validity/user",
            content=(
                "The mixed display_mode is concise; the future_access_code will be "
                f"{future_value}."
            ),
            occurred_at="2025-01-01T00:00:00Z",
            claims=(
                ClaimInput(
                    "user:mixed-validity",
                    "display_mode",
                    "concise",
                    valid_from="2025-01-01T00:00:00Z",
                ),
                ClaimInput(
                    "user:mixed-validity",
                    "future_access_code",
                    future_value,
                    valid_from="2099-01-01T00:00:00Z",
                ),
            ),
        )

        recalls = (
            self.memory.recall(
                namespace="mixed-validity/user",
                query="mixed display_mode future_access_code",
                budget_tokens=1024,
            ),
            self.memory.recall(
                namespace="mixed-validity/user",
                query="mixed display_mode future_access_code",
                as_of="2026-01-01T00:00:00Z",
                budget_tokens=1024,
            ),
        )

        for capsule in recalls:
            values = {fact.value for fact in capsule.facts}
            self.assertIn("concise", values)
            self.assertNotIn(future_value, values)
            self.assertTrue(
                any(
                    record.id in fact.evidence_ids and fact.value == "concise"
                    for fact in capsule.facts
                )
            )
            self.assertFalse(
                any(episode.memory_id == record.id for episode in capsule.episodes)
            )
            self.assertNotIn(future_value, capsule.to_prompt())

        # Exact raw evidence remains available only through deliberate expansion.
        expanded = self.memory.expand(
            namespace="mixed-validity/user",
            memory_id=record.id,
        )
        self.assertIn(future_value, expanded.content)

    def test_historical_multi_hop_graph_uses_requested_time(self) -> None:
        self.memory.remember(
            namespace="historical-graph/user",
            content="Project Apollo was managed by person:alice.",
            occurred_at="2025-01-01T00:00:00Z",
            claims=(
                ClaimInput(
                    "project:apollo",
                    "managed_by",
                    "person:alice",
                    valid_from="2025-01-01T00:00:00Z",
                ),
            ),
        )
        old_city = self.memory.remember(
            namespace="historical-graph/user",
            content="Alice lived in Pune.",
            occurred_at="2025-01-02T00:00:00Z",
            claims=(
                ClaimInput(
                    "person:alice",
                    "home_city",
                    "Pune",
                    valid_from="2025-01-02T00:00:00Z",
                ),
            ),
        )
        self.memory.remember(
            namespace="historical-graph/user",
            content="Alice will move to Chennai.",
            occurred_at="2027-01-01T00:00:00Z",
            claims=(
                ClaimInput(
                    "person:alice",
                    "home_city",
                    "Chennai",
                    valid_from="2027-01-01T00:00:00Z",
                    supersedes=old_city.claims[0].id,
                ),
            ),
        )

        capsule = self.memory.recall(
            namespace="historical-graph/user",
            query="Who managed project Apollo and where did they live?",
            as_of="2026-01-01T00:00:00Z",
            budget_tokens=1024,
        )

        values = {fact.value for fact in capsule.facts}
        self.assertIn("person:alice", values)
        self.assertIn("Pune", values)
        self.assertNotIn("Chennai", values)
        old_city_trace = next(
            item
            for item in capsule.trace
            if item.candidate_id == old_city.claims[0].id
        )
        self.assertIn("graph", {reason.channel for reason in old_city_trace.reasons})

    def test_fact_episode_deduplication_and_progressive_exact_expansion(self) -> None:
        exact_content = "The launch codename is Aurora; retain punctuation exactly."
        record = self.memory.remember(
            namespace="expansion/user",
            content=exact_content,
            source="tool://source/7",
            claims=(
                ClaimInput("project:launch", "codename", "Aurora"),
            ),
        )

        capsule = self.memory.recall(
            namespace="expansion/user",
            query="launch codename Aurora",
            budget_tokens=1024,
        )
        self.assertIn(record.id, capsule.evidence_ids)
        self.assertTrue(any(record.id in fact.evidence_ids for fact in capsule.facts))
        self.assertFalse(any(item.memory_id == record.id for item in capsule.episodes))

        prompt = json.loads(capsule.to_prompt())
        self.assertEqual(prompt["v"], "ec1")
        self.assertEqual(prompt["facts"][0]["value"], "Aurora")
        self.assertIn(record.id, prompt["facts"][0]["evidence_ids"])
        self.assertNotIn("claim_id", prompt["facts"][0])
        self.assertNotIn("status", prompt["facts"][0])
        self.assertNotIn("confidence", prompt["facts"][0])
        self.assertNotIn("episodes", prompt)
        self.assertNotIn("as_of", prompt)

        expanded = self.memory.expand(
            namespace="expansion/user",
            memory_id=record.id,
        )
        self.assertEqual(expanded.content, exact_content)
        self.assertEqual(expanded.content_hash, record.content_hash)
        self.assertEqual(expanded.source, "tool://source/7")
        self.assertEqual(expanded.id, prompt["facts"][0]["evidence_ids"][0])

    def test_context_manager_integrity_rebuild_and_purge_wrappers(self) -> None:
        database = Path(self._temporary_directory.name) / "managed.db"
        with EviCortex.open(database) as managed:
            record = managed.remember(namespace="managed/user", content="managedneedle")
            self.assertTrue(managed.verify_integrity().ok)
            managed.rebuild_indexes()
            self.assertTrue(managed.verify_integrity("managed/user").ok)
            self.assertTrue(
                managed.purge_memory(namespace="managed/user", memory_id=record.id)
            )
            with self.assertRaises(KeyError):
                managed.expand(namespace="managed/user", memory_id=record.id)


class EviCortexExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _path(self, name: str) -> Path:
        return Path(self._temporary_directory.name) / name

    def test_multi_claim_extraction_maps_spans_to_generated_ids_after_reopen(
        self,
    ) -> None:
        database = self._path("extraction-durable.db")
        extractor = CanonicalClaimExtractor()
        content = (
            "Evidence preface. "
            "Canonical claim: subject=project:atlas; predicate=started_on; "
            "value=2026-01-10; confidence=0.750; authority=calendar. "
            "Canonical claim: subject=project:atlas; predicate=released_on; "
            "value=2026-03-05; confidence=1.000; authority=release-system."
        )
        with EviCortex.open(database, extractor=extractor) as memory:
            self.assertEqual(memory.extractor_identity, extractor.identity)
            record = memory.remember_extracted(
                namespace=" tenant/a ",
                content=content,
                source="calendar://atlas",
                occurred_at="2026-03-06T00:00:00+00:00",
                metadata={"batch": 7},
            )
            run = memory.get_extraction(
                namespace="tenant/a",
                memory_id=record.id,
            )

        self.assertIsInstance(run, ExtractionRecord)
        self.assertFalse(run.abstained)
        self.assertEqual(set(run.claim_ids), {claim.id for claim in record.claims})
        self.assertEqual(
            [item.claim_index for item in run.derivations],
            [0, 1],
        )
        claims_by_id = {claim.id: claim for claim in record.claims}
        self.assertEqual(
            [claims_by_id[item.claim_id].predicate for item in run.derivations],
            ["started_on", "released_on"],
        )
        for item in run.derivations:
            self.assertEqual(content[item.start : item.end], item.canonical_text)
            self.assertEqual(item.content_sha256, record.content_hash)
            self.assertEqual(item.extractor_identity, extractor.identity)
        self.assertTrue(
            all(
                isinstance(item, ClaimProvenance)
                for item in extractor.extract(
                    EvidenceContext.from_content(
                        namespace="tenant/a",
                        content=content,
                    )
                ).provenance
            )
        )

        with EviCortex.open(database) as reopened:
            self.assertEqual(
                reopened.get_extraction("tenant/a", record.id),
                run,
            )
            self.assertEqual(
                reopened.expand(namespace="tenant/a", memory_id=record.id).content,
                content,
            )

    def test_abstention_and_namespace_scoped_lookup_are_durable(self) -> None:
        with EviCortex.open(
            self._path("extraction-abstention.db"),
            extractor=CanonicalClaimExtractor(),
        ) as memory:
            record = memory.remember_extracted(
                namespace="tenant/a",
                content="No canonical assertion is present.",
            )
            run = memory.get_extraction("tenant/a", record.id)
            self.assertTrue(run.abstained)
            self.assertEqual(record.claims, ())
            self.assertEqual(run.derivations, ())
            self.assertEqual(run.diagnostics[0].code, "no_canonical_claims")
            with self.assertRaises(KeyError) as wrong_namespace:
                memory.get_extraction("tenant/b", record.id)
            with self.assertRaises(KeyError) as missing:
                memory.get_extraction("tenant/a", "mem_missing")
            self.assertEqual(str(wrong_namespace.exception), str(missing.exception).replace(
                "mem_missing", record.id
            ))

    def test_supersedes_source_uses_only_the_explicit_resolution_snapshot(
        self,
    ) -> None:
        old_source = "synthetic/theme-old"
        content = (
            "Canonical claim: subject=user:ada; predicate=preferred_theme; "
            "value=light; valid_from=2026-02-01T00:00:00Z; "
            f"supersedes_source={old_source}; confidence=1.000; "
            "authority=application."
        )
        extractor = _IdentityCanonicalExtractor("tests/resolution-extractor-v1")
        with EviCortex.open(
            self._path("extraction-supersedes.db"),
            extractor=extractor,
        ) as memory:
            predecessor = memory.remember(
                namespace="tenant/a",
                content="Ada preferred dark.",
                occurred_at="2026-01-01T00:00:00Z",
                claims=(
                    ClaimInput(
                        "user:ada",
                        "preferred_theme",
                        "dark",
                        valid_from="2026-01-01T00:00:00Z",
                    ),
                ),
            )
            replacement = memory.remember_extracted(
                namespace="tenant/a",
                content=content,
                claim_ids_by_source={old_source: predecessor.claims[0].id},
            )

            self.assertEqual(
                replacement.claims[0].supersedes,
                predecessor.claims[0].id,
            )
            self.assertEqual(
                extractor.calls[0].claim_ids_by_source[old_source],
                predecessor.claims[0].id,
            )
            self.assertEqual(
                memory.expand(
                    namespace="tenant/a",
                    memory_id=predecessor.id,
                ).claims[0].status,
                "superseded",
            )

    def test_malformed_or_unstable_extractor_fails_before_evidence_write(
        self,
    ) -> None:
        content = (
            "Canonical claim: subject=device:q; predicate=state; value=ready; "
            "confidence=1.000; authority=registry."
        )
        with EviCortex.open(
            self._path("extraction-malformed.db"),
            extractor=_MalformedExtractor(),
        ) as memory:
            with self.assertRaisesRegex(ValueError, "exact evidence substring"):
                memory.remember_extracted(namespace="n", content=content)
            self.assertEqual(memory.stats().memories, 0)

        mutable = _IdentityCanonicalExtractor("tests/mutable-v1")
        with EviCortex.open(
            self._path("extraction-mutable.db"),
            extractor=mutable,
        ) as memory:
            mutable.identity = "tests/mutable-v2"
            with self.assertRaisesRegex(ValueError, "changed"):
                memory.remember_extracted(namespace="n", content=content)
            self.assertEqual(memory.stats().memories, 0)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            EviCortex.open(
                self._path("extraction-empty-identity.db"),
                extractor=_IdentityCanonicalExtractor(""),
            )

    def test_configured_and_explicit_override_identity_behavior(self) -> None:
        content = (
            "Canonical claim: subject=device:q; predicate=state; value=ready; "
            "confidence=1.000; authority=registry."
        )
        override = _IdentityCanonicalExtractor("tests/override-v1")
        with EviCortex.open(self._path("extraction-override.db")) as memory:
            self.assertIsNone(memory.extractor_identity)
            with self.assertRaisesRegex(ValueError, "not configured"):
                memory.remember_extracted(namespace="n", content=content)
            record = memory.remember_extracted(
                namespace="n",
                content=content,
                extractor=override,
            )
            self.assertEqual(
                memory.get_extraction("n", record.id).extractor_identity,
                override.identity,
            )

        configured = CanonicalClaimExtractor()
        with EviCortex.open(
            self._path("extraction-configured-override.db"),
            extractor=configured,
        ) as memory:
            record = memory.remember_extracted(
                namespace="n",
                content=content,
                extractor=override,
            )
            self.assertEqual(memory.extractor_identity, configured.identity)
            self.assertEqual(
                memory.get_extraction("n", record.id).extractor_identity,
                override.identity,
            )

    def test_extracted_idempotent_replay_and_output_conflict(self) -> None:
        extractor = _DiagnosticExtractor()
        with EviCortex.open(
            self._path("extraction-idempotency.db"),
            extractor=extractor,
        ) as memory:
            first = memory.remember_extracted(
                namespace="n",
                content="Evidence with a controlled abstention.",
                idempotency_key="extract-1",
            )
            replay = memory.remember_extracted(
                namespace="n",
                content="Evidence with a controlled abstention.",
                idempotency_key="extract-1",
            )
            self.assertEqual(replay.id, first.id)
            self.assertEqual(memory.stats().memories, 1)

            extractor.message = "changed output"
            with self.assertRaises(IdempotencyConflictError):
                memory.remember_extracted(
                    namespace="n",
                    content="Evidence with a controlled abstention.",
                    idempotency_key="extract-1",
                )
            self.assertEqual(memory.stats().memories, 1)
            self.assertEqual(
                memory.get_extraction("n", first.id).diagnostics[0].message,
                "initial output",
            )

    def test_extraction_and_semantic_auto_index_commit_together(self) -> None:
        embedder = _ControlledEmbedder()
        content = (
            "Canonical claim: subject=asset:one; predicate=category; "
            "value=automobile; confidence=1.000; authority=registry."
        )
        with EviCortex.open(
            self._path("extraction-semantic.db"),
            extractor=CanonicalClaimExtractor(),
            embedder=embedder,
        ) as memory:
            record = memory.remember_extracted(
                namespace="semantic/user",
                content=content,
                idempotency_key="extract-semantic-1",
            )
            calls_after_commit = len(embedder.calls)
            replay = memory.remember_extracted(
                namespace="semantic/user",
                content=content,
                idempotency_key="extract-semantic-1",
            )

            self.assertEqual(replay.id, record.id)
            self.assertTrue(memory.semantic_status("semantic/user").complete)
            self.assertEqual(len(embedder.calls), calls_after_commit)
            self.assertEqual(
                memory.get_extraction("semantic/user", record.id).claim_ids,
                (record.claims[0].id,),
            )


class EviCortexSemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _path(self, name: str) -> Path:
        return Path(self._temporary_directory.name) / name

    def test_role_aware_provider_separates_index_and_query_encoding(self) -> None:
        embedder = _RoleAwareTestEmbedder()
        with EviCortex.open(
            self._path("role-aware.db"),
            embedder=embedder,
        ) as memory:
            record = memory.remember(
                namespace="semantic/user",
                content="Opaque evidence payload.",
                claims=(ClaimInput("asset:one", "category", "coded-value"),),
            )
            capsule = memory.recall(
                namespace="semantic/user",
                query="semantically related paraphrase",
                budget_tokens=512,
            )

        self.assertIn(record.id, capsule.evidence_ids)
        self.assertEqual(len(embedder.document_calls), 1)
        self.assertEqual(embedder.query_calls, ["semantically related paraphrase"])

    def test_semantic_only_claim_match_fuses_with_trace(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-only.db"),
            embedder=embedder,
        ) as memory:
            record = memory.remember(
                namespace="semantic/user",
                content="Opaque evidence payload.",
                claims=(
                    ClaimInput("asset:one", "category", "automobile"),
                ),
            )
            capsule = memory.recall(
                namespace="semantic/user",
                query="car",
                budget_tokens=512,
            )
            repeated = memory.recall(
                namespace="semantic/user",
                query="car",
                budget_tokens=512,
            )

        self.assertIn(record.id, capsule.evidence_ids)
        self.assertEqual(capsule, repeated)
        self.assertLessEqual(capsule.estimated_tokens, 512)
        self.assertIn("automobile", {fact.value for fact in capsule.facts})
        claim_trace = next(
            item
            for item in capsule.trace
            if item.candidate_id == record.claims[0].id
        )
        self.assertIn(
            "claim_semantic",
            {reason.channel for reason in claim_trace.reasons},
        )
        self.assertEqual(embedder.calls.count(("car",)), 2)

    def test_diagnostic_hash_collision_cannot_answer_unknown_query(self) -> None:
        embedder = HashingEmbedder(dimensions=16)
        unknown_query = "What is Zephyr-999's xylophonic_allergy?"
        query_vector = embedder.embed_batch((unknown_query,))[0]
        collision_token = ""
        collision_content = ""
        for index in range(1000):
            token = f"unrelatedtoken{index}"
            content = f"A known device status is {token}."
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            projected_memory = MemoryRecord(
                id="mem_probe",
                namespace="collision/user",
                kind="episode",
                content=content,
                content_hash=digest,
                source="",
                occurred_at="2026-01-01T00:00:00.000000Z",
                recorded_at="2026-01-01T00:00:00.000000Z",
                metadata={},
                idempotency_key=None,
            )
            projected_vector = embedder.embed_batch(
                (memory_embedding_input(projected_memory).text,)
            )[0]
            if (
                sum(
                    query_value * target_value
                    for query_value, target_value in zip(
                        query_vector,
                        projected_vector,
                        strict=True,
                    )
                )
                >= 0.05
            ):
                collision_token = token
                collision_content = content
                break
        self.assertTrue(collision_token)
        with EviCortex.open(
            self._path("semantic-diagnostic-collision.db"),
            embedder=embedder,
        ) as memory:
            memory.remember(
                namespace="collision/user",
                content=collision_content,
                claims=(
                    ClaimInput(
                        "asset:collision",
                        "opaque_token",
                        collision_token,
                    ),
                ),
            )
            unknown = memory.recall(
                namespace="collision/user",
                query=unknown_query,
                budget_tokens=512,
            )
            exact = memory.recall(
                namespace="collision/user",
                query=collision_token,
                budget_tokens=512,
            )

        self.assertEqual(unknown.evidence_ids, ())
        diagnostic = next(
            item
            for item in unknown.diagnostics
            if item.code == "semantic_diagnostic_token_overlap_filtered"
        )
        self.assertIn("memories=1", diagnostic.detail)
        self.assertNotIn("diagnostic", unknown.to_prompt())
        self.assertTrue(exact.evidence_ids)

    def test_negated_accessory_flood_never_displaces_supported_device(
        self,
    ) -> None:
        namespace = "growth/user"
        target_key = "target"
        distractor_keys = tuple(f"accessory-{index:03d}" for index in range(220))
        insertion_orders = (
            (target_key, *distractor_keys),
            (*reversed(distractor_keys), target_key),
        )
        selected_sources: list[tuple[str, ...]] = []

        for run_index, insertion_order in enumerate(insertion_orders):
            with EviCortex.open(
                self._path(f"semantic-negation-{run_index}.db"),
                embedder=_TiedEmbedder(),
                auto_index=False,
                semantic_scan_cap=512,
            ) as memory:
                source_by_memory_id: dict[str, str] = {}
                source_by_candidate_id: dict[str, str] = {}
                for key in insertion_order:
                    if key == target_key:
                        source = "growth/target"
                        record = memory.remember(
                            namespace=namespace,
                            source=source,
                            occurred_at="2026-01-16T09:00:00Z",
                            content="The Quartz device has serial number QZ-4401.",
                            claims=(
                                ClaimInput(
                                    "device:quartz",
                                    "serial_number",
                                    "QZ-4401",
                                ),
                            ),
                        )
                    else:
                        index = int(key.rsplit("-", 1)[1])
                        source = f"growth/accessory/{index:03d}"
                        record = memory.remember(
                            namespace=namespace,
                            source=source,
                            occurred_at="2026-04-01T00:00:00Z",
                            content=(
                                f"Quartz accessory {index} has serial number "
                                f"QA-{index:06d}."
                            ),
                            claims=(
                                ClaimInput(
                                    f"device:quartz_accessory_{index:06d}",
                                    "serial_number",
                                    f"QA-{index:06d}",
                                ),
                            ),
                        )
                    source_by_memory_id[record.id] = source
                    source_by_candidate_id[record.id] = source
                    for claim in record.claims:
                        source_by_candidate_id[claim.id] = source

                self.assertTrue(
                    memory.sync_semantic(namespace, batch_size=512).complete
                )
                capsule = memory.recall(
                    namespace=namespace,
                    query=(
                        "What is the serial_number for the Quartz device, "
                        "not an accessory?"
                    ),
                    budget_tokens=512,
                    limit=40,
                )
                sources = tuple(
                    source_by_memory_id[memory_id]
                    for memory_id in capsule.evidence_ids
                )
                selected_sources.append(sources)
                self.assertEqual(sources, ("growth/target",))
                self.assertIn("QZ-4401", capsule.to_prompt())
                self.assertNotIn("QA-", capsule.to_prompt())
                self.assertEqual(
                    {
                        source_by_candidate_id[trace.candidate_id]
                        for trace in capsule.trace
                    },
                    {"growth/target"},
                )
                diagnostic = next(
                    item
                    for item in capsule.diagnostics
                    if item.code == "semantic_excluded_term_filtered"
                )
                self.assertIn("memories=220", diagnostic.detail)
                self.assertIn("claims=220", diagnostic.detail)
                self.assertIn("terms=accessory", diagnostic.detail)

        self.assertEqual(selected_sources[0], selected_sources[1])

    def test_exclusions_cover_claim_authority_and_memory_kind_source(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-excluded-fields.db"),
            embedder=embedder,
        ) as memory:
            target = memory.remember(
                namespace="excluded-fields/user",
                content="A car record belongs to the registry.",
                kind="episode",
                source="registry",
                claims=(
                    ClaimInput(
                        "asset:one",
                        "category",
                        "car",
                        authority="registry",
                    ),
                ),
            )
            excluded_authority = memory.remember(
                namespace="excluded-fields/user",
                content="Opaque evidence for a related asset.",
                kind="episode",
                source="registry",
                claims=(
                    ClaimInput(
                        "asset:one",
                        "category",
                        "automobile",
                        authority="calendar",
                    ),
                ),
            )
            excluded_source = memory.remember(
                namespace="excluded-fields/user",
                content="An automobile source-only record.",
                kind="episode",
                source="calendar",
            )
            excluded_kind = memory.remember(
                namespace="excluded-fields/user",
                content="A vehicle kind-only record.",
                kind="calendar",
                source="registry",
            )

            capsule = memory.recall(
                namespace="excluded-fields/user",
                query="car not calendar",
                budget_tokens=512,
            )

        self.assertEqual(capsule.evidence_ids, (target.id,))
        self.assertNotIn(excluded_authority.id, capsule.evidence_ids)
        self.assertNotIn(excluded_source.id, capsule.evidence_ids)
        self.assertNotIn(excluded_kind.id, capsule.evidence_ids)
        excluded_candidate_ids = {
            excluded_authority.id,
            excluded_authority.claims[0].id,
            excluded_source.id,
            excluded_kind.id,
        }
        self.assertFalse(
            any(
                trace.candidate_id in excluded_candidate_ids
                for trace in capsule.trace
            )
        )
        diagnostic = next(
            item
            for item in capsule.diagnostics
            if item.code == "semantic_excluded_term_filtered"
        )
        self.assertIn("memories=2", diagnostic.detail)
        self.assertIn("claims=1", diagnostic.detail)
        self.assertNotIn("calendar", capsule.to_prompt().casefold())

    def test_explicit_index_semantic_repairs_one_record(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-explicit.db"),
            embedder=embedder,
            auto_index=False,
        ) as memory:
            record = memory.remember(
                namespace="explicit/user",
                content="An automobile was recorded.",
            )
            self.assertEqual(memory.semantic_status("explicit/user").missing_count, 1)
            result = memory.index_semantic(record)
            self.assertEqual(result.written, 1)
            self.assertTrue(memory.semantic_status("explicit/user").complete)

    def test_index_semantic_rejects_uncommitted_or_forged_records(self) -> None:
        embedder = _ControlledEmbedder()
        with (
            EviCortex.open(
                self._path("semantic-owner-a.db"),
                embedder=embedder,
            ) as owner,
            EviCortex.open(
                self._path("semantic-owner-b.db"),
                embedder=embedder,
            ) as other,
        ):
            record = owner.remember(
                namespace="owner/user",
                content="An automobile owned by store A.",
            )
            calls_after_commit = len(embedder.calls)

            with self.assertRaises(SemanticProjectionError) as cross_store:
                other.index_semantic(record)
            self.assertFalse(cross_store.exception.evidence_committed)

            forged_content = "Forged replacement content."
            forged = replace(
                record,
                content=forged_content,
                content_hash=hashlib.sha256(
                    forged_content.encode("utf-8")
                ).hexdigest(),
            )
            with self.assertRaises(SemanticProjectionError) as forged_error:
                owner.index_semantic(forged)
            self.assertFalse(forged_error.exception.evidence_committed)

            self.assertTrue(
                owner.purge_memory(
                    namespace=record.namespace,
                    memory_id=record.id,
                )
            )
            with self.assertRaises(SemanticProjectionError) as purged_error:
                owner.index_semantic(record)
            self.assertFalse(purged_error.exception.evidence_committed)
            self.assertEqual(len(embedder.calls), calls_after_commit)

    def test_empty_namespace_is_vacuously_complete_but_unregistered(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-empty.db"),
            embedder=embedder,
            auto_index=False,
        ) as memory:
            empty = memory.semantic_status("empty/user")
            self.assertTrue(empty.complete)
            self.assertFalse(empty.registered)
            synced_empty = memory.sync_semantic("empty/user")
            self.assertTrue(synced_empty.complete)
            self.assertFalse(synced_empty.registered)
            self.assertEqual(embedder.calls, [])

            other = memory.remember(
                namespace="populated/user",
                content="An automobile in another namespace.",
            )
            memory.index_semantic(other)
            still_empty = memory.semantic_status("empty/user")
            self.assertTrue(still_empty.complete)
            self.assertFalse(still_empty.registered)

    def test_semantic_namespace_and_historical_time_precede_ranking(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-scope.db"),
            embedder=embedder,
        ) as memory:
            old = memory.remember(
                namespace="tenant/a",
                content="Opaque old record.",
                occurred_at="2025-01-01T00:00:00Z",
                claims=(
                    ClaimInput(
                        "asset:old",
                        "category",
                        "automobile",
                        valid_from="2025-01-01T00:00:00Z",
                    ),
                ),
            )
            future = memory.remember(
                namespace="tenant/a",
                content="Opaque future record.",
                occurred_at="2027-01-01T00:00:00Z",
                claims=(
                    ClaimInput(
                        "asset:future",
                        "category",
                        "automobile",
                        valid_from="2027-01-01T00:00:00Z",
                    ),
                ),
            )
            other = memory.remember(
                namespace="tenant/b",
                content="Opaque private record.",
                occurred_at="2025-01-01T00:00:00Z",
                claims=(
                    ClaimInput(
                        "asset:private",
                        "category",
                        "automobile",
                        valid_from="2025-01-01T00:00:00Z",
                    ),
                ),
            )
            capsule = memory.recall(
                namespace="tenant/a",
                query="car",
                as_of="2026-01-01T00:00:00Z",
                budget_tokens=512,
            )

        self.assertIn(old.id, capsule.evidence_ids)
        self.assertNotIn(future.id, capsule.evidence_ids)
        self.assertNotIn(other.id, capsule.evidence_ids)

    def test_missing_profile_sync_and_empty_namespace_diagnostics(self) -> None:
        path = self._path("semantic-sync.db")
        with EviCortex.open(path) as lexical:
            record = lexical.remember(
                namespace="sync/user",
                content="An automobile is stored here.",
            )

        embedder = _ControlledEmbedder()
        with EviCortex.open(
            path,
            embedder=embedder,
            auto_index=False,
        ) as memory:
            before = memory.recall(
                namespace="sync/user",
                query="car",
                budget_tokens=512,
            )
            self.assertNotIn(record.id, before.evidence_ids)
            self.assertIn(
                "semantic_profile_missing",
                {item.code for item in before.diagnostics},
            )
            lexical_fallback = memory.recall(
                namespace="sync/user",
                query="automobile",
                budget_tokens=512,
            )
            self.assertIn(record.id, lexical_fallback.evidence_ids)
            self.assertIn(
                "semantic_profile_missing",
                {item.code for item in lexical_fallback.diagnostics},
            )
            status = memory.semantic_status("sync/user")
            self.assertFalse(status.registered)
            self.assertEqual(status.missing_count, 1)

            repaired = memory.sync_semantic("sync/user", batch_size=1)
            self.assertTrue(repaired.complete)
            after = memory.recall(
                namespace="sync/user",
                query="car",
                budget_tokens=512,
            )
            self.assertIn(record.id, after.evidence_ids)

            empty = memory.recall(
                namespace="sync/empty",
                query="car",
                budget_tokens=512,
            )
            codes = {item.code for item in empty.diagnostics}
            self.assertIn("semantic_memory_no_vectors", codes)
            self.assertIn("semantic_claim_no_vectors", codes)

    def test_rebuild_indexes_never_hides_namespace_scoped_semantic_repair(
        self,
    ) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-explicit-rebuild.db"),
            embedder=embedder,
            auto_index=False,
        ) as memory:
            first = memory.remember(
                namespace="tenant/a",
                content="An automobile belongs to tenant A.",
            )
            memory.remember(
                namespace="tenant/b",
                content="A vehicle belongs to tenant B.",
            )
            before_a = memory.semantic_status("tenant/a")
            before_b = memory.semantic_status("tenant/b")
            self.assertEqual(before_a.memories.missing_ids, (first.id,))
            self.assertEqual(before_b.missing_count, 1)
            self.assertEqual(embedder.calls, [])

            memory.rebuild_indexes()

            rebuilt_a = memory.semantic_status("tenant/a")
            rebuilt_b = memory.semantic_status("tenant/b")
            self.assertEqual(rebuilt_a.memories.missing_ids, (first.id,))
            self.assertEqual(rebuilt_b.missing_count, 1)
            self.assertEqual(embedder.calls, [])

            repaired_a = memory.sync_semantic("tenant/a")
            self.assertTrue(repaired_a.complete)
            self.assertEqual(len(embedder.calls), 1)
            still_missing_b = memory.semantic_status("tenant/b")
            self.assertEqual(still_missing_b.missing_count, 1)
            self.assertFalse(still_missing_b.complete)

    def test_auto_index_failure_commits_evidence_and_idempotent_retry_repairs(self) -> None:
        embedder = _ControlledEmbedder(failures=1)
        with EviCortex.open(
            self._path("semantic-repair.db"),
            embedder=embedder,
        ) as memory:
            with self.assertRaises(SemanticProjectionError) as raised:
                memory.remember(
                    namespace="repair/user",
                    content="The stored object is an automobile.",
                    idempotency_key="repair-1",
                )
            self.assertTrue(raised.exception.evidence_committed)
            committed = memory.expand(
                namespace="repair/user",
                memory_id=raised.exception.memory_id,
            )

            retried = memory.remember(
                namespace="repair/user",
                content="The stored object is an automobile.",
                idempotency_key="repair-1",
            )
            self.assertEqual(retried.id, committed.id)
            self.assertEqual(len(embedder.calls), 2)
            self.assertTrue(memory.semantic_status("repair/user").complete)
            capsule = memory.recall(
                namespace="repair/user",
                query="car",
                budget_tokens=512,
            )
            self.assertIn(committed.id, capsule.evidence_ids)

    def test_clean_idempotent_replay_skips_provider_and_upsert(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-current.db"),
            embedder=embedder,
        ) as memory:
            first = memory.remember(
                namespace="current/user",
                content="An automobile projection is current.",
                idempotency_key="current-1",
            )
            self.assertEqual(len(embedder.calls), 1)
            embedder.failures_remaining = 1

            replay = memory.remember(
                namespace="current/user",
                content="An automobile projection is current.",
                idempotency_key="current-1",
            )
            self.assertEqual(replay.id, first.id)
            self.assertEqual(len(embedder.calls), 1)
            self.assertEqual(embedder.failures_remaining, 1)

            result = memory.index_semantic(first)
            self.assertEqual(result.written, 0)
            self.assertEqual(len(embedder.calls), 1)
            self.assertEqual(embedder.failures_remaining, 1)

    def test_scan_cap_discards_partial_semantic_channel(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-cap.db"),
            embedder=embedder,
            semantic_scan_cap=1,
        ) as memory:
            first = memory.remember(
                namespace="cap/user",
                content="First automobile evidence.",
            )
            second = memory.remember(
                namespace="cap/user",
                content="Second automobile evidence.",
            )
            capsule = memory.recall(
                namespace="cap/user",
                query="car",
                budget_tokens=512,
            )

        self.assertNotIn(first.id, capsule.evidence_ids)
        self.assertNotIn(second.id, capsule.evidence_ids)
        diagnostic = next(
            item
            for item in capsule.diagnostics
            if item.code == "semantic_memory_scan_truncated"
        )
        self.assertIn("hits_discarded=true", diagnostic.detail)
        self.assertFalse(
            any(
                reason.channel == "episode_semantic"
                for trace in capsule.trace
                for reason in trace.reasons
            )
        )

    def test_zero_query_skips_semantic_search(self) -> None:
        embedder = _ControlledEmbedder()
        with EviCortex.open(
            self._path("semantic-zero.db"),
            embedder=embedder,
        ) as memory:
            memory.remember(
                namespace="zero/user",
                content="An automobile record.",
            )
            capsule = memory.recall(
                namespace="zero/user",
                query="zero semantic query",
                budget_tokens=512,
            )

        self.assertEqual(capsule.evidence_ids, ())
        self.assertIn(
            "semantic_query_zero",
            {item.code for item in capsule.diagnostics},
        )
        self.assertFalse(
            any(
                "semantic" in reason.channel
                for trace in capsule.trace
                for reason in trace.reasons
            )
        )

    def test_provider_validation_and_failed_open_release_database(self) -> None:
        path = self._path("invalid-provider.db")
        with self.assertRaises(TypeError):
            EviCortex.open(path, embedder=object())  # type: ignore[arg-type]
        with EviCortex.open(path) as reopened:
            self.assertEqual(reopened.stats().memories, 0)
        path.unlink()
        self.assertFalse(path.exists())

        config_path = self._path("invalid-config.db")
        with self.assertRaises(ValueError):
            EviCortex.open(config_path, semantic_scan_cap=0)
        with EviCortex.open(config_path) as reopened:
            self.assertEqual(reopened.stats().memories, 0)
        config_path.unlink()
        self.assertFalse(config_path.exists())

        wrong = _ControlledEmbedder(wrong_dimensions=True)
        with EviCortex.open(
            self._path("invalid-vector.db"),
            embedder=wrong,
        ) as memory:
            with self.assertRaises(SemanticProjectionError) as raised:
                memory.remember(
                    namespace="provider/user",
                    content="Committed before invalid projection.",
                )
            self.assertIsInstance(
                raised.exception.__cause__,
                EmbeddingValidationError,
            )
            self.assertEqual(
                memory.expand(
                    namespace="provider/user",
                    memory_id=raised.exception.memory_id,
                ).content,
                "Committed before invalid projection.",
            )

        mutable = _ControlledEmbedder()
        with EviCortex.open(
            self._path("mutated-provider.db"),
            embedder=mutable,
            auto_index=False,
        ) as memory:
            self.assertEqual(memory.embedding_profile.dimensions, 2)  # type: ignore[union-attr]
            mutable.dimensions = 3
            with self.assertRaises(EmbeddingValidationError):
                memory.recall(
                    namespace="provider/user",
                    query="car",
                    budget_tokens=128,
                )


class _GraphSpyStore:
    """Small fixed-signature store double for graph-channel contract coverage."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], int]] = []
        self.direct = ClaimRecord(
            id="claim-direct",
            namespace="graph/user",
            memory_id="memory-direct",
            subject="project:apollo",
            predicate="owned_by",
            value="Varun",
            valid_from="2026-01-01T00:00:00Z",
            valid_to=None,
            recorded_at="2026-01-01T00:00:00Z",
            confidence=1.0,
            authority="application",
            supersedes=None,
            status="active",
        )
        self.related = ClaimRecord(
            id="claim-related",
            namespace="graph/user",
            memory_id="memory-related",
            subject="Varun",
            predicate="located_in",
            value="Chennai",
            valid_from="2026-01-02T00:00:00Z",
            valid_to=None,
            recorded_at="2026-01-02T00:00:00Z",
            confidence=1.0,
            authority="application",
            supersedes=None,
            status="active",
        )
        self.stale = ClaimRecord(
            id="claim-stale",
            namespace="graph/user",
            memory_id="memory-stale",
            subject="Varun",
            predicate="prefers_theme",
            value="light",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-04-01T00:00:00Z",
            recorded_at="2026-01-01T00:00:00Z",
            confidence=1.0,
            authority="application",
            supersedes=None,
            status="superseded",
        )

    def search_claims(
        self,
        namespace: str,
        query: str,
        as_of: str | None = None,
        limit: int = 10,
    ) -> list[ClaimHit]:
        del query, as_of, limit
        if namespace != "graph/user":
            return []
        return [ClaimHit(self.direct, rank=1.0, reason="claim lexical")]

    def search_memories(self, namespace: str, query: str, limit: int = 10) -> list[object]:
        del namespace, query, limit
        return []

    def graph_neighbors(
        self,
        namespace: str,
        seed_terms: tuple[str, ...],
        limit: int = 10,
    ) -> list[ClaimHit]:
        self.calls.append((namespace, seed_terms, limit))
        return [
            ClaimHit(self.related, rank=1.0, reason="shared entity"),
            ClaimHit(self.stale, rank=2.0, reason="obsolete neighbor"),
        ]


class _HistoricalSpyStore:
    def __init__(self) -> None:
        self.claim_calls: list[tuple[str, str, str | None, int]] = []
        self.memory_calls: list[tuple[str, str, str | None, int]] = []

    def search_claims(
        self,
        namespace: str,
        query: str,
        as_of: str | None = None,
        limit: int = 10,
    ) -> list[object]:
        self.claim_calls.append((namespace, query, as_of, limit))
        return []

    def search_memories(
        self,
        namespace: str,
        query: str,
        limit: int = 10,
        *,
        as_of: str | None = None,
    ) -> list[object]:
        self.memory_calls.append((namespace, query, as_of, limit))
        return []

    def graph_neighbors(
        self,
        namespace: str,
        seed_terms: tuple[str, ...],
        limit: int = 10,
    ) -> list[object]:
        del namespace, seed_terms, limit
        return []


class RetrievalChannelTests(unittest.TestCase):
    def test_graph_channel_uses_seed_terms_and_cannot_reintroduce_stale_fact(self) -> None:
        store = _GraphSpyStore()
        capsule = DeterministicRetriever(store).recall(
            namespace="graph/user",
            query="Apollo owner",
            budget_tokens=2048,
            limit=10,
        )

        self.assertEqual(len(store.calls), 1)
        namespace, seed_terms, _ = store.calls[0]
        self.assertEqual(namespace, "graph/user")
        self.assertIn("apollo", seed_terms)
        self.assertIn("varun", seed_terms)
        self.assertIn("claim-related", {fact.claim_id for fact in capsule.facts})
        self.assertNotIn("claim-stale", {fact.claim_id for fact in capsule.facts})
        related_trace = next(
            item for item in capsule.trace if item.candidate_id == "claim-related"
        )
        self.assertTrue(related_trace.selected)
        self.assertIn("graph", {reason.channel for reason in related_trace.reasons})

    def test_historical_cutoff_reaches_memory_search_without_mutating_query(self) -> None:
        store = _HistoricalSpyStore()
        original_query = "Which theme applied as of 2026-02-01?"
        cutoff = "2026-02-01T00:00:00Z"

        capsule = DeterministicRetriever(store).recall(
            namespace="history/user",
            query=original_query,
            as_of=cutoff,
            budget_tokens=256,
        )

        self.assertEqual(capsule.query, original_query)
        self.assertEqual(len(store.memory_calls), 1)
        namespace, search_query, received_as_of, _ = store.memory_calls[0]
        self.assertEqual(namespace, "history/user")
        self.assertEqual(received_as_of, cutoff)
        self.assertIn("theme", search_query)
        self.assertNotIn("2026", search_query)
        self.assertNotIn("02", search_query)
        self.assertNotIn("01", search_query)
        self.assertEqual(store.claim_calls[0][1], search_query)
        self.assertEqual(store.claim_calls[0][2], cutoff)


if __name__ == "__main__":
    unittest.main()
