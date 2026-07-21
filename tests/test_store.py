from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from evicortex.errors import (
    EmbeddingProfileNotFoundError,
    IdempotencyConflictError,
    IntegrityViolationError,
    StorageClosedError,
    ValidationError,
)
from evicortex.models import ClaimInput
from evicortex.plugins import (
    CanonicalClaimExtractor,
    ClaimExtraction,
    EvidenceContext,
    ExtractionDiagnostic,
)
from evicortex.semantic import (
    EmbeddingInput,
    EmbeddingProfile,
    claim_embedding_input,
    memory_embedding_input,
)
from evicortex.store import SQLiteMemoryStore


def _downgrade_fixture_to_legacy_schema(path: Path, version: int) -> None:
    if version not in {1, 2}:
        raise AssertionError("test helper only supports Prototype A v1/v2")
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            DROP TABLE extraction_diagnostics;
            DROP TABLE claim_derivations;
            DROP TABLE extractions;
            DROP INDEX claims_derivation_target;
            DROP TABLE claim_embeddings;
            DROP TABLE memory_embeddings;
            DROP TABLE embedding_profiles;
            """
        )
        if version == 1:
            connection.executescript(
                """
                DROP TRIGGER memories_fts_insert;
                DROP TRIGGER memories_fts_delete;
                DROP TRIGGER memories_fts_update;
                DROP TRIGGER claims_fts_insert;
                DROP TRIGGER claims_fts_delete;
                DROP TRIGGER claims_fts_update;
                DROP TABLE memory_fts;
                DROP TABLE claim_fts;
                ALTER TABLE claims DROP COLUMN redacted_valid_to;
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    memory_id UNINDEXED,
                    namespace UNINDEXED,
                    content,
                    kind,
                    source,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                CREATE VIRTUAL TABLE claim_fts USING fts5(
                    claim_id UNINDEXED,
                    namespace UNINDEXED,
                    subject,
                    predicate,
                    value,
                    authority,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                INSERT INTO memory_fts(memory_id, namespace, content, kind, source)
                SELECT id, namespace, content, kind, source FROM memories;
                INSERT INTO claim_fts(
                    claim_id, namespace, subject, predicate, value, authority
                )
                SELECT id, namespace, subject, predicate, value, authority
                FROM claims;
                PRAGMA user_version = 1;
                """
            )
        else:
            connection.executescript(
                """
                ALTER TABLE claims DROP COLUMN redacted_valid_to;
                PRAGMA user_version = 2;
                """
            )
        connection.commit()
    finally:
        connection.close()


def _downgrade_fixture_to_v4(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = OFF;
            DROP TABLE extraction_diagnostics;
            DROP TABLE claim_derivations;
            DROP TABLE extractions;
            DROP INDEX claims_derivation_target;
            PRAGMA user_version = 4;
            """
        )
        connection.commit()
    finally:
        connection.close()


class SQLiteMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "memory.sqlite3"
        self.store = SQLiteMemoryStore(self.database_path)

    def tearDown(self) -> None:
        self.store.close()
        self._temporary_directory.cleanup()

    def test_remember_preserves_exact_evidence_and_normalizes_utc(self) -> None:
        content = "  Exact text, including whitespace.\nSecond line.  "
        result = self.store.remember(
            namespace="tenant/a",
            content=content,
            kind="document",
            source="file:///notes/1",
            occurred_at="2026-04-18T15:00:00+05:30",
            recorded_at=datetime(2026, 4, 18, 10, 0),
            metadata={"z": [1, True], "a": "é"},
        )

        self.assertEqual(result.content, content)
        self.assertEqual(
            result.content_hash, hashlib.sha256(content.encode("utf-8")).hexdigest()
        )
        self.assertEqual(result.occurred_at, "2026-04-18T09:30:00.000000Z")
        self.assertEqual(result.recorded_at, "2026-04-18T10:00:00.000000Z")
        self.assertEqual(result.metadata, {"a": "é", "z": [1, True]})
        self.assertEqual(self.store.get_memory("tenant/a", result.id), result)

    def test_fts_uses_external_content_and_trigger_synchronization(self) -> None:
        definitions = {
            row["name"]: row["sql"]
            for row in self.store._connection.execute(  # noqa: SLF001
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name IN ('memory_fts', 'claim_fts')"
            )
        }
        memory_sql = definitions["memory_fts"].casefold()
        claim_sql = definitions["claim_fts"].casefold()
        self.assertIn("content = 'memories'", memory_sql)
        self.assertIn("content = 'claims'", claim_sql)
        self.assertNotIn("memory_id", memory_sql)
        self.assertNotIn("namespace", memory_sql)
        self.assertNotIn("claim_id", claim_sql)
        self.assertNotIn("namespace", claim_sql)
        self.assertEqual(
            self.store._connection.execute("PRAGMA user_version").fetchone()[0],  # noqa: SLF001
            5,
        )

        stored = self.store.remember(
            namespace="n",
            content="Trigger synchronized evidence",
            claims=[ClaimInput("entity:trigger", "sync_state", "ready")],
        )
        self.assertEqual(
            [hit.id for hit in self.store.search_memories("n", "synchronized")],
            [stored.id],
        )
        self.assertEqual(
            [hit.memory_id for hit in self.store.search_claims("n", "sync state")],
            [stored.id],
        )

    def test_idempotency_replays_same_request_and_rejects_conflict(self) -> None:
        claim = ClaimInput("user:1", "prefers_theme", "dark", confidence=0.9)
        first = self.store.remember(
            namespace="tenant/a",
            content="I prefer dark mode.",
            source="session/1",
            metadata={"turn": 4},
            idempotency_key="request-42",
            claims=[claim],
        )
        replay = self.store.remember(
            namespace="tenant/a",
            content="I prefer dark mode.",
            source="session/1",
            metadata={"turn": 4},
            idempotency_key="request-42",
            claims=[claim],
        )

        self.assertEqual(replay, first)
        self.assertEqual(self.store.stats("tenant/a").memories, 1)
        self.assertEqual(self.store.stats("tenant/a").claims, 1)
        with self.assertRaises(IdempotencyConflictError):
            self.store.remember(
                namespace="tenant/a",
                content="A different payload",
                idempotency_key="request-42",
            )

        other_namespace = self.store.remember(
            namespace="tenant/b",
            content="A different payload",
            idempotency_key="request-42",
        )
        self.assertNotEqual(other_namespace.id, first.id)

    def test_supersession_preserves_evidence_and_supports_historical_reads(self) -> None:
        old = self.store.remember(
            namespace="tenant/a",
            content="The user prefers light mode.",
            occurred_at="2026-01-02T08:00:00Z",
            claims=[ClaimInput("user:1", "prefers_theme", "light")],
        )
        old_claim_id = old.claims[0].id
        new = self.store.remember(
            namespace="tenant/a",
            content="The user switched to dark mode.",
            occurred_at="2026-04-18T09:30:00Z",
            claims=[
                ClaimInput(
                    "user:1",
                    "prefers_theme",
                    "dark",
                    supersedes=old_claim_id,
                )
            ],
        )

        preserved = self.store.get_memory("tenant/a", old.id)
        self.assertIsNotNone(preserved)
        assert preserved is not None
        self.assertEqual(preserved.content, "The user prefers light mode.")
        self.assertEqual(preserved.claims[0].status, "superseded")
        self.assertEqual(
            preserved.claims[0].valid_to, "2026-04-18T09:30:00.000000Z"
        )
        self.assertEqual(new.claims[0].supersedes, old_claim_id)

        current = self.store.search_claims("tenant/a", "prefers theme")
        self.assertEqual([hit.value for hit in current], ["dark"])
        historical = self.store.search_claims(
            "tenant/a", "prefers theme", as_of="2026-02-01T00:00:00Z"
        )
        self.assertEqual([hit.value for hit in historical], ["light"])
        boundary = self.store.search_claims(
            "tenant/a", "prefers theme", as_of="2026-04-18T09:30:00Z"
        )
        self.assertEqual([hit.value for hit in boundary], ["dark"])
        self.assertEqual(
            self.store.get_memory("tenant/a", old.id).content_hash, old.content_hash  # type: ignore[union-attr]
        )

    def test_supersession_is_atomic_when_predecessor_is_invalid(self) -> None:
        before = self.store.stats()
        with self.assertRaises(ValidationError):
            self.store.remember(
                namespace="tenant/a",
                content="Invalid update",
                occurred_at="2026-02-01T00:00:00Z",
                claims=[
                    ClaimInput(
                        "user:1",
                        "prefers_theme",
                        "dark",
                        supersedes="clm_missing",
                    )
                ],
            )
        self.assertEqual(self.store.stats(), before)
        self.assertEqual(self.store.search_memories("tenant/a", "invalid"), [])

    def test_namespace_isolation_precedes_every_ranking_channel(self) -> None:
        first = self.store.remember(
            namespace="tenant/a",
            content="Shared keyword orchid; secret alpha.",
            claims=[ClaimInput("project:alpha", "uses_code", "orchid")],
        )
        second = self.store.remember(
            namespace="tenant/b",
            content="Shared keyword orchid; secret beta beta beta.",
            claims=[ClaimInput("project:beta", "uses_code", "orchid")],
        )

        memory_hits = self.store.search_memories("tenant/a", "orchid beta")
        self.assertEqual([hit.id for hit in memory_hits], [first.id])
        claim_hits = self.store.search_claims("tenant/a", "orchid")
        self.assertEqual({hit.namespace for hit in claim_hits}, {"tenant/a"})
        graph_hits = self.store.graph_neighbors("tenant/a", ["orchid"], limit=10)
        self.assertEqual({hit.namespace for hit in graph_hits}, {"tenant/a"})
        self.assertIsNone(self.store.get_memory("tenant/a", second.id))
        self.assertFalse(self.store.purge_memory("tenant/a", second.id))
        self.assertIsNotNone(self.store.get_memory("tenant/b", second.id))

    def test_parameterized_queries_handle_quotes_and_fts_syntax(self) -> None:
        namespace = "tenant/'quoted'"
        stored = self.store.remember(
            namespace=namespace,
            content='A phrase with "quotes" and punctuation: safe.',
        )
        hits = self.store.search_memories(
            namespace, '"quotes" OR safe*; DROP TABLE memories;--'
        )
        self.assertEqual([hit.id for hit in hits], [stored.id])
        self.assertIsNotNone(self.store.get_memory(namespace, stored.id))

    def test_negated_term_is_filtered_in_fts_and_target_wins(self) -> None:
        target = self.store.remember(
            namespace="n",
            content="Quartz is a compact device.",
            claims=[ClaimInput("device:quartz", "category", "device")],
        )
        for index in range(40):
            self.store.remember(
                namespace="n",
                content=f"Quartz accessory {index} is an accessory for a device.",
                claims=[
                    ClaimInput(
                        f"device:quartz_accessory_{index}", "category", "accessory"
                    )
                ],
            )

        hits = self.store.search_claims(
            "n", "Quartz device, not an accessory", limit=10
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].memory_id, target.id)
        self.assertIn("excluded_matches=0", hits[0].reason)

    def test_namespace_local_pool_is_invariant_and_keeps_late_exact_hits(self) -> None:
        # More scoped distractors than the candidate pool ensures relevance must be
        # applied before LIMIT. The quoted/injection-shaped query also proves every
        # query and namespace value remains data, never executable SQL/FTS syntax.
        for index in range(180):
            self.store.remember(
                namespace="n",
                content=f"Quartz filler record {index}.",
                claims=[
                    ClaimInput(
                        f"device:distractor_{index}", "catalog_entry", "quartz"
                    )
                ],
            )
        target = self.store.remember(
            namespace="n",
            content="Quartz precision calibration target.",
            claims=[
                ClaimInput(
                    "device:quartz",
                    "precision_calibration",
                    "target",
                )
            ],
        )
        query = (
            '\"quartz\" precision calibration, not an accessory; '
            "'); DROP TABLE claims;--"
        )
        before_memories = self.store.search_memories("n", query, limit=5)
        before_claims = self.store.search_claims("n", query, limit=5)
        self.assertEqual(before_memories[0].id, target.id)
        self.assertEqual(before_claims[0].memory_id, target.id)

        # Flooding another namespace changes global FTS term statistics. It must not
        # change any pre-limit result selected inside namespace n.
        for index in range(220):
            self.store.remember(
                namespace="foreign/'quoted'",
                content=f"Foreign precision calibration signal {index}.",
                claims=[
                    ClaimInput(
                        f"foreign:device_{index}",
                        "precision_calibration",
                        "signal",
                    )
                ],
            )
        after_memories = self.store.search_memories("n", query, limit=5)
        after_claims = self.store.search_claims("n", query, limit=5)

        self.assertEqual(
            [hit.id for hit in after_memories],
            [hit.id for hit in before_memories],
        )
        self.assertEqual(
            [hit.id for hit in after_claims],
            [hit.id for hit in before_claims],
        )
        self.assertEqual({hit.namespace for hit in after_memories}, {"n"})
        self.assertEqual({hit.namespace for hit in after_claims}, {"n"})
        self.assertGreater(self.store.stats().claims, 0)

    def test_stopwords_do_not_create_false_positive_abstention_results(self) -> None:
        self.store.remember(namespace="n", content="This is a harmless sentence.")
        self.store.remember(
            namespace="n",
            content="Another item is present.",
            claims=[ClaimInput("item:ordinary", "state", "present")],
        )
        query = "What is Zephyr-999's xylophonic_allergy?"
        self.assertEqual(self.store.search_memories("n", query), [])
        self.assertEqual(self.store.search_claims("n", query), [])

    def test_current_claim_search_observes_finite_and_future_validity(self) -> None:
        now = datetime.now(timezone.utc)
        finite = self.store.remember(
            namespace="n",
            content="The maintenance window is open.",
            occurred_at=now - timedelta(hours=1),
            claims=[
                ClaimInput(
                    "window:1",
                    "window_state",
                    "open",
                    valid_to=now + timedelta(hours=1),
                )
            ],
        )
        future = self.store.remember(
            namespace="n",
            content="The maintenance window will close later.",
            occurred_at=now + timedelta(hours=2),
            claims=[ClaimInput("window:1", "window_state", "closed")],
        )

        current = self.store.search_claims("n", "window state", limit=10)
        self.assertEqual([hit.memory_id for hit in current], [finite.id])
        episodes = self.store.search_memories(
            "n", "maintenance window", as_of=now, limit=10
        )
        self.assertEqual([hit.id for hit in episodes], [finite.id])
        future_hits = self.store.search_claims(
            "n", "window state", as_of=now + timedelta(hours=3), limit=10
        )
        self.assertIn(future.id, {hit.memory_id for hit in future_hits})

    def test_graph_neighbors_prefers_exact_canonical_entity_ids(self) -> None:
        exact = self.store.remember(
            namespace="n",
            content="Atlas is owned by Alice.",
            claims=[ClaimInput("project:atlas", "owner", "person:alice")],
        )
        self.store.remember(
            namespace="n",
            content="Alice lives in Paris.",
            claims=[ClaimInput("person:alice", "lives_in", "city:paris")],
        )
        self.store.remember(
            namespace="n",
            content="Atlas Backup is owned by Bob.",
            claims=[
                ClaimInput("project:atlas_backup", "owner", "person:bob")
            ],
        )

        hits = self.store.graph_neighbors("n", ["project:atlas"], limit=10)
        self.assertEqual([hit.memory_id for hit in hits], [exact.id])
        self.assertEqual(
            hits[0].reason, "graph:exact-entity;namespace-scoped;current"
        )

    def test_graph_neighbors_respects_historical_valid_time(self) -> None:
        old = self.store.remember(
            namespace="n",
            content="Atlas was owned by Alice.",
            occurred_at="2026-01-01T00:00:00Z",
            claims=[ClaimInput("project:atlas", "owner", "person:alice")],
        )
        new = self.store.remember(
            namespace="n",
            content="Atlas is now owned by Bob.",
            occurred_at="2026-04-01T00:00:00Z",
            claims=[
                ClaimInput(
                    "project:atlas",
                    "owner",
                    "person:bob",
                    supersedes=old.claims[0].id,
                )
            ],
        )

        historical = self.store.graph_neighbors(
            "n", ["project:atlas"], as_of="2026-02-01T00:00:00Z"
        )
        current = self.store.graph_neighbors("n", ["project:atlas"])
        self.assertEqual([hit.memory_id for hit in historical], [old.id])
        self.assertEqual([hit.memory_id for hit in current], [new.id])
        self.assertIn("as-of=2026-02-01T00:00:00.000000Z", historical[0].reason)

    def test_integrity_detects_projection_damage_and_rebuild_restores_it(self) -> None:
        stored = self.store.remember(
            namespace="tenant/a",
            content="Rebuildable lexical evidence",
            claims=[ClaimInput("system:index", "state", "healthy")],
        )
        self.assertTrue(self.store.verify_integrity().ok)
        memory_row = self.store._connection.execute(  # noqa: SLF001
            "SELECT rowid, content, kind, source FROM memories WHERE id = ?",
            (stored.id,),
        ).fetchone()
        claim_row = self.store._connection.execute(  # noqa: SLF001
            "SELECT rowid, subject, predicate, value, authority "
            "FROM claims WHERE id = ?",
            (stored.claims[0].id,),
        ).fetchone()
        self.store._connection.execute(  # noqa: SLF001 - deliberate corruption test
            "INSERT INTO memory_fts(memory_fts, rowid, content, kind, source) "
            "VALUES ('delete', ?, ?, ?, ?)",
            tuple(memory_row),
        )
        self.store._connection.execute(  # noqa: SLF001 - deliberate corruption test
            "INSERT INTO claim_fts("
            "claim_fts, rowid, subject, predicate, value, authority"
            ") VALUES ('delete', ?, ?, ?, ?, ?)",
            tuple(claim_row),
        )
        self.store._connection.commit()  # noqa: SLF001

        damaged = self.store.verify_integrity()
        self.assertFalse(damaged.ok)
        self.assertEqual(damaged.corrupt_memory_ids, ())
        self.assertTrue(
            any("memory_fts:integrity" in issue for issue in damaged.index_issues)
        )
        self.assertEqual(self.store.search_memories("tenant/a", "rebuildable"), [])

        self.store.rebuild_indexes()
        restored = self.store.verify_integrity()
        self.assertTrue(restored.ok, restored.index_issues)
        self.assertEqual(
            [hit.id for hit in self.store.search_memories("tenant/a", "rebuildable")],
            [stored.id],
        )
        exact = self.store.get_memory("tenant/a", stored.id)
        self.assertIsNotNone(exact)
        assert exact is not None
        self.assertEqual(exact.content, stored.content)
        self.assertEqual(exact.content_hash, stored.content_hash)

    def test_integrity_detects_evidence_digest_corruption(self) -> None:
        stored = self.store.remember(namespace="tenant/a", content="Original evidence")
        self.store._connection.execute(  # noqa: SLF001 - deliberate corruption test
            "UPDATE memories SET content = ? WHERE id = ?", ("Tampered", stored.id)
        )
        self.store._connection.commit()  # noqa: SLF001
        report = self.store.verify_integrity("tenant/a")
        self.assertFalse(report.ok)
        self.assertEqual(report.corrupt_memory_ids, (stored.id,))

    def test_rebuild_restores_temporal_projection_as_well_as_fts(self) -> None:
        old = self.store.remember(
            namespace="n",
            content="Account tier was basic.",
            occurred_at="2026-01-01T00:00:00Z",
            claims=[ClaimInput("account:1", "account_tier", "basic")],
        )
        new = self.store.remember(
            namespace="n",
            content="Account tier became pro.",
            occurred_at="2026-02-01T00:00:00Z",
            claims=[
                ClaimInput(
                    "account:1",
                    "account_tier",
                    "pro",
                    supersedes=old.claims[0].id,
                )
            ],
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE claims SET status = 'active', valid_to = NULL WHERE id = ?",
            (old.claims[0].id,),
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE claims SET status = 'superseded', valid_to = ? WHERE id = ?",
            ("2026-01-15T00:00:00.000000Z", new.claims[0].id),
        )
        self.store._connection.execute(  # noqa: SLF001
            "INSERT INTO claim_fts(claim_fts) VALUES ('delete-all')"
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertFalse(self.store.verify_integrity().ok)

        self.store.rebuild_indexes()

        self.assertTrue(self.store.verify_integrity().ok)
        rebuilt_old = self.store.get_memory("n", old.id)
        rebuilt_new = self.store.get_memory("n", new.id)
        assert rebuilt_old is not None and rebuilt_new is not None
        self.assertEqual(rebuilt_old.claims[0].status, "superseded")
        self.assertEqual(
            rebuilt_old.claims[0].valid_to, "2026-02-01T00:00:00.000000Z"
        )
        self.assertEqual(rebuilt_new.claims[0].status, "active")
        self.assertIsNone(rebuilt_new.claims[0].valid_to)
        self.assertEqual(rebuilt_old.content, old.content)
        self.assertEqual(rebuilt_old.content_hash, old.content_hash)
        historical = self.store.search_claims(
            "n", "account tier", as_of="2026-01-15T00:00:00Z"
        )
        self.assertEqual([hit.value for hit in historical], ["basic"])

    def test_purge_closes_predecessor_at_an_unknown_redaction_boundary(self) -> None:
        original = self.store.remember(
            namespace="tenant/a",
            content="Status is queued.",
            occurred_at="2026-01-01T00:00:00Z",
            claims=[ClaimInput("job:1", "status", "queued")],
        )
        update = self.store.remember(
            namespace="tenant/a",
            content="Status is running.",
            occurred_at="2026-01-02T00:00:00Z",
            claims=[
                ClaimInput(
                    "job:1", "status", "running", supersedes=original.claims[0].id
                )
            ],
        )
        survivor = self.store.remember(
            namespace="tenant/b", content="Status is private to tenant B."
        )

        self.assertTrue(self.store.purge_memory("tenant/a", update.id))
        self.assertFalse(self.store.purge_memory("tenant/a", update.id))
        self.assertIsNone(self.store.get_memory("tenant/a", update.id))
        current = self.store.search_claims("tenant/a", "job status")
        self.assertEqual(current, [])
        restored_original = self.store.get_memory("tenant/a", original.id)
        self.assertIsNotNone(restored_original)
        assert restored_original is not None
        self.assertEqual(restored_original.claims[0].status, "active")
        self.assertEqual(
            restored_original.claims[0].valid_to,
            "2026-01-02T00:00:00.000000Z",
        )
        before_redaction = self.store.search_claims(
            "tenant/a", "job status", as_of="2026-01-01T12:00:00Z"
        )
        at_redaction = self.store.search_claims(
            "tenant/a", "job status", as_of="2026-01-02T00:00:00Z"
        )
        self.assertEqual([hit.value for hit in before_redaction], ["queued"])
        self.assertEqual(at_redaction, [])
        boundary = self.store._connection.execute(  # noqa: SLF001
            "SELECT declared_valid_to, redacted_valid_to FROM claims WHERE id = ?",
            (original.claims[0].id,),
        ).fetchone()
        self.assertIsNone(boundary["declared_valid_to"])
        self.assertEqual(
            boundary["redacted_valid_to"], "2026-01-02T00:00:00.000000Z"
        )
        self.store.rebuild_indexes()
        self.assertEqual(self.store.search_claims("tenant/a", "job status"), [])
        self.assertIsNotNone(self.store.get_memory("tenant/b", survivor.id))
        self.assertTrue(self.store.verify_integrity().ok)

        self.assertTrue(self.store.purge_memory("tenant/a", original.id))
        self.assertEqual(self.store.stats("tenant/a").memories, 0)
        self.assertEqual(self.store.stats("tenant/a").claims, 0)
        self.assertEqual(self.store.search_memories("tenant/a", "status"), [])

    def test_purge_bridges_middle_of_supersession_chain(self) -> None:
        first = self.store.remember(
            namespace="n",
            content="Phase A",
            occurred_at="2026-01-01T00:00:00Z",
            claims=[ClaimInput("job:1", "job_phase", "A")],
        )
        middle = self.store.remember(
            namespace="n",
            content="Phase B",
            occurred_at="2026-02-01T00:00:00Z",
            claims=[
                ClaimInput(
                    "job:1", "job_phase", "B", supersedes=first.claims[0].id
                )
            ],
        )
        last = self.store.remember(
            namespace="n",
            content="Phase C",
            occurred_at="2026-03-01T00:00:00Z",
            claims=[
                ClaimInput(
                    "job:1", "job_phase", "C", supersedes=middle.claims[0].id
                )
            ],
        )

        self.assertTrue(self.store.purge_memory("n", middle.id))

        rebuilt_first = self.store.get_memory("n", first.id)
        rebuilt_last = self.store.get_memory("n", last.id)
        assert rebuilt_first is not None and rebuilt_last is not None
        self.assertEqual(rebuilt_last.claims[0].supersedes, first.claims[0].id)
        self.assertEqual(rebuilt_first.claims[0].status, "superseded")
        self.assertEqual(
            rebuilt_first.claims[0].valid_to, "2026-02-01T00:00:00.000000Z"
        )
        before_gap = self.store.search_claims(
            "n", "job phase", as_of="2026-01-15T00:00:00Z"
        )
        in_unknown_gap = self.store.search_claims(
            "n", "job phase", as_of="2026-02-15T00:00:00Z"
        )
        self.assertEqual([hit.value for hit in before_gap], ["A"])
        self.assertEqual(in_unknown_gap, [])
        current = self.store.search_claims("n", "job phase")
        self.assertEqual([hit.value for hit in current], ["C"])
        self.store.rebuild_indexes()
        rebuilt_first = self.store.get_memory("n", first.id)
        assert rebuilt_first is not None
        self.assertEqual(
            rebuilt_first.claims[0].valid_to, "2026-02-01T00:00:00.000000Z"
        )

        # Repeatedly purging the surviving successor must preserve the earliest
        # redaction boundary instead of reactivating A through the present.
        self.assertTrue(self.store.purge_memory("n", last.id))
        self.assertFalse(self.store.purge_memory("n", last.id))
        self.assertIsNone(self.store.get_memory("n", middle.id))
        self.assertIsNone(self.store.get_memory("n", last.id))
        remaining = self.store.get_memory("n", first.id)
        assert remaining is not None
        self.assertEqual(remaining.claims[0].status, "active")
        self.assertEqual(
            remaining.claims[0].valid_to, "2026-02-01T00:00:00.000000Z"
        )
        self.assertEqual(self.store.search_claims("n", "job phase"), [])
        self.assertTrue(self.store.verify_integrity().ok)

    def test_validation_and_closed_lifecycle(self) -> None:
        with self.assertRaises(ValidationError):
            self.store.remember(namespace="", content="x")
        with self.assertRaises(ValidationError):
            self.store.remember(namespace="a", content="")
        with self.assertRaises(ValidationError):
            self.store.remember(
                namespace="a",
                content="x",
                metadata={"not-finite": float("nan")},
            )
        with self.assertRaises(ValidationError):
            self.store.remember(
                namespace="a",
                content="x",
                claims=[ClaimInput("s", "p", "v", confidence=1.1)],
            )
        with self.assertRaises(ValidationError):
            self.store.remember(
                namespace="a", content="x", occurred_at="not-a-timestamp"
            )

        self.store.close()
        self.store.close()
        with self.assertRaises(StorageClosedError):
            self.store.stats()

    def test_context_manager_persists_file_database(self) -> None:
        another_path = Path(self._temporary_directory.name) / "context.sqlite3"
        with SQLiteMemoryStore(another_path) as store:
            record = store.remember(namespace="n", content="persistent evidence")
        with SQLiteMemoryStore(another_path) as reopened:
            self.assertEqual(reopened.get_memory("n", record.id).content, record.content)  # type: ignore[union-attr]
            self.assertTrue(reopened.verify_integrity().ok)

    def test_v1_and_v2_databases_migrate_in_place_and_reopen(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version):
                path = Path(self._temporary_directory.name) / f"legacy-v{version}.sqlite3"
                with SQLiteMemoryStore(path) as seed:
                    old = seed.remember(
                        namespace="n",
                        content="The plan was bronze.",
                        occurred_at="2026-01-01T00:00:00Z",
                        idempotency_key="legacy-request",
                        claims=[ClaimInput("account:1", "plan_tier", "bronze")],
                    )
                    new = seed.remember(
                        namespace="n",
                        content="The plan became silver.",
                        occurred_at="2026-02-01T00:00:00Z",
                        claims=[
                            ClaimInput(
                                "account:1",
                                "plan_tier",
                                "silver",
                                supersedes=old.claims[0].id,
                            )
                        ],
                    )
                _downgrade_fixture_to_legacy_schema(path, version)

                with closing(sqlite3.connect(path)) as legacy:
                    self.assertEqual(
                        legacy.execute("PRAGMA user_version").fetchone()[0], version
                    )
                with SQLiteMemoryStore(path) as migrated:
                    self.assertEqual(
                        migrated._connection.execute(  # noqa: SLF001
                            "PRAGMA user_version"
                        ).fetchone()[0],
                        5,
                    )
                    columns = {
                        row["name"]
                        for row in migrated._connection.execute(  # noqa: SLF001
                            "PRAGMA table_info(claims)"
                        )
                    }
                    self.assertIn("redacted_valid_to", columns)
                    self.assertEqual(migrated.get_memory("n", old.id).content, old.content)  # type: ignore[union-attr]
                    self.assertEqual(migrated.get_memory("n", new.id).content, new.content)  # type: ignore[union-attr]
                    current = migrated.search_claims("n", "plan tier")
                    historical = migrated.search_claims(
                        "n", "plan tier", as_of="2026-01-15T00:00:00Z"
                    )
                    self.assertEqual([hit.value for hit in current], ["silver"])
                    self.assertEqual([hit.value for hit in historical], ["bronze"])
                    replay = migrated.remember(
                        namespace="n",
                        content="The plan was bronze.",
                        occurred_at="2026-01-01T00:00:00Z",
                        idempotency_key="legacy-request",
                        claims=[ClaimInput("account:1", "plan_tier", "bronze")],
                    )
                    self.assertEqual(replay.id, old.id)
                    appended = migrated.remember(
                        namespace="n", content="Migration trigger sentinel"
                    )
                    self.assertEqual(
                        [
                            hit.id
                            for hit in migrated.search_memories(
                                "n", "migration sentinel"
                            )
                        ],
                        [appended.id],
                    )
                    self.assertTrue(migrated.verify_integrity().ok)

                # A second open exercises the strict v5 validation path, not migration.
                with SQLiteMemoryStore(path) as reopened:
                    self.assertTrue(reopened.verify_integrity().ok)
                    self.assertEqual(reopened.get_memory("n", old.id).content_hash, old.content_hash)  # type: ignore[union-attr]

    def test_newer_and_ambiguous_unversioned_databases_are_rejected(self) -> None:
        newer_path = Path(self._temporary_directory.name) / "future.sqlite3"
        with closing(sqlite3.connect(newer_path)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        with self.assertRaisesRegex(IntegrityViolationError, "newer than supported"):
            SQLiteMemoryStore(newer_path)
        with closing(sqlite3.connect(newer_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%'"
                ).fetchone()[0],
                0,
            )

        ambiguous_path = Path(self._temporary_directory.name) / "ambiguous.sqlite3"
        with closing(sqlite3.connect(ambiguous_path)) as connection:
            connection.execute("CREATE TABLE unrelated(value TEXT)")
            connection.commit()
        with self.assertRaisesRegex(IntegrityViolationError, "unversioned non-empty"):
            SQLiteMemoryStore(ambiguous_path)
        with closing(sqlite3.connect(ambiguous_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertIsNotNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'unrelated'"
                ).fetchone()
            )

    def test_get_claim_is_namespace_scoped(self) -> None:
        stored = self.store.remember(
            namespace="tenant/a",
            content="A private claim",
            claims=[ClaimInput("account:1", "plan", "gold")],
        )
        claim = stored.claims[0]
        self.assertEqual(self.store.get_claim("tenant/a", claim.id), claim)
        self.assertIsNone(self.store.get_claim("tenant/b", claim.id))
        self.assertIsNone(self.store.get_claim("tenant/a", "clm_missing"))

    def test_extraction_persists_ordered_claim_derivations_with_generated_ids(self) -> None:
        extractor = CanonicalClaimExtractor()
        content = (
            "Evidence preface. "
            "Canonical claim: subject=project:atlas; predicate=started_on; "
            "value=2026-01-10; confidence=0.750; authority=calendar. "
            "Canonical claim: subject=project:atlas; predicate=released_on; "
            "value=2026-03-05; confidence=1.000; authority=release-system."
        )
        context = EvidenceContext.from_content(
            namespace="tenant/a",
            content=content,
            source="calendar://atlas",
            occurred_at="2026-03-06T00:00:00.000000Z",
            metadata={"batch": 7},
        )
        extraction = extractor.extract(context)
        stored = self.store.remember(
            namespace="tenant/a",
            content=content,
            source="calendar://atlas",
            occurred_at="2026-03-06T00:00:00Z",
            metadata={"batch": 7},
            extraction=extraction,
            extraction_identity=extractor.identity,
        )

        run = self.store.get_extraction("tenant/a", stored.id)
        self.assertIsNotNone(run)
        assert run is not None
        self.assertFalse(run.abstained)
        self.assertEqual(run.extractor_identity, extractor.identity)
        self.assertEqual(run.content_sha256, stored.content_hash)
        self.assertEqual([item.claim_index for item in run.derivations], [0, 1])
        self.assertEqual(len(set(run.claim_ids)), 2)
        mapped_claims = [
            self.store.get_claim("tenant/a", item.claim_id)
            for item in run.derivations
        ]
        self.assertEqual(
            [claim.predicate for claim in mapped_claims if claim is not None],
            ["started_on", "released_on"],
        )
        for item in run.derivations:
            self.assertEqual(content[item.start : item.end], item.canonical_text)
            self.assertEqual(item.content_sha256, stored.content_hash)
            self.assertEqual(item.extractor_identity, extractor.identity)
        self.assertEqual(
            self.store._connection.execute(  # noqa: SLF001
                "PRAGMA user_version"
            ).fetchone()[0],
            5,
        )

    def test_extraction_persists_abstention_and_rejected_span_diagnostics(self) -> None:
        extractor = CanonicalClaimExtractor()
        rejected_content = "Canonical claim: malformed."
        rejected_context = EvidenceContext.from_content(
            namespace="tenant/a", content=rejected_content
        )
        rejected = extractor.extract(rejected_context)
        self.assertTrue(rejected.abstained)
        stored = self.store.remember(
            namespace="tenant/a",
            content=rejected_content,
            extraction=rejected,
            extraction_identity=extractor.identity,
        )
        run = self.store.get_extraction("tenant/a", stored.id)
        self.assertIsNotNone(run)
        assert run is not None
        self.assertTrue(run.abstained)
        self.assertEqual(run.derivations, ())
        self.assertEqual(run.diagnostics[0].diagnostic_index, 0)
        self.assertEqual(run.diagnostics[0].code, "malformed_canonical_claim")
        self.assertEqual(run.diagnostics[0].start, 0)
        self.assertEqual(run.diagnostics[0].end, len(rejected_content))

        plain_content = "No structured assertion is present."
        plain = extractor.extract(
            EvidenceContext.from_content(
                namespace="tenant/a", content=plain_content
            )
        )
        plain_record = self.store.remember(
            namespace="tenant/a",
            content=plain_content,
            extraction=plain,
            extraction_identity=extractor.identity,
        )
        plain_run = self.store.get_extraction("tenant/a", plain_record.id)
        self.assertIsNotNone(plain_run)
        assert plain_run is not None
        self.assertEqual(plain_run.diagnostics[0].code, "no_canonical_claims")
        self.assertIsNone(plain_run.diagnostics[0].start)
        self.assertIsNone(plain_run.diagnostics[0].end)

    def test_malformed_extraction_is_rejected_before_any_write(self) -> None:
        extractor = CanonicalClaimExtractor()
        content = (
            "Canonical claim: subject=device:q; predicate=state; value=ready; "
            "confidence=1.000; authority=registry."
        )
        valid = extractor.extract(
            EvidenceContext.from_content(namespace="n", content=content)
        )
        forged = ClaimExtraction(
            claims=valid.claims,
            provenance=(
                replace(
                    valid.provenance[0],
                    canonical_text=valid.provenance[0].canonical_text + " forged",
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "invalid.*evidence context"):
            self.store.remember(
                namespace="n",
                content=content,
                extraction=forged,
                extraction_identity=extractor.identity,
            )
        self.assertEqual(self.store.stats().memories, 0)
        self.assertEqual(
            self.store._connection.execute(  # noqa: SLF001
                "SELECT COUNT(*) FROM extractions"
            ).fetchone()[0],
            0,
        )
        with self.assertRaisesRegex(ValidationError, "mutually exclusive"):
            self.store.remember(
                namespace="n",
                content=content,
                claims=(ClaimInput("s", "p", "v"),),
                extraction=valid,
                extraction_identity=extractor.identity,
            )
        with self.assertRaisesRegex(ValidationError, "extraction_identity"):
            self.store.remember(
                namespace="n", content=content, extraction=valid
            )
        self.assertEqual(self.store.stats().memories, 0)

    def test_extraction_provenance_participates_in_idempotency_semantics(self) -> None:
        extractor = CanonicalClaimExtractor()
        content = (
            "Canonical claim: subject=device:q; predicate=state; value=ready; "
            "confidence=1.000; authority=registry."
        )
        context = EvidenceContext.from_content(namespace="n", content=content)
        extraction = extractor.extract(context)
        first = self.store.remember(
            namespace="n",
            content=content,
            idempotency_key="extract-1",
            extraction=extraction,
            extraction_identity=extractor.identity,
        )
        replay = self.store.remember(
            namespace="n",
            content=content,
            idempotency_key="extract-1",
            extraction=extraction,
            extraction_identity=extractor.identity,
        )
        self.assertEqual(replay.id, first.id)
        self.assertEqual(
            self.store._connection.execute(  # noqa: SLF001
                "SELECT COUNT(*) FROM extractions"
            ).fetchone()[0],
            1,
        )
        changed_diagnostics = ClaimExtraction(
            claims=extraction.claims,
            provenance=extraction.provenance,
            diagnostics=(ExtractionDiagnostic("audit_note", "different output"),),
        )
        with self.assertRaises(IdempotencyConflictError):
            self.store.remember(
                namespace="n",
                content=content,
                idempotency_key="extract-1",
                extraction=changed_diagnostics,
                extraction_identity=extractor.identity,
            )
        self.assertEqual(self.store.get_extraction("n", first.id).diagnostics, ())  # type: ignore[union-attr]

    def test_get_extraction_is_namespace_scoped(self) -> None:
        extractor = CanonicalClaimExtractor()
        content = "No canonical claims."
        extraction = extractor.extract(
            EvidenceContext.from_content(namespace="tenant/a", content=content)
        )
        stored = self.store.remember(
            namespace="tenant/a",
            content=content,
            extraction=extraction,
            extraction_identity=extractor.identity,
        )
        self.assertIsNotNone(self.store.get_extraction("tenant/a", stored.id))
        self.assertIsNone(self.store.get_extraction("tenant/b", stored.id))
        self.assertIsNone(self.store.get_extraction("tenant/a", "mem_missing"))

    def test_extraction_survives_rebuild_reopen_and_cascades_on_purge(self) -> None:
        extractor = CanonicalClaimExtractor()
        content = (
            "Canonical claim: subject=device:q; predicate=state; value=ready; "
            "confidence=1.000; authority=registry."
        )
        extraction = extractor.extract(
            EvidenceContext.from_content(namespace="n", content=content)
        )
        stored = self.store.remember(
            namespace="n",
            content=content,
            extraction=extraction,
            extraction_identity=extractor.identity,
        )
        before = self.store.get_extraction("n", stored.id)
        self.store.rebuild_indexes()
        self.assertEqual(self.store.get_extraction("n", stored.id), before)
        self.store.close()
        self.store = SQLiteMemoryStore(self.database_path)
        self.assertEqual(self.store.get_extraction("n", stored.id), before)
        self.assertTrue(self.store.purge_memory("n", stored.id))
        counts = self.store._connection.execute(  # noqa: SLF001
            "SELECT (SELECT COUNT(*) FROM extractions), "
            "(SELECT COUNT(*) FROM claim_derivations), "
            "(SELECT COUNT(*) FROM extraction_diagnostics)"
        ).fetchone()
        self.assertEqual(tuple(counts), (0, 0, 0))
        self.assertTrue(self.store.verify_integrity().ok)

    def test_extraction_integrity_detects_digest_span_identity_and_mapping_damage(self) -> None:
        extractor = CanonicalClaimExtractor()
        content = (
            "Canonical claim: subject=device:q; predicate=state; value=ready; "
            "confidence=1.000; authority=registry."
        )
        base = extractor.extract(
            EvidenceContext.from_content(namespace="n", content=content)
        )
        extraction = ClaimExtraction(
            claims=base.claims,
            provenance=base.provenance,
            diagnostics=(ExtractionDiagnostic("ignored_prefix", "prefix ignored", 0, 0),),
        )
        stored = self.store.remember(
            namespace="n",
            content=content,
            extraction=extraction,
            extraction_identity=extractor.identity,
        )
        original = self.store.get_extraction("n", stored.id)
        assert original is not None

        self.store._connection.execute(  # noqa: SLF001
            "UPDATE claim_derivations SET canonical_text = canonical_text || 'x' "
            "WHERE memory_id = ?",
            (stored.id,),
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertTrue(
            any(
                issue == f"extraction:{stored.id}:provenance"
                for issue in self.store.verify_integrity("n").index_issues
            )
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE claim_derivations SET canonical_text = ? WHERE memory_id = ?",
            (original.derivations[0].canonical_text, stored.id),
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE extraction_diagnostics SET end_offset = ? WHERE memory_id = ?",
            (len(content) + 1, stored.id),
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertIn(
            f"extraction:{stored.id}:provenance",
            self.store.verify_integrity("n").index_issues,
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE extraction_diagnostics SET end_offset = 0 WHERE memory_id = ?",
            (stored.id,),
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE extractions SET content_sha256 = ? WHERE memory_id = ?",
            ("0" * 64, stored.id),
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertIn(
            f"extraction:{stored.id}:content_digest",
            self.store.verify_integrity("n").index_issues,
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE extractions SET content_sha256 = ?, extractor_identity = ? "
            "WHERE memory_id = ?",
            (stored.content_hash, " altered ", stored.id),
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertIn(
            f"extraction:{stored.id}:identity",
            self.store.verify_integrity("n").index_issues,
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE extractions SET extractor_identity = ? WHERE memory_id = ?",
            (extractor.identity, stored.id),
        )
        self.store._connection.execute(  # noqa: SLF001
            "DELETE FROM claim_derivations WHERE memory_id = ?", (stored.id,)
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertIn(
            f"extraction:{stored.id}:claim_mapping",
            self.store.verify_integrity("n").index_issues,
        )

    def test_embedding_batch_derives_ownership_and_is_all_or_nothing(self) -> None:
        profile = EmbeddingProfile("test/model", 3, False)
        first = self.store.remember(
            namespace="tenant/a",
            content="Persistent semantic evidence",
            claims=[ClaimInput("entity:1", "state", "ready")],
        )
        second = self.store.remember(namespace="tenant/b", content="Other tenant")
        inputs = (
            memory_embedding_input(first),
            claim_embedding_input(first.claims[0]),
            memory_embedding_input(second),
        )
        result = self.store.upsert_embeddings(
            profile,
            inputs,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(result.written, 3)
        self.assertEqual(result.memory_embeddings, 2)
        self.assertEqual(result.claim_embeddings, 1)
        self.assertEqual(result.namespaces, ("tenant/a", "tenant/b"))
        rows = self.store._connection.execute(  # noqa: SLF001
            "SELECT namespace, length(vector) AS bytes FROM memory_embeddings "
            "ORDER BY namespace"
        ).fetchall()
        self.assertEqual(
            [(row["namespace"], row["bytes"]) for row in rows],
            [("tenant/a", 12), ("tenant/b", 12)],
        )

        fresh_profile = EmbeddingProfile("test/rollback", 3, False)
        text = '{"missing":true}'
        missing = EmbeddingInput(
            "memory",
            "mem_missing",
            text,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        with self.assertRaisesRegex(ValidationError, "does not exist"):
            self.store.upsert_embeddings(
                fresh_profile,
                (memory_embedding_input(first), missing),
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            )
        self.assertEqual(
            self.store._connection.execute(  # noqa: SLF001
                "SELECT COUNT(*) FROM embedding_profiles "
                "WHERE profile_sha256 = ?",
                (fresh_profile.profile_sha256,),
            ).fetchone()[0],
            0,
        )

    def test_embedding_batch_rejects_stale_input_and_dimension_drift(self) -> None:
        stored = self.store.remember(namespace="n", content="Canonical source")
        embedding_input = memory_embedding_input(stored)
        profile = EmbeddingProfile("test/model", 2, False)
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE memories SET kind = 'changed' WHERE id = ?", (stored.id,)
        )
        self.store._connection.commit()  # noqa: SLF001
        with self.assertRaisesRegex(ValidationError, "digest is stale"):
            self.store.upsert_embeddings(
                profile, (embedding_input,), ((1.0, 0.0),)
            )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE memories SET kind = ? WHERE id = ?", (stored.kind, stored.id)
        )
        self.store._connection.commit()  # noqa: SLF001
        with self.assertRaises(ValueError):
            self.store.upsert_embeddings(
                profile, (embedding_input,), ((1.0, 0.0, 0.0),)
            )
        self.assertEqual(
            self.store._connection.execute(  # noqa: SLF001
                "SELECT COUNT(*) FROM embedding_profiles"
            ).fetchone()[0],
            0,
        )

    def test_semantic_memory_search_ranks_and_filters_before_bounded_scan(self) -> None:
        profile = EmbeddingProfile("test/rank", 2, False)
        first = self.store.remember(
            namespace="tenant/a",
            content="First",
            occurred_at="2026-01-01T00:00:00Z",
        )
        second = self.store.remember(
            namespace="tenant/a",
            content="Second",
            occurred_at="2026-02-01T00:00:00Z",
        )
        third = self.store.remember(
            namespace="tenant/a",
            content="Third",
            occurred_at="2026-03-01T00:00:00Z",
        )
        private = self.store.remember(
            namespace="tenant/b",
            content="Private closest",
            occurred_at="2025-01-01T00:00:00Z",
        )
        self.store.upsert_embeddings(
            profile,
            tuple(
                memory_embedding_input(item)
                for item in (first, second, third, private)
            ),
            ((1.0, 0.0), (0.8, 0.6), (-1.0, 0.0), (1.0, 0.0)),
        )
        result = self.store.semantic_search_memories(
            "tenant/a", (1.0, 0.0), profile, limit=10, scan_cap=10
        )
        self.assertEqual([hit.id for hit in result.hits], [first.id, second.id, third.id])
        self.assertAlmostEqual(result.hits[0].score, 1.0)
        self.assertAlmostEqual(result.hits[1].score, 0.8, places=6)
        self.assertEqual(result.scanned, 3)
        self.assertFalse(result.truncated)
        self.assertFalse(result.diagnostics.no_vectors)

        historical = self.store.semantic_search_memories(
            "tenant/a",
            (1.0, 0.0),
            profile,
            as_of="2026-01-15T00:00:00Z",
            scan_cap=10,
        )
        self.assertEqual([hit.id for hit in historical.hits], [first.id])
        capped = self.store.semantic_search_memories(
            "tenant/a", (1.0, 0.0), profile, limit=10, scan_cap=2
        )
        self.assertEqual(capped.scanned, 2)
        self.assertTrue(capped.truncated)
        zero = self.store.semantic_search_memories(
            "tenant/a", (0.0, 0.0), profile, scan_cap=10
        )
        self.assertTrue(zero.diagnostics.query_is_zero)
        self.assertEqual([hit.score for hit in zero.hits], [0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValidationError, "scan_cap"):
            self.store.semantic_search_memories(
                "tenant/a", (1.0, 0.0), profile, scan_cap=0
            )

        empty = self.store.semantic_search_memories(
            "tenant/missing", (1.0, 0.0), profile
        )
        self.assertTrue(empty.diagnostics.no_vectors)
        unknown = EmbeddingProfile("test/unknown", 2, False)
        with self.assertRaises(EmbeddingProfileNotFoundError):
            self.store.semantic_search_memories(
                "tenant/a", (1.0, 0.0), unknown
            )

    def test_semantic_claim_search_enforces_namespace_and_valid_time_in_sql(self) -> None:
        profile = EmbeddingProfile("test/claims", 2, False)
        old = self.store.remember(
            namespace="tenant/a",
            content="Old plan",
            occurred_at="2026-01-01T00:00:00Z",
            claims=[ClaimInput("account:1", "plan", "bronze")],
        )
        new = self.store.remember(
            namespace="tenant/a",
            content="New plan",
            occurred_at="2026-02-01T00:00:00Z",
            claims=[
                ClaimInput(
                    "account:1",
                    "plan",
                    "silver",
                    supersedes=old.claims[0].id,
                )
            ],
        )
        private = self.store.remember(
            namespace="tenant/b",
            content="Private plan",
            occurred_at="2026-01-01T00:00:00Z",
            claims=[ClaimInput("account:1", "plan", "secret")],
        )
        self.store.upsert_embeddings(
            profile,
            (
                claim_embedding_input(old.claims[0]),
                claim_embedding_input(new.claims[0]),
                claim_embedding_input(private.claims[0]),
            ),
            ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
        )
        current = self.store.semantic_search_claims(
            "tenant/a", (1.0, 0.0), profile, scan_cap=10
        )
        self.assertEqual([hit.claim.value for hit in current.hits], ["silver"])
        self.assertEqual(current.scanned, 1)
        historical = self.store.semantic_search_claims(
            "tenant/a",
            (1.0, 0.0),
            profile,
            as_of="2026-01-15T00:00:00Z",
            scan_cap=10,
        )
        self.assertEqual([hit.claim.value for hit in historical.hits], ["bronze"])
        self.assertEqual(historical.scanned, 1)

    def test_future_successor_boundary_is_consistent_across_claim_channels(
        self,
    ) -> None:
        now = datetime.now(timezone.utc)
        predecessor_start = now - timedelta(days=2)
        boundary = now + timedelta(days=1)
        successor_end = now + timedelta(days=2)
        just_before = boundary - timedelta(microseconds=1)
        predecessor = self.store.remember(
            namespace="tenant/a",
            content="The support schedule uses weekday coverage.",
            occurred_at=predecessor_start,
            claims=[
                ClaimInput(
                    "support:schedule",
                    "coverage",
                    "weekday",
                    valid_from=predecessor_start,
                )
            ],
        )
        successor = self.store.remember(
            namespace="tenant/a",
            content="The support schedule uses summer coverage.",
            occurred_at=now,
            claims=[
                ClaimInput(
                    "support:schedule",
                    "coverage",
                    "summer",
                    valid_from=boundary,
                    valid_to=successor_end,
                    supersedes=predecessor.claims[0].id,
                )
            ],
        )

        persisted_predecessor = self.store.get_claim(
            "tenant/a",
            predecessor.claims[0].id,
        )
        assert persisted_predecessor is not None
        self.assertEqual(persisted_predecessor.status, "superseded")

        def lexical(at: datetime | None) -> list[str]:
            return [
                hit.value
                for hit in self.store.search_claims(
                    "tenant/a",
                    "support schedule coverage",
                    as_of=at,
                )
            ]

        self.assertEqual(lexical(None), ["weekday"])
        self.assertEqual(lexical(just_before), ["weekday"])
        self.assertEqual(lexical(boundary), ["summer"])
        self.assertEqual(lexical(successor_end), [])
        self.assertEqual(lexical(successor_end + timedelta(days=1)), [])

        self.assertEqual(
            [
                hit.value
                for hit in self.store.graph_neighbors(
                    "tenant/a",
                    ("support:schedule",),
                )
            ],
            ["weekday"],
        )
        self.assertEqual(
            [
                hit.value
                for hit in self.store.graph_neighbors(
                    "tenant/a",
                    ("support:schedule",),
                    as_of=boundary,
                )
            ],
            ["summer"],
        )
        self.assertEqual(
            self.store.graph_neighbors(
                "tenant/a",
                ("support:schedule",),
                as_of=successor_end,
            ),
            [],
        )

        profile = EmbeddingProfile("test/future-successor", 2, False)
        self.store.upsert_embeddings(
            profile,
            (
                claim_embedding_input(predecessor.claims[0]),
                claim_embedding_input(successor.claims[0]),
            ),
            ((1.0, 0.0), (0.0, 1.0)),
        )
        current_semantic = self.store.semantic_search_claims(
            "tenant/a",
            (1.0, 0.0),
            profile,
            scan_cap=10,
        )
        boundary_semantic = self.store.semantic_search_claims(
            "tenant/a",
            (0.0, 1.0),
            profile,
            as_of=boundary,
            scan_cap=10,
        )
        expired_semantic = self.store.semantic_search_claims(
            "tenant/a",
            (0.0, 1.0),
            profile,
            as_of=successor_end,
            scan_cap=10,
        )
        self.assertEqual(
            [hit.claim.value for hit in current_semantic.hits],
            ["weekday"],
        )
        self.assertEqual(
            [hit.claim.value for hit in boundary_semantic.hits],
            ["summer"],
        )
        self.assertEqual(expired_semantic.hits, ())

    def test_embedding_sync_reports_missing_and_stale_sources(self) -> None:
        profile = EmbeddingProfile("test/sync", 2, False)
        first = self.store.remember(namespace="n", content="First")
        second = self.store.remember(namespace="n", content="Second")
        unregistered = EmbeddingProfile("test/new-profile", 2, False)
        before_backfill = self.store.embedding_sync_status(
            unregistered, "n", "memory"
        )
        self.assertFalse(before_backfill.registered)
        self.assertFalse(before_backfill.complete)
        self.assertEqual(
            before_backfill.missing_ids, tuple(sorted((first.id, second.id)))
        )
        self.store.upsert_embeddings(
            profile, (memory_embedding_input(first),), ((1.0, 0.0),)
        )
        initial = self.store.embedding_sync_status(profile, "n", "memory")
        self.assertTrue(initial.registered)
        self.assertEqual(initial.missing_ids, (second.id,))
        self.assertEqual(initial.stale_ids, ())
        self.assertEqual(initial.projected_sources, 1)
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE memory_embeddings SET input_sha256 = ? WHERE memory_id = ?",
            ("0" * 64, first.id),
        )
        self.store._connection.commit()  # noqa: SLF001
        damaged = self.store.embedding_sync_status(profile, "n", "memory")
        self.assertEqual(damaged.missing_ids, (second.id,))
        self.assertEqual(damaged.stale_ids, (first.id,))
        self.assertFalse(damaged.complete)
        self.store.upsert_embeddings(
            profile,
            (memory_embedding_input(first), memory_embedding_input(second)),
            ((1.0, 0.0), (0.0, 1.0)),
        )
        self.assertTrue(
            self.store.embedding_sync_status(profile, "n", "memory").complete
        )

    def test_embeddings_current_uses_exact_rows_and_rejects_spoofed_inputs(self) -> None:
        profile = EmbeddingProfile("test/current", 2, False)
        stored = self.store.remember(
            namespace="n",
            content="Current projection",
            claims=(ClaimInput("entity:1", "state", "ready"),),
        )
        inputs = (
            memory_embedding_input(stored),
            claim_embedding_input(stored.claims[0]),
        )
        self.assertFalse(self.store.embeddings_current(profile, inputs))
        self.store.upsert_embeddings(
            profile, inputs, ((1.0, 0.0), (0.0, 1.0))
        )
        self.assertTrue(self.store.embeddings_current(profile, inputs))
        self.store._connection.execute(  # noqa: SLF001
            "DELETE FROM claim_embeddings WHERE claim_id = ?",
            (stored.claims[0].id,),
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertFalse(self.store.embeddings_current(profile, inputs))
        self.store.upsert_embeddings(
            profile, inputs, ((1.0, 0.0), (0.0, 1.0))
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE memory_embeddings SET vector = ? WHERE memory_id = ?",
            (b"\x00", stored.id),
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertFalse(self.store.embeddings_current(profile, inputs))

        other = self.store.remember(namespace="n", content="Different canonical input")
        spoofed = EmbeddingInput(
            "memory",
            other.id,
            inputs[0].text,
            inputs[0].input_sha256,
        )
        with self.assertRaisesRegex(ValidationError, "digest is stale"):
            self.store.embeddings_current(profile, (spoofed,))
        with self.assertRaisesRegex(ValidationError, "does not exist"):
            missing_text = '{"missing":true}'
            self.store.embeddings_current(
                profile,
                (
                    EmbeddingInput(
                        "memory",
                        "mem_missing",
                        missing_text,
                        hashlib.sha256(missing_text.encode()).hexdigest(),
                    ),
                ),
            )

    def test_semantic_integrity_detects_corruption_and_search_fails_closed(self) -> None:
        profile = EmbeddingProfile("test/integrity", 2, False)
        stored = self.store.remember(namespace="n", content="Vector source")
        self.store.upsert_embeddings(
            profile, (memory_embedding_input(stored),), ((1.0, 0.0),)
        )
        self.store._connection.execute(  # noqa: SLF001
            "UPDATE memory_embeddings SET vector = ? WHERE memory_id = ?",
            (b"\x00", stored.id),
        )
        self.store._connection.commit()  # noqa: SLF001
        report = self.store.verify_integrity("n")
        self.assertFalse(report.ok)
        self.assertTrue(
            any("memory_embedding:vector" in issue for issue in report.index_issues)
        )
        with self.assertRaisesRegex(IntegrityViolationError, "invalid embedding"):
            self.store.semantic_search_memories("n", (1.0, 0.0), profile)

        self.store._connection.execute(  # noqa: SLF001
            "UPDATE embedding_profiles SET identity = 'tampered' "
            "WHERE profile_sha256 = ?",
            (profile.profile_sha256,),
        )
        self.store._connection.commit()  # noqa: SLF001
        self.assertFalse(self.store.verify_integrity().ok)
        with self.assertRaises(IntegrityViolationError):
            self.store.semantic_search_memories("n", (1.0, 0.0), profile)

    def test_semantic_rows_survive_rebuild_reopen_and_cascade_on_purge(self) -> None:
        profile = EmbeddingProfile("test/lifecycle", 2, False)
        stored = self.store.remember(
            namespace="n",
            content="Lifecycle source",
            claims=[ClaimInput("entity:1", "state", "live")],
        )
        self.store.upsert_embeddings(
            profile,
            (
                memory_embedding_input(stored),
                claim_embedding_input(stored.claims[0]),
            ),
            ((1.0, 0.0), (0.0, 1.0)),
        )
        self.store.rebuild_indexes()
        counts = self.store._connection.execute(  # noqa: SLF001
            "SELECT (SELECT COUNT(*) FROM memory_embeddings), "
            "(SELECT COUNT(*) FROM claim_embeddings)"
        ).fetchone()
        self.assertEqual(tuple(counts), (1, 1))
        self.store.close()
        self.store = SQLiteMemoryStore(self.database_path)
        self.assertEqual(
            self.store.semantic_search_memories("n", (1.0, 0.0), profile).hits[0].id,
            stored.id,
        )
        self.assertTrue(self.store.purge_memory("n", stored.id))
        counts = self.store._connection.execute(  # noqa: SLF001
            "SELECT (SELECT COUNT(*) FROM memory_embeddings), "
            "(SELECT COUNT(*) FROM claim_embeddings), "
            "(SELECT COUNT(*) FROM embedding_profiles)"
        ).fetchone()
        self.assertEqual(tuple(counts), (0, 0, 1))
        self.assertTrue(self.store.verify_integrity().ok)

    def test_v3_database_migrates_transactionally_to_v5(self) -> None:
        path = Path(self._temporary_directory.name) / "legacy-v3.sqlite3"
        with SQLiteMemoryStore(path) as seed:
            stored = seed.remember(namespace="n", content="V3 survives")
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = OFF;
                DROP TABLE extraction_diagnostics;
                DROP TABLE claim_derivations;
                DROP TABLE extractions;
                DROP INDEX claims_derivation_target;
                DROP TABLE claim_embeddings;
                DROP TABLE memory_embeddings;
                DROP TABLE embedding_profiles;
                PRAGMA user_version = 3;
                """
            )
            connection.commit()
        with SQLiteMemoryStore(path) as migrated:
            self.assertEqual(
                migrated._connection.execute(  # noqa: SLF001
                    "PRAGMA user_version"
                ).fetchone()[0],
                5,
            )
            self.assertEqual(migrated.get_memory("n", stored.id), stored)
            profile = EmbeddingProfile("test/migrated", 2, False)
            migrated.upsert_embeddings(
                profile, (memory_embedding_input(stored),), ((1.0, 0.0),)
            )
            self.assertTrue(migrated.verify_integrity().ok)

    def test_v4_database_migrates_to_v5_without_losing_semantic_rows(self) -> None:
        path = Path(self._temporary_directory.name) / "legacy-v4.sqlite3"
        profile = EmbeddingProfile("test/v4-migration", 2, False)
        with SQLiteMemoryStore(path) as seed:
            stored = seed.remember(namespace="n", content="Semantic row survives")
            seed.upsert_embeddings(
                profile, (memory_embedding_input(stored),), ((1.0, 0.0),)
            )
        _downgrade_fixture_to_v4(path)
        with closing(sqlite3.connect(path)) as legacy:
            self.assertEqual(legacy.execute("PRAGMA user_version").fetchone()[0], 4)

        extractor = CanonicalClaimExtractor()
        content = "No canonical claims in the migrated database."
        extraction = extractor.extract(
            EvidenceContext.from_content(namespace="n", content=content)
        )
        with SQLiteMemoryStore(path) as migrated:
            self.assertEqual(
                migrated._connection.execute(  # noqa: SLF001
                    "PRAGMA user_version"
                ).fetchone()[0],
                5,
            )
            semantic = migrated.semantic_search_memories(
                "n", (1.0, 0.0), profile
            )
            self.assertEqual([hit.id for hit in semantic.hits], [stored.id])
            extracted = migrated.remember(
                namespace="n",
                content=content,
                extraction=extraction,
                extraction_identity=extractor.identity,
            )
            self.assertIsNotNone(migrated.get_extraction("n", extracted.id))
            self.assertTrue(migrated.verify_integrity().ok)
        with SQLiteMemoryStore(path) as reopened:
            self.assertTrue(reopened.verify_integrity().ok)

    def test_v5_open_rejects_damaged_provenance_schema(self) -> None:
        path = Path(self._temporary_directory.name) / "damaged-v5.sqlite3"
        with SQLiteMemoryStore(path):
            pass
        with closing(sqlite3.connect(path)) as damaged:
            damaged.execute("DROP INDEX extraction_diagnostics_namespace_memory")
            damaged.commit()
        with self.assertRaisesRegex(IntegrityViolationError, "schema v5 index"):
            SQLiteMemoryStore(path)
        with closing(sqlite3.connect(path)) as unchanged:
            self.assertEqual(unchanged.execute("PRAGMA user_version").fetchone()[0], 5)

    def test_offset_normalization_preserves_temporal_order(self) -> None:
        start = datetime(2026, 1, 1, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        first = self.store.remember(
            namespace="n",
            content="First value",
            occurred_at=start,
            claims=[ClaimInput("entity:sample", "sample_state", "first")],
        )
        self.store.remember(
            namespace="n",
            content="Second value",
            occurred_at="2026-01-01T05:00:00Z",
            claims=[
                ClaimInput(
                    "entity:sample",
                    "sample_state",
                    "second",
                    supersedes=first.claims[0].id,
                )
            ],
        )
        before = self.store.search_claims(
            "n", "sample state", as_of="2026-01-01T04:30:00Z"
        )
        after = self.store.search_claims(
            "n", "sample state", as_of="2026-01-01T05:00:00Z"
        )
        self.assertEqual([hit.value for hit in before], ["first"])
        self.assertEqual([hit.value for hit in after], ["second"])


if __name__ == "__main__":
    unittest.main()
