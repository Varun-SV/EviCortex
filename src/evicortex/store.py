"""SQLite-backed immutable evidence and temporal claim storage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Iterator
from uuid import uuid4

from .errors import (
    EmbeddingProfileNotFoundError,
    FTS5UnavailableError,
    IdempotencyConflictError,
    IntegrityViolationError,
    StorageClosedError,
    ValidationError,
)
from .models import (
    ClaimDerivationRecord,
    ClaimHit,
    ClaimInput,
    ClaimRecord,
    EmbeddingSyncStatus,
    EmbeddingUpsertResult,
    ExtractionDiagnosticRecord,
    ExtractionRecord,
    IntegrityReport,
    MemoryHit,
    MemoryRecord,
    SemanticClaimHit,
    SemanticClaimSearchResult,
    SemanticMemoryHit,
    SemanticMemorySearchResult,
    SemanticSearchDiagnostics,
    StoreStats,
    TimestampInput,
)
from .plugins import (
    ClaimExtraction,
    ClaimProvenance,
    EvidenceContext,
    ExtractionDiagnostic,
    validate_claim_extraction,
)
from .semantic import (
    EmbeddingInput,
    EmbeddingProfile,
    claim_embedding_input,
    cosine_similarity,
    decode_float32_vector,
    encode_float32_vector,
    memory_embedding_input,
)


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "before",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "s",
        "should",
        "the",
        "to",
        "was",
        "what",
        "when",
        "which",
        "who",
        "with",
    }
)
_NEGATION_WORDS = frozenset({"not", "no", "without", "excluding", "except"})
_ARTICLES = frozenset({"a", "an", "the"})
_CANDIDATE_MULTIPLIER = 16
_MIN_CANDIDATES = 128
_MAX_CANDIDATES = 768
_SCHEMA_VERSION = 5
_TERM_SEPARATOR = "\x1f"

_EXTERNAL_FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE memory_fts USING fts5(
    content,
    kind,
    source,
    content = 'memories',
    content_rowid = 'rowid',
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE claim_fts USING fts5(
    subject,
    predicate,
    value,
    authority,
    content = 'claims',
    content_rowid = 'rowid',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER memories_fts_insert
AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content, kind, source)
    VALUES (new.rowid, new.content, new.kind, new.source);
END;
CREATE TRIGGER memories_fts_delete
AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, kind, source)
    VALUES ('delete', old.rowid, old.content, old.kind, old.source);
END;
CREATE TRIGGER memories_fts_update
AFTER UPDATE OF content, kind, source ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, kind, source)
    VALUES ('delete', old.rowid, old.content, old.kind, old.source);
    INSERT INTO memory_fts(rowid, content, kind, source)
    VALUES (new.rowid, new.content, new.kind, new.source);
END;

CREATE TRIGGER claims_fts_insert
AFTER INSERT ON claims BEGIN
    INSERT INTO claim_fts(rowid, subject, predicate, value, authority)
    VALUES (new.rowid, new.subject, new.predicate, new.value, new.authority);
END;
CREATE TRIGGER claims_fts_delete
AFTER DELETE ON claims BEGIN
    INSERT INTO claim_fts(
        claim_fts, rowid, subject, predicate, value, authority
    ) VALUES (
        'delete', old.rowid, old.subject, old.predicate, old.value, old.authority
    );
END;
CREATE TRIGGER claims_fts_update
AFTER UPDATE OF subject, predicate, value, authority ON claims BEGIN
    INSERT INTO claim_fts(
        claim_fts, rowid, subject, predicate, value, authority
    ) VALUES (
        'delete', old.rowid, old.subject, old.predicate, old.value, old.authority
    );
    INSERT INTO claim_fts(rowid, subject, predicate, value, authority)
    VALUES (new.rowid, new.subject, new.predicate, new.value, new.authority);
END;
"""

_DROP_FTS_SCHEMA_SQL = """
DROP TRIGGER IF EXISTS memories_fts_insert;
DROP TRIGGER IF EXISTS memories_fts_delete;
DROP TRIGGER IF EXISTS memories_fts_update;
DROP TRIGGER IF EXISTS claims_fts_insert;
DROP TRIGGER IF EXISTS claims_fts_delete;
DROP TRIGGER IF EXISTS claims_fts_update;
DROP TABLE IF EXISTS memory_fts;
DROP TABLE IF EXISTS claim_fts;
"""

_SEMANTIC_SCHEMA_SQL = """
CREATE TABLE embedding_profiles (
    profile_sha256 TEXT PRIMARY KEY,
    identity TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    normalized INTEGER NOT NULL CHECK(normalized IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE memory_embeddings (
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    profile_sha256 TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, profile_sha256),
    FOREIGN KEY(memory_id, namespace)
        REFERENCES memories(id, namespace) ON DELETE CASCADE,
    FOREIGN KEY(profile_sha256)
        REFERENCES embedding_profiles(profile_sha256) ON DELETE RESTRICT
);
CREATE INDEX memory_embeddings_namespace_profile
    ON memory_embeddings(namespace, profile_sha256, memory_id);

CREATE TABLE claim_embeddings (
    claim_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    profile_sha256 TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(claim_id, profile_sha256),
    FOREIGN KEY(claim_id, namespace)
        REFERENCES claims(id, namespace) ON DELETE CASCADE,
    FOREIGN KEY(profile_sha256)
        REFERENCES embedding_profiles(profile_sha256) ON DELETE RESTRICT
);
CREATE INDEX claim_embeddings_namespace_profile
    ON claim_embeddings(namespace, profile_sha256, claim_id);
"""

_EXTRACTION_SCHEMA_SQL = """
CREATE UNIQUE INDEX claims_derivation_target
    ON claims(id, memory_id, namespace);

CREATE TABLE extractions (
    memory_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    extractor_identity TEXT NOT NULL
        CHECK(length(trim(extractor_identity)) > 0),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    recorded_at TEXT NOT NULL,
    abstained INTEGER NOT NULL CHECK(abstained IN (0, 1)),
    UNIQUE(memory_id, namespace),
    FOREIGN KEY(memory_id, namespace)
        REFERENCES memories(id, namespace) ON DELETE CASCADE
);
CREATE INDEX extractions_namespace_memory
    ON extractions(namespace, memory_id);

CREATE TABLE claim_derivations (
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    claim_id TEXT NOT NULL UNIQUE,
    claim_index INTEGER NOT NULL CHECK(claim_index >= 0),
    start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
    canonical_text TEXT NOT NULL,
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    extractor_identity TEXT NOT NULL
        CHECK(length(trim(extractor_identity)) > 0),
    PRIMARY KEY(memory_id, claim_index),
    FOREIGN KEY(memory_id, namespace)
        REFERENCES extractions(memory_id, namespace) ON DELETE CASCADE,
    FOREIGN KEY(claim_id, memory_id, namespace)
        REFERENCES claims(id, memory_id, namespace) ON DELETE CASCADE
);
CREATE INDEX claim_derivations_namespace_memory
    ON claim_derivations(namespace, memory_id, claim_index);

CREATE TABLE extraction_diagnostics (
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    diagnostic_index INTEGER NOT NULL CHECK(diagnostic_index >= 0),
    code TEXT NOT NULL CHECK(length(trim(code)) > 0),
    message TEXT NOT NULL CHECK(length(trim(message)) > 0),
    start_offset INTEGER,
    end_offset INTEGER,
    PRIMARY KEY(memory_id, diagnostic_index),
    FOREIGN KEY(memory_id, namespace)
        REFERENCES extractions(memory_id, namespace) ON DELETE CASCADE,
    CHECK(
        (start_offset IS NULL AND end_offset IS NULL)
        OR (
            start_offset IS NOT NULL AND end_offset IS NOT NULL
            AND start_offset >= 0 AND end_offset >= start_offset
        )
    )
);
CREATE INDEX extraction_diagnostics_namespace_memory
    ON extraction_diagnostics(namespace, memory_id, diagnostic_index);
"""


@dataclass(frozen=True, slots=True)
class _PreparedClaim:
    subject: str
    predicate: str
    value: str
    valid_from: str | None
    valid_to: str | None
    confidence: float
    authority: str
    supersedes: str | None


class SQLiteMemoryStore:
    """A local ACID store whose FTS views can be rebuilt from source rows."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(self.path, timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.create_function(
                "evicortex_local_rank",
                3,
                _sqlite_local_rank,
                deterministic=True,
            )
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._initialize_schema()
        except BaseException as exc:
            self._connection.close()
            self._closed = True
            if isinstance(exc, sqlite3.OperationalError) and "fts5" in str(
                exc
            ).casefold():
                raise FTS5UnavailableError("SQLite FTS5 support is required") from exc
            raise

    @classmethod
    def open(cls, path: str | Path = ":memory:") -> SQLiteMemoryStore:
        return cls(path)

    def __enter__(self) -> SQLiteMemoryStore:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def remember(
        self,
        *,
        namespace: str,
        content: str,
        kind: str = "episode",
        source: str = "",
        occurred_at: TimestampInput = None,
        recorded_at: TimestampInput = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        claims: Iterable[ClaimInput] = (),
        extraction: ClaimExtraction | None = None,
        extraction_identity: str | None = None,
    ) -> MemoryRecord:
        """Atomically append evidence, claims, and optional extraction provenance."""

        ns = _required_label(namespace, "namespace")
        normalized_kind = _required_label(kind, "kind")
        if not isinstance(content, str) or content == "":
            raise ValidationError("content must be a non-empty string")
        if not isinstance(source, str):
            raise ValidationError("source must be a string")
        try:
            content_bytes = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValidationError("content must be valid UTF-8 text") from exc
        normalized_key = None
        if idempotency_key is not None:
            normalized_key = _required_label(idempotency_key, "idempotency_key")

        metadata_json = _canonical_metadata(metadata)
        explicit_occurred = _normalize_timestamp(occurred_at)
        explicit_recorded = _normalize_timestamp(recorded_at)
        try:
            supplied_claims = tuple(claims)
        except TypeError as exc:
            raise ValidationError("claims must be an iterable of ClaimInput values") from exc
        validated_extraction: ClaimExtraction | None = None
        normalized_extraction_identity: str | None = None
        if extraction is None:
            if extraction_identity is not None:
                raise ValidationError(
                    "extraction_identity requires an extraction result"
                )
            claim_values = supplied_claims
        else:
            if supplied_claims:
                raise ValidationError(
                    "extraction is mutually exclusive with explicit claims"
                )
            normalized_extraction_identity = _exact_extraction_identity(
                extraction_identity
            )
            try:
                evidence = EvidenceContext.from_content(
                    namespace=ns,
                    content=content,
                    source=source,
                    occurred_at=explicit_occurred,
                    metadata=json.loads(metadata_json),
                )
                validated_extraction = validate_claim_extraction(
                    evidence,
                    extraction,
                    expected_identity=normalized_extraction_identity,
                )
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "extraction is invalid for the supplied evidence context"
                ) from exc
            claim_values = validated_extraction.claims
        prepared_claims = tuple(_prepare_claim(item) for item in claim_values)
        superseded_ids = [item.supersedes for item in prepared_claims if item.supersedes]
        if len(superseded_ids) != len(set(superseded_ids)):
            raise ValidationError("one write cannot supersede the same claim more than once")

        request_hash = _request_digest(
            content=content,
            kind=normalized_kind,
            source=source,
            occurred_at=explicit_occurred,
            recorded_at=explicit_recorded,
            metadata_json=metadata_json,
            claims=prepared_claims,
            extraction=validated_extraction,
            extraction_identity=normalized_extraction_identity,
        )
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        with self._lock:
            self._ensure_open()
            with self._write_transaction():
                if normalized_key is not None:
                    existing = self._connection.execute(
                        "SELECT id, request_hash FROM memories "
                        "WHERE namespace = ? AND idempotency_key = ?",
                        (ns, normalized_key),
                    ).fetchone()
                    if existing is not None:
                        if not hmac.compare_digest(existing["request_hash"], request_hash):
                            raise IdempotencyConflictError(
                                "idempotency key already identifies a different write"
                            )
                        record = self._get_memory_unlocked(ns, existing["id"])
                        if record is None:
                            raise IntegrityViolationError("idempotency row disappeared")
                        return record

                stored_recorded = explicit_recorded or _utc_now()
                stored_occurred = explicit_occurred or stored_recorded
                memory_id = f"mem_{uuid4().hex}"
                self._connection.execute(
                    """
                    INSERT INTO memories(
                        id, namespace, kind, content, content_hash, source,
                        occurred_at, recorded_at, metadata_json,
                        idempotency_key, request_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        ns,
                        normalized_kind,
                        content,
                        content_hash,
                        source,
                        stored_occurred,
                        stored_recorded,
                        metadata_json,
                        normalized_key,
                        request_hash,
                    ),
                )
                generated_claim_ids: list[str] = []
                for item in prepared_claims:
                    valid_from = item.valid_from or stored_occurred
                    if item.valid_to is not None and item.valid_to <= valid_from:
                        raise ValidationError("claim valid_to must be later than valid_from")
                    if item.supersedes is not None:
                        predecessor = self._connection.execute(
                            "SELECT id, subject, predicate, valid_from, status "
                            "FROM claims WHERE namespace = ? AND id = ?",
                            (ns, item.supersedes),
                        ).fetchone()
                        if predecessor is None:
                            raise ValidationError(
                                f"superseded claim does not exist in namespace: {item.supersedes}"
                            )
                        if predecessor["status"] != "active":
                            raise ValidationError("only an active claim can be superseded")
                        if (
                            predecessor["subject"] != item.subject
                            or predecessor["predicate"] != item.predicate
                        ):
                            raise ValidationError(
                                "a claim may supersede only the same subject and predicate"
                            )
                        if valid_from <= predecessor["valid_from"]:
                            raise ValidationError(
                                "a superseding claim must begin after its predecessor"
                            )
                        self._connection.execute(
                            """
                            UPDATE claims
                            SET status = 'superseded',
                                valid_to = CASE
                                    WHEN valid_to IS NULL OR valid_to > ? THEN ?
                                    ELSE valid_to
                                END
                            WHERE namespace = ? AND id = ?
                            """,
                            (valid_from, valid_from, ns, item.supersedes),
                        )

                    claim_id = f"clm_{uuid4().hex}"
                    self._connection.execute(
                        """
                        INSERT INTO claims(
                            id, namespace, memory_id, subject, predicate, value,
                            valid_from, valid_to, declared_valid_to, recorded_at,
                            confidence, authority, supersedes, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                        """,
                        (
                            claim_id,
                            ns,
                            memory_id,
                            item.subject,
                            item.predicate,
                            item.value,
                            valid_from,
                            item.valid_to,
                            item.valid_to,
                            stored_recorded,
                            item.confidence,
                            item.authority,
                            item.supersedes,
                        ),
                    )
                    generated_claim_ids.append(claim_id)
                if validated_extraction is not None:
                    assert normalized_extraction_identity is not None
                    self._connection.execute(
                        """
                        INSERT INTO extractions(
                            memory_id, namespace, extractor_identity,
                            content_sha256, recorded_at, abstained
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            memory_id,
                            ns,
                            normalized_extraction_identity,
                            content_hash,
                            stored_recorded,
                            int(validated_extraction.abstained),
                        ),
                    )
                    for provenance in validated_extraction.provenance:
                        self._connection.execute(
                            """
                            INSERT INTO claim_derivations(
                                memory_id, namespace, claim_id, claim_index,
                                start_offset, end_offset, canonical_text,
                                content_sha256, extractor_identity
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                memory_id,
                                ns,
                                generated_claim_ids[provenance.claim_index],
                                provenance.claim_index,
                                provenance.start,
                                provenance.end,
                                provenance.canonical_text,
                                provenance.content_sha256,
                                provenance.extractor_identity,
                            ),
                        )
                    for diagnostic_index, diagnostic in enumerate(
                        validated_extraction.diagnostics
                    ):
                        self._connection.execute(
                            """
                            INSERT INTO extraction_diagnostics(
                                memory_id, namespace, diagnostic_index, code,
                                message, start_offset, end_offset
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                memory_id,
                                ns,
                                diagnostic_index,
                                diagnostic.code,
                                diagnostic.message,
                                diagnostic.start,
                                diagnostic.end,
                            ),
                        )
                record = self._get_memory_unlocked(ns, memory_id)
                if record is None:
                    raise IntegrityViolationError("newly inserted memory disappeared")
                return record

    def get_memory(self, namespace: str, id: str) -> MemoryRecord | None:
        ns = _required_label(namespace, "namespace")
        memory_id = _required_label(id, "id")
        with self._lock:
            self._ensure_open()
            return self._get_memory_unlocked(ns, memory_id)

    def get_claim(self, namespace: str, id: str) -> ClaimRecord | None:
        """Return one claim only when it belongs to the requested namespace."""

        ns = _required_label(namespace, "namespace")
        claim_id = _required_label(id, "id")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM claims WHERE namespace = ? AND id = ?",
                (ns, claim_id),
            ).fetchone()
            return None if row is None else self._claim_from_row(row)

    def get_extraction(
        self, namespace: str, memory_id: str
    ) -> ExtractionRecord | None:
        """Return one extraction run without disclosing across namespaces."""

        ns = _required_label(namespace, "namespace")
        target = _required_label(memory_id, "memory_id")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM extractions WHERE namespace = ? AND memory_id = ?",
                (ns, target),
            ).fetchone()
            return None if row is None else self._extraction_from_row(row)

    def upsert_embeddings(
        self,
        profile: EmbeddingProfile,
        inputs: Sequence[EmbeddingInput],
        vectors: Sequence[Sequence[float]],
    ) -> EmbeddingUpsertResult:
        """Persist one aligned memory/claim projection batch atomically.

        Namespace and claim-to-memory ownership are derived from the persisted
        targets.  Every supplied input digest is checked against a freshly
        reconstructed canonical input before any part of the batch commits.
        """

        checked_profile = _checked_embedding_profile(profile)
        frozen_inputs = tuple(inputs)
        frozen_vectors = tuple(vectors)
        if not frozen_inputs:
            raise ValidationError("embedding batch must contain at least one input")
        if len(frozen_inputs) != len(frozen_vectors):
            raise ValidationError("embedding inputs and vectors must have equal length")
        keys: list[tuple[str, str]] = []
        for item in frozen_inputs:
            if not isinstance(item, EmbeddingInput):
                raise ValidationError(
                    "embedding inputs must contain EmbeddingInput values"
                )
            keys.append((item.target_kind, item.target_id))
        if len(keys) != len(set(keys)):
            raise ValidationError("embedding batch contains a duplicate target")

        memory_count = 0
        claim_count = 0
        namespaces: set[str] = set()
        with self._lock:
            self._ensure_open()
            with self._write_transaction():
                self._upsert_embedding_profile_unlocked(checked_profile)
                updated_at = _utc_now()
                for item, vector in zip(
                    frozen_inputs, frozen_vectors, strict=True
                ):
                    namespace, canonical = (
                        self._embedding_input_for_target_unlocked(
                            item.target_kind, item.target_id
                        )
                    )
                    if not _digest_matches(
                        canonical.input_sha256, item.input_sha256
                    ):
                        raise ValidationError(
                            "embedding input digest is stale for "
                            f"{item.target_kind} {item.target_id}"
                        )
                    payload = encode_float32_vector(
                        vector,
                        dimensions=checked_profile.dimensions,
                        normalized=checked_profile.normalized,
                        allow_zero=True,
                    )
                    namespaces.add(namespace)
                    if item.target_kind == "memory":
                        self._connection.execute(
                            """
                            INSERT INTO memory_embeddings(
                                memory_id, namespace, profile_sha256,
                                input_sha256, vector, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(memory_id, profile_sha256) DO UPDATE SET
                                namespace = excluded.namespace,
                                input_sha256 = excluded.input_sha256,
                                vector = excluded.vector,
                                updated_at = excluded.updated_at
                            """,
                            (
                                item.target_id,
                                namespace,
                                checked_profile.profile_sha256,
                                canonical.input_sha256,
                                payload,
                                updated_at,
                            ),
                        )
                        memory_count += 1
                    else:
                        self._connection.execute(
                            """
                            INSERT INTO claim_embeddings(
                                claim_id, namespace, profile_sha256,
                                input_sha256, vector, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(claim_id, profile_sha256) DO UPDATE SET
                                namespace = excluded.namespace,
                                input_sha256 = excluded.input_sha256,
                                vector = excluded.vector,
                                updated_at = excluded.updated_at
                            """,
                            (
                                item.target_id,
                                namespace,
                                checked_profile.profile_sha256,
                                canonical.input_sha256,
                                payload,
                                updated_at,
                            ),
                        )
                        claim_count += 1
        return EmbeddingUpsertResult(
            profile=checked_profile,
            written=len(frozen_inputs),
            memory_embeddings=memory_count,
            claim_embeddings=claim_count,
            namespaces=tuple(sorted(namespaces)),
        )

    def embedding_sync_status(
        self,
        profile: EmbeddingProfile,
        namespace: str,
        target_kind: str,
    ) -> EmbeddingSyncStatus:
        """Report source rows whose selected semantic projection needs syncing."""

        checked_profile = _checked_embedding_profile(profile)
        ns = _required_label(namespace, "namespace")
        kind = _embedding_target_kind(target_kind)
        with self._lock:
            self._ensure_open()
            registered = self._connection.execute(
                "SELECT 1 FROM embedding_profiles WHERE profile_sha256 = ?",
                (checked_profile.profile_sha256,),
            ).fetchone() is not None
            if registered:
                self._require_persisted_profile_unlocked(checked_profile)
            if kind == "memory":
                rows = self._connection.execute(
                    """
                    SELECT m.*,
                           e.input_sha256 AS _embedding_input_sha256,
                           e.vector AS _embedding_vector
                    FROM memories AS m
                    LEFT JOIN memory_embeddings AS e
                      ON e.memory_id = m.id AND e.namespace = m.namespace
                     AND e.profile_sha256 = ?
                    WHERE m.namespace = ?
                    ORDER BY m.id
                    """,
                    (checked_profile.profile_sha256, ns),
                ).fetchall()
                id_column = "id"
            else:
                rows = self._connection.execute(
                    """
                    SELECT c.*,
                           e.input_sha256 AS _embedding_input_sha256,
                           e.vector AS _embedding_vector
                    FROM claims AS c
                    LEFT JOIN claim_embeddings AS e
                      ON e.claim_id = c.id AND e.namespace = c.namespace
                     AND e.profile_sha256 = ?
                    WHERE c.namespace = ?
                    ORDER BY c.id
                    """,
                    (checked_profile.profile_sha256, ns),
                ).fetchall()
                id_column = "id"

            missing: list[str] = []
            stale: list[str] = []
            for row in rows:
                target_id = row[id_column]
                if row["_embedding_input_sha256"] is None:
                    missing.append(target_id)
                    continue
                try:
                    expected = self._canonical_embedding_input_from_row(kind, row)
                    decode_float32_vector(
                        row["_embedding_vector"],
                        dimensions=checked_profile.dimensions,
                        normalized=checked_profile.normalized,
                        allow_zero=True,
                    )
                except (TypeError, ValueError, IntegrityViolationError):
                    stale.append(target_id)
                    continue
                if not _digest_matches(
                    expected.input_sha256, row["_embedding_input_sha256"]
                ):
                    stale.append(target_id)
            return EmbeddingSyncStatus(
                profile=checked_profile,
                namespace=ns,
                target_kind=kind,
                total_sources=len(rows),
                projected_sources=len(rows) - len(missing) - len(stale),
                missing_ids=tuple(missing),
                stale_ids=tuple(stale),
                registered=registered,
            )

    def embeddings_current(
        self,
        profile: EmbeddingProfile,
        inputs: Sequence[EmbeddingInput],
    ) -> bool:
        """Check a small exact projection set without scanning its namespace.

        Invalid or spoofed source inputs are caller errors and raise.  A valid
        source whose profile/row/digest/vector needs registration or repair
        returns ``False`` so an orchestrator can safely call its provider.
        """

        checked_profile = _checked_embedding_profile(profile)
        frozen_inputs = tuple(inputs)
        if not frozen_inputs:
            raise ValidationError("embedding input set must not be empty")
        keys: list[tuple[str, str]] = []
        for item in frozen_inputs:
            if not isinstance(item, EmbeddingInput):
                raise ValidationError(
                    "embedding inputs must contain EmbeddingInput values"
                )
            keys.append((item.target_kind, item.target_id))
        if len(keys) != len(set(keys)):
            raise ValidationError("embedding input set contains a duplicate target")

        with self._lock:
            self._ensure_open()
            resolved: list[tuple[EmbeddingInput, str, EmbeddingInput]] = []
            for item in frozen_inputs:
                namespace, canonical = self._embedding_input_for_target_unlocked(
                    item.target_kind, item.target_id
                )
                if not _digest_matches(
                    canonical.input_sha256, item.input_sha256
                ):
                    raise ValidationError(
                        "embedding input digest is stale for "
                        f"{item.target_kind} {item.target_id}"
                    )
                resolved.append((item, namespace, canonical))

            profile_row = self._connection.execute(
                "SELECT 1 FROM embedding_profiles WHERE profile_sha256 = ?",
                (checked_profile.profile_sha256,),
            ).fetchone()
            if profile_row is None:
                return False
            self._require_persisted_profile_unlocked(checked_profile)
            for item, namespace, canonical in resolved:
                if item.target_kind == "memory":
                    row = self._connection.execute(
                        "SELECT input_sha256, vector FROM memory_embeddings "
                        "WHERE namespace = ? AND memory_id = ? "
                        "AND profile_sha256 = ?",
                        (
                            namespace,
                            item.target_id,
                            checked_profile.profile_sha256,
                        ),
                    ).fetchone()
                else:
                    row = self._connection.execute(
                        "SELECT input_sha256, vector FROM claim_embeddings "
                        "WHERE namespace = ? AND claim_id = ? "
                        "AND profile_sha256 = ?",
                        (
                            namespace,
                            item.target_id,
                            checked_profile.profile_sha256,
                        ),
                    ).fetchone()
                if row is None or not _digest_matches(
                    canonical.input_sha256, row["input_sha256"]
                ):
                    return False
                try:
                    decode_float32_vector(
                        row["vector"],
                        dimensions=checked_profile.dimensions,
                        normalized=checked_profile.normalized,
                        allow_zero=True,
                    )
                except (TypeError, ValueError):
                    return False
            return True

    def semantic_search_memories(
        self,
        namespace: str,
        query_vector: Sequence[float],
        profile: EmbeddingProfile,
        limit: int = 10,
        *,
        scan_cap: int = 1024,
        as_of: TimestampInput = None,
    ) -> SemanticMemorySearchResult:
        """Rank a bounded, SQL-scoped memory projection scan by exact cosine."""

        ns = _required_label(namespace, "namespace")
        size = _positive_limit(limit)
        cap = _positive_scan_cap(scan_cap)
        normalized_as_of = _normalize_timestamp(as_of)
        checked_profile = _checked_embedding_profile(profile)
        query = _float32_query_vector(query_vector, checked_profile)
        query_is_zero = all(value == 0.0 for value in query)
        with self._lock:
            self._ensure_open()
            self._require_persisted_profile_unlocked(checked_profile)
            if normalized_as_of is None:
                rows = self._connection.execute(
                    """
                    SELECT m.*, e.input_sha256 AS _embedding_input_sha256,
                           e.vector AS _embedding_vector
                    FROM memory_embeddings AS e
                    JOIN memories AS m
                      ON m.id = e.memory_id AND m.namespace = e.namespace
                    WHERE e.namespace = ? AND e.profile_sha256 = ?
                    ORDER BY m.source, m.occurred_at, m.recorded_at,
                             m.content_hash, m.kind, m.idempotency_key,
                             e.memory_id
                    LIMIT ?
                    """,
                    (ns, checked_profile.profile_sha256, cap + 1),
                ).fetchall()
                temporal_reason = "all-time"
            else:
                rows = self._connection.execute(
                    """
                    SELECT m.*, e.input_sha256 AS _embedding_input_sha256,
                           e.vector AS _embedding_vector
                    FROM memory_embeddings AS e
                    JOIN memories AS m
                      ON m.id = e.memory_id AND m.namespace = e.namespace
                    WHERE e.namespace = ? AND e.profile_sha256 = ?
                      AND m.occurred_at <= ?
                    ORDER BY m.source, m.occurred_at, m.recorded_at,
                             m.content_hash, m.kind, m.idempotency_key,
                             e.memory_id
                    LIMIT ?
                    """,
                    (
                        ns,
                        checked_profile.profile_sha256,
                        normalized_as_of,
                        cap + 1,
                    ),
                ).fetchall()
                temporal_reason = f"as-of={normalized_as_of}"
            truncated = len(rows) > cap
            scanned_rows = rows[:cap]
            scored: list[tuple[sqlite3.Row, float]] = []
            for row in scanned_rows:
                stored = self._validated_projection_vector_from_row(
                    "memory", row, checked_profile
                )
                scored.append(
                    (
                        row,
                        cosine_similarity(
                            query,
                            stored,
                            normalized=checked_profile.normalized,
                        ),
                    )
                )
            scored.sort(
                key=lambda item: (-item[1], *_memory_row_tie_key(item[0]))
            )
            selected = scored[:size]
            records = {
                record.id: record
                for record in self._memory_records_from_rows(
                    [row for row, _score in selected]
                )
            }
            hits = tuple(
                SemanticMemoryHit(
                    memory=records[row["id"]],
                    score=score,
                    reason=(
                        "semantic:exact-cosine;namespace-scoped;"
                        f"{temporal_reason};profile="
                        f"{checked_profile.profile_sha256}"
                    ),
                )
                for row, score in selected
            )
            diagnostics = SemanticSearchDiagnostics(
                profile=checked_profile,
                scanned=len(scanned_rows),
                scan_cap=cap,
                truncated=truncated,
                query_is_zero=query_is_zero,
                no_vectors=not scanned_rows,
            )
            return SemanticMemorySearchResult(hits, diagnostics)

    def semantic_search_claims(
        self,
        namespace: str,
        query_vector: Sequence[float],
        profile: EmbeddingProfile,
        as_of: TimestampInput = None,
        limit: int = 10,
        *,
        scan_cap: int = 1024,
    ) -> SemanticClaimSearchResult:
        """Rank a bounded, temporally filtered claim scan by exact cosine."""

        ns = _required_label(namespace, "namespace")
        size = _positive_limit(limit)
        cap = _positive_scan_cap(scan_cap)
        normalized_as_of = _normalize_timestamp(as_of)
        checked_profile = _checked_embedding_profile(profile)
        query = _float32_query_vector(query_vector, checked_profile)
        query_is_zero = all(value == 0.0 for value in query)
        with self._lock:
            self._ensure_open()
            self._require_persisted_profile_unlocked(checked_profile)
            if normalized_as_of is None:
                cutoff = _utc_now()
                rows = self._connection.execute(
                    """
                    SELECT c.*, e.input_sha256 AS _embedding_input_sha256,
                           e.vector AS _embedding_vector
                    FROM claim_embeddings AS e
                    JOIN claims AS c
                      ON c.id = e.claim_id AND c.namespace = e.namespace
                    WHERE e.namespace = ? AND e.profile_sha256 = ?
                      AND c.valid_from <= ?
                      AND (c.valid_to IS NULL OR c.valid_to > ?)
                    ORDER BY c.subject, c.predicate, c.value, c.valid_from,
                             c.valid_to, c.recorded_at, c.authority,
                             e.claim_id
                    LIMIT ?
                    """,
                    (
                        ns,
                        checked_profile.profile_sha256,
                        cutoff,
                        cutoff,
                        cap + 1,
                    ),
                ).fetchall()
                temporal_reason = "current"
            else:
                rows = self._connection.execute(
                    """
                    SELECT c.*, e.input_sha256 AS _embedding_input_sha256,
                           e.vector AS _embedding_vector
                    FROM claim_embeddings AS e
                    JOIN claims AS c
                      ON c.id = e.claim_id AND c.namespace = e.namespace
                    WHERE e.namespace = ? AND e.profile_sha256 = ?
                      AND c.valid_from <= ?
                      AND (c.valid_to IS NULL OR c.valid_to > ?)
                    ORDER BY c.subject, c.predicate, c.value, c.valid_from,
                             c.valid_to, c.recorded_at, c.authority,
                             e.claim_id
                    LIMIT ?
                    """,
                    (
                        ns,
                        checked_profile.profile_sha256,
                        normalized_as_of,
                        normalized_as_of,
                        cap + 1,
                    ),
                ).fetchall()
                temporal_reason = f"as-of={normalized_as_of}"
            truncated = len(rows) > cap
            scanned_rows = rows[:cap]
            scored: list[tuple[ClaimRecord, float]] = []
            for row in scanned_rows:
                stored = self._validated_projection_vector_from_row(
                    "claim", row, checked_profile
                )
                scored.append(
                    (
                        self._claim_from_row(row),
                        cosine_similarity(
                            query,
                            stored,
                            normalized=checked_profile.normalized,
                        ),
                    )
                )
            scored.sort(key=lambda item: (-item[1], *_claim_tie_key(item[0])))
            hits = tuple(
                SemanticClaimHit(
                    claim=claim,
                    score=score,
                    reason=(
                        "semantic:exact-cosine;namespace-scoped;"
                        f"{temporal_reason};profile="
                        f"{checked_profile.profile_sha256}"
                    ),
                )
                for claim, score in scored[:size]
            )
            diagnostics = SemanticSearchDiagnostics(
                profile=checked_profile,
                scanned=len(scanned_rows),
                scan_cap=cap,
                truncated=truncated,
                query_is_zero=query_is_zero,
                no_vectors=not scanned_rows,
            )
            return SemanticClaimSearchResult(hits, diagnostics)

    def search_memories(
        self,
        namespace: str,
        query: str,
        limit: int = 10,
        *,
        as_of: TimestampInput = None,
    ) -> list[MemoryHit]:
        ns = _required_label(namespace, "namespace")
        query_terms, excluded_terms, fts_query = _prepare_fts_query(query)
        size = _positive_limit(limit)
        candidate_pool = _candidate_pool_size(size)
        normalized_as_of = _normalize_timestamp(as_of)
        if not query_terms:
            return []
        positive_blob = _encode_terms(query_terms)
        excluded_blob = _encode_terms(excluded_terms)
        with self._lock:
            self._ensure_open()
            if normalized_as_of is None:
                rows = self._connection.execute(
                    """
                    SELECT m.*, evicortex_local_rank(
                        m.content || ' ' || m.kind || ' ' || m.source, ?, ?
                    ) AS _local_rank
                    FROM memory_fts AS f
                    JOIN memories AS m ON m.rowid = f.rowid
                    WHERE m.namespace = ?
                      AND memory_fts MATCH ?
                    ORDER BY _local_rank DESC, m.occurred_at DESC,
                             m.source, m.content_hash, m.kind, m.recorded_at,
                             m.id
                    LIMIT ?
                    """,
                    (positive_blob, excluded_blob, ns, fts_query, candidate_pool),
                ).fetchall()
                temporal_reason = "all-time"
            else:
                rows = self._connection.execute(
                    """
                    SELECT m.*, evicortex_local_rank(
                        m.content || ' ' || m.kind || ' ' || m.source, ?, ?
                    ) AS _local_rank
                    FROM memory_fts AS f
                    JOIN memories AS m ON m.rowid = f.rowid
                    WHERE m.namespace = ?
                      AND m.occurred_at <= ? AND memory_fts MATCH ?
                    ORDER BY _local_rank DESC, m.occurred_at DESC,
                             m.source, m.content_hash, m.kind, m.recorded_at,
                             m.id
                    LIMIT ?
                    """,
                    (
                        positive_blob,
                        excluded_blob,
                        ns,
                        normalized_as_of,
                        fts_query,
                        candidate_pool,
                    ),
                ).fetchall()
                temporal_reason = f"as-of={normalized_as_of}"
            scored: list[tuple[sqlite3.Row, tuple[int, float, int, str]]] = []
            for row in rows:
                document_terms = _tokenize(
                    " ".join((row["content"], row["kind"], row["source"]))
                )
                coverage, frequency, excluded_matches = _lexical_score(
                    query_terms, document_terms, excluded_terms
                )
                scored.append(
                    (
                        row,
                        (-excluded_matches, coverage, frequency, row["occurred_at"]),
                    )
                )
            scored.sort(key=lambda item: _memory_row_tie_key(item[0]))
            scored.sort(key=lambda item: item[1], reverse=True)
            selected = scored[:size]
            records = {
                record.id: record
                for record in self._memory_records_from_rows(
                    [row for row, _score in selected]
                )
            }
            return [
                MemoryHit(
                    memory=records[row["id"]],
                    rank=float(position),
                    reason=(
                        "fts5:memory;namespace-scoped;"
                        f"{temporal_reason};"
                        f"coverage={score[1]:.3f};term_frequency={score[2]};"
                        f"excluded_matches={-score[0]}"
                    ),
                )
                for position, (row, score) in enumerate(selected, start=1)
            ]

    def search_claims(
        self,
        namespace: str,
        query: str,
        as_of: TimestampInput = None,
        limit: int = 10,
    ) -> list[ClaimHit]:
        ns = _required_label(namespace, "namespace")
        query_terms, excluded_terms, fts_query = _prepare_fts_query(query)
        size = _positive_limit(limit)
        candidate_pool = _candidate_pool_size(size)
        normalized_as_of = _normalize_timestamp(as_of)
        if not query_terms:
            return []
        positive_blob = _encode_terms(query_terms)
        excluded_blob = _encode_terms(excluded_terms)
        with self._lock:
            self._ensure_open()
            if normalized_as_of is None:
                current_time = _utc_now()
                rows = self._connection.execute(
                    """
                    SELECT c.*, evicortex_local_rank(
                        c.subject || ' ' || c.predicate || ' ' ||
                        c.value || ' ' || c.authority, ?, ?
                    ) AS _local_rank
                    FROM claim_fts AS f
                    JOIN claims AS c ON c.rowid = f.rowid
                    WHERE c.namespace = ?
                      AND c.valid_from <= ?
                      AND (c.valid_to IS NULL OR c.valid_to > ?)
                      AND claim_fts MATCH ?
                    ORDER BY _local_rank DESC, c.confidence DESC,
                             c.valid_from DESC, c.subject, c.predicate,
                             c.value, c.authority, c.recorded_at, c.id
                    LIMIT ?
                    """,
                    (
                        positive_blob,
                        excluded_blob,
                        ns,
                        current_time,
                        current_time,
                        fts_query,
                        candidate_pool,
                    ),
                ).fetchall()
                # The exact wall-clock cutoff is an internal SQL parameter. Keeping it
                # out of the public reason makes identical reads deterministic while
                # still describing the temporal policy that was applied.
                temporal_reason = "current"
            else:
                rows = self._connection.execute(
                    """
                    SELECT c.*, evicortex_local_rank(
                        c.subject || ' ' || c.predicate || ' ' ||
                        c.value || ' ' || c.authority, ?, ?
                    ) AS _local_rank
                    FROM claim_fts AS f
                    JOIN claims AS c ON c.rowid = f.rowid
                    WHERE c.namespace = ?
                      AND c.valid_from <= ?
                      AND (c.valid_to IS NULL OR c.valid_to > ?)
                      AND claim_fts MATCH ?
                    ORDER BY _local_rank DESC, c.confidence DESC,
                             c.valid_from DESC, c.subject, c.predicate,
                             c.value, c.authority, c.recorded_at, c.id
                    LIMIT ?
                    """,
                    (
                        positive_blob,
                        excluded_blob,
                        ns,
                        normalized_as_of,
                        normalized_as_of,
                        fts_query,
                        candidate_pool,
                    ),
                ).fetchall()
                temporal_reason = f"as-of={normalized_as_of}"
            scored: list[
                tuple[ClaimRecord, tuple[int, float, int, float, str]]
            ] = []
            for row in rows:
                claim = self._claim_from_row(row)
                document_terms = _tokenize(
                    " ".join(
                        (claim.subject, claim.predicate, claim.value, claim.authority)
                    )
                )
                coverage, frequency, excluded_matches = _lexical_score(
                    query_terms, document_terms, excluded_terms
                )
                scored.append(
                    (
                        claim,
                        (
                            -excluded_matches,
                            coverage,
                            frequency,
                            claim.confidence,
                            claim.valid_from,
                        ),
                    )
                )
            scored.sort(key=lambda item: _claim_tie_key(item[0]))
            scored.sort(key=lambda item: item[1], reverse=True)
            return [
                ClaimHit(
                    claim=claim,
                    rank=float(position),
                    reason=(
                        "fts5:claim;namespace-scoped;"
                        f"{temporal_reason};coverage={score[1]:.3f};"
                        f"term_frequency={score[2]};excluded_matches={-score[0]}"
                    ),
                )
                for position, (claim, score) in enumerate(scored[:size], start=1)
            ]

    def graph_neighbors(
        self,
        namespace: str,
        seed_terms: Sequence[str],
        limit: int = 10,
        *,
        as_of: TimestampInput = None,
    ) -> list[ClaimHit]:
        ns = _required_label(namespace, "namespace")
        size = _positive_limit(limit)
        normalized_as_of = _normalize_timestamp(as_of)
        if isinstance(seed_terms, str):
            raw_seeds: Sequence[str] = (seed_terms,)
        else:
            raw_seeds = seed_terms
        canonical_seeds = tuple(
            dict.fromkeys(
                seed.strip()
                for seed in raw_seeds
                if isinstance(seed, str) and seed.strip()
            )
        )
        entity_seeds = tuple(seed for seed in canonical_seeds if ":" in seed)
        if entity_seeds:
            placeholders = ", ".join("?" for _ in entity_seeds)
            if normalized_as_of is None:
                cutoff = _utc_now()
                status_clause = ""
                temporal_reason = "current"
            else:
                cutoff = normalized_as_of
                status_clause = ""
                temporal_reason = f"as-of={normalized_as_of}"
            sql = f"""
                SELECT * FROM claims
                WHERE namespace = ? {status_clause}
                  AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                  AND (subject IN ({placeholders}) OR value IN ({placeholders}))
                ORDER BY confidence DESC, valid_from DESC,
                         subject, predicate, value, authority, recorded_at, id
                LIMIT ?
            """
            params: tuple[object, ...] = (
                ns,
                cutoff,
                cutoff,
                *entity_seeds,
                *entity_seeds,
                size,
            )
            with self._lock:
                self._ensure_open()
                rows = self._connection.execute(sql, params).fetchall()
                return [
                    ClaimHit(
                        claim=self._claim_from_row(row),
                        rank=float(position),
                        reason=(
                            "graph:exact-entity;namespace-scoped;"
                            f"{temporal_reason}"
                        ),
                    )
                    for position, row in enumerate(rows, start=1)
                ]
        normalized = tuple(
            dict.fromkeys(
                token
                for seed in canonical_seeds
                for token in _tokenize(seed)
            )
        )
        if not normalized:
            return []
        candidates = self.search_claims(
            ns,
            " ".join(normalized),
            as_of=normalized_as_of,
            limit=max(size * 8, 64),
        )
        seed_set = set(normalized)
        scored: list[tuple[ClaimHit, int]] = []
        for hit in candidates:
            claim_terms = set(
                _tokenize(
                    " ".join((hit.subject, hit.predicate, hit.value, hit.claim.authority))
                )
            )
            overlap = len(seed_set & claim_terms)
            if overlap:
                scored.append((hit, overlap))
        scored.sort(key=lambda item: _claim_tie_key(item[0].claim))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            ClaimHit(
                claim=hit.claim,
                rank=float(position),
                reason=(
                    f"graph:shared-terms={overlap};namespace-scoped;"
                    + (
                        f"as-of={normalized_as_of}"
                        if normalized_as_of is not None
                        else "current"
                    )
                ),
            )
            for position, (hit, overlap) in enumerate(scored[:size], start=1)
        ]

    def verify_integrity(self, namespace: str | None = None) -> IntegrityReport:
        ns = None if namespace is None else _required_label(namespace, "namespace")
        with self._lock:
            self._ensure_open()
            clause = "" if ns is None else " WHERE namespace = ?"
            params: tuple[object, ...] = () if ns is None else (ns,)
            memories = self._connection.execute(
                "SELECT id, namespace, content, content_hash, kind, source FROM memories"
                + clause,
                params,
            ).fetchall()
            corrupt: list[str] = []
            for row in memories:
                try:
                    actual = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
                except (AttributeError, UnicodeEncodeError):
                    actual = ""
                if not hmac.compare_digest(actual, row["content_hash"]):
                    corrupt.append(row["id"])

            issues: list[str] = []
            claims = self._connection.execute(
                "SELECT id, namespace, subject, predicate, value, authority, "
                "supersedes, valid_from, valid_to, declared_valid_to, "
                "redacted_valid_to, status FROM claims"
                + clause,
                params,
            ).fetchall()
            expected_projection = _expected_claim_projections(claims)
            for row in claims:
                expected_status, expected_valid_to = expected_projection[row["id"]]
                if (
                    row["status"] != expected_status
                    or row["valid_to"] != expected_valid_to
                ):
                    issues.append(f"claim_projection:mismatch:{row['id']}")
            self._check_external_fts_integrity_unlocked("memory_fts", issues)
            self._check_external_fts_integrity_unlocked("claim_fts", issues)
            self._check_semantic_integrity_unlocked(ns, issues)
            self._check_extraction_integrity_unlocked(ns, issues)

            if ns is None:
                quick_check = self._connection.execute("PRAGMA quick_check").fetchone()[0]
                if quick_check != "ok":
                    issues.append(f"sqlite:{quick_check}")
                foreign_keys = self._connection.execute("PRAGMA foreign_key_check").fetchall()
                if foreign_keys:
                    issues.append(f"foreign_keys:{len(foreign_keys)}")
            return IntegrityReport(
                ok=not corrupt and not issues,
                checked_memories=len(memories),
                corrupt_memory_ids=tuple(sorted(corrupt)),
                index_issues=tuple(sorted(issues)),
            )

    def _check_external_fts_integrity_unlocked(
        self, table: str, issues: list[str]
    ) -> None:
        if table not in {"memory_fts", "claim_fts"}:
            raise AssertionError("unexpected FTS table")
        self._connection.execute("SAVEPOINT evicortex_fts_integrity")
        try:
            self._connection.execute(
                f"INSERT INTO {table}({table}, rank) VALUES (?, ?)",
                ("integrity-check", 1),
            )
        except sqlite3.DatabaseError as exc:
            self._connection.execute("ROLLBACK TO evicortex_fts_integrity")
            self._connection.execute("RELEASE evicortex_fts_integrity")
            issues.append(f"{table}:integrity:{exc}")
        else:
            self._connection.execute("RELEASE evicortex_fts_integrity")

    def _check_semantic_integrity_unlocked(
        self, namespace: str | None, issues: list[str]
    ) -> None:
        if namespace is None:
            profile_rows = self._connection.execute(
                "SELECT * FROM embedding_profiles ORDER BY profile_sha256"
            ).fetchall()
        else:
            referenced = {
                row[0]
                for table in ("memory_embeddings", "claim_embeddings")
                for row in self._connection.execute(
                    f"SELECT DISTINCT profile_sha256 FROM {table} "
                    "WHERE namespace = ?",
                    (namespace,),
                )
            }
            if referenced:
                placeholders = ", ".join("?" for _ in referenced)
                profile_rows = self._connection.execute(
                    "SELECT * FROM embedding_profiles WHERE profile_sha256 IN "
                    f"({placeholders}) ORDER BY profile_sha256",
                    tuple(sorted(referenced)),
                ).fetchall()
            else:
                profile_rows = []
        profiles: dict[str, EmbeddingProfile] = {}
        for row in profile_rows:
            profile_sha256 = row["profile_sha256"]
            try:
                profiles[profile_sha256] = _embedding_profile_from_row(row)
            except (TypeError, ValueError, IntegrityViolationError):
                issues.append(f"embedding_profile:invalid:{profile_sha256}")

        clause = "" if namespace is None else " WHERE namespace = ?"
        params: tuple[object, ...] = () if namespace is None else (namespace,)
        for target_kind, table, id_column in (
            ("memory", "memory_embeddings", "memory_id"),
            ("claim", "claim_embeddings", "claim_id"),
        ):
            projection_rows = self._connection.execute(
                f"SELECT * FROM {table}{clause} "
                f"ORDER BY namespace, {id_column}, profile_sha256",
                params,
            ).fetchall()
            for row in projection_rows:
                target_id = row[id_column]
                profile_sha256 = row["profile_sha256"]
                profile = profiles.get(profile_sha256)
                if profile is None:
                    issues.append(
                        f"{target_kind}_embedding:profile:{target_id}:"
                        f"{profile_sha256}"
                    )
                    continue
                try:
                    source_namespace, expected = (
                        self._embedding_input_for_target_unlocked(
                            target_kind, target_id
                        )
                    )
                except (TypeError, ValueError, IntegrityViolationError):
                    issues.append(
                        f"{target_kind}_embedding:source:{target_id}:"
                        f"{profile_sha256}"
                    )
                    continue
                if source_namespace != row["namespace"]:
                    issues.append(
                        f"{target_kind}_embedding:namespace:{target_id}:"
                        f"{profile_sha256}"
                    )
                if not _digest_matches(
                    expected.input_sha256, row["input_sha256"]
                ):
                    issues.append(
                        f"{target_kind}_embedding:input:{target_id}:"
                        f"{profile_sha256}"
                    )
                try:
                    decode_float32_vector(
                        row["vector"],
                        dimensions=profile.dimensions,
                        normalized=profile.normalized,
                        allow_zero=True,
                    )
                except (TypeError, ValueError):
                    issues.append(
                        f"{target_kind}_embedding:vector:{target_id}:"
                        f"{profile_sha256}"
                    )

    def _check_extraction_integrity_unlocked(
        self, namespace: str | None, issues: list[str]
    ) -> None:
        clause = "" if namespace is None else " WHERE namespace = ?"
        params: tuple[object, ...] = () if namespace is None else (namespace,)
        runs = self._connection.execute(
            "SELECT * FROM extractions" + clause + " ORDER BY namespace, memory_id",
            params,
        ).fetchall()
        for run in runs:
            memory_id = run["memory_id"]
            prefix = f"extraction:{memory_id}"
            memory = self._connection.execute(
                "SELECT * FROM memories WHERE namespace = ? AND id = ?",
                (run["namespace"], memory_id),
            ).fetchone()
            if memory is None:
                issues.append(f"{prefix}:source")

            identity = run["extractor_identity"]
            identity_valid = (
                isinstance(identity, str)
                and bool(identity.strip())
                and identity == identity.strip()
            )
            if not identity_valid:
                issues.append(f"{prefix}:identity")
            digest = run["content_sha256"]
            try:
                actual_content_digest = (
                    ""
                    if memory is None
                    else hashlib.sha256(memory["content"].encode("utf-8")).hexdigest()
                )
            except (AttributeError, UnicodeEncodeError):
                actual_content_digest = ""
            if memory is None or not (
                _digest_matches(memory["content_hash"], digest)
                and _digest_matches(actual_content_digest, digest)
            ):
                issues.append(f"{prefix}:content_digest")
            if memory is not None and run["recorded_at"] != memory["recorded_at"]:
                issues.append(f"{prefix}:recorded_at")
            abstained_value = run["abstained"]
            if abstained_value not in {0, 1} or isinstance(abstained_value, bool):
                issues.append(f"{prefix}:abstained")

            derivations = self._connection.execute(
                "SELECT * FROM claim_derivations "
                "WHERE namespace = ? AND memory_id = ? ORDER BY claim_index",
                (run["namespace"], memory_id),
            ).fetchall()
            diagnostics = self._connection.execute(
                "SELECT * FROM extraction_diagnostics "
                "WHERE namespace = ? AND memory_id = ? ORDER BY diagnostic_index",
                (run["namespace"], memory_id),
            ).fetchall()
            claim_rows = self._connection.execute(
                "SELECT * FROM claims WHERE namespace = ? AND memory_id = ?",
                (run["namespace"], memory_id),
            ).fetchall()
            claims_by_id = {row["id"]: row for row in claim_rows}

            derivation_indices = [row["claim_index"] for row in derivations]
            if derivation_indices != list(range(len(derivations))):
                issues.append(f"{prefix}:claim_index")
            diagnostic_indices = [row["diagnostic_index"] for row in diagnostics]
            if diagnostic_indices != list(range(len(diagnostics))):
                issues.append(f"{prefix}:diagnostic_index")
            mapped_claim_ids = [row["claim_id"] for row in derivations]
            if (
                len(mapped_claim_ids) != len(set(mapped_claim_ids))
                or set(mapped_claim_ids) != set(claims_by_id)
            ):
                issues.append(f"{prefix}:claim_mapping")
            should_abstain = not claim_rows
            if (
                abstained_value in {0, 1}
                and bool(abstained_value) != should_abstain
            ):
                issues.append(f"{prefix}:abstention_mismatch")

            extracted_claims: list[ClaimInput] = []
            provenance: list[ClaimProvenance] = []
            reconstruction_failed = False
            for item in derivations:
                claim = claims_by_id.get(item["claim_id"])
                if claim is None:
                    reconstruction_failed = True
                    continue
                if (
                    claim["memory_id"] != memory_id
                    or claim["namespace"] != run["namespace"]
                ):
                    issues.append(f"{prefix}:claim_mapping")
                try:
                    extracted_claims.append(
                        ClaimInput(
                            subject=claim["subject"],
                            predicate=claim["predicate"],
                            value=claim["value"],
                            valid_from=claim["valid_from"],
                            valid_to=claim["declared_valid_to"],
                            confidence=float(claim["confidence"]),
                            authority=claim["authority"],
                            supersedes=claim["supersedes"],
                        )
                    )
                    provenance.append(
                        ClaimProvenance(
                            claim_index=item["claim_index"],
                            start=item["start_offset"],
                            end=item["end_offset"],
                            canonical_text=item["canonical_text"],
                            content_sha256=item["content_sha256"],
                            extractor_identity=item["extractor_identity"],
                        )
                    )
                except (TypeError, ValueError, OverflowError):
                    reconstruction_failed = True
            extracted_diagnostics: list[ExtractionDiagnostic] = []
            for item in diagnostics:
                try:
                    extracted_diagnostics.append(
                        ExtractionDiagnostic(
                            code=item["code"],
                            message=item["message"],
                            start=item["start_offset"],
                            end=item["end_offset"],
                        )
                    )
                except (TypeError, ValueError):
                    reconstruction_failed = True

            if memory is not None and identity_valid and not reconstruction_failed:
                try:
                    evidence = EvidenceContext.from_content(
                        namespace=memory["namespace"],
                        content=memory["content"],
                        source=memory["source"],
                        occurred_at=memory["occurred_at"],
                        metadata=json.loads(memory["metadata_json"]),
                    )
                    rebuilt = ClaimExtraction(
                        claims=tuple(extracted_claims),
                        provenance=tuple(provenance),
                        diagnostics=tuple(extracted_diagnostics),
                    )
                    validate_claim_extraction(
                        evidence, rebuilt, expected_identity=identity
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    issues.append(f"{prefix}:provenance")
            elif reconstruction_failed:
                issues.append(f"{prefix}:provenance")

        scoped_join = "" if namespace is None else " AND p.namespace = ?"
        for table in ("claim_derivations", "extraction_diagnostics"):
            orphan_count = self._scalar(
                f"SELECT COUNT(*) FROM {table} AS p "
                "LEFT JOIN extractions AS e "
                "ON e.memory_id = p.memory_id AND e.namespace = p.namespace "
                f"WHERE e.memory_id IS NULL{scoped_join}",
                params,
            )
            if orphan_count:
                issues.append(f"{table}:orphaned:{orphan_count}")

    def rebuild_indexes(self) -> None:
        """Recompute deterministic claim projections and both FTS indexes.

        Embeddings are provider-derived projections, so this local operation leaves
        existing, missing, stale, or corrupt semantic rows unchanged.  The public
        facade's namespace-scoped ``sync_semantic`` method owns provider repair.
        """

        with self._lock:
            self._ensure_open()
            with self._write_transaction():
                self._recompute_claim_projections_unlocked()
                self._connection.execute(
                    "INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')"
                )
                self._connection.execute(
                    "INSERT INTO claim_fts(claim_fts) VALUES ('rebuild')"
                )

    def purge_memory(self, namespace: str, id: str) -> bool:
        ns = _required_label(namespace, "namespace")
        memory_id = _required_label(id, "id")
        with self._lock:
            self._ensure_open()
            with self._write_transaction():
                exists = self._connection.execute(
                    "SELECT 1 FROM memories WHERE namespace = ? AND id = ?",
                    (ns, memory_id),
                ).fetchone()
                if exists is None:
                    return False
                claim_rows = self._connection.execute(
                    "SELECT id, supersedes, valid_from FROM claims "
                    "WHERE namespace = ? AND memory_id = ?",
                    (ns, memory_id),
                ).fetchall()
                claim_ids = tuple(row["id"] for row in claim_rows)
                deleted_ids = set(claim_ids)
                predecessor_by_id = {
                    row["id"]: row["supersedes"] for row in claim_rows
                }
                # Retain only an evidence-free temporal boundary when a purged claim
                # had superseded surviving truth. Without this tombstone, rebuilding
                # would incorrectly extend the stale predecessor through the present.
                redaction_boundaries: dict[str, str] = {}
                for row in claim_rows:
                    predecessor = row["supersedes"]
                    visited: set[str] = set()
                    while predecessor in deleted_ids:
                        if predecessor in visited:
                            raise IntegrityViolationError(
                                "cycle in claim supersession chain"
                            )
                        visited.add(predecessor)
                        predecessor = predecessor_by_id.get(predecessor)
                    if predecessor is None:
                        continue
                    existing_boundary = redaction_boundaries.get(predecessor)
                    redaction_boundaries[predecessor] = (
                        row["valid_from"]
                        if existing_boundary is None
                        else min(existing_boundary, row["valid_from"])
                    )
                for predecessor_id, boundary in redaction_boundaries.items():
                    self._connection.execute(
                        """
                        UPDATE claims
                        SET redacted_valid_to = CASE
                            WHEN redacted_valid_to IS NULL OR redacted_valid_to > ?
                                THEN ?
                            ELSE redacted_valid_to
                        END
                        WHERE namespace = ? AND id = ?
                        """,
                        (boundary, boundary, ns, predecessor_id),
                    )
                namespace_claims = self._connection.execute(
                    "SELECT id, supersedes FROM claims WHERE namespace = ?",
                    (ns,),
                ).fetchall()
                bridges: list[tuple[str | None, str]] = []
                for row in namespace_claims:
                    if row["id"] in deleted_ids or row["supersedes"] not in deleted_ids:
                        continue
                    predecessor = row["supersedes"]
                    bridge_visited: set[str] = set()
                    while predecessor in deleted_ids:
                        if predecessor in bridge_visited:
                            raise IntegrityViolationError("cycle in claim supersession chain")
                        bridge_visited.add(predecessor)
                        predecessor = predecessor_by_id.get(predecessor)
                    bridges.append((predecessor, row["id"]))
                if claim_ids:
                    self._connection.execute(
                        "UPDATE claims SET supersedes = NULL "
                        "WHERE namespace = ? AND memory_id = ?",
                        (ns, memory_id),
                    )
                    for predecessor, child_id in bridges:
                        self._connection.execute(
                            "UPDATE claims SET supersedes = ? "
                            "WHERE namespace = ? AND id = ?",
                            (predecessor, ns, child_id),
                        )
                self._connection.execute(
                    "DELETE FROM memories WHERE namespace = ? AND id = ?",
                    (ns, memory_id),
                )
                self._recompute_claim_projections_unlocked(ns)
                return True

    def stats(self, namespace: str | None = None) -> StoreStats:
        ns = None if namespace is None else _required_label(namespace, "namespace")
        with self._lock:
            self._ensure_open()
            if ns is None:
                values = (
                    self._scalar("SELECT COUNT(*) FROM memories"),
                    self._scalar("SELECT COUNT(*) FROM claims"),
                    self._scalar("SELECT COUNT(DISTINCT namespace) FROM memories"),
                    self._scalar("SELECT COUNT(*) FROM memory_fts"),
                    self._scalar("SELECT COUNT(*) FROM claim_fts"),
                )
            else:
                param = (ns,)
                memories = self._scalar(
                    "SELECT COUNT(*) FROM memories WHERE namespace = ?", param
                )
                values = (
                    memories,
                    self._scalar("SELECT COUNT(*) FROM claims WHERE namespace = ?", param),
                    1 if memories else 0,
                    self._scalar(
                        "SELECT COUNT(*) FROM memory_fts AS f "
                        "JOIN memories AS m ON m.rowid = f.rowid "
                        "WHERE m.namespace = ?",
                        param,
                    ),
                    self._scalar(
                        "SELECT COUNT(*) FROM claim_fts AS f "
                        "JOIN claims AS c ON c.rowid = f.rowid "
                        "WHERE c.namespace = ?",
                        param,
                    ),
                )
            return StoreStats(*values)

    def _upsert_embedding_profile_unlocked(
        self, profile: EmbeddingProfile
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO embedding_profiles(
                profile_sha256, identity, dimensions, normalized, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(profile_sha256) DO NOTHING
            """,
            (
                profile.profile_sha256,
                profile.identity,
                profile.dimensions,
                int(profile.normalized),
                _utc_now(),
            ),
        )
        self._require_persisted_profile_unlocked(profile)

    def _require_persisted_profile_unlocked(
        self, profile: EmbeddingProfile
    ) -> None:
        row = self._connection.execute(
            "SELECT * FROM embedding_profiles WHERE profile_sha256 = ?",
            (profile.profile_sha256,),
        ).fetchone()
        if row is None:
            raise EmbeddingProfileNotFoundError(
                "embedding profile has no persisted projections: "
                f"{profile.profile_sha256}"
            )
        persisted = _embedding_profile_from_row(row)
        if persisted != profile:
            raise IntegrityViolationError(
                "persisted embedding profile is incompatible with requested profile: "
                f"{profile.profile_sha256}"
            )

    def _embedding_input_for_target_unlocked(
        self, target_kind: str, target_id: str
    ) -> tuple[str, EmbeddingInput]:
        kind = _embedding_target_kind(target_kind)
        target = _required_label(target_id, "embedding target_id")
        if kind == "memory":
            row = self._connection.execute(
                "SELECT * FROM memories WHERE id = ?", (target,)
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT * FROM claims WHERE id = ?", (target,)
            ).fetchone()
        if row is None:
            raise ValidationError(f"embedding {kind} target does not exist: {target}")
        return row["namespace"], self._canonical_embedding_input_from_row(kind, row)

    def _canonical_embedding_input_from_row(
        self, target_kind: str, row: sqlite3.Row
    ) -> EmbeddingInput:
        if target_kind == "memory":
            return memory_embedding_input(self._memory_from_row(row, ()))
        if target_kind == "claim":
            return claim_embedding_input(self._claim_from_row(row))
        raise AssertionError("unexpected embedding target kind")

    def _validated_projection_vector_from_row(
        self,
        target_kind: str,
        row: sqlite3.Row,
        profile: EmbeddingProfile,
    ) -> tuple[float, ...]:
        target_id = row["id"]
        try:
            canonical = self._canonical_embedding_input_from_row(target_kind, row)
        except (TypeError, ValueError, IntegrityViolationError) as exc:
            raise IntegrityViolationError(
                f"cannot reconstruct embedding input for {target_kind} {target_id}"
            ) from exc
        if not _digest_matches(
            canonical.input_sha256, row["_embedding_input_sha256"]
        ):
            raise IntegrityViolationError(
                f"stale embedding input digest for {target_kind} {target_id}"
            )
        try:
            return decode_float32_vector(
                row["_embedding_vector"],
                dimensions=profile.dimensions,
                normalized=profile.normalized,
                allow_zero=True,
            )
        except (TypeError, ValueError) as exc:
            raise IntegrityViolationError(
                f"invalid embedding vector for {target_kind} {target_id}"
            ) from exc

    def _recompute_claim_projections_unlocked(
        self, namespace: str | None = None
    ) -> None:
        """Recreate effective validity and status from durable claim links."""

        if namespace is None:
            rows = self._connection.execute(
                "SELECT id, supersedes, valid_from, declared_valid_to, "
                "redacted_valid_to FROM claims"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT id, supersedes, valid_from, declared_valid_to, "
                "redacted_valid_to FROM claims "
                "WHERE namespace = ?",
                (namespace,),
            ).fetchall()
        projections = _expected_claim_projections(rows)
        for claim_id, (status, valid_to) in projections.items():
            self._connection.execute(
                "UPDATE claims SET status = ?, valid_to = ? WHERE id = ?",
                (status, valid_to, claim_id),
            )

    def _initialize_schema(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _SCHEMA_VERSION:
            raise IntegrityViolationError(
                f"database schema version {version} is newer than supported "
                f"version {_SCHEMA_VERSION}"
            )
        if version == 0:
            existing = self._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
            ).fetchone()
            if existing is not None:
                raise IntegrityViolationError(
                    "unversioned non-empty database is not a fresh EviCortex store"
                )
            self._create_schema_v5()
        elif version in {1, 2}:
            self._migrate_legacy_schema(version)
        elif version == 3:
            self._migrate_schema_v3()
        elif version == 4:
            self._migrate_schema_v4()
        elif version == _SCHEMA_VERSION:
            self._validate_schema_v5()
        else:  # SQLite user_version is normally non-negative; be explicit regardless.
            raise IntegrityViolationError(f"unsupported database schema version {version}")

    def _create_schema_v5(self) -> None:
        script = f"""
            BEGIN IMMEDIATE;
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                idempotency_key TEXT,
                request_hash TEXT NOT NULL,
                UNIQUE(id, namespace)
            );
            CREATE UNIQUE INDEX memories_idempotency
                ON memories(namespace, idempotency_key)
                WHERE idempotency_key IS NOT NULL;
            CREATE INDEX memories_namespace_time
                ON memories(namespace, occurred_at, id);

            CREATE TABLE claims (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                declared_valid_to TEXT,
                redacted_valid_to TEXT,
                recorded_at TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                authority TEXT NOT NULL,
                supersedes TEXT REFERENCES claims(id) ON DELETE SET NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'superseded', 'disputed')),
                UNIQUE(id, namespace),
                FOREIGN KEY(memory_id, namespace)
                    REFERENCES memories(id, namespace) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX claims_one_successor
                ON claims(supersedes) WHERE supersedes IS NOT NULL;
            CREATE INDEX claims_namespace_sp_time
                ON claims(namespace, subject, predicate, valid_from, id);
            CREATE INDEX claims_memory ON claims(namespace, memory_id);
            {_EXTERNAL_FTS_SCHEMA_SQL}
            {_SEMANTIC_SCHEMA_SQL}
            {_EXTRACTION_SCHEMA_SQL}
            PRAGMA user_version = {_SCHEMA_VERSION};
            COMMIT;
        """
        self._run_schema_script(script)
        self._validate_schema_v5()

    def _migrate_schema_v3(self) -> None:
        self._validate_schema_v3()
        script = f"""
            BEGIN IMMEDIATE;
            {_SEMANTIC_SCHEMA_SQL}
            {_EXTRACTION_SCHEMA_SQL}
            PRAGMA user_version = {_SCHEMA_VERSION};
            COMMIT;
        """
        self._run_schema_script(script)
        self._validate_schema_v5()

    def _migrate_schema_v4(self) -> None:
        self._validate_schema_v4()
        script = f"""
            BEGIN IMMEDIATE;
            {_EXTRACTION_SCHEMA_SQL}
            PRAGMA user_version = {_SCHEMA_VERSION};
            COMMIT;
        """
        self._run_schema_script(script)
        self._validate_schema_v5()

    def _validate_schema_v5(self) -> None:
        self._validate_schema_v4()
        expected_columns = {
            "extractions": {
                "memory_id",
                "namespace",
                "extractor_identity",
                "content_sha256",
                "recorded_at",
                "abstained",
            },
            "claim_derivations": {
                "memory_id",
                "namespace",
                "claim_id",
                "claim_index",
                "start_offset",
                "end_offset",
                "canonical_text",
                "content_sha256",
                "extractor_identity",
            },
            "extraction_diagnostics": {
                "memory_id",
                "namespace",
                "diagnostic_index",
                "code",
                "message",
                "start_offset",
                "end_offset",
            },
        }
        for table, expected in expected_columns.items():
            columns = {
                row["name"]
                for row in self._connection.execute(f"PRAGMA table_info({table})")
            }
            if columns != expected:
                raise IntegrityViolationError(
                    f"schema v5 {table} columns do not match the storage contract"
                )
        required_foreign_keys: dict[
            str, set[tuple[str, tuple[str, ...], tuple[str, ...], str]]
        ] = {
            "extractions": {
                (
                    "memories",
                    ("memory_id", "namespace"),
                    ("id", "namespace"),
                    "CASCADE",
                )
            },
            "claim_derivations": {
                (
                    "extractions",
                    ("memory_id", "namespace"),
                    ("memory_id", "namespace"),
                    "CASCADE",
                ),
                (
                    "claims",
                    ("claim_id", "memory_id", "namespace"),
                    ("id", "memory_id", "namespace"),
                    "CASCADE",
                ),
            },
            "extraction_diagnostics": {
                (
                    "extractions",
                    ("memory_id", "namespace"),
                    ("memory_id", "namespace"),
                    "CASCADE",
                )
            },
        }
        for table, required in required_foreign_keys.items():
            grouped: dict[int, list[sqlite3.Row]] = {}
            for row in self._connection.execute(
                f"PRAGMA foreign_key_list({table})"
            ):
                grouped.setdefault(int(row["id"]), []).append(row)
            actual = set()
            for rows in grouped.values():
                ordered = sorted(rows, key=lambda row: int(row["seq"]))
                actual.add(
                    (
                        ordered[0]["table"],
                        tuple(row["from"] for row in ordered),
                        tuple(row["to"] for row in ordered),
                        ordered[0]["on_delete"].upper(),
                    )
                )
            if not required <= actual:
                raise IntegrityViolationError(
                    f"schema v5 {table} is missing provenance purge cascades"
                )
        required_indexes = {
            "claims_derivation_target": (
                "claims",
                True,
                ("id", "memory_id", "namespace"),
            ),
            "extractions_namespace_memory": (
                "extractions",
                False,
                ("namespace", "memory_id"),
            ),
            "claim_derivations_namespace_memory": (
                "claim_derivations",
                False,
                ("namespace", "memory_id", "claim_index"),
            ),
            "extraction_diagnostics_namespace_memory": (
                "extraction_diagnostics",
                False,
                ("namespace", "memory_id", "diagnostic_index"),
            ),
        }
        for name, (table, unique, expected_index_columns) in required_indexes.items():
            index_rows = {
                row["name"]: bool(row["unique"])
                for row in self._connection.execute(f"PRAGMA index_list({table})")
            }
            actual_columns = tuple(
                row["name"]
                for row in self._connection.execute(f"PRAGMA index_info({name})")
            )
            if (
                index_rows.get(name) != unique
                or actual_columns != expected_index_columns
            ):
                raise IntegrityViolationError(
                    f"schema v5 index {name} does not match the storage contract"
                )

    def _validate_schema_v4(self) -> None:
        self._validate_schema_v3()
        expected_columns = {
            "embedding_profiles": {
                "profile_sha256",
                "identity",
                "dimensions",
                "normalized",
                "created_at",
            },
            "memory_embeddings": {
                "memory_id",
                "namespace",
                "profile_sha256",
                "input_sha256",
                "vector",
                "updated_at",
            },
            "claim_embeddings": {
                "claim_id",
                "namespace",
                "profile_sha256",
                "input_sha256",
                "vector",
                "updated_at",
            },
        }
        for table, expected in expected_columns.items():
            columns = {
                row["name"]
                for row in self._connection.execute(f"PRAGMA table_info({table})")
            }
            if columns != expected:
                raise IntegrityViolationError(
                    f"schema v4 {table} columns do not match the storage contract"
                )
        for table, parent in (
            ("memory_embeddings", "memories"),
            ("claim_embeddings", "claims"),
        ):
            foreign_keys = self._connection.execute(
                f"PRAGMA foreign_key_list({table})"
            ).fetchall()
            target_actions = {
                (row["table"], row["from"], row["on_delete"].upper())
                for row in foreign_keys
            }
            expected_target = "memory_id" if parent == "memories" else "claim_id"
            if (parent, expected_target, "CASCADE") not in target_actions:
                raise IntegrityViolationError(
                    f"schema v4 {table} is missing target purge cascade"
                )
            if ("embedding_profiles", "profile_sha256", "RESTRICT") not in (
                target_actions
            ):
                raise IntegrityViolationError(
                    f"schema v4 {table} is missing profile foreign key"
                )
        indexes = {
            row["name"]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        required_indexes = {
            "memory_embeddings_namespace_profile",
            "claim_embeddings_namespace_profile",
        }
        if not required_indexes <= indexes:
            raise IntegrityViolationError(
                "schema v4 is missing namespace/profile semantic indexes"
            )

    def _migrate_legacy_schema(self, source_version: int) -> None:
        required_tables = {"memories", "claims", "memory_fts", "claim_fts"}
        existing_tables = {
            row["name"]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = sorted(required_tables - existing_tables)
        if missing:
            raise IntegrityViolationError(
                f"schema v{source_version} is missing required tables: "
                + ", ".join(missing)
            )
        claim_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(claims)")
        }
        add_boundary = (
            ""
            if "redacted_valid_to" in claim_columns
            else "ALTER TABLE claims ADD COLUMN redacted_valid_to TEXT;"
        )
        script = f"""
            BEGIN IMMEDIATE;
            {add_boundary}
            {_DROP_FTS_SCHEMA_SQL}
            {_EXTERNAL_FTS_SCHEMA_SQL}
            {_SEMANTIC_SCHEMA_SQL}
            {_EXTRACTION_SCHEMA_SQL}
            INSERT INTO memory_fts(memory_fts) VALUES ('rebuild');
            INSERT INTO claim_fts(claim_fts) VALUES ('rebuild');
            PRAGMA user_version = {_SCHEMA_VERSION};
            COMMIT;
        """
        self._run_schema_script(script)
        self._validate_schema_v5()

    def _validate_schema_v3(self) -> None:
        claim_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(claims)")
        }
        if "redacted_valid_to" not in claim_columns:
            raise IntegrityViolationError(
                "schema v3 claims table is missing redacted_valid_to"
            )
        definitions = {
            row["name"]: row["sql"] or ""
            for row in self._connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name IN ('memory_fts', 'claim_fts')"
            )
        }
        expected_content = {
            "memory_fts": "memories",
            "claim_fts": "claims",
        }
        for table, content_table in expected_content.items():
            sql = definitions.get(table, "")
            pattern = rf"content\s*=\s*['\"]{content_table}['\"]"
            if not re.search(pattern, sql, flags=re.IGNORECASE):
                raise IntegrityViolationError(
                    f"schema v3 {table} is not an external-content index"
                )
        required_triggers = {
            "memories_fts_insert",
            "memories_fts_delete",
            "memories_fts_update",
            "claims_fts_insert",
            "claims_fts_delete",
            "claims_fts_update",
        }
        triggers = {
            row["name"]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
        missing_triggers = sorted(required_triggers - triggers)
        if missing_triggers:
            raise IntegrityViolationError(
                "schema v3 is missing FTS synchronization triggers: "
                + ", ".join(missing_triggers)
            )

    def _run_schema_script(self, script: str) -> None:
        try:
            self._connection.executescript(script)
        except BaseException:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _get_memory_unlocked(self, namespace: str, memory_id: str) -> MemoryRecord | None:
        row = self._connection.execute(
            "SELECT * FROM memories WHERE namespace = ? AND id = ?",
            (namespace, memory_id),
        ).fetchone()
        return None if row is None else self._memory_from_row(row)

    def _memory_records_from_rows(
        self, rows: Sequence[sqlite3.Row]
    ) -> list[MemoryRecord]:
        if not rows:
            return []
        memory_ids = tuple(row["id"] for row in rows)
        placeholders = ", ".join("?" for _ in memory_ids)
        claim_rows = self._connection.execute(
            f"SELECT * FROM claims WHERE namespace = ? "
            f"AND memory_id IN ({placeholders}) ORDER BY recorded_at, id",
            (rows[0]["namespace"], *memory_ids),
        ).fetchall()
        claims_by_memory: dict[str, list[sqlite3.Row]] = {}
        for claim_row in claim_rows:
            claims_by_memory.setdefault(claim_row["memory_id"], []).append(claim_row)
        return [
            self._memory_from_row(row, claims_by_memory.get(row["id"], ()))
            for row in rows
        ]

    def _memory_from_row(
        self,
        row: sqlite3.Row,
        claim_rows: Sequence[sqlite3.Row] | None = None,
    ) -> MemoryRecord:
        if claim_rows is None:
            claim_rows = self._connection.execute(
                "SELECT * FROM claims WHERE namespace = ? AND memory_id = ? "
                "ORDER BY recorded_at, id",
                (row["namespace"], row["id"]),
            ).fetchall()
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise IntegrityViolationError(
                f"invalid metadata JSON for memory {row['id']}"
            ) from exc
        return MemoryRecord(
            id=row["id"],
            namespace=row["namespace"],
            kind=row["kind"],
            content=row["content"],
            content_hash=row["content_hash"],
            source=row["source"],
            occurred_at=row["occurred_at"],
            recorded_at=row["recorded_at"],
            metadata=metadata,
            idempotency_key=row["idempotency_key"],
            claims=tuple(self._claim_from_row(item) for item in claim_rows),
        )

    @staticmethod
    def _claim_from_row(row: sqlite3.Row) -> ClaimRecord:
        return ClaimRecord(
            id=row["id"],
            namespace=row["namespace"],
            memory_id=row["memory_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            recorded_at=row["recorded_at"],
            confidence=float(row["confidence"]),
            authority=row["authority"],
            supersedes=row["supersedes"],
            status=row["status"],
        )

    def _extraction_from_row(self, row: sqlite3.Row) -> ExtractionRecord:
        derivation_rows = self._connection.execute(
            "SELECT * FROM claim_derivations "
            "WHERE namespace = ? AND memory_id = ? ORDER BY claim_index",
            (row["namespace"], row["memory_id"]),
        ).fetchall()
        diagnostic_rows = self._connection.execute(
            "SELECT * FROM extraction_diagnostics "
            "WHERE namespace = ? AND memory_id = ? ORDER BY diagnostic_index",
            (row["namespace"], row["memory_id"]),
        ).fetchall()
        derivations = tuple(
            ClaimDerivationRecord(
                claim_id=item["claim_id"],
                claim_index=int(item["claim_index"]),
                start=int(item["start_offset"]),
                end=int(item["end_offset"]),
                canonical_text=item["canonical_text"],
                content_sha256=item["content_sha256"],
                extractor_identity=item["extractor_identity"],
            )
            for item in derivation_rows
        )
        diagnostics = tuple(
            ExtractionDiagnosticRecord(
                diagnostic_index=int(item["diagnostic_index"]),
                code=item["code"],
                message=item["message"],
                start=(
                    None
                    if item["start_offset"] is None
                    else int(item["start_offset"])
                ),
                end=(
                    None
                    if item["end_offset"] is None
                    else int(item["end_offset"])
                ),
            )
            for item in diagnostic_rows
        )
        return ExtractionRecord(
            memory_id=row["memory_id"],
            namespace=row["namespace"],
            extractor_identity=row["extractor_identity"],
            content_sha256=row["content_sha256"],
            recorded_at=row["recorded_at"],
            abstained=bool(row["abstained"]),
            derivations=derivations,
            diagnostics=diagnostics,
        )

    def _scalar(self, sql: str, params: tuple[object, ...] = ()) -> int:
        return int(self._connection.execute(sql, params).fetchone()[0])

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageClosedError("memory store is closed")


MemoryStore = SQLiteMemoryStore


def _required_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _exact_extraction_identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "extraction_identity must be a non-empty string when extraction is supplied"
        )
    if value != value.strip():
        raise ValidationError("extraction_identity cannot have edge whitespace")
    return value


def _normalize_timestamp(value: TimestampInput) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValidationError("timestamp must not be empty")
        if text[-1:] in {"Z", "z"}:
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValidationError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    else:
        raise ValidationError("timestamp must be an ISO string or datetime")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        utc = parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValidationError("timestamp cannot be normalized to UTC") from exc
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_metadata(metadata: Mapping[str, Any] | None) -> str:
    if metadata is None:
        value: Mapping[str, Any] = {}
    elif isinstance(metadata, Mapping):
        value = metadata
    else:
        raise ValidationError("metadata must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValidationError("metadata must contain JSON-compatible values") from exc
    if not isinstance(decoded, dict):
        raise ValidationError("metadata must encode a JSON object")
    return encoded


def _prepare_claim(value: ClaimInput) -> _PreparedClaim:
    if not isinstance(value, ClaimInput):
        raise ValidationError("claims must contain ClaimInput values")
    subject = _required_label(value.subject, "claim subject")
    predicate = _required_label(value.predicate, "claim predicate")
    if not isinstance(value.value, str) or not value.value.strip():
        raise ValidationError("claim value must be a non-empty string")
    authority = _required_label(value.authority, "claim authority")
    if isinstance(value.confidence, bool) or not isinstance(
        value.confidence, (int, float)
    ):
        raise ValidationError("claim confidence must be numeric")
    confidence = float(value.confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValidationError("claim confidence must be between 0 and 1")
    supersedes = (
        None
        if value.supersedes is None
        else _required_label(value.supersedes, "claim supersedes")
    )
    valid_from = _normalize_timestamp(value.valid_from)
    valid_to = _normalize_timestamp(value.valid_to)
    if valid_from is not None and valid_to is not None and valid_to <= valid_from:
        raise ValidationError("claim valid_to must be later than valid_from")
    return _PreparedClaim(
        subject=subject,
        predicate=predicate,
        value=value.value,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=confidence,
        authority=authority,
        supersedes=supersedes,
    )


def _request_digest(
    *,
    content: str,
    kind: str,
    source: str,
    occurred_at: str | None,
    recorded_at: str | None,
    metadata_json: str,
    claims: tuple[_PreparedClaim, ...],
    extraction: ClaimExtraction | None,
    extraction_identity: str | None,
) -> str:
    payload = {
        "content": content,
        "kind": kind,
        "source": source,
        "occurred_at": occurred_at,
        "recorded_at": recorded_at,
        "metadata": json.loads(metadata_json),
        "claims": [
            {
                "subject": item.subject,
                "predicate": item.predicate,
                "value": item.value,
                "valid_from": item.valid_from,
                "valid_to": item.valid_to,
                "confidence": item.confidence,
                "authority": item.authority,
                "supersedes": item.supersedes,
            }
            for item in claims
        ],
    }
    # Preserve the exact Prototype A/B digest for non-extraction writes so
    # idempotency keys stored by schema v1-v4 remain replayable after migration.
    if extraction is not None:
        assert extraction_identity is not None
        payload["extraction"] = {
            "extractor_identity": extraction_identity,
            "abstained": extraction.abstained,
            "provenance": [
                {
                    "claim_index": item.claim_index,
                    "start": item.start,
                    "end": item.end,
                    "canonical_text": item.canonical_text,
                    "content_sha256": item.content_sha256,
                    "extractor_identity": item.extractor_identity,
                }
                for item in extraction.provenance
            ],
            "diagnostics": [
                {
                    "code": item.code,
                    "message": item.message,
                    "start": item.start,
                    "end": item.end,
                }
                for item in extraction.diagnostics
            ],
        }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(value))


def _encode_terms(terms: Sequence[str]) -> str:
    return _TERM_SEPARATOR.join(terms)


def _sqlite_local_rank(
    document: object, positive_blob: object, excluded_blob: object
) -> float:
    """Return a namespace-independent lexical key for SQL preselection.

    FTS5 supplies only the boolean candidate set. This score reproduces the
    Python rerank's lexicographic priorities: fewer excluded matches, then more
    distinct positive terms (coverage), then greater positive term frequency.
    Each lower-priority component is mathematically bounded so it cannot outrank
    a higher-priority component.
    """

    if not isinstance(document, str) or not isinstance(positive_blob, str):
        return float("-inf")
    positive_terms = tuple(
        term for term in positive_blob.split(_TERM_SEPARATOR) if term
    )
    if not positive_terms:
        return float("-inf")
    excluded_terms = (
        tuple(term for term in excluded_blob.split(_TERM_SEPARATOR) if term)
        if isinstance(excluded_blob, str)
        else ()
    )
    counts = Counter(_tokenize(document))
    matched = sum(1 for term in positive_terms if counts[term])
    frequency = sum(counts[term] for term in positive_terms)
    excluded_matches = sum(counts[term] for term in excluded_terms)
    frequency_tiebreak = frequency / (frequency + 1.0)
    positive_score = (matched + frequency_tiebreak) / (len(positive_terms) + 1.0)
    return -float(excluded_matches) + positive_score


def _prepare_fts_query(
    query: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if not isinstance(query, str):
        raise ValidationError("query must be a string")
    raw_terms = _tokenize(query)
    excluded: list[str] = []
    syntax_positions: set[int] = set()
    for index, term in enumerate(raw_terms):
        if term not in _NEGATION_WORDS:
            continue
        syntax_positions.add(index)
        target_index = index + 1
        if target_index < len(raw_terms) and raw_terms[target_index] in _ARTICLES:
            syntax_positions.add(target_index)
            target_index += 1
        if target_index < len(raw_terms):
            excluded.append(raw_terms[target_index])
            syntax_positions.add(target_index)
    excluded_terms = tuple(dict.fromkeys(excluded))
    positive_terms = tuple(
        dict.fromkeys(
            term
            for index, term in enumerate(raw_terms)
            if index not in syntax_positions
            and term not in _STOP_WORDS
            and len(term) > 1
        )
    )
    # One-character subjects/predicates are legitimate in structured stores. Avoid
    # letting possessive fragments such as the ``s`` in "user's" add noise when a
    # query already has informative terms, but retain short identifiers when they are
    # the only searchable input.
    if not positive_terms:
        positive_terms = tuple(
            dict.fromkeys(
                term
                for index, term in enumerate(raw_terms)
                if index not in syntax_positions and term not in _STOP_WORDS
            )
        )
    if not positive_terms:
        return (), excluded_terms, ""
    positive_expression = " OR ".join(f'"{term}"' for term in positive_terms)
    if excluded_terms:
        excluded_expression = " OR ".join(f'"{term}"' for term in excluded_terms)
        expression = f"({positive_expression}) NOT ({excluded_expression})"
    else:
        expression = positive_expression
    return positive_terms, excluded_terms, expression


def _positive_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValidationError("limit must be a positive integer")
    return limit


def _positive_scan_cap(scan_cap: int) -> int:
    if (
        isinstance(scan_cap, bool)
        or not isinstance(scan_cap, int)
        or scan_cap <= 0
    ):
        raise ValidationError("scan_cap must be a positive integer")
    return scan_cap


def _embedding_target_kind(target_kind: object) -> str:
    if target_kind not in {"memory", "claim"}:
        raise ValidationError("target_kind must be 'memory' or 'claim'")
    return str(target_kind)


def _checked_embedding_profile(profile: object) -> EmbeddingProfile:
    if not isinstance(profile, EmbeddingProfile):
        raise ValidationError("profile must be an EmbeddingProfile")
    try:
        canonical = EmbeddingProfile(
            profile.identity, profile.dimensions, profile.normalized
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("embedding profile is invalid") from exc
    if not _digest_matches(canonical.profile_sha256, profile.profile_sha256):
        raise ValidationError("embedding profile digest is invalid")
    return canonical


def _embedding_profile_from_row(row: Mapping[str, Any]) -> EmbeddingProfile:
    normalized = row["normalized"]
    if normalized not in {0, 1} or isinstance(normalized, bool):
        raise IntegrityViolationError(
            "persisted embedding profile has invalid normalization flag"
        )
    try:
        profile = EmbeddingProfile(
            identity=row["identity"],
            dimensions=row["dimensions"],
            normalized=bool(normalized),
        )
    except (TypeError, ValueError) as exc:
        raise IntegrityViolationError(
            "persisted embedding profile is invalid"
        ) from exc
    if not _digest_matches(profile.profile_sha256, row["profile_sha256"]):
        raise IntegrityViolationError(
            "persisted embedding profile digest does not match its fields"
        )
    return profile


def _float32_query_vector(
    vector: Sequence[float], profile: EmbeddingProfile
) -> tuple[float, ...]:
    payload = encode_float32_vector(
        vector,
        dimensions=profile.dimensions,
        normalized=profile.normalized,
        allow_zero=True,
    )
    return decode_float32_vector(
        payload,
        dimensions=profile.dimensions,
        normalized=profile.normalized,
        allow_zero=True,
    )


def _digest_matches(expected: object, actual: object) -> bool:
    return (
        isinstance(expected, str)
        and isinstance(actual, str)
        and len(expected) == 64
        and len(actual) == 64
        and hmac.compare_digest(expected, actual)
    )


def _candidate_pool_size(limit: int) -> int:
    return min(
        _MAX_CANDIDATES,
        max(_MIN_CANDIDATES, limit * _CANDIDATE_MULTIPLIER),
    )


def _lexical_score(
    query_terms: tuple[str, ...],
    document_terms: tuple[str, ...],
    excluded_terms: tuple[str, ...] = (),
) -> tuple[float, int, int]:
    counts = Counter(document_terms)
    matched = sum(1 for term in query_terms if counts[term])
    frequency = sum(counts[term] for term in query_terms)
    excluded_matches = sum(counts[term] for term in excluded_terms)
    return matched / len(query_terms), frequency, excluded_matches


def _memory_row_tie_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Prefer durable evidence fields over generated IDs for equal ranks."""

    return (
        str(row["source"]).casefold(),
        str(row["source"]),
        str(row["occurred_at"]),
        str(row["recorded_at"]),
        str(row["content_hash"]),
        str(row["kind"]).casefold(),
        str(row["kind"]),
        "" if row["idempotency_key"] is None else str(row["idempotency_key"]),
        str(row["id"]),
    )


def _claim_tie_key(claim: ClaimRecord) -> tuple[str, ...]:
    """Prefer stable claim fields over generated IDs for equal ranks."""

    return (
        claim.subject.casefold(),
        claim.subject,
        claim.predicate.casefold(),
        claim.predicate,
        claim.value.casefold(),
        claim.value,
        claim.valid_from,
        claim.valid_to or "",
        claim.recorded_at,
        claim.authority.casefold(),
        claim.authority,
        claim.id,
    )


def _expected_claim_projections(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str | None]]:
    """Derive each claim's status/effective end from its declared end and child."""

    children: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        predecessor = row["supersedes"]
        if predecessor is None:
            continue
        if predecessor in children:
            raise IntegrityViolationError(
                f"claim {predecessor} has more than one superseding child"
            )
        children[predecessor] = row
    projections: dict[str, tuple[str, str | None]] = {}
    for row in rows:
        declared_end = row["declared_valid_to"]
        redacted_end = row["redacted_valid_to"]
        durable_end = _earliest_timestamp(declared_end, redacted_end)
        child = children.get(row["id"])
        if child is None:
            projections[row["id"]] = ("active", durable_end)
            continue
        child_start = child["valid_from"]
        effective_end = _earliest_timestamp(durable_end, child_start)
        projections[row["id"]] = ("superseded", effective_end)
    return projections


def _earliest_timestamp(*values: str | None) -> str | None:
    present = tuple(value for value in values if value is not None)
    return min(present) if present else None


__all__ = ["MemoryStore", "SQLiteMemoryStore"]
