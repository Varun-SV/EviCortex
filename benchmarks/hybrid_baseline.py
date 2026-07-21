"""A local diagnostic BM25 + exact-cosine hybrid RAG comparator.

This module is deliberately stronger than the mechanical fixed-chunk baseline:
it indexes two chunk granularities, fuses lexical and embedding ranks, removes
overlapping duplicates, diversifies results, and expands a winning child chunk
to bounded neighboring context.  It is still a diagnostic comparator.  The
bundled HashingEmbedder does feature hashing rather than semantic inference, so
results produced with it must not be described as a strong dense-RAG baseline.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Self
import unicodedata

from evicortex.budget import BudgetTooSmallError, estimate_tokens, validate_budget
from evicortex.plugins import (
    Embedder,
    HashingEmbedder,
    embed_document_batch,
    embed_query_text,
    validate_embedding_batch,
)
from evicortex.semantic import (
    EmbeddingProfile,
    cosine_similarity,
    decode_float32_vector,
    encode_float32_vector,
)


DIAGNOSTIC_HYBRID_LABEL = (
    "diagnostic local BM25 + exact-cosine hybrid RAG comparator; "
    "not a validated strong dense baseline"
)

_RRF_K_DEFAULT = 60
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
_MIN_TRUNCATED_CHARS = 16
_BOUNDARY_RE = re.compile(
    r"(?:\r?\n[ \t]*\r?\n+)|"
    r"(?:[.!?][\"')\]]*[ \t]+)|"
    r"(?:\r?\n+)",
    re.UNICODE,
)


TimestampInput = datetime | str


@dataclass(frozen=True, slots=True)
class HybridScoreReason:
    """One inspectable weighted reciprocal-rank contribution."""

    channel: str
    rank: int
    contribution: float
    raw_score: float


@dataclass(frozen=True, slots=True)
class HybridEvidence:
    """One source-preserving context window selected by the comparator."""

    chunk_id: str
    source_id: str
    namespace: str
    occurred_at: str
    content: str
    start_char: int
    end_char: int
    fused_score: float
    mmr_score: float
    reasons: tuple[HybridScoreReason, ...] = ()
    expanded: bool = False
    truncated: bool = False

    def prompt_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "source_id": self.source_id,
            "occurred_at": self.occurred_at,
            "text": self.content,
        }
        if self.truncated:
            item["truncated"] = True
        return item


@dataclass(frozen=True, slots=True)
class HybridDiagnostics:
    """Query-compute counters kept outside the model-facing prompt."""

    query_embedding_calls: int
    semantic_candidates_scored: int
    lexical_candidates_returned: int
    fused_candidates: int
    candidates_after_deduplication: int
    diagnostic_semantic_candidates_filtered: int = 0


@dataclass(frozen=True, slots=True)
class HybridRecall:
    """A deterministic, hard-budget result from the hybrid comparator."""

    namespace: str
    query: str
    budget_tokens: int
    as_of: str | None
    evidence: tuple[HybridEvidence, ...] = ()
    omitted_count: int = 0
    diagnostics: HybridDiagnostics = field(
        default_factory=lambda: HybridDiagnostics(0, 0, 0, 0, 0)
    )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.source_id for item in self.evidence))

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.to_prompt())

    def to_prompt(self) -> str:
        payload: dict[str, Any] = {"v": "hybrid-diag1"}
        if self.as_of is not None:
            payload["as_of"] = self.as_of
        if self.evidence:
            payload["evidence"] = [item.prompt_dict() for item in self.evidence]
        if self.omitted_count:
            payload["omitted_count"] = self.omitted_count
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class HybridIndexStats:
    """Small public index diagnostic used by tests and benchmark reports."""

    sources: int
    chunks: int
    vectors: int
    chunks_by_granularity: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class _Candidate:
    rowid: int
    chunk_id: str
    source_id: str
    namespace: str
    occurred_at: str
    granularity: str
    ordinal: int
    start_char: int
    end_char: int
    content: str
    content_hash: str
    vector: tuple[float, ...]
    reasons: dict[str, HybridScoreReason] = field(default_factory=dict)

    @property
    def fused_score(self) -> float:
        return math.fsum(reason.contribution for reason in self.reasons.values())


def _validated_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validated_probability(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return result


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _normalize_timestamp(value: TimestampInput, name: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as error:
            raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    else:
        raise TypeError(f"{name} must be a datetime or non-empty string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _chunk_spans(text: str, *, size: int, overlap: int) -> tuple[tuple[int, int], ...]:
    """Return boundary-snapped spans with overlap and complete source coverage.

    The nominal size is a character target, not a hard maximum.  Endpoints prefer
    a sentence, line, or paragraph boundary in the latter half of the target
    window; when none exists, the exact target is used.  Each next span begins at
    the prior endpoint minus the requested overlap, so snapping cannot create a
    gap or silently discard punctuation/whitespace.
    """

    if not text:
        return ()
    boundaries = tuple(match.end() for match in _BOUNDARY_RE.finditer(text))
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        target = min(len(text), start + size)
        if target == len(text):
            end = target
        else:
            minimum = start + max(1, size // 2)
            eligible = [
                boundary
                for boundary in boundaries
                if minimum <= boundary <= target
            ]
            end = eligible[-1] if eligible else target
        spans.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return tuple(spans)


def _lexical_tokens(value: str) -> tuple[str, ...]:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return tuple(_TOKEN_RE.findall(without_marks))


def _query_terms(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    positive: list[str] = []
    negative: list[str] = []
    negate_next = False
    for token in _lexical_tokens(query):
        if token in _NEGATION_WORDS:
            negate_next = True
            continue
        if token in _STOP_WORDS or len(token) <= 1:
            continue
        target = negative if negate_next else positive
        target.append(token)
        negate_next = False
    return tuple(dict.fromkeys(positive)), tuple(dict.fromkeys(negative))


def _fts_expression(query: str) -> tuple[str | None, tuple[str, ...]]:
    positive, negative = _query_terms(query)
    if not positive:
        return None, negative

    def quoted(term: str) -> str:
        return f'"{term.replace(chr(34), chr(34) * 2)}"'

    expression = " OR ".join(quoted(term) for term in positive)
    if len(positive) > 1:
        expression = f"({expression})"
    if negative:
        excluded = " OR ".join(quoted(term) for term in negative)
        if len(negative) > 1:
            excluded = f"({excluded})"
        expression = f"{expression} NOT {excluded}"
    return expression, negative


def _candidate_key(candidate: _Candidate) -> tuple[Any, ...]:
    return (
        -candidate.fused_score,
        candidate.source_id,
        candidate.start_char,
        candidate.end_char,
        candidate.chunk_id,
    )


def _overlap_ratio(left: _Candidate, right: _Candidate) -> float:
    if left.source_id != right.source_id:
        return 0.0
    intersection = max(
        0,
        min(left.end_char, right.end_char) - max(left.start_char, right.start_char),
    )
    denominator = min(
        left.end_char - left.start_char,
        right.end_char - right.start_char,
    )
    return intersection / denominator if denominator > 0 else 0.0


class DiagnosticHybridRAGBaseline:
    """SQLite BM25 plus exact-cosine diagnostic retrieval.

    Source rows are append-only.  Optional event-level validity and supersession
    metadata let the comparator preserve every version while filtering current or
    historical retrieval.  This is intentionally not a claim-level temporal model:
    superseding one fact suppresses the whole annotated source event.  Once a
    successor is event-visible, its valid_from is the predecessor's permanent
    effective end; expiry of a finite successor yields no active event and never
    resurrects the predecessor.  Occurrence time is the visibility proxy because
    this comparator does not implement transaction-time known_at queries.
    """

    label = DIAGNOSTIC_HYBRID_LABEL
    diagnostic_only = True

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
        chunk_chars: tuple[int, int] = (192, 576),
        chunk_overlaps: tuple[int, int] = (32, 96),
        channel_limit: int = 80,
        candidate_limit: int = 40,
        max_exact_candidates: int = 100_000,
        rrf_k: int = _RRF_K_DEFAULT,
        lexical_weight: float = 1.0,
        semantic_weight: float = 1.0,
        mmr_lambda: float = 0.72,
        minimum_semantic_score: float = 0.05,
        neighbor_radius: int = 1,
        max_expansion_chars: int = 720,
    ) -> None:
        if len(chunk_chars) != 2 or len(chunk_overlaps) != 2:
            raise ValueError("exactly two chunk granularities are required")
        sizes = tuple(
            _validated_positive_int(value, f"chunk_chars[{index}]")
            for index, value in enumerate(chunk_chars)
        )
        overlaps: list[int] = []
        for index, value in enumerate(chunk_overlaps):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"chunk_overlaps[{index}] must be an integer")
            if value < 0 or value >= sizes[index]:
                raise ValueError(
                    f"chunk_overlaps[{index}] must satisfy 0 <= overlap < size"
                )
            overlaps.append(value)
        if sizes[0] >= sizes[1]:
            raise ValueError("chunk granularities must be strictly increasing")

        provider = embedder if embedder is not None else HashingEmbedder()
        identity = getattr(provider, "identity", None)
        dimensions = getattr(provider, "dimensions", None)
        normalized = getattr(provider, "normalized", None)
        if not isinstance(identity, str) or not identity:
            raise TypeError("embedder.identity must be a non-empty string")
        _validated_positive_int(dimensions, "embedder.dimensions")
        if not isinstance(normalized, bool):
            raise TypeError("embedder.normalized must be a bool")
        if not callable(getattr(provider, "embed_batch", None)):
            raise TypeError("embedder must implement embed_batch")

        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer")
        if isinstance(neighbor_radius, bool) or not isinstance(neighbor_radius, int):
            raise TypeError("neighbor_radius must be an integer")
        if neighbor_radius < 0:
            raise ValueError("neighbor_radius cannot be negative")
        self.path = Path(path)
        self.embedder = provider
        self.embedding_profile = EmbeddingProfile.from_embedder(provider)
        self._requires_diagnostic_token_overlap = (
            getattr(provider, "diagnostic_only", False) is True
            or getattr(provider, "is_semantic", True) is False
        )
        self.chunk_chars = sizes
        self.chunk_overlaps = tuple(overlaps)
        self.channel_limit = _validated_positive_int(channel_limit, "channel_limit")
        self.candidate_limit = _validated_positive_int(
            candidate_limit, "candidate_limit"
        )
        self.max_exact_candidates = _validated_positive_int(
            max_exact_candidates, "max_exact_candidates"
        )
        self.rrf_k = rrf_k
        self.lexical_weight = float(lexical_weight)
        self.semantic_weight = float(semantic_weight)
        if not math.isfinite(self.lexical_weight) or self.lexical_weight <= 0:
            raise ValueError("lexical_weight must be finite and positive")
        if not math.isfinite(self.semantic_weight) or self.semantic_weight <= 0:
            raise ValueError("semantic_weight must be finite and positive")
        self.mmr_lambda = _validated_probability(mmr_lambda, "mmr_lambda")
        self.minimum_semantic_score = float(minimum_semantic_score)
        if (
            not math.isfinite(self.minimum_semantic_score)
            or not -1.0 <= self.minimum_semantic_score <= 1.0
        ):
            raise ValueError(
                "minimum_semantic_score must be finite and between -1 and 1"
            )
        self.neighbor_radius = neighbor_radius
        self.max_expansion_chars = _validated_positive_int(
            max_expansion_chars, "max_expansion_chars"
        )
        self._granularities = ("small", "large")
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(str(self.path))
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = DELETE")
            self._initialize_schema()
        except BaseException:
            # Constructor failures (most importantly profile incompatibility)
            # must not retain a Windows file handle on the rejected database.
            self._connection.close()
            self._closed = True
            raise

    @classmethod
    def open(cls, path: str | Path, **kwargs: Any) -> Self:
        return cls(path, **kwargs)

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS hybrid_index_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hybrid_source_events (
                    id INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    supersedes_source_id TEXT,
                    session_id TEXT,
                    speaker TEXT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(namespace, source_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS hybrid_one_successor
                    ON hybrid_source_events(namespace, supersedes_source_id)
                    WHERE supersedes_source_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS hybrid_source_scope_time
                    ON hybrid_source_events(namespace, occurred_at, valid_from);
                CREATE TABLE IF NOT EXISTS hybrid_chunks (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    UNIQUE(namespace, chunk_id),
                    FOREIGN KEY(namespace, source_id)
                        REFERENCES hybrid_source_events(namespace, source_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS hybrid_chunks_source_order
                    ON hybrid_chunks(namespace, source_id, granularity, ordinal);
                CREATE VIRTUAL TABLE IF NOT EXISTS hybrid_chunks_fts
                    USING fts5(content, tokenize='unicode61 remove_diacritics 2');
                CREATE TABLE IF NOT EXISTS hybrid_chunk_vectors (
                    chunk_rowid INTEGER PRIMARY KEY
                        REFERENCES hybrid_chunks(id) ON DELETE CASCADE,
                    vector BLOB NOT NULL
                );
                """
            )
        expected = {
            "embedding_profile_sha256": self.embedding_profile.profile_sha256,
            "embedder_identity": self.embedder.identity,
            "embedder_dimensions": str(self.embedder.dimensions),
            "embedder_normalized": json.dumps(self.embedder.normalized),
            "vector_dtype": "float32-le",
            "diagnostic_token_overlap_guard": json.dumps(
                self._requires_diagnostic_token_overlap
            ),
            "chunk_chars": json.dumps(self.chunk_chars),
            "chunk_overlaps": json.dumps(self.chunk_overlaps),
        }
        existing = {
            str(row["key"]): str(row["value"])
            for row in self._connection.execute(
                "SELECT key, value FROM hybrid_index_config"
            )
        }
        populated = any(
            int(
                self._connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )
            > 0
            for table in (
                "hybrid_source_events",
                "hybrid_chunks",
                "hybrid_chunk_vectors",
            )
        )
        missing = tuple(sorted(set(expected) - set(existing)))
        if populated and missing:
            raise ValueError(
                "populated hybrid index is missing compatibility config: "
                + ", ".join(missing)
            )
        with self._connection:
            for key, value in expected.items():
                stored = existing.get(key)
                if stored is None:
                    self._connection.execute(
                        "INSERT INTO hybrid_index_config(key, value) VALUES (?, ?)",
                        (key, value),
                    )
                elif stored != value:
                    raise ValueError(
                        f"existing hybrid index uses incompatible {key}: "
                        f"{stored!r}"
                    )

    def remember(
        self,
        *,
        namespace: str,
        content: str,
        source_id: str,
        occurred_at: TimestampInput,
        recorded_at: TimestampInput | None = None,
        session_id: str | None = None,
        speaker: str | None = None,
        metadata: Mapping[str, object] | None = None,
        valid_from: TimestampInput | None = None,
        valid_to: TimestampInput | None = None,
        supersedes_source_id: str | None = None,
    ) -> None:
        """Append one source event and both chunk/vector granularities atomically."""

        self._ensure_open()
        ns = _required_text(namespace, "namespace")
        source = _required_text(source_id, "source_id")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")
        occurred = _normalize_timestamp(occurred_at, "occurred_at")
        recorded = (
            occurred
            if recorded_at is None
            else _normalize_timestamp(recorded_at, "recorded_at")
        )
        starts = (
            occurred
            if valid_from is None
            else _normalize_timestamp(valid_from, "valid_from")
        )
        ends = (
            None if valid_to is None else _normalize_timestamp(valid_to, "valid_to")
        )
        if ends is not None and ends <= starts:
            raise ValueError("valid_to must be later than valid_from")
        supersedes = (
            None
            if supersedes_source_id is None
            else _required_text(supersedes_source_id, "supersedes_source_id")
        )
        if session_id is not None and not isinstance(session_id, str):
            raise TypeError("session_id must be a string or None")
        if speaker is not None and not isinstance(speaker, str):
            raise TypeError("speaker must be a string or None")
        try:
            metadata_json = json.dumps(
                {} if metadata is None else dict(metadata),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be JSON serializable") from error

        pieces: list[tuple[str, int, int, int, str]] = []
        for granularity, size, overlap in zip(
            self._granularities,
            self.chunk_chars,
            self.chunk_overlaps,
            strict=True,
        ):
            for ordinal, (start, end) in enumerate(
                _chunk_spans(content, size=size, overlap=overlap)
            ):
                pieces.append((granularity, ordinal, start, end, content[start:end]))
        texts = tuple(piece[4] for piece in pieces)
        vectors = validate_embedding_batch(
            embed_document_batch(self.embedder, texts),
            expected_count=len(texts),
            dimensions=self.embedder.dimensions,
            normalized=self.embedder.normalized,
            allow_zero=False,
        )

        with self._lock, self._connection:
            if supersedes is not None:
                predecessor = self._connection.execute(
                    """
                    SELECT valid_from
                    FROM hybrid_source_events
                    WHERE namespace = ? AND source_id = ?
                    """,
                    (ns, supersedes),
                ).fetchone()
                if predecessor is None:
                    raise ValueError(
                        "supersedes_source_id must identify an earlier event "
                        "in the same namespace"
                    )
                if starts <= predecessor["valid_from"]:
                    raise ValueError(
                        "a successor valid_from must be later than its predecessor"
                    )
            self._connection.execute(
                """
                INSERT INTO hybrid_source_events(
                    source_id, namespace, occurred_at, recorded_at,
                    valid_from, valid_to, supersedes_source_id,
                    session_id, speaker, content, content_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    ns,
                    occurred,
                    recorded,
                    starts,
                    ends,
                    supersedes,
                    session_id,
                    speaker,
                    content,
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    metadata_json,
                ),
            )
            for piece, vector in zip(pieces, vectors, strict=True):
                granularity, ordinal, start, end, text = piece
                chunk_id = f"{source}#{granularity}-{ordinal:06d}"
                cursor = self._connection.execute(
                    """
                    INSERT INTO hybrid_chunks(
                        chunk_id, source_id, namespace, granularity, ordinal,
                        start_char, end_char, content, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        source,
                        ns,
                        granularity,
                        ordinal,
                        start,
                        end,
                        text,
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    ),
                )
                rowid = int(cursor.lastrowid)
                self._connection.execute(
                    "INSERT INTO hybrid_chunks_fts(rowid, content) VALUES (?, ?)",
                    (rowid, text),
                )
                self._connection.execute(
                    """
                    INSERT INTO hybrid_chunk_vectors(chunk_rowid, vector)
                    VALUES (?, ?)
                    """,
                    (
                        rowid,
                        encode_float32_vector(
                            vector,
                            dimensions=self.embedder.dimensions,
                            normalized=self.embedder.normalized,
                            allow_zero=False,
                        ),
                    ),
                )

    def recall(
        self,
        *,
        namespace: str,
        query: str,
        budget_tokens: int,
        as_of: TimestampInput | None = None,
    ) -> HybridRecall:
        """Retrieve, fuse, diversify, expand, and pack under an exact prompt budget."""

        self._ensure_open()
        ns = _required_text(namespace, "namespace")
        normalized_query = _required_text(query, "query")
        budget = validate_budget(budget_tokens)
        explicit_as_of = (
            None if as_of is None else _normalize_timestamp(as_of, "as_of")
        )
        effective_at = explicit_as_of or self._snapshot_time(ns)
        positive_terms, negative_terms = _query_terms(normalized_query)
        expression, _ = _fts_expression(normalized_query)

        query_vectors = validate_embedding_batch(
            (embed_query_text(self.embedder, normalized_query),),
            expected_count=1,
            dimensions=self.embedder.dimensions,
            normalized=self.embedder.normalized,
            allow_zero=True,
        )
        query_vector = query_vectors[0]
        candidates: dict[str, _Candidate] = {}
        lexical_count = 0
        semantic_scored = 0
        diagnostic_semantic_filtered = 0

        for granularity in self._granularities:
            if expression is not None:
                lexical_rows = self._lexical_rows(
                    namespace=ns,
                    granularity=granularity,
                    expression=expression,
                    positive_terms=positive_terms,
                    effective_at=effective_at,
                )
                lexical_count += len(lexical_rows)
                self._add_ranked_channel(
                    candidates,
                    lexical_rows,
                    channel=f"bm25:{granularity}",
                    weight=self.lexical_weight,
                    raw_score=lambda row: float(row["raw_score"]),
                )

            semantic_rows, scored, filtered = self._semantic_rows(
                namespace=ns,
                granularity=granularity,
                query_vector=query_vector,
                positive_terms=positive_terms,
                negative_terms=negative_terms,
                effective_at=effective_at,
            )
            semantic_scored += scored
            diagnostic_semantic_filtered += filtered
            self._add_ranked_channel(
                candidates,
                semantic_rows,
                channel=f"cosine:{granularity}",
                weight=self.semantic_weight,
                raw_score=lambda row: float(row["raw_score"]),
            )

        fused_count = len(candidates)
        deduplicated = self._deduplicate(tuple(candidates.values()))
        diversified = self._diversify(deduplicated)[: self.candidate_limit]
        diagnostics = HybridDiagnostics(
            query_embedding_calls=1,
            semantic_candidates_scored=semantic_scored,
            lexical_candidates_returned=lexical_count,
            fused_candidates=fused_count,
            candidates_after_deduplication=len(deduplicated),
            diagnostic_semantic_candidates_filtered=(
                diagnostic_semantic_filtered
            ),
        )
        return self._pack(
            namespace=ns,
            query=normalized_query,
            budget_tokens=budget,
            as_of=explicit_as_of,
            candidates=diversified,
            total_candidates=len(deduplicated),
            diagnostics=diagnostics,
        )

    def _snapshot_time(self, namespace: str) -> str:
        row = self._connection.execute(
            "SELECT MAX(occurred_at) AS value "
            "FROM hybrid_source_events WHERE namespace = ?",
            (namespace,),
        ).fetchone()
        value = row["value"] if row is not None else None
        if value is not None:
            return str(value)
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _scope_sql() -> str:
        # The correlated NOT EXISTS derives an irreversible predecessor end:
        # the first event-visible successor whose valid_from has begun closes the
        # predecessor at that valid_from.  The successor's own valid_to is
        # intentionally absent here; it filters the successor row through the
        # outer predicate but cannot reopen an older version after expiry.
        return """
            e.namespace = ?
            AND e.occurred_at <= ?
            AND e.valid_from <= ?
            AND (e.valid_to IS NULL OR e.valid_to > ?)
            AND NOT EXISTS (
                SELECT 1
                FROM hybrid_source_events successor
                WHERE successor.namespace = e.namespace
                  AND successor.supersedes_source_id = e.source_id
                  AND successor.occurred_at <= ?
                  AND successor.valid_from <= ?
            )
        """

    def _scope_params(
        self, namespace: str, effective_at: str
    ) -> tuple[object, ...]:
        return (
            namespace,
            effective_at,
            effective_at,
            effective_at,
            effective_at,
            effective_at,
        )

    def _lexical_rows(
        self,
        *,
        namespace: str,
        granularity: str,
        expression: str,
        positive_terms: Sequence[str],
        effective_at: str,
    ) -> tuple[dict[str, Any], ...]:
        # FTS5 supplies deterministic boolean matching only.  Its built-in BM25
        # uses global index statistics, so using that score would let documents
        # in another namespace perturb this namespace's ranking.  Fetch every
        # eligible match first, then compute BM25 from the equally scoped corpus.
        matched_rows = self._connection.execute(
            f"""
            SELECT
                c.id AS rowid,
                c.chunk_id,
                c.source_id,
                c.namespace,
                e.occurred_at,
                c.granularity,
                c.ordinal,
                c.start_char,
                c.end_char,
                c.content,
                c.content_hash,
                v.vector
            FROM hybrid_chunks_fts
            JOIN hybrid_chunks c ON c.id = hybrid_chunks_fts.rowid
            JOIN hybrid_source_events e
              ON e.namespace = c.namespace AND e.source_id = c.source_id
            JOIN hybrid_chunk_vectors v ON v.chunk_rowid = c.id
            WHERE hybrid_chunks_fts MATCH ?
              AND c.granularity = ?
              AND {self._scope_sql()}
            ORDER BY c.id ASC
            LIMIT ?
            """,
            (
                expression,
                granularity,
                *self._scope_params(namespace, effective_at),
                self.max_exact_candidates + 1,
            ),
        ).fetchall()
        if len(matched_rows) > self.max_exact_candidates:
            raise RuntimeError(
                "eligible lexical matches exceed max_exact_candidates; "
                "raise the declared exact-scan limit instead of silently sampling"
            )

        corpus_rows = self._connection.execute(
            f"""
            SELECT c.content
            FROM hybrid_chunks c
            JOIN hybrid_source_events e
              ON e.namespace = c.namespace AND e.source_id = c.source_id
            JOIN hybrid_chunk_vectors v ON v.chunk_rowid = c.id
            WHERE c.granularity = ?
              AND {self._scope_sql()}
            ORDER BY c.id ASC
            LIMIT ?
            """,
            (
                granularity,
                *self._scope_params(namespace, effective_at),
                self.max_exact_candidates + 1,
            ),
        ).fetchall()
        if len(corpus_rows) > self.max_exact_candidates:
            raise RuntimeError(
                "namespace-local BM25 corpus exceeds max_exact_candidates; "
                "raise the declared exact-scan limit instead of silently sampling"
            )
        if not corpus_rows:
            return ()

        corpus_tokens = [
            _lexical_tokens(str(row["content"])) for row in corpus_rows
        ]
        document_count = len(corpus_tokens)
        average_length = math.fsum(len(tokens) for tokens in corpus_tokens) / (
            document_count or 1
        )
        document_frequency = {
            term: sum(term in set(tokens) for tokens in corpus_tokens)
            for term in positive_terms
        }
        k1 = 1.2
        length_normalization = 0.75
        ranked: list[dict[str, Any]] = []
        for row in matched_rows:
            tokens = _lexical_tokens(str(row["content"]))
            counts = Counter(tokens)
            document_length = len(tokens)
            score = 0.0
            for term in positive_terms:
                frequency = counts[term]
                if frequency == 0:
                    continue
                matching_documents = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0
                    + (
                        document_count
                        - matching_documents
                        + 0.5
                    )
                    / (matching_documents + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0
                    - length_normalization
                    + length_normalization
                    * document_length
                    / (average_length or 1.0)
                )
                score += inverse_document_frequency * (
                    frequency * (k1 + 1.0) / denominator
                )
            item = dict(row)
            item["raw_score"] = score
            ranked.append(item)
        ranked.sort(key=lambda row: (-float(row["raw_score"]), row["chunk_id"]))
        return tuple(ranked[: self.channel_limit])

    def _semantic_rows(
        self,
        *,
        namespace: str,
        granularity: str,
        query_vector: Sequence[float],
        positive_terms: Sequence[str],
        negative_terms: Sequence[str],
        effective_at: str,
    ) -> tuple[tuple[dict[str, Any], ...], int, int]:
        rows = self._connection.execute(
            f"""
            SELECT
                c.id AS rowid,
                c.chunk_id,
                c.source_id,
                c.namespace,
                e.occurred_at,
                c.granularity,
                c.ordinal,
                c.start_char,
                c.end_char,
                c.content,
                c.content_hash,
                v.vector
            FROM hybrid_chunks c
            JOIN hybrid_source_events e
              ON e.namespace = c.namespace AND e.source_id = c.source_id
            JOIN hybrid_chunk_vectors v ON v.chunk_rowid = c.id
            WHERE c.granularity = ?
              AND {self._scope_sql()}
            ORDER BY c.id ASC
            LIMIT ?
            """,
            (
                granularity,
                *self._scope_params(namespace, effective_at),
                self.max_exact_candidates + 1,
            ),
        ).fetchall()
        if len(rows) > self.max_exact_candidates:
            raise RuntimeError(
                "eligible semantic rows exceed max_exact_candidates; "
                "raise the declared exact-scan limit instead of silently sampling"
            )
        if not any(query_vector):
            return (), len(rows), 0

        negative = set(negative_terms)
        positive = set(positive_terms)
        ranked: list[dict[str, Any]] = []
        diagnostic_filtered = 0
        for row in rows:
            content_terms = {
                token.casefold() for token in _lexical_tokens(row["content"])
            }
            if negative:
                if negative & content_terms:
                    continue
            vector = decode_float32_vector(
                row["vector"],
                dimensions=self.embedder.dimensions,
                normalized=self.embedder.normalized,
                allow_zero=False,
            )
            score = cosine_similarity(
                query_vector,
                vector,
                normalized=self.embedder.normalized,
            )
            if score < self.minimum_semantic_score:
                continue
            if (
                self._requires_diagnostic_token_overlap
                and not positive.intersection(content_terms)
            ):
                # Feature hashing can produce positive cosine from bucket
                # collisions.  In diagnostic mode it is only a lexical projection,
                # so a collision without exact normalized token overlap is not
                # evidence.  Real semantic providers intentionally skip this gate.
                diagnostic_filtered += 1
                continue
            item = dict(row)
            item["raw_score"] = score
            ranked.append(item)
        ranked.sort(key=lambda row: (-float(row["raw_score"]), row["chunk_id"]))
        return (
            tuple(ranked[: self.channel_limit]),
            len(rows),
            diagnostic_filtered,
        )

    def _add_ranked_channel(
        self,
        candidates: dict[str, _Candidate],
        rows: Sequence[sqlite3.Row | Mapping[str, Any]],
        *,
        channel: str,
        weight: float,
        raw_score: Any,
    ) -> None:
        seen: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            candidate = candidates.get(chunk_id)
            if candidate is None:
                candidate = _Candidate(
                    rowid=int(row["rowid"]),
                    chunk_id=chunk_id,
                    source_id=str(row["source_id"]),
                    namespace=str(row["namespace"]),
                    occurred_at=str(row["occurred_at"]),
                    granularity=str(row["granularity"]),
                    ordinal=int(row["ordinal"]),
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    content=str(row["content"]),
                    content_hash=str(row["content_hash"]),
                    vector=decode_float32_vector(
                        row["vector"],
                        dimensions=self.embedder.dimensions,
                        normalized=self.embedder.normalized,
                        allow_zero=False,
                    ),
                )
                candidates[chunk_id] = candidate
            candidate.reasons[channel] = HybridScoreReason(
                channel=channel,
                rank=rank,
                contribution=weight / (self.rrf_k + rank),
                raw_score=float(raw_score(row)),
            )

    def _deduplicate(
        self, candidates: Sequence[_Candidate]
    ) -> tuple[_Candidate, ...]:
        retained: list[_Candidate] = []
        for candidate in sorted(candidates, key=_candidate_key):
            duplicate = next(
                (
                    item
                    for item in retained
                    if item.source_id == candidate.source_id
                    and (
                        item.content_hash == candidate.content_hash
                        or _overlap_ratio(item, candidate) >= 0.8
                    )
                ),
                None,
            )
            if duplicate is None:
                retained.append(candidate)
                continue
            for channel, reason in candidate.reasons.items():
                previous = duplicate.reasons.get(channel)
                if previous is None or reason.rank < previous.rank:
                    duplicate.reasons[channel] = reason
        return tuple(sorted(retained, key=_candidate_key))

    def _diversify(
        self, candidates: Sequence[_Candidate]
    ) -> tuple[tuple[_Candidate, float], ...]:
        remaining = list(sorted(candidates, key=_candidate_key))
        selected: list[tuple[_Candidate, float]] = []
        if not remaining:
            return ()
        maximum = max(item.fused_score for item in remaining)
        while remaining:
            scored: list[tuple[float, _Candidate]] = []
            for candidate in remaining:
                relevance = candidate.fused_score / maximum if maximum > 0 else 0.0
                redundancy = 0.0
                for prior, _ in selected:
                    similarity = max(
                        0.0,
                        cosine_similarity(
                            candidate.vector,
                            prior.vector,
                            normalized=self.embedder.normalized,
                        ),
                    )
                    if candidate.source_id == prior.source_id:
                        similarity = max(similarity, 0.85)
                    redundancy = max(redundancy, similarity)
                score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * redundancy
                )
                scored.append((score, candidate))
            score, winner = min(
                scored,
                key=lambda pair: (-pair[0], _candidate_key(pair[1])),
            )
            selected.append((winner, score))
            remaining.remove(winner)
        return tuple(selected)

    def _expanded_evidence(
        self, candidate: _Candidate, mmr_score: float
    ) -> HybridEvidence:
        neighbor_rows = self._connection.execute(
            """
            SELECT start_char, end_char
            FROM hybrid_chunks
            WHERE namespace = ?
              AND source_id = ?
              AND granularity = ?
              AND ordinal BETWEEN ? AND ?
            ORDER BY ordinal ASC
            """,
            (
                candidate.namespace,
                candidate.source_id,
                candidate.granularity,
                candidate.ordinal - self.neighbor_radius,
                candidate.ordinal + self.neighbor_radius,
            ),
        ).fetchall()
        source_row = self._connection.execute(
            """
            SELECT content
            FROM hybrid_source_events
            WHERE namespace = ? AND source_id = ?
            """,
            (candidate.namespace, candidate.source_id),
        ).fetchone()
        if source_row is None:
            raise RuntimeError("source row disappeared during recall")
        start = min(
            (int(row["start_char"]) for row in neighbor_rows),
            default=candidate.start_char,
        )
        end = max(
            (int(row["end_char"]) for row in neighbor_rows),
            default=candidate.end_char,
        )
        if end - start > self.max_expansion_chars:
            candidate_length = candidate.end_char - candidate.start_char
            spare = max(0, self.max_expansion_chars - candidate_length)
            start = max(start, candidate.start_char - spare // 2)
            end = min(end, start + self.max_expansion_chars)
            if end < candidate.end_char:
                end = candidate.end_char
                start = max(0, end - self.max_expansion_chars)
        content = str(source_row["content"])[start:end]
        return HybridEvidence(
            chunk_id=candidate.chunk_id,
            source_id=candidate.source_id,
            namespace=candidate.namespace,
            occurred_at=candidate.occurred_at,
            content=content,
            start_char=start,
            end_char=end,
            fused_score=candidate.fused_score,
            mmr_score=mmr_score,
            reasons=tuple(
                candidate.reasons[channel]
                for channel in sorted(candidate.reasons)
            ),
            expanded=(
                start < candidate.start_char or end > candidate.end_char
            ),
        )

    @staticmethod
    def _windows_overlap(
        left: HybridEvidence, right: HybridEvidence
    ) -> bool:
        if left.source_id != right.source_id:
            return False
        intersection = max(
            0,
            min(left.end_char, right.end_char)
            - max(left.start_char, right.start_char),
        )
        denominator = min(
            left.end_char - left.start_char,
            right.end_char - right.start_char,
        )
        return denominator > 0 and intersection / denominator >= 0.5

    def _pack(
        self,
        *,
        namespace: str,
        query: str,
        budget_tokens: int,
        as_of: str | None,
        candidates: Sequence[tuple[_Candidate, float]],
        total_candidates: int,
        diagnostics: HybridDiagnostics,
    ) -> HybridRecall:
        empty = HybridRecall(
            namespace=namespace,
            query=query,
            budget_tokens=budget_tokens,
            as_of=as_of,
            omitted_count=total_candidates,
            diagnostics=diagnostics,
        )
        if empty.estimated_tokens > budget_tokens:
            raise BudgetTooSmallError(budget_tokens, empty.estimated_tokens)

        selected: list[HybridEvidence] = []
        for candidate, mmr_score in candidates:
            evidence = self._expanded_evidence(candidate, mmr_score)
            if any(self._windows_overlap(evidence, item) for item in selected):
                continue
            if self._fits_selection(
                empty, [*selected, evidence], total_candidates
            ):
                selected.append(evidence)
                continue

            compact = replace(
                evidence,
                content=candidate.content,
                start_char=candidate.start_char,
                end_char=candidate.end_char,
                expanded=False,
            )
            if self._fits_selection(
                empty, [*selected, compact], total_candidates
            ):
                selected.append(compact)
                continue
            truncated = self._fit_truncated(
                empty, selected, compact, total_candidates
            )
            if truncated is not None:
                selected.append(truncated)

        result = replace(
            empty,
            evidence=tuple(selected),
            omitted_count=max(0, total_candidates - len(selected)),
        )
        if result.estimated_tokens > budget_tokens:
            raise AssertionError("hybrid prompt compiler exceeded its token budget")
        return result

    @staticmethod
    def _fits_selection(
        template: HybridRecall,
        evidence: Sequence[HybridEvidence],
        total_candidates: int,
    ) -> bool:
        proposed = replace(
            template,
            evidence=tuple(evidence),
            omitted_count=max(0, total_candidates - len(evidence)),
        )
        return proposed.estimated_tokens <= template.budget_tokens

    def _fit_truncated(
        self,
        template: HybridRecall,
        selected: Sequence[HybridEvidence],
        evidence: HybridEvidence,
        total_candidates: int,
    ) -> HybridEvidence | None:
        raw = evidence.content
        low = _MIN_TRUNCATED_CHARS
        high = len(raw)
        fitted: HybridEvidence | None = None
        while low <= high:
            midpoint = (low + high) // 2
            shortened_text = raw[:midpoint].rstrip()
            if midpoint < len(raw):
                shortened_text += "…"
            shortened = replace(
                evidence,
                content=shortened_text,
                end_char=evidence.start_char + midpoint,
                truncated=midpoint < len(raw),
            )
            if self._fits_selection(
                template,
                [*selected, shortened],
                total_candidates,
            ):
                fitted = shortened
                low = midpoint + 1
            else:
                high = midpoint - 1
        return fitted

    def stats(self) -> HybridIndexStats:
        self._ensure_open()
        sources = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM hybrid_source_events"
            ).fetchone()[0]
        )
        chunks = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM hybrid_chunks"
            ).fetchone()[0]
        )
        vectors = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM hybrid_chunk_vectors"
            ).fetchone()[0]
        )
        by_granularity = tuple(
            (str(row["granularity"]), int(row["count"]))
            for row in self._connection.execute(
                """
                SELECT granularity, COUNT(*) AS count
                FROM hybrid_chunks
                GROUP BY granularity
                ORDER BY granularity
                """
            )
        )
        return HybridIndexStats(sources, chunks, vectors, by_granularity)

    def database_bytes(self) -> int:
        self._ensure_open()
        self._connection.commit()
        if str(self.path) == ":memory:":
            return 0
        return sum(
            candidate.stat().st_size
            for candidate in (
                self.path,
                Path(f"{self.path}-wal"),
                Path(f"{self.path}-shm"),
            )
            if candidate.exists()
        )

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiagnosticHybridRAGBaseline is closed")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "DIAGNOSTIC_HYBRID_LABEL",
    "DiagnosticHybridRAGBaseline",
    "HybridDiagnostics",
    "HybridEvidence",
    "HybridIndexStats",
    "HybridRecall",
    "HybridScoreReason",
]
