"""A transparent fixed-chunk SQLite FTS5 lexical retrieval baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

try:
    # The comparison must use the exact same estimator as EviCortex.  Keeping the
    # fallback identical to Prototype A's implementation lets this module remain
    # importable while the core package is being bootstrapped.
    from evicortex.budget import estimate_tokens as _shared_estimate_tokens
except ImportError:  # pragma: no cover - exercised only before core installation
    def _shared_estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text.encode("utf-8")) + 2) // 3)


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
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


def estimate_tokens(text: str) -> int:
    """Return the exact deterministic estimator shared with EviCortex."""

    return _shared_estimate_tokens(text)


def fixed_chunks(text: str, *, size: int, overlap: int) -> tuple[str, ...]:
    """Split text into deterministic character chunks without losing source text."""

    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must satisfy 0 <= overlap < size")
    if not text:
        return ("",)

    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class BaselineEvidence:
    """One source-preserving chunk selected by the lexical baseline."""

    chunk_id: str
    source_id: str
    namespace: str
    occurred_at: str
    content: str
    estimated_tokens: int
    score: float


@dataclass(frozen=True, slots=True)
class BaselineRecall:
    """Token-bounded result returned by :class:`FixedChunkFTS5Baseline`."""

    evidence: tuple[BaselineEvidence, ...]
    estimated_tokens: int
    budget_tokens: int

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.source_id for item in self.evidence))

    def to_prompt(self) -> str:
        return "\n\n".join(
            f"[source={item.source_id} occurred_at={item.occurred_at}]\n{item.content}"
            for item in self.evidence
        )


class FixedChunkFTS5Baseline:
    """A mechanical lexical RAG baseline using fixed chunks and SQLite FTS5.

    Namespace and explicit ``as_of`` filters are applied in SQL before ranking.
    Ranking is BM25 followed by stable timestamp/source/chunk tie breakers.  This
    baseline performs no semantic retrieval, answer generation, or temporal conflict
    resolution; benchmark reports must identify it accordingly.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_chars: int = 320,
        overlap_chars: int = 48,
        candidate_limit: int = 100,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        # Validate chunk settings at construction rather than on the first write.
        fixed_chunks("x", size=chunk_chars, overlap=overlap_chars)
        self.path = Path(path)
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.candidate_limit = candidate_limit
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = DELETE")
        self._create_schema()

    @classmethod
    def open(
        cls,
        path: str | Path,
        **kwargs: object,
    ) -> "FixedChunkFTS5Baseline":
        return cls(path, **kwargs)

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_events (
                    source_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    content TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    source_id TEXT NOT NULL REFERENCES source_events(source_id),
                    namespace TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    content TEXT NOT NULL,
                    estimated_tokens INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_scope_time
                    ON chunks(namespace, occurred_at);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(content, tokenize='unicode61');
                """
            )

    def remember(
        self,
        *,
        namespace: str,
        content: str,
        source_id: str,
        occurred_at: str,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must not be empty")
        if not source_id:
            raise ValueError("source_id must not be empty")
        pieces = fixed_chunks(
            content,
            size=self.chunk_chars,
            overlap=self.overlap_chars,
        )
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO source_events(source_id, namespace, occurred_at, content)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, namespace, occurred_at, content),
            )
            del cursor
            for index, piece in enumerate(pieces):
                chunk_id = f"{source_id}#chunk-{index:04d}"
                cursor = self._connection.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, source_id, namespace, occurred_at, content,
                        estimated_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        source_id,
                        namespace,
                        occurred_at,
                        piece,
                        estimate_tokens(piece),
                    ),
                )
                self._connection.execute(
                    "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
                    (cursor.lastrowid, piece),
                )

    @staticmethod
    def _match_expression(query: str) -> str | None:
        tokens = [
            token.casefold()
            for token in _TOKEN_RE.findall(query)
            if token.casefold() not in _STOP_WORDS and len(token) > 1
        ]
        # Stable de-duplication prevents repeated query words changing BM25 weight.
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return None
        escaped = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
        return " OR ".join(escaped)

    def recall(
        self,
        *,
        namespace: str,
        query: str,
        budget_tokens: int,
        as_of: str | None = None,
    ) -> BaselineRecall:
        if budget_tokens < 0:
            raise ValueError("budget_tokens must be non-negative")
        expression = self._match_expression(query)
        if budget_tokens == 0 or expression is None:
            return BaselineRecall((), 0, budget_tokens)

        rows = self._connection.execute(
            """
            SELECT
                chunks.chunk_id,
                chunks.source_id,
                chunks.namespace,
                chunks.occurred_at,
                chunks.content,
                chunks.estimated_tokens,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks ON chunks.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
              AND chunks.namespace = ?
              AND (? IS NULL OR chunks.occurred_at <= ?)
            ORDER BY score ASC, chunks.occurred_at DESC,
                     chunks.source_id ASC, chunks.chunk_id ASC
            LIMIT ?
            """,
            (expression, namespace, as_of, as_of, self.candidate_limit),
        ).fetchall()

        selected: list[BaselineEvidence] = []
        for row in rows:
            token_count = int(row["estimated_tokens"])
            candidate = BaselineEvidence(
                chunk_id=str(row["chunk_id"]),
                source_id=str(row["source_id"]),
                namespace=str(row["namespace"]),
                occurred_at=str(row["occurred_at"]),
                content=str(row["content"]),
                estimated_tokens=token_count,
                score=float(row["score"]),
            )
            proposed = BaselineRecall(tuple((*selected, candidate)), 0, budget_tokens)
            if estimate_tokens(proposed.to_prompt()) <= budget_tokens:
                selected.append(candidate)
        result = BaselineRecall(tuple(selected), 0, budget_tokens)
        result = BaselineRecall(
            result.evidence,
            estimate_tokens(result.to_prompt()),
            budget_tokens,
        )
        assert result.estimated_tokens <= budget_tokens
        return result

    def database_bytes(self) -> int:
        self._connection.commit()
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
        self._connection.close()

    def __enter__(self) -> "FixedChunkFTS5Baseline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
