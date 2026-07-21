---
layout: docs
title: "API guide"
---

# API guide

This guide covers the public embedded-library API implemented in `evicortex` 0.3.0a0. The
project is pre-alpha; pin a commit and review changes before upgrading an integration.

## Imports

The common deterministic API is available from the package root:

```python
from evicortex import ClaimInput, EviCortex, MemoryCapsule, MemoryRecord
```

Optional local Transformers support is imported separately so importing `evicortex` never
loads a model framework:

```python
from evicortex.providers import TransformersEmbedder
```

## Open and close a store

```python
EviCortex.open(
    path=":memory:",
    *,
    embedder=None,
    extractor=None,
    auto_index=True,
    semantic_scan_cap=1024,
    minimum_semantic_score=0.05,
    reranker=None,
    rerank_limit=16,
    reranker_weight=1.25,
) -> EviCortex
```

`path` accepts `:memory:`, a string path, or a `pathlib.Path`. Prefer a context manager so
the SQLite connection closes deterministically:

```python
with EviCortex.open("memory.db") as memory:
    ...
```

`close()` is idempotent. Operations after close raise `StorageClosedError` from
`evicortex.errors`.

Provider identities and compatibility metadata are snapshotted at open time. Mutating a
configured provider's identity, dimensions, or normalization behavior causes validation to
fail instead of silently corrupting an index.

## Store exact evidence

```python
memory.remember(
    *,
    namespace: str,
    content: str,
    kind: str = "episode",
    source: str = "",
    occurred_at: datetime | str | None = None,
    recorded_at: datetime | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    idempotency_key: str | None = None,
    claims: Iterable[ClaimInput] = (),
) -> MemoryRecord
```

The evidence and explicit claims are validated and written atomically. `content` must be a
non-empty UTF-8 string; metadata must encode a JSON object. Timestamps are normalized to UTC.

`recorded_at` normally defaults to ingestion time. Supply it only for a trusted replay or
import where preserving the original recording time is intentional.

`idempotency_key` is unique per namespace. An identical replay returns the existing record;
a different request with the same key raises `IdempotencyConflictError`.

### `ClaimInput`

```python
ClaimInput(
    subject: str,
    predicate: str,
    value: str,
    valid_from: datetime | str | None = None,
    valid_to: datetime | str | None = None,
    confidence: float = 1.0,
    authority: str = "application",
    supersedes: str | None = None,
)
```

`confidence` must be between 0 and 1. `valid_to` must be later than `valid_from` when both
are supplied. `supersedes` is a prior claim ID in the same namespace; it creates a temporal
revision without deleting the earlier claim or evidence.

`MemoryRecord.claims` contains the stored `ClaimRecord` values, including generated claim
IDs and projected status.

## Extract claims before storing

```python
memory.remember_extracted(
    *,
    namespace: str,
    content: str,
    kind: str = "episode",
    source: str = "",
    occurred_at: datetime | str | None = None,
    recorded_at: datetime | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    idempotency_key: str | None = None,
    claim_ids_by_source: Mapping[str, str] | None = None,
    extractor: ClaimExtractor | None = None,
) -> MemoryRecord
```

A per-call extractor overrides the one configured in `open()`. Extraction and complete
span/provenance validation happen before the evidence transaction. Invalid provider output
therefore cannot leave a partial memory.

Use `claim_ids_by_source` when an extractor grammar explicitly names the source of a claim
to supersede. EviCortex never guesses that mapping.

Retrieve the persisted audit record with:

```python
memory.get_extraction(namespace: str, memory_id: str) -> ExtractionRecord
```

The built-in diagnostic extractor can be exercised as follows:

```python
from evicortex import CanonicalClaimExtractor, EviCortex

text = (
    "Ada selected dark mode. "
    "Canonical claim: subject=user:ada; predicate=prefers_theme; value=dark; "
    "valid_from=2026-04-18T09:30:00Z; confidence=1.000; authority=application."
)

with EviCortex.open(":memory:", extractor=CanonicalClaimExtractor()) as memory:
    record = memory.remember_extracted(namespace="demo/ada", content=text)
    audit = memory.get_extraction("demo/ada", record.id)
```

`CanonicalClaimExtractor` understands only this exact grammar. Implement the
`ClaimExtractor` protocol for another provider and return `ClaimExtraction` with one exact
`ClaimProvenance` span per claim plus any `ExtractionDiagnostic` values.

## Recall and expand

```python
memory.recall(
    *,
    namespace: str,
    query: str,
    budget_tokens: int = 512,
    as_of: datetime | str | None = None,
    limit: int = 20,
) -> MemoryCapsule
```

`budget_tokens` is a hard limit under the built-in UTF-8-bytes/3 estimator. `limit` bounds
the returned candidate count; it does not override the budget. `as_of` selects valid-time
state and is not a transaction-time `known_at` filter.

Important `MemoryCapsule` members are:

| Member | Purpose |
| --- | --- |
| `facts` | Compact structured claims with evidence IDs and score reasons |
| `episodes` | Exact-evidence excerpts, possibly marked as truncated |
| `conflicts` | Competing claim values retained for the reader |
| `evidence_ids` | Sorted unique IDs supporting packed facts and episodes |
| `omitted_count` | Candidates not included in the hard estimated-token budget |
| `trace` | Per-candidate score contributions and selection decisions |
| `diagnostics` | Non-prompt operational facts, including optional-channel state |
| `estimated_tokens` | Deterministic estimate of `to_prompt()` |
| `to_prompt()` | Stable compact JSON for a downstream model |

Retrieve an exact supporting record with:

```python
memory.expand(*, namespace: str, memory_id: str) -> MemoryRecord
```

Missing IDs and namespace mismatches both raise `KeyError`; the API does not reveal whether
another namespace owns the supplied ID.

## Semantic projection lifecycle

Configure any object implementing the `Embedder` protocol:

```python
from evicortex import EviCortex, HashingEmbedder

with EviCortex.open(
    "memory.db",
    embedder=HashingEmbedder(dimensions=256),
) as memory:
    record = memory.remember(namespace="demo/ada", content="Exact evidence")
    status = memory.semantic_status("demo/ada")
    assert status.complete
```

`HashingEmbedder` is non-semantic diagnostic feature hashing. It is suitable for testing
projection persistence and repair, not for semantic-quality claims.

With `auto_index=True`, each successful evidence write is followed by semantic projection.
Evidence is already committed when that provider call runs. A provider failure raises
`SemanticProjectionError` with `memory_id` and `evidence_committed=True`; retry repair with a
compatible provider rather than writing the evidence again.

The lifecycle methods are:

```python
memory.index_semantic(record: MemoryRecord) -> EmbeddingUpsertResult
memory.semantic_status(namespace: str) -> SemanticIndexStatus
memory.sync_semantic(namespace: str, batch_size: int = 64) -> SemanticIndexStatus
```

Set `auto_index=False` for an explicit indexing workflow, then call `index_semantic(record)`
or batch repairs with `sync_semantic(namespace)`. `SemanticIndexStatus` reports memory and
claim missing/stale IDs plus aggregate `complete`, `registered`, `missing_count`, and
`stale_count` properties.

Exact semantic search is bounded by `semantic_scan_cap`. Overflow produces a diagnostic
rather than silently claiming a complete exact scan. EviCortex does not yet provide an ANN
index.

### Optional dense embeddings

Install the optional dependencies:

```powershell
python -m pip install -e ".[local-transformers]"
```

`TransformersEmbedder` defaults to offline loading. A remote model name must use a pinned
40-character commit, and the model/tokenizer must already be available in the local cache:

```python
from evicortex import EviCortex
from evicortex.providers import TransformersEmbedder

embedder = TransformersEmbedder(
    "unsloth/bge-small-en-v1.5",
    revision="7382f1122c10708a1faa0bbe548674a14b1ffe7e",
    pooling="cls",
    query_prefix="Represent this sentence for searching relevant passages: ",
    document_prefix="",
    max_length=128,
    overflow_policy="error",
    local_files_only=True,
)

with EviCortex.open("memory.db", embedder=embedder) as memory:
    memory.sync_semantic("demo/ada")
```

This exact model configuration is the retained Phase C smoke configuration, not a general
recommendation or quality guarantee. For a local filesystem model, calculate
`local_transformers_artifact_sha256(path)` and pass the digest as
`local_artifact_sha256`. Inputs longer than `max_length` fail by default instead of being
silently truncated.

## Optional reranking

An object implementing `Reranker` receives at most `rerank_limit` filtered, temporally valid,
deduplicated `RerankCandidate` values. It returns finite `RerankResult` scores for known
candidate IDs. The provider identity and ordinal contribution appear in retrieval traces.

EviCortex exposes the seam and validation contract but does not currently bundle a
production cross-encoder adapter.

## Integrity, repair, diagnostics, and purge

```python
memory.verify_integrity(namespace: str | None = None) -> IntegrityReport
memory.rebuild_indexes() -> None
memory.stats(namespace: str | None = None) -> StoreStats
memory.purge_memory(*, namespace: str, memory_id: str) -> bool
```

- `verify_integrity()` checks evidence digests and deterministic projection consistency.
- `rebuild_indexes()` repairs deterministic claim-status and FTS views only. It preserves
  semantic rows and never invokes a provider.
- `stats()` returns counts for memories, claims, namespaces, and projection rows.
- `purge_memory()` deletes one scoped evidence record and its derived rows, returning whether
  a row was removed.

Logical purge is not forensic erasure from copied databases, backups, SQLite journals,
filesystem snapshots, or storage media. Define retention and backup deletion outside this
library.

## Errors and validation

Predictable storage exceptions live in `evicortex.errors`; semantic boundary exceptions most
often used by applications are exported from the package root. Common cases include:

| Exception | Meaning |
| --- | --- |
| `ValidationError` | Input violates a storage invariant |
| `IdempotencyConflictError` | A key was reused for a different request |
| `StorageClosedError` | The store has already closed |
| `FTS5UnavailableError` | The SQLite build lacks FTS5 |
| `IntegrityViolationError` | Persisted state violates an invariant |
| `SemanticNotConfiguredError` | A semantic-only operation has no embedder |
| `SemanticProjectionError` | Evidence may be committed, but semantic projection failed |
| `EmbeddingProfileNotFoundError` | A named projection profile is absent |
| `EmbeddingValidationError` | A provider returned incompatible vectors |

Validate and log these failures without logging sensitive evidence content. See
[FAQ and limitations](faq.md) for security and deployment boundaries.
