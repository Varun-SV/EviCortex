---
layout: docs
title: "Core concepts"
---

# Core concepts

EviCortex separates exact evidence from the mutable structures used to retrieve it. That
separation is the central design choice: a derived claim or vector may be incomplete or
wrong, while the observation from which it came remains addressable and auditable.

For the complete design contract, see [Architecture and invariants](architecture.md).

## The three planes

| Plane | Stores or produces | Key property |
| --- | --- | --- |
| Evidence ledger | Original observations, source, digest, scope, and times | Durable source of truth |
| Memory indexes | Claims, FTS rows, vectors, entities, links, and temporal status | Derived and repairable |
| Context compiler | A query-specific `MemoryCapsule` | Traceable and hard-budgeted |

The architecture sometimes calls these planes “Genome,” “Engram,” and “Transcript.” They
are software metaphors, not claims that human memory is stored in DNA or that SQLite
reproduces neurobiology.

## Evidence is not a claim

An evidence record is an immutable observation such as a conversation turn, document
fragment, tool result, or application event. Its `MemoryRecord` includes:

- a stable ID and namespace;
- the exact text and SHA-256 content digest;
- a source locator and application-defined kind;
- occurrence and recording timestamps;
- JSON-compatible metadata; and
- zero or more evidence-linked claims.

A claim is a structured interpretation:

```text
subject + predicate + value + validity + confidence + authority
```

For example, the observation “Ada switched to dark mode” can support the claim
`user:ada / prefers_theme / dark`. The claim is useful for temporal and entity-aware
retrieval, but it does not replace the observation. A user can expand the capsule's
evidence ID to recover the exact text.

## Namespace scope

All retrieval happens inside one namespace, and EviCortex applies that filter before
ranking. Expansion and extraction lookup deliberately use the same missing-ID behavior for
an absent record and a record in another namespace.

This protects against accidental cross-namespace retrieval inside the embedded library. It
does not identify the caller, authorize access, encrypt the database, or create a tenant
security boundary. Those responsibilities belong to the host application or a future
service layer.

## Time and revisions

EviCortex records several different notions of time:

- `occurred_at`: when the represented event happened;
- `recorded_at`: when the evidence was ingested, or the trusted replay time supplied by
  the caller;
- `valid_from` and `valid_to`: the interval during which a claim applies; and
- `as_of`: the valid-time point used by a recall request.

When a new claim supersedes an earlier claim, EviCortex retains both evidence records,
marks the projection of the older claim as superseded, and bounds its validity where
appropriate. This supports both current and historical questions.

`as_of` is a valid-time query. The current API does not expose a transaction-time
`known_at` query, so EviCortex should not yet be described as fully bitemporal.

Naive Python datetimes and ISO strings without a timezone are currently interpreted as UTC;
explicit offsets or `Z` are safer at application boundaries. Persisted timestamps are
normalized to UTC.

## Derived projections and semantic profiles

SQLite FTS rows and claim-status views are deterministic projections. `rebuild_indexes()`
can recreate them without changing evidence.

Embeddings are provider-derived projections. Their durable profile includes provider
identity, dimensions, normalization behavior, and canonical input hashes. EviCortex does
not silently mix projections from incompatible profiles and does not call a provider during
`rebuild_indexes()`. Use `semantic_status(namespace)` and `sync_semantic(namespace)` with a
compatible configured provider.

The built-in `HashingEmbedder` is deterministic feature hashing for tests and lifecycle
diagnostics. It is not a semantic model. The optional `TransformersEmbedder` supplies a
pinned local dense path, but current retrieval is an exact, bounded cosine scan rather than
an audited approximate-nearest-neighbor index.

## Extraction is an audited provider boundary

`remember_extracted()` sends an immutable `EvidenceContext` to a configured
`ClaimExtractor`. Before anything is written, EviCortex validates that every emitted claim
has an exact, ordered, non-overlapping source span, matching content digest, and stable
extractor identity.

The durable extraction record contains the provider identity, content digest, generated
claim IDs, source spans, and diagnostics. If the extractor abstains, exact evidence can
still be stored without claims.

The bundled `CanonicalClaimExtractor` parses only an explicit benchmark grammar. It ignores
ordinary natural-language statements and must not be treated as an LLM or production
information extractor.

## Retrieval and context compilation

Recall can combine lexical, entity, graph, temporal, optional semantic, and optional
reranking contributions. EviCortex fuses ranked candidates, filters temporal state,
deduplicates results, surfaces relevant conflicts, and packs selected evidence under the
requested budget.

The resulting `MemoryCapsule` has two audiences:

- `to_prompt()` contains only the compact JSON intended for a reader model; and
- `trace` and `diagnostics` contain operational explanations that stay outside the prompt.

A capsule can include structured facts, evidence excerpts, conflicts, stable evidence IDs,
an omitted-result count, and the valid-time boundary. Ranking reasons record the channels
and contributions that selected each candidate.

The current hard budget uses a deterministic estimate of one token per three UTF-8 bytes,
not a model tokenizer. It is mechanically enforced for the serialized capsule, but an
application must separately account for its system prompt, user message, schema, and reader
model tokenizer.

## A typical lifecycle

1. The host authenticates the caller and chooses an authorized namespace.
2. `remember()` stores exact evidence plus application-supplied claims, or
   `remember_extracted()` validates a provider result before the atomic write.
3. Deterministic indexes update in the same storage transaction.
4. If configured, semantic projections are created after the evidence commit.
5. `recall()` compiles a bounded capsule and an inspectable trace.
6. `expand()` retrieves exact evidence when more detail or verification is needed.
7. `verify_integrity()` checks evidence digests and projection consistency.
8. `rebuild_indexes()` repairs deterministic projections; `sync_semantic()` repairs one
   authorized namespace and embedding profile.
9. `purge_memory()` removes an authorized evidence record and its derived rows when retention
   or privacy requirements take precedence over historical preservation.

## What the design does not guarantee

Durable evidence does not imply perfect retrieval. Structured claims can be wrong, semantic
models can retrieve distractors, a token budget can omit useful evidence, and a downstream
reader can still generate an unsupported answer. These failure modes are why EviCortex
preserves provenance, reports diagnostics, and treats benchmark claims as configuration-
specific. See [FAQ and limitations](faq.md) and the
[benchmark contract](benchmark-contract.md).
