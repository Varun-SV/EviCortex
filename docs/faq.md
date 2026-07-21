---
layout: docs
title: "FAQ and limitations"
---

# FAQ and limitations

## Is EviCortex production-ready?

No. It is a pre-alpha embedded research library. The current implementation is useful for
experiments and local integration work, but it does not provide a network service,
authentication, authorization, encryption, distributed storage, mature migrations for
unattended production fleets, or operational service-level guarantees.

## Does EviCortex replace RAG or prove it is better than RAG?

No. EviCortex still retrieves evidence for a downstream model and is retrieval-augmented in
the broad technical sense. Its research question concerns whether evidence-preserving,
temporal structure and budget-aware compilation improve a quality/cost tradeoff relative to
named baselines.

The retained Prototype A comparison uses a mechanical lexical chunk baseline. Prototype B
uses diagnostic feature hashing. Phase C uses a real encoder on three synthetic questions.
None establishes superiority over a tuned dense, hybrid, graph, agentic, or production RAG
system. See [Benchmarking and reproducibility](benchmarking.md).

## Is a namespace a security boundary?

No. A namespace is a caller-selected logical scope that EviCortex filters before ranking.
The host application must authenticate the caller, determine authorized namespaces, and
prevent untrusted clients from selecting arbitrary scopes. The local SQLite file is not
encrypted.

## Can I store sensitive or regulated data?

Only after designing the missing controls outside this library. At minimum, consider access
control, encryption at rest and in transit, secrets management, audit logging, retention,
backup deletion, filesystem permissions, incident response, and jurisdiction-specific
requirements.

`purge_memory()` removes a scoped logical record and cascading derived rows. It cannot promise
forensic erasure from copied databases, SQLite journals, filesystem snapshots, backups, or
storage media.

## Is evidence truly immutable?

Normal indexing and claim revisions do not rewrite stored evidence. New information appends a
new record and can supersede an older claim while preserving both sources. Authorized purge
is the deliberate exception: privacy and retention requirements take precedence over an
append-only design.

## Does `as_of` provide full bitemporal querying?

No. EviCortex stores occurrence and recording time and supports valid-time claim selection
through `as_of`. It does not yet expose a transaction-time `known_at` query asking what the
system knew at a past ingestion time. Do not describe the current API as fully bitemporal.

## Does EviCortex extract facts from ordinary text?

Not by itself. Applications can pass explicit `ClaimInput` values or configure a provider
implementing `ClaimExtractor`. EviCortex validates exact span provenance and persists the
audit record.

The bundled `CanonicalClaimExtractor` recognizes only the explicit synthetic benchmark
grammar. It is intentionally not natural-language understanding. Any production extractor
needs separate held-out evaluation for false positives, missed facts, temporal boundaries,
contradictions, cost, and latency.

## Are semantic embeddings required?

No. The deterministic kernel uses SQLite FTS5 and structured retrieval without third-party
runtime dependencies. An optional embedder adds exact-semantic channels and persistent
projections.

`HashingEmbedder` is a deterministic, non-semantic diagnostic. `TransformersEmbedder` is an
optional pinned local dense adapter, but choosing and validating a model remains the
application's responsibility.

## What happens when embedding fails after a write?

With automatic semantic indexing enabled, exact evidence and claims commit before the
provider projection. A provider failure raises `SemanticProjectionError` containing the
memory ID and `evidence_committed=True`. Do not blindly write the observation again. Restore
a compatible provider and use `index_semantic(record)` or `sync_semantic(namespace)`.

## Why does `rebuild_indexes()` not rebuild embeddings?

Deterministic FTS and claim-status views can be reconstructed locally without external
effects. Embeddings depend on a specific provider profile and can be expensive or
non-deterministic across unpinned versions. `rebuild_indexes()` therefore preserves semantic
rows and never invokes a provider; `sync_semantic(namespace)` is explicit, scoped, and
profile-aware.

## Does semantic retrieval scale to 100,000 or millions of records?

That has not been established. The current semantic path uses exact cosine scoring after SQL
namespace/time filtering and enforces a scan cap. An audited approximate-nearest-neighbor
candidate layer, with recall measured against exact cosine, remains open work.

SQLite also limits write concurrency and distributed deployment. A future storage adapter
must pass the same evidence, temporal, isolation, expansion, and traceability conformance
tests.

## Does a hard capsule budget guarantee my model context limit?

No. EviCortex mechanically bounds `MemoryCapsule.to_prompt()` using a deterministic estimate
of one token per three UTF-8 bytes. That is not the exact tokenizer for a reader model, and
it covers only the capsule—not system instructions, tools, conversation messages, response
space, or transport wrappers.

Measure the complete prompt with the deployed model tokenizer and reserve headroom in the
application.

## Does an empty or non-empty capsule tell me whether a question is answerable?

Not reliably. Current tests show inadequate retrieval-level abstention on partial out-of-
distribution and semantic-smoke questions. A non-empty capsule can contain only distractors,
and an empty capsule can reflect a retrieval miss. Support verification, calibrated
thresholds, reader refusal evaluation, and application policy remain required.

## Are conflicts and confidence values probabilities?

No. EviCortex can surface incompatible claim values and carries caller/provider confidence,
but it does not currently calibrate those scores into probabilities of truth. Authority and
confidence are evidence for application policy, not a replacement for it.

## Can I use `recorded_at` for imports?

Yes, but treat it as a trusted field. The default is the actual ingestion time. Supplying a
historical value is appropriate for controlled replay or migration, but letting an untrusted
caller choose it can damage audit semantics.

Naive timestamps are currently interpreted as UTC. Prefer timezone-aware datetimes or ISO
8601 strings with `Z` or an explicit offset.

## Is the package available from PyPI?

The source distribution is named `evicortex-research`, but the repository does not claim
that a public PyPI namespace has been reserved or released. Install from a source checkout
and pin a commit until a release process is documented.

## Is the EviCortex name legally cleared?

No formal trademark clearance is claimed. The code and documentation use EviCortex as the
selected project identity; legal name clearance is a separate step before commercial use.

## Where should I start when something goes wrong?

1. Run `verify_integrity()` for the affected namespace.
2. Inspect `MemoryCapsule.trace` and `diagnostics` without exposing sensitive content in logs.
3. Check `semantic_status(namespace)` when an embedder is configured.
4. Use `rebuild_indexes()` for deterministic projections and `sync_semantic(namespace)` for
   provider-derived projections.
5. Reproduce the issue with the exact commit, database copy, provider identity, configuration,
   and query budget.

For method details, see the [API guide](api-guide.md). For evaluation issues, see the
[benchmark contract](benchmark-contract.md).
