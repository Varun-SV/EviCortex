---
layout: docs
title: "Prototype B: semantic and extraction plan"
---

# Prototype B: semantic and extraction plan

Status: the provider-neutral diagnostic slice is implemented and verified. A pinned local dense
provider and external-dataset runner now exist and pass an integration smoke. The official
dataset, reader-model, ANN, and strong-comparator gates remain open; no result from this phase
may be called a strong-RAG comparison yet.

## Purpose

Prototype A showed that explicit temporal structure can improve a small fixed-chunk lexical
comparison. Prototype B must determine whether that advantage survives when both systems get
the same query/document embedder, metadata, query compute, reranking allowance, and context
budget. It must also account for the cost and errors of producing structured claims from raw
evidence.

The phase separates three questions:

1. Does EviCortex's representation improve retrieval when the semantic provider is held fixed?
2. Does automatic claim extraction improve enough queries to repay its latency, token, and
   error costs?
3. Does the complete system beat a tuned hybrid chunk pipeline on external data and a shared
   reader model?

## Provider boundary

Optional components implement narrow, versioned contracts:

- `Embedder`: stable identity and dimensions plus order-preserving batch embeddings.
- `ClaimExtractor`: evidence-bound claims, exact source spans, and abstention diagnostics.
- `Reranker`: scores only known candidates and returns a validated subset.

Provider outputs are untrusted until their cardinality, dimensions, finite values, IDs, and
normalization promises are validated. Provider identity is part of every derived index so a
model or preprocessing change cannot silently mix incompatible vectors.

The bundled feature-hashing embedder and canonical-grammar extractor exist for deterministic
tests and lifecycle benchmarks. Feature hashing is not a neural semantic model, and the
canonical parser is not natural-language understanding.

Implemented public boundaries include `Embedder`, `ClaimExtractor`, `Reranker`,
`EvidenceContext`, validated float32 embedding batches, immutable embedding profiles, and
defensive validation of provider identity, cardinality, dimensions, finite values, exact
source spans, content digests, claim ordering, and diagnostics.

## Semantic projection

Exact evidence and claims remain the source of truth. Embeddings are replaceable projections
containing the provider profile, canonical input digest, item/evidence ID, namespace, and a
portable vector encoding. Purge cascades through them; index regeneration never rewrites
evidence.

The first executable diagnostic uses exact cosine scoring after SQL applies namespace and time
filters. A hard scan cap must produce an explicit diagnostic when exceeded; it must never
silently present an incomplete scan as exact retrieval. This provides a deterministic oracle
at 1k-10k items. It is intentionally not the 100k+ design.

That exact scorer is now persisted in schema v5. Memory and claim projections retain provider
profile and canonical-input hashes, survive reopen/rebuild, cascade on purge, report missing
or stale rows, and can be repaired explicitly with `sync_semantic(namespace)`. Deterministic
claim/FTS rebuilds do not invoke a provider or cross namespace boundaries.

Before the 100k scale gate, EviCortex needs a provider-neutral approximate-neighbor candidate
interface backed by an audited vector extension or service. That candidate layer must report
its own recall against the exact scorer. A custom LSH implementation is deferred because its
tuning and recall loss could otherwise be mistaken for a representation result.

## Raw-text extraction

An extractor runs before the atomic evidence write. If it fails, no partial claims are stored.
If it abstains, the exact evidence is still retained as a claimless observation. Every emitted
claim must persist:

- extractor identity and configuration revision;
- evidence content digest;
- exact character span or an explicit whole-evidence marker;
- confidence and authority;
- the resulting claim ID; and
- diagnostics for rejected or unresolved spans.

The deterministic benchmark parser accepts only the explicit canonical grammar already present
in the synthetic source stream. A real extractor must be evaluated on held-out human-labelled
text, including false-positive, contradiction, temporal-boundary, and supersession errors.

`EviCortex.remember_extracted` now owns this lifecycle. It invokes one configured or per-call
extractor, validates the complete result before writing, then atomically stores evidence,
claims, extractor identity, content digest, ordered span derivations, generated claim IDs, and
diagnostics. `get_extraction` provides a namespace-scoped audit record. Semantic indexing is a
post-commit projection; a provider failure reports that evidence was committed and an
idempotent retry repairs it.

## Hybrid comparator

The local diagnostic comparator will include:

- append-only source/session records and version metadata;
- sentence-aware child chunks at two granularities;
- FTS5 BM25 and the same embedder used by EviCortex;
- weighted reciprocal-rank fusion;
- duplicate removal, source diversity, and neighboring/parent expansion;
- namespace and `as_of` filtering before both ranking channels; and
- the same serialized-context estimated-token accounting and hard budget.

With the bundled hashing provider this is labelled **BM25 + hashing diagnostic**, not strong
hybrid RAG. The headline comparator requires a pinned dense encoder, query/document encoding
rules, exact model revision, optional shared cross-encoder, and equivalent compute accounting.

The diagnostic comparator is implemented with boundary-aware chunks at two granularities,
namespace/time/version filtering before ranking, namespace-local BM25, exact cosine, reciprocal
rank fusion, deduplication/MMR, neighbor expansion, and the same hard estimated-token context budget. Exact-scan
overflow fails closed. Diagnostic hashing collisions cannot become evidence without informative
literal query overlap; that safeguard is not applied to providers that declare real semantic
capability.

Explicit phrases such as “not an accessory” currently use literal token exclusion across all
channels and providers. This is deterministic and prevents a negated distractor flood, but it
can reject a source that mentions the same term while linguistically negating it. Structured
constraint parsing and negation-aware reranking remain open work.

## Implemented checkpoint

| Item | Status |
| --- | --- |
| Provider-neutral embedding/extraction/reranking contracts | Implemented and tested |
| Durable extraction spans, identity, diagnostics, and generated claim-ID mapping | Implemented in schema v5 |
| Persistent memory/claim embeddings and namespace-scoped repair | Implemented |
| Bounded lexical + exact-semantic fusion with non-prompt diagnostics | Implemented |
| Metadata-aware BM25 + hashing comparator | Implemented, diagnostic only |
| Multi-seed 100-event diagnostic smoke | Run for seeds 7, 11, and 19 |
| 1,000-event exact-vector scale point | Run with a symmetric 4,096-row cap |
| Pinned dense encoder adapter and real-model integration | Implemented; cached BGE smoke passed |
| Core bounded reranker seam | Implemented; production cross-encoder adapter not yet added |
| Shared reader and exact reader-tokenizer accounting | Not started |
| 10k exact-vector and 100k ANN scale gates | Not run |
| Strict LongMemEval adapter/runner | Implemented and tested on a synthetic schema fixture |
| Official external benchmark split | Not run; official artifact download is still required |

The retained diagnostic result is in
[Prototype B diagnostic benchmark result](results/prototype-b-diagnostic.md).

## Evidence gates

The broader Prototype B research phase is not complete until it reports:

- multi-seed synthetic curves at 512, 1k, 2k, 4k, and 8k context budgets;
- 1k and 10k exact-vector scale points plus the 100k ANN crossover;
- extraction accuracy and cost separately from retrieval accuracy and cost;
- paired representation/retriever crossover ablations;
- exact tokenizer counts and a shared reader-model answer metric;
- tuned dense-hybrid, no-memory, full-history-when-it-fits, and oracle-evidence controls; and
- at least one external conversation-memory benchmark using its official split.

Until those gates pass, allowed wording is limited to the named configuration and workload.
“More efficient than RAG” remains an unverified research hypothesis.
