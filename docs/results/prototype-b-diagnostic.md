---
layout: docs
title: "Prototype B diagnostic benchmark result"
---

# Prototype B diagnostic benchmark result

Status: verified local diagnostic checkpoint, 2026-07-21. This is not a dense-RAG,
reader-model, external-benchmark, or publication-quality comparison.

## Claim boundary

This experiment asks whether the Phase B wiring is mechanically sound when both systems use
the same deterministic projection provider. It does **not** ask whether feature hashing
understands language. `HashingEmbedder` is explicitly non-semantic, and the comparator is a
local BM25 + exact-cosine diagnostic—not a validated strong RAG system.

The candidate is EviCortex with exact evidence, versioned claims, durable extraction
provenance, persistent memory/claim vectors, lexical/entity/graph/time channels, and a bounded
capsule compiler. The comparator stores the same sources and temporal version metadata, uses
boundary-aware chunks at 192 and 576 characters, namespace-local BM25, exact cosine over the
same provider, reciprocal-rank fusion, deduplication/MMR, neighbor expansion, and the same
context budget.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m benchmarks.run_phase_b --events 100 --seed 7 --budget-tokens 512 --json
python -m benchmarks.run_phase_b --events 1000 --seed 7 --budget-tokens 512 `
  --semantic-scan-cap 4096 --json
```

Both databases are cleanly closed, symmetrically SQLite `VACUUM`-compacted, closed again,
and then measured. Compaction time is excluded from ingestion. No persistent table is omitted;
EviCortex's extraction runs, derivations, diagnostics, and vectors remain in its byte count.

| Setting | 100-event smoke | 1,000-event scale point |
| --- | ---: | ---: |
| Queries | 12 | 12 |
| Unanswerable probes | 3 | 3 |
| Source bytes | 21,798 | 221,586 |
| Dataset SHA-256 | `98898858c5d4621f44fa6a01b37f1d50955f687c5897f13755b450ba2e9a2b6e` | `109c196d585532bf6ee7f9f1f96d8b33a1d8cd57f0f1c17ec042b37fc6b59f7a` |
| Exact scan cap | 2,048 | 4,096 |
| Embedding dimensions | 256 | 256 |
| Reader model | None | None |
| Runtime | Python 3.12.0 / SQLite 3.42.0 | Python 3.12.0 / SQLite 3.42.0 |

The final runtime-source digest is
`8f5a4cc0f922eedba3a6a7d1f5be757925684fb085ea7e46fbc9a64b7c0ea4f7` over the
18 runtime files declared by the report. The worktree was dirty, so the digest—not Git HEAD
alone—is the exact code identity.

## Results

### 100 events, seed 7

| Metric | Hybrid diagnostic | EviCortex | Candidate / hybrid |
| --- | ---: | ---: | ---: |
| Evidence Recall@Budget | 100% | 100% | equal |
| Supported-answer rate | 100% | 100% | equal |
| Current / historical accuracy | 100% / 100% | 100% / 100% | equal |
| Stale leakage | 0% | 0% | equal |
| Retrieval-level no-evidence abstention | 33.3% | 33.3% | equal, inadequate |
| Namespace isolation / budget compliance | 100% / 100% | 100% / 100% | equal |
| Query errors | 0 | 0 | equal |
| Estimated context tokens, total | 3,388 | 2,216 | 0.654x |
| Retrieval p50 / p95 / p99 | 144.01 / 166.54 / 173.32 ms | 94.50 / 127.31 / 132.12 ms | 0.656x / 0.764x / 0.762x |
| Ingestion | 1,146.42 ms | 2,297.90 ms | 2.004x |
| Compacted database | 778,240 B | 831,488 B | 1.068x |
| Ingestion projection texts | 293 | 200 | 0.683x |

Three 100-event streams (seeds 7, 11, and 19) retained equal supported quality and the same
1/3 abstention result. Their arithmetic mean candidate/hybrid ratios were 0.654 for context,
0.645 for p50, 0.710 for p95, 0.699 for p99, 2.000 for ingestion, and 1.068 for compacted
storage. These are three small,
template-sharing streams and have no inferential confidence interval.

### 1,000 events, seed 7

| Metric | Hybrid diagnostic | EviCortex | Candidate / hybrid |
| --- | ---: | ---: | ---: |
| Evidence recall / supported-answer rate | 100% / 100% | 100% / 100% | equal |
| Current / historical accuracy | 100% / 100% | 100% / 100% | equal |
| Stale leakage | 0% | 0% | equal |
| Retrieval-level no-evidence abstention | 33.3% | 33.3% | equal, inadequate |
| Estimated context tokens, total | 3,388 | 2,216 | 0.654x |
| Retrieval p50 / p95 / p99 | 1,625.83 / 1,669.50 / 1,676.48 ms | 1,001.22 / 1,129.43 / 1,175.15 ms | 0.616x / 0.677x / 0.701x |
| Ingestion | 12,120.37 ms | 24,845.42 ms | 2.050x |
| Compacted database | 7,147,520 B | 6,094,848 B | 0.853x |
| Persisted projection rows/texts | 2,993 chunks | 1,000 memories + 1,000 claims | 0.668x |
| Query errors / scan truncations | 0 / 0 | 0 / 0 | equal |

The 4,096 cap is symmetric but bounds different representations. The hybrid had 2,993 chunk
vectors and scored at most 2,989 eligible candidates for one query; EviCortex had 1,000 memory
and 1,000 claim vectors in separate exact channels. This helps explain the candidate's lower
retrieval latency and storage at 1,000 events. The explicit `not an accessory` constraint also
removed 988 memory and 988 claim distractors before fusion, making the result independent of
generated-ID tie order. This is not evidence about ANN behavior at scale.

## Extraction and provenance

At 1,000 events, `remember_extracted` invoked the canonical-grammar extractor exactly 1,000
times and durably stored 1,000 extraction runs, 1,000 claim-derivation rows, 1,000 generated
claim-ID mappings, and zero diagnostics or abstentions. Every run's provider identity, source
digest, exact span, canonical text, and resulting claim ID was read back and checked. The
extractor's p95 was 0.103 ms because it parses an explicit synthetic grammar; this is not an
estimate for LLM extraction from natural language.

## The useful failure: partial-OOD abstention

The original zero-overlap probe combined an unknown entity with an unseen predicate; both
systems returned no evidence. Two harder probes revealed the limitation:

- known person + unseen `xylophonic_allergy`: both systems returned unrelated records about
  that person;
- unknown `Zephyr-999` + seen `home_city`: both systems returned records matching the known
  relation but belonging to other people.

Accordingly, `abstention_accuracy` is precisely defined as retrieval-level no-evidence
abstention and is 1/3. It does not measure whether a downstream reader could refuse to answer.
Relation-aware query planning, calibrated support verification, and reader-level evaluation
are required before claiming robust abstention.

The current query parser also treats phrases such as “not an accessory” as literal exclusion
syntax across every candidate channel. This prevents excluded-candidate floods, but it will
also reject a truthful source that itself says “the device is not an accessory.” A future
constraint parser or negation-aware reranker must replace this lexical approximation; a real
semantic provider is not, by itself, permission to ignore an explicit query constraint.

## What this establishes—and what it does not

This checkpoint establishes that durable extraction, semantic projection lifecycle,
namespace/time filtering, fusion, budgeting, provenance audit, failure diagnostics, and the
metadata-aware comparator run end to end without errors at 1,000 events. Within this exact
hashing diagnostic, EviCortex used fewer prompt tokens and projection rows, traded roughly 2x
ingestion time for lower retrieval latency, and crossed from slightly higher storage at 100
events to lower storage at 1,000 events.

It does not establish semantic recall, reader answer quality, statistical significance,
10k/100k scaling, ANN recall, extraction accuracy on ordinary text, or superiority over dense,
hybrid, graph, agentic, or production RAG. Those remain explicit gates.
