---
layout: docs
title: "Prototype A benchmark result"
---

# Prototype A benchmark result

Status: last verified local result, 2026-07-21. This is an engineering checkpoint, not a
publication-quality comparison or a general claim that EviCortex is better than RAG.

## Configuration

```powershell
$env:PYTHONPATH = "src"
python -m benchmarks.run --events 1000 --seed 7 --budget-tokens 512 --json
```

| Setting | Value |
| --- | --- |
| Events | 1,000 |
| Queries | 12: 9 answerable and 3 retrieval-abstention probes |
| Source bytes | 221,586 |
| Suite SHA-256 | `109c196d585532bf6ee7f9f1f96d8b33a1d8cd57f0f1c17ec042b37fc6b59f7a` |
| Runtime-source SHA-256 | `8f5a4cc0f922eedba3a6a7d1f5be757925684fb085ea7e46fbc9a64b7c0ea4f7` |
| Candidate limit | 40 |
| Baseline chunks | 320 characters, 48-character overlap |
| Token estimate | UTF-8 bytes / 3 |
| Reader model | None; this measures evidence support, not generated-answer accuracy |
| Runtime | Python 3.12.0, SQLite 3.42.0, local Windows machine |

The worktree was dirty, so the report records both Git HEAD and a deterministic path+content
digest of all runtime source. Both databases are cleanly closed, symmetrically `VACUUM`-
compacted, closed again, and then measured; compaction time is excluded from ingestion.

Both systems received the same canonical events, timestamps, namespace restrictions, query
stream, and serialized-context budget. Times are single-machine measurements with only 12
query probes and should be treated as directional.

## Systems compared

The baseline is **fixed-size chunks ranked by SQLite FTS5** and packed to the same budget. It
is a deliberately mechanical lexical baseline—not dense retrieval, hybrid RAG, GraphRAG, or
a tuned production stack.

The candidate is the deterministic EviCortex kernel: immutable evidence, explicit temporal
claims, lexical/entity/graph/time retrieval, stale-claim filtering, and bounded capsule
compilation. Neither system used an LLM. The generator supplies EviCortex with oracle
structured claims while placing the same canonical facts verbatim in baseline text. This
isolates representation/retrieval and excludes real extraction errors and cost.

## Single-seed result

| Metric | Fixed-chunk FTS5 | EviCortex | Interpretation |
| --- | ---: | ---: | --- |
| Evidence Recall@Budget | 88.89% | 100% | Candidate found every required source ID |
| Supported-answer rate | 88.89% | 100% | +11.11 percentage points |
| Current-fact accuracy | 50% | 100% | Update handling improved |
| Historical as-of accuracy | 100% | 100% | Equal |
| Stale-fact leakage | 100% | 0% | Superseded evidence excluded from current recall |
| Retrieval-level no-evidence abstention | 33.33% | 33.33% | Both fail the two partial-OOD probes |
| Namespace isolation / budget compliance | 100% / 100% | 100% / 100% | Equal |
| Estimated context tokens, total | 3,701 | 2,216 | Candidate used 40.1% fewer |
| Retrieval latency p50 | 0.532 ms | 3.725 ms | 7.01x; only 12 probes |
| Retrieval latency p95 | 2.865 ms | 4.614 ms | 1.61x; only 12 probes |
| Retrieval latency p99 | 4.656 ms | 4.834 ms | 1.04x; descriptive near-maximum only |
| Ingestion time | 9,617.0 ms | 11,643.5 ms | 1.21x baseline |
| Compacted database size | 1,265,664 B | 1,900,544 B | 1.502x baseline |
| Storage/source-byte ratio | 5.712x | 8.577x | Candidate stores more structure |
| Query errors | 0 | 0 | Equal |

The candidate met the quality/context and temporal gates. It did not establish robust
abstention, and it narrowly missed the optional 1.5x compacted-storage target by 0.16%.

## Three-seed scale check

```powershell
$env:PYTHONPATH = "src"
python -m benchmarks.sweep --seeds 7,11,19 --events 1000 --budget-tokens 512 `
  --bootstrap-samples 2000 --bootstrap-seed 20260721
```

| Paired aggregate | Mean | Descriptive 95% bootstrap interval |
| --- | ---: | ---: |
| Supported-answer delta | +11.11 percentage points | +11.11 to +11.11 pp |
| Evidence-recall delta | +11.11 percentage points | +11.11 to +11.11 pp |
| Retrieval-abstention delta | 0 | 0 to 0 |
| Context-token ratio | 0.594x | 0.586x to 0.599x |
| Retrieval p50 ratio | 6.99x | 6.89x to 7.06x |
| Retrieval p95 ratio | 1.538x | 1.458x to 1.610x |
| Retrieval p99 ratio | 0.960x | 0.880x to 1.038x |
| Ingestion-time ratio | 1.208x | 1.199x to 1.213x |
| Compacted database-size ratio | 1.502x | 1.502x to 1.502x |

All three streams share the same task templates and vary only generated values and
distractors. The intervals are unstable small-N descriptions, not evidence of statistical
significance. They do show the cost shape: the candidate uses about 40.6% fewer context tokens
and fixes the baseline's update failure, while paying materially higher median lookup,
ingestion, and storage costs.

## Gate status

| Gate | Status | Evidence or limitation |
| --- | --- | --- |
| Namespace filtering, digest verification, budget compliance, deterministic ordering, preserved history, and per-item traces | Pass | Last verified suite: 144 tests passed |
| ≥10% relative support improvement or ≥20% fewer tokens at equal support | Pass | Support rose to 100%; context fell 40.1% |
| No more than two-point direct-recall regression | Pass in this suite | Direct evidence remained supported |
| Lower stale leakage; no worse historical accuracy | Pass | 100%→0% stale leakage; historical stayed 100% |
| Candidate p95 no greater than 2x baseline | Pass, smoke-level only | 1.61x in the retained run |
| Additional database target no greater than 1.5x baseline | **Narrow miss** | 1.5016x after symmetric compaction |
| Multi-seed paired bootstrap report | Run, small-N only | Three template-sharing streams |
| Robust unsupported-query abstention | **Fail** | Both systems abstain only on the double-OOD case: 1/3 |
| Tuned dense-hybrid comparison | Not met | Phase B currently has only a hashing diagnostic |

## Interpretation and next evidence

Prototype A supports one narrow conclusion: on this deterministic workload, explicit temporal
structure and budget-aware compilation fixed a mechanical chunk baseline's update failure and
reduced prompt material. It does **not** show that the approach stores more information than
RAG, scales better, or beats a modern retriever.

The next decisive comparison needs a pinned dense encoder, equivalent metadata/version
history, a shared reranker and reader, exact tokenizer accounting, partial-OOD support
verification, and official external workloads. See the
[Phase B diagnostic result](prototype-b-diagnostic.md) for the implemented intermediate step.

## Biology and naming caveats

“Genome,” “Engram,” and “Transcript” are software metaphors. Human autobiographical memories
are not stored as ordinary records in DNA or RNA. EviCortex is the selected project identity,
but it has not undergone formal trademark clearance.
