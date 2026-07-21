---
layout: docs
title: "Benchmark contract"
---

# EviCortex benchmark contract

Status: binding evaluation rules for Prototype A, the Phase B diagnostic, and the reader-free
Phase C semantic/external-evaluation path.

## 1. Claim under test

Under the same evidence, storage machine, reader model, and context budget, EviCortex should
provide a better evidence-quality/context-cost trade-off than a fixed-chunk RAG baseline on
streaming, changing, multi-session memory.

This is not a claim against every system called RAG. Graph RAG, agentic retrieval, temporal
knowledge graphs, and memory systems are separate comparison classes for later prototypes.

## 2. Fair-comparison rules

Both systems must use:

- identical source events and ingestion order;
- identical namespace filters;
- identical embedding model, if embeddings are enabled;
- identical downstream reader and answer prompt;
- identical maximum context-token budget;
- warm or cold cache conditions reported explicitly;
- the same hardware and concurrency;
- separate accounting for ingestion, maintenance, retrieval, and reader costs.

Hyperparameters are tuned on a development split and frozen before the test split. Every run
records Git HEAD, dirty state, a deterministic runtime-source digest, random seed, database
size, dependency versions, and configuration. Database bytes are measured only after both
systems close and receive the same SQLite `VACUUM`; compaction time is excluded and the report
labels this policy.

## 3. Prototype A systems

### Baseline: fixed-chunk lexical RAG

- Store the complete source stream.
- Split text into fixed-size, optionally overlapping chunks.
- Rank with the same SQLite FTS5 engine.
- Add top-ranked chunks until the common token budget is exhausted.

### Candidate: EviCortex kernel

- Store the same source stream in the evidence ledger.
- Index explicit claims, entities, temporal revisions, and episodes.
- Retrieve through lexical, entity, graph, and time channels.
- Resolve current/historical intent and compile a capsule under the same budget.

Prototype A compares data structures and deterministic retrieval. It does not use an LLM
during ingestion or retrieval, avoiding a hidden cost/quality advantage.

The fixed-chunk lexical system is a **mechanical baseline**, not sufficient evidence for a
public “better than RAG” claim. Prototype B's headline baseline must be a tuned hybrid RAG:

- BM25 plus the same dense embedding model used by EviCortex;
- multi-granularity child chunks and parent/session expansion;
- reciprocal-rank fusion, deduplication/MMR, and the same reranker budget;
- speaker, entity, source, and time metadata plus append-only version history;
- token-budgeted context construction; and
- equivalent query decomposition or expansion compute when EviCortex uses it.

Also report no-memory, full-history-when-it-fits, dense-only diagnostic, and oracle-evidence
reader results. Oracle evidence separates retrieval failure from reader failure.

## 4. Workloads

The synthetic streaming suite will generate evidence IDs and gold support sets for:

1. **Direct recall** — one event directly answers the query.
2. **Paraphrased recall** — terms differ while explicit entity/claim structure remains useful.
3. **Knowledge update** — a newer value supersedes an older value.
4. **Historical query** — the older value is correct for the requested time.
5. **Temporal ordering** — answer depends on event sequence or interval overlap.
6. **Cross-session multi-hop** — two or more evidence items must be connected.
7. **Contradiction/dispute** — incompatible evidence must be surfaced, not silently merged.
8. **Abstention** — the store lacks adequate support. The suite includes double-OOD,
   known-entity/unseen-predicate, and unknown-entity/seen-predicate probes.
9. **Distractor growth** — relevant evidence stays fixed while unrelated memories scale.
10. **Namespace isolation** — another user contains a tempting but forbidden answer.

The stream is queried between insertion rounds, not only after a one-time offline build.
Gold annotations refer to original source-span IDs, never system-internal chunks or memory
nodes, so different representations remain comparable.

## 5. Primary metrics

| Metric | Definition |
| --- | --- |
| Evidence Recall@Budget | Fraction of gold evidence IDs present within the context budget |
| Evidence Precision@Budget | Fraction of returned evidence that belongs to the gold support set |
| Supported-answer rate | Questions for which the capsule contains sufficient gold evidence |
| Current-fact accuracy | Current-value questions whose selected active claim is correct |
| Historical-fact accuracy | Time-scoped questions whose selected historical claim is correct |
| Stale-fact leakage | Current queries where an obsolete value is presented as current |
| Retrieval no-evidence abstention accuracy | Fraction of annotated unanswerable probes whose packed context contains zero evidence IDs; this does not measure reader refusal or calibrated uncertainty |
| Context tokens | Estimated and tokenizer-exact counts, when a model tokenizer is present |
| Retrieval latency | p50, p95, and p99 wall time after warmup |
| Write amplification | Derived rows and bytes written per source byte |
| Storage overhead | Database bytes divided by canonical source bytes |

Secondary metrics include MRR/nDCG, duplicate context rate, index rebuild time, capsule
formatting overhead, and downstream exact match/F1 when a reader model is enabled.

For external experiments, plot quality at 512, 1k, 2k, 4k, and 8k evidence-token budgets.
Report complete reader input tokens and all ingestion/retrieval model tokens separately.
Measure lifecycle cost at declared read/write ratios so query-time savings cannot conceal
expensive consolidation.

Always report the measured-query count beside latency percentiles. On tiny smoke suites, p99
is a descriptive near-maximum rather than a stable tail-latency estimate.

## 6. Prototype A acceptance gates

The prototype is considered valid only if all correctness gates pass:

- 100% namespace-isolation tests;
- 100% original-evidence digest verification after index rebuild;
- 100% capsule budget compliance;
- deterministic ordering for identical database state and query;
- no silent overwrite when a claim is superseded;
- an exact retrieval trace for every returned item.

The performance hypothesis passes only if the held-out suite shows a Pareto improvement:

- at least 10% relative improvement in supported-answer rate **or** at least 20% fewer
  context tokens at equal supported-answer rate;
- no more than two percentage points regression on direct recall;
- lower stale-fact leakage and no worse historical-fact accuracy;
- p95 retrieval latency no greater than 2x the baseline for the same corpus size.

These thresholds are project gates, not a publication-quality statistical claim. Results must
also include bootstrap confidence intervals and per-workload breakdowns so an aggregate score
cannot hide regressions.

The local Phase B comparator is implemented, but its bundled feature-hashing provider is
non-semantic. Its results may be called a **BM25 + hashing diagnostic** only. A strong-RAG gate
still requires a pinned dense model, exact query/document encoding rules, shared reranker and
reader budgets, tokenizer-exact accounting, and external data.

## 7. Scale points

Run deterministic local tests at increasing event counts (initially 1k, 10k, and 100k). Report
source bytes because “number of memories” can hide radically different payload sizes. Later,
run LongMemEval, LoCoMo, and MemoryAgentBench using their official splits and evaluation rules.
LongMemEval-V2 and action-grounding benchmarks belong in the agent/procedural-memory phase.

## 8. Ablations

Disable one mechanism at a time:

- temporal validity resolution;
- entity/relationship expansion;
- episodic neighbor expansion;
- rank fusion;
- diversity filtering;
- structured capsule packing;
- association reinforcement.

Two crossover ablations are required once retrieval components are interchangeable:

- EviCortex's representation with the baseline retriever/context composer; and
- baseline chunks with EviCortex's retriever/context composer.

These isolate whether gains come from representation, retrieval, or prompt construction.

An architectural feature is retained only when its measured benefit justifies its write,
storage, latency, and operational complexity.

## 9. Reporting language

Allowed: “On dataset X, configuration Y improved evidence recall by Z under an N-token budget.”

Not allowed: “EviCortex never forgets,” “human-like memory,” “more efficient than RAG,” or
“stores more than RAG” without the exact scope and measurements that make the statement true.

LLM judges, when unavoidable, must be blinded to system identity and calibrated on a
stratified human-reviewed sample. Confidence intervals must resample independent streams or
conversations rather than treating correlated questions as independent observations.
