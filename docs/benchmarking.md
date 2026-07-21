---
layout: docs
title: "Benchmarking and reproducibility"
---

# Benchmarking and reproducibility

EviCortex treats benchmark configuration as part of every result. A number without its
dataset, source digest, context budget, provider identity, comparator, hardware, and query
count is not a supported project claim.

The binding rules are in the [benchmark contract](benchmark-contract.md). This guide shows
how to run the implemented checks and how to interpret the retained artifacts.

## Validation levels

| Level | Purpose | What it can establish |
| --- | --- | --- |
| Unit and integration tests | Detect implementation regressions | The tested invariants pass in the current checkout |
| Bundled examples | Human-readable API smoke | The local embedded flow executes |
| Prototype A synthetic suite | Compare deterministic temporal representation with mechanical lexical chunks | Results only for the named synthetic workload |
| Prototype B diagnostic | Exercise extraction and exact-vector lifecycle with feature hashing | Wiring and cost shape, not semantic quality |
| Phase C semantic smoke | Exercise a pinned real encoder on three synthetic LongMemEval-schema cases | Offline semantic integration, not official benchmark quality |
| Official external evaluation | Planned strong comparison with shared reader and controls | Not yet completed |

## Prepare the checkout

Use Python 3.12 and install the package in an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Record the commit and worktree state before retaining a result:

```powershell
git rev-parse HEAD
git status --short
python --version
```

The benchmark reports also calculate a deterministic path-and-content digest over declared
runtime files. When the worktree is dirty, that digest—not Git HEAD alone—is the executable
code identity.

## Run tests and examples

```powershell
python -m unittest discover -s tests -v
python examples/quickstart.py
python examples/phase_b_diagnostic.py
```

The optional cached-model integration test skips when its local model requirements are not
available. A skip is not proof that dense integration works on that machine; use the Phase C
smoke when validating that path.

If you do not install the package, set `PYTHONPATH=src` before the commands instead:

```powershell
$env:PYTHONPATH = "src"
```

## Reproduce Prototype A

Single retained configuration:

```powershell
python -m benchmarks.run --events 1000 --seed 7 --budget-tokens 512 --json
```

Three-seed scale check:

```powershell
python -m benchmarks.sweep --seeds 7,11,19 --events 1000 --budget-tokens 512 `
  --bootstrap-samples 2000 --bootstrap-seed 20260721
```

This compares EviCortex with a fixed-size SQLite FTS5 chunk baseline. It isolates the
deterministic representation and retrieval path; it is not a comparison with dense, hybrid,
GraphRAG, agentic, or production RAG.

See [the retained Prototype A result](results/prototype-a.md) for configuration, metrics,
gates, and limitations.

## Reproduce the Prototype B diagnostic

```powershell
python -m benchmarks.run_phase_b --events 100 --seed 7 --budget-tokens 512 --json
python -m benchmarks.run_phase_b --events 1000 --seed 7 --budget-tokens 512 `
  --semantic-scan-cap 4096 --json
```

Both systems use the same deterministic `HashingEmbedder`. The comparator combines BM25 and
exact cosine over multi-granularity chunks, while EviCortex combines its evidence and claim
projections. Feature hashing does not understand language, so label this a “BM25 + hashing
diagnostic,” never a dense or strong-RAG result.

See [the Phase B plan](phase-b.md) and
[retained diagnostic result](results/prototype-b-diagnostic.md).

## Reproduce the Phase C semantic smoke

Install optional model dependencies:

```powershell
python -m pip install -e ".[local-transformers]"
```

The exact pinned model must already be present in the local Transformers cache because the
runner forces offline loading:

```powershell
python -m benchmarks.run_phase_c `
  --dataset benchmarks/fixtures/longmemeval_semantic_smoke.json `
  --model unsloth/bge-small-en-v1.5 `
  --revision 7382f1122c10708a1faa0bbe548674a14b1ffe7e `
  --pooling cls `
  --query-prefix="Represent this sentence for searching relevant passages: " `
  --document-prefix= `
  --max-length 128 `
  --budget-tokens 256 `
  --cutoffs 1,2
```

The fixture has three hand-written questions and no reader model. Passing it establishes
that pinned model loading, asymmetric encoding, persistent projections, temporal filtering,
ranking, and budget packing connect end to end. It cannot establish semantic superiority,
answer correctness, statistical significance, or LongMemEval performance.

See [Phase C](phase-c.md), the
[concise smoke interpretation](results/prototype-c-semantic-smoke.md), and the
[machine-readable record](results/prototype-c-semantic-smoke.json).

## Reading the metrics

Key retrieval-stage metrics include:

- **Evidence Recall@Budget:** fraction of gold evidence IDs present after prompt packing.
- **Evidence Precision@Budget:** fraction of returned evidence belonging to the gold support
  set.
- **Supported-answer rate:** questions whose packed context contains the required gold
  evidence; this is not generated-answer accuracy.
- **Current/historical accuracy:** whether the selected valid-time claim is correct.
- **Stale-fact leakage:** obsolete evidence presented as current.
- **Retrieval no-evidence abstention:** unsupported questions for which retrieval packs no
  evidence. This is not calibrated reader refusal.
- **Context tokens:** the built-in UTF-8-bytes/3 estimate unless an exact tokenizer is named.
- **Latency percentiles:** meaningful only with query count, warm/cold policy, hardware, and
  concurrency. On tiny suites, p99 is close to a descriptive maximum.
- **Lifecycle cost:** ingestion, maintenance, retrieval, reader use, and storage must be
  reported separately.

Metrics are measured after hard-budget packing. A system does not get credit for a candidate
that would not fit in the reader context.

## Fair-comparison checklist

Before comparing another system, hold constant:

- source events, ingestion order, namespaces, and query stream;
- train/development/test split and frozen hyperparameters;
- embedding model and exact query/document preprocessing;
- reranking and query-expansion compute;
- reader model, full reader prompt, and maximum context budget;
- hardware, concurrency, and cache policy; and
- compaction, database-size, token-accounting, and latency methodology.

Gold annotations should point to original source evidence, not system-specific chunks or
graph nodes. Report no-memory, full-history-when-it-fits, dense-only, and oracle-evidence
controls where applicable; oracle evidence separates retrieval failures from reader failures.

## Language for results

Appropriate:

> On the named dataset and configuration, system A changed Evidence Recall@Budget by X under
> an N-token budget.

Not supported without broader evidence:

- “EviCortex is better or more efficient than RAG.”
- “EviCortex never forgets.”
- “The system has human-like memory.”
- “Retrieval recall proves answer accuracy.”

Current retained results identify useful successes and failures, especially temporal update
handling, higher ingestion cost in the diagnostic path, and inadequate retrieval-level
abstention. Preserve those tradeoffs when summarizing the project.

## Before retaining a new result

1. Confirm the worktree scope and record Git HEAD plus dirty state.
2. Save the complete command, configuration, seed, dependency versions, platform, and data
   digest.
3. Use the same evidence and hard prompt budget for every compared system.
4. Close and symmetrically compact SQLite databases before byte measurement when following
   the existing policy; exclude and label compaction time consistently.
5. Report per-workload results and paired uncertainty over independent streams or
   conversations, not correlated questions treated as independent samples.
6. Retain machine-readable output alongside a concise interpretation with explicit claim
   boundaries.
