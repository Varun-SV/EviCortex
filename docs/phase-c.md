---
layout: docs
title: "Phase C: pinned semantic retrieval and external evaluation"
---

# Phase C: pinned semantic retrieval and external evaluation

Status: the offline dense-embedding path, core reranker seam, strict LongMemEval adapter,
and reader-free comparison runner are implemented. A cached-model semantic smoke passes;
the official LongMemEval-S artifact, shared reader, cross-encoder adapter, and ANN scale gates
have not yet been run.

## Purpose

Phase C tests whether the evidence-ledger/context-compiler design still helps when EviCortex
and a metadata-aware hybrid chunk comparator use the same real semantic encoder. It is an
evaluation phase, not a claim that EviCortex has replaced retrieval-augmented generation.

The comparison unit is original evidence. Both systems receive the same conversation turns,
timestamps, namespace, encoder, query, candidate limits, and hard prompt budget. Metrics are
computed after prompt packing so a system does not receive credit for evidence that would not
actually fit in the reader context.

## Implemented checkpoint

- Role-aware document/query embedding dispatch with backward-compatible symmetric providers.
- Offline `TransformersEmbedder` with explicit pooling and prefixes, finite capacity checks,
  bounded overlength detection, and no silent truncation by default.
- Safetensors-only model loading with pinned 40-character model commits. Custom repository
  code additionally requires its own 40-character `code_revision`.
- Local filesystem models require a verified content digest over every file; mutable paths and
  links escaping the artifact root fail closed.
- Durable embedding identity includes preprocessing, model/tokenizer commits, float32 precision,
  implementation versions/classes, capacity, and normalization behavior.
- An opt-in reranker seam over at most 16 filtered, temporally valid, deduplicated candidates.
  Raw provider scores are recorded but used only to derive ordinal RRF contributions.
- Strict LongMemEval timestamps, gold/session-label alignment, abstention invariants, official
  cleaned-artifact SHA-256 recognition, and exact released `recall_all`/`ndcg_any` behavior.
- An offline JSON/JSON.GZ runner comparing EviCortex with the metadata-aware hybrid comparator,
  using label-free turn evidence and separate retrieval-only abstention reporting.

## Reproduction

The bundled smoke fixture is deliberately synthetic and is never labelled as the official
LongMemEval dataset:

```powershell
$env:PYTHONPATH = "src"
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

For a local filesystem model, compute and pin
`local_transformers_artifact_sha256(path)` and pass the result as
`--local-artifact-sha256`. For a provider that executes custom code, pass both
`--trust-remote-code` and an audited `--code-revision`; a model commit alone is insufficient
when `auto_map` points at another repository.

The machine-readable smoke report is
[prototype-c-semantic-smoke.json](results/prototype-c-semantic-smoke.json), with a concise
[result interpretation](results/prototype-c-semantic-smoke.md).

## Remaining evidence gates

- Download and byte-verify the official 277 MB cleaned LongMemEval-S artifact, then run a
  preregistered development/test configuration rather than tuning on the reported cases.
- Batch semantic ingestion; the current per-turn path is correct but inefficient for the full
  500-question corpus on CPU.
- Add a pinned cross-encoder adapter and compare no-reranker/top-16 reranker conditions.
- Add a shared reader and exact reader-tokenizer accounting. Retrieval recall is not answer
  correctness.
- Add no-memory, full-history, oracle-evidence, dense-only, and official strong-retriever controls.
- Calibrate unsupported-query refusal on a development split. Current score thresholds are not
  calibrated probabilities, and both smoke systems retrieve distractors for the abstention case.
- Introduce an audited ANN candidate layer and measure its recall against exact cosine before the
  10k/100k scale claims.

Until those gates pass, the only allowed claim is that the pinned local semantic plumbing and
reader-free benchmark path execute end to end on the named fixture and configuration.
