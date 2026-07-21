---
layout: docs
title: "Phase C pinned-dense semantic smoke"
---

# Phase C pinned-dense semantic smoke

Status: verified local integration checkpoint, 2026-07-21. This is a three-question synthetic
LongMemEval-schema fixture, not the official LongMemEval benchmark and not evidence of superiority
over production RAG.

## Configuration

- Fixture: 3 questions, 2 answerable, 1 abstention, 9 label-free conversation turns.
- Logical JSON SHA-256:
  `7e347a0438874099fa0ea08a72d8127478437bbe5cd3418eedfa25416dff5f88`.
- Encoder: `unsloth/bge-small-en-v1.5`, commit
  `7382f1122c10708a1faa0bbe548674a14b1ffe7e`, loaded offline from safetensors.
- Encoding: 384-dimensional normalized CLS vectors, float32, BGE query instruction, raw documents,
  128-token maximum, and `overflow_policy=error`.
- Runtime: Python 3.12.0, SQLite 3.42.0, Torch 2.10.0+cu130, Transformers 4.57.6,
  Tokenizers 0.22.2, CPU.
- Hard prompt budget: 256 estimated tokens; no reader model.
- Runtime-source SHA-256:
  `232996dcd1ba7d929cdd68c2edc1f3b39b691a86f285dfa641a4f1142fc3050f`
  over the 22 declared runtime files. The worktree was dirty, so this digest is the code identity.

The complete machine-readable record is
[prototype-c-semantic-smoke.json](prototype-c-semantic-smoke.json).

## Results

| Metric | EviCortex | Metadata-aware hybrid |
| --- | ---: | ---: |
| Answerable session `recall_all@1` / `@2` | 100% / 100% | 100% / 100% |
| Official-formula `ndcg_any@1` / `@2` | 100% / 100% | 100% / 100% |
| Hard-budget compliance | 100% | 100% |
| Retrieval-only no-evidence rate on 1 abstention | 0% | 0% |
| Mean / maximum estimated context tokens | 199.0 / 208 | 211.7 / 239 |
| Retrieval p50 / p95 | 36.13 / 43.46 ms | 38.32 / 47.23 ms |
| Ingestion time | 735.52 ms | 333.12 ms |
| Compacted database | 204,800 B | 110,592 B |
| Embedded texts | 12 | 21 |

The answerable cases merely establish that the real encoder, persistent projections, temporal
filtering, ranking, and hard-budget packing are connected correctly. The token difference is only
12.7 tokens on average across three hand-written cases; it is not statistically meaningful. The
latencies are a single cold-order exploratory pass, and the model's 18.58-second load time is
reported separately rather than hidden inside either system.

The useful failure is abstention: both systems returned unrelated evidence for the unsupported
weather question. Dense similarity did not solve support calibration. A development-calibrated
support gate and reader-level refusal evaluation remain mandatory.

## Claim boundary

This result establishes one reproducible, offline, safetensors-only semantic integration run. It
does not establish official LongMemEval quality, reader answer accuracy, tokenizer-exact context
cost, extraction accuracy, ANN scale, statistical significance, or a win over a tuned dense RAG
system.
