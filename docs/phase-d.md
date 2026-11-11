---
layout: docs
title: "Phase D: answer-level comparison against other memory systems"
---

# Phase D: answer-level comparison against other memory systems

Status: implemented and run once. Results are in
[the Phase D result page](results/phase-d.md). This page is the protocol.

## Purpose

Earlier phases measured which source records each system packed into a context. They did not
measure whether a model could answer from that context, and they compared EviCortex only with
baselines written in this repository. Phase D closes both gaps on public data:

- every system's packed context is read by the same model with the same prompt;
- every answer is graded by the same judge, which never sees which system produced it;
- the comparison includes a real external memory system (Mem0) and the two controls that
  bound the problem: no memory and the full conversation history.

## What is held constant

| Item | Value |
| --- | --- |
| Source turns | Identical rendered turns, `[session date] speaker: text`, in the same order |
| Evidence unit | One conversation turn. Gold labels are never ingested |
| Encoder | `unsloth/bge-small-en-v1.5` at commit `7382f112…`, CLS pooling, BGE query prefix, 256-token truncation. Mem0 uses the same weights through its own `BAAI/bge-small-en-v1.5` loader |
| Evidence budget | Hard cap on `estimate_tokens(context)` (UTF-8 bytes / 3), checked for every row |
| Reader | One model and one prompt for all systems |
| Judge | One model, blinded to system and budget, items from all systems shuffled together |
| Hardware | One 4-vCPU container, CPU only |

Each system packs its own native prompt format. EviCortex emits its JSON capsule, the chunk
baselines emit their JSON evidence lists, and Mem0 emits a dated list of memories. Format
overhead counts against the budget, because that is what a reader would actually receive.

## Systems

| Name | What it is | LLM at ingestion |
| --- | --- | --- |
| `evicortex` | EviCortex through `remember` and `recall` only. No claims are supplied, so the capsule holds exact-evidence episodes | No |
| `evicortex_claims` | The same store, plus temporal claims proposed by an LLM once per session and written with explicit supersession ([`claims.py`](https://github.com/Varun-SV/EviCortex/blob/main/benchmarks/competitive/claims.py)) | Yes |
| `mem0` | The open-source `mem0ai` package, unmodified, with embedded Qdrant. Its LLM calls go through the same provider-neutral client | Yes |
| `hybrid_rag` | The repository's BM25 + dense comparator with rank fusion, MMR, and neighbor expansion | No |
| `dense_rag` | Cosine top-k over fixed 600-character chunks | No |
| `bm25_chunks` | SQLite FTS5 BM25 over fixed 600-character chunks | No |
| `no_memory` | The reader sees only the question | No |
| `full_history` | The reader sees every turn. Ignores the budget | No |
| `oracle` | The reader sees only the gold evidence. Ignores the budget | No |

Not run: Graphiti/Zep, HippoRAG 2, MemMachine, and hosted memory products. Graphiti and
HippoRAG 2 need several LLM calls per turn plus a graph database or a large embedding model,
which was outside this run's compute. Published scores for those systems use different
readers, judges, and prompts and are **not** comparable with the numbers here, so none are
quoted.

## Datasets

- **LoCoMo-10** (`snap-research/locomo`, CC BY-NC 4.0): 10 two-person conversations,
  5,882 turns, 1,986 questions with turn-level evidence labels. Category 5 ("adversarial")
  questions are treated as unanswerable: the correct behaviour is to decline.
- **LongMemEval-S, cleaned release** (MIT): 500 questions, each with its own haystack of
  roughly 50 sessions. Each question is a separate namespace, and retrieval may only see
  turns dated on or before the question date.

Neither dataset is redistributed here. The loaders verify the SHA-256 of the local file
against the pinned public artifact.

## Stages

```bash
pip install -e ".[local-transformers,benchmark]"

# 1. Ingest and pack contexts for every question, system, and budget (no LLM for most systems)
python -m benchmarks.competitive.retrieve --dataset locomo --data locomo10.json --out runs/locomo

# 2. Shared reader on a seeded stratified sample
python -m benchmarks.competitive.evaluate read  --run runs/locomo --reader claude-cli:claude-haiku-5-5

# 3. Blinded judge, then an independent audit of the judge
python -m benchmarks.competitive.evaluate judge --run runs/locomo --judge claude-cli:claude-sonnet-5-5
python -m benchmarks.competitive.evaluate audit --run runs/locomo --auditor claude-cli:claude-opus-5-5

# 4. Statistics, machine-readable summary, figures
python -m benchmarks.competitive.report --run runs/locomo --out docs/results/phase-d
```

Every LLM call is cached on disk by a digest of provider, prompts, and output limit, so an
interrupted run resumes without repeating paid calls.

## Use any LLM

The reader, judge, auditor, claim extractor, and Mem0's internal LLM are all selected by a
`provider:model` string, implemented with the standard library only:

| Spec | Provider |
| --- | --- |
| `claude-cli:<model>` | Claude Code CLI in headless mode; no API key needed when the CLI is signed in |
| `anthropic:<model>` | Anthropic Messages API (`ANTHROPIC_API_KEY`) |
| `openai:<model>` | OpenAI Chat Completions (`OPENAI_API_KEY`) |
| `openai-compatible:<base_url>#<model>` | Any Chat Completions server: Ollama, vLLM, LM Studio, llama.cpp, Groq, Together, OpenRouter, and others |

For example, `--reader openai-compatible:http://localhost:11434/v1#llama3.1` reruns the
answer stage with a local model. Results depend on the reader, so a result is only
comparable with another that used the same reader and judge.

## Metrics

- **Answer accuracy:** share of answers the judge marks correct. An exact "I don't know." is
  graded by rule: wrong when an answer exists, right when the question is unanswerable.
- **Answerable and unanswerable accuracy** are reported separately, because a system that
  retrieves nothing scores perfectly on unanswerable questions.
- **Evidence recall:** share of gold turns (LoCoMo) or gold sessions (LongMemEval) present in
  the packed context. Mem0's memories map back to the two-turn batch they were extracted
  from, so its turn recall is an upper bound and is flagged as such.
- **Context tokens, reader input tokens, ingestion LLM calls/tokens/cost, ingestion wall
  time, and retrieval latency.** Latency is descriptive only: the machine was shared between
  concurrent jobs.

Uncertainty is a percentile bootstrap (2,000 resamples) over independent units: conversations
for LoCoMo, questions for LongMemEval. Differences against EviCortex are paired on the same
resampled units.

## Known limits of this protocol

- **LLM judge.** The benchmark contract asks for a judge calibrated on human-reviewed
  answers. This run substitutes agreement with a second, stronger model on a seeded sample.
  That detects an inconsistent judge; it does not detect a bias both models share.
- **One reader.** Accuracy and the ranking of systems can change with a different reader.
- **Sampled answers.** Retrieval metrics cover every question; answers cover a stratified
  sample, with fixed per-category quotas, so the pooled accuracy is not the accuracy of the
  full question set.
- **LoCoMo has 10 conversations.** Intervals resample those 10 and are wide.
- **Mem0 dates.** Open-source Mem0's `add` takes no timestamp, so its extraction prompt
  resolves "today" to the wall-clock date of the run. The session date is present in every
  message it receives, but some relative dates still resolve wrongly. This is Mem0 as
  shipped; it is a disadvantage for Mem0 on temporal questions.
- **Token budgets are estimates** (UTF-8 bytes / 3), not reader-tokenizer counts. Reader
  input tokens are reported separately from the provider's own usage data.
- **Comparator speed.** The hybrid comparator's pure-Python cosine was replaced inside the
  adapter by a NumPy equivalent, with a test that rankings are unchanged. Its latency here
  is therefore not the latency of the comparator as written.
