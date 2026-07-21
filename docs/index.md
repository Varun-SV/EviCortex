---
layout: docs
title: "Documentation"
---

# EviCortex documentation

EviCortex is a pre-alpha, local-first Python research kernel for evidence-preserving,
temporal memory in LLM applications. It stores exact source observations, derives
rebuildable views, and compiles traceable evidence into a hard estimated-token context budget.

The project is testing a research hypothesis; it has not established that it is better
than a tuned dense, hybrid, graph, agentic, or production RAG system. Start with the
[getting-started guide](getting-started.md) to run the implemented kernel.

## Choose a path

| Goal | Start here |
| --- | --- |
| Evaluate the OpenAI Build Week Developer Tools submission | [Build Week sprint](build-week.md) |
| Run the shortest deterministic demo and install the Codex plugin | [Judge guide](judge-guide.md) |
| Install EviCortex and run a temporal-memory example | [Getting started](getting-started.md) |
| Understand evidence, claims, time, projections, and capsules | [Core concepts](concepts.md) |
| Look up public classes, methods, and provider options | [API guide](api-guide.md) |
| Reproduce tests and research checkpoints | [Benchmarking and reproducibility](benchmarking.md) |
| Check current limitations, security boundaries, and common questions | [FAQ and limitations](faq.md) |

## Recommended learning path

1. Follow [Getting started](getting-started.md) and inspect the returned
   `MemoryCapsule`.
2. Read [Core concepts](concepts.md), especially namespaces, temporal revisions, and
   the difference between exact evidence and derived claims.
3. Use the [API guide](api-guide.md) when integrating the embedded library.
4. Read [FAQ and limitations](faq.md) before evaluating EviCortex for real data or a
   multi-user application.
5. Use [Benchmarking and reproducibility](benchmarking.md) before interpreting or
   publishing performance results.

## Build Week sprint

EviCortex is being prepared for the OpenAI Build Week **Developer Tools** track. The
submission combines the deterministic local memory kernel with a Codex plugin/skill: Codex
chooses when to remember, recall, expand cited evidence, or check integrity, while the core
stores and retrieves data without requiring a model or cloud service.

- [Sprint scope, architecture, attribution, and disclosures](build-week.md)
- [Five-minute judge guide](judge-guide.md)
- [Paste-ready Devpost copy and sub-three-minute video script](build-week-submission.md)
- [Production-readiness status and closure gates](production-readiness.md)
- [Supported pre-alpha operations and recovery envelope](operations.md)
- [Release and compatibility policy](release-policy.md)
- [Changelog](../CHANGELOG.md)

The submission path is demonstrable, but the project remains pre-alpha research software
and is not yet production-ready.

## Design and research record

These documents describe the deeper design, evaluation contract, and retained results:

- [Architecture and invariants](architecture.md)
- [Benchmark contract](benchmark-contract.md)
- [ADR 0001: local Python and SQLite kernel](decisions/0001-local-first-python-sqlite.md)
- [Prototype A result](results/prototype-a.md)
- [Prototype B semantic and extraction plan](phase-b.md)
- [Prototype B diagnostic result](results/prototype-b-diagnostic.md)
- [Phase C pinned semantic evaluation](phase-c.md)
- [Phase C semantic smoke result](results/prototype-c-semantic-smoke.md)
- [Phase C machine-readable smoke artifact](results/prototype-c-semantic-smoke.json)

The result files are engineering checkpoints with named configurations and explicit
claim boundaries. The synthetic suites are useful for regression testing, but they are
not substitutes for official external benchmarks, a shared reader model, calibrated
abstention, or statistically powered comparisons.

## Current implementation boundary

Implemented today:

- a Python 3.12 embedded library backed by SQLite and FTS5;
- immutable evidence records with provenance, occurrence time, and recording time;
- explicit versioned claims, current and historical valid-time recall, and conflict
  surfacing;
- lexical, entity, graph, temporal, and optional exact-semantic retrieval;
- deterministic context packing, stable evidence IDs, and retrieval traces;
- provider-neutral extraction, embedding, and reranking contracts;
- deterministic diagnostic providers and an optional pinned local Transformers adapter;
- integrity checks, deterministic index rebuilds, semantic repair, and scoped purge; and
- synthetic benchmark runners plus a strict LongMemEval-schema retrieval runner.

Not yet established or production-ready:

- authentication or authorization, encryption, a network service, or distributed storage;
- natural-language claim extraction supplied by EviCortex;
- approximate-nearest-neighbor retrieval at large scale;
- calibrated unsupported-query refusal or reader answer accuracy;
- full transaction-time (`known_at`) queries; or
- superiority over a strong RAG or memory system.

See [FAQ and limitations](faq.md) for the operational consequences of these boundaries.
