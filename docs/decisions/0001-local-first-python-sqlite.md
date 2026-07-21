---
layout: docs
title: "ADR 0001: Start with a local Python and SQLite kernel"
---

# ADR 0001: Start with a local Python and SQLite kernel

- Status: accepted for Prototype A
- Date: 2026-07-21

## Context

The repository is greenfield. The first risk is architectural validity, not distributed scale.
Requiring a vector database, graph database, hosted model, or API key would make the earliest
results slower to reproduce and would mix infrastructure effects with the memory design.

## Decision

Build the deterministic kernel in Python 3.12 using the standard library and SQLite FTS5.
Keep storage, extraction, embedding, reranking, and token-counting behind narrow interfaces.

The import package is `evicortex`. The distribution remains `evicortex-research` until the
PyPI namespace and formal trademark checks are completed; no public package name is reserved.

## Consequences

Benefits:

- one-file local setup and reproducible tests;
- ACID transactions, FTS5, indexes, and a mature query planner;
- no network, model, or paid-service dependency for the baseline;
- easy inspection of invariants and benchmark artifacts.

Costs:

- no dense semantic retrieval in Prototype A;
- graph traversal and rank fusion must be implemented in application code;
- write concurrency and very large deployments will require another adapter;
- SQLite benchmark results cannot be presented as distributed production results.

## Exit criteria

Add or replace storage only after profiling identifies a concrete limit. A replacement must
pass the same conformance suite for evidence fidelity, temporal claims, isolation, expansion,
and deterministic retrieval traces.
