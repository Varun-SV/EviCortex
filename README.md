# EviCortex

[Website](https://varun-sv.github.io/EviCortex/) ·
[Documentation](docs/index.md) ·
[![CI](https://github.com/Varun-SV/EviCortex/actions/workflows/ci.yml/badge.svg)](https://github.com/Varun-SV/EviCortex/actions/workflows/ci.yml) ·
[![License: MIT](https://img.shields.io/badge/License-MIT-57d9a3.svg)](LICENSE)

Evidence-preserving, temporal memory infrastructure for long-lived AI systems.

> **Project status:** pre-alpha research software. The kernel is executable and tested,
> but the central comparative hypothesis is not yet proven on the official benchmark.

EviCortex is exploring a narrower, testable claim than “memory that replaces RAG”:

> Can a lossless, temporal, associative memory retrieve the right evidence with fewer
> context tokens than fixed-chunk RAG as information changes and grows?

The project is inspired by biological *principles*—durable records, multiple memory
systems, consolidation, associative recall, and temporary working representations—not
by a claim that human memories are literally stored in DNA.

## Proposed architecture

EviCortex separates three concerns that conventional RAG pipelines often couple:

1. **Evidence ledger (“Genome”)** — append-only, digest-verified source material with
   provenance and both event time and ingestion time. This is the lossless source of truth.
2. **Memory indexes (“Engram”)** — rebuildable episodic, semantic, temporal, lexical,
   and graph views. Corrections create history-aware revisions instead of destroying the
   evidence that supported an older belief.
3. **Context compiler (“Transcript”)** — query-specific retrieval, conflict resolution,
   and evidence packing under a strict estimated-token budget. Returned items retain stable IDs so
   an agent can expand only the details it needs.

This is not yet proven to outperform a strong RAG system. The repository now contains a
deterministic kernel, durable extraction/embedding projections, two controlled comparators,
an offline pinned dense-encoder adapter, and a strict LongMemEval retrieval runner. The real
encoder has passed a tiny local semantic smoke, but the official external dataset, shared
reader, calibrated abstention, and ANN scale gates remain open.

## Initial scope

- Local-first Python library with SQLite/FTS5 and no required cloud service.
- Immutable observations plus valid-time, versioned claims and separate recording timestamps.
- Namespace isolation for users, agents, and applications.
- Hybrid lexical, exact-semantic, entity, graph, and temporal retrieval with a complete
  score trace.
- Provider-neutral embedding and claim-extraction contracts with durable span provenance.
- Bounded, citation-bearing memory capsules for LLM prompts.
- Reproducible fixed-chunk and metadata-aware hybrid diagnostic comparisons.

Out of scope for the first prototype: training a new foundation model, storing memories
inside model weights, claiming biological fidelity, or promising perfect recall. Durable
storage can be lossless; retrieval is probabilistic and must be measured.

Prototype A is an embedded research library, not a tenant-security boundary. A namespace is
a required retrieval scope supplied by the caller; the host application must authenticate
the caller and authorize that scope. The local SQLite file is not encrypted, and hard purge
removes logical records but does not promise forensic erasure from backups or storage media.

## Quickstart

EviCortex requires Python 3.12 and has no runtime dependencies outside the standard library.
From a source checkout:

```powershell
python -m pip install -e .
python examples/quickstart.py
python examples/phase_b_diagnostic.py
```

The public API stores exact evidence plus optional structured claims, then returns a compact
prompt payload under a hard estimated-token budget:

```python
from evicortex import ClaimInput, EviCortex

with EviCortex.open("memory.db") as memory:
    record = memory.remember(
        namespace="assistant/user-1",
        content="Ada prefers the dark interface theme.",
        source="conversation/42",
        occurred_at="2026-04-18T09:30:00Z",
        claims=[ClaimInput("user:ada", "prefers_theme", "dark")],
    )
    capsule = memory.recall(
        namespace="assistant/user-1",
        query="Which interface theme does Ada prefer?",
        budget_tokens=300,
    )
    print(capsule.to_prompt())
    print(memory.expand(namespace="assistant/user-1", memory_id=record.id).content)
```

## Project documents

- [Documentation home](docs/index.md)
- [Getting started](docs/getting-started.md)
- [Core concepts](docs/concepts.md)
- [API guide](docs/api-guide.md)
- [Benchmarking guide](docs/benchmarking.md)
- [Frequently asked questions](docs/faq.md)
- [Architecture and invariants](docs/architecture.md)
- [Benchmark contract](docs/benchmark-contract.md)
- [Initial technology decision](docs/decisions/0001-local-first-python-sqlite.md)
- [Prototype A benchmark result](docs/results/prototype-a.md)
- [Prototype B semantic/extraction status and plan](docs/phase-b.md)
- [Prototype B diagnostic benchmark result](docs/results/prototype-b-diagnostic.md)
- [Phase C pinned semantic evaluation](docs/phase-c.md)
- [Phase C semantic smoke result](docs/results/prototype-c-semantic-smoke.md)

Contributions are welcome within the current research scope. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, report security issues as
described in [SECURITY.md](SECURITY.md), and see [LICENSE](LICENSE) for the MIT terms.

## Status

Prototype A, the diagnostic slice of Prototype B, and the pinned semantic Phase C path are
executable. In the latest verified
three-seed, 1,000-event-per-seed Prototype A sweep, EviCortex retrieved every required source
while using 40.6% fewer estimated context tokens than the mechanical lexical baseline. It
also used about 50.2% more compacted database space and had slower p50/p95 retrieval.

At 1,000 events, the Phase B hashing diagnostic gave EviCortex and the metadata-aware hybrid
the same supported-answer, temporal, and namespace scores. EviCortex used 34.6% fewer prompt
tokens, 14.7% fewer compacted database bytes, and lower retrieval p95 latency, but ingestion
was 2.05x slower. Both systems achieved only 1/3 retrieval-level abstention once partial-OOD
queries were added. These measurements diagnose the representation and implementation; they
are **not** evidence that EviCortex beats dense or production RAG.

The first real-encoder smoke used a pinned cached BGE model on three synthetic
LongMemEval-schema questions. Both systems retrieved all answerable gold sessions at rank 1,
both failed the single retrieval-level abstention, and EviCortex packed 199.0 estimated tokens
on average versus 211.7 for the hybrid comparator. This verifies wiring only: it is not the
official LongMemEval dataset, is far too small for a quality claim, and has no reader model.

EviCortex is now the project and Python namespace. This engineering rename is not formal
trademark clearance; that remains a separate requirement before a commercial release.

EviCortex stores both occurrence and recording time but does not yet expose a transaction-time
`known_at` query. It should therefore not be described as fully bitemporal yet.

## Research anchors

- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [HippoRAG 2](https://arxiv.org/abs/2502.14802)
- [Zep temporal knowledge graph](https://arxiv.org/abs/2501.13956)
- [MemMachine](https://arxiv.org/abs/2604.04853)
- [H-Mem](https://arxiv.org/abs/2605.15701)
- [MemoryAgentBench](https://openreview.net/forum?id=DT7JyQC3MR)
- [Engram lifecycle review](https://pubmed.ncbi.nlm.nih.gov/37369721/)
