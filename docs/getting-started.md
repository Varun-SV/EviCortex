---
layout: docs
title: "Getting started"
---

# Getting started

This guide installs the source checkout, stores two temporally related observations,
recalls the current and historical values, and expands the exact supporting evidence.

## Requirements

- Python 3.12 or later
- SQLite with FTS5 support (included in standard CPython builds on common platforms)
- Git, if cloning the repository

The deterministic kernel has no required third-party runtime dependencies. Dense local
Transformers support is optional and covered in the [API guide](api-guide.md#optional-dense-embeddings).

## Install from a source checkout

```powershell
git clone https://github.com/Varun-SV/EviCortex.git
Set-Location EviCortex
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The distribution is currently named `evicortex-research`; the Python import package is
`evicortex`. The project is pre-alpha, so pin a commit when another application depends on
it.

## Run the bundled example

```powershell
python examples/quickstart.py
```

The example uses an in-memory database, writes a light-theme preference followed by a
superseding dark-theme preference, and prints separate current and historical capsules.

## Build the same flow step by step

Create `try_evicortex.py`:

```python
from evicortex import ClaimInput, EviCortex


NAMESPACE = "assistant/user-ada"

with EviCortex.open("evicortex.db") as memory:
    earlier = memory.remember(
        namespace=NAMESPACE,
        content="Ada prefers the light interface theme.",
        source="conversation/session-1/turn-4",
        occurred_at="2026-01-02T10:00:00Z",
        claims=[
            ClaimInput(
                subject="user:ada",
                predicate="prefers_theme",
                value="light",
            )
        ],
    )

    latest = memory.remember(
        namespace=NAMESPACE,
        content="Ada switched to the dark interface theme.",
        source="conversation/session-7/turn-12",
        occurred_at="2026-04-18T09:30:00Z",
        claims=[
            ClaimInput(
                subject="user:ada",
                predicate="prefers_theme",
                value="dark",
                supersedes=earlier.claims[0].id,
            )
        ],
    )

    current = memory.recall(
        namespace=NAMESPACE,
        query="Which interface theme does Ada prefer?",
        budget_tokens=300,
    )
    historical = memory.recall(
        namespace=NAMESPACE,
        query="Which interface theme did Ada prefer?",
        as_of="2026-02-01T00:00:00Z",
        budget_tokens=300,
    )

    print("Current:", current.to_prompt())
    print("Historical:", historical.to_prompt())
    print("Current evidence IDs:", current.evidence_ids)

    exact = memory.expand(
        namespace=NAMESPACE,
        memory_id=latest.id,
    )
    print("Exact evidence:", exact.content)
```

Run it:

```powershell
python try_evicortex.py
```

The important behavior is:

- `remember()` stores the original text and the structured claim in one transaction.
- `supersedes` points to the earlier claim; it does not erase the earlier evidence.
- `recall()` without `as_of` favors the active claim.
- `recall(..., as_of=...)` evaluates claim validity at the requested time.
- `to_prompt()` serializes only model-useful capsule content.
- `evidence_ids` and `expand()` let the application inspect the exact source record.

## Use namespaces deliberately

Every write, recall, expansion, extraction audit, and purge is scoped by a namespace.
Choose an application-owned convention such as:

```text
product/account-42/assistant-7
```

A namespace is a logical filter, not an authentication mechanism. The host application
must authenticate its caller and authorize the namespace before invoking EviCortex. Do
not accept an arbitrary namespace directly from an untrusted client.

## Make retries idempotent

Use an application-stable key when a write may be retried:

```python
record = memory.remember(
    namespace=NAMESPACE,
    content="Ada prefers concise notifications.",
    source="conversation/session-8/turn-2",
    idempotency_key="session-8/turn-2",
)
```

Replaying the same normalized request with the same key returns the existing record.
Reusing the key for different content or write parameters raises an idempotency conflict.
Keys are unique within a namespace.

## Verify the local store

For a persistent database, periodically inspect integrity and row counts:

```python
with EviCortex.open("evicortex.db") as memory:
    report = memory.verify_integrity(namespace=NAMESPACE)
    print(report.ok, report.corrupt_memory_ids, report.index_issues)
    print(memory.stats(namespace=NAMESPACE))
```

`rebuild_indexes()` repairs deterministic claim-status and FTS projections. It does not
call an embedding provider; semantic projections use the explicit namespace-scoped
`sync_semantic()` path described in the [API guide](api-guide.md#semantic-projection-lifecycle).

## Next steps

- Read [Core concepts](concepts.md) before modeling updates or contradictions.
- Use the [API guide](api-guide.md) for extraction, dense embeddings, reranking, repair,
  and purge behavior.
- Read [FAQ and limitations](faq.md) before using sensitive data.
- Use [Benchmarking and reproducibility](benchmarking.md) to run the test suite and
  interpret retained results.
