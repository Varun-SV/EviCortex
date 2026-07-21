---
layout: docs
title: "Production readiness"
---

# Production readiness

EviCortex is pre-alpha research software. It is **not production-ready today**. This page
defines the evidence required before that label can change and prevents a working prototype,
successful demo, or narrow benchmark from being mistaken for an operational guarantee.
The [operations and recovery runbook](operations.md) defines the smaller envelope that can be
tested today.

## Release decision

A production release requires every P0 gate below to be closed with a reproducible artifact.
An open P1 gate requires a documented risk acceptance and rollback plan. P2 work may remain on
the roadmap when it does not invalidate the supported deployment envelope.

| Area | Current evidence | Status | Production gate |
| --- | --- | --- | --- |
| Core correctness | 184-test deterministic suite, Ruff and Mypy gates, and three-OS Python 3.12 CI configuration | Partial | Green protected-branch evidence, property/fuzz tests for storage and budgets, and a supported-version policy |
| Packaging | Wheel and source distribution build, isolated wheel install, and CLI demo verification | Partial | Signed, reproducible release artifacts with provenance and an installation rollback path |
| Data integrity | SHA-256 evidence checks, SQLite integrity checks, schema migrations, rebuildable deterministic indexes | Partial | Documented backup, restore, crash-recovery, and migration rollback drills on production-sized data |
| Isolation | Namespace filtering precedes ranking | Partial | Host authentication/authorization reference adapter and negative integration tests; namespaces alone are not security boundaries |
| Confidentiality | No silent cloud dependency in the core | Open P0 | Encryption and secret-management guidance, file-permission checks, data classification, and threat-model review |
| Privacy and deletion | Scoped logical hard purge exists | Partial | Retention controls, backup deletion policy, audit evidence, and explicit forensic-erasure limitations |
| Retrieval quality | Controlled synthetic diagnostics and a tiny pinned semantic smoke | Open P0 | Official external benchmark, shared reader, calibrated abstention, quality thresholds, and regression gates |
| Scale | Exact scans and 1,000-event diagnostics | Open P0 | 10k/100k stress tests, ANN recall evaluation, concurrency limits, and resource envelopes |
| Availability | Embedded single-process library | Open P0 | Supported deployment topology, health/readiness checks, recovery objectives, and failure-injection tests |
| Observability | Retrieval score traces and integrity reports | Open P1 | Structured operational logs, metrics, tracing, alert thresholds, and sensitive-data redaction |
| Provider reliability | Provider identity snapshots and durable extraction provenance | Partial | Timeouts, retries, circuit breaking, quota handling, version compatibility, and offline/failure tests |
| API stability | Typed public facade, pre-1.0 compatibility policy, changelog, and documented CLI exit codes | Open P1 | A tested upgrade guide and an evidence-backed deprecation window |
| Supply chain | Minimal core dependencies, SHA-pinned Actions, Dependabot, CodeQL, and pinned optional model identity | Partial | SBOM, artifact attestations, release signing, and repository protection settings |
| Operations | Integrity/rebuild APIs and a cold-backup, restore, migration, and incident runbook | Open P0 | Tested backup tooling, destructive recovery drills, capacity thresholds, and defined recovery objectives |

## Supported envelope during pre-alpha

- Python 3.12 embedded applications using a local SQLite database with FTS5.
- One host application that authenticates users and authorizes every namespace before calling
  EviCortex.
- Workloads that fit exact local scans and the documented estimated-token accounting model.
- Evidence that may be stored unencrypted unless the host or storage layer provides encryption.
- Retrieval output treated as untrusted data before it is placed in an LLM prompt.
- One EviCortex instance per creator thread, with separate SQLite connections for separate
  workers; no supported network filesystem or high-availability topology.

Anything outside that envelope is an experiment until a named gate closes.

## P0 closure sequence

1. Freeze the supported API and persistence compatibility contract.
2. Complete a repository threat model and security scan, then fix all validated high-impact
   findings.
3. Add backup/restore and migration rollback tooling with destructive-failure drills.
4. Add a host authorization reference adapter and verify isolation through integration tests.
5. Run official quality evaluation with a shared reader and calibrated abstention.
6. Establish 10k/100k performance, concurrency, memory, and disk envelopes.
7. Add structured telemetry, operator runbooks, release provenance, and rollback procedures.
8. Run a release-candidate soak and incident simulation before publishing a stable version.

## Evidence policy

Every closed gate must link to a test, workflow run, benchmark artifact, security report, or
operational drill. A statement without a reproducible artifact is a plan, not evidence.
