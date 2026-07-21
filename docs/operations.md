---
layout: docs
title: "Operations and recovery"
---

# Operations and recovery

This runbook describes the narrow deployment envelope EviCortex supports during pre-alpha.
It reduces avoidable data-loss risk; it is not an availability, durability, or security
guarantee. Read [production readiness](production-readiness.md) before using persistent data.

## Supported operating envelope

- Run EviCortex as an embedded Python 3.12 library or CLI on one trusted local host.
- Keep each `EviCortex`/`SQLiteMemoryStore` instance on the thread that created it. Give
  separate workers separate instances and therefore separate SQLite connections.
- Authenticate users and authorize the exact namespace in the host application. A namespace
  supplied to EviCortex is a logical filter, not an access-control decision.
- Treat the database as plaintext. Use restrictive operating-system permissions and storage
  encryption supplied by the host when the data classification requires it.
- Keep the database on a local filesystem. Network filesystems, distributed writers, hot
  replicas, and high-availability topologies are outside the tested envelope.
- Treat every recalled record as untrusted data, not as instructions for an LLM or tool.

Do not store secrets, credentials, regulated data, or sensitive personal information under
this pre-alpha envelope.

## Ingestion contract

Use a stable, namespace-scoped `idempotency_key` for every retryable write. A replay with the
same canonical payload returns the existing record; a conflicting payload fails instead of
silently changing evidence.

When a semantic provider is configured, the lowest-risk batch pattern is:

1. Open with `auto_index=False`.
2. Commit immutable evidence with an idempotency key.
3. Queue the namespace for semantic synchronization.
4. Run `sync_semantic(namespace)` in a controlled worker.
5. Monitor `semantic_status(namespace)` and retry missing projections.

This separates the durable evidence commit from a fallible model call. If synchronous
`auto_index=True` raises `SemanticProjectionError` with `evidence_committed=True`, do not
recreate the evidence with a new key; repair the persisted record with `index_semantic()` or
`sync_semantic()`.

## Cold backup before upgrade or migration

Opening an older database may migrate it in place. Before the first EviCortex command or
library open under a new package version:

1. Stop every process that can access the database and confirm it has exited.
2. Copy the closed database to a versioned backup on a different failure domain.
3. Preserve the package version, application revision, database checksum, creation time,
   and source path alongside the backup.
4. Restore that backup to a disposable path, open the disposable copy with the new version,
   and run an integrity check there first.
5. Upgrade the working database only after the rehearsal succeeds.

With all connections closed, the default non-WAL configuration has one database file. If
`-wal`, `-shm`, or `-journal` sidecars exist, do not copy a hand-picked subset: keep the
application stopped and use a SQLite-aware backup process. EviCortex does not currently
provide a supported hot-backup command.

For the disposable restored copy:

```text
evicortex check --database <RESTORED_ABSOLUTE_PATH> --namespace <AUTHORIZED_NAMESPACE> --json
```

The check verifies SQLite integrity, evidence digests, and deterministic index consistency
within the requested namespace. Repeat it for every authorized namespace recorded in the
host application's inventory. It does not discover namespaces on behalf of an untrusted
caller.

## Restore and rollback

1. Stop all readers and writers.
2. Retain the failed database and logs for investigation; do not overwrite the last good
   backup.
3. Copy the selected backup to a new working path.
4. Open and check the restored copy with the package version recorded with that backup.
5. Point the application at the verified new path and perform a controlled smoke test.
6. Re-enable traffic gradually and monitor integrity, latency, provider failures, and disk
   growth.

EviCortex has no automatic point-in-time recovery, replication, recovery objective, or
backup retention policy. The host operator owns those controls and must rehearse them using
representative data before relying on persistent memory.

## Incident triggers

Stop writes and preserve evidence when any of these occurs:

- `evicortex check` exits with code `3` or reports corrupt evidence/index issues;
- SQLite reports corruption, I/O, disk-full, or locking errors;
- a migration, restore rehearsal, or schema-version check fails;
- semantic projection status remains incomplete after bounded retries; or
- an unauthorized namespace or sensitive record may have been exposed.

Do not use deterministic index rebuilds to conceal an evidence-digest failure. Rebuildable
views may be repaired; immutable evidence corruption requires restore or investigation.

## Capacity and service expectations

The published checkpoints cover small local and 1,000-event synthetic workloads. They do
not establish a production capacity limit. Set application-specific limits on input bytes,
query rate, database size, semantic scan size, and context budgets, then load test with the
same SQLite build and provider versions used in deployment. The CLI bounds each individual
file/stdin payload at 8 MiB, but the embedded Python API does not impose that CLI limit.
