---
layout: docs
title: "Release and compatibility policy"
---

# Release and compatibility policy

EviCortex is currently `0.3.0a0`: pre-alpha research software. This policy makes that label
operational and avoids implying stability that the project has not earned.

## Versioning

Package releases use Semantic Versioning. Before `1.0.0`, a minor release may make breaking
changes to the Python API, CLI JSON shape, plugin workflow, retrieval behavior, or database
requirements. Patch releases should remain compatible within the same minor line unless a
security or data-integrity issue makes that unsafe.

Every release must:

- use one release identity everywhere (PEP 440 `0.3.0a0` in Python and the equivalent
  SemVer `0.3.0-alpha.0` in plugin metadata);
- update [the changelog](../CHANGELOG.md) with breaking changes and migration notes;
- pass source, wheel, cross-platform, plugin, documentation, and full test validation;
- publish a Git tag and GitHub release containing the source distribution and wheel; and
- retain the exact commit and workflow run used to build the artifacts.

The automated release workflow does not publish to PyPI. A tag is not evidence of production
readiness, signing, reproducibility, or a support commitment.

## Python API and CLI

The names exported by `evicortex.__all__` are the intended public Python surface. Everything
else is internal unless documented otherwise. During pre-alpha, breaking public changes must
be called out in the changelog; a deprecation window is best effort rather than guaranteed.

CLI exit codes and top-level JSON keys are integration surfaces:

| Code | Meaning |
| --- | --- |
| `0` | Command completed successfully |
| `1` | Operational or storage failure |
| `2` | Invalid command or input |
| `3` | Integrity check completed and found a failure |

Consumers must tolerate additive JSON fields. Removing or changing a documented field is a
breaking change during pre-alpha and requires release notes.

## Database compatibility

The current code recognizes schema versions 1 through 5 and migrates older recognized
versions forward. It refuses a newer schema version rather than guessing. Migration is
currently automatic on open and has no built-in rollback, so operators must follow the cold
backup and rehearsal procedure in [operations and recovery](operations.md) before upgrading.

Downgrading a migrated database is unsupported. Restore a backup with the earlier package
version instead.

## Support window

Until a stable release, only the latest published pre-alpha line receives best-effort fixes.
There is no SLA, long-term-support branch, security embargo guarantee, or guaranteed migration
path between arbitrary development commits. Report vulnerabilities privately using the
instructions in [SECURITY.md](../SECURITY.md); use public GitHub issues for non-sensitive bugs.

The supported deployment envelope is defined by [production readiness](production-readiness.md),
not by the existence of a package artifact.
