# Changelog

All notable changes are recorded here. EviCortex follows Semantic Versioning for package
tags, with the important pre-1.0 qualification described in the
[release policy](docs/release-policy.md).

## Unreleased

### Added

- A dependency-free `evicortex` CLI for explicit remember, recall, evidence expansion,
  integrity checks, and a deterministic offline temporal-memory demo.
- A distributable Codex plugin and `evicortex-memory` skill with evidence-citation and
  untrusted-memory safety rules.
- Build Week judge, submission, operations, recovery, and production-readiness guidance.
- Distribution, cross-platform test, Ruff, Mypy, and CodeQL workflows plus weekly dependency
  update checks.

### Security

- CLI file and standard-input payloads are decoded strictly as UTF-8 and bounded to 8 MiB.
- Common SQLite database, journal, WAL, and shared-memory files are excluded from Git.
- GitHub Actions are pinned to immutable commit SHAs and release publishing is separated from
  the read-only artifact build job.

## 0.3.0a0 - 2026-07-20

### Added

- Evidence-preserving SQLite/FTS5 research kernel with immutable observations and stable IDs.
- Valid-time claims, supersession, current and historical recall, conflict surfacing, and
  estimated-token-bounded memory capsules.
- Rebuildable lexical, entity, graph, temporal, and optional exact-semantic projections.
- Provider-neutral extraction, embedding, and reranking contracts with durable provenance.
- Controlled diagnostic baselines, reproducible benchmark artifacts, documentation site,
  CI, contribution policy, security policy, and MIT license.

### Changed

- Standardized the project and Python namespace as EviCortex before publication.
