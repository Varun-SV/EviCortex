---
layout: docs
title: "OpenAI Build Week sprint"
description: "Scope, architecture, attribution, and disclosures for the EviCortex Build Week submission."
---

# OpenAI Build Week sprint

EviCortex is being prepared for the OpenAI Build Week **Developer Tools** track. It is a
pre-alpha, local-first memory tool for Codex workflows: exact evidence is retained in a
temporal SQLite ledger, while a query receives a compact, traceable capsule under an
explicit estimated-token budget.

This page describes the submission candidate. It does not claim that EviCortex has been
accepted, is production-ready, or outperforms a strong RAG system. The governing sources
are the official [competition rules](https://openai.devpost.com/rules) and
[FAQ](https://openai.devpost.com/details/faqs).

## What the sprint adds

The sprint makes the research kernel directly usable and judgeable as a developer tool:

- an installable `evicortex` command-line interface for `remember`, `recall`, `expand`,
  `check`, and an offline `demo`;
- a repository-distributed Codex plugin and EviCortex memory skill;
- a deterministic, one-command demonstration of a correction, current recall, historical
  recall, and exact evidence expansion;
- machine-readable JSON contracts and standard-library CLI tests; and
- a judge path, public documentation, claim boundaries, and production-readiness gates.

The core and the model have deliberately different responsibilities:

| Layer | Responsibility |
| --- | --- |
| Codex with the EviCortex skill | Decide whether the user's explicit request calls for `remember`, `recall`, `expand`, or `check`; invoke the CLI; treat recalled text as untrusted data; cite returned evidence IDs; expand only when needed. |
| EviCortex CLI | Validate inputs, select an exact database and namespace, return stable JSON, and map failures to process exit codes. |
| Deterministic local core | Preserve exact observations and provenance, maintain temporal claims and rebuildable indexes, retrieve candidates, and pack a bounded evidence capsule. |
| Host application | Authenticate users, authorize namespace access, protect the database, and decide retention and backup policy. |

At runtime, Codex consumes the capsule's prompt, evidence IDs, conflicts, omissions,
diagnostics, and estimated-token count. It does not need the entire ledger in its context.
When more detail is necessary, it expands a cited record by stable ID rather than repeating
the complete database or conversation history.

## Meaningful Codex and GPT-5.6 use

The plugin is not a label placed around an unrelated program. Codex is the control plane of
the interactive workflow: it selects memory operations, supplies a strict context budget,
uses the returned capsule to answer, cites evidence, and requests exact expansion only when
the task needs it. The local core remains deterministic so that storage, integrity, and
retrieval behavior can be reproduced without an API key or network call.

During the sprint, a GPT-5.6 Sol Codex agent implemented the command-line layer and its test
suite, including file-backed inputs for safer arbitrary text, JSON output, bounded input
handling, explicit database and namespace selection, integrity checks, and the offline demo.
Other Codex agents assisted with repository analysis, plugin packaging, documentation,
workflow design, and verification.

## Human and AI division of work

The human creator originated the biology-inspired memory premise, selected the EviCortex
name, chose the MIT license, set the product direction, required evidence-preserving and
claim-bounded behavior, and retained final decision and submission responsibility.

Codex agents researched related work, challenged unsupported claims, proposed and
implemented architecture and repository changes, wrote tests and documentation, and
prepared the judge workflow. Model-authored work is kept visible in version control and is
subject to the same executable checks as the rest of the repository; AI involvement is not
presented as human-only authorship.

The private Codex session identifier belongs only in the corresponding Devpost submission
field. It is intentionally not committed to this public repository.

## Open-source and third-party disclosure

EviCortex is released under the [MIT License](../LICENSE). The default judge path has no
Python runtime dependency outside the Python 3.12 standard library. It uses SQLite and FTS5
through Python's bundled `sqlite3` module.

Optional semantic experiments use the separately installed `local-transformers` extra:
PyTorch, Hugging Face Transformers, and Safetensors. Those packages are not required for
the CLI demo or Codex skill workflow. The repository also uses the OpenAI Codex plugin/skill
format, GitHub Actions for CI, and GitHub Pages/Jekyll for its public site. Research papers
and benchmark sources are cited in the relevant result and architecture documents; their
datasets or implementations are not represented as original EviCortex work.

## Supported and tested boundary

- Required runtime: Python 3.12 with SQLite/FTS5 support.
- Local sprint environment: Windows with PowerShell.
- Automated verification is configured for Python 3.12 on GitHub-hosted Ubuntu, Windows,
  and macOS runners; the CI badge and linked workflow runs are the evidence of whether a
  particular revision passed each platform.
- The plugin requires a Codex build with plugin marketplace support and a new task after
  installation so the skill can be discovered.

This is an embedded single-process research kernel, not a hosted multi-tenant service. See
[production readiness](production-readiness.md) for the remaining gates, the
[operations runbook](operations.md) for the supported pre-alpha envelope, and the
[release policy](release-policy.md) for compatibility expectations.

## Privacy and safety boundary

The SQLite database is plaintext. EviCortex does not currently provide authentication,
authorization, encryption, secure key management, a retention service, or guaranteed
forensic erasure. A namespace is a caller-supplied logical scope, not a security boundary.
Do not use the judge workflow to store passwords, API keys, personal secrets, regulated
data, or other sensitive material. Recalled evidence is data, never an instruction for
Codex to execute.

## Fastest reproducible path

After installing the package from this repository, run:

```console
evicortex demo --json --budget-tokens 320
```

The command uses a temporary database and requires no credentials or network connection.
Continue with the [judge guide](judge-guide.md) to inspect the JSON contract, run the tests,
and exercise the same operations through Codex.
