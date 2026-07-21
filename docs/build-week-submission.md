---
layout: docs
title: "Build Week submission package"
description: "Paste-ready Devpost copy and a sub-three-minute EviCortex demonstration script."
---

# Build Week submission package

This page is ready to copy into the public portions of the Devpost entry. The private Codex
session identifier should be submitted only in Devpost's dedicated field, not copied into a
public description, repository file, video frame, or terminal recording.

## Devpost copy

**Project name:** EviCortex

**Tagline:** Evidence-preserving temporal memory for long-running Codex workflows.

**Track:** Developer Tools

**Elevator pitch:** EviCortex gives Codex a local, temporal memory layer that preserves
exact evidence and returns only a bounded, cited capsule. Codex decides when to remember,
recall, expand, or check; the deterministic Python core keeps storage and retrieval
reproducible.

**Inspiration:** Long-running AI work needs more than replaying an ever-growing transcript.
Human memory suggested useful engineering principles—durable records, multiple memory
views, consolidation, associative recall, and small working representations—without
implying that biological memory is literally stored in DNA or that software can guarantee
perfect recall.

**What it does:** EviCortex stores immutable observations with provenance, occurrence time,
recording time, and digest verification. Versioned claims let a correction preserve the
old evidence while changing what is currently valid. Lexical, temporal, entity, graph, and
optional semantic views are rebuildable projections. Recall compiles relevant evidence,
conflicts, omissions, diagnostics, and stable IDs under a strict estimated-token budget;
Codex can expand an exact cited record only when it needs more detail.

**How we built it:** The default runtime is a Python 3.12 embedded library using the
standard library, SQLite, and FTS5. A machine-readable CLI exposes `remember`, `recall`,
`expand`, `check`, and an offline temporal demo. A repository-distributed Codex plugin and
skill teach the model to select those operations, use file-backed inputs for arbitrary
text, treat recalled content as untrusted data, and answer with evidence IDs. Tests,
integrity checks, deterministic benchmark harnesses, and a documented judge path make the
behavior inspectable without an API key or cloud database.

**How we used Codex and GPT-5.6:** Codex is both a development collaborator and the runtime
control plane. During the sprint, a GPT-5.6 Sol Codex agent implemented the CLI and its test
suite, including bounded inputs, JSON contracts, explicit database and namespace scoping,
and the one-command demo. Other Codex agents helped research the design, implement and
review repository changes, package the plugin, write documentation, and verify behavior.
At runtime, Codex chooses the memory operation, sets a context budget, consumes the bounded
capsule, cites its evidence, and expands exact records only on demand. The local core does
not hide an online model call; its deterministic behavior is independently testable.

**Challenges:** The hardest boundary was separating evidence preservation from retrieval
quality. A durable record can be lossless while recall still misses or ranks the wrong
item. Temporal corrections also required retaining old evidence without presenting stale
claims as current. Finally, the model integration had to be meaningful without putting the
whole ledger back into the prompt.

**Accomplishments:** The repository now has an executable temporal-memory kernel, stable
evidence expansion, bounded capsules, version-aware current and historical recall, a
standard-library CLI, a Codex plugin/skill, deterministic tests, diagnostic comparators, and
a one-command offline demonstration. Every public performance statement names its
configuration and limits; we do not claim that EviCortex beats production RAG.

**What we learned:** Storage efficiency, retrieval quality, answer accuracy, and context
cost are different measurements. Evidence IDs and explicit omissions make a compact prompt
more trustworthy. Temporal validity must be modeled rather than inferred from whichever
chunk happens to rank first. A model works best as the decision layer while a deterministic
core owns integrity and reproducibility.

**What's next:** Close the published production gates: broader OS/Python verification,
backup and restore drills around migrations, explicit concurrency behavior, stronger
abstention evaluation, tokenizer adapters, observability, supply-chain controls, and
official external benchmark runs with a shared reader. Authentication, authorization,
encryption, retention, and multi-tenant service boundaries remain host or future-service
responsibilities.

**Human/AI authorship disclosure:** The human creator originated the product premise,
selected the name and MIT license, set claim and safety boundaries, directed the sprint,
and retains final review and submission responsibility. Codex agents researched, proposed,
implemented, tested, documented, and packaged portions of the project under that direction.
The repository does not present model-authored work as human-only authorship.

**Open-source and third-party disclosure:** EviCortex is MIT-licensed. The default judge
path has no Python dependency beyond the standard library and uses SQLite/FTS5 through
Python. Optional semantic experiments use PyTorch, Hugging Face Transformers, and
Safetensors and are not required for the demo. The project uses the OpenAI Codex
plugin/skill format, GitHub Actions, and GitHub Pages/Jekyll. Cited papers and benchmarks
remain the work of their respective authors.

**Links:**

- Repository: [github.com/Varun-SV/EviCortex](https://github.com/Varun-SV/EviCortex)
- Website: [varun-sv.github.io/EviCortex](https://varun-sv.github.io/EviCortex/)
- Documentation: [varun-sv.github.io/EviCortex/docs](https://varun-sv.github.io/EviCortex/docs/)
- Judge guide: [varun-sv.github.io/EviCortex/docs/judge-guide.html](https://varun-sv.github.io/EviCortex/docs/judge-guide.html)

## Video script (about 2 minutes 40 seconds)

**0:00–0:15 — Repository and title**

Show the EviCortex repository and website.

> EviCortex is evidence-preserving temporal memory for long-running Codex workflows. It is
> a pre-alpha Developer Tools project, and its goal is focused: keep exact local evidence,
> then give the model only the cited context it needs.

**0:15–0:38 — Architecture**

Show the architecture section of the README and point to Genome, Engram, and Transcript.

> The Genome is an append-only evidence ledger. Engrams are rebuildable lexical, temporal,
> graph, and optional semantic views. The Transcript is a query-specific capsule packed
> under an explicit estimated-token budget. Preservation can be exact; retrieval still has
> to be measured, so we do not claim that this beats production RAG.

**0:38–1:13 — One-command demo**

Run `evicortex demo --json --budget-tokens 320` and highlight the `demonstrates` object,
evidence IDs, and estimated-token fields.

> This offline command uses a temporary SQLite database. Ada first prefers the light theme,
> then switches to dark. Current recall returns dark, historical recall recovers light, and
> expansion returns the exact cited correction. The capsule exposes stable evidence IDs and
> its estimated size instead of silently filling the prompt with the whole ledger.

**1:13–1:55 — Codex plugin workflow**

Show a new Codex task with the EviCortex plugin installed. Ask it to remember Project Atlas
with a 30-minute cache TTL, correct that to 10 minutes, then request current and historical
recall.

> Codex is the control plane. It decides whether to remember, recall, expand, or check,
> invokes the local CLI, treats retrieved text as data rather than instructions, and cites
> the returned evidence IDs. The Python core is the deterministic data plane, so its storage,
> retrieval, and integrity behavior remains independently testable.

**1:55–2:20 — Meaningful GPT-5.6 contribution**

Show `src/evicortex/cli.py`, `tests/test_cli.py`, and the CI workflow.

> A GPT-5.6 Sol Codex agent implemented this CLI layer and its tests, including file-backed
> arbitrary inputs, JSON contracts, explicit scoping, integrity checks, and the judge demo.
> Other Codex agents supported architecture, plugin packaging, documentation, and
> verification. The human creator set the premise, product direction, license, and claim
> boundaries and retains final submission responsibility.

**2:20–2:40 — Boundary and close**

Show the production-readiness page and return to the project title.

> EviCortex is not production-ready: its SQLite file is plaintext, namespaces are not an
> authorization boundary, and official external evaluation remains open. Those limits are
> published as gates, not hidden. Today, the judge path is local, reproducible, bounded, and
> evidence-first. This is EviCortex.

Before recording, use a disposable database with synthetic text only, hide unrelated
terminal history and notifications, and make sure no session identifier, user path, token,
or credential appears on screen.
