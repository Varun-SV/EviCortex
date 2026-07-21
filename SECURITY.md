# Security policy

EviCortex is pre-alpha research software. It is not yet a tenant-security boundary, encrypted
vault, or forensic-erasure system.

## Supported versions

Security fixes currently target the latest commit on `main`. No released version has a long-term
support promise yet.

## Report a vulnerability

Please do not open a public issue for a suspected vulnerability. Use the repository's
[private vulnerability reporting](https://github.com/Varun-SV/EviCortex/security/advisories/new)
form. Include affected revisions, reproduction steps, impact, and any suggested mitigation.

Do not include real secrets, credentials, private conversations, or personal data in a report.

## Current boundary

- The host application must authenticate callers and authorize every namespace.
- Local SQLite files are not encrypted by EviCortex.
- Model and dataset artifacts must be independently trusted and pinned.
- Logical purge does not guarantee forensic erasure from backups or storage media.
- Retrieved evidence is untrusted input and still requires application-level prompt-injection
  defenses before it reaches an LLM.
