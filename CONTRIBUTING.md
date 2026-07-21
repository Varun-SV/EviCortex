# Contributing to EviCortex

EviCortex is a pre-alpha research project. Contributions are welcome, but changes must preserve
the project's evidence, isolation, temporal, and reproducibility contracts.

## Development setup

EviCortex requires Python 3.12. From a source checkout in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/quickstart.py
```

No third-party package is required for the deterministic kernel. The optional local transformer
adapter has additional dependencies declared in `pyproject.toml`.

## Validate a change

Run the complete standard-library test suite and compile checks before opening a pull request:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src benchmarks examples tests
python examples/quickstart.py
python examples/phase_b_diagnostic.py
```

The cached transformer integration test is intentionally optional and only runs when its pinned
model and revision environment variables are provided.

## Pull requests

- Keep each pull request focused on one coherent change.
- Add or update tests for behavioral changes.
- Preserve namespace filtering before every ranking channel.
- Keep exact evidence immutable; derived indexes must remain rebuildable.
- Document benchmark configuration and limitations alongside every reported result.
- Do not claim superiority over RAG or another system without a controlled, reproducible study.

Use a short imperative commit message and explain what changed, why it matters, and which checks
you ran in the pull request description.

## Reporting problems

Use [GitHub Issues](https://github.com/Varun-SV/EviCortex/issues) for reproducible bugs and feature
proposals. Please follow [SECURITY.md](SECURITY.md) instead for suspected vulnerabilities.
