"""Deterministic Prototype A benchmark helpers.

The benchmark package deliberately depends only on Python's standard library.  It
compares EviCortex with a mechanical SQLite FTS5 fixed-chunk baseline; it is not a
general comparison with the broad family of systems described as RAG.
"""

from .baseline import FixedChunkFTS5Baseline
from .synthetic import SyntheticSuite, generate_suite

__all__ = ["FixedChunkFTS5Baseline", "SyntheticSuite", "generate_suite"]
