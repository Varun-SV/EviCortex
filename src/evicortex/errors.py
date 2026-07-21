"""Domain exceptions raised by the EviCortex storage kernel."""


class EviCortexError(Exception):
    """Base class for predictable EviCortex failures."""


class ValidationError(EviCortexError, ValueError):
    """Input cannot be represented without violating a storage invariant."""


class IdempotencyConflictError(EviCortexError):
    """An idempotency key was reused for a different write request."""


# A shorter spelling is kept as a convenience for API consumers.
IdempotencyConflict = IdempotencyConflictError


class StorageClosedError(EviCortexError):
    """An operation was attempted after the store was closed."""


class FTS5UnavailableError(EviCortexError):
    """The running SQLite library does not provide the required FTS5 module."""


class IntegrityViolationError(EviCortexError):
    """Persisted data failed a storage integrity invariant."""


class EmbeddingProfileNotFoundError(EviCortexError, LookupError):
    """A semantic read named a profile that has no persisted projections."""


class SemanticNotConfiguredError(EviCortexError):
    """A semantic-only operation was requested without an embedder."""


class SemanticProjectionError(EviCortexError):
    """Exact evidence committed, but its rebuildable semantic projection failed."""

    def __init__(
        self,
        message: str,
        *,
        memory_id: str,
        evidence_committed: bool = True,
    ) -> None:
        super().__init__(message)
        self.memory_id = memory_id
        self.evidence_committed = evidence_committed
