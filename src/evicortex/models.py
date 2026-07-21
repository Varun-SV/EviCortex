"""Public, dependency-free data models for the EviCortex storage kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .semantic import EmbeddingProfile


TimestampInput = datetime | str | None


@dataclass(frozen=True, slots=True)
class ClaimInput:
    """A structured interpretation to attach to one immutable evidence row."""

    subject: str
    predicate: str
    value: str
    valid_from: TimestampInput = None
    valid_to: TimestampInput = None
    confidence: float = 1.0
    authority: str = "application"
    supersedes: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """A persisted, temporally versioned claim linked to exact evidence."""

    id: str
    namespace: str
    memory_id: str
    subject: str
    predicate: str
    value: str
    valid_from: str
    valid_to: str | None
    recorded_at: str
    confidence: float
    authority: str
    supersedes: str | None
    status: str

    @property
    def evidence_id(self) -> str:
        """Alias emphasizing that ``memory_id`` identifies supporting evidence."""

        return self.memory_id


@dataclass(frozen=True, slots=True)
class ClaimDerivationRecord:
    """Exact evidence span from which one persisted claim was derived."""

    claim_id: str
    claim_index: int
    start: int
    end: int
    canonical_text: str
    content_sha256: str
    extractor_identity: str


@dataclass(frozen=True, slots=True)
class ExtractionDiagnosticRecord:
    """Persisted machine-readable reason a span emitted no claim."""

    diagnostic_index: int
    code: str
    message: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractionRecord:
    """One durable claim-extraction run tied to immutable evidence."""

    memory_id: str
    namespace: str
    extractor_identity: str
    content_sha256: str
    recorded_at: str
    abstained: bool
    derivations: tuple[ClaimDerivationRecord, ...] = ()
    diagnostics: tuple[ExtractionDiagnosticRecord, ...] = ()

    @property
    def claim_ids(self) -> tuple[str, ...]:
        return tuple(item.claim_id for item in self.derivations)


@dataclass(frozen=True, slots=True)
class MemoryInput:
    """Convenient value object for callers that stage evidence before ingestion."""

    content: str
    kind: str = "episode"
    source: str = ""
    occurred_at: TimestampInput = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    claims: Sequence[ClaimInput] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """An immutable evidence row and its currently stored claim records."""

    id: str
    namespace: str
    kind: str
    content: str
    content_hash: str
    source: str
    occurred_at: str
    recorded_at: str
    metadata: Mapping[str, Any]
    idempotency_key: str | None
    claims: tuple[ClaimRecord, ...] = ()

    @property
    def memory_id(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """A ranked evidence result with an inspectable retrieval reason."""

    memory: MemoryRecord
    rank: float
    reason: str

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def memory_id(self) -> str:
        return self.memory.id

    @property
    def namespace(self) -> str:
        return self.memory.namespace

    @property
    def content(self) -> str:
        return self.memory.content

    @property
    def content_hash(self) -> str:
        return self.memory.content_hash


@dataclass(frozen=True, slots=True)
class ClaimHit:
    """A ranked claim result with an inspectable retrieval reason."""

    claim: ClaimRecord
    rank: float
    reason: str

    @property
    def id(self) -> str:
        return self.claim.id

    @property
    def claim_id(self) -> str:
        return self.claim.id

    @property
    def namespace(self) -> str:
        return self.claim.namespace

    @property
    def memory_id(self) -> str:
        return self.claim.memory_id

    @property
    def evidence_id(self) -> str:
        return self.claim.memory_id

    @property
    def subject(self) -> str:
        return self.claim.subject

    @property
    def predicate(self) -> str:
        return self.claim.predicate

    @property
    def value(self) -> str:
        return self.claim.value


@dataclass(frozen=True, slots=True)
class SemanticMemoryHit:
    """One evidence row ranked by exact cosine similarity."""

    memory: MemoryRecord
    score: float
    reason: str

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def memory_id(self) -> str:
        return self.memory.id

    @property
    def namespace(self) -> str:
        return self.memory.namespace


@dataclass(frozen=True, slots=True)
class SemanticClaimHit:
    """One temporally eligible claim ranked by exact cosine similarity."""

    claim: ClaimRecord
    score: float
    reason: str

    @property
    def id(self) -> str:
        return self.claim.id

    @property
    def claim_id(self) -> str:
        return self.claim.id

    @property
    def memory_id(self) -> str:
        return self.claim.memory_id

    @property
    def namespace(self) -> str:
        return self.claim.namespace


@dataclass(frozen=True, slots=True)
class SemanticSearchDiagnostics:
    """Bounded-scan facts that prevent semantic retrieval from being opaque."""

    profile: EmbeddingProfile
    scanned: int
    scan_cap: int
    truncated: bool
    query_is_zero: bool
    no_vectors: bool

    @property
    def profile_sha256(self) -> str:
        return self.profile.profile_sha256


@dataclass(frozen=True, slots=True)
class SemanticMemorySearchResult:
    """Exact semantic memory hits and the diagnostics for their bounded scan."""

    hits: tuple[SemanticMemoryHit, ...]
    diagnostics: SemanticSearchDiagnostics

    @property
    def scanned(self) -> int:
        return self.diagnostics.scanned

    @property
    def truncated(self) -> bool:
        return self.diagnostics.truncated

    @property
    def profile(self) -> EmbeddingProfile:
        return self.diagnostics.profile


@dataclass(frozen=True, slots=True)
class SemanticClaimSearchResult:
    """Exact semantic claim hits and the diagnostics for their bounded scan."""

    hits: tuple[SemanticClaimHit, ...]
    diagnostics: SemanticSearchDiagnostics

    @property
    def scanned(self) -> int:
        return self.diagnostics.scanned

    @property
    def truncated(self) -> bool:
        return self.diagnostics.truncated

    @property
    def profile(self) -> EmbeddingProfile:
        return self.diagnostics.profile


@dataclass(frozen=True, slots=True)
class EmbeddingUpsertResult:
    """Summary of one all-or-nothing projection write batch."""

    profile: EmbeddingProfile
    written: int
    memory_embeddings: int
    claim_embeddings: int
    namespaces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingSyncStatus:
    """Missing and stale source IDs for one namespace/profile projection."""

    profile: EmbeddingProfile
    namespace: str
    target_kind: str
    total_sources: int
    projected_sources: int
    missing_ids: tuple[str, ...]
    stale_ids: tuple[str, ...]
    registered: bool = True

    @property
    def complete(self) -> bool:
        # An empty source set is already synchronized even when this profile has
        # never been registered.  Registration describes persisted projection
        # rows; it is not work required for a vacuous namespace.
        return (
            (self.registered or self.total_sources == 0)
            and not self.missing_ids
            and not self.stale_ids
        )

    @property
    def profile_sha256(self) -> str:
        return self.profile.profile_sha256


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Result of evidence-digest and derived-index verification."""

    ok: bool
    checked_memories: int
    corrupt_memory_ids: tuple[str, ...] = ()
    index_issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.ok


@dataclass(frozen=True, slots=True)
class StoreStats:
    """Small diagnostic snapshot of base and derived row counts."""

    memories: int
    claims: int
    namespaces: int
    memory_index_rows: int
    claim_index_rows: int

    @property
    def memory_count(self) -> int:
        return self.memories

    @property
    def claim_count(self) -> int:
        return self.claims
