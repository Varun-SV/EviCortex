"""Provider-neutral contracts for optional EviCortex Phase B components.

The storage kernel deliberately has no dependency on a particular model vendor.
This module defines the narrow boundaries that embedders, claim extractors, and
rerankers can implement without changing that kernel.  The bundled
``HashingEmbedder`` is a deterministic feature-hashing diagnostic for tests and
CI; it is not a neural model and must not be described as semantic retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
import re
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .models import ClaimInput


EmbeddingVector = tuple[float, ...]


class EmbeddingValidationError(ValueError):
    """Raised when an embedding provider violates the Phase B contract."""


def _validated_dimensions(dimensions: int) -> int:
    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise TypeError("dimensions must be an integer")
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")
    return dimensions


def validate_embedding_batch(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    dimensions: int,
    normalized: bool = False,
    allow_zero: bool = False,
    tolerance: float = 1e-6,
) -> tuple[EmbeddingVector, ...]:
    """Validate and freeze a provider batch before it enters an index.

    Validation rejects count or dimensionality drift, booleans, non-numeric and
    non-finite coordinates, and (when requested) non-unit vectors.  Zero vectors
    are rejected by default because cosine similarity is undefined for them.
    Providers with an explicit empty-input policy may opt into ``allow_zero``.
    """

    _validated_dimensions(dimensions)
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise TypeError("expected_count must be an integer")
    if expected_count < 0:
        raise ValueError("expected_count cannot be negative")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    if len(vectors) != expected_count:
        raise EmbeddingValidationError(
            f"provider returned {len(vectors)} vectors for {expected_count} texts"
        )

    frozen: list[EmbeddingVector] = []
    for vector_index, vector in enumerate(vectors):
        if isinstance(vector, (str, bytes, bytearray, memoryview)):
            raise EmbeddingValidationError(
                f"vector {vector_index} must be a numeric coordinate sequence"
            )
        if len(vector) != dimensions:
            raise EmbeddingValidationError(
                f"vector {vector_index} has {len(vector)} dimensions; "
                f"expected {dimensions}"
            )
        coordinates: list[float] = []
        for coordinate_index, coordinate in enumerate(vector):
            if isinstance(coordinate, bool):
                raise EmbeddingValidationError(
                    f"vector {vector_index} coordinate {coordinate_index} is boolean"
                )
            try:
                number = float(coordinate)
            except (TypeError, ValueError) as error:
                raise EmbeddingValidationError(
                    f"vector {vector_index} coordinate {coordinate_index} "
                    "is not numeric"
                ) from error
            if not math.isfinite(number):
                raise EmbeddingValidationError(
                    f"vector {vector_index} coordinate {coordinate_index} "
                    "is not finite"
                )
            coordinates.append(number)

        # ``hypot`` scales internally, unlike sum-of-squares, so large but finite
        # provider coordinates do not overflow merely during validation.
        norm = math.hypot(*coordinates)
        if norm == 0.0 and not allow_zero:
            raise EmbeddingValidationError(f"vector {vector_index} is a zero vector")
        if normalized and norm != 0.0 and not math.isclose(
            norm, 1.0, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise EmbeddingValidationError(
                f"vector {vector_index} has L2 norm {norm!r}; expected 1.0"
            )
        frozen.append(tuple(coordinates))
    return tuple(frozen)


@runtime_checkable
class Embedder(Protocol):
    """A batch text embedder with stable index-compatibility metadata."""

    @property
    def identity(self) -> str:
        """Stable algorithm/model identity used to detect incompatible indexes."""

    @property
    def dimensions(self) -> int:
        """The fixed number of coordinates returned for every text."""

    @property
    def normalized(self) -> bool:
        """Whether non-zero output vectors are guaranteed to have unit L2 norm."""

    def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed a batch while preserving input order and cardinality."""


@runtime_checkable
class RoleAwareEmbedder(Embedder, Protocol):
    """Optional asymmetric query/document encoding extension.

    Some retrieval models require a query instruction while documents remain
    unprefixed. Providers implementing this protocol keep that distinction at
    the provider boundary and retain ``embed_batch`` as a document-encoding
    compatibility alias. Legacy symmetric :class:`Embedder` implementations
    continue to work through the dispatch helpers below.
    """

    @property
    def identity(self) -> str:
        """Stable identity covering both query and document encoding rules."""

    @property
    def dimensions(self) -> int:
        """The shared query/document vector dimensionality."""

    @property
    def normalized(self) -> bool:
        """Whether every non-zero query/document vector is unit normalized."""

    def embed_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        """Embed indexable document text in input order."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed one retrieval query using the provider's query rule."""


def embed_document_batch(
    embedder: Embedder,
    texts: Sequence[str],
) -> Sequence[Sequence[float]]:
    """Dispatch document encoding without breaking symmetric providers."""

    if not isinstance(embedder, Embedder):
        raise TypeError("embedder does not implement the Embedder protocol")
    has_documents = callable(getattr(embedder, "embed_documents", None))
    has_query = callable(getattr(embedder, "embed_query", None))
    if has_documents != has_query:
        raise TypeError(
            "role-aware embedder must implement both embed_documents and embed_query"
        )
    if has_documents and has_query:
        return embedder.embed_documents(texts)
    return embedder.embed_batch(texts)


def embed_query_text(embedder: Embedder, text: str) -> Sequence[float]:
    """Dispatch query encoding without conflating it with document encoding."""

    if not isinstance(embedder, Embedder):
        raise TypeError("embedder does not implement the Embedder protocol")
    if not isinstance(text, str):
        raise TypeError("query text must be a string")
    has_documents = callable(getattr(embedder, "embed_documents", None))
    has_query = callable(getattr(embedder, "embed_query", None))
    if has_documents != has_query:
        raise TypeError(
            "role-aware embedder must implement both embed_documents and embed_query"
        )
    if has_documents and has_query:
        return embedder.embed_query(text)
    vectors = tuple(embedder.embed_batch((text,)))
    if len(vectors) != 1:
        raise EmbeddingValidationError(
            f"provider returned {len(vectors)} vectors for one query"
        )
    return vectors[0]


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}][^\W_]+)?", re.UNICODE)


@dataclass(frozen=True, slots=True)
class HashingEmbedder:
    """Deterministic, non-neural feature hashing for CI diagnostics only.

    Token counts are projected into signed buckets with BLAKE2b and L2
    normalized.  The vectors test plugin wiring and index lifecycle behavior;
    they do not encode meaning and are not a semantic-embedding baseline.
    """

    dimensions: int = 256

    def __post_init__(self) -> None:
        _validated_dimensions(self.dimensions)

    @property
    def identity(self) -> str:
        return f"evicortex/diagnostic-feature-hashing-blake2b-v1/{self.dimensions}"

    @property
    def normalized(self) -> bool:
        return True

    @property
    def is_semantic(self) -> bool:
        """Always false: this diagnostic does not model semantic similarity."""

        return False

    @property
    def diagnostic_only(self) -> bool:
        """Always true: production quality claims require a real comparator."""

        return True

    def embed_batch(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")

        vectors: list[EmbeddingVector] = []
        for text_index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"text {text_index} must be a string")
            buckets = [0.0] * self.dimensions
            for token in _TOKEN_PATTERN.findall(text.casefold()):
                digest = hashlib.blake2b(
                    token.encode("utf-8"),
                    digest_size=16,
                    person=b"EviCortexHashV1",
                ).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = 1.0 if digest[8] & 1 else -1.0
                buckets[bucket] += sign
            norm = math.hypot(*buckets)
            if norm != 0.0:
                buckets = [value / norm for value in buckets]
            vectors.append(tuple(buckets))

        return validate_embedding_batch(
            vectors,
            expected_count=len(texts),
            dimensions=self.dimensions,
            normalized=True,
            # Blank or punctuation-only text has no features by design.
            allow_zero=True,
        )

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        """Symmetric diagnostic document encoding."""

        return self.embed_batch(texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        """Symmetric diagnostic query encoding."""

        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        return self.embed_batch((text,))[0]


def _freeze_json_value(
    value: object,
    *,
    path: str,
    active_container_ids: set[int],
) -> object:
    """Copy JSON-like data into recursively immutable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_container_ids:
            raise ValueError(f"{path} contains a reference cycle")
        active_container_ids.add(identity)
        try:
            frozen: dict[str, object] = {}
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings")
                frozen[key] = _freeze_json_value(
                    nested,
                    path=f"{path}.{key}",
                    active_container_ids=active_container_ids,
                )
            return MappingProxyType(frozen)
        finally:
            active_container_ids.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_container_ids:
            raise ValueError(f"{path} contains a reference cycle")
        active_container_ids.add(identity)
        try:
            return tuple(
                _freeze_json_value(
                    nested,
                    path=f"{path}[{index}]",
                    active_container_ids=active_container_ids,
                )
                for index, nested in enumerate(value)
            )
        finally:
            active_container_ids.remove(identity)
    raise TypeError(
        f"{path} contains unsupported {type(value).__name__}; "
        "expected JSON-like values"
    )


def _freeze_metadata(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen = _freeze_json_value(
        values,
        path="metadata",
        active_container_ids=set(),
    )
    # The root was checked as a Mapping above.
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """An immutable snapshot supplied to a claim extractor.

    ``content_sha256`` is checked on construction, preventing an extractor from
    silently associating claims with evidence other than the bytes identified by
    the caller.  ``claim_ids_by_source`` is an immutable resolution snapshot for
    explicit ``supersedes_source`` references; it is never guessed.
    """

    namespace: str
    content: str
    content_sha256: str
    source: str = ""
    occurred_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    claim_ids_by_source: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("content must be a non-empty string")
        if not isinstance(self.content_sha256, str):
            raise TypeError("content_sha256 must be a string")
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if digest != self.content_sha256:
            raise ValueError("content_sha256 does not match content")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if self.occurred_at is not None and not isinstance(self.occurred_at, str):
            raise TypeError("occurred_at must be a string or None")
        if not isinstance(self.claim_ids_by_source, Mapping):
            raise TypeError("claim_ids_by_source must be a mapping")
        claim_ids_by_source = dict(self.claim_ids_by_source)
        for source, claim_id in claim_ids_by_source.items():
            if not isinstance(source, str) or not source:
                raise ValueError("claim_ids_by_source keys must be non-empty strings")
            if not isinstance(claim_id, str) or not claim_id:
                raise ValueError("claim_ids_by_source values must be non-empty strings")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "claim_ids_by_source",
            MappingProxyType(claim_ids_by_source),
        )

    @classmethod
    def from_content(
        cls,
        *,
        namespace: str,
        content: str,
        source: str = "",
        occurred_at: str | None = None,
        metadata: Mapping[str, object] | None = None,
        claim_ids_by_source: Mapping[str, str] | None = None,
    ) -> EvidenceContext:
        """Build a context while deriving its exact UTF-8 content digest."""

        return cls(
            namespace=namespace,
            content=content,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            source=source,
            occurred_at=occurred_at,
            metadata={} if metadata is None else metadata,
            claim_ids_by_source=(
                {} if claim_ids_by_source is None else claim_ids_by_source
            ),
        )


@dataclass(frozen=True, slots=True)
class ClaimProvenance:
    """Exact evidence span and extractor identity for one emitted claim."""

    claim_index: int
    start: int
    end: int
    canonical_text: str
    content_sha256: str
    extractor_identity: str


@dataclass(frozen=True, slots=True)
class ExtractionDiagnostic:
    """A machine-readable reason an input span produced no claim."""

    code: str
    message: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True, slots=True)
class ClaimExtraction:
    """Conservative extraction output linked to immutable evidence spans."""

    claims: tuple[ClaimInput, ...] = ()
    provenance: tuple[ClaimProvenance, ...] = ()
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        try:
            claims = tuple(self.claims)
            provenance = tuple(self.provenance)
            diagnostics = tuple(self.diagnostics)
        except TypeError as error:
            raise TypeError("extraction collections must be iterable") from error
        if not all(isinstance(claim, ClaimInput) for claim in claims):
            raise TypeError("claims must contain only ClaimInput values")
        if not all(isinstance(item, ClaimProvenance) for item in provenance):
            raise TypeError("provenance must contain only ClaimProvenance values")
        if not all(isinstance(item, ExtractionDiagnostic) for item in diagnostics):
            raise TypeError(
                "diagnostics must contain only ExtractionDiagnostic values"
            )
        if len(claims) != len(provenance):
            raise ValueError("each extracted claim must have exactly one provenance row")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def abstained(self) -> bool:
        return not self.claims


def validate_claim_extraction(
    evidence: EvidenceContext,
    result: ClaimExtraction,
    *,
    expected_identity: str | None = None,
) -> ClaimExtraction:
    """Validate and freeze untrusted claim-extractor output.

    A valid provenance row maps positionally to exactly one claim and points to
    the exact evidence substring it records.  This function should be called at
    every third-party extractor boundary before claims are persisted.
    """

    if not isinstance(evidence, EvidenceContext):
        raise TypeError("evidence must be an EvidenceContext")
    if not isinstance(result, ClaimExtraction):
        raise TypeError("result must be a ClaimExtraction")
    if expected_identity is not None:
        if not isinstance(expected_identity, str) or not expected_identity.strip():
            raise ValueError("expected_identity must be a non-empty string")
        if expected_identity != expected_identity.strip():
            raise ValueError("expected_identity cannot have edge whitespace")

    # Reconstructing takes defensive snapshots even when a third party supplied
    # list-like values through Python's dynamically typed boundary.
    frozen = ClaimExtraction(
        claims=tuple(result.claims),
        provenance=tuple(result.provenance),
        diagnostics=tuple(result.diagnostics),
    )
    previous_end = 0
    observed_identity: str | None = None
    evidence_length = len(evidence.content)
    for claim_index, provenance in enumerate(frozen.provenance):
        if isinstance(provenance.claim_index, bool) or not isinstance(
            provenance.claim_index, int
        ):
            raise ValueError("provenance claim_index must be an integer")
        if provenance.claim_index != claim_index:
            raise ValueError(
                "provenance claim_index must exactly match its claim position"
            )
        if (
            isinstance(provenance.start, bool)
            or not isinstance(provenance.start, int)
            or isinstance(provenance.end, bool)
            or not isinstance(provenance.end, int)
        ):
            raise ValueError("provenance spans must use integer offsets")
        if not 0 <= provenance.start < provenance.end <= evidence_length:
            raise ValueError("provenance span is outside evidence bounds")
        if provenance.start < previous_end:
            raise ValueError("provenance spans must be ordered and non-overlapping")
        previous_end = provenance.end
        if not isinstance(provenance.canonical_text, str) or not provenance.canonical_text:
            raise ValueError("provenance canonical_text must be non-empty")
        if evidence.content[provenance.start : provenance.end] != provenance.canonical_text:
            raise ValueError("provenance canonical_text is not the exact evidence substring")
        if provenance.content_sha256 != evidence.content_sha256:
            raise ValueError("provenance content digest does not match evidence")
        identity = provenance.extractor_identity
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("provenance extractor identity must be non-empty")
        if identity != identity.strip():
            raise ValueError("provenance extractor identity cannot have edge whitespace")
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("provenance extractor identity does not match expected identity")
        if observed_identity is None:
            observed_identity = identity
        elif identity != observed_identity:
            raise ValueError("all provenance rows must use the same extractor identity")

    for diagnostic in frozen.diagnostics:
        if not isinstance(diagnostic.code, str) or not diagnostic.code.strip():
            raise ValueError("diagnostic code must be non-empty")
        if not isinstance(diagnostic.message, str) or not diagnostic.message.strip():
            raise ValueError("diagnostic message must be non-empty")
        if (diagnostic.start is None) != (diagnostic.end is None):
            raise ValueError("diagnostic span must provide both start and end")
        if diagnostic.start is None:
            continue
        assert diagnostic.end is not None
        if (
            isinstance(diagnostic.start, bool)
            or not isinstance(diagnostic.start, int)
            or isinstance(diagnostic.end, bool)
            or not isinstance(diagnostic.end, int)
        ):
            raise ValueError("diagnostic spans must use integer offsets")
        if not 0 <= diagnostic.start <= diagnostic.end <= evidence_length:
            raise ValueError("diagnostic span is outside evidence bounds")
    return frozen


@runtime_checkable
class ClaimExtractor(Protocol):
    """Extract only claims supported by an immutable evidence snapshot."""

    @property
    def identity(self) -> str:
        """Stable extractor and prompt/grammar identity."""

    def extract(self, evidence: EvidenceContext) -> ClaimExtraction:
        """Return supported ``ClaimInput`` values plus audit diagnostics."""


_CANONICAL_MARKER = "Canonical claim:"
_CANONICAL_MARKER_PATTERN = re.compile(r"(?<!\w)Canonical claim:\s*")
_FIELD_ORDER = {
    "subject": 0,
    "predicate": 1,
    "value": 2,
    "valid_from": 3,
    "valid_to": 4,
    "supersedes_source": 5,
    "confidence": 6,
    "authority": 7,
}
_REQUIRED_FIELDS = {"subject", "predicate", "value", "confidence", "authority"}
_CANONICAL_ATOM_PATTERN = re.compile(r"[^\s;=.]+")
_CANONICAL_CONFIDENCE_PATTERN = re.compile(r"(?:0|1)\.\d{3}")


def _valid_timestamp(value: str) -> bool:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class CanonicalClaimExtractor:
    """Parse only the explicit canonical grammar emitted by the benchmark.

    Natural-language sentences are intentionally ignored.  A malformed span or
    unresolved supersession reference emits a diagnostic and no claim for that
    span, ensuring this diagnostic extractor never fills gaps by inference.
    """

    @property
    def identity(self) -> str:
        return "evicortex/canonical-claim-grammar-v1"

    def extract(self, evidence: EvidenceContext) -> ClaimExtraction:
        if not isinstance(evidence, EvidenceContext):
            raise TypeError("evidence must be an EvidenceContext")

        matches = tuple(_CANONICAL_MARKER_PATTERN.finditer(evidence.content))
        if not matches:
            return validate_claim_extraction(
                evidence,
                ClaimExtraction(
                    diagnostics=(
                        ExtractionDiagnostic(
                            code="no_canonical_claims",
                            message=f"no exact {_CANONICAL_MARKER!r} marker was present",
                        ),
                    ),
                ),
                expected_identity=self.identity,
            )

        claims: list[ClaimInput] = []
        provenance: list[ClaimProvenance] = []
        diagnostics: list[ExtractionDiagnostic] = []
        for span_index, match in enumerate(matches):
            start = match.start()
            end = (
                matches[span_index + 1].start()
                if span_index + 1 < len(matches)
                else len(evidence.content)
            )
            while end > start and evidence.content[end - 1].isspace():
                end -= 1
            canonical_text = evidence.content[start:end]
            try:
                claim = self._parse_span(canonical_text, evidence)
            except ValueError as error:
                diagnostics.append(
                    ExtractionDiagnostic(
                        code="malformed_canonical_claim",
                        message=str(error),
                        start=start,
                        end=end,
                    )
                )
                continue
            claims.append(claim)
            provenance.append(
                ClaimProvenance(
                    claim_index=len(claims) - 1,
                    start=start,
                    end=end,
                    canonical_text=canonical_text,
                    content_sha256=evidence.content_sha256,
                    extractor_identity=self.identity,
                )
            )

        return validate_claim_extraction(
            evidence,
            ClaimExtraction(tuple(claims), tuple(provenance), tuple(diagnostics)),
            expected_identity=self.identity,
        )

    @staticmethod
    def _parse_span(canonical_text: str, evidence: EvidenceContext) -> ClaimInput:
        if not canonical_text.startswith(_CANONICAL_MARKER):
            raise ValueError("canonical marker is missing")
        if not canonical_text.endswith("."):
            raise ValueError("canonical claim must end with a period")
        body = canonical_text[len(_CANONICAL_MARKER) : -1].strip()
        if not body or "\n" in body or "\r" in body or "\x00" in body:
            raise ValueError("canonical claim body must be one non-empty line")

        fields: dict[str, str] = {}
        field_order: list[int] = []
        for encoded_field in body.split(";"):
            key, separator, value = encoded_field.strip().partition("=")
            if not separator or not key or not value:
                raise ValueError("every canonical field must use non-empty key=value syntax")
            if key not in _FIELD_ORDER:
                raise ValueError(f"unsupported canonical field {key!r}")
            if key in fields:
                raise ValueError(f"duplicate canonical field {key!r}")
            if value != value.strip():
                raise ValueError(f"canonical field {key!r} has ambiguous whitespace")
            fields[key] = value
            field_order.append(_FIELD_ORDER[key])

        missing = sorted(_REQUIRED_FIELDS - fields.keys())
        if missing:
            raise ValueError(f"missing required canonical fields: {', '.join(missing)}")
        if field_order != sorted(field_order):
            raise ValueError("canonical fields are not in the defined grammar order")
        if field_order[:3] != [0, 1, 2]:
            raise ValueError("canonical claim must begin with subject, predicate, value")
        if field_order[-2:] != [6, 7]:
            raise ValueError("canonical claim must end with confidence, authority")

        for key in ("subject", "predicate", "value", "authority"):
            if not fields[key].strip():
                raise ValueError(f"canonical field {key!r} cannot be blank")
        for key in ("subject", "predicate", "authority"):
            if _CANONICAL_ATOM_PATTERN.fullmatch(fields[key]) is None:
                raise ValueError(
                    f"canonical field {key!r} must be one delimiter-free token"
                )

        if _CANONICAL_CONFIDENCE_PATTERN.fullmatch(fields["confidence"]) is None:
            raise ValueError("confidence must use the canonical three-decimal format")
        try:
            confidence = float(fields["confidence"])
        except ValueError as error:
            raise ValueError("confidence must be numeric") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")

        for key in ("valid_from", "valid_to"):
            if key in fields and not _valid_timestamp(fields[key]):
                raise ValueError(f"{key} must be an ISO-8601 timestamp with a timezone")
        if "valid_from" in fields and "valid_to" in fields:
            start = datetime.fromisoformat(fields["valid_from"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(fields["valid_to"].replace("Z", "+00:00"))
            if end <= start:
                raise ValueError("valid_to must be later than valid_from")

        supersedes = None
        if "supersedes_source" in fields:
            supersedes = evidence.claim_ids_by_source.get(fields["supersedes_source"])
            if supersedes is None:
                raise ValueError(
                    "supersedes_source has no explicit claim-ID resolution in evidence context"
                )

        return ClaimInput(
            subject=fields["subject"],
            predicate=fields["predicate"],
            value=fields["value"],
            valid_from=fields.get("valid_from"),
            valid_to=fields.get("valid_to"),
            confidence=confidence,
            authority=fields["authority"],
            supersedes=supersedes,
        )


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """Provider-neutral text candidate supplied to an optional reranker."""

    candidate_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RerankResult:
    """A finite provider score for one known candidate."""

    candidate_id: str
    score: float


@runtime_checkable
class Reranker(Protocol):
    """Optional second-stage ranker; higher returned scores rank first."""

    @property
    def identity(self) -> str:
        """Stable model/configuration identity used in retrieval traces."""

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> Sequence[RerankResult]:
        """Score any subset of known candidates without mutating evidence."""


def validate_rerank_results(
    candidates: Sequence[RerankCandidate],
    results: Sequence[RerankResult],
    *,
    require_complete: bool = False,
) -> tuple[RerankResult, ...]:
    """Reject unknown, duplicate, non-finite, or optionally partial output.

    ``require_complete`` defaults to ``False`` for provider adapters that use the
    protocol as a general scoring primitive. Retrieval pipelines that depend on a
    total ordering should enable it so a provider cannot silently drop evidence.
    """

    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be a boolean")

    known_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, RerankCandidate):
            raise TypeError("candidates must contain RerankCandidate values")
        if not isinstance(candidate.candidate_id, str) or not candidate.candidate_id:
            raise ValueError("candidate IDs must be non-empty strings")
        if not isinstance(candidate.text, str):
            raise TypeError("rerank candidate text must be a string")
        known_ids.add(candidate.candidate_id)
    if len(known_ids) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    seen: set[str] = set()
    frozen: list[RerankResult] = []
    for result in results:
        if not isinstance(result, RerankResult):
            raise TypeError("results must contain RerankResult values")
        if result.candidate_id not in known_ids:
            raise ValueError(f"reranker returned unknown candidate {result.candidate_id!r}")
        if result.candidate_id in seen:
            raise ValueError(f"reranker returned duplicate candidate {result.candidate_id!r}")
        if isinstance(result.score, bool):
            raise ValueError(
                f"reranker returned a boolean score for {result.candidate_id!r}"
            )
        try:
            score = float(result.score)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"reranker returned a non-numeric score for {result.candidate_id!r}"
            ) from error
        if not math.isfinite(score):
            raise ValueError(
                f"reranker returned a non-finite score for {result.candidate_id!r}"
            )
        seen.add(result.candidate_id)
        frozen.append(RerankResult(result.candidate_id, score))
    if require_complete and seen != known_ids:
        missing = ", ".join(repr(item) for item in sorted(known_ids - seen))
        raise ValueError(f"reranker omitted candidates: {missing}")
    return tuple(frozen)


def snapshot_reranker_identity(reranker: Reranker) -> str:
    """Validate and copy a reranker's stable trace identity."""

    if not isinstance(reranker, Reranker) or not callable(
        getattr(reranker, "rerank", None)
    ):
        raise TypeError("reranker does not implement the Reranker protocol")
    identity = reranker.identity
    if not isinstance(identity, str):
        raise TypeError("reranker identity must be a string")
    if not identity or identity != identity.strip():
        raise ValueError(
            "reranker identity must be non-empty without edge whitespace"
        )
    if not identity.isprintable():
        raise ValueError("reranker identity must contain only printable characters")
    if len(identity.encode("utf-8")) > 512:
        raise ValueError("reranker identity exceeds 512 UTF-8 bytes")
    return identity


__all__ = [
    "CanonicalClaimExtractor",
    "ClaimExtraction",
    "ClaimExtractor",
    "ClaimProvenance",
    "Embedder",
    "EmbeddingValidationError",
    "EmbeddingVector",
    "EvidenceContext",
    "ExtractionDiagnostic",
    "HashingEmbedder",
    "RoleAwareEmbedder",
    "RerankCandidate",
    "Reranker",
    "RerankResult",
    "snapshot_reranker_identity",
    "embed_document_batch",
    "embed_query_text",
    "validate_claim_extraction",
    "validate_embedding_batch",
    "validate_rerank_results",
]
