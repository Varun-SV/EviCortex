"""Portable semantic-projection primitives for EviCortex Phase B.

This module owns no database or retrieval behavior.  It defines deterministic
profiles and input formats so a later projection can detect incompatible model
configuration, rebuild safely, and exchange vectors across platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
from typing import Sequence

from .models import ClaimRecord, MemoryRecord
from .plugins import Embedder, EmbeddingValidationError, validate_embedding_batch


MAX_EMBEDDER_IDENTITY_BYTES = 512
_PROFILE_SCHEMA = "evicortex.embedding-profile.v1"
_INPUT_SCHEMA = "evicortex.embedding-input.v1"


def _canonical_json(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_dimensions(dimensions: int) -> int:
    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise TypeError("dimensions must be an integer")
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")
    return dimensions


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Immutable compatibility identity for one embedder configuration.

    The digest covers the provider's stable identity, output dimensionality, and
    normalization contract.  It identifies index compatibility, not model
    quality or semantic capability.
    """

    identity: str
    dimensions: int
    normalized: bool
    profile_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str):
            raise TypeError("embedder identity must be a string")
        if not self.identity or self.identity != self.identity.strip():
            raise ValueError("embedder identity must be non-empty without edge whitespace")
        if not self.identity.isprintable():
            raise ValueError("embedder identity must contain only printable characters")
        if len(self.identity.encode("utf-8")) > MAX_EMBEDDER_IDENTITY_BYTES:
            raise ValueError(
                "embedder identity exceeds "
                f"{MAX_EMBEDDER_IDENTITY_BYTES} UTF-8 bytes"
            )
        _validate_dimensions(self.dimensions)
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be a boolean")

        canonical = _canonical_json(
            {
                "dimensions": self.dimensions,
                "identity": self.identity,
                "normalized": self.normalized,
                "schema": _PROFILE_SCHEMA,
            }
        )
        object.__setattr__(self, "profile_sha256", _sha256_text(canonical))

    @classmethod
    def from_embedder(cls, embedder: Embedder) -> EmbeddingProfile:
        """Snapshot and validate the compatibility fields of an embedder."""

        if not isinstance(embedder, Embedder):
            raise TypeError("embedder does not implement the Embedder protocol")
        return cls(
            identity=embedder.identity,
            dimensions=embedder.dimensions,
            normalized=embedder.normalized,
        )

    @property
    def digest(self) -> str:
        """Short alias for the deterministic profile SHA-256 digest."""

        return self.profile_sha256

    def is_compatible(self, embedder: Embedder) -> bool:
        """Return whether a current provider snapshot matches this profile."""

        try:
            current = type(self).from_embedder(embedder)
        except (TypeError, ValueError):
            return False
        return current.profile_sha256 == self.profile_sha256


def encode_float32_vector(
    vector: Sequence[float],
    *,
    dimensions: int,
    normalized: bool = False,
    allow_zero: bool = True,
) -> bytes:
    """Encode a validated vector as portable little-endian IEEE-754 float32."""

    _validate_dimensions(dimensions)
    frozen = validate_embedding_batch(
        (vector,),
        expected_count=1,
        dimensions=dimensions,
        normalized=normalized,
        allow_zero=allow_zero,
    )[0]
    try:
        payload = struct.pack(f"<{dimensions}f", *frozen)
    except (OverflowError, struct.error) as error:
        raise EmbeddingValidationError(
            "vector contains a value outside the finite float32 range"
        ) from error

    # A finite Python float can overflow or round during float32 conversion.
    # Revalidation prevents a non-finite representation from being persisted.
    decoded = struct.unpack(f"<{dimensions}f", payload)
    validate_embedding_batch(
        (decoded,),
        expected_count=1,
        dimensions=dimensions,
        normalized=normalized,
        allow_zero=allow_zero,
        # Float32 quantization can perturb a unit norm by a few ULPs.
        tolerance=1e-5,
    )
    return payload


def decode_float32_vector(
    payload: bytes | bytearray | memoryview,
    *,
    dimensions: int,
    normalized: bool = False,
    allow_zero: bool = True,
) -> tuple[float, ...]:
    """Decode and validate a little-endian IEEE-754 float32 vector."""

    _validate_dimensions(dimensions)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    raw = bytes(payload)
    expected_bytes = dimensions * 4
    if len(raw) != expected_bytes:
        raise EmbeddingValidationError(
            f"float32 payload is {len(raw)} bytes; expected {expected_bytes}"
        )
    decoded = struct.unpack(f"<{dimensions}f", raw)
    return validate_embedding_batch(
        (decoded,),
        expected_count=1,
        dimensions=dimensions,
        normalized=normalized,
        allow_zero=allow_zero,
        tolerance=1e-5,
    )[0]


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
    *,
    normalized: bool = False,
) -> float:
    """Return cosine similarity, defining any zero-vector comparison as 0.0."""

    dimensions = len(left)
    _validate_dimensions(dimensions)
    left_vector, right_vector = validate_embedding_batch(
        (left, right),
        expected_count=2,
        dimensions=dimensions,
        normalized=normalized,
        allow_zero=True,
        tolerance=1e-5,
    )
    left_scale = max(abs(value) for value in left_vector)
    right_scale = max(abs(value) for value in right_vector)
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    # Scale each side independently before products or norms.  This preserves
    # cosine while avoiding overflow for otherwise valid finite coordinates.
    left_scaled = tuple(value / left_scale for value in left_vector)
    right_scaled = tuple(value / right_scale for value in right_vector)
    dot = math.fsum(
        a * b for a, b in zip(left_scaled, right_scaled, strict=True)
    )
    left_norm = math.hypot(*left_scaled)
    right_norm = math.hypot(*right_scaled)
    score = dot / (left_norm * right_norm)
    # Floating-point roundoff can otherwise leak tiny values outside cosine's
    # mathematical range.
    return max(-1.0, min(1.0, score))


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """Canonical model input and exact digest for one derived projection row."""

    target_kind: str
    target_id: str
    text: str
    input_sha256: str

    def __post_init__(self) -> None:
        if self.target_kind not in {"memory", "claim"}:
            raise ValueError("target_kind must be 'memory' or 'claim'")
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("target_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("text must be a non-empty string")
        if not isinstance(self.input_sha256, str):
            raise TypeError("input_sha256 must be a string")
        if self.input_sha256 != _sha256_text(self.text):
            raise ValueError("input_sha256 does not match canonical input text")

    @property
    def digest(self) -> str:
        return self.input_sha256


def memory_embedding_input(memory: MemoryRecord) -> EmbeddingInput:
    """Build canonical text from immutable, semantically relevant memory fields.

    Namespace, record ID, timestamps, source, and metadata remain projection
    metadata rather than model input.  This permits safe reuse when the same
    immutable content and kind occurs in different namespaces.
    """

    if not isinstance(memory, MemoryRecord):
        raise TypeError("memory must be a MemoryRecord")
    content_sha256 = _sha256_text(memory.content)
    if memory.content_hash != content_sha256:
        raise ValueError("memory content_hash does not match immutable content")
    text = _canonical_json(
        {
            "content": memory.content,
            "kind": memory.kind,
            "schema": _INPUT_SCHEMA,
            "target": "memory",
        }
    )
    return EmbeddingInput("memory", memory.id, text, _sha256_text(text))


def claim_embedding_input(claim: ClaimRecord) -> EmbeddingInput:
    """Build canonical text from the subject-predicate-value assertion only.

    Temporal validity, status, confidence, authority, and evidence IDs must be
    enforced by other retrieval channels; excluding them keeps the vector stable
    when derived temporal status changes.
    """

    if not isinstance(claim, ClaimRecord):
        raise TypeError("claim must be a ClaimRecord")
    text = _canonical_json(
        {
            "predicate": claim.predicate,
            "schema": _INPUT_SCHEMA,
            "subject": claim.subject,
            "target": "claim",
            "value": claim.value,
        }
    )
    return EmbeddingInput("claim", claim.id, text, _sha256_text(text))


__all__ = [
    "EmbeddingInput",
    "EmbeddingProfile",
    "MAX_EMBEDDER_IDENTITY_BYTES",
    "claim_embedding_input",
    "cosine_similarity",
    "decode_float32_vector",
    "encode_float32_vector",
    "memory_embedding_input",
]
