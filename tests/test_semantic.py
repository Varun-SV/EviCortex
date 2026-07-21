"""Tests for storage-independent semantic projection primitives."""

from __future__ import annotations

import hashlib
import math
import struct
import unittest

from evicortex.models import ClaimRecord, MemoryRecord
from evicortex.plugins import EmbeddingValidationError, HashingEmbedder
from evicortex.semantic import (
    EmbeddingInput,
    EmbeddingProfile,
    MAX_EMBEDDER_IDENTITY_BYTES,
    claim_embedding_input,
    cosine_similarity,
    decode_float32_vector,
    encode_float32_vector,
    memory_embedding_input,
)


class _InvalidEmbedder:
    identity = "invalid"
    dimensions = 2
    normalized = True


class EmbeddingProfileTests(unittest.TestCase):
    def test_profile_is_a_deterministic_validated_embedder_snapshot(self) -> None:
        embedder = HashingEmbedder(dimensions=64)
        first = EmbeddingProfile.from_embedder(embedder)
        second = EmbeddingProfile.from_embedder(embedder)

        self.assertEqual(first, second)
        self.assertEqual(first.identity, embedder.identity)
        self.assertEqual(first.dimensions, 64)
        self.assertTrue(first.normalized)
        self.assertEqual(first.digest, first.profile_sha256)
        self.assertEqual(
            first.profile_sha256,
            "a80f41d0f1dc88003390da2f92e9487ef5e2f6d3bee31125514db67af7c761b8",
        )
        self.assertTrue(first.is_compatible(embedder))
        self.assertFalse(first.is_compatible(HashingEmbedder(dimensions=32)))

        changed_identity = EmbeddingProfile("provider/model-v2", 64, True)
        changed_dimensions = EmbeddingProfile(first.identity, 32, True)
        changed_normalization = EmbeddingProfile(first.identity, 64, False)
        self.assertEqual(len({
            first.digest,
            changed_identity.digest,
            changed_dimensions.digest,
            changed_normalization.digest,
        }), 4)

    def test_profile_rejects_ambiguous_or_invalid_compatibility_fields(self) -> None:
        invalid_identity = (
            "",
            " leading",
            "trailing ",
            "line\nbreak",
            "x" * (MAX_EMBEDDER_IDENTITY_BYTES + 1),
        )
        for identity in invalid_identity:
            with self.subTest(identity=identity[:20]):
                with self.assertRaises(ValueError):
                    EmbeddingProfile(identity, 4, True)
        with self.assertRaises(TypeError):
            EmbeddingProfile(17, 4, True)  # type: ignore[arg-type]
        for dimensions in (0, -1):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    EmbeddingProfile("provider/model", dimensions, True)
        for dimensions in (True, 2.5):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(TypeError):
                    EmbeddingProfile("provider/model", dimensions, True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            EmbeddingProfile("provider/model", 4, 1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "Embedder protocol"):
            EmbeddingProfile.from_embedder(_InvalidEmbedder())  # type: ignore[arg-type]

    def test_identity_bound_is_measured_in_utf8_bytes(self) -> None:
        multi_byte = "é" * (MAX_EMBEDDER_IDENTITY_BYTES // 2)
        self.assertEqual(len(multi_byte.encode("utf-8")), MAX_EMBEDDER_IDENTITY_BYTES)
        self.assertEqual(EmbeddingProfile(multi_byte, 1, False).identity, multi_byte)
        with self.assertRaises(ValueError):
            EmbeddingProfile(multi_byte + "é", 1, False)


class Float32CodecTests(unittest.TestCase):
    def test_encoding_is_little_endian_and_round_trips(self) -> None:
        vector = (1.0, -2.5, 0.0)
        payload = encode_float32_vector(vector, dimensions=3)

        self.assertEqual(payload.hex(), "0000803f000020c000000000")
        self.assertEqual(decode_float32_vector(payload, dimensions=3), vector)
        self.assertEqual(
            decode_float32_vector(bytearray(payload), dimensions=3), vector
        )
        self.assertEqual(
            decode_float32_vector(memoryview(payload), dimensions=3), vector
        )

    def test_float32_quantization_is_explicit_and_deterministic(self) -> None:
        vector = (math.pi, 1.0 / 3.0)
        decoded = decode_float32_vector(
            encode_float32_vector(vector, dimensions=2), dimensions=2
        )
        self.assertEqual(decoded, struct.unpack("<2f", struct.pack("<2f", *vector)))
        self.assertNotEqual(decoded, vector)

    def test_codec_enforces_dimensions_finiteness_normalization_and_zero_policy(self) -> None:
        invalid_encodes = (
            lambda: encode_float32_vector((1.0,), dimensions=2),
            lambda: encode_float32_vector((float("nan"),), dimensions=1),
            lambda: encode_float32_vector((float("inf"),), dimensions=1),
            lambda: encode_float32_vector((1e300,), dimensions=1),
            lambda: encode_float32_vector(
                (2.0, 0.0), dimensions=2, normalized=True
            ),
            lambda: encode_float32_vector(
                (0.0, 0.0), dimensions=2, allow_zero=False
            ),
        )
        for encode in invalid_encodes:
            with self.subTest(encode=encode):
                with self.assertRaises(EmbeddingValidationError):
                    encode()

        unit = (1.0 / math.sqrt(2.0),) * 2
        unit_payload = encode_float32_vector(unit, dimensions=2, normalized=True)
        decoded_unit = decode_float32_vector(
            unit_payload, dimensions=2, normalized=True
        )
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in decoded_unit)), 1.0, places=6)

        with self.assertRaises(EmbeddingValidationError):
            decode_float32_vector(b"\x00" * 4, dimensions=2)
        with self.assertRaises(EmbeddingValidationError):
            decode_float32_vector(
                struct.pack("<f", float("nan")), dimensions=1
            )
        with self.assertRaises(EmbeddingValidationError):
            decode_float32_vector(
                struct.pack("<f", float("inf")), dimensions=1
            )
        with self.assertRaises(TypeError):
            decode_float32_vector("not bytes", dimensions=1)  # type: ignore[arg-type]

    def test_zero_vector_round_trip_is_an_explicit_default(self) -> None:
        zero = (0.0, 0.0, 0.0)
        payload = encode_float32_vector(zero, dimensions=3, normalized=True)
        self.assertEqual(
            decode_float32_vector(payload, dimensions=3, normalized=True), zero
        )
        with self.assertRaises(EmbeddingValidationError):
            decode_float32_vector(payload, dimensions=3, allow_zero=False)


class CosineSimilarityTests(unittest.TestCase):
    def test_general_and_normalized_cosine_scores(self) -> None:
        self.assertEqual(cosine_similarity((1.0, 0.0), (0.0, 1.0)), 0.0)
        self.assertEqual(cosine_similarity((2.0, 0.0), (5.0, 0.0)), 1.0)
        self.assertEqual(cosine_similarity((2.0, 0.0), (-5.0, 0.0)), -1.0)
        self.assertAlmostEqual(
            cosine_similarity((3.0, 4.0), (4.0, 3.0)), 24.0 / 25.0
        )
        self.assertEqual(
            cosine_similarity((1.0, 0.0), (0.0, -1.0), normalized=True),
            0.0,
        )

    def test_zero_vector_has_defined_neutral_score(self) -> None:
        self.assertEqual(cosine_similarity((0.0, 0.0), (1.0, 0.0)), 0.0)
        self.assertEqual(
            cosine_similarity((0.0, 0.0), (0.0, 0.0), normalized=True), 0.0
        )

    def test_huge_finite_coordinates_are_scored_without_overflow(self) -> None:
        huge = 1e308
        self.assertAlmostEqual(
            cosine_similarity((huge, huge), (huge, huge)),
            1.0,
        )
        self.assertAlmostEqual(
            cosine_similarity((huge, huge), (-huge, -huge)),
            -1.0,
        )
        self.assertEqual(
            cosine_similarity((huge, huge), (huge, -huge)),
            0.0,
        )

    def test_invalid_vectors_are_rejected(self) -> None:
        invalid = (
            lambda: cosine_similarity((), ()),
            lambda: cosine_similarity((1.0,), (1.0, 0.0)),
            lambda: cosine_similarity((float("nan"),), (1.0,)),
            lambda: cosine_similarity((2.0, 0.0), (1.0, 0.0), normalized=True),
        )
        for score in invalid:
            with self.subTest(score=score):
                with self.assertRaises((ValueError, EmbeddingValidationError)):
                    score()


def _memory(
    *,
    memory_id: str = "memory-1",
    namespace: str = "tenant-a/user",
    content: str = "The Quartz device has serial number QZ-4401.",
    kind: str = "episode",
    source: str = "session/1",
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        namespace=namespace,
        kind=kind,
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        source=source,
        occurred_at="2026-01-01T00:00:00.000000Z",
        recorded_at="2026-01-01T00:00:01.000000Z",
        metadata={"test": True},
        idempotency_key=None,
    )


def _claim(
    *,
    claim_id: str = "claim-1",
    namespace: str = "tenant-a/user",
    memory_id: str = "memory-1",
    status: str = "active",
    confidence: float = 1.0,
) -> ClaimRecord:
    return ClaimRecord(
        id=claim_id,
        namespace=namespace,
        memory_id=memory_id,
        subject="device:quartz",
        predicate="serial_number",
        value="QZ-4401",
        valid_from="2026-01-01T00:00:00.000000Z",
        valid_to=None,
        recorded_at="2026-01-01T00:00:01.000000Z",
        confidence=confidence,
        authority="registry",
        supersedes=None,
        status=status,
    )


class CanonicalEmbeddingInputTests(unittest.TestCase):
    def test_memory_input_is_canonical_and_hashes_exact_utf8_text(self) -> None:
        value = memory_embedding_input(_memory())

        self.assertEqual(value.target_kind, "memory")
        self.assertEqual(value.target_id, "memory-1")
        self.assertEqual(
            value.text,
            '{"content":"The Quartz device has serial number QZ-4401.",'
            '"kind":"episode","schema":"evicortex.embedding-input.v1",'
            '"target":"memory"}',
        )
        self.assertEqual(
            value.input_sha256,
            "7120075532265de8254d9c8fcf82e984714317759d2af2850cb36df1659a9ed3",
        )

    def test_nonsemantic_memory_metadata_does_not_invalidate_projection_input(self) -> None:
        first = memory_embedding_input(_memory())
        second = memory_embedding_input(
            _memory(
                memory_id="memory-2",
                namespace="another/tenant",
                source="another/source",
            )
        )
        changed = memory_embedding_input(_memory(content="Different evidence."))

        self.assertEqual(first.text, second.text)
        self.assertEqual(first.input_sha256, second.input_sha256)
        self.assertNotEqual(first.target_id, second.target_id)
        self.assertNotEqual(first.input_sha256, changed.input_sha256)

    def test_claim_input_excludes_temporal_projection_and_evidence_metadata(self) -> None:
        active = claim_embedding_input(_claim())
        changed_projection = claim_embedding_input(
            _claim(
                claim_id="claim-2",
                namespace="another/tenant",
                memory_id="memory-9",
                status="superseded",
                confidence=0.4,
            )
        )

        self.assertEqual(active.target_kind, "claim")
        self.assertEqual(
            active.text,
            '{"predicate":"serial_number","schema":"evicortex.embedding-input.v1",'
            '"subject":"device:quartz","target":"claim","value":"QZ-4401"}',
        )
        self.assertEqual(active.text, changed_projection.text)
        self.assertEqual(active.input_sha256, changed_projection.input_sha256)
        self.assertEqual(
            active.input_sha256,
            "c7d93dfae4a14744d3aed83623d6af52144762940ae834b1f37af1aaf8a1928c",
        )
        self.assertNotIn("active", active.text)
        self.assertNotIn("registry", active.text)

    def test_embedding_input_rejects_mismatched_digest_or_unknown_target(self) -> None:
        text = '{"schema":"evicortex.embedding-input.v1"}'
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(
            EmbeddingInput("memory", "m1", text, digest).digest,
            digest,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            EmbeddingInput("memory", "m1", text, "0" * 64)
        with self.assertRaisesRegex(ValueError, "target_kind"):
            EmbeddingInput("episode", "m1", text, digest)

    def test_helpers_require_public_record_types(self) -> None:
        with self.assertRaises(TypeError):
            memory_embedding_input(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            claim_embedding_input(object())  # type: ignore[arg-type]

        corrupted = _memory()
        object.__setattr__(corrupted, "content_hash", "0" * 64)
        with self.assertRaisesRegex(ValueError, "immutable content"):
            memory_embedding_input(corrupted)


if __name__ == "__main__":
    unittest.main()
