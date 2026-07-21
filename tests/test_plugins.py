"""Contract tests for optional provider-neutral Phase B components."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence
import unittest

from evicortex.models import ClaimInput
from evicortex.plugins import (
    CanonicalClaimExtractor,
    ClaimExtraction,
    ClaimExtractor,
    Embedder,
    EmbeddingValidationError,
    EvidenceContext,
    ExtractionDiagnostic,
    HashingEmbedder,
    RoleAwareEmbedder,
    RerankCandidate,
    Reranker,
    RerankResult,
    embed_document_batch,
    embed_query_text,
    validate_claim_extraction,
    validate_embedding_batch,
    validate_rerank_results,
)


class HashingEmbedderTests(unittest.TestCase):
    def test_runtime_contract_identity_determinism_and_batch_order(self) -> None:
        embedder = HashingEmbedder(dimensions=64)

        self.assertIsInstance(embedder, Embedder)
        self.assertEqual(
            embedder.identity,
            "evicortex/diagnostic-feature-hashing-blake2b-v1/64",
        )
        self.assertFalse(embedder.is_semantic)
        self.assertTrue(embedder.diagnostic_only)
        self.assertTrue(embedder.normalized)
        self.assertIsInstance(embedder, RoleAwareEmbedder)

        texts = ("A quiet Quartz device", "A different sentence", "")
        first = embedder.embed_batch(texts)
        second = embedder.embed_batch(texts)
        reversed_batch = embedder.embed_batch(tuple(reversed(texts)))

        self.assertEqual(first, second)
        self.assertEqual(first, tuple(reversed(reversed_batch)))
        self.assertEqual(len(first), len(texts))
        self.assertTrue(all(len(vector) == 64 for vector in first))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first[0])), 1.0)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first[1])), 1.0)
        self.assertEqual(first[2], (0.0,) * 64)
        self.assertEqual(tuple(embed_document_batch(embedder, texts)), first)
        self.assertEqual(tuple(embed_query_text(embedder, texts[0])), first[0])

    def test_role_aware_dispatch_keeps_query_and_document_rules_separate(self) -> None:
        class AsymmetricEmbedder:
            identity = "tests/asymmetric-v1"
            dimensions = 2
            normalized = True

            def embed_batch(
                self, texts: Sequence[str]
            ) -> tuple[tuple[float, float], ...]:
                return tuple((0.0, 1.0) for _ in texts)

            def embed_documents(
                self, texts: Sequence[str]
            ) -> tuple[tuple[float, float], ...]:
                return tuple((1.0, 0.0) for _ in texts)

            def embed_query(self, text: str) -> tuple[float, float]:
                if not isinstance(text, str):
                    raise TypeError
                return (0.0, 1.0)

        embedder = AsymmetricEmbedder()
        self.assertIsInstance(embedder, Embedder)
        self.assertIsInstance(embedder, RoleAwareEmbedder)
        self.assertEqual(
            tuple(embed_document_batch(embedder, ("one", "two"))),
            ((1.0, 0.0), (1.0, 0.0)),
        )
        self.assertEqual(tuple(embed_query_text(embedder, "query")), (0.0, 1.0))

        class PartialEmbedder:
            identity = "tests/partial-role-aware-v1"
            dimensions = 2
            normalized = True

            def embed_batch(
                self, texts: Sequence[str]
            ) -> tuple[tuple[float, float], ...]:
                return tuple((1.0, 0.0) for _ in texts)

            def embed_query(self, text: str) -> tuple[float, float]:
                return (1.0, 0.0)

        partial = PartialEmbedder()
        with self.assertRaisesRegex(TypeError, "both"):
            embed_document_batch(partial, ("document",))
        with self.assertRaisesRegex(TypeError, "both"):
            embed_query_text(partial, "query")

    def test_casefolding_and_repeat_token_weighting_are_stable(self) -> None:
        embedder = HashingEmbedder(dimensions=128)
        lower, upper, repeated = embedder.embed_batch(
            ("quartz device", "QUARTZ DEVICE", "quartz quartz device")
        )
        self.assertEqual(lower, upper)
        self.assertNotEqual(lower, repeated)

    def test_input_and_dimension_validation(self) -> None:
        for dimensions in (0, -1):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    HashingEmbedder(dimensions=dimensions)
        for dimensions in (True, 2.5, "64"):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(TypeError):
                    HashingEmbedder(dimensions=dimensions)  # type: ignore[arg-type]

        embedder = HashingEmbedder(dimensions=8)
        with self.assertRaises(TypeError):
            embedder.embed_batch("one input")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "text 1"):
            embedder.embed_batch(("valid", 7))  # type: ignore[arg-type]

    def test_shared_validator_rejects_provider_contract_drift(self) -> None:
        self.assertEqual(
            validate_embedding_batch(
                ((1.0, 0.0),),
                expected_count=1,
                dimensions=2,
                normalized=True,
            ),
            ((1.0, 0.0),),
        )

        invalid_calls = (
            lambda: validate_embedding_batch(
                (), expected_count=1, dimensions=2
            ),
            lambda: validate_embedding_batch(
                ((1.0,),), expected_count=1, dimensions=2
            ),
            lambda: validate_embedding_batch(
                ((float("nan"), 0.0),), expected_count=1, dimensions=2
            ),
            lambda: validate_embedding_batch(
                ((float("inf"), 0.0),), expected_count=1, dimensions=2
            ),
            lambda: validate_embedding_batch(
                ((True, 0.0),), expected_count=1, dimensions=2
            ),
            lambda: validate_embedding_batch(
                ((2.0, 0.0),),
                expected_count=1,
                dimensions=2,
                normalized=True,
            ),
            lambda: validate_embedding_batch(
                ((0.0, 0.0),), expected_count=1, dimensions=2
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(EmbeddingValidationError):
                    call()

        for encoded_vector in ("10", b"10", bytearray(b"10"), memoryview(b"10")):
            with self.subTest(encoded_vector=type(encoded_vector).__name__):
                with self.assertRaisesRegex(
                    EmbeddingValidationError, "numeric coordinate sequence"
                ):
                    validate_embedding_batch(
                        (encoded_vector,),  # type: ignore[arg-type]
                        expected_count=1,
                        dimensions=2,
                        allow_zero=True,
                    )

        huge = validate_embedding_batch(
            ((1e308, 1e308),),
            expected_count=1,
            dimensions=2,
            allow_zero=False,
        )
        self.assertEqual(huge, ((1e308, 1e308),))

        self.assertEqual(
            validate_embedding_batch(
                ((0.0, 0.0),),
                expected_count=1,
                dimensions=2,
                normalized=True,
                allow_zero=True,
            ),
            ((0.0, 0.0),),
        )


class CanonicalClaimExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = CanonicalClaimExtractor()

    @staticmethod
    def context(
        content: str,
        *,
        claim_ids_by_source: dict[str, str] | None = None,
    ) -> EvidenceContext:
        return EvidenceContext.from_content(
            namespace="tenant/user",
            content=content,
            source="session/7",
            occurred_at="2026-01-01T00:00:00Z",
            metadata={"origin": "test"},
            claim_ids_by_source=claim_ids_by_source,
        )

    def test_runtime_contract_and_exact_benchmark_grammar(self) -> None:
        self.assertIsInstance(self.extractor, ClaimExtractor)
        content = (
            "Ada selected the dark theme. "
            "Canonical claim: subject=user:ada; predicate=preferred_theme; "
            "value=dark; valid_from=2026-02-01T09:00:00Z; "
            "confidence=1.000; authority=application."
        )
        context = self.context(content)

        result = self.extractor.extract(context)

        self.assertEqual(
            result.claims,
            (
                ClaimInput(
                    subject="user:ada",
                    predicate="preferred_theme",
                    value="dark",
                    valid_from="2026-02-01T09:00:00Z",
                    confidence=1.0,
                    authority="application",
                ),
            ),
        )
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(len(result.provenance), 1)
        provenance = result.provenance[0]
        self.assertEqual(provenance.claim_index, 0)
        self.assertEqual(provenance.content_sha256, context.content_sha256)
        self.assertEqual(provenance.extractor_identity, self.extractor.identity)
        self.assertEqual(
            content[provenance.start : provenance.end],
            provenance.canonical_text,
        )

    def test_multiple_claims_preserve_order_and_exact_provenance(self) -> None:
        content = (
            "Evidence preface. "
            "Canonical claim: subject=project:atlas; predicate=started_on; "
            "value=2026-01-10; confidence=0.750; authority=calendar. "
            "Canonical claim: subject=project:atlas; predicate=released_on; "
            "value=2026-03-05; valid_to=2027-01-01T00:00:00+00:00; "
            "confidence=1.000; authority=release-system."
        )

        result = self.extractor.extract(self.context(content))

        self.assertEqual(
            [(claim.predicate, claim.value) for claim in result.claims],
            [("started_on", "2026-01-10"), ("released_on", "2026-03-05")],
        )
        self.assertEqual([item.claim_index for item in result.provenance], [0, 1])
        self.assertEqual(result.diagnostics, ())

    def test_supersedes_source_requires_explicit_snapshot_resolution(self) -> None:
        old_source = "synthetic/00000007/theme_old"
        content = (
            "Canonical claim: subject=user:ada; predicate=preferred_theme; "
            "value=dark; valid_from=2026-02-01T09:00:00Z; "
            f"supersedes_source={old_source}; confidence=1.000; "
            "authority=application."
        )

        unresolved = self.extractor.extract(self.context(content))
        resolved = self.extractor.extract(
            self.context(content, claim_ids_by_source={old_source: "claim-old-123"})
        )

        self.assertTrue(unresolved.abstained)
        self.assertEqual(unresolved.diagnostics[0].code, "malformed_canonical_claim")
        self.assertIn("no explicit claim-ID resolution", unresolved.diagnostics[0].message)
        self.assertEqual(resolved.claims[0].supersedes, "claim-old-123")

    def test_natural_language_and_lookalike_markers_abstain(self) -> None:
        inputs = (
            "Ada definitely prefers dark mode.",
            "canonical claim: subject=user:ada; predicate=theme; value=dark; "
            "confidence=1.000; authority=application.",
            "Canonical claims: subject=user:ada; predicate=theme; value=dark; "
            "confidence=1.000; authority=application.",
        )
        for content in inputs:
            with self.subTest(content=content):
                result = self.extractor.extract(self.context(content))
                self.assertTrue(result.abstained)
                self.assertEqual(result.provenance, ())
                self.assertEqual(result.diagnostics[0].code, "no_canonical_claims")

    def test_malformed_spans_abstain_without_partial_claims(self) -> None:
        bodies = (
            # Missing required authority.
            "subject=user:ada; predicate=theme; value=dark; confidence=1.000",
            # Unknown fields cannot be smuggled into the grammar.
            "subject=user:ada; predicate=theme; value=dark; invented=yes; "
            "confidence=1.000; authority=application",
            # Duplicate and out-of-order fields are ambiguous.
            "subject=user:ada; predicate=theme; value=dark; value=light; "
            "confidence=1.000; authority=application",
            "predicate=theme; subject=user:ada; value=dark; confidence=1.000; "
            "authority=application",
            # Non-finite/out-of-range confidence and invalid temporal bounds.
            "subject=user:ada; predicate=theme; value=dark; confidence=nan; "
            "authority=application",
            "subject=user:ada; predicate=theme; value=dark; confidence=1.1; "
            "authority=application",
            "subject=user:ada; predicate=theme; value=dark; "
            "valid_from=yesterday; confidence=1.000; authority=application",
            "subject=user:ada; predicate=theme; value=dark; "
            "valid_from=2026-02-01T00:00:00Z; valid_to=2026-01-01T00:00:00Z; "
            "confidence=1.000; authority=application",
            "subject=user:ada; predicate=theme; value=dark; "
            "valid_from=2026-02-01T00:00:00Z; valid_to=2026-02-01T00:00:00Z; "
            "confidence=1.000; authority=application",
            # Text after the terminal grammar cannot be absorbed into authority.
            "subject=user:ada; predicate=theme; value=dark; confidence=1.000; "
            "authority=application. trailing prose",
        )
        for body in bodies:
            with self.subTest(body=body):
                result = self.extractor.extract(
                    self.context(f"Canonical claim: {body}.")
                )
                self.assertTrue(result.abstained)
                self.assertEqual(result.provenance, ())
                self.assertEqual(
                    result.diagnostics[0].code, "malformed_canonical_claim"
                )

    def test_a_malformed_span_does_not_suppress_independent_valid_span(self) -> None:
        content = (
            "Canonical claim: subject=user:ada; predicate=theme; value=dark; "
            "confidence=2.000; authority=application. "
            "Canonical claim: subject=device:quartz; predicate=serial_number; "
            "value=QZ-4401; confidence=1.000; authority=registry."
        )

        result = self.extractor.extract(self.context(content))

        self.assertEqual(len(result.claims), 1)
        self.assertEqual(result.claims[0].value, "QZ-4401")
        self.assertEqual(len(result.provenance), 1)
        self.assertEqual(len(result.diagnostics), 1)

    def test_evidence_digest_and_top_level_context_maps_are_immutable(self) -> None:
        context = self.context(
            "Canonical claim: subject=s; predicate=p; value=v; "
            "confidence=1.000; authority=a.",
            claim_ids_by_source={"old": "claim-old"},
        )
        with self.assertRaises(TypeError):
            context.metadata["origin"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            context.claim_ids_by_source["new"] = "claim-new"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "does not match"):
            EvidenceContext(
                namespace="tenant/user",
                content="exact evidence",
                content_sha256="0" * 64,
            )

    def test_evidence_metadata_is_deeply_copied_and_frozen(self) -> None:
        nested_list: list[object] = [{"state": ["original"]}]
        metadata: dict[str, object] = {"nested": nested_list}
        context = EvidenceContext.from_content(
            namespace="tenant/user",
            content="immutable evidence",
            metadata=metadata,
        )

        nested_list.append("later")
        nested_map = nested_list[0]
        assert isinstance(nested_map, dict)
        nested_map["state"] = ["changed"]
        frozen_list = context.metadata["nested"]
        self.assertIsInstance(frozen_list, tuple)
        self.assertEqual(frozen_list[0]["state"], ("original",))  # type: ignore[index]
        with self.assertRaises(TypeError):
            frozen_list[0]["state"] = ("changed",)  # type: ignore[index]

        cyclic: dict[str, object] = {}
        cyclic["self"] = cyclic
        with self.assertRaisesRegex(ValueError, "reference cycle"):
            EvidenceContext.from_content(
                namespace="tenant/user",
                content="cyclic metadata",
                metadata=cyclic,
            )
        for unsupported in ({"set": {1, 2}}, {"bytes": bytearray(b"x")}):
            with self.subTest(unsupported=unsupported):
                with self.assertRaisesRegex(TypeError, "unsupported"):
                    EvidenceContext.from_content(
                        namespace="tenant/user",
                        content="unsupported metadata",
                        metadata=unsupported,
                    )
        with self.assertRaisesRegex(ValueError, "finite"):
            EvidenceContext.from_content(
                namespace="tenant/user",
                content="non-finite metadata",
                metadata={"score": float("nan")},
            )

    def test_extraction_validator_freezes_sequences_and_checks_exact_evidence(self) -> None:
        content = (
            "Preface. Canonical claim: subject=device:quartz; "
            "predicate=serial_number; value=QZ-4401; confidence=1.000; "
            "authority=registry."
        )
        context = self.context(content)
        canonical = self.extractor.extract(context)
        mutable_claims = list(canonical.claims)
        mutable_provenance = list(canonical.provenance)
        mutable_diagnostics: list[ExtractionDiagnostic] = []
        provider_result = ClaimExtraction(
            mutable_claims,  # type: ignore[arg-type]
            mutable_provenance,  # type: ignore[arg-type]
            mutable_diagnostics,  # type: ignore[arg-type]
        )

        mutable_claims.clear()
        mutable_provenance.clear()
        mutable_diagnostics.append(ExtractionDiagnostic("late", "late mutation"))
        validated = validate_claim_extraction(
            context,
            provider_result,
            expected_identity=self.extractor.identity,
        )

        self.assertIsInstance(validated.claims, tuple)
        self.assertIsInstance(validated.provenance, tuple)
        self.assertIsInstance(validated.diagnostics, tuple)
        self.assertEqual(validated, canonical)

    def test_extraction_validator_rejects_forged_provenance(self) -> None:
        content = (
            "Canonical claim: subject=device:quartz; predicate=serial_number; "
            "value=QZ-4401; confidence=1.000; authority=registry."
        )
        context = self.context(content)
        valid = self.extractor.extract(context)
        provenance = valid.provenance[0]

        invalid_provenance = (
            replace(provenance, claim_index=1),
            replace(provenance, start=provenance.start + 1),
            replace(provenance, end=len(content) + 1),
            replace(provenance, canonical_text=provenance.canonical_text + "x"),
            replace(provenance, content_sha256="0" * 64),
            replace(provenance, extractor_identity=""),
            replace(provenance, extractor_identity="other/extractor"),
        )
        for forged in invalid_provenance:
            with self.subTest(forged=forged):
                with self.assertRaises(ValueError):
                    validate_claim_extraction(
                        context,
                        ClaimExtraction(valid.claims, (forged,)),
                        expected_identity=self.extractor.identity,
                    )

        with self.assertRaisesRegex(ValueError, "ordered and non-overlapping"):
            first = provenance
            second = replace(provenance, claim_index=1)
            validate_claim_extraction(
                context,
                ClaimExtraction(
                    (valid.claims[0], valid.claims[0]),
                    (first, second),
                ),
                expected_identity=self.extractor.identity,
            )

    def test_extraction_validator_checks_diagnostic_bounds(self) -> None:
        context = self.context("plain evidence")
        invalid_diagnostics = (
            ExtractionDiagnostic("code", "message", 0, None),
            ExtractionDiagnostic("code", "message", -1, 1),
            ExtractionDiagnostic("code", "message", 0, 99),
            ExtractionDiagnostic("", "message"),
            ExtractionDiagnostic("code", ""),
        )
        for diagnostic in invalid_diagnostics:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaises(ValueError):
                    validate_claim_extraction(
                        context,
                        ClaimExtraction(diagnostics=(diagnostic,)),
                        expected_identity=self.extractor.identity,
                    )

    def test_claim_extraction_rejects_wrong_member_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "ClaimInput"):
            ClaimExtraction(claims=("not a claim",), provenance=())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "ClaimProvenance"):
            ClaimExtraction(
                claims=(ClaimInput("s", "p", "v"),),
                provenance=("not provenance",),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "ExtractionDiagnostic"):
            ClaimExtraction(diagnostics=("not diagnostic",))  # type: ignore[arg-type]

    def test_result_rejects_claims_without_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one provenance"):
            ClaimExtraction(claims=(ClaimInput("s", "p", "v"),))


class _TestReranker:
    @property
    def identity(self) -> str:
        return "test/reranker-v1"

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate]
    ) -> Sequence[RerankResult]:
        del query
        return tuple(
            RerankResult(candidate.candidate_id, float(len(candidate.text)))
            for candidate in candidates
        )


class RerankerContractTests(unittest.TestCase):
    def test_runtime_protocol_and_validated_subset(self) -> None:
        reranker = _TestReranker()
        candidates = (
            RerankCandidate("a", "short"),
            RerankCandidate("b", "a longer text"),
        )
        self.assertIsInstance(reranker, Reranker)
        self.assertEqual(
            validate_rerank_results(candidates, reranker.rerank("query", candidates)),
            (RerankResult("a", 5.0), RerankResult("b", 13.0)),
        )
        self.assertEqual(
            validate_rerank_results(candidates, (RerankResult("b", 0.25),)),
            (RerankResult("b", 0.25),),
        )

    def test_reranker_output_rejects_unknown_duplicate_and_nonfinite_scores(self) -> None:
        candidates = (RerankCandidate("a", "text"),)
        invalid = (
            (RerankResult("unknown", 1.0),),
            (RerankResult("a", 1.0), RerankResult("a", 0.5)),
            (RerankResult("a", float("nan")),),
            (RerankResult("a", float("inf")),),
        )
        for results in invalid:
            with self.subTest(results=results):
                with self.assertRaises(ValueError):
                    validate_rerank_results(candidates, results)

        with self.assertRaisesRegex(ValueError, "candidate IDs must be unique"):
            validate_rerank_results(
                (RerankCandidate("a", "one"), RerankCandidate("a", "two")),
                (),
            )


if __name__ == "__main__":
    unittest.main()
