"""Framework-free contract tests for the optional Transformers provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from evicortex.plugins import Embedder, RoleAwareEmbedder
from evicortex.providers import (
    EmbeddingInputTooLongError,
    TransformersDependencyError,
    TransformersEmbedder,
    local_transformers_artifact_sha256,
)


class _FakeRuntime:
    dimensions = 3
    max_positions = 32
    resolved_revision = "b" * 40
    tokenizer_resolved_revision = "b" * 40
    metadata: Mapping[str, object] = MappingProxyType(
        {"torch_version": "test", "transformers_version": "test"}
    )

    def __init__(self) -> None:
        self.calls: list[
            tuple[tuple[str, ...], str, bool, int, str]
        ] = []

    def encode(
        self,
        texts: Sequence[str],
        *,
        pooling: str,
        normalized: bool,
        max_length: int,
        overflow_policy: str,
    ) -> tuple[tuple[float, float, float], ...]:
        frozen = tuple(texts)
        self.calls.append(
            (frozen, pooling, normalized, max_length, overflow_policy)
        )
        if overflow_policy == "error" and any(len(text) > max_length for text in frozen):
            raise EmbeddingInputTooLongError("fake overlength input")
        return tuple(
            (0.0, 1.0, 0.0) if text.startswith("query: ") else (1.0, 0.0, 0.0)
            for text in frozen
        )


class TransformersEmbedderTests(unittest.TestCase):
    def _open(self, runtime: _FakeRuntime, **kwargs: object) -> TransformersEmbedder:
        defaults: dict[str, object] = {
            "revision": "b" * 40,
            "pooling": "mean",
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
            "max_length": 16,
            "batch_size": 2,
        }
        defaults.update(kwargs)
        with patch(
            "evicortex.providers.transformers._load_transformers_runtime",
            return_value=runtime,
        ):
            return TransformersEmbedder("example/model", **defaults)

    def test_identity_role_dispatch_batching_and_metadata_are_stable(self) -> None:
        first_runtime = _FakeRuntime()
        first = self._open(first_runtime)
        second = self._open(_FakeRuntime())
        changed = self._open(_FakeRuntime(), query_prefix="search: ")

        self.assertIsInstance(first, Embedder)
        self.assertIsInstance(first, RoleAwareEmbedder)
        self.assertEqual(first.identity, second.identity)
        self.assertNotEqual(first.identity, changed.identity)
        self.assertEqual(first.dimensions, 3)
        self.assertTrue(first.is_semantic)
        self.assertFalse(first.diagnostic_only)
        self.assertEqual(first.resolved_revision, "b" * 40)
        self.assertEqual(first.tokenizer_resolved_revision, "b" * 40)
        with self.assertRaises(TypeError):
            first.runtime_metadata["device"] = "changed"  # type: ignore[index]

        documents = first.embed_documents(("one", "two", "three"))
        query = first.embed_query("find one")
        self.assertEqual(documents, ((1.0, 0.0, 0.0),) * 3)
        self.assertEqual(query, (0.0, 1.0, 0.0))
        self.assertEqual(len(first_runtime.calls), 3)
        self.assertEqual(first_runtime.calls[0][0], ("passage: one", "passage: two"))
        self.assertEqual(first_runtime.calls[1][0], ("passage: three",))
        self.assertEqual(first_runtime.calls[2][0], ("query: find one",))

    def test_no_silent_truncation_and_configuration_validation(self) -> None:
        embedder = self._open(_FakeRuntime(), max_length=8)
        with self.assertRaises(EmbeddingInputTooLongError):
            embedder.embed_documents(("a document that is too long",))

        truncated = self._open(
            _FakeRuntime(),
            max_length=8,
            overflow_policy="truncate",
        )
        self.assertEqual(truncated.embed_documents(("a document that is too long",)), ((1.0, 0.0, 0.0),))
        self.assertNotEqual(embedder.identity, truncated.identity)

        invalid = (
            {"pooling": "unknown"},
            {"overflow_policy": "ignore"},
            {"max_length": 0},
            {"batch_size": 0},
            {"normalized": "yes"},
            {"trust_remote_code": True, "revision": "main"},
            {"revision": "main"},
            {"trust_remote_code": True},
            {"code_revision": "c" * 40},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                self._open(_FakeRuntime(), **values)

        with self.assertRaisesRegex(ValueError, "capacity"):
            self._open(_FakeRuntime(), max_length=33)

        remote_code = self._open(
            _FakeRuntime(),
            trust_remote_code=True,
            code_revision="c" * 40,
        )
        self.assertEqual(remote_code.identity_fields["code_revision"], "c" * 40)

    def test_local_artifacts_are_content_pinned_and_path_independent(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_path = Path(first_dir)
            second_path = Path(second_dir)
            (first_path / "config.json").write_text("{}", encoding="utf-8")
            (first_path / "model.safetensors").write_bytes(b"weights")
            (second_path / "config.json").write_text("{}", encoding="utf-8")
            (second_path / "model.safetensors").write_bytes(b"weights")
            digest = local_transformers_artifact_sha256(first_path)
            self.assertEqual(
                digest,
                local_transformers_artifact_sha256(second_path),
            )

            with patch(
                "evicortex.providers.transformers._load_transformers_runtime",
                return_value=_FakeRuntime(),
            ):
                first = TransformersEmbedder(
                    str(first_path),
                    revision="local-label-one",
                    pooling="mean",
                    max_length=16,
                    local_artifact_sha256=digest,
                )
                second = TransformersEmbedder(
                    str(second_path),
                    revision="local-label-two",
                    pooling="mean",
                    max_length=16,
                    local_artifact_sha256=digest,
                )
            self.assertEqual(first.identity, second.identity)
            self.assertNotIn(str(first_path), first.identity_fields.values())

            (first_path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                TransformersEmbedder(
                    str(first_path),
                    revision="local-label-one",
                    pooling="mean",
                    max_length=16,
                    local_artifact_sha256=digest,
                )

    def test_dependency_failure_is_actionable(self) -> None:
        with patch(
            "evicortex.providers.transformers._load_transformers_runtime",
            side_effect=TransformersDependencyError("missing optional runtime"),
        ):
            with self.assertRaisesRegex(TransformersDependencyError, "optional"):
                TransformersEmbedder(
                    "example/model",
                    revision="b" * 40,
                    pooling="mean",
                )


@unittest.skipUnless(
    os.environ.get("EVICORTEX_TEST_TRANSFORMERS_MODEL")
    and os.environ.get("EVICORTEX_TEST_TRANSFORMERS_REVISION"),
    "set pinned cached model and revision for the optional local integration test",
)
class CachedTransformersIntegrationTests(unittest.TestCase):
    def test_cached_model_embeds_query_and_documents_offline(self) -> None:
        model = os.environ["EVICORTEX_TEST_TRANSFORMERS_MODEL"]
        revision = os.environ["EVICORTEX_TEST_TRANSFORMERS_REVISION"]
        pooling = os.environ.get("EVICORTEX_TEST_TRANSFORMERS_POOLING", "mean")
        embedder = TransformersEmbedder(
            model,
            revision=revision,
            pooling=pooling,  # type: ignore[arg-type]
            query_prefix=os.environ.get("EVICORTEX_TEST_QUERY_PREFIX", ""),
            document_prefix=os.environ.get("EVICORTEX_TEST_DOCUMENT_PREFIX", ""),
            max_length=128,
            batch_size=2,
            device="cpu",
            local_files_only=True,
        )
        documents = embedder.embed_documents(("A compact car.", "A ripe banana."))
        query = embedder.embed_query("Which text describes an automobile?")
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(query), embedder.dimensions)
        self.assertTrue(all(len(vector) == embedder.dimensions for vector in documents))


if __name__ == "__main__":
    unittest.main()
