"""Offline-first Hugging Face Transformers embedding adapter.

The base EviCortex package remains standard-library-only. This module imports
PyTorch and Transformers lazily when :class:`TransformersEmbedder` is created.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
from pathlib import Path
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Literal, Protocol

from ..plugins import EmbeddingVector, validate_embedding_batch


PoolingMode = Literal["cls", "mean", "last_token"]
OverflowPolicy = Literal["error", "truncate"]
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ADAPTER_SCHEMA = "evicortex.transformers-embedder.v1"
_LOCAL_ARTIFACT_SCHEMA = b"evicortex.local-transformers-artifact.v1\0"


class TransformersDependencyError(RuntimeError):
    """Raised when the optional local-model runtime is unavailable."""


class EmbeddingInputTooLongError(ValueError):
    """Raised before inference when explicit truncation was not authorized."""


class _EmbeddingRuntime(Protocol):
    dimensions: int
    max_positions: int | None
    resolved_revision: str
    tokenizer_resolved_revision: str
    metadata: Mapping[str, object]

    def encode(
        self,
        texts: Sequence[str],
        *,
        pooling: PoolingMode,
        normalized: bool,
        max_length: int,
        overflow_policy: OverflowPolicy,
    ) -> Sequence[Sequence[float]]:
        """Encode already-prefixed text."""


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without edge whitespace")
    return value


def _non_negative_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def local_transformers_artifact_sha256(path: str | Path) -> str:
    """Hash every file in a local model artifact with path/content framing.

    A caller can pin this digest in configuration before constructing a
    :class:`TransformersEmbedder`. Hashing all files is deliberately
    conservative: even a custom-code or tokenizer-file change invalidates the
    embedding profile instead of silently mixing projections.
    """

    source = Path(path).expanduser().resolve(strict=True)
    files: tuple[tuple[str, Path], ...]
    if source.is_file():
        files = ((source.name, source),)
    elif source.is_dir():
        collected: list[tuple[str, Path]] = []
        for candidate in source.rglob("*"):
            is_junction = bool(
                getattr(candidate, "is_junction", lambda: False)()
            )
            if candidate.is_symlink() or is_junction:
                target = candidate.resolve(strict=True)
                if not target.is_relative_to(source):
                    raise ValueError(
                        "local transformer artifact contains a link outside its root"
                    )
            if candidate.is_file():
                collected.append(
                    (candidate.relative_to(source).as_posix(), candidate)
                )
        files = tuple(sorted(collected, key=lambda item: item[0]))
    else:
        raise ValueError("local transformer artifact must be a file or directory")
    if not files:
        raise ValueError("local transformer artifact contains no files")

    digest = hashlib.sha256()
    digest.update(_LOCAL_ARTIFACT_SCHEMA)
    for relative, candidate in files:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        size = candidate.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as stream:
            observed = 0
            while block := stream.read(1024 * 1024):
                observed += len(block)
                digest.update(block)
        if observed != size:
            raise RuntimeError(
                f"local transformer artifact changed while hashing: {candidate}"
            )
    return digest.hexdigest()


class _TorchTransformersRuntime:
    metadata: Mapping[str, object]

    def __init__(
        self,
        *,
        model_name_or_path: str,
        revision: str,
        code_revision: str | None,
        device: str,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            tokenizers = importlib.import_module("tokenizers")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise TransformersDependencyError(
                "TransformersEmbedder requires the 'local-transformers' extra"
            ) from error

        started = time.perf_counter()
        common = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if code_revision is not None:
            common["code_revision"] = code_revision
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path,
                **common,
            )
            model = transformers.AutoModel.from_pretrained(
                model_name_or_path,
                **common,
                dtype=torch.float32,
                use_safetensors=True,
            )
            torch_device = torch.device(device)
            model.to(torch_device)
            model.eval()
        except Exception as error:
            raise RuntimeError(
                f"failed to load pinned transformer model {model_name_or_path!r}"
            ) from error

        dimensions = getattr(model.config, "hidden_size", None)
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise ValueError("transformer model config has no integer hidden_size")
        max_positions = getattr(model.config, "max_position_embeddings", None)
        if isinstance(max_positions, bool) or not isinstance(max_positions, int):
            max_positions = None
        tokenizer_capacity = getattr(tokenizer, "model_max_length", None)
        if (
            isinstance(tokenizer_capacity, bool)
            or not isinstance(tokenizer_capacity, int)
            or tokenizer_capacity < 1
            or tokenizer_capacity >= 1_000_000_000
        ):
            tokenizer_capacity = None
        capacities = tuple(
            value
            for value in (max_positions, tokenizer_capacity)
            if value is not None
        )
        max_input_tokens = min(capacities) if capacities else None
        resolved = getattr(model.config, "_commit_hash", None)
        if not isinstance(resolved, str) or not resolved:
            resolved = revision
        tokenizer_revision = getattr(tokenizer, "_commit_hash", None)
        if not isinstance(tokenizer_revision, str) or not tokenizer_revision:
            init_kwargs = getattr(tokenizer, "init_kwargs", {})
            if isinstance(init_kwargs, Mapping):
                tokenizer_revision = init_kwargs.get("_commit_hash")
        if not isinstance(tokenizer_revision, str) or not tokenizer_revision:
            tokenizer_revision = revision
        parameter = next(model.parameters(), None)
        if parameter is None:
            raise ValueError("transformer model has no parameters")
        model_dtype = str(parameter.dtype)
        if model_dtype != "torch.float32":
            raise ValueError(
                f"transformer model did not load as enforced float32: {model_dtype}"
            )

        self.dimensions = dimensions
        self.max_positions = max_input_tokens
        self.resolved_revision = resolved
        self.tokenizer_resolved_revision = tokenizer_revision
        self.metadata = MappingProxyType(
            {
                "model_class": type(model).__qualname__,
                "model_dtype": model_dtype,
                "model_position_capacity": max_positions,
                "torch_version": str(getattr(torch, "__version__", "unknown")),
                "tokenizer_class": type(tokenizer).__qualname__,
                "tokenizer_capacity": tokenizer_capacity,
                "tokenizer_resolved_revision": tokenizer_revision,
                "tokenizers_version": str(
                    getattr(tokenizers, "__version__", "unknown")
                ),
                "transformers_version": str(
                    getattr(transformers, "__version__", "unknown")
                ),
                "device": str(torch_device),
                "load_seconds": time.perf_counter() - started,
            }
        )
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = torch_device
        self._lock = threading.Lock()

    def _tokenize(
        self,
        texts: Sequence[str],
        *,
        max_length: int,
        overflow_policy: OverflowPolicy,
    ) -> Mapping[str, Any]:
        probe_length = (
            max_length if overflow_policy == "truncate" else max_length + 1
        )
        encoded = self._tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=probe_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if overflow_policy == "error":
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for index, raw_length in enumerate(lengths):
                length = int(raw_length)
                if length > max_length:
                    raise EmbeddingInputTooLongError(
                        f"text {index} exceeds {max_length} encoded tokens; "
                        "overflow_policy='error'"
                    )
        return encoded

    def encode(
        self,
        texts: Sequence[str],
        *,
        pooling: PoolingMode,
        normalized: bool,
        max_length: int,
        overflow_policy: OverflowPolicy,
    ) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        with self._lock:
            encoded = self._tokenize(
                texts,
                max_length=max_length,
                overflow_policy=overflow_policy,
            )
            device_inputs = {
                key: value.to(self._device) for key, value in encoded.items()
            }
            with self._torch.inference_mode():
                output = self._model(**device_inputs)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    raise ValueError("transformer model returned no last_hidden_state")
                attention = device_inputs.get("attention_mask")
                if attention is None:
                    raise ValueError("tokenizer returned no attention_mask")

                if pooling == "cls":
                    pooled = hidden[:, 0]
                elif pooling == "mean":
                    mask = attention.unsqueeze(-1).to(dtype=hidden.dtype)
                    denominator = mask.sum(dim=1).clamp(min=1)
                    pooled = (hidden * mask).sum(dim=1) / denominator
                else:
                    positions = self._torch.arange(
                        attention.shape[1],
                        device=attention.device,
                    ).unsqueeze(0)
                    final_indices = (positions * attention).max(dim=1).values
                    rows = self._torch.arange(
                        attention.shape[0],
                        device=attention.device,
                    )
                    pooled = hidden[rows, final_indices]

                pooled = pooled.float()
                if normalized:
                    norms = pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    pooled = pooled / norms
                return pooled.cpu().tolist()


def _load_transformers_runtime(
    *,
    model_name_or_path: str,
    revision: str,
    code_revision: str | None,
    device: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> _EmbeddingRuntime:
    return _TorchTransformersRuntime(
        model_name_or_path=model_name_or_path,
        revision=revision,
        code_revision=code_revision,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


class TransformersEmbedder:
    """Pinned dense encoder with explicit asymmetric preprocessing.

    ``revision`` participates in the durable embedding profile. By default the
    adapter refuses network access and rejects overlength input instead of
    silently creating a vector that never observed the tail of the evidence.
    """

    is_semantic = True
    diagnostic_only = False

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str,
        pooling: PoolingMode,
        code_revision: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        max_length: int = 512,
        overflow_policy: OverflowPolicy = "error",
        batch_size: int = 16,
        normalized: bool = True,
        device: str = "cpu",
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        local_artifact_sha256: str | None = None,
    ) -> None:
        self.model_name_or_path = _required_text(
            model_name_or_path, "model_name_or_path"
        )
        self.revision = _required_text(revision, "revision")
        local_candidate = Path(self.model_name_or_path).expanduser()
        try:
            is_local_artifact = local_candidate.exists()
        except OSError:
            is_local_artifact = False
        if not is_local_artifact and not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ValueError(
                "remote model revision must be a pinned 40-character commit"
            )
        if pooling not in ("cls", "mean", "last_token"):
            raise ValueError("pooling must be 'cls', 'mean', or 'last_token'")
        if overflow_policy not in ("error", "truncate"):
            raise ValueError("overflow_policy must be 'error' or 'truncate'")
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a boolean")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")
        if not isinstance(trust_remote_code, bool):
            raise TypeError("trust_remote_code must be a boolean")
        if is_local_artifact and trust_remote_code:
            raise ValueError(
                "trust_remote_code is not supported for local filesystem artifacts"
            )
        if trust_remote_code:
            if code_revision is None or not isinstance(code_revision, str):
                raise ValueError(
                    "trust_remote_code requires a separate pinned code_revision"
                )
            if not _COMMIT_PATTERN.fullmatch(code_revision):
                raise ValueError(
                    "code_revision must be a 40-character commit when remote code is trusted"
                )
            self.code_revision: str | None = code_revision
        else:
            if code_revision is not None:
                raise ValueError(
                    "code_revision is only valid when trust_remote_code=True"
                )
            self.code_revision = None

        if is_local_artifact:
            if not isinstance(local_artifact_sha256, str) or not _SHA256_PATTERN.fullmatch(
                local_artifact_sha256
            ):
                raise ValueError(
                    "local model paths require a pinned 64-character "
                    "local_artifact_sha256"
                )
            observed_local_digest = local_transformers_artifact_sha256(local_candidate)
            if observed_local_digest != local_artifact_sha256:
                raise ValueError("local transformer artifact digest does not match its pin")
            self.local_artifact_sha256: str | None = observed_local_digest
        else:
            if local_artifact_sha256 is not None:
                raise ValueError(
                    "local_artifact_sha256 is only valid for an existing local path"
                )
            self.local_artifact_sha256 = None
        self.pooling = pooling
        self.query_prefix = _non_negative_text(query_prefix, "query_prefix")
        self.document_prefix = _non_negative_text(
            document_prefix, "document_prefix"
        )
        self.max_length = _positive_integer(max_length, "max_length")
        self.overflow_policy = overflow_policy
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.normalized = normalized
        self.device = _required_text(device, "device")
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

        runtime = _load_transformers_runtime(
            model_name_or_path=self.model_name_or_path,
            revision=self.revision,
            code_revision=self.code_revision,
            device=self.device,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        if is_local_artifact:
            post_load_digest = local_transformers_artifact_sha256(local_candidate)
            if post_load_digest != self.local_artifact_sha256:
                raise RuntimeError(
                    "local transformer artifact changed while the model was loading"
                )
        if runtime.max_positions is not None and self.max_length > runtime.max_positions:
            raise ValueError(
                f"max_length {self.max_length} exceeds model capacity "
                f"{runtime.max_positions}"
            )
        self._runtime = runtime
        self.dimensions = _positive_integer(runtime.dimensions, "model dimensions")
        self.resolved_revision = _required_text(
            runtime.resolved_revision, "resolved revision"
        )
        self.tokenizer_resolved_revision = _required_text(
            runtime.tokenizer_resolved_revision,
            "tokenizer resolved revision",
        )
        if (
            not is_local_artifact
            and (
                self.resolved_revision != self.revision
                or self.tokenizer_resolved_revision != self.revision
            )
        ):
            raise ValueError(
                "resolved model/tokenizer revision does not match the requested commit"
            )
        self.runtime_metadata = MappingProxyType(dict(runtime.metadata))

        identity_payload = {
            "dimensions": self.dimensions,
            "document_prefix": self.document_prefix,
            "inference_dtype": "float32",
            "max_length": self.max_length,
            "normalized": self.normalized,
            "overflow_policy": self.overflow_policy,
            "pooling": self.pooling,
            "query_prefix": self.query_prefix,
            "schema": _ADAPTER_SCHEMA,
            "trust_remote_code": self.trust_remote_code,
        }
        identity_payload["implementation"] = {
            key: runtime.metadata.get(key, "unknown")
            for key in (
                "model_class",
                "model_dtype",
                "tokenizer_class",
                "tokenizers_version",
                "torch_version",
                "transformers_version",
            )
        }
        if is_local_artifact:
            identity_payload["model_source"] = "content-addressed-local-artifact"
        else:
            identity_payload.update(
                {
                    "model": self.model_name_or_path,
                    "requested_revision": self.revision,
                    "resolved_revision": self.resolved_revision,
                    "tokenizer_resolved_revision": (
                        self.tokenizer_resolved_revision
                    ),
                }
            )
        if self.code_revision is not None:
            identity_payload["code_revision"] = self.code_revision
        if self.local_artifact_sha256 is not None:
            identity_payload["local_artifact_sha256"] = self.local_artifact_sha256
        canonical = json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.identity = f"{_ADAPTER_SCHEMA}/{digest}"
        self.identity_fields = MappingProxyType(identity_payload)

    def _validated_texts(self, texts: Sequence[str]) -> tuple[str, ...]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        frozen = tuple(texts)
        for index, text in enumerate(frozen):
            if not isinstance(text, str):
                raise TypeError(f"text {index} must be a string")
        return frozen

    def _encode(self, texts: Sequence[str], *, prefix: str) -> tuple[EmbeddingVector, ...]:
        frozen = self._validated_texts(texts)
        vectors: list[Sequence[float]] = []
        for start in range(0, len(frozen), self.batch_size):
            batch = tuple(prefix + text for text in frozen[start : start + self.batch_size])
            vectors.extend(
                self._runtime.encode(
                    batch,
                    pooling=self.pooling,
                    normalized=self.normalized,
                    max_length=self.max_length,
                    overflow_policy=self.overflow_policy,
                )
            )
        return validate_embedding_batch(
            vectors,
            expected_count=len(frozen),
            dimensions=self.dimensions,
            normalized=self.normalized,
            allow_zero=False,
            tolerance=1e-5,
        )

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[EmbeddingVector, ...]:
        return self._encode(texts, prefix=self.document_prefix)

    def embed_query(self, text: str) -> EmbeddingVector:
        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        return self._encode((text,), prefix=self.query_prefix)[0]

    def embed_batch(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        """Backward-compatible alias whose semantics are document encoding."""

        return self.embed_documents(texts)
