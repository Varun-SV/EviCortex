"""Optional model-provider adapters.

Importing :mod:`evicortex` never imports a model framework. Applications opt in
through this package and the corresponding packaging extra.
"""

from .transformers import (
    EmbeddingInputTooLongError,
    TransformersDependencyError,
    TransformersEmbedder,
    local_transformers_artifact_sha256,
)

__all__ = [
    "EmbeddingInputTooLongError",
    "TransformersDependencyError",
    "TransformersEmbedder",
    "local_transformers_artifact_sha256",
]
