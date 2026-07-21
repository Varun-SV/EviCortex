"""Public API for the deterministic EviCortex research kernel."""

from .engine import EviCortex, SemanticIndexStatus
from .errors import (
    EmbeddingProfileNotFoundError,
    SemanticNotConfiguredError,
    SemanticProjectionError,
)
from .models import (
    ClaimDerivationRecord,
    ClaimInput,
    ClaimRecord,
    ExtractionDiagnosticRecord,
    ExtractionRecord,
    MemoryRecord,
)
from .plugins import (
    CanonicalClaimExtractor,
    ClaimExtraction,
    ClaimExtractor,
    ClaimProvenance,
    Embedder,
    EmbeddingValidationError,
    EvidenceContext,
    ExtractionDiagnostic,
    HashingEmbedder,
    RerankCandidate,
    Reranker,
    RerankResult,
    RoleAwareEmbedder,
    embed_document_batch,
    embed_query_text,
    snapshot_reranker_identity,
    validate_claim_extraction,
    validate_embedding_batch,
    validate_rerank_results,
)
from .retrieval import CapsuleDiagnostic, MemoryCapsule
from .semantic import (
    EmbeddingInput,
    EmbeddingProfile,
    claim_embedding_input,
    memory_embedding_input,
)

__all__ = [
    "CapsuleDiagnostic",
    "CanonicalClaimExtractor",
    "ClaimDerivationRecord",
    "ClaimExtraction",
    "ClaimExtractor",
    "ClaimInput",
    "ClaimProvenance",
    "ClaimRecord",
    "Embedder",
    "EmbeddingInput",
    "EmbeddingProfile",
    "EmbeddingProfileNotFoundError",
    "EmbeddingValidationError",
    "EvidenceContext",
    "ExtractionDiagnostic",
    "ExtractionDiagnosticRecord",
    "ExtractionRecord",
    "HashingEmbedder",
    "MemoryCapsule",
    "MemoryRecord",
    "EviCortex",
    "RerankCandidate",
    "Reranker",
    "RerankResult",
    "RoleAwareEmbedder",
    "SemanticIndexStatus",
    "SemanticNotConfiguredError",
    "SemanticProjectionError",
    "claim_embedding_input",
    "embed_document_batch",
    "embed_query_text",
    "memory_embedding_input",
    "snapshot_reranker_identity",
    "validate_claim_extraction",
    "validate_embedding_batch",
    "validate_rerank_results",
]

__version__ = "0.3.0a0"
