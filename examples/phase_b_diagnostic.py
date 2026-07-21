"""Durable extraction plus diagnostic semantic indexing from a source checkout."""

from evicortex import CanonicalClaimExtractor, HashingEmbedder, EviCortex


NAMESPACE = "assistant/user-ada"
CONTENT = (
    "Ada selected the dark interface theme. "
    "Canonical claim: subject=user:ada; predicate=preferred_theme; value=dark; "
    "valid_from=2026-04-18T09:30:00Z; confidence=1.000; authority=application."
)


def main() -> None:
    # HashingEmbedder is deliberately non-semantic. It exercises projection,
    # persistence, fusion, and recovery without pretending to understand language.
    with EviCortex.open(
        ":memory:",
        extractor=CanonicalClaimExtractor(),
        embedder=HashingEmbedder(dimensions=128),
    ) as memory:
        record = memory.remember_extracted(
            namespace=NAMESPACE,
            content=CONTENT,
            source="conversation/session-7/turn-12",
            occurred_at="2026-04-18T09:30:00Z",
        )
        extraction = memory.get_extraction(NAMESPACE, record.id)
        capsule = memory.recall(
            namespace=NAMESPACE,
            query="What preferred_theme is recorded for Ada?",
            budget_tokens=300,
        )

        print("Extraction:", extraction)
        print("Capsule:", capsule.to_prompt())
        print("Semantic projection complete:", memory.semantic_status(NAMESPACE).complete)


if __name__ == "__main__":
    main()
