"""Minimal temporal-memory example for a EviCortex source checkout."""

from evicortex import ClaimInput, EviCortex


NAMESPACE = "assistant/user-ada"


def main() -> None:
    with EviCortex.open(":memory:") as memory:
        earlier = memory.remember(
            namespace=NAMESPACE,
            content="Ada prefers the light interface theme.",
            source="conversation/session-1/turn-4",
            occurred_at="2026-01-02T10:00:00Z",
            claims=[
                ClaimInput(
                    subject="user:ada",
                    predicate="prefers_theme",
                    value="light",
                )
            ],
        )
        memory.remember(
            namespace=NAMESPACE,
            content="Ada switched to the dark interface theme.",
            source="conversation/session-7/turn-12",
            occurred_at="2026-04-18T09:30:00Z",
            claims=[
                ClaimInput(
                    subject="user:ada",
                    predicate="prefers_theme",
                    value="dark",
                    supersedes=earlier.claims[0].id,
                )
            ],
        )

        current = memory.recall(
            namespace=NAMESPACE,
            query="Which interface theme does Ada prefer?",
            budget_tokens=300,
        )
        historical = memory.recall(
            namespace=NAMESPACE,
            query="Which interface theme did Ada prefer?",
            as_of="2026-02-01T00:00:00Z",
            budget_tokens=300,
        )

        print("Current capsule:", current.to_prompt())
        print("Historical capsule:", historical.to_prompt())

        evidence = memory.expand(
            namespace=NAMESPACE,
            memory_id=current.evidence_ids[0],
        )
        print("Expanded evidence:", evidence.content)


if __name__ == "__main__":
    main()
