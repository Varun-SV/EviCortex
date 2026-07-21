from __future__ import annotations

import json
from collections.abc import Sequence
import unittest

from evicortex import EviCortex, RerankCandidate, Reranker, RerankResult
from evicortex.retrieval import DeterministicRetriever


def _claim(
    claim_id: str,
    value: str,
    *,
    subject: str = "topic",
    predicate: str = "detail",
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "namespace": "n",
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "status": "active",
        "confidence": 1.0,
        "evidence_ids": (),
    }


def _episode(
    memory_id: str,
    content: str,
    *,
    content_hash: str | None = None,
) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "namespace": "n",
        "kind": "episode",
        "content": content,
        "content_hash": content_hash or f"hash-{memory_id}",
        "source": "test",
        "occurred_at": "2026-01-01T00:00:00Z",
        "recorded_at": "2026-01-01T00:00:00Z",
    }


class _Store:
    def __init__(
        self,
        *,
        claims: Sequence[dict[str, object]] = (),
        episodes: Sequence[dict[str, object]] = (),
    ) -> None:
        self.claims = tuple(claims)
        self.episodes = tuple(episodes)

    def search_claims(
        self,
        namespace: str,
        query: str,
        as_of: str | None = None,
        limit: int = 10,
    ) -> tuple[dict[str, object], ...]:
        del namespace, query, as_of
        return self.claims[:limit]

    def search_memories(
        self,
        namespace: str,
        query: str,
        as_of: str | None = None,
        limit: int = 10,
    ) -> tuple[dict[str, object], ...]:
        del namespace, query, as_of
        return self.episodes[:limit]


class _CapturingReranker:
    identity = "tests/capturing-reranker-v1"

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[tuple[str, tuple[RerankCandidate, ...]]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> Sequence[RerankResult]:
        frozen = tuple(candidates)
        self.calls.append((query, frozen))
        return tuple(
            RerankResult(candidate.candidate_id, self.scores.get(candidate.candidate_id, 0.0))
            for candidate in frozen
        )


class _PartialReranker:
    identity = "tests/partial-reranker-v1"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> Sequence[RerankResult]:
        del query
        return (RerankResult(candidates[0].candidate_id, 1.0),)


class _MutableIdentityReranker(_CapturingReranker):
    identity = "tests/mutable-reranker-v1"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> Sequence[RerankResult]:
        results = super().rerank(query, candidates)
        self.identity = "tests/mutable-reranker-v2"
        return results


class RerankerIntegrationTests(unittest.TestCase):
    def test_reranks_only_filtered_deduplicated_candidates_with_typed_ids(
        self,
    ) -> None:
        store = _Store(
            claims=(
                _claim("shared", "target"),
                _claim("zduplicate", "target"),
            ),
            episodes=(
                _episode("shared", "target episode"),
                _episode("banned", "target banned episode"),
            ),
        )
        reranker = _CapturingReranker(
            {"fact:shared": 0.0, "episode:shared": 100.0}
        )
        retriever = DeterministicRetriever(
            store,
            reranker=reranker,
            rerank_limit=10,
            reranker_weight=1000.0,
        )

        capsule = retriever.recall(
            namespace="n",
            query="target without banned",
            budget_tokens=2048,
            limit=2,
        )

        self.assertEqual(retriever.reranker_identity, reranker.identity)
        self.assertEqual(len(reranker.calls), 1)
        query, candidates = reranker.calls[0]
        self.assertEqual(query, "target without banned")
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in candidates),
            ("fact:shared", "episode:shared"),
        )
        decoded = {item.candidate_id: json.loads(item.text) for item in candidates}
        self.assertEqual(decoded["fact:shared"]["type"], "fact")
        self.assertEqual(decoded["episode:shared"]["type"], "episode")

        self.assertEqual(
            tuple((item.candidate_type, item.candidate_id) for item in capsule.trace),
            (("episode", "shared"), ("fact", "shared")),
        )
        for trace in capsule.trace:
            reason = next(item for item in trace.reasons if item.channel == "reranker")
            detail = json.loads(reason.detail)
            self.assertEqual(detail["identity"], reranker.identity)
            self.assertEqual(detail["score_use"], "ordinal_only")
            self.assertGreater(reason.contribution, 0.0)

    def test_rerank_pool_is_bounded_but_applied_before_final_limit(self) -> None:
        store = _Store(
            episodes=tuple(
                _episode(f"e{index}", f"memory item {index}")
                for index in range(1, 5)
            )
        )
        reranker = _CapturingReranker(
            {"episode:e1": 0.0, "episode:e2": 1.0}
        )
        retriever = DeterministicRetriever(
            store,
            reranker=reranker,
            rerank_limit=2,
        )

        capsule = retriever.recall(
            namespace="n",
            query="memory",
            budget_tokens=2048,
            limit=1,
        )

        self.assertEqual(
            tuple(item.candidate_id for item in reranker.calls[0][1]),
            ("episode:e1", "episode:e2"),
        )
        self.assertEqual(tuple(item.memory_id for item in capsule.episodes), ("e2",))

    def test_default_pool_keeps_cross_encoder_work_bounded(self) -> None:
        reranker = _CapturingReranker()
        retriever = DeterministicRetriever(
            _Store(
                episodes=tuple(
                    _episode(f"e{index:02d}", f"memory item {index}")
                    for index in range(20)
                )
            ),
            reranker=reranker,
        )

        retriever.recall(
            namespace="n",
            query="memory",
            budget_tokens=8192,
            limit=20,
        )

        self.assertEqual(len(reranker.calls[0][1]), 16)

    def test_equal_provider_scores_preserve_deterministic_base_order(self) -> None:
        store = _Store(
            episodes=(
                _episode("e1", "memory one"),
                _episode("e2", "memory two"),
                _episode("e3", "memory three"),
            )
        )
        reranker = _CapturingReranker()
        retriever = DeterministicRetriever(store, reranker=reranker)

        first = retriever.recall(
            namespace="n",
            query="memory",
            budget_tokens=2048,
            limit=3,
        )
        second = retriever.recall(
            namespace="n",
            query="memory",
            budget_tokens=2048,
            limit=3,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            tuple(item.memory_id for item in first.episodes),
            ("e1", "e2", "e3"),
        )
        ranks = [
            next(reason.rank for reason in item.reasons if reason.channel == "reranker")
            for item in first.episodes
        ]
        self.assertEqual(ranks, [1, 2, 3])

    def test_partial_provider_output_fails_loudly(self) -> None:
        retriever = DeterministicRetriever(
            _Store(
                episodes=(
                    _episode("e1", "memory one"),
                    _episode("e2", "memory two"),
                )
            ),
            reranker=_PartialReranker(),
        )

        with self.assertRaisesRegex(ValueError, "reranker omitted candidates"):
            retriever.recall(
                namespace="n",
                query="memory",
                budget_tokens=2048,
            )

    def test_identity_drift_is_rejected_at_the_provider_boundary(self) -> None:
        retriever = DeterministicRetriever(
            _Store(episodes=(_episode("e1", "memory one"),)),
            reranker=_MutableIdentityReranker(),
        )

        with self.assertRaisesRegex(ValueError, "identity changed during reranking"):
            retriever.recall(
                namespace="n",
                query="memory",
                budget_tokens=2048,
            )

    def test_facade_exposes_identity_and_validates_configuration(self) -> None:
        reranker = _CapturingReranker()
        self.assertIsInstance(reranker, Reranker)
        with EviCortex.open(reranker=reranker) as memory:
            self.assertEqual(memory.reranker_identity, reranker.identity)
        with EviCortex.open() as memory:
            self.assertIsNone(memory.reranker_identity)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            EviCortex.open(reranker=reranker, rerank_limit=0)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            EviCortex.open(reranker=reranker, reranker_weight=0.0)


if __name__ == "__main__":
    unittest.main()
