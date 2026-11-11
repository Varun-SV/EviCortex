"""Offline tests for the Phase D harness: no network, no model, no paid calls."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evicortex import HashingEmbedder
from evicortex.budget import estimate_tokens

from benchmarks.competitive import evaluate, report
from benchmarks.competitive.data import ABSTENTION_GOLD, load_locomo, load_longmemeval_s
from benchmarks.competitive.llm import CachedLLM, LLMResponse, make_client
from benchmarks.competitive.retrieve import evidence_metrics
from benchmarks.competitive.systems import (
    BM25Chunks,
    CachingEmbedder,
    EviCortexSystem,
    FullHistory,
    HybridRAG,
    NoMemory,
    OracleEvidence,
)
from benchmarks.competitive.types import Retrieved

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

_TOPICS = (
    ("Ada", "adopted a grey cat named Pixel"),
    ("Ben", "started learning the cello"),
    ("Ada", "moved to Lisbon for a new job"),
    ("Ben", "ran a marathon in under four hours"),
    ("Ada", "planted tomatoes on the balcony"),
    ("Ben", "bought a red bicycle"),
    ("Ada", "painted a mural of a lighthouse"),
    ("Ben", "baked sourdough bread every Sunday"),
)


def _locomo_fixture() -> list[dict]:
    sessions: dict = {"speaker_a": "Ada", "speaker_b": "Ben"}
    for number in (1, 2, 3):
        sessions[f"session_{number}_date_time"] = f"1:0{number} pm on {number} May, 2023"
        turns = []
        for index, (speaker, fact) in enumerate(_TOPICS):
            turn = {
                "speaker": speaker,
                "dia_id": f"D{number}:{index + 1}",
                "text": f"In session {number} {speaker} said that {speaker} {fact}.",
            }
            if index == 0:
                turn["blip_caption"] = "a photo of a cat"
            turns.append(turn)
        sessions[f"session_{number}"] = turns
    return [
        {
            "sample_id": "conv-test",
            "conversation": sessions,
            "qa": [
                {"question": "What pet did Ada adopt?", "answer": "a grey cat",
                 "evidence": ["D1:1"], "category": 4},
                {"question": "What instrument is Ben learning?", "answer": "cello",
                 "evidence": ["D1:2; D2:2"], "category": 1},
                {"question": "When did Ada move to Lisbon?", "answer": 2023,
                 "evidence": ["D1:3"], "category": 2},
                {"question": "What did Ben say about scuba diving?",
                 "adversarial_answer": "he loved it", "evidence": ["D9:9"], "category": 5},
            ],
        }
    ]


class _FakeClient:
    identity = "fake:model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
        self.calls += 1
        return LLMResponse(f"echo {len(prompt)}", "model", 10, 2, 0.001, 1.0)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "locomo.json"
        self.path.write_text(json.dumps(_locomo_fixture()), encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_locomo_units_questions_and_label_isolation(self) -> None:
        conversations, info = load_locomo(self.path)
        self.assertFalse(info["pinned_artifact_verified"])
        conversation = conversations[0]
        self.assertEqual(len(conversation.units), 24)
        first = conversation.units[0]
        self.assertEqual(first.unit_id, "D1:1")
        self.assertIn("[shared image: a photo of a cat]", first.rendered)
        self.assertTrue(first.rendered.startswith("[1:01 pm on 1 May, 2023] Ada: "))
        self.assertLess(first.occurred_at, conversation.units[1].occurred_at)
        single, multi, temporal, adversarial = conversation.questions
        self.assertEqual(single.category, "single-hop")
        self.assertEqual(multi.gold_unit_ids, ("D1:2", "D2:2"))
        self.assertEqual(multi.gold_session_ids, ("session_1", "session_2"))
        self.assertEqual(temporal.answer, "2023")
        self.assertTrue(adversarial.is_abstention)
        self.assertEqual(adversarial.answer, ABSTENTION_GOLD)
        self.assertEqual(adversarial.gold_unit_ids, ())
        # Gold answers never appear in what a system can ingest.
        self.assertFalse(any("he loved it" in unit.rendered for unit in conversation.units))


class LongMemEvalDataTests(unittest.TestCase):
    def test_duplicate_session_ids_are_kept_as_distinct_sessions(self) -> None:
        def case(question_id: str, question_type: str) -> dict:
            return {
                "question_id": question_id,
                "question_type": question_type,
                "question": "Which city did I move to?",
                "answer": "Lisbon",
                "question_date": "2023/05/03 (Wed) 10:00",
                "haystack_session_ids": ["s1", "s2", "s1"],
                "haystack_dates": [
                    "2023/05/01 (Mon) 09:00", "2023/05/02 (Tue) 09:00", "2023/05/04 (Thu) 09:00",
                ],
                "haystack_sessions": [
                    [{"role": "user", "content": "I like tea."},
                     {"role": "assistant", "content": ""}],
                    [{"role": "user", "content": "I moved to Lisbon.", "has_answer": True}],
                    [{"role": "user", "content": "A later, different session."}],
                ],
                "answer_session_ids": [] if question_id.endswith("_abs") else ["s2"],
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lme.json"
            path.write_text(
                json.dumps([case("q1", "knowledge-update"), case("q2_abs", "multi-session")]),
                encoding="utf-8",
            )
            conversations, info = load_longmemeval_s(path, per_type=None, seed=1)
        self.assertEqual(info["duplicate_session_ids_renamed_in_release"], 2)
        self.assertFalse(info["pinned_artifact_verified"])
        first, second = conversations
        self.assertEqual([unit.session_id for unit in first.units], ["s1", "s2", "s1#dup2"])
        self.assertEqual(first.questions[0].gold_unit_ids, ("s2:t0",))
        self.assertEqual(first.questions[0].gold_session_ids, ("s2",))
        # The release dates some gold sessions after the question, so the loader
        # sets no retrieval cutoff and passes the date to the reader only.
        self.assertIsNone(first.questions[0].question_date)
        self.assertEqual(first.questions[0].question_date_text, "2023/05/03 (Wed) 10:00")
        self.assertTrue(second.questions[0].is_abstention)
        self.assertEqual(second.questions[0].answer, ABSTENTION_GOLD)


class SystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        path = self.root / "locomo.json"
        path.write_text(json.dumps(_locomo_fixture()), encoding="utf-8")
        self.conversation = load_locomo(path)[0][0]
        self.embedder = CachingEmbedder(HashingEmbedder(dimensions=64))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _check_budget(self, system, budgets=(60, 200, 600)) -> None:
        system.ingest(self.conversation)
        try:
            for question in self.conversation.questions:
                for budget in budgets:
                    retrieved = system.retrieve(question, budget)
                    self.assertLessEqual(estimate_tokens(retrieved.context), budget)
                    known = {unit.unit_id for unit in self.conversation.units}
                    self.assertTrue(set(retrieved.unit_ids) <= known)
        finally:
            system.close()

    def test_bm25_respects_budget_and_finds_lexical_match(self) -> None:
        system = BM25Chunks(self.root / "bm25.sqlite3")
        system.ingest(self.conversation)
        try:
            retrieved = system.retrieve(self.conversation.questions[0], 2000)
        finally:
            system.close()
        self.assertIn("D1:1", retrieved.unit_ids)
        self._check_budget(BM25Chunks(self.root / "bm25b.sqlite3"))

    def test_evicortex_respects_budget_and_maps_evidence_to_turns(self) -> None:
        system = EviCortexSystem(self.root / "evicortex.sqlite3", self.embedder)
        self._check_budget(system)
        self.assertEqual(system.ingestion_cost()["llm_calls"], 0)

    def test_hybrid_respects_budget(self) -> None:
        self._check_budget(HybridRAG(self.root / "hybrid.sqlite3", self.embedder, fast_cosine=False))

    @unittest.skipUnless(HAS_NUMPY, "numpy is optional")
    def test_dense_respects_budget(self) -> None:
        from benchmarks.competitive.systems import DenseRAG

        self._check_budget(DenseRAG(self.embedder))

    @unittest.skipUnless(HAS_NUMPY, "numpy is optional")
    def test_fast_cosine_keeps_the_reference_ranking(self) -> None:
        from benchmarks import hybrid_baseline

        reference_cosine = hybrid_baseline.cosine_similarity
        try:
            reference = HybridRAG(self.root / "reference.sqlite3", self.embedder, fast_cosine=False)
            reference.ingest(self.conversation)
            expected = [
                reference.retrieve(question, budget).unit_ids
                for question in self.conversation.questions
                for budget in (120, 600)
            ]
            reference.close()
            fast = HybridRAG(self.root / "fast.sqlite3", self.embedder, fast_cosine=True)
            fast.ingest(self.conversation)
            observed = [
                fast.retrieve(question, budget).unit_ids
                for question in self.conversation.questions
                for budget in (120, 600)
            ]
            fast.close()
        finally:
            hybrid_baseline.cosine_similarity = reference_cosine
        self.assertEqual(observed, expected)

    def test_repeated_turn_ids_across_conversations_stay_separate(self) -> None:
        import dataclasses

        other = dataclasses.replace(
            self.conversation,
            namespace="locomo/other",
            questions=tuple(
                dataclasses.replace(question, namespace="locomo/other")
                for question in self.conversation.questions
            ),
        )
        for system in (
            BM25Chunks(self.root / "two.sqlite3"),
            HybridRAG(self.root / "two-hybrid.sqlite3", self.embedder, fast_cosine=False),
            EviCortexSystem(self.root / "two-evicortex.sqlite3", self.embedder),
        ):
            try:
                system.ingest(self.conversation)
                system.ingest(other)
                for conversation in (self.conversation, other):
                    retrieved = system.retrieve(conversation.questions[0], 2000)
                    self.assertTrue(retrieved.unit_ids)
                    self.assertTrue(all("|" not in unit_id for unit_id in retrieved.unit_ids))
            finally:
                system.close()

    def test_controls(self) -> None:
        question = self.conversation.questions[1]
        for system in (NoMemory(), FullHistory(), OracleEvidence(by_session=False)):
            system.ingest(self.conversation)
        empty = NoMemory()
        empty.ingest(self.conversation)
        self.assertEqual(empty.retrieve(question, 100).context, "")
        full = FullHistory()
        full.ingest(self.conversation)
        self.assertEqual(len(full.retrieve(question, 1).unit_ids), 24)
        oracle = OracleEvidence(by_session=False)
        oracle.ingest(self.conversation)
        self.assertEqual(oracle.retrieve(question, 1).unit_ids, ("D1:2", "D2:2"))

    def test_evidence_metrics(self) -> None:
        question = self.conversation.questions[1]
        session_of = {unit.unit_id: unit.session_id for unit in self.conversation.units}
        half = evidence_metrics(question, Retrieved("x", ("D1:2", "D3:1")), session_of)
        self.assertEqual(half["unit_recall"], 0.5)
        self.assertFalse(half["all_gold_units"])
        self.assertEqual(half["session_recall"], 0.5)
        unattributed = evidence_metrics(question, Retrieved("x", (), attributable=False), session_of)
        self.assertIsNone(unattributed["unit_recall"])
        abstention = evidence_metrics(self.conversation.questions[3], Retrieved("x", ("D1:1",)), session_of)
        self.assertIsNone(abstention["unit_recall"])


class LLMTests(unittest.TestCase):
    def test_cache_replays_without_calling_the_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            client = _FakeClient()
            cached = CachedLLM(client, path)
            first = cached.complete(system="s", prompt="p")
            second = cached.complete(system="s", prompt="p")
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(client.calls, 1)
            reopened = CachedLLM(_FakeClient(), path)
            self.assertTrue(reopened.complete(system="s", prompt="p").cached)
            self.assertEqual(reopened.usage()["calls"], 1)

    def test_identical_concurrent_requests_share_one_provider_call(self) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        release = threading.Event()

        class Slow(_FakeClient):
            def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
                self.calls += 1
                release.wait(timeout=5)
                return LLMResponse(f"answer {self.calls}", "model", 1, 1, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as directory:
            client = Slow()
            cached = CachedLLM(client, Path(directory) / "cache.jsonl")
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(cached.complete, system="s", prompt="p") for _ in range(4)]
                release.set()
                answers = {future.result().text for future in futures}
        self.assertEqual(client.calls, 1)
        self.assertEqual(answers, {"answer 1"})

    def test_a_failed_call_is_not_cached_and_can_be_retried(self) -> None:
        class Flaky(_FakeClient):
            def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("provider down")
                return LLMResponse("ok", "model", 1, 1, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as directory:
            cached = CachedLLM(Flaky(), Path(directory) / "cache.jsonl")
            with self.assertRaises(RuntimeError):
                cached.complete(system="s", prompt="p")
            self.assertEqual(cached.complete(system="s", prompt="p").text, "ok")

    def test_usage_can_be_limited_to_requests_this_instance_made(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            first = CachedLLM(_FakeClient(), path)
            first.complete(system="s", prompt="one")
            first.complete(system="s", prompt="two")
            shard = CachedLLM(_FakeClient(), path)
            shard.complete(system="s", prompt="one")
            self.assertEqual(shard.usage()["calls"], 2)
            self.assertEqual(shard.usage(used_only=True)["calls"], 1)

    def test_unknown_provider_cost_is_not_reported_as_zero(self) -> None:
        class Unpriced(_FakeClient):
            def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
                return LLMResponse("ok", "model", 3, 1, None, 1.0)

        with tempfile.TemporaryDirectory() as directory:
            cached = CachedLLM(Unpriced(), Path(directory) / "cache.jsonl")
            cached.complete(system="s", prompt="p")
            self.assertIsNone(cached.usage()["cost_usd"])
            self.assertEqual(cached.usage()["input_tokens"], 3)

    def test_provider_specs(self) -> None:
        self.assertEqual(make_client("claude-cli:m").identity, "claude-cli:m")
        self.assertEqual(make_client("anthropic:m").identity, "anthropic:m")
        self.assertEqual(make_client("openai:m").identity, "openai:m")
        local = make_client("openai-compatible:http://localhost:11434/v1#llama3")
        self.assertEqual(local.identity, "openai-compatible:http://localhost:11434/v1#llama3")
        with self.assertRaises(ValueError):
            make_client("unknown:m")
        with self.assertRaises(ValueError):
            make_client("openai-compatible:http://localhost:11434/v1")


class EvaluateTests(unittest.TestCase):
    def test_verdict_parsing(self) -> None:
        text = 'Here: [{"id": 3, "verdict": "correct"}, {"id": 4, "verdict": "Incorrect"}]'
        self.assertEqual(evaluate._parse_verdicts(text, [3, 4]), {3: "correct", 4: "incorrect"})
        self.assertIsNone(evaluate._parse_verdicts(text, [3]))
        self.assertIsNone(evaluate._parse_verdicts("no json", [3]))
        self.assertIsNone(evaluate._parse_verdicts('[{"id": 3, "verdict": "maybe"}]', [3]))

    def test_sampling_is_seeded_and_stratified(self) -> None:
        rows = [
            {"dataset": "locomo", "question_id": f"c{conversation}:q{index}",
             "namespace": f"locomo/c{conversation}", "category": category}
            for conversation in range(2)
            for category in ("single-hop", "temporal", "adversarial")
            for index in range({"single-hop": 0, "temporal": 100, "adversarial": 200}[category],
                               {"single-hop": 0, "temporal": 100, "adversarial": 200}[category] + 20)
        ]
        first = evaluate.select_questions(rows, seed=1)
        self.assertEqual(first, evaluate.select_questions(list(reversed(rows)), seed=1))
        self.assertNotEqual(first, evaluate.select_questions(rows, seed=2))
        self.assertEqual(len(first), 2 * (10 + 6 + 5))

    def test_judge_stage_grades_dont_know_by_rule_and_blinds_the_judge(self) -> None:
        seen: list[str] = []

        class Judge(_FakeClient):
            def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
                seen.append(prompt)
                ids = [int(line.split()[1]) for line in prompt.splitlines() if line.startswith("Item ")]
                body = json.dumps([{"id": number, "verdict": "correct"} for number in ids])
                return LLMResponse(body, "judge", 1, 1, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            answers = [
                {"key": "q1|evicortex|1000", "system": "evicortex", "budget": 1000,
                 "question": "Q1?", "gold_answer": "G1", "answer": "I don't know.",
                 "is_abstention": False},
                {"key": "q2|mem0|1000", "system": "mem0", "budget": 1000,
                 "question": "Q2?", "gold_answer": ABSTENTION_GOLD, "answer": "I don't know",
                 "is_abstention": True},
                {"key": "q4|dense_rag|1000", "system": "dense_rag", "budget": 1000,
                 "question": "Q4?", "gold_answer": ABSTENTION_GOLD, "answer": "  ",
                 "is_abstention": True},
                {"key": "q3|hybrid_rag|1000", "system": "hybrid_rag", "budget": 1000,
                 "question": "Q3?", "gold_answer": "G3", "answer": "G3 indeed",
                 "is_abstention": False},
            ]
            (run / "answers.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in answers), encoding="utf-8"
            )
            original = evaluate.make_client
            evaluate.make_client = lambda spec: Judge()
            try:
                evaluate.stage_judge(
                    argparse.Namespace(run=str(run), judge="fake:judge", seed=1, batch_size=10, workers=1)
                )
            finally:
                evaluate.make_client = original
            graded = {row["key"]: row for row in evaluate._read_jsonl(run / "judgments.jsonl")}
        self.assertEqual(graded["q1|evicortex|1000"]["verdict"], "incorrect")
        self.assertEqual(graded["q2|mem0|1000"]["verdict"], "correct")
        self.assertEqual(graded["q3|hybrid_rag|1000"]["verdict"], "correct")
        self.assertTrue(graded["q1|evicortex|1000"]["abstained"])
        # An empty completion is a failed read, not a refusal, even when unanswerable.
        self.assertEqual(graded["q4|dense_rag|1000"]["verdict"], "incorrect")
        self.assertEqual(graded["q4|dense_rag|1000"]["graded_by"], "rule:empty-answer")
        self.assertFalse(graded["q4|dense_rag|1000"]["abstained"])
        self.assertEqual(len(seen), 1)
        for name in ("evicortex", "mem0", "hybrid_rag", "1000"):
            self.assertNotIn(name, seen[0])

    def test_audit_with_no_judge_graded_answers_reports_undefined_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            row = {"key": "q1|no_memory|1000", "system": "no_memory", "category": "single-hop",
                   "question": "Q?", "gold_answer": "G", "answer": "I don't know.",
                   "is_abstention": False, "abstained": True, "verdict": "incorrect",
                   "graded_by": "rule:exact-dont-know"}
            (run / "judgments.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            original = evaluate.make_client
            evaluate.make_client = lambda spec: _FakeClient()
            try:
                evaluate.stage_audit(argparse.Namespace(
                    run=str(run), auditor="fake:auditor", sample=10, seed=1, batch_size=10, workers=1))
            finally:
                evaluate.make_client = original
            audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["sampled_items"], 0)
        self.assertIsNone(audit["agreement"])
        self.assertIsNone(audit["cohen_kappa"])

    def test_cohen_kappa(self) -> None:
        agree = [("correct", "correct"), ("incorrect", "incorrect")] * 5
        self.assertAlmostEqual(evaluate.cohen_kappa(agree), 1.0)
        mixed = [("correct", "correct")] * 4 + [("correct", "incorrect")] * 1 + [("incorrect", "incorrect")] * 5
        self.assertAlmostEqual(evaluate.cohen_kappa(mixed), 0.8)
        self.assertIsNone(evaluate.cohen_kappa([]))


class ClaimWriterTests(unittest.TestCase):
    def test_successor_reuses_the_stored_spelling_of_its_key(self) -> None:
        from evicortex import EviCortex

        from benchmarks.competitive.claims import ClaimWriter, RawClaim
        from benchmarks.competitive.types import Unit

        def unit(index: int) -> Unit:
            return Unit(f"D1:{index}", "session_1", f"2023-05-0{index}T00:00:00.000000Z",
                        "1 May 2023", "Ada", f"turn {index}")

        writer = ClaimWriter()
        with EviCortex.open(":memory:") as memory:
            for index, raw in ((1, RawClaim("Ada", "home_city", "Lisbon")),
                               (2, RawClaim("ADA", "Home_City", "Paris"))):
                claims = writer.prepare(unit(index), [raw])
                record = memory.remember(
                    namespace="t", content=unit(index).rendered,
                    occurred_at=unit(index).occurred_at, claims=claims,
                )
                writer.committed(record)
        self.assertEqual(writer.superseded, 1)
        self.assertEqual((claims[0].subject, claims[0].predicate), ("Ada", "home_city"))
        self.assertIsNotNone(claims[0].supersedes)


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is optional")
class FigureTests(unittest.TestCase):
    def test_control_only_run_produces_no_figures_and_no_error(self) -> None:
        from benchmarks.competitive.figures import write_figures

        estimate = {"n": 1, "mean": 0.2, "ci95": None}
        payload = {"runs": {"LoCoMo-10": {
            "answers": {"questions": 1, "reader": "fake", "category_counts": {"single-hop": 1},
                        "cells": [{"system": "no_memory", "budget": 1000, "accuracy": estimate,
                                   "accuracy_by_category": {"single-hop": estimate}}]},
            "retrieval": {"questions": 1, "cells": [
                {"system": "no_memory", "budget": 1000, "unit_recall": {"mean": 0.0},
                 "session_recall": {"mean": 0.0}}]},
        }}}
        with tempfile.TemporaryDirectory() as directory:
            write_figures(payload, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


class ReportTests(unittest.TestCase):
    def test_cluster_bootstrap_is_deterministic_and_brackets_the_mean(self) -> None:
        rows = [
            {"namespace": f"c{cluster}", "question_id": f"c{cluster}q{index}",
             "verdict": "correct" if (cluster + index) % 3 else "incorrect"}
            for cluster in range(6)
            for index in range(10)
        ]
        first = report._estimate(rows, report._correct)
        self.assertEqual(first, report._estimate(rows, report._correct))
        low, high = first["ci95"]
        self.assertLessEqual(low, first["mean"])
        self.assertGreaterEqual(high, first["mean"])

    def test_paired_difference_uses_matching_questions(self) -> None:
        reference = [
            {"namespace": f"c{index % 3}", "question_id": f"q{index}", "verdict": "incorrect"}
            for index in range(9)
        ]
        system = [
            {"namespace": f"c{index % 3}", "question_id": f"q{index}", "verdict": "correct"}
            for index in range(6)
        ]
        difference = report._paired_difference(system, reference, report._correct)
        self.assertEqual(difference["n"], 6)
        self.assertEqual(difference["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
