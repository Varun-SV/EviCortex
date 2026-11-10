"""Stage 1 of Phase D: ingest every system and record packed contexts.

    python -m benchmarks.competitive.retrieve --dataset locomo \
        --data /data/locomo10.json --out runs/locomo \
        --systems no_memory,full_history,oracle,bm25_chunks,dense_rag,hybrid_rag,evicortex \
        --budgets 500,1000,2000,4000

Writes ``retrieval.jsonl`` (one row per question, system, and budget, including
the exact context a reader would receive) and ``retrieval_meta.json``. Evidence
metrics are computed after hard-budget packing against original source turns.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import platform
import sqlite3
import tempfile
import time
from typing import Any

from evicortex.budget import estimate_tokens

from ..run import _code_revision
from .data import load_locomo, load_longmemeval_s
from .systems import (
    BM25Chunks,
    CachingEmbedder,
    DenseRAG,
    EviCortexSystem,
    FullHistory,
    HybridRAG,
    NoMemory,
    OracleEvidence,
)
from .types import Conversation, Question, Retrieved

ENCODER_MODEL = "unsloth/bge-small-en-v1.5"
ENCODER_REVISION = "7382f1122c10708a1faa0bbe548674a14b1ffe7e"
ENCODER_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ALL_SYSTEMS = (
    "no_memory",
    "full_history",
    "oracle",
    "bm25_chunks",
    "dense_rag",
    "hybrid_rag",
    "evicortex",
    "evicortex_claims",
    "mem0",
)


def build_embedder(max_length: int) -> CachingEmbedder:
    from evicortex.providers import TransformersEmbedder

    return CachingEmbedder(
        TransformersEmbedder(
            ENCODER_MODEL,
            revision=ENCODER_REVISION,
            pooling="cls",
            query_prefix=ENCODER_QUERY_PREFIX,
            document_prefix="",
            max_length=max_length,
            overflow_policy="truncate",
            batch_size=32,
            local_files_only=True,
        )
    )


def evidence_metrics(
    question: Question, retrieved: Retrieved, session_of: dict[str, str]
) -> dict[str, Any]:
    """Source-turn and source-session coverage of the packed context."""

    if question.is_abstention or not retrieved.attributable:
        return {"unit_recall": None, "all_gold_units": None, "session_recall": None}
    got_units = set(retrieved.unit_ids)
    got_sessions = {session_of[unit_id] for unit_id in got_units if unit_id in session_of}
    gold_units = set(question.gold_unit_ids)
    gold_sessions = set(question.gold_session_ids)
    return {
        "unit_recall": (len(gold_units & got_units) / len(gold_units)) if gold_units else None,
        "all_gold_units": (gold_units <= got_units) if gold_units else None,
        "session_recall": (
            len(gold_sessions & got_sessions) / len(gold_sessions) if gold_sessions else None
        ),
    }


def _build_systems(
    names: Sequence[str],
    *,
    dataset: str,
    root: Path,
    embedder: CachingEmbedder | None,
    arguments: argparse.Namespace,
) -> list[Any]:
    systems: list[Any] = []
    for name in names:
        if name == "no_memory":
            systems.append(NoMemory())
        elif name == "full_history":
            systems.append(FullHistory())
        elif name == "oracle":
            systems.append(OracleEvidence(by_session=dataset == "longmemeval"))
        elif name == "bm25_chunks":
            systems.append(BM25Chunks(root / "bm25.sqlite3"))
        elif name == "dense_rag":
            systems.append(DenseRAG(embedder))
        elif name == "hybrid_rag":
            systems.append(HybridRAG(root / "hybrid.sqlite3", embedder))
        elif name == "evicortex":
            systems.append(EviCortexSystem(root / "evicortex.sqlite3", embedder))
        elif name == "evicortex_claims":
            from .claims import LLMClaimSource

            systems.append(
                EviCortexSystem(
                    root / "evicortex_claims.sqlite3",
                    embedder,
                    claim_source=LLMClaimSource(
                        arguments.claims_llm,
                        cache_path=Path(arguments.out) / "claims_llm_cache.jsonl",
                        workers=arguments.workers,
                    ),
                )
            )
        elif name == "mem0":
            from .mem0_system import Mem0PerConversation

            systems.append(
                Mem0PerConversation(
                    work_root=Path(arguments.mem0_dir),
                    llm_spec=arguments.mem0_llm,
                )
            )
        else:
            raise ValueError(f"unknown system {name!r}")
    return systems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--systems", default="no_memory,full_history,oracle,bm25_chunks,dense_rag,hybrid_rag,evicortex")
    parser.add_argument("--budgets", default="500,1000,2000,4000")
    parser.add_argument("--lme-per-type", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--claims-llm", default="claude-cli:claude-haiku-5-5")
    parser.add_argument("--mem0-llm", default="claude-cli:claude-haiku-5-5")
    parser.add_argument("--mem0-dir", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default=None, help="suffix for the output file names")
    parser.add_argument(
        "--question-ids", default=None,
        help="sample.json from the read stage; only those questions are retrieved",
    )
    parser.add_argument(
        "--shard", default="0/1",
        help="k/n: process conversations whose index is k modulo n (parallel runs)",
    )
    arguments = parser.parse_args(argv)

    names = [name.strip() for name in arguments.systems.split(",") if name.strip()]
    budgets = sorted({int(value) for value in arguments.budgets.split(",")})
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    if arguments.dataset == "locomo":
        conversations, dataset_info = load_locomo(arguments.data)
    else:
        conversations, dataset_info = load_longmemeval_s(
            arguments.data, per_type=arguments.lme_per_type, seed=arguments.seed
        )
    if arguments.limit_conversations is not None:
        conversations = conversations[: arguments.limit_conversations]
    shard_index, shard_count = (int(part) for part in arguments.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard must be k/n with 0 <= k < n")
    conversations = [
        item for index, item in enumerate(conversations) if index % shard_count == shard_index
    ]
    suffix = "" if shard_count == 1 else f".shard-{shard_index}-of-{shard_count}"
    if arguments.tag:
        suffix = f".{arguments.tag}{suffix}"
    only_questions = (
        set(json.loads(Path(arguments.question_ids).read_text(encoding="utf-8"))["question_ids"])
        if arguments.question_ids
        else None
    )

    needs_embedder = any(
        name in ("dense_rag", "hybrid_rag", "evicortex", "evicortex_claims") for name in names
    )
    embedder = build_embedder(arguments.max_length) if needs_embedder else None

    query_seconds: dict[str, list[float]] = {name: [] for name in names}
    rows_written = 0
    with tempfile.TemporaryDirectory(prefix="evicortex-phase-d-") as directory:
        systems = _build_systems(
            names, dataset=arguments.dataset, root=Path(directory),
            embedder=embedder, arguments=arguments,
        )
        try:
            with (out / f"retrieval{suffix}.jsonl").open("w", encoding="utf-8") as handle:
                for index, conversation in enumerate(conversations):
                    started = time.perf_counter()
                    session_of = {unit.unit_id: unit.session_id for unit in conversation.units}
                    for system in systems:
                        system.ingest(conversation)
                    for question in conversation.questions:
                        if only_questions is not None and question.question_id not in only_questions:
                            continue
                        for system in systems:
                            done_exempt = False
                            for budget in budgets:
                                if system.budget_exempt and done_exempt:
                                    continue
                                clock = time.perf_counter()
                                retrieved = system.retrieve(question, budget)
                                elapsed = time.perf_counter() - clock
                                query_seconds[system.name].append(elapsed)
                                tokens = estimate_tokens(retrieved.context)
                                if not system.budget_exempt and tokens > budget:
                                    raise AssertionError(
                                        f"{system.name} exceeded budget {budget}: {tokens}"
                                    )
                                row = {
                                    "dataset": arguments.dataset,
                                    "question_id": question.question_id,
                                    "namespace": question.namespace,
                                    "category": question.category,
                                    "is_abstention": question.is_abstention,
                                    "system": system.name,
                                    "budget": None if system.budget_exempt else budget,
                                    "context_tokens": tokens,
                                    "retrieved_units": len(retrieved.unit_ids),
                                    "latency_ms": round(elapsed * 1000, 3),
                                    **evidence_metrics(question, retrieved, session_of),
                                    "extra": retrieved.extra,
                                    "question": question.text,
                                    "question_date_text": question.question_date_text,
                                    "gold_answer": question.answer,
                                    "context": retrieved.context,
                                }
                                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                rows_written += 1
                                done_exempt = True
                    print(
                        f"[{index + 1}/{len(conversations)}] {conversation.namespace}: "
                        f"{len(conversation.units)} units, {len(conversation.questions)} "
                        f"questions, {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
            system_meta = {
                system.name: {
                    "label": system.label,
                    "budget_exempt": system.budget_exempt,
                    "config": getattr(system, "config", {}),
                    "ingestion": system.ingestion_cost(),
                }
                for system in systems
            }
        finally:
            for system in systems:
                system.close()

    meta = {
        "schema_version": 1,
        "benchmark": "evicortex-phase-d-competitive",
        "stage": "retrieve",
        "dataset": dataset_info,
        "systems": system_meta,
        "budgets": budgets,
        "seed": arguments.seed,
        "shard": arguments.shard,
        "question_subset": arguments.question_ids is not None,
        "conversations_processed": len(conversations),
        "rows": rows_written,
        "token_estimator": "evicortex.budget.estimate_tokens (UTF-8 bytes / 3, rounded up)",
        "encoder": (
            {
                "model": ENCODER_MODEL,
                "revision": ENCODER_REVISION,
                "pooling": "cls",
                "query_prefix": ENCODER_QUERY_PREFIX,
                "max_length": arguments.max_length,
                "overflow_policy": "truncate",
                "identity": embedder.identity,
                "document_texts_encoded": embedder.document_texts_encoded,
                "query_texts_encoded": embedder.query_texts_encoded,
                "encode_seconds": round(embedder.encode_seconds, 2),
            }
            if embedder is not None
            else None
        ),
        "latency_note": (
            "Single pass on a shared 4-vCPU container. Query embeddings are cached "
            "across systems and budgets, so the first system to see a question pays "
            "the encoder cost. Latency here is descriptive only."
        ),
        "reproducibility": {
            "code_revision": _code_revision(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sqlite_version": sqlite3.sqlite_version,
        },
    }
    (out / f"retrieval_meta{suffix}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {rows_written} rows to {out / f'retrieval{suffix}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
