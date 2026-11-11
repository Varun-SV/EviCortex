"""Mem0 (open-source ``mem0ai``) behind the Phase D adapter contract.

This runs the REAL ``mem0.Memory`` class, unmodified, fully locally:

* vector store: embedded Qdrant (``path=<work_dir>/qdrant``), no server;
* embedder:     Mem0's ``huggingface`` provider (sentence-transformers) with
                ``BAAI/bge-small-en-v1.5`` (384 dims);
* LLM:          Mem0 has no Claude-CLI provider, so a small ``LLMBase`` subclass is
                registered in ``LlmFactory`` that forwards ``generate_response`` to our
                ``CachedLLM`` (``claude -p``).  Mem0's prompts and add/search logic are
                untouched; the bridge only (a) splits the message list into
                ``system`` / ``prompt`` and (b) when Mem0 asks for a JSON object, adds a
                "respond with only a JSON object" instruction and strips code fences.
* telemetry:    ``MEM0_TELEMETRY=False`` is set before ``mem0`` is imported; no graph
                store is configured.

Concurrency / process model
---------------------------
Embedded Qdrant takes an exclusive lock on its directory and Mem0's history DB is a
single SQLite file, so ONE process may use a given ``work_dir`` at a time.  Different
processes are safe as long as each one is given its own ``work_dir`` (the intended
pattern: one process per conversation namespace, ``work_dir=<root>/<namespace>``).
Within a process, ``ingest`` is sequential.  The LLM disk cache
(``<work_dir>/llm_cache.jsonl``) is per ``work_dir`` too.

Resumability
------------
``<work_dir>/progress.json`` records every finished (namespace, batch index) together
with the batch's unit ids and its LLM cost.  Re-running ``ingest`` skips finished
batches.  A crash between ``Memory.add`` returning and the progress write can replay
one batch; the replay is served from the LLM cache and Mem0's own hash de-duplication
absorbs identical memories.

Importing this module imports neither ``mem0`` nor ``torch``; both load when
``Mem0System(...)`` is constructed.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import types
from typing import Any

from benchmarks.competitive.llm import CachedLLM, make_client
from benchmarks.competitive.types import Conversation, Question, Retrieved, Unit

_JSON_INSTRUCTION = (
    "\n\nIMPORTANT OUTPUT FORMAT: Respond with only a JSON object. Do not wrap it in "
    "markdown code fences and do not add any text before or after the JSON."
)
_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _clean_json_text(text: str) -> tuple[str, bool]:
    """Strip fences / surrounding prose.  Returns (text, parsed_ok)."""

    candidate = text.strip()
    match = _FENCE.match(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        json.loads(candidate, strict=False)
        return candidate, True
    except ValueError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        inner = candidate[start : end + 1]
        try:
            json.loads(inner, strict=False)
            return inner, True
        except ValueError:
            pass
    return candidate, False


class _LLMMeter:
    """Thread-safe counters for the calls Mem0 makes through the bridge."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.fresh_calls = 0  # not served from the disk cache
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.json_requests = 0
        self.json_parse_failures = 0
        self.fenced_responses = 0

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "calls": self.calls,
                "fresh_calls": self.fresh_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd,
                "json_requests": self.json_requests,
                "json_parse_failures": self.json_parse_failures,
                "fenced_responses": self.fenced_responses,
            }

    def add(self, **fields: float) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, getattr(self, key) + value)


def _render_prompt(messages: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """Split Mem0's chat messages into (system, prompt) for ``CachedLLM.complete``."""

    def text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # OpenAI-style content parts
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return "" if content is None else str(content)

    system_parts = [text_of(m.get("content")) for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) == 1 and rest[0].get("role") == "user":
        prompt = text_of(rest[0].get("content"))
    else:
        prompt = "\n\n".join(f"{m.get('role', 'user')}: {text_of(m.get('content'))}" for m in rest)
    return "\n\n".join(part for part in system_parts if part), prompt


def _install_llm_provider(llm: CachedLLM, meter: _LLMMeter, max_tokens: int, key: str) -> str:
    """Register a Mem0 LLM provider that forwards to ``llm``; returns its provider name.

    Mem0's ``LlmConfig`` validator only accepts a fixed provider list, so the caller
    builds the config with a valid name and swaps ``config.llm.provider`` afterwards;
    ``LlmFactory.create`` itself only needs the ``provider_to_class`` entry.
    """

    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.llms.base import LLMBase
    from mem0.utils.factory import LlmFactory

    class ClaudeCLIBridgeLLM(LLMBase):
        def __init__(self, config: Any = None) -> None:
            super().__init__(config)
            if not getattr(self.config, "model", None):
                self.config.model = llm.identity

        def generate_response(
            self,
            messages: list[dict[str, Any]],
            response_format: Any = None,
            tools: Any = None,
            tool_choice: str = "auto",
            **kwargs: Any,
        ) -> str:
            if tools:
                raise NotImplementedError("tool calling is not supported by the Claude CLI bridge")
            system, prompt = _render_prompt(messages)
            wants_json = bool(response_format) and (
                not isinstance(response_format, dict)
                or response_format.get("type") in (None, "json_object", "json_schema")
            )
            if wants_json:
                system += _JSON_INSTRUCTION
            reply = llm.complete(system=system, prompt=prompt, max_tokens=max_tokens)
            text = reply.text
            fenced = text.strip().startswith("```")
            parsed_ok = True
            if wants_json:
                text, parsed_ok = _clean_json_text(text)
            meter.add(
                calls=1,
                fresh_calls=0 if reply.cached else 1,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                cost_usd=float(reply.cost_usd or 0.0),
                json_requests=1 if wants_json else 0,
                json_parse_failures=0 if parsed_ok else 1,
                fenced_responses=1 if (wants_json and fenced) else 0,
            )
            return text

    provider = f"evx_claude_cli_{key}"
    module_name = f"_evx_mem0_bridge_{key}"
    module = types.ModuleType(module_name)
    module.ClaudeCLIBridgeLLM = ClaudeCLIBridgeLLM  # type: ignore[attr-defined]
    ClaudeCLIBridgeLLM.__module__ = module_name
    sys.modules[module_name] = module
    LlmFactory.provider_to_class[provider] = (f"{module_name}.ClaudeCLIBridgeLLM", BaseLlmConfig)
    return provider


def _batch_digest(batch: Sequence[Unit]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for unit in batch:
        digest.update(unit.rendered.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _split_batches(units: Sequence[Unit], batch_turns: int) -> list[list[Unit]]:
    """Group consecutive units of the same session into batches of ``batch_turns``."""

    batches: list[list[Unit]] = []
    current: list[Unit] = []
    for unit in units:
        if current and (unit.session_id != current[0].session_id or len(current) >= batch_turns):
            batches.append(current)
            current = []
        current.append(unit)
    if current:
        batches.append(current)
    return batches


class Mem0System:
    """Adapter running the real Mem0 OSS ``Memory`` (see module docstring)."""

    name = "mem0"

    def __init__(
        self,
        *,
        work_dir: str | Path,
        llm_spec: str = "claude-cli:claude-haiku-5-5",
        embed_model: str = "BAAI/bge-small-en-v1.5",
        batch_turns: int = 2,
        search_limit: int = 100,
    ) -> None:
        if batch_turns < 1:
            raise ValueError("batch_turns must be >= 1")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.llm_spec = llm_spec
        self.embed_model = embed_model
        self.batch_turns = batch_turns
        self.search_limit = search_limit

        # Must be set before ``mem0`` is first imported.
        os.environ["MEM0_TELEMETRY"] = "False"
        os.environ.setdefault("MEM0_DIR", str(self.work_dir / "mem0_home"))

        self._meter = _LLMMeter()
        self._llm = CachedLLM(make_client(llm_spec), self.work_dir / "llm_cache.jsonl")
        self._progress_path = self.work_dir / "progress.json"
        self._progress = self._load_progress()
        # A store is only reusable with the configuration that built it; otherwise
        # old memories would be reported under the new settings.
        fingerprint = {
            "llm_spec": llm_spec, "embed_model": embed_model, "batch_turns": batch_turns,
        }
        if self._progress["batches"] and self._progress.get("config") != fingerprint:
            raise RuntimeError(
                f"{self._progress_path} was built with {self._progress.get('config')}, "
                f"not {fingerprint}; use a fresh work_dir"
            )
        self._progress["config"] = fingerprint
        self._ingest_wall = 0.0
        self._closed = False

        from mem0 import Memory
        from mem0.configs.base import MemoryConfig
        from mem0.llms.configs import LlmConfig

        key = re.sub(r"\W", "_", str(self.work_dir.resolve()))[-60:] + f"_{id(self):x}"
        provider = _install_llm_provider(self._llm, self._meter, 4096, key)
        config = MemoryConfig(
            vector_store={
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_locomo",
                    "path": str(self.work_dir / "qdrant"),
                    "embedding_model_dims": 384,
                    "on_disk": True,
                },
            },
            embedder={
                "provider": "huggingface",
                "config": {"model": embed_model, "embedding_dims": 384},
            },
            llm=LlmConfig(provider="openai", config={"model": self._llm.identity}),
            history_db_path=str(self.work_dir / "history.db"),
        )
        config.llm.provider = provider  # bypass the static provider allow-list
        self._memory = Memory(config)

    # ------------------------------------------------------------------ progress
    def _load_progress(self) -> dict[str, Any]:
        if self._progress_path.exists():
            with self._progress_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {"batches": {}}

    def _save_progress(self) -> None:
        tmp = self._progress_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._progress, handle)
        os.replace(tmp, self._progress_path)

    # -------------------------------------------------------------------- ingest
    def ingest(self, conversation: Conversation) -> None:
        namespace = conversation.namespace
        for index, batch in enumerate(_split_batches(conversation.units, self.batch_turns)):
            key = f"{namespace}|{index}"
            unit_ids = [unit.unit_id for unit in batch]
            content_sha256 = _batch_digest(batch)
            done = self._progress["batches"].get(key)
            if done is not None:
                if done["unit_ids"] != unit_ids or done.get("content_sha256") != content_sha256:
                    raise RuntimeError(
                        f"progress file disagrees with conversation at {key} (turn ids or "
                        "turn text changed); use a fresh work_dir"
                    )
                continue
            messages = [{"role": "user", "content": unit.rendered} for unit in batch]
            before = self._meter.snapshot()
            started = time.perf_counter()
            result = self._memory.add(
                messages,
                user_id=namespace,
                metadata={
                    "session_id": batch[0].session_id,
                    "date": batch[0].date_text,
                    "unit_ids": ",".join(unit_ids),
                },
                infer=True,
            )
            wall = time.perf_counter() - started
            after = self._meter.snapshot()
            events = (result or {}).get("results", []) if isinstance(result, dict) else []
            self._progress["batches"][key] = {
                "unit_ids": unit_ids,
                "content_sha256": content_sha256,
                "wall_seconds": round(wall, 3),
                "memories_added": sum(1 for e in events if e.get("event") == "ADD"),
                "llm_calls": int(after["calls"] - before["calls"]),
                "fresh_llm_calls": int(after["fresh_calls"] - before["fresh_calls"]),
                "input_tokens": int(after["input_tokens"] - before["input_tokens"]),
                "output_tokens": int(after["output_tokens"] - before["output_tokens"]),
                "cost_usd": after["cost_usd"] - before["cost_usd"],
                "json_parse_failures": int(
                    after["json_parse_failures"] - before["json_parse_failures"]
                ),
            }
            self._save_progress()
            self._ingest_wall += wall

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        from evicortex.budget import estimate_tokens

        import inspect

        search = self._memory.search
        params = inspect.signature(search).parameters
        if "filters" in params and "top_k" in params:  # mem0 >= 2.x
            response = search(
                question.text, filters={"user_id": question.namespace}, top_k=self.search_limit
            )
        else:  # mem0 1.x
            response = search(question.text, user_id=question.namespace, limit=self.search_limit)
        results = response.get("results", []) if isinstance(response, dict) else list(response)

        kept: list[str] = []
        unit_ids: list[str] = []
        seen: set[str] = set()
        context = ""
        for item in results:
            memory_text = str(item.get("memory") or "").strip()
            if not memory_text:
                continue
            metadata = item.get("metadata") or {}
            date = metadata.get("date")
            line = f"- [{date}] {memory_text}" if date else f"- {memory_text}"
            candidate = "\n".join([*kept, line])
            if estimate_tokens(candidate) > budget_tokens:
                continue  # skip lines that do not fit; shorter lower-ranked ones may
            kept.append(line)
            context = candidate
            raw_ids = metadata.get("unit_ids") or ""
            ids = raw_ids if isinstance(raw_ids, list) else str(raw_ids).split(",")
            for unit_id in ids:
                unit_id = unit_id.strip()
                if unit_id and unit_id not in seen:
                    seen.add(unit_id)
                    unit_ids.append(unit_id)
        return Retrieved(
            context=context,
            unit_ids=tuple(unit_ids),
            # Memories carry the ids of the add() batch (<= batch_turns turns) they were
            # extracted from, a superset of the true source turn(s), not a per-turn link.
            # An empty context covers no evidence, which is a known zero, not unknown.
            attributable=bool(unit_ids) or not context,
            extra={
                "memories": len(kept),
                "candidates": len(results),
                "attribution": "batch-level",
            },
        )

    # ---------------------------------------------------------------------- cost
    def ingestion_cost(self) -> dict[str, Any]:
        """Cumulative over all runs recorded in ``progress.json`` (cache replays count)."""

        rows = list(self._progress["batches"].values())
        try:
            stored = int(
                self._memory.vector_store.client.count(
                    collection_name=self._memory.collection_name
                ).count
            )
        except Exception:  # noqa: BLE001 - best effort
            stored = -1
        return {
            "adds": len(rows),
            "llm_calls": sum(r["llm_calls"] for r in rows),
            "fresh_llm_calls_this_process": int(self._meter.snapshot()["fresh_calls"]),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "wall_seconds": round(sum(r["wall_seconds"] for r in rows), 2),
            "json_parse_failures": sum(r.get("json_parse_failures", 0) for r in rows),
            "memories_stored": stored,
            "memories_reported_added": sum(r["memories_added"] for r in rows),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for target in (
            getattr(getattr(self._memory.vector_store, "client", None), "close", None),
            getattr(getattr(self._memory.db, "connection", None), "close", None),
        ):
            try:
                if target:
                    target()
            except Exception:  # noqa: BLE001
                pass


def _work_dir_for(work_root: str | Path, namespace: str) -> Path:
    return Path(work_root) / namespace.replace("/", "__")


class Mem0PerConversation:
    """One ``Mem0System`` (and one ``work_dir``) per conversation namespace.

    Ingestion can then run as one process per conversation, and the retrieval
    stage re-opens each finished store in turn. Only one store is open at a time.
    """

    name = "mem0"
    label = "Mem0 OSS (mem0ai, LLM fact extraction at ingestion)"
    budget_exempt = False
    uses_embedder = False

    def __init__(self, *, work_root: str | Path, llm_spec: str = "claude-cli:claude-haiku-5-5") -> None:
        self._work_root = Path(work_root)
        self._llm_spec = llm_spec
        self._current: tuple[str, Mem0System] | None = None
        self._costs: dict[str, dict[str, Any]] = {}
        self.config = {"llm": llm_spec, "batch_turns": 2, "search_limit": 100}

    def _system(self, namespace: str) -> "Mem0System":
        if self._current is not None and self._current[0] == namespace:
            return self._current[1]
        if self._current is not None:
            self._costs[self._current[0]] = self._current[1].ingestion_cost()
            self._current[1].close()
        system = Mem0System(
            work_dir=_work_dir_for(self._work_root, namespace), llm_spec=self._llm_spec
        )
        self._current = (namespace, system)
        return system

    def ingest(self, conversation: Conversation) -> None:
        self._system(conversation.namespace).ingest(conversation)

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved:
        return self._system(question.namespace).retrieve(question, budget_tokens)

    def ingestion_cost(self) -> dict[str, Any]:
        if self._current is not None:
            self._costs[self._current[0]] = self._current[1].ingestion_cost()
        total: dict[str, Any] = {}
        for cost in self._costs.values():
            for key, value in cost.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                total[key] = round(total.get(key, 0) + value, 6)
        total["conversations"] = len(self._costs)
        return total

    def close(self) -> None:
        if self._current is not None:
            self._costs[self._current[0]] = self._current[1].ingestion_cost()
            self._current[1].close()
            self._current = None


def main(argv: Sequence[str] | None = None) -> int:
    """Ingest one LoCoMo conversation: ``python -m ...mem0_system --data F --root D --index 0``."""

    import argparse

    from benchmarks.competitive.data import load_locomo

    parser = argparse.ArgumentParser(description="Ingest one LoCoMo conversation into Mem0")
    parser.add_argument("--data", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--llm", default="claude-cli:claude-haiku-5-5")
    arguments = parser.parse_args(argv)
    conversations, _ = load_locomo(arguments.data)
    conversation = conversations[arguments.index]
    system = Mem0System(
        work_dir=_work_dir_for(arguments.root, conversation.namespace), llm_spec=arguments.llm
    )
    try:
        system.ingest(conversation)
        print(json.dumps({"namespace": conversation.namespace, **system.ingestion_cost()}))
    finally:
        system.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
