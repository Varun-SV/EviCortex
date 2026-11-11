"""LLM claim extraction for the ``evicortex_claims`` configuration.

EviCortex stores structured, temporal claims next to exact evidence, but it ships
no extractor for natural conversation. This module is the smallest honest bridge:
one LLM call per session proposes claims, each tied to the turn that states it,
and the caller writes them through the public ``remember(claims=...)`` API.

Supersession is decided here, in plain code, because EviCortex never guesses it:
a claim supersedes the latest claim with the same subject and predicate when its
value differs and its turn is later. The extractor is shown the subject/predicate
keys already in use so an update can reuse a key instead of inventing a new one.

Any provider accepted by ``llm.make_client`` can be the extractor. Its calls,
tokens, and cost are reported as ingestion cost.

    python -m benchmarks.competitive.claims --data locomo10.json \
        --cache runs/locomo/claims_llm_cache.jsonl --workers 10
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

from evicortex import ClaimInput

from .llm import CachedLLM, make_client
from .types import Conversation, Unit

EXTRACTION_SYSTEM = (
    "You extract durable memory claims from one session of a conversation. Reply with "
    "only a JSON object: {\"claims\": [{\"turn\": \"<turn id>\", \"subject\": \"<name>\", "
    "\"predicate\": \"<snake_case attribute>\", \"value\": \"<short value>\"}]}\n"
    "Rules:\n"
    "- Be exhaustive. Record every concrete fact someone could later be asked about: "
    "events (what happened, when, where, with whom), plans, preferences, relationships, "
    "possessions, opinions, reasons, and biographical details, keeping names, places, "
    "objects, numbers, and dates. Most substantive turns yield at least one claim. Skip "
    "only greetings and small talk.\n"
    "- `turn` must be the id of the shown turn that states the fact. `subject` is the "
    "person or named thing the fact is about, written as it appears in the session.\n"
    "- Resolve relative dates against the session date and write absolute dates in the "
    "value (\"yesterday\" in a session dated 8 May 2023 becomes \"7 May 2023\"). Include "
    "the date of an event in its value whenever it is known.\n"
    "- `predicate` names one specific attribute that has a single current value for "
    "that subject, for example home_city, job_title, date_of_charity_race, "
    "pottery_class_plan, opinion_of_camping_trip. Never use generic predicates such as "
    "said, did, or mentioned.\n"
    "- When the session changes a fact whose subject and predicate appear under "
    "\"Keys already in use\", reuse exactly that subject and predicate with the new "
    "value. Otherwise create a new specific predicate.\n"
    "- At most 3 claims per turn. Values are self-contained and under 25 words."
)
MAX_CLAIMS_PER_TURN = 3
_PREDICATE = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class RawClaim:
    subject: str
    predicate: str
    value: str


def _session_prompt(date_text: str, units: Sequence[Unit], keys: Sequence[tuple[str, str]]) -> str:
    known = "\n".join(f"{subject} | {predicate}" for subject, predicate in keys) or "(none)"
    turns = "\n".join(f"{unit.unit_id} | {unit.speaker}: {unit.text}" for unit in units)
    return f"Session date: {date_text}\n\nKeys already in use:\n{known}\n\nTurns:\n{turns}"


def _parse_claims(text: str, valid_turns: set[str]) -> dict[str, list[RawClaim]]:
    items: list[Any] = []
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
            items = list(payload.get("claims") or ()) if isinstance(payload, dict) else []
        except json.JSONDecodeError:
            # One malformed object must not discard the session: salvage the
            # individually well-formed claim objects.
            for candidate in re.findall(r"\{[^{}]*\}", text):
                try:
                    items.append(json.loads(candidate))
                except json.JSONDecodeError:
                    continue
    by_turn: dict[str, list[RawClaim]] = defaultdict(list)
    for item in items:
        if not isinstance(item, dict):
            continue
        turn = str(item.get("turn", "")).strip()
        subject = " ".join(str(item.get("subject", "")).split())
        predicate = _PREDICATE.sub("_", str(item.get("predicate", "")).strip().lower()).strip("_")
        value = " ".join(str(item.get("value", "")).split())
        if turn not in valid_turns or not subject or not predicate or not value:
            continue
        if len(by_turn[turn]) < MAX_CLAIMS_PER_TURN:
            by_turn[turn].append(RawClaim(subject, predicate, value[:400]))
    return dict(by_turn)


class ClaimWriter:
    """Turns raw claims into ``ClaimInput`` values with explicit supersession."""

    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], tuple[str, str, str]] = {}
        self.superseded = 0
        self.restated = 0

    def prepare(self, unit: Unit, raw: Sequence[RawClaim]) -> tuple[ClaimInput, ...]:
        prepared: list[ClaimInput] = []
        seen: set[tuple[str, str]] = set()
        for claim in raw:
            key = (claim.subject.casefold(), claim.predicate.casefold())
            if key in seen:
                continue
            supersedes = None
            prior = self._latest.get(key)
            if prior is not None:
                prior_id, prior_value, prior_start = prior
                if prior_value.casefold() == claim.value.casefold():
                    self.restated += 1
                    continue
                if unit.occurred_at <= prior_start:
                    continue
                supersedes = prior_id
                self.superseded += 1
            seen.add(key)
            prepared.append(
                ClaimInput(
                    subject=claim.subject,
                    predicate=claim.predicate,
                    value=claim.value,
                    valid_from=unit.occurred_at,
                    confidence=0.8,
                    authority="llm-extraction",
                    supersedes=supersedes,
                )
            )
        return tuple(prepared)

    def committed(self, record: Any) -> None:
        for claim in record.claims:
            key = (claim.subject.casefold(), claim.predicate.casefold())
            self._latest[key] = (str(claim.id), claim.value, str(claim.valid_from))


class LLMClaimSource:
    def __init__(self, llm_spec: str, *, cache_path: str | Path, workers: int = 8) -> None:
        self._llm = CachedLLM(make_client(llm_spec), cache_path)
        self._workers = workers
        self._seconds = 0.0
        self._sessions = 0
        self._unparsed = 0
        self.config = {"llm": llm_spec, "unit": "one call per session"}

    def writer(self) -> ClaimWriter:
        return ClaimWriter()

    def claims_for(
        self, conversation: Conversation, stats: dict[str, int] | None = None
    ) -> dict[str, list[RawClaim]]:
        """Extract session by session so later sessions can see earlier keys."""

        started = time.perf_counter()
        sessions: dict[str, list[Unit]] = defaultdict(list)
        for unit in conversation.units:
            sessions[unit.session_id].append(unit)
        keys: dict[tuple[str, str], tuple[str, str]] = {}
        result: dict[str, list[RawClaim]] = {}
        for units in sessions.values():
            response = self._llm.complete(
                system=EXTRACTION_SYSTEM,
                prompt=_session_prompt(units[0].date_text, units, list(keys.values())),
                max_tokens=4000,
            )
            parsed = _parse_claims(response.text, {unit.unit_id for unit in units})
            self._sessions += 1
            if not parsed:
                self._unparsed += 1
            for unit in units:
                for claim in parsed.get(unit.unit_id, ()):
                    keys.setdefault(
                        (claim.subject.casefold(), claim.predicate.casefold()),
                        (claim.subject, claim.predicate),
                    )
            result.update(parsed)
        self._seconds += time.perf_counter() - started
        if stats is not None:
            stats["claims_extracted"] = stats.get("claims_extracted", 0) + sum(
                len(items) for items in result.values()
            )
        return result

    def cost(self) -> dict[str, Any]:
        # Only the requests this run made: a shared, prefilled cache must not be
        # charged in full to every shard.
        usage = self._llm.usage(used_only=True)
        return {
            "llm": self._llm.identity,
            "llm_calls": usage["calls"],
            "llm_input_tokens": usage["input_tokens"],
            "llm_output_tokens": usage["output_tokens"],
            "llm_cost_usd": usage["cost_usd"],
            "extraction_sessions": self._sessions,
            "sessions_without_parseable_claims": self._unparsed,
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Fill the extraction cache for LoCoMo, one worker per conversation."""

    from .data import load_locomo

    parser = argparse.ArgumentParser(description="Pre-extract claims for LoCoMo")
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--llm", default="claude-cli:claude-haiku-5-5")
    parser.add_argument("--workers", type=int, default=10)
    arguments = parser.parse_args(argv)
    conversations, _ = load_locomo(arguments.data)
    source = LLMClaimSource(arguments.llm, cache_path=arguments.cache)

    def extract(conversation: Conversation) -> tuple[str, int]:
        claims = source.claims_for(conversation)
        return conversation.namespace, sum(len(items) for items in claims.values())

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for namespace, count in pool.map(extract, conversations):
            print(f"{namespace}: {count} claims", flush=True)
    print(json.dumps(source.cost()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
