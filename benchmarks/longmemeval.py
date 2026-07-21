"""Strict adapter for the cleaned LongMemEval conversation-memory benchmark.

The official dataset remains an external artifact. This module validates its
released JSON schema, renders sessions deterministically for ingestion, and
computes transparent session-level retrieval metrics without a reader model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


LONGMEMEVAL_SOURCE = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_VARIANT = "longmemeval_s_cleaned.json"
LONGMEMEVAL_LICENSE = "MIT"
LONGMEMEVAL_ARTIFACT_SHA256 = MappingProxyType(
    {
        "longmemeval_s_cleaned.json": (
            "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
        ),
        "longmemeval_m_cleaned.json": (
            "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f"
        ),
        "longmemeval_oracle.json": (
            "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"
        ),
    }
)
_OFFICIAL_VARIANT_BY_SHA256 = {
    digest: variant for variant, digest in LONGMEMEVAL_ARTIFACT_SHA256.items()
}
_QUESTION_TYPES = frozenset(
    {
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "temporal-reasoning",
        "knowledge-update",
        "multi-session",
    }
)
_TIMESTAMP_PATTERN = re.compile(
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2}) "
    r"\((?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def parse_longmemeval_timestamp(value: str) -> datetime:
    """Parse one released benchmark timestamp into an ordered UTC value.

    LongMemEval timestamps do not include a real-world timezone. The adapter
    assigns UTC solely as a deterministic ordering convention required by the
    EviCortex temporal API; this must not be interpreted as location metadata.
    """

    if not isinstance(value, str):
        raise TypeError("LongMemEval timestamp must be a string")
    match = _TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            "LongMemEval timestamp must use YYYY/MM/DD (Day) HH:MM"
        )
    try:
        parsed = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=timezone.utc,
        )
    except ValueError as error:
        raise ValueError("LongMemEval timestamp contains an invalid date") from error
    if match.group("weekday") != _WEEKDAYS[parsed.weekday()]:
        raise ValueError("LongMemEval timestamp weekday does not match its date")
    return parsed


def official_longmemeval_variant(sha256: str) -> str | None:
    """Identify a byte-exact cleaned release artifact by its logical JSON hash."""

    if not isinstance(sha256, str):
        raise TypeError("sha256 must be a string")
    return _OFFICIAL_VARIANT_BY_SHA256.get(sha256.casefold())


@dataclass(frozen=True, slots=True)
class LongMemEvalTurn:
    role: str
    content: str
    has_answer: bool = False


@dataclass(frozen=True, slots=True)
class LongMemEvalSession:
    session_id: str
    date: str
    turns: tuple[LongMemEvalTurn, ...]

    @property
    def has_answer(self) -> bool:
        return any(turn.has_answer for turn in self.turns)

    def render(self) -> str:
        """Render one session without hiding dates, roles, or turn boundaries."""

        lines = [f"Session date: {self.date}"]
        lines.extend(f"{turn.role}: {turn.content}" for turn in self.turns)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LongMemEvalCase:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: tuple[LongMemEvalSession, ...]
    answer_session_ids: tuple[str, ...]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def namespace(self) -> str:
        return f"longmemeval/{self.question_id}"


@dataclass(frozen=True, slots=True)
class LongMemEvalDataset:
    path: Path
    sha256: str
    source_bytes: int
    cases: tuple[LongMemEvalCase, ...]

    @property
    def answerable_cases(self) -> tuple[LongMemEvalCase, ...]:
        return tuple(case for case in self.cases if not case.is_abstention)


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: object, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{path} must be non-empty")
    return value


def _case(value: object, index: int) -> LongMemEvalCase:
    path = f"cases[{index}]"
    item = _mapping(value, path)
    required = {
        "question_id",
        "question_type",
        "question",
        "answer",
        "question_date",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
        "answer_session_ids",
    }
    missing = sorted(required.difference(item))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")

    question_id = _string(item["question_id"], f"{path}.question_id")
    question_type = _string(item["question_type"], f"{path}.question_type")
    if question_type not in _QUESTION_TYPES:
        raise ValueError(f"{path}.question_type is not a released LongMemEval type")
    session_ids = _list(item["haystack_session_ids"], f"{path}.haystack_session_ids")
    dates = _list(item["haystack_dates"], f"{path}.haystack_dates")
    raw_sessions = _list(item["haystack_sessions"], f"{path}.haystack_sessions")
    if len({len(session_ids), len(dates), len(raw_sessions)}) != 1:
        raise ValueError(f"{path} haystack arrays must have equal lengths")

    sessions: list[LongMemEvalSession] = []
    observed_ids: set[str] = set()
    for session_index, (raw_id, raw_date, raw_turns) in enumerate(
        zip(session_ids, dates, raw_sessions, strict=True)
    ):
        session_path = f"{path}.haystack_sessions[{session_index}]"
        session_id = _string(
            raw_id,
            f"{path}.haystack_session_ids[{session_index}]",
        )
        if session_id in observed_ids:
            raise ValueError(f"{path} contains duplicate session id {session_id!r}")
        observed_ids.add(session_id)
        date = _string(raw_date, f"{path}.haystack_dates[{session_index}]")
        parse_longmemeval_timestamp(date)
        turns: list[LongMemEvalTurn] = []
        for turn_index, raw_turn in enumerate(_list(raw_turns, session_path)):
            turn_path = f"{session_path}[{turn_index}]"
            turn = _mapping(raw_turn, turn_path)
            role = _string(turn.get("role"), f"{turn_path}.role")
            if role not in ("user", "assistant"):
                raise ValueError(f"{turn_path}.role must be user or assistant")
            content = _string(
                turn.get("content"),
                f"{turn_path}.content",
                allow_empty=True,
            )
            has_answer = turn.get("has_answer", False)
            if not isinstance(has_answer, bool):
                raise ValueError(f"{turn_path}.has_answer must be a boolean")
            turns.append(LongMemEvalTurn(role, content, has_answer))
        sessions.append(LongMemEvalSession(session_id, date, tuple(turns)))

    answer_ids = tuple(
        _string(value, f"{path}.answer_session_ids[{answer_index}]")
        for answer_index, value in enumerate(
            _list(item["answer_session_ids"], f"{path}.answer_session_ids")
        )
    )
    if len(set(answer_ids)) != len(answer_ids):
        raise ValueError(f"{path}.answer_session_ids contains duplicates")
    unknown = sorted(set(answer_ids).difference(observed_ids))
    if unknown:
        raise ValueError(
            f"{path}.answer_session_ids references absent sessions: {', '.join(unknown)}"
        )
    is_abstention = question_id.endswith("_abs")
    if is_abstention and answer_ids:
        raise ValueError(f"{path} abstention question must not have answer sessions")
    if not is_abstention and not answer_ids:
        raise ValueError(f"{path} answerable question must have answer sessions")
    labelled_ids = {
        session.session_id for session in sessions if session.has_answer
    }
    if labelled_ids != set(answer_ids):
        raise ValueError(
            f"{path} has_answer session labels must exactly match answer_session_ids"
        )

    question_date = _string(item["question_date"], f"{path}.question_date")
    parse_longmemeval_timestamp(question_date)
    return LongMemEvalCase(
        question_id=question_id,
        question_type=question_type,
        question=_string(item["question"], f"{path}.question"),
        answer=_string(item["answer"], f"{path}.answer", allow_empty=True),
        question_date=question_date,
        sessions=tuple(sessions),
        answer_session_ids=answer_ids,
    )


def load_longmemeval(
    path: str | Path,
    *,
    limit: int | None = None,
) -> LongMemEvalDataset:
    """Load and validate a cleaned LongMemEval-schema JSON file.

    ``limit`` restricts validation/evaluation only. The complete JSON artifact
    is still read and decoded so its byte-exact digest can be verified.
    """

    source = Path(path)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer or None")
        if limit < 1:
            raise ValueError("limit must be at least 1")
    content = source.read_bytes()
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("LongMemEval input must be valid UTF-8 JSON") from error
    values = _list(raw, "root")
    selected = values if limit is None else values[:limit]
    cases = tuple(_case(value, index) for index, value in enumerate(selected))
    question_ids = [case.question_id for case in cases]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("LongMemEval question IDs must be unique")
    return LongMemEvalDataset(
        path=source.resolve(),
        sha256=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
        cases=cases,
    )


def session_retrieval_metrics(
    case: LongMemEvalCase,
    retrieved_session_ids: Sequence[str],
    *,
    cutoffs: Iterable[int] = (5, 10),
) -> dict[str, float | bool]:
    """Calculate explicit binary-relevance session metrics for one question.

    As in the official retrieval evaluation, abstention questions have no gold
    answer location and are skipped. ``recall_all`` is exact coverage of every
    gold session, while ``recall_any`` is the softer hit-any measure.
    """

    if not isinstance(case, LongMemEvalCase):
        raise TypeError("case must be a LongMemEvalCase")
    if isinstance(retrieved_session_ids, (str, bytes)):
        raise TypeError("retrieved_session_ids must be a sequence")
    retrieved: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(retrieved_session_ids):
        if not isinstance(value, str) or not value:
            raise ValueError(f"retrieved session id {index} must be non-empty")
        if value not in seen:
            seen.add(value)
            retrieved.append(value)
    normalized_cutoffs: list[int] = []
    for value in cutoffs:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("cutoffs must contain positive integers")
        if value not in normalized_cutoffs:
            normalized_cutoffs.append(value)
    if not normalized_cutoffs:
        raise ValueError("at least one cutoff is required")
    result: dict[str, float | bool] = {"skipped_abstention": case.is_abstention}
    if case.is_abstention:
        return result
    gold = set(case.answer_session_ids)
    if not gold:
        raise ValueError("answerable LongMemEval case has no answer_session_ids")
    for cutoff in normalized_cutoffs:
        top = retrieved[:cutoff]
        hits = [1 if item in gold else 0 for item in top]
        found = sum(hits)
        ideal_count = min(len(gold), cutoff)
        # Reproduce LongMemEval's released ``eval_utils.dcg`` exactly. It gives
        # ranks one and two weight 1, then discounts rank r by log2(r). This is
        # intentionally not the more common log2(r + 1) textbook convention.
        dcg = (
            float(hits[0])
            + sum(
                hit / math.log2(rank)
                for rank, hit in enumerate(hits[1:], start=2)
            )
            if hits
            else 0.0
        )
        ideal_hits = [1] * ideal_count
        idcg = (
            float(ideal_hits[0])
            + sum(
                hit / math.log2(rank)
                for rank, hit in enumerate(ideal_hits[1:], start=2)
            )
            if ideal_hits
            else 0.0
        )
        result[f"recall_fraction@{cutoff}"] = found / len(gold)
        result[f"recall_all@{cutoff}"] = float(found == len(gold))
        result[f"recall_any@{cutoff}"] = float(found > 0)
        # Match the official LongMemEval retrieval-log field name. At session
        # granularity a session is binary relevant when it contains any answer
        # turn, hence the upstream name ``ndcg_any``.
        result[f"ndcg_any@{cutoff}"] = dcg / idcg if idcg else 0.0
    return result


__all__ = [
    "LONGMEMEVAL_ARTIFACT_SHA256",
    "LONGMEMEVAL_LICENSE",
    "LONGMEMEVAL_SOURCE",
    "LONGMEMEVAL_VARIANT",
    "LongMemEvalCase",
    "LongMemEvalDataset",
    "LongMemEvalSession",
    "LongMemEvalTurn",
    "load_longmemeval",
    "official_longmemeval_variant",
    "parse_longmemeval_timestamp",
    "session_retrieval_metrics",
]
