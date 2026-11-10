"""Dataset adapters for Phase D: LoCoMo-10 and the cleaned LongMemEval-S release.

Neither dataset is redistributed with this repository. Both loaders verify the
SHA-256 of the caller-supplied file and report whether it matches the pinned
public artifact, so a result can always be tied to exact bytes.

Gold labels (answers, evidence turn IDs, ``has_answer`` flags) are attached to
``Question`` objects only. They are never part of a ``Unit`` and therefore can
never be ingested by a memory system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from ..longmemeval import LONGMEMEVAL_ARTIFACT_SHA256, parse_longmemeval_timestamp
from .types import Conversation, Question, Unit

LOCOMO_SOURCE = "https://github.com/snap-research/locomo (data/locomo10.json)"
LOCOMO_LICENSE = "CC BY-NC 4.0"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
ABSTENTION_GOLD = "The conversation does not contain this information."

_DIA_ID = re.compile(r"D\d+:\d+")
_SESSION_KEY = re.compile(r"session_(\d+)")


@dataclass(frozen=True, slots=True)
class _Session:
    session_id: str
    date: str
    turns: tuple[tuple[str, str, bool], ...]


@dataclass(frozen=True, slots=True)
class _Case:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: tuple[_Session, ...]
    answer_session_ids: tuple[str, ...]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _locomo_time(value: str) -> datetime:
    # e.g. "1:56 pm on 8 May, 2023". The dataset has no timezone; UTC is assigned
    # purely as a deterministic ordering convention.
    return datetime.strptime(value.strip(), "%I:%M %p on %d %B, %Y").replace(
        tzinfo=timezone.utc
    )


def load_locomo(path: str | Path) -> tuple[list[Conversation], dict[str, Any]]:
    """Load LoCoMo-10 as conversations of turn-level units plus questions."""

    source = Path(path)
    sha256 = file_sha256(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    conversations: list[Conversation] = []
    for sample in raw:
        sample_id = str(sample["sample_id"])
        namespace = f"locomo/{sample_id}"
        body = sample["conversation"]
        session_numbers = sorted(
            int(match.group(1))
            for key in body
            if (match := _SESSION_KEY.fullmatch(key)) is not None
        )
        units: list[Unit] = []
        for number in session_numbers:
            date_text = str(body[f"session_{number}_date_time"]).strip()
            started = _locomo_time(date_text)
            for index, turn in enumerate(body[f"session_{number}"]):
                text = str(turn["text"])
                caption = turn.get("blip_caption")
                if caption:
                    text = f"{text} [shared image: {caption}]"
                units.append(
                    Unit(
                        unit_id=str(turn["dia_id"]),
                        session_id=f"session_{number}",
                        # One second per turn keeps within-session order explicit.
                        occurred_at=_iso(started + timedelta(seconds=index)),
                        date_text=date_text,
                        speaker=str(turn["speaker"]),
                        text=text,
                        turn_index=index,
                    )
                )
        known = {unit.unit_id for unit in units}
        session_of = {unit.unit_id: unit.session_id for unit in units}
        questions: list[Question] = []
        for index, item in enumerate(sample["qa"]):
            category = int(item["category"])
            gold_units = tuple(
                dict.fromkeys(
                    dia_id
                    for entry in item.get("evidence") or ()
                    for dia_id in _DIA_ID.findall(str(entry))
                    if dia_id in known
                )
            )
            adversarial = category == 5
            questions.append(
                Question(
                    question_id=f"{sample_id}:q{index:03d}",
                    namespace=namespace,
                    text=str(item["question"]),
                    answer=ABSTENTION_GOLD if adversarial else str(item["answer"]),
                    category=LOCOMO_CATEGORIES[category],
                    gold_unit_ids=gold_units,
                    gold_session_ids=tuple(
                        dict.fromkeys(session_of[unit_id] for unit_id in gold_units)
                    ),
                    is_abstention=adversarial,
                )
            )
        conversations.append(
            Conversation(
                namespace=namespace,
                units=tuple(units),
                questions=tuple(questions),
                speakers=(str(body["speaker_a"]), str(body["speaker_b"])),
            )
        )
    info = {
        "name": "LoCoMo-10",
        "source": LOCOMO_SOURCE,
        "license": LOCOMO_LICENSE,
        "sha256": sha256,
        "pinned_artifact_verified": sha256 == LOCOMO_SHA256,
        "conversations": len(conversations),
        "units": sum(len(item.units) for item in conversations),
        "questions": sum(len(item.questions) for item in conversations),
        "evidence_unit": "one conversation turn (dia_id)",
        "abstention_note": (
            "Category-5 adversarial questions are treated as unanswerable: the gold "
            "behaviour is to say the conversation does not contain the information."
        ),
    }
    return conversations, info


def load_longmemeval_s(
    path: str | Path,
    *,
    per_type: int | None,
    seed: int,
) -> tuple[list[Conversation], dict[str, Any]]:
    """Load LongMemEval-S; each question is its own namespace and haystack.

    ``per_type`` draws a seeded stratified sample of that many questions per
    released question type (``None`` keeps all 500).
    """

    source = Path(path)
    sha256 = file_sha256(source)
    raw_cases = json.loads(source.read_text(encoding="utf-8"))
    renamed_sessions = 0
    cases: list[_Case] = []
    for item in raw_cases:
        # The official release repeats a session ID inside 13 haystacks, each time
        # for a different, non-answer session. They are kept as distinct sessions
        # by suffixing later occurrences; answer sessions are never affected.
        seen: dict[str, int] = {}
        sessions: list[_Session] = []
        for session_id, date, turns in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"],
            strict=True,
        ):
            count = seen.get(session_id, 0) + 1
            seen[session_id] = count
            if count > 1:
                renamed_sessions += 1
                session_id = f"{session_id}#dup{count}"
            sessions.append(
                _Session(
                    session_id=str(session_id),
                    date=str(date),
                    turns=tuple(
                        (str(turn["role"]), str(turn["content"]), bool(turn.get("has_answer", False)))
                        for turn in turns
                    ),
                )
            )
        cases.append(
            _Case(
                question_id=str(item["question_id"]),
                question_type=str(item["question_type"]),
                question=str(item["question"]),
                answer=str(item["answer"]),
                question_date=str(item["question_date"]),
                sessions=tuple(sessions),
                answer_session_ids=tuple(str(value) for value in item["answer_session_ids"]),
            )
        )
    total_cases = len(cases)
    if per_type is not None:
        rng = random.Random(seed)
        by_type: dict[str, list[_Case]] = {}
        for case in cases:
            by_type.setdefault(case.question_type, []).append(case)
        selected = []
        for question_type in sorted(by_type):
            pool = sorted(by_type[question_type], key=lambda case: case.question_id)
            rng.shuffle(pool)
            selected.extend(pool[:per_type])
        cases = sorted(selected, key=lambda case: case.question_id)

    conversations: list[Conversation] = []
    for case in cases:
        namespace = f"longmemeval/{case.question_id}"
        units: list[Unit] = []
        gold_units: list[str] = []
        ordered = sorted(
            case.sessions,
            key=lambda session: (parse_longmemeval_timestamp(session.date), session.session_id),
        )
        for session in ordered:
            started = parse_longmemeval_timestamp(session.date)
            for index, (role, content, has_answer) in enumerate(session.turns):
                if not content.strip():
                    continue
                unit_id = f"{session.session_id}:t{index}"
                units.append(
                    Unit(
                        unit_id=unit_id,
                        session_id=session.session_id,
                        occurred_at=_iso(started + timedelta(seconds=index)),
                        date_text=session.date,
                        speaker=role,
                        text=content,
                        turn_index=index,
                    )
                )
                if has_answer:
                    gold_units.append(unit_id)
        question_time = parse_longmemeval_timestamp(case.question_date)
        is_abstention = case.question_id.endswith("_abs")
        question = Question(
            question_id=case.question_id,
            namespace=namespace,
            text=case.question,
            answer=ABSTENTION_GOLD if is_abstention else str(case.answer),
            category=case.question_type,
            gold_unit_ids=tuple(gold_units),
            gold_session_ids=tuple(case.answer_session_ids),
            question_date=_iso(question_time + timedelta(hours=23, minutes=59)),
            question_date_text=case.question_date,
            is_abstention=is_abstention,
        )
        conversations.append(
            Conversation(namespace=namespace, units=tuple(units), questions=(question,))
        )
    info = {
        "name": "LongMemEval-S (cleaned)",
        "source": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "license": "MIT",
        "sha256": sha256,
        "pinned_artifact_verified": sha256 in LONGMEMEVAL_ARTIFACT_SHA256.values(),
        "total_questions_in_release": total_cases,
        "selected_questions": len(cases),
        "sampling": (
            "all questions"
            if per_type is None
            else f"seeded stratified sample: {per_type} per question type, seed {seed}"
        ),
        "units": sum(len(item.units) for item in conversations),
        "evidence_unit": "one conversation turn; gold session = answer_session_ids",
        "duplicate_session_ids_renamed_in_release": renamed_sessions,
    }
    return conversations, info
