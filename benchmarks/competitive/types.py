"""Shared value types and the adapter contract for Phase D systems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Unit:
    """One original conversation turn: the evidence unit shared by every system."""

    unit_id: str  # stable source ID used by gold annotations (e.g. LoCoMo "D3:7")
    session_id: str
    occurred_at: str  # ISO-8601 UTC ordering timestamp
    date_text: str  # the dataset's own human-readable session date
    speaker: str
    text: str  # raw turn text (plus any dataset-provided image caption)
    turn_index: int = 0

    @property
    def rendered(self) -> str:
        """The exact string every system stores for this turn."""

        return f"[{self.date_text}] {self.speaker}: {self.text}"


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    namespace: str
    text: str
    answer: str  # gold answer; never shown to a memory system or the reader
    category: str
    gold_unit_ids: tuple[str, ...] = ()
    gold_session_ids: tuple[str, ...] = ()
    question_date: str | None = None  # ISO-8601; retrieval must not see later turns
    question_date_text: str | None = None
    is_abstention: bool = False


@dataclass(frozen=True, slots=True)
class Conversation:
    namespace: str
    units: tuple[Unit, ...]
    questions: tuple[Question, ...]
    speakers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Retrieved:
    """What a system would place in the reader prompt for one question."""

    context: str
    unit_ids: tuple[str, ...]  # source turns represented in ``context``, in rank order
    attributable: bool = True  # False when the system cannot map text to source turns
    extra: dict[str, Any] = field(default_factory=dict)


class MemorySystem(Protocol):
    """Adapter contract. ``retrieve`` must respect ``budget_tokens`` as a hard cap
    on ``evicortex.budget.estimate_tokens(context)``."""

    name: str

    def ingest(self, conversation: Conversation) -> None: ...

    def retrieve(self, question: Question, budget_tokens: int) -> Retrieved: ...

    def ingestion_cost(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def units_by_id(units: Sequence[Unit]) -> dict[str, Unit]:
    return {unit.unit_id: unit for unit in units}
