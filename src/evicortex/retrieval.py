"""Deterministic multi-channel retrieval and bounded context compilation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
import inspect
import json
import math
import re
from typing import Any, Literal

from .budget import BudgetTooSmallError, estimate_tokens, validate_budget
from .errors import EmbeddingProfileNotFoundError
from .plugins import (
    Embedder,
    EmbeddingValidationError,
    RerankCandidate,
    Reranker,
    embed_query_text,
    snapshot_reranker_identity,
    validate_embedding_batch,
    validate_rerank_results,
)
from .semantic import EmbeddingProfile


_RRF_K = 60
_MAX_EPISODE_EXCERPT_CHARS = 640
_MIN_TRUNCATED_EXCERPT_CHARS = 24
_WORD_RE = re.compile(r"[\w:-]+", re.UNICODE)
_DIAGNOSTIC_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_DIAGNOSTIC_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "before",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "s",
        "should",
        "the",
        "to",
        "was",
        "what",
        "when",
        "which",
        "who",
        "with",
    }
)
_QUERY_NEGATION_WORDS = frozenset(
    {"not", "no", "without", "excluding", "except"}
)
_QUERY_ARTICLES = frozenset({"a", "an", "the"})
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ScoreReason:
    """One auditable contribution to a fused retrieval score."""

    channel: str
    rank: int
    contribution: float
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "rank": self.rank,
            "contribution": self.contribution,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CapsuleFact:
    """A compact semantic claim linked to exact supporting evidence."""

    claim_id: str
    subject: str
    predicate: str
    value: str
    status: str
    valid_from: str | None
    valid_to: str | None
    confidence: float
    evidence_ids: tuple[str, ...]
    score: float
    reasons: tuple[ScoreReason, ...] = ()

    def prompt_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "evidence_ids": list(self.evidence_ids),
        }
        if self.status.casefold() != "active":
            value["status"] = self.status
        if self.valid_from is not None:
            value["valid_from"] = self.valid_from
        if self.valid_to is not None:
            value["valid_to"] = self.valid_to
        if self.confidence != 1.0:
            value["confidence"] = self.confidence
        return value


@dataclass(frozen=True, slots=True)
class CapsuleEpisode:
    """A compact excerpt from an immutable evidence record."""

    memory_id: str
    kind: str
    excerpt: str
    source: str | None
    occurred_at: str | None
    recorded_at: str | None
    content_hash: str | None
    score: float
    reasons: tuple[ScoreReason, ...] = ()
    truncated: bool = False

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return (self.memory_id,)

    def prompt_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "evidence_id": self.memory_id,
            "excerpt": self.excerpt,
        }
        if self.kind != "episode":
            value["kind"] = self.kind
        if self.source is not None:
            value["source"] = self.source
        if self.occurred_at is not None:
            value["occurred_at"] = self.occurred_at
        if self.truncated:
            value["truncated"] = True
        return value


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """Competing claim values that should be shown rather than silently merged."""

    subject: str
    predicate: str
    values: tuple[str, ...]
    claim_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "values": list(self.values),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Final fused score and selection decision for one unique candidate."""

    candidate_id: str
    candidate_type: Literal["fact", "episode"]
    score: float
    reasons: tuple[ScoreReason, ...]
    selected: bool
    omission_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "score": self.score,
            "reasons": [reason.as_dict() for reason in self.reasons],
            "selected": self.selected,
            "omission_reason": self.omission_reason,
        }


@dataclass(frozen=True, slots=True)
class CapsuleDiagnostic:
    """Non-prompt operational fact about an optional retrieval channel."""

    code: str
    channel: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "channel": self.channel,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class MemoryCapsule:
    """A deterministic, prompt-ready projection of retrieved memory."""

    namespace: str
    query: str
    budget_tokens: int
    as_of: str | None
    facts: tuple[CapsuleFact, ...] = ()
    episodes: tuple[CapsuleEpisode, ...] = ()
    conflicts: tuple[MemoryConflict, ...] = ()
    omitted_count: int = 0
    trace: tuple[RetrievalTrace, ...] = ()
    diagnostics: tuple[CapsuleDiagnostic, ...] = ()

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        evidence: set[str] = set()
        for fact in self.facts:
            evidence.update(fact.evidence_ids)
        for episode in self.episodes:
            evidence.add(episode.memory_id)
        return tuple(sorted(evidence))

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.to_prompt())

    def to_prompt(self) -> str:
        """Serialize only model-useful memory, with stable deterministic ordering."""

        payload: dict[str, Any] = {"v": "ec1"}
        if self.as_of is not None:
            payload["as_of"] = self.as_of
        if self.facts:
            payload["facts"] = [fact.prompt_dict() for fact in self.facts]
        if self.episodes:
            payload["episodes"] = [episode.prompt_dict() for episode in self.episodes]
        if self.conflicts:
            payload["conflicts"] = [
                conflict.prompt_dict() for conflict in self.conflicts
            ]
        if self.omitted_count:
            payload["omitted_count"] = self.omitted_count
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(slots=True)
class _Candidate:
    stable_id: str
    candidate_type: Literal["fact", "episode"]
    item: CapsuleFact | CapsuleEpisode
    reasons: list[ScoreReason] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(reason.contribution for reason in self.reasons)


def _get(value: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is _MISSING:
        joined = ", ".join(names)
        raise AttributeError(f"{type(value).__name__} has none of: {joined}")
    return default


def _unwrap_hit(value: Any, candidate_type: Literal["fact", "episode"]) -> Any:
    names = ("claim", "record", "item") if candidate_type == "fact" else (
        "memory",
        "record",
        "item",
    )
    return _get(value, *names, default=value)


def _record_namespace(
    value: Any,
    candidate_type: Literal["fact", "episode"],
) -> str | None:
    record = _unwrap_hit(value, candidate_type)
    namespace = _get(record, "namespace", default=None)
    return None if namespace is None else str(namespace)


def _as_iso(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        normalized = value
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    else:
        raise TypeError("as_of and validity timestamps must be ISO strings or datetimes")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _evidence_ids(record: Any) -> tuple[str, ...]:
    raw = _get(record, "evidence_ids", default=())
    if isinstance(raw, str):
        values: Iterable[Any] = (raw,)
    else:
        values = raw or ()
    result = {str(value) for value in values if value is not None and str(value)}
    singular = _get(record, "memory_id", "evidence_id", default=None)
    if singular is not None and str(singular):
        result.add(str(singular))
    return tuple(sorted(result))


def _to_fact(hit: Any) -> CapsuleFact:
    record = _unwrap_hit(hit, "fact")
    claim_id = str(_get(record, "claim_id", "id"))
    confidence = float(_get(record, "confidence", default=1.0))
    return CapsuleFact(
        claim_id=claim_id,
        subject=str(_get(record, "subject")),
        predicate=str(_get(record, "predicate")),
        value=str(_get(record, "value", "object")),
        status=str(_get(record, "status", default="active")),
        valid_from=_as_iso(_get(record, "valid_from", default=None)),
        valid_to=_as_iso(_get(record, "valid_to", default=None)),
        confidence=confidence,
        evidence_ids=_evidence_ids(record),
        score=0.0,
    )


def _to_episode(hit: Any) -> CapsuleEpisode:
    record = _unwrap_hit(hit, "episode")
    memory_id = str(_get(record, "memory_id", "id"))
    content = str(_get(record, "content", "text", default=""))
    excerpt = content[:_MAX_EPISODE_EXCERPT_CHARS]
    truncated = len(excerpt) != len(content)
    if truncated:
        excerpt = excerpt.rstrip() + "…"
    return CapsuleEpisode(
        memory_id=memory_id,
        kind=str(_get(record, "kind", default="episode")),
        excerpt=excerpt,
        source=_optional_text(_get(record, "source", default=None)),
        occurred_at=_as_iso(_get(record, "occurred_at", default=None)),
        recorded_at=_as_iso(_get(record, "recorded_at", default=None)),
        content_hash=_optional_text(_get(record, "content_hash", default=None)),
        score=0.0,
        truncated=truncated,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _source_detail(hit: Any) -> str:
    parts: list[str] = []
    raw = _get(hit, "score", "rank_score", "rank", default=None)
    if raw is not None:
        try:
            parts.append(f"source_rank={float(raw):.12g}")
        except (TypeError, ValueError):
            parts.append(f"source_rank={raw}")
    reason = _get(hit, "reason", default=None)
    if reason:
        # Store adapters may report the wall-clock instant used for a "current"
        # SQL filter.  It is useful diagnostically but would make two identical
        # recalls compare unequal, so retain the mode while removing that volatile
        # timestamp from the deterministic retrieval trace.
        stable_reason = ";".join(
            "current" if segment.startswith("current=") else segment
            for segment in str(reason).split(";")
        )
        parts.append(stable_reason)
    return "; ".join(parts)


def _claim_key(fact: CapsuleFact) -> tuple[Any, ...]:
    return (
        fact.subject.casefold(),
        fact.predicate.casefold(),
        fact.value.casefold(),
        fact.valid_from,
        fact.valid_to,
    )


def _fact_applies(
    fact: CapsuleFact,
    as_of: datetime | None,
    *,
    current_time: datetime | None = None,
) -> bool:
    status = fact.status.casefold()
    if as_of is None:
        # ``superseded`` is a durable chain marker, not an immediate visibility
        # revocation.  The predecessor's exclusive ``valid_to`` boundary decides
        # when a future-effective successor actually replaces it.
        if status in {"retracted", "deleted", "inactive"}:
            return False
        point = current_time or datetime.now(timezone.utc)
    else:
        if status in {"retracted", "deleted"}:
            return False
        point = as_of
    valid_from = _parse_datetime(fact.valid_from)
    valid_to = _parse_datetime(fact.valid_to)
    if valid_from is not None and point < valid_from:
        return False
    if valid_to is not None and point >= valid_to:
        return False
    return True


def _episode_hit_applies(
    hit: Any,
    as_of: datetime | None,
    *,
    current_time: datetime,
) -> bool:
    """Keep claimless evidence, or evidence whose every claim is applicable.

    Evidence search is intentionally a separate retrieval channel, but it must not
    bypass temporal resolution.  Raw text can contain values from every linked claim,
    and Prototype A cannot safely redact arbitrary substrings.  Therefore one stale,
    future, retracted, or otherwise unreadable claim makes the whole raw episode
    ineligible; applicable claims remain available through the structured fact channel.
    """

    record = _unwrap_hit(hit, "episode")
    claims = tuple(_get(record, "claims", default=()) or ())
    if not claims:
        return True
    for claim in claims:
        try:
            fact = _to_fact(claim)
        except (AttributeError, TypeError, ValueError):
            return False
        if not _fact_applies(fact, as_of, current_time=current_time):
            return False
    return True


def _historical_search_query(
    query: str,
    as_of: str | datetime | None,
    point: datetime | None,
) -> str:
    """Remove the explicit cutoff date from a historical retrieval query.

    The original query remains part of :class:`MemoryCapsule`; this derivative is
    used only for lexical candidate generation.  Without the rewrite, common ISO
    components such as ``2026`` and ``01`` can dominate FTS matches while adding no
    relevance beyond the separately supplied temporal cutoff.
    """

    if point is None:
        return query

    cleaned = query
    exact_values: list[str] = [_as_iso(point) or "", point.date().isoformat()]
    if isinstance(as_of, str):
        exact_values.insert(0, as_of.strip())
    for value in dict.fromkeys(item for item in exact_values if item):
        cleaned = re.sub(re.escape(value), " ", cleaned, flags=re.IGNORECASE)

    year = str(point.year)
    month = str(point.month)
    day = str(point.day)
    numeric_patterns = (
        rf"\b{year}\s*[-/.]\s*0?{month}\s*[-/.]\s*0?{day}\b",
        rf"\b0?{month}\s*[-/.]\s*0?{day}\s*[-/.]\s*{year}\b",
        rf"\b0?{day}\s*[-/.]\s*0?{month}\s*[-/.]\s*{year}\b",
        rf"\b{year}\s+0?{month}\s+0?{day}\b",
    )
    for pattern in numeric_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    month_name = point.strftime("%B")
    month_abbreviation = point.strftime("%b")
    named_month = rf"(?:{re.escape(month_name)}|{re.escape(month_abbreviation)})"
    named_patterns = (
        rf"\b{named_month}\s+0?{day}(?:st|nd|rd|th)?\s*,?\s*{year}\b",
        rf"\b0?{day}(?:st|nd|rd|th)?\s+{named_month}\s*,?\s*{year}\b",
    )
    for pattern in named_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", cleaned).strip(" \t,;:()[]{}")


def _query_overlap(query_terms: frozenset[str], fact: CapsuleFact) -> int:
    fact_terms = {
        term.casefold()
        for term in _DIAGNOSTIC_TOKEN_RE.findall(
            f"{fact.subject} {fact.predicate} {fact.value}"
        )
    }
    return len(query_terms & fact_terms)


def _query_terms(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Mirror the store's positive/excluded term policy without private imports."""

    raw_terms = tuple(
        token.casefold() for token in _DIAGNOSTIC_TOKEN_RE.findall(query.casefold())
    )
    syntax_positions: set[int] = set()
    excluded: list[str] = []
    for index, term in enumerate(raw_terms):
        if term not in _QUERY_NEGATION_WORDS:
            continue
        syntax_positions.add(index)
        target_index = index + 1
        if (
            target_index < len(raw_terms)
            and raw_terms[target_index] in _QUERY_ARTICLES
        ):
            syntax_positions.add(target_index)
            target_index += 1
        if target_index < len(raw_terms):
            excluded.append(raw_terms[target_index])
            syntax_positions.add(target_index)
    positive = tuple(
        dict.fromkeys(
            term
            for index, term in enumerate(raw_terms)
            if index not in syntax_positions
            and term not in _DIAGNOSTIC_STOP_WORDS
            and len(term) > 1
        )
    )
    if positive:
        return positive, tuple(dict.fromkeys(excluded))
    fallback = tuple(
        dict.fromkeys(
            term
            for index, term in enumerate(raw_terms)
            if index not in syntax_positions
            and term not in _DIAGNOSTIC_STOP_WORDS
        )
    )
    return fallback, tuple(dict.fromkeys(excluded))


def _positive_query_terms(query: str) -> tuple[str, ...]:
    return _query_terms(query)[0]


def _normalized_terms(text: str) -> frozenset[str]:
    return frozenset(
        token.casefold()
        for token in _DIAGNOSTIC_TOKEN_RE.findall(text.casefold())
    )


def _hit_matches_excluded_terms(
    hit: Any,
    target: Literal["memory", "claim"],
    excluded_terms: frozenset[str],
) -> bool:
    if not excluded_terms:
        return False
    record = _unwrap_hit(hit, "episode" if target == "memory" else "fact")
    if target == "memory":
        text = " ".join(
            (
                str(_get(record, "content", "text", default="")),
                str(_get(record, "kind", default="")),
                str(_get(record, "source", default="")),
            )
        )
    else:
        text = " ".join(
            (
                str(_get(record, "subject", default="")),
                str(_get(record, "predicate", default="")),
                str(_get(record, "value", "object", default="")),
                str(_get(record, "authority", default="")),
            )
        )
    return bool(excluded_terms.intersection(_normalized_terms(text)))


def _diagnostic_token_overlap(
    query_terms: frozenset[str],
    hit: Any,
    target: Literal["memory", "claim"],
) -> bool:
    """Require literal token evidence when a provider is not semantic."""

    record = _unwrap_hit(hit, "episode" if target == "memory" else "fact")
    if target == "memory":
        text = str(_get(record, "content", "text", default=""))
    else:
        text = " ".join(
            (
                str(_get(record, "subject", default="")),
                str(_get(record, "predicate", default="")),
                str(_get(record, "value", "object", default="")),
            )
        )
    target_terms = _normalized_terms(text)
    return bool(query_terms.intersection(target_terms))


def _candidate_content_sort_key(candidate: _Candidate) -> tuple[str, ...]:
    if candidate.candidate_type == "fact":
        fact = candidate.item
        assert isinstance(fact, CapsuleFact)
        return (
            fact.subject.casefold(),
            fact.subject,
            fact.predicate.casefold(),
            fact.predicate,
            fact.value.casefold(),
            fact.value,
            fact.valid_from or "",
            fact.valid_to or "",
        )
    episode = candidate.item
    assert isinstance(episode, CapsuleEpisode)
    return (
        (episode.source or "").casefold(),
        episode.source or "",
        episode.occurred_at or "",
        episode.recorded_at or "",
        episode.content_hash or "",
        episode.kind.casefold(),
        episode.kind,
        episode.excerpt.casefold(),
        episode.excerpt,
    )


def _candidate_matches_excluded_terms(
    candidate: _Candidate,
    excluded_terms: frozenset[str],
) -> bool:
    if not excluded_terms:
        return False
    if candidate.candidate_type == "fact":
        fact = candidate.item
        assert isinstance(fact, CapsuleFact)
        text = f"{fact.subject} {fact.predicate} {fact.value}"
    else:
        episode = candidate.item
        assert isinstance(episode, CapsuleEpisode)
        text = " ".join(
            (
                episode.excerpt,
                episode.kind,
                episode.source or "",
            )
        )
    return bool(excluded_terms.intersection(_normalized_terms(text)))


def _candidate_sort_key(candidate: _Candidate) -> tuple[Any, ...]:
    type_order = 0 if candidate.candidate_type == "fact" else 1
    return (
        -candidate.score,
        type_order,
        *_candidate_content_sort_key(candidate),
        candidate.stable_id,
    )


def _typed_candidate_id(candidate: _Candidate) -> str:
    """Return a collision-free provider ID without changing public trace IDs."""

    return f"{candidate.candidate_type}:{candidate.stable_id}"


def _rerank_candidate_text(candidate: _Candidate) -> str:
    """Serialize one candidate into stable, provider-neutral reranker input."""

    if candidate.candidate_type == "fact":
        fact = candidate.item
        assert isinstance(fact, CapsuleFact)
        payload: dict[str, Any] = {
            "predicate": fact.predicate,
            "subject": fact.subject,
            "type": "fact",
            "value": fact.value,
        }
        if fact.status.casefold() != "active":
            payload["status"] = fact.status
        if fact.valid_from is not None:
            payload["valid_from"] = fact.valid_from
        if fact.valid_to is not None:
            payload["valid_to"] = fact.valid_to
    else:
        episode = candidate.item
        assert isinstance(episode, CapsuleEpisode)
        payload = {
            "excerpt": episode.excerpt,
            "kind": episode.kind,
            "type": "episode",
        }
        if episode.source is not None:
            payload["source"] = episode.source
        if episode.occurred_at is not None:
            payload["occurred_at"] = episode.occurred_at
        if episode.recorded_at is not None:
            payload["recorded_at"] = episode.recorded_at
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _find_conflicts(facts: Sequence[CapsuleFact]) -> tuple[MemoryConflict, ...]:
    grouped: dict[tuple[str, str], list[CapsuleFact]] = defaultdict(list)
    for fact in facts:
        grouped[(fact.subject.casefold(), fact.predicate.casefold())].append(fact)

    conflicts: list[MemoryConflict] = []
    for key in sorted(grouped):
        group = grouped[key]
        values = {fact.value for fact in group}
        if len(values) <= 1:
            continue
        conflicts.append(
            MemoryConflict(
                subject=min((fact.subject for fact in group), key=str.casefold),
                predicate=min((fact.predicate for fact in group), key=str.casefold),
                values=tuple(sorted(values, key=lambda item: (item.casefold(), item))),
                claim_ids=tuple(sorted(fact.claim_id for fact in group)),
                evidence_ids=tuple(
                    sorted({item for fact in group for item in fact.evidence_ids})
                ),
            )
        )
    return tuple(conflicts)


def _call_with_supported_keywords(method: Any, **kwargs: Any) -> Any:
    """Call a store method while tolerating optional adapter parameters."""

    signature = inspect.signature(method)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return method(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return method(**supported)


def _semantic_scan_cap(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("semantic_scan_cap must be a positive integer")
    return value


def _minimum_semantic_score(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("minimum_semantic_score must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("minimum_semantic_score must be a finite number") from error
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("minimum_semantic_score must be between -1 and 1")
    return score


def _reranker_weight(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("reranker_weight must be a finite positive number")
    try:
        weight = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "reranker_weight must be a finite positive number"
        ) from error
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("reranker_weight must be greater than zero")
    return weight


def _rerank_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("rerank_limit must be a positive integer")
    return value


class DeterministicRetriever:
    """Fuse store channels with reciprocal-rank fusion and compile a capsule."""

    def __init__(
        self,
        store: Any,
        *,
        embedder: Embedder | None = None,
        embedding_profile: EmbeddingProfile | None = None,
        semantic_scan_cap: int = 1024,
        minimum_semantic_score: float = 0.05,
        reranker: Reranker | None = None,
        rerank_limit: int = 16,
        reranker_weight: float = 1.25,
    ) -> None:
        self._store = store
        self._embedder = embedder
        if embedder is None:
            if embedding_profile is not None:
                raise ValueError("embedding_profile requires an embedder")
            self._embedding_profile = None
        else:
            profile = (
                EmbeddingProfile.from_embedder(embedder)
                if embedding_profile is None
                else embedding_profile
            )
            if not isinstance(profile, EmbeddingProfile):
                raise TypeError("embedding_profile must be an EmbeddingProfile")
            if not profile.is_compatible(embedder):
                raise EmbeddingValidationError(
                    "embedding_profile is incompatible with the embedder"
                )
            self._embedding_profile = profile
        self._requires_diagnostic_token_overlap = (
            embedder is not None
            and (
                getattr(embedder, "diagnostic_only", False) is True
                or getattr(embedder, "is_semantic", True) is False
            )
        )
        self._semantic_scan_cap = _semantic_scan_cap(semantic_scan_cap)
        self._minimum_semantic_score = _minimum_semantic_score(
            minimum_semantic_score
        )
        self._reranker = reranker
        self._reranker_identity = (
            None if reranker is None else snapshot_reranker_identity(reranker)
        )
        self._rerank_limit = _rerank_limit(rerank_limit)
        self._reranker_weight = _reranker_weight(reranker_weight)

    @property
    def reranker_identity(self) -> str | None:
        """Immutable reranker identity snapshot used in retrieval traces."""

        return self._reranker_identity

    def recall(
        self,
        *,
        namespace: str,
        query: str,
        budget_tokens: int,
        as_of: str | datetime | None = None,
        limit: int = 20,
    ) -> MemoryCapsule:
        budget = validate_budget(budget_tokens)
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        normalized_namespace = namespace.strip()
        normalized_query = query.strip()
        as_of_dt = _parse_datetime(as_of)
        as_of_iso = _as_iso(as_of_dt)
        search_query = _historical_search_query(
            normalized_query,
            as_of,
            as_of_dt,
        )
        positive_terms_raw, excluded_terms_raw = _query_terms(search_query)
        positive_terms = frozenset(positive_terms_raw)
        excluded_terms = frozenset(excluded_terms_raw)
        current_time = datetime.now(timezone.utc)
        fetch_limit = max(limit * 3, limit)

        candidates: dict[tuple[str, str], _Candidate] = {}
        claim_hits = (
            tuple(
                _call_with_supported_keywords(
                    self._store.search_claims,
                    namespace=normalized_namespace,
                    query=search_query,
                    as_of=as_of,
                    limit=fetch_limit,
                )
                or ()
            )
            if search_query
            else ()
        )
        memory_hits = (
            tuple(
                hit
                for hit in (
                    _call_with_supported_keywords(
                        self._store.search_memories,
                        namespace=normalized_namespace,
                        query=search_query,
                        as_of=as_of,
                        limit=fetch_limit,
                    )
                    or ()
                )
                if _episode_hit_applies(
                    hit,
                    as_of_dt,
                    current_time=current_time,
                )
            )
            if search_query
            else ()
        )
        (
            semantic_claim_hits,
            semantic_memory_hits,
            semantic_diagnostics,
        ) = self._semantic_hits(
            namespace=normalized_namespace,
            query=normalized_query,
            as_of=as_of,
            as_of_dt=as_of_dt,
            current_time=current_time,
            limit=fetch_limit,
            positive_terms=positive_terms,
            excluded_terms=excluded_terms,
        )

        self._add_channel(
            candidates,
            claim_hits,
            "fact",
            "claim_lexical",
            1.15,
            normalized_namespace,
        )

        self._add_channel(
            candidates,
            memory_hits,
            "episode",
            "episode_lexical",
            1.0,
            normalized_namespace,
        )
        self._add_channel(
            candidates,
            semantic_claim_hits,
            "fact",
            "claim_semantic",
            1.05,
            normalized_namespace,
        )
        self._add_channel(
            candidates,
            semantic_memory_hits,
            "episode",
            "episode_semantic",
            0.95,
            normalized_namespace,
        )

        # Explicit query exclusions are eligibility constraints, not weak negative
        # ranking signals.  Apply them before temporal/entity scoring so a flood of
        # negated candidates cannot gain additional fusion contributions.
        for key, candidate in tuple(candidates.items()):
            if _candidate_matches_excluded_terms(candidate, excluded_terms):
                del candidates[key]

        applicable_facts: list[_Candidate] = []
        for key, candidate in tuple(candidates.items()):
            if candidate.candidate_type != "fact":
                continue
            fact = candidate.item
            assert isinstance(fact, CapsuleFact)
            if not _fact_applies(fact, as_of_dt, current_time=current_time):
                del candidates[key]
                continue
            if fact.status.casefold() == "superseded":
                # The durable row records lineage immediately, while valid-time
                # determines effective truth.  Do not tell the model that a fact
                # is already superseded before its exclusive end boundary.
                candidate.item = replace(fact, status="active")
            applicable_facts.append(candidate)

        temporal_order = sorted(
            applicable_facts,
            key=lambda candidate: (
                _candidate_content_sort_key(candidate),
                candidate.stable_id,
            ),
        )
        temporal_order.sort(
            key=lambda candidate: _temporal_sort_value(candidate.item),
            reverse=True,
        )
        for rank, candidate in enumerate(temporal_order, start=1):
            candidate.reasons.append(
                ScoreReason(
                    channel="historical" if as_of_dt is not None else "current",
                    rank=rank,
                    contribution=0.7 / (_RRF_K + rank),
                    detail=(f"as_of={as_of_iso}" if as_of_iso is not None else "active claim"),
                )
            )

        entity_order = sorted(
            (
                (overlap, candidate)
                for candidate in applicable_facts
                if (overlap := _query_overlap(positive_terms, candidate.item)) > 0
            ),
            key=lambda pair: (
                -pair[0],
                _candidate_content_sort_key(pair[1]),
                pair[1].stable_id,
            ),
        )
        for rank, (overlap, candidate) in enumerate(entity_order, start=1):
            candidate.reasons.append(
                ScoreReason(
                    channel="entity",
                    rank=rank,
                    contribution=0.8 / (_RRF_K + rank),
                    detail=f"query_term_overlap={overlap}",
                )
            )

        graph_seeds = self._graph_seed_terms(
            search_query,
            candidates.values(),
            limit,
        )
        graph_hits = self._graph_hits(
            normalized_namespace,
            graph_seeds,
            self._semantic_scan_cap if excluded_terms else fetch_limit,
            as_of=as_of,
        )
        graph_hits = tuple(
            hit
            for hit in graph_hits
            if not _hit_matches_excluded_terms(
                hit,
                "claim" if _infer_candidate_type(hit) == "fact" else "memory",
                excluded_terms,
            )
        )
        self._add_mixed_graph_channel(
            candidates,
            graph_hits,
            normalized_namespace,
            as_of=as_of_dt,
            current_time=current_time,
        )

        for key, candidate in tuple(candidates.items()):
            if _candidate_matches_excluded_terms(candidate, excluded_terms):
                del candidates[key]

        # Graph expansion is another candidate source, not a bypass around temporal
        # resolution.  Re-apply the same rule so an obsolete neighbor cannot leak back
        # into a current capsule after the initial claim search was filtered.
        for key, candidate in tuple(candidates.items()):
            if candidate.candidate_type != "fact":
                continue
            fact = candidate.item
            assert isinstance(fact, CapsuleFact)
            if not _fact_applies(fact, as_of_dt, current_time=current_time):
                del candidates[key]
            elif fact.status.casefold() == "superseded":
                candidate.item = replace(fact, status="active")

        deduplicated = self._deduplicate(candidates.values())
        ranked = sorted(deduplicated, key=_candidate_sort_key)
        ranked = self._rerank(normalized_query, ranked)
        ordered = ranked[:limit]
        capsule = self._pack(
            namespace=normalized_namespace,
            query=normalized_query,
            budget_tokens=budget,
            as_of=as_of_iso,
            candidates=ordered,
            additional_omitted=max(0, len(ranked) - len(ordered)),
        )
        return replace(capsule, diagnostics=semantic_diagnostics)

    def _rerank(
        self,
        query: str,
        ranked: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """Fuse ordinal reranker output over a bounded, deduplicated pool."""

        if self._reranker is None or not ranked:
            return list(ranked)
        assert self._reranker_identity is not None
        if snapshot_reranker_identity(self._reranker) != self._reranker_identity:
            raise ValueError(
                "reranker identity changed after retriever construction"
            )

        pool = tuple(ranked[: self._rerank_limit])
        provider_candidates = tuple(
            RerankCandidate(
                candidate_id=_typed_candidate_id(candidate),
                text=_rerank_candidate_text(candidate),
            )
            for candidate in pool
        )
        raw_results = self._reranker.rerank(query, provider_candidates)
        if snapshot_reranker_identity(self._reranker) != self._reranker_identity:
            raise ValueError("reranker identity changed during reranking")
        results = validate_rerank_results(
            provider_candidates,
            raw_results,
            require_complete=True,
        )

        base_position = {
            candidate.candidate_id: position
            for position, candidate in enumerate(provider_candidates)
        }
        candidate_by_id = {
            _typed_candidate_id(candidate): candidate for candidate in pool
        }
        ordinal_results = sorted(
            results,
            key=lambda result: (
                -result.score,
                base_position[result.candidate_id],
            ),
        )
        for rank, result in enumerate(ordinal_results, start=1):
            detail = json.dumps(
                {
                    "identity": self._reranker_identity,
                    "provider_score": result.score,
                    "schema": "evicortex.rerank.v1",
                    "score_use": "ordinal_only",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            candidate_by_id[result.candidate_id].reasons.append(
                ScoreReason(
                    channel="reranker",
                    rank=rank,
                    contribution=self._reranker_weight / (_RRF_K + rank),
                    detail=detail,
                )
            )
        return sorted(ranked, key=_candidate_sort_key)

    def _semantic_hits(
        self,
        *,
        namespace: str,
        query: str,
        as_of: str | datetime | None,
        as_of_dt: datetime | None,
        current_time: datetime,
        limit: int,
        positive_terms: frozenset[str],
        excluded_terms: frozenset[str],
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[CapsuleDiagnostic, ...]]:
        if self._embedder is None or self._embedding_profile is None:
            return (), (), ()
        if not hasattr(self._store, "semantic_search_memories") or not hasattr(
            self._store,
            "semantic_search_claims",
        ):
            return (
                (),
                (),
                (
                    CapsuleDiagnostic(
                        "semantic_store_unavailable",
                        "semantic",
                        "configured store has no semantic search API",
                    ),
                ),
            )

        profile = self._embedding_profile
        if not profile.is_compatible(self._embedder):
            raise EmbeddingValidationError(
                "embedder compatibility fields changed after retriever construction"
            )
        raw_vectors = (embed_query_text(self._embedder, query),)
        if not profile.is_compatible(self._embedder):
            raise EmbeddingValidationError(
                "embedder compatibility fields changed during query embedding"
            )
        query_vector = validate_embedding_batch(
            raw_vectors,
            expected_count=1,
            dimensions=profile.dimensions,
            normalized=profile.normalized,
            allow_zero=True,
        )[0]
        if all(value == 0.0 for value in query_vector):
            return (
                (),
                (),
                (
                    CapsuleDiagnostic(
                        "semantic_query_zero",
                        "semantic",
                        f"profile={profile.profile_sha256}",
                    ),
                ),
            )

        semantic_limit = self._semantic_scan_cap if excluded_terms else limit
        try:
            memory_result = _call_with_supported_keywords(
                self._store.semantic_search_memories,
                namespace=namespace,
                query_vector=query_vector,
                profile=profile,
                limit=semantic_limit,
                scan_cap=self._semantic_scan_cap,
                as_of=as_of,
            )
            claim_result = _call_with_supported_keywords(
                self._store.semantic_search_claims,
                namespace=namespace,
                query_vector=query_vector,
                profile=profile,
                as_of=as_of,
                limit=semantic_limit,
                scan_cap=self._semantic_scan_cap,
            )
        except EmbeddingProfileNotFoundError:
            return (
                (),
                (),
                (
                    CapsuleDiagnostic(
                        "semantic_profile_missing",
                        "semantic",
                        f"profile={profile.profile_sha256}",
                    ),
                ),
            )

        diagnostics: list[CapsuleDiagnostic] = []
        memory_diagnostics = _get(memory_result, "diagnostics")
        claim_diagnostics = _get(claim_result, "diagnostics")
        if bool(_get(memory_diagnostics, "query_is_zero", default=False)) or bool(
            _get(claim_diagnostics, "query_is_zero", default=False)
        ):
            diagnostics.append(
                CapsuleDiagnostic(
                    "semantic_query_zero",
                    "semantic",
                    f"profile={profile.profile_sha256};float32=true",
                )
            )
            return (), (), tuple(diagnostics)

        memory_hits = self._usable_semantic_channel(
            memory_result,
            "memory",
            diagnostics,
        )
        claim_hits = self._usable_semantic_channel(
            claim_result,
            "claim",
            diagnostics,
        )
        unfiltered_memory_count = len(memory_hits)
        unfiltered_claim_count = len(claim_hits)
        memory_hits = tuple(
            hit
            for hit in memory_hits
            if not _hit_matches_excluded_terms(
                hit,
                "memory",
                excluded_terms,
            )
        )
        claim_hits = tuple(
            hit
            for hit in claim_hits
            if not _hit_matches_excluded_terms(
                hit,
                "claim",
                excluded_terms,
            )
        )
        excluded_memories = unfiltered_memory_count - len(memory_hits)
        excluded_claims = unfiltered_claim_count - len(claim_hits)
        if excluded_memories or excluded_claims:
            diagnostics.append(
                CapsuleDiagnostic(
                    "semantic_excluded_term_filtered",
                    "semantic",
                    (
                        f"memories={excluded_memories};"
                        f"claims={excluded_claims};"
                        f"terms={','.join(sorted(excluded_terms))}"
                    ),
                )
            )
        if self._requires_diagnostic_token_overlap:
            query_terms = positive_terms
            unfiltered_memory_count = len(memory_hits)
            unfiltered_claim_count = len(claim_hits)
            memory_hits = tuple(
                hit
                for hit in memory_hits
                if _diagnostic_token_overlap(query_terms, hit, "memory")
            )
            claim_hits = tuple(
                hit
                for hit in claim_hits
                if _diagnostic_token_overlap(query_terms, hit, "claim")
            )
            discarded_memories = unfiltered_memory_count - len(memory_hits)
            discarded_claims = unfiltered_claim_count - len(claim_hits)
            if discarded_memories or discarded_claims:
                diagnostics.append(
                    CapsuleDiagnostic(
                        "semantic_diagnostic_token_overlap_filtered",
                        "semantic",
                        (
                            f"memories={discarded_memories};"
                            f"claims={discarded_claims};"
                            f"profile={profile.profile_sha256}"
                        ),
                    )
                )
        memory_hits = tuple(
            hit
            for hit in memory_hits
            if _episode_hit_applies(
                hit,
                as_of_dt,
                current_time=current_time,
            )
        )
        return claim_hits, memory_hits, tuple(diagnostics)

    def _usable_semantic_channel(
        self,
        result: Any,
        target: Literal["memory", "claim"],
        diagnostics: list[CapsuleDiagnostic],
    ) -> tuple[Any, ...]:
        result_diagnostics = _get(result, "diagnostics")
        scanned = int(_get(result_diagnostics, "scanned", default=0))
        scan_cap = int(
            _get(
                result_diagnostics,
                "scan_cap",
                default=self._semantic_scan_cap,
            )
        )
        profile = self._embedding_profile
        assert profile is not None
        if bool(_get(result_diagnostics, "truncated", default=False)):
            diagnostics.append(
                CapsuleDiagnostic(
                    f"semantic_{target}_scan_truncated",
                    f"semantic_{target}",
                    (
                        f"scanned={scanned};scan_cap={scan_cap};"
                        f"profile={profile.profile_sha256};hits_discarded=true"
                    ),
                )
            )
            return ()
        if bool(_get(result_diagnostics, "no_vectors", default=False)):
            diagnostics.append(
                CapsuleDiagnostic(
                    f"semantic_{target}_no_vectors",
                    f"semantic_{target}",
                    f"profile={profile.profile_sha256}",
                )
            )
            return ()

        hits = tuple(_get(result, "hits", default=()) or ())
        eligible = tuple(
            hit
            for hit in hits
            if float(_get(hit, "score")) >= self._minimum_semantic_score
        )
        if len(eligible) != len(hits):
            diagnostics.append(
                CapsuleDiagnostic(
                    f"semantic_{target}_below_threshold",
                    f"semantic_{target}",
                    (
                        f"discarded={len(hits) - len(eligible)};"
                        f"minimum_score={self._minimum_semantic_score:.12g}"
                    ),
                )
            )
        return eligible

    def _add_channel(
        self,
        candidates: dict[tuple[str, str], _Candidate],
        hits: Sequence[Any],
        candidate_type: Literal["fact", "episode"],
        channel: str,
        weight: float,
        namespace: str,
    ) -> None:
        seen: set[str] = set()
        eligible_hits = [
            hit
            for hit in hits
            if (hit_namespace := _record_namespace(hit, candidate_type)) is None
            or hit_namespace == namespace
        ]
        for rank, hit in enumerate(eligible_hits, start=1):
            item: CapsuleFact | CapsuleEpisode
            item = _to_fact(hit) if candidate_type == "fact" else _to_episode(hit)
            stable_id = item.claim_id if isinstance(item, CapsuleFact) else item.memory_id
            if stable_id in seen:
                continue
            seen.add(stable_id)
            key = (candidate_type, stable_id)
            candidate = candidates.setdefault(
                key,
                _Candidate(
                    stable_id=stable_id,
                    candidate_type=candidate_type,
                    item=item,
                ),
            )
            candidate.reasons.append(
                ScoreReason(
                    channel=channel,
                    rank=rank,
                    contribution=weight / (_RRF_K + rank),
                    detail=_source_detail(hit),
                )
            )

    def _add_mixed_graph_channel(
        self,
        candidates: dict[tuple[str, str], _Candidate],
        hits: Sequence[Any],
        namespace: str,
        *,
        as_of: datetime | None,
        current_time: datetime,
    ) -> None:
        eligible_hits: list[tuple[Any, Literal["fact", "episode"]]] = []
        for hit in hits:
            candidate_type = _infer_candidate_type(hit)
            hit_namespace = _record_namespace(hit, candidate_type)
            if hit_namespace is not None and hit_namespace != namespace:
                continue
            if candidate_type == "episode" and not _episode_hit_applies(
                hit,
                as_of,
                current_time=current_time,
            ):
                continue
            eligible_hits.append((hit, candidate_type))
        for rank, (hit, candidate_type) in enumerate(eligible_hits, start=1):
            try:
                item = _to_fact(hit) if candidate_type == "fact" else _to_episode(hit)
            except AttributeError:
                continue
            stable_id = item.claim_id if isinstance(item, CapsuleFact) else item.memory_id
            key = (candidate_type, stable_id)
            candidate = candidates.setdefault(
                key,
                _Candidate(
                    stable_id=stable_id,
                    candidate_type=candidate_type,
                    item=item,
                ),
            )
            candidate.reasons.append(
                ScoreReason(
                    channel="graph",
                    rank=rank,
                    contribution=0.65 / (_RRF_K + rank),
                    detail=_source_detail(hit),
                )
            )

    def _graph_seed_terms(
        self,
        query: str,
        candidates: Iterable[_Candidate],
        limit: int,
    ) -> tuple[str, ...]:
        ranked = sorted(candidates, key=_candidate_sort_key)
        seeds: list[str] = list(_positive_query_terms(query))
        seeds.extend(
            term.casefold()
            for term in _WORD_RE.findall(query)
            if ":" in term
            and any(part in seeds for part in term.casefold().split(":"))
        )
        for candidate in ranked:
            if candidate.candidate_type == "fact":
                fact = candidate.item
                assert isinstance(fact, CapsuleFact)
                seeds.extend(
                    term
                    for term in (fact.subject, fact.predicate)
                    if term.strip().casefold() not in _DIAGNOSTIC_STOP_WORDS
                )
                if ":" in fact.value:
                    seeds.append(fact.value)
                seeds.extend(_positive_query_terms(fact.value))
            if len(seeds) >= min(limit * 2, 12):
                break
        normalized = (seed.strip().casefold() for seed in seeds)
        return tuple(dict.fromkeys(seed for seed in normalized if seed))

    def _graph_hits(
        self,
        namespace: str,
        seed_terms: Sequence[str],
        limit: int,
        *,
        as_of: str | datetime | None,
    ) -> tuple[Any, ...]:
        if not seed_terms or not hasattr(self._store, "graph_neighbors"):
            return ()
        method = self._store.graph_neighbors
        parameters = inspect.signature(method).parameters
        if "seed_terms" in parameters:
            result = _call_with_supported_keywords(
                method,
                namespace=namespace,
                seed_terms=tuple(seed_terms),
                limit=limit,
                as_of=as_of,
            )
            return tuple(result or ())
        if "memory_ids" in parameters or "seed_ids" in parameters:
            result = _call_with_supported_keywords(
                method,
                namespace=namespace,
                memory_ids=tuple(seed_terms),
                seed_ids=tuple(seed_terms),
                limit=limit,
                as_of=as_of,
            )
            return tuple(result or ())

        combined: list[Any] = []
        for memory_id in seed_terms:
            result = _call_with_supported_keywords(
                method,
                namespace=namespace,
                memory_id=memory_id,
                limit=limit,
                as_of=as_of,
            )
            combined.extend(result or ())
        return tuple(combined[:limit])

    def _deduplicate(self, candidates: Iterable[_Candidate]) -> list[_Candidate]:
        ordered = sorted(candidates, key=_candidate_sort_key)
        seen_claims: dict[tuple[Any, ...], _Candidate] = {}
        seen_episodes: set[tuple[str, str]] = set()
        output: list[_Candidate] = []
        for candidate in ordered:
            if candidate.candidate_type == "fact":
                fact = candidate.item
                assert isinstance(fact, CapsuleFact)
                identity = _claim_key(fact)
                if identity in seen_claims:
                    retained = seen_claims[identity]
                    retained_fact = retained.item
                    assert isinstance(retained_fact, CapsuleFact)
                    retained.item = replace(
                        retained_fact,
                        confidence=max(retained_fact.confidence, fact.confidence),
                        evidence_ids=tuple(
                            sorted(set(retained_fact.evidence_ids) | set(fact.evidence_ids))
                        ),
                    )
                    continue
                seen_claims[identity] = candidate
            else:
                episode = candidate.item
                assert isinstance(episode, CapsuleEpisode)
                identity = (
                    "hash" if episode.content_hash else "id",
                    episode.content_hash or episode.memory_id,
                )
                if identity in seen_episodes:
                    continue
                seen_episodes.add(identity)
            output.append(candidate)
        return output

    def _pack(
        self,
        *,
        namespace: str,
        query: str,
        budget_tokens: int,
        as_of: str | None,
        candidates: Sequence[_Candidate],
        additional_omitted: int = 0,
    ) -> MemoryCapsule:
        total = len(candidates) + additional_omitted
        empty = MemoryCapsule(
            namespace=namespace,
            query=query,
            budget_tokens=budget_tokens,
            as_of=as_of,
            omitted_count=total,
        )
        minimum = empty.estimated_tokens
        if minimum > budget_tokens:
            raise BudgetTooSmallError(budget_tokens, minimum)

        selected_facts: list[CapsuleFact] = []
        selected_episodes: list[CapsuleEpisode] = []
        decisions: dict[tuple[str, str], tuple[bool, str | None]] = {}

        for candidate in candidates:
            if candidate.candidate_type == "fact":
                fact = candidate.item
                assert isinstance(fact, CapsuleFact)
                fact = replace(
                    fact,
                    score=candidate.score,
                    reasons=tuple(candidate.reasons),
                )
                duplicate_episode_ids = set(fact.evidence_ids)
                retained_episodes = [
                    episode
                    for episode in selected_episodes
                    if episode.memory_id not in duplicate_episode_ids
                ]
                proposed_facts = [*selected_facts, fact]
                proposed = self._capsule_for_selection(
                    namespace,
                    query,
                    budget_tokens,
                    as_of,
                    proposed_facts,
                    retained_episodes,
                    total,
                )
                if proposed.estimated_tokens <= budget_tokens:
                    for episode in selected_episodes:
                        if episode.memory_id in duplicate_episode_ids:
                            decisions[("episode", episode.memory_id)] = (
                                False,
                                "deduplicated_by_fact",
                            )
                    selected_episodes = retained_episodes
                    selected_facts = proposed_facts
                    decisions[("fact", fact.claim_id)] = (True, None)
                else:
                    decisions[("fact", fact.claim_id)] = (False, "budget")
                continue

            episode = candidate.item
            assert isinstance(episode, CapsuleEpisode)
            if any(episode.memory_id in fact.evidence_ids for fact in selected_facts):
                decisions[("episode", episode.memory_id)] = (
                    False,
                    "deduplicated_by_fact",
                )
                continue
            episode = replace(
                episode,
                score=candidate.score,
                reasons=tuple(candidate.reasons),
            )
            fitted = self._fit_episode(
                namespace,
                query,
                budget_tokens,
                as_of,
                selected_facts,
                selected_episodes,
                episode,
                total,
            )
            if fitted is None:
                decisions[("episode", episode.memory_id)] = (False, "budget")
            else:
                selected_episodes.append(fitted)
                decisions[("episode", episode.memory_id)] = (True, None)

        capsule = self._capsule_for_selection(
            namespace,
            query,
            budget_tokens,
            as_of,
            selected_facts,
            selected_episodes,
            total,
        )
        traces: list[RetrievalTrace] = []
        for candidate in candidates:
            selected, reason = decisions.get(
                (candidate.candidate_type, candidate.stable_id),
                (False, "not_selected"),
            )
            traces.append(
                RetrievalTrace(
                    candidate_id=candidate.stable_id,
                    candidate_type=candidate.candidate_type,
                    score=candidate.score,
                    reasons=tuple(candidate.reasons),
                    selected=selected,
                    omission_reason=reason,
                )
            )
        capsule = replace(capsule, trace=tuple(traces))
        if capsule.estimated_tokens > budget_tokens:
            raise AssertionError("memory capsule compiler exceeded its token budget")
        return capsule

    def _capsule_for_selection(
        self,
        namespace: str,
        query: str,
        budget_tokens: int,
        as_of: str | None,
        facts: Sequence[CapsuleFact],
        episodes: Sequence[CapsuleEpisode],
        total_candidates: int,
    ) -> MemoryCapsule:
        selected_count = len(facts) + len(episodes)
        return MemoryCapsule(
            namespace=namespace,
            query=query,
            budget_tokens=budget_tokens,
            as_of=as_of,
            facts=tuple(facts),
            episodes=tuple(episodes),
            conflicts=_find_conflicts(facts),
            omitted_count=max(0, total_candidates - selected_count),
        )

    def _fit_episode(
        self,
        namespace: str,
        query: str,
        budget_tokens: int,
        as_of: str | None,
        facts: Sequence[CapsuleFact],
        episodes: Sequence[CapsuleEpisode],
        episode: CapsuleEpisode,
        total_candidates: int,
    ) -> CapsuleEpisode | None:
        proposed = self._capsule_for_selection(
            namespace,
            query,
            budget_tokens,
            as_of,
            facts,
            [*episodes, episode],
            total_candidates,
        )
        if proposed.estimated_tokens <= budget_tokens:
            return episode

        raw_excerpt = episode.excerpt.rstrip("…")
        low = _MIN_TRUNCATED_EXCERPT_CHARS
        high = len(raw_excerpt)
        fitted: CapsuleEpisode | None = None
        while low <= high:
            midpoint = (low + high) // 2
            shortened = replace(
                episode,
                excerpt=raw_excerpt[:midpoint].rstrip() + "…",
                truncated=True,
            )
            proposed = self._capsule_for_selection(
                namespace,
                query,
                budget_tokens,
                as_of,
                facts,
                [*episodes, shortened],
                total_candidates,
            )
            if proposed.estimated_tokens <= budget_tokens:
                fitted = shortened
                low = midpoint + 1
            else:
                high = midpoint - 1
        return fitted


def _temporal_sort_value(item: CapsuleFact | CapsuleEpisode) -> str:
    if isinstance(item, CapsuleFact):
        return item.valid_from or ""
    return item.occurred_at or item.recorded_at or ""


def _infer_candidate_type(hit: Any) -> Literal["fact", "episode"]:
    unwrapped_claim = _get(hit, "claim", default=None)
    if unwrapped_claim is not None:
        return "fact"
    record = _get(hit, "memory", "record", "item", default=hit)
    if _get(record, "claim_id", default=None) is not None:
        return "fact"
    return "episode"


__all__ = [
    "CapsuleDiagnostic",
    "CapsuleEpisode",
    "CapsuleFact",
    "DeterministicRetriever",
    "MemoryCapsule",
    "MemoryConflict",
    "RetrievalTrace",
    "ScoreReason",
]
