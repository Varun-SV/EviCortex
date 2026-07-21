"""Seeded streaming workloads with source-level gold evidence annotations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random


PRIMARY_NAMESPACE = "tenant-alpha/user-001"
OTHER_NAMESPACE = "tenant-beta/user-001"
MIN_EVENTS = 12


@dataclass(frozen=True, slots=True)
class ClaimSpec:
    subject: str
    predicate: str
    value: str
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 1.0
    authority: str = "application"
    supersedes_source_id: str | None = None

    def canonical_text(self) -> str:
        fields = [
            f"subject={self.subject}",
            f"predicate={self.predicate}",
            f"value={self.value}",
        ]
        if self.valid_from is not None:
            fields.append(f"valid_from={self.valid_from}")
        if self.valid_to is not None:
            fields.append(f"valid_to={self.valid_to}")
        if self.supersedes_source_id is not None:
            fields.append(f"supersedes_source={self.supersedes_source_id}")
        fields.append(f"confidence={self.confidence:.3f}")
        fields.append(f"authority={self.authority}")
        return "Canonical claim: " + "; ".join(fields) + "."


@dataclass(frozen=True, slots=True)
class SyntheticEvent:
    sequence: int
    source_id: str
    namespace: str
    occurred_at: str
    content: str
    claims: tuple[ClaimSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryCase:
    query_id: str
    task: str
    after_sequence: int
    namespace: str
    query: str
    gold_evidence_ids: tuple[str, ...]
    forbidden_evidence_ids: tuple[str, ...] = ()
    as_of: str | None = None
    expected_answerable: bool = True
    expected_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SyntheticSuite:
    seed: int
    requested_event_count: int
    events: tuple[SyntheticEvent, ...]
    queries: tuple[QueryCase, ...]

    @property
    def source_bytes(self) -> int:
        return sum(len(event.content.encode("utf-8")) for event in self.events)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _event(
    *,
    sequence: int,
    source_id: str,
    namespace: str,
    occurred_at: str,
    sentence: str,
    claims: tuple[ClaimSpec, ...],
) -> SyntheticEvent:
    # Structured facts are appended verbatim so the lexical baseline receives the
    # same subject/predicate/value representation supplied to EviCortex.
    canonical = " ".join(claim.canonical_text() for claim in claims)
    content = f"{sentence.strip()} {canonical}".strip()
    return SyntheticEvent(
        sequence=sequence,
        source_id=source_id,
        namespace=namespace,
        occurred_at=occurred_at,
        content=content,
        claims=claims,
    )


def generate_suite(*, event_count: int, seed: int) -> SyntheticSuite:
    """Create a reproducible event stream and source-ID-based gold questions."""

    if event_count < MIN_EVENTS:
        raise ValueError(f"event_count must be at least {MIN_EVENTS}")
    rng = random.Random(seed)
    prefix = f"synthetic/{seed:08x}"

    person = rng.choice(("Ada", "Ira", "Mira", "Noor"))
    person_key = f"user:{person.casefold()}"
    direct_city = rng.choice(("Pune", "Mysuru", "Jaipur", "Surat"))
    manager = rng.choice(("Mina", "Ravi", "Tara", "Zoya"))
    manager_city = rng.choice(tuple(city for city in ("Kochi", "Indore", "Nagpur", "Agra") if city != direct_city))
    project_a = rng.choice(("Atlas", "Helios", "Nimbus", "Orion"))
    project_b = rng.choice(tuple(name for name in ("Atlas", "Helios", "Nimbus", "Orion") if name != project_a))

    moments = {
        "direct": "2026-01-01T09:00:00Z",
        "theme_old": "2026-01-02T09:00:00Z",
        "project_start": "2026-01-10T09:00:00Z",
        "manager": "2026-01-12T09:00:00Z",
        "manager_city": "2026-01-13T09:00:00Z",
        "conflict_a": "2026-01-14T09:00:00Z",
        "secret_primary": "2026-01-15T09:00:00Z",
        "secret_other": "2026-01-15T09:01:00Z",
        "growth_target": "2026-01-16T09:00:00Z",
        "theme_new": "2026-02-01T09:00:00Z",
        "project_release": "2026-03-05T09:00:00Z",
        "conflict_b": "2026-03-06T09:00:00Z",
    }
    source = {name: f"{prefix}/{name}" for name in moments}

    specifications = [
        (
            "direct",
            PRIMARY_NAMESPACE,
            f"{person}'s home city is {direct_city}.",
            (ClaimSpec(person_key, "home_city", direct_city, moments["direct"]),),
        ),
        (
            "theme_old",
            PRIMARY_NAMESPACE,
            f"{person} selected the light interface theme.",
            (ClaimSpec(person_key, "preferred_theme", "light", moments["theme_old"]),),
        ),
        (
            "project_start",
            PRIMARY_NAMESPACE,
            f"Project {project_a} started on 2026-01-10.",
            (ClaimSpec(f"project:{project_a.casefold()}", "started_on", "2026-01-10", moments["project_start"]),),
        ),
        (
            "manager",
            PRIMARY_NAMESPACE,
            f"Project {project_b} is managed by {manager}.",
            (ClaimSpec(f"project:{project_b.casefold()}", "managed_by", f"person:{manager.casefold()}", moments["manager"]),),
        ),
        (
            "manager_city",
            PRIMARY_NAMESPACE,
            f"{manager}'s home city is {manager_city}.",
            (ClaimSpec(f"person:{manager.casefold()}", "home_city", manager_city, moments["manager_city"]),),
        ),
        (
            "conflict_a",
            PRIMARY_NAMESPACE,
            "The calendar records review Omega in the Blue room.",
            (ClaimSpec("meeting:review_omega", "meeting_room", "Blue", moments["conflict_a"], authority="calendar"),),
        ),
        (
            "secret_primary",
            PRIMARY_NAMESPACE,
            f"The recovery code for {person} is ALPHA-17.",
            (ClaimSpec(person_key, "recovery_code", "ALPHA-17", moments["secret_primary"]),),
        ),
        (
            "secret_other",
            OTHER_NAMESPACE,
            f"The recovery code for {person} is OMEGA-99.",
            (ClaimSpec(person_key, "recovery_code", "OMEGA-99", moments["secret_other"]),),
        ),
        (
            "growth_target",
            PRIMARY_NAMESPACE,
            "The Quartz device has serial number QZ-4401.",
            (ClaimSpec("device:quartz", "serial_number", "QZ-4401", moments["growth_target"]),),
        ),
        (
            "theme_new",
            PRIMARY_NAMESPACE,
            f"{person} changed the interface preference to the dark theme.",
            (
                ClaimSpec(
                    person_key,
                    "preferred_theme",
                    "dark",
                    moments["theme_new"],
                    supersedes_source_id=source["theme_old"],
                ),
            ),
        ),
        (
            "project_release",
            PRIMARY_NAMESPACE,
            f"Project {project_a} was released on 2026-03-05.",
            (ClaimSpec(f"project:{project_a.casefold()}", "released_on", "2026-03-05", moments["project_release"]),),
        ),
        (
            "conflict_b",
            PRIMARY_NAMESPACE,
            "The facilities note records review Omega in the Green room.",
            (ClaimSpec("meeting:review_omega", "meeting_room", "Green", moments["conflict_b"], authority="facilities"),),
        ),
    ]

    events: list[SyntheticEvent] = []
    for sequence, (name, namespace, sentence, claims) in enumerate(specifications, start=1):
        events.append(
            _event(
                sequence=sequence,
                source_id=source[name],
                namespace=namespace,
                occurred_at=moments[name],
                sentence=sentence,
                claims=claims,
            )
        )

    distractor_base = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    for index in range(event_count - MIN_EVENTS):
        sequence = len(events) + 1
        suffix = rng.randrange(100_000, 999_999)
        subject = f"device:quartz_accessory_{index:06d}"
        value = f"QA-{suffix}"
        occurred_at = _iso(distractor_base + timedelta(minutes=index))
        claim = ClaimSpec(subject, "serial_number", value, occurred_at)
        events.append(
            _event(
                sequence=sequence,
                source_id=f"{prefix}/distractor/{index:08d}",
                namespace=PRIMARY_NAMESPACE,
                occurred_at=occurred_at,
                sentence=f"Quartz accessory {index} has serial number {value}.",
                claims=(claim,),
            )
        )

    final_sequence = len(events)
    queries = (
        QueryCase(
            "direct-home-city",
            "direct",
            final_sequence,
            PRIMARY_NAMESPACE,
            f"What is {person}'s home_city?",
            (source["direct"],),
            expected_values=(direct_city,),
        ),
        QueryCase(
            "current-before-update",
            "current",
            2,
            PRIMARY_NAMESPACE,
            f"What is {person}'s current preferred_theme?",
            (source["theme_old"],),
            expected_values=("light",),
        ),
        QueryCase(
            "current-after-update",
            "current",
            final_sequence,
            PRIMARY_NAMESPACE,
            f"What is {person}'s current preferred_theme?",
            (source["theme_new"],),
            (source["theme_old"],),
            expected_values=("dark",),
        ),
        QueryCase(
            "historical-before-update",
            "historical",
            final_sequence,
            PRIMARY_NAMESPACE,
            f"What was {person}'s preferred_theme as of 2026-01-15?",
            (source["theme_old"],),
            (source["theme_new"],),
            as_of="2026-01-15T23:59:59Z",
            expected_values=("light",),
        ),
        QueryCase(
            "temporal-project-order",
            "temporal",
            final_sequence,
            PRIMARY_NAMESPACE,
            f"Compare when project {project_a} started_on and released_on. Did it start first?",
            (source["project_start"], source["project_release"]),
            expected_values=("yes",),
        ),
        QueryCase(
            "multi-hop-manager-city",
            "multi_hop",
            final_sequence,
            PRIMARY_NAMESPACE,
            f"For project {project_b}, what home_city belongs to the person in managed_by?",
            (source["manager"], source["manager_city"]),
            expected_values=(manager_city,),
        ),
        QueryCase(
            "contradiction-meeting-room",
            "contradiction",
            final_sequence,
            PRIMARY_NAMESPACE,
            "What meeting_room is recorded for review Omega? Surface conflicting records.",
            (source["conflict_a"], source["conflict_b"]),
            expected_values=("Blue", "Green"),
        ),
        QueryCase(
            "abstention-unknown",
            "abstention",
            final_sequence,
            PRIMARY_NAMESPACE,
            "What is Zephyr-999's xylophonic_allergy?",
            (),
            expected_answerable=False,
        ),
        QueryCase(
            "abstention-known-entity-unseen-predicate",
            "abstention",
            final_sequence,
            PRIMARY_NAMESPACE,
            f"What is {person}'s xylophonic_allergy?",
            (),
            expected_answerable=False,
        ),
        QueryCase(
            "abstention-unknown-entity-seen-predicate",
            "abstention",
            final_sequence,
            PRIMARY_NAMESPACE,
            "What is Zephyr-999's home_city?",
            (),
            expected_answerable=False,
        ),
        QueryCase(
            "distractor-growth-serial",
            "distractor_growth",
            final_sequence,
            PRIMARY_NAMESPACE,
            "What is the serial_number for the Quartz device, not an accessory?",
            (source["growth_target"],),
            expected_values=("QZ-4401",),
        ),
        QueryCase(
            "namespace-recovery-code",
            "namespace_isolation",
            final_sequence,
            PRIMARY_NAMESPACE,
            f"What is the recovery_code for {person}?",
            (source["secret_primary"],),
            (source["secret_other"],),
            expected_values=("ALPHA-17",),
        ),
    )
    return SyntheticSuite(seed, event_count, tuple(events), queries)
