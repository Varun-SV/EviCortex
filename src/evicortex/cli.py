"""Dependency-free command-line interface for EviCortex.

The CLI deliberately requires an explicit database and namespace for every command
that touches application data.  This keeps namespace selection visible at the trust
boundary instead of falling back to a surprising process-wide default.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Mapping, Sequence, TextIO

from . import __version__
from .budget import BudgetTooSmallError, DEFAULT_BUDGET_TOKENS
from .engine import EviCortex
from .errors import EviCortexError
from .models import ClaimInput, MemoryRecord
from .retrieval import MemoryCapsule


EXIT_OK = 0
EXIT_OPERATIONAL_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_INTEGRITY_ERROR = 3
MAX_INPUT_BYTES = 8 * 1024 * 1024


class CLIInputError(ValueError):
    """An actionable problem with command-line or file input."""


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_scope_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        required=True,
        metavar="PATH",
        help="SQLite database path (required; use :memory: explicitly if desired)",
    )
    parser.add_argument(
        "--namespace",
        required=True,
        metavar="NAME",
        help="authorized logical namespace (required; there is no implicit default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit one machine-readable JSON object",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the public argument parser."""

    parser = argparse.ArgumentParser(
        prog="evicortex",
        description="Evidence-preserving, temporal local memory CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON object (also accepted after a command)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    remember = subparsers.add_parser(
        "remember", help="store immutable evidence and optional structured claims"
    )
    _add_scope_options(remember)
    content = remember.add_mutually_exclusive_group(required=True)
    content.add_argument("--content", help="evidence text")
    content.add_argument(
        "--content-file",
        metavar="PATH",
        help="read evidence as UTF-8 from PATH; use - for stdin",
    )
    remember.add_argument("--kind", default="episode", help="evidence kind")
    remember.add_argument("--source", default="", help="source citation label")
    remember.add_argument("--occurred-at", help="ISO-8601 event time")
    remember.add_argument("--recorded-at", help="ISO-8601 recording time")
    metadata = remember.add_mutually_exclusive_group()
    metadata.add_argument("--metadata", help="JSON object with evidence metadata")
    metadata.add_argument(
        "--metadata-file",
        metavar="PATH",
        help="read a UTF-8 JSON object from PATH; use - for stdin",
    )
    remember.add_argument(
        "--claim",
        action="append",
        default=[],
        metavar="JSON",
        help="claim JSON object; may be repeated",
    )
    remember.add_argument(
        "--claim-file",
        action="append",
        default=[],
        metavar="PATH",
        help="UTF-8 claim JSON object or array; may be repeated; use - for stdin",
    )
    remember.add_argument("--idempotency-key", help="namespace-scoped retry key")

    recall = subparsers.add_parser(
        "recall", help="retrieve a deterministic capsule under a hard token budget"
    )
    _add_scope_options(recall)
    query = recall.add_mutually_exclusive_group(required=True)
    query.add_argument("--query", help="retrieval query")
    query.add_argument(
        "--query-file",
        metavar="PATH",
        help="read query as UTF-8 from PATH; use - for stdin",
    )
    recall.add_argument(
        "--budget-tokens", type=_positive_integer, default=DEFAULT_BUDGET_TOKENS
    )
    recall.add_argument("--as-of", help="ISO-8601 historical cutoff")
    recall.add_argument("--limit", type=_positive_integer, default=20)

    expand = subparsers.add_parser(
        "expand", help="return exact immutable evidence by cited memory ID"
    )
    _add_scope_options(expand)
    expand.add_argument("--memory-id", required=True, help="cited evidence ID")

    check = subparsers.add_parser(
        "check", help="verify integrity and show counts for one namespace"
    )
    _add_scope_options(check)

    demo = subparsers.add_parser(
        "demo", help="run a deterministic offline temporal-memory demonstration"
    )
    demo.add_argument(
        "--database",
        metavar="PATH",
        help="persist the demo in PATH (otherwise a temporary database is used)",
    )
    demo.add_argument("--namespace", default="evicortex/demo")
    demo.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    demo.add_argument(
        "--budget-tokens", type=_positive_integer, default=300
    )
    return parser


def _read_stdin_utf8(*, label: str, stdin: TextIO) -> str:
    """Read bounded UTF-8 from stdin without depending on the console locale."""

    binary = getattr(stdin, "buffer", None)
    if binary is not None:
        try:
            raw = binary.read(MAX_INPUT_BYTES + 1)
        except OSError as error:
            raise CLIInputError(f"cannot read {label} from stdin: {error}") from error
        if not isinstance(raw, bytes):
            raise CLIInputError(f"cannot read {label} as bytes from stdin")
        if len(raw) > MAX_INPUT_BYTES:
            raise CLIInputError(f"{label} exceeds the {MAX_INPUT_BYTES}-byte limit")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CLIInputError(f"{label} from stdin is not valid UTF-8") from error

    try:
        text = stdin.read(MAX_INPUT_BYTES + 1)
    except OSError as error:
        raise CLIInputError(f"cannot read {label} from stdin: {error}") from error
    if not isinstance(text, str):
        raise CLIInputError(f"cannot read {label} as text from stdin")
    return _bounded_inline(text, label=label)


def _read_utf8(path: str, *, label: str, stdin: TextIO) -> str:
    if path == "-":
        return _read_stdin_utf8(label=label, stdin=stdin)
    try:
        with Path(path).open("rb") as source:
            raw = source.read(MAX_INPUT_BYTES + 1)
    except OSError as error:
        raise CLIInputError(f"cannot read {label} file {path!r}: {error}") from error
    if len(raw) > MAX_INPUT_BYTES:
        raise CLIInputError(f"{label} exceeds the {MAX_INPUT_BYTES}-byte limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CLIInputError(f"{label} file {path!r} is not valid UTF-8") from error


def _bounded_inline(value: str, *, label: str) -> str:
    if len(value.encode("utf-8")) > MAX_INPUT_BYTES:
        raise CLIInputError(f"{label} exceeds the {MAX_INPUT_BYTES}-byte limit")
    return value


def _validate_database_path(database: str, *, must_exist: bool) -> None:
    if database == ":memory:":
        return
    path = Path(database)
    if path.exists():
        if not path.is_file():
            raise CLIInputError(f"database path is not a file: {database!r}")
        return
    if must_exist:
        raise CLIInputError(
            f"database does not exist: {database!r}; create it with remember first"
        )
    parent = path.parent
    if not parent.exists():
        raise CLIInputError(f"database parent directory does not exist: {str(parent)!r}")
    if not parent.is_dir():
        raise CLIInputError(f"database parent path is not a directory: {str(parent)!r}")


def _scope(
    args: argparse.Namespace, *, database_must_exist: bool = False
) -> tuple[str, str]:
    """Validate database selection and return its normalized namespace."""

    if not args.database.strip():
        raise CLIInputError("--database must be a non-empty path")
    _validate_database_path(args.database, must_exist=database_must_exist)
    namespace = args.namespace.strip()
    if not namespace:
        raise CLIInputError("--namespace must be non-empty")
    return args.database, namespace


def _validate_remember_stdin(args: argparse.Namespace) -> None:
    consumers = [
        args.content_file == "-",
        args.metadata_file == "-",
        *(path == "-" for path in args.claim_file),
    ]
    if sum(consumers) > 1:
        raise CLIInputError(
            "stdin ('-') may be selected by only one of --content-file, "
            "--metadata-file, or --claim-file"
        )


def _json_value(text: str, *, label: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise CLIInputError(
            f"{label} is not valid JSON at line {error.lineno}, column {error.colno}"
        ) from error


def _metadata(args: argparse.Namespace, stdin: TextIO) -> Mapping[str, Any] | None:
    if args.metadata is None and args.metadata_file is None:
        return None
    raw = (
        _bounded_inline(args.metadata, label="metadata")
        if args.metadata is not None
        else _read_utf8(args.metadata_file, label="metadata", stdin=stdin)
    )
    value = _json_value(raw, label="metadata")
    if not isinstance(value, dict):
        raise CLIInputError("metadata must be a JSON object")
    return value


_CLAIM_FIELDS = {
    "subject",
    "predicate",
    "value",
    "valid_from",
    "valid_to",
    "confidence",
    "authority",
    "supersedes",
}


def _claim_from_value(value: Any, *, label: str) -> ClaimInput:
    if not isinstance(value, dict):
        raise CLIInputError(f"{label} must be a JSON object")
    unknown = sorted(set(value) - _CLAIM_FIELDS)
    if unknown:
        raise CLIInputError(f"{label} has unknown field(s): {', '.join(unknown)}")
    missing = [name for name in ("subject", "predicate", "value") if name not in value]
    if missing:
        raise CLIInputError(f"{label} is missing required field(s): {', '.join(missing)}")
    try:
        return ClaimInput(**value)
    except TypeError as error:
        raise CLIInputError(f"invalid {label}: {error}") from error


def _claims(args: argparse.Namespace, stdin: TextIO) -> tuple[ClaimInput, ...]:
    values: list[Any] = []
    for index, raw in enumerate(args.claim, start=1):
        values.append(
            _json_value(_bounded_inline(raw, label=f"claim {index}"), label=f"claim {index}")
        )
    for index, path in enumerate(args.claim_file, start=1):
        value = _json_value(
            _read_utf8(path, label=f"claim file {index}", stdin=stdin),
            label=f"claim file {index}",
        )
        values.extend(value if isinstance(value, list) else [value])
    return tuple(
        _claim_from_value(value, label=f"claim {index}")
        for index, value in enumerate(values, start=1)
    )


def _serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _serializable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return value


def _memory_payload(record: MemoryRecord) -> dict[str, Any]:
    return _serializable(record)


def _capsule_payload(capsule: MemoryCapsule) -> dict[str, Any]:
    payload = _serializable(capsule)
    payload.update(
        evidence_ids=list(capsule.evidence_ids),
        estimated_tokens=capsule.estimated_tokens,
        prompt=capsule.to_prompt(),
    )
    return payload


def _emit_json(payload: Mapping[str, Any], stdout: TextIO) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        file=stdout,
    )


def _command_remember(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    database, namespace = _scope(args)
    _validate_remember_stdin(args)
    content = (
        _bounded_inline(args.content, label="content")
        if args.content is not None
        else _read_utf8(args.content_file, label="content", stdin=stdin)
    )
    with EviCortex.open(database) as memory:
        record = memory.remember(
            namespace=namespace,
            content=content,
            kind=args.kind,
            source=args.source,
            occurred_at=args.occurred_at,
            recorded_at=args.recorded_at,
            metadata=_metadata(args, stdin),
            idempotency_key=args.idempotency_key,
            claims=_claims(args, stdin),
        )
    if args.json:
        _emit_json({"command": "remember", "memory": _memory_payload(record)}, stdout)
    else:
        print(
            f"Remembered {record.id} in namespace {record.namespace} "
            f"({len(record.claims)} claim(s)).",
            file=stdout,
        )
    return EXIT_OK


def _command_recall(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    database, namespace = _scope(args, database_must_exist=True)
    query = (
        _bounded_inline(args.query, label="query")
        if args.query is not None
        else _read_utf8(args.query_file, label="query", stdin=stdin)
    )
    with EviCortex.open(database) as memory:
        capsule = memory.recall(
            namespace=namespace,
            query=query,
            budget_tokens=args.budget_tokens,
            as_of=args.as_of,
            limit=args.limit,
        )
    if args.json:
        _emit_json({"command": "recall", "capsule": _capsule_payload(capsule)}, stdout)
    else:
        print(capsule.to_prompt(), file=stdout)
        evidence = ", ".join(capsule.evidence_ids) or "none"
        print(
            f"Evidence IDs: {evidence} (estimated {capsule.estimated_tokens}/"
            f"{capsule.budget_tokens} tokens)",
            file=stdout,
        )
    return EXIT_OK


def _command_expand(args: argparse.Namespace, stdout: TextIO) -> int:
    database, namespace = _scope(args, database_must_exist=True)
    with EviCortex.open(database) as memory:
        record = memory.expand(namespace=namespace, memory_id=args.memory_id)
    if args.json:
        _emit_json({"command": "expand", "memory": _memory_payload(record)}, stdout)
    else:
        print(f"Evidence ID: {record.id}", file=stdout)
        print(f"Namespace: {record.namespace}", file=stdout)
        print(f"Content SHA-256: {record.content_hash}", file=stdout)
        print(record.content, file=stdout)
    return EXIT_OK


def _command_check(args: argparse.Namespace, stdout: TextIO) -> int:
    database, namespace = _scope(args, database_must_exist=True)
    with EviCortex.open(database) as memory:
        report = memory.verify_integrity(namespace=namespace)
        stats = memory.stats(namespace=namespace)
    payload = {
        "command": "check",
        "namespace": namespace,
        "integrity": _serializable(report),
        "stats": _serializable(stats),
    }
    if args.json:
        _emit_json(payload, stdout)
    else:
        print(
            f"Integrity: {'ok' if report.ok else 'FAILED'} "
            f"({report.checked_memories} memories checked)",
            file=stdout,
        )
        print(f"Memories: {stats.memories}; claims: {stats.claims}", file=stdout)
        for memory_id in report.corrupt_memory_ids:
            print(f"Corrupt evidence: {memory_id}", file=stdout)
        for issue in report.index_issues:
            print(f"Index issue: {issue}", file=stdout)
    return EXIT_OK if report.ok else EXIT_INTEGRITY_ERROR


def _populate_demo(database: str, namespace: str, budget_tokens: int) -> dict[str, Any]:
    with EviCortex.open(database) as memory:
        earlier = memory.remember(
            namespace=namespace,
            content="Ada prefers the light interface theme.",
            source="demo/session-1/turn-4",
            occurred_at="2026-01-02T10:00:00Z",
            recorded_at="2026-01-02T10:00:00Z",
            idempotency_key="evicortex-demo/theme-light/v1",
            claims=(
                ClaimInput(
                    subject="user:ada", predicate="prefers_theme", value="light"
                ),
            ),
        )
        latest = memory.remember(
            namespace=namespace,
            content="Ada switched to the dark interface theme.",
            source="demo/session-7/turn-12",
            occurred_at="2026-04-18T09:30:00Z",
            recorded_at="2026-04-18T09:30:00Z",
            idempotency_key="evicortex-demo/theme-dark/v1",
            claims=(
                ClaimInput(
                    subject="user:ada",
                    predicate="prefers_theme",
                    value="dark",
                    supersedes=earlier.claims[0].id,
                ),
            ),
        )
        current = memory.recall(
            namespace=namespace,
            query="Which interface theme does Ada prefer?",
            budget_tokens=budget_tokens,
        )
        historical = memory.recall(
            namespace=namespace,
            query="Which interface theme did Ada prefer?",
            as_of="2026-02-01T00:00:00Z",
            budget_tokens=budget_tokens,
        )
        exact = memory.expand(namespace=namespace, memory_id=latest.id)
    return {
        "command": "demo",
        "namespace": namespace,
        "current": _capsule_payload(current),
        "historical": _capsule_payload(historical),
        "expanded": _memory_payload(exact),
        "demonstrates": {
            "current_value": "dark",
            "historical_value": "light",
            "expanded_evidence_id": latest.id,
            "exact_expansion": exact.content == latest.content,
        },
    }


def _command_demo(args: argparse.Namespace, stdout: TextIO) -> int:
    namespace = args.namespace.strip()
    if not namespace:
        raise CLIInputError("--namespace must be non-empty")
    if args.database is not None:
        if not args.database.strip():
            raise CLIInputError("--database must be a non-empty path")
        _validate_database_path(args.database, must_exist=False)
        payload = _populate_demo(args.database, namespace, args.budget_tokens)
        payload["database"] = args.database
    else:
        with tempfile.TemporaryDirectory(prefix="evicortex-demo-") as directory:
            database = str(Path(directory) / "demo.sqlite3")
            payload = _populate_demo(database, namespace, args.budget_tokens)
        payload["database"] = "temporary"
    if args.json:
        _emit_json(payload, stdout)
    else:
        print("EviCortex offline temporal demo", file=stdout)
        print(f"Namespace: {payload['namespace']}", file=stdout)
        print("Current recall:", file=stdout)
        print(payload["current"]["prompt"], file=stdout)
        print(
            "Current evidence IDs: " + ", ".join(payload["current"]["evidence_ids"]),
            file=stdout,
        )
        print("Historical recall (as of 2026-02-01):", file=stdout)
        print(payload["historical"]["prompt"], file=stdout)
        print(
            "Historical evidence IDs: "
            + ", ".join(payload["historical"]["evidence_ids"]),
            file=stdout,
        )
        print(
            f"Exact expansion [{payload['expanded']['id']}]: "
            f"{payload['expanded']['content']}",
            file=stdout,
        )
    return EXIT_OK


def _error_message(error: BaseException) -> str:
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code.

    Streams are injectable to keep integrations and standard-library tests free of
    global redirection.  ``argparse`` still owns syntax errors and their conventional
    exit code 2.
    """

    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    try:
        if args.command == "remember":
            return _command_remember(args, input_stream, output_stream)
        if args.command == "recall":
            return _command_recall(args, input_stream, output_stream)
        if args.command == "expand":
            return _command_expand(args, output_stream)
        if args.command == "check":
            return _command_check(args, output_stream)
        if args.command == "demo":
            return _command_demo(args, output_stream)
        raise AssertionError(f"unhandled command: {args.command}")
    except (CLIInputError, BudgetTooSmallError, ValueError, TypeError) as error:
        message = _error_message(error)
        if getattr(args, "json", False):
            _emit_json(
                {"error": {"code": "invalid_input", "message": message}},
                error_stream,
            )
        else:
            print(f"evicortex: invalid input: {message}", file=error_stream)
        return EXIT_USAGE_ERROR
    except (EviCortexError, KeyError, sqlite3.Error, OSError) as error:
        message = _error_message(error)
        if getattr(args, "json", False):
            _emit_json(
                {"error": {"code": "operation_failed", "message": message}},
                error_stream,
            )
        else:
            print(f"evicortex: operation failed: {message}", file=error_stream)
        return EXIT_OPERATIONAL_ERROR


__all__ = ["build_parser", "main"]
