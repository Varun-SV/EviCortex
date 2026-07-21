from __future__ import annotations

from contextlib import closing
from io import BytesIO, StringIO, TextIOWrapper
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from evicortex.cli import (
    EXIT_OPERATIONAL_ERROR,
    EXIT_USAGE_ERROR,
    main,
)


class CLITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = str(Path(self.temporary.name) / "memory.sqlite3")

    def run_cli(
        self, *arguments: str, stdin: str = ""
    ) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        code = main(
            list(arguments),
            stdin=StringIO(stdin),
            stdout=stdout,
            stderr=stderr,
        )
        return code, stdout.getvalue(), stderr.getvalue()

    def remember(
        self,
        namespace: str,
        content: str,
        *extra: str,
    ) -> dict[str, object]:
        code, stdout, stderr = self.run_cli(
            "remember",
            "--database",
            self.database,
            "--namespace",
            namespace,
            "--content",
            content,
            "--json",
            *extra,
        )
        self.assertEqual(code, 0, stderr)
        return json.loads(stdout)["memory"]

    def test_remember_and_expand_round_trip_exact_evidence(self) -> None:
        record = self.remember(
            "tenant-a",
            "The exact evidence, including punctuation!",
            "--source",
            "conversation/7",
            "--metadata",
            '{"priority":2}',
            "--claim",
            '{"subject":"user","predicate":"likes","value":"tea"}',
        )

        code, stdout, stderr = self.run_cli(
            "expand",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--memory-id",
            str(record["id"]),
            "--json",
        )

        self.assertEqual(code, 0, stderr)
        expanded = json.loads(stdout)["memory"]
        self.assertEqual(expanded["content"], "The exact evidence, including punctuation!")
        self.assertEqual(expanded["content_hash"], record["content_hash"])
        self.assertEqual(expanded["metadata"], {"priority": 2})
        self.assertEqual(expanded["claims"][0]["value"], "tea")

    def test_content_file_stdin_and_claim_file_are_safe_inputs(self) -> None:
        claim_path = Path(self.temporary.name) / "claims.json"
        claim_path.write_text(
            '[{"subject":"Ada","predicate":"prefers","value":"concise"}]',
            encoding="utf-8",
        )
        hostile_text = "line one\n$(not executed); `still evidence`\nline three"

        code, stdout, stderr = self.run_cli(
            "remember",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--content-file",
            "-",
            "--claim-file",
            str(claim_path),
            "--json",
            stdin=hostile_text,
        )

        self.assertEqual(code, 0, stderr)
        memory = json.loads(stdout)["memory"]
        self.assertEqual(memory["content"], hostile_text)
        self.assertEqual(memory["claims"][0]["value"], "concise")

    def test_stdin_is_decoded_as_utf8_not_the_host_text_encoding(self) -> None:
        stdin = TextIOWrapper(BytesIO("café".encode()), encoding="cp1252")
        stdout = StringIO()
        stderr = StringIO()
        code = main(
            [
                "remember",
                "--database",
                self.database,
                "--namespace",
                "tenant-a",
                "--content-file",
                "-",
                "--json",
            ],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["memory"]["content"], "café")

        invalid_stdin = TextIOWrapper(BytesIO(b"\xff"), encoding="cp1252")
        stderr = StringIO()
        code = main(
            [
                "remember",
                "--database",
                self.database,
                "--namespace",
                "tenant-a",
                "--content-file",
                "-",
            ],
            stdin=invalid_stdin,
            stdout=StringIO(),
            stderr=stderr,
        )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("stdin is not valid UTF-8", stderr.getvalue())

    def test_only_one_remember_input_can_consume_stdin(self) -> None:
        code, _, stderr = self.run_cli(
            "remember",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--content-file",
            "-",
            "--claim-file",
            "-",
            stdin="evidence",
        )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("may be selected by only one", stderr)

    def test_query_file_and_hard_budget_are_reported(self) -> None:
        self.remember("tenant-a", "Ada stores the amber recovery phrase.")
        query_path = Path(self.temporary.name) / "query.txt"
        query_path.write_text("What is Ada's recovery phrase?", encoding="utf-8")

        code, stdout, stderr = self.run_cli(
            "recall",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--query-file",
            str(query_path),
            "--budget-tokens",
            "160",
            "--json",
        )

        self.assertEqual(code, 0, stderr)
        capsule = json.loads(stdout)["capsule"]
        self.assertLessEqual(capsule["estimated_tokens"], 160)
        self.assertEqual(capsule["budget_tokens"], 160)
        self.assertTrue(capsule["evidence_ids"])
        self.assertIn("amber", capsule["prompt"])

    def test_read_commands_never_create_a_misspelled_database(self) -> None:
        missing = str(Path(self.temporary.name) / "misspelled.sqlite3")
        for command, command_args in (
            ("recall", ["--query", "anything"]),
            ("expand", ["--memory-id", "mem_missing"]),
            ("check", []),
        ):
            with self.subTest(command=command):
                code, stdout, stderr = self.run_cli(
                    command,
                    "--database",
                    missing,
                    "--namespace",
                    "tenant-a",
                    "--json",
                    *command_args,
                )
                self.assertEqual(code, EXIT_USAGE_ERROR)
                self.assertEqual(stdout, "")
                self.assertIn("database does not exist", stderr)
                self.assertFalse(Path(missing).exists())

    def test_database_path_shape_errors_are_actionable(self) -> None:
        directory = Path(self.temporary.name) / "database-directory"
        directory.mkdir()
        code, _, stderr = self.run_cli(
            "check",
            "--database",
            str(directory),
            "--namespace",
            "tenant-a",
        )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("not a file", stderr)

        absent_parent_database = Path(self.temporary.name) / "absent" / "memory.db"
        code, _, stderr = self.run_cli(
            "remember",
            "--database",
            str(absent_parent_database),
            "--namespace",
            "tenant-a",
            "--content",
            "evidence",
        )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("parent directory does not exist", stderr)
        self.assertFalse(absent_parent_database.exists())

    def test_explicit_in_memory_database_remains_supported(self) -> None:
        code, stdout, stderr = self.run_cli(
            "check",
            "--database",
            ":memory:",
            "--namespace",
            "tenant-a",
            "--json",
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["stats"]["memories"], 0)

    def test_budget_too_small_is_actionable_json_without_traceback(self) -> None:
        self.remember("tenant-a", "A seed record makes the database explicit.")
        code, stdout, stderr = self.run_cli(
            "recall",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--query",
            "anything",
            "--budget-tokens",
            "1",
            "--json",
        )

        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertEqual(stdout, "")
        error = json.loads(stderr)["error"]
        self.assertEqual(error["code"], "invalid_input")
        self.assertIn("at least", error["message"])
        self.assertNotIn("Traceback", stderr)

    def test_namespace_isolation_applies_to_recall_expand_and_check(self) -> None:
        alpha = self.remember("tenant-a", "Alpha owns the cobalt token.")
        beta = self.remember("tenant-b", "Beta owns the secret cobalt token.")

        code, stdout, stderr = self.run_cli(
            "recall",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--query",
            "cobalt token",
            "--json",
        )
        self.assertEqual(code, 0, stderr)
        capsule = json.loads(stdout)["capsule"]
        self.assertIn(alpha["id"], capsule["evidence_ids"])
        self.assertNotIn(beta["id"], capsule["evidence_ids"])
        self.assertNotIn("Beta", stdout)

        code, stdout, stderr = self.run_cli(
            "expand",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--memory-id",
            str(beta["id"]),
            "--json",
        )
        self.assertEqual(code, EXIT_OPERATIONAL_ERROR)
        self.assertEqual(stdout, "")
        self.assertNotIn("Beta", stderr)

        code, stdout, stderr = self.run_cli(
            "check",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--json",
        )
        self.assertEqual(code, 0, stderr)
        checked = json.loads(stdout)
        self.assertEqual(checked["stats"]["memories"], 1)
        self.assertEqual(checked["stats"]["namespaces"], 1)
        self.assertEqual(checked["integrity"]["checked_memories"], 1)

    def test_check_normalizes_namespace_and_uses_integrity_exit_code(self) -> None:
        record = self.remember("tenant-a", "Evidence that will be corrupted.")
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "UPDATE memories SET content = ? WHERE id = ?",
                ("tampered", record["id"]),
            )
            connection.commit()

        code, stdout, stderr = self.run_cli(
            "check",
            "--database",
            self.database,
            "--namespace",
            "  tenant-a  ",
            "--json",
        )
        self.assertEqual(code, 3, stderr)
        result = json.loads(stdout)
        self.assertEqual(result["namespace"], "tenant-a")
        self.assertFalse(result["integrity"]["ok"])
        self.assertEqual(result["integrity"]["corrupt_memory_ids"], [record["id"]])

    def test_invalid_json_and_unknown_claim_fields_are_actionable(self) -> None:
        code, _, stderr = self.run_cli(
            "remember",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--content",
            "evidence",
            "--claim",
            '{"subject":"a","predicate":"b","value":"c","typo":1}',
        )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("unknown field(s): typo", stderr)
        self.assertNotIn("Traceback", stderr)

        code, _, stderr = self.run_cli(
            "remember",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--content",
            "evidence",
            "--metadata",
            "[1,2]",
        )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("JSON object", stderr)

    def test_input_size_and_utf8_errors_do_not_write(self) -> None:
        bad_path = Path(self.temporary.name) / "bad.bin"
        bad_path.write_bytes(b"\xff\xfe")
        code, _, stderr = self.run_cli(
            "remember",
            "--database",
            self.database,
            "--namespace",
            "tenant-a",
            "--content-file",
            str(bad_path),
        )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("not valid UTF-8", stderr)

        with patch("evicortex.cli.MAX_INPUT_BYTES", 4):
            code, _, stderr = self.run_cli(
                "remember",
                "--database",
                self.database,
                "--namespace",
                "tenant-a",
                "--content",
                "12345",
            )
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("4-byte limit", stderr)

    def test_data_commands_require_explicit_database_and_namespace(self) -> None:
        for arguments in (
            ["remember", "--content", "x"],
            ["recall", "--query", "x"],
            ["expand", "--memory-id", "mem_x"],
            ["check"],
        ):
            with self.subTest(command=arguments[0]):
                with self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertEqual(raised.exception.code, EXIT_USAGE_ERROR)

    def test_demo_is_offline_temporal_cited_and_exact(self) -> None:
        # The output switch is accepted globally as well as after subcommands.
        code, stdout, stderr = self.run_cli("--json", "demo")

        self.assertEqual(code, 0, stderr)
        demo = json.loads(stdout)
        self.assertEqual(demo["database"], "temporary")
        self.assertEqual(demo["demonstrates"]["current_value"], "dark")
        self.assertEqual(demo["demonstrates"]["historical_value"], "light")
        self.assertTrue(demo["demonstrates"]["exact_expansion"])
        self.assertEqual(demo["current"]["facts"][0]["value"], "dark")
        self.assertEqual(demo["historical"]["facts"][0]["value"], "light")
        self.assertIn(
            demo["expanded"]["id"], demo["current"]["evidence_ids"]
        )
        self.assertEqual(
            demo["expanded"]["content"],
            "Ada switched to the dark interface theme.",
        )

    def test_demo_with_database_is_idempotent(self) -> None:
        first_code, first_stdout, first_stderr = self.run_cli(
            "demo", "--database", self.database, "--json"
        )
        second_code, second_stdout, second_stderr = self.run_cli(
            "demo", "--database", self.database, "--json"
        )
        self.assertEqual(first_code, 0, first_stderr)
        self.assertEqual(second_code, 0, second_stderr)
        first = json.loads(first_stdout)
        second = json.loads(second_stdout)
        self.assertEqual(first["current"]["evidence_ids"], second["current"]["evidence_ids"])
        self.assertEqual(first["historical"]["evidence_ids"], second["historical"]["evidence_ids"])


if __name__ == "__main__":
    unittest.main()
