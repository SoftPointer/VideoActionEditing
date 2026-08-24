from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import materialize_graft_a_lite_terminal_admission_v1 as materializer  # noqa: E402


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_sha(value: object) -> str:
    return _sha(materializer.canonical_json_bytes(value))


def _line(value: object) -> bytes:
    return materializer.canonical_json_bytes(value) + b"\n"


def _seal(core: dict[str, object]) -> dict[str, object]:
    return {**deepcopy(core), "receipt_digest": _object_sha(core)}


def _sacct_raw(
    *,
    state: str = "COMPLETED",
    exit_code: str = "0:0",
    start: str = "2026-08-10T12:00:00",
    end: str = "2026-08-10T12:00:05",
    elapsed: str = "00:00:05",
    node: str = "auh7-1b-gpu-186",
    job_id: str = materializer.JOB_ID,
) -> bytes:
    return (
        "|".join((job_id, state, exit_code, start, end, elapsed, node)) + "|\n"
    ).encode("ascii")


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.release = self.root / "release"
        self.release.mkdir()
        self.stem = self.release / "job132549-canary4"
        self.manifest = self.stem.with_name(
            f"{self.stem.name}{materializer.MANIFEST_SUFFIX}"
        )
        self.producer = self.stem.with_name(
            f"{self.stem.name}{materializer.PRODUCER_SUFFIX}"
        )
        self.execution = self.stem.with_name(
            f"{self.stem.name}{materializer.EXECUTION_SUFFIX}"
        )
        self.submission = self.stem.with_name(
            f"{self.stem.name}{materializer.SUBMISSION_SUFFIX}"
        )
        self.output = self.stem.with_name(
            f"{self.stem.name}{materializer.TERMINAL_SUFFIX}"
        )
        self.rows: list[dict[str, object]] = []
        for index, (iid, split, update, confirmation) in enumerate(
            materializer.CANARY4
        ):
            core = {
                "schema_version": materializer.ROW_SCHEMA,
                "release_mode": "canary4",
                "row_index": index,
                "iid": iid,
                "split": split,
                "optimizer_update_authorized": update,
                "optimizer_confirmation_only": confirmation,
            }
            self.rows.append({**core, "row_digest": _object_sha(core)})
        self.execution_job_id = materializer.JOB_ID
        self.submission_job_id = materializer.JOB_ID
        self.submission_job_success: object = None
        self.submission_terminal = False
        self.write_artifacts()

    @staticmethod
    def _write(path: Path, raw: bytes) -> None:
        if path.exists():
            path.chmod(0o644)
            path.unlink()
        path.write_bytes(raw)
        path.chmod(0o444)

    def write_artifacts(self) -> None:
        rows = []
        for row in self.rows:
            core = deepcopy(row)
            core.pop("row_digest", None)
            rows.append({**core, "row_digest": _object_sha(core)})
        self.rows = rows
        manifest_raw = b"".join(_line(row) for row in rows)
        self._write(self.manifest, manifest_raw)

        producer_core = {
            "schema_version": materializer.PRODUCER_SCHEMA,
            "status": "complete",
            "release_mode": "canary4",
            "artifact": {
                "manifest_rows": 4,
                "manifest_sha256": _sha(manifest_raw),
            },
        }
        producer = _seal(producer_core)
        producer_raw = _line(producer)
        self._write(self.producer, producer_raw)

        execution_core = {
            "schema_version": materializer.EXECUTION_SCHEMA,
            "status": "complete",
            "successful_return": True,
            "builder_successful_return": True,
            "slurm": {"job_id": self.execution_job_id},
            "outputs": {
                "logical_output_stem": str(self.stem),
                "manifest_rows": 4,
                "producer_receipt_digest": producer["receipt_digest"],
                "manifest": {"sha256": _sha(manifest_raw)},
                "producer_receipt": {"sha256": _sha(producer_raw)},
            },
            "failure_semantics": {
                "consumer_must_also_require_slurm_completed_exit_zero": True,
                "receipt_alone_proves_successful_process_return": False,
            },
        }
        execution = _seal(execution_core)
        execution_raw = _line(execution)
        self._write(self.execution, execution_raw)

        submission_core = {
            "schema_version": materializer.SUBMISSION_SCHEMA,
            "status": "submitted",
            "submission_success": True,
            "job_success": self.submission_job_success,
            "job_terminal_state_observed": self.submission_terminal,
            "submitted_job": {"job_id": self.submission_job_id},
            "outputs": {
                "logical_output_stem": str(self.stem),
                "submission_receipt_path": str(self.submission),
            },
        }
        submission = _seal(submission_core)
        submission_raw = _line(submission)
        self._write(self.submission, submission_raw)
        self.pins = materializer.TerminalArtifactPins(
            manifest_sha256=_sha(manifest_raw),
            producer_receipt_sha256=_sha(producer_raw),
            execution_receipt_sha256=_sha(execution_raw),
            submission_receipt_sha256=_sha(submission_raw),
            materializer_implementation_sha256=_sha(
                Path(materializer.__file__).read_bytes()
            ),
            materializer_runtime_sha256="f" * 64,
        )

    def observation(
        self, raw: bytes | None = None
    ) -> materializer._ParsedSacctObservation:
        return materializer._parse_sacct_stdout(
            _sacct_raw() if raw is None else raw,
            runtime_sha="f" * 64,
        )

    def materialize(self, raw: bytes | None = None) -> materializer.PublishedAdmission:
        observation = self.observation(raw)
        with mock.patch.object(
            materializer, "_observe_production_sacct", return_value=observation
        ):
            return materializer.materialize_graft_a_lite_terminal_admission(
                manifest_path=self.manifest,
                producer_receipt_path=self.producer,
                execution_receipt_path=self.execution,
                submission_receipt_path=self.submission,
                output_path=self.output,
                pins=self.pins,
            )


class MaterializeGraftALiteTerminalAdmissionTests(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory[str], _Fixture]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary, _Fixture(Path(temporary.name))

    def test_production_shape_is_canonical_create_only_0444_and_cross_bound(self) -> None:
        _, fixture = self.fixture()
        result = fixture.materialize()
        self.assertEqual(result.path, fixture.output)
        self.assertEqual(stat.S_IMODE(result.path.stat().st_mode), 0o444)
        self.assertEqual(result.path.stat().st_nlink, 1)
        raw = result.path.read_bytes()
        self.assertEqual(_sha(raw), result.sha256)
        receipt = json.loads(raw)
        self.assertEqual(_line(receipt), raw)
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(declared, _object_sha(unsigned))
        self.assertEqual(receipt["schema_version"], materializer.TERMINAL_SCHEMA)
        self.assertEqual(receipt["status"], "admitted")
        self.assertEqual(
            receipt["sacct_admission"]["queried_fields"],
            list(materializer.ADMISSION_DECISION_FIELDS),
        )
        self.assertEqual(receipt["sacct_admission"]["state"], "COMPLETED")
        self.assertEqual(receipt["sacct_admission"]["exit_code"], "0:0")
        self.assertEqual(
            receipt["sacct_admission"]["selected_record_sha256"],
            _object_sha(
                dict(
                    zip(
                        materializer.SACCT_QUERY_FIELDS,
                        _sacct_raw().decode("ascii").rstrip("|\n").split("|"),
                    )
                )
            ),
        )
        bindings = receipt["artifact_bindings"]
        self.assertEqual(bindings["manifest_file_sha256"], fixture.pins.manifest_sha256)
        self.assertEqual(
            bindings["producer_receipt_file_sha256"],
            fixture.pins.producer_receipt_sha256,
        )
        self.assertEqual(
            bindings["execution_receipt_file_sha256"],
            fixture.pins.execution_receipt_sha256,
        )
        self.assertEqual(
            bindings["submission_receipt_file_sha256"],
            fixture.pins.submission_receipt_sha256,
        )
        for name in (
            "producer_receipt_digest",
            "execution_receipt_digest",
            "submission_receipt_digest",
        ):
            self.assertRegex(bindings[name], r"^[0-9a-f]{64}$")
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        materializer_record = receipt["materializer"]
        self.assertEqual(
            materializer_record["implementation_sha256"],
            _sha(Path(materializer.__file__).read_bytes()),
        )
        self.assertEqual(materializer_record["runtime_sha256"], "f" * 64)
        self.assertTrue(materializer_record["independent_of_submitted_job_process"])

    def test_existing_output_is_preserved_and_blocks_before_sacct(self) -> None:
        _, fixture = self.fixture()
        fixture.output.write_bytes(b"preexisting")
        fixture.output.chmod(0o444)
        with mock.patch.object(materializer, "_observe_production_sacct") as observer:
            with self.assertRaisesRegex(materializer.TerminalAdmissionError, "already exists"):
                materializer.materialize_graft_a_lite_terminal_admission(
                    manifest_path=fixture.manifest,
                    producer_receipt_path=fixture.producer,
                    execution_receipt_path=fixture.execution,
                    submission_receipt_path=fixture.submission,
                    output_path=fixture.output,
                    pins=fixture.pins,
                )
        observer.assert_not_called()
        self.assertEqual(fixture.output.read_bytes(), b"preexisting")

    def test_fake_cpu_seam_is_explicitly_test_only_and_cannot_publish(self) -> None:
        _, fixture = self.fixture()
        with mock.patch.object(
            materializer.subprocess, "run", side_effect=AssertionError("must not run")
        ):
            result = materializer._build_with_test_sacct(
                manifest_path=fixture.manifest,
                producer_receipt_path=fixture.producer,
                execution_receipt_path=fixture.execution,
                submission_receipt_path=fixture.submission,
                pins=fixture.pins,
                fake_sacct_stdout=_sacct_raw(),
            )
        self.assertIsInstance(result, materializer.TestOnlyAdmissionObservation)
        self.assertTrue(result.test_only)
        self.assertFalse(result.publication_eligible)
        receipt = json.loads(result.receipt_bytes)
        self.assertEqual(receipt["schema_version"], materializer.TEST_TERMINAL_SCHEMA)
        self.assertEqual(receipt["status"], "test_only_untrusted")
        self.assertEqual(receipt["sacct_admission"]["source"], "test_only_fake_sacct")
        output = materializer._pin_output_parent(
            fixture.output, expected_stem=fixture.stem
        )
        self.addCleanup(os.close, output.fd)
        self.assertNotIn(
            "payload",
            inspect.signature(materializer._publish_production_at).parameters,
        )
        with self.assertRaises(TypeError):
            materializer._publish_production_at(output, result)  # type: ignore[arg-type]
        self.assertFalse(fixture.output.exists())

    def test_tamper_and_resealed_semantic_changes_fail_closed(self) -> None:
        _, fixture = self.fixture()
        fixture.manifest.chmod(0o644)
        fixture.manifest.write_bytes(fixture.manifest.read_bytes() + b" ")
        fixture.manifest.chmod(0o444)
        with self.assertRaisesRegex(materializer.TerminalAdmissionError, "external pin"):
            fixture.materialize()

        _, fixture = self.fixture()
        fixture.submission_job_success = True
        fixture.submission_terminal = True
        fixture.write_artifacts()
        with self.assertRaisesRegex(materializer.TerminalAdmissionError, "remain non-terminal"):
            fixture.materialize()

        _, fixture = self.fixture()
        fixture.execution_job_id = "132548"
        fixture.write_artifacts()
        with self.assertRaisesRegex(materializer.TerminalAdmissionError, "execution receipt binding"):
            fixture.materialize()

    def test_bad_scheduler_status_exit_time_node_and_extra_rows_fail(self) -> None:
        for raw, message in (
            (_sacct_raw(state="FAILED", exit_code="1:0"), "not COMPLETED"),
            (_sacct_raw(exit_code="1:0"), "not COMPLETED"),
            (_sacct_raw(start="Unknown"), "Start time"),
            (_sacct_raw(end="2026-08-10T11:59:59"), "End precedes"),
            (_sacct_raw(elapsed="bad"), "Elapsed differs"),
            (_sacct_raw(elapsed="00:00:04"), "inconsistent"),
            (_sacct_raw(node="Unknown"), "NodeList differs"),
            (_sacct_raw() + _sacct_raw(job_id="132549.batch"), "exactly one"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(materializer.TerminalAdmissionError, message):
                    materializer._parse_sacct_stdout(
                        raw,
                        runtime_sha="f" * 64,
                    )

    def test_artifact_replacement_race_after_sacct_has_no_receipt(self) -> None:
        _, fixture = self.fixture()

        def observe_and_replace(
            **_: object,
        ) -> materializer._ParsedSacctObservation:
            observation = fixture.observation()
            fixture.manifest.unlink()
            fixture.manifest.write_bytes(b"replacement")
            fixture.manifest.chmod(0o444)
            return observation

        with mock.patch.object(
            materializer, "_observe_production_sacct", side_effect=observe_and_replace
        ):
            with self.assertRaisesRegex(materializer.TerminalAdmissionError, "identity changed"):
                materializer.materialize_graft_a_lite_terminal_admission(
                    manifest_path=fixture.manifest,
                    producer_receipt_path=fixture.producer,
                    execution_receipt_path=fixture.execution,
                    submission_receipt_path=fixture.submission,
                    output_path=fixture.output,
                    pins=fixture.pins,
                )
        self.assertFalse(fixture.output.exists())

    def test_modes_hardlinks_suffix_and_relative_paths_fail(self) -> None:
        temporary, fixture = self.fixture()
        fixture.execution.chmod(0o644)
        with self.assertRaisesRegex(materializer.TerminalAdmissionError, "mode-0444"):
            fixture.materialize()

        _, fixture = self.fixture()
        hardlink = Path(temporary.name).resolve() / "producer-hardlink"
        os.link(fixture.producer, hardlink)
        with self.assertRaisesRegex(materializer.TerminalAdmissionError, "link-count-one"):
            fixture.materialize()

        _, fixture = self.fixture()
        with self.assertRaisesRegex(materializer.TerminalAdmissionError, "sibling suffix"):
            materializer.materialize_graft_a_lite_terminal_admission(
                manifest_path=fixture.manifest,
                producer_receipt_path=fixture.producer,
                execution_receipt_path=fixture.execution,
                submission_receipt_path=fixture.submission,
                output_path=fixture.release / "wrong.json",
                pins=fixture.pins,
            )

    def test_public_api_and_cli_accept_no_live_status_or_command_override(self) -> None:
        parameters = inspect.signature(
            materializer.materialize_graft_a_lite_terminal_admission
        ).parameters
        for forbidden in (
            "state",
            "exit_code",
            "sacct_stdout",
            "sacct_path",
            "command",
            "job_id",
        ):
            self.assertNotIn(forbidden, parameters)
        options = {
            option
            for action in materializer.build_parser()._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--state",
            "--exit-code",
            "--sacct-stdout",
            "--sacct-path",
            "--job-id",
        ):
            self.assertNotIn(forbidden, options)
        source = inspect.getsource(materializer._observe_production_sacct)
        self.assertIn("SACCT_PATH", source)
        self.assertEqual(materializer.SACCT_PATH, "/usr/bin/sacct")
        self.assertEqual(
            materializer.SACCT_QUERY_FIELDS,
            ("JobIDRaw", "State", "ExitCode", "Start", "End", "Elapsed", "NodeList"),
        )
        self.assertNotIn(
            "production_boundary",
            inspect.signature(materializer._parse_sacct_stdout).parameters,
        )

    def test_external_materializer_implementation_and_runtime_pins_are_required(self) -> None:
        _, fixture = self.fixture()
        fixture.pins = materializer.TerminalArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            materializer_implementation_sha256="e" * 64,
            materializer_runtime_sha256=fixture.pins.materializer_runtime_sha256,
        )
        with mock.patch.object(materializer, "_observe_production_sacct") as observer:
            with self.assertRaisesRegex(
                materializer.TerminalAdmissionError, "implementation differs"
            ):
                fixture.materialize()
        observer.assert_not_called()
        self.assertFalse(fixture.output.exists())

        _, fixture = self.fixture()
        fixture.pins = materializer.TerminalArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            materializer_implementation_sha256=(
                fixture.pins.materializer_implementation_sha256
            ),
            materializer_runtime_sha256="e" * 64,
        )
        with self.assertRaisesRegex(
            materializer.TerminalAdmissionError, "runtime binding differs"
        ):
            fixture.materialize()
        self.assertFalse(fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
