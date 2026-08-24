#!/usr/bin/env python3
"""Contract tests for the HOLD-only bone-removed-v2 create-once controller."""

from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "case01_bone_removed_v2_create_once_controller_v1.py"
SPEC = importlib.util.spec_from_file_location("bone_removed_v2_create_once_controller", PROGRAM)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)

ACCEPTANCE_PROGRAM = ROOT / "tools" / "case01_bone_removed_v2_acceptance_v1.py"
ACCEPTANCE_SPEC = importlib.util.spec_from_file_location(
    "bone_removed_v2_acceptance_for_controller_test", ACCEPTANCE_PROGRAM
)
assert ACCEPTANCE_SPEC is not None and ACCEPTANCE_SPEC.loader is not None
ACCEPTANCE = importlib.util.module_from_spec(ACCEPTANCE_SPEC)
ACCEPTANCE_SPEC.loader.exec_module(ACCEPTANCE)

ACCEPTANCE_TEST_PROGRAM = Path(__file__).with_name(
    "test_case01_bone_removed_v2_acceptance_v1.py"
)
ACCEPTANCE_TEST_SPEC = importlib.util.spec_from_file_location(
    "bone_removed_v2_acceptance_fixture_for_controller_test",
    ACCEPTANCE_TEST_PROGRAM,
)
assert ACCEPTANCE_TEST_SPEC is not None and ACCEPTANCE_TEST_SPEC.loader is not None
ACCEPTANCE_FIXTURE = importlib.util.module_from_spec(ACCEPTANCE_TEST_SPEC)
ACCEPTANCE_TEST_SPEC.loader.exec_module(ACCEPTANCE_FIXTURE)


def digest(tag: str) -> str:
    return hashlib.sha256(tag.encode("ascii")).hexdigest()


def populated_pins() -> dict:
    stage = Path("/authority/output/.bone-removed-v2-stage")
    final = Path("/authority/output/bone-removed-v2-bundle")
    return {
        "controller_program_path": "/authority/controller.py",
        "source": {
            "path": "/authority/source.mp4",
            "sha256": CONTROLLER.SOURCE_SHA256,
            "size": CONTROLLER.SOURCE_SIZE,
        },
        "sam2_receipt": {
            "path": "/authority/sam2/receipt.json",
            "sha256": CONTROLLER.SAM2_RECEIPT_SHA256,
            "size": CONTROLLER.SAM2_RECEIPT_SIZE,
        },
        "support_review_receipt": {
            "path": "/authority/support/review.json",
            "sha256": digest("support-review-file"),
            "size": 8000,
            "review_digest": digest("support-review-object"),
        },
        "python_runtime_manifest": {
            "path": "/authority/manifests/python-runtime.json",
            "sha256": digest("python-runtime-manifest"),
            "size": 1000,
            "tree_digest": digest("python-runtime-tree"),
            "tree_root": "/authority/runtime",
        },
        "vace_checkpoint_manifest": {
            "path": "/authority/manifests/vace-checkpoint.json",
            "sha256": digest("vace-checkpoint-manifest"),
            "size": 2000,
            "tree_digest": digest("vace-checkpoint-tree"),
            "tree_root": "/authority/checkpoint",
        },
        "vace_source_manifest": {
            "path": "/authority/manifests/vace-source.json",
            "sha256": digest("vace-source-manifest"),
            "size": 3000,
            "tree_digest": digest("vace-source-tree"),
            "tree_root": "/authority/VACE",
        },
        "python_bin": {
            "path": "/authority/runtime/bin/python3.12",
            "sha256": digest("python-bin"),
            "size": 9000,
        },
        "gpu_visible_device": "0",
        "child_environment": {
            "PYTHONHASHSEED": "20260822",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
            "PATH": "/authority/runtime/bin",
        },
        "bundle_staging_root": str(stage),
        "bundle_final_root": str(final),
        "asset_staging_root": str(stage / "assets"),
        "evidence_staging_root": str(stage / "evidence"),
        "asset_final_root": str(final / "assets"),
        "evidence_final_root": str(final / "evidence"),
        "attempt_receipt": "/authority/output/bone-removed-v2.attempt.json",
        "publication_receipt": str(final / "evidence" / "create_only_publication.json"),
        "producer_receipt": str(final / "evidence" / "producer_receipt.json"),
    }


def result_from_acceptance_fixture(fixture: dict) -> dict:
    return {
        "generator_fragment": copy.deepcopy(fixture["generator"]),
        "support_review_receipt": copy.deepcopy(fixture["support"]["review_receipt"]),
        "support_frame_masks": copy.deepcopy(fixture["support"]["frame_masks"]),
        "delivery_contract": {
            key: fixture["delivery_candidate"][key]
            for key in (
                "authority_scope", "identity_authority", "canonical_is_identity_authority"
            )
        },
        "construction_audit": copy.deepcopy(fixture["construction_audit"]),
        "claim_limits": copy.deepcopy(fixture["claim_limits"]),
    }


class HoldBeforeIOTests(unittest.TestCase):
    def test_frozen_state_and_every_deployment_class_are_blocked(self) -> None:
        self.assertEqual(CONTROLLER.EXECUTION_STATE, "HOLD_PRE_IO")
        blocked = set(CONTROLLER.blocked_pin_names())
        for expected in (
            "controller_program_path",
            "source.path",
            "sam2_receipt.path",
            "support_review_receipt.review_digest",
            "python_runtime_manifest.tree_digest",
            "vace_checkpoint_manifest.tree_digest",
            "vace_source_manifest.tree_digest",
            "python_bin.path",
            "gpu_visible_device",
            "child_environment",
            "bundle_staging_root",
            "bundle_final_root",
            "asset_staging_root",
            "evidence_staging_root",
            "asset_final_root",
            "evidence_final_root",
            "attempt_receipt",
            "publication_receipt",
            "producer_receipt",
        ):
            self.assertIn(expected, blocked)

    def test_main_returns_96_without_filesystem_or_subprocess_io(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(CONTROLLER.sys, "stderr", stderr), \
             mock.patch.object(CONTROLLER.os, "open") as open_mock, \
             mock.patch.object(CONTROLLER.os, "stat") as stat_mock, \
             mock.patch.object(CONTROLLER.subprocess, "Popen") as popen_mock, \
             mock.patch.object(CONTROLLER, "execute_create_once") as execute_mock:
            code = CONTROLLER.main([])
        self.assertEqual(code, 96)
        self.assertIn("HOLD_BEFORE_IO", stderr.getvalue())
        open_mock.assert_not_called()
        stat_mock.assert_not_called()
        popen_mock.assert_not_called()
        execute_mock.assert_not_called()

    def test_command_line_cannot_fill_or_override_blocked_pins(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(CONTROLLER.sys, "stderr", stderr), \
             mock.patch.object(CONTROLLER, "execute_create_once") as execute_mock:
            code = CONTROLLER.main(["--gpu-visible-device", "0"])
        self.assertEqual(code, 96)
        self.assertIn("blocked_pins=", stderr.getvalue())
        execute_mock.assert_not_called()

    def test_armed_state_still_holds_when_any_pin_is_blocked(self) -> None:
        pins = populated_pins()
        pins["vace_source_manifest"]["tree_digest"] = CONTROLLER.BLOCKED
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "HOLD_BEFORE_IO"):
            CONTROLLER.admit_pre_io(state=CONTROLLER.ARMED_STATE, pins=pins)

    def test_complete_typed_pins_pass_only_the_pure_admission(self) -> None:
        CONTROLLER.admit_pre_io(
            state=CONTROLLER.ARMED_STATE,
            pins=populated_pins(),
        )

    def test_hold_state_rejects_even_complete_pins(self) -> None:
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "state='HOLD_PRE_IO'"):
            CONTROLLER.admit_pre_io(pins=populated_pins())


class PinContractTests(unittest.TestCase):
    def test_hard_pins_frozen_generator_and_acceptance(self) -> None:
        self.assertEqual(
            (CONTROLLER.GENERATOR_SHA256, CONTROLLER.GENERATOR_SIZE),
            ("f6dc4edb5ea3da03e14dd00399a800c3af545379bd0030aeab0fc8e2a205ce86", 85957),
        )
        self.assertEqual(
            (CONTROLLER.ACCEPTANCE_SHA256, CONTROLLER.ACCEPTANCE_SIZE),
            ("7a11c7bc2c0e37b8f00dfcb21da7755f57856a433166e8c978fd400bbde16c51", 148113),
        )
        self.assertEqual(CONTROLLER.GENERATOR_PATH, ACCEPTANCE.GENERATOR_PATH)
        self.assertEqual(CONTROLLER.GENERATOR_SHA256, ACCEPTANCE.GENERATOR_SHA256)
        self.assertEqual(CONTROLLER.GENERATOR_SIZE, ACCEPTANCE.GENERATOR_SIZE)

    def test_controller_self_authority_avoids_impossible_embedded_hash_cycle(self) -> None:
        self.assertIn("controller_program_path", CONTROLLER.DYNAMIC_PINS)
        self.assertNotIn("controller_program", CONTROLLER.DYNAMIC_PINS)
        self.assertEqual(
            CONTROLLER.DYNAMIC_PINS["controller_program_path"], CONTROLLER.BLOCKED
        )

    def test_schema_names_match_frozen_acceptance(self) -> None:
        self.assertEqual(CONTROLLER.ATTEMPT_SCHEMA, ACCEPTANCE.ATTEMPT_SCHEMA)
        self.assertEqual(CONTROLLER.PUBLICATION_SCHEMA, ACCEPTANCE.PUBLICATION_SCHEMA)
        self.assertEqual(CONTROLLER.PRODUCER_SCHEMA, ACCEPTANCE.RECEIPT_SCHEMA)
        self.assertEqual(CONTROLLER.TREE_MANIFEST_SCHEMA, ACCEPTANCE.TREE_MANIFEST_SCHEMA)

    def test_rejects_bool_as_exact_integer_size(self) -> None:
        pins = populated_pins()
        pins["python_bin"]["size"] = True
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "size differs"):
            CONTROLLER.validate_dynamic_pins(pins)

    def test_rejects_uppercase_digest(self) -> None:
        pins = populated_pins()
        pins["support_review_receipt"]["sha256"] = "A" * 64
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "SHA-256 differs"):
            CONTROLLER.validate_dynamic_pins(pins)

    def test_rejects_relative_output(self) -> None:
        pins = populated_pins()
        pins["bundle_staging_root"] = "relative/stage"
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "not absolute"):
            CONTROLLER.validate_dynamic_pins(pins)

    def test_rejects_different_publication_parent(self) -> None:
        pins = populated_pins()
        pins["bundle_final_root"] = "/different-parent/final"
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "same-parent"):
            CONTROLLER.validate_dynamic_pins(pins)

    def test_rejects_attempt_receipt_inside_bundle(self) -> None:
        pins = populated_pins()
        pins["attempt_receipt"] = pins["bundle_staging_root"] + "/attempt.json"
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "inside bundle_staging_root"):
            CONTROLLER.validate_dynamic_pins(pins)

    def test_rejects_unapproved_inherited_environment_key(self) -> None:
        pins = populated_pins()
        pins["child_environment"]["HOME"] = "/home/operator"
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "unapproved key"):
            CONTROLLER.validate_dynamic_pins(pins)

    def test_rejects_multiple_gpu_device_list(self) -> None:
        pins = populated_pins()
        pins["gpu_visible_device"] = "0,1"
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "GPU device"):
            CONTROLLER.validate_dynamic_pins(pins)


class ReceiptAndChildTests(unittest.TestCase):
    def test_attempt_receipt_matches_frozen_acceptance_abi(self) -> None:
        pins = populated_pins()
        preflight = {
            "authority_replay_digest": digest("authority-replay"),
            "model_authorities": [
                {
                    "role": role,
                    "path": pins[role.replace("_tree", "_manifest")]["path"],
                    "sha256": pins[role.replace("_tree", "_manifest")]["sha256"],
                    "size": pins[role.replace("_tree", "_manifest")]["size"],
                }
                for role in CONTROLLER.MODEL_AUTHORITY_ROLES
            ],
        }
        value = CONTROLLER._build_attempt(
            pins, preflight, digest("controller")
        )
        self.assertEqual(value["schema_version"], ACCEPTANCE.ATTEMPT_SCHEMA)
        self.assertEqual(value["status"], "RESERVED_FRESH_BEFORE_GENERATION")
        payload = dict(value)
        observed = payload.pop("attempt_digest")
        self.assertEqual(observed, CONTROLLER.object_sha256(payload))
        self.assertTrue(all(value["preflight"].values()))

    def test_publication_receipt_is_exact_create_only_contract(self) -> None:
        pins = populated_pins()
        attempt = {"attempt_token": digest("attempt-token")}
        assets = {
            name: {"path": f"{pins['asset_final_root']}/{name}", "sha256": digest(name), "size": 1}
            for name in ("support", "canonical_candidate", "delivery_candidate")
        }
        value = CONTROLLER._build_publication(
            pins, attempt, digest("controller"), assets
        )
        self.assertEqual(value["schema_version"], ACCEPTANCE.PUBLICATION_SCHEMA)
        self.assertEqual(value["status"], "PUBLISHED_FRESH_NO_REPLACE")
        self.assertTrue(value["publication"]["atomic_rename_noreplace"])
        self.assertFalse(value["publication"]["overwrite_performed"])
        payload = dict(value)
        observed = payload.pop("publication_digest")
        self.assertEqual(observed, CONTROLLER.object_sha256(payload))

    def test_producer_builder_passes_frozen_structural_validator(self) -> None:
        fixture = ACCEPTANCE_FIXTURE.producer_receipt()
        pins = populated_pins()
        pins["source"] = copy.deepcopy(fixture["source"])
        pins["sam2_receipt"] = copy.deepcopy(fixture["mask_authority"]["receipt"])
        result = result_from_acceptance_fixture(fixture)
        built = CONTROLLER._build_producer(
            pins,
            result,
            fixture["create_only_authority"]["controller_program"],
            fixture["create_only_authority"]["attempt_receipt"],
            fixture["create_only_authority"]["publication_receipt"],
            {
                "support": fixture["support"]["tube"],
                "canonical_candidate": fixture["canonical_candidate"]["video"],
                "delivery_candidate": fixture["delivery_candidate"]["video"],
            },
        )
        ACCEPTANCE.validate_producer_receipt(built)
        self.assertFalse(built["claim_limits"]["scientific_claim_authorized"])
        self.assertFalse(built["claim_limits"]["generation_execution_lineage_verified"])

    def test_attempt_and_publication_cross_bind_under_frozen_acceptance(self) -> None:
        fixture = ACCEPTANCE_FIXTURE.producer_receipt()
        pins = populated_pins()
        pins["source"] = copy.deepcopy(fixture["source"])
        pins["sam2_receipt"] = copy.deepcopy(fixture["mask_authority"]["receipt"])
        controller_row = fixture["create_only_authority"]["controller_program"]
        model_rows = fixture["generator"]["model_authorities"]
        preflight = {
            "authority_replay_digest": digest("authority-replay"),
            "model_authorities": copy.deepcopy(model_rows),
        }
        attempt = CONTROLLER._build_attempt(
            pins, preflight, controller_row["sha256"]
        )
        assets = {
            name: {
                "path": str(Path(pins["asset_final_root"]) / (name + ".bin")),
                "sha256": digest(name),
                "size": 100,
            }
            for name in ("support", "canonical_candidate", "delivery_candidate")
        }
        publication = CONTROLLER._build_publication(
            pins, attempt, controller_row["sha256"], assets
        )
        attempt_row = ACCEPTANCE_FIXTURE.file_row("attempt.json")
        publication_row = ACCEPTANCE_FIXTURE.file_row("publication.json")
        producer = CONTROLLER._build_producer(
            pins,
            result_from_acceptance_fixture(fixture),
            controller_row,
            attempt_row,
            publication_row,
            assets,
        )
        ACCEPTANCE.validate_producer_receipt(producer)
        replay = ACCEPTANCE.validate_create_only_receipts(
            attempt, publication, producer=producer
        )
        self.assertEqual(replay["attempt_token"], attempt["attempt_token"])
        self.assertEqual(replay["final_root"], pins["asset_final_root"])

    def test_controller_wraps_frozen_receipt_replay_failure_as_attempt_failure(self) -> None:
        attempt = {
            "attempt_token": digest("attempt-token"),
            "final_root": "/authority/final/assets",
            "staging_root": "/authority/stage/assets",
        }
        publication = {"publication_digest": digest("publication")}
        producer = {"receipt_digest": digest("producer")}
        acceptance = mock.Mock()
        failure = RuntimeError("synthetic frozen replay rejection")
        acceptance.validate_create_only_receipts.side_effect = failure

        with self.assertRaisesRegex(
            CONTROLLER.AttemptFailed,
            "frozen create-only receipt replay failed",
        ) as caught:
            CONTROLLER._replay_create_only_receipts(
                acceptance,
                attempt,
                publication,
                producer,
            )

        acceptance.validate_create_only_receipts.assert_called_once_with(
            attempt,
            publication,
            producer=producer,
        )
        self.assertIs(caught.exception.__cause__, failure)

    def test_generator_argv_is_one_explicit_run_with_all_authorities(self) -> None:
        pins = populated_pins()
        argv = CONTROLLER._generator_argv(pins)
        self.assertEqual(argv[:3], [pins["python_bin"]["path"], CONTROLLER.GENERATOR_PATH, "run"])
        for flag in (
            "--source-video", "--sam2-receipt", "--support-review-receipt",
            "--vace-source-manifest", "--vace-checkpoint-manifest",
            "--python-runtime-manifest", "--asset-staging-root",
            "--evidence-staging-root", "--asset-final-root", "--evidence-final-root",
        ):
            self.assertEqual(argv.count(flag), 1)

    def test_generator_success_uses_exactly_one_popen(self) -> None:
        pins = populated_pins()
        process = mock.Mock()
        process.pid = 4242
        process.communicate.return_value = (b'{"result":1}\n', b"")
        process.returncode = 0
        process.stdin = None
        process.stdout = mock.Mock(closed=True)
        process.stderr = mock.Mock(closed=True)
        process.poll.return_value = 0
        with mock.patch.object(
            CONTROLLER.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(
            CONTROLLER, "_process_group_absent", return_value=True
        ), mock.patch.object(
            CONTROLLER, "_signal_process_group"
        ) as signal_group:
            result, evidence, zero_token = CONTROLLER._run_generator_once(pins)
        self.assertEqual(result, {"result": 1})
        self.assertEqual(evidence["direct_generator_child_invocations"], 1)
        self.assertEqual(evidence["return_code"], 0)
        self.assertFalse(evidence["automatic_retry_performed"])
        self.assertEqual(evidence["exact_argv"], CONTROLLER._generator_argv(pins))
        self.assertEqual(evidence["exact_environment"], pins["child_environment"])
        self.assertEqual(evidence["saved_process_group_id"], 4242)
        self.assertTrue(evidence["normal_exit_passive_grace_performed"])
        self.assertFalse(evidence["normal_exit_signal_sent"])
        self.assertTrue(evidence["terminal_pipes_closed"])
        self.assertTrue(evidence["process_group_zero"])
        self.assertIs(zero_token, CONTROLLER._PROCESS_GROUP_ZERO_PROVEN)
        popen.assert_called_once()
        process.communicate.assert_called_once_with(
            timeout=CONTROLLER.GENERATOR_TIMEOUT_SECONDS
        )
        signal_group.assert_not_called()

    def test_keyboard_interrupt_while_saving_spawned_pgid_seals_group(self) -> None:
        pins = populated_pins()

        class InterruptOnFirstPidRead:
            def __init__(self) -> None:
                self.pid_reads = 0

            @property
            def pid(self) -> int:
                self.pid_reads += 1
                if self.pid_reads == 1:
                    raise KeyboardInterrupt("synthetic pre-communicate interrupt")
                return 4545

        process = InterruptOnFirstPidRead()
        with mock.patch.object(
            CONTROLLER.subprocess,
            "Popen",
            return_value=process,
        ) as popen, mock.patch.object(
            CONTROLLER,
            "_seal_process_group",
        ) as seal, self.assertRaisesRegex(
            KeyboardInterrupt,
            "pre-communicate interrupt",
        ):
            CONTROLLER._run_generator_once(pins)

        popen.assert_called_once()
        self.assertEqual(process.pid_reads, 2)
        seal.assert_called_once_with(process, 4545)

    def test_keyboard_interrupt_after_communicate_seals_group(self) -> None:
        pins = populated_pins()
        process = mock.Mock()
        process.pid = 4646
        process.communicate.return_value = (b'{"result":1}\n', b"")
        interrupt = KeyboardInterrupt("synthetic post-communicate interrupt")
        with mock.patch.object(
            CONTROLLER.subprocess,
            "Popen",
            return_value=process,
        ) as popen, mock.patch.object(
            CONTROLLER,
            "_process_group_absent",
            side_effect=interrupt,
        ) as absent, mock.patch.object(
            CONTROLLER,
            "_seal_process_group",
        ) as seal, self.assertRaisesRegex(
            KeyboardInterrupt,
            "post-communicate interrupt",
        ):
            CONTROLLER._run_generator_once(pins)

        popen.assert_called_once()
        process.communicate.assert_called_once_with(
            timeout=CONTROLLER.GENERATOR_TIMEOUT_SECONDS
        )
        absent.assert_called_once_with(
            4646,
            CONTROLLER.PROCESS_TERM_GRACE_SECONDS,
        )
        seal.assert_called_once_with(process, 4646)

    def test_generator_failure_is_not_retried(self) -> None:
        pins = populated_pins()
        process = mock.Mock()
        process.pid = 4343
        process.communicate.return_value = (b"", b"failed")
        process.returncode = 2
        process.stdin = None
        process.stdout = mock.Mock(closed=True)
        process.stderr = mock.Mock(closed=True)
        process.poll.return_value = 2
        with mock.patch.object(
            CONTROLLER.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(
            CONTROLLER, "_process_group_absent", return_value=True
        ):
            with self.assertRaisesRegex(CONTROLLER.AttemptFailed, "single generator attempt failed"):
                CONTROLLER._run_generator_once(pins)
        popen.assert_called_once()
        process.communicate.assert_called_once_with(
            timeout=CONTROLLER.GENERATOR_TIMEOUT_SECONDS
        )

    def test_timeout_seals_saved_process_group_and_never_retries(self) -> None:
        pins = populated_pins()
        process = mock.Mock()
        process.pid = 5151
        process.returncode = None
        timeout = subprocess.TimeoutExpired(["generator"], 1)
        process.communicate.side_effect = timeout
        with mock.patch.object(
            CONTROLLER.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(
            CONTROLLER, "_seal_process_group"
        ) as seal, self.assertRaisesRegex(
            CONTROLLER.AttemptFailed, "attempt timed out"
        ) as caught:
            CONTROLLER._run_generator_once(pins)
        popen.assert_called_once()
        seal.assert_called_once_with(process, 5151)
        self.assertIs(caught.exception.__cause__, timeout)

    def test_persistent_terminal_group_is_sealed_then_fails_closed(self) -> None:
        pins = populated_pins()
        process = mock.Mock()
        process.pid = 5252
        process.returncode = 0
        process.communicate.return_value = (b'{"result":1}\n', b"")
        with mock.patch.object(
            CONTROLLER.subprocess, "Popen", return_value=process
        ), mock.patch.object(
            CONTROLLER, "_process_group_absent", return_value=False
        ), mock.patch.object(
            CONTROLLER, "_seal_process_group"
        ) as seal, self.assertRaisesRegex(
            CONTROLLER.AttemptFailed, "required cleanup after passive grace"
        ):
            CONTROLLER._run_generator_once(pins)
        seal.assert_called_once_with(process, 5252)

    def test_communicate_error_with_unproven_cleanup_quarantines(self) -> None:
        pins = populated_pins()
        process = mock.Mock()
        process.pid = 5353
        primary = RuntimeError("hostile communicate error")
        cleanup = OSError("hostile cleanup error")
        process.communicate.side_effect = primary
        with mock.patch.object(
            CONTROLLER.subprocess, "Popen", return_value=process
        ), mock.patch.object(
            CONTROLLER, "_seal_process_group", side_effect=cleanup
        ) as seal, self.assertRaisesRegex(
            CONTROLLER.ProcessGroupZeroUnproven,
            "process/pipe zero gate is unproven",
        ) as caught:
            CONTROLLER._run_generator_once(pins)
        seal.assert_called_once_with(process, 5353)
        self.assertIs(caught.exception.__cause__, primary)

    def test_early_leader_and_term_ignoring_descendant_reach_esrch(self) -> None:
        source = (
            "import os,signal,sys,time\n"
            "child=os.fork()\n"
            "if child == 0:\n"
            " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            " sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            " while True: time.sleep(1)\n"
            "time.sleep(0.05)\n"
            "os._exit(0)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", source],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.35)
            process.poll()
            with mock.patch.object(
                CONTROLLER, "PROCESS_TERM_GRACE_SECONDS", 0.10
            ), mock.patch.object(
                CONTROLLER, "PROCESS_KILL_GRACE_SECONDS", 1.50
            ):
                CONTROLLER._seal_process_group(process, process.pid)
            self.assertTrue(CONTROLLER._process_group_absent(process.pid, 0.5))
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def test_pipe_close_error_does_not_bypass_process_group_zero_gate(self) -> None:
        source = (
            "import signal,sys,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", source],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
        original_stdout = process.stdout

        class BrokenClose:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.closed = False

            def close(self):
                if not self.closed:
                    self.closed = True
                    self.wrapped.close()
                raise OSError("synthetic pipe close failure")

        try:
            self.assertIsNotNone(original_stdout)
            self.assertEqual(original_stdout.readline(), b"READY\n")
            process.stdout = BrokenClose(original_stdout)
            with mock.patch.object(
                CONTROLLER, "PROCESS_TERM_GRACE_SECONDS", 0.10
            ), mock.patch.object(
                CONTROLLER, "PROCESS_KILL_GRACE_SECONDS", 1.50
            ), self.assertRaisesRegex(
                CONTROLLER.AttemptFailed, "terminal pipe seal"
            ):
                CONTROLLER._seal_process_group(process, process.pid)
            self.assertTrue(CONTROLLER._process_group_absent(process.pid, 0.5))
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def test_strict_json_rejects_duplicate_keys_and_pretty_bytes(self) -> None:
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "duplicate JSON key"):
            CONTROLLER._parse_canonical_object(b'{"x":1,"x":2}\n', "test")
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "not canonical"):
            CONTROLLER._parse_canonical_object(b'{\n  "x": 1\n}\n', "test")


class PublicationAndCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temp.name).resolve()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _quarantine_execution_fixture(self):
        pins = populated_pins()
        stage = self.root / ".stage"
        final = self.root / "final"
        pins.update(
            {
                "bundle_staging_root": str(stage),
                "bundle_final_root": str(final),
                "asset_staging_root": str(stage / "assets"),
                "evidence_staging_root": str(stage / "evidence"),
                "asset_final_root": str(final / "assets"),
                "evidence_final_root": str(final / "evidence"),
                "attempt_receipt": str(self.root / "attempt.json"),
                "publication_receipt": str(
                    final / "evidence" / "create_only_publication.json"
                ),
                "producer_receipt": str(
                    final / "evidence" / "producer_receipt.json"
                ),
            }
        )
        preflight = {
            "authority_replay_digest": digest("authority-replay"),
            "model_authorities": [],
        }
        controller_row = {
            "path": "/authority/controller.py",
            "sha256": digest("controller"),
            "size": 100,
        }
        return pins, stage, final, preflight, controller_row

    def assert_stage_is_quarantined(self, stage: Path, final: Path) -> None:
        self.assertTrue(stage.is_dir())
        self.assertTrue((stage / "assets").is_dir())
        self.assertTrue((stage / "evidence").is_dir())
        self.assertFalse(final.exists())

    def test_same_parent_bundle_publication_is_one_noreplace_operation(self) -> None:
        stage = self.root / ".stage"
        final = self.root / "final"
        (stage / "assets").mkdir(parents=True)
        (stage / "evidence").mkdir()
        (stage / "assets" / "canonical.mkv").write_bytes(b"canonical")
        (stage / "evidence" / "receipt.json").write_bytes(b"receipt")
        row = stage.lstat()
        identity = (row.st_dev, row.st_ino)

        def rename_once(source: Path, destination: Path) -> None:
            os.rename(source, destination)

        with mock.patch.object(
            CONTROLLER, "_rename_noreplace", side_effect=rename_once
        ) as rename, mock.patch.object(
            CONTROLLER, "_fsync_tree"
        ) as fsync_tree, mock.patch.object(
            CONTROLLER, "_fsync_directory"
        ) as fsync_dir:
            CONTROLLER._publish_bundle(stage, final, identity)
        rename.assert_called_once_with(stage, final)
        fsync_tree.assert_called_once_with(stage)
        fsync_dir.assert_called_once_with(self.root)
        self.assertFalse(stage.exists())
        self.assertEqual((final / "assets" / "canonical.mkv").read_bytes(), b"canonical")

    def test_publication_rejects_cross_parent_before_libc(self) -> None:
        with mock.patch.object(CONTROLLER.ctypes, "CDLL") as library:
            with self.assertRaisesRegex(CONTROLLER.ControllerHold, "not same-parent"):
                CONTROLLER._rename_noreplace(
                    Path("/authority/a/stage"), Path("/authority/b/final")
                )
        library.assert_not_called()

    def test_rename_success_parent_fsync_failure_has_no_success_receipt(self) -> None:
        stage = self.root / ".stage"
        final = self.root / "final"
        (stage / "assets").mkdir(parents=True)
        (stage / "evidence").mkdir()
        (stage / "assets" / "canonical.mkv").write_bytes(b"canonical")
        identity_row = stage.lstat()
        identity = (identity_row.st_dev, identity_row.st_ino)

        def rename_once(source: Path, destination: Path) -> None:
            os.rename(source, destination)

        with mock.patch.object(
            CONTROLLER, "_rename_noreplace", side_effect=rename_once
        ), mock.patch.object(
            CONTROLLER, "_fsync_tree"
        ), mock.patch.object(
            CONTROLLER,
            "_fsync_directory",
            side_effect=OSError("synthetic parent fsync failure"),
        ), self.assertRaisesRegex(OSError, "parent fsync failure"):
            CONTROLLER._publish_bundle(stage, final, identity)

        self.assertFalse(stage.exists())
        self.assertTrue(final.is_dir())
        self.assertFalse(
            (final / "evidence" / "create_only_publication.json").exists()
        )

    def test_cleanup_removes_only_captured_private_inode(self) -> None:
        stage = self.root / ".stage"
        (stage / "nested").mkdir(parents=True)
        (stage / "nested" / "partial.bin").write_bytes(b"partial")
        row = stage.lstat()
        with mock.patch.object(CONTROLLER, "_fsync_directory"):
            CONTROLLER._cleanup_private_stage(stage, (row.st_dev, row.st_ino))
        self.assertFalse(stage.exists())

    def test_cleanup_refuses_replaced_stage_inode(self) -> None:
        stage = self.root / ".stage"
        stage.mkdir()
        row = stage.lstat()
        stage.rmdir()
        stage.mkdir()
        with self.assertRaisesRegex(CONTROLLER.AttemptFailed, "changed staging inode"):
            CONTROLLER._cleanup_private_stage(stage, (row.st_dev, row.st_ino))
        self.assertTrue(stage.exists())

    def test_create_only_writer_never_overwrites(self) -> None:
        path = self.root / "receipt.json"
        path.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            CONTROLLER._write_create_only(path, b"replacement", 0o400)
        self.assertEqual(path.read_bytes(), b"existing")

    def test_regular_tree_rejects_symlink(self) -> None:
        tree = self.root / "tree"
        tree.mkdir()
        target = self.root / "target"
        target.write_bytes(b"target")
        (tree / "link").symlink_to(target)
        with self.assertRaisesRegex(CONTROLLER.ControllerHold, "topology|symlink"):
            CONTROLLER._scan_regular_tree(tree)

    def _scandir_error_patch(self, blocked: Path):
        real_scandir = os.scandir

        def injected(path):
            if Path(path) == blocked:
                raise PermissionError(
                    errno.EACCES,
                    "synthetic unreadable subtree",
                    str(blocked),
                )
            return real_scandir(path)

        return mock.patch.object(CONTROLLER.os, "scandir", side_effect=injected)

    def test_regular_tree_scandir_error_fails_closed(self) -> None:
        tree = self.root / "tree"
        blocked = tree / "blocked"
        blocked.mkdir(parents=True)
        (blocked / "hidden.bin").write_bytes(b"hidden")

        with self._scandir_error_patch(blocked), self.assertRaisesRegex(
            CONTROLLER.AttemptFailed,
            "filesystem tree walk failed",
        ) as caught:
            CONTROLLER._scan_regular_tree(tree)

        self.assertIsInstance(caught.exception.__cause__, PermissionError)

    def test_fsync_tree_scandir_error_fails_closed(self) -> None:
        tree = self.root / "tree"
        blocked = tree / "blocked"
        blocked.mkdir(parents=True)
        (blocked / "hidden.bin").write_bytes(b"hidden")

        with self._scandir_error_patch(blocked), mock.patch.object(
            CONTROLLER,
            "_fsync_directory",
        ) as fsync_directory, self.assertRaisesRegex(
            CONTROLLER.AttemptFailed,
            "filesystem tree walk failed",
        ):
            CONTROLLER._fsync_tree(tree)

        fsync_directory.assert_not_called()

    def test_cleanup_scandir_error_retains_unreadable_subtree(self) -> None:
        stage = self.root / ".stage"
        blocked = stage / "blocked"
        blocked.mkdir(parents=True)
        hidden = blocked / "partial.bin"
        hidden.write_bytes(b"partial")
        row = stage.lstat()

        with self._scandir_error_patch(blocked), mock.patch.object(
            CONTROLLER,
            "_fsync_directory",
        ) as fsync_directory, self.assertRaisesRegex(
            CONTROLLER.AttemptFailed,
            "filesystem tree walk failed",
        ):
            CONTROLLER._cleanup_private_stage(
                stage,
                (row.st_dev, row.st_ino),
            )

        fsync_directory.assert_not_called()
        self.assertTrue(stage.is_dir())
        self.assertEqual(hidden.read_bytes(), b"partial")

    def test_unproven_process_group_quarantines_stage_without_cleanup(self) -> None:
        pins = populated_pins()
        stage = self.root / ".stage"
        final = self.root / "final"
        pins.update(
            {
                "bundle_staging_root": str(stage),
                "bundle_final_root": str(final),
                "asset_staging_root": str(stage / "assets"),
                "evidence_staging_root": str(stage / "evidence"),
                "asset_final_root": str(final / "assets"),
                "evidence_final_root": str(final / "evidence"),
                "attempt_receipt": str(self.root / "attempt.json"),
                "publication_receipt": str(
                    final / "evidence" / "create_only_publication.json"
                ),
                "producer_receipt": str(
                    final / "evidence" / "producer_receipt.json"
                ),
            }
        )
        preflight = {
            "authority_replay_digest": digest("authority-replay"),
            "model_authorities": [],
        }
        controller_row = {
            "path": "/authority/controller.py",
            "sha256": digest("controller"),
            "size": 100,
        }
        failure = CONTROLLER.ProcessGroupZeroUnproven("synthetic live group")
        with mock.patch.object(
            CONTROLLER,
            "_preflight",
            return_value=(None, None, preflight, {}, controller_row),
        ), mock.patch.object(
            CONTROLLER, "_run_generator_once", side_effect=failure
        ), mock.patch.object(
            CONTROLLER, "_cleanup_private_stage"
        ) as cleanup, self.assertRaises(
            CONTROLLER.ProcessGroupZeroUnproven
        ):
            CONTROLLER.execute_create_once(pins)
        cleanup.assert_not_called()
        self.assertTrue(stage.is_dir())
        self.assertTrue((stage / "assets").is_dir())
        self.assertTrue((stage / "evidence").is_dir())
        self.assertFalse(final.exists())

    def test_sys_exc_info_interrupt_quarantines_full_stage(self) -> None:
        pins, stage, final, preflight, controller_row = (
            self._quarantine_execution_fixture()
        )
        process = mock.Mock()
        process.pid = 6161
        process.communicate.side_effect = RuntimeError(
            "synthetic lifecycle primary error"
        )
        real_exc_info = sys.exc_info
        exc_info_calls = 0

        def interrupt_first_exc_info():
            nonlocal exc_info_calls
            exc_info_calls += 1
            if exc_info_calls == 1:
                raise KeyboardInterrupt("synthetic finally interrupt")
            return real_exc_info()

        with mock.patch.object(
            CONTROLLER,
            "_preflight",
            return_value=(None, None, preflight, {}, controller_row),
        ), mock.patch.object(
            CONTROLLER,
            "_fsync_directory",
        ), mock.patch.object(
            CONTROLLER.subprocess,
            "Popen",
            return_value=process,
        ), mock.patch.object(
            CONTROLLER.sys,
            "exc_info",
            side_effect=interrupt_first_exc_info,
        ), mock.patch.object(
            CONTROLLER,
            "_seal_process_group",
        ) as seal, mock.patch.object(
            CONTROLLER,
            "_cleanup_private_stage",
        ) as cleanup, self.assertRaisesRegex(
            KeyboardInterrupt,
            "synthetic finally interrupt",
        ):
            CONTROLLER.execute_create_once(pins)

        self.assertGreaterEqual(exc_info_calls, 1)
        seal.assert_not_called()
        cleanup.assert_not_called()
        self.assert_stage_is_quarantined(stage, final)

    def test_popen_call_store_equivalent_interrupt_quarantines_full_stage(self) -> None:
        pins, stage, final, preflight, controller_row = (
            self._quarantine_execution_fixture()
        )
        spawned = []

        def spawn_then_interrupt(*args, **kwargs):
            # Models an async exception after the OS-side spawn succeeded but
            # before Python stored the returned Popen wrapper in its local.
            spawned.append((args, kwargs))
            raise KeyboardInterrupt("synthetic Popen CALL-to-STORE interrupt")

        with mock.patch.object(
            CONTROLLER,
            "_preflight",
            return_value=(None, None, preflight, {}, controller_row),
        ), mock.patch.object(
            CONTROLLER,
            "_fsync_directory",
        ), mock.patch.object(
            CONTROLLER.subprocess,
            "Popen",
            side_effect=spawn_then_interrupt,
        ), mock.patch.object(
            CONTROLLER,
            "_cleanup_private_stage",
        ) as cleanup, self.assertRaisesRegex(
            KeyboardInterrupt,
            "Popen CALL-to-STORE interrupt",
        ):
            CONTROLLER.execute_create_once(pins)

        self.assertEqual(len(spawned), 1)
        cleanup.assert_not_called()
        self.assert_stage_is_quarantined(stage, final)

    def test_unknown_zero_token_quarantines_full_stage(self) -> None:
        pins, stage, final, preflight, controller_row = (
            self._quarantine_execution_fixture()
        )
        with mock.patch.object(
            CONTROLLER,
            "_preflight",
            return_value=(None, None, preflight, {}, controller_row),
        ), mock.patch.object(
            CONTROLLER,
            "_fsync_directory",
        ), mock.patch.object(
            CONTROLLER,
            "_run_generator_once",
            return_value=({}, {}, object()),
        ) as child, mock.patch.object(
            CONTROLLER,
            "_cleanup_private_stage",
        ) as cleanup, self.assertRaisesRegex(
            CONTROLLER.ProcessGroupZeroUnproven,
            "no affirmative process-group-zero token",
        ):
            CONTROLLER.execute_create_once(pins)

        child.assert_called_once_with(pins)
        cleanup.assert_not_called()
        self.assert_stage_is_quarantined(stage, final)

    def test_affirmative_zero_token_allows_post_child_stage_cleanup(self) -> None:
        pins, stage, final, preflight, controller_row = (
            self._quarantine_execution_fixture()
        )
        with mock.patch.object(
            CONTROLLER,
            "_preflight",
            return_value=(None, None, preflight, {}, controller_row),
        ), mock.patch.object(
            CONTROLLER,
            "_fsync_directory",
        ), mock.patch.object(
            CONTROLLER,
            "_run_generator_once",
            return_value=(
                {},
                {},
                CONTROLLER._PROCESS_GROUP_ZERO_PROVEN,
            ),
        ) as child, self.assertRaisesRegex(
            CONTROLLER.ControllerHold,
            "generator result keys differ",
        ):
            CONTROLLER.execute_create_once(pins)

        child.assert_called_once_with(pins)
        self.assertFalse(stage.exists())
        self.assertFalse(final.exists())

    def test_full_body_orders_attempt_child_receipts_then_one_bundle_publish(self) -> None:
        pins = populated_pins()
        stage = self.root / ".stage"
        final = self.root / "final"
        pins.update(
            {
                "bundle_staging_root": str(stage),
                "bundle_final_root": str(final),
                "asset_staging_root": str(stage / "assets"),
                "evidence_staging_root": str(stage / "evidence"),
                "asset_final_root": str(final / "assets"),
                "evidence_final_root": str(final / "evidence"),
                "attempt_receipt": str(self.root / "attempt.json"),
                "publication_receipt": str(
                    final / "evidence" / "create_only_publication.json"
                ),
                "producer_receipt": str(final / "evidence" / "producer_receipt.json"),
            }
        )
        fixture = ACCEPTANCE_FIXTURE.producer_receipt()
        pins["source"] = copy.deepcopy(fixture["source"])
        pins["sam2_receipt"] = copy.deepcopy(fixture["mask_authority"]["receipt"])
        controller_row = copy.deepcopy(
            fixture["create_only_authority"]["controller_program"]
        )
        preflight = {
            "authority_replay_digest": digest("authority-replay"),
            "model_authorities": copy.deepcopy(fixture["generator"]["model_authorities"]),
            "support_review_receipt": copy.deepcopy(
                fixture["support"]["review_receipt"]
            ),
            "support_frame_masks": copy.deepcopy(fixture["support"]["frame_masks"]),
        }
        events = []

        def row(path: Path) -> dict:
            payload = path.read_bytes()
            return {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }

        def generator_once(_pins: dict) -> dict:
            events.append("child")
            asset_root = Path(_pins["asset_staging_root"])
            support = asset_root / "support_ffv1.mkv"
            canonical = asset_root / "bone_removed_v2_canonical_ffv1.mkv"
            delivery = asset_root / "bone_removed_v2_delivery_h264.mp4"
            support.write_bytes(b"support")
            canonical.write_bytes(b"canonical")
            delivery.write_bytes(b"delivery")
            result = result_from_acceptance_fixture(fixture)
            result.update(
                {
                    "support_tube_stage": row(support),
                    "canonical_stage": row(canonical),
                    "delivery_stage": row(delivery),
                    "execution_evidence": {
                        "status": "UNPUBLISHED_PENDING_CONTROLLER_ATTESTATION",
                        "generation_execution_lineage_verified": False,
                    },
                }
            )
            child_evidence = {
                "exact_argv": CONTROLLER._generator_argv(_pins),
                "exact_environment": dict(_pins["child_environment"]),
                "return_code": 0,
                "stdout_sha256": digest("stdout"),
                "stdout_size": 100,
                "stderr_sha256": digest("stderr"),
                "stderr_size": 0,
                "stdout_is_canonical_generator_result": True,
                "direct_generator_child_invocations": 1,
                "start_new_session": True,
                "saved_process_group_id": 4242,
                "normal_exit_passive_grace_performed": True,
                "normal_exit_signal_sent": False,
                "terminal_pipes_closed": True,
                "process_group_zero": True,
                "automatic_retry_performed": False,
            }
            return (
                result,
                child_evidence,
                CONTROLLER._PROCESS_GROUP_ZERO_PROVEN,
            )

        real_write = CONTROLLER._write_create_only

        def tracked_write(path: Path, payload: bytes, mode: int = 0o400) -> dict:
            events.append("write:" + Path(path).name)
            return real_write(path, payload, mode)

        def publish_once(source: Path, destination: Path, _identity: tuple) -> None:
            events.append("publish")
            os.rename(source, destination)

        def tracked_fsync(path: Path) -> None:
            events.append("fsync:" + str(Path(path)))

        def tracked_verify(value: dict, label: str, *, nlink1: bool = False) -> bytes:
            del value, nlink1
            events.append("verify:" + label)
            return b"verified"

        real_receipt_replay = ACCEPTANCE.validate_create_only_receipts

        def tracked_receipt_replay(
            attempt: dict,
            publication: dict,
            *,
            producer: dict,
        ) -> dict:
            events.append("receipt-replay")
            return dict(
                real_receipt_replay(
                    attempt,
                    publication,
                    producer=producer,
                )
            )

        with mock.patch.object(
            CONTROLLER,
            "_preflight",
            return_value=(None, ACCEPTANCE, preflight, {}, controller_row),
        ), mock.patch.object(
            CONTROLLER, "_run_generator_once", side_effect=generator_once
        ) as child, mock.patch.object(
            CONTROLLER, "_write_create_only", side_effect=tracked_write
        ), mock.patch.object(
            CONTROLLER, "_publish_bundle", side_effect=publish_once
        ) as publish, mock.patch.object(
            CONTROLLER, "_fsync_directory", side_effect=tracked_fsync
        ), mock.patch.object(
            CONTROLLER, "_verify_file_pin", side_effect=tracked_verify
        ), mock.patch.object(
            CONTROLLER,
            "_read_stable_file",
            return_value=(b"controller", controller_row),
        ), mock.patch.object(
            CONTROLLER,
            "_verify_manifest_pin",
        ), mock.patch.object(
            ACCEPTANCE,
            "validate_create_only_receipts",
            side_effect=tracked_receipt_replay,
        ) as receipt_replay:
            result = CONTROLLER.execute_create_once(pins)

        self.assertEqual(result["status"], "COMPLETE_CANDIDATE_PENDING_INDEPENDENT_ACCEPTANCE")
        self.assertFalse(result["semantic_acceptance_performed"])
        self.assertFalse(result["scientific_claim_authorized"])
        self.assertFalse(result["automatic_retry_performed"])
        child.assert_called_once_with(pins)
        publish.assert_called_once()
        self.assertLess(events.index("write:attempt.json"), events.index("child"))
        self.assertLess(events.index("write:producer_receipt.json"), events.index("publish"))
        self.assertLess(events.index("publish"), events.index("write:create_only_publication.json"))
        publication_write = events.index("write:create_only_publication.json")
        publication_fsync = events.index(
            "fsync:" + str(Path(pins["publication_receipt"]).parent),
            publication_write,
        )
        publication_verify = events.index(
            "verify:published publication receipt",
            publication_fsync,
        )
        replay_event = events.index("receipt-replay")
        self.assertLess(publication_write, publication_fsync)
        self.assertLess(publication_fsync, publication_verify)
        self.assertLess(publication_verify, replay_event)
        receipt_replay.assert_called_once()
        self.assertFalse(stage.exists())
        self.assertTrue((final / "evidence" / "producer_receipt.json").is_file())
        controller_evidence = final / "evidence" / "controller_execution_evidence.json"
        self.assertTrue(controller_evidence.is_file())
        evidence_value = json.loads(controller_evidence.read_text(encoding="ascii"))
        self.assertEqual(evidence_value["child_process"]["direct_generator_child_invocations"], 1)
        self.assertFalse(evidence_value["generation_execution_lineage_verified"])


if __name__ == "__main__":
    unittest.main()
