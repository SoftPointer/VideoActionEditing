#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = REPO_ROOT / "methods" / "bernini_action_editing"
SUBJECT_PATH = METHOD_ROOT / "recover_fresh_world8_level_a_compare_only_0817_v1.py"
AUDIT_ROOT = METHOD_ROOT / "audits"
AUTHORITY_PATH = (
    AUDIT_ROOT / "fresh_world8_level_a_compare_only_recovery_v1_AUTHORITY.json"
)
SACCT_PATH = AUDIT_ROOT / "fresh_world8_level_a_compare_only_recovery_v1_SACCT_ROWS.json"
RECOVERY_PINS = (
    AUDIT_ROOT / "fresh_world8_level_a_compare_only_recovery_v1_DEPLOYMENT_PINS.json"
)
OLD_RELEASE_MANIFEST = (
    AUDIT_ROOT / "fresh_world8_level_a_r2_p2_launchbound_v2_RELEASE_MANIFEST.json"
)
OLD_LAUNCH_CORE = (
    AUDIT_ROOT / "fresh_world8_level_a_r2_p2_launchbound_v2_LAUNCH_AUTHORITY_CORE.json"
)
OLD_DEPLOYMENT_PINS = (
    AUDIT_ROOT / "fresh_world8_level_a_r2_p2_launchbound_v2_DEPLOYMENT_PINS.json"
)
EVIDENCE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "level_a_launchbound_v2_node279_20260817"
    / "remote_evidence"
)


def load_subject(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recovery = load_subject(
    SUBJECT_PATH, "recover_fresh_world8_level_a_compare_only_0817_v1_test"
)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_inputs(**overrides):
    values = {
        "old_release_manifest": OLD_RELEASE_MANIFEST.resolve(),
        "old_launch_authority_core": OLD_LAUNCH_CORE.resolve(),
        "old_deployment_pins": OLD_DEPLOYMENT_PINS.resolve(),
        "evidence_root": EVIDENCE_ROOT.resolve(),
        "sacct_rows_json": SACCT_PATH.resolve(),
    }
    values.update(overrides)
    return values


def load_attempts():
    authority, _ = recovery.load_frozen_authority()
    _, core = recovery.validate_old_trust_roots(
        OLD_RELEASE_MANIFEST.resolve(),
        OLD_LAUNCH_CORE.resolve(),
        OLD_DEPLOYMENT_PINS.resolve(),
        authority=authority,
    )
    _, loaded = recovery.validate_evidence_files(
        EVIDENCE_ROOT.resolve(), authority=authority
    )
    attempts = recovery.load_and_validate_attempts(
        loaded, authority=authority, core=core
    )
    return authority, attempts


class FrozenAuthorityTests(unittest.TestCase):
    def test_authority_and_sacct_are_canonical_and_compiled_not_caller_supplied(self) -> None:
        authority_payload = AUTHORITY_PATH.read_bytes()
        authority = json.loads(authority_payload.decode("utf-8"))
        self.assertEqual(
            authority_payload, recovery.canonical_json_bytes(authority) + b"\n"
        )
        self.assertEqual(sha(authority_payload), recovery.AUTHORITY_SHA256)
        loaded, loaded_sha = recovery.load_frozen_authority()
        self.assertEqual(loaded, authority)
        self.assertEqual(loaded_sha, recovery.AUTHORITY_SHA256)
        sacct_payload = SACCT_PATH.read_bytes()
        sacct = json.loads(sacct_payload.decode("utf-8"))
        self.assertEqual(sacct_payload, recovery.canonical_json_bytes(sacct) + b"\n")
        self.assertEqual(sha(sacct_payload), authority["sacct"]["rows_file_sha256"])
        parser_actions = {action.dest for action in recovery.build_parser()._actions}
        self.assertNotIn("authority", parser_actions)

    def test_authority_freezes_exact_old_roots_steps_evidence_and_false_claims(self) -> None:
        authority, _ = recovery.load_frozen_authority()
        trust = authority["old_trust_roots"]
        self.assertEqual(
            trust["release_manifest_sha256"], sha(OLD_RELEASE_MANIFEST.read_bytes())
        )
        self.assertEqual(
            trust["launch_authority_core_sha256"], sha(OLD_LAUNCH_CORE.read_bytes())
        )
        self.assertEqual(
            trust["deployment_pins_sha256"], sha(OLD_DEPLOYMENT_PINS.read_bytes())
        )
        self.assertEqual(authority["sacct"]["step_ids"], ["140846.367", "140846.368"])
        self.assertEqual(len(authority["evidence_files"]), 17)
        self.assertTrue(authority["old_campaign"]["failed_fail_closed"])
        self.assertFalse(authority["old_campaign"]["root_success_present"])
        claims = authority["output_claims"]
        self.assertTrue(claims["recovered_parity_only"])
        self.assertTrue(claims["old_campaign_failed"])
        for key in (
            "gpu_relaunched",
            "promotion_authorized",
            "formal_training_started",
            "counts_as_d0",
            "scientific_claim_authorized",
            "full_bernini_renderer_forward_executed",
            "offline_product_inference_completed",
            "full40_denoise_executed",
            "mp4_emitted",
        ):
            self.assertFalse(claims[key], key)

    def test_subject_has_no_gpu_slurm_network_or_old_driver_execution_surface(self) -> None:
        source = SUBJECT_PATH.read_text(encoding="utf-8")
        executable = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for forbidden in (
            "import torch",
            "subprocess.run",
            "os.system",
            "srun ",
            "scancel",
            "scontrol",
            "ssh ",
            "action_edit_fresh_world8_level_a_driver_0817_v1 import",
        ):
            self.assertNotIn(forbidden, executable)

    def test_outer_recovery_pins_freeze_subject_test_authority_and_spec(self) -> None:
        pins = json.loads(RECOVERY_PINS.read_text(encoding="utf-8"))
        self.assertEqual(
            pins["schema_version"],
            "bernini-action-edit-fresh-world8-level-a-compare-only-recovery-deployment-pins-v1",
        )
        self.assertEqual(pins["status"], "LOCAL_FROZEN_NOT_DEPLOYED")
        members = {row["path"]: row for row in pins["files"]}
        for path in (SUBJECT_PATH, Path(__file__).resolve(), AUTHORITY_PATH, SACCT_PATH):
            row = members[path.name]
            self.assertEqual(row["sha256"], sha(path.read_bytes()))
            self.assertEqual(row["size"], len(path.read_bytes()))
            self.assertEqual(row["mode"], 0o444)
        self.assertEqual(
            pins["old_inputs"]["deployment_pins_sha256"],
            sha(OLD_DEPLOYMENT_PINS.read_bytes()),
        )
        self.assertTrue(pins["claims"]["compare_only"])
        self.assertFalse(pins["claims"]["gpu_relaunch_authorized"])
        self.assertFalse(pins["claims"]["promotion_authorized"])


class ExactRecoveryTests(unittest.TestCase):
    def test_exact_local_mirror_validates_and_builds_bounded_receipt(self) -> None:
        receipt = recovery.validate_inputs(**exact_inputs())
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_digest")
        self.assertEqual(claimed, recovery.object_sha256(unsigned))
        self.assertEqual(receipt["status"], "RECOVERED_PARITY_ONLY")
        self.assertEqual(receipt["recovery_reason"], "cross_clock_domain")
        self.assertEqual(receipt["old_campaign"]["status"], "FAILED_FAIL_CLOSED")
        self.assertFalse(receipt["old_campaign"]["parity_receipt_published"])
        self.assertTrue(receipt["recovered_parity"]["exact_parity"])
        self.assertEqual(receipt["recovered_parity"]["world8_launches"], 2)
        self.assertEqual(
            receipt["recovered_parity"]["distinct_fresh_process_sessions"], 16
        )
        self.assertEqual(
            receipt["recovered_parity"]["world8_consensus_sha256"],
            "3ed461d99927d2e7b059b78b1d3c6ef039b3d8d3b66d5392b39960eca99dc75e",
        )
        self.assertFalse(receipt["gpu_relaunched"])
        self.assertFalse(receipt["promotion_authorized"])
        self.assertFalse(receipt["formal_training_started"])
        self.assertFalse(receipt["counts_as_d0"])
        self.assertAlmostEqual(
            receipt["clock_domain_evidence"]["attempts"]["A"][
                "absolute_start_claim_skew_seconds"
            ],
            44.807141304,
            places=6,
        )
        self.assertAlmostEqual(
            receipt["clock_domain_evidence"]["attempts"]["B"][
                "absolute_start_claim_skew_seconds"
            ],
            45.339496851,
            places=6,
        )

    def test_recover_publishes_only_one_create_only_receipt_in_fresh_root(self) -> None:
        source_hashes_before = {
            path: sha(path.read_bytes())
            for path in (
                OLD_RELEASE_MANIFEST,
                OLD_LAUNCH_CORE,
                OLD_DEPLOYMENT_PINS,
                AUTHORITY_PATH,
                SACCT_PATH,
                EVIDENCE_ROOT / "run_A" / "bundle.consumer_receipt",
                EVIDENCE_ROOT / "run_B" / "bundle.consumer_receipt",
            )
        }
        receipt = recovery.validate_inputs(**exact_inputs())
        with tempfile.TemporaryDirectory() as temporary:
            root = (Path(temporary) / "new-recovery-root").resolve()
            root.mkdir(mode=0o700)
            destination = recovery.publish_create_only(root, receipt)
            self.assertEqual([item.name for item in root.iterdir()], [recovery.OUTPUT_FILENAME])
            self.assertEqual(destination.read_bytes(), recovery.canonical_json_bytes(receipt))
            info = destination.lstat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(info.st_nlink, 1)
            with self.assertRaises(recovery.RecoveryError):
                recovery.publish_create_only(root, receipt)
        source_hashes_after = {path: sha(path.read_bytes()) for path in source_hashes_before}
        self.assertEqual(source_hashes_before, source_hashes_after)

    def test_wrong_old_release_core_or_sacct_file_hash_is_rejected(self) -> None:
        for key, source in (
            ("old_release_manifest", OLD_RELEASE_MANIFEST),
            ("old_launch_authority_core", OLD_LAUNCH_CORE),
            ("old_deployment_pins", OLD_DEPLOYMENT_PINS),
            ("sacct_rows_json", SACCT_PATH),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                target = (Path(temporary) / source.name).resolve()
                target.write_bytes(source.read_bytes() + b" ")
                os.chmod(target, 0o444)
                with self.assertRaises(recovery.RecoveryError):
                    recovery.validate_inputs(**exact_inputs(**{key: target}))

    def test_wrong_evidence_hash_and_extra_member_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = (Path(temporary) / "evidence").resolve()
            shutil.copytree(EVIDENCE_ROOT, copied, copy_function=shutil.copy2)
            target = copied / "A" / "SUCCESS"
            os.chmod(target, 0o600)
            target.write_bytes(target.read_bytes() + b" ")
            os.chmod(target, 0o444)
            with self.assertRaises(recovery.RecoveryError):
                recovery.validate_inputs(**exact_inputs(evidence_root=copied))
        with tempfile.TemporaryDirectory() as temporary:
            copied = (Path(temporary) / "evidence").resolve()
            shutil.copytree(EVIDENCE_ROOT, copied, copy_function=shutil.copy2)
            extra = copied / "shadow.json"
            extra.write_text("{}", encoding="utf-8")
            with self.assertRaises(recovery.RecoveryError):
                recovery.validate_inputs(**exact_inputs(evidence_root=copied))


class HostilePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority, self.attempts = load_attempts()
        self.rows = deepcopy(self.authority["sacct"]["rows"])
        self.times = deepcopy(self.authority["nfs_mtime_ns"])

    def test_clock_policy_accepts_exact_cross_clock_observation(self) -> None:
        result = recovery.validate_clock_policy(self.rows, self.times)
        self.assertTrue(result["different_clock_domains"])
        self.assertTrue(result["nfs_domain_order_verified"])
        self.assertTrue(result["slurm_domain_order_verified"])

    def test_more_than_sixty_second_start_claim_skew_is_rejected(self) -> None:
        start = datetime.fromisoformat(self.rows[0]["Start"]).replace(
            tzinfo=timezone.utc
        ).timestamp()
        self.times["A"]["intent"] = int((start + 60.000001) * 1_000_000_000)
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_clock_policy(self.rows, self.times)

    def test_wrong_step_or_terminal_sacct_row_is_rejected(self) -> None:
        for field, value in (
            ("JobIDRaw", "140846.366"),
            ("State", "FAILED"),
            ("ExitCode", "1:0"),
            ("NodeList", "auh7-1b-gpu-247"),
            ("AllocTRES", "cpu=32,gres/gpu:mi210=4,gres/gpu=4,mem=60G,node=1"),
        ):
            hostile = deepcopy(self.rows)
            hostile[0][field] = value
            with self.subTest(field=field), self.assertRaises(recovery.RecoveryError):
                recovery.validate_sacct_rows(hostile, authority=self.authority)

    def test_NFS_and_slurm_order_violations_are_rejected(self) -> None:
        hostile_times = deepcopy(self.times)
        hostile_times["A"]["receipt"] = hostile_times["A"]["intent"] - 1
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_clock_policy(self.rows, hostile_times)

        hostile_times = deepcopy(self.times)
        hostile_times["A"]["SUCCESS"] = hostile_times["B"]["intent"] + 1
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_clock_policy(self.rows, hostile_times)

        hostile_rows = deepcopy(self.rows)
        hostile_rows[0]["End"] = "2026-08-17T04:32:13"
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_clock_policy(hostile_rows, self.times)

        hostile_rows = deepcopy(self.rows)
        hostile_rows[0]["End"] = "2026-08-17T04:25:57"
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_clock_policy(hostile_rows, self.times)

    def test_reused_session_or_step_is_rejected(self) -> None:
        b_receipt = deepcopy(self.attempts["B"].receipt)
        b_receipt["world8_consensus"]["rank_local_fresh_process_sessions"][0] = (
            self.attempts["A"].receipt["world8_consensus"][
                "rank_local_fresh_process_sessions"
            ][0]
        )
        hostile = {
            **self.attempts,
            "B": replace(self.attempts["B"], receipt=b_receipt),
        }
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_cross_attempt_parity(hostile)

        b_receipt = deepcopy(self.attempts["B"].receipt)
        b_receipt["launch_binding"]["slurm_numeric_step"] = "140846.367"
        hostile = {
            **self.attempts,
            "B": replace(self.attempts["B"], receipt=b_receipt),
        }
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_cross_attempt_parity(hostile)

    def test_projection_tamper_is_rejected_even_when_session_fields_are_allowed(self) -> None:
        b_receipt = deepcopy(self.attempts["B"].receipt)
        b_receipt["fresh_loaded_fixed_forward_fingerprint"]["tensor_set_sha256"] = (
            "0" * 64
        )
        hostile = {
            **self.attempts,
            "B": replace(self.attempts["B"], receipt=b_receipt),
        }
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_cross_attempt_parity(hostile)

    def test_parent_control_or_terminal_chain_tamper_cannot_be_resigned_by_caller(self) -> None:
        controller = deepcopy(self.attempts["A"].controller)
        controller["parent_signalled"] = True
        unsigned = dict(controller)
        unsigned.pop("status_digest")
        controller["status_digest"] = recovery.object_sha256(unsigned)
        with self.assertRaises(recovery.RecoveryError):
            recovery._validate_controller("A", controller, authority=self.authority)


if __name__ == "__main__":
    unittest.main()
