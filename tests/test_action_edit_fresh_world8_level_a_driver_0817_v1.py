#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = REPO_ROOT / "methods" / "bernini_action_editing"
DRIVER_PATH = METHOD_ROOT / "action_edit_fresh_world8_level_a_driver_0817_v1.py"
AUDIT_MANIFEST = (
    METHOD_ROOT
    / "audits"
    / "fresh_world8_level_a_r2_p2_launchbound_v2_RELEASE_MANIFEST.json"
)
LAUNCH_CORE = (
    METHOD_ROOT / "audits" /
    "fresh_world8_level_a_r2_p2_launchbound_v2_LAUNCH_AUTHORITY_CORE.json"
)
DEPLOYMENT_PINS = (
    METHOD_ROOT / "audits" /
    "fresh_world8_level_a_r2_p2_launchbound_v2_DEPLOYMENT_PINS.json"
)
SCRIPT_ROOT = METHOD_ROOT / "scripts"
CONTROLLER = SCRIPT_ROOT / "auh_launch_fresh_world8_level_a_r2_p2_node279_job140846_v1.sh"
STEP = SCRIPT_ROOT / "auh_fresh_world8_level_a_r2_p2_node279_step_v1.sh"
RANK = SCRIPT_ROOT / "auh_fresh_world8_level_a_r2_p2_node279_rank_exec_v1.sh"


def load_subject(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = load_subject(DRIVER_PATH, "action_edit_fresh_world8_level_a_driver_0817_v1_test")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ReleaseFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (Path(self.temporary.name) / "release").resolve()
        self.root.mkdir(mode=0o700)
        for relative in driver.EXPECTED_RELEASE_PATHS:
            source = DRIVER_PATH if relative == driver.DRIVER_FILENAME else METHOD_ROOT / relative
            shutil.copyfile(source, self.root / relative)
            os.chmod(self.root / relative, 0o444)
        self.driver_sha = sha((self.root / driver.DRIVER_FILENAME).read_bytes())
        rows = []
        for relative in driver.EXPECTED_RELEASE_PATHS:
            payload = (self.root / relative).read_bytes()
            rows.append(
                {
                    "path": relative,
                    "mode": 0o444,
                    "size": len(payload),
                    "sha256": sha(payload),
                }
            )
        manifest = {
            "schema_version": driver.RELEASE_SCHEMA,
            "member_root": driver.RELEASE_MEMBER_ROOT,
            "files": rows,
        }
        self.manifest_payload = driver.canonical_json_bytes(manifest)
        self.manifest = self.root / driver.RELEASE_MANIFEST_FILENAME
        self.manifest.write_bytes(self.manifest_payload)
        os.chmod(self.manifest, 0o444)
        os.chmod(self.root, 0o555)
        self.addCleanup(self._make_writable)

    def _make_writable(self) -> None:
        try:
            os.chmod(self.root, 0o700)
            for item in self.root.iterdir():
                if not item.is_symlink():
                    os.chmod(item, 0o600)
        except OSError:
            pass

    def authenticate(self):
        return driver.validate_deployment_release(
            self.manifest,
            expected_manifest_sha256=sha(self.manifest_payload),
            expected_driver_sha256=self.driver_sha,
        )

    def mutate_root(self, function) -> None:
        os.chmod(self.root, 0o755)
        function()
        os.chmod(self.root, 0o555)


class DeploymentReleaseTests(ReleaseFixture):
    def test_exact_eleven_member_release_authenticates(self) -> None:
        release = self.authenticate()
        self.assertEqual(len(release.members), 11)
        self.assertEqual(set(release.source_bytes), set(driver.EXPECTED_RELEASE_PATHS))
        self.assertEqual(release.driver_sha256, self.driver_sha)

    def test_extra_cache_or_binary_member_is_rejected(self) -> None:
        for name in ("__pycache__", "shadow.pyc", "shadow.so"):
            with self.subTest(name=name):
                def add() -> None:
                    path = self.root / name
                    if name == "__pycache__":
                        path.mkdir()
                    else:
                        path.write_bytes(b"forbidden")

                self.mutate_root(add)
                with self.assertRaises(driver.LevelADriverError):
                    self.authenticate()
                os.chmod(self.root, 0o755)
                path = self.root / name
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
                os.chmod(self.root, 0o555)

    def test_member_symlink_hardlink_mode_and_bytes_are_rejected(self) -> None:
        victim = self.root / driver.PRODUCT_FILENAME
        original = victim.read_bytes()

        def symlink() -> None:
            victim.unlink()
            victim.symlink_to(self.root / "train_lora.py")

        self.mutate_root(symlink)
        with self.assertRaises(driver.LevelADriverError):
            self.authenticate()

        os.chmod(self.root, 0o755)
        victim.unlink()
        victim.write_bytes(original)
        os.chmod(victim, 0o444)
        outside = (Path(self.temporary.name) / "outside.py").resolve()
        outside.write_bytes(original)
        victim.unlink()
        os.link(outside, victim)
        os.chmod(self.root, 0o555)
        with self.assertRaises(driver.LevelADriverError):
            self.authenticate()

        os.chmod(self.root, 0o755)
        victim.unlink()
        outside.unlink()
        victim.write_bytes(original)
        os.chmod(victim, 0o644)
        os.chmod(self.root, 0o555)
        with self.assertRaises(driver.LevelADriverError):
            self.authenticate()

        os.chmod(self.root, 0o755)
        os.chmod(victim, 0o444)
        payload = bytearray(victim.read_bytes())
        payload[-1] ^= 1
        os.chmod(victim, 0o644)
        victim.write_bytes(payload)
        os.chmod(victim, 0o444)
        os.chmod(self.root, 0o555)
        with self.assertRaises(driver.LevelADriverError):
            self.authenticate()

    def test_wrong_driver_or_manifest_sha_is_rejected(self) -> None:
        with self.assertRaises(driver.LevelADriverError):
            driver.validate_deployment_release(
                self.manifest,
                expected_manifest_sha256="0" * 64,
                expected_driver_sha256=self.driver_sha,
            )
        with self.assertRaises(driver.LevelADriverError):
            driver.validate_deployment_release(
                self.manifest,
                expected_manifest_sha256=sha(self.manifest_payload),
                expected_driver_sha256="0" * 64,
            )

    def test_consumer_is_compiled_from_authenticated_source_bytes(self) -> None:
        release = self.authenticate()
        sys.modules.pop(driver.CONSUMER_MODULE, None)
        consumer = driver.load_consumer_from_authenticated_bytes(release)
        self.addCleanup(sys.modules.pop, driver.CONSUMER_MODULE, None)
        self.assertEqual(Path(consumer.__file__), self.root / driver.CONSUMER_FILENAME)
        self.assertTrue(callable(consumer.consume_frozen_r2_world8_checkpoint))
        self.assertTrue(callable(consumer.compare_fresh_world8_consumer_receipts))


class CanonicalIOTests(unittest.TestCase):
    def test_noncanonical_duplicate_and_nonfinite_json_are_rejected(self) -> None:
        for payload in (
            b'{"b":1,"a":2}',
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            b'{"a":1}\n',
        ):
            with self.subTest(payload=payload), self.assertRaises(driver.LevelADriverError):
                driver.strict_json_bytes(payload, label="hostile")
        self.assertEqual(driver.strict_json_bytes(b'{"a":1}', label="good"), {"a": 1})

    def test_atomic_receipt_is_canonical_sealed_and_single_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "out"
            root.mkdir(mode=0o700)
            destination = root / "receipt"
            digest = driver._atomic_write_canonical(
                destination, {"z": 2, "a": 1}, expected_parent_mode=0o700
            )
            self.assertEqual(destination.read_bytes(), b'{"a":1,"z":2}')
            info = destination.lstat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(info.st_nlink, 1)
            self.assertEqual(digest, sha(destination.read_bytes()))
            with self.assertRaises(driver.LevelADriverError):
                driver._atomic_write_canonical(
                    destination, {"a": 1}, expected_parent_mode=0o700
                )

    def test_create_only_publication_cannot_overwrite_and_finishes_nlink_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "out"
            root.mkdir(mode=0o700)
            destination = root / "validation.json"
            digest = driver._create_only_publish_canonical(
                destination, {"receipt_file_sha256": "a" * 64},
                expected_parent_mode=0o700,
            )
            self.assertEqual(digest, sha(destination.read_bytes()))
            self.assertEqual(stat.S_IMODE(destination.lstat().st_mode), 0o444)
            self.assertEqual(destination.lstat().st_nlink, 1)
            before = destination.read_bytes()
            with self.assertRaises(driver.LevelADriverError):
                driver._create_only_publish_canonical(
                    destination, {"receipt_file_sha256": "b" * 64},
                    expected_parent_mode=0o700,
                )
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["validation.json"]
            )


def level_a_receipt(
    offset: int,
    *,
    label: str = "A",
    step: int = 101,
    intent_sha: str = "a" * 64,
    core_sha: str = "b" * 64,
    claim_mtime_ns: int = 1,
    manifest_sha: str = "c" * 64,
    driver_sha: str = "d" * 64,
    step_sha: str = "e" * 64,
    rank_sha: str = "f" * 64,
) -> dict[str, object]:
    sessions = [f"{offset + rank:064x}" for rank in range(8)]
    raw = {key: None for key in driver.RAW_RECEIPT_KEYS}
    raw.update({
        "schema_version": "bernini-action-edit-fresh-consumer-receipt-v1",
        "method": "bernini-action-edit-checkpoint-consumer-0817-v1",
        "authority": driver.AUTHORITY,
        "complete": True,
        "promotable": False,
        "promotion_authorized": False,
        "formal_training_started": False,
        "counts_as_d0": False,
        "scientific_claim_authorized": False,
        "action_quality_claim_authorized": False,
        "checkpoint_step": 2,
        "checkpoint_parameter_sha256": driver.PINNED_P2_PARAMETER_SHA256,
        "loaded_parameter_sha256": driver.PINNED_P2_PARAMETER_SHA256,
        "campaign_receipt_sha256": driver.PINNED_R2_CAMPAIGN_RECEIPT_SHA256,
        "checkpoint_metadata_sha256": "3" * 64,
        "release_manifest_sha256": driver.PINNED_R2_RELEASE_MANIFEST_SHA256,
        "runner_source_sha256": "4" * 64,
        "predictor_source_sha256": "5" * 64,
        "conditioner_state_abi_sha256": "6" * 64,
        "consumer_source_sha256": driver.PINNED_CONSUMER_SHA256,
        "product_bridge_source_sha256": driver.PINNED_PRODUCT_SHA256,
        "fresh_process_session_id": sessions[0],
        "fresh_loaded_fixed_forward_executed": True,
        "fresh_loaded_fixed_forward_fingerprint": {
            "schema_version": "bernini-action-edit-fixed-forward-fingerprint-v1",
            "tensor_set_sha256": "9" * 64,
        },
        "world8_consumer_complete": True,
        "fresh_world8_process_forward_exact_consensus_verified": True,
        "fresh_world8_process_forward_scope": (
            "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
        ),
        "full_bernini_renderer_forward_executed": False,
        "checkpoint_bytes_conditioner_exact30_fresh_consumer_go": True,
        "offline_product_inference_completed": False,
        "full40_denoise_executed": False,
        "mp4_emitted": False,
        "training_attached_reference_absent": True,
        "training_attached_reference_present": False,
        "training_attached_conditioner_cell_reference_present": False,
        "training_attached_conditioner_cell_reference_absent": True,
        "training_attached_full_renderer_reference_present": False,
        "training_attached_full_renderer_reference_absent": True,
        "training_to_fresh_forward_parity_verified": False,
        "conditioner_cell_training_to_fresh_forward_parity_verified": False,
        "full_bernini_renderer_training_to_fresh_forward_parity_verified": False,
        "fresh_a_b_parity_verified": False,
        "fixed_forward_process_rng_unchanged": True,
        "fixed_forward_trainable_bytes_unchanged": True,
        "training_gradient_checkpoint_hooks_installed": False,
        "offline_single_forward_exact30_hooks_installed": True,
        "world8_consensus": None,
    })
    common = dict(raw)
    for key in driver.POST_CONSENSUS_ONLY_KEYS:
        common.pop(key)
    common.pop("fresh_process_session_id")
    raw["world8_consensus"] = {
            "world_size": 8,
            "rank_order": list(range(8)),
            "consumer_receipt_sha256": driver.object_sha256(common),
            "all8_exact_consensus": True,
            "eight_distinct_fresh_process_sessions": True,
            "rank_local_fresh_process_sessions": sessions,
    }
    launch_binding = {
        "schema_version": driver.LAUNCH_BINDING_SCHEMA,
        "attempt": label,
        "attempt_intent_sha256": intent_sha,
        "attempt_claim_mtime_ns": claim_mtime_ns,
        "launch_authority_core_sha256": core_sha,
        "parent_job_id": driver.PARENT_JOB_ID,
        "slurm_numeric_step": f"{driver.PARENT_JOB_ID}.{step}",
        "node": driver.PINNED_NODE,
        "job_name": driver.ATTEMPT_JOB_NAMES[label],
        "release_manifest_sha256": manifest_sha,
        "driver_source_sha256": driver_sha,
        "launcher_hash_chain": {
            "step_payload_sha256": step_sha,
            "rank_exec_sha256": rank_sha,
        },
    }
    unsigned = {**raw, "launch_binding": launch_binding}
    return {**unsigned, "receipt_digest": driver.object_sha256(unsigned)}


class ReceiptAuthorityTests(unittest.TestCase):
    def test_exact_published_receipt_recomputes_both_digests(self) -> None:
        receipt = level_a_receipt(100)
        receipt_digest, consensus_digest = driver._validate_level_a_receipt(receipt)
        self.assertEqual(receipt_digest, receipt["receipt_digest"])
        self.assertEqual(
            consensus_digest,
            receipt["world8_consensus"]["consumer_receipt_sha256"],
        )

    def test_fabricated_missing_extra_and_self_digest_are_rejected(self) -> None:
        good = level_a_receipt(100)
        hostile = []
        missing = dict(good); missing.pop("receipt_digest"); hostile.append(missing)
        extra = dict(good); extra["fabricated"] = True; hostile.append(extra)
        wrong = dict(good); wrong["receipt_digest"] = "0" * 64; hostile.append(wrong)
        for receipt in hostile:
            with self.subTest(keys=set(receipt)), self.assertRaises(driver.LevelADriverError):
                driver._validate_level_a_receipt(receipt)

    def test_recomputed_outer_digest_cannot_hide_wrong_world8_consensus(self) -> None:
        receipt = level_a_receipt(100)
        receipt["world8_consensus"]["consumer_receipt_sha256"] = "0" * 64
        unsigned = dict(receipt); unsigned.pop("receipt_digest")
        receipt["receipt_digest"] = driver.object_sha256(unsigned)
        with self.assertRaises(driver.LevelADriverError):
            driver._validate_level_a_receipt(receipt)

    def test_launch_intent_and_jobname_tamper_are_rejected(self) -> None:
        for key, value in (
            ("attempt_intent_sha256", "0" * 64),
            ("job_name", "historical-step"),
            ("slurm_numeric_step", "140846.batch"),
        ):
            receipt = level_a_receipt(100)
            receipt["launch_binding"][key] = value
            unsigned = dict(receipt); unsigned.pop("receipt_digest")
            receipt["receipt_digest"] = driver.object_sha256(unsigned)
            with self.subTest(key=key):
                if key == "attempt_intent_sha256":
                    good_intent_sha = "a" * 64
                    authority = driver.LaunchAuthority(
                        path=Path("/tmp/core"), sha256="b" * 64,
                        raw={
                            "release": {
                                "manifest_sha256": "c" * 64,
                                "driver_sha256": "d" * 64,
                            },
                            "launcher_hash_chain": {
                                "step_payload_sha256": "e" * 64,
                                "rank_exec_sha256": "f" * 64,
                            },
                            "attempts": {
                                "A": {"intent_sha256": good_intent_sha},
                                "B": {"intent_sha256": "1" * 64},
                            },
                        },
                    )
                    intent = driver.AttemptIntent(
                        path=Path("/tmp/intent"), sha256=good_intent_sha,
                        mtime_ns=1,
                        raw={"attempt": "A", "job_name": driver.ATTEMPT_JOB_NAMES["A"]},
                    )
                    with self.assertRaises(driver.LevelADriverError):
                        driver._validate_level_a_receipt(
                            receipt, authority=authority, intent=intent
                        )
                else:
                    with self.assertRaises(driver.LevelADriverError):
                        driver._validate_level_a_receipt(receipt)

    def _intent(self, label: str, mtime_ns: int) -> driver.AttemptIntent:
        return driver.AttemptIntent(
            path=Path(f"/tmp/{label}"), sha256=("a" if label == "A" else "b") * 64,
            mtime_ns=mtime_ns,
            raw={"attempt": label, "job_name": driver.ATTEMPT_JOB_NAMES[label]},
        )

    def _sacct_rows(self) -> list[dict[str, str]]:
        base = {
            "State": "COMPLETED", "ExitCode": "0:0", "NodeList": driver.PINNED_NODE,
            "NNodes": "1", "NTasks": "1",
            "AllocTRES": "cpu=32,gres/gpu:mi210=8,gres/gpu=8,mem=60G,node=1",
            "Start": "2026-08-17T10:00:02", "End": "2026-08-17T10:01:02",
        }
        return [
            {**base, "JobIDRaw": "140846.101", "JobName": driver.ATTEMPT_JOB_NAMES["A"]},
            {**base, "JobIDRaw": "140846.102", "JobName": driver.ATTEMPT_JOB_NAMES["B"]},
        ]

    def test_sacct_requires_distinct_exact_terminal_steps(self) -> None:
        # Claim at Unix epoch 1 is intentionally well before the fixed sacct rows.
        a = self._intent("A", 1_000_000_000)
        b = self._intent("B", 1_000_000_000)
        ra = level_a_receipt(100, label="A", step=101, intent_sha=a.sha256)
        rb = level_a_receipt(200, label="B", step=102, intent_sha=b.sha256)
        rows = self._sacct_rows()
        driver.validate_terminal_sacct_rows(
            rows, receipt_a=ra, receipt_b=rb, intent_a=a, intent_b=b
        )
        for field, value in (
            ("JobName", "wrong"), ("State", "FAILED"), ("ExitCode", "1:0"),
            ("NTasks", "8"),
        ):
            bad = [dict(row) for row in rows]; bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(driver.LevelADriverError):
                driver.validate_terminal_sacct_rows(
                    bad, receipt_a=ra, receipt_b=rb, intent_a=a, intent_b=b
                )

    def test_historical_sacct_step_before_claim_is_rejected(self) -> None:
        future_claim = 2_000_000_000_000_000_000
        a = self._intent("A", future_claim); b = self._intent("B", future_claim)
        ra = level_a_receipt(100, label="A", step=101, intent_sha=a.sha256)
        rb = level_a_receipt(200, label="B", step=102, intent_sha=b.sha256)
        with self.assertRaises(driver.LevelADriverError):
            driver.validate_terminal_sacct_rows(
                self._sacct_rows(), receipt_a=ra, receipt_b=rb, intent_a=a, intent_b=b
            )

    def test_terminal_success_cannot_rebind_receipt_or_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt_root = Path(temporary).resolve() / "A"
            output_root = Path(temporary).resolve() / "out-A"
            attempt_root.mkdir(mode=0o700); output_root.mkdir(mode=0o700)
            intent_sha, core_sha = "a" * 64, "b" * 64
            authority = driver.LaunchAuthority(
                path=Path("/tmp/core"), sha256=core_sha,
                raw={
                    "release": {"manifest_sha256": "c" * 64, "driver_sha256": "d" * 64},
                    "launcher_hash_chain": {
                        "step_payload_sha256": "e" * 64,
                        "rank_exec_sha256": "f" * 64,
                    },
                    "attempts": {"A": {"intent_sha256": intent_sha}},
                },
            )
            intent = driver.AttemptIntent(
                path=attempt_root / "STARTED" / "intent.json", sha256=intent_sha,
                mtime_ns=1,
                raw={
                    "attempt": "A", "job_name": driver.ATTEMPT_JOB_NAMES["A"],
                    "attempt_root": str(attempt_root), "output_root": str(output_root),
                },
            )
            receipt = level_a_receipt(
                100, label="A", step=101, intent_sha=intent_sha, core_sha=core_sha,
                manifest_sha="c" * 64, driver_sha="d" * 64,
            )
            receipt_payload = driver.canonical_json_bytes(receipt)
            receipt_sha = sha(receipt_payload)
            receipt_digest, consensus_sha = driver._validate_level_a_receipt(
                receipt, authority=authority, intent=intent
            )
            terminal_unsigned = {
                "schema_version": driver.TERMINAL_AUTHORITY_SCHEMA,
                "method": driver.METHOD, "authority": driver.AUTHORITY,
                "status": "SUCCESS", "attempt": "A",
                "attempt_intent_sha256": intent_sha,
                "launch_authority_core_sha256": core_sha,
                "controller_status_sha256": "2" * 64,
                "receipt_validation_sha256": "3" * 64,
                "consecutive_full_validations": 2,
                "receipt_file_sha256": receipt_sha,
                "receipt_digest": receipt_digest,
                "world8_consensus_sha256": consensus_sha,
                "slurm_numeric_step": "140846.101", "node": driver.PINNED_NODE,
                "job_name": driver.ATTEMPT_JOB_NAMES["A"],
                "receipt_validated": True, "parent_untouched": True,
                "automatic_relaunch_authorized": False,
                "promotion_authorized": False,
            }
            terminal = {
                **terminal_unsigned,
                "terminal_digest": driver.object_sha256(terminal_unsigned),
            }
            terminal_path = attempt_root / "terminal.authority.json"
            terminal_path.write_bytes(driver.canonical_json_bytes(terminal)); os.chmod(terminal_path, 0o444)
            terminal_sha = sha(terminal_path.read_bytes())
            success_unsigned = {
                "schema_version": driver.TERMINAL_SUCCESS_SCHEMA,
                "method": driver.METHOD, "authority": driver.AUTHORITY,
                "status": "SUCCESS", "attempt": "A",
                "attempt_intent_sha256": intent_sha,
                "launch_authority_core_sha256": core_sha,
                "terminal_authority_sha256": terminal_sha,
                "receipt_file_sha256": receipt_sha,
                "receipt_digest": receipt_digest,
                "world8_consensus_sha256": consensus_sha,
                "slurm_numeric_step": "140846.101", "node": driver.PINNED_NODE,
                "job_name": driver.ATTEMPT_JOB_NAMES["A"],
                "parent_untouched": True, "automatic_relaunch_authorized": False,
                "promotion_authorized": False,
            }
            success = {**success_unsigned, "success_digest": driver.object_sha256(success_unsigned)}
            success_path = attempt_root / "SUCCESS"
            success_path.write_bytes(driver.canonical_json_bytes(success)); os.chmod(success_path, 0o444)
            driver._read_terminal_pair(
                terminal_value=terminal_path, success_value=success_path,
                receipt=receipt, receipt_file_sha256=receipt_sha,
                authority=authority, intent=intent,
            )
            os.chmod(success_path, 0o600)
            success["receipt_file_sha256"] = "0" * 64
            unsigned = dict(success); unsigned.pop("success_digest")
            success["success_digest"] = driver.object_sha256(unsigned)
            success_path.write_bytes(driver.canonical_json_bytes(success)); os.chmod(success_path, 0o444)
            with self.assertRaises(driver.LevelADriverError):
                driver._read_terminal_pair(
                    terminal_value=terminal_path, success_value=success_path,
                    receipt=receipt, receipt_file_sha256=receipt_sha,
                    authority=authority, intent=intent,
                )


class FrozenDeploymentArtifactTests(unittest.TestCase):
    def test_audit_manifest_is_exact_current_eleven_source_closure(self) -> None:
        payload = AUDIT_MANIFEST.read_bytes()
        value = json.loads(payload.decode("utf-8"))
        self.assertEqual(value["schema_version"], driver.RELEASE_SCHEMA)
        self.assertEqual(
            [row["path"] for row in value["files"]],
            list(driver.EXPECTED_RELEASE_PATHS),
        )
        self.assertEqual(len(value["files"]), 11)
        for row in value["files"]:
            source = METHOD_ROOT / row["path"]
            source_payload = source.read_bytes()
            self.assertEqual(row["mode"], 0o444)
            self.assertEqual(row["size"], len(source_payload))
            self.assertEqual(row["sha256"], sha(source_payload))
        self.assertEqual(
            sha(payload),
            "f9e9f8542ec701cc9890fed919695980b989fd6d731eb914a5588edb1de4eeaa",
        )

    def test_launch_authority_core_is_canonical_and_binds_the_nonrecursive_chain(self) -> None:
        payload = LAUNCH_CORE.read_bytes()
        value = json.loads(payload.decode("utf-8"))
        self.assertEqual(
            payload,
            driver.canonical_json_bytes(value) + b"\n",
        )
        self.assertEqual(value["schema_version"], driver.LAUNCH_AUTHORITY_SCHEMA)
        self.assertEqual(value["roots"], dict(driver.PINNED_ROOTS))
        self.assertEqual(value["release"]["manifest_sha256"], sha(AUDIT_MANIFEST.read_bytes()))
        self.assertEqual(value["release"]["driver_sha256"], sha(DRIVER_PATH.read_bytes()))
        self.assertEqual(value["launcher_hash_chain"]["step_payload_sha256"], sha(STEP.read_bytes()))
        self.assertEqual(value["launcher_hash_chain"]["rank_exec_sha256"], sha(RANK.read_bytes()))
        self.assertEqual(value["topology"]["attempt_order"], ["A", "B"])
        self.assertNotIn("controller_sha256", value)
        controller = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn(sha(payload), controller)
        for label in ("A", "B"):
            self.assertEqual(value["attempts"][label]["job_name"], driver.ATTEMPT_JOB_NAMES[label])
            self.assertIn(value["attempts"][label]["intent_sha256"], controller)
            intent = {
                "schema_version": driver.ATTEMPT_INTENT_SCHEMA,
                "method": driver.METHOD,
                "authority": driver.AUTHORITY,
                "attempt": label,
                "parent_job_id": driver.PARENT_JOB_ID,
                "node": driver.PINNED_NODE,
                "job_name": value["attempts"][label]["job_name"],
                "release_root": value["roots"]["release"],
                "launch_root": value["roots"]["launch"],
                "attempt_root": value["attempts"][label]["attempt_root"],
                "output_root": value["attempts"][label]["output_root"],
                "checkpoint_step": 2,
                "world_size": 8,
                "dp_size": 2,
                "sp_size": 4,
                "release_manifest_sha256": value["release"]["manifest_sha256"],
                "driver_sha256": value["release"]["driver_sha256"],
                "consumer_sha256": value["release"]["consumer_sha256"],
                "product_bridge_sha256": value["release"]["product_bridge_sha256"],
                "step_payload_sha256": value["launcher_hash_chain"]["step_payload_sha256"],
                "rank_exec_sha256": value["launcher_hash_chain"]["rank_exec_sha256"],
                "automatic_relaunch_authorized": False,
                "parent_control_authorized": False,
            }
            self.assertEqual(driver.object_sha256(intent), value["attempts"][label]["intent_sha256"])

    def test_shell_payloads_parse_and_form_one_way_hash_chain(self) -> None:
        for script in (CONTROLLER, STEP, RANK):
            result = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        controller = CONTROLLER.read_text(encoding="utf-8")
        step = STEP.read_text(encoding="utf-8")
        rank = RANK.read_text(encoding="utf-8")
        driver_sha = sha(DRIVER_PATH.read_bytes())
        step_sha = sha(STEP.read_bytes())
        rank_sha = sha(RANK.read_bytes())
        manifest_sha = sha(AUDIT_MANIFEST.read_bytes())
        for text in (controller, step, rank):
            self.assertIn(driver_sha, text)
        for text in (controller, step):
            self.assertIn(rank_sha, text)
            self.assertIn(manifest_sha, text)
        self.assertIn(step_sha, controller)
        self.assertIn("--max-restarts=0", step)
        self.assertIn("--no_python", step)
        self.assertIn("run_attempt A\nrun_attempt B", controller)
        self.assertIn("--job-name=\"${job_name}\"", controller)
        self.assertIn("for poll in $(seq 1 60)", controller)
        self.assertEqual(controller.count("validate-receipt \\\n"), 2)
        self.assertNotIn('--output-validation "', controller)
        self.assertEqual(controller.count("publish-validation-pair \\\n"), 1)
        self.assertIn("first_status == 0 && second_status == 0", controller)
        self.assertIn('"${first_probe}" == "${second_probe}"', controller)
        self.assertIn("|| continue", controller)
        self.assertIn("receipt-validation-1.json", controller)
        self.assertIn("receipt-validation-2.json", controller)
        self.assertIn("/usr/bin/sacct", DRIVER_PATH.read_text(encoding="utf-8"))

    def test_node279_step_census_cannot_miss_compressed_multinode_nodelists(self) -> None:
        controller = CONTROLLER.read_text(encoding="utf-8")
        step = STEP.read_text(encoding="utf-8")
        for source in (controller, step):
            self.assertIn('/usr/bin/squeue --steps -w "${node}"', source)
            self.assertIn('-j "${job_id}"', source)
            self.assertIn('index($0, prefix) == 1', source)
            self.assertNotIn('$2 == wanted', source)
            self.assertNotIn("-o '%i|%N'", source)
        self.assertIn('"${node_numeric_steps}" == "${current_step}"', step)

    def test_launch_is_two_fresh_world8_attempts_without_parent_control_or_promotion(self) -> None:
        controller = CONTROLLER.read_text(encoding="utf-8")
        step = STEP.read_text(encoding="utf-8")
        rank = RANK.read_text(encoding="utf-8")
        executable_text = "\n".join(
            line
            for text in (controller, step, rank)
            for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        for forbidden in ("scancel", "scontrol", "killall", "pkill", "kill -"):
            self.assertNotIn(forbidden, executable_text)
        self.assertEqual(step.count("--nproc_per_node=8"), 1)
        self.assertIn("WORLD_SIZE:-}", rank)
        self.assertIn("TORCHELASTIC_MAX_RESTARTS", rank)
        self.assertIn("STARTED", controller)
        self.assertIn("automatic_relaunch_authorized=false", controller)
        self.assertIn('"parent_control_authorized":False', controller)
        self.assertIn('"receipt_validated": False', DRIVER_PATH.read_text(encoding="utf-8"))
        self.assertIn("terminal.authority.json", controller)
        self.assertIn("controller.status.json", controller)
        for false_claim in (
            "full_bernini_renderer_forward_executed=false",
            "offline_product_inference_completed=false",
            "full40_denoise_executed=false",
            "mp4_emitted=false",
            "promotion_authorized=false",
        ):
            self.assertIn(false_claim, controller)

    def test_outer_deployment_pins_bind_controller_without_a_hash_cycle(self) -> None:
        pins = json.loads(DEPLOYMENT_PINS.read_text(encoding="utf-8"))
        self.assertEqual(
            pins["schema_version"],
            "bernini-action-edit-fresh-world8-level-a-deployment-pins-v2",
        )
        self.assertEqual(pins["tag"], driver.TAG)
        self.assertEqual(pins["launch_authority_core"]["sha256"], sha(LAUNCH_CORE.read_bytes()))
        launchers = {row["path"]: row for row in pins["launchers"]}
        for path in (CONTROLLER, STEP, RANK):
            row = launchers[path.name]
            self.assertEqual(row["sha256"], sha(path.read_bytes()))
            self.assertEqual(row["size"], len(path.read_bytes()))
            self.assertEqual(row["mode"], 0o555)
        self.assertEqual(
            pins["release"]["manifest_sha256"], sha(AUDIT_MANIFEST.read_bytes())
        )
        self.assertEqual(pins["release"]["driver_sha256"], sha(DRIVER_PATH.read_bytes()))
        self.assertTrue(pins["hash_chain"]["outer_pins_bind_controller"])
        self.assertFalse(pins["hash_chain"]["launch_core_binds_controller"])
        self.assertEqual(
            pins["test"]["sha256"], sha(Path(__file__).resolve().read_bytes())
        )


if __name__ == "__main__":
    unittest.main()
