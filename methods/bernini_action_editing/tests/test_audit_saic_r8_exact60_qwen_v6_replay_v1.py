from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = METHOD_ROOT / "audit_saic_r8_exact60_qwen_v6_replay_v1.py"
LAUNCHER_PATH = (
    METHOD_ROOT
    / "scripts"
    / "auh_audit_saic_r8_exact60_qwen_v6_replay_v1.sh"
)
EXPECTED_SOURCE_SHA256 = (
    "45cfd756d5929126d591023a1c2b74b953dacd5431434ed99959e83bc53f7782"
)
EXPECTED_LAUNCHER_SHA256 = (
    "9ff1e7494c6a7d409c3c07bcc1c56a7bd08aeaa9a323812d8daa5334749572f4"
)

SPEC = importlib.util.spec_from_file_location("saic_r8_qwen_v6_replay", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def observation(**updates):
    value = {
        "schema_version": MODULE.qwen_v6.MODEL_OUTPUT_SCHEMA,
        "start_state_match": "yes",
        "requested_branch_change_present": "yes",
        "requested_change_fidelity": "exact",
        "requested_attribute_already_present_at_start": "not_applicable",
        "target_action_progress": "none",
        "terminal_state_reached": "no",
        "temporal_order_coherent": "yes",
        "identity_geometry_stable": "yes",
        "protected_scene_stable": "yes",
        "camera_motion_level": "conspicuous",
        "appearance_change_level": "none",
        "observed_evidence": ["F0 to F8 shows the registered camera change."],
    }
    value.update(updates)
    return MODULE.qwen_v6.validate_model_output(value)


class SaicR8QwenV6ReplayTests(unittest.TestCase):
    def test_actual_triplet_and_dependency_hash_pins(self) -> None:
        self.assertEqual(MODULE.file_sha256(MODULE_PATH), EXPECTED_SOURCE_SHA256)
        self.assertEqual(MODULE.file_sha256(LAUNCHER_PATH), EXPECTED_LAUNCHER_SHA256)
        self.assertEqual(
            MODULE.file_sha256(METHOD_ROOT / MODULE.QWEN_SOURCE_NAME),
            MODULE.QWEN_SOURCE_SHA256,
        )
        self.assertEqual(
            MODULE.EXPECTED_OLD_LAUNCHER_SHA256,
            "38f63226963b7d780639c3e7250916cde1f5a5e1012870d08cfcd8be03793a5a",
        )
        self.assertEqual(
            MODULE.EXPECTED_RECORDS_SHA256,
            "d885317804e62d9f58f183476f538f3e5dbba9f21579ddb8971ad160a48f38c4",
        )
        self.assertEqual(
            MODULE.EXPECTED_SUMMARY_SHA256,
            "c6e5a995267ddb5779481c9837bc5458a1b1217a25ea70c2ee99d5d8d02445c7",
        )

    def test_exact_formal_r8_hash_and_digest_pins(self) -> None:
        self.assertEqual(
            (MODULE.EXPECTED_TERMINAL_SHA256, MODULE.EXPECTED_TERMINAL_RECEIPT_DIGEST),
            (
                "07a6ec7ccbe165d89aa8757985537ef18d62eea5d08e245e452b607dee5bd29a",
                "a8fe672840d597445a2164660a38bdfeb4fa51ccfbc3b822c3af8adb6d6519e5",
            ),
        )
        self.assertEqual(
            (MODULE.EXPECTED_MASTER_SHA256, MODULE.EXPECTED_MASTER_RECEIPT_DIGEST),
            (
                "c5528a08fa976c0dbfb16984a35df3169c2d013a73fabd982ad45f45d5defc61",
                "8d28c170f5c8fdc5e76bdfb55bb89a5a819f02beb483c005f87d6898c5d8ae33",
            ),
        )
        self.assertEqual(
            MODULE.EXPECTED_DEEP,
            {
                "sp4-a": {
                    "sha256": "2c5b47c306a7cd7895278c3bc668bc8c895328ff7c528afcab8b4ccbdd83a67e",
                    "receipt_digest": "3a8bec49270bd426a360969141b404d21a43d37469b6876c0bc9d43e0124ac48",
                },
                "sp4-b": {
                    "sha256": "fca0e039259babae6188a8912a5990d16fe6584d9e8d8092eb02e036d83865d3",
                    "receipt_digest": "748a547808e47460292f409e37b7e9306f81907a6ffc463934dfee63740eb3ef",
                },
            },
        )

    def test_nonappearance_attribute_field_hole_is_closed(self) -> None:
        hostile = observation(
            requested_attribute_already_present_at_start="yes"
        )
        self.assertEqual(
            MODULE.qwen_v6.deterministic_branch_gate("camera_only", hostile),
            (True, []),
        )
        self.assertEqual(
            MODULE.corrected_branch_gate("camera_only", hostile),
            (False, ["appearance_start_field_misapplied"], True),
        )
        clean = observation()
        self.assertEqual(
            MODULE.corrected_branch_gate("camera_only", clean),
            (True, [], False),
        )

    def test_appearance_start_absence_remains_exactly_no(self) -> None:
        clean = observation(
            camera_motion_level="none",
            appearance_change_level="localized",
            requested_attribute_already_present_at_start="no",
        )
        self.assertEqual(
            MODULE.corrected_branch_gate("appearance_only", clean),
            (True, [], False),
        )
        for value, failure in (
            ("yes", "requested_attribute_present_at_start"),
            ("not_applicable", "appearance_start_field_misapplied"),
            ("uncertain", "insufficient_visual_evidence"),
        ):
            with self.subTest(value=value):
                hostile = observation(
                    camera_motion_level="none",
                    appearance_change_level="localized",
                    requested_attribute_already_present_at_start=value,
                )
                passed, failures, newly_detected = MODULE.corrected_branch_gate(
                    "appearance_only", hostile
                )
                self.assertFalse(passed)
                self.assertIn(failure, failures)
                self.assertFalse(newly_detected)

    def test_expected_corrected_pass_and_four_field_violations_are_frozen(self) -> None:
        self.assertEqual(
            MODULE.EXPECTED_CORRECTED_PASS_IDS,
            {
                "saic-topup-v2-99cde432839f4240-appearance_only-s2026082203"
            },
        )
        self.assertEqual(len(MODULE.EXPECTED_FIELD_VIOLATION_IDS), 4)
        self.assertIn(
            "saic-topup-v2-7b88a1ca1f804f41-camera_only-s2026082102",
            MODULE.EXPECTED_FIELD_VIOLATION_IDS,
        )

    def test_raw_pin_rejects_one_byte_mutation_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.write_bytes(b"sealed bytes\n")
            expected = hashlib.sha256(original.read_bytes()).hexdigest()
            raw, actual = MODULE._load_raw_pinned(
                original, expected, label="hostile fixture"
            )
            self.assertEqual(raw, b"sealed bytes\n")
            self.assertEqual(actual, expected)
            original.write_bytes(b"sealed byteS\n")
            with self.assertRaises(MODULE.QwenReplayError):
                MODULE._load_raw_pinned(original, expected, label="hostile fixture")
            target = root / "target"
            target.write_bytes(b"sealed bytes\n")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(MODULE.QwenReplayError):
                MODULE._load_raw_pinned(link, expected, label="hostile symlink")

    def test_receipt_rejects_valid_raw_pin_with_bad_embedded_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            unsigned = {"schema_version": "fixture", "count": 60}
            value = {**unsigned, "receipt_digest": MODULE.object_sha256(unsigned)}
            path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")
            raw_sha = MODULE.file_sha256(path)
            loaded, _ = MODULE._load_receipt_pinned(
                path,
                raw_sha,
                value["receipt_digest"],
                label="fixture receipt",
            )
            self.assertEqual(loaded, value)
            hostile = dict(value)
            hostile["count"] = 59
            path.write_text(json.dumps(hostile, sort_keys=True), encoding="ascii")
            hostile_raw_sha = MODULE.file_sha256(path)
            with self.assertRaises(MODULE.QwenReplayError):
                MODULE._load_receipt_pinned(
                    path,
                    hostile_raw_sha,
                    value["receipt_digest"],
                    label="hostile receipt",
                )

    def test_duplicate_json_keys_fail_closed(self) -> None:
        with self.assertRaises(MODULE.QwenReplayError):
            MODULE.decode_json(b'{"a":1,"a":1}', label="duplicate fixture")

    def test_output_is_create_only_and_all_authority_is_false(self) -> None:
        self.assertTrue(MODULE.AUTHORITY)
        self.assertTrue(all(value is False for value in MODULE.AUTHORITY.values()))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            MODULE._write_create_only(output, {"authority": MODULE.AUTHORITY})
            self.assertTrue(output.is_file())
            with self.assertRaises(MODULE.QwenReplayError):
                MODULE._write_create_only(output, {"authority": MODULE.AUTHORITY})

    def test_source_and_launcher_contain_no_model_execution(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)
        self.assertNotIn("QwenAuditor", called_names)
        self.assertNotIn("from_pretrained", called_names)
        self.assertNotIn("generate", called_names)
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torchrun", launcher)
        self.assertNotIn("--model", launcher)
        self.assertIn("expected_job_id=134964", launcher)
        self.assertIn("expected_node=auh7-1b-gpu-283", launcher)
        self.assertIn(f"readonly source_sha={EXPECTED_SOURCE_SHA256}", launcher)
        for digest in (
            MODULE.EXPECTED_OLD_LAUNCHER_SHA256,
            MODULE.EXPECTED_RECORDS_SHA256,
            MODULE.EXPECTED_SUMMARY_SHA256,
            MODULE.EXPECTED_TERMINAL_SHA256,
            MODULE.EXPECTED_MASTER_SHA256,
            MODULE.EXPECTED_DEEP["sp4-a"]["sha256"],
            MODULE.EXPECTED_DEEP["sp4-b"]["sha256"],
        ):
            self.assertIn(digest, launcher)

    def test_available_actual_qwen_files_match_pins_and_projection(self) -> None:
        records_path = Path("/private/tmp/fresh60_qwen_v6_records.jsonl")
        summary_path = Path("/private/tmp/fresh60_qwen_v6_summary.json")
        launcher_path = Path("/private/tmp/launch_saic_fresh60_qwen_v6_69eec35.sh")
        if not all(path.is_file() for path in (records_path, summary_path, launcher_path)):
            self.skipTest("developer copies of sealed Qwen evidence are unavailable")
        self.assertEqual(MODULE.file_sha256(records_path), MODULE.EXPECTED_RECORDS_SHA256)
        self.assertEqual(MODULE.file_sha256(summary_path), MODULE.EXPECTED_SUMMARY_SHA256)
        self.assertEqual(
            MODULE.file_sha256(launcher_path), MODULE.EXPECTED_OLD_LAUNCHER_SHA256
        )
        rows = [
            json.loads(line)
            for line in records_path.read_text(encoding="ascii").splitlines()
        ]
        old_passes = set()
        corrected_passes = set()
        violations = set()
        for row in rows:
            old_passed, old_failures = MODULE.qwen_v6.deterministic_branch_gate(
                row["branch"], row["validated_observation"]
            )
            self.assertIs(old_passed, row["deterministic_branch_gate_passed"])
            self.assertEqual(old_failures, row["deterministic_failure_codes"])
            corrected, _, field_violation = MODULE.corrected_branch_gate(
                row["branch"], row["validated_observation"]
            )
            if old_passed:
                old_passes.add(row["candidate_id"])
            if corrected:
                corrected_passes.add(row["candidate_id"])
            if field_violation:
                violations.add(row["candidate_id"])
        self.assertEqual(old_passes, MODULE.EXPECTED_OLD_PASS_IDS)
        self.assertEqual(corrected_passes, MODULE.EXPECTED_CORRECTED_PASS_IDS)
        self.assertEqual(violations, MODULE.EXPECTED_FIELD_VIOLATION_IDS)


if __name__ == "__main__":
    unittest.main()
