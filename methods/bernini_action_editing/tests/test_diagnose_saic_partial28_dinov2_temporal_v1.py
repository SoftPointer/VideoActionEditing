from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_partial28_dinov2_temporal_v1 as diagnostic


class Partial28DINOContractTest(unittest.TestCase):
    def test_partition_is_exactly_once_and_balanced(self) -> None:
        partitions = [
            diagnostic.partition_indices(28, rank, 8) for rank in range(8)
        ]
        self.assertEqual([len(row) for row in partitions], [4, 4, 4, 4, 3, 3, 3, 3])
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(28)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_authority_is_fail_closed(self) -> None:
        self.assertEqual(
            diagnostic.AUTHORITY_CLOSURE,
            {
                "diagnostic_only": True,
                "raw_proxy_evidence_only": True,
                "identity_authority": False,
                "identity_preservation_verified": False,
                "event_authority": False,
                "event_verified": False,
                "scientific_claim_authorized": False,
                "selection_authorized": False,
                "ranking_authorized": False,
                "training_target_authorized": False,
                "optimizer_or_parameter_update_authorized": False,
            },
        )

    def test_create_only_is_canonical_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "receipt.json"
            value = {"z": 2, "a": 1}
            diagnostic._write_create_only(path, value)
            self.assertEqual(path.read_bytes(), b'{"a":1,"z":2}\n')
            with self.assertRaises(diagnostic.Partial28DINOError):
                diagnostic._write_create_only(path, value)

    def test_generation_receipt_binds_exact81_mp4_and_false_authority(self) -> None:
        root_sha = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            attempt = root / "attempt"
            attempt.mkdir()
            mp4 = attempt / "t2v.mp4"
            mp4.write_bytes(b"sealed-exact81-placeholder")
            native = attempt / "receipt.json"
            native.write_bytes(b"{}\n")
            envelope = root / "envelope.json"
            envelope.write_bytes(b"{}\n")
            candidate = {
                "candidate_id": "saic-topup-v2-test-appearance_only-s1",
                "ordinal": 0,
                "iid": "test",
                "row_id": "row-test",
                "actor_family": "human",
                "analysis_split": "fit",
                "branch": "appearance_only",
                "seed": 1,
                "event_verified": False,
                "identity_preservation_verified": False,
                "seed_selection_authorized": False,
                "training_target_authorized": False,
                "optimizer_authorized": False,
            }
            receipt = {field: None for field in diagnostic.topup_generate._ATTEMPT_FIELDS}
            receipt.update(
                {
                    "schema_version": diagnostic.topup_generate.SCHEMA_VERSION,
                    "bank_id": diagnostic.topup_generate.contract.BANK_ID,
                    "top_up_only": True,
                    "root_spec_raw_sha256": root_sha,
                    "sampling_contract": diagnostic.topup_generate.contract.SAMPLING_CONTRACT,
                    "semantic_input_closure": diagnostic.topup_generate.contract.SEMANTIC_INPUT_CLOSURE,
                    "geometry_proxy_contract": diagnostic.topup_generate.contract.GEOMETRY_PROXY_CONTRACT,
                    "artifact_authority": diagnostic.topup_generate.contract.ARTIFACT_AUTHORITY,
                    "candidate": candidate,
                    "candidate_envelope_path": str(envelope),
                    "candidate_envelope_sha256": diagnostic.file_sha256(envelope),
                    "native_receipt_path": str(native),
                    "native_receipt_sha256": diagnostic.file_sha256(native),
                    "native_receipt_digest": "b" * 64,
                    "artifacts": {
                        "mp4": {
                            "path": str(mp4),
                            "sha256": diagnostic.file_sha256(mp4),
                            "frame_count": 81,
                            "fps": 25,
                        }
                    },
                    "event_audit_status": "pending_detached_full81_review",
                    "event_verified": False,
                    "identity_preservation_verified": False,
                    "seed_selection_authorized": False,
                    "training_target_authorized": False,
                    "optimizer_or_parameter_update_authorized": False,
                }
            )
            unsigned = dict(receipt)
            unsigned.pop("receipt_digest")
            receipt["receipt_digest"] = diagnostic.object_sha256(unsigned)
            receipt_path = attempt / diagnostic.ATTEMPT_BASENAME
            receipt_path.write_bytes(diagnostic.canonical_json_bytes(receipt) + b"\n")

            checked = diagnostic.validate_attempt_receipt(
                receipt_path, expected_root_spec_sha256=root_sha
            )
            self.assertEqual(checked["mp4_sha256"], diagnostic.file_sha256(mp4))
            self.assertEqual(checked["declared_frame_count"], 81)
            self.assertFalse(checked["upstream_event_verified"])
            self.assertFalse(checked["upstream_selection_authorized"])

            tampered = json.loads(receipt_path.read_text("utf-8"))
            tampered["event_verified"] = True
            unsigned = dict(tampered)
            unsigned.pop("receipt_digest")
            tampered["receipt_digest"] = diagnostic.object_sha256(unsigned)
            receipt_path.write_bytes(diagnostic.canonical_json_bytes(tampered) + b"\n")
            with self.assertRaises(diagnostic.Partial28DINOError):
                diagnostic.validate_attempt_receipt(
                    receipt_path, expected_root_spec_sha256=root_sha
                )

    def test_temporal_proxy_exposes_traces_but_no_threshold_or_authority(self) -> None:
        import torch

        global_feature = torch.nn.functional.normalize(
            torch.arange(17 * 8, dtype=torch.float32).reshape(17, 8) + 1.0,
            dim=-1,
        )
        dense_feature = torch.nn.functional.normalize(
            torch.arange(17 * 4 * 8, dtype=torch.float32).reshape(17, 4, 8) + 1.0,
            dim=-1,
        )
        value = diagnostic.temporal_proxy(global_feature, dense_feature)
        self.assertEqual(len(value["global_adjacent_mapped_cosine"]), 16)
        self.assertEqual(len(value["global_frame0_to_later_mapped_cosine"]), 16)
        self.assertEqual(len(value["dense_adjacent_token_median_mapped_cosine"]), 16)
        self.assertIsNone(value["thresholds"])
        self.assertFalse(value["identity_authority"])
        self.assertFalse(value["event_authority"])
        self.assertFalse(value["scientific_claim_authorized"])
        self.assertFalse(value["selection_authorized"])

    def test_launcher_maps_eight_distinct_visible_devices(self) -> None:
        launcher = (
            METHOD_ROOT / "scripts" / "auh_diagnose_saic_partial28_dinov2_temporal_v1.sh"
        ).read_text("utf-8")
        self.assertIn("for rank in 0 1 2 3 4 5 6 7", launcher)
        self.assertIn('ROCR_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn(
            "env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL",
            launcher,
        )
        self.assertNotIn('HIP_VISIBLE_DEVICES="$rank"', launcher)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn("aggregate is forbidden", launcher)


if __name__ == "__main__":
    unittest.main()
