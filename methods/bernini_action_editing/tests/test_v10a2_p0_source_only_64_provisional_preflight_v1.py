from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import v10a2_p0_source_only_64_provisional_preflight_v1 as preflight


REGISTRY_PATH = (
    METHOD_ROOT / "assets" / "v10a2_p0_source_only_64_provisional_v1.json"
)
ACTUAL_MANIFEST_PATH = (
    METHOD_ROOT
    / "assets"
    / "target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rehash_registry(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("provisional_registry_sha256", None)
    result["provisional_registry_sha256"] = preflight.object_sha256(result)
    return result


def build(*, registry: dict | None = None, actual_manifest: dict | None = None) -> dict:
    return dict(
        preflight.build_preflight_receipt(
            registry=load(REGISTRY_PATH) if registry is None else registry,
            registry_file_sha256=preflight.EXPECTED_REGISTRY_FILE_SHA256,
            actual_manifest=(
                load(ACTUAL_MANIFEST_PATH)
                if actual_manifest is None
                else actual_manifest
            ),
            actual_manifest_file_sha256=preflight.file_sha256(
                ACTUAL_MANIFEST_PATH
            ),
        )
    )


class V10A2P0SourceOnly64ProvisionalPreflightV1Tests(unittest.TestCase):
    def assert_semantic_tamper_rejected(
        self, registry: dict, message: str
    ) -> None:
        sealed = rehash_registry(registry)
        with mock.patch.object(
            preflight,
            "EXPECTED_REGISTRY_SELF_SHA256",
            sealed["provisional_registry_sha256"],
        ):
            with self.assertRaisesRegex(
                preflight.V10A2P0ProvisionalPreflightError, message
            ):
                build(registry=sealed)

    def test_checked_in_registry_has_canonical_self_hash(self) -> None:
        registry = load(REGISTRY_PATH)
        expected = registry.pop("provisional_registry_sha256")
        self.assertEqual(expected, preflight.EXPECTED_REGISTRY_SELF_SHA256)
        self.assertEqual(preflight.object_sha256(registry), expected)
        self.assertEqual(
            preflight.file_sha256(REGISTRY_PATH),
            preflight.EXPECTED_REGISTRY_FILE_SHA256,
        )

    def test_current_receipt_is_integrity_pass_but_hard_p0_no(self) -> None:
        receipt = build()
        self.assertEqual(receipt["status"], preflight.RECEIPT_STATUS)
        self.assertTrue(receipt["integrity_verified"])
        self.assertTrue(receipt["sanitized_field_contract_verified"])
        self.assertEqual(receipt["candidate_count"], 64)
        self.assertEqual(
            receipt["selected_strata"],
            {stratum: 16 for stratum in preflight.STRATA},
        )
        self.assertEqual(receipt["exact_uuid_path_media_overlap_with_actual"], 0)
        self.assertFalse(receipt["perceptual_exclusion_complete"])
        self.assertFalse(receipt["frozen_observer_qualification_complete"])
        self.assertFalse(receipt["official_source_only_registry"])
        self.assertFalse(receipt["p0_slot_pretraining_authorized"])
        self.assertFalse(receipt["gpu_launch_authorized"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["optimizer_created"])
        self.assertEqual(receipt["parameter_updates"], 0)
        self.assertEqual(receipt["v10a2_blocker"], preflight.V10A2_BLOCKER)
        self.assertEqual(receipt["known_unresolved"], list(preflight.KNOWN_UNRESOLVED))
        receipt_hash = receipt.pop("receipt_sha256")
        self.assertEqual(preflight.object_sha256(receipt), receipt_hash)

    def test_direct_tamper_fails_self_hash(self) -> None:
        registry = load(REGISTRY_PATH)
        registry["rows"][0]["size_bytes"] += 1
        with self.assertRaisesRegex(
            preflight.V10A2P0ProvisionalPreflightError, "self hash differs"
        ):
            build(registry=registry)

    def test_rehashed_duplicate_candidate_is_rejected(self) -> None:
        registry = load(REGISTRY_PATH)
        duplicate = copy.deepcopy(registry["rows"][0])
        duplicate["ordinal"] = 1
        registry["rows"][1] = duplicate
        self.assert_semantic_tamper_rejected(registry, "UUIDs are not unique")

    def test_rehashed_actual_uuid_path_and_media_leak_is_rejected(self) -> None:
        registry = load(REGISTRY_PATH)
        actual = load(ACTUAL_MANIFEST_PATH)["pairs"][0]
        leaked = registry["rows"][0]
        leaked["uuid"] = actual["uuid"]
        leaked["source_video_path"] = actual["source_video_path"]
        leaked["size_bytes"] = actual["source_media"]["size_bytes"]
        leaked["source_media_sha256"] = actual["source_media"]["sha256"]
        leaked["selection_rank_sha256"] = preflight._expected_rank(actual["uuid"])
        registry["rows"][:16] = sorted(
            registry["rows"][:16],
            key=lambda row: (row["selection_rank_sha256"], row["uuid"]),
        )
        for ordinal, row in enumerate(registry["rows"]):
            row["ordinal"] = ordinal
        self.assert_semantic_tamper_rejected(
            registry, "candidate UUID leaks into the actual split"
        )

    def test_rehashed_quality_threshold_tamper_is_rejected(self) -> None:
        registry = load(REGISTRY_PATH)
        registry["rows"][0]["quality"]["imaging_quality"] = 0.59
        self.assert_semantic_tamper_rejected(
            registry, "candidate quality threshold fails"
        )

    def test_rehashed_selection_rank_tamper_is_rejected(self) -> None:
        registry = load(REGISTRY_PATH)
        registry["rows"][0]["selection_rank_sha256"] = "a" * 64
        self.assert_semantic_tamper_rejected(
            registry, "candidate selection rank differs"
        )

    def test_rehashed_caption_field_leak_is_rejected(self) -> None:
        registry = load(REGISTRY_PATH)
        registry["rows"][0]["caption"] = "forbidden text"
        self.assert_semantic_tamper_rejected(
            registry, "forbidden text-bearing key"
        )

    def test_rehashed_fake_ready_status_is_rejected(self) -> None:
        registry = load(REGISTRY_PATH)
        registry["status"] = "READY"
        self.assert_semantic_tamper_rejected(
            registry, "cannot emit a ready or authority status"
        )

    def test_rehashed_fake_p0_authority_is_rejected(self) -> None:
        registry = load(REGISTRY_PATH)
        registry["authority"]["p0_slot_pretraining_authorized"] = True
        self.assert_semantic_tamper_rejected(
            registry, "authority must remain hard P0 NO"
        )

    def test_registry_whitespace_change_breaks_file_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            changed = Path(temporary_directory) / "registry.json"
            changed.write_bytes(REGISTRY_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                preflight.V10A2P0ProvisionalPreflightError,
                "provisional registry bytes differ",
            ):
                preflight.run_preflight(changed, ACTUAL_MANIFEST_PATH)

    def test_actual_manifest_tamper_is_rejected(self) -> None:
        actual = load(ACTUAL_MANIFEST_PATH)
        actual["pairs"][0]["formal_sft_authorized"] = True
        with self.assertRaisesRegex(
            preflight.V10A2P0ProvisionalPreflightError,
            "actual manifest self hash differs",
        ):
            build(actual_manifest=actual)


if __name__ == "__main__":
    unittest.main()
