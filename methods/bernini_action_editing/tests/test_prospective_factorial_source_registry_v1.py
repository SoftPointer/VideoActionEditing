from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prospective_factorial_source_registry_v1 as registry  # noqa: E402


def sha(index: int) -> str:
    return f"{index:064x}"


def source(index: int, family: str, split: str) -> dict:
    return {
        "schema_version": registry.SOURCE_SCHEMA,
        "source_id": f"source-{index:02d}",
        "source_media_path": f"/sealed/source-{index:02d}.mp4",
        "source_media_sha256": sha(index + 100),
        "action_family": family,
        "review_decision": "accepted_prospective",
        "review_note": "independent source review accepted typed initial state",
        "assigned_split": split,
        "registered_seeds": [index * 10 + 1, index * 10 + 2],
    }


def complete_spec() -> dict:
    families = ["dog-stand-to-sit", "human-one-knee-to-stand"]
    rows = []
    index = 1
    for family in families:
        for split in registry.SPLITS:
            rows.append(source(index, family, split))
            index += 1
    return {
        "schema_version": registry.SPEC_SCHEMA,
        "population_id": "factorial-canary-v1",
        "created_utc": "2026-08-13T14:00:00Z",
        "minimum_sources_per_family_per_split": 1,
        "required_seeds_per_source": 2,
        "branch_order": list(registry.BRANCHES),
        "action_families": families,
        "evidence_bindings": [
            {
                "evidence_id": "source-audit-v1",
                "path": "/sealed/source-audit.json",
                "sha256": sha(1),
                "role": "source-only review; no target or score access",
            }
        ],
        "excluded_sources": [
            {
                "schema_version": registry.EXCLUSION_SCHEMA,
                "source_id": "opened-old-source",
                "source_media_sha256": sha(2),
                "reason": "opened_confirmation",
                "note": "previous primary read; never reused",
            }
        ],
        "sources": sorted(rows, key=lambda row: row["source_id"]),
    }


class ProspectiveFactorialSourceRegistryTests(unittest.TestCase):
    def test_balanced_registry_allows_generation_but_not_optimizer(self) -> None:
        receipt = registry.seal_registry(complete_spec())
        self.assertEqual(
            receipt["status"], "balanced_population_frozen_branch_generation_allowed"
        )
        self.assertTrue(receipt["branch_generation_allowed"])
        self.assertFalse(receipt["optimizer_step_allowed"])
        self.assertFalse(receipt["scores_consumed"])
        self.assertEqual(receipt["missing_quotas"], [])

    def test_incomplete_registry_reports_exact_missing_quota(self) -> None:
        spec = complete_spec()
        spec["minimum_sources_per_family_per_split"] = 2
        receipt = registry.seal_registry(spec)
        self.assertFalse(receipt["branch_generation_allowed"])
        self.assertEqual(len(receipt["missing_quotas"]), 6)
        self.assertTrue(all(item["missing"] == 1 for item in receipt["missing_quotas"]))

    def test_opened_confirmation_cannot_leak_back(self) -> None:
        spec = complete_spec()
        spec["sources"][0]["source_id"] = "opened-old-source"
        with self.assertRaisesRegex(registry.ProspectiveRegistryError, "excluded source leaked"):
            registry.seal_registry(spec)

    def test_media_hash_exclusion_cannot_be_renamed_away(self) -> None:
        spec = complete_spec()
        spec["sources"][0]["source_media_sha256"] = sha(2)
        with self.assertRaisesRegex(registry.ProspectiveRegistryError, "excluded source leaked"):
            registry.seal_registry(spec)

    def test_unaccepted_source_cannot_own_split_or_seed(self) -> None:
        spec = complete_spec()
        spec["sources"][0]["review_decision"] = "rejected_typed_state"
        with self.assertRaisesRegex(registry.ProspectiveRegistryError, "cannot own split"):
            registry.seal_registry(spec)

    def test_source_quality_rejection_has_no_generation_authority(self) -> None:
        spec = complete_spec()
        spec["sources"][0].update(
            {
                "review_decision": "rejected_source_quality",
                "assigned_split": None,
                "registered_seeds": [],
            }
        )
        receipt = registry.seal_registry(spec)
        self.assertFalse(receipt["branch_generation_allowed"])
        self.assertNotIn(
            spec["sources"][0]["source_id"],
            {row["source_id"] for row in receipt["accepted_sources"]},
        )

    def test_reserve_can_be_recorded_without_authority(self) -> None:
        spec = complete_spec()
        reserve = {
            **source(50, "dog-stand-to-sit", "fit"),
            "review_decision": "reserve_unsealed",
            "assigned_split": None,
            "registered_seeds": [],
        }
        spec["sources"].append(reserve)
        spec["sources"].sort(key=lambda row: row["source_id"])
        receipt = registry.seal_registry(spec)
        self.assertTrue(receipt["branch_generation_allowed"])
        self.assertNotIn("source-50", {row["source_id"] for row in receipt["accepted_sources"]})

    def test_branch_order_is_closed(self) -> None:
        spec = complete_spec()
        spec["branch_order"][-1] = "wrong_scene"
        with self.assertRaisesRegex(registry.ProspectiveRegistryError, "branch order differs"):
            registry.seal_registry(spec)

    def test_two_seeds_are_non_negotiable(self) -> None:
        spec = complete_spec()
        spec["sources"][0]["registered_seeds"] = [1]
        with self.assertRaisesRegex(registry.ProspectiveRegistryError, "exactly 2"):
            registry.seal_registry(spec)

    def test_confirmation_digest_changes_with_confirmation_identity(self) -> None:
        first = registry.seal_registry(complete_spec())
        spec = complete_spec()
        confirmation = next(
            row for row in spec["sources"] if row["assigned_split"] == "confirmation"
        )
        confirmation["registered_seeds"] = [9001, 9002]
        second = registry.seal_registry(spec)
        self.assertNotEqual(
            first["confirmation_registry_digest"], second["confirmation_registry_digest"]
        )

    def test_cli_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            output = root / "registry.json"
            spec_path.write_text(json.dumps(complete_spec()), encoding="utf-8")
            command = [
                sys.executable,
                str(Path(registry.__file__).resolve()),
                "--spec",
                str(spec_path),
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(written["branch_generation_allowed"])
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)

    def test_input_is_not_mutated(self) -> None:
        spec = complete_spec()
        before = copy.deepcopy(spec)
        registry.seal_registry(spec)
        self.assertEqual(spec, before)


if __name__ == "__main__":
    unittest.main()
