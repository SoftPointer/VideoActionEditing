from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "pair_v7_multicondition_geometry_authority.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v7_multicondition_geometry_authority as authority


class MulticonditionAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.event_ids = [str(row["event_id"]) for row in authority.FIXED_EVENTS]
        self.samples = [str(row["source_sample_id"]) for row in authority.FIXED_EVENTS]
        self.families = [str(row["action_family"]) for row in authority.FIXED_EVENTS]
        self.splits = ["fit", "fit", "confirmation", "confirmation"]
        self.seeds = [2026080825, 2026080827, 2026080826, 2026080828]
        self._create_fixture()

    def _write(self, name: str, payload: bytes) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path.resolve()

    def _write_json(self, name: str, value) -> Path:
        return self._write(name, authority.canonical_json_bytes(value) + b"\n")

    def _write_safetensor(
        self, name: str, key: str, shape: list[int], fill: float
    ) -> Path:
        count = 1
        for item in shape:
            count *= item
        raw = struct.pack("<f", fill) * count
        descriptor = {
            key: {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, len(raw)],
            }
        }
        header = json.dumps(
            descriptor, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        header += b" " * ((-len(header)) % 8)
        payload = len(header).to_bytes(8, "little") + header + raw
        return self._write(name, payload)

    def _create_fixture(self) -> None:
        shape = [1, 16, 21, 2, 2]
        draft_events = []
        fixed_events = []
        for index in range(4):
            source = self._write(
                f"source-{index}.mp4", f"source-video-{index}".encode("ascii")
            )
            clean = self._write_safetensor(
                f"clean-{index}.safetensors",
                "normalized_clean_latent",
                shape,
                0.125 + index,
            )
            gaussian = self._write_safetensor(
                f"gaussian-{index}.safetensors",
                "official_initial_gaussian",
                shape,
                -0.25 - index,
            )
            clean_tensor = authority._inspect_tensor_artifact(
                clean, "normalized_clean_latent", label="clean"
            )
            gaussian_tensor = authority._inspect_tensor_artifact(
                gaussian, "official_initial_gaussian", label="Gaussian"
            )
            expected = {
                "event_id": self.event_ids[index],
                "source_sample_id": self.samples[index],
                "action_family": self.families[index],
                "analysis_split": self.splits[index],
                "pair_wave": self.splits[index],
                "dp_arm": index % 2,
                "generation_seed": self.seeds[index],
                "latent_shape": shape,
                "source_video_file_sha256": authority._file_sha256(source),
                "clean_latent_file_sha256": authority._file_sha256(clean),
                "clean_latent_tensor_sha256": clean_tensor.tensor_sha256,
                "official_gaussian_file_sha256": authority._file_sha256(
                    gaussian
                ),
                "official_gaussian_tensor_sha256": (
                    gaussian_tensor.tensor_sha256
                ),
            }
            fixed_events.append(expected)
            draft_events.append(
                {
                    **expected,
                    "source_video_path": str(source),
                    "clean_latent_path": str(clean),
                    "clean_latent_tensor_key": "normalized_clean_latent",
                    "official_gaussian_path": str(gaussian),
                    "official_gaussian_tensor_key": "official_initial_gaussian",
                }
            )

        pilot_unsigned = {
            "schema_version": "bernini-pair-v7-phase-a-geometry-audit-v3",
            "audit_complete": True,
            "geometry_audit_passed": True,
            "schedule_policy": {"schedule_index": 33},
            "action_manifest": {"candidate_ids": self.event_ids[:2]},
            "parameter_mutation_performed": False,
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "parameter_add_called": False,
            **authority._NO_AUTHORITY,
        }
        pilot = authority._seal(pilot_unsigned, field="receipt_digest")
        self.pilot_path = self._write_json("pilot/receipt.json", pilot)
        self.pilot_file_sha = authority._file_sha256(self.pilot_path)
        self.pilot_receipt_digest = pilot["receipt_digest"]

        self.fixed_patch = patch.object(
            authority, "FIXED_EVENTS", tuple(fixed_events)
        )
        self.pilot_file_patch = patch.object(
            authority, "EXPECTED_PILOT_FILE_SHA256", self.pilot_file_sha
        )
        self.pilot_digest_patch = patch.object(
            authority,
            "EXPECTED_PILOT_RECEIPT_DIGEST",
            self.pilot_receipt_digest,
        )
        self.fixed_patch.start()
        self.pilot_file_patch.start()
        self.pilot_digest_patch.start()
        self.addCleanup(self.fixed_patch.stop)
        self.addCleanup(self.pilot_file_patch.stop)
        self.addCleanup(self.pilot_digest_patch.stop)

        self.draft = {
            "schema_version": authority.DRAFT_SCHEMA,
            "checkpoint_tree_sha256": (
                authority.EXPECTED_CHECKPOINT_TREE_SHA256
            ),
            "action_adapter_schema_sha256": (
                authority.EXPECTED_ACTION_ADAPTER_SCHEMA_SHA256
            ),
            "primary_schedule_indices": [16, 35],
            "source_noise_master_seed": 20260808,
            "pilot_receipt_path": str(self.pilot_path),
            "pilot_receipt_file_sha256": self.pilot_file_sha,
            "pilot_receipt_digest": self.pilot_receipt_digest,
            "events": draft_events,
            "cast_validation_performed": False,
            **authority._NO_AUTHORITY,
        }
        self.draft_path = self._write_json("draft.json", self.draft)

    def _author(self):
        output = self.root / "plan.json"
        plan = authority.author_preregistration(
            draft_path=self.draft_path, output_path=output
        )
        loaded = authority.validate_preregistration(
            plan_path=output,
            expected_plan_file_sha256=authority._file_sha256(output),
        )
        return output, plan, loaded

    def _reseal_plan(self, plan: dict) -> dict:
        unsigned = copy.deepcopy(plan)
        unsigned.pop("preregistration_digest", None)
        return authority._seal(unsigned, field="preregistration_digest")

    def _write_mutated_plan(self, plan: dict, name: str) -> Path:
        path = self._write_json(name, self._reseal_plan(plan))
        return path

    def _assert_rejected_plan(self, path: Path, pattern: str) -> None:
        with self.assertRaisesRegex(
            authority.PairV7MulticonditionAuthorityError, pattern
        ):
            authority.validate_preregistration(
                plan_path=path,
                expected_plan_file_sha256=authority._file_sha256(path),
            )

    def test_authors_fixed_four_by_two_plan_without_runtime_authority(self) -> None:
        output, plan, loaded = self._author()
        self.assertEqual(plan, loaded)
        self.assertEqual(plan["primary_schedule_indices"], [16, 35])
        self.assertEqual(
            [row["condition_id"] for row in plan["primary_cells"]],
            ["fit-s16", "fit-s35", "confirmation-s16", "confirmation-s35"],
        )
        self.assertEqual(
            plan["global_common_direction_spec"]["action_component_count"], 8
        )
        self.assertEqual(
            plan["global_common_direction_spec"]["identity_probe_count"], 64
        )
        self.assertFalse(plan["pilot_exclusion"]["included_in_primary_gate"])
        self.assertEqual(
            plan["pilot_exclusion"]["included_in_primary_condition_ids"], []
        )
        self.assertFalse(plan["cast_validation_performed"])
        self.assertFalse(plan["geometry_measurement_authorized"])
        for field in authority._NO_AUTHORITY:
            self.assertFalse(plan[field])
        with self.assertRaisesRegex(
            authority.PairV7MulticonditionAuthorityError, "create-only"
        ):
            authority.author_preregistration(
                draft_path=self.draft_path, output_path=output
            )

    def test_rejects_primary_cell_change_even_when_resealed(self) -> None:
        _, plan, _ = self._author()
        mutated = copy.deepcopy(plan)
        cell = mutated["primary_cells"][0]
        cell["schedule"] = dict(authority.SCHEDULES[35])
        unsigned_cell = dict(cell)
        unsigned_cell.pop("cell_digest")
        mutated["primary_cells"][0] = authority._seal(
            unsigned_cell, field="cell_digest"
        )
        path = self._write_mutated_plan(mutated, "mutated-cell.json")
        self._assert_rejected_plan(path, "primary cell closure differs")

    def test_rejects_source_change_even_when_resealed(self) -> None:
        _, plan, _ = self._author()
        mutated = copy.deepcopy(plan)
        event = mutated["events"][0]
        event["source_sample_id"] = "leaked-source"
        event["source_noise_key_sha256"] = authority._source_noise_key(
            "leaked-source"
        )
        unsigned_event = dict(event)
        unsigned_event.pop("event_digest")
        mutated["events"][0] = authority._seal(
            unsigned_event, field="event_digest"
        )
        path = self._write_mutated_plan(mutated, "mutated-source.json")
        self._assert_rejected_plan(path, "fixed core4 field differs")

    def test_rejects_master_or_generation_seed_change(self) -> None:
        _, plan, _ = self._author()
        master = copy.deepcopy(plan)
        master["source_noise_contract"]["master_seed"] = 20260809
        path = self._write_mutated_plan(master, "mutated-master-seed.json")
        self._assert_rejected_plan(path, "source-noise seed/derivation differs")

        generation = copy.deepcopy(plan)
        event = generation["events"][1]
        event["generation_seed"] += 1
        unsigned_event = dict(event)
        unsigned_event.pop("event_digest")
        generation["events"][1] = authority._seal(
            unsigned_event, field="event_digest"
        )
        path = self._write_mutated_plan(
            generation, "mutated-generation-seed.json"
        )
        self._assert_rejected_plan(path, "fixed core4 field differs")

        numeric_type = copy.deepcopy(plan)
        numeric_type["source_noise_contract"]["master_seed"] = 20260808.0
        path = self._write_mutated_plan(numeric_type, "float-master-seed.json")
        self._assert_rejected_plan(path, "source-noise seed/derivation differs")

    def test_rejects_split_change_even_when_resealed(self) -> None:
        _, plan, _ = self._author()
        mutated = copy.deepcopy(plan)
        event = mutated["events"][0]
        event["analysis_split"] = "confirmation"
        event["pair_wave"] = "confirmation"
        unsigned_event = dict(event)
        unsigned_event.pop("event_digest")
        mutated["events"][0] = authority._seal(
            unsigned_event, field="event_digest"
        )
        path = self._write_mutated_plan(mutated, "mutated-split.json")
        self._assert_rejected_plan(path, "fixed core4 field differs")

    def test_rejects_pilot_leakage_even_when_resealed(self) -> None:
        _, plan, _ = self._author()
        mutated = copy.deepcopy(plan)
        mutated["pilot_exclusion"]["included_in_primary_gate"] = True
        mutated["pilot_exclusion"]["included_in_primary_condition_ids"] = [
            "fit-s33"
        ]
        path = self._write_mutated_plan(mutated, "pilot-leakage.json")
        self._assert_rejected_plan(path, "pilot leakage")

        changed_boundary = copy.deepcopy(plan)
        changed_boundary["pilot_exclusion"][
            "observed_before_this_preregistration"
        ] = False
        path = self._write_mutated_plan(
            changed_boundary, "pilot-boundary-change.json"
        )
        self._assert_rejected_plan(path, "pilot exclusion closure differs")

    def test_rejects_schedule_rule_or_artifact_claim_change_when_resealed(
        self,
    ) -> None:
        _, plan, _ = self._author()
        schedule = copy.deepcopy(plan)
        schedule["schedule_selection_rule"] = "selected_after_primary_results"
        path = self._write_mutated_plan(schedule, "mutated-schedule-rule.json")
        self._assert_rejected_plan(
            path, "preregistration authority boundary differs"
        )

        artifact = copy.deepcopy(plan)
        artifact["artifact_validation"][
            "all_gaussian_tensor_hashes_verified"
        ] = False
        path = self._write_mutated_plan(
            artifact, "mutated-artifact-validation.json"
        )
        self._assert_rejected_plan(path, "artifact validation closure differs")

    def test_draft_rejects_schedule_source_seed_split_and_tensor_changes(self) -> None:
        mutations = {
            "schedule": lambda value: value["primary_schedule_indices"].__setitem__(
                0, 17
            ),
            "source": lambda value: value["events"][0].__setitem__(
                "source_sample_id", "other-source"
            ),
            "seed": lambda value: value.__setitem__(
                "source_noise_master_seed", 20260809
            ),
            "split": lambda value: value["events"][2].__setitem__(
                "analysis_split", "fit"
            ),
            "tensor": lambda value: value["events"][3].__setitem__(
                "clean_latent_tensor_sha256", "0" * 64
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                draft = copy.deepcopy(self.draft)
                mutate(draft)
                path = self._write_json(f"draft-{name}.json", draft)
                with self.assertRaises(
                    authority.PairV7MulticonditionAuthorityError
                ):
                    authority.author_preregistration(
                        draft_path=path,
                        output_path=self.root / f"plan-{name}.json",
                    )

    def test_external_artifact_change_is_detected_post_authoring(self) -> None:
        output, _, _ = self._author()
        source = Path(self.draft["events"][0]["source_video_path"])
        source.write_bytes(source.read_bytes() + b"changed")
        with self.assertRaisesRegex(
            authority.PairV7MulticonditionAuthorityError,
            "bound source video changed",
        ):
            authority.validate_preregistration(
                plan_path=output,
                expected_plan_file_sha256=authority._file_sha256(output),
            )

    def test_evidence_skeleton_explicitly_cannot_authorize_runtime(self) -> None:
        plan_path, plan, _ = self._author()
        output = self.root / "evidence-skeleton.json"
        skeleton = authority.author_evidence_skeleton(
            plan_path=plan_path,
            expected_plan_file_sha256=authority._file_sha256(plan_path),
            output_path=output,
        )
        self.assertEqual(
            skeleton["preregistration_digest"], plan["preregistration_digest"]
        )
        self.assertFalse(skeleton["cast_validation_performed"])
        self.assertFalse(skeleton["geometry_measurement_authorized"])
        self.assertFalse(skeleton["runtime_launch_authorized"])
        self.assertIn("missing_cast", skeleton["status"])
        for field in authority._NO_AUTHORITY:
            self.assertFalse(skeleton[field])

    def test_static_source_has_no_training_or_cast_success_path(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("parameter.add_(", source)
        self.assertIn('"cast_validation_performed": False', source)
        self.assertIn('"geometry_measurement_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
