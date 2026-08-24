from __future__ import annotations

import copy
import json
from pathlib import Path
import stat
import tempfile
import unittest

from methods.bernini_action_editing import case01_oracle_object_trajectory_v1 as subject


ROOT = Path(__file__).resolve().parents[3]
STAGE0 = ROOT / "artifacts/object_grounded_case01_0821_sam2_masklets_r2"
G0 = (
    ROOT
    / "methods/bernini_action_editing/assets/"
    "case01_288545b9c031491a_g0_sparse_annotations_v1.json"
)
SOURCE = (
    ROOT
    / "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
    "videos/exact_original.mp4"
)
REMOVED = (
    ROOT
    / "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
    "videos/bone_removed.mp4"
)
FROZEN = ROOT / "artifacts/case01_oracle_object_trajectory_v1/scaffold.json"


def reseal(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("artifact_digest", None)
    result["artifact_digest"] = subject.object_sha256(result)
    return result


class Case01OracleObjectTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        cls.rebuilt = subject.build_artifact(
            stage0_root=STAGE0,
            g0_sparse_path=G0,
            source_video=SOURCE,
            bone_removed_video=REMOVED,
        )

    def test_real_authorities_rebuild_byte_exact_artifact(self) -> None:
        self.assertEqual(self.rebuilt, self.frozen)
        self.assertEqual(
            subject.canonical_json_bytes(self.rebuilt) + b"\n",
            FROZEN.read_bytes(),
        )
        self.assertEqual(
            subject._validate_artifact(self.frozen),
            self.frozen,
        )

    def test_phase_layout_and_conservation_are_exact(self) -> None:
        phases = self.frozen["latent_phases"]
        self.assertEqual(len(phases), 21)
        for index, row in enumerate(phases):
            self.assertEqual(row["phase_index"], index)
            self.assertEqual(row["frame_window"], list(subject._phase_window(index)))
            self.assertEqual(
                len(row["source_bone_tokens"]),
                len(row["target_bone_tokens"]),
            )
            self.assertFalse(
                set(row["dog_identity_core_tokens"])
                & (
                    set(row["origin_clear_tokens"])
                    | set(row["target_bone_tokens"])
                )
            )
        self.assertEqual(phases[0]["bone_shift_patch_xy"], [0, 0])
        self.assertNotEqual(phases[-1]["bone_shift_patch_xy"], [0, 0])
        self.assertEqual(phases[-1]["typed_stage"], "hold")

    def test_resealed_authority_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.frozen)
        tampered["authority"]["source_video"]["sha256"] = "0" * 64
        with self.assertRaises(subject.Case01TrajectoryError):
            subject._validate_artifact(reseal(tampered))

    def test_resealed_shift_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.frozen)
        tampered["latent_phases"][-1]["bone_shift_patch_xy"] = [0, 0]
        with self.assertRaises(subject.Case01TrajectoryError):
            subject._validate_artifact(reseal(tampered))

    def test_resealed_patient_identity_overlap_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.frozen)
        row = tampered["latent_phases"][-1]
        row["dog_identity_core_tokens"] = sorted(
            set(row["dog_identity_core_tokens"]) | {row["target_bone_tokens"][0]}
        )
        with self.assertRaises(subject.Case01TrajectoryError):
            subject._validate_artifact(reseal(tampered))

    def test_create_only_writer_seals_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "scaffold.json"
            subject._write_create_only(target, self.frozen)
            self.assertEqual(target.read_bytes(), FROZEN.read_bytes())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
            with self.assertRaises(subject.Case01TrajectoryError):
                subject._write_create_only(target, self.frozen)


if __name__ == "__main__":
    unittest.main()
