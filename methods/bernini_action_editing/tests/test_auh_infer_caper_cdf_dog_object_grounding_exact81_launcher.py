#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_infer_caper_cdf_dog_object_grounding_exact81_all8.sbatch"
)


class AUHCaperCDFDogObjectGroundingLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax_and_all8_slurm_contract(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("#SBATCH --cpus-per-task=32", self.source)
        self.assertIn("#SBATCH --mem=256G", self.source)
        self.assertIn("--nproc_per_node=4", self.source)
        self.assertIn('launch_group "cdf-dog-historical-seed-2027" 0,1,2,3', self.source)
        self.assertIn('launch_group "cdf-dog-fresh-seed-2026081701" 4,5,6,7', self.source)

    def test_two_world4_groups_start_before_either_wait(self) -> None:
        historical_launch = self.source.index(
            'launch_group "cdf-dog-historical-seed-2027" 0,1,2,3'
        )
        fresh_launch = self.source.index(
            'launch_group "cdf-dog-fresh-seed-2026081701" 4,5,6,7'
        )
        first_wait = self.source.index('wait "${historical_pid}"')
        self.assertLess(historical_launch, fresh_launch)
        self.assertLess(fresh_launch, first_wait)
        self.assertIn(
            '"topology": "one_wave_two_concurrent_world4_sp4_groups_on_one_8gpu_node"',
            self.source,
        )
        self.assertIn('"one_concurrent_wave": True', self.source)

    def test_actual_source_and_registry_are_fail_closed(self) -> None:
        self.assertIn(
            "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/288545b9c031491a/source.mp4",
            self.source,
        )
        self.assertIn(
            "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18",
            self.source,
        )
        self.assertIn(
            "f91327227384d4d29308d43895fe71d2fa9b4666438b9ec99bf6c65e7b7283c8",
            self.source,
        )
        self.assertIn('sha256sum "${source_video}"', self.source)
        self.assertIn('sha256sum "${registry}"', self.source)
        self.assertIn('sha256sum "${source_archive}"', self.source)
        self.assertIn("git get-tar-commit-id", self.source)
        self.assertIn("git -C \"${repo_root}\" diff --quiet", self.source)
        self.assertIn("git -C \"${repo_root}\" diff --cached --quiet", self.source)

    def test_runtime_closure_and_preflight_tests_are_sealed(self) -> None:
        required = (
            "infer_caper_cdf_dog_object_grounding_exact81_canary.py",
            "caper_cdf_dog_object_grounding_exact81_v1.json",
            "infer_t2v_v2v_branch_homotopy_canary.py",
            "t2v_v2v_branch_homotopy_v1.py",
            "t2v_v2v_branch_homotopy_runtime_v1.py",
            "tri_branch_unipc.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "t2v_v2v_branch_homotopy_core4_v1.json",
            "native_branch_homotopy_core4_v1.json",
            "wrong_family_prompt_swap_pilot_registry_v1.json",
            "test_infer_caper_cdf_dog_object_grounding_exact81_canary.py",
            "test_auh_infer_caper_cdf_dog_object_grounding_exact81_launcher.py",
        )
        for name in required:
            self.assertIn(name, self.source)
        self.assertIn('[[ -f "${path}" && ! -L "${path}" ]]', self.source)
        self.assertIn("source_closure_sha256=", self.source)
        self.assertIn("launcher_sha256=", self.source)

    def test_exact_three_arms_and_exact81_shift5_postflight(self) -> None:
        self.assertIn('"native-source-video-only-v2v-endpoint"', self.source)
        self.assertIn('"pure-target-only-t2v-endpoint"', self.source)
        self.assertIn('"t2v-v2v-branch-homotopy-095-075"', self.source)
        self.assertIn('sampling.get("frame_count") != 81', self.source)
        self.assertIn('sampling.get("latent_phases") != 21', self.source)
        self.assertIn('sampling.get("num_inference_steps") != 40', self.source)
        self.assertIn('sampling.get("fps") != 25', self.source)
        self.assertIn('sampling.get("flow_shift_from_renderer_config") != 5.0', self.source)
        self.assertIn('observed_hw != {(496, 480)}', self.source)
        self.assertIn('[1, 16, 21, 62, 60]', self.source)
        self.assertIn("hom.get(\"transformer_forwards\") != 160", self.source)

    def test_object_event_gate_is_receipted_without_success_claim(self) -> None:
        for stage in ("approach", "contact", "grip", "lift", "hold"):
            self.assertRegex(self.source, rf'\b{re.escape(stage)}\b')
        for correspondence in (
            "source_dog_identity",
            "source_bone_identity",
            "source_dog_mouth_anatomical_identity",
        ):
            self.assertIn(correspondence, self.source)
        self.assertIn(
            'grounding.get("gate_status") != "not_automatically_adjudicated"',
            self.source,
        )
        self.assertIn(
            '"automatic_object_event_success_claim_authorized": False',
            self.source,
        )
        self.assertIn('"object_grounding_outcome_recorded": False', self.source)
        self.assertIn('"manual_review_required": True', self.source)

    def test_same_source_seeds_are_not_counted_as_identities(self) -> None:
        self.assertIn(
            '"same_source_cells_are_seed_replicates_not_independent_identities": True',
            self.source,
        )
        self.assertIn(
            '"aggregate_as_independent_identities_authorized": False', self.source
        )
        self.assertIn('"single_example_conclusion_authorized": False', self.source)
        self.assertIn('"seeds": [seed for _, seed, _ in cells]', self.source)
        self.assertNotIn('"independent_identity_count": 2', self.source)

    def test_checkpoint_freeze_hash_and_no_update_are_postflighted(self) -> None:
        self.assertIn("checkpoint-content-manifest", self.source)
        self.assertIn('freeze.get("base_frozen") is not True', self.source)
        self.assertIn('freeze.get("adapter_modules_absent") is not True', self.source)
        self.assertIn(
            'freeze.get("exact_parameter_and_buffer_bytes_hashed") is not True',
            self.source,
        )
        self.assertIn(
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            self.source,
        )
        self.assertIn(
            "2d2b4591ac053ec25c6371b01a5a6746679e5793", self.source
        )
        self.assertIn(
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d", self.source
        )
        self.assertIn(
            "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512",
            self.source,
        )
        self.assertIn('row.get("training_performed") is not False', self.source)
        self.assertIn('row.get("optimizer_created") is not False', self.source)
        self.assertIn('row.get("parameter_update") is not False', self.source)
        self.assertIn('"training_performed": False', self.source)
        self.assertIn('"optimizer_created": False', self.source)
        self.assertIn('"parameter_update": False', self.source)


if __name__ == "__main__":
    unittest.main()
