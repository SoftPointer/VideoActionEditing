from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    METHOD_ROOT / "assets" / "mev840_action_graph_multiappearance_anchor_bank_r2.json"
)
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_generate_mev840_action_graph_multiappearance_anchor_r2.sh"
)
EXPECTED_MANIFEST_SHA256 = (
    "be8dba8d32b63d79660f46f38fa3f66926b3fa29947d6898726556ce130336db"
)


class AUHMEV840ActionGraphMultiappearanceAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_bytes = MANIFEST.read_bytes()
        cls.manifest = json.loads(cls.manifest_bytes)
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")

    def test_manifest_and_launcher_are_syntax_valid_and_content_bound(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.manifest_bytes).hexdigest(),
            EXPECTED_MANIFEST_SHA256,
        )
        self.assertIn(
            f"readonly expected_manifest_sha256={EXPECTED_MANIFEST_SHA256}",
            self.launcher,
        )
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.launcher, re.DOTALL)
        self.assertEqual(len(python_blocks), 2)
        for block in python_blocks:
            ast.parse(block)

    def test_authority_is_representation_only_frozen_t2v(self) -> None:
        self.assertEqual(
            self.manifest["schema_version"],
            "mev840-self-generated-action-graph-anchor-bank-r2",
        )
        authorization = self.manifest["authorization"]
        self.assertTrue(authorization["native_frozen_t2v_anchor_generation"])
        for field in (
            "source_edit_route",
            "source_edit_decode",
            "training",
            "lora",
            "parameter_updates",
            "real_target_read",
            "external_reference_read",
            "mask_flow_pose_track_trajectory_read",
        ):
            self.assertIs(authorization[field], False)
        self.assertEqual(authorization["optimizer_steps"], 0)
        self.assertEqual(
            self.manifest["supersedes_failed_preflight"]["slurm_steps"],
            ["143808.468", "143808.469", "143808.470"],
        )
        self.assertFalse(
            self.manifest["supersedes_failed_preflight"]["outputs_created"]
        )
        self.assertFalse(
            self.manifest["generic_action"][
                "appearance_fields_authorized_in_representation"
            ]
        )
        self.assertFalse(
            self.manifest["generic_action"][
                "source_instance_identity_fields_authorized_in_representation"
            ]
        )
        self.assertEqual(
            self.manifest["geometry_source"]["native_t2v_conditioning_count"], 0
        )

    def test_three_appearance_disjoint_variants_share_the_generic_action(self) -> None:
        variants = self.manifest["variants"]
        self.assertEqual([row["variant_id"] for row in variants], ["v1", "v2", "v3"])
        self.assertEqual(
            [row["seed"] for row in variants],
            [2026082101, 2026082102, 2026082103],
        )
        self.assertEqual(len({row["appearance_family"] for row in variants}), 3)
        for row in variants:
            prompt = row["prompt"]
            self.assertEqual(
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                row["prompt_sha256"],
            )
            self.assertIn("turns his head to the left", prompt.lower()) if row[
                "variant_id"
            ] != "v2" else self.assertIn("turns her head to the left", prompt.lower())
            self.assertIn("places the same", prompt.lower())
            self.assertIn("continuous static-camera shot", prompt.lower())
        all_prompts = " ".join(row["prompt"].lower() for row in variants)
        for source_specific in (
            "mint green",
            "reddish-blonde",
            "blue and white striped pillars",
            "modern gym",
            "treadmill",
        ):
            self.assertNotIn(source_specific, all_prompts)

    def test_node_map_is_three_independent_sp4_workers(self) -> None:
        self.assertEqual(self.manifest["slurm"]["job_id"], "143808")
        self.assertEqual(self.manifest["slurm"]["gpus_per_variant"], 4)
        self.assertEqual(
            self.manifest["slurm"]["node_variant_map"],
            {
                "v1": "auh7-1b-gpu-268",
                "v2": "auh7-1b-gpu-315",
                "v3": "auh7-1b-gpu-233",
            },
        )
        for fragment in (
            '[[ "${SLURM_GPUS_ON_NODE:-}" == "4" ]]',
            '[[ "${ROCR_VISIBLE_DEVICES:-}" == "0,1,2,3" ]]',
            "--nproc_per_node=4",
            "--arms t2v",
            "--num-inference-steps 40",
        ):
            self.assertIn(fragment, self.launcher)
        self.assertNotIn("--nproc_per_node=8", self.launcher)

    def test_native_invocation_has_no_target_route_or_training_inputs(self) -> None:
        command = self.launcher.split(
            '"${python_bin}" -B -m torch.distributed.run', 1
        )[1].split('"${python_bin}" -B - \\\n+', 1)[0]
        for required in (
            '--source-video "${source_video}"',
            "--expected-source-sha256 a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646",
            '--action-prompt "${prompt}"',
            '--expected-action-prompt-sha256 "${prompt_sha256}"',
            "--arms t2v",
        ):
            self.assertIn(required, command)
        for forbidden in (
            "--target",
            "--anchor-video",
            "--reference",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--route",
            "--lora",
            "--train",
            "--optimizer",
        ):
            self.assertNotIn(forbidden, command)

    def test_postflight_seals_exact81_and_zero_update_claim(self) -> None:
        for fragment in (
            'native["conditioning"]["t2v"]["full_source_video_count"] == 0',
            'native["conditioning"]["t2v"]["source_derived_reference_count"] == 0',
            'native["freeze_certificate"]["trainable_parameter_elements"] == 0',
            'native["interpretation"]["training_performed"] is False',
            'native["outputs"]["t2v"]["frame_count"] == 81',
            'native["outputs"]["t2v"]["fps"] == 25',
            'frame_count == 81 and fps == Fraction(25, 1)',
            '"real_target_read": False',
            '"REPRESENTATION_ONLY_ANCHOR_COMPLETE"',
        ):
            self.assertIn(fragment, self.launcher)

    def test_worker_neither_submits_nor_mutates_repository(self) -> None:
        self.assertNotRegex(self.launcher, r"(?m)^\s*(?:srun|sbatch|scancel)(?:\s|$)")
        self.assertNotRegex(
            self.launcher,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|restore|switch)\b",
        )


if __name__ == "__main__":
    unittest.main()
