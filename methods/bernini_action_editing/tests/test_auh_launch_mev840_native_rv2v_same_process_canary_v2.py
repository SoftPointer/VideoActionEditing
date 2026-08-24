from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "assets" / "mev840_native_rv2v_same_process_prompt_matrix_v2.json"
LAUNCHER = ROOT / "scripts" / "auh_launch_mev840_native_rv2v_same_process_canary_v2.sh"


class SameProcessMechanicalCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority_bytes = AUTHORITY.read_bytes()
        cls.authority = json.loads(cls.authority_bytes.decode("ascii"))
        cls.script = LAUNCHER.read_text(encoding="ascii")

    def test_authority_exact_prompts_and_execution_order(self) -> None:
        self.assertEqual(
            self.authority["schema"],
            "mev840-native-rv2v-same-process-prompt-matrix-v2",
        )
        pairing = self.authority["same_process_pairing"]
        self.assertEqual(pairing["execution_order"], ["p0a", "p1", "p2", "p0b"])
        self.assertEqual(
            pairing["prompt_label_by_execution_cell"],
            {"p0a": "P0", "p1": "P1", "p2": "P2", "p0b": "P0"},
        )
        for key in (
            "same_scheduler_object_all_calls",
            "no_manual_model_or_scheduler_state_reset_between_calls",
            "rope_unregistered_state_observed_not_mutated",
        ):
            self.assertIs(pairing[key], True)
        for key in (
            "positive_tokens_and_embedding_world4_exact_per_cell",
            "negative_tokens_and_embedding_world4_exact_across_cells",
            "scheduler_effective_reset_fields_exact_across_calls",
            "scheduler_stale_timestep_list_recorded",
            "scheduler_stale_timestep_list_inactive_before_order2_overwrite_proved",
        ):
            self.assertIs(self.authority["dynamic_observer_gates"][key], True)
        self.assertEqual(
            self.authority["runtime_authority"],
            {
                "unipc_source": {
                    "path": "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/diffusers/schedulers/scheduling_unipc_multistep.py",
                    "sha256": "5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872",
                },
                "mechanical_slurm": {
                    "job_id": "147873",
                    "node": "auh7-1b-gpu-284",
                    "world_size": 4,
                },
                "nearest_finite_cgroup_limit_bytes": 64 * 1024**3,
            },
        )
        for key in (
            "unipc_source_plain_file_sha256_exact",
            "rope_exact_module_set_single_expert",
            "rope_initial_values_unchanged_allowing_cpu_to_p0a_device_transition",
            "rope_p1_p2_p0b_full_state_exact_to_p0a",
            "slurm_job_node_step_world4_bound_in_receipt",
            "cgroup_ancestor_chain_and_nearest_finite_limit_recorded",
            "cgroup_oom_oom_kill_oom_group_kill_delta_zero",
            "terminal_sacct_completed_exit0_maxrss_required",
        ):
            self.assertIs(self.authority["dynamic_observer_gates"][key], True)
        self.assertNotIn(
            "scheduler_schedule_and_reset_state_exact_across_calls",
            self.authority["dynamic_observer_gates"],
        )
        pins = {
            "P0": (370, "effdf094385a4f2486391efc008150b7436a8137c1d5766864a678ed6e0c749f", 162, "604bff69e9f43990de2efd7c26e64d15b4f1e92d9c165d182c7e2707f9299251"),
            "P1": (677, "248410295a0dd4226b478bedaa46cd23f0dd4d406d4d262c692c4006f4481aef", 231, "ccd07d417c3a11ee698b11a07922a55a9f8c32d5bb40d69fa3d2541b4c7e0e0b"),
            "P2": (835, "63d4cda9cedca68487cdd9c5c951c2fe63226483d8975487114c221e38d1b4e5", 276, "79293cc4c429e4b49734221d86800cc906726535901da2c0e5cc4dce648fbc11"),
        }
        for label, (byte_count, raw_sha, token_count, task_sha) in pins.items():
            row = self.authority["prompts"][label]
            raw = row["full_prompt_utf8"].encode("utf-8")
            self.assertEqual(len(raw), byte_count)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), raw_sha)
            self.assertEqual(row["full_prompt_utf8_sha256"], raw_sha)
            self.assertEqual(row["untruncated_token_count"], token_count)
            self.assertEqual(row["final_task_prompt_utf8_sha256"], task_sha)
            self.assertEqual(row["terminal_token_id"], 1)

    def test_mechanical_mode_is_two_step_latent_only_and_cannot_launch_formal(self) -> None:
        mode = self.authority["execution_modes"]["mechanical_canary"]
        self.assertEqual(mode["seed"], 2028)
        self.assertEqual(mode["job_id"], "147873")
        self.assertEqual(mode["node"], "auh7-1b-gpu-284")
        self.assertEqual(mode["num_inference_steps"], 2)
        self.assertIs(mode["decode_mp4"], False)
        self.assertIs(mode["scientific_candidate"], False)
        self.assertIs(mode["formal_launch_authorized_by_canary"], False)
        self.assertIn("launch-canary|worker-canary|postflight-canary", self.script)
        self.assertNotIn("launch-formal)", self.script)
        self.assertIn("--num-inference-steps 2 --seed 2028 --skip-video-decode", self.script)

    def test_launcher_is_syntax_valid_exact_scratch_and_fail_closed_until_runner_pin(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        authority_sha = hashlib.sha256(self.authority_bytes).hexdigest()
        self.assertIn(f"readonly authority_sha={authority_sha}", self.script)
        for digest in (
            "46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115",
            "e104031526236f16e94a4753c31ad8048b1a65345b1913212c35e421fcad48ae",
            "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42",
            "5c28e672bcdd86da3c7d3a94ba9e07b644421cea6c5945fb163fa7b871c2af0a",
            "21a23222ef69781850a8d3a8735713274d07f53d7cd41eae9de41303067c65a3",
            "5eaad43a5be4d21fdffeb802162adeaef702187562e8de5a58e600de5c2840aa",
            "5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872",
        ):
            self.assertIn(digest, self.script)
        self.assertIn("extracted exact19 member set differs", self.script)
        self.assertIn("authorized exact20 closure differs", self.script)
        self.assertIn("control-root exact member closure differs", self.script)
        self.assertIn("cfile=sys.argv[2]", self.script)
        self.assertIn('MIOPEN_USER_DB_PATH="$scratch/cache/miopen-user"', self.script)
        match = re.search(r"^readonly paired_runner_sha=(\S+)$", self.script, re.MULTILINE)
        self.assertIsNotNone(match)
        value = match.group(1)
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            self.assertEqual(value, "UNBOUND_INDEPENDENT_REVIEW_REQUIRED")
            self.assertIn('fail "paired runner is not independently frozen"', self.script)
        else:
            self.assertEqual(
                value,
                "21a23222ef69781850a8d3a8735713274d07f53d7cd41eae9de41303067c65a3",
            )

    def test_generator_input_allowlist_excludes_target_and_legacy_controls(self) -> None:
        generator = self.authority["generator_contract"]
        self.assertEqual(
            generator["accepted_external_conditions"],
            ["source_video", "positive_prompt_matrix"],
        )
        for key in (
            "target_video_read",
            "target_action_json_read",
            "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian_read",
            "anchor_rgb_kv_latent_gaussian_read",
            "legacy_activity25_qk_read",
        ):
            self.assertIs(generator[key], False)
        self.assertNotIn("target_action_oracle", self.script)
        self.assertNotIn("activity25.pt", self.script.lower())
        self.assertNotIn("anchor_qk_transport.py", self.script.lower())
        self.assertNotIn("--action-prompt", self.script)
        self.assertNotIn("--arms", self.script)


if __name__ == "__main__":
    unittest.main()
