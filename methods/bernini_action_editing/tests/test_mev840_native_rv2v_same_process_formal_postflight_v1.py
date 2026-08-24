from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock
import copy

from methods.bernini_action_editing.tools import (
    audit_mev840_native_rv2v_same_process_formal_v1 as auditor,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "assets" / "mev840_native_rv2v_same_process_formal_v1.json"
RUNNER = ROOT / "infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1.py"
LAUNCHER = ROOT / "scripts" / "auh_launch_mev840_native_rv2v_same_process_formal_v1.sh"
AUDITOR = ROOT / "tools" / "audit_mev840_native_rv2v_same_process_formal_v1.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(*, current: int, baseline: dict | None = None) -> dict:
    path = "/slurm/job/step"
    events = {"oom": 3, "oom_kill": 1, "oom_group_kill": 0}
    value = {
        "leaf_relative_path": path,
        "ancestors_through_nearest_finite": [
            {
                "relative_path": path,
                "memory_current": current,
                "memory_max": auditor.CGROUP_LIMIT_BYTES,
                "memory_events": events,
            }
        ],
        "nearest_finite_relative_path": path,
        "nearest_finite_limit_bytes": auditor.CGROUP_LIMIT_BYTES,
        "nearest_finite_current_bytes": current,
        "headroom_bytes": auditor.CGROUP_LIMIT_BYTES - current,
        "minimum_required_headroom_bytes": auditor.CGROUP_MIN_HEADROOM_BYTES,
        "headroom_gate_passed": True,
        "effective_64_gib_limit": True,
        "oom_event_baseline_by_path": None,
        "oom_event_delta_by_path": None,
        "oom_oom_kill_oom_group_kill_delta_zero": None,
    }
    if baseline is not None:
        value["oom_event_baseline_by_path"] = {path: events}
        value["oom_event_delta_by_path"] = {
            path: {"oom": 0, "oom_kill": 0, "oom_group_kill": 0}
        }
        value["oom_oom_kill_oom_group_kill_delta_zero"] = True
    return value


class FormalPostflightTests(unittest.TestCase):
    def test_frozen_authority_runner_and_exact13_contract(self) -> None:
        value = json.loads(AUTHORITY.read_text(encoding="ascii"))
        self.assertEqual(value["schema"], "mev840-native-rv2v-same-process-formal-v1")
        self.assertEqual(value["execution_mode"]["num_inference_steps"], 40)
        self.assertEqual(value["execution_mode"]["exact_regular_file_count_per_seed"], 13)
        self.assertEqual(value["execution_mode"]["decode_cells"], ["p0a", "p1", "p2"])
        self.assertEqual(value["execution_mode"]["latent_only_replay_cells"], ["p0b"])
        self.assertEqual(_sha(AUTHORITY), auditor.AUTHORITY_SHA)
        self.assertEqual(_sha(RUNNER), auditor.RUNNER_SHA)

    def test_launcher_is_per_seed_formal_and_pinned(self) -> None:
        completed = subprocess.run(["bash", "-n", str(LAUNCHER)], check=False)
        self.assertEqual(completed.returncode, 0)
        text = LAUNCHER.read_text(encoding="ascii")
        self.assertIn("launch-formal SEED|worker-formal SEED|postflight-formal SEED", text)
        self.assertIn("2027) echo \"143808 auh7-1b-gpu-292\"", text)
        self.assertIn("2028) echo \"147873 auh7-1b-gpu-284\"", text)
        self.assertIn("--num-inference-steps 40", text)
        self.assertNotIn("--skip-video-decode", text)
        self.assertNotIn("UNBOUND_FORMAL", text)
        self.assertEqual(auditor.launcher_pin(text, "paired_runner_sha"), auditor.RUNNER_SHA)
        self.assertEqual(auditor.launcher_pin(text, "authority_sha"), auditor.AUTHORITY_SHA)
        self.assertEqual(auditor.launcher_pin(text, "postflight_sha"), _sha(AUDITOR))
        self.assertEqual(text.count("printf '%s\\n' \"$!\" >\"$pid_file\""), 1)
        self.assertEqual(text.count(': >"$pid_file"'), 1)
        self.assertEqual(text.count(': >"$log"'), 1)
        self.assertIn("require_seed2027_terminal_postflight_before_seed2028", text)
        self.assertIn("seed=2027 step=143808\\.[0-9]+", text)
        self.assertIn('--output-dir "${output_root}/seed2027" --seed 2027', text)

    def test_auditor_has_no_mechanical_execution_assumptions(self) -> None:
        text = AUDITOR.read_text(encoding="utf-8")
        for stale in (
            '"num_inference_steps": 2',
            '"mechanical_canary": True',
            "exact10",
            "MECHANICAL_JOB_ID",
            "MECHANICAL_NODE",
            "same-process-mechanical-postflight",
        ):
            self.assertNotIn(stale, text)
        args = auditor.build_parser().parse_args(
            [
                "--output-dir", "/tmp/output",
                "--seed", "2028",
                "--authority", "/tmp/authority",
                "--launcher", "/tmp/launcher",
                "--runner", "/tmp/runner",
            ]
        )
        self.assertEqual(args.seed, 2028)

    def test_cgroup_headroom_and_zero_delta_gate(self) -> None:
        baseline = _snapshot(current=1024)
        current = _snapshot(current=2048, baseline=baseline)
        auditor._validate_cgroup_snapshot(baseline, label="baseline", baseline=None)
        auditor._validate_cgroup_snapshot(current, label="current", baseline=baseline)
        bad = dict(current)
        bad["headroom_bytes"] = auditor.CGROUP_MIN_HEADROOM_BYTES - 1
        with self.assertRaisesRegex(auditor.AuditError, "cgroup hierarchy/limit"):
            auditor._validate_cgroup_snapshot(bad, label="bad", baseline=baseline)

        mismatched_summary = copy.deepcopy(current)
        mismatched_summary["nearest_finite_current_bytes"] += 1
        mismatched_summary["headroom_bytes"] -= 1
        with self.assertRaisesRegex(auditor.AuditError, "nearest finite cgroup"):
            auditor._validate_cgroup_snapshot(
                mismatched_summary, label="summary-mismatch", baseline=baseline
            )

        hidden_oom = copy.deepcopy(current)
        hidden_oom["ancestors_through_nearest_finite"][0]["memory_events"]["oom"] += 1
        with self.assertRaisesRegex(auditor.AuditError, "OOM event delta"):
            auditor._validate_cgroup_snapshot(
                hidden_oom, label="hidden-oom", baseline=baseline
            )

    def test_resource_lifecycle_rejects_barrier_setup_overlap(self) -> None:
        value = {
            "schema_version": "bernini-native-t2v-resource-lifecycle-v4",
            "serialized_host_checkpoint_load_required": True,
            "renderer_deserialized_and_moved_to_rank_gpu_under_lock": True,
            "host_allocator_trim_called_before_load_lock_release": True,
            "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup": True,
            "world4_load_completion_receipt_before_native_sampling": True,
            "renderer_retired_before_rank_zero_vae_load": True,
            "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
            "t2v_vae_weights_loaded_before_sampling": False,
            "t2v_vae_decode_rank": 0,
            "sampling_model_and_vae_not_host_resident_concurrently_for_t2v": True,
            "t2v_text_encoder_gpu_residency_required": False,
            "t2v_text_encoder_cpu_offload_bypass_active": False,
            "t2v_text_encoder_retired_only_with_renderer": False,
            "world4_t2v_text_encoder_gpu_residency_gate": None,
            "renderer_scheduler_and_rope_aliases_retired_before_rank_zero_decode_vae_load": True,
            "source_conditions_and_noise_captures_retired_before_rank_zero_decode_vae_load": True,
            "world4_predecode_retirement_barrier_completed": True,
            "rank_zero_decode_vae_loaded_only_after_renderer_retirement": True,
            "rank_zero_decode_vae_cpu_materialization_count_after_decode": 0,
            "rank_zero_decode_vae_retired_before_final_memory_gate": True,
            "all_rank_conditions_and_held_latents_retired_before_final_memory_gate": True,
            "world4_post_decode_retirement_barrier_completed": True,
            "world4_load_completion_gate": {
                "schema_version": "bernini-native-world4-renderer-load-completion-gate-v1",
                "world_size": 4,
                "hostname": "auh7-1b-gpu-292",
                "ranks": [0, 1, 2, 3],
                "renderer_gpu_resident_trimmed_monotonic_ns_by_rank": [1, 2, 3, 4],
                "load_completion_barrier_returned_monotonic_ns_by_rank": [10, 11, 12, 13],
                "source_tokenizer_setup_entered_monotonic_ns_by_rank": [20, 21, 22, 23],
                "native_sampling_entered_monotonic_ns_by_rank": [30, 31, 32, 33],
                "world4_barrier_completed_before_source_tokenizer_setup": True,
                "all_four_renderer_loads_complete_before_any_source_tokenizer_setup": True,
                "all_four_renderer_loads_complete_before_first_native_sampling": True,
            },
        }
        auditor._validate_resource_lifecycle(value, expected_node="auh7-1b-gpu-292")
        overlap = copy.deepcopy(value)
        overlap["world4_load_completion_gate"][
            "load_completion_barrier_returned_monotonic_ns_by_rank"
        ] = [18, 19, 20, 21]
        with self.assertRaisesRegex(auditor.AuditError, "lifecycle differs"):
            auditor._validate_resource_lifecycle(
                overlap, expected_node="auh7-1b-gpu-292"
            )

    def test_terminal_sacct_is_seed_node_and_headroom_bound(self) -> None:
        row = "143808.42|COMPLETED|0:0|65000000K|auh7-1b-gpu-292\n"
        completed = subprocess.CompletedProcess([], 0, stdout=row, stderr="")
        with mock.patch.object(auditor.subprocess, "run", return_value=completed):
            result = auditor.terminal_sacct_evidence(
                "143808.42",
                expected_job_id="143808",
                expected_node="auh7-1b-gpu-292",
            )
        self.assertTrue(result["accounting_terminal"])
        self.assertEqual(result["node"], "auh7-1b-gpu-292")


if __name__ == "__main__":
    unittest.main()
