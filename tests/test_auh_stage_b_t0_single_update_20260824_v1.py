#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import unittest


REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "md" / "action_editing" / "20260824_reward"
TEMPLATE = DOC / "stage_b_t0_single_update_retry5_authority_addendum.template.json"
ACTIVE = DOC / "stage_b_t0_single_update_retry5_authority_addendum.json"
LAUNCHER = (
    REPO
    / "methods"
    / "bernini_action_editing"
    / "scripts"
    / "auh_stage_b_t0_single_update_20260824_retry5.sh"
)


class StageBT0SingleUpdateStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        cls.script = LAUNCHER.read_text(encoding="utf-8")

    def test_template_is_explicitly_inert_until_final_hash_lock(self) -> None:
        authority = self.authority
        self.assertEqual(
            authority["schema_version"],
            "bernini-action-repr-stage-b-t0-single-update-authority-addendum-v1",
        )
        self.assertEqual(
            authority["activation"],
            {
                "state": "DRAFT_HASH_PLACEHOLDERS_NOT_AUTHORIZED",
                "create_once": True,
                "all_placeholders_replaced": False,
                "activation_rule": (
                    "copy_to_stage_b_t0_single_update_authority_addendum.json_"
                    "once_only_after_runner_tests_and_launcher_are_final_then_"
                    "replace_every_explicit_sha256_placeholder_and_set_state_"
                    "ACTIVE_CREATE_ONCE_AUTHORITY"
                ),
                "template_itself_authorizes_optimizer_creation": False,
            },
        )
        pins = authority["source_hash_pins"]
        self.assertEqual(
            set(pins),
            {
                "methods/bernini_action_editing/train_action_repr_target_t0_canary_v1.py",
                "methods/bernini_action_editing/action_representation_joint_objective_v1.py",
                "methods/bernini_action_editing/action_repr_g2a_adapter_v1.py",
                "methods/bernini_action_editing/audit_action_repr_g2a_world4_v1.py",
                "methods/bernini_action_editing/materialize_decoded_middle_action_repr_v1.py",
                "methods/bernini_action_editing/dense_flow_token_adapter_v1.py",
                "methods/bernini_action_editing/exact_local_video_materializer_v1.py",
                "methods/bernini_action_editing/train_lora.py",
                "methods/bernini_action_editing/train_self_generated_action_quotient_v1.py",
                "methods/bernini_action_editing/scripts/auh_stage_b_t0_single_update_20260824_retry5.sh",
                "methods/bernini_action_editing/tests/test_train_action_repr_target_t0_canary_v1.py",
                "tests/test_auh_stage_b_t0_single_update_20260824_v1.py",
            },
        )
        placeholders = [value for value in pins.values() if value.startswith("__FINAL_")]
        self.assertEqual(len(placeholders), 6)
        self.assertTrue(all(value.endswith("_SHA256__") for value in placeholders))
        self.assertEqual(
            pins["methods/bernini_action_editing/audit_action_repr_g2a_world4_v1.py"],
            "8f6f13e76bcba0defd9af7576912eb55eddd76c1cc74ba334e1b11d5e0dd359d",
        )
        self.assertEqual(
            pins[
                "methods/bernini_action_editing/"
                "materialize_decoded_middle_action_repr_v1.py"
            ],
            "f3fa0138ffcff997a604567c0951bf7f9aba74ae6cb66acb943eddef2aa6a1ac",
        )

    def test_active_retry5_authority_is_create_once_and_source_pinned(self) -> None:
        active = json.loads(ACTIVE.read_text(encoding="utf-8"))
        self.assertEqual(
            active["document_role"],
            "create_once_stage_b_target_t0_single_update_authority",
        )
        self.assertEqual(active["activation"]["state"], "ACTIVE_CREATE_ONCE_AUTHORITY")
        self.assertTrue(active["activation"]["create_once"])
        self.assertTrue(active["activation"]["all_placeholders_replaced"])
        self.assertNotIn("__FINAL_", ACTIVE.read_text(encoding="utf-8"))
        self.assertEqual(
            active["runtime_paths"]["fresh_source_root_name"],
            "source_stage_b_t0_retry5",
        )
        self.assertEqual(
            active["runtime_paths"]["fresh_stage_root_name"],
            "stage_b_t0_retry5",
        )
        self_relative = "tests/test_auh_stage_b_t0_single_update_20260824_v1.py"
        for relative, expected in active["source_hash_pins"].items():
            self.assertRegex(expected, r"^[0-9a-f]{64}$")
            if relative == self_relative:
                continue
            observed = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)

    def test_upstream_gates_and_fixed_target_only_scope(self) -> None:
        upstream = self.authority["upstream_gate_evidence"]
        self.assertEqual(
            upstream["manifest"]["sha256"],
            "c78e42f0661e5905407505037ce322d32d67ffec0b70b1cab466f895dc8d0632",
        )
        self.assertEqual(
            upstream["g1_target"]["receipt_sha256"],
            "974471a39e8c904cc5234b244d2e59fbf6540d5e951cc2c285454486c8f8066e",
        )
        self.assertEqual(upstream["g1_target"]["required_status"], "passed")
        self.assertEqual(
            upstream["production_g2a"]["receipt_sha256"],
            "7ea0ab20709d942ca51a3062f2306407be8f9d0f4445926dca57af9b83fc3f09",
        )
        self.assertEqual(
            upstream["production_g2a"]["receipt_digest"],
            "a39b1be65887532a24378fda25f517bb7b8edb60831e91aaf55468b70f2802b7",
        )
        scope = self.authority["canary_scope"]
        self.assertEqual(scope["arm"], "T0")
        self.assertEqual(scope["case_id"], "0be6494dfac3")
        self.assertEqual(scope["case_split"], "fit")
        self.assertEqual(scope["world_size"], 4)
        self.assertEqual(scope["sequence_parallel_size"], 4)
        self.assertEqual(
            self.authority["distributed_contract"]["runtime_backend"], "nccl/rccl"
        )
        distributed = self.authority["distributed_contract"]
        self.assertTrue(distributed["slurm_rocr_visible_devices_mapping_preserved"])
        self.assertTrue(distributed["slurm_rocr_visible_devices_mapping_receipted"])
        self.assertEqual(
            distributed["slurm_rocr_visible_devices_exact_device_count"], 4
        )
        self.assertEqual(
            distributed["slurm_rocr_visible_devices_physical_range_inclusive"],
            [0, 7],
        )
        self.assertTrue(distributed["hip_visible_devices_must_be_empty"])
        self.assertTrue(distributed["cuda_visible_devices_must_be_empty"])
        self.assertEqual(scope["optimization_steps"], 1)
        self.assertTrue(scope["parameter_updates_required"])
        self.assertTrue(scope["target_representation"])
        for forbidden in (
            "TP",
            "sourcecopy",
            "selfgen",
            "graph",
            "automatic_expansion",
            "longer_training_authorized",
            "decode_authorized_by_this_addendum",
        ):
            self.assertFalse(scope[forbidden], forbidden)

    def test_projection_optimizer_gradient_and_firewall_contracts(self) -> None:
        representation = self.authority["representation_contract"]
        self.assertFalse(representation["target_rgb_allowed_in_trainer"])
        self.assertFalse(representation["target_video_cli_argument_allowed"])
        self.assertFalse(representation["target_vae_or_clean_latent_allowed_in_trainer"])
        self.assertFalse(representation["absolute_target_hidden_or_qkv_allowed_in_trainer"])
        self.assertEqual(
            representation["fixed_jl"],
            {
                "kind": "case_independent_fixed_rademacher_jl",
                "seed": 2026082401,
                "input_width": 1536,
                "output_width": 256,
                "tensor_sha256": (
                    "6291f3e65908fc8500b7529873f1165011b8bd61916b19d82d696f2485a01dbe"
                ),
                "fitted_on_input_video": False,
                "applied_differentiably_to_student_trace": True,
                "teacher_cache_remains_detached": True,
            },
        )
        self.assertEqual(
            representation["phase_activity"],
            {
                "phase0_active": False,
                "onset_active": True,
                "terminal_active": True,
                "phase0_may_be_relabelled_active": False,
            },
        )
        gradients = self.authority["counterfactual_gradient_contract"]
        self.assertEqual(
            gradients["required_controls_in_order"],
            ["zero", "temporal_shuffle", "reverse", "incomplete", "wrong_action"],
        )
        self.assertTrue(gradients["no_grad_hinge_prepass"])
        self.assertEqual(gradients["correct_side_gradient_passes"], 1)
        self.assertEqual(gradients["separate_control_gradient_passes"], 5)
        self.assertFalse(
            gradients["detached_control_scores_without_control_side_gradient_are_sufficient"]
        )
        optimizer = self.authority["optimizer_contract"]
        self.assertEqual(optimizer["kind"], "AdamW")
        self.assertEqual(optimizer["learning_rate"], 1.0e-4)
        self.assertEqual(optimizer["weight_decay"], 0.0)
        self.assertEqual(optimizer["steps_exact"], 1)
        self.assertTrue(optimizer["second_step_forbidden"])
        self.assertTrue(optimizer["resume_forbidden"])
        firewall = self.authority["parameter_firewall"]
        self.assertTrue(firewall["base_generator_frozen"])
        self.assertTrue(firewall["vae_frozen"])
        self.assertTrue(firewall["text_encoder_frozen"])
        self.assertFalse(firewall["lora_enabled"])
        self.assertEqual(
            firewall["trainable_roles_exact"], ["motion_adapter", "middle_projector"]
        )

    def test_launcher_parses_and_rejects_draft_or_unpinned_authority(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("STAGE_B_T0_AUTHORITY_SHA256", self.script)
        self.assertIn("require_authority_hash", self.script)
        self.assertIn("ACTIVE_CREATE_ONCE_AUTHORITY", self.script)
        self.assertIn("draft/revoked Stage-B authority", self.script)
        self.assertIn("unresolved source placeholder", self.script)
        self.assertIn("grep -Eq '__FINAL_[A-Z0-9_]+_SHA256__'", self.script)
        self.assertIn('source_root="$experiment_root/source_stage_b_t0_retry5"', self.script)
        self.assertIn('stage_root="$experiment_root/stage_b_t0_retry5"', self.script)
        self.assertIn('test ! -e "$output_root"', self.script)
        self.assertIn("O_EXCL", (REPO / "methods" / "bernini_action_editing" / "train_action_repr_target_t0_canary_v1.py").read_text(encoding="utf-8"))

    def test_launcher_has_one_bounded_world4_runner_surface(self) -> None:
        labels = set(
            re.findall(r"^  ([a-z][a-z0-9-]*)\)\n", self.script, flags=re.MULTILINE)
        )
        self.assertEqual(labels, {"preflight", "launch", "worker", "status"})
        self.assertNotRegex(self.script, r"\b(?:sbatch|scancel|squeue)\b")
        self.assertIn('srun --jobid="$job" --exclusive --exact', self.script)
        self.assertIn("--gres=gpu:mi210:4", self.script)
        self.assertIn("--mem=0", self.script)
        self.assertIn("--nproc_per_node=4", self.script)
        self.assertIn("/usr/bin/timeout --signal=TERM --kill-after=60s 45m", self.script)
        self.assertIn("TORCH_NCCL_ASYNC_ERROR_HANDLING=1", self.script)
        self.assertNotRegex(self.script, r"(?m)^\s*export\s+ROCR_VISIBLE_DEVICES=")
        self.assertNotRegex(
            self.script,
            r"(?m)^\s*unset\s+.*(?:HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES)",
        )
        self.assertIn('slurm_rocr="${ROCR_VISIBLE_DEVICES-}"', self.script)
        self.assertIn(
            'export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES="$slurm_rocr"',
            self.script,
        )
        self.assertIn('test -z "${HIP_VISIBLE_DEVICES-}"', self.script)
        self.assertIn('test -z "${CUDA_VISIBLE_DEVICES-}"', self.script)
        self.assertIn("UserId=guangyi.chen(2012)", self.script)
        self.assertIn("Account=faculty-acc", self.script)
        self.assertIn("QOS=bgqos", self.script)
        self.assertIn("Partition=faculty", self.script)
        self.assertIn('validate_parent_gpu_tres "$alloc_tres"', self.script)
        self.assertIn('[[ "$generic_count" =~ ^(4|8)$ ]]', self.script)
        self.assertIn(
            'visible_count="$($python_bin -c \'import torch; print(torch.cuda.device_count())\')"',
            self.script,
        )
        self.assertNotIn("147871", self.script)
        self.assertNotIn("147873", self.script)
        self.assertNotIn("147881", self.script)
        invocation = re.search(
            r'"\$runner" \\\n(?P<body>.*?)\n      --output "\$output_root"',
            self.script,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(invocation)
        body = invocation.group("body")
        expected_flags = {
            "--authorization-addendum",
            "--manifest",
            "--g1-admission-receipt",
            "--g2a-receipt",
            "--bernini-root",
            "--veomni-root",
            "--checkpoint",
        }
        self.assertEqual(set(re.findall(r"--[a-z0-9-]+", body)), expected_flags)
        for forbidden in (
            "--case-id",
            "--target-video",
            "--target-latent",
            "--sourcecopy",
            "--selfgen",
            "--graph",
            "--tp",
            "--steps",
            "--learning-rate",
            "--resume",
        ):
            self.assertNotIn(forbidden, self.script)

    def test_launcher_replays_result_contract_and_stops_before_decode(self) -> None:
        self.assertIn("validate_published_t0_output", self.script)
        self.assertIn(".optimization_steps", self.script)
        self.assertIn(".parameter_updates > 0", self.script)
        self.assertIn(".training.all_five_control_gradient_passes_executed", self.script)
        self.assertIn(".training.renderer_base_identity_versions_bytes_unchanged", self.script)
        self.assertIn("step0000/adapter_model.safetensors", self.script)
        self.assertIn("step0001/adapter_model.safetensors", self.script)
        self.assertNotRegex(self.script, r"\b(?:ffmpeg|review_web|index\.html)\b")
        self.assertNotIn("decode)", self.script)
        self.assertEqual(
            self.authority["claim_boundary"],
            "optimizer_integration_canary_only_not_ours_not_quality_not_decoded_video",
        )
        self.assertFalse(self.authority["next_order_boundary"]["longer_T0_before_matched_decode"])
        self.assertFalse(self.authority["next_order_boundary"]["TP_authorized"])

    def _run_step_token_contract(self, value: str | None) -> subprocess.CompletedProcess:
        match = re.search(
            r"(?ms)^normalize_step_token\(\) \{\n.*?^\}\n",
            self.script,
        )
        self.assertIsNotNone(match)
        snippet = (
            "set -u\n"
            "fail() { echo \"TOKEN_ERROR:$*\" >&2; exit 19; }\n"
            f"{match.group(0)}\n"
            "normalize_step_token\n"
        )
        environment = os.environ.copy()
        environment["SLURM_JOB_ID"] = "151642"
        if value is None:
            environment.pop("SLURM_STEP_ID", None)
        else:
            environment["SLURM_STEP_ID"] = value
        return subprocess.run(
            ["bash", "-c", snippet],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_direct_sbatch_without_step_id_uses_strict_batch_token(self) -> None:
        completed = self._run_step_token_contract(None)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "batch\n")
        self.assertNotIn("TOKEN_ERROR:", completed.stderr)
        self.assertIn('local raw="${SLURM_STEP_ID:-batch}"', self.script)
        self.assertIn('step_token="$(normalize_step_token)"', self.script)
        self.assertIn('${SLURM_JOB_ID}-${step_token}', self.script)
        self.assertNotIn('${SLURM_JOB_ID}-${SLURM_STEP_ID}', self.script)

    def test_step_token_path_separators_and_nonwhitelist_values_fail_closed(self) -> None:
        for value in (
            "../escape",
            "nested/path",
            "nested\\path",
            "white space",
            "-leading",
            "x" * 65,
        ):
            with self.subTest(value=value):
                completed = self._run_step_token_contract(value)
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, "")
                self.assertIn("TOKEN_ERROR:", completed.stderr)
        accepted = self._run_step_token_contract("123.extern")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(accepted.stdout, "123.extern\n")

    def _run_slurm_rocr_mapping_contract(self, value: str) -> subprocess.CompletedProcess:
        match = re.search(
            r"(?ms)^validate_slurm_rocr_mapping\(\) \{\n.*?^\}\n",
            self.script,
        )
        self.assertIsNotNone(match)
        snippet = (
            "set -u\n"
            "fail() { echo \"ROCR_ERROR:$*\" >&2; exit 29; }\n"
            f"{match.group(0)}\n"
            "validate_slurm_rocr_mapping \"$1\"\n"
        )
        return subprocess.run(
            ["bash", "-c", snippet, "rocr-contract", value],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

    def test_slurm_rocr_mapping_accepts_any_legal_exact_four_device_subset(self) -> None:
        for value in ("0,1,2,3", "4,5,6,7", "0,2,5,7", "1,3,4,6"):
            with self.subTest(value=value):
                completed = self._run_slurm_rocr_mapping_contract(value)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout, f"{value}\n")
                self.assertNotIn("ROCR_ERROR:", completed.stderr)

    def test_slurm_rocr_mapping_rejects_empty_duplicate_override_and_path_values(self) -> None:
        rejected = (
            "",
            "0,1,2",
            "0,1,2,3,4",
            "0,1,2,2",
            "0,1,2,8",
            "0,1,2,-1",
            "0,1,2,a",
            "0,1,2, 3",
            "0,1,2,/3",
            "0,1,2,\\3",
            "0,1,,3",
            "00,1,2,3",
        )
        for value in rejected:
            with self.subTest(value=value):
                completed = self._run_slurm_rocr_mapping_contract(value)
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, "")
                self.assertIn("ROCR_ERROR:", completed.stderr)

    def _run_parent_gpu_contract(self, alloc_tres: str) -> subprocess.CompletedProcess:
        match = re.search(
            r"(?ms)^validate_parent_gpu_tres\(\) \{\n.*?^\}\n",
            self.script,
        )
        self.assertIsNotNone(match)
        snippet = (
            "set -u\n"
            "fail() { echo \"GPU_TRES_ERROR:$*\" >&2; exit 23; }\n"
            f"{match.group(0)}\n"
            "validate_parent_gpu_tres \"$1\"\n"
        )
        return subprocess.run(
            ["bash", "-c", snippet, "gpu-contract", alloc_tres],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

    def test_parent_allocation_accepts_exactly_four_or_eight_mi210_only(self) -> None:
        for count in (4, 8):
            with self.subTest(accepted=count):
                completed = self._run_parent_gpu_contract(
                    f"cpu=16,mem=64G,node=1,gres/gpu={count},"
                    f"gres/gpu:mi210={count}"
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout, f"{count}\n")
                self.assertNotIn("GPU_TRES_ERROR:", completed.stderr)

        rejected = (
            "cpu=16,mem=64G,node=1,gres/gpu=1,gres/gpu:mi210=1",
            "cpu=16,mem=64G,node=1,gres/gpu=2,gres/gpu:mi210=2",
            "cpu=16,mem=64G,node=1,gres/gpu=6,gres/gpu:mi210=6",
            "cpu=16,mem=64G,node=1,gres/gpu=16,gres/gpu:mi210=16",
            "cpu=16,mem=64G,node=1,gres/gpu=8,gres/gpu:mi210=4",
            "cpu=16,mem=64G,node=1,gres/gpu=4,gres/gpu:mi210=8",
            "cpu=16,mem=64G,node=1,gres/gpu=8",
            "cpu=16,mem=64G,node=1,gres/gpu:mi210=8",
            "cpu=16,mem=64G,node=1,gres/gpu=8,gres/gpu=8,gres/gpu:mi210=8",
        )
        for alloc_tres in rejected:
            with self.subTest(rejected=alloc_tres):
                completed = self._run_parent_gpu_contract(alloc_tres)
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, "")
                self.assertIn("GPU_TRES_ERROR:", completed.stderr)


if __name__ == "__main__":
    unittest.main()
