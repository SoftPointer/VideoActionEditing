#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_infer_saic_source_state_flow_transport_all8_v1.sbatch"
)
RUNNER = METHOD_ROOT / "infer_saic_source_state_flow_transport_v1.py"


class SAICSourceStateFlowTransportAll8LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")
        cls.runner_text = RUNNER.read_text(encoding="utf-8")

    def test_shell_and_all_embedded_python_are_syntactically_valid(self) -> None:
        # Use the system interpreter explicitly.  A MacPorts ``bash`` selected
        # through PATH can inherit the already-imported PyTorch/OpenMP runtime
        # and intermittently abort while opening its shared-memory transport.
        subprocess.run(["/bin/bash", "-n", str(LAUNCHER)], check=True)
        snippets = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", self.text, re.DOTALL)
        self.assertEqual(len(snippets), 4)
        for index, source in enumerate(snippets):
            compile(source, f"{LAUNCHER.name}:heredoc-{index}", "exec")

    def test_one_all8_node_is_two_concurrent_world4_sp4_groups(self) -> None:
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertIn("#SBATCH --qos=bgqos", self.text)
        self.assertIn(
            "#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,"
            "auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-289,"
            "auh7-1b-gpu-318",
            self.text,
        )
        wave = self.text[self.text.index("run_wave() {") : self.text.index(
            'echo "[saic-ssft-all8] topology='
        )]
        self.assertIn('run_cell dog "${dog_row}"', wave)
        self.assertIn('"0,1,2,3" "${dog_port}"', wave)
        self.assertIn('run_cell human "${human_row}"', wave)
        self.assertIn('"4,5,6,7" "${human_port}"', wave)
        self.assertEqual(wave.count("2>&1 &"), 2)
        self.assertIn("dog_pid=$!", wave)
        self.assertIn("human_pid=$!", wave)
        self.assertIn('wait "${dog_pid}"', wave)
        self.assertIn('wait "${human_pid}"', wave)
        self.assertEqual(self.text.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.text)
        self.assertIn('ROCR_VISIBLE_DEVICES="${visible}"', self.text)
        self.assertIn(
            "exec env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES "
            "-u GPU_DEVICE_ORDINAL",
            self.text,
        )

    def test_first_pilot_is_exactly_t0_i0_for_fixed_dog_and_human_cells(self) -> None:
        self.assertIn("readonly pilot_arms=(T0 I0)", self.text)
        self.assertNotRegex(self.text, r"readonly pilot_arms=.*(?:T1|IAVG|I1A?)")
        self.assertIn("readonly dog_row=fit-dog-00-7b88a1ca1f804f41", self.text)
        self.assertIn("readonly dog_seed=2026082101", self.text)
        self.assertIn("readonly human_row=fit-human-00-a35b590961d24694", self.text)
        self.assertIn("readonly human_seed=2026082121", self.text)
        self.assertIn('for arm in "${pilot_arms[@]}"; do', self.text)
        self.assertIn('run_wave "${arm}"', self.text)
        self.assertNotRegex(self.text, r"for\s+(?:seed|row_id|candidate)\s+in")
        self.assertIn("--branch forward", self.text)

    def test_each_source_is_materialized_once_on_world1_before_any_arm(self) -> None:
        phase = self.text[
            self.text.index("# Materialize each selected source coordinate exactly once") :
            self.text.index("job_mod=$((SLURM_JOB_ID % 20000))")
        ]
        self.assertIn("materialize_saic_source_clean_latent_v1.py", self.text)
        self.assertIn("test_materialize_saic_source_clean_latent_v1.py", self.text)
        self.assertIn('materialize_source dog "${dog_row}"', phase)
        self.assertIn('"${dog_source_sha256}" 0 "${dog_source_clean}"', phase)
        self.assertIn('materialize_source human "${human_row}"', phase)
        self.assertIn('"${human_source_sha256}" 4 "${human_source_clean}"', phase)
        self.assertEqual(phase.count("2>&1 &"), 2)
        self.assertIn("WORLD_SIZE=1 RANK=0 LOCAL_RANK=0", phase)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1", phase)
        self.assertIn("--device cuda:0", phase)
        self.assertIn("source-only WORLD1 materialization failed", phase)
        self.assertIn("source.clean-latent.safetensors", self.text)
        self.assertIn("refusing source-clean-latent cache reuse", phase)
        self.assertLess(
            self.text.index("source materialization binding order differs"),
            self.text.index("run_wave() {"),
        )

    def test_runner_limitation_and_two_model_loads_are_explicit(self) -> None:
        self.assertIn("runner_multi_arm=false model_loads_per_group=2", self.text)
        self.assertIn('"runner_supports_multi_arm_model_reuse": False', self.text)
        self.assertIn('"model_loads_in_group": 2', self.text)
        self.assertIn('"model_loads_per_group": 2', self.text)
        self.assertIn('model.get("one_arm_per_process_group") is not True', self.text)
        self.assertIn('model.get("model_once_multi_arm_supported") is not False', self.text)
        self.assertIn('model.get("unsupported_model_once") is not True', self.text)
        self.assertIn("SAIC_SSFT_ACK_RUNNER_RELOADS_MODEL_PER_ARM", self.text)
        self.assertNotIn("shared_step_audit_restored", self.text)
        self.assertIn(
            "runner native attempt/success counter closure differs", self.text
        )
        self.assertIn(
            '"native_raw_transformer_forward_attempt_and_success_count"',
            self.text,
        )
        self.assertIn("expected_guided = 80", self.text)
        self.assertIn('expected_raw = 160 if arm == "T0" else 240', self.text)
        self.assertIn(
            'cert.get("core_native_guided_count_reconciled") is not True',
            self.text,
        )
        self.assertIn(
            'cert.get("core_native_raw_forward_count_reconciled") is not True',
            self.text,
        )

    def test_launcher_consumes_the_final_runner_cli_and_receipt_schema(self) -> None:
        for flag in (
            "--bernini-root",
            "--veomni-root",
            "--checkpoint",
            "--checkpoint-content-manifest",
            "--source-manifest",
            "--event-bank",
            "--source-clean-latent",
            "--source-clean-latent-receipt",
            "--expected-source-clean-latent-sha256",
            "--expected-source-clean-latent-receipt-sha256",
            "--expected-source-clean-tensor-raw-sha256",
            "--row-id",
            "--branch",
            "--rollout-seed",
            "--arm",
            "--output",
            "--method-source-revision",
            "--method-source-archive",
            "--durable-method-source-archive",
            "--method-source-archive-sha256",
        ):
            self.assertIn(flag, self.text)
            self.assertIn(flag, self.runner_text)
        for schema_key in (
            "normalized_clean_latent",
            "published_normalized_clean_latent_mode",
            "candidate_zero_noise_sha256",
            "source_manifest_bound_files_verified",
            "source_manifest_terminal_raw_sha256_verified",
            "event_bank_terminal_raw_sha256_verified",
            "selected_source_video_terminal_raw_sha256_verified",
            "native_guided_query_attempt_count",
            "native_guided_query_success_count",
            "native_raw_transformer_forward_attempt_count",
            "native_raw_transformer_forward_success_count",
            "core_native_guided_count_reconciled",
            "core_native_raw_forward_count_reconciled",
            "one_arm_per_process_group",
            "model_once_multi_arm_supported",
            "unsupported_model_once",
            "sealed_source_coordinate",
            "loaded_from_sealed_source_coordinate",
            "encoded_in_runner",
        ):
            self.assertIn(schema_key, self.text)
            self.assertIn(schema_key, self.runner_text)
        self.assertNotIn("SharedStepAudit", self.runner_text)
        self.assertNotIn("shared_step_audit_restored", self.runner_text)

    def test_exact81_25fps_exact40_shift5_are_preflight_and_postflight(self) -> None:
        for token in (
            "runner.FRAME_COUNT != 81",
            "runner.FPS != 25",
            "runner.NUM_INFERENCE_STEPS != 40",
            "runner.FLOW_SHIFT != 5.0",
            'schedule.get("num_frames") != 81',
            'schedule.get("latent_frames") != 21',
            'schedule.get("fps") != 25',
            'schedule.get("num_inference_steps") != 40',
            'schedule.get("flow_shift") != 5.0',
        ):
            self.assertIn(token, self.text)
        self.assertIn("exact81=true fps=25 exact40=true shift=5", self.text)
        self.assertIn("runner T0/I0 condition-bandwidth contract differs", self.text)
        self.assertIn("runner reference-frame0 digest closure differs", self.text)
        for forbidden in (
            "target_video",
            "target_or_oracle_frame",
            "mask_or_swept_tube",
            "pose_flow_track_or_trajectory",
            "motion_donor",
            "external_reference",
        ):
            self.assertIn(f'sealed.get("{forbidden}") is not False', self.text)

    def test_archive_is_revision_bound_scoped_link_free_and_read_only(self) -> None:
        for token in (
            "git get-tar-commit-id",
            "member.issym()",
            "member.islnk()",
            "member.isdev()",
            "member.isfifo()",
            "name in seen",
            "not scoped",
            "archive lacks frozen SSFT runtime closure",
            'find "${task_scratch}/source-tree" -type f -exec chmod a-w',
            'find "${task_scratch}/source-tree" -type d -exec chmod a-w',
            'check_hash "${source_archive}" "${source_archive_sha256}" '
            "source_archive_terminal",
        ):
            self.assertIn(token, self.text)
        self.assertIn(
            '[[ "$(hash_file "${BASH_SOURCE[0]}")" == '
            '"$(hash_file "${launcher}")" ]]',
            self.text,
        )
        self.assertNotRegex(
            self.text,
            re.compile(r"(?m)^\s*git\s+(?:add|commit|push|reset|clean)\b"),
        )

    def test_runtime_closure_tests_and_content_audits_precede_gpu(self) -> None:
        tests = self.text.index("# CPU contract closure must pass")
        source_tree_audit = self.text.index("train_lora.validate_source_trees")
        checkpoint_audit = self.text.index("source_audit.validate_checkpoint_content")
        source_media_audit = self.text.index("fixed source-video bytes changed")
        output = self.text.index('mkdir -- "${output_root}"')
        torchrun = self.text.index('"${python_bin}" -B -m torch.distributed.run')
        self.assertLess(source_tree_audit, checkpoint_audit)
        self.assertLess(checkpoint_audit, source_media_audit)
        self.assertLess(source_media_audit, tests)
        self.assertLess(source_media_audit, output)
        self.assertLess(tests, output)
        self.assertLess(output, torchrun)
        for name in (
            "test_saic_source_state_flow_transport_v1.py",
            "test_saic_native_source_state_field_v1.py",
            "test_materialize_saic_source_clean_latent_v1.py",
            "test_infer_saic_source_state_flow_transport_v1.py",
            "test_auh_infer_saic_source_state_flow_transport_all8_v1.py",
        ):
            self.assertGreaterEqual(self.text.count(name), 2)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", self.text)
        self.assertIn('PYTHONPYCACHEPREFIX="${group_scratch}/pycache"', self.text)
        self.assertIn("private bytecode prefix must remain empty", self.text)
        self.assertIn(
            'PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}"',
            self.text,
        )
        self.assertNotIn("ffprobe_bin", self.text)
        self.assertNotIn("command -v ffprobe", self.text)
        self.assertIn(
            'sealed.get("source_manifest_bound_files_verified") is not False',
            self.text,
        )
        self.assertIn(
            'sealed.get("source_manifest_terminal_raw_sha256_verified") is not True',
            self.text,
        )
        self.assertIn(
            'sealed.get("event_bank_terminal_raw_sha256_verified") is not True',
            self.text,
        )
        self.assertIn(
            'sealed.get("selected_source_video_terminal_raw_sha256_verified") is not True',
            self.text,
        )

    def test_every_external_identity_has_a_pinned_hash_or_revision(self) -> None:
        expected = (
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9",
            "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f",
            "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927",
            "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a",
            "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
            "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed",
        )
        for digest in expected:
            self.assertIn(digest, self.text)
        self.assertIn("source manifest canonical content SHA-256 differs", self.text)
        self.assertIn("event bank canonical content SHA-256 differs", self.text)
        self.assertIn("fixed source-video bytes changed", self.text)
        self.assertIn(
            "runner-bound source bytes changed before parent publication", self.text
        )

    def test_outputs_are_fresh_create_only_and_failure_is_diagnostic(self) -> None:
        self.assertIn("resume and overwrite are forbidden", self.text)
        self.assertIn("output parent must not be filesystem root", self.text)
        self.assertIn('[[ ! -e "${output}" && ! -L "${output}" ]]', self.text)
        self.assertIn('[[ ! -e "${output}.receipt.json"', self.text)
        self.assertIn(
            '[[ ! -e "${output}.normalized-clean-latent.safetensors"',
            self.text,
        )
        self.assertIn("os.O_EXCL", self.text)
        self.assertIn("os.link(temporary, path)", self.text)
        self.assertIn("os.fsync(handle.fileno())", self.text)
        self.assertIn("os.fsync(directory)", self.text)
        self.assertIn("source-clean materialization mode differs before consumer", self.text)
        self.assertIn("dog_source_clean_terminal", self.text)
        self.assertIn("human_source_clean_terminal", self.text)
        self.assertIn("tail -n 240", self.text)
        self.assertIn('if [[ "${published}" != true', self.text)
        self.assertIn("fresh output root device/inode identity differs", self.text)
        self.assertIn("fresh task scratch device/inode identity differs", self.text)
        self.assertIn(
            '[[ "$(stat -c \'%d:%i\' -- "${output_root}")" == '
            '"${output_device_inode}" ]]',
            self.text,
        )
        self.assertIn('rm -rf -- "${output_root}"', self.text)
        self.assertEqual(self.text.rstrip().splitlines()[-1], "published=true")

    def test_parent_reopens_two_group_receipts_and_four_media_latent_pairs(self) -> None:
        for token in (
            '"child_group_receipt_count": 2',
            '"runner_receipt_count": 4',
            '"media_count": 4',
            '"normalized_clean_latent_count": 4',
            '"source_latent_hash_count": 4',
            '"generated_latent_hash_count": 4',
            '"source_latent_raw_sha256"',
            '"materialized_source_tensor_raw_sha256"',
            '"source_latent_equals_single_materialization": True',
            '"generated_latent_raw_sha256"',
            '"normalized_clean_latent_file_sha256"',
            '"normalized_clean_latent_tensor_raw_sha256"',
            '"normalized_clean_latent_reopened_and_verified": True',
            '"normalized_clean_latent_file_and_tensor_hashes_verified": True',
            '"child_receipt_reopened_and_verified": True',
            '"runner_receipts_reopened": True',
            '"media_hashes_verified": True',
            '"latent_hashes_verified": True',
            '"candidate_zero_noise_sha256"',
            '"t0_i0_candidate_zero_noise_identical": True',
            '"candidate_zero_noise_equality_verified_within_each_source": True',
            '"t0_i0_source_clean_latent_identical": True',
            "all8-parent-receipt.json",
            '"source_materialization_count": 2',
            '"source_materialization_receipt_count": 2',
            '"source_materialization_consumer_count": 4',
            '"runner_source_latents_equal_materialized_tensors": True',
        ):
            self.assertIn(token, self.text)
        self.assertIn("raw != canonical(value) + b\"\\n\"", self.text)
        self.assertIn("object_sha(unsigned) != declared", self.text)
        self.assertIn("plain_file(media, 0o444)", self.text)
        self.assertIn("plain_file(clean_latent, 0o444)", self.text)
        self.assertIn("plain_file(path, 0o444)", self.text)
        self.assertIn(
            'actual_clean_tensor_sha != cert.get("generated_latent_raw_sha256")',
            self.text,
        )
        self.assertIn(
            'attempts[0]["source_latent_raw_sha256"] != materialization["tensor_raw_sha256"]',
            self.text,
        )

    def test_materializer_and_consumers_are_hash_closed_without_causal_claim(self) -> None:
        for token in (
            '"full_source_vae_encode_count": 1',
            '"consumed_by_arms": ["T0", "I0"]',
            '"consumer_count": 2',
            '"source_clean_vae_encode_count": 2',
            '"consumer_count": 4',
            '"fresh_create_only_no_cache_reuse": True',
            '"runner_source_reencoding": False',
            '"predecessor_job_132359_cross_arm_source_latent_mismatch_observed": True',
            '"cause_attribution_authorized": False',
            '"single_materialization_removes_repeated_encode_boundary": True',
            '"quality_or_semantic_success_implied": False',
            "source materializer artifact/tensor closure differs",
            "source materializer one-encode closure differs",
            "source materializer WORLD1/authority closure differs",
            'encoding.get("encoded_in_runner") is not False',
            'encoding.get("posterior_statistic") != "latent_dist.mode"',
            'encoding.get("callable_signature")',
            '!= "(vae, x: torch.Tensor) -> torch.Tensor"',
            'sealed.get("accepted_roles") != materializer_accepted_roles',
            'sealed.get("forbidden_roles") != materializer_forbidden_roles',
            "runner sealed source-coordinate provenance differs",
            "runner sealed source-coordinate {stage} rehash differs",
            'sealed_coordinate != certificate_coordinate',
            'transport.get("complete_source_video_vae_encoded_in_runner") is not False',
            'sealed_coordinate.get("terminal_rehash_recorded_in_this_receipt") is not False',
            'sealed_coordinate.get("terminal_rehash_required_for_process_success") is not True',
            '"runner_terminal_source_coordinate_rehash_enforced_by_successful_exit": True',
            '"source_materializations_reopened_after_all_consumers": True',
        ):
            self.assertIn(token, self.text)
        self.assertNotIn('"(vae, video)"', self.text)
        self.assertNotRegex(self.text, re.compile(r"(?i)miopen.{0,80}(?:cause|caused|root cause)"))

    def test_parent_explicitly_has_no_selection_training_optimizer_or_success(self) -> None:
        for key in (
            "quality_authority",
            "selection_authority",
            "semantic_action_success",
            "identity_preservation_success",
            "training_authority",
            "optimizer_authority",
            "training_update_allowed",
            "optimizer_step_allowed",
            "checkpoint_or_lora_artifact",
            "production_claim_authorized",
        ):
            self.assertIn(f'"{key}": False', self.text)
        runner_call = self.text[
            self.text.index('"${runner}" \\') : self.text.index("run_wave() {")
        ]
        self.assertNotRegex(
            runner_call,
            r"--(?:train|optimizer|update|resume|selection|reward|quality)(?:\s|=)",
        )
        self.assertNotIn("sbatch ", self.text)


if __name__ == "__main__":
    unittest.main()
