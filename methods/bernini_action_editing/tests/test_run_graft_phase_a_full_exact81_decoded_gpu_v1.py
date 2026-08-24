#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER = METHOD_ROOT / "run_graft_phase_a_full_exact81_decoded_gpu_v1.py"
CORE = METHOD_ROOT / "graft_phase_a_full_exact81_decoded_v1.py"
ACTIVE_CORE = METHOD_ROOT / "train_graft_phase_a_active14_transaction_v1.py"
ACTIVE_RUNNER = METHOD_ROOT / "run_graft_phase_a_active14_transaction_gpu_v1.py"


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


class FullExact81RunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.core = CORE.read_text(encoding="utf-8")

    def test_same_process_active14_api_is_two_phase(self):
        main = function_source(RUNNER, "main")
        callbacks = function_source(RUNNER, "_callbacks_factory")
        self.assertIn(
            "prepare, finalize, publish_after_outer_close = _callbacks_factory",
            main,
        )
        self.assertIn("replay_active14_for_downstream", main)
        self.assertIn("prepare=prepare, finalize=finalize", main)
        self.assertLess(
            callbacks.index("def prepare("), callbacks.index("def finalize(")
        )
        self.assertIn('"preparation_completed": True', callbacks)
        self.assertIn('"published": False', callbacks)
        self.assertIn('"finalize_completed": True', callbacks)
        self.assertIn('"active14_commit_receipt_digest"', callbacks)
        self.assertIn('"preparation_receipt_digest"', callbacks)
        self.assertIn('"published": False', callbacks)
        self.assertIn("def publish_after_outer_close()", callbacks)
        self.assertIn("_rename_directory_noreplace(stage, final)", callbacks)
        self.assertIn("renameat2", self.runner)
        self.assertIn("atomic publish reservation admission differs", self.runner)
        self.assertIn("os.mkdir(target.name, 0o700, dir_fd=parent_fd)", self.runner)
        self.assertIn("reservation.st_nlink < 2", self.runner)
        self.assertIn("os.fchmod(reservation_fd, 0o000)", self.runner)
        self.assertIn("stat.S_IMODE(reservation.st_mode) != 0o000", self.runner)
        self.assertIn("exact final-path directory is the publication lock", self.runner)
        self.assertIn("retain the reservation exactly as", self.runner)
        self.assertNotIn("os.rmdir(target.name, dir_fd=parent_fd)", self.runner)
        self.assertIn("os.listdir(reservation_fd)", self.runner)
        self.assertIn("src_dir_fd=parent_fd", self.runner)
        self.assertIn("dst_dir_fd=parent_fd", self.runner)
        self.assertIn("atomic publish parent identity changed", self.runner)
        self.assertIn("final.is_symlink()", callbacks)
        self.assertLess(
            main.index("replay_active14_for_downstream"),
            main.index("publish_after_outer_close()"),
        )

    def test_active14_upstream_contract_is_final_field14_v2(self):
        active_core = ACTIVE_CORE.read_text(encoding="utf-8")
        active_runner = ACTIVE_RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'UPSTREAM_SCHEMA_VERSION = "bernini-graft-phase-a-field14-world8-parent-v2"',
            active_core,
        )
        self.assertIn('"field14_source_commit"', active_runner)
        self.assertIn("args.expected_field14_source_commit", active_runner)
        self.assertIn("def replay_active14_for_downstream(", active_runner)
        self.assertIn("prepare: PrepareCallback", active_runner)
        self.assertIn("finalize: FinalizeCallback", active_runner)

    def test_active14_scheduler_parent_receipt_is_independently_sealed(self):
        loader = function_source(RUNNER, "_load_upstream_active14_parent")
        main = function_source(RUNNER, "main")
        callbacks = function_source(RUNNER, "_callbacks_factory")
        self.assertIn("ACTIVE14_PARENT_SCHEMA_VERSION", loader)
        self.assertIn("expected_upstream_active14_receipt_sha256", loader)
        self.assertIn("active14 parent receipt digest differs", loader)
        self.assertIn("full_expected_active14_source_commit", loader)
        self.assertIn("full_expected_active14_plan_sha256", loader)
        self.assertIn("full_expected_active14_launcher_sha256", loader)
        self.assertIn("expected_upstream_field14_receipt_sha256", loader)
        self.assertIn("weights_inherited_from_dependency_job", loader)
        self.assertLess(
            main.index("_load_upstream_active14_parent"),
            main.index("replay_active14_for_downstream"),
        )
        self.assertIn("upstream_active14_parent_receipt_sha256", callbacks)
        self.assertIn("upstream_active14_parent_receipt_digest", callbacks)

    def test_rollout_is_fresh_gaussian_source_conditioned_exact40(self):
        rollout = function_source(RUNNER, "_exact40_action_rollout")
        step = function_source(RUNNER, "_step_state")
        projection = function_source(RUNNER, "_route_trace_projection")
        self.assertIn("keyed_fresh_gaussian", rollout)
        self.assertIn("full-exact81-action-seed-", rollout)
        self.assertIn("for index in range(full81.NUM_INFERENCE_STEPS)", rollout)
        self.assertIn("context.confirmation.atlas_frames", rollout)
        self.assertLess(
            rollout.index("with torch.no_grad():"),
            rollout.index("context.handle.build_atlas("),
        )
        self.assertIn("atlas.tokens.grad_fn is not None", rollout)
        self.assertIn("source_latent=context.confirmation.source_latent", step)
        self.assertEqual(step.count("context.diffusion.scheduler.step("), 1)
        self.assertIn("condition=context.negative_condition", step)
        self.assertIn("condition=context.action_condition", step)
        self.assertIn('"target_video_used": False', step)
        self.assertIn('"clean_source_initial_latent_used": False', step)
        self.assertIn("full81.validate_sealed_mapping", projection)
        self.assertIn(
            'raw.get("sequence_parallel_rank") != context.topology.sp_rank',
            projection,
        )
        self.assertIn("short_runner.rebinder.mid_low_sigma_gate", projection)
        self.assertIn("schedule_index < 26 and expected_gate != 0.0", projection)
        self.assertIn("schedule_index >= 26", projection)
        self.assertNotIn('"sequence_parallel_rank":', projection)
        self.assertIn("negative_route_projection", step)
        self.assertNotIn("dict(negative_route.receipt())", step)

    def test_decoder_binds_exact81_25fps_and_both_artifact_hashes(self):
        decoder = function_source(RUNNER, "_decode_and_seal_arm")
        callbacks = function_source(RUNNER, "_callbacks_factory")
        geometry = function_source(RUNNER, "_validate_continuation_geometry")
        self.assertIn("AutoencoderKLWan.from_pretrained", decoder)
        self.assertIn("_vae_decode(vae, endpoint)", decoder)
        self.assertIn('artifact_role="native_sampler_proposal"', decoder)
        self.assertIn("safe_open(str(latent_path)", decoder)
        self.assertIn("staged latent does not bind exact endpoint", decoder)
        self.assertIn("endpoint_after_decode", decoder)
        self.assertIn("VAE decode mutated exact40 endpoint", decoder)
        self.assertIn("safetensors_roundtrip_verified", decoder)
        self.assertIn("full81.FRAME_COUNT", decoder)
        self.assertIn("_open_frozen_ffprobe", decoder)
        self.assertIn("_probe_with_frozen_ffprobe", decoder)
        self.assertIn('role="normalized-clean-latent"', decoder)
        self.assertIn('role="decoded-exact81-video"', decoder)
        self.assertIn("opened_nofollow_and_revalidated", self.runner)
        self.assertIn('metadata["source_derived_bucket_hw"]', geometry)
        self.assertIn("DERIVED_BUCKET_HW_BY_DP_ARM", geometry)
        self.assertIn("exact81 endpoint bucket geometry differs", decoder)
        self.assertLess(
            callbacks.index("_decode_and_seal_arm("),
            callbacks.index("trainable_after_decode ="),
        )
        self.assertIn("exact81 decode changed parameter bytes", callbacks)
        self.assertIn("decode_statuses", callbacks)
        self.assertIn("one or more exact81 arm decodes failed", callbacks)

    def test_no_checkpoint_optimizer_or_semantic_evaluator_surface(self):
        for forbidden in (
            "torch.save",
            "torch.optim",
            "load_state_dict(",
            "action_score",
            "identity_score",
            "quality_score",
            "selected_candidate",
            "target_video_path",
        ):
            self.assertNotIn(forbidden, self.runner)
        for forbidden_import in (
            "saic_source_state_flow_transport_v1",
            "source_state_flow_step",
            "clean_source initial latent",
        ):
            self.assertNotIn(forbidden_import, self.runner)
            self.assertNotIn(forbidden_import, self.core)
        self.assertIn('"checkpoint_written": False', self.runner)
        self.assertIn('"visual_semantics_evaluated": False', self.runner)

    def test_failure_after_publish_is_quarantined(self):
        main = function_source(RUNNER, "main")
        self.assertIn("failed-postcommit-job", main)
        self.assertIn("failed-staging-job", main)
        self.assertIn("candidate.lstat()", main)
        self.assertIn("stat.S_ISDIR", main)
        self.assertIn('os.environ.get("RANK", "") in ("", "0")', main)
        self.assertIn("os.rename(candidate, quarantine)", main)
        self.assertIn("_fsync_directory(output.parent)", main)
        self.assertLess(
            main.index("publish_after_outer_close()"),
            main.index("except BaseException:"),
        )

    def test_core_requires_full_trace_media_and_false_authority(self):
        for evidence in (
            "official_unipc_step_count",
            "exact40 state chain is discontinuous",
            "decoded_tensor_shape",
            "both_exact81_decoded",
            "all_parameter_bytes_unchanged_during_decode",
            "decoded_media_semantically_evaluated",
        ):
            self.assertIn(evidence, self.core)
        for authority in (
            "action_authority",
            "identity_authority",
            "quality_authority",
            "scientific_success_claimed",
        ):
            self.assertIn(authority, self.core)


if __name__ == "__main__":
    unittest.main()
