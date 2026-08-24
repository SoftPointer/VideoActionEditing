#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import infer_guided_source_aligned_controller_oracle as guided_runner  # noqa: E402
import infer_saic_source_state_flow_transport_v1 as runner  # noqa: E402
import saic_pure_t2v_event_bank_v1 as event_bank  # noqa: E402
import saic_source_state_flow_transport_v1 as transport  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fake_schedule(spec: runner.ArmSpec) -> runner.NativeScheduleBundle:
    sigmas = torch.linspace(1.0, 0.0, 41, dtype=torch.float32)
    timesteps = torch.arange(999, 959, -1, dtype=torch.int64)
    return runner.bind_native_schedule_objects(
        scheduler_sigmas=sigmas,
        runtime_timesteps=timesteps,
        spec=spec,
        expected_pinned_schedule_sha256=None,
        expected_scheduler_sigma_fp32_sha256=None,
    )


def adapter_diagnostics(spec: runner.ArmSpec) -> dict[str, object]:
    seals = (("diffusion", sha("diffusion")), ("transformer", sha("transformer")))
    reference_patches = spec.expected_guided_queries if spec.uses_reference_frame0 else 0
    return {
        "field_regime": spec.field_regime,
        "guided_query_count": spec.expected_guided_queries,
        "raw_transformer_forward_count": spec.expected_raw_forwards,
        "patch_query_count": spec.expected_guided_queries,
        "patch_reference_count": reference_patches,
        "guided_query_attempt_count": spec.expected_guided_queries,
        "guided_query_success_count": spec.expected_guided_queries,
        "raw_transformer_forward_attempt_count": spec.expected_raw_forwards,
        "raw_transformer_forward_success_count": spec.expected_raw_forwards,
        "patch_query_attempt_count": spec.expected_guided_queries,
        "patch_query_success_count": spec.expected_guided_queries,
        "patch_reference_attempt_count": reference_patches,
        "patch_reference_success_count": reference_patches,
        "vendor_single_attempt_count": (
            0 if spec.uses_reference_frame0 else spec.expected_guided_queries
        ),
        "vendor_single_success_count": (
            0 if spec.uses_reference_frame0 else spec.expected_guided_queries
        ),
        "vendor_chain_attempt_count": (
            spec.expected_guided_queries if spec.uses_reference_frame0 else 0
        ),
        "vendor_chain_success_count": (
            spec.expected_guided_queries if spec.uses_reference_frame0 else 0
        ),
        "expected_guided_query_count": spec.expected_guided_queries,
        "expected_raw_transformer_forward_count": spec.expected_raw_forwards,
        "next_step_index": 40,
        "next_candidate_index": 0,
        "next_role": "target",
        "initial_full_model_content_audit": True,
        "final_full_model_content_audit": True,
        "rollout_complete": True,
        "adapter_failed": False,
        "failure_stage": None,
        "initial_model_content_seal_sha256_by_module": seals,
        "final_model_content_seal_sha256_by_module": seals,
        "model_checkpoint_use_verified": False,
        "target_tail_direct_view": True,
        "optimizer_step_allowed": False,
        "training_update_allowed": False,
        "semantic_action_success": False,
    }


def core_diagnostics(spec: runner.ArmSpec) -> dict[str, object]:
    guidance = runner.guidance_contract(spec)
    return {
        "guided_velocity_query_count": spec.expected_guided_queries,
        "raw_transformer_forward_count": spec.expected_raw_forwards,
        "field_regime": spec.field_regime,
        "visual_condition_scope": (
            "source_frame0_only_no_future_motion"
            if spec.uses_reference_frame0
            else "none"
        ),
        "guidance_mode": spec.guidance_mode,
        "guidance_contract_sha256": runner.legacy.object_sha256(guidance),
        "guidance_scale": 4.0,
        "image_guidance_scale": guidance["image_guidance_scale"],
        "guidance_chain_scales": tuple(guidance["guidance_chain_scales"]),
        "apg_eta": 0.5,
        "apg_norm_threshold": 50.0,
        "apg_norm_thresholds": tuple(guidance["apg_norm_thresholds"]),
        "apg_momentum": 0.0,
        "apg_momenta": tuple(guidance["apg_momenta"]),
        "branch_order": tuple(guidance["branch_order"]),
        "noise_bank_digest_verified": True,
        "raw_transformer_forward_count_verified": False,
        "native_request_execution_verified": False,
        "model_checkpoint_use_verified": False,
        "noise_distribution_verified": False,
        "optimizer_step_allowed": False,
        "training_update_allowed": False,
        "semantic_action_success": False,
    }


def source_coordinate_certificate() -> dict[str, object]:
    provenance = {
        "schema_version": runner.SOURCE_CLEAN_RECEIPT_SCHEMA,
        "artifact_path": "/tmp/source.clean-latent.safetensors",
        "artifact_sha256": sha("source-artifact"),
        "artifact_size": 123,
        "artifact_mode": "0444",
        "receipt_path": "/tmp/source.clean-latent.safetensors.receipt.json",
        "receipt_file_sha256": sha("source-receipt-file"),
        "receipt_digest": sha("source-receipt-content"),
        "tensor_key": runner.SOURCE_CLEAN_TENSOR_KEY,
        "tensor_raw_sha256": sha("source-latent"),
        "shape": [1, 16, 21, 60, 104],
        "dtype": "torch.float32",
        "source_manifest_raw_sha256": sha("source-manifest-raw"),
        "source_manifest_content_sha256": sha("source-manifest-content"),
        "row_id": "fit-dog-00-7b88a1ca1f804f41",
        "source_video_sha256": sha("source-video"),
        "checkpoint_tree_sha256": sha("checkpoint"),
        "checkpoint_content_manifest_audit": {},
        "source_derived_bucket_hw": [480, 832],
        "materializer_method_source_revision": "1" * 40,
        "materializer_method_source_archive_sha256": sha("method-archive"),
        "materializer_runtime_source_index_sha256": sha("materializer-runtime"),
        "loaded_from_sealed_source_coordinate": True,
        "encoded_in_runner": False,
        "runner_reencoding_verified": False,
        "inference_available_source_video": True,
        "ground_truth": False,
        "quality_authority": False,
        "semantic_action_success": False,
        "identity_preservation_success": False,
        "training_authority": False,
        "optimizer_authority": False,
        "cpu_to_gpu_byte_exact": True,
        "rank0_broadcast_before_renderer": True,
        "all_rank_identity_after_broadcast": True,
        "terminal_rehash_recorded_in_this_receipt": False,
        "terminal_rehash_required_for_process_success": True,
    }
    for stage in ("pre_rollout", "pre_publish"):
        provenance[f"{stage}_rehash"] = {
            "stage": stage,
            "artifact_sha256": provenance["artifact_sha256"],
            "receipt_file_sha256": provenance["receipt_file_sha256"],
            "receipt_digest": provenance["receipt_digest"],
            "tensor_raw_sha256": provenance["tensor_raw_sha256"],
            "retained_descriptor_identity_verified": True,
            "canonical_path_identity_verified": True,
            "mode_0444_verified": True,
            "canonical_receipt_and_digest_verified": True,
            "tensor_reopened_byte_exact": True,
        }
    return provenance


def rank_rows(spec: runner.ArmSpec) -> list[dict[str, object]]:
    coordinate = source_coordinate_certificate()
    certificate = {
        "arm": spec.arm,
        "source_video_sha256": sha("source-video"),
        "source_latent_raw_sha256": sha("source-latent"),
        "sealed_source_coordinate": coordinate,
        "loaded_from_sealed_source_coordinate": True,
        "source_clean_encoded_in_runner": False,
        "generated_latent_raw_sha256": sha("generated-latent"),
        "noise_bank_sha256": sha("noise-bank"),
        "candidate_zero_noise_sha256": sha("candidate-zero"),
        "native_schedule_sha256": sha("native-schedule"),
        "core_sigma_schedule_sha256": sha("core-schedule"),
        "model_receipt_sha256": sha("model-receipt"),
        "runtime_source_index_sha256": sha("runtime-source"),
        "native_guided_query_attempt_count": spec.expected_guided_queries,
        "native_guided_query_success_count": spec.expected_guided_queries,
        "native_raw_transformer_forward_attempt_count": spec.expected_raw_forwards,
        "native_raw_transformer_forward_success_count": spec.expected_raw_forwards,
        "core_native_guided_count_reconciled": True,
        "core_native_raw_forward_count_reconciled": True,
        "model_freeze_unchanged": True,
        "native_adapter": adapter_diagnostics(spec),
        "transport_core": core_diagnostics(spec),
        "transport_core_diagnostics_sha256": sha("core-diagnostics"),
    }
    return [
        {
            "rank": rank,
            "local_rank": rank,
            "world_size": 4,
            "ulysses_size": 4,
            "certificate": deepcopy(certificate),
        }
        for rank in range(4)
    ]


class ArmContractTests(unittest.TestCase):
    def test_arm_order_and_exact_counts(self) -> None:
        self.assertEqual(
            runner.ARM_NAMES, ("T0", "T1", "I0", "IAVG", "I1", "I1A")
        )
        t0 = runner.arm_spec("T0")
        i0 = runner.arm_spec("I0")
        for spec in (t0, i0):
            self.assertEqual(spec.candidate_schedule, (1,) * 40)
            self.assertEqual(spec.expected_guided_queries, 80)
            self.assertFalse(spec.anc_enabled)
        self.assertEqual(t0.expected_raw_forwards, 160)
        self.assertEqual(i0.expected_raw_forwards, 240)
        for name in ("T1", "IAVG", "I1", "I1A"):
            spec = runner.arm_spec(name)
            self.assertEqual(spec.candidate_schedule[:4], (5, 5, 5, 1))
            self.assertEqual(spec.expected_guided_queries, 104)
            self.assertTrue(spec.anc_enabled)
            self.assertEqual(
                spec.expected_raw_forwards,
                208 if name == "T1" else 312,
            )
        self.assertFalse(runner.arm_spec("T1").uses_reference_frame0)
        self.assertTrue(runner.arm_spec("I0").uses_reference_frame0)
        self.assertTrue(runner.arm_spec("I1A").anchor_latent_phase_zero)

    def test_guidance_contract_distinguishes_bandwidth_and_mechanism(self) -> None:
        contracts = {
            name: runner.guidance_contract(runner.arm_spec(name))
            for name in runner.ARM_NAMES
        }
        digests = {name: runner.legacy.object_sha256(value) for name, value in contracts.items()}
        self.assertEqual(len(set(digests.values())), len(runner.ARM_NAMES))
        self.assertEqual(contracts["T0"]["visual_condition"], "none")
        self.assertEqual(contracts["T0"]["per_guided_query_raw_forwards"], 2)
        self.assertEqual(
            contracts["I0"]["visual_condition"],
            "independently_vae_encoded_source_rgb_frame0",
        )
        self.assertEqual(contracts["I0"]["per_guided_query_raw_forwards"], 3)
        self.assertEqual(contracts["I0"]["image_guidance_scale"], 4.5)
        self.assertEqual(contracts["I0"]["guidance_chain_scales"], [4.5, 4.0])
        self.assertEqual(
            contracts["I0"]["branch_order"],
            list(transport.EXPECTED_R2V_I0_BRANCH_ORDER),
        )
        self.assertFalse(contracts["I1"]["full_source_video_field_tokens"])

    def test_reference_encoder_receipt_is_zero_only_for_text_arms(self) -> None:
        text_receipt, text_digest = runner.build_reference_encoder_receipt(
            spec=runner.arm_spec("T0"),
            model_receipt_sha256=sha("model"),
            checkpoint_tree_sha256=sha("checkpoint"),
            vae_z_dim=16,
            bernini_pipeline_sha256=sha("pipeline"),
        )
        image_receipt, image_digest = runner.build_reference_encoder_receipt(
            spec=runner.arm_spec("I0"),
            model_receipt_sha256=sha("model"),
            checkpoint_tree_sha256=sha("checkpoint"),
            vae_z_dim=16,
            bernini_pipeline_sha256=sha("pipeline"),
        )
        self.assertEqual(text_digest, "0" * 64)
        self.assertFalse(text_receipt["used"])
        self.assertNotEqual(image_digest, "0" * 64)
        self.assertFalse(image_receipt["temporal_video_latent_slice_used"])

    def test_cli_has_only_sealed_caption_selection_not_free_form_text(self) -> None:
        parser = runner.build_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        for required in (
            "--source-manifest",
            "--event-bank",
            "--row-id",
            "--branch",
            "--rollout-seed",
            "--source-clean-latent",
            "--source-clean-latent-receipt",
            "--expected-source-clean-latent-sha256",
            "--expected-source-clean-latent-receipt-sha256",
            "--expected-source-clean-tensor-raw-sha256",
        ):
            self.assertIn(required, options)
        for forbidden in (
            "--source-video",
            "--source-caption",
            "--target-caption",
            "--instruction",
            "--mask",
            "--pose",
            "--flow",
        ):
            self.assertNotIn(forbidden, options)

    def test_public_arm_and_prompt_registry_rebinding_cannot_change_contract(self) -> None:
        original_arms = runner.ARM_SPECS
        original_names = runner.ARM_NAMES
        original_prompts = runner.TASK_SYSTEM_PROMPTS
        try:
            runner.ARM_SPECS = {"EVIL": runner.arm_spec("T0")}
            runner.ARM_NAMES = ("EVIL",)
            runner.TASK_SYSTEM_PROMPTS = {"t2v": "forged"}
            self.assertEqual(runner.arm_spec("T0").arm, "T0")
            with self.assertRaises(runner.SAICInferenceError):
                runner.arm_spec("EVIL")
            self.assertEqual(
                runner.build_task_prompt(
                    "t2v", "Canonical body.", prompt_cleaner=lambda value: value
                ),
                runner.T2V_SYSTEM_PROMPT + "Canonical body.",
            )
            parser = runner.build_parser()
            arm_action = next(
                action for action in parser._actions if "--arm" in action.option_strings
            )
            self.assertEqual(tuple(arm_action.choices), original_names)
        finally:
            runner.ARM_SPECS = original_arms
            runner.ARM_NAMES = original_names
            runner.TASK_SYSTEM_PROMPTS = original_prompts


class SealedCaptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = source_set.ASSET_PATH.resolve()
        cls.event_path = event_bank.ASSET_PATH.resolve()
        cls.manifest = source_set.load_manifest(cls.source_path)
        source_set.validate_manifest(cls.manifest, verify_bound_files=False)
        cls.spec, _ = event_bank.load_sealed_spec(
            cls.event_path,
            expected_raw_sha256=runner.EVENT_BANK_RAW_SHA256,
            source_manifest_path=cls.source_path,
        )

    def test_pinned_asset_raw_hashes_match(self) -> None:
        self.assertEqual(
            source_set.file_sha256(self.source_path),
            runner.SOURCE_MANIFEST_RAW_SHA256,
        )
        self.assertEqual(
            event_bank.file_sha256(self.event_path), runner.EVENT_BANK_RAW_SHA256
        )

    def test_dog_forward_caption_is_unique_and_not_naively_concatenated(self) -> None:
        cell = runner.resolve_sealed_forward_cell(
            self.manifest,
            self.spec,
            row_id="fit-dog-00-7b88a1ca1f804f41",
            branch="forward",
            rollout_seed=2026082101,
        )
        self.assertIn("settle into a stable sit", cell["target_caption_body"])
        self.assertNotIn("dog remains essentially still", cell["target_caption_body"])
        self.assertIn("dog remains essentially still", cell["source_caption_body"])
        self.assertEqual(
            sha(cell["target_caption_body"]),
            cell["target_caption_body_utf8_sha256"],
        )

    def test_human_forward_caption_uses_canonical_rise_event(self) -> None:
        cell = runner.resolve_sealed_forward_cell(
            self.manifest,
            self.spec,
            row_id="fit-human-00-a35b590961d24694",
            branch="forward",
            rollout_seed=2026082121,
        )
        self.assertIn("rise smoothly", cell["target_caption_body"])
        self.assertIn("one-knee kneeling pose", cell["source_caption_body"])
        self.assertFalse(cell["event_verified"])
        self.assertFalse(cell["optimizer_authorized"])

    def test_wrong_branch_or_seed_fails_closed(self) -> None:
        with self.assertRaisesRegex(runner.SAICInferenceError, "branch=forward"):
            runner.resolve_sealed_forward_cell(
                self.manifest,
                self.spec,
                row_id="fit-dog-00-7b88a1ca1f804f41",
                branch="reverse",
                rollout_seed=2026082101,
            )
        with self.assertRaisesRegex(runner.SAICInferenceError, "exactly one"):
            runner.resolve_sealed_forward_cell(
                self.manifest,
                self.spec,
                row_id="fit-dog-00-7b88a1ca1f804f41",
                branch="forward",
                rollout_seed=1,
            )

    def test_load_sealed_cell_binds_both_raw_and_content_hashes(self) -> None:
        cell, assets = runner.load_sealed_caption_cell(
            source_manifest_path=self.source_path,
            source_manifest_raw_sha256=runner.SOURCE_MANIFEST_RAW_SHA256,
            event_bank_path=self.event_path,
            event_bank_raw_sha256=runner.EVENT_BANK_RAW_SHA256,
            row_id="fit-dog-00-7b88a1ca1f804f41",
            branch="forward",
            rollout_seed=2026082101,
            verify_bound_source_files=False,
        )
        self.assertEqual(cell["candidate_id"], "saic-7b88a1ca1f804f41-forward-s2026082101")
        self.assertEqual(
            assets["source_manifest_raw_sha256"], runner.SOURCE_MANIFEST_RAW_SHA256
        )
        self.assertEqual(assets["event_bank_raw_sha256"], runner.EVENT_BANK_RAW_SHA256)
        self.assertRegex(assets["source_manifest_content_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(assets["event_bank_content_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(assets["source_manifest_bound_files_verified"])

    def test_main_limits_bound_media_reads_to_the_selected_cell(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        main = source[source.index("def main(") :]
        self.assertIn("verify_bound_source_files=False", main)
        self.assertNotIn("verify_bound_source_files=True", main)
        self.assertIn("prepare_hashed_source_snapshot(source_path)", main)
        self.assertIn('source_metadata.get("frame_count") != FRAME_COUNT', main)
        self.assertIn('source_metadata.get("fps") != FPS', main)
        self.assertIn("revalidate_terminal_sealed_input_bytes(", main)
        self.assertNotIn("source_clean = _vae_encode(", main)
        self.assertIn("source_pixels[:, :, 0:1, :, :].contiguous()", main)
        self.assertIn("load_sealed_source_coordinate(", main)
        self.assertIn('stage="pre_rollout"', main)
        self.assertIn('stage="pre_publish"', main)
        self.assertIn('stage="terminal"', main)

    def test_terminal_semantic_bytes_are_rehashed_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            manifest_path = base / "manifest.json"
            event_path = base / "events.json"
            source_path = base / "source.mp4"
            manifest_path.write_bytes(b"manifest")
            event_path.write_bytes(b"events")
            source_path.write_bytes(b"source")
            assets = {
                "source_manifest_path": str(manifest_path),
                "source_manifest_raw_sha256": runner.legacy.file_sha256(manifest_path),
                "event_bank_path": str(event_path),
                "event_bank_raw_sha256": runner.legacy.file_sha256(event_path),
            }
            cell = {"source_video_sha256": runner.legacy.file_sha256(source_path)}
            result = runner.revalidate_terminal_sealed_input_bytes(
                sealed_assets=assets,
                sealed_cell=cell,
                selected_source_path=source_path,
            )
            self.assertTrue(result["source_manifest_terminal_raw_sha256_verified"])
            event_path.write_bytes(b"changed")
            with self.assertRaisesRegex(runner.SAICInferenceError, "event bank bytes"):
                runner.revalidate_terminal_sealed_input_bytes(
                    sealed_assets=assets,
                    sealed_cell=cell,
                    selected_source_path=source_path,
                )

    def test_task_prompt_has_exact_single_prefix(self) -> None:
        body = "A complete source-content caption."
        prompt = runner.build_task_prompt("t2v", body, prompt_cleaner=lambda value: value)
        self.assertEqual(prompt, runner.T2V_SYSTEM_PROMPT + body)
        self.assertEqual(prompt.count(runner.T2V_SYSTEM_PROMPT), 1)
        with self.assertRaises(runner.SAICInferenceError):
            runner.build_task_prompt("t2v", f" {body}", prompt_cleaner=lambda value: value)


class NoiseAndScheduleTests(unittest.TestCase):
    def test_candidate_zero_noise_is_exactly_shared_across_k1_and_k5(self) -> None:
        shape = (1, 16, 21, 2, 2)
        k1 = runner.build_fresh_noise_bank(
            shape=shape,
            device=torch.device("cpu"),
            master_seed=2026082101,
            candidate_schedule=runner.K1_SCHEDULE,
        )
        k5 = runner.build_fresh_noise_bank(
            shape=shape,
            device=torch.device("cpu"),
            master_seed=2026082101,
            candidate_schedule=runner.K5_EARLY_SCHEDULE,
        )
        self.assertTrue(all(torch.equal(left[0], right[0]) for left, right in zip(k1, k5)))
        self.assertEqual(
            runner.candidate_zero_noise_sha256(k1),
            runner.candidate_zero_noise_sha256(k5),
        )
        self.assertNotEqual(
            transport.noise_bank_sha256(k1, candidate_schedule=runner.K1_SCHEDULE),
            transport.noise_bank_sha256(
                k5, candidate_schedule=runner.K5_EARLY_SCHEDULE
            ),
        )

    def test_keyed_noise_seed_is_stable_and_cell_specific(self) -> None:
        first = runner.keyed_noise_seed(7, 0, 0)
        self.assertEqual(first, runner.keyed_noise_seed(7, 0, 0))
        self.assertNotEqual(first, runner.keyed_noise_seed(7, 0, 1))
        self.assertNotEqual(first, runner.keyed_noise_seed(7, 1, 0))
        with self.assertRaises(runner.SAICInferenceError):
            runner.keyed_noise_seed(-1, 0, 0)

    def test_schedule_binds_direct_views_and_arm_mechanism(self) -> None:
        t0 = fake_schedule(runner.arm_spec("T0"))
        t1 = fake_schedule(runner.arm_spec("T1"))
        self.assertEqual(len(t0.sigma_schedule), 41)
        self.assertTrue(t0.scalar_views_share_scheduler_storage)
        self.assertTrue(t0.timestep_views_share_runtime_storage)
        self.assertEqual(
            t0.core_sigma_schedule_sha256, t1.core_sigma_schedule_sha256
        )
        self.assertNotEqual(t0.native_schedule_sha256, t1.native_schedule_sha256)
        self.assertEqual(t0.sigma_schedule[-1], 0.0)

    def test_schedule_rejects_non_descending_official_timesteps(self) -> None:
        sigmas = torch.linspace(1.0, 0.0, 41, dtype=torch.float32)
        times = torch.arange(999, 959, -1, dtype=torch.int64)
        times[3] = times[2]
        with self.assertRaisesRegex(runner.SAICInferenceError, "strictly descending"):
            runner.bind_native_schedule_objects(
                scheduler_sigmas=sigmas,
                runtime_timesteps=times,
                spec=runner.arm_spec("T0"),
                expected_pinned_schedule_sha256=None,
                expected_scheduler_sigma_fp32_sha256=None,
            )


@unittest.skipUnless(
    importlib.util.find_spec("safetensors") is not None,
    "local torch test environment has no safetensors",
)
class SealedSourceCoordinateTests(unittest.TestCase):
    @staticmethod
    def _bundle(base: Path) -> tuple[argparse.Namespace, dict[str, object]]:
        from safetensors.torch import save as save_safetensors

        artifact_path = base / "source.clean-latent.safetensors"
        receipt_path = base / "source.clean-latent.safetensors.receipt.json"
        manifest_path = base / "manifest.json"
        source_path = base / "source.mp4"
        checkpoint_path = base / "checkpoint"
        checkpoint_path.mkdir()
        manifest_path.write_bytes(b"sealed-manifest")
        source_path.write_bytes(b"sealed-source")
        source_latent = torch.arange(
            1 * 16 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(1, 16, 21, 2, 2)
        source_pixels = torch.arange(
            1 * 3 * 81 * 16 * 16, dtype=torch.float32
        ).reshape(1, 3, 81, 16, 16)
        artifact_path.write_bytes(
            save_safetensors(
                {runner.SOURCE_CLEAN_TENSOR_KEY: source_latent},
                metadata=dict(runner.SOURCE_CLEAN_ARTIFACT_METADATA),
            )
        )
        artifact_sha256 = runner.legacy.file_sha256(artifact_path)
        tensor_sha256 = runner.tensor_raw_sha256(source_latent)
        checkpoint_identity = {
            "manifest_path": str(base / "checkpoint.manifest"),
            "manifest_sha256_computed": sha("checkpoint-manifest"),
            "manifest_sha256_expected": sha("checkpoint-manifest"),
            "verified_file_count": 23,
            "every_file_sha256_verified": True,
            "verified_entries_digest": sha("checkpoint-entries"),
        }
        materializer_hashes = {
            member: sha(f"materializer-source:{member}")
            for member in runner.SOURCE_CLEAN_MATERIALIZER_ARCHIVE_MEMBERS
        }
        method_provenance = {
            "revision": "1" * 40,
            "scratch_archive_path": str(base / "scratch.tar"),
            "durable_archive_path": str(base / "durable.tar"),
            "archive_sha256": sha("archive"),
            "runtime_source_sha256": dict(materializer_hashes),
            "runtime_source_index_sha256": runner.legacy.object_sha256(
                materializer_hashes
            ),
            "archive_safe_scoped_duplicate_free_link_free": True,
            "revision_label_matches_archive_comment": True,
            "git_revision_verified_by_runner": False,
            "bytecode_policy": {
                "pythondontwritebytecode_environment": "1",
                "dont_write_bytecode": True,
                "pythonpycacheprefix_environment": str(base / "pycache"),
                "runtime_pycache_prefix": str(base / "pycache"),
                "resolved_private_empty_pycache_prefix": str(base / "pycache"),
                "method_source_pycache_ignored": True,
            },
        }
        source_metadata = {
            "decoded_from_private_byte_snapshot": True,
            "frame_count": 81,
            "fps": 25,
            "reported_fps": 25.0,
            "source_input_hw": [16, 16],
            "source_derived_bucket_hw": [16, 16],
            "max_pixels": 245760,
            "stride": 16,
            "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
            "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
            "resize": "torchvision_bicubic_antialias_true",
            "external_shared_i0": False,
        }
        sealed_cell = {
            "row_id": "fit-dog-00-source",
            "iid": "source",
            "analysis_split": "fit",
            "actor_family": "dog",
            "source_video_sha256": runner.legacy.file_sha256(source_path),
        }
        sealed_assets = {
            "source_manifest_path": str(manifest_path),
            "source_manifest_raw_sha256": runner.legacy.file_sha256(
                manifest_path
            ),
            "source_manifest_content_sha256": sha("manifest-content"),
            "source_manifest_schema_version": "bernini-saic-reversible-source-set-v1",
            "source_manifest_dataset_id": "saic-reversible-source-set-exact81-v1",
        }
        encoder_identity = {
            "encoder_symbol": "bernini.pipeline._vae_encode",
            "callable_module": "bernini.pipeline",
            "callable_name": "_vae_encode",
            "callable_qualname": "_vae_encode",
            "callable_signature": "(vae, x: torch.Tensor) -> torch.Tensor",
        }
        artifact = {
            "schema_version": runner.SOURCE_CLEAN_ARTIFACT_SCHEMA,
            "path": str(artifact_path),
            "file_sha256": artifact_sha256,
            "size_bytes": artifact_path.stat().st_size,
            "mode": "0444",
            "tensor_key": runner.SOURCE_CLEAN_TENSOR_KEY,
            "tensor_raw_sha256": tensor_sha256,
            "shape": list(source_latent.shape),
            "dtype": "torch.float32",
            "metadata": dict(runner.SOURCE_CLEAN_ARTIFACT_METADATA),
        }
        receipt = {
            "schema_version": runner.SOURCE_CLEAN_RECEIPT_SCHEMA,
            "method": runner.SOURCE_CLEAN_MATERIALIZER_METHOD,
            "artifact": artifact,
            "sealed_inputs": {
                "accepted_roles": list(runner.SOURCE_CLEAN_ACCEPTED_INPUT_ROLES),
                "forbidden_roles": list(runner.SOURCE_CLEAN_FORBIDDEN_INPUT_ROLES),
                "source_manifest_path": str(manifest_path),
                "source_manifest_raw_sha256": sealed_assets[
                    "source_manifest_raw_sha256"
                ],
                "source_manifest_content_sha256": sealed_assets[
                    "source_manifest_content_sha256"
                ],
                "source_manifest_schema_version": sealed_assets[
                    "source_manifest_schema_version"
                ],
                "source_manifest_dataset_id": sealed_assets[
                    "source_manifest_dataset_id"
                ],
                "source_manifest_bound_files_verified": False,
                "row_id": sealed_cell["row_id"],
                "iid": sealed_cell["iid"],
                "analysis_split": sealed_cell["analysis_split"],
                "actor_family": sealed_cell["actor_family"],
                "source_video_path": str(source_path),
                "source_video_sha256": sealed_cell["source_video_sha256"],
                "source_video_rehashed_after_encode": True,
                "source_manifest_terminal_events_verified": False,
                "optimizer_authorized": False,
            },
            "preprocessing": {
                **source_metadata,
                "source_pixels_shape": list(source_pixels.shape),
                "source_pixels_dtype": "torch.float32",
                "source_pixels_raw_sha256": runner.tensor_raw_sha256(
                    source_pixels
                ),
            },
            "model_closure": {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_tree_sha256": sha("checkpoint-tree"),
                "checkpoint_content_manifest_audit": checkpoint_identity,
                "bernini_revision": "2" * 40,
                "veomni_revision": "3" * 40,
                "bernini_inference_files": {
                    "bernini/pipeline.py": sha("pipeline")
                },
                "bernini_inference_files_index_sha256": runner.legacy.object_sha256(
                    {"bernini/pipeline.py": sha("pipeline")}
                ),
                "method_source_revision": method_provenance["revision"],
                "method_source_archive_sha256": method_provenance[
                    "archive_sha256"
                ],
                "runtime_source_index_sha256": method_provenance[
                    "runtime_source_index_sha256"
                ],
                "method_provenance": method_provenance,
            },
            "encoding": {
                **encoder_identity,
                "encoded_in_runner": False,
                "full_source_vae_encode_count": 1,
                "total_vae_encode_count": 1,
                "posterior_statistic": "latent_dist.mode",
                "sampling": False,
                "torch_inference_mode": True,
                "source_pixels_mutated": False,
                "source_pixels_before_sha256": runner.tensor_raw_sha256(
                    source_pixels
                ),
                "source_pixels_after_sha256": runner.tensor_raw_sha256(
                    source_pixels
                ),
                "vae_dtype": "torch.float32",
                "vae_eval": True,
                "vae_requires_grad": False,
                "latent_frame_count": 21,
                "finite": True,
            },
            "runtime": {
                "device_requested": "cuda:0",
                "world_size": 1,
                "distributed_initialized": False,
                "python_version": "3.10.0",
                "torch_version": "2.4.0",
                "hip_version": "6.3",
                "diffusers_version": "0.31.0",
                "safetensors_version": "0.4.5",
            },
            "authority": {
                "quality_claim_authorized": False,
                "semantic_action_success_authorized": False,
                "ground_truth_authorized": False,
                "training_target_authorized": False,
                "selection_authorized": False,
                "optimizer_step_authorized": False,
                "checkpoint_or_lora_artifact": False,
                "production_claim_authorized": False,
            },
        }
        receipt["receipt_digest"] = runner.legacy.object_sha256(receipt)
        receipt_path.write_bytes(
            runner.legacy.canonical_json_bytes(receipt) + b"\n"
        )
        artifact_path.chmod(0o444)
        receipt_path.chmod(0o444)
        args = argparse.Namespace(
            source_clean_latent=str(artifact_path),
            source_clean_latent_receipt=str(receipt_path),
            expected_source_clean_latent_sha256=artifact_sha256,
            expected_source_clean_latent_receipt_sha256=(
                runner.legacy.file_sha256(receipt_path)
            ),
            expected_source_clean_tensor_raw_sha256=tensor_sha256,
            expected_checkpoint_tree_sha256=sha("checkpoint-tree"),
        )
        context = {
            "sealed_cell": sealed_cell,
            "sealed_assets": sealed_assets,
            "source_path": source_path,
            "source_tensor": source_pixels,
            "source_metadata": source_metadata,
            "checkpoint_path": checkpoint_path,
            "checkpoint_identity": checkpoint_identity,
            "bernini_revision": "2" * 40,
            "veomni_revision": "3" * 40,
            "bernini_inference_files": {"bernini/pipeline.py": sha("pipeline")},
            "method_provenance": {
                **method_provenance,
                "runtime_source_sha256": dict(materializer_hashes),
            },
            "expected_shape": source_latent.shape,
            "encoder_identity": encoder_identity,
        }
        return args, context

    def _load(self, args: argparse.Namespace, context: dict[str, object]):
        encoder_identity = context.pop("encoder_identity")
        try:
            with mock.patch.object(
                runner,
                "runtime_vae_encoder_identity",
                return_value=encoder_identity,
            ):
                return runner.load_sealed_source_coordinate(args, **context)
        finally:
            context["encoder_identity"] = encoder_identity

    def test_loads_cpu_tensor_and_rehashes_all_three_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args, context = self._bundle(Path(root).resolve())
            coordinate = self._load(args, context)
            try:
                self.assertEqual(coordinate.tensor.device.type, "cpu")
                self.assertEqual(
                    runner.tensor_raw_sha256(coordinate.tensor),
                    args.expected_source_clean_tensor_raw_sha256,
                )
                for stage in ("pre_rollout", "pre_publish", "terminal"):
                    result = runner.revalidate_sealed_source_coordinate(
                        coordinate, stage=stage
                    )
                    self.assertEqual(result["stage"], stage)
                    self.assertTrue(result["tensor_reopened_byte_exact"])
            finally:
                runner.close_sealed_source_coordinate(coordinate)

    def test_rejects_receipt_binding_tamper_even_with_updated_digest(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args, context = self._bundle(Path(root).resolve())
            receipt_path = Path(args.source_clean_latent_receipt)
            receipt_path.chmod(0o644)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["sealed_inputs"]["row_id"] = "other-row"
            receipt.pop("receipt_digest")
            receipt["receipt_digest"] = runner.legacy.object_sha256(receipt)
            receipt_path.write_bytes(
                runner.legacy.canonical_json_bytes(receipt) + b"\n"
            )
            receipt_path.chmod(0o444)
            args.expected_source_clean_latent_receipt_sha256 = (
                runner.legacy.file_sha256(receipt_path)
            )
            with self.assertRaisesRegex(runner.SAICInferenceError, "row_id"):
                self._load(args, context)

    def test_rehash_rejects_path_replacement_after_admission(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args, context = self._bundle(Path(root).resolve())
            coordinate = self._load(args, context)
            artifact_path = Path(args.source_clean_latent)
            original = artifact_path.read_bytes()
            artifact_path.unlink()
            artifact_path.write_bytes(original)
            artifact_path.chmod(0o444)
            try:
                with self.assertRaisesRegex(
                    runner.SAICInferenceError, "sealed identity changed"
                ):
                    runner.revalidate_sealed_source_coordinate(
                        coordinate, stage="pre_publish"
                    )
            finally:
                runner.close_sealed_source_coordinate(coordinate)

    def test_read_only_mode_and_expected_hashes_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args, context = self._bundle(Path(root).resolve())
            Path(args.source_clean_latent).chmod(0o644)
            with self.assertRaisesRegex(runner.SAICInferenceError, "0444"):
                self._load(args, context)
        with tempfile.TemporaryDirectory() as root:
            args, context = self._bundle(Path(root).resolve())
            args.expected_source_clean_latent_sha256 = sha("wrong")
            with self.assertRaisesRegex(runner.SAICInferenceError, "SHA-256"):
                self._load(args, context)


class RuntimeAndReceiptTests(unittest.TestCase):
    def test_all_rank_runtime_requires_exact_native_seals_and_counts(self) -> None:
        spec = runner.arm_spec("I1")
        runtime = runner.validate_all_rank_runtime(rank_rows(spec), spec=spec)
        self.assertTrue(runtime["all_rank_exact"])
        self.assertEqual(
            runtime["certificate"]["native_raw_transformer_forward_success_count"],
            spec.expected_raw_forwards,
        )

        tampered = rank_rows(spec)
        tampered[2]["certificate"]["generated_latent_raw_sha256"] = sha("other")
        with self.assertRaisesRegex(runner.SAICInferenceError, "differs across ranks"):
            runner.validate_all_rank_runtime(tampered, spec=spec)

        tampered = rank_rows(spec)
        tampered[0]["certificate"]["native_adapter"][
            "final_model_content_seal_sha256_by_module"
        ] = (("diffusion", sha("changed")), ("transformer", sha("transformer")))
        for index in range(1, 4):
            tampered[index]["certificate"] = deepcopy(tampered[0]["certificate"])
        with self.assertRaisesRegex(runner.SAICInferenceError, "finalization"):
            runner.validate_all_rank_runtime(tampered, spec=spec)

    def test_model_receipt_is_stable_and_discloses_adapter_limit(self) -> None:
        kwargs = dict(
            checkpoint_identity={"manifest": sha("manifest")},
            checkpoint_tree_sha256=sha("checkpoint"),
            bernini_revision="1" * 40,
            veomni_revision="2" * 40,
            bernini_inference_files={"bernini/pipeline.py": sha("pipeline")},
            runtime_source_index_sha256=sha("runtime"),
            renderer_config={"shift": 5.0, "use_unipc": True},
            freeze_certificate={"base_frozen": True},
        )
        first, first_digest = runner.build_model_receipt(**kwargs)
        second, second_digest = runner.build_model_receipt(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_digest, second_digest)
        self.assertTrue(first["checkpoint_provenance_established_externally"])
        self.assertFalse(first["native_adapter_model_checkpoint_use_verified"])

    def test_receipt_has_no_quality_evaluator_training_or_optimizer_authority(self) -> None:
        spec = runner.arm_spec("T0")
        schedule = fake_schedule(spec)
        runtime = runner.validate_all_rank_runtime(rank_rows(spec), spec=spec)
        args = argparse.Namespace(expected_checkpoint_tree_sha256=sha("checkpoint"))
        cell = {
            "row_id": "fit-dog-00-7b88a1ca1f804f41",
            "iid": "7b88a1ca1f804f41",
            "candidate_id": "saic-7b88a1ca1f804f41-forward-s2026082101",
            "branch": "forward",
            "rollout_seed": 2026082101,
            "source_video_sha256": sha("source-video"),
            "source_caption_body": "A dog stands still.",
            "source_caption_body_utf8_sha256": sha("A dog stands still."),
            "target_caption_body": "The same dog sits down.",
            "target_caption_body_utf8_sha256": sha("The same dog sits down."),
        }
        prompts = {
            "source": runner.T2V_SYSTEM_PROMPT + cell["source_caption_body"],
            "target": runner.T2V_SYSTEM_PROMPT + cell["target_caption_body"],
            "negative": "negative",
        }
        receipt = runner.build_receipt(
            args=args,
            spec=spec,
            sealed_cell=cell,
            sealed_assets={"source_manifest_raw_sha256": sha("sm")},
            source_path=Path("/tmp/source.mp4"),
            source_metadata={"source_derived_bucket_hw": [480, 832]},
            prompts=prompts,
            prompt_identities={"source": {}, "target": {}, "negative": {}},
            checkpoint_identity={"manifest": sha("manifest")},
            model_receipt={"base_frozen": True},
            model_receipt_sha256=sha("model"),
            method_provenance={"runtime_source_index_sha256": sha("runtime")},
            bernini_revision="1" * 40,
            veomni_revision="2" * 40,
            bernini_inference_files={"bernini/pipeline.py": sha("pipeline")},
            schedule=schedule,
            guidance_contract_value=runner.guidance_contract(spec),
            noise_bank_sha256=sha("noise"),
            candidate_zero_sha256=sha("candidate-zero"),
            sealed_source_coordinate=source_coordinate_certificate(),
            source_latent_identity={"identity": {"raw_storage_sha256": sha("source-latent")}},
            reference_identity=None,
            reference_encoder_receipt={"used": False},
            reference_encoder_sha256="0" * 64,
            runtime=runtime,
            runtime_versions={"torch": torch.__version__},
            output_identity={"path": "/tmp/output.mp4", "sha256": sha("video"), "size": 10},
            normalized_clean_latent_identity={
                "path": "/tmp/output.mp4.normalized-clean-latent.safetensors",
                "sha256": sha("latent-file"),
                "size": 20,
                "tensor_raw_sha256": sha("generated-latent"),
                "transport_endpoint_before_vae_decode": True,
                "ground_truth": "false",
                "selected_for_training": "false",
            },
            transaction_token="token",
        )
        authority = receipt["authority"]
        for key in (
            "quality_authority",
            "evaluator_authority",
            "semantic_action_success",
            "training_authority",
            "optimizer_authority",
            "training_update_allowed",
            "optimizer_step_allowed",
        ):
            self.assertFalse(authority[key])
        self.assertFalse(receipt["sealed_inputs"]["target_video"])
        self.assertFalse(receipt["sealed_inputs"]["mask_or_swept_tube"])
        self.assertEqual(
            receipt["output"]["normalized_clean_latent"]["tensor_raw_sha256"],
            sha("generated-latent"),
        )
        self.assertTrue(
            receipt["transport"]["sealed_source_coordinate"]
            ["loaded_from_sealed_source_coordinate"]
        )
        self.assertFalse(
            receipt["transport"]["complete_source_video_vae_encoded_in_runner"]
        )
        digest = receipt.pop("receipt_digest")
        self.assertEqual(digest, runner.legacy.object_sha256(receipt))

    def test_published_pair_is_read_only_and_receipt_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / "sample.mp4"
            receipt_path = Path(root) / "sample.mp4.receipt.json"
            video.write_bytes(b"not-a-real-video-but-byte-stable")
            payload = {"schema_version": "test", "value": 7}
            payload["receipt_digest"] = runner.legacy.object_sha256(payload)
            receipt_path.write_bytes(runner.legacy.canonical_json_bytes(payload) + b"\n")
            video_identity = guided_runner.artifact_identity(video)
            receipt_identity = guided_runner.artifact_identity(receipt_path)
            result = runner.seal_published_pair_read_only(
                video,
                receipt_path,
                expected_video_identity=video_identity,
                expected_receipt_identity=receipt_identity,
            )
            self.assertTrue(result["receipt_reopened"])
            self.assertEqual(stat.S_IMODE(video.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o444)

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") is not None,
        "local torch test environment has no safetensors",
    )
    def test_clean_latent_publication_is_create_only_and_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = (
                Path(root)
                / "sample.mp4.normalized-clean-latent.safetensors"
            )
            latent = torch.arange(
                1 * 16 * 21 * 2 * 2, dtype=torch.float32
            ).reshape(1, 16, 21, 2, 2)
            identity = runner.publish_normalized_clean_latent_owned(
                latent, path, transaction_token="test-token"
            )
            self.assertEqual(identity["tensor_raw_sha256"], runner.tensor_raw_sha256(latent))
            self.assertFalse(identity["quality_or_selection_authority"])
            before = path.read_bytes()
            with self.assertRaises(runner.SAICInferenceError):
                runner.publish_normalized_clean_latent_owned(
                    latent.add(1), path, transaction_token="other-token"
                )
            self.assertEqual(path.read_bytes(), before)
            video = Path(root) / "sample.mp4"
            receipt_path = Path(root) / "sample.mp4.receipt.json"
            video.write_bytes(b"video")
            receipt = {
                "schema_version": "test",
                "output": {
                    "normalized_clean_latent": {
                        "tensor_raw_sha256": identity["tensor_raw_sha256"]
                    }
                },
            }
            receipt["receipt_digest"] = runner.legacy.object_sha256(receipt)
            receipt_path.write_bytes(
                runner.legacy.canonical_json_bytes(receipt) + b"\n"
            )
            video_identity = guided_runner.artifact_identity(video)
            receipt_identity = guided_runner.artifact_identity(receipt_path)
            runner.seal_published_bundle_read_only(
                video,
                path,
                receipt_path,
                expected_video_identity=video_identity,
                expected_clean_latent_identity=identity,
                expected_receipt_identity=receipt_identity,
            )
            reopened = runner.reopen_published_bundle_read_only(
                video,
                path,
                receipt_path,
                expected_video_identity=video_identity,
                expected_clean_latent_identity=identity,
                expected_receipt_identity=receipt_identity,
            )
            self.assertTrue(reopened["all_three_mode_0444"])

    def test_runtime_hash_registry_includes_runner_core_adapter_and_assets(self) -> None:
        hashes = runner.runtime_source_hashes()
        for relative in (
            "infer_saic_source_state_flow_transport_v1.py",
            "saic_source_state_flow_transport_v1.py",
            "saic_native_source_state_field_v1.py",
            "dclr_runtime_contract.py",
            "source_aligned_controller.py",
            "tri_branch_unipc.py",
            "source_kv_replay.py",
            "source_kv_route_batches.py",
            "infer_source_value_residual_oracle.py",
            "source_value_residual.py",
            "materialize_saic_source_clean_latent_v1.py",
            "tools/build_renderer_dataset.py",
            "assets/saic_reversible_source_set_v1.json",
            "assets/saic_pure_t2v_event_bank_v1.json",
        ):
            key = f"methods/bernini_action_editing/{relative}"
            self.assertRegex(hashes[key], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
