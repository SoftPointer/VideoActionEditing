from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_full644_camera_stabilizer as inference  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


SHA1 = "1" * 40
SHA256 = "2" * 64


def _args(**overrides) -> argparse.Namespace:
    values = {
        "bernini_root": "/vendor/bernini",
        "veomni_root": "/vendor/veomni",
        "checkpoint": "/checkpoint/Bernini-R-1.3B",
        "adapter_checkpoint": "/checkpoint/full644/checkpoint-00000644",
        "source_video": "/data/source.mp4",
        "instruction": "Make the dog sit and turn its head right.",
        "output": "/output/result.mp4",
        "beta": 0.25,
        "camera_estimator": "global_svd",
        "num_inference_steps": 40,
        "seed": 2027,
        "expected_source_sha256": "3" * 64,
        "expected_instruction_sha256": "4" * 64,
        "expected_bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA256,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _adapter_identity() -> dict[str, object]:
    return {
        "checkpoint_root": "/checkpoint/full644/checkpoint-00000644",
        "adapter_config_sha256": inference.FULL644_ADAPTER_CONFIG_SHA256,
        "adapter_model_sha256": inference.FULL644_ADAPTER_SHA256,
        "training_receipt_file_sha256": (
            inference.FULL644_TRAINING_RECEIPT_FILE_SHA256
        ),
        "training_receipt_digest": inference.FULL644_TRAINING_RECEIPT_DIGEST,
        "training_global_step": 644,
        "training_method_source_revision": "5" * 40,
        "training_method_source_archive_sha256": "6" * 64,
        "target_module_count": 240,
        "adapter_tensor_count": 480,
    }


def _core_step_trace(beta: float, *, bypassed: bool) -> dict[str, object]:
    return {
        "schema_version": inference.camera.SCHEMA_VERSION,
        "method": inference.camera.METHOD_NAME,
        "bypassed": bypassed,
        "bypass_reason": "zero_beta" if beta == 0.0 else (
            "all_phases_degenerate" if bypassed else None
        ),
        "beta_mode": "scalar",
        "beta_per_phase": [[beta] * 21],
        "basis_built": not bypassed,
        "basis_reused": not bypassed,
        "source_basis_detached": not bypassed,
        "retained_rank": None if bypassed else [[8] * 21],
        "condition_number": None if bypassed else [[2.0] * 21],
        "valid_phase": None if bypassed else [[True] * 21],
        "camera_component_rms_before": None if bypassed else [[0.2] * 21],
        "camera_component_rms_after": None if bypassed else [[0.1] * 21],
        "source_camera_component_rms": None if bypassed else [[0.1] * 21],
        "applied_correction_rms": None if bypassed else [[0.01] * 21],
        "noncamera_invariance_max_abs": 0.0,
        "noncamera_invariance_rms": 0.0,
        "noncamera_invariance_tolerance": 0.00004,
        "invariant_satisfied": True,
    }


def _grid_core_step_trace(beta: float, *, bypassed: bool) -> dict[str, object]:
    return {
        "schema_version": inference.grid_stabilizer.SCHEMA_VERSION,
        "method": inference.grid_stabilizer.METHOD_NAME,
        "bypassed": bypassed,
        "bypass_reason": "zero_beta" if bypassed else None,
        "beta_mode": "scalar",
        "beta_per_phase": [[beta] * 21],
        "geometry_built": False,
        "geometry_reused": not bypassed,
        "estimator": "fixed_grid_median_MAD_trimmed_robust_consensus",
        "consensus_scope": "independent_per_batch_and_latent_phase",
        "consensus_valid": None if bypassed else [[True] * 21],
        "correction_rms": None if bypassed else [[0.01] * 21],
        "geometry_valid_tile_count": None if bypassed else [[16] * 21],
        "fit_valid_tile_count": None if bypassed else [[14] * 21],
        "inlier_tile_count": None if bypassed else [[12] * 21],
        "spatial_coverage_valid": None if bypassed else [[True] * 21],
        "consensus_coefficient_max_abs": (
            None if bypassed else [[0.03] * 21]
        ),
        "tile_relative_fit_residual_max": (
            None if bypassed else [[0.20] * 21]
        ),
        "invalid_phases_exact_action": True,
    }


def _grid_rank_authority(
    beta: float,
    *,
    step_index: int,
    world_size: int = 4,
    group_rank: int = 0,
) -> dict[str, object]:
    if beta == 0.0:
        return {
            "schema_version": inference.GRID_RANK_AUTHORITY_SCHEMA,
            "mode": "zero_beta_exact_action_no_collective",
            "authority_group": "torch.distributed.group.WORLD",
            "world_size": world_size,
            "group_rank": group_rank,
            "process_group_rank0": 0,
            "rank0_authoritative_broadcast": False,
            "source_clean_cross_rank_exact": None,
            "source_clean_rank0_reference_exact": None,
            "source_clean_certified_this_step": False,
            "action_clean_cross_rank_exact": None,
            "action_clean_rank0_reference_exact": None,
            "local_candidate_success_all_ranks": None,
            "pre_broadcast_max_abs_disagreement": None,
            "post_broadcast_exact": None,
            "post_broadcast_proof": None,
            "exact_comparison_proof": (
                inference.GRID_RANK_AUTHORITY_EXACT_PROOF
            ),
            "all_rank_participant_count": 0,
            "collective_sequence": [],
        }
    sequence = []
    if step_index == 0:
        sequence.extend(
            [
                "source_rank0_reference_broadcast",
                "source_rank0_reference_exact_all_reduce_min",
            ]
        )
    sequence.extend(
        [
            "action_rank0_reference_broadcast",
            "action_rank0_reference_exact_all_reduce_min",
            "candidate_success_all_reduce_min",
            "executed_clean_broadcast_from_process_group_rank0",
            "pre_broadcast_disagreement_all_reduce_max",
        ]
    )
    return {
        "schema_version": inference.GRID_RANK_AUTHORITY_SCHEMA,
        "mode": "distributed_rank0_authoritative",
        "authority_group": "torch.distributed.group.WORLD",
        "world_size": world_size,
        "group_rank": group_rank,
        "process_group_rank0": 0,
        "rank0_authoritative_broadcast": True,
        "source_clean_cross_rank_exact": True,
        "source_clean_rank0_reference_exact": True,
        "source_clean_certified_this_step": step_index == 0,
        "action_clean_cross_rank_exact": True,
        "action_clean_rank0_reference_exact": True,
        "local_candidate_success_all_ranks": True,
        "pre_broadcast_max_abs_disagreement": float(step_index) * 1e-7,
        "post_broadcast_exact": True,
        "post_broadcast_proof": (
            "rank0_broadcast_then_all_rank_disagreement_MAX_completed"
        ),
        "exact_comparison_proof": inference.GRID_RANK_AUTHORITY_EXACT_PROOF,
        "all_rank_participant_count": world_size,
        "collective_sequence": sequence,
    }


def _valid_execution(
    beta: float = 0.25,
    *,
    estimator: str = "global_svd",
):
    branches = []
    cameras = []
    for index in range(40):
        sigma = 1.0 - index / 41.0
        timestep = 1000.0 - index
        branches.append(
            tri.TriBranchStepRecord(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                model_id="transformer_1",
                transformer_forwards=3,
                shared_negative_forwards=1,
                action_forwards=1,
                noop_forwards=1,
                original_scheduler_calls=1,
                callback_correction_rms=0.01,
                raw_action_noop_delta_rms=0.1,
                guided_action_noop_delta_rms=0.2,
                guided_action_noop_delta_l2=2.0,
                action_noop_exact_parity=False,
                effective_guidance_scale=4.0,
                official_action_parity_rms_error=0.0,
                official_action_parity_max_abs_error=0.0,
                official_action_exact_parity=True,
                sample_dtype="torch.float32",
                branch_velocity_dtype="torch.bfloat16",
                official_model_output_dtype="torch.float32",
            )
        )
        grid = estimator == "grid_consensus"
        cameras.append(
            inference.CameraStepRecord(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                beta=beta,
                action_passthrough_object_exact=(beta == 0.0),
                basis_built_this_step=(
                    not grid and beta != 0.0 and index == 0
                ),
                basis_reused_from_prior_step=(
                    not grid and beta != 0.0 and index > 0
                ),
                core_trace=(
                    _grid_core_step_trace(beta, bypassed=(beta == 0.0))
                    if grid
                    else _core_step_trace(beta, bypassed=(beta == 0.0))
                ),
                estimator=estimator,
                geometry_built_this_step=(
                    grid and beta != 0.0 and index == 0
                ),
                geometry_reused_from_prior_step=(
                    grid and beta != 0.0 and index > 0
                ),
                grid_rank_authority=(
                    _grid_rank_authority(beta, step_index=index)
                    if grid
                    else None
                ),
            )
        )
    return (
        tri.TriBranchTrace(records=branches, sample_calls=1),
        inference.CameraExecutionTrace(
            beta=beta,
            estimator=estimator,
            records=cameras,
        ),
    )


def _legacy_receipt() -> dict[str, object]:
    value = {
        "schema_version": inference.legacy.INFERENCE_RECEIPT_SCHEMA,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA256,
        "adapter": {
            "adapter_model_sha256": inference.FULL644_ADAPTER_SHA256,
            "training_receipt_digest": inference.FULL644_TRAINING_RECEIPT_DIGEST,
            "training_global_step": 644,
            "tensor_count": 480,
            "strictly_reloaded": True,
            "safe_merged_for_inference": True,
        },
        "input": {
            "accepted_model_conditions": ["source_video", "edit_instruction"],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "external_mask_or_swept_tube": False,
            "external_tracking_pose_or_trajectory": False,
            "reference_image_or_video": False,
            "external_shared_i0": False,
        },
        "sampling": inference.exact_sampler_contract(),
        "prompt_contract": {},
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    value["receipt_digest"] = inference.object_sha256(value)
    return value


class _TraceReceipt:
    def __init__(self, value: dict[str, object]):
        self.value = value

    def to_receipt(self) -> dict[str, object]:
        return dict(self.value)


class Full644CameraInferenceContractTests(unittest.TestCase):
    def test_cli_is_source_instruction_full644_and_beta_only(self) -> None:
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertTrue(
            {
                "bernini_root",
                "veomni_root",
                "checkpoint",
                "adapter_checkpoint",
                "source_video",
                "instruction",
                "output",
                "beta",
                "camera_estimator",
                "method_source_revision",
                "method_source_archive_sha256",
            }
            <= destinations
        )
        forbidden = {
            "target",
            "target_video",
            "support",
            "support_video",
            "mask",
            "flow",
            "optical_flow",
            "pose",
            "track",
            "swept_tube",
            "trajectory",
            "reference",
            "reference_image",
            "reference_video",
            "edited_first_frame",
            "first_frame_anchor",
        }
        self.assertTrue(destinations.isdisjoint(forbidden))
        parsed = parser.parse_args(
            [
                "--bernini-root",
                "/b",
                "--veomni-root",
                "/v",
                "--checkpoint",
                "/c",
                "--adapter-checkpoint",
                "/a",
                "--source-video",
                "/s.mp4",
                "--instruction",
                "sit",
                "--output",
                "/o.mp4",
                "--beta",
                "0.2",
                "--expected-source-sha256",
                "3" * 64,
                "--expected-instruction-sha256",
                "4" * 64,
                "--method-source-revision",
                SHA1,
                "--method-source-archive-sha256",
                SHA256,
            ]
        )
        self.assertEqual(parsed.num_inference_steps, 40)
        self.assertEqual(parsed.seed, 2027)
        self.assertEqual(parsed.camera_estimator, "global_svd")

        grid_arguments = [
            "--camera-estimator",
            "grid_consensus",
        ] + [
            item
            for pair in (
                ("--bernini-root", "/b"),
                ("--veomni-root", "/v"),
                ("--checkpoint", "/c"),
                ("--adapter-checkpoint", "/a"),
                ("--source-video", "/s.mp4"),
                ("--instruction", "sit"),
                ("--output", "/o.mp4"),
                ("--beta", "0.2"),
                ("--expected-source-sha256", "3" * 64),
                ("--expected-instruction-sha256", "4" * 64),
                ("--method-source-revision", SHA1),
                ("--method-source-archive-sha256", SHA256),
            )
            for item in pair
        ]
        self.assertEqual(
            parser.parse_args(grid_arguments).camera_estimator,
            "grid_consensus",
        )

    def test_cli_locks_beta_solver_seed_and_provenance(self) -> None:
        inference.validate_cli(_args())
        for changed in (
            {"beta": -0.1},
            {"beta": 1.1},
            {"beta": float("nan")},
            {"num_inference_steps": 39},
            {"seed": 42},
            {"camera_estimator": "unknown"},
            {"expected_source_sha256": "bad"},
            {"method_source_revision": "bad"},
            {"source_video": "relative.mp4"},
        ):
            with self.subTest(changed=changed), self.assertRaises(
                inference.CameraStabilizerInferenceError
            ):
                inference.validate_cli(_args(**changed))

    def test_sampler_is_exact_81_frame_40_step_seed_2027_unipc(self) -> None:
        value = inference.exact_sampler_contract()
        self.assertEqual(value["num_frames"], 81)
        self.assertEqual(value["num_inference_steps"], 40)
        self.assertEqual(value["seed"], 2027)
        self.assertEqual(value["guidance_mode"], "v2v_apg")
        self.assertEqual(value["flow_shift"], 5.0)

    def test_full644_validator_pins_weights_config_receipt_and_step(self) -> None:
        bundle = SimpleNamespace(
            checkpoint_root=Path("/a"),
            adapter_config_path=Path("/a/adapter/adapter_config.json"),
            adapter_model_path=Path("/a/adapter/adapter_model.safetensors"),
            training_receipt_path=Path("/a/receipt.json"),
        )
        hashes = {
            bundle.adapter_config_path: inference.FULL644_ADAPTER_CONFIG_SHA256,
            bundle.adapter_model_path: inference.FULL644_ADAPTER_SHA256,
            bundle.training_receipt_path: (
                inference.FULL644_TRAINING_RECEIPT_FILE_SHA256
            ),
        }
        receipt = {
            "receipt_digest": inference.FULL644_TRAINING_RECEIPT_DIGEST,
            "global_step": 644,
            "target_module_count": 240,
            "method_source_revision": "5" * 40,
            "method_source_archive_sha256": "6" * 64,
        }
        with mock.patch.object(
            inference.legacy,
            "file_sha256",
            side_effect=lambda path: hashes[path],
        ), mock.patch.object(inference.legacy, "_read_json", return_value=receipt):
            identity = inference.validate_full644_adapter_bundle(bundle)
        self.assertEqual(
            identity["adapter_model_sha256"],
            inference.FULL644_ADAPTER_SHA256,
        )
        self.assertEqual(identity["target_module_count"], 240)
        self.assertEqual(identity["adapter_tensor_count"], 480)

        bad_hashes = dict(hashes)
        bad_hashes[bundle.adapter_model_path] = "0" * 64
        with mock.patch.object(
            inference.legacy,
            "file_sha256",
            side_effect=lambda path: bad_hashes[path],
        ), mock.patch.object(inference.legacy, "_read_json", return_value=receipt):
            with self.assertRaises(inference.CameraStabilizerInferenceError):
                inference.validate_full644_adapter_bundle(bundle)

    def test_callback_calls_core_and_beta_zero_is_exact_object_passthrough(self) -> None:
        source = object()
        action = object()
        noop = object()
        fields = SimpleNamespace(
            step_index=0,
            timestep=999.0,
            sigma=0.99,
            action_guided_clean=action,
            noop_guided_clean=noop,
        )
        callback = inference.CameraTangentCallback(
            source_clean_field=source, beta=0.0
        )
        result = SimpleNamespace(
            executed_clean_field=action,
            trace=_TraceReceipt(_core_step_trace(0.0, bypassed=True)),
        )
        with mock.patch.object(
            inference.camera, "stabilize_camera_tangent", return_value=result
        ) as stabilize, mock.patch.object(
            inference.camera, "build_camera_tangent_basis"
        ) as build_basis:
            self.assertIs(callback(fields), action)
        build_basis.assert_not_called()
        stabilize.assert_called_once_with(
            source,
            action,
            noop,
            beta=0.0,
            enabled=True,
            camera_edit_requested=False,
            config=callback.config,
            precomputed_basis=None,
        )
        self.assertTrue(callback.trace.records[0].action_passthrough_object_exact)

        callback = inference.CameraTangentCallback(
            source_clean_field=source, beta=0.0
        )
        wrong = SimpleNamespace(
            executed_clean_field=object(),
            trace=_TraceReceipt(_core_step_trace(0.0, bypassed=True)),
        )
        with mock.patch.object(
            inference.camera, "stabilize_camera_tangent", return_value=wrong
        ), self.assertRaisesRegex(
            inference.CameraStabilizerInferenceError, "object identity"
        ):
            callback(fields)

    def test_grid_callback_beta_zero_is_exact_without_geometry_or_noop_input(self) -> None:
        source = object()
        action = object()
        fields = SimpleNamespace(
            step_index=0,
            timestep=999.0,
            sigma=0.99,
            action_guided_clean=action,
            noop_guided_clean=object(),
        )
        callback = inference.CameraGridConsensusCallback(
            source_clean_field=source,
            beta=0.0,
        )
        result = SimpleNamespace(
            executed_clean_field=action,
            trace=_TraceReceipt(_grid_core_step_trace(0.0, bypassed=True)),
        )
        with mock.patch.object(
            inference.grid_stabilizer,
            "stabilize_camera_consensus",
            return_value=result,
        ) as stabilize, mock.patch.object(
            inference.grid_camera,
            "build_fixed_grid_camera_geometry",
        ) as build_geometry:
            self.assertIs(callback(fields), action)
        build_geometry.assert_not_called()
        stabilize.assert_called_once_with(
            source,
            action,
            beta=0.0,
            config=callback.config,
            precomputed_geometry=None,
        )
        record = callback.trace.records[0]
        self.assertEqual(record.estimator, "grid_consensus")
        self.assertFalse(record.geometry_built_this_step)
        self.assertFalse(record.geometry_reused_from_prior_step)
        self.assertEqual(
            record.grid_rank_authority["mode"],
            "zero_beta_exact_action_no_collective",
        )
        self.assertEqual(record.grid_rank_authority["collective_sequence"], [])
        self.assertFalse(
            record.grid_rank_authority["rank0_authoritative_broadcast"]
        )

    def test_trace_certifies_parity_120_forwards_and_beta_zero_control(self) -> None:
        tri_trace, camera_trace = _valid_execution(beta=0.0)
        value = inference.validate_execution_trace(tri_trace, camera_trace)
        certificate = value["certificate"]
        self.assertEqual(certificate["step_count"], 40)
        self.assertEqual(certificate["transformer_forwards"], 120)
        self.assertEqual(certificate["official_action_apg_exact_steps"], 40)
        self.assertEqual(certificate["original_unipc_calls"], 40)
        self.assertEqual(certificate["camera_basis_build_count"], 0)
        self.assertEqual(certificate["camera_basis_reuse_count"], 0)
        self.assertTrue(certificate["beta_zero_exact_full644_passthrough"])
        self.assertFalse(certificate["custom_integrator"])
        self.assertRegex(value["trace_digest"], r"^[0-9a-f]{64}$")

        tri_trace.records[4] = replace(
            tri_trace.records[4], official_action_exact_parity=False
        )
        with self.assertRaisesRegex(
            inference.CameraStabilizerInferenceError, "exact parity"
        ):
            inference.validate_execution_trace(tri_trace, camera_trace)

        tri_trace, camera_trace = _valid_execution(beta=0.5)
        active = inference.validate_execution_trace(tri_trace, camera_trace)
        self.assertEqual(active["certificate"]["camera_basis_build_count"], 1)
        self.assertEqual(active["certificate"]["camera_basis_reuse_count"], 39)
        self.assertEqual(active["certificate"]["camera_geometry_build_count"], 0)
        self.assertEqual(active["certificate"]["camera_geometry_reuse_count"], 0)

        for beta in (0.0, 0.5):
            with self.subTest(grid_beta=beta):
                tri_trace, camera_trace = _valid_execution(
                    beta=beta,
                    estimator="grid_consensus",
                )
                grid = inference.validate_execution_trace(
                    tri_trace,
                    camera_trace,
                )
                certificate = grid["certificate"]
                self.assertEqual(certificate["transformer_forwards"], 120)
                self.assertEqual(certificate["official_action_apg_exact_steps"], 40)
                self.assertEqual(certificate["original_unipc_calls"], 40)
                self.assertEqual(certificate["camera_estimator"], "grid_consensus")
                self.assertEqual(certificate["camera_basis_build_count"], 0)
                self.assertEqual(certificate["camera_basis_reuse_count"], 0)
                self.assertEqual(
                    certificate["camera_geometry_build_count"],
                    0 if beta == 0.0 else 1,
                )
                self.assertEqual(
                    certificate["camera_geometry_reuse_count"],
                    0 if beta == 0.0 else 39,
                )
                self.assertEqual(
                    certificate["camera_estimator_state_build_count"],
                    0 if beta == 0.0 else 1,
                )
                self.assertEqual(
                    certificate["camera_estimator_state_reuse_count"],
                    0 if beta == 0.0 else 39,
                )
                authority = certificate["grid_rank_authority"]
                self.assertEqual(authority["world_size"], 4)
                self.assertEqual(authority["receipt_group_rank"], 0)
                self.assertEqual(authority["rank0_receipt_aggregated_steps"], 40)
                if beta == 0.0:
                    self.assertEqual(authority["zero_beta_no_collective_steps"], 40)
                    self.assertEqual(
                        authority["rank0_authoritative_broadcast_steps"], 0
                    )
                else:
                    self.assertTrue(authority["source_clean_cross_rank_exact"])
                    self.assertEqual(
                        authority["source_clean_exact_certification_steps"], 1
                    )
                    self.assertEqual(
                        authority["action_clean_cross_rank_exact_steps"], 40
                    )
                    self.assertEqual(
                        authority["rank0_authoritative_broadcast_steps"], 40
                    )
                    self.assertEqual(authority["post_broadcast_exact_steps"], 40)
                    self.assertEqual(
                        len(
                            authority[
                                "pre_broadcast_max_abs_disagreement_by_step"
                            ]
                        ),
                        40,
                    )

    def test_grid_trace_rejects_missing_or_forged_rank_authority(self) -> None:
        tri_trace, camera_trace = _valid_execution(
            beta=0.5,
            estimator="grid_consensus",
        )
        camera_trace.records[7] = replace(
            camera_trace.records[7],
            grid_rank_authority=None,
        )
        with self.assertRaisesRegex(
            inference.CameraStabilizerInferenceError,
            "rank authority evidence",
        ):
            inference.validate_execution_trace(tri_trace, camera_trace)

        tri_trace, camera_trace = _valid_execution(
            beta=0.5,
            estimator="grid_consensus",
        )
        forged = dict(camera_trace.records[9].grid_rank_authority)
        forged["post_broadcast_exact"] = False
        camera_trace.records[9] = replace(
            camera_trace.records[9],
            grid_rank_authority=forged,
        )
        with self.assertRaisesRegex(
            inference.CameraStabilizerInferenceError,
            "authority evidence differs",
        ):
            inference.validate_execution_trace(tri_trace, camera_trace)

    def test_grid_receipt_rejects_non_rank0_execution_summary(self) -> None:
        tri_trace, camera_trace = _valid_execution(
            beta=0.5,
            estimator="grid_consensus",
        )
        for index, record in enumerate(camera_trace.records):
            evidence = dict(record.grid_rank_authority)
            evidence["group_rank"] = 1
            camera_trace.records[index] = replace(
                record,
                grid_rank_authority=evidence,
            )
        execution = inference.validate_execution_trace(tri_trace, camera_trace)
        with mock.patch.object(
            inference,
            "_method_hashes",
            return_value={"runner": "7" * 64},
        ), self.assertRaisesRegex(
            inference.CameraStabilizerInferenceError,
            "not aggregated by Ulysses process-group rank 0",
        ):
            inference.augment_inference_receipt(
                _legacy_receipt(),
                args=_args(beta=0.5, camera_estimator="grid_consensus"),
                adapter_identity=_adapter_identity(),
                noop_identity={"frozen_t5": True},
                execution_trace=execution,
            )

    def test_augmented_receipt_binds_full644_training_and_camera_trace(self) -> None:
        tri_trace, camera_trace = _valid_execution(beta=0.0)
        execution = inference.validate_execution_trace(tri_trace, camera_trace)
        with mock.patch.object(
            inference, "_method_hashes", return_value={"runner": "7" * 64}
        ), mock.patch.object(
            inference.camera,
            "camera_stabilizer_contract_receipt",
            return_value={"schema_version": "camera-contract-v1"},
        ):
            value = inference.augment_inference_receipt(
                _legacy_receipt(),
                args=_args(beta=0.0),
                adapter_identity=_adapter_identity(),
                noop_identity={"frozen_t5": True},
                execution_trace=execution,
            )
        self.assertEqual(value["schema_version"], inference.INFERENCE_RECEIPT_SCHEMA)
        self.assertEqual(
            value["full644_adapter"]["adapter_model_sha256"],
            inference.FULL644_ADAPTER_SHA256,
        )
        self.assertEqual(
            value["full644_adapter"]["training_receipt_digest"],
            inference.FULL644_TRAINING_RECEIPT_DIGEST,
        )
        self.assertTrue(value["full644_adapter"]["strict_240_module_scope"])
        self.assertTrue(value["full644_adapter"]["strict_480_tensor_reload"])
        self.assertTrue(value["full644_adapter"]["safe_merge_before_camera_hook"])
        self.assertEqual(value["camera_stabilizer"]["beta"], 0.0)
        self.assertEqual(
            value["camera_stabilizer"]["execution"]["certificate"][
                "transformer_forwards"
            ],
            120,
        )
        self.assertEqual(
            value["input"]["accepted_external_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertFalse(value["input"]["support_accessed_by_inference"])
        unsigned = dict(value)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(inference.object_sha256(unsigned), declared)

    def test_grid_receipt_uses_robust_contract_without_orthogonal_claim(self) -> None:
        tri_trace, camera_trace = _valid_execution(
            beta=0.5,
            estimator="grid_consensus",
        )
        execution = inference.validate_execution_trace(tri_trace, camera_trace)
        with mock.patch.object(
            inference,
            "_method_hashes",
            return_value={
                "runner": "7" * 64,
                "grid_core": "8" * 64,
                "grid_wrapper": "9" * 64,
            },
        ), mock.patch.object(
            inference.camera,
            "camera_stabilizer_contract_receipt",
        ) as global_contract:
            value = inference.augment_inference_receipt(
                _legacy_receipt(),
                args=_args(beta=0.5, camera_estimator="grid_consensus"),
                adapter_identity=_adapter_identity(),
                noop_identity={"frozen_t5": True},
                execution_trace=execution,
            )
        global_contract.assert_not_called()
        camera_receipt = value["camera_stabilizer"]
        self.assertEqual(camera_receipt["estimator"], "grid_consensus")
        self.assertEqual(
            camera_receipt["execution"]["certificate"][
                "camera_geometry_build_count"
            ],
            1,
        )
        self.assertEqual(
            camera_receipt["execution"]["certificate"][
                "camera_geometry_reuse_count"
            ],
            39,
        )
        authority = camera_receipt["execution"]["certificate"][
            "grid_rank_authority"
        ]
        self.assertEqual(authority["world_size"], 4)
        self.assertEqual(authority["receipt_group_rank"], 0)
        self.assertEqual(authority["rank0_authoritative_broadcast_steps"], 40)
        self.assertEqual(authority["post_broadcast_exact_steps"], 40)
        serialized = json.dumps(camera_receipt).lower()
        self.assertNotIn("orthogonal", serialized)
        self.assertNotIn("noncamera_invariance", serialized)

    def test_method_hash_closure_includes_both_estimator_implementations(self) -> None:
        hashes = inference._method_hashes()
        self.assertEqual(
            set(hashes),
            {
                "infer_full644_camera_stabilizer.py",
                "generator_native_camera_stabilizer.py",
                "fixed_grid_camera_consensus.py",
                "fixed_grid_camera_consensus_stabilizer.py",
                "tri_branch_unipc.py",
                "infer_lora.py",
                "motion_residual.py",
            },
        )
        for digest in hashes.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_main_delegates_strict_safe_merge_then_wraps_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.mp4"
            source.touch()
            instruction = "Make the dog sit and turn its head right."
            args = _args(
                source_video=str(source),
                instruction=instruction,
                expected_instruction_sha256=inference.hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
            )
            adapter = SimpleNamespace()
            merged = SimpleNamespace()
            events: list[str] = []
            observed_legacy_argv: list[str] = []

            original_loader = inference.legacy._strict_load_and_merge_adapter
            original_tokenizer = inference.legacy._tokenize_training_prompt
            original_writer = inference.legacy._atomic_write_json

            def strict_loader(base_model, observed_adapter, expected_targets):
                del base_model
                self.assertIs(observed_adapter, adapter)
                self.assertEqual(
                    list(expected_targets),
                    inference.legacy.expected_lora_target_modules(),
                )
                events.append("legacy_strict_480_tensor_safe_merge")
                return merged, 480

            def legacy_main(argv):
                observed_legacy_argv.extend(argv)
                model, count = inference.legacy._strict_load_and_merge_adapter(
                    object(), adapter, inference.legacy.expected_lora_target_modules()
                )
                self.assertEqual(count, 480)
                self.assertIsInstance(model, inference._CameraWrappedModel)
                self.assertIs(model._model, merged)
                events.append("camera_wrapper_after_merge")
                return 19

            with mock.patch.object(
                inference, "build_parser"
            ) as parser_factory, mock.patch.object(
                inference.legacy, "_plain_file", return_value=source
            ), mock.patch.object(
                inference.legacy, "resolve_adapter_bundle", return_value=adapter
            ), mock.patch.object(
                inference.legacy,
                "file_sha256",
                return_value=args.expected_source_sha256,
            ), mock.patch.object(
                inference,
                "validate_full644_adapter_bundle",
                return_value=_adapter_identity(),
            ), mock.patch.object(
                inference.legacy,
                "_strict_load_and_merge_adapter",
                new=strict_loader,
            ), mock.patch.object(
                inference.legacy, "main", new=legacy_main
            ):
                parser_factory.return_value.parse_args.return_value = args
                status = inference.main([])

        self.assertEqual(status, 19)
        self.assertEqual(
            events,
            ["legacy_strict_480_tensor_safe_merge", "camera_wrapper_after_merge"],
        )
        self.assertIn("--adapter-checkpoint", observed_legacy_argv)
        self.assertNotIn("--target-video", observed_legacy_argv)
        self.assertIs(inference.legacy._strict_load_and_merge_adapter, original_loader)
        self.assertIs(inference.legacy._tokenize_training_prompt, original_tokenizer)
        self.assertIs(inference.legacy._atomic_write_json, original_writer)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class Full644CameraTensorIntegrationTests(unittest.TestCase):
    def test_mocked_ulysses_rank1_executes_rank0_candidate_with_audited_order(self) -> None:
        class FakeDistributed:
            class group:
                WORLD = "world4"

            class ReduceOp:
                MIN = "min"
                MAX = "max"

            def __init__(self, source, action, candidate):
                self.rank0_values = [source, action, candidate]
                self.broadcast_index = 0
                self.calls = []

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size(group=None):
                return 4

            @staticmethod
            def get_rank(group=None):
                return 1

            def broadcast(self, value, *, src, group):
                self.calls.append(("broadcast", src, group))
                value.copy_(self.rank0_values[self.broadcast_index])
                self.broadcast_index += 1

            def all_reduce(self, value, *, op, group):
                self.calls.append(("all_reduce", op, group, float(value.item())))

        source = torch.randn(1, 2, 21, 4, 5, dtype=torch.float32)
        action = torch.randn_like(source)
        rank0_candidate = action + 0.25
        local_candidate = action - 0.5
        fake = FakeDistributed(source, action, rank0_candidate)
        authority = inference._GridRank0Authority(source, dist_module=fake)

        self.assertTrue(authority.certify_source())
        self.assertTrue(authority.certify_action(action))
        self.assertTrue(
            authority.require_all_candidates_succeeded(True, reference=action)
        )
        executed, evidence = authority.execute_rank0_authoritative(
            local_candidate,
            action_exact=True,
            source_certified_this_step=True,
        )

        self.assertTrue(torch.equal(executed, rank0_candidate))
        self.assertEqual(evidence["world_size"], 4)
        self.assertEqual(evidence["group_rank"], 1)
        self.assertTrue(evidence["rank0_authoritative_broadcast"])
        self.assertTrue(evidence["source_clean_cross_rank_exact"])
        self.assertTrue(evidence["action_clean_cross_rank_exact"])
        self.assertTrue(evidence["post_broadcast_exact"])
        self.assertAlmostEqual(
            evidence["pre_broadcast_max_abs_disagreement"],
            0.75,
            places=6,
        )
        self.assertEqual(
            evidence["collective_sequence"],
            [
                "source_rank0_reference_broadcast",
                "source_rank0_reference_exact_all_reduce_min",
                "action_rank0_reference_broadcast",
                "action_rank0_reference_exact_all_reduce_min",
                "candidate_success_all_reduce_min",
                "executed_clean_broadcast_from_process_group_rank0",
                "pre_broadcast_disagreement_all_reduce_max",
            ],
        )
        self.assertEqual(
            [item[0] for item in fake.calls],
            [
                "broadcast",
                "all_reduce",
                "broadcast",
                "all_reduce",
                "all_reduce",
                "broadcast",
                "all_reduce",
            ],
        )

    def test_mocked_ulysses_source_mismatch_fails_before_grid_estimation(self) -> None:
        class MismatchDistributed:
            class group:
                WORLD = "world4"

            class ReduceOp:
                MIN = "min"

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size(group=None):
                return 4

            @staticmethod
            def get_rank(group=None):
                return 2

            @staticmethod
            def broadcast(value, *, src, group):
                value.add_(1.0)

            @staticmethod
            def all_reduce(value, *, op, group):
                return None

        source = torch.zeros(1, 1, 21, 2, 2, dtype=torch.float32)
        authority = inference._GridRank0Authority(
            source,
            dist_module=MismatchDistributed(),
        )
        with self.assertRaisesRegex(
            inference.CameraStabilizerInferenceError,
            "source_clean_field is not exact",
        ):
            authority.certify_source()

    def test_real_core_beta_zero_preserves_action_object_and_trace(self) -> None:
        source = torch.randn(1, 2, 21, 8, 8, dtype=torch.float32)
        action = torch.randn_like(source)
        noop = torch.randn_like(source)
        fields = SimpleNamespace(
            step_index=0,
            timestep=999.0,
            sigma=0.99,
            action_guided_clean=action,
            noop_guided_clean=noop,
        )
        callback = inference.CameraTangentCallback(
            source_clean_field=source, beta=0.0
        )
        self.assertIs(callback(fields), action)
        record = callback.trace.records[0]
        self.assertTrue(record.action_passthrough_object_exact)
        self.assertEqual(record.core_trace["bypass_reason"], "zero_beta")
        self.assertFalse(record.core_trace["basis_built"])
        self.assertFalse(record.core_trace["basis_reused"])

    def test_real_core_active_projection_satisfies_trace_adapter(self) -> None:
        generator = torch.Generator().manual_seed(7)
        source = torch.randn(
            1, 2, 21, 8, 8, dtype=torch.float32, generator=generator
        )
        noop = torch.randn(
            1, 2, 21, 8, 8, dtype=torch.float32, generator=generator
        )
        action = noop + 0.2 * torch.randn(
            1, 2, 21, 8, 8, dtype=torch.float32, generator=generator
        )
        fields = SimpleNamespace(
            step_index=0,
            timestep=999.0,
            sigma=0.99,
            action_guided_clean=action,
            noop_guided_clean=noop,
        )
        callback = inference.CameraTangentCallback(
            source_clean_field=source, beta=0.5
        )
        executed = callback(fields)
        self.assertIsNot(executed, action)
        record = callback.trace.records[0]
        basis = callback._precomputed_basis
        self.assertFalse(record.action_passthrough_object_exact)
        self.assertTrue(record.basis_built_this_step)
        self.assertFalse(record.basis_reused_from_prior_step)
        self.assertTrue(record.core_trace["basis_built"])
        self.assertTrue(record.core_trace["basis_reused"])
        self.assertTrue(record.core_trace["source_basis_detached"])
        self.assertTrue(record.core_trace["invariant_satisfied"])

        for index in range(1, 40):
            fields.step_index = index
            fields.timestep = 999.0 - index
            fields.sigma = 0.99 - index * 0.01
            callback(fields)
        self.assertIs(callback._precomputed_basis, basis)
        second = callback.trace.records[1]
        self.assertFalse(second.basis_built_this_step)
        self.assertTrue(second.basis_reused_from_prior_step)
        self.assertEqual(callback.trace.as_dict()["basis_build_count"], 1)
        self.assertEqual(callback.trace.as_dict()["basis_reuse_count"], 39)

    def test_real_grid_callback_zero_bypass_and_active_geometry_reuse(self) -> None:
        generator = torch.Generator().manual_seed(1709)
        source = torch.randn(
            1,
            2,
            21,
            20,
            24,
            dtype=torch.float32,
            generator=generator,
        )
        action = torch.randn(
            1,
            2,
            21,
            20,
            24,
            dtype=torch.float32,
            generator=generator,
        )
        fields = SimpleNamespace(
            step_index=0,
            timestep=999.0,
            sigma=0.99,
            action_guided_clean=action,
            noop_guided_clean=torch.randn_like(action),
        )

        zero = inference.CameraGridConsensusCallback(
            source_clean_field=source,
            beta=0.0,
        )
        self.assertIs(zero(fields), action)
        self.assertIsNone(zero._precomputed_geometry)
        self.assertEqual(zero.trace.as_dict()["geometry_build_count"], 0)
        self.assertEqual(zero.trace.as_dict()["geometry_reuse_count"], 0)

        active = inference.CameraGridConsensusCallback(
            source_clean_field=source,
            beta=0.5,
        )
        active(fields)
        geometry = active._precomputed_geometry
        first = active.trace.records[0]
        self.assertIsNotNone(geometry)
        self.assertTrue(first.geometry_built_this_step)
        self.assertFalse(first.geometry_reused_from_prior_step)
        self.assertFalse(first.basis_built_this_step)
        self.assertTrue(first.core_trace["geometry_reused"])
        self.assertNotIn("invariant_satisfied", first.core_trace)

        for index in range(1, 40):
            fields.step_index = index
            fields.timestep = 999.0 - index
            fields.sigma = 0.99 - index * 0.01
            active(fields)
        self.assertIs(active._precomputed_geometry, geometry)
        trace = active.trace.as_dict()
        self.assertEqual(trace["estimator"], "grid_consensus")
        self.assertEqual(trace["basis_build_count"], 0)
        self.assertEqual(trace["basis_reuse_count"], 0)
        self.assertEqual(trace["geometry_build_count"], 1)
        self.assertEqual(trace["geometry_reuse_count"], 39)
        authority = active.trace.records[0].grid_rank_authority
        self.assertEqual(authority["mode"], "single_process_rank0")
        self.assertEqual(authority["world_size"], 1)
        self.assertTrue(authority["source_clean_cross_rank_exact"])
        self.assertTrue(authority["action_clean_cross_rank_exact"])
        self.assertFalse(authority["rank0_authoritative_broadcast"])
        self.assertTrue(authority["post_broadcast_exact"])
        self.assertEqual(authority["collective_sequence"], [])
        self.assertTrue(
            active.trace.records[0].grid_rank_authority[
                "source_clean_certified_this_step"
            ]
        )
        self.assertFalse(
            active.trace.records[1].grid_rank_authority[
                "source_clean_certified_this_step"
            ]
        )


if __name__ == "__main__":
    unittest.main()
