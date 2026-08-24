from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import latent_temporal_event_critic_dataset as dataset  # noqa: E402
import run_starc_core4_critic_pilot_v1 as runner  # noqa: E402


try:
    import torch  # noqa: E402
    from safetensors.torch import save_file  # noqa: E402
except ImportError:  # pragma: no cover - dependency-light local host
    torch = None
    save_file = None


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seal(value: dict) -> dict:
    return {**value, "receipt_digest": runner.object_sha256(value)}


def write_json(path: Path, value: dict) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return runner.file_sha256(path)


def denial_fields() -> dict:
    return {
        "training_performed": False,
        "optimizer_authorized": False,
        "editor_optimizer_authorized": False,
        "scientific_critic_claim_authorized": False,
        "generated_media_editor_use_authorized": False,
    }


def common_materializer_fields() -> dict:
    return {
        "root_spec_binding": {"digest": sha_text("root-spec")},
        "bank_binding": {"digest": sha_text("bank")},
        "detached_event_label_binding": {
            "labels_are_external_and_detached": True,
            "labels_may_enter_model_condition": False,
        },
        "critic_use_binding": {
            "authorized_use": dataset.CRITIC_ONLY_USE,
        },
        "model_binding": {"digest": sha_text("model")},
        "runtime_binding": {"digest": sha_text("runtime")},
    }


def build_valid_live_vjp(
    root: Path,
    *,
    graph: runner.PilotManifestGraph,
    config_receipt: dict,
    checkpoint_receipt: dict,
    config_receipt_path: Path,
    checkpoint_receipt_path: Path,
) -> dict[str, object]:
    source = METHOD_ROOT / "starch_live_vjp_bridge_v1.py"
    revision = "1" * 40
    archive = root / "live-bridge-source.tar"
    source_bytes = source.read_bytes()
    with tarfile.open(
        archive,
        mode="w",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": revision},
    ) as handle:
        info = tarfile.TarInfo(runner.LIVE_VJP_BRIDGE_ARCHIVE_MEMBER)
        info.size = len(source_bytes)
        info.mode = 0o644
        info.mtime = 0
        handle.addfile(info, io.BytesIO(source_bytes))

    checkpoint_root = root / "bernini-checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "weights").mkdir()
    checkpoint_payloads = {
        "config.json": b"{}\n",
        "weights/model.bin": b"frozen Bernini test weights\n",
    }
    manifest_lines = []
    for relative, payload in sorted(checkpoint_payloads.items()):
        destination = checkpoint_root / relative
        destination.write_bytes(payload)
        manifest_lines.append(
            f"{hashlib.sha256(payload).hexdigest()}  ./{relative}"
        )
    checkpoint_content_manifest = root / "bernini-checkpoint.sha256"
    checkpoint_content_manifest.write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii"
    )
    checkpoint_manifest_sha = runner.file_sha256(checkpoint_content_manifest)
    checkpoint_entries = [
        {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
        for relative, payload in sorted(checkpoint_payloads.items())
    ]

    instruction_sha = sha_text("the actor completes the requested motion")
    clean_sha = sha_text("current-clean-latent")
    candidate_manifest = seal(
        {
            "schema_version": runner.LIVE_VJP_CANDIDATE_SCHEMA,
            "candidate_id": "current-rv2v-candidate",
            "source_video_sha256": sha_text("source-video"),
            "instruction_sha256": instruction_sha,
            "current_clean_latent_tensor_sha256": clean_sha,
            "latent_shape": [1, 16, 21, 60, 62],
            "patch_order": "phase_major_then_patch_row_major",
            "external_inference_inputs": ["source_video", "instruction"],
            "auxiliary_spatial_inputs": [],
        }
    )
    candidate_manifest_path = root / "current-candidate.json"
    candidate_manifest_sha = write_json(candidate_manifest_path, candidate_manifest)
    candidate_sketch = runner.reconstruct_geometry_spatial_sketch_binding(30, 31)
    rank_evidence = [
        {
            "rank": index,
            "shape": [1, 21, 16, 1536],
            "action_digest": sha_text(f"rank-gradient-{index}"),
            "noop_digest": sha_text(f"rank-noop-gradient-{index}"),
            "norm": 0.25 + index,
            "finite_nonzero": True,
            "action_is_exact_negative_noop": True,
        }
        for index in range(4)
    ]
    sp4_proof = {
        "world_size": 4,
        "implementation": runner.LIVE_VJP_SP4_IMPLEMENTATION,
        "rank_local_hidden_global_shape": [1, 21, 930, 1536],
        "autograd_collective_tensor_shape": [1, 21, 16, 1536],
        "dynamic_spatial_sketch_critic_tensor_sha256": candidate_sketch[
            "critic_tensor_sha256"
        ],
        "preflight_replica_contract_digest": sha_text("replica-contract"),
        "replica_graph_input_consensus_observed": True,
        "replicated_score_consensus_digest": sha_text("replicated-score"),
        "all_rank_hidden_backward_evidence_digest": runner.object_sha256(
            {
                "schema_version": "bernini-starc-all-rank-hidden-vjp-v2",
                "ordered_rank_evidence": rank_evidence,
            }
        ),
        "forward_autograd_connected": True,
        "backward_reached_all_rank_local_hidden_shards": True,
        "detached_or_object_collective_used": False,
        "ordered_rank_hidden_backward_evidence": rank_evidence,
        "rank_gradient_tensor_digests": [row["action_digest"] for row in rank_evidence],
    }
    sp4_proof["proof_digest"] = runner.object_sha256(sp4_proof)
    path = root / "vjp.json"
    receipt = seal(
        {
            "schema_version": runner.LIVE_VJP_BINDING_SCHEMA,
            "critic_binding": {
                "checkpoint_path": checkpoint_receipt["checkpoint_path"],
                "checkpoint_file_sha256": checkpoint_receipt[
                    "checkpoint_file_sha256"
                ],
                "checkpoint_state_content_digest": checkpoint_receipt[
                    "checkpoint_state_content_digest"
                ],
                "checkpoint_receipt_path": str(checkpoint_receipt_path),
                "checkpoint_receipt_file_sha256": runner.file_sha256(
                    checkpoint_receipt_path
                ),
                "checkpoint_receipt_digest": checkpoint_receipt["receipt_digest"],
                "config_receipt_path": str(config_receipt_path),
                "config_receipt_file_sha256": runner.file_sha256(config_receipt_path),
                "config_receipt_digest": config_receipt["receipt_digest"],
            },
            "materializer_binding": {
                "master_path": str(graph.master_path),
                "master_file_sha256": graph.master_file_sha256,
                "master_receipt_digest": graph.master_receipt_digest,
                "population_content_digest": graph.content_digest,
            },
            "live_bridge_binding": {
                "source_path": str(source),
                "source_file_sha256": runner.file_sha256(source),
                "source_archive_path": str(archive),
                "source_archive_file_sha256": runner.file_sha256(archive),
                "source_archive_bridge_member_path": (
                    runner.LIVE_VJP_BRIDGE_ARCHIVE_MEMBER
                ),
                "source_archive_bridge_member_sha256": runner.file_sha256(source),
                "source_git_revision": revision,
                "backend_id": runner.LIVE_VJP_BACKEND_ID,
                "bernini_commit": runner.BERNINI_OFFICIAL_COMMIT,
                "veomni_commit": runner.VEOMNI_TESTED_COMMIT,
                "checkpoint_root": str(checkpoint_root),
                "checkpoint_tree_sha256": sha_text("frozen-checkpoint-tree"),
                "checkpoint_content_manifest_path": str(checkpoint_content_manifest),
                "checkpoint_content_manifest_file_sha256": checkpoint_manifest_sha,
                "checkpoint_content_verified_file_count": len(checkpoint_payloads),
                "checkpoint_content_verified_entries_digest": runner.object_sha256(
                    checkpoint_entries
                ),
                "adapter_enabled": False,
                "frozen_bernini_and_critic": True,
            },
            "current_rv2v_clean_latent": {
                "candidate_id": candidate_manifest["candidate_id"],
                "candidate_manifest_path": str(candidate_manifest_path),
                "candidate_manifest_file_sha256": candidate_manifest_sha,
                "candidate_manifest_receipt_digest": candidate_manifest[
                    "receipt_digest"
                ],
                "source_video_sha256": candidate_manifest["source_video_sha256"],
                "instruction_sha256": instruction_sha,
                "tensor_sha256": clean_sha,
                "tensor_shape": [1, 16, 21, 60, 62],
                "tensor_dtype": "torch.float32",
                "requires_grad": True,
                "generated_t2v_owner_or_target": False,
                "patch_grid_height_width": [30, 31],
                "patch_positions": 930,
                "spatial_sketch_binding": candidate_sketch,
            },
            "same_state_hidden_query": {
                "native_schedule_index": dataset.PILOT_HIDDEN_QUERY[
                    "native_schedule_index"
                ],
                "physical_sigma": dataset.PILOT_HIDDEN_QUERY["sigma"],
                "native_timestep": dataset.PILOT_HIDDEN_QUERY["native_timestep"],
                "hook_coordinate": dataset.PILOT_HIDDEN_QUERY["hook_coordinate"],
                "action_text_tensor_sha256": sha_text("action-text"),
                "noop_text_tensor_sha256": sha_text("noop-text"),
                "action_x_sigma_tensor_sha256": sha_text("shared-x-sigma"),
                "noop_x_sigma_tensor_sha256": sha_text("shared-x-sigma"),
                "action_and_noop_received_same_python_x_sigma_object": True,
                "action_and_noop_x_sigma_value_equal": True,
                "source_condition_consumed": False,
            },
            "sp4_differentiable_collective_proof": sp4_proof,
            "gradient_audit": {
                "tensor_sha256": sha_text("current-clean-gradient"),
                "tensor_shape": [1, 16, 21, 60, 62],
                "tensor_dtype": "torch.float32",
                "gradient_norm": 0.125,
                "minimum_norm": 1.0e-12,
                "finite": True,
                "nonzero": True,
                "reached_current_rv2v_clean_latent": True,
            },
            "generated_t2v_target_consumed": False,
            "editor_parameter_or_optimizer_present": False,
            "editor_optimizer_authorized": False,
            "scientific_critic_claim_authorized": False,
        }
    )
    return {
        "receipt_path": path,
        "receipt_file_sha256": write_json(path, receipt),
        "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
        "checkpoint_content_file_count": len(checkpoint_payloads),
        "checkpoint_tree_sha256": sha_text("frozen-checkpoint-tree"),
    }


class MaterializerFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.master_path = root / runner.MASTER_FILENAME
        self.group_paths: dict[str, Path] = {}
        self.master = self._build()
        self.master_sha = write_json(self.master_path, self.master)

    def _build(self) -> dict:
        group_episode_ids = {
            "sp4-a": ("fit-dog", "confirmation-dog"),
            "sp4-b": ("fit-human", "confirmation-human"),
        }
        split_by_episode = {
            "fit-dog": "fit",
            "confirmation-dog": "confirmation",
            "fit-human": "fit",
            "confirmation-human": "confirmation",
        }
        latent_shape_by_episode = {
            "fit-dog": [1, 16, 21, 60, 62],
            "confirmation-dog": [1, 16, 21, 64, 58],
            "fit-human": [1, 16, 21, 68, 54],
            "confirmation-human": [1, 16, 21, 60, 62],
        }
        sketch_by_episode = {
            episode_id: runner.reconstruct_geometry_spatial_sketch_binding(
                shape[3] // 2, shape[4] // 2
            )
            for episode_id, shape in latent_shape_by_episode.items()
        }
        group_bindings = []
        episode_order = []
        for group_id in runner.GROUP_ORDER:
            group_root = self.root / group_id
            group_root.mkdir()
            episodes = group_episode_ids[group_id]
            episode_order.extend(episodes)
            arm_bindings = []
            for episode_id in episodes:
                split = split_by_episode[episode_id]
                episode_root = group_root / episode_id
                episode_root.mkdir()
                clean_path = episode_root / "native-clean.safetensors"
                clean_path.write_bytes(
                    f"authenticated-clean:{group_id}:{episode_id}".encode("ascii")
                )
                clean_file_sha = runner.file_sha256(clean_path)
                clean_raw_sha = sha_text(f"clean-raw:{episode_id}")
                clean_content_sha = sha_text(f"clean-content:{episode_id}")
                clean_tensor_sha = sha_text(f"clean-tensor:{episode_id}")
                clean_shape = latent_shape_by_episode[episode_id]
                clean_numel = 1
                for extent in clean_shape:
                    clean_numel *= extent
                clean_auth_unsigned = {
                    "shape": clean_shape,
                    "dtype": "torch.float32",
                    "numel": clean_numel,
                    "byte_count": clean_numel * 4,
                    "raw_value_sha256": clean_raw_sha,
                    "content_sha256": clean_content_sha,
                    "authenticated_container_path": str(clean_path),
                    "authenticated_container_sha256": clean_file_sha,
                    "single_tensor_container_reopened_byte_exact": True,
                    "safetensors_metadata": {
                        "coordinate": "bernini_normalized_clean_vae_latent",
                        "frame_contract": "exact81_latent21",
                        "artifact_role": "native_sampler_proposal",
                        "source": "native_sampler_before_vae_decode",
                    },
                    "historical_native_coordinate_role_roundtrip_verified": True,
                    "recorded_value_hashes_present": False,
                    "historical_native_receipt_value_hashes_absent": True,
                    "strict_recorded_value_identity_verified": False,
                    "native_receipt_value_hashes_synthesized": False,
                    "producer_time_value_digest_claimed_by_materializer": False,
                    "observed_value_hashes_recomputed_after_authenticated_reopen": True,
                    "value_identity_observation_time": (
                        "materializer_authenticated_reopen"
                    ),
                    "identity_authority": (
                        "authenticated_single_tensor_container_sha256_and_native_fp32_roundtrip"
                    ),
                }
                clean_authentication = {
                    **clean_auth_unsigned,
                    "binding_digest": runner.object_sha256(clean_auth_unsigned),
                }
                for role in dataset.ARM_ROLES:
                    arm_root = group_root / episode_id / role
                    arm_root.mkdir()
                    artifact_path = arm_root / "starc-block15-hidden-residual.safetensors"
                    artifact_path.write_bytes(
                        f"placeholder:{group_id}:{episode_id}:{role}".encode("ascii")
                    )
                    artifact_file_sha = runner.file_sha256(artifact_path)
                    artifact_tensor_sha = sha_text(
                        f"tensor:{group_id}:{episode_id}:{role}"
                    )
                    receipt_path = (
                        arm_root / "starc-block15-hidden-arm-receipt-v1.json"
                    )
                    receipt = seal(
                        {
                            "schema_version": runner.ARM_SCHEMA,
                            "group_id": group_id,
                            "episode_id": episode_id,
                            "split": split,
                            "role": role,
                            "label": 1 if role == "positive" else 0,
                            "action_family_id": "action-family",
                            "actor_group_id": "actor-group",
                            "scene_group_id": "scene-group",
                            "action_group_id": "action-group",
                            "seed": 17,
                            "source_candidate_binding": {
                                "digest": sha_text(f"source:{episode_id}:{role}")
                            },
                            "event_label_binding": {
                                "labels_are_external_and_detached": True,
                                "labels_may_enter_model_condition": False,
                            },
                            "critic_use_binding": {
                                "authorized_use": dataset.CRITIC_ONLY_USE,
                            },
                            "latent_binding": {
                                "path": str(clean_path),
                                "file_sha256": clean_file_sha,
                                "tensor_key": "normalized_clean_latent",
                                "stored_dtype": "torch.float32",
                                "shape": clean_shape,
                                "raw_value_sha256": clean_raw_sha,
                                "content_sha256": clean_content_sha,
                                "tensor_sha256": clean_tensor_sha,
                                "clean_latent_authentication": clean_authentication,
                                "source_shape": clean_shape,
                                "transformed_shape": clean_shape,
                                "temporal_transform": dataset.TEMPORAL_TRANSFORM_BY_ROLE[
                                    role
                                ],
                                "transform_applied_before_noising": True,
                                "transformed_tensor_sha256": sha_text(
                                    f"transformed:{episode_id}:{role}"
                                ),
                                "generated_clean_latent_used_only_as_frozen_hidden_query": True,
                            },
                            "official_gaussian_binding": {
                                "digest": sha_text(f"gaussian:{episode_id}:{role}")
                            },
                            "prompt_binding": {
                                "digest": sha_text(f"prompt:{episode_id}:{role}")
                            },
                            "same_state_query_binding": {
                                "native_schedule_index": dataset.PILOT_HIDDEN_QUERY[
                                    "native_schedule_index"
                                ],
                                "sigma": dataset.PILOT_HIDDEN_QUERY["sigma"],
                                "native_timestep": dataset.PILOT_HIDDEN_QUERY[
                                    "native_timestep"
                                ],
                                "action_and_noop_share_exact_x_sigma_object": True,
                                "action_and_noop_share_exact_rotary_object": True,
                                "action_and_noop_share_exact_timestep_object": True,
                                "shared_tensor_bytes_unchanged": True,
                                "block0_input_and_attn1_exact_parity": True,
                            },
                            "hidden_binding": {
                                "hook_coordinate": dataset.PILOT_HIDDEN_QUERY[
                                    "hook_coordinate"
                                ],
                                "action_global_sketch_shape": list(
                                    runner.RESIDUAL_SHAPE
                                ),
                                "noop_global_sketch_shape": list(
                                    runner.RESIDUAL_SHAPE
                                ),
                                "residual_shape": list(runner.RESIDUAL_SHAPE),
                                "patch_positions": (
                                    latent_shape_by_episode[episode_id][3]
                                    * latent_shape_by_episode[episode_id][4]
                                    // 4
                                ),
                                "patch_grid_height_width": [
                                    latent_shape_by_episode[episode_id][3] // 2,
                                    latent_shape_by_episode[episode_id][4] // 2,
                                ],
                                "full_hidden_persisted": False,
                            },
                            "spatial_sketch_binding": sketch_by_episode[episode_id],
                            "artifact": {
                                "path": str(artifact_path),
                                "file_sha256": artifact_file_sha,
                                "tensor_key": runner.RESIDUAL_TENSOR_KEY,
                                "tensor_shape": list(runner.RESIDUAL_SHAPE),
                                "tensor_dtype": runner.RESIDUAL_DTYPE,
                                "tensor_sha256": artifact_tensor_sha,
                                "detached_finite_fp32": True,
                            },
                            "model_binding": {"digest": sha_text("model")},
                            "runtime_binding": {"digest": sha_text("runtime")},
                            "model_forward_count": 2,
                            "labels_entered_model_condition": False,
                            **denial_fields(),
                        }
                    )
                    receipt_file_sha = write_json(receipt_path, receipt)
                    arm_bindings.append(
                        {
                            "episode_id": episode_id,
                            "split": split,
                            "role": role,
                            "label": 1 if role == "positive" else 0,
                            "receipt_path": str(receipt_path),
                            "receipt_file_sha256": receipt_file_sha,
                            "receipt_digest": receipt["receipt_digest"],
                            "artifact_path": str(artifact_path),
                            "artifact_file_sha256": artifact_file_sha,
                            "artifact_tensor_sha256": artifact_tensor_sha,
                        }
                    )
            group = seal(
                {
                    "schema_version": runner.GROUP_SCHEMA,
                    "group_id": group_id,
                    **common_materializer_fields(),
                    "episode_order": list(episodes),
                    "episode_splits": {
                        episode_id: split_by_episode[episode_id]
                        for episode_id in episodes
                    },
                    "arm_order": list(dataset.ARM_ROLES),
                    "spatial_sketch_bindings_by_episode": {
                        episode_id: sketch_by_episode[episode_id]
                        for episode_id in episodes
                    },
                    "candidate_count": 20,
                    "episode_count": 2,
                    "arm_count": 26,
                    "tensor_artifact_count": 26,
                    "model_forward_count": 52,
                    "arm_bindings": arm_bindings,
                    **denial_fields(),
                }
            )
            group_path = group_root / f"starc-core4-hidden-group-{group_id}-v1.json"
            group_file_sha = write_json(group_path, group)
            self.group_paths[group_id] = group_path
            group_bindings.append(
                {
                    "group_id": group_id,
                    "manifest_path": str(group_path),
                    "manifest_file_sha256": group_file_sha,
                    "receipt_digest": group["receipt_digest"],
                    "episode_order": list(episodes),
                    "episode_splits": {
                        episode_id: split_by_episode[episode_id]
                        for episode_id in episodes
                    },
                    "arm_count": 26,
                    "model_forward_count": 52,
                }
            )
        return seal(
            {
                "schema_version": runner.MASTER_SCHEMA,
                **common_materializer_fields(),
                "group_order": list(runner.GROUP_ORDER),
                "group_bindings": group_bindings,
                "episode_order": episode_order,
                "episode_splits": split_by_episode,
                "arm_order": list(dataset.ARM_ROLES),
                "spatial_sketch_bindings_by_episode": sketch_by_episode,
                "candidate_count": 40,
                "episode_count": 4,
                "arm_count": 52,
                "tensor_artifact_count": 52,
                "model_forward_count": 104,
                "fit_episode_count": 2,
                "confirmation_episode_count": 2,
                "confirmation_consumed_by_optimizer": False,
                **denial_fields(),
            }
        )

    def rewrite_master(self) -> None:
        self.master = seal(
            {key: value for key, value in self.master.items() if key != "receipt_digest"}
        )
        self.master_sha = write_json(self.master_path, self.master)


class FixedProtocolTests(unittest.TestCase):
    def test_hyperparameters_are_preregistered_and_not_cli_tunable(self) -> None:
        self.assertEqual(runner.FIXED_SEED, 20260808031)
        self.assertEqual(runner.FIXED_OPTIMIZER_STEPS, 200)
        self.assertEqual(runner.FIXED_HYPERPARAMETERS["learning_rate"], 2.0e-4)
        self.assertEqual(runner.FIXED_HYPERPARAMETERS["weight_decay"], 1.0e-2)
        self.assertEqual(
            runner.FIXED_HYPERPARAMETERS["maximum_gradient_norm"], 1.0
        )
        options = {
            option
            for action in runner.build_parser()._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--seed",
            "--optimizer-steps",
            "--learning-rate",
            "--weight-decay",
            "--maximum-gradient-norm",
            "--minimum-margin",
            "--hook-coordinate",
            "--layer",
            "--live-vjp-receipt",
            "--expected-live-vjp-receipt-sha256",
        ):
            self.assertNotIn(forbidden, options)

    def test_two_stage_cli_keeps_live_vjp_out_of_fit_evaluate(self) -> None:
        parser = runner.build_parser()
        subparser_action = next(
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
        )
        fit_options = {
            option
            for action in subparser_action.choices["fit-evaluate"]._actions
            for option in action.option_strings
        }
        finalize_options = {
            option
            for action in subparser_action.choices["finalize"]._actions
            for option in action.option_strings
        }
        self.assertNotIn("--live-vjp-receipt", fit_options)
        self.assertNotIn("--expected-live-vjp-receipt-sha256", fit_options)
        self.assertIn("--live-vjp-receipt", finalize_options)
        self.assertIn("--expected-live-vjp-receipt-sha256", finalize_options)

    def test_orchestration_opens_confirmation_only_after_fit_and_reload(self) -> None:
        text = (METHOD_ROOT / "run_starc_core4_critic_pilot_v1.py").read_text(
            encoding="utf-8"
        )
        fit_at = text.index("fit_trace = train_fixed_fit_cells")
        reload_at = text.index("frozen_critic, checkpoint_receipt = save_reload_final_checkpoint")
        confirmation_at = text.index(
            'confirmation_episodes = load_split_tensors(\n        graph, split="confirmation"'
        )
        self.assertLess(fit_at, reload_at)
        self.assertLess(reload_at, confirmation_at)
        self.assertIn("only_final_checkpoint_saved", text)
        self.assertIn("confirmation_samples_consumed_by_optimizer", text)
        self.assertNotIn("best_model", text)


class ManifestGraphTests(unittest.TestCase):
    def test_all_three_geometry_specific_sketch_digests_are_pinned(self) -> None:
        expected = {
            (30, 31): (
                "5a75404b60cadddb29ac7473fc4596d7ebfcd306acfb3fa1a6bc6575a228a246",
                "be43863f6a000fb00083798610e3993200c24e5fd94dcb2ef7d4e3858618dde7",
                "4a8330c77079671f6515bda07acc21f0d060176c4c07d2609ad2553acf657561",
            ),
            (32, 29): (
                "260d47275c7d407512ff4fca9fa20d2223eaa29b6e4d151b7495e51721980df4",
                "9fdee154009d0d4283716a4e93abe4df2dde5241065040eaf05bd2c9a9f2fa64",
                "be52cac4d90f0a5a70368d25fef2fb1edb4d346fb10598329f5bb7e8e7285ede",
            ),
            (34, 27): (
                "f48f9577ec829cc67bd5f9da09721bebccec7e6c92b18f5322e25ab76f19192a",
                "d05582d93963ae8de876171526f00671b7fbe0ca27841b1ab4c32b196afbc911",
                "9cc6e96d5909542189ca43ea2ff54efda6a44b302483890629b82d2ecad7f7ba",
            ),
        }
        for grid, digests in expected.items():
            binding = runner.reconstruct_geometry_spatial_sketch_binding(*grid)
            self.assertEqual(
                (
                    binding["matrix_raw_bytes_sha256"],
                    binding["matrix_value_sha256"],
                    binding["critic_tensor_sha256"],
                ),
                digests,
            )

    def test_authenticates_master_group_arm_graph_without_opening_tensor_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = MaterializerFixture(Path(directory).resolve())
            graph = runner.StarcMaterializerAdapter.load(
                fixture.master_path, expected_master_sha256=fixture.master_sha
            )
            self.assertEqual(graph.group_order, runner.GROUP_ORDER)
            self.assertEqual(len(graph.arms), 52)
            self.assertEqual(graph.episode_ids("fit"), ("fit-dog", "fit-human"))
            self.assertEqual(
                graph.episode_ids("confirmation"),
                ("confirmation-dog", "confirmation-human"),
            )
            self.assertEqual(len(graph.arms_for_split("fit")), 26)
            self.assertRegex(graph.content_digest, r"^[0-9a-f]{64}$")

    def test_historical_clean_binding_denies_synthesized_producer_hash_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = MaterializerFixture(Path(directory).resolve())
            group = json.loads(
                fixture.group_paths["sp4-a"].read_text(encoding="ascii")
            )
            arm_path = Path(group["arm_bindings"][0]["receipt_path"])
            arm = json.loads(arm_path.read_text(encoding="ascii"))
            latent = arm["latent_binding"]
            for field, replacement in (
                ("recorded_value_hashes_present", True),
                ("historical_native_receipt_value_hashes_absent", False),
                ("native_receipt_value_hashes_synthesized", True),
                ("producer_time_value_digest_claimed_by_materializer", True),
                ("value_identity_observation_time", "producer_time"),
            ):
                with self.subTest(field=field):
                    authentication = dict(latent["clean_latent_authentication"])
                    authentication[field] = replacement
                    unsigned = {
                        key: value
                        for key, value in authentication.items()
                        if key != "binding_digest"
                    }
                    authentication["binding_digest"] = runner.object_sha256(unsigned)
                    with self.assertRaisesRegex(
                        runner.StarcCriticPilotError,
                        "historical clean latent authentication differs",
                    ):
                        runner._validate_historical_clean_latent_authentication(
                            authentication,
                            latent=latent,
                            source_shape=latent["source_shape"],
                            label="fixture arm",
                        )

    def test_group_to_arm_artifact_cross_binding_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = MaterializerFixture(Path(directory).resolve())
            group_path = fixture.group_paths["sp4-a"]
            group = json.loads(group_path.read_text(encoding="ascii"))
            unsigned = {key: value for key, value in group.items() if key != "receipt_digest"}
            unsigned["arm_bindings"][0]["artifact_file_sha256"] = sha_text(
                "substituted artifact"
            )
            group = seal(unsigned)
            group_file_sha = write_json(group_path, group)
            fixture.master["group_bindings"][0]["manifest_file_sha256"] = group_file_sha
            fixture.master["group_bindings"][0]["receipt_digest"] = group[
                "receipt_digest"
            ]
            fixture.rewrite_master()
            with self.assertRaisesRegex(
                runner.StarcCriticPilotError, "tensor contract differs"
            ):
                runner.StarcMaterializerAdapter.load(
                    fixture.master_path,
                    expected_master_sha256=fixture.master_sha,
                )

    def test_confirmation_authority_escalation_fails_even_when_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = MaterializerFixture(Path(directory).resolve())
            group_path = fixture.group_paths["sp4-a"]
            group = json.loads(group_path.read_text(encoding="ascii"))
            group_unsigned = {
                key: value for key, value in group.items() if key != "receipt_digest"
            }
            confirmation_binding = group_unsigned["arm_bindings"][13]
            receipt_path = Path(confirmation_binding["receipt_path"])
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            receipt_unsigned = {
                key: value for key, value in receipt.items() if key != "receipt_digest"
            }
            receipt_unsigned["editor_optimizer_authorized"] = True
            receipt = seal(receipt_unsigned)
            confirmation_binding["receipt_file_sha256"] = write_json(
                receipt_path, receipt
            )
            confirmation_binding["receipt_digest"] = receipt["receipt_digest"]
            group = seal(group_unsigned)
            group_file_sha = write_json(group_path, group)
            fixture.master["group_bindings"][0]["manifest_file_sha256"] = group_file_sha
            fixture.master["group_bindings"][0]["receipt_digest"] = group[
                "receipt_digest"
            ]
            fixture.rewrite_master()
            with self.assertRaisesRegex(
                runner.StarcCriticPilotError, "editor_optimizer authority"
            ):
                runner.StarcMaterializerAdapter.load(
                    fixture.master_path,
                    expected_master_sha256=fixture.master_sha,
                )

    def test_geometry_specific_sketch_substitution_fails_when_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = MaterializerFixture(Path(directory).resolve())
            group_path = fixture.group_paths["sp4-a"]
            group = json.loads(group_path.read_text(encoding="ascii"))
            group_unsigned = {
                key: value for key, value in group.items() if key != "receipt_digest"
            }
            arm_binding = group_unsigned["arm_bindings"][13]
            receipt_path = Path(arm_binding["receipt_path"])
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            receipt_unsigned = {
                key: value for key, value in receipt.items() if key != "receipt_digest"
            }
            # This arm is P928; substituting the valid P930 family member must
            # fail even if every enclosing JSON receipt is freshly resealed.
            receipt_unsigned["spatial_sketch_binding"] = (
                runner.reconstruct_geometry_spatial_sketch_binding(30, 31)
            )
            receipt = seal(receipt_unsigned)
            arm_binding["receipt_file_sha256"] = write_json(receipt_path, receipt)
            arm_binding["receipt_digest"] = receipt["receipt_digest"]
            group = seal(group_unsigned)
            group_file_sha = write_json(group_path, group)
            fixture.master["group_bindings"][0][
                "manifest_file_sha256"
            ] = group_file_sha
            fixture.master["group_bindings"][0]["receipt_digest"] = group[
                "receipt_digest"
            ]
            fixture.rewrite_master()
            with self.assertRaisesRegex(
                runner.StarcCriticPilotError, "geometry-specific spatial sketch"
            ):
                runner.StarcMaterializerAdapter.load(
                    fixture.master_path,
                    expected_master_sha256=fixture.master_sha,
                )


class HeldoutGateTests(unittest.TestCase):
    @staticmethod
    def outputs(*, weak_role: str | None = None) -> dict:
        result = {}
        for episode_id in ("confirmation-dog", "confirmation-human"):
            role_outputs = {}
            for role in dataset.ARM_ROLES:
                score = 1.0 if role == "positive" else 0.0
                if episode_id == "confirmation-human" and role == weak_role:
                    score = 0.85
                role_outputs[role] = {
                    "score": score,
                    "milestone_scores": {
                        "actor_object_binding": score,
                        "transition": score,
                        "chronology": score,
                        "terminal_hold": score,
                    },
                }
            result[episode_id] = role_outputs
        return result

    def test_gate_emits_exact_24_noncompensating_margins(self) -> None:
        gate = runner.make_heldout_margin_gate(
            self.outputs(),
            expected_episode_ids=("confirmation-dog", "confirmation-human"),
        )
        self.assertEqual(gate["margin_count"], 24)
        self.assertEqual(gate["passed_margin_count"], 24)
        self.assertTrue(gate["all_24_role_margins_passed"])
        self.assertFalse(gate["role_margins_averaged_or_compensated"])
        self.assertTrue(
            gate["worth_fixed_topup_generation_recommended_by_this_gate"]
        )
        self.assertFalse(gate["editor_optimizer_authorized"])

    def test_one_weak_role_fails_whole_margin_gate(self) -> None:
        gate = runner.make_heldout_margin_gate(
            self.outputs(weak_role="semantic_wrong_actor"),
            expected_episode_ids=("confirmation-dog", "confirmation-human"),
        )
        self.assertEqual(gate["passed_margin_count"], 23)
        self.assertFalse(gate["all_24_role_margins_passed"])
        self.assertFalse(
            gate["worth_fixed_topup_generation_recommended_by_this_gate"]
        )
        self.assertIn(
            "margin:confirmation-human:semantic_wrong_actor",
            gate["failure_reasons"],
        )

    def test_missing_live_vjp_is_explicit_fail_closed(self) -> None:
        gate = runner.validate_live_vjp_receipt(None, None)
        self.assertFalse(gate["provided"])
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["reason"],
            "live_current_rv2v_input_vjp_composite_receipt_missing",
        )

    def test_valid_live_vjp_is_checkpoint_runtime_and_tensor_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture = MaterializerFixture(root)
            graph = runner.StarcMaterializerAdapter.load(
                fixture.master_path, expected_master_sha256=fixture.master_sha
            )
            critic_path = root / "critic.safetensors"
            critic_path.write_bytes(b"critic checkpoint")
            config = {"receipt_digest": sha_text("critic-config")}
            config_path = root / "critic-config.json"
            config_sha = write_json(config_path, config)
            checkpoint = {
                "checkpoint_path": str(critic_path),
                "checkpoint_file_sha256": runner.file_sha256(critic_path),
                "checkpoint_state_content_digest": sha_text("critic-state"),
                "receipt_digest": sha_text("critic-checkpoint-receipt"),
            }
            checkpoint_receipt_path = root / "critic-checkpoint-receipt.json"
            checkpoint_receipt_sha = write_json(
                checkpoint_receipt_path, checkpoint
            )
            live = build_valid_live_vjp(
                root,
                graph=graph,
                config_receipt=config,
                checkpoint_receipt=checkpoint,
                config_receipt_path=config_path,
                checkpoint_receipt_path=checkpoint_receipt_path,
            )
            with mock.patch.object(
                runner,
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
                live["checkpoint_content_manifest_sha256"],
            ), mock.patch.object(
                runner,
                "BERNINI_CHECKPOINT_CONTENT_FILE_COUNT",
                live["checkpoint_content_file_count"],
            ), mock.patch.object(
                runner,
                "BERNINI_CHECKPOINT_TREE_SHA256",
                live["checkpoint_tree_sha256"],
            ):
                binding = runner.validate_live_vjp_receipt(
                    live["receipt_path"],
                    live["receipt_file_sha256"],
                    graph=graph,
                    config_receipt=config,
                    checkpoint_receipt=checkpoint,
                    config_receipt_path=config_path,
                    config_receipt_file_sha256=config_sha,
                    checkpoint_receipt_path=checkpoint_receipt_path,
                    checkpoint_receipt_file_sha256=checkpoint_receipt_sha,
                )
            self.assertTrue(binding["provided"])
            self.assertTrue(binding["passed"])
            self.assertTrue(binding["checkpoint_and_runtime_bound"])
            self.assertEqual(
                binding["receipt_file_sha256"], live["receipt_file_sha256"]
            )
            decoded = json.loads(Path(live["receipt_path"]).read_text(encoding="ascii"))
            collective = decoded["sp4_differentiable_collective_proof"]
            self.assertEqual(
                collective["rank_local_hidden_global_shape"],
                [1, 21, 930, 1536],
            )
            self.assertEqual(
                collective["autograd_collective_tensor_shape"],
                [1, 21, 16, 1536],
            )
            self.assertEqual(
                decoded["current_rv2v_clean_latent"][
                    "candidate_manifest_file_sha256"
                ],
                binding["current_candidate_manifest_file_sha256"],
            )

    def test_old_bare_live_vjp_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture = MaterializerFixture(root)
            graph = runner.StarcMaterializerAdapter.load(
                fixture.master_path, expected_master_sha256=fixture.master_sha
            )
            path = root / "bare-vjp.json"
            bare = {
                "schema_version": "bernini-ltec-current-clean-latent-gradient-audit-v1",
                "gradient_shape": [1, 16, 21, 60, 62],
                "gradient_norm": 1.0,
                "finite": True,
                "nonzero": True,
                "minimum_norm": 1.0e-12,
                "passed": True,
                "generated_t2v_target_consumed": False,
                "editor_optimizer_authorized": False,
            }
            expected = write_json(path, bare)
            config = {"receipt_digest": sha_text("config")}
            config_path = root / "config.json"
            config_sha = write_json(config_path, config)
            checkpoint = {
                "checkpoint_file_sha256": sha_text("checkpoint"),
                "checkpoint_state_content_digest": sha_text("state"),
                "receipt_digest": sha_text("checkpoint-receipt"),
            }
            checkpoint_path = root / "checkpoint-receipt.json"
            checkpoint_sha = write_json(checkpoint_path, checkpoint)
            with self.assertRaisesRegex(runner.StarcCriticPilotError, "digest"):
                runner.validate_live_vjp_receipt(
                    path,
                    expected,
                    graph=graph,
                    config_receipt=config,
                    checkpoint_receipt=checkpoint,
                    config_receipt_path=config_path,
                    config_receipt_file_sha256=config_sha,
                    checkpoint_receipt_path=checkpoint_path,
                    checkpoint_receipt_file_sha256=checkpoint_sha,
                )


class TwoStageFinalizerTests(unittest.TestCase):
    def test_finalize_replays_json_only_and_never_reopens_residual_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            materializer_root = root / "materializer"
            materializer_root.mkdir()
            fixture = MaterializerFixture(materializer_root)
            graph = runner.StarcMaterializerAdapter.load(
                fixture.master_path, expected_master_sha256=fixture.master_sha
            )

            pilot_root = root / "pilot"
            pilot_root.mkdir()
            critic_config = dict(runner.GEOMETRY_NEUTRAL_CRITIC_CONFIG)
            config_receipt = seal(
                {
                    "schema_version": runner.CONFIG_SCHEMA,
                    "run_schema_version": runner.SCHEMA_VERSION,
                    "fixed_hyperparameters": dict(runner.FIXED_HYPERPARAMETERS),
                    "fixed_hyperparameter_digest": runner.object_sha256(
                        runner.FIXED_HYPERPARAMETERS
                    ),
                    "critic_config": critic_config,
                    "critic_config_content_digest": runner.object_sha256(
                        critic_config
                    ),
                    "pre_sketched_head_contract": {
                        "entrypoint": "forward_sketched_residual_only",
                        "input_shape": list(runner.RESIDUAL_SHAPE),
                        "geometry_neutral_after_fixed_sketch": True,
                        "constructor_spatial_buffer": "inert_16x16_identity_never_consumed",
                        "constructor_spatial_buffer_checkpointed": False,
                        "full_hidden_forward_authorized": False,
                        "geometry_specific_sketches_authenticated_by_materializer": True,
                        "trainable_parameter_count": 128,
                    },
                    "materializer_master_path": str(graph.master_path),
                    "materializer_master_file_sha256": graph.master_file_sha256,
                    "materializer_master_receipt_digest": graph.master_receipt_digest,
                    "materialized_population_content_digest": graph.content_digest,
                    "spatial_sketch_bindings_by_episode": {
                        episode_id: dict(binding)
                        for episode_id, binding in graph.spatial_sketch_bindings_by_episode.items()
                    },
                    "fit_episode_order": list(graph.episode_ids("fit")),
                    "confirmation_episode_order": list(
                        graph.episode_ids("confirmation")
                    ),
                    "confirmation_tensor_load_phase": "after_step200_checkpoint_reload_and_freeze",
                    "nuisance_basis_used": False,
                    "core4_scientific_claim_authorized": False,
                    "editor_optimizer_present_or_authorized": False,
                }
            )
            config_path = pilot_root / runner.CONFIG_FILENAME
            config_sha = write_json(config_path, config_receipt)

            fit_ids = list(graph.episode_ids("fit"))
            trace_receipt = seal(
                {
                    "schema_version": runner.TRACE_SCHEMA,
                    "fixed_hyperparameters": dict(runner.FIXED_HYPERPARAMETERS),
                    "fit_episode_order": fit_ids,
                    "fit_artifact_count": 26,
                    "optimizer_step_count": runner.FIXED_OPTIMIZER_STEPS,
                    "both_fit_cells_consumed_every_step": True,
                    "confirmation_manifest_metadata_authenticated_before_fit": True,
                    "confirmation_tensor_artifacts_opened_before_fit_complete": False,
                    "confirmation_samples_consumed_by_optimizer": False,
                    "checkpoint_selection": "final_step_200_only",
                    "best_checkpoint_saved": False,
                    "early_stopping_performed": False,
                    "editor_parameter_present": False,
                    "steps": [
                        {
                            "step": step,
                            "loss": 0.0,
                            "gradient_norm_before_clip": 1.0,
                            "minimum_fit_group_margin": 1.0,
                            "episode_ids": fit_ids,
                        }
                        for step in range(1, runner.FIXED_OPTIMIZER_STEPS + 1)
                    ],
                }
            )
            trace_path = pilot_root / runner.TRACE_FILENAME
            trace_sha = write_json(trace_path, trace_receipt)

            checkpoint_path = pilot_root / runner.CHECKPOINT_FILENAME
            checkpoint_path.write_bytes(b"synthetic-head-only-checkpoint")
            checkpoint_receipt = seal(
                {
                    "schema_version": runner.CHECKPOINT_SCHEMA,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_file_sha256": runner.file_sha256(checkpoint_path),
                    "checkpoint_state_content_digest": sha_text(
                        "checkpoint-state-content"
                    ),
                    "checkpoint_tensor_count": 2,
                    "checkpoint_scope": "geometry_neutral_pre_sketched_critic_head_only",
                    "excluded_constructor_buffer_keys": list(
                        runner.NON_HEAD_STATE_KEYS
                    ),
                    "config_receipt_digest": config_receipt["receipt_digest"],
                    "optimizer_step": runner.FIXED_OPTIMIZER_STEPS,
                    "only_final_checkpoint_saved": True,
                    "best_checkpoint_saved": False,
                    "confirmation_sample_seen_before_checkpoint_save": False,
                    "state_tensor_byte_parity_after_fresh_load": True,
                    "fit_score_parity_after_fresh_load": True,
                    "critic_frozen_after_reload": True,
                    "editor_checkpoint_or_parameter_present": False,
                    "editor_optimizer_authorized": False,
                }
            )
            checkpoint_receipt_path = (
                pilot_root / runner.CHECKPOINT_RECEIPT_FILENAME
            )
            checkpoint_receipt_sha = write_json(
                checkpoint_receipt_path, checkpoint_receipt
            )

            outputs = HeldoutGateTests.outputs()
            heldout = runner.make_heldout_margin_gate(
                outputs,
                expected_episode_ids=graph.episode_ids("confirmation"),
            )
            provisional_receipt = seal(
                {
                    "schema_version": runner.PROVISIONAL_GATE_SCHEMA,
                    "run_schema_version": runner.SCHEMA_VERSION,
                    "gate_stage": "fit-evaluate-provisional",
                    "finalization_required": True,
                    "materializer_binding": {
                        "master_path": str(graph.master_path),
                        "master_file_sha256": graph.master_file_sha256,
                        "master_receipt_digest": graph.master_receipt_digest,
                        "group_manifest_file_sha256s": list(
                            graph.group_manifest_file_sha256s
                        ),
                        "group_receipt_digests": list(
                            graph.group_receipt_digests
                        ),
                        "population_content_digest": graph.content_digest,
                        "spatial_sketch_bindings_by_episode": {
                            episode_id: dict(binding)
                            for episode_id, binding in graph.spatial_sketch_bindings_by_episode.items()
                        },
                    },
                    "config_binding": {
                        "path": str(config_path),
                        "file_sha256": config_sha,
                        "receipt_digest": config_receipt["receipt_digest"],
                    },
                    "fit_trace_binding": {
                        "path": str(trace_path),
                        "file_sha256": trace_sha,
                        "receipt_digest": trace_receipt["receipt_digest"],
                        "optimizer_step_count": runner.FIXED_OPTIMIZER_STEPS,
                    },
                    "checkpoint_binding": {
                        "receipt_path": str(checkpoint_receipt_path),
                        "receipt_file_sha256": checkpoint_receipt_sha,
                        "receipt_digest": checkpoint_receipt["receipt_digest"],
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_file_sha256": checkpoint_receipt[
                            "checkpoint_file_sha256"
                        ],
                        "checkpoint_state_content_digest": checkpoint_receipt[
                            "checkpoint_state_content_digest"
                        ],
                    },
                    "fit_protocol": {
                        "fit_episode_order": fit_ids,
                        "both_fit_cells_consumed_every_step": True,
                        "optimizer_steps": runner.FIXED_OPTIMIZER_STEPS,
                        "only_final_checkpoint_saved": True,
                        "early_stopping_performed": False,
                        "confirmation_samples_consumed_by_optimizer": False,
                    },
                    "confirmation_protocol": {
                        "confirmation_episode_order": list(
                            graph.episode_ids("confirmation")
                        ),
                        "tensor_artifact_count": 26,
                        "critic_forward_count": 26,
                        "each_tensor_artifact_read_once": True,
                        "evaluated_once_after_checkpoint_reload_and_freeze": True,
                        "used_for_checkpoint_threshold_layer_or_hyperparameter_selection": False,
                        "outputs": outputs,
                    },
                    "heldout_margin_gate": heldout,
                    "live_current_rv2v_input_vjp_gate": runner.validate_live_vjp_receipt(
                        None, None
                    ),
                    "worth_fixed_topup_generation": False,
                    "scientific_critic_claim_authorized": False,
                    "action_editing_success_claim_authorized": False,
                    "editor_optimizer_present": False,
                    "editor_optimizer_authorized": False,
                    "generated_rgb_or_latent_used_as_editor_target_condition_donor_or_noise": False,
                    "failure_reasons": [
                        *heldout["failure_reasons"],
                        "live_current_rv2v_input_vjp_composite_receipt_pending",
                    ],
                }
            )
            provisional_path = (
                pilot_root / runner.PROVISIONAL_GATE_RECEIPT_FILENAME
            )
            provisional_sha = write_json(provisional_path, provisional_receipt)
            live = build_valid_live_vjp(
                root,
                graph=graph,
                config_receipt=config_receipt,
                checkpoint_receipt=checkpoint_receipt,
                config_receipt_path=config_path,
                checkpoint_receipt_path=checkpoint_receipt_path,
            )
            final_root = root / "final"
            with mock.patch.object(
                runner,
                "_runtime_modules",
                side_effect=AssertionError("finalize imported tensor runtime"),
            ), mock.patch.object(
                runner,
                "_load_residual_tensor",
                side_effect=AssertionError("finalize reopened residual tensor"),
            ), mock.patch.object(
                runner,
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
                live["checkpoint_content_manifest_sha256"],
            ), mock.patch.object(
                runner,
                "BERNINI_CHECKPOINT_CONTENT_FILE_COUNT",
                live["checkpoint_content_file_count"],
            ), mock.patch.object(
                runner,
                "BERNINI_CHECKPOINT_TREE_SHA256",
                live["checkpoint_tree_sha256"],
            ):
                final = runner.run_finalize(
                    pilot_output_dir=pilot_root,
                    expected_provisional_gate_sha256=provisional_sha,
                    live_vjp_receipt=live["receipt_path"],
                    expected_live_vjp_receipt_sha256=live[
                        "receipt_file_sha256"
                    ],
                    output_dir=final_root,
                )
            self.assertTrue(final["worth_fixed_topup_generation"])
            self.assertTrue(final["live_current_rv2v_input_vjp_gate"]["passed"])
            self.assertFalse(final["finalization_retrained_critic"])
            self.assertFalse(
                final["finalization_loaded_fit_or_confirmation_tensor_artifact"]
            )
            self.assertFalse(
                final["finalization_changed_threshold_layer_or_hyperparameter"]
            )
            self.assertTrue((final_root / runner.FINAL_GATE_RECEIPT_FILENAME).is_file())

            # Restore write permission solely so TemporaryDirectory can clean up.
            os.chmod(final_root, 0o700)
            os.chmod(final_root / runner.FINAL_GATE_RECEIPT_FILENAME, 0o600)


@unittest.skipIf(torch is None, "torch/safetensors are unavailable")
class TensorAndCheckpointPrimitiveTests(unittest.TestCase):
    def test_exact_residual_loader_checks_file_and_value_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "residual.safetensors"
            tensor = torch.linspace(
                -1.0,
                1.0,
                steps=1 * 21 * 16 * 1536,
                dtype=torch.float32,
            ).reshape(runner.RESIDUAL_SHAPE)
            save_file({runner.RESIDUAL_TENSOR_KEY: tensor}, str(path))
            tensor_digest = runner.materializer_tensor_sha256(tensor)
            sketch_binding = runner.reconstruct_geometry_spatial_sketch_binding(
                30, 31
            )
            binding = runner.ArmArtifactBinding(
                group_id="sp4-a",
                episode_id="fit-dog",
                split="fit",
                role="positive",
                label=1,
                receipt_path=path,
                receipt_file_sha256=runner.file_sha256(path),
                receipt_digest=sha_text("receipt"),
                artifact_path=path,
                artifact_file_sha256=runner.file_sha256(path),
                artifact_tensor_sha256=tensor_digest,
                tensor_key=runner.RESIDUAL_TENSOR_KEY,
                tensor_shape=runner.RESIDUAL_SHAPE,
                tensor_dtype=runner.RESIDUAL_DTYPE,
                source_latent_shape=(1, 16, 21, 60, 62),
                patch_grid_height_width=(30, 31),
                spatial_sketch_binding=sketch_binding,
            )
            loaded = runner._load_residual_tensor(binding, device="cpu")
            self.assertTrue(torch.equal(loaded, tensor))

    def test_state_digest_and_exact_parity_detect_mutation(self) -> None:
        state = {
            "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "ids": torch.arange(3, dtype=torch.int64),
        }
        repeated = {name: value.clone() for name, value in state.items()}
        self.assertEqual(
            runner.checkpoint_state_content_digest(state),
            runner.checkpoint_state_content_digest(repeated),
        )
        runner._assert_exact_state_parity(state, repeated)
        reordered = {"ids": repeated["ids"], "weight": repeated["weight"]}
        runner._assert_exact_state_parity(state, reordered)
        missing = {"weight": repeated["weight"]}
        with self.assertRaisesRegex(
            runner.StarcCriticPilotError, "key closure differs after reload"
        ):
            runner._assert_exact_state_parity(state, missing)
        mutated = deepcopy(repeated)
        mutated["weight"][0, 0] += 1.0
        with self.assertRaisesRegex(
            runner.StarcCriticPilotError, "differs after reload"
        ):
            runner._assert_exact_state_parity(state, mutated)


if __name__ == "__main__":
    unittest.main()
