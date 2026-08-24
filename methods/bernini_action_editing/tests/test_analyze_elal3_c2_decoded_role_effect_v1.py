from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from methods.bernini_action_editing.tests.test_decode_elal3_c2_simulator_oracle_q_v1 import (
    exact14_rows,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
SOURCE = METHOD_ROOT / "analyze_elal3_c2_decoded_role_effect_v1.py"
SPEC = importlib.util.spec_from_file_location("analyze_elal3_c2_decoded_role_effect_v1", SOURCE)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def portable_replay(stage: str, fixed: dict[str, object]) -> dict[str, object]:
    return {
        "stage": stage,
        "fixed_release_binding": fixed,
        "fixed_release_binding_digest": analysis.object_sha256(fixed),
        "physical_runtime_replay_passed": True,
    }


def portable_checkpoint_tree() -> dict[str, object]:
    parameter_order = [f"block.{index}.lora_a" for index in range(480)] + [
        f"block.{index}.elal3_c0_v1.weight" for index in range(188)
    ]
    inventory = []
    for index, name in enumerate(parameter_order):
        numel = 198_722_947 if index == 667 else 1
        inventory.append(
            {
                "name": name,
                "shape": [numel],
                "dtype": "torch.float32",
                "numel": numel,
            }
        )
    records = []
    for step in (0, 10):
        names = ["adapter-and-elal3.pt"]
        if step == 10:
            names.append("optimizer.pt")
        names.append("CHECKPOINT_RECEIPT.json")
        files = [
            {
                "name": name,
                "sha256": sha(f"step{step}:{name}".encode()),
                "size": index + 1,
                "mode": 0o444,
                "nlink": 1,
                "held_fd_double_hash_verified": True,
                "named_identity_replayed": True,
            }
            for index, name in enumerate(names)
        ]
        optimizer_digest = sha(b"optimizer") if step == 10 else None
        optimizer = (
            {
                "state_entry_count": 668,
                "param_group_count": 1,
                "parameter_count": 668,
                "parameter_inventory_digest": analysis.object_sha256(inventory),
                "optimizer_step": 10,
                "exp_avg_nonzero_parameter_count": 668,
                "exp_avg_sq_nonzero_parameter_count": 668,
                "state_keys_by_parameter": [
                    {
                        "parameter_id": index,
                        "state_keys": ["exp_avg", "exp_avg_sq", "step"],
                    }
                    for index in range(668)
                ],
                "tree_digest": optimizer_digest,
            }
            if step == 10
            else None
        )
        unsigned = {
            "schema_version": analysis.CHECKPOINT_SCHEMA,
            "step": step,
            "file_order": names,
            "directory_entries": names,
            "directory_mode": 0o500,
            "files": files,
            "adapter_payload_tree_digest": sha(f"adapter:{step}".encode()),
            "parameter_order": parameter_order,
            "parameter_inventory": inventory,
            "optimizer_payload_tree_digest": optimizer_digest,
            "optimizer_state_inventory": optimizer,
            "checkpoint_receipt_digest": sha(f"receipt:{step}".encode()),
            "trainable_parameter_sha256": sha(f"parameters:{step}".encode()),
            "strict_reload_pass": True,
        }
        records.append(
            {**unsigned, "portable_record_digest": analysis.object_sha256(unsigned)}
        )
    return {
        "schema_version": "bernini-elal3-c2-sealed-checkpoint-tree-v1",
        "expected_steps": [0, 10],
        "directory_entries": ["checkpoint-00000000", "checkpoint-00000010"],
        "directory_mode": 0o500,
        "portable_checkpoint_records": records,
        "portable_checkpoint_tree_digest": analysis.object_sha256(records),
        "physical_origin_replay_passed": True,
    }


def exact23_fixed() -> dict[str, object]:
    files = [
        {
            "row_index": index,
            "relative_path": relative,
            "sha256": sha(relative.encode()),
            "size": index + 1,
            "mode": 0o644,
            "nlink": 1,
            "held_fd_double_hash_verified": True,
            "held_openat_parent_chain_replayed": True,
        }
        for index, relative in enumerate(analysis.CHECKPOINT_EXACT23_RELATIVE_PATHS)
    ]
    closure_unsigned = {
        "noncache_file_count": 23,
        "noncache_files": sorted(analysis.CHECKPOINT_EXACT23_RELATIVE_PATHS),
        "noncache_directory_count": 6,
        "noncache_directories": sorted(analysis.CHECKPOINT_EXACT23_DIRECTORIES),
        "canonical_dot_cache_only_exclusion": True,
        "noncache_symlinks_rejected": True,
    }
    return {
        "manifest_relative_path": "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
        "manifest_sha256": analysis.CHECKPOINT_EXACT23_MANIFEST_SHA256,
        "manifest_size": analysis.CHECKPOINT_EXACT23_MANIFEST_SIZE,
        "file_count": 23,
        "files": files,
        "noncache_load_precedence_closure": {
            **closure_unsigned,
            "closure_digest": analysis.object_sha256(closure_unsigned),
        },
        "checkpoint_root_expected_by_renderer_and_tokenizer": True,
    }


def execution_fixed() -> dict[str, object]:
    files = [
        {
            "row_index": index,
            "relative_path": relative,
            "sha256": digest,
            "size": index + 1,
            "mode": 0o444,
            "nlink": 1,
            "held_fd_double_hash_verified": True,
            "held_openat_parent_chain_replayed": True,
        }
        for index, (relative, digest) in enumerate(
            analysis.BERNINI_EXECUTION_SHA256.items()
        )
    ]
    veomni = [
        {
            "row_index": index,
            "relative_path": relative,
            "sha256": sha(relative.encode()),
            "size": index + 1,
            "mode": 0o644,
            "nlink": 1,
            "held_fd_double_hash_verified": True,
            "held_openat_parent_chain_replayed": True,
            "actual_imported_module_file_verified": True,
        }
        for index, relative in enumerate(analysis.VEOMNI_EXECUTION_PATHS)
    ]
    return {
        "bernini_commit": analysis.BERNINI_COMMIT,
        "veomni_commit": analysis.VEOMNI_COMMIT,
        "file_count": 10,
        "files": files,
        "veomni_actual_imported_module_count": 2,
        "veomni_actual_imported_modules": veomni,
        "actual_imported_modules_and_callable_ownership_verified": True,
    }


def lerp(points: list[tuple[int, float, float]]) -> list[list[float]]:
    result = []
    for frame in range(analysis.FRAME_COUNT):
        left = points[0]
        right = points[-1]
        for begin, end in zip(points, points[1:]):
            if begin[0] <= frame <= end[0]:
                left, right = begin, end
                break
        alpha = 0.0 if left[0] == right[0] else (frame - left[0]) / float(right[0] - left[0])
        result.append([
            (1.0 - alpha) * left[1] + alpha * right[1],
            (1.0 - alpha) * left[2] + alpha * right[2],
        ])
    return result


def block_tracks(role: bool = False) -> dict[str, list[list[float]]]:
    target = {
        "agent": lerp([(0, .18, .75), (48, .39, .52), (64, .39, .52), (80, .39, .52)]),
        "patient": lerp([(0, .10, .42), (48, .35, .42), (64, .35, .42), (80, .35, .42)]),
        "object": lerp([(0, .44, .75), (48, .45, .42), (64, .45, .42), (80, .45, .42)]),
    }
    if not role:
        return target
    return {
        "agent": target["patient"],
        "patient": target["agent"],
        "object": target["object"],
    }


def hand_tracks(role: bool = False) -> dict[str, list[list[float]]]:
    target = {
        "agent": lerp([(0, .17, .52), (56, .60, .52), (64, .60, .52), (80, .60, .52)]),
        "patient": lerp([(0, .82, .52), (56, .71, .52), (64, .71, .52), (80, .71, .52)]),
        "object": lerp([(0, .24, .52), (56, .68, .52), (64, .68, .52), (80, .68, .52)]),
    }
    if not role:
        return target
    return {
        "agent": target["patient"],
        "patient": target["agent"],
        "object": target["object"],
    }


def annotation(tracks: dict[str, list[list[float]]], *, occluded_entity: str) -> dict[str, object]:
    frames = []
    for index in range(analysis.FRAME_COUNT):
        entities = []
        for entity in analysis.ENTITY_ORDER:
            point = tracks[entity][index]
            previous = tracks[entity][max(0, index - 1)]
            dx = int(round((point[0] - previous[0]) * 127.0))
            dy = int(round((point[1] - previous[1]) * 95.0))
            entities.append({
                "entity_id": entity,
                "center_xy": [int(round(point[0] * 127)), int(round(point[1] * 95))],
                "signed_track_dxdy_from_previous_frame": [dx, dy],
                "visibility_fraction": .5 if entity == occluded_entity and 38 <= index <= 44 else 1.0,
            })
        frames.append({"frame_index": index, "entities": entities})
    return {"frames": frames}


def reliable(track: dict[str, list[list[float]]]) -> dict[str, object]:
    return {
        "identity_recovery_reliable": True,
        "tracks_xy_normalized": track,
        "entities": {},
    }


class ELAL3C2DecodedAnalysisTests(unittest.TestCase):
    def test_portable_checkpoint_exact2_is_deeply_closed(self) -> None:
        tree = portable_checkpoint_tree()
        analysis.validate_portable_checkpoint_tree_receipt_v1(
            tree, label="test tree"
        )
        reloads = [
            {
                "step": record["step"],
                "adapter_sha256": record["files"][0]["sha256"],
                "trainable_parameter_sha256": record[
                    "trainable_parameter_sha256"
                ],
                "parameter_count": 198_723_614,
                "lora_tensors": 480,
                "elal3_tensors": 188,
                "strict_origin_physical_runtime_reload_verified": True,
            }
            for record in tree["portable_checkpoint_records"]
        ]
        analysis.validate_checkpoint_reload_receipts_v1(
            reloads, portable_tree=tree, label="test reload"
        )
        wrong_adapter = copy.deepcopy(reloads)
        wrong_adapter[1]["adapter_sha256"] = sha(b"resigned adapter decoy")
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "reload/portable record join"
        ):
            analysis.validate_checkpoint_reload_receipts_v1(
                wrong_adapter, portable_tree=tree, label="test reload"
            )
        injected = copy.deepcopy(tree)
        injected["portable_checkpoint_records"][0]["path"] = "/tmp/origin"
        injected["portable_checkpoint_records"][0]["portable_record_digest"] = (
            analysis.object_sha256(
                {
                    key: value
                    for key, value in injected["portable_checkpoint_records"][0].items()
                    if key != "portable_record_digest"
                }
            )
        )
        injected["portable_checkpoint_tree_digest"] = analysis.object_sha256(
            injected["portable_checkpoint_records"]
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "exact2 record"
        ):
            analysis.validate_portable_checkpoint_tree_receipt_v1(
                injected, label="test tree"
            )

        resigned_count = copy.deepcopy(tree)
        record0 = resigned_count["portable_checkpoint_records"][0]
        record0["parameter_inventory"][-1]["shape"] = [198_722_946]
        record0["parameter_inventory"][-1]["numel"] = 198_722_946
        unsigned = dict(record0)
        unsigned.pop("portable_record_digest")
        record0["portable_record_digest"] = analysis.object_sha256(unsigned)
        resigned_count["portable_checkpoint_tree_digest"] = analysis.object_sha256(
            resigned_count["portable_checkpoint_records"]
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "parameter count"
        ):
            analysis.validate_portable_checkpoint_tree_receipt_v1(
                resigned_count, label="test tree"
            )

    def test_exact23_and_execution_portable_replays_reject_resigned_hostiles(self) -> None:
        exact23 = portable_replay("pre_load", exact23_fixed())
        analysis.validate_checkpoint_exact23_replay_v1(
            exact23, expected_stage="pre_load", label="exact23"
        )
        bad_exact23 = copy.deepcopy(exact23)
        bad_exact23["fixed_release_binding"]["files"][4]["relative_path"] = (
            "decoy/config.json"
        )
        bad_exact23["fixed_release_binding_digest"] = analysis.object_sha256(
            bad_exact23["fixed_release_binding"]
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "file row"
        ):
            analysis.validate_checkpoint_exact23_replay_v1(
                bad_exact23, expected_stage="pre_load", label="exact23"
            )

        execution = portable_replay("post_deserialize", execution_fixed())
        analysis.validate_bernini_execution_replay_v1(
            execution, expected_stage="post_deserialize", label="execution"
        )
        bad_execution = copy.deepcopy(execution)
        bad_execution["fixed_release_binding"]["files"][0]["sha256"] = "f" * 64
        bad_execution["fixed_release_binding_digest"] = analysis.object_sha256(
            bad_execution["fixed_release_binding"]
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "Bernini exact10"
        ):
            analysis.validate_bernini_execution_replay_v1(
                bad_execution,
                expected_stage="post_deserialize",
                label="execution",
            )

    def test_decode_exact25_source_replay_joins_release_and_training(self) -> None:
        receipt = {
            "training": {
                "receipt_sha256": sha(b"training receipt"),
                "origin_attestation_sha256": sha(b"origin attestation"),
            },
            "decode_release": {
                "decoder_source_sha256": sha(b"decoder"),
                "helper_source_sha256": sha(b"helper"),
                "analyzer_source_sha256": sha(b"analyzer"),
                "sha256": sha(b"release"),
            },
        }
        expected = dict(analysis.RUNTIME_SOURCE_BINDINGS)
        expected.update(
            {
                "artifact:latent_bundle": (
                    analysis.LATENT_BUNDLE_SHA256,
                    analysis.LATENT_BUNDLE_SIZE,
                ),
                "artifact:latent_bundle_receipt": (
                    analysis.LATENT_BUNDLE_RECEIPT_SHA256,
                    analysis.LATENT_BUNDLE_RECEIPT_SIZE,
                ),
                "artifact:materializer_run_complete": (
                    analysis.MATERIALIZER_RUN_COMPLETE_SHA256,
                    analysis.MATERIALIZER_RUN_COMPLETE_SIZE,
                ),
                "artifact:experiment_contract": (
                    analysis.EXPERIMENT_CONTRACT_SHA256,
                    analysis.EXPERIMENT_CONTRACT_SIZE,
                ),
                "artifact:external_authority": (
                    analysis.EXTERNAL_AUTHORITY_SHA256,
                    analysis.EXTERNAL_AUTHORITY_SIZE,
                ),
                "artifact:model_authority": (
                    analysis.MODEL_AUTHORITY_SHA256,
                    analysis.MODEL_AUTHORITY_SIZE,
                ),
                "artifact:checkpoint_exact23_manifest": (
                    analysis.CHECKPOINT_EXACT23_MANIFEST_SHA256,
                    analysis.CHECKPOINT_EXACT23_MANIFEST_SIZE,
                ),
                "artifact:exact10_training_receipt": (
                    receipt["training"]["receipt_sha256"],
                    None,
                ),
                "artifact:exact10_origin_attestation": (
                    receipt["training"]["origin_attestation_sha256"],
                    None,
                ),
                "decode:decoder": (
                    receipt["decode_release"]["decoder_source_sha256"],
                    None,
                ),
                "decode:c1_helper": (
                    receipt["decode_release"]["helper_source_sha256"],
                    None,
                ),
                "decode:analyzer": (
                    receipt["decode_release"]["analyzer_source_sha256"],
                    None,
                ),
                "decode:release_manifest": (
                    receipt["decode_release"]["sha256"],
                    None,
                ),
            }
        )
        self.assertEqual(len(expected), analysis.DECODE_SOURCE_COUNT)
        rows = [
            {
                "name": name,
                "sha256": expected[name][0],
                "size": (
                    expected[name][1]
                    if expected[name][1] is not None
                    else index + 1
                ),
                "mode": (
                    0o644
                    if name in analysis.DECODE_MUTABLE_CONTROL_SOURCE_NAMES
                    else 0o444
                ),
                "nlink": 1,
            }
            for index, name in enumerate(sorted(expected))
        ]
        fixed = {
            "source_count": analysis.DECODE_SOURCE_COUNT,
            "sources": rows,
            "all_sources_held_fd_replayed": True,
        }
        replay = {
            "stage": "final_pre_publish",
            "fixed_binding": fixed,
            "fixed_binding_digest": analysis.object_sha256(fixed),
            "world4_rank_consensus": True,
        }
        analysis.validate_decode_source_replay_v1(
            replay,
            expected_stage="final_pre_publish",
            receipt=receipt,
            label="source replay",
        )
        all_immutable = copy.deepcopy(replay)
        for row in all_immutable["fixed_binding"]["sources"]:
            row["mode"] = 0o444
        all_immutable["fixed_binding_digest"] = analysis.object_sha256(
            all_immutable["fixed_binding"]
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "source row"
        ):
            analysis.validate_decode_source_replay_v1(
                all_immutable,
                expected_stage="final_pre_publish",
                receipt=receipt,
                label="all-0444 incompatible control projection",
            )
        hostile = copy.deepcopy(replay)
        runtime = next(
            row
            for row in hostile["fixed_binding"]["sources"]
            if row["name"] == "runtime:c2_trainer"
        )
        runtime["sha256"] = "e" * 64
        hostile["fixed_binding_digest"] = analysis.object_sha256(
            hostile["fixed_binding"]
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "source row"
        ):
            analysis.validate_decode_source_replay_v1(
                hostile,
                expected_stage="final_pre_publish",
                receipt=receipt,
                label="source replay",
            )

    def test_analyzer_revalidates_decoder_exact10_branch_abi(self) -> None:
        rows = exact14_rows()
        for item in rows[4:]:
            step, q_key = analysis.GENERATED_STEPS_AND_Q[item["key"]]
            analysis.validate_decode_generated_branch_v1(
                item["branch_receipt"],
                branch=item["key"],
                expected_step=step,
                expected_q=q_key,
                expected_seed=20260821,
            )
        hostile = exact14_rows()[6]["branch_receipt"]
        hostile["world4_rank_receipts"][3]["initial_sampling_noise"][
            "requested_device"
        ] = "cuda:0"
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "rank3"
        ):
            analysis.validate_decode_generated_branch_v1(
                hostile,
                branch="trained_correct_q",
                expected_step=10,
                expected_q="target",
                expected_seed=20260821,
            )

    def test_decode_schedule_and_noise_survive_canonical_receipt_roundtrip(
        self,
    ) -> None:
        reference = {
            "schedule_sha256": analysis.EXACT40_SCHEDULE_SHA256,
            "timesteps": list(range(40)),
            "positive_sigmas": [1.0] * 40,
            "positive_sigmas_float32_be_hex": ["3f800000"] * 40,
            "terminal_sigma": 0.0,
            "terminal_sigma_float32_be_hex": "00000000",
        }
        reference_digest = analysis.object_sha256(reference)
        noise_sha = sha(b"matched native noise")
        fragment = {
            "exact40_unipc_schedule": {
                "pre_sample_reference": reference,
                "per_generated_branch": {
                    branch: {
                        "schedule_sha256": analysis.EXACT40_SCHEDULE_SHA256,
                        "audit_object_sha256": reference_digest,
                        "matches_pre_sample_reference": True,
                    }
                    for branch in analysis.GENERATED_BRANCHES
                },
                "all_exact10_generated_branches_match_reference": True,
            },
            "matched_initial_sampling_noise": {
                "generated_branch_order": list(analysis.GENERATED_BRANCHES),
                "spatial_tensor_sha256": noise_sha,
                "sha256_by_branch": {
                    branch: noise_sha for branch in analysis.GENERATED_BRANCHES
                },
                "same_native_initial_sampling_noise_for_all_exact10_generated_branches": True,
                "observer_only_external_noise_injection": False,
            },
        }
        persisted = json.loads(analysis.canonical_json_bytes(fragment))
        self.assertNotEqual(
            list(persisted["exact40_unipc_schedule"]["per_generated_branch"]),
            list(analysis.GENERATED_BRANCHES),
        )
        self.assertNotEqual(
            list(persisted["matched_initial_sampling_noise"]["sha256_by_branch"]),
            list(analysis.GENERATED_BRANCHES),
        )
        analysis.validate_decode_schedule_and_matched_noise_v1(
            persisted, label="canonical decode receipt"
        )
        for outer, inner in (
            ("exact40_unipc_schedule", "per_generated_branch"),
            ("matched_initial_sampling_noise", "sha256_by_branch"),
        ):
            for mutation in ("missing", "extra", "renamed"):
                with self.subTest(outer=outer, mutation=mutation):
                    hostile = copy.deepcopy(persisted)
                    mapping = hostile[outer][inner]
                    victim = analysis.GENERATED_BRANCHES[0]
                    if mutation == "missing":
                        mapping.pop(victim)
                    elif mutation == "extra":
                        mapping["attacker_extra"] = copy.deepcopy(
                            next(iter(mapping.values()))
                        )
                    else:
                        mapping["attacker_renamed"] = mapping.pop(victim)
                    with self.assertRaisesRegex(
                        analysis.ELAL3C2DecodedAnalysisError,
                        "(schedule closure|sampling-noise closure)",
                    ):
                        analysis.validate_decode_schedule_and_matched_noise_v1(
                            hostile, label="hostile canonical decode receipt"
                        )

    def test_fixed_palette_tracker_accepts_exact_clean_entities(self) -> None:
        frames = np.zeros((81, 96, 128, 3), dtype=np.uint8)
        frames[:] = (24, 42, 55)
        for index in range(81):
            for offset, entity in enumerate(analysis.ENTITY_ORDER):
                x = 16 + offset * 34 + index // 8
                y = 25 + offset * 22
                frames[index, y - 3:y + 4, x - 3:x + 4] = analysis.PALETTE_RGB[entity]
        receipt = analysis.track_colored_entities_v1(frames)
        self.assertTrue(receipt["identity_recovery_reliable"])
        self.assertEqual(set(receipt["tracks_xy_normalized"]), set(analysis.ENTITY_ORDER))
        self.assertTrue(all(row["observed_frames"] == 81 for row in receipt["entities"].values()))

    def test_fixed_palette_tracker_fails_closed_on_blank_video(self) -> None:
        frames = np.full((81, 96, 128, 3), (24, 42, 55), dtype=np.uint8)
        receipt = analysis.track_colored_entities_v1(frames)
        self.assertFalse(receipt["identity_recovery_reliable"])
        self.assertIsNone(receipt["tracks_xy_normalized"])
        self.assertGreaterEqual(len(receipt["no_go_reasons"]), 1)

    def test_fixed_palette_tracker_fails_closed_on_multimodal_entity(self) -> None:
        frames = np.full((81, 96, 128, 3), (24, 42, 55), dtype=np.uint8)
        for index in range(81):
            frames[index, 18:23, 8:13] = analysis.PALETTE_RGB["agent"]
            frames[index, 70:75, 108:113] = analysis.PALETTE_RGB["agent"]
            frames[index, 42:49, 48:55] = analysis.PALETTE_RGB["patient"]
            frames[index, 60:67, 72:79] = analysis.PALETTE_RGB["object"]
        receipt = analysis.track_colored_entities_v1(frames)
        self.assertFalse(receipt["identity_recovery_reliable"])
        self.assertIsNone(receipt["tracks_xy_normalized"])
        self.assertTrue(
            any("agent:palette_identity_unreliable" == row for row in receipt["no_go_reasons"])
        )

    def _evaluate(self, row_id: str) -> dict[str, object]:
        if row_id == analysis.ROW_IDS[0]:
            target = block_tracks()
            role = block_tracks(role=True)
            occ = "patient"
        else:
            target = hand_tracks()
            role = hand_tracks(role=True)
            occ = "object"
        branches = {key: reliable(target) for key in analysis.TRACK_REQUIRED_BRANCHES}
        branches["trained_full_role_swap_q"] = reliable(role)
        branches["trained_role_only_mismatch_q"] = reliable(role)
        return analysis.evaluate_row_tracks_v1(
            row_id=row_id,
            tracks_by_branch=branches,
            target_annotation=annotation(target, occluded_entity=occ),
            role_annotation=annotation(role, occluded_entity=occ),
        )

    def test_blocking_preregistered_gates_pass_on_exact_tracks(self) -> None:
        result = self._evaluate(analysis.ROW_IDS[0])
        self.assertEqual(result["status"], "DECODED_GATES_PASS")
        self.assertTrue(result["all_preregistered_decoded_gates_pass"])
        self.assertTrue(all(row["passed"] for row in result["gates"].values()))

    def test_handover_preregistered_gates_pass_on_exact_tracks(self) -> None:
        result = self._evaluate(analysis.ROW_IDS[1])
        self.assertEqual(result["status"], "DECODED_GATES_PASS")
        self.assertEqual(
            result["gates"]["secondary_effect"]["effect"],
            "object_ownership_agent_to_receiver",
        )

    def test_role_only_wrong_direction_is_no_go(self) -> None:
        target = block_tracks()
        role = block_tracks(role=True)
        branches = {key: reliable(target) for key in analysis.TRACK_REQUIRED_BRANCHES}
        branches["trained_full_role_swap_q"] = reliable(role)
        # Opposite of the preregistered contrast: mismatch moves farther beyond target.
        anti = {
            entity: [
                [1.2 * target[entity][i][axis] - .2 * role[entity][i][axis] for axis in (0, 1)]
                for i in range(81)
            ]
            for entity in analysis.ENTITY_ORDER
        }
        branches["trained_role_only_mismatch_q"] = reliable(anti)
        result = analysis.evaluate_row_tracks_v1(
            row_id=analysis.ROW_IDS[0],
            tracks_by_branch=branches,
            target_annotation=annotation(target, occluded_entity="patient"),
            role_annotation=annotation(role, occluded_entity="patient"),
        )
        self.assertEqual(result["status"], "DECODED_GATES_NO_GO")
        self.assertFalse(result["gates"]["role_only_matched_vs_mismatch"]["passed"])

    def test_2x2_diagonal_swap_collapse_is_no_go(self) -> None:
        target = block_tracks()
        role = block_tracks(role=True)
        branches = {key: reliable(target) for key in analysis.TRACK_REQUIRED_BRANCHES}
        branches["trained_full_role_swap_q"] = reliable(target)
        branches["trained_role_only_mismatch_q"] = reliable(role)
        result = analysis.evaluate_row_tracks_v1(
            row_id=analysis.ROW_IDS[0],
            tracks_by_branch=branches,
            target_annotation=annotation(target, occluded_entity="patient"),
            role_annotation=annotation(role, occluded_entity="patient"),
        )
        self.assertFalse(
            result["gates"][
                "event_participant_union_correct_vs_full_role_swap_2x2"
            ]["passed"]
        )

    def test_terminal_secondary_and_occlusion_gates_each_fail_closed(self) -> None:
        target = block_tracks()
        terminal_bad = {key: [list(point) for point in value] for key, value in target.items()}
        for index in range(65, 81):
            terminal_bad["agent"][index][1] = min(1.0, .52 + .035 * (index - 65))
        self.assertFalse(analysis.terminal_hold_gate_v1(terminal_bad, target)["passed"])

        secondary_bad = {key: [list(point) for point in value] for key, value in target.items()}
        secondary_bad["patient"] = [[.10, .42] for _ in range(81)]
        self.assertFalse(
            analysis.secondary_effect_gate_v1(analysis.ROW_IDS[0], secondary_bad)["passed"]
        )

        identity_bad = {key: [list(point) for point in value] for key, value in target.items()}
        for index in range(45, 81):
            identity_bad["patient"][index] = list(target["agent"][index])
        gate = analysis.occlusion_identity_gate_v1(
            annotation(target, occluded_entity="patient"), identity_bad, target
        )
        self.assertFalse(gate["passed"])

    def test_cross_seed_requires_same_positive_direction(self) -> None:
        rows = {
            arm: {row: self._evaluate(row) for row in analysis.ROW_IDS}
            for arm in analysis.ARM_IDS
        }
        self.assertTrue(analysis.cross_seed_direction_gate_v1(rows)["passed"])
        rows[analysis.ARM_IDS[2]][analysis.ROW_IDS[1]]["gates"][
            "role_only_matched_vs_mismatch"
        ]["normalized_predicted_vs_clean_role_contrast"] = -0.1
        result = analysis.cross_seed_direction_gate_v1(rows)
        self.assertFalse(result["passed"])
        self.assertFalse(result["per_row"][analysis.ROW_IDS[1]]["same_positive_direction"])

    def test_malformed_reliable_track_receipt_is_structural_error(self) -> None:
        target = block_tracks()
        role = block_tracks(role=True)
        branches = {key: reliable(target) for key in analysis.TRACK_REQUIRED_BRANCHES}
        branches["trained_full_role_swap_q"] = reliable(role)
        branches["trained_role_only_mismatch_q"] = reliable(role)
        branches["trained_correct_q"] = reliable(target)
        target_annotation = annotation(target, occluded_entity="patient")
        role_annotation = annotation(role, occluded_entity="patient")
        branches["trained_correct_q"]["tracks_xy_normalized"]["agent"] = [[.1, .2]]
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "exact81"
        ):
            analysis.evaluate_row_tracks_v1(
                row_id=analysis.ROW_IDS[0],
                tracks_by_branch=branches,
                target_annotation=target_annotation,
                role_annotation=role_annotation,
            )

    def test_any_unreliable_branch_produces_explicit_no_go_without_scores(self) -> None:
        target = block_tracks()
        role = block_tracks(role=True)
        branches = {key: reliable(target) for key in analysis.TRACK_REQUIRED_BRANCHES}
        branches["trained_zero_q"] = {
            "identity_recovery_reliable": False,
            "tracks_xy_normalized": None,
            "no_go_reasons": ["agent:palette_identity_unreliable"],
        }
        result = analysis.evaluate_row_tracks_v1(
            row_id=analysis.ROW_IDS[0],
            tracks_by_branch=branches,
            target_annotation=annotation(target, occluded_entity="patient"),
            role_annotation=annotation(role, occluded_entity="patient"),
        )
        self.assertEqual(result["status"], "NO_GO_UNRELIABLE_COLOR_IDENTITY")
        self.assertIsNone(result["gates"])

    def test_static_cli_requires_exact_six_unique_coordinates(self) -> None:
        parser = analysis.parser()
        base = [
            "--packet-root", "/tmp/packet",
            "--core-source", "/tmp/elal3_c0_v1.py",
            "--expected-core-source-sha256",
            analysis.FROZEN_ELAL3_CORE_SOURCE_SHA256,
            "--label-source", "/tmp/elal3_simulator_c2_label_v1.py",
            "--expected-label-source-sha256",
            analysis.FROZEN_C2_LABEL_SOURCE_SHA256,
            "--expected-analyzer-source-sha256", "2" * 64,
            "--output-root", "/tmp/fresh-output",
            "--ack-simulator-oracle-q-only",
            "--ack-no-formal-c2",
            "--ack-no-exact160",
            "--ack-no-scientific-or-real-video-claim",
        ]
        specs = []
        for arm in analysis.ARM_IDS:
            for row in analysis.ROW_IDS:
                specs += ["--decode", f"{arm}:{row}:/tmp/{arm}-{row}.json:{'3' * 64}"]
        args = parser.parse_args(base + specs)
        analysis.validate_static_args_v1(args)
        stale = list(base)
        stale[stale.index(analysis.FROZEN_C2_LABEL_SOURCE_SHA256)] = "1" * 64
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "frozen runtime:c2_label"
        ):
            analysis.validate_static_args_v1(parser.parse_args(stale + specs))
        wrong_name = list(base)
        wrong_name[wrong_name.index("/tmp/elal3_simulator_c2_label_v1.py")] = (
            "/tmp/attacker-label.py"
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "basename"
        ):
            analysis.validate_static_args_v1(
                parser.parse_args(wrong_name + specs)
            )
        stale_core = list(base)
        stale_core[stale_core.index(analysis.FROZEN_ELAL3_CORE_SOURCE_SHA256)] = (
            "4" * 64
        )
        with self.assertRaisesRegex(
            analysis.ELAL3C2DecodedAnalysisError, "frozen runtime:elal3_core"
        ):
            analysis.validate_static_args_v1(
                parser.parse_args(stale_core + specs)
            )
        with self.assertRaisesRegex(analysis.ELAL3C2DecodedAnalysisError, "exact six"):
            analysis.validate_static_args_v1(parser.parse_args(base + specs[:-2]))

    def test_frozen_label_source_binding_rejects_fully_resigned_arbitrary_helper(
        self,
    ) -> None:
        frozen = METHOD_ROOT / analysis.FROZEN_C2_LABEL_SOURCE_NAME
        frozen_core = METHOD_ROOT / analysis.FROZEN_ELAL3_CORE_SOURCE_NAME
        self.assertEqual(
            (sha(frozen.read_bytes()), frozen.stat().st_size),
            (
                analysis.FROZEN_C2_LABEL_SOURCE_SHA256,
                analysis.FROZEN_C2_LABEL_SOURCE_SIZE,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / analysis.FROZEN_C2_LABEL_SOURCE_NAME
            core_path = Path(temporary).resolve() / analysis.FROZEN_ELAL3_CORE_SOURCE_NAME
            path.write_bytes(frozen.read_bytes())
            path.chmod(0o444)
            core_path.write_bytes(frozen_core.read_bytes())
            core_path.chmod(0o444)
            binding = analysis.validate_frozen_label_source_v1(
                path,
                expected_sha256=analysis.FROZEN_C2_LABEL_SOURCE_SHA256,
                label="positive frozen label helper",
            )
            self.assertEqual(
                (binding["sha256"], binding["size"], binding["mode"], binding["nlink"]),
                (
                    analysis.FROZEN_C2_LABEL_SOURCE_SHA256,
                    analysis.FROZEN_C2_LABEL_SOURCE_SIZE,
                    0o444,
                    1,
                ),
            )
            module, executed_binding, core_binding = analysis.load_frozen_label_module_v1(
                path,
                expected_sha256=analysis.FROZEN_C2_LABEL_SOURCE_SHA256,
                core_path=core_path,
                expected_core_sha256=analysis.FROZEN_ELAL3_CORE_SOURCE_SHA256,
            )
            self.assertTrue(callable(module.load_oracle_q_label_v1))
            self.assertTrue(executed_binding["executed_bytes_are_held_replay"])
            self.assertTrue(core_binding["executed_bytes_are_held_replay"])
            sys.modules.pop("elal3_simulator_c2_label_v1", None)
            sys.modules.pop("elal3_c0_v1", None)

            original_compile = compile
            swapped = False

            def swap_during_compile(source, filename, mode, **kwargs):
                nonlocal swapped
                if not swapped and filename == str(path):
                    swapped = True
                    path.chmod(0o644)
                    path.write_bytes(frozen.read_bytes())
                    path.chmod(0o444)
                return original_compile(source, filename, mode, **kwargs)

            with mock.patch("builtins.compile", side_effect=swap_during_compile):
                with self.assertRaisesRegex(
                    analysis.ELAL3C2DecodedAnalysisError,
                    "changed across held-byte execution",
                ):
                    analysis.load_frozen_label_module_v1(
                        path,
                        expected_sha256=analysis.FROZEN_C2_LABEL_SOURCE_SHA256,
                        core_path=core_path,
                        expected_core_sha256=analysis.FROZEN_ELAL3_CORE_SOURCE_SHA256,
                    )
            self.assertNotIn("elal3_simulator_c2_label_v1", sys.modules)
            self.assertNotIn("elal3_c0_v1", sys.modules)
            previous_core = sys.modules.get("elal3_c0_v1")
            sys.modules["elal3_c0_v1"] = object()
            try:
                with self.assertRaisesRegex(
                    analysis.ELAL3C2DecodedAnalysisError, "module caches"
                ):
                    analysis.load_frozen_label_module_v1(
                        path,
                        expected_sha256=analysis.FROZEN_C2_LABEL_SOURCE_SHA256,
                        core_path=core_path,
                        expected_core_sha256=analysis.FROZEN_ELAL3_CORE_SOURCE_SHA256,
                    )
            finally:
                if previous_core is None:
                    sys.modules.pop("elal3_c0_v1", None)
                else:
                    sys.modules["elal3_c0_v1"] = previous_core
            path.chmod(0o644)
            arbitrary = b"PACKET_TRUTH = 'forged'\n"
            path.write_bytes(arbitrary)
            path.chmod(0o444)
            with self.assertRaisesRegex(
                analysis.ELAL3C2DecodedAnalysisError, "CLI SHA differs"
            ):
                analysis.validate_frozen_label_source_v1(
                    path,
                    expected_sha256=sha(arbitrary),
                    label="fully resigned arbitrary label helper",
                )

    def test_analysis_video_decode_uses_retained_fd_and_rejects_path_swap(
        self,
    ) -> None:
        class Frame:
            width = 8
            height = 6
            format = type("Format", (), {"name": "yuv420p"})()

            @staticmethod
            def to_ndarray(*, format):
                self.assertEqual(format, "rgb24")
                return np.zeros((6, 8, 3), dtype=np.uint8)

        class Streams(list):
            def __init__(self):
                rate = type("Rate", (), {"numerator": 25, "denominator": 1})()
                stream = type("Stream", (), {"average_rate": rate})()
                super().__init__([stream])
                self.video = [stream]
                self.audio = []

        class Container:
            streams = Streams()

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            @staticmethod
            def decode(*, video):
                self.assertEqual(video, 0)
                return [Frame() for _ in range(analysis.FRAME_COUNT)]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "video.mp4"
            raw = b"analysis-held-video-fixture"
            path.write_bytes(raw)
            calls = []

            def open_retained(stream, *, mode):
                self.assertFalse(isinstance(stream, (str, Path)))
                self.assertEqual(mode, "r")
                calls.append(stream)
                return Container()

            previous = sys.modules.get("av")
            sys.modules["av"] = type("FakeAv", (), {"open": staticmethod(open_retained)})()
            try:
                frames = analysis.decode_video_rgb24_v1(
                    path, expected_sha256=sha(raw), expected_hw=(6, 8)
                )
                self.assertEqual(frames.shape, (analysis.FRAME_COUNT, 6, 8, 3))
                self.assertEqual(len(calls), 1)

                def swap_path_during_open(stream, *, mode):
                    path.unlink()
                    path.write_bytes(raw)
                    return Container()

                sys.modules["av"] = type(
                    "SwapAv", (), {"open": staticmethod(swap_path_during_open)}
                )()
                with self.assertRaisesRegex(
                    analysis.ELAL3C2DecodedAnalysisError,
                    "identity changed across decode",
                ):
                    analysis.decode_video_rgb24_v1(
                        path, expected_sha256=sha(raw), expected_hw=(6, 8)
                    )
            finally:
                if previous is None:
                    sys.modules.pop("av", None)
                else:
                    sys.modules["av"] = previous


if __name__ == "__main__":
    unittest.main()
