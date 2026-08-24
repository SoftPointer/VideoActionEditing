#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import audit_pair_v7_phase_a2_multicondition_geometry as runtime  # noqa: E402
import pair_v7_multicondition_geometry_authority as prereg  # noqa: E402
import postflight_pair_v7_phase_a2_multicondition as postflight  # noqa: E402
import source_self_runtime as distributed_runtime  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class PhaseA2MulticonditionRuntimeTests(unittest.TestCase):
    def test_read_only_multinode_world8_requires_exact_local_sp4_mapping(self) -> None:
        environment = {
            "WORLD_SIZE": "8",
            "RANK": "6",
            "LOCAL_RANK": "2",
            "LOCAL_WORLD_SIZE": "4",
        }
        with self.assertRaises(distributed_runtime.SourceSelfRuntimeError):
            distributed_runtime.distributed_contract(environment)
        contract = distributed_runtime.distributed_contract(
            environment, allow_multinode_dp2_sp4=True
        )
        self.assertEqual(contract.arm_index, 1)
        self.assertEqual(contract.sp_rank, 2)
        self.assertEqual(contract.local_world_size, 4)
        for invalid in (
            {**environment, "LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "0"},
            {**environment, "LOCAL_RANK": "1"},
        ):
            with self.assertRaises(distributed_runtime.SourceSelfRuntimeError):
                distributed_runtime.distributed_contract(
                    invalid, allow_multinode_dp2_sp4=True
                )

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.shape = [1, 16, 21, 2, 2]
        self.candidates: list[dict] = []
        self.child_paths: list[Path] = []
        self.events: list[dict] = []
        self._build_fixture()

    def _write(self, relative: str, payload: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path.resolve(strict=True)

    @staticmethod
    def _file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _captions(tag: str) -> dict[str, str]:
        return {
            branch: f"raw {tag} {branch}"
            for branch in runtime.BRANCH_ORDER
        }

    @staticmethod
    def _rebuild(captions):
        return {
            branch: f"T2V::{captions[branch]}"
            for branch in runtime.BRANCH_ORDER
        }

    def _write_child(
        self,
        index: int,
        *,
        candidate_id: str,
        split: str,
        family: str,
        source_sha: str,
        clean_tensor_sha: str,
        noise_tensor_sha: str,
        captions: dict[str, str],
    ) -> dict:
        prompts = self._rebuild(captions)
        unsigned = {
            "schema_version": "test-cast-v4-score",
            "candidate_id": candidate_id,
            "analysis_split": split,
            "action_family_id": family,
            "semantic_branch": "action",
            "root_spec_raw_sha256": _sha("root-spec"),
            "frozen_checkpoint_receipt_digest": _sha("frozen-checkpoint"),
            "checkpoint_content_binding": {
                "tree_sha256": _sha("checkpoint-tree")
            },
            "geometry_source_video_sha256": source_sha,
            "full_t2v_caption_by_branch": captions,
            "clean_latent_tensor_sha256": clean_tensor_sha,
            "official_gaussian_tensor_sha256": noise_tensor_sha,
            "prompt_by_branch": prompts,
            "frozen_t2v_packet_binding": {"candidate_shape": self.shape},
            "raw_global_action_energy_score": 1.0 + index / 100.0,
            "scientific_action_editing_claim": False,
        }
        child = {**unsigned, "receipt_digest": runtime.object_sha256(unsigned)}
        path = self._write(
            f"cast/candidate-{index:02d}/score.json",
            runtime.canonical_json_bytes(child) + b"\n",
        )
        self.child_paths.append(path)
        return {
            "candidate_id": candidate_id,
            "path": str(path),
            "file_sha256": self._file_sha(path),
            "receipt_digest": child["receipt_digest"],
            "analysis_split": split,
            "action_family_id": family,
            "semantic_branch": "action",
            "root_spec_raw_sha256": unsigned["root_spec_raw_sha256"],
            "frozen_checkpoint_receipt_digest": unsigned[
                "frozen_checkpoint_receipt_digest"
            ],
            "checkpoint_content_binding": unsigned[
                "checkpoint_content_binding"
            ],
            "geometry_source_video_sha256": source_sha,
            "full_t2v_caption_by_branch": captions,
            "clean_latent_tensor_sha256": clean_tensor_sha,
            "official_gaussian_tensor_sha256": noise_tensor_sha,
            "prompt_by_branch": prompts,
            "candidate_shape": self.shape,
            "raw_global_action_energy_score": unsigned[
                "raw_global_action_energy_score"
            ],
        }

    def _build_fixture(self) -> None:
        pairs = ("fit", "fit", "confirmation", "confirmation")
        families = ("fit-a", "fit-b", "confirmation-a", "confirmation-b")
        event_ids = tuple(f"event-{index}" for index in range(4))
        for index, (pair, family, event_id) in enumerate(
            zip(pairs, families, event_ids)
        ):
            source = self._write(
                f"artifacts/source-{index}.mp4", f"source-{index}".encode()
            )
            clean = self._write(
                f"artifacts/clean-{index}.safetensors", f"clean-{index}".encode()
            )
            noise = self._write(
                f"artifacts/noise-{index}.safetensors", f"noise-{index}".encode()
            )
            clean_tensor_sha = _sha(f"clean-tensor-{index}")
            noise_tensor_sha = _sha(f"noise-tensor-{index}")
            event_unsigned = {
                "schema_version": "test-event",
                "event_id": event_id,
                "source_sample_id": f"sample-{index}",
                "action_family": family,
                "analysis_split": pair,
                "pair_wave": pair,
                "dp_arm": index % 2,
                "generation_seed": 100 + index,
                "source_video_path": str(source),
                "source_video_file_sha256": self._file_sha(source),
                "clean_latent_path": str(clean),
                "clean_latent_file_sha256": self._file_sha(clean),
                "clean_latent_tensor_key": "normalized_clean_latent",
                "clean_latent_tensor_sha256": clean_tensor_sha,
                "official_gaussian_path": str(noise),
                "official_gaussian_file_sha256": self._file_sha(noise),
                "official_gaussian_tensor_key": "official_initial_gaussian",
                "official_gaussian_tensor_sha256": noise_tensor_sha,
                "latent_shape": self.shape,
                "frame_count": 81,
                "fps": 25.0,
                "source_noise_key_sha256": prereg._source_noise_key(
                    f"sample-{index}"
                ),
                "geometry_measurement_authorized": False,
                "optimizer_authorized": False,
                "parameter_update_authorized": False,
            }
            self.events.append(
                {
                    **event_unsigned,
                    "event_digest": runtime.object_sha256(event_unsigned),
                }
            )
            captions = self._captions(event_id)
            self.candidates.append(
                self._write_child(
                    index,
                    candidate_id=event_id,
                    split=pair,
                    family=family,
                    source_sha=self._file_sha(source),
                    clean_tensor_sha=clean_tensor_sha,
                    noise_tensor_sha=noise_tensor_sha,
                    captions=captions,
                )
            )

        for index in range(4, 40):
            split = "fit" if index < 20 else "confirmation"
            self.candidates.append(
                self._write_child(
                    index,
                    candidate_id=f"unused-{index}",
                    split=split,
                    family=f"unused-family-{index}",
                    source_sha=_sha(f"unused-source-{index}"),
                    clean_tensor_sha=_sha(f"unused-clean-{index}"),
                    noise_tensor_sha=_sha(f"unused-noise-{index}"),
                    captions=self._captions(f"unused-{index}"),
                )
            )

        plan_unsigned = {
            "schema_version": prereg.PLAN_SCHEMA,
            "geometry_measurement_authorized": False,
            "optimizer_authorized": False,
            "parameter_update_authorized": False,
            "primary_schedule_indices": [16, 35],
            "event_count": 4,
            "events": self.events,
            "primary_condition_count": 4,
            "primary_cells": [
                {
                    "pair_wave": pair,
                    "schedule": {"schedule_index": schedule},
                }
                for pair in runtime.PAIR_IDS
                for schedule in runtime.SCHEDULE_INDICES
            ],
            "global_common_direction_spec": prereg._global_common_direction_spec(),
            "primary_gate_definition": prereg._primary_gate_definition(),
            "checkpoint_tree_sha256": _sha("checkpoint-tree"),
            "action_adapter_schema_sha256": _sha("adapter-schema"),
        }
        self.plan = {
            **plan_unsigned,
            "preregistration_digest": prereg.object_sha256(plan_unsigned),
        }
        self.plan_path = self._write(
            "plan.json", runtime.canonical_json_bytes(self.plan) + b"\n"
        )
        self.plan_file_sha = self._file_sha(self.plan_path)

    def _bind(self, *, plan=None, candidates=None, plan_path=None, plan_sha=None):
        return runtime.bind_plan_events_to_cast_candidates(
            plan=self.plan if plan is None else plan,
            plan_path=self.plan_path if plan_path is None else plan_path,
            plan_file_sha256=self.plan_file_sha if plan_sha is None else plan_sha,
            candidates=self.candidates if candidates is None else candidates,
            cast_group_receipt_digests=[_sha("group-a"), _sha("group-b")],
            cast_method_binding={"archive_sha256": _sha("cast-method")},
            cast_root_binding={"root_sha256": _sha("cast-root")},
            prompt_rebuilder=self._rebuild,
        )

    def _rewrite_candidate_child(self, index: int, updates: dict) -> list[dict]:
        candidates = copy.deepcopy(self.candidates)
        path = Path(candidates[index]["path"])
        child = json.loads(path.read_text(encoding="ascii"))
        unsigned = dict(child)
        unsigned.pop("receipt_digest")
        unsigned.update(copy.deepcopy(updates))
        child = {**unsigned, "receipt_digest": runtime.object_sha256(unsigned)}
        path.write_bytes(runtime.canonical_json_bytes(child) + b"\n")
        for field, value in updates.items():
            if field == "frozen_t2v_packet_binding":
                candidates[index]["candidate_shape"] = value["candidate_shape"]
            else:
                candidates[index][field] = copy.deepcopy(value)
        candidates[index]["receipt_digest"] = child["receipt_digest"]
        candidates[index]["file_sha256"] = self._file_sha(path)
        return candidates

    def test_binds_exact_four_events_to_complete_forty_child_bank(self) -> None:
        manifest, binding = self._bind()
        self.assertEqual(len(manifest.events), 4)
        self.assertEqual(
            [(row.pair_id, row.dp_arm) for row in manifest.events],
            [("fit", 0), ("fit", 1), ("confirmation", 0), ("confirmation", 1)],
        )
        self.assertEqual(binding["cast_candidate_receipt_count"], 40)
        self.assertTrue(
            binding["combined_read_only_geometry_measurement_authorized"]
        )
        self.assertFalse(binding["optimizer_constructed"])
        self.assertFalse(binding["parameter_mutation_performed"])
        first = binding["selected_events"][0]
        self.assertEqual(
            first["prompt_by_branch"], self.candidates[0]["prompt_by_branch"]
        )
        self.assertEqual(
            first["full_t2v_caption_by_branch"],
            self.candidates[0]["full_t2v_caption_by_branch"],
        )
        self.assertEqual(
            first["official_gaussian_tensor_sha256"],
            self.events[0]["official_gaussian_tensor_sha256"],
        )

    def test_rejects_prompt_or_raw_caption_semantic_drift(self) -> None:
        prompts = copy.deepcopy(self.candidates[0]["prompt_by_branch"])
        prompts["action"] = "wrong prompt"
        with self.assertRaisesRegex(runtime.PairV7PhaseA2Error, "prompt/raw-caption"):
            self._bind(
                candidates=self._rewrite_candidate_child(
                    0, {"prompt_by_branch": prompts}
                )
            )

        # Restore fixture bytes, then alter only the raw caption semantics.
        self.setUp()
        captions = copy.deepcopy(
            self.candidates[0]["full_t2v_caption_by_branch"]
        )
        captions["reverse"] = "wrong raw caption"
        with self.assertRaisesRegex(runtime.PairV7PhaseA2Error, "prompt/raw-caption"):
            self._bind(
                candidates=self._rewrite_candidate_child(
                    0, {"full_t2v_caption_by_branch": captions}
                )
            )

    def test_rejects_wrong_split_source_or_tensor_binding(self) -> None:
        cases = (
            {"analysis_split": "confirmation"},
            {"geometry_source_video_sha256": _sha("wrong-source")},
            {"clean_latent_tensor_sha256": _sha("wrong-clean")},
            {"official_gaussian_tensor_sha256": _sha("wrong-noise")},
        )
        for updates in cases:
            with self.subTest(updates=updates):
                temporary = copy.deepcopy(self.candidates)
                path = Path(temporary[0]["path"])
                original = path.read_bytes()
                try:
                    temporary = self._rewrite_candidate_child(0, updates)
                    with self.assertRaisesRegex(
                        runtime.PairV7PhaseA2Error, "CAST binding differs"
                    ):
                        self._bind(candidates=temporary)
                finally:
                    path.write_bytes(original)

    def test_rejects_candidate_projection_not_bound_to_child_bytes(self) -> None:
        candidates = copy.deepcopy(self.candidates)
        candidates[0]["prompt_by_branch"]["action"] = "unsealed caller value"
        with self.assertRaisesRegex(
            runtime.PairV7PhaseA2Error, "validator projection differs"
        ):
            self._bind(candidates=candidates)

    def test_rejects_plan_path_bytes_or_incomplete_cast_bank(self) -> None:
        with self.assertRaisesRegex(runtime.PairV7PhaseA2Error, "path/bytes"):
            self._bind(plan_sha=_sha("wrong-plan"))
        with self.assertRaisesRegex(runtime.PairV7PhaseA2Error, "forty unique"):
            self._bind(candidates=self.candidates[:-1])

    def test_world_input_consensus_requires_all_eight_exact_values(self) -> None:
        unsigned = {
            "schema_version": runtime.WORLD_INPUT_SCHEMA,
            "action_condition_count": 8,
            "identity_probe_count": 64,
        }
        row = {**unsigned, "input_digest": runtime.object_sha256(unsigned)}
        self.assertEqual(
            runtime.validate_world_input_consensus([row] * 8),
            row["input_digest"],
        )
        mismatched = [copy.deepcopy(row) for _ in range(8)]
        mismatched[7]["identity_probe_count"] = 63
        with self.assertRaisesRegex(runtime.PairV7PhaseA2Error, "consensus"):
            runtime.validate_world_input_consensus(mismatched)
        with self.assertRaisesRegex(runtime.PairV7PhaseA2Error, "count/type"):
            runtime.validate_world_input_consensus([row] * 7)

    def test_no_go_solver_receipt_is_valid_and_string_bool_is_rejected(self) -> None:
        authority_digest = _sha("live-toctou-authority")
        world_input_digest = _sha("sealed-world-input")
        unsigned = {
            "schema_version": "test-multicondition-transport",
            "primary_replication_go": False,
            "geometry_audit_passed": False,
            "failure_codes": ["GLOBAL_COMMON_DIRECTION_NO_GO"],
            "action_condition_count": 8,
            "identity_probe_count": 64,
            "multicondition_authority_digest": authority_digest,
            "validated_world_input_digest": world_input_digest,
            "parameter_mutation_performed": False,
            "gradient_or_adapter_artifact_written": False,
        }
        transport = {
            **unsigned,
            "receipt_digest": runtime.object_sha256(unsigned),
        }
        envelope = {
            "ok": True,
            "transport_receipt": transport,
            "transport_receipt_digest": transport["receipt_digest"],
            "primary_replication_go": False,
        }
        checked, digest, go = runtime.validate_root_solver_result(
            envelope,
            bank_binding_digest=authority_digest,
            expected_world_input_digest=world_input_digest,
        )
        self.assertEqual(checked, transport)
        self.assertEqual(digest, transport["receipt_digest"])
        self.assertIs(go, False)
        poisoned = copy.deepcopy(envelope)
        poisoned["primary_replication_go"] = "False"
        with self.assertRaisesRegex(runtime.PairV7PhaseA2Error, "type differs"):
            runtime.validate_root_solver_result(
                poisoned,
                bank_binding_digest=authority_digest,
                expected_world_input_digest=world_input_digest,
            )

        mismatched = copy.deepcopy(envelope)
        with self.assertRaisesRegex(
            runtime.PairV7PhaseA2Error, "transport receipt closure differs"
        ):
            runtime.validate_root_solver_result(
                mismatched,
                bank_binding_digest=authority_digest,
                expected_world_input_digest=_sha("different-world-input"),
            )

    def _synthetic_final_receipt(self, *, go: bool) -> dict:
        _manifest, bank = self._bind()
        bank_digest = bank["receipt_digest"]
        transport_unsigned = {
            "schema_version": "test-multicondition-transport",
            "primary_replication_go": go,
            "geometry_audit_passed": go,
            "failure_codes": [] if go else ["GLOBAL_COMMON_DIRECTION_NO_GO"],
            "action_condition_count": 8,
            "identity_probe_count": 64,
            "multicondition_authority_digest": bank_digest,
            "validated_world_input_digest": _sha("world-input"),
            "parameter_mutation_performed": False,
            "gradient_or_adapter_artifact_written": False,
        }
        transport = {
            **transport_unsigned,
            "receipt_digest": runtime.object_sha256(transport_unsigned),
        }
        runtime_archive = _sha("runtime-archive")
        runtime_revision = "a" * 40
        solver = runtime._seal(
            {
                "schema_version": runtime.WORLD_SOLVER_AUTHORITY_SCHEMA,
                "world_size": 8,
                "topology": "WORLD8-DP2xUlysses-SP4",
                "input_consensus": True,
                "input_digest": _sha("world-input"),
                "input_consensus_rank_count": 8,
                "final_toctou_bank_binding_digest": bank_digest,
                "manifest_digest": bank_digest,
                "runtime_source_archive_sha256": runtime_archive,
                "runtime_source_revision": runtime_revision,
                "solver_execution_rank": 0,
                "solver_execution_device": "cpu",
                "solver_input_dtype": "torch.float32",
                "solver_internal_geometry_dtype": "torch.float64",
                "solver_execution_count": 1,
                "single_global_direction_solve": True,
                "local_project_then_average": False,
                "transport_receipt_digest": transport["receipt_digest"],
                "result_consensus": True,
                "raw_gradient_artifact_written": False,
                "safe_direction_artifact_written": False,
                "phase_b_requires_independent_remeasurement": True,
                "phase_b_must_apply_remeasured_direction_in_memory": True,
                "receipt_can_reconstruct_safe_direction": False,
                "parameter_mutation_performed": False,
            }
        )
        action_rows = []
        identity_rows = []
        for event in bank["selected_events"]:
            pair = event["pair_id"]
            source = event["source_sample_id"]
            for schedule in runtime.SCHEDULE_INDICES:
                action_rows.append(
                    {
                        "pair_id": pair,
                        "source_sample_id": source,
                        "schedule_index": schedule,
                    }
                )
                coordinate = _sha(f"{pair}.{source}.s{schedule}")
                for family in runtime.IDENTITY_FAMILIES:
                    for sketch in range(runtime.SKETCH_COUNT):
                        identity_rows.append(
                            {
                                "pair_id": pair,
                                "source_sample_id": source,
                                "schedule_index": schedule,
                                "family": family,
                                "sketch_index": sketch,
                                "source_coordinate_receipt_digest": coordinate,
                            }
                        )
        return runtime._seal(
            {
                "schema_version": runtime.RUN_RECEIPT_SCHEMA,
                "method_name": runtime.METHOD_NAME,
                "audit_complete": True,
                "geometry_audit_performed": True,
                "geometry_audit_passed": go,
                "primary_replication_go": go,
                "optimizer_constructed": False,
                "optimizer_step_called": False,
                "candidate_delta_constructed": False,
                "parameter_add_called": False,
                "parameter_mutation_performed": False,
                "parameter_update_authorized": False,
                "scientific_action_editing_success_claim": False,
                "topology": "WORLD8-DP2xUlysses-SP4",
                "frame_count": 81,
                "fps": 25.0,
                "primary_pair_ids": list(runtime.PAIR_IDS),
                "primary_schedule_indices": list(runtime.SCHEDULE_INDICES),
                "action_condition_count": 8,
                "identity_probe_count": 64,
                "preregistration": {
                    "file_sha256": self.plan_file_sha,
                    "preregistration_digest": self.plan[
                        "preregistration_digest"
                    ],
                    "preregistration_alone_geometry_measurement_authorized": False,
                },
                "live_cast_bank_binding": bank,
                "gradient_information_flow": {
                    "pure_t2v_action_gradient_count": 8,
                    "deployment_identity_probe_count": 64,
                    "unprojected_rows_preserved_until_root_solver": True,
                    "local_project_then_average": False,
                    "mask_flow_pose_track_or_trajectory_used": False,
                    "raw_gradient_artifact_written": False,
                },
                "action_gradient_metadata": action_rows,
                "identity_probe_metadata": identity_rows,
                "world_solver_authority": solver,
                "multicondition_transport_receipt": transport,
                "phase_b_handoff": {
                    "phase_a2_safe_direction_persisted": False,
                    "receipt_can_reconstruct_safe_direction": False,
                    "if_primary_replication_go_then_next_job_must_remeasure": True,
                    "phase_b_must_apply_remeasured_direction_in_memory": True,
                    "phase_b_is_separate_root_authorized_job": True,
                },
                "rank_runtime_provenance": [[{} for _ in range(4)] for _ in range(8)],
                "runtime_source": {
                    "revision": runtime_revision,
                    "archive_sha256": runtime_archive,
                    "post_audit_unchanged": True,
                },
            }
        )

    def test_postflight_accepts_scientific_no_go_without_authorizing_phase_b(self) -> None:
        receipt = self._synthetic_final_receipt(go=False)
        result = postflight.validate_phase_a2_receipt(receipt)
        self.assertTrue(result["postflight_passed"])
        self.assertFalse(result["primary_replication_go"])
        self.assertFalse(result["phase_b_authorized"])
        self.assertEqual(result["phase_b_next_action"], "terminate_no_go_no_update")
        self.assertFalse(result["receipt_can_reconstruct_safe_direction"])

    def test_identity_families_must_share_each_source_coordinate(self) -> None:
        rows = []
        for pair_index, pair in enumerate(runtime.PAIR_IDS):
            for source_index in range(2):
                source = f"{pair}-source-{source_index}"
                for schedule in runtime.SCHEDULE_INDICES:
                    coordinate = _sha(f"{pair}.{source}.s{schedule}")
                    for family in runtime.IDENTITY_FAMILIES:
                        for sketch in range(runtime.SKETCH_COUNT):
                            rows.append(
                                {
                                    "pair_id": pair,
                                    "source_sample_id": source,
                                    "schedule_index": schedule,
                                    "family": family,
                                    "sketch_index": sketch,
                                    "source_coordinate_receipt_digest": coordinate,
                                }
                            )
        cells = runtime.validate_cross_family_identity_coordinate_closure(rows)
        self.assertEqual(len(cells), 8)
        self.assertTrue(
            all(row["cross_family_coordinate_consensus"] for row in cells)
        )
        poisoned = copy.deepcopy(rows)
        poisoned[4]["source_coordinate_receipt_digest"] = _sha(
            "wrong-family-coordinate"
        )
        with self.assertRaisesRegex(
            runtime.PairV7PhaseA2Error, "do not share one source coordinate"
        ):
            runtime.validate_cross_family_identity_coordinate_closure(poisoned)

    def test_static_read_only_exact81_multicondition_closure(self) -> None:
        source = (METHOD_ROOT / runtime.__file__).read_text(encoding="utf-8") if not Path(runtime.__file__).is_absolute() else Path(runtime.__file__).read_text(encoding="utf-8")
        self.assertIn("SCHEDULE_INDICES = (16, 35)", source)
        self.assertIn("EXPECTED_ACTION_COUNT = 8", source)
        self.assertIn("EXPECTED_IDENTITY_COUNT = 64", source)
        self.assertIn("solver_execution_count\": 1", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("decode_latents", source)
        self.assertNotIn("safe_direction_artifact_written\": True", source)

    def test_parser_requires_explicit_read_only_acknowledgements(self) -> None:
        parser = runtime.build_parser()
        acknowledgement_actions = {
            action.dest: action.default
            for action in parser._actions
            if action.dest.startswith("ack_")
        }
        self.assertEqual(
            acknowledgement_actions,
            {
                "ack_root_reviewed_phase_a2_launch": False,
                "ack_no_parameter_mutation_no_success_claim": False,
            },
        )

    def test_deferred_import_closure_uses_real_repository_modules(self) -> None:
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        for absent_alias in (
            "checkpoint_content_audit",
            "pair_v5_cagd_v3",
            "distributed_runtime_contract",
        ):
            self.assertNotIn(absent_alias, source)
        for real_module in (
            "infer_source_kv_carrier_oracle.py",
            "pair_v5_t2v_guidance_distill.py",
            "source_self_runtime.py",
            "pair_v7_multicondition_nullspace_transport.py",
        ):
            self.assertTrue((METHOD_ROOT / real_module).is_file(), real_module)

    def test_auh_launcher_pins_world8_and_receipt_only_phase_b_handoff(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts"
            / "auh_audit_pair_v7_phase_a2_multicondition_dp2sp4.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", launcher)
        self.assertIn("--nproc_per_node=8", launcher)
        self.assertIn("--nnodes=2 --nproc_per_node=4", launcher)
        self.assertIn("PAIR_V7_TORCHRUN_NODE_RANK", launcher)
        self.assertIn("PAIR_V7_TORCHRUN_MASTER_ADDR", launcher)
        self.assertIn("PAIR_V7_VISIBLE_DEVICES", launcher)
        self.assertIn('runtime_binding_archive="${runtime_archive}"', launcher)
        self.assertIn(
            '--runtime-source-archive "${runtime_binding_archive}"', launcher
        )
        self.assertIn(
            "registered two-node GPU quartet must be physical 0,1,2,3",
            launcher,
        )
        self.assertIn(
            "cfd065c6ad84c4598b76ae5d0c390fe69bcbcaaad643045af4ac719c030f52df",
            launcher,
        )
        self.assertIn(
            "ad8a9d6ae462c28f48bf8c20cb649903ccca3941af6660e0cab4be214eba1790",
            launcher,
        )
        self.assertIn(
            "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95",
            launcher,
        )
        self.assertIn("postflight_pair_v7_phase_a2_multicondition.py", launcher)
        self.assertIn("separate Phase-B job to remeasure", launcher)
        self.assertIn("applies the solved direction in memory", launcher)
        self.assertIn("output must contain only receipt.json", launcher)
        self.assertNotIn("optimizer.step", launcher)
        self.assertNotIn("decode_latents", launcher)


if __name__ == "__main__":
    unittest.main()
