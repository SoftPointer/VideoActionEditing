from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_pair_v5_t2v_calibration_bank as runner
import pair_v5_t2v_calibration_bank_spec as contract
from tools import author_pair_v5_t2v_calibration_bank as author


LAUNCHER = METHOD_ROOT / "scripts/auh_pair_v5_t2v_calibration_bank_dual4.sbatch"
FIRST8 = METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
CORE4 = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_selection_v1.json"
RESERVE4 = METHOD_ROOT / "assets/pair_v5_t2v_calibration_reserve4_selection_v1.json"
CORE4_BANK = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v1.json"
CORE4_BANK_SHA256 = "655484e8a19774c21618c74f825057c1ed9b22014214907f8354e991f07799ec"


def _caption(split: str, branch: str) -> str:
    return (
        f"A continuous studio portrait shows the registered {split} performer and a separate "
        f"background actor. The main performer executes the complete {branch.replace('_', ' ')} "
        "motion description with explicit temporal order and a stable terminal state. The shot "
        "remains coherent under constant illumination without cuts or unrelated scene changes."
    )


def _candidate(branch: str, index: int, split: str) -> dict[str, object]:
    caption = _caption(split, branch)
    actor = f"actor-{split}"
    scene = f"scene-{split}"
    action_group = f"action-instance-{split}"
    return {
        "candidate_id": f"turn-head-{split}-{index:02d}-{branch}",
        "analysis_split": split,
        "action_family_id": "turn-head-right",
        "calibration_group_id": f"cell-turn-head-{split}",
        "prompt_group_id": f"{actor}--{scene}",
        "action_family_group_id": action_group,
        "actor_group_id": actor,
        "scene_group_id": scene,
        "action_group_id": action_group,
        "geometry_source_video": "/dataset/geometry-exact81.mp4",
        "geometry_source_video_sha256": "a" * 64,
        "geometry_contract": contract.GEOMETRY_CONTRACT,
        "semantic_branch": branch,
        "full_t2v_caption": caption,
        "full_t2v_caption_utf8_sha256": hashlib.sha256(caption.encode()).hexdigest(),
        "caption_contract": contract.CAPTION_CONTRACT,
        "seed": 17 if split == "fit" else 29,
    }


def _spec() -> dict[str, object]:
    fit = [
        _candidate(branch, index, "fit")
        for index, branch in enumerate(contract.MACE_BRANCH_ORDER)
    ]
    confirmation = [
        _candidate(branch, index, "confirmation")
        for index, branch in enumerate(contract.MACE_BRANCH_ORDER)
    ]
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "sampling_contract": contract.SAMPLING_CONTRACT,
        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
        "artifact_use_contract": contract.ARTIFACT_USE_CONTRACT,
        "split_contract": contract.SPLIT_CONTRACT,
        "groups": [
            {"group_id": "sp4-a", "visible_gpus": [0, 1, 2, 3], "candidates": fit},
            {"group_id": "sp4-b", "visible_gpus": [4, 5, 6, 7], "candidates": confirmation},
        ],
    }


def _write_spec(root: Path, value: object | None = None) -> tuple[Path, str]:
    path = root / "sealed.json"
    path.write_bytes(contract.canonical_json_bytes(_spec() if value is None else value) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _file_artifact(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _valid_resource_lifecycle() -> dict[str, object]:
    residency_rows = [
        {
            "rank": rank,
            "local_rank": rank,
            "hostname": "test-node",
            "guard_required": True,
            "guard_active": True,
            "module_path": "model.t5_text_encoder",
            "exact_positional_cpu_offload_request_only": True,
            "cpu_offload_requests_observed": 1,
            "cpu_offload_requests_suppressed": 1,
            "successful_cpu_materializations": 0,
            "delegated_to_requests": 1,
            "parameter_device_before": f"cuda:{rank}",
            "parameter_device_after": f"cuda:{rank}",
            "storage_fingerprint_before": f"{rank + 1:064x}",
            "storage_fingerprint_after": f"{rank + 1:064x}",
            "guard_method_restored": True,
            "vmrss_kib": 1000 + rank,
            "vmhwm_kib": 2000 + rank,
            "gpu_memory_limit_gib": runner.native.T2V_GPU_MEMORY_LIMIT_GIB,
            "gpu_memory_limit_bytes": runner.native.T2V_GPU_MEMORY_LIMIT_BYTES,
            "gpu_total_memory_bytes": 64 * 1024**3,
            "gpu_peak_allocated_bytes": 31 * 1024**3 + rank,
            "gpu_peak_reserved_bytes": 32 * 1024**3 + rank,
            "gpu_peak_reserved_within_limit": True,
        }
        for rank in range(4)
    ]
    return {
        **runner.native.T2V_RESOURCE_LIFECYCLE_CONTRACT,
        "world4_load_completion_gate": {
            "schema_version": runner.native.WORLD4_LOAD_COMPLETION_GATE_SCHEMA,
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "renderer_gpu_resident_trimmed_monotonic_ns_by_rank": [
                101,
                102,
                103,
                104,
            ],
            "load_completion_barrier_returned_monotonic_ns_by_rank": [
                201,
                202,
                203,
                204,
            ],
            "source_tokenizer_setup_entered_monotonic_ns_by_rank": [
                301,
                302,
                303,
                304,
            ],
            "native_sampling_entered_monotonic_ns_by_rank": [401, 402, 403, 404],
            "world4_barrier_completed_before_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_any_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_first_native_sampling": True,
        },
        "world4_t2v_text_encoder_gpu_residency_gate": {
            "schema_version": (
                runner.native.T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA
            ),
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "module_path": "model.t5_text_encoder",
            "rank_evidence": residency_rows,
            "all_rank_exactly_one_cpu_offload_request_suppressed": True,
            "all_rank_zero_successful_cpu_materializations": True,
            "all_rank_gpu_resident_before_and_after_sampling": True,
            "all_rank_storage_fingerprint_unchanged": True,
            "all_rank_guard_method_restored": True,
            "all_rank_peak_reserved_within_52_gib": True,
        },
    }


def _native_receipt(
    output: Path,
    candidate: dict[str, object],
    *,
    gaussian_payload: bytes,
    gaussian_container_payload: Optional[bytes] = None,
) -> dict[str, object]:
    mp4 = _file_artifact(output / "t2v.mp4", f"mp4:{candidate['candidate_id']}".encode())
    clean = _file_artifact(
        output / "t2v.normalized-clean-latent.safetensors",
        f"latent:{candidate['candidate_id']}".encode(),
    )
    clean.update(
        {
            "shape": [1, 16, 21, 62, 60],
            "native_sampler_before_vae_decode": True,
            "mp4_decode_reencode_used": False,
        }
    )
    mp4.update(
        {
            "frame_count": 81,
            "fps": 25,
            "height": 496,
            "width": 480,
            "normalized_clean_latent": clean,
        }
    )
    gaussian = _file_artifact(
        output / "t2v.official-initial-gaussian.safetensors",
        gaussian_payload
        if gaussian_container_payload is None
        else gaussian_container_payload,
    )
    gaussian.update(
        {
            "shape": [1, 16, 21, 62, 60],
            "dtype": "torch.float32",
            "stored_dtype": "torch.float32",
            "generator_initial_seed": candidate["seed"],
            "captured_from_native_sampler": True,
            "external_initial_noise_injection": False,
            "source_or_target_derived": False,
            "observer_changed_return_value": False,
            "official_randn_tensor_call_count": 1,
            "raw_value_sha256": hashlib.sha256(gaussian_payload).hexdigest(),
            "content_sha256": hashlib.sha256(b"content:" + gaussian_payload).hexdigest(),
        }
    )
    guidance = contract.SAMPLING_CONTRACT["guidance"]
    receipt: dict[str, object] = {
        "schema_version": runner.native.SCHEMA_VERSION,
        "method": runner.native.METHOD,
        "arms": ["t2v"],
        "input": {
            "source_video_sha256": candidate["geometry_source_video_sha256"],
            "action_prompt_utf8_sha256": candidate["full_t2v_caption_utf8_sha256"],
            "target_video": False,
            "external_reference_image_or_video": False,
            "external_mask_flow_pose_track_trajectory": False,
            "external_first_frame_anchor": False,
        },
        "preprocessing": {
            "frame_count": 81,
            "fps": 25,
            "source_derived_bucket_hw": [496, 480],
        },
        "conditioning": {
            "t2v": {
                "full_source_video_count": 0,
                "source_derived_reference_count": 0,
                "source_frame_indices": [],
                "reference_encoding": "none",
                "source_ids": {
                    "target_source_id": 0,
                    "video_source_ids": [],
                    "reference_source_ids": [],
                    "conditioning_source_count": 0,
                    "max_conditioning_source_id": 0,
                    "within_pretrained_source_ids_1_through_5": True,
                    "source_id_interpolation_required": False,
                },
            }
        },
        "condition_identities": {
            "rank_zero_broadcasts": {"references": {}, "full_source_video": None},
            "references": {},
            "full_source_video": None,
        },
        "source_condition_artifact": None,
        "sampling": {
            "t2v": {
                "num_frames": 81,
                "num_inference_steps": 40,
                "guidance_mode": "t2v_apg",
                "seed": candidate["seed"],
                "omega_txt": guidance["omega_txt"],
                "omega_vid": guidance["omega_vid"],
                "omega_img": guidance["omega_img"],
                "target_initialization": contract.TARGET_INITIALIZATION,
                "target_mixed_with_source_latent": False,
                "custom_sampler_or_scheduler": False,
                "ulysses_size": 4,
            }
        },
        "latent_geometry": {"video_latent_shape": [1, 16, 21, 62, 60]},
        "resource_lifecycle": _valid_resource_lifecycle(),
        "outputs": {"t2v": mp4},
        "initial_noise_artifacts": {"t2v": gaussian},
        "interpretation": {"training_performed": False},
    }
    receipt["receipt_digest"] = contract.sha256_bytes(contract.canonical_json_bytes(receipt))
    (output / "receipt.json").write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
    return receipt


class PairV5T2VCalibrationSpecTests(unittest.TestCase):
    def test_exact_mace_order_exact81_dual_sp4_and_source_free_closure(self) -> None:
        normalized = contract.validate_root_spec(_spec())
        self.assertEqual(
            contract.MACE_BRANCH_ORDER,
            (
                "action", "noop", "incomplete", "reverse", "shuffle",
                "wrong_actor", "wrong_object", "camera_only", "appearance_only",
                "generic_wrong_motion",
            ),
        )
        self.assertEqual(normalized["sampling_contract"]["num_frames"], 81)
        self.assertEqual(normalized["sampling_contract"]["latent_frames"], 21)
        self.assertEqual(normalized["sampling_contract"]["num_inference_steps"], 40)
        self.assertTrue(normalized["sampling_contract"]["same_cell_official_gaussian_reuse_by_seed"])
        self.assertEqual(
            [group["visible_gpus"] for group in normalized["groups"]],
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )
        self.assertFalse(contract.SEMANTIC_INPUT_CLOSURE["source_content_conditioning"])
        self.assertFalse(contract.SEMANTIC_INPUT_CLOSURE["proposal_media_as_condition"])

    def test_closed_candidate_rejects_privileged_slots(self) -> None:
        for forbidden in (
            "source_latent", "source_reference", "target_video", "proposal_media",
            "donor", "pseudo_target", "student_input", "initial_noise", "mask",
            "flow", "pose", "track", "trajectory",
        ):
            value = _spec()
            value["groups"][0]["candidates"][0][forbidden] = "/forbidden"
            with self.subTest(forbidden=forbidden), self.assertRaises(
                contract.PairT2VCalibrationSpecError
            ):
                contract.validate_root_spec(value)

    def test_cell_requires_ordered_ten_branches_same_seed_geometry_and_sp4(self) -> None:
        mutations = []
        missing = _spec(); missing["groups"][1]["candidates"].pop(); mutations.append(missing)
        reordered = _spec()
        reordered["groups"][0]["candidates"][0], reordered["groups"][0]["candidates"][1] = (
            reordered["groups"][0]["candidates"][1], reordered["groups"][0]["candidates"][0]
        )
        mutations.append(reordered)
        seed_drift = _spec(); seed_drift["groups"][1]["candidates"][0]["seed"] = 18; mutations.append(seed_drift)
        cross_sp4 = _spec(); cross_sp4["groups"][1]["candidates"].append(cross_sp4["groups"][0]["candidates"].pop()); mutations.append(cross_sp4)
        for index, value in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(contract.PairT2VCalibrationSpecError):
                contract.validate_root_spec(value)

    def test_split_is_actor_scene_action_group_disjoint_with_family_coverage(self) -> None:
        for axis in contract.SPLIT_GROUP_AXES:
            value = _spec()
            fit_value = value["groups"][0]["candidates"][0][axis]
            for row in value["groups"][1]["candidates"]:
                row[axis] = fit_value
                if axis == "action_group_id":
                    row["action_family_group_id"] = fit_value
                elif axis in ("actor_group_id", "scene_group_id"):
                    row["prompt_group_id"] = (
                        f"{row['actor_group_id']}--{row['scene_group_id']}"
                    )
            with self.subTest(axis=axis), self.assertRaisesRegex(
                contract.PairT2VCalibrationSpecError, "overlap"
            ):
                contract.validate_root_spec(value)
        missing_family = _spec()
        for row in missing_family["groups"][1]["candidates"]:
            row["action_family_id"] = "different-family"
        with self.assertRaisesRegex(contract.PairT2VCalibrationSpecError, "fit and confirmation"):
            contract.validate_root_spec(missing_family)

    def test_raw_hash_seals_materialized_branch_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = _write_spec(root)
            manifest = contract.materialize_plan(
                spec_path=path, expected_sha256=digest, output_dir=root / "plan"
            )
            self.assertEqual(len(manifest["candidate_records"]), 20)
            envelope = contract.load_candidate_envelope(
                manifest["candidate_records"][7]["path"], digest
            )
            self.assertEqual(envelope["split_contract"], contract.SPLIT_CONTRACT)
            self.assertRegex(envelope["candidate_envelope_sha256"], r"^[0-9a-f]{64}$")
            with self.assertRaises(contract.PairT2VCalibrationSpecError):
                contract.load_sealed_spec(path, "f" * 64)


class PairV5T2VCalibrationRuntimeTests(unittest.TestCase):
    def test_native_receipt_rejects_unproved_serial_load_or_deferred_vae(self) -> None:
        for field in (
            "serialized_host_checkpoint_load_required",
            "world4_renderer_retirement_barrier_before_rank_zero_vae_load",
            "sampling_model_and_vae_not_host_resident_concurrently_for_t2v",
            "t2v_text_encoder_gpu_residency_required",
            "t2v_text_encoder_cpu_offload_bypass_active",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp).resolve()
                candidate = _spec()["groups"][0]["candidates"][0]
                receipt = _native_receipt(
                    output, candidate, gaussian_payload=b"one-gaussian"
                )
                receipt["resource_lifecycle"][field] = False
                unsigned = dict(receipt)
                unsigned.pop("receipt_digest")
                receipt["receipt_digest"] = contract.sha256_bytes(
                    contract.canonical_json_bytes(unsigned)
                )
                with self.assertRaisesRegex(
                    contract.PairT2VCalibrationSpecError,
                    "resource lifecycle differs",
                ):
                    runner._verify_native_receipt(receipt, candidate)

    def test_native_receipt_rejects_missing_and_resigned_old_lifecycle_schemas(
        self,
    ) -> None:
        for mutation in (
            "missing-lifecycle",
            "old-native-schema",
            "old-lifecycle-schema",
            "old-load-gate-schema",
            "old-t5-residency-gate-schema",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp).resolve()
                candidate = _spec()["groups"][0]["candidates"][0]
                receipt = _native_receipt(
                    output, candidate, gaussian_payload=b"one-gaussian"
                )
                if mutation == "missing-lifecycle":
                    receipt.pop("resource_lifecycle")
                elif mutation == "old-native-schema":
                    receipt["schema_version"] = (
                        "bernini-native-identity-generation-canary-v1"
                    )
                elif mutation == "old-lifecycle-schema":
                    receipt["resource_lifecycle"]["schema_version"] = (
                        "bernini-native-t2v-resource-lifecycle-v2"
                    )
                elif mutation == "old-load-gate-schema":
                    receipt["resource_lifecycle"]["world4_load_completion_gate"][
                        "schema_version"
                    ] = "bernini-native-world4-renderer-load-completion-gate-v0"
                else:
                    receipt["resource_lifecycle"][
                        "world4_t2v_text_encoder_gpu_residency_gate"
                    ]["schema_version"] = (
                        "bernini-native-t2v-text-encoder-gpu-residency-gate-v0"
                    )
                unsigned = dict(receipt)
                unsigned.pop("receipt_digest")
                receipt["receipt_digest"] = contract.sha256_bytes(
                    contract.canonical_json_bytes(unsigned)
                )
                expected = (
                    "resource lifecycle"
                    if mutation == "missing-lifecycle"
                    else (
                        "pinned frozen T2V-only arm"
                        if mutation == "old-native-schema"
                        else "resource lifecycle differs"
                    )
                )
                with self.assertRaisesRegex(
                    contract.PairT2VCalibrationSpecError, expected
                ):
                    runner._verify_native_receipt(receipt, candidate)

    def test_pair_receipt_loader_rejects_unknown_schema_even_with_valid_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair-receipt.json"
            receipt = {field: None for field in runner._PAIR_RECEIPT_FIELDS}
            receipt["schema_version"] = (
                "pair-v5-frozen-bernini-t2v-calibration-receipt-v999"
            )
            unsigned = dict(receipt)
            unsigned.pop("receipt_digest")
            receipt["receipt_digest"] = contract.sha256_bytes(
                contract.canonical_json_bytes(unsigned)
            )
            path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
            with self.assertRaisesRegex(
                contract.PairT2VCalibrationSpecError, "receipt schema differs"
            ):
                runner._load_pair_receipt(path)

    def test_runner_forwards_only_t2v_geometry_prompt_and_cell_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = _write_spec(root)
            manifest = contract.materialize_plan(
                spec_path=path, expected_sha256=digest, output_dir=root / "plan"
            )
            captured: list[list[str]] = []
            old_main, old_bind = runner.native.main, runner.bind_receipt
            old_values = (runner.native.OMEGA_TEXT, runner.native.OMEGA_VIDEO, runner.native.OMEGA_IMAGE)
            old_env = {key: os.environ.get(key) for key in ("ROCR_VISIBLE_DEVICES", "RANK")}
            try:
                os.environ.update({"ROCR_VISIBLE_DEVICES": "0,1,2,3", "RANK": "1"})
                runner.native.main = lambda argv: captured.append(list(argv)) or 0
                runner.bind_receipt = lambda args, envelope: Path(args.output_dir)
                for index, record in enumerate(manifest["candidate_records"][:2]):
                    status = runner.main(
                        [
                            "--candidate-spec", record["path"],
                            "--expected-root-spec-sha256", digest,
                            "--output-dir", str(root / f"output-{index}"),
                            "--bernini-root", "/bernini", "--veomni-root", "/veomni",
                            "--checkpoint", "/checkpoint",
                            "--checkpoint-content-manifest", "/checkpoint.sha256",
                            "--method-source-revision", "b" * 40,
                            "--method-source-archive-sha256", "c" * 64,
                        ]
                    )
                    self.assertEqual(status, 0)
            finally:
                runner.native.main, runner.bind_receipt = old_main, old_bind
                runner.native.OMEGA_TEXT, runner.native.OMEGA_VIDEO, runner.native.OMEGA_IMAGE = old_values
                for key, prior in old_env.items():
                    if prior is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = prior
            self.assertEqual(len(captured), 2)
            for argv in captured:
                self.assertEqual(argv[argv.index("--arms") + 1], "t2v")
                self.assertEqual(argv[argv.index("--num-inference-steps") + 1], "40")
                self.assertEqual(argv[argv.index("--seed") + 1], "17")
                self.assertNotIn("rv2v", argv)
                for forbidden in ("--target", "--reference", "--mask", "--flow", "--initial-noise"):
                    self.assertNotIn(forbidden, argv)

    def test_full_bank_receipt_proves_same_gaussian_without_event_or_update_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = _write_spec(root)
            manifest = contract.materialize_plan(
                spec_path=path, expected_sha256=digest, output_dir=root / "plan"
            )
            rendered = root / "rendered"; rendered.mkdir()
            old_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
            try:
                for record in manifest["candidate_records"]:
                    envelope = contract.load_candidate_envelope(record["path"], digest)
                    candidate = envelope["candidate"]
                    output = rendered / candidate["candidate_id"]; output.mkdir()
                    gaussian_payload = f"gaussian:{candidate['analysis_split']}".encode()
                    _native_receipt(
                        output,
                        candidate,
                        gaussian_payload=gaussian_payload,
                        # Exercise the real runtime case: the sampled tensor
                        # value is identical inside a cell, while independent
                        # safetensors containers can have distinct headers.
                        gaussian_container_payload=(
                            gaussian_payload
                            + b":container:"
                            + candidate["semantic_branch"].encode()
                        ),
                    )
                    os.environ["ROCR_VISIBLE_DEVICES"] = ",".join(
                        str(item) for item in envelope["visible_gpus"]
                    )
                    runner.bind_receipt(argparse.Namespace(output_dir=str(output)), envelope)
            finally:
                if old_visible is None:
                    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                else:
                    os.environ["ROCR_VISIBLE_DEVICES"] = old_visible
            bank = runner.audit_rendered_bank(
                root_spec=path, expected_sha256=digest, output_dir=rendered
            )
            self.assertEqual(bank["candidate_count"], 20)
            self.assertEqual(bank["cell_count"], 2)
            self.assertTrue(bank["fit_confirmation_all_registered_axes_disjoint"])
            self.assertTrue(
                all(
                    row["all_ten_official_gaussian_tensor_values_byte_equal"]
                    for row in bank["same_cell_gaussian_proofs"]
                )
            )
            self.assertTrue(
                all(
                    len(set(row["official_gaussian_file_sha256_by_branch"].values()))
                    == 10
                    for row in bank["same_cell_gaussian_proofs"]
                )
            )
            self.assertFalse(bank["interpretation"]["event_qualification_performed"])
            self.assertFalse(bank["interpretation"]["optimizer_authorized"])
            self.assertFalse(bank["interpretation"]["t2v_negative_media_are_rv2v_policy_candidates"])


class PairV5T2VAuthoringTests(unittest.TestCase):
    def test_core4_sealed_bank_is_exact40_candidates_and_hash_bound(self) -> None:
        bank, digest = contract.load_sealed_spec(CORE4_BANK, CORE4_BANK_SHA256)
        rows = [row for group in bank["groups"] for row in group["candidates"]]
        self.assertEqual(digest, CORE4_BANK_SHA256)
        self.assertEqual(len(rows), 40)
        self.assertEqual({row["analysis_split"] for row in rows}, {"fit", "confirmation"})
        self.assertEqual(
            {row["action_family_id"] for row in rows},
            {"dog-sit-facing-camera", "human-rise-to-stand"},
        )
        self.assertEqual(
            {
                (row["analysis_split"], row["action_family_id"])
                for row in rows
            },
            {
                ("fit", "dog-sit-facing-camera"),
                ("confirmation", "dog-sit-facing-camera"),
                ("fit", "human-rise-to-stand"),
                ("confirmation", "human-rise-to-stand"),
            },
        )

    def test_core4_is_first_job_40_candidates_and_first8_registry_is_80(self) -> None:
        value = json.loads(FIRST8.read_bytes())
        cells = value["cells"]
        self.assertEqual(
            {cell["iid"] for cell in cells},
            {
                "00435ad621c44fac", "0c6915018a5f4d9b", "33322eb8ec1e4703",
                "71ba57892bd043df", "7b88a1ca1f804f41", "841b5e0080a1441d",
                "a35b590961d24694", "a66e6818e4144928",
            },
        )
        for family in ("dog-sit-facing-camera", "human-rise-to-stand"):
            rows = [cell for cell in cells if cell["action_family_id"] == family]
            self.assertEqual({row["analysis_split"] for row in rows}, {"fit", "confirmation"})
            self.assertEqual(len({row["actor_group_id"] for row in rows}), 2)
        self.assertTrue(all(list(cell["branch_descriptions"]) == list(contract.MACE_BRANCH_ORDER) for cell in cells))
        for cell in cells:
            scene = cell["scene_caption"].lower()
            self.assertNotIn("second adult", scene)
            self.assertNotIn("another adult", scene)
            self.assertNotIn("second dog", scene)
            self.assertNotIn("another dog", scene)
            wrong_actor = cell["branch_descriptions"]["wrong_actor"].lower()
            self.assertTrue("background" in wrong_actor or "distant" in wrong_actor)
            camera_only = cell["branch_descriptions"]["camera_only"].lower()
            for invented_plural in ("both adults", "both people", "both dogs"):
                self.assertNotIn(invented_plural, camera_only)
        first8 = author.inspect_authoring(
            authoring_path=FIRST8,
            expected_authoring_sha256=hashlib.sha256(FIRST8.read_bytes()).hexdigest(),
        )
        core4 = author.inspect_authoring(
            authoring_path=CORE4,
            expected_authoring_sha256=hashlib.sha256(CORE4.read_bytes()).hexdigest(),
        )
        reserve4 = author.inspect_authoring(
            authoring_path=RESERVE4,
            expected_authoring_sha256=hashlib.sha256(RESERVE4.read_bytes()).hexdigest(),
        )
        self.assertEqual(first8["candidate_count"], 80)
        self.assertEqual(core4["candidate_count"], 40)
        self.assertEqual(reserve4["candidate_count"], 40)
        core_value = json.loads(CORE4.read_bytes())
        reserve_value = json.loads(RESERVE4.read_bytes())
        self.assertTrue(core_value["first_gpu_job_default"])
        self.assertFalse(reserve_value["first_gpu_job_default"])
        self.assertEqual(
            set(core4["selected_iids"]),
            {"7b88a1ca1f804f41", "841b5e0080a1441d", "a35b590961d24694", "a66e6818e4144928"},
        )

    def test_helper_computes_geometry_and_caption_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry_fit = root / "fit.mp4"; geometry_fit.write_bytes(b"fit exact81 geometry")
            geometry_confirm = root / "confirm.mp4"; geometry_confirm.write_bytes(b"confirm exact81 geometry")
            cells = []
            for split, geometry, group in (
                ("fit", geometry_fit, "sp4-a"),
                ("confirmation", geometry_confirm, "sp4-b"),
            ):
                cells.append(
                    {
                        "iid": f"iid-{split}",
                        "analysis_split": split,
                        "action_family_id": "test-action-family",
                        "actor_group_id": f"actor-{split}",
                        "scene_group_id": f"scene-{split}",
                        "action_group_id": f"action-{split}",
                        "execution_group": group,
                        "geometry_source_video": str(geometry),
                        "seed": 11 if split == "fit" else 29,
                        "scene_caption": "A continuous full body view shows one main performer and one distant secondary performer.",
                        "branch_descriptions": {
                            branch: f"The registered {branch.replace('_', ' ')} branch performs a distinct complete motion and holds its terminal state clearly."
                            for branch in contract.MACE_BRANCH_ORDER
                        },
                        "camera_caption": "The continuous shot retains stable lighting and coherent temporal detail throughout.",
                    }
                )
            draft = {
                "schema_version": contract.AUTHORING_SCHEMA_VERSION,
                "bank_id": "test-bank",
                "expected_cell_count": 2,
                "cells": cells,
            }
            draft_path = root / "draft.json"
            draft_path.write_bytes(contract.canonical_json_bytes(draft) + b"\n")
            draft_sha = hashlib.sha256(draft_path.read_bytes()).hexdigest()
            output = root / "sealed-bank.json"
            receipt = author.write_bank(
                authoring_path=draft_path,
                expected_authoring_sha256=draft_sha,
                output_path=output,
            )
            bank, output_sha = contract.load_sealed_spec(output, receipt["output_raw_sha256"])
            self.assertEqual(output_sha, receipt["output_raw_sha256"])
            self.assertEqual(receipt["candidate_count"], 20)
            fit = bank["groups"][0]["candidates"][0]
            self.assertEqual(fit["geometry_source_video_sha256"], hashlib.sha256(geometry_fit.read_bytes()).hexdigest())
            self.assertEqual(fit["full_t2v_caption_utf8_sha256"], hashlib.sha256(fit["full_t2v_caption"].encode()).hexdigest())


class PairV5T2VCalibrationLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_and_exact_dual_sp4_concurrency(self) -> None:
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.text)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.text)
        self.assertEqual(self.text.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.text)
        self.assertIn("candidates_serial_within_group=true", self.text)

    def test_launcher_is_t2v_only_and_runs_strong_bank_audit(self) -> None:
        self.assertIn("PAIR_V5_T2V_BANK_SPEC_SHA256", self.text)
        self.assertIn('[[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]]', self.text)
        invocation = self.text.split(
            '"${method_root}/infer_pair_v5_t2v_calibration_bank.py"', 1
        )[1].split("\n  done", 1)[0]
        for forbidden in (
            "--target", "--donor", "--pseudo-target", "--student-input", "--mask",
            "--flow", "--pose", "--track", "--trajectory", "--initial-noise",
        ):
            self.assertNotIn(forbidden, invocation)
        self.assertIn("--audit-bank --root-spec", self.text)
        self.assertIn("40-branch ``core4``", self.text)
        self.assertIn("same_cell_gaussian=required", self.text)
        self.assertIn("PAIR_V5_T2V_CALIBRATION_BANK_DUAL4_STRONG_AUDIT_OK", self.text)


if __name__ == "__main__":
    unittest.main()
