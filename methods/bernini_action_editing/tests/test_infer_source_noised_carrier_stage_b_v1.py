from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "infer_source_noised_carrier_stage_b_v1.py"
ROLE_PATH = METHOD_ROOT / "source_self_role_repaint.py"
SHARED_RUNTIME_PATH = METHOD_ROOT / "source_self_runtime.py"

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

SPEC = importlib.util.spec_from_file_location("stage_b_infer_tested", RUNTIME_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def digest(value: object) -> str:
    return runtime.object_sha256(value)


def with_digest(value: dict[str, object]) -> dict[str, object]:
    return {**value, "digest": digest(value)}


def write_receipt(directory: Path, receipt: dict[str, object]) -> None:
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_digest", None)
    receipt.clear()
    receipt.update(unsigned)
    receipt["receipt_digest"] = digest(unsigned)
    (directory / "receipt.json").write_bytes(runtime.canonical_json_bytes(receipt) + b"\n")


def resign_compute_and_receipt(directory: Path, receipt: dict[str, object]) -> None:
    compute = receipt["compute_consensus"]
    assert isinstance(compute, dict)
    compute["records_sha256"] = digest(receipt["records"])
    compute_unsigned = copy.deepcopy(compute)
    compute_unsigned.pop("digest", None)
    compute["digest"] = digest(compute_unsigned)
    write_receipt(directory, receipt)


class FakeNumpyDType:
    def __init__(self, kind: str, itemsize: int, name: str) -> None:
        self.kind = kind
        self.itemsize = itemsize
        self.name = name

    def __str__(self) -> str:
        return self.name


class FakeNumpyArray:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        kind: str = "f",
        itemsize: int = 4,
        dtype_name: str = "float32",
        finite: bool = True,
        value_min: float = 0.0,
        value_max: float = 1.0,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = FakeNumpyDType(kind, itemsize, dtype_name)
        self.finite = finite
        self.value_min = value_min
        self.value_max = value_max

    def min(self) -> float:
        return self.value_min

    def max(self) -> float:
        return self.value_max


def fake_numpy_module() -> types.ModuleType:
    module = types.ModuleType("numpy")
    module.ndarray = FakeNumpyArray

    class FiniteResult:
        def __init__(self, value: bool) -> None:
            self.value = value

        def all(self) -> bool:
            return self.value

    module.isfinite = lambda decoded: FiniteResult(decoded.finite)
    return module


def make_probe_pair(root: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    mode = "registered-probes"
    output_names = runtime.OUTPUT_FILES_BY_MODE[mode]
    common_source = {
        "path": "/sealed/source.mp4",
        "sha256": runtime.REGISTERED_IID_SEALED_ROW_AUTHORITY[
            "00435ad621c44fac"
        ]["source_video_sha256"],
        "model_condition_from_runtime_reencode": False,
        "display_and_sealed_row_cross_binding_only": True,
    }
    common_dataset = {
        "root": "/sealed/dataset",
        "parquet_sha256": runtime.EXPECTED_DATASET_PARQUET_SHA256,
        "materialization_spec_sha256": runtime.EXPECTED_DATASET_MATERIALIZATION_SPEC_SHA256,
        "receipt_sha256": runtime.EXPECTED_DATASET_RECEIPT_SHA256,
        "receipt_digest": runtime.EXPECTED_DATASET_RECEIPT_DIGEST,
        "iid": "00435ad621c44fac",
        "row_digest": runtime.REGISTERED_IID_SEALED_ROW_AUTHORITY[
            "00435ad621c44fac"
        ]["row_digest"],
        "source_video_sha256": runtime.REGISTERED_IID_SEALED_ROW_AUTHORITY[
            "00435ad621c44fac"
        ]["source_video_sha256"],
        "clean_posterior_blob_sha256": runtime.REGISTERED_IID_SEALED_ROW_AUTHORITY[
            "00435ad621c44fac"
        ]["clean_posterior_blob_sha256"],
        "style_id": 2,
        "style_posterior_blob_sha256": runtime.REGISTERED_IID_SEALED_ROW_AUTHORITY[
            "00435ad621c44fac"
        ]["style_posterior_blob_sha256"],
        "reference_index_set": [0, 40, 80],
        "reference_order": [0, 80, 40],
        "reference_posterior_blob_sha256_in_order": runtime.REGISTERED_IID_SEALED_ROW_AUTHORITY[
            "00435ad621c44fac"
        ]["reference_posterior_blob_sha256_in_order"],
        "six_independent_training_materialization_encodes_reused": True,
    }
    tensor_binding = {
        "clean_sha256": "c" * 64,
        "style_donor_sha256": "d" * 64,
        "reference_sha256_in_order": ["e" * 64, "f" * 64, "0" * 64],
        "epsilon_sha256": "1" * 63 + "0",
        "seed": runtime.INFERENCE_SEED,
    }
    token_binding = with_digest(
        {
            "input_ids_sha256": "2" * 63 + "0",
            "attention_mask_sha256": "3" * 63 + "0",
            "t5_input_lens_sha256": "4" * 63 + "0",
            "text_lens": [17],
            "embedding_sha256": "5" * 63 + "0",
        }
    )
    instruction = {
        "text": runtime.stage_b.GENERIC_INSTRUCTION,
        "utf8_sha256": hashlib.sha256(
            runtime.stage_b.GENERIC_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "matches_stage_b_training_generic_instruction": True,
        "token_and_embedding_binding": token_binding,
    }
    visual = {
        "role_order": [
            "forward_noised_registered_style_donor",
            "independent_clean_ref_slot0",
            "independent_clean_ref_slot1",
            "independent_clean_ref_slot2",
            "target",
        ],
        "source_ids": [1, 2, 3, 4, 0],
        "same_custom_pack_every_query": True,
        "online_rgb_corruption_or_vae_reencode": False,
        "clean_posterior_used_as_donor": False,
    }
    schedule = with_digest(
        {
            "schedule_sha256": runtime.exact40.SCHEDULE_SHA256,
            "fresh_scheduler_instance_for_this_arm": False,
            "scheduler_steps_executed": 0,
            "stateless_registered_schedule_coordinates_only": True,
        }
    )
    placement = with_digest(
        {
            "world_size": 4,
            "local_world_size": 2,
            "nodes": 2,
            "ranks_per_node": 2,
            "ulysses_sp_size": 4,
            "sp4_crosses_nodes": True,
            "rank_hostname_local_rank": [
                {"rank": 0, "local_rank": 0, "hostname": "n0"},
                {"rank": 1, "local_rank": 1, "hostname": "n0"},
                {"rank": 2, "local_rank": 0, "hostname": "n1"},
                {"rank": 3, "local_rank": 1, "hostname": "n1"},
            ],
        }
    )
    common_runtime = {
        "physical_placement": placement,
        "torch": "2.7.1",
        "torch_hip": "6.3",
        "transformers": "4.49",
        "diffusers": "0.34",
        "source_self_runtime_dependency": {
            "path": "/release/source_self_runtime.py",
            "sha256": runtime.EXPECTED_SOURCE_SELF_RUNTIME_SHA256,
        },
    }
    common_model = {
        "bernini_commit": "a" * 40,
        "veomni_commit": "b" * 40,
        "checkpoint_tree_sha256": "c" * 64,
        "checkpoint_content": {"verified_file_count": 23},
        "pinned_inference_source_files": {"bernini/pipeline.py": "d" * 64},
        "single_expert": "transformer_1",
    }
    anchor = {
        "present": False,
        "path": None,
        "sha256": None,
        "full_video_must_be_embedded_by_web_report": False,
        "used_as_model_condition": False,
        "opened_for_hash_binding_only": False,
        "decoded_by_model_runtime": False,
        "vae_encoded_by_model_runtime": False,
        "routed_to_transformer": False,
        "latent_or_rgb_transplanted": False,
    }
    receipts = []
    directories = []
    for arm_index, arm in enumerate(("frozen_base", "trained")):
        directory = root / arm
        directory.mkdir()
        directories.append(directory)
        artifacts = {}
        outputs = []
        records = []
        for coordinate, name in zip(runtime.stage_b.validate_registered_schedule(), output_names):
            payload = f"{arm}-{name}".encode("ascii")
            (directory / name).write_bytes(payload)
            artifact_sha = hashlib.sha256(payload).hexdigest()
            latent_sha = hashlib.sha256(f"{arm}-latent-{name}".encode("ascii")).hexdigest()
            artifacts[name] = artifact_sha
            outputs.append(
                {
                    "name": name,
                    "sha256": artifact_sha,
                    "frames": 81,
                    "fps": 25.0,
                    "hw": [592, 400],
                    "decode_input_latent_sha256": latent_sha,
                    "decode_input_latent_shape": [1, 16, 21, 74, 50],
                    "vae_frozen_eval": True,
                }
            )
            binding = with_digest(
                {
                    "epsilon_sha256": tensor_binding["epsilon_sha256"],
                    "clean_target_sha256": tensor_binding["clean_sha256"],
                    "clean_donor_sha256": tensor_binding["style_donor_sha256"],
                    "noised_target_sha256": "6" * 63 + str(coordinate.optimizer_step_zero_based),
                    "noised_donor_sha256": "7" * 63 + str(coordinate.optimizer_step_zero_based),
                    "same_epsilon_object_target_donor": True,
                    "same_sigma_target_donor": True,
                    "sigma_float32_be_hex": coordinate.sigma_float32_be_hex,
                }
            )
            records.append(
                {
                    **coordinate.receipt(),
                    "target_equation_exact": True,
                    "donor_equation_exact": True,
                    "training_realization_replayed": False,
                    "predicted_clean_equation": "x0_hat=x_target_sigma-sigma*v_raw_conditional",
                    "raw_conditional_single_forward": True,
                    "raw_prediction_dtype": "torch.bfloat16",
                    "noisy_target_dtype": "torch.float32",
                    "sigma_dtype_device": "torch.float32/cpu/0d",
                    "numeric_program": "fp32_noisy-minus-cpu_fp32_sigma-times-bf16_raw_velocity",
                    "route_enabled": True,
                    "prepared_input_digest": "8" * 64,
                    "binding": binding,
                    "predicted_clean_sha256": latent_sha,
                }
            )
        parameter_sha = (
            "9" * 64
            if arm == "frozen_base"
            else runtime.EXPECTED_STAGE_B_FINAL_PARAMETER_SHA256
        )
        adapter = {
            "arm": arm,
            "weights": (
                "all_adapter_tensors_exact_zero"
                if arm == "frozen_base"
                else "strict_stage_b_training_adapter"
            ),
            "route_wrapper_installed": True,
            "all_adapter_tensors_exact_zero": arm == "frozen_base",
            "parameter_sha256": parameter_sha,
            "file_sha256": None if arm == "frozen_base" else runtime.EXPECTED_STAGE_B_ADAPTER_SHA256,
            "training_receipt_sha256": None if arm == "frozen_base" else runtime.EXPECTED_STAGE_B_TRAINING_RECEIPT_SHA256,
            "training_receipt_digest": None if arm == "frozen_base" else runtime.EXPECTED_STAGE_B_TRAINING_RECEIPT_DIGEST,
            "strict_load": None,
        }
        if arm == "frozen_base":
            adapter["same_role_q_o_wrapper_as_trained"] = True
        else:
            adapter["same_role_q_o_wrapper_as_frozen_base"] = True
            strict_load = {
                "schema_version": "bernini-source-noised-carrier-strict-load-v1",
                "path": "/sealed/training/adapter.safetensors",
                "file_sha256": runtime.EXPECTED_STAGE_B_ADAPTER_SHA256,
                "exact40_schedule_sha256": runtime.stage_b.EXPECTED_EXACT40_SCHEDULE_SHA256,
                "registered_schedule_indices": list(
                    runtime.stage_b.REGISTERED_SCHEDULE_INDICES
                ),
                "metadata": {
                    "schema_version": runtime.stage_b.ADAPTER_FILE_SCHEMA,
                    "role_adapter_schema_version": "bernini-source-self-role-repaint-adapter-v1",
                    "block_indices_json": runtime.canonical_json_bytes(list(range(23))).decode("ascii"),
                    "projections_json": runtime.canonical_json_bytes(
                        ["attn1.to_q", "attn1.to_out.0"]
                    ).decode("ascii"),
                    "target_row_only": "true",
                    "role_embedding": "donor_reference_target",
                    "lora_rank": str(runtime.stage_b.LORA_RANK),
                    "lora_alpha_hex": runtime.stage_b.LORA_ALPHA.hex(),
                    "exact40_schedule_sha256": runtime.stage_b.EXPECTED_EXACT40_SCHEDULE_SHA256,
                    "registered_schedule_indices_json": runtime.canonical_json_bytes(
                        list(runtime.stage_b.REGISTERED_SCHEDULE_INDICES)
                    ).decode("ascii"),
                    "target_and_donor_same_epsilon": "true",
                    "forward_noising_only": "true",
                    "inversion_claimed": "false",
                    "matched_carrier_runtime_required": "true",
                },
                "tensor_count": 93,
                "tensor_names_sha256": "e" * 64,
                "strict_tensor_closure": True,
                "base_parameters_frozen": True,
                "forward_noising_only": True,
                "inversion_claimed": False,
            }
            adapter["strict_load"] = with_digest(strict_load)
        preforward = with_digest(
            {
                "dataset_iid": common_dataset["iid"],
                "style_id": 2,
                "clean_sha256": tensor_binding["clean_sha256"],
                "donor_sha256": tensor_binding["style_donor_sha256"],
                "reference_sha256_in_order": tensor_binding["reference_sha256_in_order"],
                "epsilon_sha256": tensor_binding["epsilon_sha256"],
                "text_binding_digest": token_binding["digest"],
                "adapter_parameter_sha256": parameter_sha,
                "source_ids": [1, 2, 3, 4, 0],
            }
        )
        compute = with_digest(
            {
                "records_sha256": digest(records),
                "latent_outputs_sha256": [item["decode_input_latent_sha256"] for item in outputs],
                "epsilon_sha256": tensor_binding["epsilon_sha256"],
                "adapter_parameter_sha256": parameter_sha,
                "text_embedding_sha256": token_binding["embedding_sha256"],
                "runtime_schedule_audit_digest": schedule["digest"],
            }
        )
        if arm == "frozen_base":
            adapter_binding = {
                "training_dataset_exact_match": None,
                "training_model_exact_match": None,
                "training_artifacts_sha256": None,
                "training_final_parameter_sha256": None,
                "training_noise_seeds": None,
                "training_epsilon_sha256": None,
                "inference_seed_absent_from_training_noise_seeds": None,
                "inference_epsilon_absent_from_training_realizations": None,
                "reason": "frozen_base_does_not_open_training_artifacts",
            }
        else:
            adapter_binding = with_digest(
                {
                    "training_dataset_exact_match": True,
                    "training_model_exact_match": True,
                    "training_artifacts_sha256": {
                        "adapter.safetensors": runtime.EXPECTED_STAGE_B_ADAPTER_SHA256,
                        "optimizer.pt": runtime.EXPECTED_STAGE_B_OPTIMIZER_SHA256,
                        "history.json": runtime.EXPECTED_STAGE_B_HISTORY_SHA256,
                    },
                    "training_final_parameter_sha256": parameter_sha,
                    "training_noise_seeds": list(range(8)),
                    "training_epsilon_sha256": [f"{index:x}" * 64 for index in range(8)],
                    "inference_seed_absent_from_training_noise_seeds": True,
                    "inference_epsilon_absent_from_training_realizations": True,
                }
            )
        receipt = {
            "schema_version": runtime.RECEIPT_SCHEMA,
            "complete": True,
            "arm": arm,
            "mode_contract": runtime.mode_contract(mode),
            "source": copy.deepcopy(common_source),
            "dataset": copy.deepcopy(common_dataset),
            "tensor_binding": copy.deepcopy(tensor_binding),
            "instruction": copy.deepcopy(instruction),
            "visual_pack": copy.deepcopy(visual),
            "execution_counts": {
                "raw_conditional_forward_calls": 4,
                "unconditional_forward_calls": 0,
                "cfg_or_apg_combinations": 0,
                "scheduler_steps": 0,
                "decoded_videos": 4,
            },
            "runtime_schedule_audit": copy.deepcopy(schedule),
            "compute_consensus": compute,
            "preforward_input_consensus": preforward,
            "adapter": adapter,
            "adapter_runtime_binding": adapter_binding,
            "records": records,
            "outputs": outputs,
            "artifacts": artifacts,
            "anchor_action_display": copy.deepcopy(anchor),
            "model": copy.deepcopy(common_model),
            "runtime": copy.deepcopy(common_runtime),
            "method_source_revision": "e" * 40,
            "method_source_archive_sha256": "f" * 64,
            "inversion_claimed": False,
            "method_success_claimed": False,
            "scientific_claim_authorized": False,
        }
        write_receipt(directory, receipt)
        receipts.append(receipt)
    return directories[0], directories[1], receipts[0], receipts[1]


def make_full40_pair(root: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    base, trained, base_receipt, trained_receipt = make_probe_pair(root)
    mode = "full40-evolved-target-all40-route-extrapolation"
    initial_target_sha = hashlib.sha256(b"shared-initial-packed-epsilon").hexdigest()
    for directory, receipt in (
        (base, base_receipt),
        (trained, trained_receipt),
    ):
        arm = receipt["arm"]
        for name in runtime.OUTPUT_FILES_BY_MODE["registered-probes"]:
            (directory / name).unlink()
        output_name = runtime.OUTPUT_FILES_BY_MODE[mode][0]
        payload = f"{arm}-{output_name}".encode("ascii")
        (directory / output_name).write_bytes(payload)
        artifact_sha = hashlib.sha256(payload).hexdigest()
        final_unpacked_sha = hashlib.sha256(
            f"{arm}-final-unpacked".encode("ascii")
        ).hexdigest()
        previous = initial_target_sha
        records = []
        for index in range(40):
            after = hashlib.sha256(f"{arm}-packed-after-{index}".encode("ascii")).hexdigest()
            prepared = hashlib.sha256(
                (
                    "shared-prepared-step-0"
                    if index == 0
                    else f"{arm}-prepared-step-{index}"
                ).encode("ascii")
            ).hexdigest()
            records.append(
                {
                    "schedule_index": index,
                    "timestep_int64": runtime.exact40.PINNED_TIMESTEPS[index],
                    "sigma_float32_be_hex": runtime.exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
                    "registered_training_coordinate": index
                    in runtime.stage_b.REGISTERED_SCHEDULE_INDICES,
                    "route_enabled": True,
                    "target_reanchored": False,
                    "target_matches_stateless_training_formula": False,
                    "initial_target_is_seeded_epsilon_only": index == 0,
                    "donor_same_initial_epsilon_forward_noised": True,
                    "packed_condition_roundtrip_bit_exact": True,
                    "raw_prediction_dtype": "torch.bfloat16",
                    "packed_target_pre_post_dtype": "torch.float32",
                    "packed_target_shape": [1, 21840, 64],
                    "scheduler_step_index_before": None if index == 0 else index,
                    "scheduler_step_index_after": index + 1,
                    "prepared_input_digest": prepared,
                    "noised_style_donor_sha256": hashlib.sha256(
                        f"shared-donor-{index}".encode("ascii")
                    ).hexdigest(),
                    "packed_target_sha256_before_step": previous,
                    "raw_prediction_sha256": hashlib.sha256(
                        f"{arm}-prediction-{index}".encode("ascii")
                    ).hexdigest(),
                    "target_sha256_after_step": after,
                }
            )
            previous = after
        schedule = with_digest(
            {
                "schedule_sha256": runtime.exact40.SCHEDULE_SHA256,
                "timesteps": list(runtime.exact40.PINNED_TIMESTEPS),
                "positive_sigmas": list(runtime.exact40.PINNED_POSITIVE_SIGMAS),
                "positive_sigmas_float32_be_hex": list(
                    runtime.exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
                ),
                "terminal_sigma": 0.0,
                "terminal_sigma_float32_be_hex": runtime.exact40.TERMINAL_SIGMA_FLOAT32_HEX,
                "fresh_scheduler_instance_for_this_arm": True,
                "scheduler_steps_executed": 40,
                "raw_bf16_prediction_passed_without_precast": True,
                "packed_fp32_target_state_evolved": True,
                "scheduler_cursor_pre_post_exact": True,
                "terminal_step_index": 40,
                "final_packed_target_sha256": previous,
                "final_unpacked_decode_latent_sha256": final_unpacked_sha,
            }
        )
        receipt["mode_contract"] = runtime.mode_contract(mode)
        receipt["execution_counts"] = {
            "raw_conditional_forward_calls": 40,
            "unconditional_forward_calls": 0,
            "cfg_or_apg_combinations": 0,
            "scheduler_steps": 40,
            "decoded_videos": 1,
        }
        receipt["runtime_schedule_audit"] = schedule
        receipt["records"] = records
        receipt["outputs"] = [
            {
                "name": output_name,
                "sha256": artifact_sha,
                "frames": 81,
                "fps": 25.0,
                "hw": [592, 400],
                "decode_input_latent_sha256": final_unpacked_sha,
                "decode_input_latent_shape": [1, 16, 21, 74, 50],
                "vae_frozen_eval": True,
            }
        ]
        receipt["artifacts"] = {output_name: artifact_sha}
        compute = receipt["compute_consensus"]
        compute["records_sha256"] = digest(records)
        compute["latent_outputs_sha256"] = [final_unpacked_sha]
        compute["runtime_schedule_audit_digest"] = schedule["digest"]
        compute_unsigned = copy.deepcopy(compute)
        compute_unsigned.pop("digest", None)
        compute["digest"] = digest(compute_unsigned)
        write_receipt(directory, receipt)
    return base, trained, base_receipt, trained_receipt


class StaticRuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_runtime_and_training_file_pins(self) -> None:
        self.assertEqual(
            runtime.EXPECTED_SOURCE_SELF_RUNTIME_SHA256,
            "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
        )
        self.assertEqual(
            hashlib.sha256(ROLE_PATH.read_bytes()).hexdigest(),
            "bf212ac4effcd5b3975eefc61e01c71cba366969ec92cf2ff186765ddec43f2e",
        )
        self.assertEqual(
            runtime.REGISTERED_IID_REFERENCE_ORDER,
            {
                "0014a41e55e44670": [0, 40, 80],
                "00435ad621c44fac": [0, 80, 40],
            },
        )
        self.assertEqual(runtime.EXPECTED_SEALED_OUTPUT_HW, (592, 400))
        self.assertEqual(
            runtime.EXPECTED_SEALED_DECODE_LATENT_SHAPE,
            (1, 16, 21, 74, 50),
        )
        self.assertEqual(
            runtime.EXPECTED_STAGE_B_ADAPTER_SHA256,
            "f85de3518ec88ac86a33fcb574c328ae3bca581ea3a1f86a648ef051b14ec16c",
        )
        self.assertEqual(
            runtime.EXPECTED_STAGE_B_TRAINING_RECEIPT_SHA256,
            "7c61d23d00e442a7fada318ade279db1131db57da88d068233325f004cd1dca9",
        )
        self.assertEqual(
            runtime.EXPECTED_STAGE_B_FINAL_PARAMETER_SHA256,
            "710e1715abfa993e7fead4dbd3740de199f0418da5811391a93f5ed22c929f85",
        )

    def test_only_two_scientifically_distinct_modes_exist(self) -> None:
        self.assertEqual(
            runtime.MODES,
            (
                "registered-probes",
                "full40-evolved-target-all40-route-extrapolation",
            ),
        )
        probe = runtime.mode_contract(runtime.MODES[0])
        rollout = runtime.mode_contract(runtime.MODES[1])
        self.assertTrue(probe["training_equation_pack_schedule_coordinate_exact"])
        self.assertFalse(probe["training_realization_replayed"])
        self.assertTrue(probe["one_shared_inference_epsilon_across_registered_probes"])
        self.assertTrue(rollout["rollout_distribution_shift_from_stateless_training"])
        self.assertFalse(rollout["registered_target_training_formula_match_claimed"])
        self.assertEqual(rollout["target_solver_kind"], "noise_to_clean_unipc_denoising")

    def test_iid_style_mapping_and_fresh_seed_fail_closed(self) -> None:
        self.assertEqual(
            runtime.REGISTERED_IID_STYLE,
            {"0014a41e55e44670": 1, "00435ad621c44fac": 2},
        )
        parser = runtime.build_parser()
        destinations = [action.dest for action in parser._actions]
        self.assertEqual(destinations.count("instruction"), 1)
        args = argparse.Namespace(
            arm="frozen_base",
            adapter_checkpoint=None,
            expected_adapter_sha256=None,
            expected_training_receipt_sha256=None,
            anchor_action_video=None,
            expected_anchor_action_sha256=None,
            seed=runtime.INFERENCE_SEED,
            instruction=runtime.stage_b.GENERIC_INSTRUCTION,
            expected_bernini_commit="a" * 40,
            expected_veomni_commit="b" * 40,
            method_source_revision="c" * 40,
            expected_checkpoint_tree_sha256="d" * 64,
            expected_checkpoint_content_manifest_sha256="e" * 64,
            method_source_archive_sha256="f" * 64,
            expected_materialization_spec_sha256=runtime.EXPECTED_DATASET_MATERIALIZATION_SPEC_SHA256,
            expected_dataset_parquet_sha256=runtime.EXPECTED_DATASET_PARQUET_SHA256,
            expected_dataset_receipt_sha256=runtime.EXPECTED_DATASET_RECEIPT_SHA256,
            expected_dataset_receipt_digest=runtime.EXPECTED_DATASET_RECEIPT_DIGEST,
            expected_source_sha256="5" * 64,
        )
        runtime.validate_cli(args)
        args.seed += 1
        with self.assertRaises(runtime.StageBInferenceError):
            runtime.validate_cli(args)

    def test_p0_execution_fragments_are_present(self) -> None:
        for fragment in (
            "strict_load_stage_b_adapter",
            "runtime.prepare_output_transaction",
            "publish_inference_transaction",
            "validate_two_node_two_rank_placement",
            "full40 packed/spatial condition roundtrip is not bit exact",
            "raw_bf16_prediction_passed_without_precast",
            "scheduler_cursor_pre_post_exact",
            "final_unpacked_decode_latent_sha256",
            "inference_epsilon_absent_from_training_realizations",
            "collective_rank_call",
            "patch embedding",
            "validate_pair_receipts",
            "verify-pair",
            "save_validated_vae_decoded_clip",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("strict_load_source_noised_carrier_adapter(", self.source)
        self.assertNotIn("torch.isfinite(video)", self.source)


class PureContractTests(unittest.TestCase):
    def test_vae_decode_numpy_contract_forwards_original_clip_to_writer(self) -> None:
        fake_numpy = fake_numpy_module()
        for itemsize, dtype_name in ((2, "float16"), (4, "float32"), (8, "float64")):
            with self.subTest(dtype=dtype_name), mock.patch.dict(
                sys.modules, {"numpy": fake_numpy}
            ):
                decoded = FakeNumpyArray(
                    (81, 2, 3, 3), itemsize=itemsize, dtype_name=dtype_name
                )
                writer = mock.Mock()
                output_path = Path("/sealed/stage/probe.mp4")
                audit = runtime.save_validated_vae_decoded_clip(
                    decoded,
                    output_path=output_path,
                    expected_height=2,
                    expected_width=3,
                    fps=25,
                    save_output_fn=writer,
                )
                writer.assert_called_once()
                positional, keyword = writer.call_args
                self.assertIs(positional[0], decoded)
                self.assertEqual(positional[1], str(output_path))
                self.assertEqual(keyword, {"fps": 25})
                self.assertEqual(audit["array_type"], "numpy.ndarray")
                self.assertEqual(audit["shape"], [81, 2, 3, 3])
                self.assertEqual(audit["dtype"], dtype_name)
                self.assertTrue(audit["finite"])
                self.assertTrue(audit["normalized_zero_one"])
                self.assertEqual(audit["value_min"], 0.0)
                self.assertEqual(audit["value_max"], 1.0)

    def test_vae_decode_numpy_contract_rejects_wrong_type_dtype_shape_and_values(self) -> None:
        shape = (81, 2, 3, 3)

        class MockTensor:
            ndim = 4
            dtype = "torch.float32"
            shape = (81, 2, 3, 3)

        bad_values: dict[str, object] = {
            "tensor": MockTensor(),
            "bool": FakeNumpyArray(shape, kind="b", itemsize=1, dtype_name="bool"),
            "integer": FakeNumpyArray(shape, kind="i", itemsize=4, dtype_name="int32"),
            "object": FakeNumpyArray(shape, kind="O", itemsize=8, dtype_name="object"),
            "complex": FakeNumpyArray(shape, kind="c", itemsize=8, dtype_name="complex64"),
            "frame_shape": FakeNumpyArray((80, 2, 3, 3)),
            "height_shape": FakeNumpyArray((81, 1, 3, 3)),
            "channel_shape": FakeNumpyArray((81, 2, 3, 4)),
            "nan": FakeNumpyArray(shape, finite=False),
            "inf": FakeNumpyArray(shape, finite=False),
            "below_zero": FakeNumpyArray(shape, value_min=-1.0e-7),
            "above_one": FakeNumpyArray(shape, value_max=1.0000001),
        }

        for label, decoded in bad_values.items():
            with self.subTest(label=label), mock.patch.dict(
                sys.modules, {"numpy": fake_numpy_module()}
            ):
                writer = mock.Mock()
                with self.assertRaises(runtime.StageBInferenceError):
                    runtime.save_validated_vae_decoded_clip(
                        decoded,
                        output_path=Path("/sealed/stage/probe.mp4"),
                        expected_height=2,
                        expected_width=3,
                        fps=25,
                        save_output_fn=writer,
                    )
                writer.assert_not_called()

    def test_rank_zero_decode_contract_failure_is_broadcast_to_all_ranks(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.__path__ = []
        fake_dist = types.ModuleType("torch.distributed")
        broadcast_payload: list[object] = []

        def get_world_size(*, group: object) -> int:
            self.assertIsNotNone(group)
            return 4

        def all_gather_object(
            gathered: list[object], value: object, *, group: object
        ) -> None:
            self.assertIsNotNone(group)
            gathered[:] = [value] * 4

        def rank_zero_broadcast(
            values: list[object], *, src: int, group: object
        ) -> None:
            self.assertEqual(src, 0)
            self.assertIsNotNone(group)
            broadcast_payload[:] = copy.deepcopy(values)

        fake_dist.get_world_size = get_world_size
        fake_dist.all_gather_object = all_gather_object
        fake_dist.broadcast_object_list = rank_zero_broadcast
        fake_torch.distributed = fake_dist
        modules = {"torch": fake_torch, "torch.distributed": fake_dist}
        world_group = object()
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                runtime.StageBInferenceError,
                "cannot publish inference transaction.*decoded numpy contract",
            ):
                runtime.publish_inference_transaction(
                    Path("/sealed/output"),
                    Path("/sealed/stage"),
                    None,
                    rank=0,
                    world_group=world_group,
                    rank_zero_error="decoded numpy contract failed",
                )

        self.assertEqual(len(broadcast_payload), 1)
        payload = broadcast_payload[0]
        self.assertIsInstance(payload, dict)
        self.assertFalse(payload["ok"])
        self.assertIn("decoded numpy contract failed", payload["error"])

        def nonzero_broadcast(
            values: list[object], *, src: int, group: object
        ) -> None:
            self.assertEqual(src, 0)
            self.assertIsNotNone(group)
            values[:] = copy.deepcopy(broadcast_payload)

        fake_dist.broadcast_object_list = nonzero_broadcast
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                runtime.StageBInferenceError,
                "cannot publish inference transaction.*decoded numpy contract",
            ):
                runtime.publish_inference_transaction(
                    Path("/sealed/output"),
                    Path("/sealed/stage"),
                    None,
                    rank=1,
                    world_group=world_group,
                    rank_zero_error=None,
                )

    def test_checkpoint_manifest_validation_is_exact_and_hostile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            rows = []
            for index in range(runtime.CHECKPOINT_CONTENT_FILE_COUNT):
                path = checkpoint / f"f{index:02d}.bin"
                path.write_bytes(f"payload-{index}".encode("ascii"))
                rows.append(f"{runtime.file_sha256(path)}  ./f{index:02d}.bin")
            manifest = root / "manifest.sha256"
            manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
            receipt = runtime.validate_checkpoint_content(
                checkpoint,
                manifest,
                expected_manifest_sha256=runtime.file_sha256(manifest),
            )
            self.assertEqual(receipt["verified_file_count"], 23)
            (checkpoint / "f07.bin").write_bytes(b"tampered")
            with self.assertRaises(runtime.StageBInferenceError):
                runtime.validate_checkpoint_content(
                    checkpoint,
                    manifest,
                    expected_manifest_sha256=runtime.file_sha256(manifest),
                )

    def test_pair_cli_create_only_rejects_existing_output_before_validation(self) -> None:
        # A real pair is exercised by the controller postflight.  This focused
        # unit test locks the create-only O_EXCL publication primitive.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "pair.json"
            output.write_text("occupied", encoding="ascii")
            original = runtime.validate_pair_receipts
            runtime.validate_pair_receipts = lambda *_: {
                "schema_version": runtime.PAIR_RECEIPT_SCHEMA,
                "receipt_digest": "0" * 64,
            }
            try:
                with self.assertRaises(runtime.StageBInferenceError):
                    runtime.verify_pair_main(
                        [
                            "--base-dir", "/unused/base",
                            "--trained-dir", "/unused/trained",
                            "--mode", runtime.MODES[0],
                            "--output", str(output),
                        ]
                    )
            finally:
                runtime.validate_pair_receipts = original
            self.assertEqual(output.read_text(encoding="ascii"), "occupied")

    def test_synthetic_valid_probe_pair_passes_model_free_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, base_receipt, _ = make_probe_pair(Path(temporary).resolve())
            self.assertEqual(base_receipt["dataset"]["iid"], "00435ad621c44fac")
            self.assertEqual(base_receipt["dataset"]["style_id"], 2)
            self.assertEqual(base_receipt["dataset"]["reference_order"], [0, 80, 40])
            pair = runtime.validate_pair_receipts(base, trained, "registered-probes")
            self.assertEqual(pair["schema_version"], runtime.PAIR_RECEIPT_SCHEMA)
            self.assertTrue(pair["only_adapter_parameter_values_intentionally_differ"])
            self.assertEqual(len(pair["receipt_digest"]), 64)

    def test_pair_rejects_digest_consistent_internal_link_mutations(self) -> None:
        mutations = {
            "epsilon_compute_link": lambda receipt: receipt["tensor_binding"].__setitem__(
                "epsilon_sha256", "a" * 64
            ),
            "record_decode_link": lambda receipt: receipt["records"][0].__setitem__(
                "predicted_clean_sha256", "b" * 64
            ),
            "output_latent_geometry": lambda receipt: receipt["outputs"][0].__setitem__(
                "decode_input_latent_shape", [1, 16, 21, 75, 50]
            ),
            "output_media_geometry": lambda receipt: receipt["outputs"][0].__setitem__(
                "hw", [600, 400]
            ),
            "anchor_condition": lambda receipt: receipt["anchor_action_display"].__setitem__(
                "used_as_model_condition", True
            ),
            "runtime_dependency": lambda receipt: receipt["runtime"][
                "source_self_runtime_dependency"
            ].__setitem__("sha256", "c" * 64),
            "unconditional_forward": lambda receipt: receipt["execution_counts"].__setitem__(
                "unconditional_forward_calls", 1
            ),
            "extra_namespace": lambda receipt: receipt.__setitem__("unexpected", True),
            "extra_nested_key": lambda receipt: receipt["dataset"].__setitem__(
                "unexpected", True
            ),
            "missing_nested_key": lambda receipt: receipt["runtime"].pop("torch_hip"),
            "training_epsilon_duplicate": lambda receipt: receipt[
                "adapter_runtime_binding"
            ]["training_epsilon_sha256"].__setitem__(
                7, receipt["adapter_runtime_binding"]["training_epsilon_sha256"][0]
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                base, trained, _, trained_receipt = make_probe_pair(Path(temporary).resolve())
                mutation(trained_receipt)
                write_receipt(trained, trained_receipt)
                with self.assertRaises(runtime.StageBInferenceError):
                    runtime.validate_pair_receipts(base, trained, "registered-probes")

    def test_pair_rejects_both_arm_resigned_alternate_output_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, base_receipt, trained_receipt = make_probe_pair(
                Path(temporary).resolve()
            )
            for directory, receipt in (
                (base, base_receipt),
                (trained, trained_receipt),
            ):
                for output in receipt["outputs"]:
                    output["hw"] = [480, 832]
                    output["decode_input_latent_shape"] = [1, 16, 21, 60, 104]
                write_receipt(directory, receipt)
            with self.assertRaises(runtime.StageBInferenceError):
                runtime.validate_pair_receipts(base, trained, "registered-probes")

    def test_pair_rejects_both_arm_resigned_probe_equation_exploit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, base_receipt, trained_receipt = make_probe_pair(
                Path(temporary).resolve()
            )
            for directory, receipt in (
                (base, base_receipt),
                (trained, trained_receipt),
            ):
                receipt["records"][0]["target_equation_exact"] = False
                resign_compute_and_receipt(directory, receipt)
            with self.assertRaises(runtime.StageBInferenceError):
                runtime.validate_pair_receipts(base, trained, "registered-probes")

    def test_pair_rejects_resigned_equal_adapter_parameter_exploit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, base_receipt, trained_receipt = make_probe_pair(
                Path(temporary).resolve()
            )
            base_parameter = base_receipt["adapter"]["parameter_sha256"]
            trained_receipt["adapter"]["parameter_sha256"] = base_parameter
            trained_receipt["adapter_runtime_binding"][
                "training_final_parameter_sha256"
            ] = base_parameter
            binding_unsigned = copy.deepcopy(trained_receipt["adapter_runtime_binding"])
            binding_unsigned.pop("digest", None)
            trained_receipt["adapter_runtime_binding"]["digest"] = digest(binding_unsigned)
            trained_receipt["preforward_input_consensus"][
                "adapter_parameter_sha256"
            ] = base_parameter
            preforward_unsigned = copy.deepcopy(
                trained_receipt["preforward_input_consensus"]
            )
            preforward_unsigned.pop("digest", None)
            trained_receipt["preforward_input_consensus"]["digest"] = digest(
                preforward_unsigned
            )
            trained_receipt["compute_consensus"][
                "adapter_parameter_sha256"
            ] = base_parameter
            resign_compute_and_receipt(trained, trained_receipt)
            with self.assertRaises(runtime.StageBInferenceError):
                runtime.validate_pair_receipts(base, trained, "registered-probes")

    def test_pair_rejects_both_arm_resigned_reference_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, base_receipt, trained_receipt = make_probe_pair(
                Path(temporary).resolve()
            )
            for directory, receipt in (
                (base, base_receipt),
                (trained, trained_receipt),
            ):
                receipt["dataset"]["reference_order"] = [0, 40, 80]
                write_receipt(directory, receipt)
            with self.assertRaises(runtime.StageBInferenceError):
                runtime.validate_pair_receipts(base, trained, "registered-probes")

    def test_pair_rejects_self_consistent_alternate_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, _, trained_receipt = make_probe_pair(
                Path(temporary).resolve()
            )
            alternate_adapter = "0" * 64
            alternate_optimizer = "1" * 64
            alternate_history = "2" * 64
            alternate_receipt_sha = "3" * 64
            alternate_receipt_digest = "4" * 64
            alternate_final = "5" * 64
            adapter = trained_receipt["adapter"]
            adapter["parameter_sha256"] = alternate_final
            adapter["file_sha256"] = alternate_adapter
            adapter["training_receipt_sha256"] = alternate_receipt_sha
            adapter["training_receipt_digest"] = alternate_receipt_digest
            adapter["strict_load"]["file_sha256"] = alternate_adapter
            strict_unsigned = copy.deepcopy(adapter["strict_load"])
            strict_unsigned.pop("digest", None)
            adapter["strict_load"]["digest"] = digest(strict_unsigned)
            binding = trained_receipt["adapter_runtime_binding"]
            binding["training_artifacts_sha256"] = {
                "adapter.safetensors": alternate_adapter,
                "optimizer.pt": alternate_optimizer,
                "history.json": alternate_history,
            }
            binding["training_final_parameter_sha256"] = alternate_final
            binding_unsigned = copy.deepcopy(binding)
            binding_unsigned.pop("digest", None)
            binding["digest"] = digest(binding_unsigned)
            trained_receipt["preforward_input_consensus"][
                "adapter_parameter_sha256"
            ] = alternate_final
            preforward_unsigned = copy.deepcopy(
                trained_receipt["preforward_input_consensus"]
            )
            preforward_unsigned.pop("digest", None)
            trained_receipt["preforward_input_consensus"]["digest"] = digest(
                preforward_unsigned
            )
            trained_receipt["compute_consensus"][
                "adapter_parameter_sha256"
            ] = alternate_final
            resign_compute_and_receipt(trained, trained_receipt)
            with self.assertRaises(runtime.StageBInferenceError):
                runtime.validate_pair_receipts(base, trained, "registered-probes")

    def test_probe_authority_rejects_resigned_coordinate_schedule_and_dtype(self) -> None:
        cases = {
            "timestep": lambda receipt: receipt["records"][0].__setitem__(
                "timestep_int64", receipt["records"][0]["timestep_int64"] - 1
            ),
            "sigma": lambda receipt: receipt["records"][0].__setitem__(
                "sigma_float32_be_hex", "00000000"
            ),
            "binding_sigma": lambda receipt: receipt["records"][0]["binding"].__setitem__(
                "sigma_float32_be_hex", "00000000"
            ),
            "dtype": lambda receipt: receipt["records"][0].__setitem__(
                "raw_prediction_dtype", "torch.float32"
            ),
            "extra_record_key": lambda receipt: receipt["records"][0].__setitem__(
                "unexpected", True
            ),
            "schedule_value": lambda receipt: receipt["runtime_schedule_audit"].__setitem__(
                "scheduler_steps_executed", 1
            ),
            "schedule_missing": lambda receipt: receipt["runtime_schedule_audit"].pop(
                "stateless_registered_schedule_coordinates_only"
            ),
            "schedule_extra": lambda receipt: receipt["runtime_schedule_audit"].__setitem__(
                "unexpected", True
            ),
        }
        for label, mutation in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                base, trained, base_receipt, trained_receipt = make_probe_pair(
                    Path(temporary).resolve()
                )
                for directory, receipt in (
                    (base, base_receipt),
                    (trained, trained_receipt),
                ):
                    mutation(receipt)
                    if label == "binding_sigma":
                        binding = receipt["records"][0]["binding"]
                        binding_unsigned = copy.deepcopy(binding)
                        binding_unsigned.pop("digest", None)
                        binding["digest"] = digest(binding_unsigned)
                    if label.startswith("schedule_"):
                        schedule = receipt["runtime_schedule_audit"]
                        schedule_unsigned = copy.deepcopy(schedule)
                        schedule_unsigned.pop("digest", None)
                        schedule["digest"] = digest(schedule_unsigned)
                        receipt["compute_consensus"][
                            "runtime_schedule_audit_digest"
                        ] = schedule["digest"]
                    resign_compute_and_receipt(directory, receipt)
                with self.assertRaises(runtime.StageBInferenceError):
                    runtime.validate_pair_receipts(base, trained, "registered-probes")

    def test_synthetic_valid_full40_pair_allows_only_post_step0_trajectory_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, base_receipt, trained_receipt = make_full40_pair(
                Path(temporary).resolve()
            )
            self.assertEqual(
                base_receipt["records"][0]["prepared_input_digest"],
                trained_receipt["records"][0]["prepared_input_digest"],
            )
            self.assertNotEqual(
                base_receipt["records"][1]["prepared_input_digest"],
                trained_receipt["records"][1]["prepared_input_digest"],
            )
            self.assertNotEqual(
                base_receipt["records"][1]["packed_target_sha256_before_step"],
                trained_receipt["records"][1]["packed_target_sha256_before_step"],
            )
            pair = runtime.validate_pair_receipts(
                base,
                trained,
                "full40-evolved-target-all40-route-extrapolation",
            )
            self.assertTrue(pair["only_adapter_parameter_values_intentionally_differ"])

    def test_full40_authority_rejects_resigned_cursor_schedule_chain_and_dtype(self) -> None:
        cases = {
            "cursor_before": lambda receipt: receipt["records"][9].__setitem__(
                "scheduler_step_index_before", 8
            ),
            "cursor_after": lambda receipt: receipt["records"][9].__setitem__(
                "scheduler_step_index_after", 9
            ),
            "timestep": lambda receipt: receipt["records"][6].__setitem__(
                "timestep_int64", receipt["records"][6]["timestep_int64"] - 1
            ),
            "sigma": lambda receipt: receipt["records"][7].__setitem__(
                "sigma_float32_be_hex", "00000000"
            ),
            "registered": lambda receipt: receipt["records"][16].__setitem__(
                "registered_training_coordinate", False
            ),
            "dtype": lambda receipt: receipt["records"][2].__setitem__(
                "raw_prediction_dtype", "torch.float32"
            ),
            "target_chain": lambda receipt: receipt["records"][3].__setitem__(
                "packed_target_sha256_before_step", "0" * 64
            ),
            "record_extra": lambda receipt: receipt["records"][4].__setitem__(
                "unexpected", True
            ),
            "schedule_terminal_cursor": lambda receipt: receipt[
                "runtime_schedule_audit"
            ].__setitem__("terminal_step_index", 39),
            "schedule_timestep": lambda receipt: receipt["runtime_schedule_audit"][
                "timesteps"
            ].__setitem__(5, 0),
            "schedule_sigma": lambda receipt: receipt["runtime_schedule_audit"][
                "positive_sigmas_float32_be_hex"
            ].__setitem__(5, "00000000"),
            "schedule_missing": lambda receipt: receipt["runtime_schedule_audit"].pop(
                "scheduler_cursor_pre_post_exact"
            ),
            "schedule_extra": lambda receipt: receipt["runtime_schedule_audit"].__setitem__(
                "unexpected", True
            ),
        }
        mode = "full40-evolved-target-all40-route-extrapolation"
        for label, mutation in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                base, trained, base_receipt, trained_receipt = make_full40_pair(
                    Path(temporary).resolve()
                )
                for directory, receipt in (
                    (base, base_receipt),
                    (trained, trained_receipt),
                ):
                    mutation(receipt)
                    if label.startswith("schedule_"):
                        schedule = receipt["runtime_schedule_audit"]
                        schedule_unsigned = copy.deepcopy(schedule)
                        schedule_unsigned.pop("digest", None)
                        schedule["digest"] = digest(schedule_unsigned)
                        receipt["compute_consensus"][
                            "runtime_schedule_audit_digest"
                        ] = schedule["digest"]
                    resign_compute_and_receipt(directory, receipt)
                with self.assertRaises(runtime.StageBInferenceError):
                    runtime.validate_pair_receipts(base, trained, mode)

    def test_full40_pair_rejects_initial_prepared_input_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, trained, _, trained_receipt = make_full40_pair(
                Path(temporary).resolve()
            )
            trained_receipt["records"][0]["prepared_input_digest"] = "0" * 64
            resign_compute_and_receipt(trained, trained_receipt)
            with self.assertRaises(runtime.StageBInferenceError):
                runtime.validate_pair_receipts(
                    base,
                    trained,
                    "full40-evolved-target-all40-route-extrapolation",
                )


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class TensorPackingTests(unittest.TestCase):
    def test_wan_pack_unpack_roundtrip_and_probe_numeric_dtype(self) -> None:
        import torch

        runtime._load_heavy_runtime_modules()
        generator = torch.Generator().manual_seed(17)
        spatial = torch.randn((1, 16, 21, 6, 8), generator=generator).float().contiguous()
        packed = runtime._pack_field(spatial, spatial.shape)
        restored = runtime._unpack_field(packed, spatial.shape)
        self.assertTrue(torch.equal(restored, spatial))
        noisy = torch.randn(spatial.shape, generator=generator).float()
        velocity = torch.randn(spatial.shape, generator=generator).to(torch.bfloat16)
        sigma = torch.tensor(0.625, dtype=torch.float32, device="cpu")
        result = noisy - sigma * velocity
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, noisy - sigma * velocity))


if __name__ == "__main__":
    unittest.main()
