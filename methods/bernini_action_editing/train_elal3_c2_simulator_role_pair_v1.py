#!/usr/bin/env python3
"""Preregistered real-Bernini C2 simulator oracle-q diagnostic.

This is an exact-two-row, ten-update diagnostic.  It compares a duplicate
compute control with paired target/role-swap supervision.  Both arms start
from the same official Bernini-R 1.3B base, install the existing full-w64
ELAL-3 module, and use a small non-zero depth/width-derived residual gate.

This program is NOT source+instruction inference, formal C2, exact160,
real-video evidence, or a scientific result.  Oracle q is derived from the
simulator clean media.  Frozen Bernini supplies initialization and a step-zero
safety reference only; it is never a velocity teacher or loss target.

The C1 trainer and ELAL-3 core are imported as frozen implementation
dependencies.  They are not modified by this file.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import weakref
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_elal3_c1_simulator_overfit_v1 as c1


METHOD = "bernini-elal3-c2-simulator-role-pair-v1"
RECEIPT_SCHEMA = "bernini-elal3-c2-simulator-role-pair-training-receipt-v1"
PREFLIGHT_SCHEMA = "bernini-elal3-c2-simulator-role-pair-preflight-receipt-v1"
CHECKPOINT_SCHEMA = "bernini-elal3-c2-simulator-role-pair-checkpoint-v1"
CROSS_ARM_GATE_SCHEMA = "bernini-elal3-c2-cross-arm-preflight-gate-v1"
FRESH1_ACCEPTANCE_GATE_SCHEMA = (
    "bernini-elal3-c2-fresh1-acceptance-gate-v1"
)
FRESH1_ORIGIN_ATTESTATION_SCHEMA = (
    "bernini-elal3-c2-fresh1-origin-attestation-v1"
)
EXACT10_ORIGIN_ATTESTATION_SCHEMA = (
    "bernini-elal3-c2-exact10-origin-attestation-v1"
)
EXPERIMENT_CONTRACT_SCHEMA = (
    "bernini-elal3-c2-role-binding-experiment-contract-v1"
)
EXPERIMENT_CONTRACT_SHA256 = (
    "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"
)
EXPERIMENT_CONTRACT_SIZE = 8_553
EXPERIMENT_CONTRACT_DIGEST = (
    "18462dcfbeb017e48a7ed6816559667fa8de1911081261cdc103bc6dd9a229d6"
)
EXTERNAL_AUTHORITY_SCHEMA = (
    "bernini-elal3-c2-simulator-oracle-q-derivative-authority-v1"
)
EXTERNAL_AUTHORITY_SHA256 = (
    "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a"
)
EXTERNAL_AUTHORITY_DIGEST = (
    "936e91cf3d1d39dd7f45d5f7a4d510dadcbcb4c2f89a8d22581638fccdefd599"
)
MODEL_AUTHORITY_SCHEMA = "bernini-elal3-c2-real-model-authority-v1"
MODEL_AUTHORITY_SHA256 = (
    "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d"
)
MODEL_AUTHORITY_DIGEST = (
    "c2c0c9037dea2fd56aa13ac56416bf38c6167686c75b69f0b4b568c82e670c1f"
)
PACKET_MANIFEST_SHA256 = (
    "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
)
C1_TRAINER_SHA256 = (
    "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3"
)
C1_CORE_SHA256 = (
    "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862"
)
C1_TRAINER_SIZE = 90_600
C1_CORE_SIZE = 31_330
C2_LABEL_SHA256 = (
    "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11"
)
C2_LABEL_SIZE = 76_939
C2_MATERIALIZER_SHA256 = (
    "b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f"
)
C2_MATERIALIZER_SIZE = 50_334
TRAIN_LORA_SHA256 = (
    "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5"
)
TRAIN_LORA_SIZE = 66_931
PACKED_LORA_SHA256 = (
    "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6"
)
PACKED_LORA_SIZE = 30_419
RUNTIME_SHA256 = (
    "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f"
)
RUNTIME_SIZE = 36_607
SIGMA_SHA256 = (
    "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3"
)
SIGMA_SIZE = 17_956
CHECKPOINT_EXACT23_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_EXACT23_MANIFEST_SIZE = 2_350
BERNINI_PARALLEL_INIT_SHA256 = (
    "ef16834c0af0e4e2201db37fbbd3a13be6622ac8e09d076a6e6bf68543c9bc29"
)
BERNINI_PARALLEL_INIT_SIZE = 1_485
CHECKPOINT_EXACT23_RELATIVE_PATHS = (
    ".gitattributes",
    "README.md",
    "assets/arena.png",
    "assets/bernini-icon.png",
    "config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
LATENT_BUNDLE_SCHEMA = "bernini-elal3-simulator-c2-exact16-latent-bundle-v1"
LATENT_RECEIPT_SCHEMA = (
    "bernini-elal3-simulator-c2-exact16-latent-bundle-receipt-v1"
)
MATERIALIZER_RUN_COMPLETE_SCHEMA = (
    "bernini-elal3-c2-exact16-materializer-run-complete-v1"
)
# Frozen retry2 release.  The failed/withdrawn retry1 is deliberately absent
# from this consumer, so it cannot be selected through a CLI-only digest.
LATENT_BUNDLE_SHA256 = (
    "b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
)
LATENT_BUNDLE_SIZE = 78_277_976
LATENT_BUNDLE_RECEIPT_SHA256 = (
    "a1ca0d3c015a54d61c8a71d00bc78688dab20d6592ba30ddf73b0ea18e7d70ee"
)
LATENT_BUNDLE_RECEIPT_SIZE = 52_752
LATENT_BUNDLE_RECEIPT_DIGEST = (
    "225255f5ada73848686b240c4a53001c9dd65b1373da2b293c2da8c2ec14f35d"
)
MATERIALIZER_RUN_COMPLETE_SHA256 = (
    "c6eee4766943c7959a2c1ad9b8b6b4e823dec054b31d2fdfb5d03aacd9f7e1ac"
)
MATERIALIZER_RUN_COMPLETE_SIZE = 2_666
MATERIALIZER_RUN_COMPLETE_DIGEST = (
    "186d10a0635a826ebb9bd34dcbc9af7cd23ae45881877c2d252981290edf6d6d"
)
ROW_IDS = (
    "c2-three-entity-blocking-response",
    "c2-three-entity-handover-occlusion",
)
VARIANT_ORDER = (
    "source",
    "target",
    "anchor",
    "wrong_agent",
    "wrong_object",
    "role_swap",
    "reverse",
    "phase_shuffle",
)
TRAIN_VARIANTS = ("target", "role_swap")
ARM_DUPLICATE = "A_duplicate_control"
ARM_ROLE_PAIR = "B_paired_role"
ARM_ROLE_REPLICA = "B_paired_role_replica"
ARM_IDS = (ARM_DUPLICATE, ARM_ROLE_PAIR, ARM_ROLE_REPLICA)
ARM_PLACEMENT = {
    ARM_DUPLICATE: ("141620", "auh7-1b-gpu-226", 20260821),
    ARM_ROLE_PAIR: ("141618", "auh7-1b-gpu-249", 20260821),
    ARM_ROLE_REPLICA: ("141619", "auh7-1b-gpu-257", 20260822),
}
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
BLOCKS = 30
HIDDEN = 1536
MAX_STEPS = 10
LATENT_SHAPE = (1, 16, 21, 52, 70)
PATCH_GRID = (21, 26, 35)
BUCKET_HW = (416, 560)
DEFAULT_LR = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
MEMORY_FRACTION_GATE = 0.5
ACTIVATION_CHECKPOINT_PROFILE = c1.ACTIVATION_CHECKPOINT_PROFILE
ACTIVATION_CHECKPOINT_BLOCKS = c1.ACTIVATION_CHECKPOINT_BLOCKS
ACTIVATION_UNCHECKPOINTED_BLOCKS = c1.ACTIVATION_UNCHECKPOINTED_BLOCKS
BERNINI_COMMIT = c1.BERNINI_COMMIT
VEOMNI_COMMIT = c1.VEOMNI_COMMIT
CHECKPOINT_TREE_SHA256 = c1.CHECKPOINT_TREE_SHA256
LORA_RANK = 256
LORA_AFFINES = 240
ELAL3_FULL_W64_PARAMETERS = 9_979_934
EXPECTED_TRAINABLE_PARAMETERS = 198_723_614
CLAIM_BOUNDARIES = {
    "teacher_forced_oracle_q_simulator_diagnostic_only": True,
    "formal_c2_authorized": False,
    "exact160_authorized": False,
    "real_video_claim_authorized": False,
    "scientific_claim_authorized": False,
    "source_instruction_inference_authorized": False,
}
CONTROLLED_GAIN_FORMULA = "1/(30*sqrt(1536))"
CONTROLLED_GAIN_FLOAT32_HEX = "3a5ef53f"
CONTROLLED_GAIN = struct.unpack(">f", bytes.fromhex(CONTROLLED_GAIN_FLOAT32_HEX))[0]
STEP0_RATIO_BOUND = 1.0 / math.sqrt(HIDDEN)
# The exact ten training coordinates selected by the frozen sigma source.  The
# evaluation coordinate is deliberately separate and exactly (999, 1.0).
TRAINING_SIGMA_EXACT10 = tuple(
    {
        "optimizer_step": index,
        "cycle_index": 0,
        "schedule_index": index,
        "timestep": timestep,
        "sigma": float(struct.unpack(">f", bytes.fromhex(sigma_hex))[0]),
        "sigma_float32_be_hex": sigma_hex,
    }
    for index, (timestep, sigma_hex) in enumerate(
        zip(
            (999, 994, 989, 984, 978, 972, 965, 959, 952, 945),
            (
                "3f7fffef",
                "3f7eb1f9",
                "3f7d560b",
                "3f7beb53",
                "3f7a70da",
                "3f78e594",
                "3f77485b",
                "3f7597f0",
                "3f73d2f4",
                "3f71f7e6",
            ),
        )
    )
)
ROLE_ONLY_CELL_ORDER = tuple(
    (row_id, clean_variant)
    for row_id in ROW_IDS
    for clean_variant in TRAIN_VARIANTS
)
LATENT_TENSOR_ORDER = tuple(
    f"r{row_index:02d}__{row_id.replace('-', '_')}__{variant}"
    for row_index, row_id in enumerate(ROW_IDS)
    for variant in VARIANT_ORDER
)
PREFLIGHT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "method",
        "arm_id",
        "branch_recipe",
        "holder_job_id",
        "node",
        "seed",
        "preflight_only",
        "completed_optimizer_steps",
        "optimizer_constructed",
        "resume_consumed",
        "recipe_version_digest",
        "common_comparison_payload",
        "common_comparison_payload_digest",
        "initial_trainable_sha256",
        "row_input_noise_schedule_digest",
        "second_branch_descriptor",
        "actual_shape_preflight",
        "step0_gain_safety",
        "step0_full_q_route",
        "step0_role_only_cells",
        "step0_role_only_input_invariants",
        "step0_role_only_input_invariants_validation",
        "step0_evaluation_forward_evidence",
        "step0_evaluation_forward_evidence_validation",
        "all_preflight_hard_gates_pass",
        "experiment_contract_sha256",
        "external_authority_sha256",
        "model_authority_sha256",
        "latent_bundle_sha256",
        "runner_source_sha256",
        "source_pins",
        "claim_boundaries",
        "pre_publish_closure_replays",
        "receipt_digest",
    }
)
CROSS_ARM_ALLOWED_DIFFERENCES = (
    "arm_id",
    "branch_recipe",
    "holder_job_id",
    "node",
    "second_branch_descriptor",
    "actual_shape_preflight.runtime_telemetry",
    "step0_gain_safety.runtime_telemetry",
    "pre_publish_closure_replays.runtime_telemetry",
)
COMMON_PREFLIGHT_PAYLOAD_FIELDS = frozenset(
    {
        "experiment_contract_sha256",
        "experiment_contract_digest",
        "external_authority_sha256",
        "external_authority_digest",
        "model_authority_sha256",
        "model_authority_digest",
        "latent_bundle_sha256",
        "latent_bundle_receipt_sha256",
        "materializer_run_complete_sha256",
        "materializer_run_complete_digest",
        "checkpoint_exact23_manifest_sha256",
        "checkpoint_exact23_binding_digest",
        "bernini_execution_source_binding_digest",
        "latent_tensor_order",
        "latent_tensor_order_digest",
        "latent_tensor_rows",
        "source_pins",
        "trainable_parameter_count",
        "trainable_inventory_digest",
        "initial_trainable_sha256",
        "row_common_target_inputs",
        "common_target_branch_schedule",
        "row_input_noise_schedule_digest",
        "step0_common_target_prediction_sha256_by_row",
        "step0_full_q_route_digest",
        "step0_role_only_cells_digest",
        "step0_role_only_input_invariants_digest",
        "step0_evaluation_forward_evidence_digest",
        "optimizer_recipe",
        "rng_recipe",
        "execution_memory_contract",
        "fresh_official_base",
        "resume_consumed",
    }
)
FRESH1_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "method",
        "arm_id",
        "branch_recipe",
        "holder_job_id",
        "node",
        "seed",
        "preflight_only",
        "requested_optimizer_steps",
        "completed_optimizer_steps",
        "optimizer_constructed",
        "optimizer_state_empty_before_first_update",
        "resume_consumed",
        "fresh_official_base",
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "parameters_changed",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
        "own_preflight_binding",
        "cross_arm_gate_binding",
        "actual_shape_training",
        "gradient_gate",
        "memory_gate",
        "checkpoint_gate",
        "history",
        "history_validation",
        "all_fresh1_acceptance_gates_pass",
        "experiment_contract_sha256",
        "external_authority_sha256",
        "model_authority_sha256",
        "latent_bundle_sha256",
        "runner_source_sha256",
        "source_pins",
        "claim_boundaries",
        "pre_publish_closure_replays",
        "receipt_digest",
    }
)
EXACT10_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "method",
        "arm_id",
        "branch_recipe",
        "holder_job_id",
        "node",
        "seed",
        "requested_optimizer_steps",
        "completed_optimizer_steps",
        "optimizer_constructed",
        "optimizer_state_empty_before_first_update",
        "fresh_official_base",
        "resume_consumed",
        "fresh1_checkpoint_consumed",
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "parameters_changed",
        "own_preflight_binding",
        "cross_arm_gate_binding",
        "fresh1_acceptance_gate_binding",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
        "history",
        "history_validation",
        "checkpoint_records",
        "checkpoint_tree_closure",
        "step0_full_q_route",
        "step0_role_only_cells",
        "step0_role_only_input_invariants",
        "step0_role_only_input_invariants_validation",
        "step0_evaluation_forward_evidence",
        "step0_evaluation_forward_evidence_validation",
        "step10_evidence",
        "step10_gate",
        "latent_hard_gates_pass",
        "latent_hard_gate_error",
        "decoded_track_effect_gate_pending",
        "selection_eligible",
        "selection_requires_decoded_track_effect_conjunction",
        "primary_metric_if_all_gates_pass",
        "weighted_metric_sum_used",
        "pre_publish_closure_replays",
        "experiment_contract_sha256",
        "external_authority_sha256",
        "model_authority_sha256",
        "latent_bundle_sha256",
        "runner_source_sha256",
        "source_pins",
        "claim_boundaries",
        "formal_c2_authorized",
        "exact160_authorized",
        "real_video_claim_authorized",
        "scientific_claim_authorized",
        "source_instruction_inference",
        "elapsed_seconds",
        "receipt_digest",
    }
)


class ELAL3C2TrainingError(RuntimeError):
    """Raised before accepting an ambiguous C2 update or result."""


@dataclass(frozen=True)
class EvaluationCoordinateV1:
    timestep: int = 999
    sigma: float = 1.0

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "timestep": self.timestep,
            "renderer_timestep_dtype": "torch.int64",
            "renderer_timestep_cpu_origin": True,
            "sigma": self.sigma,
            "sigma_float32_be_hex": struct.pack(">f", self.sigma).hex(),
            "x_sigma_equals_epsilon": True,
        }


def fail(message: str) -> NoReturn:
    raise ELAL3C2TrainingError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3C2TrainingError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return c1.file_sha256(path)


def _read_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: Optional[int],
    label: str,
) -> Mapping[str, Any]:
    if expected_size is not None and path.stat().st_size != expected_size:
        fail(f"{label} byte size differs")
    try:
        return c1.read_bound_json(
            path,
            expected_sha256=expected_sha256,
            label=label,
            require_canonical_newline=False,
        )
    except c1.ELAL3C1TrainingError as error:
        raise ELAL3C2TrainingError(str(error)) from error


def _require_sha256_v1(value: Any, *, label: str) -> str:
    try:
        return c1._require_sha(value, label=label)
    except c1.ELAL3C1TrainingError as error:
        raise ELAL3C2TrainingError(str(error)) from error


def _require_sealed_json_path_v1(path: Path, *, label: str) -> Path:
    requested = path.expanduser()
    try:
        resolved = requested.resolve(strict=True)
        info = requested.lstat()
    except OSError as error:
        raise ELAL3C2TrainingError(f"{label} is unavailable") from error
    if (
        not requested.is_absolute()
        or requested != resolved
        or requested.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
    ):
        fail(f"{label} is not one sealed canonical 0444 file")
    return resolved


def _read_sealed_json_held_fd_v1(
    path: Path, *, expected_sha256: str, label: str
) -> Mapping[str, Any]:
    """Read canonical JSON twice through one held, no-follow, sealed FD."""

    expected = _require_sha256_v1(expected_sha256, label=f"{label} expected SHA")
    resolved = _require_sealed_json_path_v1(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(resolved), flags)
    try:
        before = os.fstat(descriptor)
        named_before = resolved.stat()

        def read_pass() -> bytes:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)

        first = read_pass()
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = read_pass()
        after = os.fstat(descriptor)
        named_after = resolved.stat()
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        first != second
        or identity(before) != identity(after)
        or identity(before) != identity(named_before)
        or identity(before) != identity(named_after)
        or hashlib.sha256(first).hexdigest() != expected
    ):
        fail(f"{label} held-FD double-read identity/hash differs")
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = item
        return result
    try:
        value = json.loads(
            first.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: fail(
                f"{label} contains non-finite JSON token {token}"
            ),
        )
    except ELAL3C2TrainingError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3C2TrainingError(f"{label} is not strict JSON") from error
    if not isinstance(value, Mapping) or first != canonical_json_bytes(value) + b"\n":
        fail(f"{label} is not canonical JSON with one newline")
    return value


def _validate_common_preflight_payload_v1(
    common: Mapping[str, Any],
    *,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    expected_seed: int,
) -> Mapping[str, Any]:
    """Validate the exact A/B-comparable target-branch recipe, not just a digest."""

    if set(common) != COMMON_PREFLIGHT_PAYLOAD_FIELDS:
        fail("common preflight payload field closure differs")
    scalar_sha_fields = (
        "experiment_contract_sha256",
        "experiment_contract_digest",
        "external_authority_sha256",
        "external_authority_digest",
        "model_authority_sha256",
        "model_authority_digest",
        "latent_bundle_sha256",
        "latent_bundle_receipt_sha256",
        "materializer_run_complete_sha256",
        "materializer_run_complete_digest",
        "checkpoint_exact23_manifest_sha256",
        "checkpoint_exact23_binding_digest",
        "bernini_execution_source_binding_digest",
        "latent_tensor_order_digest",
        "trainable_inventory_digest",
        "initial_trainable_sha256",
        "row_input_noise_schedule_digest",
        "step0_full_q_route_digest",
        "step0_role_only_cells_digest",
        "step0_role_only_input_invariants_digest",
        "step0_evaluation_forward_evidence_digest",
    )
    if any(
        _require_sha256_v1(common.get(field), label=f"common {field}")
        != common.get(field)
        for field in scalar_sha_fields
    ):
        fail("common preflight SHA field differs")
    if (
        common.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or common.get("experiment_contract_digest") != EXPERIMENT_CONTRACT_DIGEST
        or common.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or common.get("external_authority_digest") != EXTERNAL_AUTHORITY_DIGEST
        or common.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or common.get("model_authority_digest") != MODEL_AUTHORITY_DIGEST
        or common.get("latent_bundle_sha256") != expected_bundle_sha256
        or common.get("materializer_run_complete_sha256")
        != MATERIALIZER_RUN_COMPLETE_SHA256
        or common.get("materializer_run_complete_digest")
        != MATERIALIZER_RUN_COMPLETE_DIGEST
        or common.get("checkpoint_exact23_manifest_sha256")
        != CHECKPOINT_EXACT23_MANIFEST_SHA256
        or common.get("source_pins") != dict(expected_source_pins)
        or common.get("trainable_parameter_count")
        != EXPECTED_TRAINABLE_PARAMETERS
        or common.get("fresh_official_base") is not True
        or common.get("resume_consumed") is not False
        or common.get("latent_tensor_order") != list(LATENT_TENSOR_ORDER)
        or common.get("latent_tensor_order_digest")
        != object_sha256(list(LATENT_TENSOR_ORDER))
    ):
        fail("common preflight authority/recipe binding differs")

    latent_rows = common.get("latent_tensor_rows")
    if not isinstance(latent_rows, list) or len(latent_rows) != 16:
        fail("common preflight exact16 tensor rows differ")
    for index, (key, row) in enumerate(zip(LATENT_TENSOR_ORDER, latent_rows)):
        row_index = index // len(VARIANT_ORDER)
        variant = VARIANT_ORDER[index % len(VARIANT_ORDER)]
        if (
            not isinstance(row, Mapping)
            or row.get("tensor_key") != key
            or row.get("row_index") != row_index
            or row.get("row_id") != ROW_IDS[row_index]
            or row.get("variant") != variant
            or row.get("shape") != list(LATENT_SHAPE)
            or row.get("dtype") != "torch.float32"
        ):
            fail(f"common preflight latent row differs: {key}")
        _require_sha256_v1(
            row.get("tensor_sha256"), label=f"common {key} tensor SHA"
        )
        _require_sha256_v1(
            row.get("source_media_sha256"),
            label=f"common {key} source media SHA",
        )

    row_inputs = common.get("row_common_target_inputs")
    expected_input_fields = {
        "row_index",
        "row_id",
        "source_tensor_sha256",
        "target_tensor_sha256",
        "role_swap_tensor_sha256",
        "instruction_sha256",
        "target_q_digest",
        "role_swap_q_digest",
        "target_label_digest",
        "role_swap_label_digest",
        "target_mismatch_digest",
        "role_swap_mismatch_digest",
    }
    if not isinstance(row_inputs, list) or len(row_inputs) != 2:
        fail("common preflight target input exact2 closure differs")
    for row_index, row in enumerate(row_inputs):
        source_tensor_row = latent_rows[
            row_index * len(VARIANT_ORDER) + VARIANT_ORDER.index("source")
        ]
        target_tensor_row = latent_rows[
            row_index * len(VARIANT_ORDER) + VARIANT_ORDER.index("target")
        ]
        role_tensor_row = latent_rows[
            row_index * len(VARIANT_ORDER) + VARIANT_ORDER.index("role_swap")
        ]
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_input_fields
            or row.get("row_index") != row_index
            or row.get("row_id") != ROW_IDS[row_index]
            or row.get("source_tensor_sha256")
            != source_tensor_row.get("tensor_sha256")
            or row.get("target_tensor_sha256")
            != target_tensor_row.get("tensor_sha256")
            or row.get("role_swap_tensor_sha256")
            != role_tensor_row.get("tensor_sha256")
        ):
            fail("common preflight target row identity differs")
        for field in expected_input_fields - {"row_index", "row_id"}:
            _require_sha256_v1(
                row.get(field), label=f"common row {row_index} {field}"
            )

    prediction_rows = common.get("step0_common_target_prediction_sha256_by_row")
    if (
        not isinstance(prediction_rows, list)
        or len(prediction_rows) != 2
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {"row_index", "row_id", "prediction_sha256", "hash_projection"}
            or row.get("row_index") != row_index
            or row.get("row_id") != ROW_IDS[row_index]
            or _require_sha256_v1(
                row.get("prediction_sha256"),
                label=f"step0 target prediction row {row_index}",
            )
            != row.get("prediction_sha256")
            or validate_prediction_hash_projection_receipt_v1(
                row.get("hash_projection"),
                expected_prediction_sha256=str(row.get("prediction_sha256")),
                expected_original_device_type="cuda",
                expected_original_device_index=row_index * SP_SIZE,
                expected_original_dtype=PREDICTION_HASH_PROJECTION_PRODUCTION_DTYPE,
                expected_original_stride=PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE,
                expected_original_storage_offset=(
                    PREDICTION_HASH_PROJECTION_PRODUCTION_STORAGE_OFFSET
                ),
                expected_original_requires_grad=False,
                expected_original_is_contiguous=True,
                label=f"common step0 target prediction row {row_index}",
            )
            != row.get("hash_projection")
            for row_index, row in enumerate(prediction_rows)
        )
    ):
        fail("common preflight step0 target prediction closure differs")

    schedule = common.get("common_target_branch_schedule")
    if not isinstance(schedule, Mapping) or set(schedule) != {
        "training_exact10_common_target_branch",
        "evaluation_sigma1_by_row",
        "rng",
        "schedule_digest",
    }:
        fail("common target branch schedule envelope differs")
    unsigned_schedule = dict(schedule)
    schedule_digest = unsigned_schedule.pop("schedule_digest", None)
    train_rows = schedule.get("training_exact10_common_target_branch")
    eval_rows = schedule.get("evaluation_sigma1_by_row")
    if (
        schedule.get("rng") != "cpu_float32_torch_standard_normal"
        or schedule_digest != object_sha256(unsigned_schedule)
        or schedule_digest != common.get("row_input_noise_schedule_digest")
        or not isinstance(train_rows, list)
        or len(train_rows) != MAX_STEPS * len(ROW_IDS)
        or not isinstance(eval_rows, list)
        or len(eval_rows) != len(ROW_IDS)
    ):
        fail("common target branch schedule digest/count differs")
    for linear, row in enumerate(train_rows):
        step_zero, row_index = divmod(linear, len(ROW_IDS))
        expected_fields = {
            "step_zero",
            "row_index",
            "row_id",
            "sigma_coordinate",
            "epsilon_seed",
            "epsilon_sha256",
            "epsilon_shape",
            "epsilon_dtype",
            "target_sha256",
            "noisy_target_sha256",
            "target_velocity_sha256",
            "common_target_input_digest",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_fields
            or row.get("step_zero") != step_zero
            or row.get("row_index") != row_index
            or row.get("row_id") != ROW_IDS[row_index]
            or row.get("epsilon_shape") != list(LATENT_SHAPE)
            or row.get("epsilon_dtype") != "torch.float32"
            or row.get("epsilon_seed")
            != training_noise_seed_v1(expected_seed, step_zero, row_index)
            or row.get("sigma_coordinate") != TRAINING_SIGMA_EXACT10[step_zero]
            or row.get("target_sha256")
            != row_inputs[row_index].get("target_tensor_sha256")
        ):
            fail("common target training schedule row differs")
        _require_sha256_v1(
            row.get("epsilon_sha256"), label="common training epsilon SHA"
        )
        if row.get("epsilon_sha256") != c1.tensor_sha256_v1(
            cpu_epsilon_v1(row["epsilon_seed"])
        ):
            fail("common training epsilon bytes differ from registered CPU RNG")
        for field in (
            "target_sha256",
            "noisy_target_sha256",
            "target_velocity_sha256",
            "common_target_input_digest",
        ):
            _require_sha256_v1(
                row.get(field), label=f"common training {field}"
            )
        input_row = {
            "row_index": row_index,
            "row_id": ROW_IDS[row_index],
            "sigma_coordinate": row["sigma_coordinate"],
            "epsilon_seed": row["epsilon_seed"],
            "epsilon_sha256": row["epsilon_sha256"],
            "target_sha256": row["target_sha256"],
            "noisy_target_sha256": row["noisy_target_sha256"],
            "target_velocity_sha256": row["target_velocity_sha256"],
        }
        if row.get("common_target_input_digest") != object_sha256(input_row):
            fail("common training target input digest differs")
    for row_index, row in enumerate(eval_rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "row_index",
                "row_id",
                "sigma_float32_be_hex",
                "x_sigma_equals_epsilon",
                "epsilon_seed",
                "epsilon_sha256",
                "target_sha256",
                "noisy_target_sha256",
                "target_velocity_sha256",
                "common_target_input_digest",
            }
            or row.get("row_index") != row_index
            or row.get("row_id") != ROW_IDS[row_index]
            or row.get("sigma_float32_be_hex") != "3f800000"
            or row.get("x_sigma_equals_epsilon") is not True
            or row.get("epsilon_seed")
            != evaluation_noise_seed_v1(expected_seed, row_index)
            or row.get("target_sha256")
            != row_inputs[row_index].get("target_tensor_sha256")
            or row.get("noisy_target_sha256") != row.get("epsilon_sha256")
        ):
            fail("common target evaluation schedule row differs")
        _require_sha256_v1(
            row.get("epsilon_sha256"), label="common evaluation epsilon SHA"
        )
        if row.get("epsilon_sha256") != c1.tensor_sha256_v1(
            cpu_epsilon_v1(row["epsilon_seed"])
        ):
            fail("common evaluation epsilon bytes differ from registered CPU RNG")
        for field in (
            "target_sha256",
            "noisy_target_sha256",
            "target_velocity_sha256",
            "common_target_input_digest",
        ):
            _require_sha256_v1(
                row.get(field), label=f"common evaluation {field}"
            )
        evaluation_input = {
            key: row[key]
            for key in (
                "row_index",
                "row_id",
                "sigma_float32_be_hex",
                "x_sigma_equals_epsilon",
                "epsilon_seed",
                "epsilon_sha256",
                "target_sha256",
                "noisy_target_sha256",
                "target_velocity_sha256",
            )
        }
        if row.get("common_target_input_digest") != object_sha256(evaluation_input):
            fail("common evaluation target input digest differs")

    if common.get("optimizer_recipe") != {
        "class": "torch.optim.AdamW",
        "learning_rate": DEFAULT_LR,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
        "allowed_completed_steps": [1, MAX_STEPS],
        "optimizer_state_before_first_update": "empty",
        "resume": False,
    }:
        fail("common preflight optimizer recipe differs")
    if common.get("rng_recipe") != {
        "model_initialization_seed": expected_seed,
        "training_epsilon": "training_noise_seed_v1(seed,step_zero,row_index)",
        "evaluation_epsilon": "100*arm_seed+row_index",
        "epsilon_device": "cpu",
        "epsilon_dtype": "torch.float32",
    }:
        fail("common preflight RNG recipe differs")
    if common.get("execution_memory_contract") != execution_memory_contract_v1():
        fail("common preflight sequential/checkpoint memory contract differs")
    return common


def _validate_preflight_receipt_value_v1(
    receipt: Mapping[str, Any],
    *,
    arm_id: str,
    holder_job_id: str,
    node: str,
    seed: int,
    expected_receipt_digest: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    common = receipt.get("common_comparison_payload")
    actual = receipt.get("actual_shape_preflight")
    gain = receipt.get("step0_gain_safety")
    expected_recipe = (
        "target_duplicate_exact2"
        if arm_id == ARM_DUPLICATE
        else "target_and_role_swap_exact2"
    )
    second = receipt.get("second_branch_descriptor")
    if not isinstance(common, Mapping):
        fail("preflight common comparison payload is missing")
    _validate_common_preflight_payload_v1(
        common,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        expected_seed=seed,
    )
    step0_full = receipt.get("step0_full_q_route")
    step0_cells = receipt.get("step0_role_only_cells")
    step0_invariants = receipt.get("step0_role_only_input_invariants")
    step0_invariants_validation = validate_role_only_invariant_receipts_v1(
        step0_invariants, stage="step0"
    )
    step0_forward = receipt.get("step0_evaluation_forward_evidence")
    step0_forward_validation = []
    if not isinstance(step0_forward, list) or len(step0_forward) != 2:
        fail("preflight step0 evaluation forward exact2 differs")
    for row_index, row in enumerate(step0_forward):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "row_id",
                "input_payload",
                "actual_forward_evidence",
                "validation",
                "observation_validation",
            }
            or row.get("row_id") != ROW_IDS[row_index]
        ):
            fail("preflight step0 evaluation forward row differs")
        replay = validate_evaluation_forward_evidence_v1(
            row["actual_forward_evidence"],
            row_id=ROW_IDS[row_index],
            sp_rank=0,
            input_payload=row["input_payload"],
        )
        observation_replay = validate_evaluation_observation_binding_v1(
            full_q_route=step0_full[ROW_IDS[row_index]],
            role_only_cells=step0_cells[row_index * 2 : row_index * 2 + 2],
            actual_forward_evidence=row["actual_forward_evidence"],
            row_id=ROW_IDS[row_index],
            stage="step0",
        )
        if (
            row.get("validation") != replay
            or row.get("observation_validation") != observation_replay
            or common["step0_common_target_prediction_sha256_by_row"][row_index][
                "prediction_sha256"
            ]
            != row["actual_forward_evidence"]["full_target"][
                "actual_input_receipt"
            ]["prediction_sha256"]
        ):
            fail("preflight step0 evaluation forward validation differs")
        step0_forward_validation.append(replay)
    closure = receipt.get("pre_publish_closure_replays")
    if isinstance(actual, Mapping):
        _validate_all8_graph_rows_closed_v1(
            actual.get("runtime_telemetry"),
            arm_id=arm_id,
            completed_step=None,
            label=f"{arm_id} preflight",
            expected_common_payload=common,
        )
    gain_runtime = gain.get("runtime_telemetry") if isinstance(gain, Mapping) else None
    if (
        not isinstance(gain_runtime, list)
        or len(gain_runtime) != WORLD_SIZE
        or [row.get("world_rank") for row in gain_runtime if isinstance(row, Mapping)]
        != list(range(WORLD_SIZE))
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("gain_probe"), Mapping)
            or row["gain_probe"].get("finite_nonzero_bounded") is not True
            or row["gain_probe"].get("parameter_bytes_restored_after_probe")
            is not True
            for row in gain_runtime
        )
    ):
        fail(f"{arm_id} preflight gain runtime exact8 closure differs")
    step0_cells_order = (
        tuple(
            (row.get("row_id"), row.get("clean_variant"))
            for row in step0_cells
        )
        if isinstance(step0_cells, list)
        and all(isinstance(row, Mapping) for row in step0_cells)
        else ()
    )
    if (
        set(receipt) != PREFLIGHT_RECEIPT_FIELDS
        or receipt.get("schema_version") != PREFLIGHT_SCHEMA
        or receipt.get("status") != "PRECHECK_COMPLETE_NO_OPTIMIZER_NO_UPDATE"
        or receipt.get("method") != METHOD
        or receipt.get("arm_id") != arm_id
        or receipt.get("branch_recipe") != expected_recipe
        or receipt.get("holder_job_id") != holder_job_id
        or receipt.get("node") != node
        or receipt.get("seed") != seed
        or receipt.get("preflight_only") is not True
        or receipt.get("completed_optimizer_steps") != 0
        or receipt.get("optimizer_constructed") is not False
        or receipt.get("resume_consumed") is not False
        or receipt.get("all_preflight_hard_gates_pass") is not True
        or receipt.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or receipt.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or receipt.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or receipt.get("latent_bundle_sha256") != expected_bundle_sha256
        or receipt.get("runner_source_sha256") != expected_runner_sha256
        or receipt.get("source_pins") != dict(expected_source_pins)
        or receipt.get("claim_boundaries") != CLAIM_BOUNDARIES
        or not isinstance(closure, Mapping)
        or closure.get("runtime_sources_pre_final_bit_exact") is not True
        or closure.get("model_exact9_pre_post_final_stable") is not True
        or closure.get("bundle_exact16_pre_final_stable") is not True
        or closure.get("materializer_run_complete_pre_final_stable") is not True
        or closure.get("materializer_run_complete_sha256")
        != MATERIALIZER_RUN_COMPLETE_SHA256
        or closure.get("materializer_run_complete_digest")
        != MATERIALIZER_RUN_COMPLETE_DIGEST
        or closure.get("checkpoint_exact23_pre_post_final_stable") is not True
        or closure.get("checkpoint_exact23_manifest_sha256")
        != CHECKPOINT_EXACT23_MANIFEST_SHA256
        or closure.get("checkpoint_exact23_binding_digest")
        != common.get("checkpoint_exact23_binding_digest")
        or closure.get("bernini_execution_sources_pre_post_final_stable")
        is not True
        or closure.get("bernini_execution_source_binding_digest")
        != common.get("bernini_execution_source_binding_digest")
        or closure.get("oracle_labels_pre_final_stable") is not True
        or digest != expected_receipt_digest
        or digest != object_sha256(unsigned)
        or not isinstance(common, Mapping)
        or set(common) != COMMON_PREFLIGHT_PAYLOAD_FIELDS
        or receipt.get("common_comparison_payload_digest")
        != object_sha256(common)
        or common.get("step0_full_q_route_digest") != object_sha256(step0_full)
        or common.get("step0_role_only_cells_digest") != object_sha256(step0_cells)
        or common.get("step0_role_only_input_invariants_digest")
        != object_sha256(step0_invariants)
        or receipt.get("step0_role_only_input_invariants_validation")
        != step0_invariants_validation
        or common.get("step0_evaluation_forward_evidence_digest")
        != object_sha256(step0_forward)
        or receipt.get("step0_evaluation_forward_evidence_validation")
        != step0_forward_validation
        or not isinstance(step0_full, Mapping)
        or set(step0_full) != set(ROW_IDS)
        or any(
            not isinstance(step0_full[row_id], Mapping)
            or step0_full[row_id].get("claim_name")
            != "oracle_route_controllability_only"
            or step0_full[row_id].get(
                "participant_role_binding_claim_forbidden"
            )
            is not True
            for row_id in ROW_IDS
        )
        or step0_cells_order != ROLE_ONLY_CELL_ORDER
        or common.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or common.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or common.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or common.get("latent_bundle_sha256") != expected_bundle_sha256
        or common.get("source_pins") != dict(expected_source_pins)
        or common.get("initial_trainable_sha256")
        != receipt.get("initial_trainable_sha256")
        or common.get("row_input_noise_schedule_digest")
        != receipt.get("row_input_noise_schedule_digest")
        or common.get("fresh_official_base") is not True
        or common.get("resume_consumed") is not False
        or not isinstance(actual, Mapping)
        or actual.get("actual_shape_two_branch_forward_pass") is not True
        or actual.get("all30_each_branch_used") is not True
        or actual.get("sp4_partition_all8_pass") is not True
        or actual.get("memory_all8_strictly_gt_half") is not True
        or actual.get("cross_arm_collective_used") is not False
        or actual.get("strict_sequential_branch_graphs") is not True
        or actual.get("preflight_grad_enabled_training_graph") is not True
        or actual.get("preflight_backward_executed") is not False
        or actual.get("simultaneous_live_autograd_branch_graphs_maximum") != 1
        or actual.get("activation_checkpoint_profile")
        != ACTIVATION_CHECKPOINT_PROFILE
        or actual.get("activation_checkpointed_blocks")
        != list(ACTIVATION_CHECKPOINT_BLOCKS)
        or actual.get("activation_checkpoint_nonreentrant") is not True
        or actual.get("activation_checkpoint_elal_route_context_replay") is not True
        or actual.get("memory_peak_true_tensors_only") is not True
        or actual.get("dummy_or_padding_allocations") is not False
        or actual.get("common_target_prediction_sha256_by_row")
        != common.get("step0_common_target_prediction_sha256_by_row")
        or not isinstance(gain, Mapping)
        or gain.get("all_rows_finite_nonzero_bounded_and_restored") is not True
        or not isinstance(second, Mapping)
        or second.get("recipe") != expected_recipe
        or second.get("exactly_one_whitelisted_difference_from_common_target")
        is not True
    ):
        fail(f"{arm_id} preflight closed receipt differs")
    return receipt


def validate_own_preflight_receipt_v1(
    path: Path,
    *,
    expected_sha256: str,
    arm_id: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
) -> Mapping[str, Any]:
    if arm_id not in ARM_PLACEMENT:
        fail("own preflight arm differs")
    receipt = _read_sealed_json_held_fd_v1(
        path,
        expected_sha256=expected_sha256,
        label=f"{arm_id} own preflight receipt",
    )
    job_id, node, seed = ARM_PLACEMENT[arm_id]
    return _validate_preflight_receipt_value_v1(
        receipt,
        arm_id=arm_id,
        holder_job_id=job_id,
        node=node,
        seed=seed,
        expected_receipt_digest=str(receipt.get("receipt_digest")),
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
    )


def validate_cross_arm_preflight_gate_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replay both sealed A/B preflights before any optimizer construction."""

    _require_sha256_v1(expected_sha256, label="cross-arm gate expected SHA")
    gate = _read_sealed_json_held_fd_v1(
        path,
        expected_sha256=expected_sha256,
        label="C2 cross-arm preflight gate",
    )
    unsigned_gate = dict(gate)
    gate_digest = unsigned_gate.pop("gate_digest", None)
    expected_gate_fields = {
        "schema_version",
        "status",
        "experiment_contract_sha256",
        "external_authority_sha256",
        "model_authority_sha256",
        "latent_bundle_sha256",
        "runner_source_sha256",
        "source_pins",
        "recipe_version_digest",
        "allowed_preflight_receipt_differences",
        "preflight_receipts",
        "common_initial_trainable_sha256",
        "common_row_input_noise_schedule_digest",
        "common_comparison_payload_digest",
        "updates_executed_before_gate",
        "gate_digest",
    }
    if (
        set(gate) != expected_gate_fields
        or gate.get("schema_version") != CROSS_ARM_GATE_SCHEMA
        or gate.get("status") != "CROSS_ARM_PREFLIGHT_GATE_PASS"
        or gate.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or gate.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or gate.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or gate.get("latent_bundle_sha256") != expected_bundle_sha256
        or gate.get("runner_source_sha256") != expected_runner_sha256
        or gate.get("source_pins") != dict(expected_source_pins)
        or gate.get("allowed_preflight_receipt_differences")
        != list(CROSS_ARM_ALLOWED_DIFFERENCES)
        or gate.get("updates_executed_before_gate") != 0
        or gate_digest != object_sha256(unsigned_gate)
    ):
        fail("cross-arm preflight gate envelope/digest differs")
    rows = gate.get("preflight_receipts")
    if not isinstance(rows, list) or len(rows) != 2:
        fail("cross-arm gate requires exact A/B preflight receipts")
    expected_arms = (
        (ARM_DUPLICATE, "141620", "auh7-1b-gpu-226", 20260821),
        (ARM_ROLE_PAIR, "141618", "auh7-1b-gpu-249", 20260821),
    )
    replayed: list[Mapping[str, Any]] = []
    for row, (arm_id, job_id, node, seed) in zip(rows, expected_arms):
        if not isinstance(row, Mapping) or set(row) != {
            "arm_id",
            "holder_job_id",
            "node",
            "seed",
            "path",
            "sha256",
            "receipt_digest",
        }:
            fail("cross-arm gate preflight row schema differs")
        if (
            (row.get("arm_id"), row.get("holder_job_id"), row.get("node"), row.get("seed"))
            != (arm_id, job_id, node, seed)
            or type(row.get("path")) is not str
        ):
            fail("cross-arm gate arm/job/node/seed differs")
        receipt_path = Path(str(row["path"]))
        receipt_sha = _require_sha256_v1(
            row.get("sha256"), label=f"{arm_id} preflight receipt SHA"
        )
        receipt_digest_expected = _require_sha256_v1(
            row.get("receipt_digest"),
            label=f"{arm_id} preflight receipt digest",
        )
        receipt = _read_sealed_json_held_fd_v1(
            receipt_path,
            expected_sha256=receipt_sha,
            label=f"{arm_id} sealed preflight receipt",
        )
        replayed.append(
            _validate_preflight_receipt_value_v1(
                receipt,
                arm_id=arm_id,
                holder_job_id=job_id,
                node=node,
                seed=seed,
                expected_receipt_digest=receipt_digest_expected,
                expected_runner_sha256=expected_runner_sha256,
                expected_bundle_sha256=expected_bundle_sha256,
                expected_source_pins=expected_source_pins,
            )
        )
    initial = [row.get("initial_trainable_sha256") for row in replayed]
    schedules = [row.get("row_input_noise_schedule_digest") for row in replayed]
    comparable: list[dict[str, Any]] = []
    for receipt in replayed:
        value = json.loads(canonical_json_bytes(receipt).decode("ascii"))
        for key in (
            "arm_id",
            "branch_recipe",
            "holder_job_id",
            "node",
            "second_branch_descriptor",
            "receipt_digest",
        ):
            value.pop(key, None)
        for parent in ("actual_shape_preflight", "step0_gain_safety"):
            nested = value.get(parent)
            if isinstance(nested, dict):
                nested.pop("runtime_telemetry", None)
        closure = value.get("pre_publish_closure_replays")
        if isinstance(closure, dict):
            closure.pop("runtime_telemetry", None)
        comparable.append(value)
    if (
        len(set(initial)) != 1
        or len(set(schedules)) != 1
        or initial[0] != gate.get("common_initial_trainable_sha256")
        or schedules[0] != gate.get("common_row_input_noise_schedule_digest")
        or replayed[0].get("common_comparison_payload_digest")
        != replayed[1].get("common_comparison_payload_digest")
        or replayed[0].get("common_comparison_payload_digest")
        != gate.get("common_comparison_payload_digest")
        or replayed[0].get("recipe_version_digest")
        != replayed[1].get("recipe_version_digest")
        or replayed[0].get("recipe_version_digest")
        != gate.get("recipe_version_digest")
        or canonical_json_bytes(comparable[0]) != canonical_json_bytes(comparable[1])
    ):
        fail("A/B preflight has a non-whitelisted recipe/input/step0 difference")
    return {
        "gate_sha256": expected_sha256,
        "gate_digest": gate_digest,
        "a_preflight_sha256": rows[0]["sha256"],
        "b_preflight_sha256": rows[1]["sha256"],
        "common_initial_trainable_sha256": initial[0],
        "common_row_input_noise_schedule_digest": schedules[0],
        "common_comparison_payload_digest": gate[
            "common_comparison_payload_digest"
        ],
        "recipe_version_digest": gate["recipe_version_digest"],
        "updates_executed_before_gate": 0,
    }


def _checkpoint_common_from_receipt_v1(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "method": METHOD,
        "arm_id": receipt.get("arm_id"),
        "seed": receipt.get("seed"),
        "fresh_official_base": True,
        "resume_consumed": False,
        "initial_trainable_sha256": receipt.get("initial_trainable_sha256"),
        "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
        "latent_bundle_sha256": receipt.get("latent_bundle_sha256"),
        "formal_c2_authorized": False,
        "exact160_authorized": False,
        "source_instruction_inference": False,
    }


def _validate_fresh1_receipt_value_v1(
    receipt: Mapping[str, Any],
    *,
    arm_id: str,
    expected_receipt_digest: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    cross_gate_sha256: str,
    cross_gate_digest: str,
    cross_recipe_version_digest: str,
    cross_common_initial_trainable_sha256: str,
    cross_common_row_input_noise_schedule_digest: str,
    cross_common_comparison_payload_digest: str,
) -> Mapping[str, Any]:
    """Replay one exact-one-update engineering receipt and its own preflight."""

    if arm_id not in ARM_PLACEMENT:
        fail("fresh1 receipt arm differs")
    job_id, node, seed = ARM_PLACEMENT[arm_id]
    recipe = (
        "target_duplicate_exact2"
        if arm_id == ARM_DUPLICATE
        else "target_and_role_swap_exact2"
    )
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    own = receipt.get("own_preflight_binding")
    cross = receipt.get("cross_arm_gate_binding")
    actual = receipt.get("actual_shape_training")
    gradient = receipt.get("gradient_gate")
    memory = receipt.get("memory_gate")
    checkpoint = receipt.get("checkpoint_gate")
    closure = receipt.get("pre_publish_closure_replays")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "step0_create_only_reload_pass",
        "step1_create_only_reload_pass",
        "step1_parameter_sha256",
        "step0_checkpoint_record",
        "step1_checkpoint_record",
        "checkpoint_tree_closure",
    }:
        fail(f"{arm_id} fresh1 checkpoint gate closure differs")
    checkpoint_common = _checkpoint_common_from_receipt_v1(receipt)
    validate_checkpoint_record_v1(
        checkpoint["step0_checkpoint_record"],
        expected_step=0,
        expected_parameter_sha256=str(receipt.get("initial_trainable_sha256")),
        optimizer_required=False,
        expected_common=checkpoint_common,
    )
    checkpoint_tree_replay = seal_and_validate_checkpoint_tree_v1(
        Path(checkpoint["step0_checkpoint_record"]["path"]).parent,
        records=[
            checkpoint["step0_checkpoint_record"],
            checkpoint["step1_checkpoint_record"],
        ],
        expected_steps=(0, 1),
        expected_parameter_sha256_by_step={
            0: str(receipt.get("initial_trainable_sha256")),
            1: str(receipt.get("final_trainable_sha256")),
        },
        expected_common=checkpoint_common,
    )
    if checkpoint.get("checkpoint_tree_closure") != checkpoint_tree_replay:
        fail(f"{arm_id} fresh1 checkpoint tree closure differs")
    validate_checkpoint_record_v1(
        checkpoint["step1_checkpoint_record"],
        expected_step=1,
        expected_parameter_sha256=str(receipt.get("final_trainable_sha256")),
        optimizer_required=True,
        expected_common=checkpoint_common,
    )
    history_validation = validate_training_history_v1(
        receipt.get("history"),
        arm_id=arm_id,
        seed=seed,
        expected_steps=1,
        initial_parameter_sha256=str(receipt.get("initial_trainable_sha256")),
        final_parameter_sha256=str(receipt.get("final_trainable_sha256")),
    )
    if (
        set(receipt) != FRESH1_RECEIPT_FIELDS
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status")
        != "FRESH1_ENGINEERING_ACCEPTANCE_COMPLETE"
        or receipt.get("method") != METHOD
        or receipt.get("arm_id") != arm_id
        or receipt.get("branch_recipe") != recipe
        or receipt.get("holder_job_id") != job_id
        or receipt.get("node") != node
        or receipt.get("seed") != seed
        or receipt.get("preflight_only") is not False
        or receipt.get("requested_optimizer_steps") != 1
        or receipt.get("completed_optimizer_steps") != 1
        or receipt.get("optimizer_constructed") is not True
        or receipt.get("optimizer_state_empty_before_first_update") is not True
        or receipt.get("resume_consumed") is not False
        or receipt.get("fresh_official_base") is not True
        or receipt.get("parameters_changed") is not True
        or receipt.get("initial_trainable_sha256")
        == receipt.get("final_trainable_sha256")
        or receipt.get("all_fresh1_acceptance_gates_pass") is not True
        or receipt.get("history_validation") != history_validation
        or receipt.get("experiment_contract_sha256")
        != EXPERIMENT_CONTRACT_SHA256
        or receipt.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or receipt.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or receipt.get("latent_bundle_sha256") != expected_bundle_sha256
        or receipt.get("runner_source_sha256") != expected_runner_sha256
        or receipt.get("source_pins") != dict(expected_source_pins)
        or receipt.get("claim_boundaries") != CLAIM_BOUNDARIES
        or not isinstance(closure, Mapping)
        or closure.get("runtime_sources_pre_final_bit_exact") is not True
        or closure.get("model_exact9_pre_post_final_stable") is not True
        or closure.get("bundle_exact16_pre_final_stable") is not True
        or closure.get("materializer_run_complete_pre_final_stable") is not True
        or closure.get("materializer_run_complete_sha256")
        != MATERIALIZER_RUN_COMPLETE_SHA256
        or closure.get("materializer_run_complete_digest")
        != MATERIALIZER_RUN_COMPLETE_DIGEST
        or closure.get("checkpoint_exact23_pre_post_final_stable") is not True
        or closure.get("checkpoint_exact23_manifest_sha256")
        != CHECKPOINT_EXACT23_MANIFEST_SHA256
        or type(closure.get("checkpoint_exact23_binding_digest")) is not str
        or len(closure.get("checkpoint_exact23_binding_digest")) != 64
        or closure.get("bernini_execution_sources_pre_post_final_stable")
        is not True
        or type(closure.get("bernini_execution_source_binding_digest")) is not str
        or len(closure.get("bernini_execution_source_binding_digest")) != 64
        or closure.get("oracle_labels_pre_final_stable") is not True
        or closure.get("checkpoint_tree_pre_publish_stable") is not True
        or closure.get("checkpoint_tree_binding_digest")
        != checkpoint_tree_replay.get("tree_binding_digest")
        or closure.get("checkpoint_portable_tree_digest")
        != checkpoint_tree_replay.get("portable_checkpoint_tree_digest")
        or digest != expected_receipt_digest
        or digest != object_sha256(unsigned)
        or not isinstance(own, Mapping)
        or set(own) != {"path", "sha256", "receipt_digest"}
        or not isinstance(cross, Mapping)
        or set(cross)
        != {
            "path",
            "sha256",
            "gate_digest",
            "recipe_version_digest",
        }
        or cross.get("sha256") != cross_gate_sha256
        or cross.get("gate_digest") != cross_gate_digest
        or cross.get("recipe_version_digest") != cross_recipe_version_digest
        or not isinstance(actual, Mapping)
        or actual.get("actual_shape_two_branch_forward_pass") is not True
        or actual.get("all30_each_branch_used") is not True
        or actual.get("sp4_partition_all8_pass") is not True
        or actual.get("cross_arm_collective_used") is not False
        or actual.get("strict_sequential_branch_forward_backward") is not True
        or actual.get("fixed_branch_coefficients") != [0.5, 0.5]
        or actual.get("first_graph_released_before_second_forward") is not True
        or actual.get("simultaneous_live_autograd_branch_graphs_maximum") != 1
        or actual.get("reduce_clip_optimizer_after_both_branches") is not True
        or actual.get("activation_checkpoint_profile")
        != ACTIVATION_CHECKPOINT_PROFILE
        or actual.get("activation_checkpointed_blocks")
        != list(ACTIVATION_CHECKPOINT_BLOCKS)
        or actual.get("activation_checkpoint_nonreentrant") is not True
        or actual.get("activation_checkpoint_elal_route_context_replay") is not True
        or not isinstance(gradient, Mapping)
        or gradient.get("all_trainable_parameters_have_finite_gradients")
        is not True
        or gradient.get("all30_elal_nonzero_after_manual_sp4_dp2_reduction")
        is not True
        or gradient.get("finite_nonzero_synchronized_gradient_norm") is not True
        or not isinstance(memory, Mapping)
        or memory.get("all8_peak_allocated_strictly_gt_half") is not True
        or memory.get("dummy_or_padding_allocations") is not False
        or checkpoint.get("step0_create_only_reload_pass") is not True
        or checkpoint.get("step1_create_only_reload_pass") is not True
        or checkpoint.get("step1_parameter_sha256")
        != receipt.get("final_trainable_sha256")
    ):
        fail(f"{arm_id} fresh1 closed receipt differs")
    for field in (
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
    ):
        _require_sha256_v1(receipt.get(field), label=f"fresh1 {field}")
    own_sha = _require_sha256_v1(
        own.get("sha256"), label=f"{arm_id} own preflight SHA"
    )
    own_digest = _require_sha256_v1(
        own.get("receipt_digest"), label=f"{arm_id} own preflight digest"
    )
    if type(own.get("path")) is not str or type(cross.get("path")) is not str:
        fail("fresh1 predecessor path ABI differs")
    own_receipt = _read_sealed_json_held_fd_v1(
        Path(own["path"]),
        expected_sha256=own_sha,
        label=f"{arm_id} fresh1 own preflight replay",
    )
    own_receipt = _validate_preflight_receipt_value_v1(
        own_receipt,
        arm_id=arm_id,
        holder_job_id=job_id,
        node=node,
        seed=seed,
        expected_receipt_digest=own_digest,
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
    )
    history_validation_with_schedule = validate_training_history_v1(
        receipt.get("history"),
        arm_id=arm_id,
        seed=seed,
        expected_steps=1,
        initial_parameter_sha256=str(receipt.get("initial_trainable_sha256")),
        final_parameter_sha256=str(receipt.get("final_trainable_sha256")),
        expected_common_payload=own_receipt["common_comparison_payload"],
    )
    if (
        receipt.get("initial_trainable_sha256")
        != own_receipt.get("initial_trainable_sha256")
        or receipt.get("common_comparison_payload_digest")
        != own_receipt.get("common_comparison_payload_digest")
        or receipt.get("row_input_noise_schedule_digest")
        != own_receipt.get("row_input_noise_schedule_digest")
        or receipt.get("history_validation") != history_validation_with_schedule
        or receipt["pre_publish_closure_replays"].get(
            "checkpoint_exact23_binding_digest"
        )
        != own_receipt["common_comparison_payload"].get(
            "checkpoint_exact23_binding_digest"
        )
        or receipt["pre_publish_closure_replays"].get(
            "bernini_execution_source_binding_digest"
        )
        != own_receipt["common_comparison_payload"].get(
            "bernini_execution_source_binding_digest"
        )
    ):
        fail("fresh1 fresh-init/input/RNG replay differs from own preflight")
    if arm_id in (ARM_DUPLICATE, ARM_ROLE_PAIR) and (
        receipt.get("initial_trainable_sha256")
        != cross_common_initial_trainable_sha256
        or receipt.get("row_input_noise_schedule_digest")
        != cross_common_row_input_noise_schedule_digest
        or receipt.get("common_comparison_payload_digest")
        != cross_common_comparison_payload_digest
    ):
        fail("A/B fresh1 receipt differs from sealed cross-arm common payload")
    return receipt


def _canonical_json_file_sha256_v1(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


def _portable_tool_binding_v1(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    row = _sealed_file_row_v1(path, label=label)
    if row["sha256"] != _require_sha256_v1(expected_sha256, label=f"{label} SHA"):
        fail(f"{label} release SHA differs")
    return {
        "name": row["name"],
        "sha256": row["sha256"],
        "size": row["size"],
        "mode": row["mode"],
        "nlink": row["nlink"],
    }


def _validate_portable_checkpoint_tree_v1(
    value: Any, *, expected_steps: Sequence[int], expected_parameters: Sequence[str]
) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "expected_steps",
        "directory_entries",
        "directory_mode",
        "portable_checkpoint_records",
        "portable_checkpoint_tree_digest",
        "physical_origin_replay_passed",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version") != "bernini-elal3-c2-sealed-checkpoint-tree-v1"
        or value.get("expected_steps") != list(expected_steps)
        or value.get("directory_entries")
        != [f"checkpoint-{step:08d}" for step in expected_steps]
        or value.get("directory_mode") != 0o500
        or value.get("physical_origin_replay_passed") is not True
    ):
        fail("portable checkpoint tree envelope differs")
    rows = value.get("portable_checkpoint_records")
    if not isinstance(rows, list) or len(rows) != len(expected_steps):
        fail("portable checkpoint record count differs")
    for row, step, parameter_sha in zip(rows, expected_steps, expected_parameters):
        unsigned = dict(row) if isinstance(row, Mapping) else {}
        digest = unsigned.pop("portable_record_digest", None)
        expected_row_fields = {
            "schema_version",
            "step",
            "file_order",
            "directory_entries",
            "directory_mode",
            "files",
            "adapter_payload_tree_digest",
            "parameter_order",
            "parameter_inventory",
            "optimizer_payload_tree_digest",
            "optimizer_state_inventory",
            "checkpoint_receipt_digest",
            "trainable_parameter_sha256",
            "strict_reload_pass",
            "portable_record_digest",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_fields
            or digest != object_sha256(unsigned)
            or row.get("schema_version") != CHECKPOINT_SCHEMA
            or row.get("step") != step
            or row.get("directory_mode") != 0o500
            or row.get("trainable_parameter_sha256") != parameter_sha
            or row.get("strict_reload_pass") is not True
            or not isinstance(row.get("parameter_order"), list)
            or len(row["parameter_order"]) != 668
            or len(set(row["parameter_order"])) != 668
            or not isinstance(row.get("parameter_inventory"), list)
            or len(row["parameter_inventory"]) != 668
        ):
            fail("portable checkpoint record closure differs")
        inventory = row["parameter_inventory"]
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "shape", "dtype", "numel"}
            or item.get("name") != row["parameter_order"][index]
            or type(item.get("shape")) is not list
            or not item["shape"]
            or any(type(dimension) is not int or dimension <= 0 for dimension in item["shape"])
            or math.prod(item["shape"]) != item.get("numel")
            or item.get("dtype") not in {"torch.float32", "torch.bfloat16"}
            for index, item in enumerate(inventory)
        ) or (
            sum(".lora_" in name for name in row["parameter_order"]) != 480
            or sum(".elal3_c0_v1." in name for name in row["parameter_order"]) != 188
        ):
            fail("portable checkpoint parameter inventory/name partition differs")
        expected_files = ["adapter-and-elal3.pt"]
        if step != 0:
            expected_files.append("optimizer.pt")
        expected_files.append("CHECKPOINT_RECEIPT.json")
        files = row.get("files")
        if (
            row.get("file_order") != expected_files
            or row.get("directory_entries") != expected_files
            or not isinstance(files, list)
            or [item.get("name") for item in files] != expected_files
            or any(
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "name",
                    "sha256",
                    "size",
                    "mode",
                    "nlink",
                    "held_fd_double_hash_verified",
                    "named_identity_replayed",
                }
                or item.get("mode") != 0o444
                or item.get("nlink") != 1
                or item.get("held_fd_double_hash_verified") is not True
                or item.get("named_identity_replayed") is not True
                or type(item.get("size")) is not int
                or item.get("size") <= 0
                for item in files
            )
        ):
            fail("portable checkpoint exact file rows differ")
        for field in (
            "adapter_payload_tree_digest",
            "checkpoint_receipt_digest",
            "trainable_parameter_sha256",
            "portable_record_digest",
        ):
            _require_sha256_v1(row.get(field), label=f"portable checkpoint {field}")
        optimizer_digest = row.get("optimizer_payload_tree_digest")
        optimizer_inventory = row.get("optimizer_state_inventory")
        if step == 0:
            if optimizer_digest is not None or optimizer_inventory is not None:
                fail("portable step0 unexpectedly carries optimizer state")
        else:
            if (
                _require_sha256_v1(
                    optimizer_digest, label="portable optimizer payload"
                )
                != optimizer_digest
                or not isinstance(optimizer_inventory, Mapping)
                or set(optimizer_inventory)
                != {
                    "state_entry_count",
                    "param_group_count",
                    "parameter_count",
                    "parameter_inventory_digest",
                    "optimizer_step",
                    "exp_avg_nonzero_parameter_count",
                    "exp_avg_sq_nonzero_parameter_count",
                    "state_keys_by_parameter",
                    "tree_digest",
                }
                or optimizer_inventory.get("state_entry_count") != 668
                or optimizer_inventory.get("param_group_count") != 1
                or optimizer_inventory.get("parameter_count") != 668
                or optimizer_inventory.get("parameter_inventory_digest")
                != object_sha256(inventory)
                or optimizer_inventory.get("optimizer_step") != step
                or optimizer_inventory.get("tree_digest") != optimizer_digest
                or optimizer_inventory.get("exp_avg_nonzero_parameter_count")
                not in range(1, 669)
                or optimizer_inventory.get("exp_avg_sq_nonzero_parameter_count")
                not in range(1, 669)
                or optimizer_inventory.get("state_keys_by_parameter")
                != [
                    {
                        "parameter_id": index,
                        "state_keys": ["exp_avg", "exp_avg_sq", "step"],
                    }
                    for index in range(668)
                ]
            ):
                fail("portable optimizer exact668/step/moment closure differs")
    if value.get("portable_checkpoint_tree_digest") != object_sha256(rows):
        fail("portable checkpoint tree digest differs")
    return value


def _validate_fresh1_origin_attestation_value_v1(
    attestation: Any,
    *,
    arm_id: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    cross_gate: Mapping[str, Any],
    expected_origin_verifier_binding: Mapping[str, Any],
    expected_gate_controller_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "status",
        "stage",
        "arm_id",
        "holder_job_id",
        "node",
        "seed",
        "receipt_sha256",
        "receipt_size",
        "receipt_digest",
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
        "history_digest",
        "portable_checkpoint_tree",
        "portable_checkpoint_tree_digest",
        "cross_arm_gate_sha256",
        "cross_arm_gate_digest",
        "cross_arm_recipe_version_digest",
        "runner_source_sha256",
        "latent_bundle_sha256",
        "source_pins",
        "experiment_contract_sha256",
        "external_authority_sha256",
        "model_authority_sha256",
        "materializer_run_complete_sha256",
        "materializer_run_complete_digest",
        "checkpoint_exact23_binding_digest",
        "bernini_execution_source_binding_digest",
        "origin_verifier_binding",
        "gate_controller_binding",
        "physical_origin_replay_passed",
        "closed_validator_passed",
        "attestation_digest",
    }
    job_id, node, seed = ARM_PLACEMENT[arm_id]
    unsigned = dict(attestation) if isinstance(attestation, Mapping) else {}
    digest = unsigned.pop("attestation_digest", None)
    if (
        not isinstance(attestation, Mapping)
        or set(attestation) != fields
        or attestation.get("schema_version") != FRESH1_ORIGIN_ATTESTATION_SCHEMA
        or attestation.get("status") != "FRESH1_ORIGIN_PHYSICAL_REPLAY_PASS"
        or attestation.get("stage") != "fresh1"
        or (attestation.get("arm_id"), attestation.get("holder_job_id"), attestation.get("node"), attestation.get("seed"))
        != (arm_id, job_id, node, seed)
        or attestation.get("runner_source_sha256") != expected_runner_sha256
        or attestation.get("latent_bundle_sha256") != expected_bundle_sha256
        or attestation.get("source_pins") != dict(expected_source_pins)
        or attestation.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or attestation.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or attestation.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or attestation.get("materializer_run_complete_sha256") != MATERIALIZER_RUN_COMPLETE_SHA256
        or attestation.get("materializer_run_complete_digest") != MATERIALIZER_RUN_COMPLETE_DIGEST
        or attestation.get("cross_arm_gate_sha256") != cross_gate.get("gate_sha256")
        or attestation.get("cross_arm_gate_digest") != cross_gate.get("gate_digest")
        or attestation.get("cross_arm_recipe_version_digest") != cross_gate.get("recipe_version_digest")
        or attestation.get("physical_origin_replay_passed") is not True
        or attestation.get("closed_validator_passed") is not True
        or attestation.get("initial_trainable_sha256")
        == attestation.get("final_trainable_sha256")
        or (
            arm_id in (ARM_DUPLICATE, ARM_ROLE_PAIR)
            and (
                attestation.get("initial_trainable_sha256")
                != cross_gate.get("common_initial_trainable_sha256")
                or attestation.get("common_comparison_payload_digest")
                != cross_gate.get("common_comparison_payload_digest")
                or attestation.get("row_input_noise_schedule_digest")
                != cross_gate.get("common_row_input_noise_schedule_digest")
            )
        )
        or type(attestation.get("receipt_size")) is not int
        or attestation.get("receipt_size") <= 0
        or digest != object_sha256(unsigned)
    ):
        fail("fresh1 origin attestation envelope differs")
    for field in (
        "receipt_sha256",
        "receipt_digest",
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
        "history_digest",
        "portable_checkpoint_tree_digest",
        "checkpoint_exact23_binding_digest",
        "bernini_execution_source_binding_digest",
        "attestation_digest",
    ):
        _require_sha256_v1(attestation.get(field), label=f"fresh1 attestation {field}")
    for tool_name in ("origin_verifier_binding", "gate_controller_binding"):
        tool = attestation.get(tool_name)
        if (
            not isinstance(tool, Mapping)
            or set(tool) != {"name", "sha256", "size", "mode", "nlink"}
            or type(tool.get("name")) is not str
            or type(tool.get("size")) is not int
            or tool.get("size") <= 0
            or tool.get("mode") != 0o444
            or tool.get("nlink") != 1
        ):
            fail("fresh1 attestation verifier/controller binding differs")
        _require_sha256_v1(tool.get("sha256"), label=f"{tool_name} SHA")
    if (
        attestation.get("origin_verifier_binding")
        != dict(expected_origin_verifier_binding)
        or attestation.get("gate_controller_binding")
        != dict(expected_gate_controller_binding)
    ):
        fail("fresh1 attestation tool release pin differs")
    _validate_portable_checkpoint_tree_v1(
        attestation.get("portable_checkpoint_tree"),
        expected_steps=(0, 1),
        expected_parameters=(
            attestation["initial_trainable_sha256"],
            attestation["final_trainable_sha256"],
        ),
    )
    if attestation.get("portable_checkpoint_tree_digest") != attestation["portable_checkpoint_tree"].get("portable_checkpoint_tree_digest"):
        fail("fresh1 attestation portable checkpoint join differs")
    return attestation


def build_fresh1_origin_attestation_v1(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    arm_id: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    cross_gate: Mapping[str, Any],
    origin_verifier_path: Path,
    expected_origin_verifier_sha256: str,
    gate_controller_path: Path,
    expected_gate_controller_sha256: str,
) -> Mapping[str, Any]:
    """Origin-only physical replay; ordinary trainer main never calls this."""

    receipt = _read_sealed_json_held_fd_v1(
        receipt_path,
        expected_sha256=expected_receipt_sha256,
        label=f"{arm_id} origin fresh1 receipt",
    )
    validated = _validate_fresh1_receipt_value_v1(
        receipt,
        arm_id=arm_id,
        expected_receipt_digest=str(receipt.get("receipt_digest")),
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        cross_gate_sha256=str(cross_gate.get("gate_sha256")),
        cross_gate_digest=str(cross_gate.get("gate_digest")),
        cross_recipe_version_digest=str(cross_gate.get("recipe_version_digest")),
        cross_common_initial_trainable_sha256=str(cross_gate.get("common_initial_trainable_sha256")),
        cross_common_row_input_noise_schedule_digest=str(cross_gate.get("common_row_input_noise_schedule_digest")),
        cross_common_comparison_payload_digest=str(cross_gate.get("common_comparison_payload_digest")),
    )
    tree = validated["checkpoint_gate"]["checkpoint_tree_closure"]
    portable_tree = {
        key: tree[key]
        for key in (
            "schema_version",
            "expected_steps",
            "directory_entries",
            "directory_mode",
            "portable_checkpoint_records",
            "portable_checkpoint_tree_digest",
            "physical_origin_replay_passed",
        )
    }
    unsigned = {
        "schema_version": FRESH1_ORIGIN_ATTESTATION_SCHEMA,
        "status": "FRESH1_ORIGIN_PHYSICAL_REPLAY_PASS",
        "stage": "fresh1",
        "arm_id": arm_id,
        "holder_job_id": ARM_PLACEMENT[arm_id][0],
        "node": ARM_PLACEMENT[arm_id][1],
        "seed": ARM_PLACEMENT[arm_id][2],
        "receipt_sha256": expected_receipt_sha256,
        "receipt_size": len(canonical_json_bytes(receipt)) + 1,
        "receipt_digest": validated["receipt_digest"],
        "initial_trainable_sha256": validated["initial_trainable_sha256"],
        "final_trainable_sha256": validated["final_trainable_sha256"],
        "common_comparison_payload_digest": validated["common_comparison_payload_digest"],
        "row_input_noise_schedule_digest": validated["row_input_noise_schedule_digest"],
        "history_digest": validated["history_validation"]["history_digest"],
        "portable_checkpoint_tree": portable_tree,
        "portable_checkpoint_tree_digest": portable_tree["portable_checkpoint_tree_digest"],
        "cross_arm_gate_sha256": cross_gate["gate_sha256"],
        "cross_arm_gate_digest": cross_gate["gate_digest"],
        "cross_arm_recipe_version_digest": cross_gate["recipe_version_digest"],
        "runner_source_sha256": expected_runner_sha256,
        "latent_bundle_sha256": expected_bundle_sha256,
        "source_pins": dict(expected_source_pins),
        "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
        "external_authority_sha256": EXTERNAL_AUTHORITY_SHA256,
        "model_authority_sha256": MODEL_AUTHORITY_SHA256,
        "materializer_run_complete_sha256": MATERIALIZER_RUN_COMPLETE_SHA256,
        "materializer_run_complete_digest": MATERIALIZER_RUN_COMPLETE_DIGEST,
        "checkpoint_exact23_binding_digest": validated["pre_publish_closure_replays"]["checkpoint_exact23_binding_digest"],
        "bernini_execution_source_binding_digest": validated["pre_publish_closure_replays"]["bernini_execution_source_binding_digest"],
        "origin_verifier_binding": _portable_tool_binding_v1(
            origin_verifier_path,
            expected_sha256=expected_origin_verifier_sha256,
            label="fresh1 origin verifier",
        ),
        "gate_controller_binding": _portable_tool_binding_v1(
            gate_controller_path,
            expected_sha256=expected_gate_controller_sha256,
            label="fresh1 gate controller",
        ),
        "physical_origin_replay_passed": True,
        "closed_validator_passed": True,
    }
    attestation = {**unsigned, "attestation_digest": object_sha256(unsigned)}
    return _validate_fresh1_origin_attestation_value_v1(
        attestation,
        arm_id=arm_id,
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        cross_gate=cross_gate,
        expected_origin_verifier_binding=unsigned["origin_verifier_binding"],
        expected_gate_controller_binding=unsigned["gate_controller_binding"],
    )


def validate_fresh1_origin_attestation_v1(
    path: Path,
    *,
    expected_sha256: str,
    arm_id: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    cross_gate: Mapping[str, Any],
    expected_origin_verifier_binding: Mapping[str, Any],
    expected_gate_controller_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _read_sealed_json_held_fd_v1(
        path,
        expected_sha256=expected_sha256,
        label=f"{arm_id} portable fresh1 origin attestation",
    )
    return _validate_fresh1_origin_attestation_value_v1(
        value,
        arm_id=arm_id,
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        cross_gate=cross_gate,
        expected_origin_verifier_binding=expected_origin_verifier_binding,
        expected_gate_controller_binding=expected_gate_controller_binding,
    )


def validate_fresh1_acceptance_gate_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    cross_gate: Mapping[str, Any],
    expected_origin_verifier_binding: Mapping[str, Any],
    expected_gate_controller_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate embedded portable attestations; never dereference origin paths."""

    gate = _read_sealed_json_held_fd_v1(
        path, expected_sha256=expected_sha256, label="C2 fresh1 acceptance gate"
    )
    unsigned = dict(gate)
    digest = unsigned.pop("gate_digest", None)
    fields = {
        "schema_version",
        "status",
        "experiment_contract_sha256",
        "external_authority_sha256",
        "model_authority_sha256",
        "latent_bundle_sha256",
        "runner_source_sha256",
        "source_pins",
        "cross_arm_gate_sha256",
        "cross_arm_gate_digest",
        "cross_arm_recipe_version_digest",
        "origin_verifier_binding",
        "gate_controller_binding",
        "fresh1_origin_attestations",
        "exact_fresh1_attestation_count",
        "all_three_origin_physical_replays_passed",
        "exact10_resume_from_fresh1_forbidden",
        "gate_digest",
    }
    if (
        set(gate) != fields
        or gate.get("schema_version") != FRESH1_ACCEPTANCE_GATE_SCHEMA
        or gate.get("status") != "FRESH1_ACCEPTANCE_GATE_PASS"
        or gate.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or gate.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or gate.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or gate.get("latent_bundle_sha256") != expected_bundle_sha256
        or gate.get("runner_source_sha256") != expected_runner_sha256
        or gate.get("source_pins") != dict(expected_source_pins)
        or gate.get("cross_arm_gate_sha256") != cross_gate.get("gate_sha256")
        or gate.get("cross_arm_gate_digest") != cross_gate.get("gate_digest")
        or gate.get("cross_arm_recipe_version_digest") != cross_gate.get("recipe_version_digest")
        or gate.get("origin_verifier_binding")
        != dict(expected_origin_verifier_binding)
        or gate.get("gate_controller_binding")
        != dict(expected_gate_controller_binding)
        or gate.get("exact_fresh1_attestation_count") != 3
        or gate.get("all_three_origin_physical_replays_passed") is not True
        or gate.get("exact10_resume_from_fresh1_forbidden") is not True
        or digest != object_sha256(unsigned)
    ):
        fail("fresh1 acceptance gate envelope differs")
    rows = gate.get("fresh1_origin_attestations")
    if not isinstance(rows, list) or len(rows) != 3:
        fail("fresh1 gate requires exact3 embedded origin attestations")
    validated = []
    for row, arm_id in zip(rows, ARM_IDS):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"arm_id", "attestation_sha256", "attestation_digest", "attestation"}
            or row.get("arm_id") != arm_id
            or row.get("attestation_sha256") != _canonical_json_file_sha256_v1(row.get("attestation"))
            or row.get("attestation_digest") != row.get("attestation", {}).get("attestation_digest")
        ):
            fail("fresh1 gate portable attestation row differs")
        validated.append(
            _validate_fresh1_origin_attestation_value_v1(
                row["attestation"],
                arm_id=arm_id,
                expected_runner_sha256=expected_runner_sha256,
                expected_bundle_sha256=expected_bundle_sha256,
                expected_source_pins=expected_source_pins,
                cross_gate=cross_gate,
                expected_origin_verifier_binding=expected_origin_verifier_binding,
                expected_gate_controller_binding=expected_gate_controller_binding,
            )
        )
    return {
        "gate_sha256": expected_sha256,
        "gate_digest": digest,
        "cross_arm_gate_sha256": cross_gate["gate_sha256"],
        "cross_arm_gate_digest": cross_gate["gate_digest"],
        "cross_arm_recipe_version_digest": cross_gate["recipe_version_digest"],
        "fresh1_attestation_sha256_by_arm": {
            arm_id: rows[index]["attestation_sha256"]
            for index, arm_id in enumerate(ARM_IDS)
        },
        "all_three_portable_origin_attestations_replayed": True,
        "exact10_must_fresh_initialize": True,
        "resume_from_fresh1_forbidden": True,
    }


def validate_experiment_contract_v1(
    path: Path, *, expected_sha256: str
) -> Mapping[str, Any]:
    if expected_sha256 != EXPERIMENT_CONTRACT_SHA256:
        fail("experiment contract literal SHA differs")
    value = _read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_size=EXPERIMENT_CONTRACT_SIZE,
        label="C2 experiment contract",
    )
    unsigned = dict(value)
    digest = unsigned.pop("contract_digest", None)
    if (
        value.get("schema_version") != EXPERIMENT_CONTRACT_SCHEMA
        or digest != EXPERIMENT_CONTRACT_DIGEST
        or object_sha256(unsigned) != digest
        or tuple(value.get("authorized_row_ids", ())) != ROW_IDS
        or value.get("packet_manifest_sha256") != PACKET_MANIFEST_SHA256
        or value.get("status")
        != "PREREGISTERED_C2_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
    ):
        fail("experiment contract identity/envelope differs")
    initialization = value.get("initialization")
    topology = value.get("topology")
    objective = value.get("objective_contract")
    comparison = value.get("comparison_contract")
    gates = value.get("preregistered_gates")
    claims = value.get("claim_boundaries")
    bindings = value.get("authority_bindings")
    selection = value.get("selection_rule")
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("residual_gain_formula") != CONTROLLED_GAIN_FORMULA
        or initialization.get("residual_gain_float32_bits")
        != f"0x{CONTROLLED_GAIN_FLOAT32_HEX}"
        or initialization.get("residual_gain_count") != BLOCKS
        or initialization.get("fresh_official_frozen_base_required") is not True
        or initialization.get("c1_checkpoint_consumption_forbidden") is not True
        or not isinstance(topology, Mapping)
        or topology.get("world_size") != WORLD_SIZE
        or topology.get("sequence_parallel_size") != SP_SIZE
        or topology.get("data_parallel_size") != DP_SIZE
        or topology.get("global_optimizer_updates") != MAX_STEPS
        or topology.get("dp_row_mapping")
        != {"0": ROW_IDS[0], "1": ROW_IDS[1]}
        or not isinstance(objective, Mapping)
        or objective.get("branch_reduction") != "strict_arithmetic_mean"
        or objective.get("shared_epsilon_within_exact2_branches") is not True
        or objective.get("arm_a_branches")
        != [
            "target_q_to_target_clean_latent",
            "exact_duplicate_target_q_to_target_clean_latent",
        ]
        or objective.get("arm_b_branches")
        != [
            "target_q_to_target_clean_latent",
            "role_swap_q_to_role_swap_clean_latent",
        ]
        or not isinstance(comparison, Mapping)
        or comparison.get("arms_are_independent_world8_runs") is not True
        or comparison.get("cross_arm_data_parallel_all_reduce_forbidden")
        is not True
        or comparison.get("a_b_initial_trainable_parameter_digest_bit_identical")
        is not True
        or not isinstance(gates, Mapping)
        or gates.get("evaluation_energy_abi", {}).get("renderer_timestep_value")
        != 999
        or gates.get("evaluation_energy_abi", {}).get("renderer_timestep_dtype")
        != "torch.int64"
        or gates.get("evaluation_energy_abi", {}).get("sigma_float32") != 1.0
        or gates.get("evaluation_energy_abi", {}).get("x_sigma") != "epsilon"
        or gates.get("evaluation_energy_abi", {}).get("epsilon_shape")
        != list(LATENT_SHAPE)
        or gates.get("role_only_gate", {}).get("cell_count") != 4
        or gates.get("role_only_gate", {}).get("fixed_fields")
        != [
            "source",
            "instruction",
            "x_sigma",
            "epsilon",
            "q_local",
            "q_phase",
            "q_terminal",
            "q_camera",
            "validity",
            "spatial_masks",
            "semantic_role_code_order",
        ]
        or gates.get("role_only_gate", {}).get("swapped_fields")
        != ["q_entity_slots", "directed_q_relation_edges"]
        or gates.get("role_only_gate", {}).get("required_mapping_evidence")
        != (
            "matched_uses_clean_variant_slot_entity_ids_and_mismatch_uses_"
            "opposite_variant_slot_entity_ids;the_two_physical_entity_to_"
            "semantic_slot_mappings_must_differ"
        )
        or tuple(
            (row.get("row_id"), row.get("clean_variant"))
            for row in gates.get("role_only_gate", {}).get("cells", ())
        )
        != ROLE_ONLY_CELL_ORDER
        or gates.get("full_q_route_gate", {}).get("claim_name")
        != "oracle_route_controllability_only"
        or gates.get("full_q_route_gate", {}).get(
            "participant_role_binding_claim_forbidden"
        )
        is not True
        or not isinstance(claims, Mapping)
        or claims.get("teacher_forced_oracle_q_simulator_diagnostic_only")
        is not True
        or any(claims.get(key) is not False for key in (
            "exact160",
            "formal_c2",
            "production_model",
            "real_video_generalization",
            "scientific_promotion",
            "source_instruction_inference",
        ))
        or not isinstance(bindings, Mapping)
        or bindings.get("derivative_authority_file_sha256")
        != EXTERNAL_AUTHORITY_SHA256
        or bindings.get("derivative_authority_digest")
        != EXTERNAL_AUTHORITY_DIGEST
        or bindings.get("model_authority_file_sha256") != MODEL_AUTHORITY_SHA256
        or bindings.get("model_authority_digest") != MODEL_AUTHORITY_DIGEST
        or not isinstance(selection, Mapping)
        or selection.get("primary_metric")
        != "minimum_of_four_role_only_matched_vs_mismatch_margins"
        or selection.get("all_hard_gates_must_pass_before_ranking") is not True
    ):
        fail("experiment contract closed recipe differs")
    arms = value.get("arm_rows")
    expected_arms = [
        {
            "arm_id": ARM_DUPLICATE,
            "holder_job_id": "141620",
            "node": "auh7-1b-gpu-226",
            "recipe": "target_duplicate_exact2",
            "seed": 20260821,
        },
        {
            "arm_id": ARM_ROLE_PAIR,
            "holder_job_id": "141618",
            "node": "auh7-1b-gpu-249",
            "recipe": "target_and_role_swap_exact2",
            "seed": 20260821,
        },
        {
            "arm_id": ARM_ROLE_REPLICA,
            "holder_job_id": "141619",
            "node": "auh7-1b-gpu-257",
            "recipe": "target_and_role_swap_exact2",
            "seed": 20260822,
        },
    ]
    if arms != expected_arms:
        fail("experiment arm/node/seed registry differs")
    return dict(value)


def validate_external_authority_v1(
    path: Path, *, expected_sha256: str
) -> Mapping[str, Any]:
    if expected_sha256 != EXTERNAL_AUTHORITY_SHA256:
        fail("external authority literal SHA differs")
    value = _read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_size=None,
        label="C2 external optimizer authority",
    )
    unsigned = dict(value)
    digest = unsigned.pop("authority_digest", None)
    restrictions = value.get("training_objective_restrictions")
    if (
        value.get("schema_version") != EXTERNAL_AUTHORITY_SCHEMA
        or digest != EXTERNAL_AUTHORITY_DIGEST
        or object_sha256(unsigned) != digest
        or tuple(value.get("authorized_row_ids", ())) != ROW_IDS
        or value.get("max_optimizer_updates_per_arm") != MAX_STEPS
        or value.get("oracle_q_teacher_forced_required") is not True
        or value.get("fresh_optimizer_run_required") is not True
        or value.get("packet_manifest_sha256") != PACKET_MANIFEST_SHA256
        or value.get("packet_status_preserved") != "ELAL3_SIM_DIAGNOSTIC"
        or value.get("status")
        != "AUTHORIZED_C2_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
        or not isinstance(restrictions, Mapping)
        or dict(restrictions)
        != {
            "frozen_base_velocity_reference_forbidden": True,
            "frozen_teacher_self_distillation_forbidden": True,
            "hand_tuned_reward_scalar_forbidden": True,
            "target_grounded_event_and_context_flow_only": True,
        }
    ):
        fail("external authority closed scope differs")
    return dict(value)


def validate_runtime_arm_placement_v1(
    authority: Mapping[str, Any], contract: Mapping[str, Any], *, arm_id: str
) -> Mapping[str, Any]:
    if arm_id not in ARM_PLACEMENT:
        fail("unknown preregistered arm")
    job, node, seed = ARM_PLACEMENT[arm_id]
    allowed = authority.get("allowed_nodes")
    if (
        not isinstance(allowed, list)
        or {"holder_job_id": job, "node": node} not in allowed
        or not any(row.get("arm_id") == arm_id for row in contract.get("arm_rows", ()))
    ):
        fail("arm is not authorized at preregistered placement")
    runtime_node = os.uname().nodename.split(".", 1)[0]
    runtime_job = os.environ.get("SLURM_JOB_ID")
    if runtime_node != node or runtime_job != job:
        fail(
            f"runtime placement differs for {arm_id}: "
            f"job={runtime_job}, node={runtime_node}"
        )
    return {"arm_id": arm_id, "holder_job_id": job, "node": node, "seed": seed}


def validate_model_authority_v1(
    path: Path,
    *,
    expected_sha256: str,
    bernini_root: Path,
    checkpoint_root: Path,
) -> Mapping[str, Any]:
    if expected_sha256 != MODEL_AUTHORITY_SHA256:
        fail("model authority literal SHA differs")
    value = _read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_size=None,
        label="C2 real-model authority",
    )
    unsigned = dict(value)
    digest = unsigned.pop("authority_digest", None)
    constraints = value.get("constraints")
    rows = value.get("files")
    if (
        value.get("schema_version") != MODEL_AUTHORITY_SCHEMA
        or digest != MODEL_AUTHORITY_DIGEST
        or object_sha256(unsigned) != digest
        or tuple(value.get("authorized_row_ids", ())) != ROW_IDS
        or value.get("file_count") != 9
        or not isinstance(rows, list)
        or len(rows) != 9
        or value.get("bernini_root") != str(bernini_root)
        or value.get("checkpoint_root") != str(checkpoint_root)
        or not isinstance(constraints, Mapping)
        or dict(constraints)
        != {
            "allowed_operation": "elal3_c2_simulator_oracle_q_optimizer_diagnostic",
            "exact160_authorized": False,
            "formal_c2_authorized": False,
            "max_optimizer_updates_per_arm": MAX_STEPS,
            "real_video_claim_authorized": False,
            "scientific_claim_authorized": False,
            "source_instruction_inference_claim_authorized": False,
        }
    ):
        fail("real-model authority envelope differs")
    python_root = Path(str(value.get("python_env_root"))).resolve(strict=True)
    roots = {
        "bernini": bernini_root,
        "checkpoint": checkpoint_root,
        "python_env": python_root,
    }
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "mode",
            "relative_path",
            "root",
            "sha256",
            "size",
        }:
            fail("real-model authority file row differs")
        root_name = row.get("root")
        relative = row.get("relative_path")
        if root_name not in roots or type(relative) is not str:
            fail("real-model authority file root/path differs")
        key = (root_name, relative)
        if key in seen:
            fail("real-model authority duplicate file row")
        seen.add(key)
        try:
            actual = c1._plain_relative(roots[root_name], relative, label=f"C2 model pin {key}")
        except c1.ELAL3C1TrainingError as error:
            raise ELAL3C2TrainingError(str(error)) from error
        info = actual.stat()
        if (
            stat.S_IMODE(info.st_mode) != row.get("mode")
            or info.st_size != row.get("size")
            or file_sha256(actual) != row.get("sha256")
        ):
            fail(f"real-model authority pin differs: {key}")
    required = {
        ("bernini", "bernini/pipeline.py"),
        ("checkpoint", "transformer/config.json"),
        (
            "checkpoint",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ),
        *(("checkpoint", f"transformer/{name}") for name in c1.MODEL_SHARDS),
        ("checkpoint", "vae/config.json"),
        ("checkpoint", "vae/diffusion_pytorch_model.safetensors"),
        ("python_env", "diffusers/__init__.py"),
        (
            "python_env",
            "diffusers/models/autoencoders/autoencoder_kl_wan.py",
        ),
    }
    if seen != required:
        fail("real-model authority exact9 closure differs")
    return dict(value)


def require_model_authority_replay_identity_v1(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, stage: str
) -> Mapping[str, Any]:
    if stage not in {"post_deserialize", "final_pre_publish"}:
        fail("model authority replay stage differs")
    before = canonical_json_bytes(reference)
    after = canonical_json_bytes(candidate)
    if before != after:
        fail(f"model authority {stage} replay differs")
    return {
        "stage": stage,
        "authority_sha256": MODEL_AUTHORITY_SHA256,
        "authority_digest": MODEL_AUTHORITY_DIGEST,
        "replayed_object_sha256": hashlib.sha256(after).hexdigest(),
        "exact9_rehashed_by_rank_zero": True,
        "world8_barrier_before_replay": True,
        "world8_broadcast_identity_verified": True,
    }


def validate_model_authority_strong_v1(
    *,
    materializer_module: Any,
    path: Path,
    expected_sha256: str,
    bernini_root: Path,
    checkpoint_root: Path,
    pipeline_module: Any,
    diffusers_module: Any,
    wan_module: Any,
) -> Mapping[str, Any]:
    """Use the frozen materializer's held-openat exact9/import validator."""

    try:
        value, authority_binding, file_bindings = (
            materializer_module.validate_model_authority(
                path,
                expected_sha256,
                bernini_root=bernini_root,
                checkpoint_root=checkpoint_root,
            )
        )
        imported = materializer_module.validate_imported_model_modules(
            model_authority=value,
            model_file_bindings=file_bindings,
            pipeline_module=pipeline_module,
            diffusers_module=diffusers_module,
            wan_module=wan_module,
        )
    except materializer_module.ELAL3SimulatorC2VAEError as error:
        raise ELAL3C2TrainingError(str(error)) from error
    if (
        value.get("authority_digest") != MODEL_AUTHORITY_DIGEST
        or value.get("schema_version") != MODEL_AUTHORITY_SCHEMA
        or len(file_bindings) != 9
        or len(imported) != 3
    ):
        fail("strong C2 exact9 model authority result differs")
    result = {
        "authority": dict(value),
        "authority_file_binding": authority_binding,
        "exact9_file_bindings": file_bindings,
        "actual_imported_model_module_bindings": imported,
        "held_openat_nofollow_nlink1": True,
        "callable_ownership_verified": True,
    }
    return {**result, "strong_replay_digest": object_sha256(result)}


def replay_strong_model_authority_world8_v1(
    *,
    dist: Any,
    group: Any,
    rank: int,
    reference: Mapping[str, Any],
    materializer_module: Any,
    authority_path: Path,
    expected_sha256: str,
    bernini_root: Path,
    checkpoint_root: Path,
    pipeline_module: Any,
    diffusers_module: Any,
    wan_module: Any,
    stage: str,
) -> Mapping[str, Any]:
    if stage not in {"post_deserialize", "final_pre_publish"}:
        fail("strong model replay stage differs")
    dist.barrier(group=group)
    box: list[Any] = [None]
    if rank == 0:
        try:
            candidate = validate_model_authority_strong_v1(
                materializer_module=materializer_module,
                path=authority_path,
                expected_sha256=expected_sha256,
                bernini_root=bernini_root,
                checkpoint_root=checkpoint_root,
                pipeline_module=pipeline_module,
                diffusers_module=diffusers_module,
                wan_module=wan_module,
            )
            if canonical_json_bytes(reference) != canonical_json_bytes(candidate):
                fail(f"strong model authority {stage} replay differs")
            box[0] = {"ok": True, "candidate": candidate}
        except Exception as error:
            box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(box, src=0, group=group)
    result = box[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"rank-zero strong model authority replay failed: {result!r}")
    if canonical_json_bytes(reference) != canonical_json_bytes(result["candidate"]):
        fail("broadcast strong model authority replay differs")
    dist.barrier(group=group)
    return {
        "stage": stage,
        "authority_sha256": MODEL_AUTHORITY_SHA256,
        "authority_digest": MODEL_AUTHORITY_DIGEST,
        "strong_replay_digest": reference["strong_replay_digest"],
        "exact9_held_openat_replayed": True,
        "actual_imported_modules_and_callable_ownership_replayed": True,
        "world8_broadcast_identity_verified": True,
    }


@dataclass(frozen=True)
class C2LatentBundleV1:
    local_row_index: int
    local_row_id: str
    local_tensors: Mapping[str, Any]
    tensor_rows: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]
    bundle_sha256: str
    receipt_sha256: str
    original_materialization_path: str
    runtime_path: str
    live_bundle_binding: Mapping[str, Any]

    def tensor(self, variant: str) -> Any:
        if variant not in ("source", "target", "role_swap"):
            fail("runtime requested a non-retained C2 latent variant")
        return self.local_tensors[variant]


def _expected_safetensors_metadata_v1() -> Mapping[str, str]:
    return {
        "schema_version": LATENT_BUNDLE_SCHEMA,
        "row_ids": json.dumps(list(ROW_IDS), separators=(",", ":")),
        "variant_order": json.dumps(list(VARIANT_ORDER), separators=(",", ":")),
        "tensor_order": json.dumps(list(LATENT_TENSOR_ORDER), separators=(",", ":")),
        "tensor_order_digest": object_sha256(list(LATENT_TENSOR_ORDER)),
        "tensor_count": "16",
        "bucket_hw": "416,560",
        "latent_shape_each": "1,16,21,52,70",
        "dtype_each": "F32",
    }


def validate_bundle_relocation_binding_v1(
    *,
    receipt_bundle_binding: Mapping[str, Any],
    live_bundle_binding: Mapping[str, Any],
    runtime_path: Path,
    expected_sha256: str,
    expected_size: int,
) -> Mapping[str, Any]:
    """Bind relocated live bytes while retaining the materializer path as provenance."""

    original_path = receipt_bundle_binding.get("path")
    if (
        type(original_path) is not str
        or not original_path
        or receipt_bundle_binding.get("sha256") != expected_sha256
        or receipt_bundle_binding.get("size") != expected_size
        or receipt_bundle_binding.get("mode") != 0o444
        or receipt_bundle_binding.get("nlink") != 1
        or receipt_bundle_binding.get("held_fd_double_read_verified") is not True
        or receipt_bundle_binding.get("held_openat_parent_chain_replayed") is not True
        or live_bundle_binding.get("path") != str(runtime_path)
        or live_bundle_binding.get("sha256") != expected_sha256
        or live_bundle_binding.get("size") != expected_size
        or live_bundle_binding.get("mode") != 0o444
        or live_bundle_binding.get("nlink") != 1
        or live_bundle_binding.get("held_fd_double_read_verified") is not True
        or live_bundle_binding.get("held_openat_parent_chain_replayed") is not True
    ):
        fail("exact16 original/live relocation binding differs")
    return {
        "original_materialization_path": original_path,
        "runtime_path": str(runtime_path),
        "same_path_required": False,
        "same_sealed_sha256_size_required": True,
        "sha256": expected_sha256,
        "size": expected_size,
    }


def validate_materializer_run_complete_v1(
    path: Path,
    *,
    expected_sha256: str,
    label_module: Any,
) -> Mapping[str, Any]:
    """Consume the one frozen retry2 materializer release attestation."""

    if expected_sha256 != MATERIALIZER_RUN_COMPLETE_SHA256:
        fail("materializer RUN_COMPLETE CLI SHA differs from retry2 literal")
    payload, binding = label_module.stable_read_path(
        path,
        label="C2 retry2 materializer RUN_COMPLETE consumer",
        expected_sha256=expected_sha256,
        expected_mode=0o444,
        allowed_root=path.parent,
    )
    if binding.get("size") != MATERIALIZER_RUN_COMPLETE_SIZE:
        fail("materializer RUN_COMPLETE registered size differs")
    value = label_module._canonical_json_payload(
        payload, label="C2 retry2 materializer RUN_COMPLETE consumer"
    )
    expected_top = {
        "schema_version",
        "status",
        "holder_job_id",
        "node",
        "materialized",
        "materializer_internal_pre_post_final_replay_passed",
        "mode_contract",
        "packet_closure",
        "packet_manifest_sha256",
        "release",
        "formal_c2_authorized",
        "exact160_authorized",
        "real_video_generalization_authorized",
        "scientific_claim_authorized",
        "source_instruction_inference_authorized",
        "run_digest",
    }
    unsigned = dict(value)
    digest = unsigned.pop("run_digest", None)
    materialized = value.get("materialized")
    release = value.get("release")
    if (
        set(value) != expected_top
        or value.get("schema_version") != MATERIALIZER_RUN_COMPLETE_SCHEMA
        or value.get("status") != "COMPLETE_SIMULATOR_C2_EXACT16_ONLY"
        or value.get("holder_job_id") != "141620"
        or value.get("node") != "auh7-1b-gpu-226"
        or digest != MATERIALIZER_RUN_COMPLETE_DIGEST
        or digest != object_sha256(unsigned)
        or value.get("packet_manifest_sha256") != PACKET_MANIFEST_SHA256
        or value.get("materializer_internal_pre_post_final_replay_passed")
        is not True
        or any(
            value.get(field) is not False
            for field in (
                "formal_c2_authorized",
                "exact160_authorized",
                "real_video_generalization_authorized",
                "scientific_claim_authorized",
                "source_instruction_inference_authorized",
            )
        )
        or value.get("mode_contract")
        != {
            "archive_member_mode": "0444",
            "fresh_runtime_extract_file_mode": "0644_required_by_consumer",
            "fresh_runtime_extract_root_mode": "0555",
            "published_bundle_and_receipt_mode": "0444",
        }
        or value.get("packet_closure")
        != {
            "live_c2_media_annotation_receipt_triples_pre_post": 16,
            "manifest_declared_media_count": 24,
        }
        or not isinstance(materialized, Mapping)
        or set(materialized)
        != {
            "bundle_relative_path",
            "bundle_sha256",
            "bundle_size",
            "receipt_digest",
            "receipt_relative_path",
            "receipt_sha256",
            "receipt_size",
            "tensor_count",
            "tensor_order",
        }
        or materialized.get("bundle_relative_path")
        != "materialized/c2-exact16-latents.safetensors"
        or materialized.get("bundle_sha256") != LATENT_BUNDLE_SHA256
        or materialized.get("bundle_size") != LATENT_BUNDLE_SIZE
        or materialized.get("receipt_relative_path")
        != "materialized/latent-bundle-receipt.json"
        or materialized.get("receipt_sha256") != LATENT_BUNDLE_RECEIPT_SHA256
        or materialized.get("receipt_size") != LATENT_BUNDLE_RECEIPT_SIZE
        or materialized.get("receipt_digest") != LATENT_BUNDLE_RECEIPT_DIGEST
        or materialized.get("tensor_count") != 16
        or materialized.get("tensor_order") != list(LATENT_TENSOR_ORDER)
        or not isinstance(release, Mapping)
        or set(release)
        != {
            "archive_sha256",
            "archive_size",
            "external_controller_sha256",
            "external_controller_size",
            "launcher_sha256",
            "launcher_size",
            "manifest_sha256",
            "manifest_size",
        }
    ):
        fail("materializer retry2 RUN_COMPLETE envelope/digest differs")
    for field, item in release.items():
        if field.endswith("_sha256"):
            _require_sha256_v1(item, label=f"materializer release {field}")
        elif type(item) is not int or item <= 0:
            fail(f"materializer release {field} size differs")
    return {
        "schema_version": MATERIALIZER_RUN_COMPLETE_SCHEMA,
        "status": value["status"],
        "file_sha256": expected_sha256,
        "file_size": MATERIALIZER_RUN_COMPLETE_SIZE,
        "run_digest": digest,
        "bundle_sha256": LATENT_BUNDLE_SHA256,
        "bundle_size": LATENT_BUNDLE_SIZE,
        "receipt_sha256": LATENT_BUNDLE_RECEIPT_SHA256,
        "receipt_size": LATENT_BUNDLE_RECEIPT_SIZE,
        "receipt_digest": LATENT_BUNDLE_RECEIPT_DIGEST,
        "retry_generation": "retry2_only",
        "held_file_binding": dict(binding),
    }


def load_c2_latent_bundle_v1(
    *,
    bundle_path: Path,
    expected_bundle_sha256: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
    packet_root: Path,
    local_row_index: int,
    label_module: Any,
    materializer_module: Any,
) -> C2LatentBundleV1:
    """Held-byte exact16 consumer; retain only source/target/role-swap locally."""

    import torch
    try:
        from safetensors.torch import load as load_safetensors
    except ImportError as error:
        raise ELAL3C2TrainingError("safetensors runtime is required") from error
    if (
        LATENT_BUNDLE_SHA256 is None
        or LATENT_BUNDLE_SIZE is None
        or LATENT_BUNDLE_RECEIPT_SHA256 is None
        or LATENT_BUNDLE_RECEIPT_SIZE is None
        or LATENT_BUNDLE_RECEIPT_DIGEST is None
    ):
        fail("exact16 bundle external release literals are not frozen")
    if (
        expected_bundle_sha256 != LATENT_BUNDLE_SHA256
        or expected_receipt_sha256 != LATENT_BUNDLE_RECEIPT_SHA256
        or local_row_index not in (0, 1)
    ):
        fail("exact16 bundle CLI pins/local row differ")
    receipt_payload, receipt_binding = label_module.stable_read_path(
        receipt_path,
        label="C2 exact16 latent receipt consumer",
        expected_sha256=expected_receipt_sha256,
        expected_mode=0o444,
        allowed_root=receipt_path.parent,
    )
    if receipt_binding["size"] != LATENT_BUNDLE_RECEIPT_SIZE:
        fail("exact16 receipt registered size differs")
    receipt = label_module._canonical_json_payload(
        receipt_payload, label="C2 exact16 latent receipt consumer"
    )
    unsigned = dict(receipt)
    receipt_digest = unsigned.pop("receipt_digest", None)
    expected_top = {
        "schema_version",
        "status",
        "bundle",
        "bundle_format",
        "tensor_order",
        "tensor_order_digest",
        "tensor_rows",
        "row_ids",
        "variant_order",
        "exact_media_count",
        "bucket_hw",
        "latent_shape_each",
        "safetensors_metadata",
        "published_bundle_verification",
        "packet_binding",
        "derivative_authority_binding",
        "experiment_contract_binding",
        "real_model_authority_binding",
        "runtime_source_bindings",
        "imported_model_module_bindings",
        "runtime",
        "encoding",
        "authority",
        "receipt_digest",
    }
    if (
        set(receipt) != expected_top
        or receipt.get("schema_version") != LATENT_RECEIPT_SCHEMA
        or receipt.get("status") != "ELAL3_SIMULATOR_C2_EXACT16_VAE_GO"
        or receipt.get("bundle_format") != "safetensors-exact16-fp32-v1"
        or receipt.get("tensor_order") != list(LATENT_TENSOR_ORDER)
        or receipt.get("tensor_order_digest")
        != object_sha256(list(LATENT_TENSOR_ORDER))
        or receipt.get("row_ids") != list(ROW_IDS)
        or receipt.get("variant_order") != list(VARIANT_ORDER)
        or receipt.get("exact_media_count") != 16
        or receipt.get("bucket_hw") != list(BUCKET_HW)
        or receipt.get("latent_shape_each") != list(LATENT_SHAPE)
        or receipt.get("safetensors_metadata")
        != _expected_safetensors_metadata_v1()
        or receipt_digest != LATENT_BUNDLE_RECEIPT_DIGEST
        or receipt_digest != object_sha256(unsigned)
    ):
        fail("exact16 receipt closed envelope/digest differs")
    bundle_binding = receipt.get("bundle")
    if not isinstance(bundle_binding, Mapping):
        fail("exact16 bundle materialization binding differs")
    derivative = receipt.get("derivative_authority_binding")
    experiment = receipt.get("experiment_contract_binding")
    model = receipt.get("real_model_authority_binding")
    if (
        not isinstance(derivative, Mapping)
        or derivative.get("file_sha256") != EXTERNAL_AUTHORITY_SHA256
        or derivative.get("authority_digest") != EXTERNAL_AUTHORITY_DIGEST
        or derivative.get("verified_before_and_after_encoding") is not True
        or not isinstance(experiment, Mapping)
        or experiment.get("file_sha256") != EXPERIMENT_CONTRACT_SHA256
        or experiment.get("contract_digest") != EXPERIMENT_CONTRACT_DIGEST
        or experiment.get("verified_before_and_after_encoding") is not True
        or not isinstance(model, Mapping)
        or model.get("file_sha256") != MODEL_AUTHORITY_SHA256
        or model.get("authority_digest") != MODEL_AUTHORITY_DIGEST
        or model.get("verified_before_and_after_encoding") is not True
        or len(model.get("verified_file_bindings", ())) != 9
    ):
        fail("exact16 authority closure differs")
    model_files = model.get("verified_file_bindings")
    expected_model_rows = (
        ("bernini", "bernini/pipeline.py"),
        ("checkpoint", "transformer/config.json"),
        (
            "checkpoint",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        ),
        (
            "checkpoint",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        ),
        (
            "checkpoint",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ),
        ("checkpoint", "vae/config.json"),
        ("checkpoint", "vae/diffusion_pytorch_model.safetensors"),
        ("python_env", "diffusers/__init__.py"),
        (
            "python_env",
            "diffusers/models/autoencoders/autoencoder_kl_wan.py",
        ),
    )
    model_binding_fields = {
        "authority_relative_path",
        "authority_root",
        "authority_row_index",
        "device",
        "held_fd_double_hash_verified",
        "held_openat_parent_chain_replayed",
        "inode",
        "mode",
        "nlink",
        "path",
        "sha256",
        "size",
    }
    if not isinstance(model_files, list) or len(model_files) != 9:
        fail("exact16 exact9 model file bindings differ")
    for index, (row, (root, relative)) in enumerate(
        zip(model_files, expected_model_rows)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != model_binding_fields
            or row.get("authority_row_index") != index
            or row.get("authority_root") != root
            or row.get("authority_relative_path") != relative
            or type(row.get("path")) is not str
            or not row.get("path")
            or type(row.get("device")) is not int
            or type(row.get("inode")) is not int
            or type(row.get("size")) is not int
            or row.get("size") <= 0
            or row.get("mode") not in (0o444, 0o644)
            or row.get("nlink") != 1
            or row.get("held_fd_double_hash_verified") is not True
            or row.get("held_openat_parent_chain_replayed") is not True
        ):
            fail(f"exact16 exact9 model binding row differs: {index}")
        _require_sha256_v1(row.get("sha256"), label=f"model binding {index} SHA")
    imported = receipt.get("imported_model_module_bindings")
    expected_imports = (
        (0, "bernini.pipeline"),
        (7, "diffusers"),
        (8, "diffusers.models.autoencoders.autoencoder_kl_wan"),
    )
    imported_rows = imported.get("modules") if isinstance(imported, Mapping) else None
    if (
        not isinstance(imported, Mapping)
        or set(imported)
        != {"module_count", "modules", "verified_before_and_after_encoding"}
        or imported.get("module_count") != 3
        or imported.get("verified_before_and_after_encoding") is not True
        or not isinstance(imported_rows, list)
        or len(imported_rows) != 3
    ):
        fail("exact16 imported model module binding closure differs")
    imported_fields = model_binding_fields | {
        "import_index",
        "import_name",
        "actual_module_file_verified",
    }
    for import_index, (row, (authority_index, import_name)) in enumerate(
        zip(imported_rows, expected_imports)
    ):
        donor = model_files[authority_index]
        if (
            not isinstance(row, Mapping)
            or set(row) != imported_fields
            or row.get("import_index") != import_index
            or row.get("import_name") != import_name
            or row.get("authority_row_index") != authority_index
            or row.get("actual_module_file_verified") is not True
            or any(row.get(field) != donor.get(field) for field in model_binding_fields)
        ):
            fail(f"exact16 imported module donor binding differs: {import_name}")
    sources = receipt.get("runtime_source_bindings")
    if (
        not isinstance(sources, Mapping)
        or sources.get("source_count") != 6
        or sources.get("verified_actual_import_module_files") is not True
        or sources.get("verified_before_and_after_encoding") is not True
        or sources.get("trainer_consumption_requires_external_release_pin")
        is not True
        or not isinstance(sources.get("sources"), list)
    ):
        fail("exact16 materialization source closure differs")
    source_map = {
        row.get("module_name"): row for row in sources["sources"]
        if isinstance(row, Mapping)
    }
    materializer_embedded = materializer_module.RUNTIME_SOURCE_PINS
    wanted_sources: Mapping[str, tuple[str, int, str]] = {
        "materialize_elal3_simulator_c2_vae_v1": (
            C2_MATERIALIZER_SHA256,
            C2_MATERIALIZER_SIZE,
            "materialize_elal3_simulator_c2_vae_v1.py",
        ),
        "elal3_simulator_c2_label_v1": (
            C2_LABEL_SHA256,
            C2_LABEL_SIZE,
            "elal3_simulator_c2_label_v1.py",
        ),
        "elal3_c0_v1": (C1_CORE_SHA256, C1_CORE_SIZE, "elal3_c0_v1.py"),
        "train_lora": (TRAIN_LORA_SHA256, TRAIN_LORA_SIZE, "train_lora.py"),
        "tools.materialize_vae": (
            materializer_embedded["tools.materialize_vae"]["sha256"],
            materializer_embedded["tools.materialize_vae"]["size"],
            "tools/materialize_vae.py",
        ),
        "tools.build_renderer_dataset": (
            materializer_embedded["tools.build_renderer_dataset"]["sha256"],
            materializer_embedded["tools.build_renderer_dataset"]["size"],
            "tools/build_renderer_dataset.py",
        ),
    }
    if any(
        name not in source_map
        or source_map[name].get("sha256") != sha
        or source_map[name].get("size") != size
        or source_map[name].get("relative_path") != relative
        or source_map[name].get("source_index")
        != list(wanted_sources).index(name)
        or source_map[name].get("mode") != 0o644
        or source_map[name].get("nlink") != 1
        or source_map[name].get("held_fd_double_hash_verified") is not True
        or source_map[name].get("held_openat_parent_chain_replayed") is not True
        or source_map[name].get("actual_module_file_verified") is not True
        for name, (sha, size, relative) in wanted_sources.items()
    ) or set(source_map) != set(wanted_sources):
        fail("exact16 materializer runtime source release pins differ")
    encoding = receipt.get("encoding")
    authority = receipt.get("authority")
    if (
        not isinstance(encoding, Mapping)
        or encoding.get("vae_encode_count") != 16
        or any(
            encoding.get(key) is not True
            for key in (
                "each_media_independently_full_video_vae_encoded",
                "source_media_consumed_from_authenticated_held_fd_bytes",
                "packet_replayed_before_and_after_encoding",
                "model_files_double_hashed_before_and_after_encoding",
                "derivative_authority_replayed_before_and_after_encoding",
                "experiment_contract_replayed_before_and_after_encoding",
                "runtime_sources_replayed_before_and_after_encoding",
                "bundle_serialized_to_bytes_then_create_only_written",
                "published_bundle_reloaded_and_exact16_verified",
            )
        )
        or not isinstance(authority, Mapping)
        or authority.get("teacher_forced_oracle_q_required_for_optimizer_use")
        is not True
        or any(
            authority.get(key) is not False
            for key in (
                "formal_c2_authorized",
                "exact160_authorized",
                "scientific_claim_authorized",
                "real_video_data",
                "source_instruction_inference_authorized",
                "materializer_source_independently_authorized_here",
            )
        )
    ):
        fail("exact16 encoding/claim closure differs")
    rows = receipt.get("tensor_rows")
    if not isinstance(rows, list) or len(rows) != 16:
        fail("exact16 tensor receipt rows differ")
    packet = label_module.load_verified_c2_packet(packet_root)
    for index, (key, row) in enumerate(zip(LATENT_TENSOR_ORDER, rows)):
        row_index = index // len(VARIANT_ORDER)
        variant = VARIANT_ORDER[index % len(VARIANT_ORDER)]
        row_id = ROW_IDS[row_index]
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "tensor_key",
                "row_index",
                "row_id",
                "variant",
                "shape",
                "dtype",
                "tensor_sha256",
                "source_media_sha256",
            }
            or row.get("tensor_key") != key
            or row.get("row_index") != row_index
            or row.get("row_id") != row_id
            or row.get("variant") != variant
            or row.get("shape") != list(LATENT_SHAPE)
            or row.get("dtype") != "torch.float32"
            or row.get("source_media_sha256")
            != packet.rows[row_id].row["media"][variant]["sha256"]
        ):
            fail(f"exact16 tensor row/provenance differs: {key}")
        _require_sha256_v1(row.get("tensor_sha256"), label=f"{key} tensor SHA")
    bundle_payload, live_bundle_binding = label_module.stable_read_path(
        bundle_path,
        label="C2 exact16 latent bundle consumer",
        expected_sha256=expected_bundle_sha256,
        expected_mode=0o444,
        allowed_root=bundle_path.parent,
    )
    if live_bundle_binding["size"] != LATENT_BUNDLE_SIZE:
        fail("exact16 bundle registered size differs")
    relocation_binding = validate_bundle_relocation_binding_v1(
        receipt_bundle_binding=bundle_binding,
        live_bundle_binding=live_bundle_binding,
        runtime_path=bundle_path,
        expected_sha256=expected_bundle_sha256,
        expected_size=LATENT_BUNDLE_SIZE,
    )
    header = materializer_module._strict_safetensors_header(bundle_payload)
    if header.get("__metadata__") != _expected_safetensors_metadata_v1():
        fail("exact16 safetensors metadata differs")
    tensors = load_safetensors(bundle_payload)
    if tuple(sorted(tensors)) != tuple(sorted(LATENT_TENSOR_ORDER)):
        fail("exact16 safetensors key closure differs")
    retained: dict[str, Any] = {}
    for index, (key, row) in enumerate(zip(LATENT_TENSOR_ORDER, rows)):
        tensor = tensors[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or tuple(tensor.shape) != LATENT_SHAPE
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or not bool(torch.isfinite(tensor).all().item())
            or c1.tensor_sha256_v1(tensor) != row["tensor_sha256"]
        ):
            fail(f"exact16 safetensors tensor differs: {key}")
        if index // len(VARIANT_ORDER) == local_row_index:
            variant = VARIANT_ORDER[index % len(VARIANT_ORDER)]
            if variant in ("source", "target", "role_swap"):
                retained[variant] = tensor.clone(memory_format=torch.contiguous_format)
    published = receipt.get("published_bundle_verification")
    expected_published_rows = [
        {
            "tensor_key": row["tensor_key"],
            "tensor_sha256": row["tensor_sha256"],
            "shape": list(LATENT_SHAPE),
            "dtype": "torch.float32",
            "equals_prewrite_memory_tensor": True,
        }
        for row in rows
    ]
    if (
        not isinstance(published, Mapping)
        or published.get("serialized_payload_sha256") != expected_bundle_sha256
        or published.get("serialized_payload_size") != LATENT_BUNDLE_SIZE
        or published.get("metadata") != _expected_safetensors_metadata_v1()
        or published.get("exact16_keys_verified") is not True
        or published.get("all_tensors_reloaded_from_serialized_bytes") is not True
        or published.get("tensor_rows") != expected_published_rows
    ):
        fail("exact16 published reload verification differs")
    if set(retained) != {"source", "target", "role_swap"}:
        fail("exact16 local runtime retained tensor closure differs")
    return C2LatentBundleV1(
        local_row_index=local_row_index,
        local_row_id=ROW_IDS[local_row_index],
        local_tensors=retained,
        tensor_rows=tuple(dict(row) for row in rows),
        receipt=receipt,
        bundle_sha256=expected_bundle_sha256,
        receipt_sha256=expected_receipt_sha256,
        original_materialization_path=relocation_binding[
            "original_materialization_path"
        ],
        runtime_path=relocation_binding["runtime_path"],
        live_bundle_binding=dict(live_bundle_binding),
    )


def replay_model_authority_world8_v1(
    *,
    dist: Any,
    group: Any,
    rank: int,
    reference: Mapping[str, Any],
    authority_path: Path,
    expected_sha256: str,
    bernini_root: Path,
    checkpoint_root: Path,
    stage: str,
) -> Mapping[str, Any]:
    dist.barrier(group=group)
    box: list[Any] = [None]
    if rank == 0:
        try:
            candidate = validate_model_authority_v1(
                authority_path,
                expected_sha256=expected_sha256,
                bernini_root=bernini_root,
                checkpoint_root=checkpoint_root,
            )
            box[0] = {
                "ok": True,
                "value": candidate,
                "receipt": require_model_authority_replay_identity_v1(
                    reference, candidate, stage=stage
                ),
            }
        except Exception as error:
            box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(box, src=0, group=group)
    result = box[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"rank-zero C2 model authority replay failed: {result!r}")
    receipt = require_model_authority_replay_identity_v1(
        reference, result["value"], stage=stage
    )
    if receipt != result.get("receipt"):
        fail("C2 model authority broadcast replay receipt differs")
    dist.barrier(group=group)
    return receipt


def validate_runtime_sources_strong_v1(
    *,
    runner_sha256: str,
    runner_size: int,
    materializer_module: Any,
    label_module: Any,
    elal_module: Any,
    legacy_module: Any,
    packed_module: Any,
    runtime_module: Any,
    sigma_module: Any,
) -> Mapping[str, Any]:
    """Held-openat/no-follow validation of every imported local execution source."""

    rows = (
        ("c2_trainer", sys.modules[__name__], Path(__file__), runner_sha256, runner_size),
        (
            "c1_trainer",
            c1,
            METHOD_ROOT / "train_elal3_c1_simulator_overfit_v1.py",
            C1_TRAINER_SHA256,
            C1_TRAINER_SIZE,
        ),
        (
            "elal3_core",
            elal_module,
            METHOD_ROOT / "elal3_c0_v1.py",
            C1_CORE_SHA256,
            C1_CORE_SIZE,
        ),
        (
            "c2_label",
            label_module,
            METHOD_ROOT / "elal3_simulator_c2_label_v1.py",
            C2_LABEL_SHA256,
            C2_LABEL_SIZE,
        ),
        (
            "c2_materializer",
            materializer_module,
            METHOD_ROOT / "materialize_elal3_simulator_c2_vae_v1.py",
            C2_MATERIALIZER_SHA256,
            C2_MATERIALIZER_SIZE,
        ),
        (
            "train_lora",
            legacy_module,
            METHOD_ROOT / "train_lora.py",
            TRAIN_LORA_SHA256,
            TRAIN_LORA_SIZE,
        ),
        (
            "packed_lora",
            packed_module,
            METHOD_ROOT / "packed_preservation_lora_v2.py",
            PACKED_LORA_SHA256,
            PACKED_LORA_SIZE,
        ),
        (
            "world8_runtime",
            runtime_module,
            METHOD_ROOT / "source_self_runtime.py",
            RUNTIME_SHA256,
            RUNTIME_SIZE,
        ),
        (
            "sigma_strata",
            sigma_module,
            METHOD_ROOT / "inference_sigma_strata.py",
            SIGMA_SHA256,
            SIGMA_SIZE,
        ),
    )
    if runner_size <= 0:
        fail("runner source size differs")
    bindings: dict[str, Any] = {}
    for name, module, registered, sha, size in rows:
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            fail(f"runtime source module file is absent: {name}")
        imported = Path(module_file).resolve(strict=True)
        registered = registered.resolve(strict=True)
        if imported != registered:
            fail(f"runtime imported module path differs: {name}")
        try:
            binding = materializer_module.stable_stream_hash_path(
                registered,
                label=f"C2 trainer runtime source {name}",
                expected_sha256=sha,
                expected_size=size,
                expected_mode=0o444,
                allowed_root=METHOD_ROOT,
            )
        except materializer_module.ELAL3SimulatorC2VAEError as error:
            raise ELAL3C2TrainingError(str(error)) from error
        bindings[name] = {
            **binding,
            "actual_imported_module_file_verified": True,
        }
    callable_owners = {
        "materializer.validate_model_authority": (
            materializer_module.validate_model_authority,
            materializer_module,
        ),
        "label.load_oracle_q_label_v1": (
            label_module.load_oracle_q_label_v1,
            label_module,
        ),
        "label.build_role_only_hybrid_v1": (
            label_module.build_role_only_hybrid_v1,
            label_module,
        ),
        "legacy.validate_source_trees": (
            legacy_module.validate_source_trees,
            legacy_module,
        ),
        "packed.select_projection_specs": (
            packed_module.select_projection_specs,
            packed_module,
        ),
        "runtime.distributed_contract": (
            runtime_module.distributed_contract,
            runtime_module,
        ),
        "sigma.select_sigma_stratum": (
            sigma_module.select_sigma_stratum,
            sigma_module,
        ),
    }
    for label, (function, owner) in callable_owners.items():
        if (
            not callable(function)
            or getattr(function, "__module__", None) != owner.__name__
        ):
            fail(f"runtime callable ownership differs: {label}")
    result = {
        "source_count": len(bindings),
        "sources": bindings,
        "all_modes": "0444",
        "all_nlink1_no_follow_held_openat_double_hash": True,
        "actual_imported_module_files_verified": True,
        "callable_ownership_verified": True,
    }
    return {**result, "source_closure_digest": object_sha256(result)}


def validate_checkpoint_exact23_world8_v1(
    *,
    dist: Any,
    group: Any,
    rank: int,
    checkpoint_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    label_module: Any,
    materializer_module: Any,
    stage: str,
    reference: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Rank-zero held-openat exact23 hash, broadcast, and replay closure."""

    if stage not in {"pre_load", "post_deserialize", "final_pre_publish"}:
        fail("checkpoint exact23 replay stage differs")
    if expected_manifest_sha256 != CHECKPOINT_EXACT23_MANIFEST_SHA256:
        fail("checkpoint exact23 manifest CLI SHA differs from literal")
    dist.barrier(group=group)
    box: list[Any] = [None]
    if rank == 0:
        try:
            payload, manifest_live = label_module.stable_read_path(
                manifest_path,
                label=f"checkpoint exact23 manifest {stage}",
                expected_sha256=expected_manifest_sha256,
                expected_mode=0o444,
                allowed_root=manifest_path.parent,
            )
            if manifest_live.get("size") != CHECKPOINT_EXACT23_MANIFEST_SIZE:
                fail("checkpoint exact23 manifest size differs")
            try:
                text = payload.decode("ascii")
            except UnicodeDecodeError as error:
                raise ELAL3C2TrainingError(
                    "checkpoint exact23 manifest is not ASCII"
                ) from error
            if not text.endswith("\n") or "\r" in text:
                fail("checkpoint exact23 manifest newline ABI differs")
            lines = text[:-1].split("\n")
            if len(lines) != 23:
                fail("checkpoint exact23 manifest row count differs")
            registered = []
            for index, (line, relative) in enumerate(
                zip(lines, CHECKPOINT_EXACT23_RELATIVE_PATHS)
            ):
                expected_suffix = f"  ./{relative}"
                sha = line[:64]
                if (
                    len(line) != 64 + len(expected_suffix)
                    or line[64:] != expected_suffix
                ):
                    fail(f"checkpoint exact23 manifest row differs: {index}")
                _require_sha256_v1(sha, label=f"checkpoint exact23 row {index}")
                registered.append((relative, sha))
            expected_file_set = set(CHECKPOINT_EXACT23_RELATIVE_PATHS)
            expected_directory_set = {
                "assets",
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "vae",
            }

            def scan_noncache_tree() -> Mapping[str, Any]:
                files: set[str] = set()
                directories: set[str] = set()
                for current, dir_names, file_names in os.walk(
                    checkpoint_root, topdown=True, followlinks=False
                ):
                    current_path = Path(current)
                    current_relative = current_path.relative_to(checkpoint_root)
                    if current_relative == Path("."):
                        if ".cache" in dir_names:
                            cache = current_path / ".cache"
                            cache_info = cache.lstat()
                            if cache.is_symlink() or not stat.S_ISDIR(cache_info.st_mode):
                                fail("checkpoint canonical .cache entry differs")
                            dir_names.remove(".cache")
                    for name in tuple(dir_names):
                        item = current_path / name
                        info = item.lstat()
                        if item.is_symlink() or not stat.S_ISDIR(info.st_mode):
                            fail("checkpoint non-cache directory entry differs")
                        directories.add(
                            str(item.relative_to(checkpoint_root)).replace(os.sep, "/")
                        )
                    for name in file_names:
                        item = current_path / name
                        info = item.lstat()
                        if item.is_symlink() or not stat.S_ISREG(info.st_mode):
                            fail("checkpoint non-cache file entry differs")
                        files.add(
                            str(item.relative_to(checkpoint_root)).replace(os.sep, "/")
                        )
                if files != expected_file_set or directories != expected_directory_set:
                    fail("checkpoint exact23 non-cache load-precedence closure differs")
                value = {
                    "noncache_file_count": len(files),
                    "noncache_files": sorted(files),
                    "noncache_directory_count": len(directories),
                    "noncache_directories": sorted(directories),
                    "canonical_dot_cache_only_exclusion": True,
                    "noncache_symlinks_rejected": True,
                }
                return {**value, "closure_digest": object_sha256(value)}

            tree_before = scan_noncache_tree()
            runtime_rows = []
            fixed_rows = []
            for index, (relative, sha) in enumerate(registered):
                try:
                    live = materializer_module.stable_stream_hash_path(
                        checkpoint_root / relative,
                        label=f"checkpoint exact23 {stage} row {index}",
                        expected_sha256=sha,
                        expected_size=None,
                        expected_mode=0o644,
                        allowed_root=checkpoint_root,
                    )
                except materializer_module.ELAL3SimulatorC2VAEError as error:
                    raise ELAL3C2TrainingError(str(error)) from error
                runtime_rows.append({"row_index": index, "relative_path": relative, **live})
                fixed_rows.append(
                    {
                        "row_index": index,
                        "relative_path": relative,
                        "sha256": live["sha256"],
                        "size": live["size"],
                        "mode": live["mode"],
                        "nlink": live["nlink"],
                        "held_fd_double_hash_verified": live[
                            "held_fd_double_hash_verified"
                        ],
                        "held_openat_parent_chain_replayed": live[
                            "held_openat_parent_chain_replayed"
                        ],
                    }
                )
            tree_after = scan_noncache_tree()
            if tree_before != tree_after:
                fail("checkpoint exact23 non-cache tree changed during hash")
            fixed = {
                "manifest_relative_path": (
                    "audits/bernini_r13_ff4c5d4_checkpoint.sha256"
                ),
                "manifest_sha256": expected_manifest_sha256,
                "manifest_size": CHECKPOINT_EXACT23_MANIFEST_SIZE,
                "file_count": 23,
                "files": fixed_rows,
                "noncache_load_precedence_closure": tree_before,
                "checkpoint_root_expected_by_renderer_and_tokenizer": True,
            }
            result = {
                "stage": stage,
                "fixed_release_binding": fixed,
                "fixed_release_binding_digest": object_sha256(fixed),
                "runtime_telemetry": {
                    "manifest_live_binding": manifest_live,
                    "checkpoint_file_bindings": runtime_rows,
                },
            }
            if reference is not None and (
                result["fixed_release_binding"]
                != reference.get("fixed_release_binding")
                or result["fixed_release_binding_digest"]
                != reference.get("fixed_release_binding_digest")
            ):
                fail(f"checkpoint exact23 {stage} differs from pre-load bytes")
            box[0] = {"ok": True, "value": result}
        except Exception as error:
            box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(box, src=0, group=group)
    if not isinstance(box[0], Mapping) or box[0].get("ok") is not True:
        fail(f"rank-zero checkpoint exact23 validation failed: {box[0]!r}")
    result = box[0]["value"]
    if (
        result.get("stage") != stage
        or result.get("fixed_release_binding_digest")
        != object_sha256(result.get("fixed_release_binding"))
        or (
            reference is not None
            and result.get("fixed_release_binding_digest")
            != reference.get("fixed_release_binding_digest")
        )
    ):
        fail("broadcast checkpoint exact23 receipt differs")
    dist.barrier(group=group)
    return result


def validate_bernini_execution_sources_world8_v1(
    *,
    dist: Any,
    group: Any,
    rank: int,
    bernini_root: Path,
    veomni_root: Path,
    legacy_module: Any,
    materializer_module: Any,
    renderer_module: Any,
    transformer_wan_module: Any,
    parallel_module: Any,
    parallel_state_module: Any,
    veomni_parallel_state_module: Any,
    veomni_sequence_comm_module: Any,
    renderer_config_class: Any,
    renderer_model_class: Any,
    rotary_class: Any,
    init_parallel_function: Any,
    stage: str,
    reference: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Bind actual imported Bernini execution modules to held official bytes."""

    if stage not in {"pre_load", "post_deserialize", "final_pre_publish"}:
        fail("Bernini execution source replay stage differs")
    expected = dict(legacy_module.BERNINI_PINNED_FILE_HASHES)
    expected["bernini/parallel/__init__.py"] = BERNINI_PARALLEL_INIT_SHA256
    if len(expected) != 10:
        fail("Bernini execution source exact10 registry differs")
    module_rows = (
        (renderer_module, "bernini/models/renderer.py"),
        (transformer_wan_module, "bernini/models/transformer_wan.py"),
        (parallel_module, "bernini/parallel/__init__.py"),
        (parallel_state_module, "bernini/parallel/state.py"),
    )
    for module, relative in module_rows:
        module_file = getattr(module, "__file__", None)
        if (
            type(module_file) is not str
            or Path(module_file).resolve(strict=True)
            != (bernini_root / relative).resolve(strict=True)
        ):
            fail(f"actual Bernini imported module path differs: {relative}")
    veomni_module_rows = (
        (
            veomni_parallel_state_module,
            "veomni/distributed/parallel_state.py",
        ),
        (
            veomni_sequence_comm_module,
            "veomni/distributed/sequence_parallel/comm.py",
        ),
    )
    for module, relative in veomni_module_rows:
        module_file = getattr(module, "__file__", None)
        if (
            module.__name__ not in sys.modules
            or type(module_file) is not str
            or Path(module_file).resolve(strict=True)
            != (veomni_root / relative).resolve(strict=True)
            or Path(module_file).is_symlink()
            or not Path(module_file).is_file()
        ):
            fail(f"actual VeOmni imported module path differs: {relative}")
    callable_rows = (
        (renderer_config_class, renderer_module.__name__),
        (renderer_model_class, renderer_module.__name__),
        (rotary_class, transformer_wan_module.__name__),
        (init_parallel_function, parallel_state_module.__name__),
    )
    if any(
        not callable(value) or getattr(value, "__module__", None) != owner
        for value, owner in callable_rows
    ):
        fail("actual Bernini class/function ownership differs")
    dist.barrier(group=group)
    box: list[Any] = [None]
    if rank == 0:
        try:
            try:
                roots = legacy_module.validate_source_trees(
                    bernini_root,
                    veomni_root,
                    expected_bernini_commit=BERNINI_COMMIT,
                    expected_veomni_commit=VEOMNI_COMMIT,
                )
            except legacy_module.TrainingContractError as error:
                raise ELAL3C2TrainingError(str(error)) from error
            fixed_rows = []
            runtime_rows = []
            for index, (relative, sha) in enumerate(expected.items()):
                expected_size = (
                    BERNINI_PARALLEL_INIT_SIZE
                    if relative == "bernini/parallel/__init__.py"
                    else None
                )
                try:
                    live = materializer_module.stable_stream_hash_path(
                        bernini_root / relative,
                        label=f"Bernini execution {stage} row {index}",
                        expected_sha256=sha,
                        expected_size=expected_size,
                        expected_mode=0o444,
                        allowed_root=bernini_root,
                    )
                except materializer_module.ELAL3SimulatorC2VAEError as error:
                    raise ELAL3C2TrainingError(str(error)) from error
                runtime_rows.append({"row_index": index, "relative_path": relative, **live})
                fixed_rows.append(
                    {
                        "row_index": index,
                        "relative_path": relative,
                        "sha256": live["sha256"],
                        "size": live["size"],
                        "mode": live["mode"],
                        "nlink": live["nlink"],
                        "held_fd_double_hash_verified": True,
                        "held_openat_parent_chain_replayed": True,
                    }
                )
            veomni_fixed_rows = []
            veomni_runtime_rows = []
            for index, (_, relative) in enumerate(veomni_module_rows):
                try:
                    live = materializer_module.stable_stream_hash_path(
                        veomni_root / relative,
                        label=f"VeOmni execution {stage} row {index}",
                        expected_sha256=None,
                        expected_size=None,
                        expected_mode=0o644,
                        allowed_root=veomni_root,
                    )
                except materializer_module.ELAL3SimulatorC2VAEError as error:
                    raise ELAL3C2TrainingError(str(error)) from error
                veomni_runtime_rows.append(
                    {"row_index": index, "relative_path": relative, **live}
                )
                veomni_fixed_rows.append(
                    {
                        "row_index": index,
                        "relative_path": relative,
                        "sha256": live["sha256"],
                        "size": live["size"],
                        "mode": live["mode"],
                        "nlink": live["nlink"],
                        "held_fd_double_hash_verified": True,
                        "held_openat_parent_chain_replayed": True,
                        "actual_imported_module_file_verified": True,
                    }
                )
            fixed = {
                "bernini_commit": roots[2],
                "veomni_commit": roots[3],
                "file_count": 10,
                "files": fixed_rows,
                "veomni_actual_imported_module_count": 2,
                "veomni_actual_imported_modules": veomni_fixed_rows,
                "actual_imported_modules_and_callable_ownership_verified": True,
            }
            result = {
                "stage": stage,
                "fixed_release_binding": fixed,
                "fixed_release_binding_digest": object_sha256(fixed),
                "runtime_telemetry": {
                    "bernini": runtime_rows,
                    "veomni": veomni_runtime_rows,
                },
            }
            if reference is not None and (
                result["fixed_release_binding"]
                != reference.get("fixed_release_binding")
                or result["fixed_release_binding_digest"]
                != reference.get("fixed_release_binding_digest")
            ):
                fail(f"Bernini execution source {stage} differs from pre-load")
            box[0] = {"ok": True, "value": result}
        except Exception as error:
            box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(box, src=0, group=group)
    if not isinstance(box[0], Mapping) or box[0].get("ok") is not True:
        fail(f"rank-zero Bernini execution source replay failed: {box[0]!r}")
    result = box[0]["value"]
    if (
        result.get("stage") != stage
        or result.get("fixed_release_binding_digest")
        != object_sha256(result.get("fixed_release_binding"))
        or (
            reference is not None
            and result.get("fixed_release_binding_digest")
            != reference.get("fixed_release_binding_digest")
        )
    ):
        fail("broadcast Bernini execution source receipt differs")
    dist.barrier(group=group)
    return result


def controlled_gain_v1() -> float:
    computed = 1.0 / (BLOCKS * math.sqrt(HIDDEN))
    bits = struct.pack(">f", computed).hex()
    if bits != CONTROLLED_GAIN_FLOAT32_HEX or CONTROLLED_GAIN <= 0.0:
        fail("controlled non-zero gate formula/bits differ")
    return CONTROLLED_GAIN


def install_controlled_nonzero_gates_v1(handle: Any) -> Mapping[str, Any]:
    import torch

    injections = tuple(getattr(getattr(handle, "components", None), "injections", ()))
    if len(injections) != BLOCKS:
        fail("controlled gate requires exact30 ELAL injections")
    gain = controlled_gain_v1()
    parameter_ids: set[int] = set()
    with torch.no_grad():
        for index, injection in enumerate(injections):
            parameter = getattr(injection, "residual_gain", None)
            if (
                not isinstance(parameter, torch.nn.Parameter)
                or parameter.numel() != 1
                or parameter.dtype != torch.float32
                or id(parameter) in parameter_ids
            ):
                fail(f"ELAL residual gain ABI differs at block {index}")
            parameter.fill_(gain)
            parameter_ids.add(id(parameter))
    bits = [
        struct.pack(">f", float(injection.residual_gain.detach().cpu().item())).hex()
        for injection in injections
    ]
    if bits != [CONTROLLED_GAIN_FLOAT32_HEX] * BLOCKS:
        fail("controlled non-zero gain installation differs")
    return {
        "formula": CONTROLLED_GAIN_FORMULA,
        "float32_be_hex": CONTROLLED_GAIN_FLOAT32_HEX,
        "value": gain,
        "count": BLOCKS,
        "all_small_nonzero": True,
        "adaptive_gain_search": False,
        "other_projection_initialization_changed": False,
    }


@contextmanager
def temporary_gate_zero_probe_v1(handle: Any) -> Iterator[None]:
    """Temporarily bypass ELAL injection and restore every scalar bit-exactly."""

    import torch

    injections = tuple(getattr(getattr(handle, "components", None), "injections", ()))
    if len(injections) != BLOCKS:
        fail("gate-zero probe requires exact30 injections")
    saved = [item.residual_gain.detach().clone() for item in injections]
    try:
        with torch.no_grad():
            for item in injections:
                item.residual_gain.zero_()
        yield
    finally:
        with torch.no_grad():
            for item, value in zip(injections, saved):
                item.residual_gain.copy_(value)
        if any(
            not torch.equal(
                item.residual_gain.detach().reshape(-1).view(torch.uint8),
                value.reshape(-1).view(torch.uint8),
            )
            for item, value in zip(injections, saved)
        ):
            fail("gate-zero probe failed to restore parameter bytes")


def step0_gain_safety_receipt_v1(
    gate_zero_prediction: Any,
    controlled_prediction: Any,
    *,
    parameter_digest_before: str,
    parameter_digest_after: str,
) -> Mapping[str, Any]:
    import torch

    if (
        not isinstance(gate_zero_prediction, torch.Tensor)
        or not isinstance(controlled_prediction, torch.Tensor)
        or gate_zero_prediction.shape != controlled_prediction.shape
        or gate_zero_prediction.numel() == 0
        or not bool(torch.isfinite(gate_zero_prediction).all().item())
        or not bool(torch.isfinite(controlled_prediction).all().item())
        or parameter_digest_before != parameter_digest_after
    ):
        fail("step0 gain probe tensors or parameter restoration differ")
    baseline_norm = torch.linalg.vector_norm(gate_zero_prediction.float())
    delta = controlled_prediction.float() - gate_zero_prediction.float()
    delta_norm = torch.linalg.vector_norm(delta)
    if float(baseline_norm.item()) <= 0.0 or float(delta_norm.item()) <= 0.0:
        fail("step0 controlled action delta must be finite non-zero")
    ratio = float((delta_norm / baseline_norm).item())
    if not math.isfinite(ratio) or not 0.0 < ratio < STEP0_RATIO_BOUND:
        fail("step0 controlled action delta exceeds the preregistered bound")
    return {
        "gain_float32_be_hex": CONTROLLED_GAIN_FLOAT32_HEX,
        "gate_zero_prediction_norm": float(baseline_norm.item()),
        "controlled_delta_norm": float(delta_norm.item()),
        "relative_delta_ratio": ratio,
        "strict_upper_bound": STEP0_RATIO_BOUND,
        "finite_nonzero_bounded": True,
        "parameter_bytes_restored_after_probe": True,
        "semantic_diagonal_dominance_required": False,
        "adaptive_gain_search": False,
    }


PREDICTION_HASH_PROJECTION_FIELDS = frozenset(
    {
        "schema_version",
        "original_dtype",
        "original_shape",
        "original_device_type",
        "original_device_index",
        "original_layout",
        "original_stride",
        "original_storage_offset",
        "original_is_contiguous",
        "original_requires_grad",
        "projection_operation",
        "projection_dtype",
        "projection_shape",
        "projection_device_type",
        "projection_device_index",
        "projection_layout",
        "projection_stride",
        "projection_storage_offset",
        "projection_is_contiguous",
        "projection_requires_grad",
        "prediction_sha256",
        "audit_projection_only",
        "forward_output_tensor_not_replaced",
        "original_tensor_identity_shape_stride_version_unchanged",
        "projection_receipt_digest",
    }
)
PREDICTION_HASH_PROJECTION_ALLOWED_DTYPES = frozenset(
    {"torch.bfloat16", "torch.float32"}
)
PREDICTION_HASH_PROJECTION_PRODUCTION_DTYPE = "torch.bfloat16"
PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE = (
    2 * 19_110 * c1.PATCH_VALUES,
    c1.PATCH_VALUES,
    1,
)
PREDICTION_HASH_PROJECTION_PRODUCTION_STORAGE_OFFSET = 0
PREDICTION_HASH_PROJECTION_DIAGNOSTIC_FIELDS = (
    "original_dtype",
    "original_shape",
    "original_device_type",
    "original_device_index",
    "original_layout",
    "original_stride",
    "original_storage_offset",
    "original_is_contiguous",
    "original_requires_grad",
    "projection_dtype",
    "projection_shape",
    "projection_device_type",
    "projection_device_index",
    "projection_layout",
    "projection_stride",
    "projection_storage_offset",
    "projection_is_contiguous",
    "projection_requires_grad",
)


def prediction_hash_projection_failure_metadata_v1(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return bounded tensor metadata only; never include data, pointers, or identity."""

    enum_values = {
        "original_dtype": PREDICTION_HASH_PROJECTION_ALLOWED_DTYPES,
        "projection_dtype": PREDICTION_HASH_PROJECTION_ALLOWED_DTYPES,
        "original_device_type": frozenset({"cpu", "cuda"}),
        "projection_device_type": frozenset({"cpu", "cuda"}),
        "original_layout": frozenset({"torch.strided"}),
        "projection_layout": frozenset({"torch.strided"}),
    }
    sequence_fields = {
        "original_shape",
        "original_stride",
        "projection_shape",
        "projection_stride",
    }
    optional_integer_fields = {"original_device_index", "projection_device_index"}
    integer_fields = {"original_storage_offset", "projection_storage_offset"}
    boolean_fields = {
        "original_is_contiguous",
        "original_requires_grad",
        "projection_is_contiguous",
        "projection_requires_grad",
    }
    result: dict[str, Any] = {}
    for key in PREDICTION_HASH_PROJECTION_DIAGNOSTIC_FIELDS:
        value = receipt.get(key)
        if key in enum_values:
            result[key] = value if value in enum_values[key] else "<invalid>"
        elif key in sequence_fields:
            result[key] = (
                list(value)
                if isinstance(value, list)
                and 1 <= len(value) <= 8
                and all(type(item) is int and 0 <= item < 2**63 for item in value)
                else "<invalid>"
            )
        elif key in optional_integer_fields:
            result[key] = (
                value
                if value is None or (type(value) is int and 0 <= value < 65_536)
                else "<invalid>"
            )
        elif key in integer_fields:
            result[key] = (
                value
                if type(value) is int and 0 <= value < 2**63
                else "<invalid>"
            )
        elif key in boolean_fields:
            result[key] = value if type(value) is bool else "<invalid>"
        else:  # pragma: no cover - tuple and classifier are frozen together.
            fail("prediction hash projection diagnostic classifier differs")
    if tuple(result) != PREDICTION_HASH_PROJECTION_DIAGNOSTIC_FIELDS:
        fail("prediction hash projection diagnostic key closure differs")
    return result


def validate_prediction_hash_projection_receipt_v1(
    receipt: Any,
    *,
    expected_prediction_sha256: Optional[str] = None,
    expected_original_device_type: Optional[str] = None,
    expected_original_device_index: Optional[int] = None,
    expected_original_dtype: Optional[str] = None,
    expected_original_stride: Optional[Sequence[int]] = None,
    expected_original_storage_offset: Optional[int] = None,
    expected_original_requires_grad: Optional[bool] = None,
    expected_original_is_contiguous: Optional[bool] = None,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != PREDICTION_HASH_PROJECTION_FIELDS:
        fail(f"{label} prediction hash projection field closure differs")
    unsigned = dict(receipt)
    digest = unsigned.pop("projection_receipt_digest", None)
    shape = receipt.get("original_shape")
    stride = receipt.get("original_stride")
    projection_shape = receipt.get("projection_shape")
    projection_stride = receipt.get("projection_stride")
    device_type = receipt.get("original_device_type")
    device_index = receipt.get("original_device_index")
    canonical_stride: list[int] = []
    running = 1
    if isinstance(shape, list):
        canonical_stride = [0] * len(shape)
        for index in range(len(shape) - 1, -1, -1):
            canonical_stride[index] = running
            if type(shape[index]) is int:
                running *= shape[index]
    logical_original_contiguous = True
    running = 1
    if isinstance(shape, list) and isinstance(stride, list) and len(shape) == len(stride):
        for size, item_stride in zip(reversed(shape), reversed(stride)):
            if size != 1:
                if item_stride != running:
                    logical_original_contiguous = False
                    break
                running *= size
    else:
        logical_original_contiguous = False
    if (
        receipt.get("schema_version")
        != "bernini-elal3-c2-eval-prediction-hash-projection-v1"
        or not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
        or not isinstance(stride, list)
        or len(stride) != len(shape)
        or any(type(item) is not int or item < 0 for item in stride)
        or projection_shape != shape
        or shape != [1, 19_110, c1.PATCH_VALUES]
        or not isinstance(projection_stride, list)
        or len(projection_stride) != len(shape)
        or any(type(item) is not int or item < 0 for item in projection_stride)
        or device_type not in {"cpu", "cuda"}
        or (device_type == "cpu" and device_index is not None)
        or (device_type == "cuda" and (type(device_index) is not int or device_index < 0))
        or (
            expected_original_device_type is not None
            and device_type != expected_original_device_type
        )
        or (
            expected_original_device_index is not None
            and device_index != expected_original_device_index
        )
        or receipt.get("original_dtype")
        not in PREDICTION_HASH_PROJECTION_ALLOWED_DTYPES
        or receipt.get("projection_dtype") != receipt.get("original_dtype")
        or (
            expected_original_dtype is not None
            and receipt.get("original_dtype") != expected_original_dtype
        )
        or type(receipt.get("original_is_contiguous")) is not bool
        or receipt.get("original_is_contiguous") is not logical_original_contiguous
        or (
            expected_original_is_contiguous is not None
            and receipt.get("original_is_contiguous")
            is not expected_original_is_contiguous
        )
        or type(receipt.get("original_requires_grad")) is not bool
        or (
            expected_original_requires_grad is not None
            and receipt.get("original_requires_grad")
            is not expected_original_requires_grad
        )
        or receipt.get("original_layout") != "torch.strided"
        or type(receipt.get("original_storage_offset")) is not int
        or receipt.get("original_storage_offset") < 0
        or (
            expected_original_stride is not None
            and stride != list(expected_original_stride)
        )
        or (
            expected_original_storage_offset is not None
            and receipt.get("original_storage_offset")
            != expected_original_storage_offset
        )
        or receipt.get("projection_operation")
        != "detach_then_to_cpu_then_explicit_dense_copy_audit_only"
        or receipt.get("projection_device_type") != "cpu"
        or receipt.get("projection_device_index") is not None
        or receipt.get("projection_layout") != "torch.strided"
        or projection_stride != canonical_stride
        or receipt.get("projection_storage_offset") != 0
        or receipt.get("projection_is_contiguous") is not True
        or receipt.get("projection_requires_grad") is not False
        or receipt.get("audit_projection_only") is not True
        or receipt.get("forward_output_tensor_not_replaced") is not True
        or receipt.get("original_tensor_identity_shape_stride_version_unchanged") is not True
        or digest != object_sha256(unsigned)
    ):
        diagnostic = prediction_hash_projection_failure_metadata_v1(receipt)
        fail(
            f"{label} prediction hash projection semantic closure differs; "
            f"bounded_metadata={canonical_json_bytes(diagnostic).decode('ascii')}"
        )
    prediction_sha = _require_sha256_v1(
        receipt.get("prediction_sha256"), label=f"{label} prediction SHA"
    )
    if expected_prediction_sha256 is not None and prediction_sha != _require_sha256_v1(
        expected_prediction_sha256, label=f"{label} expected prediction SHA"
    ):
        fail(f"{label} prediction hash projection SHA join differs")
    return receipt


def prediction_hash_projection_v1(value: Any, *, label: str) -> Mapping[str, Any]:
    """Hash a canonical audit copy without replacing/mutating model output."""

    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or str(value.dtype) not in PREDICTION_HASH_PROJECTION_ALLOWED_DTYPES
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all().item())
    ):
        fail(f"{label} prediction tensor ABI differs")
    before = {
        "identity": id(value),
        "data_ptr": value.untyped_storage().data_ptr(),
        "version": value._version,
        "shape": tuple(int(item) for item in value.shape),
        "stride": tuple(int(item) for item in value.stride()),
        "storage_offset": int(value.storage_offset()),
        "device": value.device,
        "dtype": value.dtype,
        "layout": value.layout,
        "requires_grad": bool(value.requires_grad),
    }
    detached_cpu = value.detach().to(device="cpu")
    projected = torch.empty(
        before["shape"],
        dtype=detached_cpu.dtype,
        device="cpu",
        memory_format=torch.contiguous_format,
    )
    projected.copy_(detached_cpu)
    if (
        projected.device.type != "cpu"
        or projected.device.index is not None
        or projected.dtype != value.dtype
        or projected.layout != torch.strided
        or tuple(int(item) for item in projected.shape) != before["shape"]
        or not projected.is_contiguous()
        or projected.requires_grad
        or int(projected.storage_offset()) != 0
    ):
        fail(f"{label} canonical CPU contiguous audit projection differs")
    prediction_sha = c1.tensor_sha256_v1(projected)
    unchanged = (
        id(value) == before["identity"]
        and value.untyped_storage().data_ptr() == before["data_ptr"]
        and value._version == before["version"]
        and tuple(int(item) for item in value.shape) == before["shape"]
        and tuple(int(item) for item in value.stride()) == before["stride"]
        and int(value.storage_offset()) == before["storage_offset"]
        and value.device == before["device"]
        and value.dtype == before["dtype"]
        and value.layout == before["layout"]
        and bool(value.requires_grad) == before["requires_grad"]
    )
    unsigned = {
        "schema_version": "bernini-elal3-c2-eval-prediction-hash-projection-v1",
        "original_dtype": str(before["dtype"]),
        "original_shape": list(before["shape"]),
        "original_device_type": before["device"].type,
        "original_device_index": before["device"].index,
        "original_layout": str(before["layout"]),
        "original_stride": list(before["stride"]),
        "original_storage_offset": before["storage_offset"],
        "original_is_contiguous": bool(value.is_contiguous()),
        "original_requires_grad": before["requires_grad"],
        "projection_operation": "detach_then_to_cpu_then_explicit_dense_copy_audit_only",
        "projection_dtype": str(projected.dtype),
        "projection_shape": [int(item) for item in projected.shape],
        "projection_device_type": projected.device.type,
        "projection_device_index": projected.device.index,
        "projection_layout": str(projected.layout),
        "projection_stride": [int(item) for item in projected.stride()],
        "projection_storage_offset": int(projected.storage_offset()),
        "projection_is_contiguous": bool(projected.is_contiguous()),
        "projection_requires_grad": bool(projected.requires_grad),
        "prediction_sha256": prediction_sha,
        "audit_projection_only": True,
        "forward_output_tensor_not_replaced": True,
        "original_tensor_identity_shape_stride_version_unchanged": unchanged,
    }
    receipt = {**unsigned, "projection_receipt_digest": object_sha256(unsigned)}
    return validate_prediction_hash_projection_receipt_v1(
        receipt,
        expected_prediction_sha256=prediction_sha,
        expected_original_device_type=value.device.type,
        expected_original_device_index=value.device.index,
        expected_original_dtype=str(value.dtype),
        expected_original_stride=before["stride"],
        expected_original_storage_offset=before["storage_offset"],
        expected_original_requires_grad=before["requires_grad"],
        label=label,
    )


def prediction_hash_projection_consensus_view_v1(
    receipt: Mapping[str, Any],
    *,
    expected_prediction_sha256: str,
    expected_original_device_index: int,
    label: str,
) -> Mapping[str, Any]:
    validate_prediction_hash_projection_receipt_v1(
        receipt,
        expected_prediction_sha256=expected_prediction_sha256,
        expected_original_device_type="cuda",
        expected_original_device_index=expected_original_device_index,
        expected_original_dtype=PREDICTION_HASH_PROJECTION_PRODUCTION_DTYPE,
        expected_original_stride=PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE,
        expected_original_storage_offset=(
            PREDICTION_HASH_PROJECTION_PRODUCTION_STORAGE_OFFSET
        ),
        expected_original_requires_grad=False,
        expected_original_is_contiguous=True,
        label=label,
    )
    return {
        key: value
        for key, value in receipt.items()
        if key not in {"original_device_index", "projection_receipt_digest"}
    }


def _partition_energy_v1(
    prediction: Any, target_velocity: Any, event_mask: Any, context_mask: Any
) -> tuple[Any, Mapping[str, Any]]:
    try:
        return c1.partitioned_flow_matching_loss_v1(
            prediction, target_velocity, event_mask, context_mask
        )
    except c1.ELAL3C1TrainingError as error:
        raise ELAL3C2TrainingError(str(error)) from error


def text_lens_runtime_list_abi_v1(text_lens: Any) -> Mapping[str, Any]:
    """Bind Bernini's real text-length ABI without changing shared_step input.

    The pinned renderer returns ``[self.max_sequence_length] * len(input_lens)``.
    This experiment is exact B=1 and max_sequence_length=512, hence the only
    accepted runtime value is the built-in Python list ``[512]``.  A CPU int64
    tensor is materialized solely for the legacy byte digest; it is never
    passed to ``shared_step``.
    """

    import torch

    if (
        type(text_lens) is not list
        or len(text_lens) != 1
        or any(type(item) is not int for item in text_lens)
        or any(item < 1 or item > 512 for item in text_lens)
        or text_lens != [512]
    ):
        fail("C2 runtime text_lens must be exact built-in Python list [512]")
    typed = {
        "schema_version": "bernini-elal3-c2-text-lens-python-list-abi-v1",
        "container": "python_list",
        "length": 1,
        "element_type": "python_int",
        "values": list(text_lens),
        "allowed_value_range_inclusive": [1, 512],
        "exact_runtime_value_required": [512],
    }
    tensor = torch.tensor(text_lens, dtype=torch.int64, device="cpu").contiguous()
    if tensor.tolist() != text_lens or tuple(tensor.shape) != (1,):
        fail("C2 text_lens audit-only CPU int64 materialization differs")
    return {
        **typed,
        "canonical_typed_list_digest": object_sha256(typed),
        "audit_only_cpu_int64_tensor_sha256": c1.tensor_sha256_v1(tensor),
        "shared_step_received_original_python_list": True,
        "tensor_substitution_into_shared_step": False,
    }


def predict_target_c2_v1(
    *,
    renderer: Any,
    packed: Mapping[str, Any],
    coordinate: Any,
    text_lens: Any,
    text_embs: Any,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Call Bernini with a receipt-visible CPU int64 timestep origin."""

    import torch

    text_lens_before = text_lens_runtime_list_abi_v1(text_lens)
    original_text_lens_identity = id(text_lens)
    timestep_cpu = torch.tensor(
        [int(coordinate.timestep)], dtype=torch.int64, device="cpu"
    )
    if (
        timestep_cpu.device.type != "cpu"
        or timestep_cpu.dtype != torch.int64
        or tuple(timestep_cpu.shape) != (1,)
    ):
        fail("renderer timestep CPU/int64 ABI differs")
    rotary = packed["rotary"].permute(1, 0, 2).unsqueeze(0)
    value = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=packed["embedded"],
        timesteps=timestep_cpu.to(device=packed["embedded"].device),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[packed["total_tokens"]],
        batch_text_seqlen=text_lens,
    )
    text_lens_after = text_lens_runtime_list_abi_v1(text_lens)
    if (
        id(text_lens) != original_text_lens_identity
        or text_lens_after != text_lens_before
    ):
        fail("renderer mutated/replaced the C2 runtime text_lens list")
    target = value[:, packed["source_tokens"] :, :]
    if tuple(target.shape) != (1, packed["target_tokens"], c1.PATCH_VALUES):
        fail("official Bernini C2 target prediction geometry differs")
    return (
        target,
        {
            "timestep_cpu_origin": True,
            "timestep_dtype": "torch.int64",
            "timestep_value": int(timestep_cpu.item()),
            "sigma_float32_be_hex": struct.pack(">f", float(coordinate.sigma)).hex(),
        },
        text_lens_before,
    )


ACTUAL_BRANCH_RECEIPT_FIELDS = frozenset(
    {
        "row_id",
        "input_variant",
        "label_binding_digest",
        "actual_q_tensor_rows",
        "actual_q_tensor_rows_digest",
        "source_sha256",
        "clean_target_sha256",
        "epsilon_sha256",
        "noisy_target_sha256",
        "target_velocity_sha256",
        "event_mask_vae_sha256",
        "context_mask_vae_sha256",
        "text_lens_runtime_abi",
        "text_lens_sha256",
        "text_embs_sha256",
        "coordinate",
        "coordinate_kind",
        "renderer_timestep_receipt",
        "route_identity",
        "registered_sp4_partition",
        "all30_hooks_used",
        "prediction_sha256",
        "packed_target_velocity_sha256",
        "packed_event_mask_sha256",
        "packed_context_mask_sha256",
        "actual_input_digest",
    }
)
ACTUAL_Q_FIELDS = (
    "q_local",
    "q_entity",
    "q_relation",
    "q_phase",
    "q_terminal",
    "q_camera",
    "entity_presence",
    "temporal_valid",
    "relation_valid",
    "phase_valid",
)
ACTUAL_Q_ABI = {
    "q_local": ([1, 21, 26, 35, 64], "torch.float32"),
    "q_entity": ([1, 3, 21, 256], "torch.float32"),
    "q_relation": ([1, 6, 21, 128], "torch.float32"),
    "q_phase": ([1, 21, 128], "torch.float32"),
    "q_terminal": ([1, 9, 256], "torch.float32"),
    "q_camera": ([1, 21, 128], "torch.float32"),
    "entity_presence": ([1, 3], "torch.bool"),
    "temporal_valid": ([1, 3, 21], "torch.bool"),
    "relation_valid": ([1, 6, 21], "torch.bool"),
    "phase_valid": ([1, 21], "torch.bool"),
}


def _validate_actual_branch_receipt_closed_v1(
    value: Any, *, label: str
) -> Mapping[str, Any]:
    """Validate bytes emitted by one renderer forward, never caller decoys."""

    if not isinstance(value, Mapping) or set(value) != ACTUAL_BRANCH_RECEIPT_FIELDS:
        fail(f"{label} actual branch receipt field closure differs")
    unsigned = dict(value)
    digest = unsigned.pop("actual_input_digest", None)
    q_rows = value.get("actual_q_tensor_rows")
    coordinate = value.get("coordinate")
    coordinate_kind = value.get("coordinate_kind")
    timestep_receipt = value.get("renderer_timestep_receipt")
    partition = value.get("registered_sp4_partition")
    if (
        digest != object_sha256(unsigned)
        or value.get("actual_q_tensor_rows_digest") != object_sha256(q_rows)
        or value.get("row_id") not in ROW_IDS
        or value.get("input_variant") not in {
            "target",
            "role_swap",
            "target_role_mismatch",
            "role_swap_role_mismatch",
        }
        or type(value.get("route_identity")) is not str
        or not value.get("route_identity")
        or value.get("all30_hooks_used") is not True
        or not isinstance(q_rows, Mapping)
        # Canonical JSON sorts mapping keys, so semantic exact10 replay must
        # close the exact field set rather than depend on transient insertion
        # order from the live Python producer.
        or set(q_rows) != set(ACTUAL_Q_FIELDS)
        or coordinate_kind not in {"training_sigma_stratum", "evaluation_sigma1"}
        or not isinstance(coordinate, Mapping)
        or not isinstance(timestep_receipt, Mapping)
        or timestep_receipt
        != {
            "timestep_cpu_origin": True,
            "timestep_dtype": "torch.int64",
            "timestep_value": coordinate.get("timestep"),
            "sigma_float32_be_hex": coordinate.get("sigma_float32_be_hex"),
        }
        or not isinstance(partition, Mapping)
        or partition.get("sp_rank") not in range(SP_SIZE)
        or value.get("text_lens_runtime_abi")
        != text_lens_runtime_list_abi_v1([512])
        or value.get("text_lens_sha256")
        != value["text_lens_runtime_abi"].get(
            "audit_only_cpu_int64_tensor_sha256"
        )
    ):
        fail(f"{label} actual branch receipt semantic closure differs")
    if coordinate_kind == "training_sigma_stratum":
        optimizer_step = coordinate.get("optimizer_step")
        if (
            type(optimizer_step) is not int
            or optimizer_step not in range(MAX_STEPS)
            or coordinate != TRAINING_SIGMA_EXACT10[optimizer_step]
        ):
            fail(f"{label} training sigma coordinate differs")
    elif coordinate != EvaluationCoordinateV1().as_dict():
        fail(f"{label} evaluation sigma-one coordinate differs")
    for name in ACTUAL_Q_FIELDS:
        row = q_rows.get(name)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"shape", "dtype", "sha256"}
            or (row.get("shape"), row.get("dtype")) != ACTUAL_Q_ABI[name]
        ):
            fail(f"{label} actual q row {name} differs")
        _require_sha256_v1(row.get("sha256"), label=f"{label} {name} SHA")
    for field in ACTUAL_BRANCH_RECEIPT_FIELDS - {
        "row_id",
        "input_variant",
        "actual_q_tensor_rows",
        "coordinate",
        "coordinate_kind",
        "renderer_timestep_receipt",
        "route_identity",
        "registered_sp4_partition",
        "all30_hooks_used",
        "text_lens_runtime_abi",
    }:
        _require_sha256_v1(value.get(field), label=f"{label} {field}")
    return value


def _validate_actual_branch_pair_closed_v1(
    first: Any, second: Any, *, arm_id: str, label: str
) -> None:
    _validate_actual_branch_receipt_closed_v1(first, label=f"{label} first")
    _validate_actual_branch_receipt_closed_v1(second, label=f"{label} second")
    if first["route_identity"] == second["route_identity"]:
        fail(f"{label} branch forward identities are not distinct")
    fixed = (
        "row_id",
        "source_sha256",
        "epsilon_sha256",
        "text_lens_runtime_abi",
        "text_lens_sha256",
        "text_embs_sha256",
        "coordinate",
        "coordinate_kind",
        "renderer_timestep_receipt",
        "registered_sp4_partition",
    )
    if any(first[field] != second[field] for field in fixed):
        fail(f"{label} branch source/noise/text/coordinate differs")
    if arm_id == ARM_DUPLICATE:
        if first.get("input_variant") != "target" or second.get("input_variant") != "target":
            fail(f"{label} duplicate arm variants differ")
        ignored = {"route_identity", "actual_input_digest"}
        if any(first[field] != second[field] for field in ACTUAL_BRANCH_RECEIPT_FIELDS - ignored):
            fail(f"{label} duplicate arm actual branch bytes differ")
    elif arm_id in (ARM_ROLE_PAIR, ARM_ROLE_REPLICA):
        if (
            first.get("input_variant") != "target"
            or second.get("input_variant") != "role_swap"
        ):
            fail(f"{label} paired-role arm variants differ")
        required_different = (
            "label_binding_digest",
            "actual_q_tensor_rows_digest",
            "clean_target_sha256",
            "target_velocity_sha256",
            "event_mask_vae_sha256",
            "context_mask_vae_sha256",
            "packed_target_velocity_sha256",
            "packed_event_mask_sha256",
            "packed_context_mask_sha256",
        )
        if any(first[field] == second[field] for field in required_different):
            fail(f"{label} paired-role actual q/clean/velocity/masks are not distinct")
    else:
        fail(f"{label} arm differs")


def exact_two_branch_objective_v1(
    *,
    arm_id: str,
    prediction_target: Any,
    velocity_target: Any,
    event_target: Any,
    context_target: Any,
    prediction_second: Any,
    velocity_second: Any,
    event_second: Any,
    context_second: Any,
    first_actual_input_receipt: Mapping[str, Any],
    second_actual_input_receipt: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    """Exact A duplicate or B paired-role mean; no reward/margin scalar."""

    import torch

    if arm_id not in ARM_IDS:
        fail("two-branch objective arm differs")
    inputs = (first_actual_input_receipt, second_actual_input_receipt)
    _validate_actual_branch_pair_closed_v1(
        first_actual_input_receipt,
        second_actual_input_receipt,
        arm_id=arm_id,
        label="exact2 objective",
    )
    first_forward_identity = first_actual_input_receipt["route_identity"]
    second_forward_identity = second_actual_input_receipt["route_identity"]
    if any(
        not isinstance(value, torch.Tensor)
        for value in (
            prediction_target,
            velocity_target,
            event_target,
            context_target,
            prediction_second,
            velocity_second,
            event_second,
            context_second,
        )
    ):
        fail("exact2 branch tensor ABI differs")
    for index, (item, prediction, velocity, event, context) in enumerate(
        zip(
            inputs,
            (prediction_target, prediction_second),
            (velocity_target, velocity_second),
            (event_target, event_second),
            (context_target, context_second),
        )
    ):
        actual_tensor_digests = {
            "prediction_sha256": c1.tensor_sha256_v1(
                prediction.detach().contiguous().cpu()
            ),
            "packed_target_velocity_sha256": c1.tensor_sha256_v1(
                velocity.detach().contiguous().cpu()
            ),
            "packed_event_mask_sha256": c1.tensor_sha256_v1(
                event.detach().contiguous().cpu()
            ),
            "packed_context_mask_sha256": c1.tensor_sha256_v1(
                context.detach().contiguous().cpu()
            ),
        }
        if any(item.get(key) != value for key, value in actual_tensor_digests.items()):
            fail(f"exact2 branch {index} objective tensors differ from forward receipt")
    duplicate = arm_id == ARM_DUPLICATE
    if duplicate:
        if (
            first_actual_input_receipt["input_variant"] != "target"
            or second_actual_input_receipt["input_variant"] != "target"
            or any(
                first_actual_input_receipt[field]
                != second_actual_input_receipt[field]
                for field in (
                    "label_binding_digest",
                    "actual_q_tensor_rows_digest",
                    "clean_target_sha256",
                    "noisy_target_sha256",
                    "target_velocity_sha256",
                    "event_mask_vae_sha256",
                    "context_mask_vae_sha256",
                )
            )
            or prediction_target is prediction_second
            or prediction_target.untyped_storage().data_ptr()
            == prediction_second.untyped_storage().data_ptr()
            or any(
                not torch.equal(
                    first.detach().contiguous().view(torch.uint8),
                    second.detach().contiguous().view(torch.uint8),
                )
                for first, second in (
                    (prediction_target, prediction_second),
                    (velocity_target, velocity_second),
                    (event_target, event_second),
                    (context_target, context_second),
                )
            )
        ):
            fail("arm A is not an exact target duplicate control")
    first, first_receipt = _partition_energy_v1(
        prediction_target, velocity_target, event_target, context_target
    )
    second, second_receipt = _partition_energy_v1(
        prediction_second, velocity_second, event_second, context_second
    )
    total = (first + second) * 0.5
    if not bool(total.isfinite().item()):
        fail("two-branch objective is non-finite")
    receipt = {
        "arm_id": arm_id,
        "recipe": (
            "target_duplicate_exact2"
            if duplicate
            else "target_and_role_swap_exact2"
        ),
        "branch_names": (
            ["target", "target_exact_duplicate"]
            if duplicate
            else ["target", "role_swap"]
        ),
        "branch_losses": [first_receipt, second_receipt],
        "branch_reduction": "strict_arithmetic_mean",
        "fixed_branch_coefficients": [0.5, 0.5],
        "shared_epsilon_bit_exact": True,
        "first_forward_identity": first_forward_identity,
        "second_forward_identity": second_forward_identity,
        "first_actual_input_digest": first_actual_input_receipt[
            "actual_input_digest"
        ],
        "second_actual_input_digest": second_actual_input_receipt[
            "actual_input_digest"
        ],
        "actual_branch_inputs_closed_and_verified": True,
        "two_distinct_all30_forward_executions": True,
        "duplicate_control": duplicate,
        "duplicate_prediction_target_velocity_and_masks_bit_exact": duplicate,
        "paired_role_supervision": not duplicate,
        "tunable_loss_weights": False,
        "frozen_teacher_used": False,
        "frozen_velocity_reference_used": False,
        "reward_used": False,
        "total_loss": float(total.detach().item()),
    }
    return total, receipt


def execution_memory_contract_v1() -> Mapping[str, Any]:
    """Frozen memory schedule; changes execution, never the exact2 objective."""

    return {
        "activation_checkpoint_profile": ACTIVATION_CHECKPOINT_PROFILE,
        "activation_checkpointed_blocks": list(ACTIVATION_CHECKPOINT_BLOCKS),
        "activation_uncheckpointed_blocks": list(ACTIVATION_UNCHECKPOINTED_BLOCKS),
        "activation_checkpoint_nonreentrant": True,
        "activation_checkpoint_elal_route_context_replay": True,
        "training_branch_execution": (
            "strict_sequential_forward_backward_release_then_next"
        ),
        "preflight_branch_execution": (
            "strict_sequential_grad_enabled_forward_release_then_next_no_backward"
        ),
        "branch_order": ["target", "arm_registered_second"],
        "fixed_branch_coefficients": [0.5, 0.5],
        "gradient_reduce_clip_optimizer_after_both_branches": True,
        "simultaneous_live_autograd_branch_graphs_maximum": 1,
        "true_tensor_peak_only": True,
        "dummy_or_padding_allocations": False,
    }


def detach_branch_loss_evidence_v1(
    branch: Mapping[str, Any], *, label: str
) -> tuple[Any, Mapping[str, Any]]:
    """Compute one branch loss and return a canonical tensor-free receipt."""

    required = {
        "prediction",
        "target_velocity",
        "event_mask",
        "context_mask",
        "hook_receipt",
        "registered_sp4_partition",
        "actual_input_receipt",
    }
    if not isinstance(branch, Mapping) or not required.issubset(branch):
        fail(f"{label} branch graph envelope differs")
    actual = branch["actual_input_receipt"]
    _validate_actual_branch_receipt_closed_v1(actual, label=f"{label} actual input")
    tensors = {
        "prediction_sha256": branch["prediction"],
        "packed_target_velocity_sha256": branch["target_velocity"],
        "packed_event_mask_sha256": branch["event_mask"],
        "packed_context_mask_sha256": branch["context_mask"],
    }
    if any(
        actual.get(name)
        != c1.tensor_sha256_v1(tensor.detach().contiguous().cpu())
        for name, tensor in tensors.items()
    ):
        fail(f"{label} branch loss tensors differ from actual forward receipt")
    loss, loss_receipt = _partition_energy_v1(
        branch["prediction"],
        branch["target_velocity"],
        branch["event_mask"],
        branch["context_mask"],
    )
    evidence_unsigned = {
        "actual_input_receipt": dict(actual),
        "hook_receipt": dict(branch["hook_receipt"]),
        "registered_sp4_partition": dict(branch["registered_sp4_partition"]),
        "branch_loss": dict(loss_receipt),
        "portable_tensor_free": True,
    }
    # Canonical serialization is also the hostile guard against retaining a
    # Tensor/grad_fn inside evidence that outlives this branch graph.
    portable = json.loads(canonical_json_bytes(evidence_unsigned).decode("ascii"))
    evidence = {
        **portable,
        "branch_evidence_digest": object_sha256(portable),
    }
    return loss, evidence


def sequential_two_branch_objective_receipt_v1(
    *,
    arm_id: str,
    first_evidence: Mapping[str, Any],
    second_evidence: Mapping[str, Any],
    execution_mode: str,
) -> Mapping[str, Any]:
    """Join two already-detached sequential branches without retaining graphs."""

    if execution_mode not in {"preflight_forward_only", "training_forward_backward"}:
        fail("sequential exact2 execution mode differs")
    expected_evidence = {
        "actual_input_receipt",
        "hook_receipt",
        "registered_sp4_partition",
        "branch_loss",
        "portable_tensor_free",
        "branch_evidence_digest",
    }
    for index, evidence in enumerate((first_evidence, second_evidence)):
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != expected_evidence
            or evidence.get("portable_tensor_free") is not True
        ):
            fail(f"sequential exact2 branch {index} detached evidence differs")
        unsigned = dict(evidence)
        digest = unsigned.pop("branch_evidence_digest", None)
        if digest != object_sha256(unsigned):
            fail(f"sequential exact2 branch {index} evidence digest differs")
    first_actual = first_evidence["actual_input_receipt"]
    second_actual = second_evidence["actual_input_receipt"]
    _validate_actual_branch_pair_closed_v1(
        first_actual,
        second_actual,
        arm_id=arm_id,
        label="sequential exact2 objective",
    )
    duplicate = arm_id == ARM_DUPLICATE
    first_loss = float(first_evidence["branch_loss"]["total_loss"])
    second_loss = float(second_evidence["branch_loss"]["total_loss"])
    total = struct.unpack(
        ">f",
        struct.pack(
            ">f",
            struct.unpack(">f", struct.pack(">f", first_loss + second_loss))[0]
            * 0.5,
        ),
    )[0]
    receipt = {
        "arm_id": arm_id,
        "recipe": (
            "target_duplicate_exact2"
            if duplicate
            else "target_and_role_swap_exact2"
        ),
        "branch_names": (
            ["target", "target_exact_duplicate"]
            if duplicate
            else ["target", "role_swap"]
        ),
        "branch_losses": [
            dict(first_evidence["branch_loss"]),
            dict(second_evidence["branch_loss"]),
        ],
        "branch_reduction": "strict_arithmetic_mean",
        "fixed_branch_coefficients": [0.5, 0.5],
        "shared_epsilon_bit_exact": True,
        "first_forward_identity": first_actual["route_identity"],
        "second_forward_identity": second_actual["route_identity"],
        "first_actual_input_digest": first_actual["actual_input_digest"],
        "second_actual_input_digest": second_actual["actual_input_digest"],
        "actual_branch_inputs_closed_and_verified": True,
        "two_distinct_all30_forward_executions": True,
        "duplicate_control": duplicate,
        "duplicate_prediction_target_velocity_and_masks_bit_exact": duplicate,
        "paired_role_supervision": not duplicate,
        "tunable_loss_weights": False,
        "frozen_teacher_used": False,
        "frozen_velocity_reference_used": False,
        "reward_used": False,
        "total_loss": total,
        "execution_mode": execution_mode,
        "branch_execution_schedule": (
            "strict_sequential_forward_backward_release_then_next"
            if execution_mode == "training_forward_backward"
            else "strict_sequential_grad_enabled_forward_release_then_next_no_backward"
        ),
        "coefficient_applied_before_each_backward": (
            execution_mode == "training_forward_backward"
        ),
        "first_graph_released_before_second_forward": True,
        "simultaneous_live_autograd_branch_graphs_maximum": 1,
        "detached_portable_branch_receipts_before_graph_release": True,
        "gradient_reduce_clip_optimizer_after_both_branches": (
            execution_mode == "training_forward_backward"
        ),
        "preflight_backward_executed": False,
    }
    _validate_objective_receipt_closed_v1(
        receipt, arm_id=arm_id, label="sequential exact2 objective"
    )
    return receipt


def validate_branch_lifecycle_receipt_v1(
    value: Any, *, execution_mode: str, label: str
) -> Mapping[str, Any]:
    training = execution_mode == "training_forward_backward"
    fields = {
        "execution_mode",
        "activation_checkpoint_profile",
        "activation_checkpointed_blocks",
        "activation_uncheckpointed_blocks",
        "activation_checkpoint_nonreentrant",
        "activation_checkpoint_elal_route_context_replay",
        "first_backward_completed",
        "second_backward_completed",
        "first_prediction_weakref_released_before_second_forward",
        "second_prediction_weakref_released_before_post_branch_work",
        "first_graph_deleted_before_second_forward",
        "second_graph_deleted_before_post_branch_work",
        "inter_branch_gc_collect_called",
        "inter_branch_cuda_empty_cache_called",
        "second_forward_started_after_first_release",
        "simultaneous_live_autograd_branch_graphs_maximum",
        "first_gradient_tensors_preserved_across_graph_release",
        "gradient_reduce_clip_optimizer_after_both_branches",
        "preflight_grad_enabled_training_graph",
        "preflight_backward_executed",
        "peak_semantics",
        "first_branch_peak_allocated_bytes",
        "post_first_release_allocated_bytes",
        "second_branch_peak_allocated_bytes",
        "post_second_release_allocated_bytes",
        "dummy_or_padding_allocations",
    }
    if (
        execution_mode not in {"preflight_forward_only", "training_forward_backward"}
        or not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("execution_mode") != execution_mode
        or value.get("activation_checkpoint_profile") != ACTIVATION_CHECKPOINT_PROFILE
        or value.get("activation_checkpointed_blocks")
        != list(ACTIVATION_CHECKPOINT_BLOCKS)
        or value.get("activation_uncheckpointed_blocks")
        != list(ACTIVATION_UNCHECKPOINTED_BLOCKS)
        or value.get("activation_checkpoint_nonreentrant") is not True
        or value.get("activation_checkpoint_elal_route_context_replay") is not True
        or value.get("first_backward_completed") is not training
        or value.get("second_backward_completed") is not training
        or value.get("first_prediction_weakref_released_before_second_forward")
        is not True
        or value.get("second_prediction_weakref_released_before_post_branch_work")
        is not True
        or value.get("first_graph_deleted_before_second_forward") is not True
        or value.get("second_graph_deleted_before_post_branch_work") is not True
        or value.get("inter_branch_gc_collect_called") is not True
        or value.get("inter_branch_cuda_empty_cache_called") is not True
        or value.get("second_forward_started_after_first_release") is not True
        or value.get("simultaneous_live_autograd_branch_graphs_maximum") != 1
        or value.get("first_gradient_tensors_preserved_across_graph_release")
        is not (True if training else None)
        or value.get("gradient_reduce_clip_optimizer_after_both_branches")
        is not training
        or value.get("preflight_grad_enabled_training_graph")
        is not (None if training else True)
        or value.get("preflight_backward_executed") is not False
        or value.get("peak_semantics")
        != (
            "maximum_of_sequential_true_branch_graphs_with_retained_parameter_gradients"
            if training
            else "maximum_of_sequential_true_grad_enabled_branch_graphs_without_backward"
        )
        or type(value.get("first_branch_peak_allocated_bytes")) is not int
        or value.get("first_branch_peak_allocated_bytes") <= 0
        or type(value.get("post_first_release_allocated_bytes")) is not int
        or value.get("post_first_release_allocated_bytes") <= 0
        or value.get("post_first_release_allocated_bytes")
        > value.get("first_branch_peak_allocated_bytes")
        or type(value.get("second_branch_peak_allocated_bytes")) is not int
        or value.get("second_branch_peak_allocated_bytes")
        < value.get("first_branch_peak_allocated_bytes")
        or type(value.get("post_second_release_allocated_bytes")) is not int
        or value.get("post_second_release_allocated_bytes") <= 0
        or value.get("post_second_release_allocated_bytes")
        > value.get("second_branch_peak_allocated_bytes")
        or value.get("dummy_or_padding_allocations") is not False
    ):
        fail(f"{label} sequential branch lifecycle differs")
    return value


def gradient_accumulation_guard_v1(
    named: Sequence[tuple[str, Any]], *, label: str
) -> tuple[tuple[Any, ...], ...]:
    """Capture tensor identity/version without retaining any gradient tensor."""

    rows = []
    for name, parameter in named:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            fail(f"{label} lacks an explicit accumulated gradient: {name}")
        rows.append(
            (
                name,
                id(gradient),
                int(gradient.data_ptr()),
                int(gradient._version),
                tuple(int(item) for item in gradient.shape),
                str(gradient.dtype),
                str(gradient.device),
            )
        )
    if len(rows) != 668:
        fail(f"{label} exact668 accumulated-gradient closure differs")
    return tuple(rows)


def validate_gradient_accumulation_guard_v1(
    named: Sequence[tuple[str, Any]],
    reference: Sequence[tuple[Any, ...]],
    *,
    label: str,
) -> None:
    if gradient_accumulation_guard_v1(named, label=label) != tuple(reference):
        fail(f"{label} first-branch accumulated gradients changed during graph release")


def role_only_swap_invariants_v1(
    matched: Any,
    opposite: Any,
    mismatch: Any,
    *,
    matched_masks: Mapping[str, Any],
    mismatch_masks: Mapping[str, Any],
    matched_source: Any,
    mismatch_source: Any,
    matched_instruction: str,
    mismatch_instruction: str,
    matched_sigma: float,
    mismatch_sigma: float,
    matched_x_sigma: Any,
    mismatch_x_sigma: Any,
    matched_epsilon: Any,
    mismatch_epsilon: Any,
    matched_slot_entity_ids: Sequence[str],
    opposite_slot_entity_ids: Sequence[str],
    mismatch_slot_entity_ids: Sequence[str],
    matched_slot_roles: Sequence[str],
    opposite_slot_roles: Sequence[str],
    mismatch_slot_roles: Sequence[str],
    matched_role_code_order: Sequence[str],
    opposite_role_code_order: Sequence[str],
    mismatch_role_code_order: Sequence[str],
) -> Mapping[str, Any]:
    """Prove the hybrid mismatch changes only registered role-bearing fields.

    ``mismatch`` must copy q_entity/q_relation byte-for-byte from ``opposite``
    while retaining every other latent field, every spatial mask, and the
    semantic-role code order byte-for-byte from ``matched``.  ELAL-3 slots are
    canonical semantic-role slots.  The global role-code channel ABI/order is
    fixed, while the active one-hot values inside the donor q_entity are
    allowed to change (notably co_agent -> receiver in the handover row).
    """

    import torch

    fixed = (
        "q_local",
        "q_phase",
        "q_terminal",
        "q_camera",
        "entity_presence",
        "temporal_valid",
        "relation_valid",
        "phase_valid",
    )
    swapped = ("q_entity", "q_relation")
    mask_names = (
        "event_mask_patch",
        "context_mask_patch",
        "event_mask_vae",
        "context_mask_vae",
        "role_amodal_mask_patch",
        "role_visible_mask_patch",
        "role_event_mask_patch",
        "role_event_mask_vae",
    )
    fixed_equal: dict[str, bool] = {}
    swapped_from_opposite: dict[str, bool] = {}
    swapped_different_from_matched: dict[str, bool] = {}
    for name in fixed:
        left, right = getattr(matched, name), getattr(mismatch, name)
        fixed_equal[name] = bool(
            torch.equal(
                left.detach().contiguous().view(torch.uint8),
                right.detach().contiguous().view(torch.uint8),
            )
        )
    for name in swapped:
        left = getattr(matched, name)
        right = getattr(mismatch, name)
        donor = getattr(opposite, name)
        swapped_from_opposite[name] = bool(
            torch.equal(
                donor.detach().contiguous().view(torch.uint8),
                right.detach().contiguous().view(torch.uint8),
            )
        )
        swapped_different_from_matched[name] = not bool(
            torch.equal(
                left.detach().contiguous().view(torch.uint8),
                right.detach().contiguous().view(torch.uint8),
            )
        )
    if set(matched_masks) != set(mask_names) or set(mismatch_masks) != set(mask_names):
        fail("role-only spatial mask field closure differs")
    mask_equal = {
        name: bool(
            torch.equal(
                matched_masks[name].detach().contiguous().view(torch.uint8),
                mismatch_masks[name].detach().contiguous().view(torch.uint8),
            )
        )
        for name in mask_names
    }
    renderer_tensor_pairs = {
        "source": (matched_source, mismatch_source),
        "x_sigma": (matched_x_sigma, mismatch_x_sigma),
        "epsilon": (matched_epsilon, mismatch_epsilon),
    }
    renderer_fixed_equal: dict[str, bool] = {}
    for name, (left, right) in renderer_tensor_pairs.items():
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            fail(f"role-only fixed renderer {name} tensor ABI differs")
        renderer_fixed_equal[name] = bool(
            torch.equal(
                left.detach().contiguous().view(torch.uint8),
                right.detach().contiguous().view(torch.uint8),
            )
        )
    renderer_fixed_equal["x_sigma_equals_epsilon_matched"] = bool(
        torch.equal(
            matched_x_sigma.detach().contiguous().view(torch.uint8),
            matched_epsilon.detach().contiguous().view(torch.uint8),
        )
    )
    renderer_fixed_equal["x_sigma_equals_epsilon_mismatch"] = bool(
        torch.equal(
            mismatch_x_sigma.detach().contiguous().view(torch.uint8),
            mismatch_epsilon.detach().contiguous().view(torch.uint8),
        )
    )
    instruction_fixed = (
        isinstance(matched_instruction, str)
        and bool(matched_instruction)
        and mismatch_instruction == matched_instruction
    )
    sigma_fixed_one = (
        struct.pack(">f", float(matched_sigma)).hex() == "3f800000"
        and struct.pack(">f", float(mismatch_sigma)).hex() == "3f800000"
    )
    role_code_orders = tuple(
        tuple(order)
        for order in (
            matched_role_code_order,
            opposite_role_code_order,
            mismatch_role_code_order,
        )
    )
    role_code_order = role_code_orders[0]
    role_code_order_fixed = (
        bool(role_code_order)
        and len(set(role_code_order)) == len(role_code_order)
        and all(isinstance(role, str) and role for role in role_code_order)
        and role_code_orders[1:] == (role_code_order, role_code_order)
    )
    role_channel_abi = {
        "q_entity_role_code_channels": [19, 27],
        "q_relation_endpoint_slot_channels": [9, 11],
    }
    for field, stop in (("q_entity", 27), ("q_relation", 11)):
        if any(
            not isinstance(getattr(item, field), torch.Tensor)
            or getattr(item, field).ndim < 1
            or getattr(item, field).shape[-1] < stop
            for item in (matched, opposite, mismatch)
        ):
            fail(f"role-only {field} channel ABI differs")
    matched_slots = tuple(matched_slot_entity_ids)
    opposite_slots = tuple(opposite_slot_entity_ids)
    mismatch_slots = tuple(mismatch_slot_entity_ids)
    matched_roles = tuple(matched_slot_roles)
    opposite_roles = tuple(opposite_slot_roles)
    mismatch_roles = tuple(mismatch_slot_roles)
    expected_opposite = tuple(matched_slots[index] for index in (1, 0, 2))
    valid_active_roles = (
        len(matched_roles) == 3
        and len(opposite_roles) == 3
        and len(set(matched_roles)) == 3
        and len(set(opposite_roles)) == 3
        and all(role in role_code_order for role in matched_roles + opposite_roles)
        and mismatch_roles == opposite_roles
    )
    matched_slot_mapping = {
        entity_id: slot_index for slot_index, entity_id in enumerate(matched_slots)
    }
    mismatch_slot_mapping = {
        entity_id: slot_index for slot_index, entity_id in enumerate(mismatch_slots)
    }
    matched_active_roles = dict(zip(matched_slots, matched_roles))
    mismatch_active_roles = dict(zip(mismatch_slots, mismatch_roles))
    physical_mapping_changed = matched_slot_mapping != mismatch_slot_mapping
    if (
        len(matched_slots) != 3
        or len(set(matched_slots)) != 3
        or any(not isinstance(entity_id, str) or not entity_id for entity_id in matched_slots)
        or opposite_slots != expected_opposite
        or mismatch_slots != opposite_slots
        or not valid_active_roles
        or not role_code_order_fixed
        or not physical_mapping_changed
        or not all(fixed_equal.values())
        or not all(mask_equal.values())
        or not all(renderer_fixed_equal.values())
        or not instruction_fixed
        or not sigma_fixed_one
        or not all(swapped_from_opposite.values())
        or not all(swapped_different_from_matched.values())
    ):
        fail("role-only intervention fixed/swapped tensor closure differs")
    return {
        "fixed_fields_bit_exact": fixed_equal,
        "spatial_masks_fixed_bit_exact": mask_equal,
        "renderer_fixed_fields": {
            **renderer_fixed_equal,
            "instruction_bit_exact": True,
            "instruction_sha256": hashlib.sha256(
                matched_instruction.encode("utf-8")
            ).hexdigest(),
            "sigma_float32_be_hex": "3f800000",
        },
        "swapped_fields_copied_bit_exact_from_opposite": swapped_from_opposite,
        "swapped_fields_bit_different_from_matched": swapped_different_from_matched,
        "semantic_role_code_order_receipt": {
            "role_code_order": list(role_code_order),
            "role_code_order_fixed_across_variants": True,
            **role_channel_abi,
            "active_role_values_required_equal_across_variants": False,
        },
        "matched_slot_entity_ids": list(matched_slots),
        "opposite_slot_entity_ids": list(opposite_slots),
        "mismatch_slot_entity_ids": list(mismatch_slots),
        "matched_slot_roles": list(matched_roles),
        "opposite_slot_roles": list(opposite_roles),
        "mismatch_slot_roles": list(mismatch_roles),
        "physical_entity_slot_permutation": [1, 0, 2],
        "matched_physical_entity_to_semantic_slot": matched_slot_mapping,
        "mismatch_physical_entity_to_semantic_slot": mismatch_slot_mapping,
        "matched_physical_entity_to_active_role": matched_active_roles,
        "mismatch_physical_entity_to_active_role": mismatch_active_roles,
        "physical_entity_to_semantic_slot_mapping_different": True,
    }


def validate_role_only_invariant_receipts_v1(
    rows: Any, *, stage: str
) -> Mapping[str, Any]:
    if stage not in {"step0", "step10"}:
        fail("role-only invariant stage differs")
    if not isinstance(rows, list) or len(rows) != 4:
        fail("role-only invariant exact4 closure differs")
    fixed_names = {
        "q_local",
        "q_phase",
        "q_terminal",
        "q_camera",
        "entity_presence",
        "temporal_valid",
        "relation_valid",
        "phase_valid",
    }
    mask_names = {
        "event_mask_patch",
        "context_mask_patch",
        "event_mask_vae",
        "context_mask_vae",
        "role_amodal_mask_patch",
        "role_visible_mask_patch",
        "role_event_mask_patch",
        "role_event_mask_vae",
    }
    for row, (row_id, clean_variant) in zip(rows, ROLE_ONLY_CELL_ORDER):
        invariant = row.get("invariant") if isinstance(row, Mapping) else None
        coordinate = row.get("evaluation_coordinate") if isinstance(row, Mapping) else None
        role_code = (
            invariant.get("semantic_role_code_order_receipt")
            if isinstance(invariant, Mapping)
            else None
        )
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"row_id", "clean_variant", "evaluation_coordinate", "invariant"}
            or (row.get("row_id"), row.get("clean_variant"))
            != (row_id, clean_variant)
            or coordinate != EvaluationCoordinateV1().as_dict()
            or not isinstance(invariant, Mapping)
            or set(invariant.get("fixed_fields_bit_exact", ())) != fixed_names
            or not all(invariant["fixed_fields_bit_exact"].values())
            or set(invariant.get("spatial_masks_fixed_bit_exact", ())) != mask_names
            or not all(invariant["spatial_masks_fixed_bit_exact"].values())
            or invariant.get("renderer_fixed_fields", {}).get("source") is not True
            or invariant.get("renderer_fixed_fields", {}).get("x_sigma") is not True
            or invariant.get("renderer_fixed_fields", {}).get("epsilon") is not True
            or invariant.get("renderer_fixed_fields", {}).get(
                "x_sigma_equals_epsilon_matched"
            )
            is not True
            or invariant.get("renderer_fixed_fields", {}).get(
                "x_sigma_equals_epsilon_mismatch"
            )
            is not True
            or invariant.get("renderer_fixed_fields", {}).get(
                "instruction_bit_exact"
            )
            is not True
            or invariant.get("renderer_fixed_fields", {}).get(
                "sigma_float32_be_hex"
            )
            != "3f800000"
            or set(
                invariant.get("swapped_fields_copied_bit_exact_from_opposite", ())
            )
            != {"q_entity", "q_relation"}
            or not all(
                invariant["swapped_fields_copied_bit_exact_from_opposite"].values()
            )
            or set(
                invariant.get("swapped_fields_bit_different_from_matched", ())
            )
            != {"q_entity", "q_relation"}
            or not all(
                invariant["swapped_fields_bit_different_from_matched"].values()
            )
            or not isinstance(role_code, Mapping)
            or role_code.get("role_code_order_fixed_across_variants") is not True
            or role_code.get("q_entity_role_code_channels") != [19, 27]
            or role_code.get("q_relation_endpoint_slot_channels") != [9, 11]
            or role_code.get("active_role_values_required_equal_across_variants")
            is not False
            or invariant.get("physical_entity_slot_permutation") != [1, 0, 2]
            or invariant.get("opposite_slot_entity_ids")
            != invariant.get("mismatch_slot_entity_ids")
            or invariant.get("opposite_slot_roles")
            != invariant.get("mismatch_slot_roles")
            or invariant.get("matched_physical_entity_to_semantic_slot")
            == invariant.get("mismatch_physical_entity_to_semantic_slot")
            or invariant.get("physical_entity_to_semantic_slot_mapping_different")
            is not True
        ):
            fail(f"role-only invariant receipt differs: {row_id}/{clean_variant}")
        _require_sha256_v1(
            invariant["renderer_fixed_fields"].get("instruction_sha256"),
            label="role-only invariant instruction SHA",
        )
    value = {
        "stage": stage,
        "exact4_order": [list(item) for item in ROLE_ONLY_CELL_ORDER],
        "only_q_entity_q_relation_swapped": True,
        "fixed_renderer_latent_masks_and_role_code_abi_proven": True,
        "invariant_digest": object_sha256(rows),
    }
    return value


def full_q_route_matrix_v1(
    *,
    prediction_target_q: Any,
    prediction_role_q: Any,
    velocity_target: Any,
    velocity_role: Any,
    event_target: Any,
    context_target: Any,
    event_role: Any,
    context_role: Any,
) -> Mapping[str, Any]:
    energies: dict[str, float] = {}
    input_bindings: dict[str, Mapping[str, str]] = {}
    for q_name, prediction in (
        ("target_q", prediction_target_q),
        ("role_swap_q", prediction_role_q),
    ):
        for truth_name, velocity, event, context in (
            ("target", velocity_target, event_target, context_target),
            ("role_swap", velocity_role, event_role, context_role),
        ):
            energy, _ = _partition_energy_v1(prediction, velocity, event, context)
            key = f"{q_name}__{truth_name}"
            energies[key] = float(energy.detach().item())
            input_bindings[key] = {
                "prediction_sha256": c1.tensor_sha256_v1(
                    prediction.detach().contiguous().cpu()
                ),
                "packed_target_velocity_sha256": c1.tensor_sha256_v1(
                    velocity.detach().contiguous().cpu()
                ),
                "packed_event_mask_sha256": c1.tensor_sha256_v1(
                    event.detach().contiguous().cpu()
                ),
                "packed_context_mask_sha256": c1.tensor_sha256_v1(
                    context.detach().contiguous().cpu()
                ),
            }
    target_margin = energies["target_q__role_swap"] - energies["target_q__target"]
    role_margin = energies["role_swap_q__target"] - energies["role_swap_q__role_swap"]
    return {
        "claim_name": "oracle_route_controllability_only",
        "participant_role_binding_claim_forbidden": True,
        "energies": energies,
        "energy_input_bindings": input_bindings,
        "target_q_row_margin": target_margin,
        "role_swap_q_row_margin": role_margin,
        "signed_diagonal_margin": min(target_margin, role_margin),
        "strict_diagonal_dominance": target_margin > 0.0 and role_margin > 0.0,
    }


def role_only_cell_v1(
    *,
    row_id: str,
    clean_variant: str,
    prediction_matched: Any,
    prediction_mismatch: Any,
    velocity_clean: Any,
    velocity_opposite: Any,
    event_clean: Any,
    context_clean: Any,
    contrast_union_mask: Any,
    require_positive_contrast: bool = True,
) -> Mapping[str, Any]:
    import torch

    if row_id not in ROW_IDS or clean_variant not in TRAIN_VARIANTS:
        fail("role-only cell identity differs")
    matched, _ = _partition_energy_v1(
        prediction_matched, velocity_clean, event_clean, context_clean
    )
    mismatch, _ = _partition_energy_v1(
        prediction_mismatch, velocity_clean, event_clean, context_clean
    )
    if (
        not isinstance(contrast_union_mask, torch.Tensor)
        or contrast_union_mask.dtype != torch.bool
        or contrast_union_mask.shape != prediction_matched.shape
        or not bool(contrast_union_mask.any().item())
    ):
        fail("role-only contrast union mask differs")
    predicted = (
        prediction_matched.float() - prediction_mismatch.float()
    )[contrast_union_mask]
    clean = (velocity_clean.float() - velocity_opposite.float())[
        contrast_union_mask
    ]
    predicted_norm = torch.linalg.vector_norm(predicted)
    clean_norm = torch.linalg.vector_norm(clean)
    predicted_norm_value = float(predicted_norm.item())
    clean_norm_value = float(clean_norm.item())
    if type(require_positive_contrast) is not bool or clean_norm_value <= 0.0:
        fail("role-only clean contrast norm/requirement differs")
    if predicted_norm_value <= 0.0:
        if require_positive_contrast:
            fail("role-only predicted contrast norm must be strictly positive")
        cosine: Optional[float] = None
    else:
        cosine = float(
            (torch.dot(predicted, clean) / (predicted_norm * clean_norm)).item()
        )
    if cosine is not None and not math.isfinite(cosine):
        fail("role-only normalized contrast is non-finite")
    matched_value = float(matched.detach().item())
    mismatch_value = float(mismatch.detach().item())
    return {
        "row_id": row_id,
        "clean_variant": clean_variant,
        "opposite_variant": (
            "role_swap" if clean_variant == "target" else "target"
        ),
        "matched_energy": matched_value,
        "mismatch_energy": mismatch_value,
        "margin_mismatch_minus_matched": mismatch_value - matched_value,
        "normalized_predicted_vs_clean_role_contrast": cosine,
        "positive_contrast_alignment": cosine is not None and cosine > 0.0,
        "positive_contrast_required_at_this_stage": require_positive_contrast,
        "predicted_contrast_l2": predicted_norm_value,
        "clean_contrast_l2": clean_norm_value,
        "input_bindings": {
            "matched_prediction_sha256": c1.tensor_sha256_v1(
                prediction_matched.detach().contiguous().cpu()
            ),
            "mismatch_prediction_sha256": c1.tensor_sha256_v1(
                prediction_mismatch.detach().contiguous().cpu()
            ),
            "packed_clean_velocity_sha256": c1.tensor_sha256_v1(
                velocity_clean.detach().contiguous().cpu()
            ),
            "packed_opposite_velocity_sha256": c1.tensor_sha256_v1(
                velocity_opposite.detach().contiguous().cpu()
            ),
            "packed_event_mask_sha256": c1.tensor_sha256_v1(
                event_clean.detach().contiguous().cpu()
            ),
            "packed_context_mask_sha256": c1.tensor_sha256_v1(
                context_clean.detach().contiguous().cpu()
            ),
            "contrast_union_mask_sha256": c1.tensor_sha256_v1(
                contrast_union_mask.detach().contiguous().cpu()
            ),
        },
    }


def validate_step10_gates_v1(
    *,
    step0_full_q: Mapping[str, Mapping[str, Any]],
    step10_full_q: Mapping[str, Mapping[str, Any]],
    step0_role_only: Sequence[Mapping[str, Any]],
    step10_role_only: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if set(step0_full_q) != set(ROW_IDS) or set(step10_full_q) != set(ROW_IDS):
        fail("full-q route matrix exact2-row closure differs")
    full_rows: dict[str, Any] = {}
    for row_id in ROW_IDS:
        before = float(step0_full_q[row_id].get("signed_diagonal_margin"))
        after = float(step10_full_q[row_id].get("signed_diagonal_margin"))
        passed = step10_full_q[row_id].get("strict_diagonal_dominance") is True
        full_rows[row_id] = {
            "margin_step0": before,
            "margin_step10": after,
            "step0_margin_recorded_not_a_full_q_threshold": before,
            "pass": passed,
        }
    before_cells = {
        (row.get("row_id"), row.get("clean_variant")): row
        for row in step0_role_only
    }
    after_cells = {
        (row.get("row_id"), row.get("clean_variant")): row
        for row in step10_role_only
    }
    if tuple(before_cells) != ROLE_ONLY_CELL_ORDER or tuple(after_cells) != ROLE_ONLY_CELL_ORDER:
        fail("role-only exact4 ordered cell closure differs")
    role_rows: list[Mapping[str, Any]] = []
    for key in ROLE_ONLY_CELL_ORDER:
        before = float(before_cells[key].get("margin_mismatch_minus_matched"))
        after = float(after_cells[key].get("margin_mismatch_minus_matched"))
        cosine = float(
            after_cells[key].get("normalized_predicted_vs_clean_role_contrast")
        )
        passed = after > max(0.0, before) and cosine > 0.0
        role_rows.append(
            {
                "row_id": key[0],
                "clean_variant": key[1],
                "margin_step0": before,
                "margin_step10": after,
                "required_step10_gt": max(0.0, before),
                "normalized_contrast_step10": cosine,
                "pass": passed,
            }
        )
    all_pass = all(row["pass"] for row in full_rows.values()) and all(
        row["pass"] for row in role_rows
    )
    if not all_pass:
        fail("step10 route/role-only preregistered gates failed")
    primary = min(float(row["margin_step10"]) for row in role_rows)
    return {
        "status": "STEP10_C2_ORACLE_ROUTE_AND_ROLE_ONLY_GATES_PASS",
        "full_q_claim": "oracle_route_controllability_only",
        "full_q_rows": full_rows,
        "role_only_cells": role_rows,
        "primary_metric": "minimum_of_four_role_only_matched_vs_mismatch_margins",
        "primary_metric_value": primary,
        "all_hard_gates_pass": True,
        "weighted_metric_sum_used": False,
    }


def evaluation_noise_seed_v1(arm_seed: int, row_index: int) -> int:
    if type(arm_seed) is not int or arm_seed < 0 or row_index not in (0, 1):
        fail("evaluation noise seed coordinates differ")
    return 100 * arm_seed + row_index


def training_noise_seed_v1(arm_seed: int, step_zero: int, row_index: int) -> int:
    if (
        type(arm_seed) is not int
        or arm_seed < 0
        or step_zero not in range(MAX_STEPS)
        or row_index not in (0, 1)
    ):
        fail("training noise seed coordinates differ")
    payload = canonical_json_bytes(
        {
            "method": METHOD,
            "purpose": "shared_target_branch_flow_epsilon",
            "seed": arm_seed,
            "step_zero": step_zero,
            "row_index": row_index,
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def cpu_epsilon_v1(seed: int) -> Any:
    import torch

    if type(seed) is not int or not 0 <= seed < 2**63:
        fail("epsilon seed differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(
        LATENT_SHAPE, generator=generator, dtype=torch.float32, device="cpu"
    ).contiguous()
    if (
        tuple(value.shape) != LATENT_SHAPE
        or not bool(torch.isfinite(value).all().item())
        or value.requires_grad
    ):
        fail("CPU fp32 epsilon ABI differs")
    return value


def build_local_noise_schedule_receipt_v1(
    *, arm_seed: int, row_index: int, target_clean: Any, sigma_module: Any
) -> Mapping[str, Any]:
    """Hash one DP row's exact target/noise/noisy-input schedule on CPU."""

    import torch

    if (
        row_index not in (0, 1)
        or not isinstance(target_clean, torch.Tensor)
        or target_clean.dtype != torch.float32
        or target_clean.device.type != "cpu"
        or tuple(target_clean.shape) != LATENT_SHAPE
        or not target_clean.is_contiguous()
        or target_clean.requires_grad
    ):
        fail("local common target schedule clean latent ABI differs")
    target_sha = c1.tensor_sha256_v1(target_clean)
    rows: list[Mapping[str, Any]] = []
    for step_zero in range(MAX_STEPS):
        coordinate = sigma_module.select_sigma_stratum(step_zero)
        coordinate_row = coordinate.as_dict()
        seed = training_noise_seed_v1(arm_seed, step_zero, row_index)
        epsilon = cpu_epsilon_v1(seed)
        sigma = float(coordinate.sigma)
        noisy = ((1.0 - sigma) * target_clean + sigma * epsilon).contiguous()
        velocity = (epsilon - target_clean).contiguous()
        input_row = {
            "row_index": row_index,
            "row_id": ROW_IDS[row_index],
            "sigma_coordinate": coordinate_row,
            "epsilon_seed": seed,
            "epsilon_sha256": c1.tensor_sha256_v1(epsilon),
            "target_sha256": target_sha,
            "noisy_target_sha256": c1.tensor_sha256_v1(noisy),
            "target_velocity_sha256": c1.tensor_sha256_v1(velocity),
        }
        rows.append(
            {
                "step_zero": step_zero,
                **input_row,
                "epsilon_shape": list(LATENT_SHAPE),
                "epsilon_dtype": "torch.float32",
                "common_target_input_digest": object_sha256(input_row),
            }
        )
    seed = evaluation_noise_seed_v1(arm_seed, row_index)
    epsilon = cpu_epsilon_v1(seed)
    evaluation_input = {
        "row_index": row_index,
        "row_id": ROW_IDS[row_index],
        "sigma_float32_be_hex": "3f800000",
        "x_sigma_equals_epsilon": True,
        "epsilon_seed": seed,
        "epsilon_sha256": c1.tensor_sha256_v1(epsilon),
        "target_sha256": target_sha,
        "noisy_target_sha256": c1.tensor_sha256_v1(epsilon),
        "target_velocity_sha256": c1.tensor_sha256_v1(
            (epsilon - target_clean).contiguous()
        ),
    }
    evaluation = {
        **evaluation_input,
        "common_target_input_digest": object_sha256(evaluation_input),
    }
    return {
        "row_index": row_index,
        "row_id": ROW_IDS[row_index],
        "training_rows": rows,
        "evaluation_row": evaluation,
        "local_schedule_digest": object_sha256(
            {"training_rows": rows, "evaluation_row": evaluation}
        ),
    }


def merge_noise_schedule_receipts_v1(
    rows_by_dp: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Merge the two independently computed DP rows into canonical step-major order."""

    indexed = {
        row.get("row_index"): row
        for row in rows_by_dp
        if isinstance(row, Mapping)
    }
    if set(indexed) != {0, 1} or any(
        indexed[index].get("row_id") != ROW_IDS[index]
        or not isinstance(indexed[index].get("training_rows"), list)
        or len(indexed[index]["training_rows"]) != MAX_STEPS
        or not isinstance(indexed[index].get("evaluation_row"), Mapping)
        for index in (0, 1)
    ):
        fail("DP2 common target schedule merge closure differs")
    rows = [
        indexed[row_index]["training_rows"][step_zero]
        for step_zero in range(MAX_STEPS)
        for row_index in (0, 1)
    ]
    evaluation = [indexed[index]["evaluation_row"] for index in (0, 1)]
    result = {
        "training_exact10_common_target_branch": rows,
        "evaluation_sigma1_by_row": evaluation,
        "rng": "cpu_float32_torch_standard_normal",
    }
    return {**result, "schedule_digest": object_sha256(result)}


def build_noise_schedule_receipt_v1(
    *, arm_seed: int, targets_by_row: Mapping[int, Any], sigma_module: Any
) -> Mapping[str, Any]:
    """Convenience builder used by tests/controllers with both clean rows."""

    if set(targets_by_row) != {0, 1}:
        fail("common target schedule requires exact2 clean rows")
    return merge_noise_schedule_receipts_v1(
        [
            build_local_noise_schedule_receipt_v1(
                arm_seed=arm_seed,
                row_index=row_index,
                target_clean=targets_by_row[row_index],
                sigma_module=sigma_module,
            )
            for row_index in (0, 1)
        ]
    )


def renderer_branch_forward_v1(
    *,
    transformer: Any,
    renderer: Any,
    elal_handle: Any,
    elal_module: Any,
    source: Any,
    clean_target: Any,
    epsilon: Any,
    coordinate: Any,
    oracle_label: Any,
    rope: Any,
    device: Any,
    text_lens: Any,
    text_embs: Any,
    sp_rank: int,
    route_identity: str,
) -> Mapping[str, Any]:
    """One actual-shape Bernini branch with one independently audited route."""

    import torch

    latent = getattr(oracle_label, "latent", None)
    label_receipt = getattr(oracle_label, "receipt", None)
    truth_event_mask_vae = getattr(oracle_label, "event_mask_vae", None)
    truth_context_mask_vae = getattr(oracle_label, "context_mask_vae", None)
    if latent is None or not isinstance(label_receipt, Mapping):
        fail("C2 branch oracle label ABI differs")
    packed = dict(
        c1.prepare_flow_v1(
            source=source,
            target=clean_target,
            epsilon=epsilon,
            coordinate=coordinate,
            rope=rope,
            device=device,
        )
    )
    if packed["patch_grid"] != PATCH_GRID:
        fail("C2 runtime flow patch grid differs")
    partition = c1.registered_sp4_partition_v1(
        total_tokens=packed["total_tokens"],
        condition_tokens=packed["source_tokens"],
        sp_rank=sp_rank,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        embedded = transformer.patch_embedding(packed["input_patches"]).flatten(1).unsqueeze(0)
    if tuple(embedded.shape) != (1, packed["total_tokens"], HIDDEN):
        fail("C2 pre-SP packed embedding geometry differs")
    packed["embedded"] = embedded
    with torch.autocast(device_type="cuda", enabled=False):
        memory = elal_handle.build_memory(latent)
    route = elal_module.ELAL3RouteV1(
        total_tokens=packed["total_tokens"],
        condition_tokens=packed["source_tokens"],
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=SP_SIZE,
        memory=memory,
        route_identity=route_identity,
    )
    audit_start = len(elal_handle.audit_records)
    with elal_handle.route(route):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction, timestep_receipt, text_lens_runtime_abi = predict_target_c2_v1(
                renderer=renderer,
                packed=packed,
                coordinate=coordinate,
                text_lens=text_lens,
                text_embs=text_embs,
            )
    forward_records = elal_handle.audit_records[audit_start:]
    try:
        hook_receipt = c1.hook_audit_v1(forward_records)
        event_mask = c1.pack_vae_partition_mask_v1(
            truth_event_mask_vae, target_velocity=packed["target_velocity"]
        )
        context_mask = c1.pack_vae_partition_mask_v1(
            truth_context_mask_vae, target_velocity=packed["target_velocity"]
        )
    except c1.ELAL3C1TrainingError as error:
        raise ELAL3C2TrainingError(str(error)) from error
    if any(value != 1 for value in hook_receipt["calls_by_block"].values()):
        fail("one branch did not execute each of all30 hooks exactly once")
    q_fields = (
        "q_local",
        "q_entity",
        "q_relation",
        "q_phase",
        "q_terminal",
        "q_camera",
        "entity_presence",
        "temporal_valid",
        "relation_valid",
        "phase_valid",
    )
    actual_q_rows = {
        name: {
            "shape": [int(item) for item in getattr(latent, name).shape],
            "dtype": str(getattr(latent, name).dtype),
            "sha256": c1.tensor_sha256_v1(
                getattr(latent, name).detach().contiguous().cpu()
            ),
        }
        for name in q_fields
    }
    registered_q_rows = label_receipt.get("q_tensor_rows")
    if registered_q_rows is not None:
        if actual_q_rows != registered_q_rows:
            fail("C2 branch actual oracle-q differs from authenticated label receipt")
        input_variant = label_receipt.get("media_variant")
        label_binding_digest = label_receipt.get("label_digest")
    else:
        fixed_proof = label_receipt.get("fixed_tensor_proof")
        opposite_proof = label_receipt.get("opposite_tensor_proof")
        if (
            not isinstance(fixed_proof, Mapping)
            or not isinstance(opposite_proof, Mapping)
            or any(
                actual_q_rows[name] != fixed_proof.get(name, {}).get("result")
                for name in (
                    "q_local",
                    "q_phase",
                    "q_terminal",
                    "q_camera",
                    "entity_presence",
                    "temporal_valid",
                    "relation_valid",
                    "phase_valid",
                )
            )
            or any(
                actual_q_rows[name] != opposite_proof.get(name, {}).get("result")
                for name in ("q_entity", "q_relation")
            )
        ):
            fail("C2 branch actual hybrid oracle-q differs from donor proof")
        input_variant = f"{label_receipt.get('matched_variant')}_role_mismatch"
        label_binding_digest = label_receipt.get("hybrid_digest")
    row_id = label_receipt.get("row_id")
    if row_id not in ROW_IDS or type(input_variant) is not str:
        fail("C2 branch row/variant receipt differs")
    label_binding_digest = _require_sha256_v1(
        label_binding_digest, label="C2 branch label binding"
    )
    sigma = float(coordinate.sigma)
    noisy_target = (
        (1.0 - sigma) * clean_target[0].contiguous()
        + sigma * epsilon[0].contiguous()
    ).contiguous()
    actual_input = {
        "row_id": row_id,
        "input_variant": input_variant,
        "label_binding_digest": label_binding_digest,
        "actual_q_tensor_rows": actual_q_rows,
        "actual_q_tensor_rows_digest": object_sha256(actual_q_rows),
        "source_sha256": c1.tensor_sha256_v1(source),
        "clean_target_sha256": c1.tensor_sha256_v1(clean_target),
        "epsilon_sha256": c1.tensor_sha256_v1(epsilon),
        "noisy_target_sha256": c1.tensor_sha256_v1(
            noisy_target.unsqueeze(0).contiguous()
        ),
        "target_velocity_sha256": c1.tensor_sha256_v1(
            (epsilon - clean_target).contiguous()
        ),
        "event_mask_vae_sha256": c1.tensor_sha256_v1(
            truth_event_mask_vae.detach().contiguous().cpu()
        ),
        "context_mask_vae_sha256": c1.tensor_sha256_v1(
            truth_context_mask_vae.detach().contiguous().cpu()
        ),
        "text_lens_runtime_abi": text_lens_runtime_abi,
        "text_lens_sha256": text_lens_runtime_abi[
            "audit_only_cpu_int64_tensor_sha256"
        ],
        "text_embs_sha256": c1.tensor_sha256_v1(
            text_embs.detach().contiguous().cpu()
        ),
        "coordinate": coordinate.as_dict(),
        "coordinate_kind": (
            "evaluation_sigma1"
            if isinstance(coordinate, EvaluationCoordinateV1)
            else "training_sigma_stratum"
        ),
        "renderer_timestep_receipt": dict(timestep_receipt),
        "route_identity": route_identity,
        "registered_sp4_partition": partition,
        "all30_hooks_used": True,
        "prediction_sha256": c1.tensor_sha256_v1(
            prediction.detach().contiguous().cpu()
        ),
        "packed_target_velocity_sha256": c1.tensor_sha256_v1(
            packed["target_velocity"].detach().contiguous().cpu()
        ),
        "packed_event_mask_sha256": c1.tensor_sha256_v1(
            event_mask.detach().contiguous().cpu()
        ),
        "packed_context_mask_sha256": c1.tensor_sha256_v1(
            context_mask.detach().contiguous().cpu()
        ),
    }
    actual_input = {
        **actual_input,
        "actual_input_digest": object_sha256(actual_input),
    }
    return {
        "prediction": prediction,
        "target_velocity": packed["target_velocity"],
        "event_mask": event_mask,
        "context_mask": context_mask,
        "packed": packed,
        "embedded": embedded,
        "memory": memory,
        "route": route,
        "route_identity": route_identity,
        "hook_receipt": hook_receipt,
        "registered_sp4_partition": partition,
        "timestep_receipt": timestep_receipt,
        "actual_input_receipt": actual_input,
    }


def validate_evaluation_forward_evidence_v1(
    evidence: Any,
    *,
    row_id: str,
    sp_rank: int,
    input_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    order = (
        "full_target",
        "full_role_swap",
        "mismatch_target",
        "mismatch_role_swap",
    )
    # This mapping is persisted through canonical ``sort_keys=True`` JSON.
    # Key insertion order is therefore not evidence; close the exact member
    # set and consume every member through the frozen semantic order below.
    if not isinstance(evidence, Mapping) or set(evidence) != set(order):
        fail("evaluation exact4 actual forward member closure differs")
    inputs: dict[str, Mapping[str, Any]] = {}
    for name in order:
        row = evidence[name]
        if not isinstance(row, Mapping) or set(row) != {
            "actual_input_receipt",
            "hook_receipt",
            "registered_sp4_partition",
            "timestep_receipt",
        }:
            fail("evaluation actual forward evidence field closure differs")
        actual = _validate_actual_branch_receipt_closed_v1(
            row["actual_input_receipt"], label=f"evaluation {row_id} {name}"
        )
        _validate_hook_receipt_closed_v1(
            row["hook_receipt"], label=f"evaluation {row_id} {name}"
        )
        _validate_partition_closed_v1(
            row["registered_sp4_partition"],
            sp_rank=sp_rank,
            label=f"evaluation {row_id} {name}",
        )
        if (
            actual["registered_sp4_partition"] != row["registered_sp4_partition"]
            or actual["renderer_timestep_receipt"] != row["timestep_receipt"]
            or actual["coordinate_kind"] != "evaluation_sigma1"
            or actual["coordinate"] != EvaluationCoordinateV1().as_dict()
            or actual["row_id"] != row_id
        ):
            fail("evaluation actual forward coordinate/hook join differs")
        inputs[name] = actual
    expected_variants = {
        "full_target": "target",
        "full_role_swap": "role_swap",
        "mismatch_target": "target_role_mismatch",
        "mismatch_role_swap": "role_swap_role_mismatch",
    }
    if any(inputs[name]["input_variant"] != variant for name, variant in expected_variants.items()):
        fail("evaluation actual forward route variants differ")
    fixed_all = (
        "row_id",
        "source_sha256",
        "epsilon_sha256",
        "noisy_target_sha256",
        "text_lens_runtime_abi",
        "text_lens_sha256",
        "text_embs_sha256",
        "coordinate",
        "coordinate_kind",
        "renderer_timestep_receipt",
        "registered_sp4_partition",
    )
    anchor = inputs["full_target"]
    if any(inputs[name][field] != anchor[field] for name in order[1:] for field in fixed_all):
        fail("evaluation exact4 source/noise/text/coordinate differs")
    if (
        anchor["source_sha256"] != input_payload.get("source_sha256")
        or anchor["epsilon_sha256"] != input_payload.get("epsilon_sha256")
        or inputs["full_target"]["clean_target_sha256"] != input_payload.get("target_sha256")
        or inputs["full_role_swap"]["clean_target_sha256"] != input_payload.get("role_swap_sha256")
        or inputs["full_target"]["label_binding_digest"] != input_payload.get("target_label_digest")
        or inputs["full_role_swap"]["label_binding_digest"] != input_payload.get("role_swap_label_digest")
        or inputs["full_target"]["actual_q_tensor_rows_digest"] != input_payload.get("target_q_digest")
        or inputs["full_role_swap"]["actual_q_tensor_rows_digest"] != input_payload.get("role_swap_q_digest")
        or inputs["mismatch_target"]["label_binding_digest"] != input_payload.get("target_mismatch_digest")
        or inputs["mismatch_role_swap"]["label_binding_digest"] != input_payload.get("role_swap_mismatch_digest")
    ):
        fail("evaluation exact4 input/label provenance join differs")
    for clean_variant in TRAIN_VARIANTS:
        matched = inputs[f"full_{clean_variant}"]
        mismatch = inputs[f"mismatch_{clean_variant}"]
        opposite = inputs[
            "full_role_swap" if clean_variant == "target" else "full_target"
        ]
        if any(
            matched[field] != mismatch[field]
            for field in (
                "clean_target_sha256",
                "target_velocity_sha256",
                "event_mask_vae_sha256",
                "context_mask_vae_sha256",
                "packed_target_velocity_sha256",
                "packed_event_mask_sha256",
                "packed_context_mask_sha256",
            )
        ):
            fail("evaluation role-only mismatch changed clean/velocity/masks")
        fixed_q = (
            "q_local",
            "q_phase",
            "q_terminal",
            "q_camera",
            "entity_presence",
            "temporal_valid",
            "relation_valid",
            "phase_valid",
        )
        if (
            any(
                matched["actual_q_tensor_rows"][field]
                != mismatch["actual_q_tensor_rows"][field]
                for field in fixed_q
            )
            or any(
                opposite["actual_q_tensor_rows"][field]
                != mismatch["actual_q_tensor_rows"][field]
                for field in ("q_entity", "q_relation")
            )
            or any(
                matched["actual_q_tensor_rows"][field]
                == mismatch["actual_q_tensor_rows"][field]
                for field in ("q_entity", "q_relation")
            )
        ):
            fail("evaluation role-only q donor/fixed-field proof differs")
    value = {
        "exact4_actual_renderer_forwards_closed": True,
        "full_q_and_role_only_q_donors_verified": True,
        "q_shortcut_forbidden": True,
        "evidence_digest": object_sha256(evidence),
    }
    return value


def validate_evaluation_observation_binding_v1(
    *,
    full_q_route: Any,
    role_only_cells: Any,
    actual_forward_evidence: Mapping[str, Any],
    row_id: str,
    stage: str,
) -> Mapping[str, Any]:
    inputs = {
        name: actual_forward_evidence[name]["actual_input_receipt"]
        for name in actual_forward_evidence
    }
    energy_keys = (
        "target_q__target",
        "target_q__role_swap",
        "role_swap_q__target",
        "role_swap_q__role_swap",
    )
    if (
        not isinstance(full_q_route, Mapping)
        or set(full_q_route)
        != {
            "claim_name",
            "participant_role_binding_claim_forbidden",
            "energies",
            "energy_input_bindings",
            "target_q_row_margin",
            "role_swap_q_row_margin",
            "signed_diagonal_margin",
            "strict_diagonal_dominance",
        }
        or full_q_route.get("claim_name") != "oracle_route_controllability_only"
        or full_q_route.get("participant_role_binding_claim_forbidden") is not True
        or not isinstance(full_q_route.get("energies"), Mapping)
        or set(full_q_route["energies"]) != set(energy_keys)
        or not isinstance(full_q_route.get("energy_input_bindings"), Mapping)
        or set(full_q_route["energy_input_bindings"]) != set(energy_keys)
        or any(not math.isfinite(float(full_q_route["energies"][key])) for key in energy_keys)
    ):
        fail("evaluation full-q scalar observation closure differs")
    expected_energy_inputs = {}
    for q_variant, q_name in (("target", "target_q"), ("role_swap", "role_swap_q")):
        prediction = inputs[f"full_{q_variant}"]
        for truth_variant in TRAIN_VARIANTS:
            truth = inputs[f"full_{truth_variant}"]
            expected_energy_inputs[f"{q_name}__{truth_variant}"] = {
                "prediction_sha256": prediction["prediction_sha256"],
                "packed_target_velocity_sha256": truth["packed_target_velocity_sha256"],
                "packed_event_mask_sha256": truth["packed_event_mask_sha256"],
                "packed_context_mask_sha256": truth["packed_context_mask_sha256"],
            }
    energies = full_q_route["energies"]
    target_margin = energies["target_q__role_swap"] - energies["target_q__target"]
    role_margin = energies["role_swap_q__target"] - energies["role_swap_q__role_swap"]
    if (
        full_q_route["energy_input_bindings"] != expected_energy_inputs
        or full_q_route.get("target_q_row_margin") != target_margin
        or full_q_route.get("role_swap_q_row_margin") != role_margin
        or full_q_route.get("signed_diagonal_margin") != min(target_margin, role_margin)
        or full_q_route.get("strict_diagonal_dominance")
        is not (target_margin > 0.0 and role_margin > 0.0)
    ):
        fail("evaluation full-q scalar/input binding differs")
    if not isinstance(role_only_cells, list) or len(role_only_cells) != 2:
        fail("evaluation role-only scalar exact2 differs")
    for cell, clean_variant in zip(role_only_cells, TRAIN_VARIANTS):
        matched = inputs[f"full_{clean_variant}"]
        mismatch = inputs[f"mismatch_{clean_variant}"]
        opposite_variant = "role_swap" if clean_variant == "target" else "target"
        opposite = inputs[f"full_{opposite_variant}"]
        binding = cell.get("input_bindings") if isinstance(cell, Mapping) else None
        cell_fields = {
            "row_id",
            "clean_variant",
            "opposite_variant",
            "matched_energy",
            "mismatch_energy",
            "margin_mismatch_minus_matched",
            "normalized_predicted_vs_clean_role_contrast",
            "positive_contrast_alignment",
            "positive_contrast_required_at_this_stage",
            "predicted_contrast_l2",
            "clean_contrast_l2",
            "input_bindings",
        }
        cosine = (
            cell.get("normalized_predicted_vs_clean_role_contrast")
            if isinstance(cell, Mapping)
            else None
        )
        if (
            not isinstance(cell, Mapping)
            or set(cell) != cell_fields
            or cell.get("row_id") != row_id
            or cell.get("clean_variant") != clean_variant
            or cell.get("opposite_variant") != opposite_variant
            or not isinstance(binding, Mapping)
            or binding.get("matched_prediction_sha256") != matched["prediction_sha256"]
            or binding.get("mismatch_prediction_sha256") != mismatch["prediction_sha256"]
            or binding.get("packed_clean_velocity_sha256") != matched["packed_target_velocity_sha256"]
            or binding.get("packed_opposite_velocity_sha256") != opposite["packed_target_velocity_sha256"]
            or binding.get("packed_event_mask_sha256") != matched["packed_event_mask_sha256"]
            or binding.get("packed_context_mask_sha256") != matched["packed_context_mask_sha256"]
            or _require_sha256_v1(
                binding.get("contrast_union_mask_sha256"),
                label="evaluation contrast union mask",
            )
            != binding.get("contrast_union_mask_sha256")
            or not math.isfinite(float(cell.get("matched_energy", math.nan)))
            or not math.isfinite(float(cell.get("mismatch_energy", math.nan)))
            or not math.isfinite(float(cell.get("predicted_contrast_l2", math.nan)))
            or not math.isfinite(float(cell.get("clean_contrast_l2", math.nan)))
            or cell.get("predicted_contrast_l2") < 0.0
            or cell.get("clean_contrast_l2") <= 0.0
            or (
                cosine is not None
                and (
                    not math.isfinite(float(cosine))
                    or not -1.0 <= float(cosine) <= 1.0
                )
            )
            or (stage == "step10" and cosine is None)
            or cell.get("margin_mismatch_minus_matched")
            != cell.get("mismatch_energy") - cell.get("matched_energy")
            or cell.get("positive_contrast_required_at_this_stage")
            is not (stage == "step10")
            or cell.get("positive_contrast_alignment")
            is not (
                cell.get("normalized_predicted_vs_clean_role_contrast") is not None
                and cell.get("normalized_predicted_vs_clean_role_contrast") > 0.0
            )
        ):
            fail("evaluation role-only scalar/input binding differs")
    value = {
        "full_q_and_role_only_scalars_bound_to_actual_forward_hashes": True,
        "margins_and_boolean_gates_recomputed": True,
        "observation_binding_digest": object_sha256(
            {"full_q_route": full_q_route, "role_only_cells": role_only_cells}
        ),
    }
    return value


def evaluate_local_row_v1(
    *,
    stage: str,
    arm_seed: int,
    row_index: int,
    row_id: str,
    source: Any,
    target_clean: Any,
    role_clean: Any,
    target_label: Any,
    role_label: Any,
    target_mismatch: Any,
    role_mismatch: Any,
    instruction: str,
    transformer: Any,
    renderer: Any,
    elal_handle: Any,
    elal_module: Any,
    rope: Any,
    device: Any,
    text_lens: Any,
    text_embs: Any,
    sp_rank: int,
) -> tuple[Mapping[str, Any], Any]:
    """Evaluate one row's full-q matrix and two role-only cells at sigma=1."""

    import torch

    if row_index not in (0, 1) or ROW_IDS[row_index] != row_id:
        fail("evaluation row coordinate differs")
    coordinate = EvaluationCoordinateV1()
    epsilon_seed = evaluation_noise_seed_v1(arm_seed, row_index)
    epsilon = cpu_epsilon_v1(epsilon_seed)
    epsilon_sha = c1.tensor_sha256_v1(epsilon)
    label_by_variant = {"target": target_label, "role_swap": role_label}
    mismatch_by_variant = {
        "target": target_mismatch,
        "role_swap": role_mismatch,
    }
    clean_by_variant = {"target": target_clean, "role_swap": role_clean}
    full: dict[str, Mapping[str, Any]] = {}
    mismatch_predictions: dict[str, Mapping[str, Any]] = {}
    invariant_receipts: dict[str, Mapping[str, Any]] = {}
    mask_names = (
        "event_mask_patch",
        "context_mask_patch",
        "event_mask_vae",
        "context_mask_vae",
        "role_amodal_mask_patch",
        "role_visible_mask_patch",
        "role_event_mask_patch",
        "role_event_mask_vae",
    )
    with torch.no_grad():
        for clean_variant in TRAIN_VARIANTS:
            label = label_by_variant[clean_variant]
            clean = clean_by_variant[clean_variant]
            full[clean_variant] = renderer_branch_forward_v1(
                transformer=transformer,
                renderer=renderer,
                elal_handle=elal_handle,
                elal_module=elal_module,
                source=source,
                clean_target=clean,
                epsilon=epsilon,
                coordinate=coordinate,
                oracle_label=label,
                rope=rope,
                device=device,
                text_lens=text_lens,
                text_embs=text_embs,
                sp_rank=sp_rank,
                route_identity=(
                    f"{row_id}:{stage}:full:{clean_variant}:sp{sp_rank}"
                ),
            )
            hybrid = mismatch_by_variant[clean_variant]
            opposite_variant = (
                "role_swap" if clean_variant == "target" else "target"
            )
            opposite = label_by_variant[opposite_variant]
            invariant_receipts[clean_variant] = {
                "row_id": row_id,
                "clean_variant": clean_variant,
                "evaluation_coordinate": coordinate.as_dict(),
                "invariant": role_only_swap_invariants_v1(
                    label.latent,
                    opposite.latent,
                    hybrid.latent,
                matched_masks={name: getattr(label, name) for name in mask_names},
                mismatch_masks={name: getattr(hybrid, name) for name in mask_names},
                matched_source=source,
                mismatch_source=source,
                matched_instruction=instruction,
                mismatch_instruction=instruction,
                matched_sigma=coordinate.sigma,
                mismatch_sigma=coordinate.sigma,
                matched_x_sigma=epsilon,
                mismatch_x_sigma=epsilon,
                matched_epsilon=epsilon,
                mismatch_epsilon=epsilon,
                matched_slot_entity_ids=label.receipt["slot_entity_ids"],
                opposite_slot_entity_ids=opposite.receipt["slot_entity_ids"],
                mismatch_slot_entity_ids=hybrid.receipt[
                    "opposite_slot_entity_ids"
                ],
                matched_slot_roles=label.receipt["slot_roles"],
                opposite_slot_roles=opposite.receipt["slot_roles"],
                mismatch_slot_roles=hybrid.receipt["opposite_slot_roles"],
                matched_role_code_order=label.receipt["role_code_order"],
                opposite_role_code_order=opposite.receipt["role_code_order"],
                    mismatch_role_code_order=opposite.receipt["role_code_order"],
                ),
            }
            mismatch_predictions[clean_variant] = renderer_branch_forward_v1(
                transformer=transformer,
                renderer=renderer,
                elal_handle=elal_handle,
                elal_module=elal_module,
                source=source,
                clean_target=clean,
                epsilon=epsilon,
                coordinate=coordinate,
                oracle_label=hybrid,
                rope=rope,
                device=device,
                text_lens=text_lens,
                text_embs=text_embs,
                sp_rank=sp_rank,
                route_identity=(
                    f"{row_id}:{stage}:role-only-mismatch:{clean_variant}:sp{sp_rank}"
                ),
            )
    timestep_rows = [
        branch["timestep_receipt"]
        for branch in (*full.values(), *mismatch_predictions.values())
    ]
    if any(
        row != {
            "timestep_cpu_origin": True,
            "timestep_dtype": "torch.int64",
            "timestep_value": 999,
            "sigma_float32_be_hex": "3f800000",
        }
        for row in timestep_rows
    ):
        fail("evaluation renderer timestep/sigma receipt differs")
    full_matrix = full_q_route_matrix_v1(
        prediction_target_q=full["target"]["prediction"],
        prediction_role_q=full["role_swap"]["prediction"],
        velocity_target=full["target"]["target_velocity"],
        velocity_role=full["role_swap"]["target_velocity"],
        event_target=full["target"]["event_mask"],
        context_target=full["target"]["context_mask"],
        event_role=full["role_swap"]["event_mask"],
        context_role=full["role_swap"]["context_mask"],
    )
    contrast_union = full["target"]["event_mask"] | full["role_swap"]["event_mask"]
    cells = []
    for clean_variant in TRAIN_VARIANTS:
        opposite_variant = "role_swap" if clean_variant == "target" else "target"
        cells.append(
            role_only_cell_v1(
                row_id=row_id,
                clean_variant=clean_variant,
                prediction_matched=full[clean_variant]["prediction"],
                prediction_mismatch=mismatch_predictions[clean_variant]["prediction"],
                velocity_clean=full[clean_variant]["target_velocity"],
                velocity_opposite=full[opposite_variant]["target_velocity"],
                event_clean=full[clean_variant]["event_mask"],
                context_clean=full[clean_variant]["context_mask"],
                contrast_union_mask=contrast_union,
                require_positive_contrast=stage == "step10",
            )
        )
    target_prediction = full["target"]["prediction"].detach().clone()
    input_payload = {
        "row_index": row_index,
        "row_id": row_id,
        "source_sha256": c1.tensor_sha256_v1(source),
        "target_sha256": c1.tensor_sha256_v1(target_clean),
        "role_swap_sha256": c1.tensor_sha256_v1(role_clean),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "epsilon_seed": epsilon_seed,
        "epsilon_sha256": epsilon_sha,
        "coordinate": coordinate.as_dict(),
        "target_label_digest": target_label.receipt["label_digest"],
        "role_swap_label_digest": role_label.receipt["label_digest"],
        "target_q_digest": target_label.receipt["q_tensor_rows_digest"],
        "role_swap_q_digest": role_label.receipt["q_tensor_rows_digest"],
        "target_mismatch_digest": target_mismatch.receipt["hybrid_digest"],
        "role_swap_mismatch_digest": role_mismatch.receipt["hybrid_digest"],
    }
    branch_lookup = {
        "full_target": full["target"],
        "full_role_swap": full["role_swap"],
        "mismatch_target": mismatch_predictions["target"],
        "mismatch_role_swap": mismatch_predictions["role_swap"],
    }
    actual_forward_evidence = {
        name: {
            "actual_input_receipt": branch["actual_input_receipt"],
            "hook_receipt": branch["hook_receipt"],
            "registered_sp4_partition": branch["registered_sp4_partition"],
            "timestep_receipt": branch["timestep_receipt"],
        }
        for name, branch in branch_lookup.items()
    }
    actual_forward_evidence_validation = validate_evaluation_forward_evidence_v1(
        actual_forward_evidence,
        row_id=row_id,
        sp_rank=sp_rank,
        input_payload=input_payload,
    )
    observation_binding_validation = validate_evaluation_observation_binding_v1(
        full_q_route=full_matrix,
        role_only_cells=cells,
        actual_forward_evidence=actual_forward_evidence,
        row_id=row_id,
        stage=stage,
    )
    return (
        {
            "stage": stage,
            "row_id": row_id,
            "input_payload": input_payload,
            "input_payload_digest": object_sha256(input_payload),
            "full_q_route": full_matrix,
            "role_only_cells": cells,
            "role_only_input_invariants": invariant_receipts,
            "actual_forward_evidence": actual_forward_evidence,
            "actual_forward_evidence_validation": actual_forward_evidence_validation,
            "observation_binding_validation": observation_binding_validation,
            "timestep_receipts": timestep_rows,
            "all_four_actual_shape_forwards_all30": True,
        },
        target_prediction,
    )


def _torch_tree_fingerprint_v1(value: Any) -> Mapping[str, Any]:
    """Canonical, byte-sensitive fingerprint for torch checkpoint trees."""

    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
            "sha256": c1.tensor_sha256_v1(tensor),
        }
    if isinstance(value, Mapping):
        rows = []
        for key, item in value.items():
            if type(key) not in (str, int, float, bool) and key is not None:
                fail("torch checkpoint mapping key ABI differs")
            rows.append(
                {
                    "key_type": type(key).__name__,
                    "key": key,
                    "value": _torch_tree_fingerprint_v1(item),
                }
            )
        rows.sort(key=lambda row: canonical_json_bytes([row["key_type"], row["key"]]))
        return {"kind": "mapping", "entries": rows}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_torch_tree_fingerprint_v1(item) for item in value],
        }
    if value is None or type(value) in (str, int, float, bool):
        return {"kind": "scalar", "type": type(value).__name__, "value": value}
    fail(f"unsupported torch checkpoint value type: {type(value).__name__}")


def _optimizer_state_inventory_v1(
    payload: Any,
    *,
    expected_parameter_inventory: Sequence[Mapping[str, Any]],
    expected_step: int,
) -> Mapping[str, Any]:
    import torch

    expected_parameter_count = 668
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"state", "param_groups"}
        or not isinstance(payload.get("state"), Mapping)
        or not isinstance(payload.get("param_groups"), list)
        or len(payload["param_groups"]) != 1
        or len(payload["state"]) != expected_parameter_count
        or len(expected_parameter_inventory) != expected_parameter_count
    ):
        fail("C2 AdamW exact state envelope differs")
    group = payload["param_groups"][0]
    if (
        not isinstance(group, Mapping)
        or set(group)
        != {
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "foreach",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
            "params",
        }
        or group.get("lr") != DEFAULT_LR
        or tuple(group.get("betas", ())) != (0.9, 0.95)
        or group.get("eps") != 1.0e-8
        or float(group.get("weight_decay", math.nan)) != 0.0
        or group.get("amsgrad") is not False
        or group.get("foreach") is not None
        or group.get("maximize") is not False
        or group.get("capturable") is not False
        or group.get("differentiable") is not False
        or group.get("fused") is not None
        or group.get("decoupled_weight_decay") is not True
        or group.get("params") != list(range(expected_parameter_count))
        or set(payload["state"]) != set(range(expected_parameter_count))
    ):
        fail("C2 AdamW fixed param-group/ID closure differs")
    rows = [
        {
            "parameter_id": parameter_id,
            "state_keys": sorted(str(key) for key in item),
        }
        for parameter_id, item in sorted(payload["state"].items())
        if isinstance(item, Mapping)
    ]
    if (
        len(rows) != expected_parameter_count
        or any(row["state_keys"] != ["exp_avg", "exp_avg_sq", "step"] for row in rows)
    ):
        fail("C2 AdamW per-parameter state closure differs")
    for parameter_id, parameter_row in enumerate(expected_parameter_inventory):
        state = payload["state"][parameter_id]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step_tensor = state["step"]
        expected_shape = tuple(parameter_row["shape"])
        expected_dtype = parameter_row["dtype"]
        if (
            not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or tuple(exp_avg.shape) != expected_shape
            or tuple(exp_avg_sq.shape) != expected_shape
            or str(exp_avg.dtype) != expected_dtype
            or str(exp_avg_sq.dtype) != expected_dtype
            or not bool(torch.isfinite(exp_avg).all().item())
            or not bool(torch.isfinite(exp_avg_sq).all().item())
            or not isinstance(step_tensor, torch.Tensor)
            or step_tensor.numel() != 1
            or not bool(torch.isfinite(step_tensor).all().item())
            or float(step_tensor.item()) != float(expected_step)
        ):
            fail("C2 AdamW state does not correspond to adapter parameter/step")
    exp_avg_nonzero = sum(
        bool(torch.count_nonzero(payload["state"][index]["exp_avg"]).item())
        for index in range(expected_parameter_count)
    )
    exp_avg_sq_nonzero = sum(
        bool(torch.count_nonzero(payload["state"][index]["exp_avg_sq"]).item())
        for index in range(expected_parameter_count)
    )
    if exp_avg_nonzero <= 0 or exp_avg_sq_nonzero <= 0:
        fail("C2 AdamW loaded moments are globally zero")
    tree_digest = object_sha256(_torch_tree_fingerprint_v1(payload))
    return {
        "state_entry_count": len(payload["state"]),
        "param_group_count": len(payload["param_groups"]),
        "parameter_count": expected_parameter_count,
        "parameter_inventory_digest": object_sha256(
            list(expected_parameter_inventory)
        ),
        "optimizer_step": expected_step,
        "exp_avg_nonzero_parameter_count": exp_avg_nonzero,
        "exp_avg_sq_nonzero_parameter_count": exp_avg_sq_nonzero,
        "state_keys_by_parameter": rows,
        "tree_digest": tree_digest,
    }


def _sealed_file_row_v1(path: Path, *, label: str) -> Mapping[str, Any]:
    """Double-hash one sealed regular file through one no-follow FD."""

    requested = path.expanduser()
    resolved = requested.resolve(strict=True)
    info = requested.lstat()
    if (
        not requested.is_absolute()
        or requested != resolved
        or requested.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
    ):
        fail(f"{label} is not one sealed absolute 0444 nlink1 file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(resolved), flags)
    try:
        before = os.fstat(descriptor)

        def hash_pass() -> tuple[str, int]:
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    return digest.hexdigest(), size
                digest.update(chunk)
                size += len(chunk)

        first_sha, first_size = hash_pass()
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_sha, second_size = hash_pass()
        after = os.fstat(descriptor)
        named_after = resolved.stat()
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        first_sha != second_sha
        or first_size != second_size
        or first_size != before.st_size
        or identity(info) != identity(before)
        or identity(before) != identity(after)
        or identity(before) != identity(named_after)
    ):
        fail(f"{label} held-FD identity/hash replay differs")
    return {
        "name": resolved.name,
        "path": str(resolved),
        "sha256": first_sha,
        "size": first_size,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
        "device": before.st_dev,
        "inode": before.st_ino,
        "held_fd_double_hash_verified": True,
        "named_identity_replayed": True,
    }


def _load_sealed_torch_payload_v1(
    path: Path, *, expected_row: Mapping[str, Any], label: str
) -> Any:
    """Deserialize only from the same held FD whose bytes are rehashed."""

    import torch

    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(resolved), flags)
    try:
        before = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            expected_row.get("device") != before.st_dev
            or expected_row.get("inode") != before.st_ino
            or expected_row.get("size") != before.st_size
            or expected_row.get("mode") != stat.S_IMODE(before.st_mode)
            or expected_row.get("nlink") != before.st_nlink
        ):
            fail(f"{label} held payload identity differs from sealed row")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_row.get("sha256"):
            fail(f"{label} held payload hash differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        os.lseek(descriptor, 0, os.SEEK_SET)
        replay = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            replay.update(chunk)
        after = os.fstat(descriptor)
        named_after = resolved.stat()
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        named_identity = (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_mode,
            named_after.st_nlink,
            named_after.st_size,
            named_after.st_mtime_ns,
            named_after.st_ctime_ns,
        )
        if (
            replay.hexdigest() != expected_row.get("sha256")
            or identity != after_identity
            or identity != named_identity
        ):
            fail(f"{label} held payload post-deserialize replay differs")
        return payload
    finally:
        os.close(descriptor)


def validate_checkpoint_record_v1(
    record: Mapping[str, Any],
    *,
    expected_step: int,
    expected_parameter_sha256: str,
    optimizer_required: bool,
    expected_common: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replay sealed checkpoint artifacts, including optimizer tensor bytes."""

    import torch

    fields = {
        "schema_version",
        "step",
        "path",
        "directory_entries",
        "directory_mode",
        "file_order",
        "files",
        "adapter_payload_tree_digest",
        "parameter_order",
        "parameter_inventory",
        "optimizer_payload_tree_digest",
        "optimizer_state_inventory",
        "checkpoint_receipt_digest",
        "trainable_parameter_sha256",
        "strict_adapter_roundtrip_pass",
        "strict_optimizer_roundtrip_pass",
        "strict_metadata_sealed_replay_pass",
        "strict_reload_pass",
    }
    expected_order = ["adapter-and-elal3.pt"]
    if optimizer_required:
        expected_order.append("optimizer.pt")
    expected_order.append("CHECKPOINT_RECEIPT.json")
    if (
        not isinstance(record, Mapping)
        or set(record) != fields
        or record.get("schema_version") != CHECKPOINT_SCHEMA
        or record.get("step") != expected_step
        or record.get("file_order") != expected_order
        or record.get("directory_entries") != expected_order
        or record.get("directory_mode") != 0o500
        or record.get("trainable_parameter_sha256") != expected_parameter_sha256
        or record.get("strict_adapter_roundtrip_pass") is not True
        or record.get("strict_optimizer_roundtrip_pass") is not optimizer_required
        or record.get("strict_metadata_sealed_replay_pass") is not True
        or record.get("strict_reload_pass") is not True
        or type(record.get("path")) is not str
    ):
        fail("C2 checkpoint record envelope differs")
    root = Path(record["path"])
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or root.is_symlink()
        or root.name != f"checkpoint-{expected_step:08d}"
        or stat.S_IMODE(root.stat().st_mode) != 0o500
    ):
        fail("C2 checkpoint sealed directory differs")
    rows = record.get("files")
    if not isinstance(rows, list) or len(rows) != len(expected_order):
        fail("C2 checkpoint exact file closure differs")
    try:
        actual_entries = sorted(item.name for item in root.iterdir())
    except OSError as error:
        raise ELAL3C2TrainingError("C2 checkpoint directory enumeration failed") from error
    if actual_entries != sorted(expected_order) or any(
        item.is_symlink() or not item.is_file() for item in root.iterdir()
    ):
        fail("C2 checkpoint directory contains an unregistered entry")
    replayed = [
        _sealed_file_row_v1(root / name, label=f"checkpoint {expected_step} {name}")
        for name in expected_order
    ]
    if canonical_json_bytes(replayed) != canonical_json_bytes(rows):
        fail("C2 checkpoint sealed file replay differs")
    adapter = _load_sealed_torch_payload_v1(
        root / "adapter-and-elal3.pt",
        expected_row=replayed[0],
        label=f"checkpoint {expected_step} adapter",
    )
    adapter_fields = {
        "schema_version",
        "step",
        "parameter_order",
        "lora_state",
        "elal3_full_w64_state",
        "teacher_forced_oracle_q",
        "formal_c2_authorized",
        "exact160_authorized",
        "scientific_claim_authorized",
        "real_video_claim_authorized",
        "source_instruction_inference",
        "resume_source",
    }
    parameter_order = adapter.get("parameter_order") if isinstance(adapter, Mapping) else None
    lora_state = adapter.get("lora_state") if isinstance(adapter, Mapping) else None
    elal_state = adapter.get("elal3_full_w64_state") if isinstance(adapter, Mapping) else None
    if (
        not isinstance(adapter, Mapping)
        or set(adapter) != adapter_fields
        or adapter.get("schema_version") != CHECKPOINT_SCHEMA
        or adapter.get("step") != expected_step
        or adapter.get("teacher_forced_oracle_q") is not True
        or adapter.get("formal_c2_authorized") is not False
        or adapter.get("exact160_authorized") is not False
        or adapter.get("scientific_claim_authorized") is not False
        or adapter.get("real_video_claim_authorized") is not False
        or adapter.get("source_instruction_inference") is not False
        or adapter.get("resume_source") is not False
        or not isinstance(parameter_order, list)
        or len(parameter_order) != 668
        or len(set(parameter_order)) != 668
        or not isinstance(lora_state, Mapping)
        or len(lora_state) != 480
        or not isinstance(elal_state, Mapping)
        or len(elal_state) != 188
        or set(parameter_order) != set(lora_state) | set(elal_state)
        or set(lora_state) & set(elal_state)
        or any(".lora_" not in name for name in lora_state)
        or any(".elal3_c0_v1." not in name for name in elal_state)
        or object_sha256(_torch_tree_fingerprint_v1(adapter))
        != record.get("adapter_payload_tree_digest")
    ):
        fail("C2 checkpoint adapter payload replay differs")
    loaded_named = [
        (name, lora_state[name] if name in lora_state else elal_state[name])
        for name in parameter_order
    ]
    parameter_inventory = c1.trainable_inventory_v1(loaded_named)
    if (
        parameter_order != record.get("parameter_order")
        or parameter_inventory != record.get("parameter_inventory")
        or c1.trainable_digest_v1(loaded_named) != expected_parameter_sha256
    ):
        fail("C2 checkpoint adapter tensor bytes/name order differ from trainables")
    optimizer_digest = None
    if optimizer_required:
        optimizer_payload = _load_sealed_torch_payload_v1(
            root / "optimizer.pt",
            expected_row=replayed[1],
            label=f"checkpoint {expected_step} optimizer",
        )
        optimizer_inventory = _optimizer_state_inventory_v1(
            optimizer_payload,
            expected_parameter_inventory=parameter_inventory,
            expected_step=expected_step,
        )
        optimizer_digest = optimizer_inventory["tree_digest"]
        if (
            optimizer_digest != record.get("optimizer_payload_tree_digest")
            or optimizer_inventory != record.get("optimizer_state_inventory")
        ):
            fail("C2 checkpoint optimizer payload replay differs")
    elif (
        record.get("optimizer_payload_tree_digest") is not None
        or record.get("optimizer_state_inventory") is not None
    ):
        fail("step0 checkpoint unexpectedly binds optimizer state")
    metadata_row = replayed[-1]
    metadata = _read_sealed_json_held_fd_v1(
        root / "CHECKPOINT_RECEIPT.json",
        expected_sha256=metadata_row["sha256"],
        label=f"checkpoint {expected_step} metadata",
    )
    unsigned = dict(metadata)
    receipt_digest = unsigned.pop("receipt_digest", None)
    expected_metadata_fields = set(expected_common) | {
        "schema_version",
        "step",
        "adapter_file",
        "adapter_sha256",
        "optimizer_file",
        "optimizer_sha256",
        "adapter_payload_tree_digest",
        "parameter_order",
        "parameter_inventory",
        "optimizer_payload_tree_digest",
        "optimizer_state_inventory",
        "strict_weights_only_reload_verified",
        "strict_optimizer_state_roundtrip_verified",
        "trainable_parameter_sha256",
        "resume_source",
        "receipt_digest",
    }
    if (
        set(metadata) != expected_metadata_fields
        or any(metadata.get(key) != value for key, value in expected_common.items())
        or
        receipt_digest != record.get("checkpoint_receipt_digest")
        or receipt_digest != object_sha256(unsigned)
        or metadata.get("schema_version") != CHECKPOINT_SCHEMA
        or metadata.get("step") != expected_step
        or metadata.get("adapter_sha256") != replayed[0]["sha256"]
        or metadata.get("optimizer_sha256")
        != (replayed[1]["sha256"] if optimizer_required else None)
        or metadata.get("trainable_parameter_sha256")
        != expected_parameter_sha256
        or metadata.get("adapter_payload_tree_digest")
        != record.get("adapter_payload_tree_digest")
        or metadata.get("parameter_order") != parameter_order
        or metadata.get("parameter_inventory") != parameter_inventory
        or metadata.get("optimizer_payload_tree_digest") != optimizer_digest
        or metadata.get("strict_optimizer_state_roundtrip_verified")
        is not optimizer_required
        or metadata.get("strict_weights_only_reload_verified") is not True
        or metadata.get("resume_source") is not False
    ):
        fail("C2 checkpoint metadata self/release binding differs")
    return record


def save_checkpoint_v1(
    *,
    root: Path,
    step: int,
    named: Sequence[tuple[str, Any]],
    optimizer: Any,
    common: Mapping[str, Any],
    save_optimizer: bool,
) -> Mapping[str, Any]:
    """Create-only C2 checkpoint; never emit a C1/formal/source claim."""

    import torch

    final = root / f"checkpoint-{step:08d}"
    if final.exists() or final.is_symlink():
        fail(f"refusing to overwrite C2 checkpoint: {final}")
    final.mkdir(mode=0o700)
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in named}
    lora = {name: value for name, value in state.items() if ".lora_" in name}
    elal = {name: value for name, value in state.items() if ".elal3_c0_v1." in name}
    if len(lora) != 480 or len(elal) != 188 or set(lora) | set(elal) != set(state):
        fail("C2 checkpoint LoRA/ELAL partition differs")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "parameter_order": list(state),
        "lora_state": lora,
        "elal3_full_w64_state": elal,
        "teacher_forced_oracle_q": True,
        "formal_c2_authorized": False,
        "exact160_authorized": False,
        "scientific_claim_authorized": False,
        "real_video_claim_authorized": False,
        "source_instruction_inference": False,
        "resume_source": False,
    }
    adapter_payload_tree_digest = object_sha256(
        _torch_tree_fingerprint_v1(payload)
    )
    adapter_path = final / "adapter-and-elal3.pt"
    try:
        c1.create_only_torch_save(adapter_path, payload)
    except c1.ELAL3C1TrainingError as error:
        raise ELAL3C2TrainingError(str(error)) from error
    loaded = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if object_sha256(_torch_tree_fingerprint_v1(loaded)) != adapter_payload_tree_digest:
        fail("strict C2 checkpoint reload differs")
    optimizer_path: Optional[Path] = None
    optimizer_payload_tree_digest: Optional[str] = None
    optimizer_state_inventory: Optional[Mapping[str, Any]] = None
    parameter_inventory = c1.trainable_inventory_v1(named)
    if save_optimizer:
        optimizer_path = final / "optimizer.pt"
        live_optimizer_state = optimizer.state_dict()
        live_optimizer_fingerprint = _torch_tree_fingerprint_v1(
            live_optimizer_state
        )
        optimizer_payload_tree_digest = object_sha256(live_optimizer_fingerprint)
        optimizer_state_inventory = _optimizer_state_inventory_v1(
            live_optimizer_state,
            expected_parameter_inventory=parameter_inventory,
            expected_step=step,
        )
        c1.create_only_torch_save(optimizer_path, live_optimizer_state)
        loaded_optimizer = torch.load(
            optimizer_path, map_location="cpu", weights_only=True
        )
        if (
            not isinstance(loaded_optimizer, Mapping)
            or set(loaded_optimizer) != {"state", "param_groups"}
            or object_sha256(_torch_tree_fingerprint_v1(loaded_optimizer))
            != optimizer_payload_tree_digest
            or _optimizer_state_inventory_v1(
                loaded_optimizer,
                expected_parameter_inventory=parameter_inventory,
                expected_step=step,
            )
            != optimizer_state_inventory
        ):
            fail("strict C2 optimizer checkpoint reload differs")
    metadata = {
        **dict(common),
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "adapter_file": adapter_path.name,
        "adapter_sha256": file_sha256(adapter_path),
        "optimizer_file": optimizer_path.name if optimizer_path else None,
        "optimizer_sha256": file_sha256(optimizer_path) if optimizer_path else None,
        "adapter_payload_tree_digest": adapter_payload_tree_digest,
        "parameter_order": list(state),
        "parameter_inventory": parameter_inventory,
        "optimizer_payload_tree_digest": optimizer_payload_tree_digest,
        "optimizer_state_inventory": optimizer_state_inventory,
        "strict_weights_only_reload_verified": True,
        "strict_optimizer_state_roundtrip_verified": save_optimizer,
        "trainable_parameter_sha256": c1.trainable_digest_v1(named),
        "resume_source": False,
    }
    metadata_path = final / "CHECKPOINT_RECEIPT.json"
    checkpoint_receipt_digest = object_sha256(metadata)
    c1.atomic_create_json(
        metadata_path, {**metadata, "receipt_digest": checkpoint_receipt_digest}
    )
    os.chmod(final, 0o500)
    file_order = [adapter_path.name]
    if optimizer_path is not None:
        file_order.append(optimizer_path.name)
    file_order.append(metadata_path.name)
    record = {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "path": str(final),
        "file_order": file_order,
        "directory_entries": file_order,
        "directory_mode": 0o500,
        "files": [
            _sealed_file_row_v1(final / name, label=f"checkpoint {step} {name}")
            for name in file_order
        ],
        "adapter_payload_tree_digest": adapter_payload_tree_digest,
        "parameter_order": list(state),
        "parameter_inventory": parameter_inventory,
        "optimizer_payload_tree_digest": optimizer_payload_tree_digest,
        "optimizer_state_inventory": optimizer_state_inventory,
        "checkpoint_receipt_digest": checkpoint_receipt_digest,
        "trainable_parameter_sha256": metadata["trainable_parameter_sha256"],
        "strict_adapter_roundtrip_pass": True,
        "strict_optimizer_roundtrip_pass": save_optimizer,
        "strict_metadata_sealed_replay_pass": True,
        "strict_reload_pass": True,
    }
    return validate_checkpoint_record_v1(
        record,
        expected_step=step,
        expected_parameter_sha256=metadata["trainable_parameter_sha256"],
        optimizer_required=save_optimizer,
        expected_common=common,
    )


def _portable_checkpoint_record_v1(record: Mapping[str, Any]) -> Mapping[str, Any]:
    file_rows = [
        {
            key: row[key]
            for key in (
                "name",
                "sha256",
                "size",
                "mode",
                "nlink",
                "held_fd_double_hash_verified",
                "named_identity_replayed",
            )
        }
        for row in record["files"]
    ]
    value = {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": record["step"],
        "file_order": record["file_order"],
        "directory_entries": record["directory_entries"],
        "directory_mode": record["directory_mode"],
        "files": file_rows,
        "adapter_payload_tree_digest": record["adapter_payload_tree_digest"],
        "parameter_order": record["parameter_order"],
        "parameter_inventory": record["parameter_inventory"],
        "optimizer_payload_tree_digest": record["optimizer_payload_tree_digest"],
        "optimizer_state_inventory": record["optimizer_state_inventory"],
        "checkpoint_receipt_digest": record["checkpoint_receipt_digest"],
        "trainable_parameter_sha256": record["trainable_parameter_sha256"],
        "strict_reload_pass": record["strict_reload_pass"],
    }
    return {**value, "portable_record_digest": object_sha256(value)}


def seal_and_validate_checkpoint_tree_v1(
    root: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    expected_steps: Sequence[int],
    expected_parameter_sha256_by_step: Mapping[int, str],
    expected_common: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Seal the parent and replay the exact checkpoint tree before publication."""

    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        fail("C2 checkpoint parent path differs")
    if [record.get("step") for record in records] != list(expected_steps):
        fail("C2 checkpoint record step order differs")
    expected_entries = [f"checkpoint-{step:08d}" for step in expected_steps]
    os.chmod(root, 0o500)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open(str(root), flags)
    try:
        before = os.fstat(descriptor)
        entries_before = sorted(os.listdir(descriptor))
        if (
            stat.S_IMODE(before.st_mode) != 0o500
            or entries_before != sorted(expected_entries)
        ):
            fail("C2 checkpoint parent exact sealed entries differ")
        replayed = [
            validate_checkpoint_record_v1(
                record,
                expected_step=step,
                expected_parameter_sha256=expected_parameter_sha256_by_step[step],
                optimizer_required=step != 0,
                expected_common=expected_common,
            )
            for record, step in zip(records, expected_steps)
        ]
        entries_after = sorted(os.listdir(descriptor))
        after = os.fstat(descriptor)
        named_after = root.stat()
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_rdev,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        entries_after != entries_before
        or identity(before) != identity(after)
        or identity(before) != identity(named_after)
    ):
        fail("C2 checkpoint parent changed during held-dir replay")
    portable = [_portable_checkpoint_record_v1(record) for record in replayed]
    fixed = {
        "schema_version": "bernini-elal3-c2-sealed-checkpoint-tree-v1",
        "expected_steps": list(expected_steps),
        "directory_entries": expected_entries,
        "directory_mode": 0o500,
        "portable_checkpoint_records": portable,
        "portable_checkpoint_tree_digest": object_sha256(portable),
        "physical_origin_replay_passed": True,
    }
    return {
        **fixed,
        "origin_path": str(root),
        "origin_device": before.st_dev,
        "origin_inode": before.st_ino,
        "tree_binding_digest": object_sha256(fixed),
    }


def require_bundle_release_literals_v1(
    *,
    expected_bundle_sha256: str,
    expected_receipt_sha256: str,
    expected_run_complete_sha256: str,
) -> None:
    if any(
        value is None
        for value in (
            LATENT_BUNDLE_SHA256,
            LATENT_BUNDLE_SIZE,
            LATENT_BUNDLE_RECEIPT_SHA256,
            LATENT_BUNDLE_RECEIPT_SIZE,
            LATENT_BUNDLE_RECEIPT_DIGEST,
            MATERIALIZER_RUN_COMPLETE_SHA256,
            MATERIALIZER_RUN_COMPLETE_SIZE,
            MATERIALIZER_RUN_COMPLETE_DIGEST,
        )
    ):
        fail("exact16 C2 bundle release literals are not frozen; no model was loaded")
    if (
        expected_bundle_sha256 != LATENT_BUNDLE_SHA256
        or expected_receipt_sha256 != LATENT_BUNDLE_RECEIPT_SHA256
        or expected_run_complete_sha256 != MATERIALIZER_RUN_COMPLETE_SHA256
    ):
        fail("exact16 C2 retry2 release CLI pins differ from literals")


def source_pin_map_v1(source_closure: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = source_closure.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "c2_trainer",
        "c1_trainer",
        "elal3_core",
        "c2_label",
        "c2_materializer",
        "train_lora",
        "packed_lora",
        "world8_runtime",
        "sigma_strata",
    }:
        fail("runtime source pin map closure differs")
    relative_paths = {
        "c2_trainer": "train_elal3_c2_simulator_role_pair_v1.py",
        "c1_trainer": "train_elal3_c1_simulator_overfit_v1.py",
        "elal3_core": "elal3_c0_v1.py",
        "c2_label": "elal3_simulator_c2_label_v1.py",
        "c2_materializer": "materialize_elal3_simulator_c2_vae_v1.py",
        "train_lora": "train_lora.py",
        "packed_lora": "packed_preservation_lora_v2.py",
        "world8_runtime": "source_self_runtime.py",
        "sigma_strata": "inference_sigma_strata.py",
    }
    result: dict[str, Any] = {}
    for name, row in sources.items():
        if not isinstance(row, Mapping):
            fail("runtime source pin row ABI differs")
        result[name] = {
            "relative_path": relative_paths[name],
            "sha256": _require_sha256_v1(
                row.get("sha256"), label=f"{name} source SHA"
            ),
            "size": row.get("size"),
            "mode": row.get("mode"),
            "nlink": row.get("nlink"),
            "held_fd_double_hash_verified": row.get(
                "held_fd_double_hash_verified"
            ),
            "held_openat_parent_chain_replayed": row.get(
                "held_openat_parent_chain_replayed"
            ),
            "actual_imported_module_file_verified": row.get(
                "actual_imported_module_file_verified"
            ),
        }
    if len(result) != 9 or any(
        type(row["relative_path"]) is not str
        or not row["relative_path"]
        or type(row["size"]) is not int
        or row["size"] <= 0
        or row["mode"] != 0o444
        or row["nlink"] != 1
        or row["held_fd_double_hash_verified"] is not True
        or row["held_openat_parent_chain_replayed"] is not True
        or row["actual_imported_module_file_verified"] is not True
        for row in result.values()
    ):
        fail("runtime source pin map row ABI differs")
    value = {
        "source_count": 9,
        "sources": result,
        "all_modes": "0444",
        "all_nlink1_no_follow_held_openat_double_hash": True,
        "actual_imported_module_files_verified": True,
        "callable_ownership_verified": source_closure.get(
            "callable_ownership_verified"
        ),
        "runtime_absolute_paths_devices_inodes_excluded": True,
    }
    if value["callable_ownership_verified"] is not True:
        fail("runtime source callable ownership receipt differs")
    return {**value, "release_pin_digest": object_sha256(value)}


def optimizer_recipe_v1() -> Mapping[str, Any]:
    return {
        "class": "torch.optim.AdamW",
        "learning_rate": DEFAULT_LR,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
        "allowed_completed_steps": [1, MAX_STEPS],
        "optimizer_state_before_first_update": "empty",
        "resume": False,
    }


def recipe_version_digest_v1(
    *,
    bundle_sha256: str,
    source_pins: Mapping[str, Any],
    checkpoint_exact23_binding_digest: str,
    bernini_execution_source_binding_digest: str,
) -> str:
    return object_sha256(
        {
            "method": METHOD,
            "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
            "external_authority_sha256": EXTERNAL_AUTHORITY_SHA256,
            "model_authority_sha256": MODEL_AUTHORITY_SHA256,
            "latent_bundle_sha256": bundle_sha256,
            "materializer_run_complete_sha256": (
                MATERIALIZER_RUN_COMPLETE_SHA256
            ),
            "materializer_run_complete_digest": (
                MATERIALIZER_RUN_COMPLETE_DIGEST
            ),
            "checkpoint_exact23_manifest_sha256": (
                CHECKPOINT_EXACT23_MANIFEST_SHA256
            ),
            "checkpoint_exact23_binding_digest": (
                checkpoint_exact23_binding_digest
            ),
            "bernini_execution_source_binding_digest": (
                bernini_execution_source_binding_digest
            ),
            "source_pins": dict(source_pins),
            "world_size": WORLD_SIZE,
            "sp_size": SP_SIZE,
            "dp_size": DP_SIZE,
            "lora_affines": LORA_AFFINES,
            "lora_rank": LORA_RANK,
            "elal_variant": "full-w64",
            "controlled_gain_float32_be_hex": CONTROLLED_GAIN_FLOAT32_HEX,
            "optimizer": optimizer_recipe_v1(),
        }
    )


def memory_receipt_v1(device: Any, *, world_rank: int) -> Mapping[str, Any]:
    import torch

    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = allocated / float(total)
    return {
        "world_rank": world_rank,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "device_total_bytes": total,
        "peak_allocated_fraction": fraction,
        "strictly_greater_than_half": fraction > MEMORY_FRACTION_GATE,
        "dummy_or_padding_allocations": False,
    }


def _validate_hook_receipt_closed_v1(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"all30_used", "calls_by_block"}
        or value.get("all30_used") is not True
        or value.get("calls_by_block")
        != {str(index): 1 for index in range(BLOCKS)}
    ):
        fail(f"{label} all30 hook closure differs")


def _validate_partition_closed_v1(value: Any, *, sp_rank: int, label: str) -> None:
    expected = {
        "sp_rank": sp_rank,
        "local_start": sp_rank * c1.LOCAL_SP_TOKENS,
        "local_stop": (sp_rank + 1) * c1.LOCAL_SP_TOKENS,
        "local_tokens": c1.LOCAL_SP_TOKENS,
        "source_only": sp_rank in (0, 1),
        "target_only": sp_rank in (2, 3),
    }
    if value != expected:
        fail(f"{label} registered SP4 partition differs")


def _validate_objective_receipt_closed_v1(
    value: Any, *, arm_id: str, label: str
) -> None:
    duplicate = arm_id == ARM_DUPLICATE
    expected_recipe = (
        "target_duplicate_exact2" if duplicate else "target_and_role_swap_exact2"
    )
    expected_branches = (
        ["target", "target_exact_duplicate"]
        if duplicate
        else ["target", "role_swap"]
    )
    expected_fields = {
        "arm_id",
        "recipe",
        "branch_names",
        "branch_losses",
        "branch_reduction",
        "fixed_branch_coefficients",
        "shared_epsilon_bit_exact",
        "first_forward_identity",
        "second_forward_identity",
        "first_actual_input_digest",
        "second_actual_input_digest",
        "actual_branch_inputs_closed_and_verified",
        "two_distinct_all30_forward_executions",
        "duplicate_control",
        "duplicate_prediction_target_velocity_and_masks_bit_exact",
        "paired_role_supervision",
        "tunable_loss_weights",
        "frozen_teacher_used",
        "frozen_velocity_reference_used",
        "reward_used",
        "total_loss",
        "execution_mode",
        "branch_execution_schedule",
        "coefficient_applied_before_each_backward",
        "first_graph_released_before_second_forward",
        "simultaneous_live_autograd_branch_graphs_maximum",
        "detached_portable_branch_receipts_before_graph_release",
        "gradient_reduce_clip_optimizer_after_both_branches",
        "preflight_backward_executed",
    }
    execution_mode = value.get("execution_mode") if isinstance(value, Mapping) else None
    training = execution_mode == "training_forward_backward"
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("arm_id") != arm_id
        or value.get("recipe") != expected_recipe
        or value.get("branch_names") != expected_branches
        or value.get("branch_reduction") != "strict_arithmetic_mean"
        or value.get("fixed_branch_coefficients") != [0.5, 0.5]
        or value.get("shared_epsilon_bit_exact") is not True
        or value.get("actual_branch_inputs_closed_and_verified") is not True
        or value.get("two_distinct_all30_forward_executions") is not True
        or value.get("duplicate_control") is not duplicate
        or value.get("duplicate_prediction_target_velocity_and_masks_bit_exact")
        is not duplicate
        or value.get("paired_role_supervision") is not (not duplicate)
        or value.get("tunable_loss_weights") is not False
        or value.get("frozen_teacher_used") is not False
        or value.get("frozen_velocity_reference_used") is not False
        or value.get("reward_used") is not False
        or execution_mode
        not in {"preflight_forward_only", "training_forward_backward"}
        or value.get("branch_execution_schedule")
        != (
            "strict_sequential_forward_backward_release_then_next"
            if training
            else "strict_sequential_grad_enabled_forward_release_then_next_no_backward"
        )
        or value.get("coefficient_applied_before_each_backward") is not training
        or value.get("first_graph_released_before_second_forward") is not True
        or value.get("simultaneous_live_autograd_branch_graphs_maximum") != 1
        or value.get("detached_portable_branch_receipts_before_graph_release")
        is not True
        or value.get("gradient_reduce_clip_optimizer_after_both_branches")
        is not training
        or value.get("preflight_backward_executed") is not False
        or type(value.get("first_forward_identity")) is not str
        or type(value.get("second_forward_identity")) is not str
        or value.get("first_forward_identity") == value.get("second_forward_identity")
        or not math.isfinite(float(value.get("total_loss", math.nan)))
    ):
        fail(f"{label} exact2 objective closure differs")
    _require_sha256_v1(
        value.get("first_actual_input_digest"),
        label=f"{label} first actual input digest",
    )
    _require_sha256_v1(
        value.get("second_actual_input_digest"),
        label=f"{label} second actual input digest",
    )
    losses = value.get("branch_losses")
    if not isinstance(losses, list) or len(losses) != 2:
        fail(f"{label} exact2 branch loss closure differs")
    for index, row in enumerate(losses):
        branch_fields = {
            "objective",
            "event_loss",
            "context_loss",
            "total_loss",
            "event_elements",
            "context_elements",
            "fixed_partition_coefficients",
            "tunable_loss_weights",
            "simulator_signed_motion_used_as_diffusion_velocity",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != branch_fields
            or row.get("objective")
            != "bernini_fm_target_velocity_target_event_context_equal_partition_mean"
            or row.get("fixed_partition_coefficients") != [0.5, 0.5]
            or row.get("tunable_loss_weights") is not False
            or row.get("simulator_signed_motion_used_as_diffusion_velocity")
            is not False
            or type(row.get("event_elements")) is not int
            or row.get("event_elements") <= 0
            or type(row.get("context_elements")) is not int
            or row.get("context_elements") <= 0
            or row.get("event_elements") + row.get("context_elements")
            != 21 * 26 * 35 * c1.PATCH_VALUES
            or any(
                not math.isfinite(float(row.get(field, math.nan)))
                for field in ("event_loss", "context_loss", "total_loss")
            )
        ):
            fail(f"{label} branch {index} loss closure differs")
        event = float(row["event_loss"])
        context = float(row["context_loss"])
        expected_branch_total = struct.unpack(
            ">f",
            struct.pack(
                ">f",
                struct.unpack(">f", struct.pack(">f", event + context))[0] * 0.5,
            ),
        )[0]
        if struct.pack(">f", float(row["total_loss"])) != struct.pack(
            ">f", expected_branch_total
        ):
            fail(f"{label} branch {index} arithmetic mean differs")
    expected_total = struct.unpack(
        ">f",
        struct.pack(
            ">f",
            struct.unpack(
                ">f", struct.pack(">f", float(losses[0]["total_loss"]) + float(losses[1]["total_loss"]))
            )[0]
            * 0.5,
        ),
    )[0]
    if struct.pack(">f", float(value["total_loss"])) != struct.pack(">f", expected_total):
        fail(f"{label} exact2 arithmetic mean differs")


def _validate_memory_row_closed_v1(
    row: Any, *, expected_world_rank: int, label: str
) -> None:
    if (
        not isinstance(row, Mapping)
        or row.get("world_rank") != expected_world_rank
        or type(row.get("peak_allocated_bytes")) is not int
        or row.get("peak_allocated_bytes") <= 0
        or type(row.get("peak_reserved_bytes")) is not int
        or row.get("peak_reserved_bytes") < row.get("peak_allocated_bytes")
        or type(row.get("device_total_bytes")) is not int
        or row.get("device_total_bytes") <= 0
        or not math.isfinite(float(row.get("peak_allocated_fraction", math.nan)))
        or row.get("peak_allocated_fraction")
        != row.get("peak_allocated_bytes") / float(row.get("device_total_bytes"))
        or row.get("peak_allocated_fraction") <= MEMORY_FRACTION_GATE
        or row.get("strictly_greater_than_half") is not True
        or row.get("dummy_or_padding_allocations") is not False
    ):
        fail(f"{label} true allocated-memory closure differs")


def _validate_all8_graph_rows_closed_v1(
    rows: Any,
    *,
    arm_id: str,
    completed_step: Optional[int],
    label: str,
    expected_common_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        fail(f"{label} exact all8 graph row count differs")
    for world_rank, row in enumerate(rows):
        row_index = world_rank // SP_SIZE
        sp_rank = world_rank % SP_SIZE
        if (
            not isinstance(row, Mapping)
            or row.get("world_rank") != world_rank
            or row.get("row_index") != row_index
            or row.get("row_id") != ROW_IDS[row_index]
            or row.get("sp_rank") != sp_rank
        ):
            fail(f"{label} rank/DP2xSP4 placement differs")
        _validate_hook_receipt_closed_v1(row.get("first_hook"), label=label)
        _validate_hook_receipt_closed_v1(row.get("second_hook"), label=label)
        _validate_partition_closed_v1(
            row.get("first_partition"), sp_rank=sp_rank, label=label
        )
        _validate_partition_closed_v1(
            row.get("second_partition"), sp_rank=sp_rank, label=label
        )
        _validate_objective_receipt_closed_v1(
            row.get("objective"), arm_id=arm_id, label=label
        )
        execution_mode = (
            "preflight_forward_only"
            if completed_step is None
            else "training_forward_backward"
        )
        lifecycle = validate_branch_lifecycle_receipt_v1(
            row.get("branch_lifecycle"),
            execution_mode=execution_mode,
            label=label,
        )
        first_input = row.get("first_actual_input_receipt")
        second_input = row.get("second_actual_input_receipt")
        _validate_actual_branch_pair_closed_v1(
            first_input,
            second_input,
            arm_id=arm_id,
            label=f"{label} portable actual branch pair",
        )
        expected_step_zero = 0 if completed_step is None else completed_step - 1
        if (
            first_input.get("coordinate") != TRAINING_SIGMA_EXACT10[expected_step_zero]
            or first_input.get("coordinate_kind") != "training_sigma_stratum"
            or row["objective"].get("first_actual_input_digest")
            != first_input.get("actual_input_digest")
            or row["objective"].get("second_actual_input_digest")
            != second_input.get("actual_input_digest")
            or row["objective"].get("actual_branch_inputs_closed_and_verified")
            is not True
            or row["objective"].get("execution_mode") != execution_mode
            or lifecycle.get("execution_mode") != execution_mode
        ):
            fail(f"{label} actual branch input/objective join differs")
        if expected_common_payload is not None:
            row_inputs = expected_common_payload["row_common_target_inputs"]
            schedule = expected_common_payload["common_target_branch_schedule"]
            schedule_row = schedule["training_exact10_common_target_branch"][
                expected_step_zero * len(ROW_IDS) + row_index
            ]
            if any(
                first_input[field] != expected
                for field, expected in (
                    ("source_sha256", row_inputs[row_index]["source_tensor_sha256"]),
                    ("clean_target_sha256", row_inputs[row_index]["target_tensor_sha256"]),
                    ("epsilon_sha256", schedule_row["epsilon_sha256"]),
                    ("noisy_target_sha256", schedule_row["noisy_target_sha256"]),
                    ("target_velocity_sha256", schedule_row["target_velocity_sha256"]),
                    ("coordinate", schedule_row["sigma_coordinate"]),
                )
            ):
                fail(f"{label} actual target branch differs from sealed common schedule")
        if completed_step is None:
            if row.get("optimizer_constructed") is not False:
                fail(f"{label} preflight unexpectedly constructed optimizer")
            _validate_memory_row_closed_v1(
                row.get("memory"), expected_world_rank=world_rank, label=label
            )
            continue
        target_owner = sp_rank in (2, 3)
        if (
            row.get("target_owner") is not target_owner
            or not math.isfinite(
                float(row.get("local_elal_gradient_norm_before_reduction", math.nan))
            )
            or (
                target_owner
                and row.get("local_elal_gradient_norm_before_reduction") <= 0.0
            )
            or (
                not target_owner
                and row.get("local_elal_gradient_norm_before_reduction") != 0.0
            )
        ):
            fail(f"{label} local SP4 gradient ownership differs")
        gradient = row.get("gradient_audit")
        block_norms = gradient.get("post_sp_reduction_elal_block_norms") if isinstance(gradient, Mapping) else None
        if (
            not isinstance(gradient, Mapping)
            or gradient.get("completed_step") != completed_step
            or gradient.get("sp_rank") != sp_rank
            or gradient.get("local_target_owner") is not target_owner
            or gradient.get("all_local_gradients_present_finite_before_and_after_reduction")
            is not True
            or gradient.get("source_only_sp_graph_zero_installed") is not True
            or gradient.get("post_sp_reduction_all30_nonzero") is not True
            or not math.isfinite(
                float(gradient.get("post_sp_reduction_elal_memory_builder_norm", math.nan))
            )
            or gradient.get("post_sp_reduction_elal_memory_builder_norm") <= 0.0
            or not isinstance(block_norms, Mapping)
            or set(block_norms) != {str(index) for index in range(BLOCKS)}
            or any(not math.isfinite(float(item)) or item <= 0.0 for item in block_norms.values())
            or gradient.get("lora_B_positive") != LORA_AFFINES
            or type(gradient.get("lora_A_positive")) is not int
            or gradient.get("lora_A_positive") not in range(LORA_AFFINES + 1)
            or (completed_step >= 2 and gradient.get("lora_A_positive") != LORA_AFFINES)
            or type(gradient.get("per_parameter_zero_count")) is not int
            or gradient.get("per_parameter_zero_count") not in range(669)
        ):
            fail(f"{label} synchronized all30 gradient audit differs")


def validate_training_history_v1(
    history: Any,
    *,
    arm_id: str,
    seed: int,
    expected_steps: int,
    initial_parameter_sha256: str,
    final_parameter_sha256: str,
    expected_common_payload: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if not isinstance(history, list) or len(history) != expected_steps:
        fail("C2 training history exact step count differs")
    seen = {initial_parameter_sha256}
    for index, step in enumerate(history):
        completed = index + 1
        if (
            not isinstance(step, Mapping)
            or step.get("step") != completed
            or step.get("optimizer_step_executed") is not True
            or step.get("memory_gate_all8_pass") is not True
            or not math.isfinite(float(step.get("synchronized_gradient_norm", math.nan)))
            or step.get("synchronized_gradient_norm") <= 0.0
            or not math.isfinite(float(step.get("preclip_gradient_norm", math.nan)))
            or step.get("preclip_gradient_norm") <= 0.0
            or step.get("noise_seeds_by_row")
            != [
                training_noise_seed_v1(seed, index, row_index)
                for row_index in (0, 1)
            ]
        ):
            fail(f"C2 training history step {completed} envelope differs")
        parameter_sha = _require_sha256_v1(
            step.get("parameter_sha256"), label=f"history step {completed} parameter"
        )
        if parameter_sha in seen:
            fail("C2 training history parameter chain repeated")
        seen.add(parameter_sha)
        coordinate = step.get("sigma_coordinate")
        if (
            coordinate != TRAINING_SIGMA_EXACT10[index]
        ):
            fail("C2 training history sigma coordinate differs")
        _validate_all8_graph_rows_closed_v1(
            step.get("all8_actual_graph_receipts"),
            arm_id=arm_id,
            completed_step=completed,
            label=f"history step {completed}",
            expected_common_payload=expected_common_payload,
        )
        memory = step.get("memory_world8")
        if not isinstance(memory, list) or len(memory) != WORLD_SIZE:
            fail("C2 training history memory exact8 differs")
        for world_rank, row in enumerate(memory):
            _validate_memory_row_closed_v1(
                row,
                expected_world_rank=world_rank,
                label=f"history step {completed}",
            )
    if history[-1].get("parameter_sha256") != final_parameter_sha256:
        fail("C2 training history final parameter digest differs")
    value = {
        "exact_step_count": expected_steps,
        "all_steps_all8_all30_gradient_memory_closed": True,
        "parameter_chain_unique_and_final_bound": True,
        "history_digest": object_sha256(history),
    }
    return value


def _validate_exact10_receipt_value_v1(
    receipt: Mapping[str, Any],
    *,
    arm_id: str,
    expected_receipt_digest: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    expected_origin_verifier_binding: Mapping[str, Any],
    expected_gate_controller_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Closed, origin-physical validator used before publish and attestation."""

    if arm_id not in ARM_PLACEMENT:
        fail("exact10 receipt arm differs")
    job_id, node, seed = ARM_PLACEMENT[arm_id]
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    latent_pass = receipt.get("latent_hard_gates_pass") is True
    expected_status = (
        "EXACT10_LATENT_GATES_PASS_DECODED_REVIEW_PENDING"
        if latent_pass
        else "EXACT10_LATENT_GATES_NO_GO"
    )
    if (
        set(receipt) != EXACT10_RECEIPT_FIELDS
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != expected_status
        or receipt.get("method") != METHOD
        or receipt.get("arm_id") != arm_id
        or receipt.get("branch_recipe")
        != (
            "target_duplicate_exact2"
            if arm_id == ARM_DUPLICATE
            else "target_and_role_swap_exact2"
        )
        or (receipt.get("holder_job_id"), receipt.get("node"), receipt.get("seed"))
        != (job_id, node, seed)
        or receipt.get("requested_optimizer_steps") != MAX_STEPS
        or receipt.get("completed_optimizer_steps") != MAX_STEPS
        or receipt.get("optimizer_constructed") is not True
        or receipt.get("optimizer_state_empty_before_first_update") is not True
        or receipt.get("fresh_official_base") is not True
        or receipt.get("resume_consumed") is not False
        or receipt.get("fresh1_checkpoint_consumed") is not False
        or receipt.get("parameters_changed") is not True
        or receipt.get("initial_trainable_sha256")
        == receipt.get("final_trainable_sha256")
        or receipt.get("decoded_track_effect_gate_pending") is not True
        or receipt.get("selection_eligible") is not False
        or receipt.get("selection_requires_decoded_track_effect_conjunction") is not True
        or receipt.get("weighted_metric_sum_used") is not False
        or receipt.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or receipt.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or receipt.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or receipt.get("latent_bundle_sha256") != expected_bundle_sha256
        or receipt.get("runner_source_sha256") != expected_runner_sha256
        or receipt.get("source_pins") != dict(expected_source_pins)
        or receipt.get("claim_boundaries") != CLAIM_BOUNDARIES
        or receipt.get("formal_c2_authorized") is not False
        or receipt.get("exact160_authorized") is not False
        or receipt.get("real_video_claim_authorized") is not False
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("source_instruction_inference") is not False
        or not math.isfinite(float(receipt.get("elapsed_seconds", math.nan)))
        or receipt.get("elapsed_seconds") < 0.0
        or digest != expected_receipt_digest
        or digest != object_sha256(unsigned)
    ):
        fail("exact10 receipt envelope/self-digest differs")
    for field in (
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
    ):
        _require_sha256_v1(receipt.get(field), label=f"exact10 {field}")

    own = receipt.get("own_preflight_binding")
    if not isinstance(own, Mapping) or set(own) != {"path", "sha256", "receipt_digest"}:
        fail("exact10 own preflight binding differs")
    own_receipt = _read_sealed_json_held_fd_v1(
        Path(own["path"]),
        expected_sha256=str(own["sha256"]),
        label=f"{arm_id} exact10 own preflight",
    )
    own_receipt = _validate_preflight_receipt_value_v1(
        own_receipt,
        arm_id=arm_id,
        holder_job_id=job_id,
        node=node,
        seed=seed,
        expected_receipt_digest=str(own["receipt_digest"]),
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
    )
    if (
        receipt.get("initial_trainable_sha256") != own_receipt.get("initial_trainable_sha256")
        or receipt.get("common_comparison_payload_digest") != own_receipt.get("common_comparison_payload_digest")
        or receipt.get("row_input_noise_schedule_digest") != own_receipt.get("row_input_noise_schedule_digest")
        or receipt.get("step0_full_q_route") != own_receipt.get("step0_full_q_route")
        or receipt.get("step0_role_only_cells") != own_receipt.get("step0_role_only_cells")
        or receipt.get("step0_role_only_input_invariants")
        != own_receipt.get("step0_role_only_input_invariants")
        or receipt.get("step0_role_only_input_invariants_validation")
        != own_receipt.get("step0_role_only_input_invariants_validation")
        or receipt.get("step0_evaluation_forward_evidence")
        != own_receipt.get("step0_evaluation_forward_evidence")
        or receipt.get("step0_evaluation_forward_evidence_validation")
        != own_receipt.get("step0_evaluation_forward_evidence_validation")
    ):
        fail("exact10 fresh init/common/step0 differs from own preflight")

    cross_binding = receipt.get("cross_arm_gate_binding")
    if not isinstance(cross_binding, Mapping) or set(cross_binding) != {
        "path", "sha256", "gate_digest", "recipe_version_digest"
    }:
        fail("exact10 cross gate binding differs")
    cross_gate = validate_cross_arm_preflight_gate_v1(
        Path(cross_binding["path"]),
        expected_sha256=str(cross_binding["sha256"]),
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
    )
    if (
        cross_gate.get("gate_digest") != cross_binding.get("gate_digest")
        or cross_gate.get("recipe_version_digest") != cross_binding.get("recipe_version_digest")
        or (
            arm_id in (ARM_DUPLICATE, ARM_ROLE_PAIR)
            and (
                receipt.get("initial_trainable_sha256") != cross_gate.get("common_initial_trainable_sha256")
                or receipt.get("common_comparison_payload_digest") != cross_gate.get("common_comparison_payload_digest")
                or receipt.get("row_input_noise_schedule_digest") != cross_gate.get("common_row_input_noise_schedule_digest")
            )
        )
    ):
        fail("exact10 cross-arm common binding differs")

    fresh_binding = receipt.get("fresh1_acceptance_gate_binding")
    expected_fresh_fields = {
        "path",
        "sha256",
        "gate_sha256",
        "gate_digest",
        "cross_arm_gate_sha256",
        "cross_arm_gate_digest",
        "cross_arm_recipe_version_digest",
        "fresh1_attestation_sha256_by_arm",
        "all_three_portable_origin_attestations_replayed",
        "exact10_must_fresh_initialize",
        "resume_from_fresh1_forbidden",
    }
    if not isinstance(fresh_binding, Mapping) or set(fresh_binding) != expected_fresh_fields:
        fail("exact10 fresh1 gate binding differs")
    gate_value = _read_sealed_json_held_fd_v1(
        Path(fresh_binding["path"]),
        expected_sha256=str(fresh_binding["sha256"]),
        label="exact10 portable fresh1 gate",
    )
    fresh_gate = validate_fresh1_acceptance_gate_v1(
        Path(fresh_binding["path"]),
        expected_sha256=str(fresh_binding["sha256"]),
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        cross_gate=cross_gate,
        expected_origin_verifier_binding=expected_origin_verifier_binding,
        expected_gate_controller_binding=expected_gate_controller_binding,
    )
    if any(fresh_binding.get(key) != value for key, value in fresh_gate.items()):
        fail("exact10 fresh1 portable gate replay differs")

    history_validation = validate_training_history_v1(
        receipt.get("history"),
        arm_id=arm_id,
        seed=seed,
        expected_steps=MAX_STEPS,
        initial_parameter_sha256=str(receipt.get("initial_trainable_sha256")),
        final_parameter_sha256=str(receipt.get("final_trainable_sha256")),
        expected_common_payload=own_receipt["common_comparison_payload"],
    )
    if receipt.get("history_validation") != history_validation:
        fail("exact10 history validation receipt differs")

    checkpoint_records = receipt.get("checkpoint_records")
    checkpoint_common = _checkpoint_common_from_receipt_v1(receipt)
    if not isinstance(checkpoint_records, list) or len(checkpoint_records) != 2:
        fail("exact10 checkpoint exact2 closure differs")
    checkpoint_tree = seal_and_validate_checkpoint_tree_v1(
        Path(checkpoint_records[0]["path"]).parent,
        records=checkpoint_records,
        expected_steps=(0, MAX_STEPS),
        expected_parameter_sha256_by_step={
            0: str(receipt.get("initial_trainable_sha256")),
            MAX_STEPS: str(receipt.get("final_trainable_sha256")),
        },
        expected_common=checkpoint_common,
    )
    if receipt.get("checkpoint_tree_closure") != checkpoint_tree:
        fail("exact10 checkpoint tree replay differs")

    validate_role_only_invariant_receipts_v1(
        receipt.get("step0_role_only_input_invariants"), stage="step0"
    )
    evidence = receipt.get("step10_evidence")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "full_q_route",
            "role_only_cells",
            "role_only_input_invariants",
            "role_only_input_invariants_validation",
            "target_prediction_sha256_by_row",
            "target_prediction_hash_projection_by_row",
            "actual_forward_evidence_by_row",
        }
        or evidence.get("role_only_input_invariants_validation")
        != validate_role_only_invariant_receipts_v1(
            evidence.get("role_only_input_invariants"), stage="step10"
        )
        or not isinstance(evidence.get("full_q_route"), Mapping)
        or set(evidence["full_q_route"]) != set(ROW_IDS)
        or not isinstance(evidence.get("role_only_cells"), list)
        or tuple(
            (row.get("row_id"), row.get("clean_variant"))
            for row in evidence["role_only_cells"]
        )
        != ROLE_ONLY_CELL_ORDER
        or not isinstance(evidence.get("target_prediction_sha256_by_row"), list)
        or len(evidence["target_prediction_sha256_by_row"]) != 2
        or not isinstance(
            evidence.get("target_prediction_hash_projection_by_row"), list
        )
        or len(evidence["target_prediction_hash_projection_by_row"]) != 2
    ):
        fail("exact10 step10 evidence closure differs")
    forward_rows = evidence.get("actual_forward_evidence_by_row")
    if not isinstance(forward_rows, list) or len(forward_rows) != 2:
        fail("exact10 step10 actual forward exact2 differs")
    for row_index, row in enumerate(forward_rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "row_id",
                "input_payload",
                "actual_forward_evidence",
                "validation",
                "observation_validation",
            }
            or row.get("row_id") != ROW_IDS[row_index]
        ):
            fail("exact10 step10 actual forward row differs")
        replay = validate_evaluation_forward_evidence_v1(
            row["actual_forward_evidence"],
            row_id=ROW_IDS[row_index],
            sp_rank=0,
            input_payload=row["input_payload"],
        )
        observation_replay = validate_evaluation_observation_binding_v1(
            full_q_route=evidence["full_q_route"][ROW_IDS[row_index]],
            role_only_cells=evidence["role_only_cells"][row_index * 2 : row_index * 2 + 2],
            actual_forward_evidence=row["actual_forward_evidence"],
            row_id=ROW_IDS[row_index],
            stage="step10",
        )
        if (
            row.get("validation") != replay
            or row.get("observation_validation") != observation_replay
            or evidence["target_prediction_sha256_by_row"][row_index]
            != row["actual_forward_evidence"]["full_target"]["actual_input_receipt"]["prediction_sha256"]
        ):
            fail("exact10 step10 actual forward validation differs")
    for item in evidence["target_prediction_sha256_by_row"]:
        _require_sha256_v1(item, label="exact10 target prediction SHA")
    for row_index, projection in enumerate(
        evidence["target_prediction_hash_projection_by_row"]
    ):
        validate_prediction_hash_projection_receipt_v1(
            projection,
            expected_prediction_sha256=evidence["target_prediction_sha256_by_row"][
                row_index
            ],
            expected_original_device_type="cuda",
            expected_original_device_index=row_index * SP_SIZE,
            expected_original_dtype=PREDICTION_HASH_PROJECTION_PRODUCTION_DTYPE,
            expected_original_stride=PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE,
            expected_original_storage_offset=(
                PREDICTION_HASH_PROJECTION_PRODUCTION_STORAGE_OFFSET
            ),
            expected_original_requires_grad=False,
            expected_original_is_contiguous=True,
            label=f"exact10 target prediction projection row {row_index}",
        )
    try:
        replayed_gate = validate_step10_gates_v1(
            step0_full_q=receipt["step0_full_q_route"],
            step10_full_q=evidence["full_q_route"],
            step0_role_only=receipt["step0_role_only_cells"],
            step10_role_only=evidence["role_only_cells"],
        )
        replay_error = None
    except ELAL3C2TrainingError as error:
        replayed_gate = None
        replay_error = str(error)
    if (
        receipt.get("step10_gate") != replayed_gate
        or receipt.get("latent_hard_gate_error") != replay_error
        or latent_pass is not (replayed_gate is not None)
        or receipt.get("primary_metric_if_all_gates_pass")
        != (replayed_gate.get("primary_metric_value") if replayed_gate else None)
    ):
        fail("exact10 recomputed latent gate/status differs")
    closure = receipt.get("pre_publish_closure_replays")
    common = own_receipt["common_comparison_payload"]
    if (
        not isinstance(closure, Mapping)
        or closure.get("runtime_sources_pre_final_bit_exact") is not True
        or closure.get("model_exact9_pre_post_final_stable") is not True
        or closure.get("bundle_exact16_pre_final_stable") is not True
        or closure.get("materializer_run_complete_pre_final_stable") is not True
        or closure.get("materializer_run_complete_sha256") != MATERIALIZER_RUN_COMPLETE_SHA256
        or closure.get("materializer_run_complete_digest") != MATERIALIZER_RUN_COMPLETE_DIGEST
        or closure.get("checkpoint_exact23_pre_post_final_stable") is not True
        or closure.get("checkpoint_exact23_binding_digest") != common.get("checkpoint_exact23_binding_digest")
        or closure.get("bernini_execution_sources_pre_post_final_stable") is not True
        or closure.get("bernini_execution_source_binding_digest") != common.get("bernini_execution_source_binding_digest")
        or closure.get("oracle_labels_pre_final_stable") is not True
        or closure.get("checkpoint_tree_pre_publish_stable") is not True
        or closure.get("checkpoint_tree_binding_digest") != checkpoint_tree.get("tree_binding_digest")
        or closure.get("checkpoint_portable_tree_digest") != checkpoint_tree.get("portable_checkpoint_tree_digest")
    ):
        fail("exact10 final source/model/bundle/checkpoint closure differs")
    return receipt


def _validate_exact10_origin_attestation_value_v1(
    attestation: Any,
    *,
    arm_id: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    expected_cross_gate_binding: Mapping[str, Any],
    expected_fresh1_gate_binding: Mapping[str, Any],
    expected_origin_verifier_binding: Mapping[str, Any],
    expected_gate_controller_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "status",
        "stage",
        "arm_id",
        "holder_job_id",
        "node",
        "seed",
        "receipt_sha256",
        "receipt_size",
        "receipt_digest",
        "receipt_status",
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
        "history_digest",
        "portable_checkpoint_tree",
        "portable_checkpoint_tree_digest",
        "cross_arm_gate_sha256",
        "cross_arm_gate_digest",
        "fresh1_acceptance_gate_sha256",
        "fresh1_acceptance_gate_digest",
        "latent_hard_gates_pass",
        "decoded_track_effect_gate_pending",
        "runner_source_sha256",
        "latent_bundle_sha256",
        "source_pins",
        "experiment_contract_sha256",
        "external_authority_sha256",
        "model_authority_sha256",
        "materializer_run_complete_sha256",
        "materializer_run_complete_digest",
        "checkpoint_exact23_binding_digest",
        "bernini_execution_source_binding_digest",
        "origin_verifier_binding",
        "gate_controller_binding",
        "physical_origin_replay_passed",
        "closed_validator_passed",
        "attestation_digest",
    }
    job_id, node, seed = ARM_PLACEMENT[arm_id]
    cross_fields = {"gate_sha256", "gate_digest", "recipe_version_digest"}
    fresh_fields = {
        "gate_sha256",
        "gate_digest",
        "cross_arm_gate_sha256",
        "cross_arm_gate_digest",
        "cross_arm_recipe_version_digest",
    }
    if (
        not isinstance(expected_cross_gate_binding, Mapping)
        or set(expected_cross_gate_binding) != cross_fields
        or not isinstance(expected_fresh1_gate_binding, Mapping)
        or set(expected_fresh1_gate_binding) != fresh_fields
    ):
        fail("exact10 expected predecessor gate binding closure differs")
    for label, binding in (
        ("cross", expected_cross_gate_binding),
        ("fresh1", expected_fresh1_gate_binding),
    ):
        for key, value in binding.items():
            _require_sha256_v1(value, label=f"exact10 expected {label} {key}")
    if (
        expected_fresh1_gate_binding["cross_arm_gate_sha256"]
        != expected_cross_gate_binding["gate_sha256"]
        or expected_fresh1_gate_binding["cross_arm_gate_digest"]
        != expected_cross_gate_binding["gate_digest"]
        or expected_fresh1_gate_binding["cross_arm_recipe_version_digest"]
        != expected_cross_gate_binding["recipe_version_digest"]
    ):
        fail("exact10 expected predecessor gates do not form one chain")
    unsigned = dict(attestation) if isinstance(attestation, Mapping) else {}
    digest = unsigned.pop("attestation_digest", None)
    expected_status = (
        "EXACT10_LATENT_GATES_PASS_DECODED_REVIEW_PENDING"
        if attestation.get("latent_hard_gates_pass") is True
        else "EXACT10_LATENT_GATES_NO_GO"
    ) if isinstance(attestation, Mapping) else None
    if (
        not isinstance(attestation, Mapping)
        or set(attestation) != fields
        or attestation.get("schema_version") != EXACT10_ORIGIN_ATTESTATION_SCHEMA
        or attestation.get("status") != "EXACT10_ORIGIN_PHYSICAL_REPLAY_PASS"
        or attestation.get("stage") != "exact10"
        or (attestation.get("arm_id"), attestation.get("holder_job_id"), attestation.get("node"), attestation.get("seed"))
        != (arm_id, job_id, node, seed)
        or attestation.get("receipt_status") != expected_status
        or attestation.get("initial_trainable_sha256") == attestation.get("final_trainable_sha256")
        or attestation.get("decoded_track_effect_gate_pending") is not True
        or attestation.get("runner_source_sha256") != expected_runner_sha256
        or attestation.get("latent_bundle_sha256") != expected_bundle_sha256
        or attestation.get("source_pins") != dict(expected_source_pins)
        or attestation.get("cross_arm_gate_sha256")
        != expected_cross_gate_binding["gate_sha256"]
        or attestation.get("cross_arm_gate_digest")
        != expected_cross_gate_binding["gate_digest"]
        or attestation.get("fresh1_acceptance_gate_sha256")
        != expected_fresh1_gate_binding["gate_sha256"]
        or attestation.get("fresh1_acceptance_gate_digest")
        != expected_fresh1_gate_binding["gate_digest"]
        or attestation.get("experiment_contract_sha256") != EXPERIMENT_CONTRACT_SHA256
        or attestation.get("external_authority_sha256") != EXTERNAL_AUTHORITY_SHA256
        or attestation.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or attestation.get("materializer_run_complete_sha256") != MATERIALIZER_RUN_COMPLETE_SHA256
        or attestation.get("materializer_run_complete_digest") != MATERIALIZER_RUN_COMPLETE_DIGEST
        or attestation.get("origin_verifier_binding") != dict(expected_origin_verifier_binding)
        or attestation.get("gate_controller_binding") != dict(expected_gate_controller_binding)
        or attestation.get("physical_origin_replay_passed") is not True
        or attestation.get("closed_validator_passed") is not True
        or type(attestation.get("receipt_size")) is not int
        or attestation.get("receipt_size") <= 0
        or digest != object_sha256(unsigned)
    ):
        fail("exact10 origin attestation envelope differs")
    for field in (
        "receipt_sha256",
        "receipt_digest",
        "initial_trainable_sha256",
        "final_trainable_sha256",
        "common_comparison_payload_digest",
        "row_input_noise_schedule_digest",
        "history_digest",
        "portable_checkpoint_tree_digest",
        "cross_arm_gate_sha256",
        "cross_arm_gate_digest",
        "fresh1_acceptance_gate_sha256",
        "fresh1_acceptance_gate_digest",
        "checkpoint_exact23_binding_digest",
        "bernini_execution_source_binding_digest",
        "attestation_digest",
    ):
        _require_sha256_v1(attestation.get(field), label=f"exact10 attestation {field}")
    for binding_name in ("origin_verifier_binding", "gate_controller_binding"):
        binding = attestation[binding_name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"name", "sha256", "size", "mode", "nlink"}
            or binding.get("mode") != 0o444
            or binding.get("nlink") != 1
            or type(binding.get("size")) is not int
            or binding.get("size") <= 0
        ):
            fail("exact10 attestation tool binding differs")
        _require_sha256_v1(binding.get("sha256"), label=f"{binding_name} SHA")
    _validate_portable_checkpoint_tree_v1(
        attestation.get("portable_checkpoint_tree"),
        expected_steps=(0, MAX_STEPS),
        expected_parameters=(
            attestation["initial_trainable_sha256"],
            attestation["final_trainable_sha256"],
        ),
    )
    if attestation.get("portable_checkpoint_tree_digest") != attestation["portable_checkpoint_tree"]["portable_checkpoint_tree_digest"]:
        fail("exact10 attestation portable checkpoint join differs")
    return attestation


def build_exact10_origin_attestation_v1(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    arm_id: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    expected_cross_gate_binding: Mapping[str, Any],
    expected_fresh1_gate_binding: Mapping[str, Any],
    origin_verifier_path: Path,
    expected_origin_verifier_sha256: str,
    gate_controller_path: Path,
    expected_gate_controller_sha256: str,
) -> Mapping[str, Any]:
    origin_binding = _portable_tool_binding_v1(
        origin_verifier_path,
        expected_sha256=expected_origin_verifier_sha256,
        label="exact10 origin verifier",
    )
    controller_binding = _portable_tool_binding_v1(
        gate_controller_path,
        expected_sha256=expected_gate_controller_sha256,
        label="exact10 gate controller",
    )
    receipt = _read_sealed_json_held_fd_v1(
        receipt_path,
        expected_sha256=expected_receipt_sha256,
        label=f"{arm_id} origin exact10 receipt",
    )
    validated = _validate_exact10_receipt_value_v1(
        receipt,
        arm_id=arm_id,
        expected_receipt_digest=str(receipt.get("receipt_digest")),
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        expected_origin_verifier_binding=origin_binding,
        expected_gate_controller_binding=controller_binding,
    )
    tree = validated["checkpoint_tree_closure"]
    portable_tree = {
        key: tree[key]
        for key in (
            "schema_version",
            "expected_steps",
            "directory_entries",
            "directory_mode",
            "portable_checkpoint_records",
            "portable_checkpoint_tree_digest",
            "physical_origin_replay_passed",
        )
    }
    closure = validated["pre_publish_closure_replays"]
    fresh = validated["fresh1_acceptance_gate_binding"]
    actual_cross_binding = {
        "gate_sha256": validated["cross_arm_gate_binding"]["sha256"],
        "gate_digest": validated["cross_arm_gate_binding"]["gate_digest"],
        "recipe_version_digest": validated["cross_arm_gate_binding"][
            "recipe_version_digest"
        ],
    }
    actual_fresh_binding = {
        "gate_sha256": fresh["gate_sha256"],
        "gate_digest": fresh["gate_digest"],
        "cross_arm_gate_sha256": fresh["cross_arm_gate_sha256"],
        "cross_arm_gate_digest": fresh["cross_arm_gate_digest"],
        "cross_arm_recipe_version_digest": fresh[
            "cross_arm_recipe_version_digest"
        ],
    }
    if (
        actual_cross_binding != dict(expected_cross_gate_binding)
        or actual_fresh_binding != dict(expected_fresh1_gate_binding)
    ):
        fail("exact10 origin receipt predecessor gates differ from sealed expectations")
    unsigned = {
        "schema_version": EXACT10_ORIGIN_ATTESTATION_SCHEMA,
        "status": "EXACT10_ORIGIN_PHYSICAL_REPLAY_PASS",
        "stage": "exact10",
        "arm_id": arm_id,
        "holder_job_id": ARM_PLACEMENT[arm_id][0],
        "node": ARM_PLACEMENT[arm_id][1],
        "seed": ARM_PLACEMENT[arm_id][2],
        "receipt_sha256": expected_receipt_sha256,
        "receipt_size": len(canonical_json_bytes(receipt)) + 1,
        "receipt_digest": validated["receipt_digest"],
        "receipt_status": validated["status"],
        "initial_trainable_sha256": validated["initial_trainable_sha256"],
        "final_trainable_sha256": validated["final_trainable_sha256"],
        "common_comparison_payload_digest": validated["common_comparison_payload_digest"],
        "row_input_noise_schedule_digest": validated["row_input_noise_schedule_digest"],
        "history_digest": validated["history_validation"]["history_digest"],
        "portable_checkpoint_tree": portable_tree,
        "portable_checkpoint_tree_digest": portable_tree["portable_checkpoint_tree_digest"],
        "cross_arm_gate_sha256": validated["cross_arm_gate_binding"]["sha256"],
        "cross_arm_gate_digest": validated["cross_arm_gate_binding"]["gate_digest"],
        "fresh1_acceptance_gate_sha256": fresh["sha256"],
        "fresh1_acceptance_gate_digest": fresh["gate_digest"],
        "latent_hard_gates_pass": validated["latent_hard_gates_pass"],
        "decoded_track_effect_gate_pending": True,
        "runner_source_sha256": expected_runner_sha256,
        "latent_bundle_sha256": expected_bundle_sha256,
        "source_pins": dict(expected_source_pins),
        "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
        "external_authority_sha256": EXTERNAL_AUTHORITY_SHA256,
        "model_authority_sha256": MODEL_AUTHORITY_SHA256,
        "materializer_run_complete_sha256": MATERIALIZER_RUN_COMPLETE_SHA256,
        "materializer_run_complete_digest": MATERIALIZER_RUN_COMPLETE_DIGEST,
        "checkpoint_exact23_binding_digest": closure["checkpoint_exact23_binding_digest"],
        "bernini_execution_source_binding_digest": closure["bernini_execution_source_binding_digest"],
        "origin_verifier_binding": origin_binding,
        "gate_controller_binding": controller_binding,
        "physical_origin_replay_passed": True,
        "closed_validator_passed": True,
    }
    attestation = {**unsigned, "attestation_digest": object_sha256(unsigned)}
    return _validate_exact10_origin_attestation_value_v1(
        attestation,
        arm_id=arm_id,
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        expected_cross_gate_binding=expected_cross_gate_binding,
        expected_fresh1_gate_binding=expected_fresh1_gate_binding,
        expected_origin_verifier_binding=origin_binding,
        expected_gate_controller_binding=controller_binding,
    )


def validate_exact10_origin_attestation_v1(
    path: Path,
    *,
    expected_sha256: str,
    arm_id: str,
    expected_runner_sha256: str,
    expected_bundle_sha256: str,
    expected_source_pins: Mapping[str, Any],
    expected_cross_gate_binding: Mapping[str, Any],
    expected_fresh1_gate_binding: Mapping[str, Any],
    expected_origin_verifier_binding: Mapping[str, Any],
    expected_gate_controller_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _read_sealed_json_held_fd_v1(
        path,
        expected_sha256=expected_sha256,
        label=f"{arm_id} portable exact10 origin attestation",
    )
    return _validate_exact10_origin_attestation_value_v1(
        value,
        arm_id=arm_id,
        expected_runner_sha256=expected_runner_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_source_pins=expected_source_pins,
        expected_cross_gate_binding=expected_cross_gate_binding,
        expected_fresh1_gate_binding=expected_fresh1_gate_binding,
        expected_origin_verifier_binding=expected_origin_verifier_binding,
        expected_gate_controller_binding=expected_gate_controller_binding,
    )


def aggregate_preoptimizer_evidence_v1(
    gathered: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bundle: C2LatentBundleV1,
    materializer_run_binding: Mapping[str, Any],
    checkpoint_exact23_binding: Mapping[str, Any],
    bernini_execution_source_binding: Mapping[str, Any],
    source_pins: Mapping[str, Any],
    initial_trainable_sha256: str,
    trainable_inventory_digest: str,
) -> Mapping[str, Any]:
    """Turn exact all8 actual work into the sealed preflight comparison payload."""

    if len(gathered) != WORLD_SIZE:
        fail("preoptimizer evidence requires exact all8 rows")
    leaders: list[Mapping[str, Any]] = []
    for row_index in (0, 1):
        group_rows = list(gathered[row_index * SP_SIZE : (row_index + 1) * SP_SIZE])
        if (
            len(group_rows) != SP_SIZE
            or [row.get("world_rank") for row in group_rows]
            != list(range(row_index * SP_SIZE, (row_index + 1) * SP_SIZE))
            or any(row.get("row_index") != row_index for row in group_rows)
            or any(row.get("row_id") != ROW_IDS[row_index] for row in group_rows)
        ):
            fail("preoptimizer DP2xSP4 evidence placement differs")
        scalar_evaluation = lambda item: {
            key: value
            for key, value in item.items()
            if key not in {
                "actual_forward_evidence",
                "actual_forward_evidence_validation",
            }
        }
        projection_views = [
            prediction_hash_projection_consensus_view_v1(
                row.get("target_prediction_hash_projection"),
                expected_prediction_sha256=str(row.get("target_prediction_sha256")),
                expected_original_device_index=int(row["world_rank"]),
                label=f"preoptimizer row {row_index} SP{sp_index}",
            )
            for sp_index, row in enumerate(group_rows)
        ]
        if any(
            canonical_json_bytes(group_rows[0][key])
            != canonical_json_bytes(candidate[key])
            for candidate in group_rows[1:]
            for key in ("local_schedule", "target_prediction_sha256", "gain_safety")
        ) or any(
            canonical_json_bytes(scalar_evaluation(group_rows[0]["evaluation"]))
            != canonical_json_bytes(scalar_evaluation(candidate["evaluation"]))
            for candidate in group_rows[1:]
        ) or any(
            canonical_json_bytes(projection_views[0])
            != canonical_json_bytes(candidate)
            for candidate in projection_views[1:]
        ):
            fail("preoptimizer SP4 row evidence lacks bit-exact consensus")
        leaders.append(group_rows[0])
    schedule = merge_noise_schedule_receipts_v1(
        [leader["local_schedule"] for leader in leaders]
    )
    step0_full = {
        ROW_IDS[index]: leaders[index]["evaluation"]["full_q_route"]
        for index in (0, 1)
    }
    step0_cells = [
        cell
        for leader in leaders
        for cell in leader["evaluation"]["role_only_cells"]
    ]
    if tuple(
        (row.get("row_id"), row.get("clean_variant")) for row in step0_cells
    ) != ROLE_ONLY_CELL_ORDER:
        fail("preoptimizer step0 role-only exact4 order differs")
    step0_invariants = [
        leader["evaluation"]["role_only_input_invariants"][clean_variant]
        for leader in leaders
        for clean_variant in TRAIN_VARIANTS
    ]
    step0_invariants_validation = validate_role_only_invariant_receipts_v1(
        step0_invariants, stage="step0"
    )
    step0_forward_evidence = [
        {
            "row_id": ROW_IDS[index],
            "input_payload": leaders[index]["evaluation"]["input_payload"],
            "actual_forward_evidence": leaders[index]["evaluation"]["actual_forward_evidence"],
            "validation": leaders[index]["evaluation"]["actual_forward_evidence_validation"],
            "observation_validation": leaders[index]["evaluation"]["observation_binding_validation"],
        }
        for index in (0, 1)
    ]
    step0_forward_validation = []
    for row_index, row in enumerate(step0_forward_evidence):
        replay = validate_evaluation_forward_evidence_v1(
            row["actual_forward_evidence"],
            row_id=ROW_IDS[row_index],
            sp_rank=0,
            input_payload=row["input_payload"],
        )
        observation_replay = validate_evaluation_observation_binding_v1(
            full_q_route=step0_full[ROW_IDS[row_index]],
            role_only_cells=step0_cells[row_index * 2 : row_index * 2 + 2],
            actual_forward_evidence=row["actual_forward_evidence"],
            row_id=ROW_IDS[row_index],
            stage="step0",
        )
        if (
            row["validation"] != replay
            or row["observation_validation"] != observation_replay
        ):
            fail("preoptimizer evaluation forward validation differs")
        step0_forward_validation.append(replay)
    common_row_inputs = []
    prediction_rows = []
    for row_index, leader in enumerate(leaders):
        payload = leader["evaluation"]["input_payload"]
        common_row_inputs.append(
            {
                "row_index": row_index,
                "row_id": ROW_IDS[row_index],
                "source_tensor_sha256": payload["source_sha256"],
                "target_tensor_sha256": payload["target_sha256"],
                "role_swap_tensor_sha256": payload["role_swap_sha256"],
                "instruction_sha256": payload["instruction_sha256"],
                "target_q_digest": leader["target_q_digest"],
                "role_swap_q_digest": payload["role_swap_q_digest"],
                "target_label_digest": payload["target_label_digest"],
                "role_swap_label_digest": payload["role_swap_label_digest"],
                "target_mismatch_digest": payload["target_mismatch_digest"],
                "role_swap_mismatch_digest": payload["role_swap_mismatch_digest"],
            }
        )
        prediction_rows.append(
            {
                "row_index": row_index,
                "row_id": ROW_IDS[row_index],
                "prediction_sha256": leader["target_prediction_sha256"],
                "hash_projection": leader["target_prediction_hash_projection"],
            }
        )
    common = {
        "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
        "experiment_contract_digest": EXPERIMENT_CONTRACT_DIGEST,
        "external_authority_sha256": EXTERNAL_AUTHORITY_SHA256,
        "external_authority_digest": EXTERNAL_AUTHORITY_DIGEST,
        "model_authority_sha256": MODEL_AUTHORITY_SHA256,
        "model_authority_digest": MODEL_AUTHORITY_DIGEST,
        "latent_bundle_sha256": bundle.bundle_sha256,
        "latent_bundle_receipt_sha256": bundle.receipt_sha256,
        "materializer_run_complete_sha256": materializer_run_binding[
            "file_sha256"
        ],
        "materializer_run_complete_digest": materializer_run_binding[
            "run_digest"
        ],
        "checkpoint_exact23_manifest_sha256": (
            CHECKPOINT_EXACT23_MANIFEST_SHA256
        ),
        "checkpoint_exact23_binding_digest": checkpoint_exact23_binding[
            "fixed_release_binding_digest"
        ],
        "bernini_execution_source_binding_digest": (
            bernini_execution_source_binding["fixed_release_binding_digest"]
        ),
        "latent_tensor_order": list(LATENT_TENSOR_ORDER),
        "latent_tensor_order_digest": object_sha256(list(LATENT_TENSOR_ORDER)),
        "latent_tensor_rows": [dict(row) for row in bundle.tensor_rows],
        "source_pins": dict(source_pins),
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "trainable_inventory_digest": trainable_inventory_digest,
        "initial_trainable_sha256": initial_trainable_sha256,
        "row_common_target_inputs": common_row_inputs,
        "common_target_branch_schedule": schedule,
        "row_input_noise_schedule_digest": schedule["schedule_digest"],
        "step0_common_target_prediction_sha256_by_row": prediction_rows,
        "step0_full_q_route_digest": object_sha256(step0_full),
        "step0_role_only_cells_digest": object_sha256(step0_cells),
        "step0_role_only_input_invariants_digest": object_sha256(
            step0_invariants
        ),
        "step0_evaluation_forward_evidence_digest": object_sha256(
            step0_forward_evidence
        ),
        "optimizer_recipe": optimizer_recipe_v1(),
        "rng_recipe": {
            "model_initialization_seed": seed,
            "training_epsilon": (
                "training_noise_seed_v1(seed,step_zero,row_index)"
            ),
            "evaluation_epsilon": "100*arm_seed+row_index",
            "epsilon_device": "cpu",
            "epsilon_dtype": "torch.float32",
        },
        "execution_memory_contract": execution_memory_contract_v1(),
        "fresh_official_base": True,
        "resume_consumed": False,
    }
    _validate_common_preflight_payload_v1(
        common,
        expected_bundle_sha256=bundle.bundle_sha256,
        expected_source_pins=source_pins,
        expected_seed=seed,
    )
    memory_rows = [row["runtime_telemetry"]["memory"] for row in gathered]
    if (
        [row.get("world_rank") for row in memory_rows] != list(range(WORLD_SIZE))
        or any(row.get("strictly_greater_than_half") is not True for row in memory_rows)
    ):
        fail("preoptimizer actual-shape memory gate differs")
    actual = {
        "actual_shape_two_branch_forward_pass": True,
        "all30_each_branch_used": True,
        "sp4_partition_all8_pass": True,
        "memory_all8_strictly_gt_half": True,
        "cross_arm_collective_used": False,
        "strict_sequential_branch_graphs": True,
        "preflight_grad_enabled_training_graph": True,
        "preflight_backward_executed": False,
        "simultaneous_live_autograd_branch_graphs_maximum": 1,
        "activation_checkpoint_profile": ACTIVATION_CHECKPOINT_PROFILE,
        "activation_checkpointed_blocks": list(ACTIVATION_CHECKPOINT_BLOCKS),
        "activation_checkpoint_nonreentrant": True,
        "activation_checkpoint_elal_route_context_replay": True,
        "memory_peak_true_tensors_only": True,
        "dummy_or_padding_allocations": False,
        "common_target_prediction_sha256_by_row": prediction_rows,
        "runtime_telemetry": [row["runtime_telemetry"] for row in gathered],
    }
    gain = {
        "all_rows_finite_nonzero_bounded_and_restored": True,
        "rows": [leader["gain_safety"] for leader in leaders],
        "runtime_telemetry": [
            {
                "world_rank": row["world_rank"],
                "gain_probe": row["gain_safety"],
            }
            for row in gathered
        ],
    }
    return {
        "common": common,
        "step0_full_q_route": step0_full,
        "step0_role_only_cells": step0_cells,
        "step0_role_only_input_invariants": step0_invariants,
        "step0_role_only_input_invariants_validation": (
            step0_invariants_validation
        ),
        "step0_evaluation_forward_evidence": step0_forward_evidence,
        "step0_evaluation_forward_evidence_validation": step0_forward_validation,
        "actual_shape_preflight": actual,
        "step0_gain_safety": gain,
        "leaders": leaders,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--arm-id", choices=ARM_IDS, required=True)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--checkpoint-exact23-manifest", required=True)
    value.add_argument(
        "--expected-checkpoint-exact23-manifest-sha256", required=True
    )
    value.add_argument("--packet-root", required=True)
    value.add_argument("--latent-bundle", required=True)
    value.add_argument("--expected-latent-bundle-sha256", required=True)
    value.add_argument("--latent-bundle-receipt", required=True)
    value.add_argument("--expected-latent-bundle-receipt-sha256", required=True)
    value.add_argument("--materializer-run-complete", required=True)
    value.add_argument(
        "--expected-materializer-run-complete-sha256", required=True
    )
    value.add_argument("--external-authority", required=True)
    value.add_argument("--expected-external-authority-sha256", required=True)
    value.add_argument("--model-authority", required=True)
    value.add_argument("--expected-model-authority-sha256", required=True)
    value.add_argument("--experiment-contract", required=True)
    value.add_argument("--expected-experiment-contract-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--max-steps", type=int, choices=(1, MAX_STEPS), required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--own-preflight-receipt")
    value.add_argument("--expected-own-preflight-receipt-sha256")
    value.add_argument("--cross-arm-preflight-gate")
    value.add_argument("--expected-cross-arm-preflight-gate-sha256")
    value.add_argument("--fresh1-acceptance-gate")
    value.add_argument("--expected-fresh1-acceptance-gate-sha256")
    value.add_argument("--fresh1-origin-verifier-name")
    value.add_argument("--expected-fresh1-origin-verifier-sha256")
    value.add_argument("--expected-fresh1-origin-verifier-size", type=int)
    value.add_argument("--fresh1-gate-controller-name")
    value.add_argument("--expected-fresh1-gate-controller-sha256")
    value.add_argument("--expected-fresh1-gate-controller-size", type=int)
    value.add_argument("--expected-runner-source-sha256", required=True)
    value.add_argument("--expected-c1-trainer-source-sha256", required=True)
    value.add_argument("--expected-elal3-core-source-sha256", required=True)
    value.add_argument("--expected-c2-label-source-sha256", required=True)
    value.add_argument("--expected-c2-materializer-source-sha256", required=True)
    value.add_argument("--expected-train-lora-source-sha256", required=True)
    value.add_argument("--expected-packed-lora-source-sha256", required=True)
    value.add_argument("--expected-runtime-source-sha256", required=True)
    value.add_argument("--expected-sigma-source-sha256", required=True)
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--ack-simulator-oracle-q-diagnostic-only", action="store_true")
    value.add_argument("--ack-not-source-instruction-inference", action="store_true")
    value.add_argument("--ack-not-formal-c2", action="store_true")
    value.add_argument("--ack-not-exact160", action="store_true")
    value.add_argument("--ack-no-real-video-or-scientific-claim", action="store_true")
    return value


def validate_args_static_v1(args: argparse.Namespace) -> None:
    if not all(
        (
            args.ack_simulator_oracle_q_diagnostic_only,
            args.ack_not_source_instruction_inference,
            args.ack_not_formal_c2,
            args.ack_not_exact160,
            args.ack_no_real_video_or_scientific_claim,
        )
    ):
        fail("all five C2 oracle diagnostic acknowledgements are mandatory")
    if args.max_steps not in (1, MAX_STEPS):
        fail("C2 staged diagnostic requires exactly one or ten optimizer updates")
    expected_seed = ARM_PLACEMENT[args.arm_id][2]
    if args.seed != expected_seed:
        fail("arm seed differs from preregistered experiment contract")
    if args.expected_external_authority_sha256 != EXTERNAL_AUTHORITY_SHA256:
        fail("external authority CLI SHA differs from literal")
    if args.expected_model_authority_sha256 != MODEL_AUTHORITY_SHA256:
        fail("model authority CLI SHA differs from literal")
    if args.expected_experiment_contract_sha256 != EXPERIMENT_CONTRACT_SHA256:
        fail("experiment contract CLI SHA differs from literal")
    if (
        args.expected_materializer_run_complete_sha256
        != MATERIALIZER_RUN_COMPLETE_SHA256
    ):
        fail("materializer RUN_COMPLETE CLI SHA differs from retry2 literal")
    if args.expected_c1_trainer_source_sha256 != C1_TRAINER_SHA256:
        fail("frozen C1 trainer dependency SHA differs")
    if (
        args.expected_checkpoint_exact23_manifest_sha256
        != CHECKPOINT_EXACT23_MANIFEST_SHA256
    ):
        fail("checkpoint exact23 manifest CLI SHA differs from literal")
    if args.expected_elal3_core_source_sha256 != C1_CORE_SHA256:
        fail("frozen ELAL3 core dependency SHA differs")
    expected_runtime_pins = {
        "expected_c2_label_source_sha256": C2_LABEL_SHA256,
        "expected_c2_materializer_source_sha256": C2_MATERIALIZER_SHA256,
        "expected_train_lora_source_sha256": TRAIN_LORA_SHA256,
        "expected_packed_lora_source_sha256": PACKED_LORA_SHA256,
        "expected_runtime_source_sha256": RUNTIME_SHA256,
        "expected_sigma_source_sha256": SIGMA_SHA256,
    }
    if any(getattr(args, name) != expected for name, expected in expected_runtime_pins.items()):
        fail("runtime source CLI SHA differs from release literal")
    gate_pairs = (
        (args.own_preflight_receipt, args.expected_own_preflight_receipt_sha256),
        (
            args.cross_arm_preflight_gate,
            args.expected_cross_arm_preflight_gate_sha256,
        ),
        (args.fresh1_acceptance_gate, args.expected_fresh1_acceptance_gate_sha256),
    )
    if args.preflight_only:
        if any(path is not None or sha is not None for path, sha in gate_pairs):
            fail("preflight must not consume downstream acceptance gates")
    else:
        required = gate_pairs[:2] if args.max_steps == 1 else gate_pairs
        if any(path is None or sha is None for path, sha in required):
            fail("non-preflight stage lacks required sealed predecessor gates")
        if args.max_steps == 1 and any(
            value is not None for value in gate_pairs[2]
        ):
            fail("fresh1 must not consume the later fresh1 acceptance gate")
        for _, sha in required:
            _require_sha256_v1(sha, label="staged gate SHA")
        portable_tool_args = (
            args.fresh1_origin_verifier_name,
            args.expected_fresh1_origin_verifier_sha256,
            args.expected_fresh1_origin_verifier_size,
            args.fresh1_gate_controller_name,
            args.expected_fresh1_gate_controller_sha256,
            args.expected_fresh1_gate_controller_size,
        )
        if args.max_steps == MAX_STEPS:
            if any(value is None for value in portable_tool_args):
                fail("exact10 lacks portable origin verifier/controller release pins")
            _require_sha256_v1(
                args.expected_fresh1_origin_verifier_sha256,
                label="fresh1 origin verifier SHA",
            )
            _require_sha256_v1(
                args.expected_fresh1_gate_controller_sha256,
                label="fresh1 gate controller SHA",
            )
            if (
                args.expected_fresh1_origin_verifier_size <= 0
                or args.expected_fresh1_gate_controller_size <= 0
                or Path(args.fresh1_origin_verifier_name).name
                != args.fresh1_origin_verifier_name
                or Path(args.fresh1_gate_controller_name).name
                != args.fresh1_gate_controller_name
            ):
                fail("portable verifier/controller name/size ABI differs")
        elif any(value is not None for value in portable_tool_args):
            fail("preflight/fresh1 must not consume future portable tool pins")
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or "elal3_c2" not in output.name.lower()
    ):
        fail("output must be one fresh absolute ELAL3_C2 path")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run preflight, fresh1, or fresh10 with every predecessor gate replayed."""

    args = parser().parse_args(argv)
    validate_args_static_v1(args)
    require_bundle_release_literals_v1(
        expected_bundle_sha256=args.expected_latent_bundle_sha256,
        expected_receipt_sha256=args.expected_latent_bundle_receipt_sha256,
        expected_run_complete_sha256=(
            args.expected_materializer_run_complete_sha256
        ),
    )
    runner_path = Path(__file__).resolve(strict=True)
    runner_sha = file_sha256(runner_path)
    runner_size = runner_path.stat().st_size
    if runner_sha != args.expected_runner_source_sha256:
        fail("C2 runner source SHA differs")

    import gc
    import time
    import train_lora as legacy

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=BERNINI_COMMIT,
                expected_veomni_commit=VEOMNI_COMMIT,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise ELAL3C2TrainingError(str(error)) from error
    if (
        transformer_config.get("num_layers") != BLOCKS
        or transformer_config.get("hidden_size") not in (None, HIDDEN)
        or transformer_config.get("num_attention_heads") != 12
        or transformer_config.get("attention_head_dim") != 128
    ):
        fail("Bernini-R 1.3B transformer geometry differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    import torch.utils.checkpoint as torch_checkpoint
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    import bernini.models.renderer as bernini_renderer_module
    import bernini.models.transformer_wan as bernini_transformer_wan_module
    import bernini.parallel as bernini_parallel_module
    import bernini.parallel.state as bernini_parallel_state_module
    import veomni.distributed.parallel_state as veomni_parallel_state_module
    import veomni.distributed.sequence_parallel.comm as veomni_sequence_comm_module
    import bernini.pipeline as bernini_pipeline
    import diffusers
    import diffusers.models.autoencoders.autoencoder_kl_wan as diffusers_wan
    import elal3_c0_v1 as elal3
    import elal3_simulator_c2_label_v1 as label_module
    import materialize_elal3_simulator_c2_vae_v1 as materializer_module
    import packed_preservation_lora_v2 as packed_lora
    import source_self_runtime as runtime
    import inference_sigma_strata as sigma_strata

    source_closure_pre = validate_runtime_sources_strong_v1(
        runner_sha256=runner_sha,
        runner_size=runner_size,
        materializer_module=materializer_module,
        label_module=label_module,
        elal_module=elal3,
        legacy_module=legacy,
        packed_module=packed_lora,
        runtime_module=runtime,
        sigma_module=sigma_strata,
    )
    source_pins = source_pin_map_v1(source_closure_pre)
    experiment_path = Path(args.experiment_contract).expanduser().resolve(strict=True)
    external_path = Path(args.external_authority).expanduser().resolve(strict=True)
    model_path = Path(args.model_authority).expanduser().resolve(strict=True)
    checkpoint_exact23_manifest_path = (
        Path(args.checkpoint_exact23_manifest).expanduser().resolve(strict=True)
    )
    packet_root = Path(args.packet_root).expanduser().resolve(strict=True)
    bundle_path = Path(args.latent_bundle).expanduser().resolve(strict=True)
    bundle_receipt_path = (
        Path(args.latent_bundle_receipt).expanduser().resolve(strict=True)
    )
    materializer_run_path = (
        Path(args.materializer_run_complete).expanduser().resolve(strict=True)
    )
    materializer_run_binding = validate_materializer_run_complete_v1(
        materializer_run_path,
        expected_sha256=args.expected_materializer_run_complete_sha256,
        label_module=label_module,
    )
    experiment_contract = validate_experiment_contract_v1(
        experiment_path,
        expected_sha256=args.expected_experiment_contract_sha256,
    )
    external_authority = validate_external_authority_v1(
        external_path,
        expected_sha256=args.expected_external_authority_sha256,
    )
    placement = validate_runtime_arm_placement_v1(
        external_authority, experiment_contract, arm_id=args.arm_id
    )
    contract = runtime.distributed_contract()
    if (
        contract.world_size != WORLD_SIZE
        or contract.local_world_size != WORLD_SIZE
        or contract.topology.sp_size != SP_SIZE
        or contract.topology.dp_size != DP_SIZE
        or contract.arm_index not in (0, 1)
        or contract.sp_rank not in range(SP_SIZE)
    ):
        fail("C2 trainer requires one independent exact WORLD8 DP2xSP4 job")
    device = runtime.initialise_distributed(contract)
    parallel = runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    c1.seed_everything(args.seed)
    checkpoint_exact23_pre = validate_checkpoint_exact23_world8_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        checkpoint_root=checkpoint,
        manifest_path=checkpoint_exact23_manifest_path,
        expected_manifest_sha256=(
            args.expected_checkpoint_exact23_manifest_sha256
        ),
        label_module=label_module,
        materializer_module=materializer_module,
        stage="pre_load",
    )
    bernini_execution_pre = validate_bernini_execution_sources_world8_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        legacy_module=legacy,
        materializer_module=materializer_module,
        renderer_module=bernini_renderer_module,
        transformer_wan_module=bernini_transformer_wan_module,
        parallel_module=bernini_parallel_module,
        parallel_state_module=bernini_parallel_state_module,
        veomni_parallel_state_module=veomni_parallel_state_module,
        veomni_sequence_comm_module=veomni_sequence_comm_module,
        renderer_config_class=BerniniRendererConfig,
        renderer_model_class=BerniniRendererModel,
        rotary_class=WanRotaryPosEmbed,
        init_parallel_function=init_parallel_state,
        stage="pre_load",
    )

    strong_box: list[Any] = [None]
    if contract.rank == 0:
        try:
            strong_box[0] = {
                "ok": True,
                "value": validate_model_authority_strong_v1(
                    materializer_module=materializer_module,
                    path=model_path,
                    expected_sha256=args.expected_model_authority_sha256,
                    bernini_root=bernini_root,
                    checkpoint_root=checkpoint,
                    pipeline_module=bernini_pipeline,
                    diffusers_module=diffusers,
                    wan_module=diffusers_wan,
                ),
            }
        except Exception as error:
            strong_box[0] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(strong_box, src=0, group=parallel.world_group)
    if not isinstance(strong_box[0], Mapping) or strong_box[0].get("ok") is not True:
        fail(f"rank-zero strong model authority validation failed: {strong_box[0]!r}")
    strong_model_pre = strong_box[0]["value"]

    row_index = int(contract.arm_index)
    row_id = ROW_IDS[row_index]
    bundle = load_c2_latent_bundle_v1(
        bundle_path=bundle_path,
        expected_bundle_sha256=args.expected_latent_bundle_sha256,
        receipt_path=bundle_receipt_path,
        expected_receipt_sha256=args.expected_latent_bundle_receipt_sha256,
        packet_root=packet_root,
        local_row_index=row_index,
        label_module=label_module,
        materializer_module=materializer_module,
    )
    labels = {
        variant: label_module.load_oracle_q_label_v1(
            packet_root,
            row_id=row_id,
            media_variant=variant,
            patch_grid=PATCH_GRID,
            external_authority_path=external_path,
            external_authority_sha256=EXTERNAL_AUTHORITY_SHA256,
            experiment_contract_path=experiment_path,
            experiment_contract_sha256=EXPERIMENT_CONTRACT_SHA256,
            device=device,
            dtype=torch.float32,
        )
        for variant in TRAIN_VARIANTS
    }
    target_mismatch = label_module.build_role_only_hybrid_v1(
        labels["target"], labels["role_swap"]
    )
    role_mismatch = label_module.build_role_only_hybrid_v1(
        labels["role_swap"], labels["target"]
    )

    # The model seed is reset immediately at the fresh constructor boundary;
    # validation/label work above cannot perturb A/B initial bytes.
    c1.seed_everything(args.seed)
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with c1.serialized_model_load_v1():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        specs = packed_lora.select_projection_specs(renderer, "all-attention")
        model = get_peft_model(
            renderer,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_RANK,
                lora_dropout=0.0,
                bias="none",
                target_modules=[item.name for item in specs],
            ),
        )
        transformer = model.get_base_model().diff_dec.transformer
        elal_handle = elal3.install_elal3_c0_v1(
            transformer, variant="full", attention_width=64, hidden_size=HIDDEN
        )
        gate_installation = install_controlled_nonzero_gates_v1(elal_handle)
        checkpoint_blocks = c1.install_selective_activation_checkpointing_v1(
            model, context_fn=elal3.elal3_checkpoint_context_fn_v1
        )
        if checkpoint_blocks != ACTIVATION_CHECKPOINT_BLOCKS:
            fail("C2 selective activation-checkpoint exact8 schedule differs")
        model.to(device)
    post_deserialize_model_replay = replay_strong_model_authority_world8_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        reference=strong_model_pre,
        materializer_module=materializer_module,
        authority_path=model_path,
        expected_sha256=args.expected_model_authority_sha256,
        bernini_root=bernini_root,
        checkpoint_root=checkpoint,
        pipeline_module=bernini_pipeline,
        diffusers_module=diffusers,
        wan_module=diffusers_wan,
        stage="post_deserialize",
    )
    model.train()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or model.__class__.__name__ in {
        "FullyShardedDataParallel",
        "DistributedDataParallel",
    }:
        fail("C2 manual SP4/DP2 ownership forbids DDP/FSDP")
    base_renderer = model.get_base_model()
    base_renderer.t5_text_encoder.eval()
    if any(
        parameter.dtype != torch.float32
        for parameter in elal_handle.components.parameters()
    ):
        fail("C2 ELAL3 full-w64 parameters must remain FP32")
    named = c1.exact_trainable_named_parameters_v1(model)
    inventory = c1.trainable_inventory_v1(named)
    inventory_digest = object_sha256(inventory)
    packed_lora.validate_lora_installation(model, specs)
    runtime.digest_consensus(
        inventory_digest,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="C2 all240-r256 plus ELAL3 full-w64 inventory",
    )
    initial_digest = c1.synchronize_initial_parameters_v1(
        named, parallel.world_group
    )
    if any(
        struct.pack(
            ">f", float(injection.residual_gain.detach().cpu().item())
        ).hex()
        != CONTROLLED_GAIN_FLOAT32_HEX
        for injection in elal_handle.components.injections
    ):
        fail("C2 controlled gain changed during initial synchronization")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    instruction = str(labels["target"].verified_row.row["instruction"])
    text_lens, text_embs, text_receipt = c1.materialize_text_condition_v1(
        tokenizer=tokenizer,
        renderer=base_renderer,
        runtime=runtime,
        instruction=instruction,
        device=device,
    )
    base_renderer.t5_text_encoder = None
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if base_renderer.t5_text_encoder is not None:
        fail("frozen T5 was not retired before any optimizer construction")
    checkpoint_exact23_post = validate_checkpoint_exact23_world8_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        checkpoint_root=checkpoint,
        manifest_path=checkpoint_exact23_manifest_path,
        expected_manifest_sha256=(
            args.expected_checkpoint_exact23_manifest_sha256
        ),
        label_module=label_module,
        materializer_module=materializer_module,
        stage="post_deserialize",
        reference=checkpoint_exact23_pre,
    )
    bernini_execution_post = validate_bernini_execution_sources_world8_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        legacy_module=legacy,
        materializer_module=materializer_module,
        renderer_module=bernini_renderer_module,
        transformer_wan_module=bernini_transformer_wan_module,
        parallel_module=bernini_parallel_module,
        parallel_state_module=bernini_parallel_state_module,
        veomni_parallel_state_module=veomni_parallel_state_module,
        veomni_sequence_comm_module=veomni_sequence_comm_module,
        renderer_config_class=BerniniRendererConfig,
        renderer_model_class=BerniniRendererModel,
        rotary_class=WanRotaryPosEmbed,
        init_parallel_function=init_parallel_state,
        stage="post_deserialize",
        reference=bernini_execution_pre,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)

    # Every stage replays the same actual-shape, zero-update preoptimizer work.
    coordinate0 = sigma_strata.select_sigma_stratum(0)
    epsilon0 = cpu_epsilon_v1(training_noise_seed_v1(args.seed, 0, row_index))
    second_variant = "target" if args.arm_id == ARM_DUPLICATE else "role_swap"
    second_label = labels[second_variant]
    second_clean = bundle.tensor(second_variant)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    if not torch.is_grad_enabled():
        fail("C2 preflight must use a grad-enabled real training graph")
    with torch_checkpoint.set_checkpoint_early_stop(False):
        training_first = renderer_branch_forward_v1(
            transformer=transformer,
            renderer=base_renderer,
            elal_handle=elal_handle,
            elal_module=elal3,
            source=bundle.tensor("source"),
            clean_target=bundle.tensor("target"),
            epsilon=epsilon0,
            coordinate=coordinate0,
            oracle_label=labels["target"],
            rope=rope,
            device=device,
            text_lens=text_lens,
            text_embs=text_embs,
            sp_rank=contract.sp_rank,
            route_identity=f"{row_id}:preoptimizer:target:sp{contract.sp_rank}",
        )
        if training_first["prediction"].grad_fn is None:
            fail("C2 preflight first branch is not a real autograd graph")
        first_prediction_ref = weakref.ref(training_first["prediction"])
        preoptimizer_first_loss, preoptimizer_first_evidence = (
            detach_branch_loss_evidence_v1(
                training_first, label="C2 preoptimizer first"
            )
        )
    if not bool(torch.isfinite(preoptimizer_first_loss.detach()).item()):
        fail("C2 preoptimizer first actual-shape loss is non-finite")
    torch.cuda.synchronize(device)
    preoptimizer_first_peak = int(torch.cuda.max_memory_allocated(device))
    del training_first, preoptimizer_first_loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    if first_prediction_ref() is not None:
        fail("C2 preoptimizer first branch graph survived inter-branch release")
    preoptimizer_post_first_release = int(torch.cuda.memory_allocated(device))
    with torch_checkpoint.set_checkpoint_early_stop(False):
        training_second = renderer_branch_forward_v1(
            transformer=transformer,
            renderer=base_renderer,
            elal_handle=elal_handle,
            elal_module=elal3,
            source=bundle.tensor("source"),
            clean_target=second_clean,
            epsilon=epsilon0.clone(),
            coordinate=coordinate0,
            oracle_label=second_label,
            rope=rope,
            device=device,
            text_lens=text_lens,
            text_embs=text_embs,
            sp_rank=contract.sp_rank,
            route_identity=(
                f"{row_id}:preoptimizer:{second_variant}:second:sp{contract.sp_rank}"
            ),
        )
        if training_second["prediction"].grad_fn is None:
            fail("C2 preflight second branch is not a real autograd graph")
        second_prediction_ref = weakref.ref(training_second["prediction"])
        preoptimizer_second_loss, preoptimizer_second_evidence = (
            detach_branch_loss_evidence_v1(
                training_second, label="C2 preoptimizer second"
            )
        )
    if not bool(torch.isfinite(preoptimizer_second_loss.detach()).item()):
        fail("C2 preoptimizer second actual-shape loss is non-finite")
    torch.cuda.synchronize(device)
    preoptimizer_second_peak = int(torch.cuda.max_memory_allocated(device))
    del training_second, preoptimizer_second_loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    if second_prediction_ref() is not None:
        fail("C2 preoptimizer second branch graph survived post-branch release")
    preoptimizer_post_second_release = int(torch.cuda.memory_allocated(device))
    preoptimizer_objective = sequential_two_branch_objective_receipt_v1(
        arm_id=args.arm_id,
        first_evidence=preoptimizer_first_evidence,
        second_evidence=preoptimizer_second_evidence,
        execution_mode="preflight_forward_only",
    )
    preoptimizer_lifecycle = {
        "execution_mode": "preflight_forward_only",
        "activation_checkpoint_profile": ACTIVATION_CHECKPOINT_PROFILE,
        "activation_checkpointed_blocks": list(ACTIVATION_CHECKPOINT_BLOCKS),
        "activation_uncheckpointed_blocks": list(ACTIVATION_UNCHECKPOINTED_BLOCKS),
        "activation_checkpoint_nonreentrant": True,
        "activation_checkpoint_elal_route_context_replay": True,
        "first_backward_completed": False,
        "second_backward_completed": False,
        "first_prediction_weakref_released_before_second_forward": True,
        "second_prediction_weakref_released_before_post_branch_work": True,
        "first_graph_deleted_before_second_forward": True,
        "second_graph_deleted_before_post_branch_work": True,
        "inter_branch_gc_collect_called": True,
        "inter_branch_cuda_empty_cache_called": True,
        "second_forward_started_after_first_release": True,
        "simultaneous_live_autograd_branch_graphs_maximum": 1,
        "first_gradient_tensors_preserved_across_graph_release": None,
        "gradient_reduce_clip_optimizer_after_both_branches": False,
        "preflight_grad_enabled_training_graph": True,
        "preflight_backward_executed": False,
        "peak_semantics": "maximum_of_sequential_true_grad_enabled_branch_graphs_without_backward",
        "first_branch_peak_allocated_bytes": preoptimizer_first_peak,
        "post_first_release_allocated_bytes": preoptimizer_post_first_release,
        "second_branch_peak_allocated_bytes": preoptimizer_second_peak,
        "post_second_release_allocated_bytes": preoptimizer_post_second_release,
        "dummy_or_padding_allocations": False,
    }
    validate_branch_lifecycle_receipt_v1(
        preoptimizer_lifecycle,
        execution_mode="preflight_forward_only",
        label="C2 preoptimizer",
    )
    preoptimizer_memory = memory_receipt_v1(device, world_rank=contract.rank)
    preoptimizer_runtime = {
        "world_rank": contract.rank,
        "row_index": row_index,
        "row_id": row_id,
        "sp_rank": contract.sp_rank,
        "first_hook": preoptimizer_first_evidence["hook_receipt"],
        "second_hook": preoptimizer_second_evidence["hook_receipt"],
        "first_partition": preoptimizer_first_evidence["registered_sp4_partition"],
        "second_partition": preoptimizer_second_evidence["registered_sp4_partition"],
        "first_actual_input_receipt": preoptimizer_first_evidence[
            "actual_input_receipt"
        ],
        "second_actual_input_receipt": preoptimizer_second_evidence[
            "actual_input_receipt"
        ],
        "objective": preoptimizer_objective,
        "branch_lifecycle": preoptimizer_lifecycle,
        "memory": preoptimizer_memory,
        "optimizer_constructed": False,
    }

    evaluation0, controlled_target_prediction = evaluate_local_row_v1(
        stage="step0",
        arm_seed=args.seed,
        row_index=row_index,
        row_id=row_id,
        source=bundle.tensor("source"),
        target_clean=bundle.tensor("target"),
        role_clean=bundle.tensor("role_swap"),
        target_label=labels["target"],
        role_label=labels["role_swap"],
        target_mismatch=target_mismatch,
        role_mismatch=role_mismatch,
        instruction=instruction,
        transformer=transformer,
        renderer=base_renderer,
        elal_handle=elal_handle,
        elal_module=elal3,
        rope=rope,
        device=device,
        text_lens=text_lens,
        text_embs=text_embs,
        sp_rank=contract.sp_rank,
    )
    controlled_target_prediction_projection = prediction_hash_projection_v1(
        controlled_target_prediction,
        label="step0 controlled target prediction",
    )
    controlled_target_prediction_sha = controlled_target_prediction_projection[
        "prediction_sha256"
    ]
    gain_digest_before = c1.trainable_digest_v1(named)
    if gain_digest_before != initial_digest:
        fail("step0 gain probe did not start from exact initial bytes")
    with temporary_gate_zero_probe_v1(elal_handle), torch.no_grad():
        gate_zero = renderer_branch_forward_v1(
            transformer=transformer,
            renderer=base_renderer,
            elal_handle=elal_handle,
            elal_module=elal3,
            source=bundle.tensor("source"),
            clean_target=bundle.tensor("target"),
            epsilon=cpu_epsilon_v1(evaluation_noise_seed_v1(args.seed, row_index)),
            coordinate=EvaluationCoordinateV1(),
            oracle_label=labels["target"],
            rope=rope,
            device=device,
            text_lens=text_lens,
            text_embs=text_embs,
            sp_rank=contract.sp_rank,
            route_identity=f"{row_id}:step0:gate-zero:sp{contract.sp_rank}",
        )
    gain_digest_after = c1.trainable_digest_v1(named)
    gain_safety = step0_gain_safety_receipt_v1(
        gate_zero["prediction"],
        controlled_target_prediction,
        parameter_digest_before=gain_digest_before,
        parameter_digest_after=gain_digest_after,
    )
    del gate_zero, controlled_target_prediction
    local_schedule = build_local_noise_schedule_receipt_v1(
        arm_seed=args.seed,
        row_index=row_index,
        target_clean=bundle.tensor("target"),
        sigma_module=sigma_strata,
    )
    local_preoptimizer = {
        "world_rank": contract.rank,
        "row_index": row_index,
        "row_id": row_id,
        "local_schedule": local_schedule,
        "evaluation": evaluation0,
        "target_prediction_sha256": controlled_target_prediction_sha,
        "target_prediction_hash_projection": controlled_target_prediction_projection,
        "target_q_digest": labels["target"].receipt["q_tensor_rows_digest"],
        "gain_safety": gain_safety,
        "runtime_telemetry": preoptimizer_runtime,
    }
    gathered_preoptimizer: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_preoptimizer, local_preoptimizer, group=parallel.world_group
    )
    aggregate_box: list[Any] = [None]
    if contract.rank == 0:
        try:
            aggregate_box[0] = {
                "ok": True,
                "value": aggregate_preoptimizer_evidence_v1(
                    gathered_preoptimizer,
                    seed=args.seed,
                    bundle=bundle,
                    materializer_run_binding=materializer_run_binding,
                    checkpoint_exact23_binding=checkpoint_exact23_pre,
                    bernini_execution_source_binding=bernini_execution_pre,
                    source_pins=source_pins,
                    initial_trainable_sha256=initial_digest,
                    trainable_inventory_digest=inventory_digest,
                ),
            }
        except Exception as error:
            aggregate_box[0] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(aggregate_box, src=0, group=parallel.world_group)
    if not isinstance(aggregate_box[0], Mapping) or aggregate_box[0].get("ok") is not True:
        fail(f"C2 preoptimizer aggregate failed: {aggregate_box[0]!r}")
    aggregate = aggregate_box[0]["value"]
    common_payload = aggregate["common"]
    common_digest = object_sha256(common_payload)
    recipe_digest = recipe_version_digest_v1(
        bundle_sha256=bundle.bundle_sha256,
        source_pins=source_pins,
        checkpoint_exact23_binding_digest=checkpoint_exact23_pre[
            "fixed_release_binding_digest"
        ],
        bernini_execution_source_binding_digest=bernini_execution_pre[
            "fixed_release_binding_digest"
        ],
    )

    def final_closure_replay() -> Mapping[str, Any]:
        source_final = validate_runtime_sources_strong_v1(
            runner_sha256=runner_sha,
            runner_size=runner_size,
            materializer_module=materializer_module,
            label_module=label_module,
            elal_module=elal3,
            legacy_module=legacy,
            packed_module=packed_lora,
            runtime_module=runtime,
            sigma_module=sigma_strata,
        )
        if canonical_json_bytes(source_final) != canonical_json_bytes(source_closure_pre):
            fail("runtime source closure changed before publish")
        materializer_run_final = validate_materializer_run_complete_v1(
            materializer_run_path,
            expected_sha256=args.expected_materializer_run_complete_sha256,
            label_module=label_module,
        )
        if canonical_json_bytes(materializer_run_final) != canonical_json_bytes(
            materializer_run_binding
        ):
            fail("materializer retry2 RUN_COMPLETE changed before publish")
        checkpoint_exact23_final = validate_checkpoint_exact23_world8_v1(
            dist=dist,
            group=parallel.world_group,
            rank=contract.rank,
            checkpoint_root=checkpoint,
            manifest_path=checkpoint_exact23_manifest_path,
            expected_manifest_sha256=(
                args.expected_checkpoint_exact23_manifest_sha256
            ),
            label_module=label_module,
            materializer_module=materializer_module,
            stage="final_pre_publish",
            reference=checkpoint_exact23_pre,
        )
        bernini_execution_final = validate_bernini_execution_sources_world8_v1(
            dist=dist,
            group=parallel.world_group,
            rank=contract.rank,
            bernini_root=bernini_root,
            veomni_root=veomni_root,
            legacy_module=legacy,
            materializer_module=materializer_module,
            renderer_module=bernini_renderer_module,
            transformer_wan_module=bernini_transformer_wan_module,
            parallel_module=bernini_parallel_module,
            parallel_state_module=bernini_parallel_state_module,
            veomni_parallel_state_module=veomni_parallel_state_module,
            veomni_sequence_comm_module=veomni_sequence_comm_module,
            renderer_config_class=BerniniRendererConfig,
            renderer_model_class=BerniniRendererModel,
            rotary_class=WanRotaryPosEmbed,
            init_parallel_function=init_parallel_state,
            stage="final_pre_publish",
            reference=bernini_execution_pre,
        )
        if (
            checkpoint_exact23_final["fixed_release_binding_digest"]
            != checkpoint_exact23_post["fixed_release_binding_digest"]
            or bernini_execution_final["fixed_release_binding_digest"]
            != bernini_execution_post["fixed_release_binding_digest"]
        ):
            fail("training checkpoint/source post/final replay differs")
        model_final = replay_strong_model_authority_world8_v1(
            dist=dist,
            group=parallel.world_group,
            rank=contract.rank,
            reference=strong_model_pre,
            materializer_module=materializer_module,
            authority_path=model_path,
            expected_sha256=args.expected_model_authority_sha256,
            bernini_root=bernini_root,
            checkpoint_root=checkpoint,
            pipeline_module=bernini_pipeline,
            diffusers_module=diffusers,
            wan_module=diffusers_wan,
            stage="final_pre_publish",
        )
        if (
            model_final["strong_replay_digest"]
            != post_deserialize_model_replay["strong_replay_digest"]
        ):
            fail("model exact9 post-deserialize/final identity replay differs")
        bundle_final = load_c2_latent_bundle_v1(
            bundle_path=bundle_path,
            expected_bundle_sha256=args.expected_latent_bundle_sha256,
            receipt_path=bundle_receipt_path,
            expected_receipt_sha256=args.expected_latent_bundle_receipt_sha256,
            packet_root=packet_root,
            local_row_index=row_index,
            label_module=label_module,
            materializer_module=materializer_module,
        )
        if (
            bundle_final.receipt != bundle.receipt
            or any(
                c1.tensor_sha256_v1(bundle_final.tensor(variant))
                != c1.tensor_sha256_v1(bundle.tensor(variant))
                for variant in ("source", "target", "role_swap")
            )
        ):
            fail("exact16 bundle changed before publish")
        final_label_digests = {}
        for variant in TRAIN_VARIANTS:
            candidate = label_module.load_oracle_q_label_v1(
                packet_root,
                row_id=row_id,
                media_variant=variant,
                patch_grid=PATCH_GRID,
                external_authority_path=external_path,
                external_authority_sha256=EXTERNAL_AUTHORITY_SHA256,
                experiment_contract_path=experiment_path,
                experiment_contract_sha256=EXPERIMENT_CONTRACT_SHA256,
                device=device,
                dtype=torch.float32,
            )
            if candidate.receipt["label_digest"] != labels[variant].receipt["label_digest"]:
                fail("oracle label changed before publish")
            final_label_digests[variant] = candidate.receipt["label_digest"]
        local_closure = {
            "row_index": row_index,
            "row_id": row_id,
            "runtime_sources_pre_final_bit_exact": True,
            "runtime_source_release_pin_digest": source_pins["release_pin_digest"],
            "model_exact9_pre_post_final_stable": True,
            "model_authority_sha256": MODEL_AUTHORITY_SHA256,
            "model_authority_digest": MODEL_AUTHORITY_DIGEST,
            "bundle_exact16_pre_final_stable": True,
            "bundle_sha256": bundle.bundle_sha256,
            "bundle_receipt_sha256": bundle.receipt_sha256,
            "materializer_run_complete_pre_final_stable": True,
            "materializer_run_complete_sha256": materializer_run_binding[
                "file_sha256"
            ],
            "materializer_run_complete_digest": materializer_run_binding[
                "run_digest"
            ],
            "checkpoint_exact23_pre_post_final_stable": True,
            "checkpoint_exact23_manifest_sha256": (
                CHECKPOINT_EXACT23_MANIFEST_SHA256
            ),
            "checkpoint_exact23_binding_digest": checkpoint_exact23_pre[
                "fixed_release_binding_digest"
            ],
            "bernini_execution_sources_pre_post_final_stable": True,
            "bernini_execution_source_binding_digest": bernini_execution_pre[
                "fixed_release_binding_digest"
            ],
            "oracle_labels_pre_final_stable": True,
            "oracle_label_digests": final_label_digests,
            "runtime_telemetry": {
                "runtime_source_pre_and_final_binding": source_final,
                "runtime_source_identity_closure_digest": source_final[
                    "source_closure_digest"
                ],
                "model_pre_binding": strong_model_pre,
                "model_strong_identity_replay_digest": model_final[
                    "strong_replay_digest"
                ],
                "model_post_deserialize_identity_replay_digest": (
                    post_deserialize_model_replay["strong_replay_digest"]
                ),
                "checkpoint_exact23_final_runtime": checkpoint_exact23_final[
                    "runtime_telemetry"
                ],
                "bernini_execution_final_runtime": bernini_execution_final[
                    "runtime_telemetry"
                ],
            },
        }
        gathered_closure: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(
            gathered_closure, local_closure, group=parallel.world_group
        )
        leaders = []
        for index in (0, 1):
            rows = gathered_closure[index * SP_SIZE : (index + 1) * SP_SIZE]
            if (
                any(row.get("row_index") != index for row in rows)
                or any(
                    canonical_json_bytes(rows[0]) != canonical_json_bytes(row)
                    for row in rows[1:]
                )
            ):
                fail("final closure replay lacks SP4 row consensus")
            leaders.append(rows[0])
        fixed_keys = (
            "runtime_sources_pre_final_bit_exact",
            "runtime_source_release_pin_digest",
            "model_exact9_pre_post_final_stable",
            "model_authority_sha256",
            "model_authority_digest",
            "bundle_exact16_pre_final_stable",
            "bundle_sha256",
            "bundle_receipt_sha256",
            "materializer_run_complete_pre_final_stable",
            "materializer_run_complete_sha256",
            "materializer_run_complete_digest",
            "checkpoint_exact23_pre_post_final_stable",
            "checkpoint_exact23_manifest_sha256",
            "checkpoint_exact23_binding_digest",
            "bernini_execution_sources_pre_post_final_stable",
            "bernini_execution_source_binding_digest",
            "oracle_labels_pre_final_stable",
        )
        if any(
            leaders[0][key] != leaders[1][key]
            for key in fixed_keys
        ):
            fail("final closure common exact2 binding differs")
        return {
            **{key: leaders[0][key] for key in fixed_keys},
            "runtime_telemetry": leaders[0]["runtime_telemetry"],
            "oracle_label_digests_by_row": {
                leader["row_id"]: leader["oracle_label_digests"]
                for leader in leaders
            },
            "exact2_rows_replayed": True,
        }

    if args.preflight_only:
        closure_replays = final_closure_replay()
        job_id, node, _ = ARM_PLACEMENT[args.arm_id]
        second_descriptor = {
            "recipe": (
                "target_duplicate_exact2"
                if args.arm_id == ARM_DUPLICATE
                else "target_and_role_swap_exact2"
            ),
            "second_variant": second_variant,
            "row_descriptors": [
                {
                    "row_id": leader["row_id"],
                    "target_q_digest": leader["target_q_digest"],
                    "second_variant": second_variant,
                }
                for leader in aggregate["leaders"]
            ],
            "exactly_one_whitelisted_difference_from_common_target": True,
        }
        unsigned_preflight = {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "PRECHECK_COMPLETE_NO_OPTIMIZER_NO_UPDATE",
            "method": METHOD,
            "arm_id": args.arm_id,
            "branch_recipe": second_descriptor["recipe"],
            "holder_job_id": job_id,
            "node": node,
            "seed": args.seed,
            "preflight_only": True,
            "completed_optimizer_steps": 0,
            "optimizer_constructed": False,
            "resume_consumed": False,
            "recipe_version_digest": recipe_digest,
            "common_comparison_payload": common_payload,
            "common_comparison_payload_digest": common_digest,
            "initial_trainable_sha256": initial_digest,
            "row_input_noise_schedule_digest": common_payload[
                "row_input_noise_schedule_digest"
            ],
            "second_branch_descriptor": second_descriptor,
            "actual_shape_preflight": aggregate["actual_shape_preflight"],
            "step0_gain_safety": aggregate["step0_gain_safety"],
            "step0_full_q_route": aggregate["step0_full_q_route"],
            "step0_role_only_cells": aggregate["step0_role_only_cells"],
            "step0_role_only_input_invariants": aggregate[
                "step0_role_only_input_invariants"
            ],
            "step0_role_only_input_invariants_validation": aggregate[
                "step0_role_only_input_invariants_validation"
            ],
            "step0_evaluation_forward_evidence": aggregate[
                "step0_evaluation_forward_evidence"
            ],
            "step0_evaluation_forward_evidence_validation": aggregate[
                "step0_evaluation_forward_evidence_validation"
            ],
            "all_preflight_hard_gates_pass": True,
            "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
            "external_authority_sha256": EXTERNAL_AUTHORITY_SHA256,
            "model_authority_sha256": MODEL_AUTHORITY_SHA256,
            "latent_bundle_sha256": bundle.bundle_sha256,
            "runner_source_sha256": runner_sha,
            "source_pins": dict(source_pins),
            "claim_boundaries": dict(CLAIM_BOUNDARIES),
            "pre_publish_closure_replays": closure_replays,
        }
        preflight = {
            **unsigned_preflight,
            "receipt_digest": object_sha256(unsigned_preflight),
        }
        _validate_preflight_receipt_value_v1(
            preflight,
            arm_id=args.arm_id,
            holder_job_id=job_id,
            node=node,
            seed=args.seed,
            expected_receipt_digest=preflight["receipt_digest"],
            expected_runner_sha256=runner_sha,
            expected_bundle_sha256=bundle.bundle_sha256,
            expected_source_pins=source_pins,
        )
        output = Path(args.output)
        if contract.rank == 0:
            output.mkdir(mode=0o700)
            c1.atomic_create_json(output / "PRECHECK_RECEIPT.json", preflight)
            os.chmod(output, 0o555)
        dist.barrier(group=parallel.world_group)
        return 0

    # Recompute-first discipline: no optimizer object exists above this line.
    own_preflight = validate_own_preflight_receipt_v1(
        Path(args.own_preflight_receipt),
        expected_sha256=args.expected_own_preflight_receipt_sha256,
        arm_id=args.arm_id,
        expected_runner_sha256=runner_sha,
        expected_bundle_sha256=bundle.bundle_sha256,
        expected_source_pins=source_pins,
    )
    cross_gate = validate_cross_arm_preflight_gate_v1(
        Path(args.cross_arm_preflight_gate),
        expected_sha256=args.expected_cross_arm_preflight_gate_sha256,
        expected_runner_sha256=runner_sha,
        expected_bundle_sha256=bundle.bundle_sha256,
        expected_source_pins=source_pins,
    )
    if (
        initial_digest != own_preflight["initial_trainable_sha256"]
        or common_digest != own_preflight["common_comparison_payload_digest"]
        or common_payload["row_input_noise_schedule_digest"]
        != own_preflight["row_input_noise_schedule_digest"]
        or recipe_digest != own_preflight["recipe_version_digest"]
        or recipe_digest != cross_gate["recipe_version_digest"]
    ):
        fail("fresh stage recomputation differs from its own sealed preflight")
    if args.arm_id in (ARM_DUPLICATE, ARM_ROLE_PAIR) and (
        initial_digest != cross_gate["common_initial_trainable_sha256"]
        or common_digest != cross_gate["common_comparison_payload_digest"]
        or common_payload["row_input_noise_schedule_digest"]
        != cross_gate["common_row_input_noise_schedule_digest"]
    ):
        fail("A/B fresh stage differs from sealed cross-arm common inputs")
    fresh1_gate = None
    fresh1_gate_binding = None
    if args.max_steps == MAX_STEPS:
        expected_origin_verifier_binding = {
            "name": args.fresh1_origin_verifier_name,
            "sha256": args.expected_fresh1_origin_verifier_sha256,
            "size": args.expected_fresh1_origin_verifier_size,
            "mode": 0o444,
            "nlink": 1,
        }
        expected_gate_controller_binding = {
            "name": args.fresh1_gate_controller_name,
            "sha256": args.expected_fresh1_gate_controller_sha256,
            "size": args.expected_fresh1_gate_controller_size,
            "mode": 0o444,
            "nlink": 1,
        }
        fresh1_gate = validate_fresh1_acceptance_gate_v1(
            Path(args.fresh1_acceptance_gate),
            expected_sha256=args.expected_fresh1_acceptance_gate_sha256,
            expected_runner_sha256=runner_sha,
            expected_bundle_sha256=bundle.bundle_sha256,
            expected_source_pins=source_pins,
            cross_gate=cross_gate,
            expected_origin_verifier_binding=expected_origin_verifier_binding,
            expected_gate_controller_binding=expected_gate_controller_binding,
        )
        fresh1_gate_binding = {
            "path": str(Path(args.fresh1_acceptance_gate).resolve(strict=True)),
            "sha256": args.expected_fresh1_acceptance_gate_sha256,
            **fresh1_gate,
        }

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=DEFAULT_LR,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    if optimizer.state or any(group.get("params") is None for group in optimizer.param_groups):
        fail("fresh optimizer state is not empty before first update")
    output = Path(args.output)
    checkpoints = output / "checkpoints"
    if contract.rank == 0:
        output.mkdir(mode=0o700)
        checkpoints.mkdir(mode=0o700)
    dist.barrier(group=parallel.world_group)
    checkpoint_common = {
        "method": METHOD,
        "arm_id": args.arm_id,
        "seed": args.seed,
        "fresh_official_base": True,
        "resume_consumed": False,
        "initial_trainable_sha256": initial_digest,
        "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
        "latent_bundle_sha256": bundle.bundle_sha256,
        "formal_c2_authorized": False,
        "exact160_authorized": False,
        "source_instruction_inference": False,
    }
    checkpoint_records: list[Any] = []
    checkpoint_box: list[Any] = [None]
    if contract.rank == 0:
        checkpoint_box[0] = save_checkpoint_v1(
            root=checkpoints,
            step=0,
            named=named,
            optimizer=optimizer,
            common=checkpoint_common,
            save_optimizer=False,
        )
    dist.broadcast_object_list(checkpoint_box, src=0, group=parallel.world_group)
    checkpoint_records.append(checkpoint_box[0])

    history: list[Mapping[str, Any]] = []
    parameter_digests = {initial_digest}
    started = time.monotonic()
    for step_zero in range(args.max_steps):
        completed = step_zero + 1
        coordinate = sigma_strata.select_sigma_stratum(step_zero)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        epsilon = cpu_epsilon_v1(
            training_noise_seed_v1(args.seed, step_zero, row_index)
        )
        clean_second = bundle.tensor(second_variant)
        label_second = labels[second_variant]
        with torch_checkpoint.set_checkpoint_early_stop(False):
            first = renderer_branch_forward_v1(
                transformer=transformer,
                renderer=base_renderer,
                elal_handle=elal_handle,
                elal_module=elal3,
                source=bundle.tensor("source"),
                clean_target=bundle.tensor("target"),
                epsilon=epsilon,
                coordinate=coordinate,
                oracle_label=labels["target"],
                rope=rope,
                device=device,
                text_lens=text_lens,
                text_embs=text_embs,
                sp_rank=contract.sp_rank,
                route_identity=(
                    f"{row_id}:train:step{completed}:target:sp{contract.sp_rank}"
                ),
            )
            first_prediction_ref = weakref.ref(first["prediction"])
            first_branch_loss, first_evidence = detach_branch_loss_evidence_v1(
                first, label=f"C2 training step {completed} first"
            )
            first_weighted_loss = first_branch_loss * 0.5
            first_backward_loss = first_weighted_loss + c1.all_trainable_graph_zero_v1(
                named, reference=first_weighted_loss
            )
            if not bool(torch.isfinite(first_backward_loss.detach()).item()):
                fail("C2 first sequential branch objective is non-finite")
            first_backward_loss.backward()
        first_gradient_guard = gradient_accumulation_guard_v1(
            named, label=f"C2 training step {completed} first branch"
        )
        torch.cuda.synchronize(device)
        first_branch_peak = int(torch.cuda.max_memory_allocated(device))
        del first, first_branch_loss, first_weighted_loss, first_backward_loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        if first_prediction_ref() is not None:
            fail("C2 first training graph survived before second forward")
        validate_gradient_accumulation_guard_v1(
            named,
            first_gradient_guard,
            label=f"C2 training step {completed} graph release",
        )
        post_first_release_allocated = int(torch.cuda.memory_allocated(device))
        with torch_checkpoint.set_checkpoint_early_stop(False):
            second = renderer_branch_forward_v1(
                transformer=transformer,
                renderer=base_renderer,
                elal_handle=elal_handle,
                elal_module=elal3,
                source=bundle.tensor("source"),
                clean_target=clean_second,
                epsilon=epsilon.clone(),
                coordinate=coordinate,
                oracle_label=label_second,
                rope=rope,
                device=device,
                text_lens=text_lens,
                text_embs=text_embs,
                sp_rank=contract.sp_rank,
                route_identity=(
                    f"{row_id}:train:step{completed}:{second_variant}:second:"
                    f"sp{contract.sp_rank}"
                ),
            )
            second_prediction_ref = weakref.ref(second["prediction"])
            second_branch_loss, second_evidence = detach_branch_loss_evidence_v1(
                second, label=f"C2 training step {completed} second"
            )
            second_weighted_loss = second_branch_loss * 0.5
            # The first branch's single graph-zero term already created the
            # exact668 local gradient buffers.  Repeating it here is
            # mathematically redundant and would allocate needless full-size
            # zero-gradient temporaries while those buffers are retained.
            second_backward_loss = second_weighted_loss
            if not bool(torch.isfinite(second_backward_loss.detach()).item()):
                fail("C2 second sequential branch objective is non-finite")
            second_backward_loss.backward()
        torch.cuda.synchronize(device)
        second_branch_peak = int(torch.cuda.max_memory_allocated(device))
        del second, second_branch_loss, second_weighted_loss, second_backward_loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        if second_prediction_ref() is not None:
            fail("C2 second training graph survived before gradient reduction")
        post_second_release_allocated = int(torch.cuda.memory_allocated(device))
        objective_receipt = sequential_two_branch_objective_receipt_v1(
            arm_id=args.arm_id,
            first_evidence=first_evidence,
            second_evidence=second_evidence,
            execution_mode="training_forward_backward",
        )
        branch_lifecycle = {
            "execution_mode": "training_forward_backward",
            "activation_checkpoint_profile": ACTIVATION_CHECKPOINT_PROFILE,
            "activation_checkpointed_blocks": list(ACTIVATION_CHECKPOINT_BLOCKS),
            "activation_uncheckpointed_blocks": list(ACTIVATION_UNCHECKPOINTED_BLOCKS),
            "activation_checkpoint_nonreentrant": True,
            "activation_checkpoint_elal_route_context_replay": True,
            "first_backward_completed": True,
            "second_backward_completed": True,
            "first_prediction_weakref_released_before_second_forward": True,
            "second_prediction_weakref_released_before_post_branch_work": True,
            "first_graph_deleted_before_second_forward": True,
            "second_graph_deleted_before_post_branch_work": True,
            "inter_branch_gc_collect_called": True,
            "inter_branch_cuda_empty_cache_called": True,
            "second_forward_started_after_first_release": True,
            "simultaneous_live_autograd_branch_graphs_maximum": 1,
            "first_gradient_tensors_preserved_across_graph_release": True,
            "gradient_reduce_clip_optimizer_after_both_branches": True,
            "preflight_grad_enabled_training_graph": None,
            "preflight_backward_executed": False,
            "peak_semantics": "maximum_of_sequential_true_branch_graphs_with_retained_parameter_gradients",
            "first_branch_peak_allocated_bytes": first_branch_peak,
            "post_first_release_allocated_bytes": post_first_release_allocated,
            "second_branch_peak_allocated_bytes": second_branch_peak,
            "post_second_release_allocated_bytes": post_second_release_allocated,
            "dummy_or_padding_allocations": False,
        }
        validate_branch_lifecycle_receipt_v1(
            branch_lifecycle,
            execution_mode="training_forward_backward",
            label=f"C2 training step {completed}",
        )
        local_ready = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            for _, parameter in named
        )
        if not runtime.world_all_true(local_ready, group=parallel.world_group):
            fail("C2 some SP rank lacks finite explicit local gradients")
        local_elal_sq = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named
            if ".elal3_c0_v1." in name
        )
        target_owner = contract.sp_rank in (2, 3)
        if (target_owner and local_elal_sq <= 0.0) or (
            not target_owner and local_elal_sq != 0.0
        ):
            fail("C2 local SP4 target-owner/source-only gradient split differs")
        synchronized_norm = c1.synchronize_gradients_v1(named, parallel)
        gradient_receipt = c1.gradient_audit_v1(
            named, completed_step=completed, sp_rank=contract.sp_rank
        )
        preclip = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], DEFAULT_MAX_GRAD_NORM
            ).item()
        )
        if not math.isfinite(preclip) or preclip <= 0.0:
            fail("C2 preclip gradient norm is zero/non-finite")
        optimizer.step()
        torch.cuda.synchronize(device)
        local_memory = memory_receipt_v1(device, world_rank=contract.rank)
        memory_world: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(memory_world, local_memory, group=parallel.world_group)
        if (
            [row["world_rank"] for row in memory_world] != list(range(WORLD_SIZE))
            or any(row["strictly_greater_than_half"] is not True for row in memory_world)
        ):
            fail("C2 per-rank >50% true-training memory gate failed")
        parameter_digest = c1.trainable_digest_v1(named)
        runtime.digest_consensus(
            parameter_digest,
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"C2 post-update parameter step {completed}",
        )
        if parameter_digest in parameter_digests:
            fail("C2 optimizer update did not change exact trainable bytes")
        parameter_digests.add(parameter_digest)
        local_step = {
            "world_rank": contract.rank,
            "row_index": row_index,
            "row_id": row_id,
            "sp_rank": contract.sp_rank,
            "target_owner": target_owner,
            "objective": objective_receipt,
            "branch_lifecycle": branch_lifecycle,
            "first_hook": first_evidence["hook_receipt"],
            "second_hook": second_evidence["hook_receipt"],
            "first_partition": first_evidence["registered_sp4_partition"],
            "second_partition": second_evidence["registered_sp4_partition"],
            "first_actual_input_receipt": first_evidence["actual_input_receipt"],
            "second_actual_input_receipt": second_evidence["actual_input_receipt"],
            "local_elal_gradient_norm_before_reduction": math.sqrt(local_elal_sq),
            "gradient_audit": gradient_receipt,
        }
        gathered_step: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_step, local_step, group=parallel.world_group)
        if contract.rank == 0:
            step_receipt = {
                "step": completed,
                "sigma_coordinate": coordinate.as_dict(),
                "noise_seeds_by_row": [
                    training_noise_seed_v1(args.seed, step_zero, index)
                    for index in (0, 1)
                ],
                "synchronized_gradient_norm": synchronized_norm,
                "preclip_gradient_norm": preclip,
                "parameter_sha256": parameter_digest,
                "all8_actual_graph_receipts": gathered_step,
                "memory_world8": memory_world,
                "memory_gate_all8_pass": True,
                "optimizer_step_executed": True,
            }
            history.append(step_receipt)
            print(json.dumps(step_receipt, sort_keys=True), flush=True)
        del first_evidence, second_evidence, epsilon, first_gradient_guard

    final_digest = c1.trainable_digest_v1(named)
    step10_gate = None
    step10_evidence = None
    latent_gate_pass = True
    latent_gate_error = None
    if args.max_steps == MAX_STEPS:
        evaluation10, prediction10 = evaluate_local_row_v1(
            stage="step10",
            arm_seed=args.seed,
            row_index=row_index,
            row_id=row_id,
            source=bundle.tensor("source"),
            target_clean=bundle.tensor("target"),
            role_clean=bundle.tensor("role_swap"),
            target_label=labels["target"],
            role_label=labels["role_swap"],
            target_mismatch=target_mismatch,
            role_mismatch=role_mismatch,
            instruction=instruction,
            transformer=transformer,
            renderer=base_renderer,
            elal_handle=elal_handle,
            elal_module=elal3,
            rope=rope,
            device=device,
            text_lens=text_lens,
            text_embs=text_embs,
            sp_rank=contract.sp_rank,
        )
        prediction10_projection = prediction_hash_projection_v1(
            prediction10,
            label="step10 controlled target prediction",
        )
        local_eval10 = {
            "world_rank": contract.rank,
            "row_index": row_index,
            "evaluation": evaluation10,
            "target_prediction_sha256": prediction10_projection[
                "prediction_sha256"
            ],
            "target_prediction_hash_projection": prediction10_projection,
        }
        del prediction10
        gathered_eval10: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_eval10, local_eval10, group=parallel.world_group)
        leaders10 = []
        for index in (0, 1):
            group_rows = gathered_eval10[index * SP_SIZE : (index + 1) * SP_SIZE]
            if (
                len(group_rows) != SP_SIZE
                or [row.get("world_rank") for row in group_rows]
                != list(range(index * SP_SIZE, (index + 1) * SP_SIZE))
                or any(row.get("row_index") != index for row in group_rows)
            ):
                fail("C2 step10 DP2xSP4 evaluation placement differs")
            scalar_view = lambda item: {
                key: value
                for key, value in item.items()
                if key not in {
                    "actual_forward_evidence",
                    "actual_forward_evidence_validation",
                }
            }
            projection_views = [
                prediction_hash_projection_consensus_view_v1(
                    row.get("target_prediction_hash_projection"),
                    expected_prediction_sha256=str(
                        row.get("target_prediction_sha256")
                    ),
                    expected_original_device_index=int(row["world_rank"]),
                    label=f"step10 row {index} SP{sp_index}",
                )
                for sp_index, row in enumerate(group_rows)
            ]
            if any(
                canonical_json_bytes(scalar_view(group_rows[0]["evaluation"]))
                != canonical_json_bytes(scalar_view(row["evaluation"]))
                or group_rows[0]["target_prediction_sha256"]
                != row["target_prediction_sha256"]
                for row in group_rows[1:]
            ) or any(
                canonical_json_bytes(projection_views[0])
                != canonical_json_bytes(row)
                for row in projection_views[1:]
            ):
                fail("C2 step10 SP4 evaluation consensus differs")
            leaders10.append(group_rows[0])
        full10 = {
            ROW_IDS[index]: leaders10[index]["evaluation"]["full_q_route"]
            for index in (0, 1)
        }
        cells10 = [
            cell
            for leader in leaders10
            for cell in leader["evaluation"]["role_only_cells"]
        ]
        invariants10 = [
            leader["evaluation"]["role_only_input_invariants"][clean_variant]
            for leader in leaders10
            for clean_variant in TRAIN_VARIANTS
        ]
        invariants10_validation = validate_role_only_invariant_receipts_v1(
            invariants10, stage="step10"
        )
        step10_evidence = {
            "full_q_route": full10,
            "role_only_cells": cells10,
            "role_only_input_invariants": invariants10,
            "role_only_input_invariants_validation": invariants10_validation,
            "target_prediction_sha256_by_row": [
                leader["target_prediction_sha256"] for leader in leaders10
            ],
            "target_prediction_hash_projection_by_row": [
                leader["target_prediction_hash_projection"] for leader in leaders10
            ],
            "actual_forward_evidence_by_row": [
                {
                    "row_id": ROW_IDS[index],
                    "input_payload": leaders10[index]["evaluation"]["input_payload"],
                    "actual_forward_evidence": leaders10[index]["evaluation"]["actual_forward_evidence"],
                    "validation": leaders10[index]["evaluation"]["actual_forward_evidence_validation"],
                    "observation_validation": leaders10[index]["evaluation"]["observation_binding_validation"],
                }
                for index in (0, 1)
            ],
        }
        try:
            step10_gate = validate_step10_gates_v1(
                step0_full_q=aggregate["step0_full_q_route"],
                step10_full_q=full10,
                step0_role_only=aggregate["step0_role_only_cells"],
                step10_role_only=cells10,
            )
        except ELAL3C2TrainingError as error:
            latent_gate_pass = False
            latent_gate_error = str(error)

    final_checkpoint_box: list[Any] = [None]
    if contract.rank == 0:
        final_checkpoint_box[0] = save_checkpoint_v1(
            root=checkpoints,
            step=args.max_steps,
            named=named,
            optimizer=optimizer,
            common=checkpoint_common,
            save_optimizer=True,
        )
    dist.broadcast_object_list(
        final_checkpoint_box, src=0, group=parallel.world_group
    )
    checkpoint_records.append(final_checkpoint_box[0])
    checkpoint_tree_box: list[Any] = [None]
    expected_checkpoint_steps = (0, args.max_steps)
    expected_checkpoint_parameters = {
        0: initial_digest,
        args.max_steps: final_digest,
    }
    if contract.rank == 0:
        checkpoint_tree_box[0] = seal_and_validate_checkpoint_tree_v1(
            checkpoints,
            records=checkpoint_records,
            expected_steps=expected_checkpoint_steps,
            expected_parameter_sha256_by_step=expected_checkpoint_parameters,
            expected_common=checkpoint_common,
        )
    dist.broadcast_object_list(
        checkpoint_tree_box, src=0, group=parallel.world_group
    )
    checkpoint_tree_closure = checkpoint_tree_box[0]
    closure_replays = final_closure_replay()
    if contract.rank == 0:
        checkpoint_tree_replay = seal_and_validate_checkpoint_tree_v1(
            checkpoints,
            records=checkpoint_records,
            expected_steps=expected_checkpoint_steps,
            expected_parameter_sha256_by_step=expected_checkpoint_parameters,
            expected_common=checkpoint_common,
        )
        if checkpoint_tree_replay != checkpoint_tree_closure:
            fail("checkpoint tree changed across final authority/source replay")
    dist.barrier(group=parallel.world_group)
    closure_replays = {
        **closure_replays,
        "checkpoint_tree_pre_publish_stable": True,
        "checkpoint_tree_binding_digest": checkpoint_tree_closure[
            "tree_binding_digest"
        ],
        "checkpoint_portable_tree_digest": checkpoint_tree_closure[
            "portable_checkpoint_tree_digest"
        ],
    }
    own_binding = {
        "path": str(Path(args.own_preflight_receipt).resolve(strict=True)),
        "sha256": args.expected_own_preflight_receipt_sha256,
        "receipt_digest": own_preflight["receipt_digest"],
    }
    cross_binding = {
        "path": str(Path(args.cross_arm_preflight_gate).resolve(strict=True)),
        "sha256": args.expected_cross_arm_preflight_gate_sha256,
        "gate_digest": cross_gate["gate_digest"],
        "recipe_version_digest": cross_gate["recipe_version_digest"],
    }
    if contract.rank == 0:
        if args.max_steps == 1:
            unsigned_receipt = {
                "schema_version": RECEIPT_SCHEMA,
                "status": "FRESH1_ENGINEERING_ACCEPTANCE_COMPLETE",
                "method": METHOD,
                "arm_id": args.arm_id,
                "branch_recipe": (
                    "target_duplicate_exact2"
                    if args.arm_id == ARM_DUPLICATE
                    else "target_and_role_swap_exact2"
                ),
                "holder_job_id": placement["holder_job_id"],
                "node": placement["node"],
                "seed": args.seed,
                "preflight_only": False,
                "requested_optimizer_steps": 1,
                "completed_optimizer_steps": 1,
                "optimizer_constructed": True,
                "optimizer_state_empty_before_first_update": True,
                "resume_consumed": False,
                "fresh_official_base": True,
                "initial_trainable_sha256": initial_digest,
                "final_trainable_sha256": final_digest,
                "parameters_changed": final_digest != initial_digest,
                "common_comparison_payload_digest": common_digest,
                "row_input_noise_schedule_digest": common_payload[
                    "row_input_noise_schedule_digest"
                ],
                "own_preflight_binding": own_binding,
                "cross_arm_gate_binding": cross_binding,
                "actual_shape_training": {
                    "actual_shape_two_branch_forward_pass": True,
                    "all30_each_branch_used": True,
                    "sp4_partition_all8_pass": True,
                    "cross_arm_collective_used": False,
                    "strict_sequential_branch_forward_backward": True,
                    "fixed_branch_coefficients": [0.5, 0.5],
                    "first_graph_released_before_second_forward": True,
                    "simultaneous_live_autograd_branch_graphs_maximum": 1,
                    "reduce_clip_optimizer_after_both_branches": True,
                    "activation_checkpoint_profile": ACTIVATION_CHECKPOINT_PROFILE,
                    "activation_checkpointed_blocks": list(
                        ACTIVATION_CHECKPOINT_BLOCKS
                    ),
                    "activation_checkpoint_nonreentrant": True,
                    "activation_checkpoint_elal_route_context_replay": True,
                },
                "gradient_gate": {
                    "all_trainable_parameters_have_finite_gradients": True,
                    "all30_elal_nonzero_after_manual_sp4_dp2_reduction": True,
                    "finite_nonzero_synchronized_gradient_norm": True,
                },
                "memory_gate": {
                    "all8_peak_allocated_strictly_gt_half": True,
                    "dummy_or_padding_allocations": False,
                },
                "checkpoint_gate": {
                    "step0_create_only_reload_pass": checkpoint_records[0][
                        "strict_reload_pass"
                    ],
                    "step1_create_only_reload_pass": checkpoint_records[1][
                        "strict_reload_pass"
                    ],
                    "step1_parameter_sha256": final_digest,
                    "step0_checkpoint_record": checkpoint_records[0],
                    "step1_checkpoint_record": checkpoint_records[1],
                    "checkpoint_tree_closure": checkpoint_tree_closure,
                },
                "history": history,
                "history_validation": validate_training_history_v1(
                    history,
                    arm_id=args.arm_id,
                    seed=args.seed,
                    expected_steps=1,
                    initial_parameter_sha256=initial_digest,
                    final_parameter_sha256=final_digest,
                    expected_common_payload=common_payload,
                ),
                "all_fresh1_acceptance_gates_pass": True,
                "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
                "external_authority_sha256": EXTERNAL_AUTHORITY_SHA256,
                "model_authority_sha256": MODEL_AUTHORITY_SHA256,
                "latent_bundle_sha256": bundle.bundle_sha256,
                "runner_source_sha256": runner_sha,
                "source_pins": dict(source_pins),
                "claim_boundaries": dict(CLAIM_BOUNDARIES),
                "pre_publish_closure_replays": closure_replays,
            }
            receipt = {
                **unsigned_receipt,
                "receipt_digest": object_sha256(unsigned_receipt),
            }
            _validate_fresh1_receipt_value_v1(
                receipt,
                arm_id=args.arm_id,
                expected_receipt_digest=receipt["receipt_digest"],
                expected_runner_sha256=runner_sha,
                expected_bundle_sha256=bundle.bundle_sha256,
                expected_source_pins=source_pins,
                cross_gate_sha256=cross_gate["gate_sha256"],
                cross_gate_digest=cross_gate["gate_digest"],
                cross_recipe_version_digest=cross_gate["recipe_version_digest"],
                cross_common_initial_trainable_sha256=cross_gate[
                    "common_initial_trainable_sha256"
                ],
                cross_common_row_input_noise_schedule_digest=cross_gate[
                    "common_row_input_noise_schedule_digest"
                ],
                cross_common_comparison_payload_digest=cross_gate[
                    "common_comparison_payload_digest"
                ],
            )
        else:
            unsigned_receipt = {
                "schema_version": RECEIPT_SCHEMA,
                "status": (
                    "EXACT10_LATENT_GATES_PASS_DECODED_REVIEW_PENDING"
                    if latent_gate_pass
                    else "EXACT10_LATENT_GATES_NO_GO"
                ),
                "method": METHOD,
                "arm_id": args.arm_id,
                "branch_recipe": (
                    "target_duplicate_exact2"
                    if args.arm_id == ARM_DUPLICATE
                    else "target_and_role_swap_exact2"
                ),
                "holder_job_id": placement["holder_job_id"],
                "node": placement["node"],
                "seed": args.seed,
                "requested_optimizer_steps": MAX_STEPS,
                "completed_optimizer_steps": MAX_STEPS,
                "optimizer_constructed": True,
                "optimizer_state_empty_before_first_update": True,
                "fresh_official_base": True,
                "resume_consumed": False,
                "fresh1_checkpoint_consumed": False,
                "initial_trainable_sha256": initial_digest,
                "final_trainable_sha256": final_digest,
                "parameters_changed": final_digest != initial_digest,
                "own_preflight_binding": own_binding,
                "cross_arm_gate_binding": cross_binding,
                "fresh1_acceptance_gate_binding": fresh1_gate_binding,
                "common_comparison_payload_digest": common_digest,
                "row_input_noise_schedule_digest": common_payload[
                    "row_input_noise_schedule_digest"
                ],
                "history": history,
                "history_validation": validate_training_history_v1(
                    history,
                    arm_id=args.arm_id,
                    seed=args.seed,
                    expected_steps=MAX_STEPS,
                    initial_parameter_sha256=initial_digest,
                    final_parameter_sha256=final_digest,
                    expected_common_payload=common_payload,
                ),
                "checkpoint_records": checkpoint_records,
                "checkpoint_tree_closure": checkpoint_tree_closure,
                "step0_full_q_route": aggregate["step0_full_q_route"],
                "step0_role_only_cells": aggregate["step0_role_only_cells"],
                "step0_role_only_input_invariants": aggregate[
                    "step0_role_only_input_invariants"
                ],
                "step0_role_only_input_invariants_validation": aggregate[
                    "step0_role_only_input_invariants_validation"
                ],
                "step0_evaluation_forward_evidence": aggregate[
                    "step0_evaluation_forward_evidence"
                ],
                "step0_evaluation_forward_evidence_validation": aggregate[
                    "step0_evaluation_forward_evidence_validation"
                ],
                "step10_evidence": step10_evidence,
                "step10_gate": step10_gate,
                "latent_hard_gates_pass": latent_gate_pass,
                "latent_hard_gate_error": latent_gate_error,
                "decoded_track_effect_gate_pending": True,
                "selection_eligible": False,
                "selection_requires_decoded_track_effect_conjunction": True,
                "primary_metric_if_all_gates_pass": (
                    step10_gate.get("primary_metric_value")
                    if step10_gate is not None
                    else None
                ),
                "weighted_metric_sum_used": False,
                "pre_publish_closure_replays": closure_replays,
                "experiment_contract_sha256": EXPERIMENT_CONTRACT_SHA256,
                "external_authority_sha256": EXTERNAL_AUTHORITY_SHA256,
                "model_authority_sha256": MODEL_AUTHORITY_SHA256,
                "latent_bundle_sha256": bundle.bundle_sha256,
                "runner_source_sha256": runner_sha,
                "source_pins": dict(source_pins),
                "claim_boundaries": dict(CLAIM_BOUNDARIES),
                "formal_c2_authorized": False,
                "exact160_authorized": False,
                "real_video_claim_authorized": False,
                "scientific_claim_authorized": False,
                "source_instruction_inference": False,
                "elapsed_seconds": time.monotonic() - started,
            }
            receipt = {
                **unsigned_receipt,
                "receipt_digest": object_sha256(unsigned_receipt),
            }
            _validate_exact10_receipt_value_v1(
                receipt,
                arm_id=args.arm_id,
                expected_receipt_digest=receipt["receipt_digest"],
                expected_runner_sha256=runner_sha,
                expected_bundle_sha256=bundle.bundle_sha256,
                expected_source_pins=source_pins,
                expected_origin_verifier_binding=expected_origin_verifier_binding,
                expected_gate_controller_binding=expected_gate_controller_binding,
            )
        c1.atomic_create_json(output / "TRAINING_RECEIPT.json", receipt)
        os.chmod(output, 0o555)
    dist.barrier(group=parallel.world_group)
    return 0 if latent_gate_pass else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ELAL3C2TrainingError as error:
        print(f"ELAL3_C2_TRAINING_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
