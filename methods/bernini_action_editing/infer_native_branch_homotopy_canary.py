#!/usr/bin/env python3
"""Frozen exact81 Bernini native-branch homotopy canary.

For one pre-registered dog or human cell this runner compares three matched
trajectories on one official Bernini Gaussian and seed:

``native-full-source-endpoint``
    The pinned ``v2v_apg`` guidance implementation with the full source video,
    four independently VAE-encoded source frames, and the mode-native VR2V
    editing prompt.  This is deliberately a hybrid condition/task contract,
    not stock Bernini ``rv2v`` deployment parity.

``r2v4-reference-only-endpoint``
    True ``r2v_apg``: exactly four source references, no full source video,
    and the mode-native R2V subject-to-video prompt.  Its three native forwards
    are empty/negative, image/negative, and image/action.

``native-branch-homotopy-090-060``
    The full-source endpoint is the host sampler.  A reversible instance hook
    reconstructs the same-step R2V-4 endpoint from the already packed I branch
    and replaces only the velocity passed to the one original UniPC step with
    ``(1-h)*v_full+h*v_r2v4``.  ``h`` is FP32 smoothstep, one at sigma >= .90
    and zero at sigma <= .60.

The positive task prefix and visual regime intentionally co-vary: using the
VR2V prefix for R2V would turn the high endpoint into an out-of-distribution
competence test.  All arms share the same target action-caption body and the
same verbatim negative prompt.  The model is frozen; no target video, mask,
track, pose, flow, custom initial noise, optimizer, or parameter update is
accepted.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402
from infer_native_self_guided_action_field_canary import (  # noqa: E402
    _strong_model_freeze_certificate,
)
from native_branch_homotopy_runtime_v1 import (  # noqa: E402
    NativeBranchHomotopyRuntimeConfig,
    NativeBranchHomotopyRuntimePatch,
)
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "frozen-bernini-native-visual-branch-homotopy-canary"
SCHEMA_VERSION = "bernini-native-branch-homotopy-canary-receipt-v1"
REGISTRY_SCHEMA_VERSION = "bernini-native-branch-homotopy-core4-v1"
FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
FPS = 25
ULYSSES_SIZE = 4
REFERENCE_INDICES = (0, 27, 53, 80)
ARM_ORDER = (
    "native-full-source-endpoint",
    "r2v4-reference-only-endpoint",
    "native-branch-homotopy-090-060",
)
CELL_ORDER = (
    "fit-dog-7b88",
    "fit-human-a35b",
    "confirmation-dog-841b",
    "confirmation-human-a66e",
)
WAVE_ORDER = ("wave1-fit-compatibility", "wave2-heldout-confirmation")
WAVE_CELLS = {
    WAVE_ORDER[0]: CELL_ORDER[:2],
    WAVE_ORDER[1]: CELL_ORDER[2:],
}
COHORT_BY_WAVE = {
    WAVE_ORDER[0]: "fit_identity_compatibility",
    WAVE_ORDER[1]: "heldout_confirmation_action_rescue",
}
HOMOTOPY_ARM = ARM_ORDER[2]
GUIDANCE_BY_ARM = {
    ARM_ORDER[0]: "v2v_apg",
    ARM_ORDER[1]: "r2v_apg",
    ARM_ORDER[2]: "v2v_apg",
}
EXPECTED_FORWARD_COUNT_PER_STEP = {
    ARM_ORDER[0]: 2,
    ARM_ORDER[1]: 3,
    ARM_ORDER[2]: 5,
}
SCHEDULE_SHA256 = "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
HIGH_ENDPOINT_STEP_INDICES = tuple(range(0, 15))
TRANSITION_STEP_INDICES = tuple(range(15, 31))
LOW_ENDPOINT_STEP_INDICES = tuple(range(31, 40))
NATIVE_UNIPC40_TIMESTEPS = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)
NATIVE_UNIPC40_SIGMAS = (
    0.9999989867210388, 0.9949031472206116, 0.9895941615104675,
    0.9840595126152039, 0.978284478187561, 0.9722530841827393,
    0.9659478068351746, 0.9593496322631836, 0.9524376392364502,
    0.9451888799667358, 0.9375780820846558, 0.9295775294303894,
    0.9211564660072327, 0.912280797958374, 0.9029127359390259,
    0.893010139465332, 0.8825258612632751, 0.871407151222229,
    0.8595945835113525, 0.8470211625099182, 0.8336109519004822,
    0.8192774057388306, 0.8039219379425049, 0.7874310612678528,
    0.7696741223335266, 0.7504994869232178, 0.7297303080558777,
    0.7071589827537537, 0.6825404167175293, 0.6555827856063843,
    0.6259360909461975, 0.5931769013404846, 0.55678790807724,
    0.5161304473876953, 0.4704066216945648, 0.41860657930374146,
    0.3594328761100769, 0.2911904454231262, 0.21162153780460358,
    0.11765105277299881,
)
R2V4_BINDING_CLAUSE = (
    "The same main subject is shown across image0, image1, image2, and image3. "
    "Preserve that subject's identity across those four source views. "
)
CONFIRMATION_SEED_REGISTRY_RELATIVE = (
    "assets/wrong_family_prompt_swap_pilot_registry_v1.json"
)
CONFIRMATION_SEED_REGISTRY_SHA256 = (
    "f8978311ee7db7f524e827b49747f74ec1a5e0d568e2bbada3fd212225f20cff"
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class NativeBranchHomotopyCanaryError(RuntimeError):
    """Raised before incomplete or ambiguous evidence is published."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _object_sha256(value: Any) -> str:
    return _sha256_bytes(native.legacy.canonical_json_bytes(value))


def _plain_json(
    value: str | Path, *, label: str
) -> tuple[Path, Mapping[str, Any]]:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise NativeBranchHomotopyCanaryError(f"{label} path differs")
    path = requested.resolve(strict=True)
    if path != requested or not path.is_file() or path.is_symlink():
        raise NativeBranchHomotopyCanaryError(f"{label} must be a plain absolute file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise NativeBranchHomotopyCanaryError(f"{label} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise NativeBranchHomotopyCanaryError(f"{label} root differs")
    return path, payload


def _registry_cell(
    registry: Mapping[str, Any], *, cell_id: str
) -> Mapping[str, Any]:
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("method") != METHOD
        or registry.get("arm_order") != list(ARM_ORDER)
    ):
        raise NativeBranchHomotopyCanaryError("registry root differs")
    contract = registry.get("contract")
    if not isinstance(contract, Mapping):
        raise NativeBranchHomotopyCanaryError("registry contract differs")
    low = contract.get("native_full_source_endpoint")
    high = contract.get("r2v4_reference_only_endpoint")
    homotopy = contract.get("homotopy")
    prompts = contract.get("prompt_homotopy_disclosure")
    schedule = contract.get("apg_and_scheduler")
    embeddings = contract.get("embedding_contract")
    if (
        not isinstance(low, Mapping)
        or low.get("guidance_mode") != "v2v_apg"
        or low.get("full_source_video_count") != 1
        or low.get("independently_vae_encoded_source_reference_count") != 4
        or low.get("visual_condition_fixed_in_negative_and_action_forwards") is not True
        or low.get("transformer_forwards_per_step") != 2
        or low.get("stock_vr2v_deployment_parity") is not False
        or not isinstance(high, Mapping)
        or high.get("guidance_mode") != "r2v_apg"
        or high.get("full_source_video_count") != 0
        or high.get("independently_vae_encoded_source_reference_count") != 4
        or high.get("forward_order") != ["none_negative", "I_negative", "I_action"]
        or high.get("transformer_forwards_per_step") != 3
        or not isinstance(homotopy, Mapping)
        or homotopy.get("high_sigma") != 0.9
        or homotopy.get("low_sigma") != 0.6
        or homotopy.get("fp32_interpolation_before_one_official_scheduler_step") is not True
        or homotopy.get("hard_switch") is not False
        or not isinstance(prompts, Mapping)
        or prompts.get("same_target_action_caption_body") is not True
        or prompts.get("same_renderer_negative_embedding") is not True
        or prompts.get("task_prefix_and_visual_regime_change_together") is not True
        or prompts.get("shared_vr2v_positive_embedding_across_endpoints") is not False
        or not isinstance(schedule, Mapping)
        or schedule.get("flow_shift_from_renderer_config") != 5.0
        or schedule.get("omega_image") != 4.5
        or schedule.get("omega_text") != 4.0
        or schedule.get("eta") != 0.5
        or schedule.get("norm_thresholds") != [50.0, 50.0]
        or schedule.get("momentum") != 0.0
        or schedule.get("unipc_steps") != NUM_INFERENCE_STEPS
        or schedule.get("exact40_shift5_schedule_sha256") != SCHEDULE_SHA256
        or not isinstance(embeddings, Mapping)
        or embeddings.get("positive_shape") != [1, 512, 4096]
        or embeddings.get("negative_shape") != [1, 512, 4096]
        or contract.get("source_reference_indices") != list(REFERENCE_INDICES)
        or contract.get("references_independently_vae_encoded_from_rgb") is not True
        or contract.get("references_sliced_from_full_video_latent") is not False
        or contract.get("frame_count") != FRAME_COUNT
        or contract.get("latent_phases") != LATENT_PHASES
        or contract.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or contract.get("frozen_model") is not True
        or contract.get("training") is not False
        or contract.get("parameter_update") is not False
    ):
        raise NativeBranchHomotopyCanaryError("registry scientific contract differs")
    regions = schedule.get("homotopy_regions")
    if (
        not isinstance(regions, Mapping)
        or regions.get("high_r2v4_weight_one_step_indices")
        != list(HIGH_ENDPOINT_STEP_INDICES)
        or regions.get("strict_transition_step_indices")
        != list(TRANSITION_STEP_INDICES)
        or regions.get("low_v2v_weight_one_step_indices")
        != list(LOW_ENDPOINT_STEP_INDICES)
    ):
        raise NativeBranchHomotopyCanaryError("registry exact40 regions differ")
    population = registry.get("population_design")
    expected_waves = [
        {
            "wave_id": wave,
            "cohort": COHORT_BY_WAVE[wave],
            "cell_ids": list(WAVE_CELLS[wave]),
        }
        for wave in WAVE_ORDER
    ]
    if (
        not isinstance(population, Mapping)
        or population.get("wave_order") != list(WAVE_ORDER)
        or population.get("waves") != expected_waves
        or population.get("fit_and_confirmation_never_aggregated") is not True
        or population.get("single_example_conclusion_authorized") is not False
    ):
        raise NativeBranchHomotopyCanaryError("registry population design differs")
    seed_audit = population.get("seed_collision_audit")
    if (
        not isinstance(seed_audit, Mapping)
        or seed_audit.get("all_four_seeds_unique") is not True
        or seed_audit.get("collision_count") != 0
    ):
        raise NativeBranchHomotopyCanaryError("registry seed collision audit differs")
    seed_provenance = population.get("confirmation_seed_provenance_contract")
    if (
        not isinstance(seed_provenance, Mapping)
        or seed_provenance.get("status") != "unrendered_preregistered_seed"
        or seed_provenance.get("source_registry_repo_relative_path")
        != "methods/bernini_action_editing/assets/wrong_family_prompt_swap_pilot_registry_v1.json"
        or seed_provenance.get("source_registry_sha256")
        != CONFIRMATION_SEED_REGISTRY_SHA256
        or seed_provenance.get("source_registry_field") != "fresh_seed"
        or seed_provenance.get("prior_auh_job_id") != 131492
        or seed_provenance.get("prior_job_state") != "FAILED"
        or seed_provenance.get("prior_elapsed_seconds") != 9
        or seed_provenance.get("prior_artifact_mp4_count") != 0
        or seed_provenance.get("prior_artifact_latent_count") != 0
        or seed_provenance.get("prior_candidate_rendered") is not False
        or seed_provenance.get("generic_fresh_seed_claim") is not False
    ):
        raise NativeBranchHomotopyCanaryError(
            "confirmation unrendered seed provenance differs"
        )
    cells = registry.get("cells")
    if (
        not isinstance(cells, list)
        or [row.get("cell_id") for row in cells if isinstance(row, Mapping)]
        != list(CELL_ORDER)
    ):
        raise NativeBranchHomotopyCanaryError("registry cells differ")
    observed_seeds = [row.get("seed") for row in cells]
    if (
        seed_audit.get("seeds_in_cell_order") != observed_seeds
        or len(observed_seeds) != 4
        or len(set(observed_seeds)) != 4
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in observed_seeds)
    ):
        raise NativeBranchHomotopyCanaryError("registry live seed collision audit differs")
    matches = [
        row
        for row in cells
        if isinstance(row, Mapping) and row.get("cell_id") == cell_id
    ]
    if len(matches) != 1:
        raise NativeBranchHomotopyCanaryError("registry cell lookup differs")
    cell = matches[0]
    required = {
        "cell_id",
        "wave_id",
        "cohort",
        "actor_kind",
        "source_iid",
        "source_video",
        "source_video_sha256",
        "target_action_caption",
        "target_action_caption_sha256",
        "seed",
        "bucket_hw",
        "latent_shape",
        "selected_before_generation",
    }
    if cell.get("cohort") == "heldout_confirmation_action_rescue":
        required.add("seed_collision_evidence")
    if set(cell) != required:
        raise NativeBranchHomotopyCanaryError("registry cell schema differs")
    caption = cell.get("target_action_caption")
    if (
        not isinstance(caption, str)
        or not caption.strip()
        or _sha256_text(caption) != cell.get("target_action_caption_sha256")
        or _SHA256.fullmatch(str(cell.get("source_video_sha256"))) is None
        or type(cell.get("seed")) is not int
        or not 0 <= int(cell["seed"]) < 2**63
        or cell.get("selected_before_generation") is not True
        or cell.get("wave_id") not in WAVE_ORDER
        or cell.get("cohort") != COHORT_BY_WAVE.get(str(cell.get("wave_id")))
        or cell_id not in WAVE_CELLS.get(str(cell.get("wave_id")), ())
        or cell.get("actor_kind") not in ("dog", "human")
        or list(cell.get("latent_shape", ()))[:3] != [1, 16, LATENT_PHASES]
        or len(cell.get("bucket_hw", ())) != 2
    ):
        raise NativeBranchHomotopyCanaryError("registry cell content differs")
    if cell.get("cohort") == "heldout_confirmation_action_rescue":
        evidence = cell.get("seed_collision_evidence")
        expected_evidence = {
            key: value
            for key, value in seed_provenance.items()
            if key != "generic_fresh_seed_claim"
        }
        expected_evidence["seed_reuse_collision_with_rendered_media"] = False
        if evidence != expected_evidence:
            raise NativeBranchHomotopyCanaryError(
                "confirmation cell seed collision evidence differs"
            )
        seed_registry_path = METHOD_ROOT / CONFIRMATION_SEED_REGISTRY_RELATIVE
        if (
            not seed_registry_path.is_file()
            or seed_registry_path.is_symlink()
            or native.legacy.file_sha256(seed_registry_path)
            != CONFIRMATION_SEED_REGISTRY_SHA256
        ):
            raise NativeBranchHomotopyCanaryError(
                "confirmation seed authority file differs"
            )
        try:
            seed_registry = json.loads(seed_registry_path.read_text(encoding="utf-8"))
            source_cells = seed_registry["cells"]
        except Exception as error:
            raise NativeBranchHomotopyCanaryError(
                "confirmation seed authority JSON differs"
            ) from error
        matches = [
            row
            for row in source_cells
            if isinstance(row, Mapping) and row.get("iid") == cell.get("source_iid")
        ]
        if (
            len(matches) != 1
            or matches[0].get("fresh_seed") != cell.get("seed")
            or matches[0].get("fresh_media_status")
            != "unrendered_and_unseen_at_registry_seal"
        ):
            raise NativeBranchHomotopyCanaryError(
                "confirmation preregistered seed binding differs"
            )
    return cell


def build_mode_native_prompt(
    mode: str, caption: str, *, prompt_cleaner: Any
) -> str:
    """Build the two trained-task prompts without hiding their co-change."""

    if mode == "low-vr2v":
        system = native.TASK_SYSTEM_PROMPTS["vr2v"]
        body = native.TASK_BINDING_CLAUSES["rv2v"] + caption
    elif mode == "high-r2v4":
        system = native.TASK_SYSTEM_PROMPTS["r2v"]
        body = R2V4_BINDING_CLAUSE + caption
    else:
        raise NativeBranchHomotopyCanaryError("prompt mode differs")
    cleaned = prompt_cleaner(body)
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise NativeBranchHomotopyCanaryError("prompt cleaner returned empty text")
    return system + cleaned


def sampling_contract(arm: str, *, seed: int) -> Mapping[str, Any]:
    if arm not in ARM_ORDER or type(seed) is not int or not 0 <= seed < 2**63:
        raise NativeBranchHomotopyCanaryError("sampling arm/seed differs")
    native_arm = "r2v" if arm == ARM_ORDER[1] else "rv2v"
    value = native.native_sampling_contract(
        native_arm, steps=NUM_INFERENCE_STEPS, seed=seed
    )
    value["guidance_mode"] = GUIDANCE_BY_ARM[arm]
    value["omega_img"] = 4.5
    value["omega_txt"] = 4.0
    value["flow_shift"] = 5.0
    value["eta"] = 0.5
    value["norm_threshold"] = (50.0, 50.0)
    value["momentum"] = 0.0
    if (
        value.get("num_frames") != FRAME_COUNT
        or value.get("num_inference_steps") != NUM_INFERENCE_STEPS
    ):
        raise NativeBranchHomotopyCanaryError("native sampling contract differs")
    return value


def pinned_exact40_schedule_receipt() -> Mapping[str, Any]:
    value = {
        "scheduler": "UniPCMultistepScheduler",
        "prediction_type": "flow_prediction",
        "use_flow_sigmas": True,
        "flow_shift": 5.0,
        "num_inference_steps": 40,
        "model_forward_count": 40,
        "terminal_sigma_excluded_from_training": 0.0,
        "timesteps": list(NATIVE_UNIPC40_TIMESTEPS),
        "sigma_float64_hex": [float(item).hex() for item in NATIVE_UNIPC40_SIGMAS],
        "sampling_distribution": "uniform_without_replacement_over_exact40_per_cycle",
    }
    result = {**value, "digest": _object_sha256(value)}
    if result["digest"] != SCHEDULE_SHA256:
        raise NativeBranchHomotopyCanaryError("pinned exact40 schedule digest differs")
    return result


def validate_homotopy_runtime_trace(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Bind the live hook trace to the pre-extracted exact40 shift-5 schedule."""

    schedule = pinned_exact40_schedule_receipt()
    rows = trace.get("trace") if isinstance(trace, Mapping) else None
    if (
        schedule.get("digest") != SCHEDULE_SHA256
        or trace.get("steps") != NUM_INFERENCE_STEPS
        or trace.get("transformer_forwards") != 5 * NUM_INFERENCE_STEPS
        or trace.get("low_vi_forwards") != 2 * NUM_INFERENCE_STEPS
        or trace.get("high_r2v4_forwards") != 3 * NUM_INFERENCE_STEPS
        or trace.get("patch_vae_latent_calls") != 10 * NUM_INFERENCE_STEPS
        or trace.get("original_scheduler_calls") != NUM_INFERENCE_STEPS
        or trace.get("low_official_apg_exact_parity_all_steps") is not True
        or trace.get("smoothstep_sigma_low") != 0.6
        or trace.get("smoothstep_sigma_high") != 0.9
        or not isinstance(rows, list)
        or len(rows) != NUM_INFERENCE_STEPS
    ):
        raise NativeBranchHomotopyCanaryError("homotopy runtime trace root differs")
    timesteps = schedule["timesteps"]
    sigmas = NATIVE_UNIPC40_SIGMAS
    for index, (row, expected_timestep, expected_sigma) in enumerate(
        zip(rows, timesteps, sigmas)
    ):
        expected_endpoint = (
            "high_r2v4_apg"
            if index in HIGH_ENDPOINT_STEP_INDICES
            else (
                "transition"
                if index in TRANSITION_STEP_INDICES
                else "low_official_v2v_apg"
            )
        )
        high_weight = row.get("high_r2v4_weight") if isinstance(row, Mapping) else None
        weight_ok = (
            high_weight == 1.0
            if index in HIGH_ENDPOINT_STEP_INDICES
            else (
                isinstance(high_weight, float) and 0.0 < high_weight < 1.0
                if index in TRANSITION_STEP_INDICES
                else high_weight == 0.0
            )
        )
        if (
            not isinstance(row, Mapping)
            or row.get("step_index") != index
            or row.get("timestep") != float(expected_timestep)
            or row.get("sigma") != expected_sigma
            or row.get("endpoint") != expected_endpoint
            or not weight_ok
            or row.get("transformer_forwards") != 5
            or row.get("low_vi_forwards") != 2
            or row.get("high_r2v4_forwards") != 3
            or row.get("original_scheduler_calls") != 1
            or row.get("patch_call_count") != 10
            or row.get("low_official_apg_exact_parity") is not True
            or row.get("freeze_safe_no_grad_outputs") is not True
        ):
            raise NativeBranchHomotopyCanaryError(
                f"homotopy exact40 step {index} differs"
            )
    return schedule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--cell-id", choices=CELL_ORDER, required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--runtime-source-closure-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=native.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=native.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_registry_sha256",
        "runtime_source_archive_sha256",
        "runtime_source_closure_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise NativeBranchHomotopyCanaryError(f"{name} differs")
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise NativeBranchHomotopyCanaryError(f"{name} differs")
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise NativeBranchHomotopyCanaryError("Bernini revision differs")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise NativeBranchHomotopyCanaryError("VeOmni revision differs")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise NativeBranchHomotopyCanaryError("checkpoint tree differs")


def _conditions_for_arm(
    arm: str, *, source_latent: Any, references: Mapping[int, Any]
) -> Mapping[str, Any]:
    if arm not in ARM_ORDER or tuple(references) != REFERENCE_INDICES:
        raise NativeBranchHomotopyCanaryError("arm/reference condition differs")
    return {
        "image_vae_latents": None,
        "multi_video_vae_latents": (
            None if arm == ARM_ORDER[1] else [source_latent]
        ),
        "multi_image_vae_latents": [references[index] for index in REFERENCE_INDICES],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    registry_path, registry = _plain_json(args.registry, label="registry")
    if native.legacy.file_sha256(registry_path) != args.expected_registry_sha256:
        raise NativeBranchHomotopyCanaryError("registry file digest differs")
    cell = _registry_cell(registry, cell_id=args.cell_id)
    output_dir = native._resolve_fresh_output_dir(args.output_dir)
    source_requested = Path(str(cell["source_video"])).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise NativeBranchHomotopyCanaryError("source path differs")
    source_path = source_requested.resolve(strict=True)
    if (
        source_path != source_requested
        or not source_path.is_file()
        or native.legacy.file_sha256(source_path) != cell["source_video_sha256"]
    ):
        raise NativeBranchHomotopyCanaryError("source bytes differ")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise NativeBranchHomotopyCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise NativeBranchHomotopyCanaryError("attention heads do not divide Ulysses4")
    inference_file_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("r2v") != native.TASK_SYSTEM_PROMPTS["r2v"]:
        raise NativeBranchHomotopyCanaryError("runtime R2V system prompt differs")
    if SYSTEM_PROMPTS.get("vr2v") != native.TASK_SYSTEM_PROMPTS["vr2v"]:
        raise NativeBranchHomotopyCanaryError("runtime VR2V system prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise NativeBranchHomotopyCanaryError("runtime negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != ULYSSES_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise NativeBranchHomotopyCanaryError("runtime requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if (
        not isinstance(checkpoint_rows[0], Mapping)
        or checkpoint_rows[0].get("ok") is not True
    ):
        raise NativeBranchHomotopyCanaryError(
            f"checkpoint validation failed: {checkpoint_rows[0]}"
        )
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = source_audit.prepare_hashed_source_snapshot(
        source_path
    )
    bucket_hw = tuple(int(item) for item in cell["bucket_hw"])
    latent_shape = tuple(int(item) for item in cell["latent_shape"])
    if (
        source_sha != cell["source_video_sha256"]
        or source_metadata.get("frame_count") != FRAME_COUNT
        or tuple(source_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
    ):
        raise NativeBranchHomotopyCanaryError("source exact81 geometry differs")

    caption = str(cell["target_action_caption"])
    low_prompt = build_mode_native_prompt(
        "low-vr2v", caption, prompt_cleaner=prompt_clean
    )
    high_prompt = build_mode_native_prompt(
        "high-r2v4", caption, prompt_cleaner=prompt_clean
    )
    if low_prompt == high_prompt:
        raise NativeBranchHomotopyCanaryError("mode-native prompts unexpectedly alias")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    low_ids, low_mask = native.legacy._tokenize_training_prompt(tokenizer, low_prompt)
    high_ids, high_mask = native.legacy._tokenize_training_prompt(tokenizer, high_prompt)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise NativeBranchHomotopyCanaryError("renderer exact40 shift5 UniPC differs")
    model = BerniniRendererModel(config)
    model.eval().requires_grad_(False)
    freeze_before = _strong_model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    reference_shape = (1, 16, 1, latent_shape[3], latent_shape[4])
    if distributed.rank == 0:
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            source_latent = _vae_encode(vae, source_pixels).contiguous()
            references = {
                index: _vae_encode(
                    vae, source_pixels[:, :, index : index + 1].contiguous()
                ).contiguous()
                for index in REFERENCE_INDICES
            }
        del source_pixels
    else:
        source_latent = torch.empty(latent_shape, device=device, dtype=torch.float32)
        references = {
            index: torch.empty(reference_shape, device=device, dtype=torch.float32)
            for index in REFERENCE_INDICES
        }
    dist.broadcast(source_latent, src=0)
    for index in REFERENCE_INDICES:
        dist.broadcast(references[index], src=0)
    if tuple(source_latent.shape) != latent_shape or any(
        tuple(value.shape) != reference_shape for value in references.values()
    ):
        raise NativeBranchHomotopyCanaryError("source condition geometry differs")
    condition_identities = {
        "source_video": native._all_rank_tensor_identity(
            source_latent,
            label=f"{args.cell_id}_source_video",
            world_size=ULYSSES_SIZE,
        ),
        "references": {
            str(index): native._all_rank_tensor_identity(
                value,
                label=f"{args.cell_id}_source_reference_{index}",
                world_size=ULYSSES_SIZE,
            )
            for index, value in references.items()
        },
        "references_independently_vae_encoded_from_rgb": True,
        "references_sliced_from_full_video_latent": False,
    }
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.to(device)
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        low_embeds = model.encode_prompt(low_ids.to(device), low_mask.to(device)).detach()
        high_embeds = model.encode_prompt(high_ids.to(device), high_mask.to(device)).detach()
        uncond_embeds = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    expected_embed_shape = (1, 512, 4096)
    if any(
        tuple(value.shape) != expected_embed_shape
        for value in (low_embeds, high_embeds, uncond_embeds)
    ):
        raise NativeBranchHomotopyCanaryError("mode-native embedding geometry differs")
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    wan_source_sha = sampler_contract.validate_runtime_source_identity(
        bernini_commit=bernini_revision,
        wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
    )
    sampler_contract._validate_scheduler_contract(
        diffusion.scheduler, expected_flow_shift=5.0
    )
    if diffusion.transformer_2 is not None:
        raise NativeBranchHomotopyCanaryError("canary requires Bernini-R 1.3B single DiT")
    target_patch_tokens = LATENT_PHASES * (bucket_hw[0] // 16) * (bucket_hw[1] // 16)
    reference_patch_tokens = target_patch_tokens // LATENT_PHASES
    expected_condition_prefix_tokens = target_patch_tokens + 4 * reference_patch_tokens

    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    initial_noise_identities: dict[str, Any] = {}
    runtime_traces: dict[str, Any] = {}
    live_schedule_receipt: Optional[Mapping[str, Any]] = None
    with torch.inference_mode():
        for arm in ARM_ORDER:
            patch = None
            if arm == HOMOTOPY_ARM:
                patch = NativeBranchHomotopyRuntimePatch(
                    diffusion,
                    r2v_action_prompt_embeds=high_embeds,
                    config=NativeBranchHomotopyRuntimeConfig(
                        target_latent_shape=latent_shape,
                        expected_steps=NUM_INFERENCE_STEPS,
                        expected_flow_shift=5.0,
                        omega_image=4.5,
                        omega_text=4.0,
                        eta=0.5,
                        image_norm_threshold=50.0,
                        text_norm_threshold=50.0,
                        momentum=0.0,
                    ),
                )
                patch.install()
            prompt_embeds = high_embeds if arm == ARM_ORDER[1] else low_embeds
            sample_kwargs = {
                "prompt_embeds": prompt_embeds,
                "uncond_prompt_embeds": uncond_embeds,
                **_conditions_for_arm(
                    arm, source_latent=source_latent, references=references
                ),
                "width": bucket_hw[1],
                "height": bucket_hw[0],
                "device": device,
                **sampling_contract(arm, seed=int(cell["seed"])),
            }
            try:
                result, capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda kwargs=sample_kwargs: diffusion.sample(**kwargs),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=latent_shape,
                    expected_device=device,
                    expected_seed=int(cell["seed"]),
                )
            finally:
                if patch is not None:
                    patch.restore()
            if patch is not None:
                runtime_traces[arm] = dict(patch.finalize())
                live_schedule_receipt = validate_homotopy_runtime_trace(
                    runtime_traces[arm]
                )
            else:
                runtime_traces[arm] = {
                    "native_endpoint": True,
                    "guidance_mode": GUIDANCE_BY_ARM[arm],
                    "expected_transformer_forwards_per_step_from_pinned_vendor": (
                        EXPECTED_FORWARD_COUNT_PER_STEP[arm]
                    ),
                    "runtime_hook_installed": False,
                    "vendor_source_sha256": wan_source_sha,
                }
            if (
                not isinstance(result, torch.Tensor)
                or tuple(result.shape) != latent_shape
                or result.dtype != torch.float32
                or result.requires_grad
                or result.grad_fn is not None
                or not bool(torch.isfinite(result).all().item())
            ):
                raise NativeBranchHomotopyCanaryError("native sampler result differs")
            stored = result.detach().to(device="cpu").contiguous()
            generated[arm] = stored
            generated_identities[arm] = native._all_rank_tensor_identity(
                stored, label=f"{args.cell_id}_{arm}", world_size=ULYSSES_SIZE
            )
            initial_noise[arm] = capture
            initial_noise_identities[arm] = native._all_rank_tensor_identity(
                capture.tensor,
                label=f"{args.cell_id}_{arm}_official_initial_gaussian",
                world_size=ULYSSES_SIZE,
            )

    noise_hashes = {capture.raw_value_sha256 for capture in initial_noise.values()}
    if len(noise_hashes) != 1 or live_schedule_receipt is None:
        raise NativeBranchHomotopyCanaryError("arms did not share one official Gaussian")
    trace_digest = _object_sha256(runtime_traces)
    trace_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(trace_rows, trace_digest)
    if len(set(trace_rows)) != 1:
        raise NativeBranchHomotopyCanaryError("branch traces differ across SP4 ranks")
    freeze_after = _strong_model_freeze_certificate(model)
    if freeze_after != freeze_before or any(p.requires_grad for p in model.parameters()):
        raise NativeBranchHomotopyCanaryError("frozen model changed")
    model.to("cpu")
    torch.cuda.empty_cache()

    after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            after_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            after_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(after_rows, src=0)
    if (
        not isinstance(after_rows[0], Mapping)
        or after_rows[0].get("identity") != checkpoint_identity
    ):
        raise NativeBranchHomotopyCanaryError("checkpoint content changed")

    if distributed.rank == 0:
        output_dir.mkdir(parents=False, exist_ok=False)
        noise_artifacts = {
            arm: native._save_initial_noise_atomically(
                output_dir / f"{arm}.official-initial-gaussian.safetensors",
                initial_noise[arm],
                all_rank_identity=initial_noise_identities[arm],
            )
            for arm in ARM_ORDER
        }
        generated_for_decode = {
            arm: value.to(device=device).contiguous() for arm, value in generated.items()
        }
        try:
            outputs = native._save_outputs(
                output_dir=output_dir,
                generated=generated_for_decode,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
        finally:
            generated_for_decode.clear()
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "cell_id": args.cell_id,
            "wave_id": cell["wave_id"],
            "cohort": cell["cohort"],
            "actor_kind": cell["actor_kind"],
            "input": {
                "source_video": str(source_path),
                "source_video_sha256": source_sha,
                "target_action_caption": caption,
                "target_action_caption_sha256": _sha256_text(caption),
                "seed_collision_evidence": cell.get("seed_collision_evidence"),
                "target_video": False,
                "custom_initial_noise": False,
                "generated_owner_media": False,
                "mask_track_pose_flow": False,
            },
            "prompts": {
                "same_target_action_caption_body": True,
                "same_verbatim_negative_prompt_all_arms": True,
                "low_vr2v_full_prompt_sha256": _sha256_text(low_prompt),
                "high_r2v4_full_prompt_sha256": _sha256_text(high_prompt),
                "positive_task_prefix_and_visual_regime_change_together": True,
                "shared_vr2v_positive_embedding_across_endpoints": False,
                "embedding_shape": list(expected_embed_shape),
                "low_is_official_v2v_apg_guidance_but_not_stock_vr2v_deployment_parity": True,
            },
            "sampling": {
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "fps": FPS,
                "seed": int(cell["seed"]),
                "arm_order": list(ARM_ORDER),
                "guidance_by_arm": dict(GUIDANCE_BY_ARM),
                "same_official_gaussian_all_arms": True,
                "official_gaussian_raw_sha256": next(iter(noise_hashes)),
                "source_reference_indices": list(REFERENCE_INDICES),
                "target_patch_tokens": target_patch_tokens,
                "reference_patch_tokens": reference_patch_tokens,
                "low_condition_prefix_tokens": expected_condition_prefix_tokens,
                "flow_shift_from_renderer_config": float(config.shift),
                "omega_img": 4.5,
                "omega_txt": 4.0,
                "eta": 0.5,
                "norm_thresholds": [50.0, 50.0],
                "momentum": 0.0,
                "schedule_sha256": SCHEDULE_SHA256,
                "schedule_receipt": live_schedule_receipt,
                "high_endpoint_step_indices": list(HIGH_ENDPOINT_STEP_INDICES),
                "transition_step_indices": list(TRANSITION_STEP_INDICES),
                "low_endpoint_step_indices": list(LOW_ENDPOINT_STEP_INDICES),
                "native_target_initialization": native.TARGET_INITIALIZATION,
            },
            "condition_identities": condition_identities,
            "runtime_traces": runtime_traces,
            "runtime_trace_digest": trace_digest,
            "generated_identities": generated_identities,
            "initial_noise_artifacts": noise_artifacts,
            "outputs": outputs,
            "checkpoint": checkpoint_identity,
            "freeze_certificate": freeze_after,
            "source_revisions": {
                "bernini": bernini_revision,
                "veomni": veomni_revision,
                "wan_diffusion_sha256": wan_source_sha,
                "runtime_method": args.runtime_source_revision,
                "runtime_source_archive_sha256": args.runtime_source_archive_sha256,
                "runtime_source_closure_sha256": args.runtime_source_closure_sha256,
                "launcher_source_sha256": args.launcher_source_sha256,
                "inference_files": inference_file_hashes,
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "diffusers": diffusers_version,
                "transformers": transformers_version,
            },
            "training_performed": False,
            "optimizer_created": False,
            "parameter_update": False,
            "scientific_or_action_editing_claim_authorized": False,
            "fit_and_confirmation_never_aggregated": True,
            "single_example_conclusion_authorized": False,
        }
        receipt["receipt_digest"] = _object_sha256(receipt)
        value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(native.legacy.canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del source_latent, references, generated, initial_noise
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "CELL_ORDER",
    "COHORT_BY_WAVE",
    "GUIDANCE_BY_ARM",
    "HIGH_ENDPOINT_STEP_INDICES",
    "HOMOTOPY_ARM",
    "LOW_ENDPOINT_STEP_INDICES",
    "METHOD",
    "NativeBranchHomotopyCanaryError",
    "R2V4_BINDING_CLAUSE",
    "SCHEDULE_SHA256",
    "SCHEMA_VERSION",
    "TRANSITION_STEP_INDICES",
    "WAVE_CELLS",
    "WAVE_ORDER",
    "_registry_cell",
    "build_mode_native_prompt",
    "build_parser",
    "sampling_contract",
]
