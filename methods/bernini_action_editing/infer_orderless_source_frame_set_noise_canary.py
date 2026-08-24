#!/usr/bin/env python3
"""Decode the isolated exact81 orderless source-frame-set noise canary.

This runner reuses Bernini's pinned native identity-generation loader,
preprocessing, prompt, RV2V-4 conditioning, sampler, decoder, and provenance
helpers.  It changes only the tensor returned at the one module-global
``bernini.models.wan_diffusion.randn_tensor`` lookup.  The original callable
is always invoked first with unchanged arguments.

For one pre-registered dog or human cell, five exact40/exact81 rollouts share
the same source condition, complete action caption, seed, official Gaussian,
UniPC scheduler, guidance, and four source RGB references:

* ``official_gaussian``;
* ``correct_source_rho005`` and ``wrong_source_rho005``; and
* ``correct_source_rho010`` and ``wrong_source_rho010``.

The wrong-source carrier comes from an upstream-authoring-sealed,
distinct-identity, same-broad-actor-class and same-action-family confirmation
source.  This is an action-family-matched source-specificity control, not a
pure identity control: the human pair is kneel-versus-crouch and has a sealed
native-aspect mismatch.  It is resized directly from raw RGB into the current
target bucket, then encoded by four independent ``T=1`` VAE calls at frames
``[0,27,53,80]``.  These indices select set members, so pose occupancy can
leak.  The operator receives neither indices nor order and is permutation
invariant.  The wrong source is never a renderer condition.

This is a frozen candidate bank, not training, reward fitting, scoring, or arm
selection.  No target, mask, flow, pose, track, trajectory, donor, trainer, or
critic is accepted.  Active endpoints are explicitly non-Gaussian.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import orderless_source_frame_set_noise as source_set_noise  # noqa: E402


SCHEMA_VERSION = "bernini-orderless-source-frame-set-noise-canary-receipt-v1"
SPEC_SCHEMA = "bernini-orderless-source-frame-set-noise-core2-spec-v1"
AUTHORING_SCHEMA = "pair-v5-pure-t2v-calibration-authoring-v1"
AUTHORING_BANK_ID = "pair5-t2v-first8-v1"
AUTHORING_SPEC_SHA256 = (
    "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
)
METHOD = "frozen-bernini-orderless-source-frame-set-noise-canary"
FRAME_COUNT = 81
FPS = 25
LATENT_PHASES = 21
REFERENCE_INDICES = (0, 27, 53, 80)
NUM_INFERENCE_STEPS = 40
WORLD_SIZE = 4
SP_SIZE = 4
ARM_ORDER = (
    "official_gaussian",
    "correct_source_rho005",
    "wrong_source_rho005",
    "correct_source_rho010",
    "wrong_source_rho010",
)
ARM_RHO = {
    "official_gaussian": 0.0,
    "correct_source_rho005": 0.05,
    "wrong_source_rho005": 0.05,
    "correct_source_rho010": 0.10,
    "wrong_source_rho010": 0.10,
}
ARM_CARRIER_ROLE = {
    "official_gaussian": "correct",
    "correct_source_rho005": "correct",
    "wrong_source_rho005": "wrong",
    "correct_source_rho010": "correct",
    "wrong_source_rho010": "wrong",
}
TEMPORARY_MUTATION_SURFACE = ("bernini.models.wan_diffusion.randn_tensor",)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[0-9a-f]{16}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SourceSetNoiseCanaryError(RuntimeError):
    """Raised before ambiguous or unsealed canary evidence is published."""


@dataclass(frozen=True)
class SourceSetNoiseCapture:
    arm: str
    rho: float
    carrier_source_role: str
    official_gaussian: Any
    sampler_initial_noise: Any
    source_set_prototype: Optional[Any]
    temporal_dc_carrier: Optional[Any]
    official_gaussian_raw_value_sha256: str
    sampler_initial_noise_raw_value_sha256: str
    source_frame_multiset_sha256: str
    source_set_prototype_sha256: Optional[str]
    temporal_dc_carrier_sha256: Optional[str]
    source_set_prototype_raw_value_sha256: Optional[str]
    temporal_dc_carrier_raw_value_sha256: Optional[str]
    operator_receipt: Mapping[str, Any]
    operator_receipt_sha256: str
    requested_shape: tuple[int, ...]
    requested_dtype: str
    requested_device: str
    generator_device: str
    generator_initial_seed: int
    original_randn_call_count: int
    external_initial_noise_injection: bool
    original_return_object_forwarded: bool


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SourceSetNoiseCanaryError(
            f"value is not finite canonical ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _output_staging_directory(final: Path) -> Path:
    stage = Path(tempfile.mkdtemp(dir=final.parent, prefix=f".{final.name}.partial-"))
    stage.chmod(0o755)
    if stage.parent != final.parent or stage.is_symlink():
        raise SourceSetNoiseCanaryError("output staging directory escaped parent")
    return stage


def _rebase_artifact_paths(value: Any, *, old_root: Path, new_root: Path) -> Any:
    old = str(old_root)
    new = str(new_root)
    if isinstance(value, str):
        if value == old:
            return new
        if value.startswith(old + os.sep):
            return new + value[len(old) :]
        return value
    if isinstance(value, Mapping):
        return {
            key: _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for item in value
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SourceSetNoiseCanaryError("refusing to overwrite receipt")
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _commit_output_transaction(*, staging: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise SourceSetNoiseCanaryError("refusing to replace output directory")
    if staging.parent != final.parent or not staging.is_dir() or staging.is_symlink():
        raise SourceSetNoiseCanaryError("output staging directory differs")
    _fsync_directory(staging)
    os.replace(staging, final)
    _fsync_directory(final.parent)


def file_sha256(path: Path) -> str:
    return native.legacy.file_sha256(path)


def _sha(value: Any, *, length: int, label: str) -> str:
    text = str(value)
    pattern = _SHA1 if length == 40 else _SHA256
    if pattern.fullmatch(text) is None:
        raise SourceSetNoiseCanaryError(f"{label} must be lowercase SHA-{length}")
    return text


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise SourceSetNoiseCanaryError(f"{label} must be an absolute plain file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not resolved.is_file() or resolved.is_symlink():
        raise SourceSetNoiseCanaryError(f"{label} must be a canonical plain file")
    return resolved


def _expected_spec_contract() -> Mapping[str, Any]:
    return {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "latent_phases": LATENT_PHASES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "reference_indices": list(REFERENCE_INDICES),
        "arm_order": list(ARM_ORDER),
        "rho_order": [ARM_RHO[arm] for arm in ARM_ORDER],
        "carrier_source_role_order": [ARM_CARRIER_ROLE[arm] for arm in ARM_ORDER],
        "topology": "two_isolated_world4_on_one_8gpu_node",
        "wrong_source_policy": (
            "authoring_sealed_confirmation_source_with_distinct_identity_same_"
            "broad_actor_class_count_and_action_family_used_as_action_family_"
            "matched_source_specificity_control_not_pure_identity_control_"
            "resized_directly_from_raw_rgb_to_target_bucket_then_four_"
            "independent_T1_VAE_calls"
        ),
        "selection_provenance": {
            "asset_path": (
                "methods/bernini_action_editing/assets/"
                "pair_v5_t2v_calibration_first8_authoring_v1.json"
            ),
            "schema_version": AUTHORING_SCHEMA,
            "bank_id": AUTHORING_BANK_ID,
            "file_sha256": AUTHORING_SPEC_SHA256,
            "row_digest_rule": "sha256_canonical_ascii_json_sort_keys_compact",
            "source_analysis_split": "fit",
            "wrong_source_analysis_split": "confirmation",
            "same_action_family_id_required": True,
        },
        "same_source_action_seed_scheduler_guidance_across_five_arms": True,
        "full_source_video_latent_available_to_carrier": False,
        "wrong_source_available_to_model_conditioning": False,
        "target_video": False,
        "mask_flow_pose_track_trajectory": False,
        "trainer_or_critic_instantiated": False,
        "training_harness_executed": False,
        "validation_only_train_lora_module_imported": True,
        "identity_orbit_or_identity_adapter_baseline_or_prior": False,
        "required_posthoc_gate_names": [
            "same_rho_correct_vs_wrong_carrier_source_specificity",
            "old_motion_direction_and_order_leakage_nonincrease",
            "target_action_nonregression_vs_official_gaussian",
            (
                "identity_blur_flicker_camera_quality_nonregression_vs_"
                "official_gaussian"
            ),
        ],
        "posthoc_gates_executed_by_canary": False,
        "best_arm_selection": False,
    }


def load_cell_spec(
    path: str | Path,
    *,
    expected_file_sha256: str,
    cell_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Path, str]:
    """Load one cell and its pre-registered content-matched wrong source."""

    spec_path = _plain_file(path, label="core2 source-set spec")
    observed_sha = file_sha256(spec_path)
    if observed_sha != _sha(
        expected_file_sha256, length=64, label="core2 spec file SHA-256"
    ):
        raise SourceSetNoiseCanaryError("core2 spec file SHA-256 differs")
    try:
        root = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceSetNoiseCanaryError("core2 spec is not valid JSON") from error
    if (
        not isinstance(root, dict)
        or set(root) != {"schema_version", "contract", "cells"}
        or root.get("schema_version") != SPEC_SCHEMA
        or root.get("contract") != _expected_spec_contract()
    ):
        raise SourceSetNoiseCanaryError("core2 spec schema/contract differs")
    cells = root.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise SourceSetNoiseCanaryError("core2 spec must contain dog and human")
    if [row.get("cell_id") for row in cells if isinstance(row, Mapping)] != [
        "dog",
        "human",
    ]:
        raise SourceSetNoiseCanaryError("core2 cell order differs")
    required = {
        "cell_id",
        "actor_kind",
        "source_iid",
        "source_authoring_row_index",
        "source_authoring_row_sha256",
        "source_video",
        "source_video_sha256",
        "wrong_source_iid",
        "wrong_source_authoring_row_index",
        "wrong_source_authoring_row_sha256",
        "wrong_source_video",
        "wrong_source_video_sha256",
        "control_role",
        "geometry_pre_registration",
        "wrong_source_content_match",
        "action_caption",
        "action_caption_utf8_sha256",
        "seed",
        "selected_before_generation",
    }
    expected_matches = {
        "dog": {
            "distinct_source_identity": True,
            "same_broad_actor_class": True,
            "same_actor_count": True,
            "same_authoring_action_family_id": True,
            "same_initial_pose_family": True,
            "initial_pose_match_grade": "same_four_legged_standing_family",
            "fixed_camera_in_both": True,
            "same_exact_scene": False,
            "same_native_geometry": True,
            "direct_resize_geometry_confound": False,
            "pure_identity_control": False,
            "selection_source": (
                "preregistered_confirmation_cell_of_same_action_family"
            ),
            "selected_before_canary_generation": True,
        },
        "human": {
            "distinct_source_identity": True,
            "same_broad_actor_class": True,
            "same_actor_count": True,
            "same_authoring_action_family_id": True,
            "same_initial_pose_family": False,
            "initial_pose_match_grade": (
                "broad_ground_level_bent_leg_state_only_kneel_versus_crouch"
            ),
            "fixed_camera_in_both": True,
            "same_exact_scene": False,
            "same_native_geometry": False,
            "direct_resize_geometry_confound": True,
            "pure_identity_control": False,
            "selection_source": (
                "preregistered_confirmation_cell_of_same_action_family"
            ),
            "selected_before_canary_generation": True,
        },
    }
    expected_geometries = {
        "dog": {
            "source_input_hw": [704, 736],
            "source_native_bucket_hw": [480, 496],
            "wrong_source_input_hw": [704, 736],
            "wrong_source_native_bucket_hw": [480, 496],
            "wrong_source_direct_resize_target_bucket_hw": [480, 496],
            "wrong_source_direct_resize_anisotropic_aspect_distortion_fraction": (
                0.011730205278592365
            ),
            "differential_geometry_confound": False,
        },
        "human": {
            "source_input_hw": [768, 704],
            "source_native_bucket_hw": [512, 464],
            "wrong_source_input_hw": [896, 704],
            "wrong_source_native_bucket_hw": [544, 432],
            "wrong_source_direct_resize_target_bucket_hw": [512, 464],
            "wrong_source_direct_resize_anisotropic_aspect_distortion_fraction": (
                0.15340909090909083
            ),
            "differential_geometry_confound": True,
        },
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in cells:
        if not isinstance(row, Mapping) or set(row) != required:
            raise SourceSetNoiseCanaryError("core2 cell field closure differs")
        row_id = row["cell_id"]
        if row_id not in {"dog", "human"} or row["actor_kind"] != row_id:
            raise SourceSetNoiseCanaryError("cell id/actor role differs")
        if row["wrong_source_content_match"] != expected_matches[row_id]:
            raise SourceSetNoiseCanaryError("wrong source is not content matched")
        if row["geometry_pre_registration"] != expected_geometries[row_id]:
            raise SourceSetNoiseCanaryError("source geometry pre-registration differs")
        if (
            not isinstance(row["source_iid"], str)
            or _IID.fullmatch(row["source_iid"]) is None
            or row["source_iid"] == row["wrong_source_iid"]
            or not isinstance(row["wrong_source_iid"], str)
            or _IID.fullmatch(row["wrong_source_iid"]) is None
        ):
            raise SourceSetNoiseCanaryError("source identity differs")
        if row["control_role"] != (
            "action_family_matched_source_specificity_control_not_pure_identity_control"
        ):
            raise SourceSetNoiseCanaryError("wrong-source control role differs")
        for prefix in ("source", "wrong_source"):
            index = row[f"{prefix}_authoring_row_index"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 8:
                raise SourceSetNoiseCanaryError("authoring row index differs")
            _sha(
                row[f"{prefix}_authoring_row_sha256"],
                length=64,
                label=f"{prefix} authoring row SHA-256",
            )
        if row["selected_before_generation"] is not True:
            raise SourceSetNoiseCanaryError("cell was not selected before generation")
        caption = row["action_caption"]
        if not isinstance(caption, str) or not caption.strip() or "\x00" in caption:
            raise SourceSetNoiseCanaryError("action caption is invalid")
        if hashlib.sha256(caption.encode("utf-8")).hexdigest() != _sha(
            row["action_caption_utf8_sha256"],
            length=64,
            label="action caption SHA-256",
        ):
            raise SourceSetNoiseCanaryError("action caption SHA-256 differs")
        seed = row["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise SourceSetNoiseCanaryError("cell seed lies outside [0,2^63)")
        source = _plain_file(row["source_video"], label=f"{row_id} source")
        if file_sha256(source) != _sha(
            row["source_video_sha256"], length=64, label="source video SHA-256"
        ):
            raise SourceSetNoiseCanaryError(f"{row_id} source SHA-256 differs")
        wrong_source = _plain_file(
            row["wrong_source_video"], label=f"{row_id} content-matched wrong source"
        )
        wrong_sha = _sha(
            row["wrong_source_video_sha256"],
            length=64,
            label="wrong source video SHA-256",
        )
        if (
            file_sha256(wrong_source) != wrong_sha
            or wrong_sha == row["source_video_sha256"]
            or wrong_source == source
        ):
            raise SourceSetNoiseCanaryError(
                f"{row_id} wrong source is absent, changed, or aliases correct"
            )
        by_id[row_id] = dict(row)
    if set(by_id) != {"dog", "human"}:
        raise SourceSetNoiseCanaryError("core2 cell identities differ")
    selected = by_id.get(cell_id)
    if selected is None:
        raise SourceSetNoiseCanaryError("requested cell is absent")
    wrong = {
        "iid": selected["wrong_source_iid"],
        "source_video": selected["wrong_source_video"],
        "source_video_sha256": selected["wrong_source_video_sha256"],
        "actor_kind": selected["actor_kind"],
        "content_match": dict(selected["wrong_source_content_match"]),
        "control_role": selected["control_role"],
        "condition_role": "noise_carrier_only",
    }
    return root, selected, wrong, spec_path, observed_sha


def load_authoring_provenance(
    path: str | Path,
    *,
    expected_file_sha256: str,
    cell: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Path, str]:
    """Bind both source choices to rows in the sealed upstream authoring bank."""

    authoring_path = _plain_file(path, label="pair-v5 first8 authoring spec")
    expected_sha = _sha(
        expected_file_sha256, length=64, label="authoring spec file SHA-256"
    )
    if expected_sha != AUTHORING_SPEC_SHA256:
        raise SourceSetNoiseCanaryError("unsupported authoring spec SHA-256")
    observed_sha = file_sha256(authoring_path)
    if observed_sha != expected_sha:
        raise SourceSetNoiseCanaryError("authoring spec file SHA-256 differs")
    try:
        root = json.loads(authoring_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceSetNoiseCanaryError("authoring spec is not valid JSON") from error
    if (
        not isinstance(root, dict)
        or set(root) != {"schema_version", "bank_id", "expected_cell_count", "cells"}
        or root.get("schema_version") != AUTHORING_SCHEMA
        or root.get("bank_id") != AUTHORING_BANK_ID
        or root.get("expected_cell_count") != 8
        or not isinstance(root.get("cells"), list)
        or len(root["cells"]) != 8
    ):
        raise SourceSetNoiseCanaryError("authoring spec schema/bank closure differs")

    selected_rows: list[Mapping[str, Any]] = []
    for prefix, expected_split in (("source", "fit"), ("wrong_source", "confirmation")):
        index = int(cell[f"{prefix}_authoring_row_index"])
        row = root["cells"][index]
        if not isinstance(row, Mapping):
            raise SourceSetNoiseCanaryError("authoring row is not a mapping")
        if object_sha256(row) != cell[f"{prefix}_authoring_row_sha256"]:
            raise SourceSetNoiseCanaryError("authoring row digest differs")
        if (
            row.get("iid") != cell[f"{prefix}_iid"]
            or row.get("analysis_split") != expected_split
            or row.get("geometry_source_video") != cell[f"{prefix}_video"]
            or not isinstance(row.get("action_family_id"), str)
            or not row["action_family_id"]
        ):
            raise SourceSetNoiseCanaryError("authoring source row binding differs")
        selected_rows.append(dict(row))
    source_row, wrong_row = selected_rows
    if (
        source_row["action_family_id"] != wrong_row["action_family_id"]
        or source_row.get("actor_group_id") == wrong_row.get("actor_group_id")
        or source_row.get("scene_group_id") == wrong_row.get("scene_group_id")
        or source_row.get("iid") == wrong_row.get("iid")
    ):
        raise SourceSetNoiseCanaryError(
            "wrong source is not a distinct same-action-family confirmation row"
        )
    branches = source_row.get("branch_descriptions")
    if not isinstance(branches, Mapping) or not isinstance(branches.get("action"), str):
        raise SourceSetNoiseCanaryError("source authoring action branch differs")
    expected_caption = " ".join(
        (source_row.get("scene_caption", ""), branches["action"], source_row.get("camera_caption", ""))
    )
    if expected_caption != cell["action_caption"]:
        raise SourceSetNoiseCanaryError("canary action caption differs from authoring row")
    return root, source_row, wrong_row, authoring_path, observed_sha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-spec", required=True)
    parser.add_argument("--expected-cell-spec-sha256", required=True)
    parser.add_argument("--authoring-spec", required=True)
    parser.add_argument("--expected-authoring-spec-sha256", required=True)
    parser.add_argument("--cell-id", choices=("dog", "human"), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256", required=True
    )
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=native.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    return parser


def validate_cli(args: argparse.Namespace) -> Path:
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise SourceSetNoiseCanaryError("canary is fixed to exact40")
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_cell_spec_sha256",
        "expected_authoring_spec_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise SourceSetNoiseCanaryError("unsupported Bernini revision")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise SourceSetNoiseCanaryError("unsupported VeOmni revision")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise SourceSetNoiseCanaryError("unsupported checkpoint tree")
    if (
        args.expected_checkpoint_content_manifest_sha256
        != native.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise SourceSetNoiseCanaryError("unsupported checkpoint content manifest")
    output = Path(args.output_dir).expanduser()
    if (
        not output.is_absolute()
        or output == Path("/")
        or _SAFE_NAME.fullmatch(output.name) is None
    ):
        raise SourceSetNoiseCanaryError("output-dir must be absolute, non-root, safe")
    return native._resolve_fresh_output_dir(output)


def _tensor_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    try:
        return native.value_audit.tensor_identity(value, label=label)
    except Exception as error:
        raise SourceSetNoiseCanaryError(str(error)) from error


def _sample_with_source_set_noise_arm(
    *,
    sample_fn: Callable[[], Any],
    wan_diffusion_module: Any,
    arm: str,
    correct_frame_latents_cpu: Sequence[Any],
    wrong_frame_latents_cpu: Sequence[Any],
    expected_shape: Sequence[int],
    expected_device: Any,
    expected_seed: int,
    canonical_randn_tensor: Optional[Callable[..., Any]] = None,
) -> tuple[Any, SourceSetNoiseCapture]:
    """Replace only the official initial Gaussian return for one arm."""

    try:
        import torch

        if canonical_randn_tensor is None:
            from diffusers.utils.torch_utils import randn_tensor as canonical
        else:
            canonical = canonical_randn_tensor
    except ImportError as error:  # pragma: no cover - AUH runtime supplies deps
        raise SourceSetNoiseCanaryError("noise hook requires torch/diffusers") from error
    if arm not in ARM_ORDER or not callable(sample_fn):
        raise SourceSetNoiseCanaryError("noise arm/sample callable differs")
    expected = tuple(int(item) for item in expected_shape)
    if len(expected) != 5 or expected[:3] != (1, 16, LATENT_PHASES):
        raise SourceSetNoiseCanaryError("native target shape is not exact81")
    if type(expected_seed) is not int or not 0 <= expected_seed < 2**63:
        raise SourceSetNoiseCanaryError("expected seed differs")
    role = ARM_CARRIER_ROLE[arm]
    rho = ARM_RHO[arm]
    frames = (
        tuple(correct_frame_latents_cpu)
        if role == "correct"
        else tuple(wrong_frame_latents_cpu)
    )
    original = getattr(wan_diffusion_module, "randn_tensor", None)
    if original is not canonical:
        raise SourceSetNoiseCanaryError("pinned randn_tensor is already replaced")
    calls: list[Mapping[str, Any]] = []

    def injected_randn_tensor(*call_args: Any, **call_kwargs: Any) -> Any:
        shape_value = call_args[0] if call_args else call_kwargs.get("shape")
        try:
            requested_shape = tuple(int(item) for item in shape_value)
        except Exception as error:
            raise SourceSetNoiseCanaryError("native randn shape differs") from error
        generator = call_kwargs.get("generator")
        if not isinstance(generator, torch.Generator):
            raise SourceSetNoiseCanaryError("native noise lacks one torch.Generator")
        official_native = original(*call_args, **call_kwargs)
        if (
            not isinstance(official_native, torch.Tensor)
            or tuple(int(item) for item in official_native.shape) != expected
            or official_native.dtype != torch.float32
            or official_native.device != torch.device(expected_device)
            or not official_native.is_contiguous()
            or official_native.requires_grad
            or not bool(torch.isfinite(official_native).all().item())
        ):
            raise SourceSetNoiseCanaryError("official native Gaussian differs")
        official_cpu = (
            official_native.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        result = source_set_noise.build_orderless_source_frame_set_noise(
            canonical_gaussian=official_cpu,
            independent_frame_latents=frames,
            rho=rho,
        )
        if rho == 0.0:
            if result.initial_noise is not official_cpu:
                raise SourceSetNoiseCanaryError("rho0 lost exact CPU Gaussian alias")
            injected_native = official_native
        else:
            injected_native = result.initial_noise.to(
                device=official_native.device, dtype=official_native.dtype
            ).contiguous()
            if injected_native is official_native or torch.equal(
                injected_native, official_native
            ):
                raise SourceSetNoiseCanaryError("active source carrier changed no noise")
        injected_cpu = (
            injected_native.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        prototype_cpu = (
            None
            if result.source_set_prototype is None
            else result.source_set_prototype.detach().cpu().contiguous().clone()
        )
        carrier_cpu = (
            None
            if result.temporal_dc_carrier is None
            else result.temporal_dc_carrier.detach().cpu().contiguous().clone()
        )
        calls.append(
            {
                "requested_shape": requested_shape,
                "requested_dtype": str(call_kwargs.get("dtype")),
                "requested_device": str(call_kwargs.get("device")),
                "generator_device": str(generator.device),
                "generator_initial_seed": int(generator.initial_seed()),
                "official": official_cpu,
                "injected": injected_cpu,
                "prototype": prototype_cpu,
                "carrier": carrier_cpu,
                "original_return_forwarded": injected_native is official_native,
                "operator_receipt": dict(result.receipt),
                "operator_receipt_sha256": result.receipt_sha256,
                "diagnostics": result.diagnostics.as_dict(),
            }
        )
        return injected_native

    setattr(injected_randn_tensor, "_orderless_source_set_noise_injector", True)
    setattr(wan_diffusion_module, "randn_tensor", injected_randn_tensor)
    wrapper_unchanged = True
    try:
        sample_result = sample_fn()
    finally:
        wrapper_unchanged = (
            getattr(wan_diffusion_module, "randn_tensor", None)
            is injected_randn_tensor
        )
        setattr(wan_diffusion_module, "randn_tensor", original)
    if (
        not wrapper_unchanged
        or getattr(wan_diffusion_module, "randn_tensor", None) is not original
    ):
        raise SourceSetNoiseCanaryError("randn_tensor hook did not restore exactly")
    if len(calls) != 1:
        raise SourceSetNoiseCanaryError("native sampler must call randn_tensor once")
    call = calls[0]
    if (
        call["requested_shape"] != expected
        or call["requested_dtype"] != str(torch.float32)
        or call["requested_device"] != str(torch.device(expected_device))
        or call["generator_device"] != "cpu"
        or call["generator_initial_seed"] != expected_seed
    ):
        raise SourceSetNoiseCanaryError("native RNG request/seed differs")
    official_identity = _tensor_identity(call["official"], label="official_gaussian")
    injected_identity = _tensor_identity(call["injected"], label="sampler_initial_noise")
    diagnostics = call["diagnostics"]
    active = rho > 0.0
    if (
        bool(call["original_return_forwarded"]) is active
        or (not active and official_identity["raw_storage_sha256"]
            != injected_identity["raw_storage_sha256"])
        or diagnostics.get("rho") != rho
        or diagnostics.get("source_frame_order_consumed") is not False
        or diagnostics.get("source_temporal_phase_consumed") is not False
    ):
        raise SourceSetNoiseCanaryError("operator arm contract differs")
    return sample_result, SourceSetNoiseCapture(
        arm=arm,
        rho=rho,
        carrier_source_role=role,
        official_gaussian=call["official"],
        sampler_initial_noise=call["injected"],
        source_set_prototype=call["prototype"],
        temporal_dc_carrier=call["carrier"],
        official_gaussian_raw_value_sha256=str(
            official_identity["raw_storage_sha256"]
        ),
        sampler_initial_noise_raw_value_sha256=str(
            injected_identity["raw_storage_sha256"]
        ),
        source_frame_multiset_sha256=str(
            diagnostics["source_frame_multiset_sha256"]
        ),
        source_set_prototype_sha256=diagnostics["source_set_prototype_sha256"],
        temporal_dc_carrier_sha256=diagnostics["transported_carrier_sha256"],
        source_set_prototype_raw_value_sha256=(
            None
            if call["prototype"] is None
            else str(
                _tensor_identity(
                    call["prototype"], label="source_set_prototype_raw"
                )["raw_storage_sha256"]
            )
        ),
        temporal_dc_carrier_raw_value_sha256=(
            None
            if call["carrier"] is None
            else str(
                _tensor_identity(
                    call["carrier"], label="temporal_dc_carrier_raw"
                )["raw_storage_sha256"]
            )
        ),
        operator_receipt=call["operator_receipt"],
        operator_receipt_sha256=str(call["operator_receipt_sha256"]),
        requested_shape=expected,
        requested_dtype=call["requested_dtype"],
        requested_device=call["requested_device"],
        generator_device=call["generator_device"],
        generator_initial_seed=int(call["generator_initial_seed"]),
        original_randn_call_count=1,
        external_initial_noise_injection=active,
        original_return_object_forwarded=bool(call["original_return_forwarded"]),
    )


def _prepare_source_snapshot_at_bucket(
    source_path: Path,
    *,
    bucket_hw: Sequence[int],
) -> tuple[Any, Mapping[str, Any], str]:
    """Hash-snapshot and resize a matched wrong source into target bucket."""

    from tools import materialize_vae

    target_hw = (int(bucket_hw[0]), int(bucket_hw[1]))
    if any(value <= 0 or value % 16 for value in target_hw):
        raise SourceSetNoiseCanaryError("target bucket is not positive stride16")
    before = source_path.stat()
    source_sha = file_sha256(source_path)
    with tempfile.TemporaryDirectory(prefix="bernini-wrong-source-snapshot-") as root:
        snapshot = Path(root) / "source.mp4"
        shutil.copyfile(source_path, snapshot)
        snapshot_sha = file_sha256(snapshot)
        if snapshot_sha != source_sha:
            raise SourceSetNoiseCanaryError("wrong-source snapshot SHA differs")
        frames, reported_fps, source_hw = materialize_vae._decode_exact_video(snapshot)
        native.legacy.validate_exact_video_metadata(int(frames.shape[0]), reported_fps)
        resized = materialize_vae._resize_video(frames, target_hw, None)
    after = source_path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or file_sha256(source_path) != source_sha
    ):
        raise SourceSetNoiseCanaryError("wrong source changed during snapshot decode")
    expected_shape = (3, FRAME_COUNT, target_hw[0], target_hw[1])
    if tuple(int(item) for item in resized.shape) != expected_shape:
        raise SourceSetNoiseCanaryError("wrong-source target resize shape differs")
    native_bucket = materialize_vae.source_aspect_bucket(
        *source_hw,
        max_pixels=native.legacy.MAX_PIXELS,
        stride=native.legacy.SPATIAL_STRIDE,
    )
    height_scale = float(target_hw[0]) / float(source_hw[0])
    width_scale = float(target_hw[1]) / float(source_hw[1])
    anisotropic_aspect_distortion = max(
        height_scale / width_scale, width_scale / height_scale
    ) - 1.0
    metadata = {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "reported_fps": float(reported_fps),
        "source_input_hw": list(source_hw),
        "source_native_bucket_hw": list(native_bucket),
        "target_cell_bucket_hw": list(target_hw),
        "native_bucket_matches_target_cell_bucket": native_bucket == target_hw,
        "direct_resize_height_scale": height_scale,
        "direct_resize_width_scale": width_scale,
        "direct_resize_anisotropic_aspect_distortion_fraction": (
            anisotropic_aspect_distortion
        ),
        "decoded_from_private_byte_snapshot": True,
        "snapshot_sha256": snapshot_sha,
        "direct_raw_rgb_to_target_bucket_resize": True,
        "double_resize_used": False,
        "resize": "torchvision_bicubic_antialias_true",
        "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
        "used_as_model_condition": False,
        "used_as_noise_carrier_only": True,
    }
    return resized.unsqueeze(0), metadata, source_sha


def _save_tensor_artifact(
    path: Path,
    value: Any,
    *,
    key: str,
    metadata: Mapping[str, str],
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or path.suffix != ".safetensors":
        raise SourceSetNoiseCanaryError("tensor artifact path must be fresh")
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.ndim != 5 or not bool(torch.isfinite(tensor).all().item()):
        raise SourceSetNoiseCanaryError("tensor artifact must be finite rank5")
    identity = _tensor_identity(tensor, label=key)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file({key: tensor}, str(temporary), metadata=dict(metadata))
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [key]:
                raise SourceSetNoiseCanaryError("tensor artifact key differs")
            restored = opened.get_tensor(key).contiguous()
            restored_metadata = dict(opened.metadata() or {})
        if not torch.equal(restored, tensor):
            raise SourceSetNoiseCanaryError("tensor artifact roundtrip differs")
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "tensor_key": key,
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "raw_value_sha256": identity["raw_storage_sha256"],
        "content_sha256": identity["content_sha256"],
        "metadata": restored_metadata,
        "roundtrip_byte_exact": True,
    }


def _save_frame_set_artifact(
    path: Path,
    frames: Sequence[Any],
    *,
    role: str,
    multiset_sha256: str,
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or role not in {"correct", "wrong"}:
        raise SourceSetNoiseCanaryError("frame-set artifact path/role differs")
    if len(frames) != len(REFERENCE_INDICES):
        raise SourceSetNoiseCanaryError("frame set must contain four refs")
    tensors = {
        f"frame_{index:03d}": frame.detach().cpu().float().contiguous()
        for index, frame in zip(REFERENCE_INDICES, frames)
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "coordinate": "independent_RGB_frame_to_Wan_VAE_T1_latents",
            "carrier_source_role": role,
            "caller_member_selection_indices": json.dumps(
                list(REFERENCE_INDICES), separators=(",", ":")
            ),
            "caller_selection_indices_consumed": "true",
            "operator_received_frame_indices": "false",
            "operator_set_sequence_order_consumed": "false",
            "unordered_pose_occupancy_leakage_possible": "true",
            "full_video_latent_consumed": "false",
            "frame_multiset_sha256": multiset_sha256,
        },
    )
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        keys = list(opened.keys())
        restored = {key: opened.get_tensor(key).contiguous() for key in keys}
        metadata = dict(opened.metadata() or {})
    if keys != sorted(tensors) or any(
        not torch.equal(restored[key], tensors[key]) for key in keys
    ):
        raise SourceSetNoiseCanaryError("frame-set artifact roundtrip differs")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "keys": keys,
        "frame_multiset_sha256": multiset_sha256,
        "caller_member_selection_indices": list(REFERENCE_INDICES),
        "caller_selection_indices_consumed": True,
        "operator_received_frame_indices": False,
        "operator_set_sequence_order_consumed": False,
        "unordered_pose_occupancy_leakage_possible": True,
        "full_video_latent_consumed": False,
        "metadata": metadata,
        "roundtrip_byte_exact": True,
    }


def _validate_generated(value: Any, *, shape: Sequence[int], device: Any, label: str) -> Any:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or tuple(int(item) for item in value.shape) != tuple(int(item) for item in shape)
        or value.dtype != torch.float32
        or value.device != torch.device(device)
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SourceSetNoiseCanaryError(f"{label} native endpoint differs")
    return value


def _sampling_contract(seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract(
        "rv2v", steps=NUM_INFERENCE_STEPS, seed=seed
    )
    if (
        value["num_frames"] != FRAME_COUNT
        or value["num_inference_steps"] != NUM_INFERENCE_STEPS
    ):
        raise SourceSetNoiseCanaryError("native exact40/exact81 contract differs")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = validate_cli(args)
    root_spec, cell, wrong_cell, spec_path, spec_sha = load_cell_spec(
        args.cell_spec,
        expected_file_sha256=args.expected_cell_spec_sha256,
        cell_id=args.cell_id,
    )
    (
        authoring_root,
        source_authoring_row,
        wrong_authoring_row,
        authoring_path,
        authoring_sha,
    ) = load_authoring_provenance(
        args.authoring_spec,
        expected_file_sha256=args.expected_authoring_spec_sha256,
        cell=cell,
    )
    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, label="checkpoint content manifest"
    )
    if file_sha256(checkpoint_manifest) != args.expected_checkpoint_content_manifest_sha256:
        raise SourceSetNoiseCanaryError("checkpoint manifest SHA-256 differs")
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
        raise SourceSetNoiseCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % SP_SIZE:
        raise SourceSetNoiseCanaryError("checkpoint heads are not SP4-compatible")
    inference_hashes = native.legacy.validate_inference_source_files(bernini_root)
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

    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise SourceSetNoiseCanaryError("native negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise SourceSetNoiseCanaryError("canary requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    try:
        checkpoint_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_rows[0] = {
                    "ok": True,
                    "identity": native.source_audit.validate_checkpoint_content(
                        checkpoint,
                        checkpoint_manifest,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                checkpoint_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_rows, src=0)
        if checkpoint_rows[0].get("ok") is not True:
            raise SourceSetNoiseCanaryError(
                f"checkpoint validation failed: {checkpoint_rows[0]}"
            )
        checkpoint_identity = dict(checkpoint_rows[0]["identity"])

        source_path = _plain_file(cell["source_video"], label="correct source")
        wrong_source_path = _plain_file(
            wrong_cell["source_video"], label="wrong source"
        )
        source_tensor, source_metadata, source_sha = (
            native.source_audit.prepare_hashed_source_snapshot(source_path)
        )
        if source_sha != cell["source_video_sha256"]:
            raise SourceSetNoiseCanaryError("correct source SHA-256 differs")
        bucket_hw = tuple(
            int(item) for item in source_metadata["source_derived_bucket_hw"]
        )
        wrong_tensor, wrong_metadata, wrong_sha = _prepare_source_snapshot_at_bucket(
            wrong_source_path, bucket_hw=bucket_hw
        )
        if wrong_sha != wrong_cell["source_video_sha256"]:
            raise SourceSetNoiseCanaryError("wrong source SHA-256 differs")
        expected_same_geometry = bool(
            wrong_cell["content_match"]["same_native_geometry"]
        )
        observed_same_geometry = bool(
            wrong_metadata["native_bucket_matches_target_cell_bucket"]
        )
        expected_geometry_confound = bool(
            wrong_cell["content_match"]["direct_resize_geometry_confound"]
        )
        observed_distortion = float(
            wrong_metadata["direct_resize_anisotropic_aspect_distortion_fraction"]
        )
        geometry_pre_registration = cell["geometry_pre_registration"]
        if (
            source_metadata["source_input_hw"]
            != geometry_pre_registration["source_input_hw"]
            or source_metadata["source_derived_bucket_hw"]
            != geometry_pre_registration["source_native_bucket_hw"]
            or wrong_metadata["source_input_hw"]
            != geometry_pre_registration["wrong_source_input_hw"]
            or wrong_metadata["source_native_bucket_hw"]
            != geometry_pre_registration["wrong_source_native_bucket_hw"]
            or wrong_metadata["target_cell_bucket_hw"]
            != geometry_pre_registration[
                "wrong_source_direct_resize_target_bucket_hw"
            ]
            or abs(
                observed_distortion
                - geometry_pre_registration[
                    "wrong_source_direct_resize_anisotropic_aspect_distortion_fraction"
                ]
            )
            > 1.0e-12
            or expected_geometry_confound
            is not geometry_pre_registration["differential_geometry_confound"]
            or observed_same_geometry is not expected_same_geometry
            or expected_geometry_confound is observed_same_geometry
            or (expected_geometry_confound and observed_distortion <= 0.01)
        ):
            raise SourceSetNoiseCanaryError(
                "wrong-source geometry/confound differs from pre-registration"
            )

        task_prompt = native.build_task_prompt(
            "rv2v", cell["action_caption"], prompt_cleaner=prompt_clean
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            subfolder="tokenizer",
            **native.legacy.tokenizer_load_kwargs(),
        )
        positive_ids, positive_mask = native.legacy._tokenize_training_prompt(
            tokenizer, task_prompt
        )
        negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
            tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
        )

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **native.legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        native.legacy.trainer.validate_renderer_config_mapping(
            config.to_dict(), checkpoint
        )
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            raise SourceSetNoiseCanaryError("renderer is not native UniPC shift5")
        model = BerniniRendererModel(config)
        model.eval().requires_grad_(False)
        freeze_before = native.source_audit.model_freeze_certificate(model)
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False)
        vae.to(device)
        correct_pixels = source_tensor.to(device=device, dtype=torch.float32)
        wrong_pixels = wrong_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            full_source_latent = _vae_encode(vae, correct_pixels).contiguous()
            correct_refs = {
                index: _vae_encode(
                    vae, correct_pixels[:, :, index : index + 1].contiguous()
                ).contiguous()
                for index in REFERENCE_INDICES
            }
            wrong_refs = {
                index: _vae_encode(
                    vae, wrong_pixels[:, :, index : index + 1].contiguous()
                ).contiguous()
                for index in REFERENCE_INDICES
            }
        broadcasts = {
            "full_correct_source": native._broadcast_condition_from_rank_zero(
                full_source_latent,
                label="full_correct_source",
                world_size=WORLD_SIZE,
            ),
            "correct_refs": {
                str(index): native._broadcast_condition_from_rank_zero(
                    value,
                    label=f"correct_ref_{index}",
                    world_size=WORLD_SIZE,
                )
                for index, value in correct_refs.items()
            },
            "wrong_carrier_refs": {
                str(index): native._broadcast_condition_from_rank_zero(
                    value,
                    label=f"wrong_carrier_ref_{index}",
                    world_size=WORLD_SIZE,
                )
                for index, value in wrong_refs.items()
            },
        }
        geometry = native._latent_geometry_receipt(
            bucket_hw=bucket_hw, z_dim=int(vae.config.z_dim)
        )
        video_shape = tuple(int(item) for item in geometry["video_latent_shape"])
        ref_shape = tuple(int(item) for item in geometry["reference_latent_shape"])
        if (
            tuple(full_source_latent.shape) != video_shape
            or video_shape[:3] != (1, 16, LATENT_PHASES)
            or any(tuple(value.shape) != ref_shape for value in correct_refs.values())
            or any(tuple(value.shape) != ref_shape for value in wrong_refs.values())
        ):
            raise SourceSetNoiseCanaryError("source/reference geometry differs")
        condition_identities = {
            "full_correct_source": native._all_rank_tensor_identity(
                full_source_latent,
                label="full_correct_source",
                world_size=WORLD_SIZE,
            ),
            "correct_refs": {
                str(index): native._all_rank_tensor_identity(
                    value, label=f"correct_ref_{index}", world_size=WORLD_SIZE
                )
                for index, value in correct_refs.items()
            },
            "wrong_carrier_refs": {
                str(index): native._all_rank_tensor_identity(
                    value,
                    label=f"wrong_carrier_ref_{index}",
                    world_size=WORLD_SIZE,
                )
                for index, value in wrong_refs.items()
            },
        }
        correct_frames_cpu = tuple(
            correct_refs[index].detach().cpu().float().contiguous().clone()
            for index in REFERENCE_INDICES
        )
        wrong_frames_cpu = tuple(
            wrong_refs[index].detach().cpu().float().contiguous().clone()
            for index in REFERENCE_INDICES
        )
        vae.to("cpu")
        del source_tensor, wrong_tensor, correct_pixels, wrong_pixels
        torch.cuda.empty_cache()
        model.to(device)
        conditions = {
            "image_vae_latents": None,
            "multi_video_vae_latents": [full_source_latent],
            "multi_image_vae_latents": [
                correct_refs[index] for index in REFERENCE_INDICES
            ],
        }
        generated: dict[str, Any] = {}
        generated_identities: dict[str, Any] = {}
        captures: dict[str, SourceSetNoiseCapture] = {}
        seed = int(cell["seed"])
        for arm in ARM_ORDER:
            with torch.inference_mode():
                endpoint, capture = _sample_with_source_set_noise_arm(
                    sample_fn=lambda: model.sample(
                        input_ids=positive_ids.to(device),
                        attention_mask=positive_mask.to(device),
                        uncond_input_ids=negative_ids.to(device),
                        uncond_attention_mask=negative_mask.to(device),
                        **conditions,
                        width=bucket_hw[1],
                        height=bucket_hw[0],
                        device=device,
                        **_sampling_contract(seed),
                    ),
                    wan_diffusion_module=wan_diffusion,
                    arm=arm,
                    correct_frame_latents_cpu=correct_frames_cpu,
                    wrong_frame_latents_cpu=wrong_frames_cpu,
                    expected_shape=video_shape,
                    expected_device=device,
                    expected_seed=seed,
                )
            endpoint = _validate_generated(
                endpoint, shape=video_shape, device=device, label=arm
            )
            generated[arm] = endpoint.detach().cpu().contiguous()
            generated_identities[arm] = native._all_rank_tensor_identity(
                generated[arm], label=f"generated_{arm}", world_size=WORLD_SIZE
            )
            native._all_rank_tensor_identity(
                capture.official_gaussian,
                label=f"official_gaussian_{arm}",
                world_size=WORLD_SIZE,
            )
            native._all_rank_tensor_identity(
                capture.sampler_initial_noise,
                label=f"sampler_initial_noise_{arm}",
                world_size=WORLD_SIZE,
            )
            if capture.source_set_prototype is not None:
                native._all_rank_tensor_identity(
                    capture.source_set_prototype,
                    label=f"source_set_prototype_{arm}",
                    world_size=WORLD_SIZE,
                )
                native._all_rank_tensor_identity(
                    capture.temporal_dc_carrier,
                    label=f"temporal_dc_carrier_{arm}",
                    world_size=WORLD_SIZE,
                )
            captures[arm] = capture
            del endpoint
            torch.cuda.empty_cache()

        official_hashes = {
            capture.official_gaussian_raw_value_sha256
            for capture in captures.values()
        }
        if len(official_hashes) != 1:
            raise SourceSetNoiseCanaryError("five arms do not share one Gaussian")
        official = captures["official_gaussian"]
        if (
            official.external_initial_noise_injection
            or not official.original_return_object_forwarded
            or official.official_gaussian_raw_value_sha256
            != official.sampler_initial_noise_raw_value_sha256
        ):
            raise SourceSetNoiseCanaryError("official rho0 arm is not native exact")
        for arm in ARM_ORDER[1:]:
            capture = captures[arm]
            if (
                not capture.external_initial_noise_injection
                or capture.original_return_object_forwarded
                or capture.official_gaussian_raw_value_sha256
                != official.official_gaussian_raw_value_sha256
            ):
                raise SourceSetNoiseCanaryError("active arm lost official parent")
        for role in ("correct", "wrong"):
            low = captures[f"{role}_source_rho005"]
            high = captures[f"{role}_source_rho010"]
            if (
                low.source_frame_multiset_sha256
                != high.source_frame_multiset_sha256
                or low.source_set_prototype_sha256 != high.source_set_prototype_sha256
                or low.temporal_dc_carrier_sha256 != high.temporal_dc_carrier_sha256
            ):
                raise SourceSetNoiseCanaryError(
                    f"{role} rho arms do not share one source carrier"
                )
        if (
            captures["correct_source_rho005"].source_frame_multiset_sha256
            == captures["wrong_source_rho005"].source_frame_multiset_sha256
            or captures["correct_source_rho005"].temporal_dc_carrier_sha256
            == captures["wrong_source_rho005"].temporal_dc_carrier_sha256
        ):
            raise SourceSetNoiseCanaryError("wrong-source carrier aliases correct")
        for rho_suffix in ("rho005", "rho010"):
            if (
                captures[
                    f"correct_source_{rho_suffix}"
                ].sampler_initial_noise_raw_value_sha256
                == captures[
                    f"wrong_source_{rho_suffix}"
                ].sampler_initial_noise_raw_value_sha256
            ):
                raise SourceSetNoiseCanaryError(
                    f"{rho_suffix} correct/wrong injected endpoints alias"
                )
        for role in ("correct", "wrong"):
            if (
                captures[
                    f"{role}_source_rho005"
                ].sampler_initial_noise_raw_value_sha256
                == captures[
                    f"{role}_source_rho010"
                ].sampler_initial_noise_raw_value_sha256
            ):
                raise SourceSetNoiseCanaryError(
                    f"{role} rho005/rho010 injected endpoints alias"
                )

        freeze_after = native.source_audit.model_freeze_certificate(model)
        if freeze_after != freeze_before or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise SourceSetNoiseCanaryError("frozen model changed")
        model.to("cpu")
        torch.cuda.empty_cache()
        after_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                after_rows[0] = {
                    "ok": True,
                    "identity": native.source_audit.validate_checkpoint_content(
                        checkpoint,
                        checkpoint_manifest,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                after_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(after_rows, src=0)
        if (
            after_rows[0].get("ok") is not True
            or after_rows[0].get("identity") != checkpoint_identity
        ):
            raise SourceSetNoiseCanaryError("checkpoint changed during decode")

        if distributed.rank == 0:
            stage = _output_staging_directory(output_dir)
            correct_multiset = captures[
                "correct_source_rho005"
            ].source_frame_multiset_sha256
            wrong_multiset = captures[
                "wrong_source_rho005"
            ].source_frame_multiset_sha256
            frame_set_artifacts = {
                "correct": _save_frame_set_artifact(
                    stage / "correct-source-frame-set.safetensors",
                    correct_frames_cpu,
                    role="correct",
                    multiset_sha256=correct_multiset,
                ),
                "wrong": _save_frame_set_artifact(
                    stage / "wrong-source-frame-set.safetensors",
                    wrong_frames_cpu,
                    role="wrong",
                    multiset_sha256=wrong_multiset,
                ),
            }
            official_artifact = _save_tensor_artifact(
                stage / "official-parent-gaussian.safetensors",
                official.official_gaussian,
                key="official_parent_gaussian",
                metadata={
                    "coordinate": "unpacked_clean_epsilon_before_patch_packing",
                    "source": "return_of_pinned_bernini_wan_diffusion_randn_tensor",
                    "external_initial_noise_injection": "false",
                },
            )
            if (
                official_artifact["raw_value_sha256"]
                != official.official_gaussian_raw_value_sha256
            ):
                raise SourceSetNoiseCanaryError(
                    "saved official Gaussian differs from observed parent"
                )
            arm_rows: list[Mapping[str, Any]] = []
            for arm in ARM_ORDER:
                capture = captures[arm]
                initial_artifact = _save_tensor_artifact(
                    stage / f"{arm}.sampler-initial-noise.safetensors",
                    capture.sampler_initial_noise,
                    key="sampler_initial_noise",
                    metadata={
                        "coordinate": "unpacked_clean_epsilon_before_patch_packing",
                        "arm": arm,
                        "rho": str(capture.rho),
                        "carrier_source_role": capture.carrier_source_role,
                        "external_initial_noise_injection": str(
                            capture.external_initial_noise_injection
                        ).lower(),
                    },
                )
                if (
                    initial_artifact["raw_value_sha256"]
                    != capture.sampler_initial_noise_raw_value_sha256
                    or object_sha256(capture.operator_receipt)
                    != capture.operator_receipt_sha256
                ):
                    raise SourceSetNoiseCanaryError(
                        f"{arm} initial noise/operator receipt artifact differs"
                    )
                prototype_artifact = None
                carrier_artifact = None
                if capture.source_set_prototype is not None:
                    prototype_artifact = _save_tensor_artifact(
                        stage / f"{arm}.source-set-prototype.safetensors",
                        capture.source_set_prototype,
                        key="source_set_prototype",
                        metadata={
                            "coordinate": "Wan_T1_latent_robust_set_barycenter",
                            "arm": arm,
                            "carrier_source_role": capture.carrier_source_role,
                            "frame_order_consumed": "false",
                        },
                    )
                    carrier_artifact = _save_tensor_artifact(
                        stage / f"{arm}.temporal-dc-carrier.safetensors",
                        capture.temporal_dc_carrier,
                        key="temporal_dc_carrier",
                        metadata={
                            "coordinate": "unpacked_clean_epsilon_temporal_dc",
                            "arm": arm,
                            "carrier_source_role": capture.carrier_source_role,
                            "strict_temporal_dc": "true",
                        },
                    )
                    if (
                        prototype_artifact["raw_value_sha256"]
                        != capture.source_set_prototype_raw_value_sha256
                        or carrier_artifact["raw_value_sha256"]
                        != capture.temporal_dc_carrier_raw_value_sha256
                    ):
                        raise SourceSetNoiseCanaryError(
                            f"{arm} prototype/carrier artifact differs"
                        )
                unsigned_arm = {
                    "arm": arm,
                    "rho": capture.rho,
                    "carrier_source_role": capture.carrier_source_role,
                    "official_parent_gaussian_raw_value_sha256": (
                        capture.official_gaussian_raw_value_sha256
                    ),
                    "sampler_initial_noise_raw_value_sha256": (
                        capture.sampler_initial_noise_raw_value_sha256
                    ),
                    "source_frame_multiset_sha256": (
                        capture.source_frame_multiset_sha256
                    ),
                    "source_set_prototype_sha256": (
                        capture.source_set_prototype_sha256
                    ),
                    "temporal_dc_carrier_sha256": (
                        capture.temporal_dc_carrier_sha256
                    ),
                    "source_set_prototype_raw_value_sha256": (
                        capture.source_set_prototype_raw_value_sha256
                    ),
                    "temporal_dc_carrier_raw_value_sha256": (
                        capture.temporal_dc_carrier_raw_value_sha256
                    ),
                    "operator_receipt": dict(capture.operator_receipt),
                    "operator_receipt_sha256": capture.operator_receipt_sha256,
                    "official_randn_call_count": capture.original_randn_call_count,
                    "external_initial_noise_injection": (
                        capture.external_initial_noise_injection
                    ),
                    "original_return_object_forwarded": (
                        capture.original_return_object_forwarded
                    ),
                    "sampler_initial_noise_artifact": initial_artifact,
                    "source_set_prototype_artifact": prototype_artifact,
                    "temporal_dc_carrier_artifact": carrier_artifact,
                    "candidate_only": True,
                    "score": None,
                    "rank": None,
                    "selected": False,
                }
                arm_rows.append(
                    {**unsigned_arm, "arm_receipt_digest": object_sha256(unsigned_arm)}
                )
            generated_device = {
                arm: value.to(device=device).contiguous()
                for arm, value in generated.items()
            }
            try:
                outputs = native._save_outputs(
                    output_dir=stage,
                    generated=generated_device,
                    vae=vae,
                    bucket_hw=bucket_hw,
                    device=device,
                    save_output_fn=save_output,
                )
            finally:
                generated_device.clear()
            receipt: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "stage": "isolated_exact40_exact81_five_arm_canary",
                "runtime_source": {
                    "revision": args.runtime_source_revision,
                    "archive_sha256": args.runtime_source_archive_sha256,
                    "launcher_sha256": args.launcher_source_sha256,
                },
                "cell_spec": {
                    "path": str(spec_path),
                    "file_sha256": spec_sha,
                    "schema_version": root_spec["schema_version"],
                    "cell": dict(cell),
                    "wrong_source_control": dict(wrong_cell),
                    "selected_before_generation": True,
                    "generated_quality_used_for_selection": False,
                },
                "upstream_authoring_provenance": {
                    "path": str(authoring_path),
                    "file_sha256": authoring_sha,
                    "schema_version": authoring_root["schema_version"],
                    "bank_id": authoring_root["bank_id"],
                    "source_row_index": cell["source_authoring_row_index"],
                    "source_row_sha256": cell["source_authoring_row_sha256"],
                    "source_row": dict(source_authoring_row),
                    "wrong_source_row_index": (
                        cell["wrong_source_authoring_row_index"]
                    ),
                    "wrong_source_row_sha256": (
                        cell["wrong_source_authoring_row_sha256"]
                    ),
                    "wrong_source_row": dict(wrong_authoring_row),
                    "same_action_family_id_verified": True,
                    "selection_precedes_canary_generation": True,
                },
                "model": {
                    "bernini_commit": bernini_revision,
                    "veomni_commit": veomni_revision,
                    "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                    "checkpoint_content": checkpoint_identity,
                    "checkpoint_unchanged": True,
                    "all_parameters_frozen": True,
                    "inference_files": inference_hashes,
                },
                "topology": {
                    "world_size": WORLD_SIZE,
                    "ulysses_size": SP_SIZE,
                    "local_group_role": cell["actor_kind"],
                    "all_rank_tensor_checks_executed": True,
                },
                "source_conditioning": {
                    "correct_source_path": str(source_path),
                    "correct_source_sha256": source_sha,
                    "correct_source_metadata": source_metadata,
                    "wrong_source_path": str(wrong_source_path),
                    "wrong_source_sha256": wrong_sha,
                    "wrong_source_metadata": wrong_metadata,
                    "wrong_source_control_role": wrong_cell["control_role"],
                    "wrong_source_is_pure_identity_control": False,
                    "wrong_source_geometry_confound_present": (
                        expected_geometry_confound
                    ),
                    "bucket_hw": list(bucket_hw),
                    "reference_indices": list(REFERENCE_INDICES),
                    "caller_selection_indices_consumed": True,
                    "operator_received_frame_indices": False,
                    "operator_set_sequence_order_consumed": False,
                    "unordered_pose_occupancy_leakage_possible": True,
                    "correct_reference_encoding": (
                        "four_independent_RGB_frame_to_Wan_VAE_T1_calls"
                    ),
                    "wrong_reference_encoding": (
                        "four_independent_RGB_frame_to_Wan_VAE_T1_calls_after_"
                        "direct_raw_RGB_target_bucket_resize"
                    ),
                    "references_sliced_from_full_video_latent": False,
                    "wrong_source_used_as_model_condition": False,
                    "renderer_conditions_same_across_all_five_arms": True,
                    "broadcasts": broadcasts,
                    "identities": condition_identities,
                    "frame_set_artifacts": frame_set_artifacts,
                },
                "prompt": {
                    "complete_action_caption": cell["action_caption"],
                    "complete_action_caption_utf8_sha256": (
                        cell["action_caption_utf8_sha256"]
                    ),
                    "native_task_prompt": task_prompt,
                    "native_task_prompt_utf8_sha256": hashlib.sha256(
                        task_prompt.encode("utf-8")
                    ).hexdigest(),
                    "same_across_all_five_arms": True,
                },
                "sampling": {
                    "frame_count": FRAME_COUNT,
                    "fps": FPS,
                    "latent_phases": LATENT_PHASES,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "seed": seed,
                    "native_sampling_contract": _sampling_contract(seed),
                    "native_unipc_shift5": True,
                    "native_guidance_replaced": False,
                    "native_scheduler_replaced": False,
                    "sample_method_replaced": False,
                    "sample_one_step_replaced": False,
                    "temporary_mutation_surface": list(TEMPORARY_MUTATION_SURFACE),
                    "official_randn_tensor_called_first_with_unchanged_arguments": True,
                    "same_source_action_seed_scheduler_guidance_across_five_arms": True,
                    "exact40": True,
                    "exact81": True,
                },
                "noise_coordinate": {
                    "operator_callable": (
                        "orderless_source_frame_set_noise."
                        "build_orderless_source_frame_set_noise"
                    ),
                    "inference_binding": (
                        "official_randn_tensor_return_before_native_rearrange_and_"
                        "patch_packing"
                    ),
                    "training_binding_if_separately_authorized": (
                        "canonical_epsilon_after_sampling_before_flow_sigma_"
                        "interpolation_and_patch_packing"
                    ),
                    "same_operator_and_unpacked_clean_epsilon_coordinate": True,
                    "training_integration_executed": False,
                    "alternate_training_noise_builder_authorized": False,
                },
                "official_parent_gaussian": official_artifact,
                "arm_order": list(ARM_ORDER),
                "arms": arm_rows,
                "matched_five_arm_audit": {
                    "one_official_parent_gaussian_across_five_arms": True,
                    "correct_rhos_share_one_frame_multiset_prototype_carrier": True,
                    "wrong_rhos_share_one_frame_multiset_prototype_carrier": True,
                    "correct_and_wrong_carriers_differ": True,
                    "rho0_original_object_forwarded": True,
                    "active_arms_external_non_gaussian_injection": True,
                    "wrong_source_is_carrier_only": True,
                    "wrong_source_is_action_family_matched_source_specificity_control": True,
                    "wrong_source_is_pure_identity_control": False,
                },
                "scientific_gate_contract": {
                    "identity_orbit_decoded_role": "excluded_external_no_go",
                    "identity_orbit_adapter_loaded": False,
                    "identity_adapter_loaded": False,
                    "identity_adapter_used_as_baseline": False,
                    "identity_adapter_used_as_prior": False,
                    "all_gates_evaluated": False,
                    "gate_results_authorized": False,
                    "arm_selection_authorized": False,
                    "gates": [
                        {
                            "name": (
                                "same_rho_correct_vs_wrong_carrier_source_"
                                "specificity"
                            ),
                            "comparisons": [
                                "correct_source_rho005_vs_wrong_source_rho005",
                                "correct_source_rho010_vs_wrong_source_rho010",
                            ],
                            "measures_carrier_itself": True,
                            "evaluated": False,
                        },
                        {
                            "name": (
                                "old_motion_direction_and_order_leakage_"
                                "nonincrease"
                            ),
                            "reference": "official_gaussian",
                            "evaluated": False,
                        },
                        {
                            "name": (
                                "target_action_nonregression_vs_official_gaussian"
                            ),
                            "reference": "official_gaussian",
                            "evaluated": False,
                        },
                        {
                            "name": (
                                "identity_blur_flicker_camera_quality_"
                                "nonregression_vs_official_gaussian"
                            ),
                            "reference": "official_gaussian",
                            "evaluated": False,
                        },
                    ],
                },
                "generated_identities": generated_identities,
                "outputs": outputs,
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "diffusers": diffusers_version,
                    "transformers": transformers_version,
                },
                "interpretation": {
                    "candidate_count": 5,
                    "training_performed": False,
                    "training_harness_executed": False,
                    "validation_only_train_lora_module_imported": True,
                    "trainer_instantiated": False,
                    "critic_loaded": False,
                    "reward_computed": False,
                    "optimizer": None,
                    "backward": False,
                    "best_arm_selected": False,
                    "ranking_performed": False,
                    "target_video": False,
                    "mask": False,
                    "flow": False,
                    "pose": False,
                    "track": False,
                    "trajectory": False,
                    "action_success_evaluated": False,
                    "identity_preservation_evaluated": False,
                    "carrier_is_index_free": False,
                    "operator_is_permutation_invariant_after_member_selection": True,
                    "source_pose_occupancy_leakage_possible": True,
                    "human_wrong_source_geometry_and_initial_pose_confound_acknowledged": (
                        cell["cell_id"] != "human"
                        or (
                            expected_geometry_confound
                            and not wrong_cell["content_match"][
                                "same_initial_pose_family"
                            ]
                        )
                    ),
                    "scientific_claim_authorized": False,
                    "editor_optimizer_authorized": False,
                },
            }
            receipt = _rebase_artifact_paths(
                receipt, old_root=stage, new_root=output_dir
            )
            receipt["receipt_digest"] = object_sha256(receipt)
            _write_receipt(stage / "receipt.json", receipt)
            _commit_output_transaction(staging=stage, final=output_dir)
            print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORING_BANK_ID",
    "AUTHORING_SCHEMA",
    "AUTHORING_SPEC_SHA256",
    "ARM_CARRIER_ROLE",
    "ARM_ORDER",
    "ARM_RHO",
    "FRAME_COUNT",
    "METHOD",
    "REFERENCE_INDICES",
    "SCHEMA_VERSION",
    "SPEC_SCHEMA",
    "SourceSetNoiseCanaryError",
    "SourceSetNoiseCapture",
    "build_parser",
    "canonical_json_bytes",
    "load_authoring_provenance",
    "load_cell_spec",
    "main",
    "object_sha256",
    "validate_cli",
]
