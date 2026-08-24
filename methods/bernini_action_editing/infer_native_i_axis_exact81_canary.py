#!/usr/bin/env python3
"""Run the frozen exact81 Bernini native-I-axis seven-arm canary.

For one pre-registered dog or human cell this runner executes two sealed seeds
and, for each seed, the matched arms ``N-C,N-W,G-C,G-W,G-P,G-D,G-S``.  Every
arm uses the same full correct source video, action caption, target geometry,
official Gaussian, exact40 UniPC schedule and frozen Bernini-R 1.3B weights.

``N`` arms are byte-parity native RV2V observations.  ``G`` arms install the
reversible same-step hook in :mod:`native_i_axis_guidance`; only exact40 steps
33--37 receive ``g=.25`` and steps 38--39 pass the original native scheduler
input object unchanged.  Every selected RGB reference is independently VAE
encoded as ``T=1``.  No target, mask, pose, flow, track, trajectory, trainer,
optimizer, reward or adapter is accepted.

The runner publishes each pre-decode FP32 normalized latent, companion MP4,
official Gaussian, per-branch/per-step call trace and raw-value digests in one
staged directory transaction.  It does not score, rank or select an arm.
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
import infer_orderless_source_frame_set_noise_canary as prior  # noqa: E402
import native_i_axis_guidance as i_axis  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


SCHEMA_VERSION = "bernini-native-i-axis-exact81-canary-receipt-v1"
SPEC_SCHEMA_VERSION = "bernini-native-i-axis-exact81-core2-spec-v1"
METHOD = i_axis.METHOD
FRAME_COUNT = 81
LATENT_PHASES = 21
FPS = 25
NUM_INFERENCE_STEPS = 40
WORLD_SIZE = 4
SP_SIZE = 4
CORRECT_REFERENCE_INDICES = i_axis.CORRECT_REFERENCE_INDICES
PHASE_SHIFT_REFERENCE_INDICES = i_axis.PHASE_SHIFT_REFERENCE_INDICES
ALL_CORRECT_REFERENCE_INDICES = tuple(
    sorted(set(CORRECT_REFERENCE_INDICES + PHASE_SHIFT_REFERENCE_INDICES))
)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class NativeIAxisCanaryError(RuntimeError):
    """Raised before incomplete or ambiguous evidence is published."""


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
        raise NativeIAxisCanaryError(f"receipt is not finite canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _expected_spec_contract() -> Mapping[str, Any]:
    return {
        "method": METHOD,
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "fps": FPS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_mode": "rv2v",
        "raw_velocity_formula": "vG=vN+4.5*g*((vI-v0)-(vVIu-vV))",
        "native_velocity_formula": (
            "vN=v0+1.25*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
        ),
        "apg": False,
        "gate": {
            "active_step_indices": list(i_axis.ACTIVE_STEP_INDICES),
            "active_value": i_axis.ACTIVE_GATE,
            "all_other_step_indices": 0.0,
            "final_exact_native_parity_step_indices": list(
                i_axis.FINAL_NATIVE_PARITY_INDICES
            ),
        },
        "arm_order": list(i_axis.ARM_ORDER),
        "canonical_reference_indices": list(CORRECT_REFERENCE_INDICES),
        "permuted_reference_indices": list(i_axis.PERMUTED_REFERENCE_INDICES),
        "phase_shift_reference_indices": list(PHASE_SHIFT_REFERENCE_INDICES),
        "permutation_control": (
            "same_four_tensor_objects_reordered_across_native_source_id_slots_"
            "not_a_chronological_motion_shuffle"
        ),
        "wrong_source_control": (
            "weak_action_family_matched_source_specificity_diagnostic_not_a_"
            "pure_identity_control"
        ),
        "seed_count_per_cell": 2,
        "topology": (
            "two_concurrent_world4_sp4_groups_on_one_8gpu_node_one_family_per_"
            "group_two_seeds_sequentially"
        ),
        "target_initialization": native.TARGET_INITIALIZATION,
        "same_x_t_t_target_geometry_within_cell_seed_across_seven_arms": True,
        "reference_encoding": (
            "each_selected_RGB_frame_independently_encoded_as_Wan_VAE_T1"
        ),
        "predecode_fp32_latent_required": True,
        "mp4_required": True,
        "per_branch_per_step_call_and_digest_required": True,
        "training": False,
        "optimizer": False,
        "mask": False,
        "pose": False,
        "flow": False,
        "track": False,
        "trajectory": False,
        "selected_before_generation": True,
    }


def _plain_file(value: str | Path, *, label: str) -> Path:
    try:
        return prior._plain_file(value, label=label)
    except Exception as error:
        raise NativeIAxisCanaryError(str(error)) from error


def load_cell_spec(
    path: str | Path,
    *,
    expected_file_sha256: str,
    cell_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, str]:
    spec_path = _plain_file(path, label="native-I core2 spec")
    observed_sha = native.legacy.file_sha256(spec_path)
    if _SHA256.fullmatch(expected_file_sha256 or "") is None or observed_sha != expected_file_sha256:
        raise NativeIAxisCanaryError("native-I spec SHA-256 differs")
    try:
        root = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeIAxisCanaryError("native-I spec is not valid JSON") from error
    if (
        not isinstance(root, dict)
        or set(root) != {"schema_version", "contract", "cells"}
        or root.get("schema_version") != SPEC_SCHEMA_VERSION
        or root.get("contract") != _expected_spec_contract()
    ):
        raise NativeIAxisCanaryError("native-I spec schema/contract differs")
    cells = root.get("cells")
    if (
        not isinstance(cells, list)
        or [row.get("cell_id") for row in cells if isinstance(row, Mapping)]
        != ["dog", "human"]
    ):
        raise NativeIAxisCanaryError("native-I spec must contain dog then human")
    selected = next((row for row in cells if row.get("cell_id") == cell_id), None)
    required = {
        "cell_id", "actor_kind", "source_iid", "source_video",
        "source_video_sha256", "wrong_source_iid", "wrong_source_video",
        "wrong_source_video_sha256", "wrong_source_geometry_confound",
        "wrong_source_pure_identity_control", "action_caption",
        "action_caption_utf8_sha256", "seeds", "selected_before_generation",
    }
    if not isinstance(selected, dict) or set(selected) != required:
        raise NativeIAxisCanaryError("native-I cell schema differs")
    caption = selected["action_caption"]
    if (
        not isinstance(caption, str)
        or not caption.strip()
        or hashlib.sha256(caption.encode("utf-8")).hexdigest()
        != selected["action_caption_utf8_sha256"]
        or not selected["selected_before_generation"]
        or selected["wrong_source_pure_identity_control"] is not False
    ):
        raise NativeIAxisCanaryError("cell caption/control registration differs")
    seeds = selected["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
    ):
        raise NativeIAxisCanaryError("cell must have two distinct presealed seeds")
    for name in ("source_video_sha256", "wrong_source_video_sha256", "action_caption_utf8_sha256"):
        if _SHA256.fullmatch(str(selected[name])) is None:
            raise NativeIAxisCanaryError(f"cell {name} differs")
    return root, selected, spec_path, observed_sha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-spec", required=True)
    parser.add_argument("--expected-cell-spec-sha256", required=True)
    parser.add_argument("--cell-id", choices=("dog", "human"), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--runtime-source-revision", required=True)
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
    return parser


def validate_cli(args: argparse.Namespace) -> Path:
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise NativeIAxisCanaryError("native-I canary is fixed to exact40")
    for name in ("runtime_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise NativeIAxisCanaryError(f"{name} must be full lowercase SHA-1")
    for name in (
        "expected_cell_spec_sha256", "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256", "runtime_source_closure_sha256",
        "launcher_source_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise NativeIAxisCanaryError(f"{name} must be lowercase SHA-256")
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise NativeIAxisCanaryError("unsupported Bernini revision")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise NativeIAxisCanaryError("unsupported VeOmni revision")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise NativeIAxisCanaryError("unsupported checkpoint tree")
    if (
        args.expected_checkpoint_content_manifest_sha256
        != native.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise NativeIAxisCanaryError("unsupported checkpoint content manifest")
    requested = Path(args.output_dir).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or _SAFE_NAME.fullmatch(requested.name) is None
    ):
        raise NativeIAxisCanaryError("output-dir must be absolute, non-root and safe")
    try:
        return native._resolve_fresh_output_dir(requested)
    except Exception as error:
        raise NativeIAxisCanaryError(str(error)) from error


def _references_for_arm(
    arm: str,
    *,
    correct: Mapping[int, Any],
    wrong: Mapping[int, Any],
) -> tuple[Any, ...]:
    contract = i_axis.arm_reference_contract(arm)
    indices = tuple(contract["reference_indices_in_list_order"])
    role = contract["reference_role"]
    if role == "none":
        return ()
    bank = wrong if role == "wrong" else correct
    try:
        result = tuple(bank[index] for index in indices)
    except KeyError as error:
        raise NativeIAxisCanaryError(f"reference bank lacks index {error.args[0]}") from error
    if arm == "G-P":
        canonical = tuple(correct[index] for index in CORRECT_REFERENCE_INDICES)
        expected = i_axis.permute_reference_objects(canonical)
        if any(left is not right for left, right in zip(result, expected)):
            raise NativeIAxisCanaryError("G-P is not the registered object permutation")
    return result


def _sampling_contract(seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract("rv2v", steps=NUM_INFERENCE_STEPS, seed=seed)
    if value["num_frames"] != FRAME_COUNT or value["guidance_mode"] != "rv2v":
        raise NativeIAxisCanaryError("native sampling contract differs")
    return value


def _gather_equal(value: Any, *, world_size: int, label: str) -> Mapping[str, Any]:
    import torch.distributed as dist

    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        raise NativeIAxisCanaryError(f"WORLD4 ranks disagree on {label}")
    return {"all_rank_exact": True, "value": rows[0]}


def _candidate_key(seed: int, arm: str) -> str:
    value = f"seed-{seed}__{arm}"
    if _SAFE_NAME.fullmatch(value) is None:
        raise NativeIAxisCanaryError("candidate artifact key is unsafe")
    return value


def validate_exact40_trace(trace: Mapping[str, Any], *, arm: str) -> Mapping[str, Any]:
    """Fail closed on the per-step execution receipt before artifact publish."""

    if arm not in i_axis.ARM_ORDER or not isinstance(trace, Mapping):
        raise NativeIAxisCanaryError("trace arm/type differs")
    steps = trace.get("steps")
    active_i_forward = arm in {"G-C", "G-W", "G-P", "G-S"}
    expected_forwards = NUM_INFERENCE_STEPS * (5 if active_i_forward else 4)
    expected_active = (
        list(i_axis.ACTIVE_STEP_INDICES) if arm in i_axis.GATED_ARMS else []
    )
    if (
        trace.get("step_count") != NUM_INFERENCE_STEPS
        or trace.get("expected_transformer_forwards") != expected_forwards
        or trace.get("observed_transformer_forwards") != expected_forwards
        or not isinstance(steps, list)
        or len(steps) != NUM_INFERENCE_STEPS
        or [row.get("step_index") for row in steps]
        != list(range(NUM_INFERENCE_STEPS))
        or [row.get("step_index") for row in steps if row.get("gate_active")]
        != expected_active
    ):
        raise NativeIAxisCanaryError("exact40 trace root/gate closure differs")
    expected_branch_names = {
        "none_uncond", "V_uncond", "VI_uncond", "VI_cond"
    }
    if arm in i_axis.GATED_ARMS:
        expected_branch_names.add("I_uncond")
    target_tokens = {row.get("target_tokens") for row in steps}
    for index, row in enumerate(steps):
        calls = row.get("branch_call_counts")
        hashes = row.get("branch_target_raw_sha256")
        if (
            not isinstance(calls, Mapping)
            or set(calls) != expected_branch_names
            or any(
                value != (0 if name == "I_uncond" and arm == "G-D" else 1)
                for name, value in calls.items()
            )
            or not isinstance(hashes, Mapping)
            or set(hashes) != expected_branch_names
            or any(_SHA256.fullmatch(str(value)) is None for value in hashes.values())
            or row.get("transformer_forward_count") != (5 if active_i_forward else 4)
            or row.get("original_scheduler_call_count") != 1
            or row.get("native_formula_exact_parity") is not True
            or _SHA256.fullmatch(str(row.get("native_velocity_raw_sha256"))) is None
            or _SHA256.fullmatch(str(row.get("executed_velocity_raw_sha256"))) is None
        ):
            raise NativeIAxisCanaryError(f"exact40 step {index} call/digest closure differs")
        if index in i_axis.FINAL_NATIVE_PARITY_INDICES and not (
            row.get("final_native_parity") is True
            and row.get("scheduler_received_original_model_output_object") is True
            and row.get("native_velocity_raw_sha256")
            == row.get("executed_velocity_raw_sha256")
        ):
            raise NativeIAxisCanaryError("final two native parity receipt differs")
        if arm in i_axis.NATIVE_ARMS | {"G-D"} and row.get(
            "native_velocity_raw_sha256"
        ) != row.get("executed_velocity_raw_sha256"):
            raise NativeIAxisCanaryError("native/ref-drop control changed velocity")
        if arm == "G-D" and row.get("i_axis_degenerate_alias_none") is not True:
            raise NativeIAxisCanaryError("G-D did not record degenerate I alias")
    if len(target_tokens) != 1 or next(iter(target_tokens), 0) in (None, 0):
        raise NativeIAxisCanaryError("target geometry changed across exact40")
    value = {
        "passed": True,
        "arm": arm,
        "step_count": NUM_INFERENCE_STEPS,
        "expected_transformer_forwards": expected_forwards,
        "active_gate_indices": expected_active,
        "final_native_parity_indices": list(i_axis.FINAL_NATIVE_PARITY_INDICES),
        "target_tokens": next(iter(target_tokens)),
        "all_branch_call_counts_and_digests_present": True,
        "one_original_unipc_call_per_step": True,
    }
    return {**value, "digest": object_sha256(value)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = validate_cli(args)
    root_spec, cell, spec_path, spec_sha = load_cell_spec(
        args.cell_spec,
        expected_file_sha256=args.expected_cell_spec_sha256,
        cell_id=args.cell_id,
    )
    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, label="checkpoint content manifest"
    )
    if native.legacy.file_sha256(checkpoint_manifest) != args.expected_checkpoint_content_manifest_sha256:
        raise NativeIAxisCanaryError("checkpoint manifest SHA-256 differs")
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
        raise NativeIAxisCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % SP_SIZE:
        raise NativeIAxisCanaryError("checkpoint heads are not SP4-compatible")
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
        raise NativeIAxisCanaryError("native negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise NativeIAxisCanaryError("canary requires AUH WORLD4/SP4 ROCm")
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
        if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
            raise NativeIAxisCanaryError(f"checkpoint validation failed: {checkpoint_rows[0]}")
        checkpoint_identity = dict(checkpoint_rows[0]["identity"])

        source_path = _plain_file(cell["source_video"], label="correct source")
        wrong_source_path = _plain_file(cell["wrong_source_video"], label="wrong source")
        source_tensor, source_metadata, source_sha = (
            native.source_audit.prepare_hashed_source_snapshot(source_path)
        )
        if source_sha != cell["source_video_sha256"]:
            raise NativeIAxisCanaryError("correct source SHA-256 differs")
        bucket_hw = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])
        wrong_tensor, wrong_metadata, wrong_sha = prior._prepare_source_snapshot_at_bucket(
            wrong_source_path, bucket_hw=bucket_hw
        )
        if wrong_sha != cell["wrong_source_video_sha256"]:
            raise NativeIAxisCanaryError("wrong source SHA-256 differs")
        observed_wrong_confound = not bool(
            wrong_metadata["native_bucket_matches_target_cell_bucket"]
        )
        if observed_wrong_confound is not bool(cell["wrong_source_geometry_confound"]):
            raise NativeIAxisCanaryError("wrong-source geometry confound differs")

        full_prompt = native.build_task_prompt(
            "rv2v", cell["action_caption"], prompt_cleaner=prompt_clean
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
        )
        positive_ids, positive_mask = native.legacy._tokenize_training_prompt(
            tokenizer, full_prompt
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
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            raise NativeIAxisCanaryError("renderer is not native UniPC shift5")
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
                for index in ALL_CORRECT_REFERENCE_INDICES
            }
            wrong_refs = {
                index: _vae_encode(
                    vae, wrong_pixels[:, :, index : index + 1].contiguous()
                ).contiguous()
                for index in CORRECT_REFERENCE_INDICES
            }
        broadcasts = {
            "full_correct_source": native._broadcast_condition_from_rank_zero(
                full_source_latent, label="full_correct_source", world_size=WORLD_SIZE
            ),
            "correct_refs": {
                str(index): native._broadcast_condition_from_rank_zero(
                    value, label=f"correct_ref_{index}", world_size=WORLD_SIZE
                )
                for index, value in correct_refs.items()
            },
            "wrong_refs": {
                str(index): native._broadcast_condition_from_rank_zero(
                    value, label=f"wrong_ref_{index}", world_size=WORLD_SIZE
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
            raise NativeIAxisCanaryError("source/reference exact81 geometry differs")
        condition_identities = {
            "full_correct_source": native._all_rank_tensor_identity(
                full_source_latent, label="full_correct_source", world_size=WORLD_SIZE
            ),
            "correct_refs": {
                str(index): native._all_rank_tensor_identity(
                    value, label=f"correct_ref_{index}", world_size=WORLD_SIZE
                )
                for index, value in correct_refs.items()
            },
            "wrong_refs": {
                str(index): native._all_rank_tensor_identity(
                    value, label=f"wrong_ref_{index}", world_size=WORLD_SIZE
                )
                for index, value in wrong_refs.items()
            },
            "rank_zero_broadcasts": broadcasts,
        }
        vae.to("cpu")
        del source_tensor, wrong_tensor, correct_pixels, wrong_pixels
        torch.cuda.empty_cache()
        model.to(device)

        diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
        wan_source_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        sampler_contract._validate_scheduler_contract(
            diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
        )

        generated: dict[str, Any] = {}
        generated_identities: dict[str, Any] = {}
        captures: dict[str, Any] = {}
        capture_rank_identities: dict[str, Any] = {}
        traces: dict[str, Any] = {}
        trace_rank_evidence: dict[str, Any] = {}
        candidate_rows: list[Mapping[str, Any]] = []
        for seed in cell["seeds"]:
            for arm in i_axis.ARM_ORDER:
                key = _candidate_key(seed, arm)
                refs = _references_for_arm(
                    arm, correct=correct_refs, wrong=wrong_refs
                )
                hook = i_axis.NativeIAxisGuidanceHook(
                    diffusion,
                    arm=arm,
                    expected_steps=NUM_INFERENCE_STEPS,
                    expected_bernini_commit=bernini_revision,
                    observed_wan_diffusion_sha256=wan_source_sha,
                )
                hook.install()
                try:
                    with torch.inference_mode():
                        endpoint, capture = native._sample_with_native_initial_noise_observer(
                            sample_fn=lambda selected_refs=refs, selected_seed=seed: model.sample(
                                input_ids=positive_ids.to(device),
                                attention_mask=positive_mask.to(device),
                                uncond_input_ids=negative_ids.to(device),
                                uncond_attention_mask=negative_mask.to(device),
                                image_vae_latents=None,
                                multi_video_vae_latents=[full_source_latent],
                                multi_image_vae_latents=(
                                    list(selected_refs) if selected_refs else None
                                ),
                                width=bucket_hw[1],
                                height=bucket_hw[0],
                                device=device,
                                **_sampling_contract(selected_seed),
                            ),
                            wan_diffusion_module=wan_diffusion,
                            expected_shape=video_shape,
                            expected_device=device,
                            expected_seed=seed,
                        )
                finally:
                    hook.restore()
                if (
                    not isinstance(endpoint, torch.Tensor)
                    or endpoint.device != device
                    or endpoint.dtype != torch.float32
                    or endpoint.requires_grad
                    or endpoint.grad_fn is not None
                    or not endpoint.is_contiguous()
                    or tuple(int(item) for item in endpoint.shape) != video_shape
                    or not bool(torch.isfinite(endpoint).all().item())
                ):
                    raise NativeIAxisCanaryError(f"{key} native endpoint differs")
                if hook.sample_calls != 1 or not hook.restored:
                    raise NativeIAxisCanaryError(f"{key} hook lifecycle differs")
                trace = dict(hook.trace)
                trace_gate = validate_exact40_trace(trace, arm=arm)
                generated[key] = endpoint.detach().cpu().contiguous()
                generated_identities[key] = native._all_rank_tensor_identity(
                    generated[key], label=f"generated_{key}", world_size=WORLD_SIZE
                )
                captures[key] = capture
                capture_rank_identities[key] = native._all_rank_tensor_identity(
                    capture.tensor,
                    label=f"official_initial_gaussian_{key}",
                    world_size=WORLD_SIZE,
                )
                traces[key] = trace
                trace_rank_evidence[key] = _gather_equal(
                    trace["trace_digest"], world_size=WORLD_SIZE, label=f"trace_{key}"
                )
                ref_contract = i_axis.arm_reference_contract(arm)
                ref_raw = []
                bank_identity = (
                    condition_identities["wrong_refs"]
                    if ref_contract["reference_role"] == "wrong"
                    else condition_identities["correct_refs"]
                )
                if ref_contract["reference_role"] != "none":
                    ref_raw = [
                        bank_identity[str(index)]["identity"]["raw_storage_sha256"]
                        for index in ref_contract["reference_indices_in_list_order"]
                    ]
                unsigned_candidate = {
                    "candidate_key": key,
                    "seed": seed,
                    "arm": arm,
                    "arm_contract": dict(ref_contract),
                    "reference_raw_storage_sha256_in_list_order": ref_raw,
                    "trace_digest": trace["trace_digest"],
                    "exact40_trace_gate": trace_gate,
                    "official_initial_gaussian_raw_value_sha256": capture.raw_value_sha256,
                    "generated_identity": generated_identities[key],
                    "score": None,
                    "rank": None,
                    "selected": False,
                }
                candidate_rows.append(
                    {
                        **unsigned_candidate,
                        "candidate_receipt_digest": object_sha256(unsigned_candidate),
                    }
                )
                del endpoint
                torch.cuda.empty_cache()

        if len(candidate_rows) != 2 * len(i_axis.ARM_ORDER):
            raise NativeIAxisCanaryError("candidate count differs")
        for seed in cell["seeds"]:
            seed_hashes = {
                captures[_candidate_key(seed, arm)].raw_value_sha256
                for arm in i_axis.ARM_ORDER
            }
            if len(seed_hashes) != 1:
                raise NativeIAxisCanaryError("same-seed arms lost common Gaussian")
        seed_parent_hashes = {
            captures[_candidate_key(seed, "N-C")].raw_value_sha256
            for seed in cell["seeds"]
        }
        if len(seed_parent_hashes) != 2:
            raise NativeIAxisCanaryError("two sealed seeds produced the same Gaussian")
        for seed in cell["seeds"]:
            canonical_row = next(
                row for row in candidate_rows
                if row["seed"] == seed and row["arm"] == "G-C"
            )
            permuted_row = next(
                row for row in candidate_rows
                if row["seed"] == seed and row["arm"] == "G-P"
            )
            if sorted(canonical_row["reference_raw_storage_sha256_in_list_order"]) != sorted(
                permuted_row["reference_raw_storage_sha256_in_list_order"]
            ):
                raise NativeIAxisCanaryError("G-P changed the byte-exact ref multiset")

        freeze_after = native.source_audit.model_freeze_certificate(model)
        if freeze_after != freeze_before or any(parameter.requires_grad for parameter in model.parameters()):
            raise NativeIAxisCanaryError("frozen model certificate changed")
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
            not isinstance(after_rows[0], Mapping)
            or after_rows[0].get("ok") is not True
            or after_rows[0].get("identity") != checkpoint_identity
        ):
            raise NativeIAxisCanaryError("checkpoint changed during canary")

        if distributed.rank == 0:
            stage = prior._output_staging_directory(output_dir)
            source_artifact = native._save_normalized_clean_latent_atomically(
                stage / "source.normalized-clean-latent.safetensors",
                full_source_latent,
                artifact_role="source_video_condition",
            )
            correct_ref_artifacts = {
                str(index): prior._save_tensor_artifact(
                    stage / f"correct-reference-{index:03d}.safetensors",
                    value,
                    key="reference_latent",
                    metadata={
                        "coordinate": "independent_RGB_frame_to_Wan_VAE_T1",
                        "source_role": "correct",
                        "frame_index": str(index),
                    },
                )
                for index, value in correct_refs.items()
            }
            wrong_ref_artifacts = {
                str(index): prior._save_tensor_artifact(
                    stage / f"wrong-reference-{index:03d}.safetensors",
                    value,
                    key="reference_latent",
                    metadata={
                        "coordinate": "independent_RGB_frame_to_Wan_VAE_T1",
                        "source_role": "wrong_weak_diagnostic",
                        "frame_index": str(index),
                    },
                )
                for index, value in wrong_refs.items()
            }
            initial_noise_artifacts = {
                key: native._save_initial_noise_atomically(
                    stage / f"{key}.official-initial-gaussian.safetensors",
                    captures[key],
                    all_rank_identity=capture_rank_identities[key],
                )
                for key in generated
            }
            generated_device = {
                key: value.to(device=device).contiguous()
                for key, value in generated.items()
            }
            outputs = native._save_outputs(
                output_dir=stage,
                generated=generated_device,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
            unsigned_receipt: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "stage": "frozen_exact40_exact81_two_seed_seven_arm_canary",
                "cell_spec": {
                    "path": str(spec_path),
                    "file_sha256": spec_sha,
                    "schema_version": root_spec["schema_version"],
                    "contract": root_spec["contract"],
                    "cell": cell,
                },
                "runtime_source": {
                    "revision": args.runtime_source_revision,
                    "closure_sha256": args.runtime_source_closure_sha256,
                    "launcher_sha256": args.launcher_source_sha256,
                },
                "pinned_sources": {
                    "bernini_commit": bernini_revision,
                    "veomni_commit": veomni_revision,
                    "wan_diffusion_path": str(Path(wan_diffusion.__file__).resolve()),
                    "wan_diffusion_sha256": wan_source_sha,
                    "bernini_inference_files": inference_hashes,
                },
                "checkpoint": {
                    "path": str(checkpoint),
                    "tree_sha256": args.expected_checkpoint_tree_sha256,
                    "content_before_and_after": checkpoint_identity,
                    "unchanged": True,
                },
                "source": {
                    "path": str(source_path),
                    "sha256": source_sha,
                    "metadata": source_metadata,
                    "normalized_clean_latent_artifact": source_artifact,
                    "correct_reference_artifacts": correct_ref_artifacts,
                    "condition_identities": condition_identities,
                },
                "wrong_source": {
                    "path": str(wrong_source_path),
                    "sha256": wrong_sha,
                    "metadata": wrong_metadata,
                    "reference_artifacts": wrong_ref_artifacts,
                    "weak_diagnostic_only": True,
                    "pure_identity_control": False,
                    "geometry_confound_present": bool(cell["wrong_source_geometry_confound"]),
                    "full_video_conditioned": False,
                },
                "prompt": {
                    "action_caption_utf8_sha256": cell["action_caption_utf8_sha256"],
                    "full_native_prompt_utf8_sha256": hashlib.sha256(
                        full_prompt.encode("utf-8")
                    ).hexdigest(),
                    "same_across_all_arms_and_seeds": True,
                },
                "sampling": {
                    "exact40": True,
                    "exact81": True,
                    "frame_count": FRAME_COUNT,
                    "latent_phases": LATENT_PHASES,
                    "fps": FPS,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "seeds": list(cell["seeds"]),
                    "arm_order": list(i_axis.ARM_ORDER),
                    "same_official_gaussian_within_seed": True,
                    "same_x_t_t_target_geometry_within_seed": True,
                    "hook_contract": i_axis.hook_contract(),
                },
                "candidates": candidate_rows,
                "traces": traces,
                "trace_all_rank_evidence": trace_rank_evidence,
                "generated_identities": generated_identities,
                "initial_noise_artifacts": initial_noise_artifacts,
                "outputs": outputs,
                "frozen_model": freeze_after,
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                },
                "interpretation": {
                    "candidate_count": len(candidate_rows),
                    "training_performed": False,
                    "trainer_instantiated": False,
                    "optimizer": None,
                    "backward": False,
                    "model_weights_written": False,
                    "adapter_loaded": False,
                    "apg": False,
                    "cache_or_cross_trajectory_replay": False,
                    "target_video": False,
                    "mask": False,
                    "pose": False,
                    "flow": False,
                    "track": False,
                    "trajectory": False,
                    "score_computed": False,
                    "ranking_performed": False,
                    "best_arm_selected": False,
                    "G_P_is_chronological_motion_shuffle": False,
                    "G_P_is_byte_exact_reference_object_multiset_permutation": True,
                    "wrong_source_is_weak_diagnostic": True,
                    "wrong_source_is_pure_identity_control": False,
                    "action_success_evaluated": False,
                    "identity_success_evaluated": False,
                    "scientific_claim_authorized": False,
                },
            }
            unsigned_receipt = prior._rebase_artifact_paths(
                unsigned_receipt, old_root=stage, new_root=output_dir
            )
            receipt = {
                **unsigned_receipt,
                "receipt_digest": object_sha256(unsigned_receipt),
            }
            prior._write_receipt(stage / "receipt.json", receipt)
            prior._commit_output_transaction(staging=stage, final=output_dir)
            print(canonical_json_bytes(receipt).decode("ascii"), flush=True)

        dist.barrier()
        del full_source_latent, correct_refs, wrong_refs, generated, captures
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_CORRECT_REFERENCE_INDICES",
    "NativeIAxisCanaryError",
    "SPEC_SCHEMA_VERSION",
    "_expected_spec_contract",
    "_references_for_arm",
    "build_parser",
    "load_cell_spec",
    "main",
    "validate_exact40_trace",
    "validate_cli",
]
