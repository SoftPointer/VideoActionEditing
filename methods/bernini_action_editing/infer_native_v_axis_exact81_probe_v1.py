#!/usr/bin/env python3
"""Run the frozen exact81 Bernini native full-video V-axis causal probe.

For each sealed dog/human cell and seed, this runner decodes three matched
arms in registry order: ``V-on``, ``V-off`` and ``wrong-V``.  The source,
instruction, four independently encoded correct-source image references,
official Gaussian, exact40 UniPC scheduler, target geometry and frozen model
are held fixed within a seed.  ``V-off`` changes only the coefficient of the
standalone ``(vV-v0)`` term from 1.25 to zero.  ``wrong-V`` replaces only the
full-video condition; the correct image references and instruction remain.

Every arm publishes its complete 81-frame MP4, pre-decode FP32 normalized
latent, observed initial Gaussian, exact40 branch trace and explicit receipt.
No feature scorer, reward, ranking, selection, target video or training is
available in this program.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_orderless_source_frame_set_noise_canary as prior  # noqa: E402
import native_v_axis_guidance_v1 as v_axis  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


SCHEMA_VERSION = "bernini-native-v-axis-exact81-probe-receipt-v1"
SPEC_SCHEMA_VERSION = "bernini-native-v-axis-exact81-core2-spec-v1"
METHOD = v_axis.METHOD
FRAME_COUNT = 81
LATENT_PHASES = 21
FPS = 25
NUM_INFERENCE_STEPS = 40
WORLD_SIZE = 4
SP_SIZE = 4
REFERENCE_INDICES = (0, 27, 53, 80)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class NativeVAxisProbeError(RuntimeError):
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
        raise NativeVAxisProbeError(f"receipt is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@contextmanager
def _serialized_host_checkpoint_load() -> Any:
    """Allow only one WORLD4 rank to deserialize model weights at a time."""

    value = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK")
    if value is None:
        raise NativeVAxisProbeError("NATIVE_V_AXIS_LOAD_LOCK is required on 64GiB")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise NativeVAxisProbeError("serialized checkpoint-load lock differs")
    with path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _trim_host_allocator() -> bool:
    """Return unused PyTorch deserialization arenas to the Slurm cgroup."""

    import ctypes

    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    malloc_trim = libc.malloc_trim
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)
    return True


def _strong_model_freeze_certificate(model: Any) -> Mapping[str, Any]:
    """Hash the exact frozen model state once on rank zero."""

    import torch

    if not isinstance(model, torch.nn.Module) or bool(model.training):
        raise NativeVAxisProbeError("frozen model must be one eval module")
    modules = [
        (str(name), f"{type(module).__module__}.{type(module).__qualname__}")
        for name, module in model.named_modules()
    ]
    if len({name for name, _ in modules}) != len(modules):
        raise NativeVAxisProbeError("frozen model module names repeat")
    if any(
        "lora" in name.lower() or "lora" in class_name.lower()
        for name, class_name in modules
    ):
        raise NativeVAxisProbeError("frozen model contains adapter modules")
    named_state = [
        *(("parameter", name, value) for name, value in model.named_parameters()),
        *(("buffer", name, value) for name, value in model.named_buffers()),
    ]
    names = [f"{kind}.{name}" for kind, name, _ in named_state]
    if len(set(names)) != len(names):
        raise NativeVAxisProbeError("frozen model state names repeat")
    content = hashlib.sha256()
    rows: list[Mapping[str, Any]] = []
    counts = {"parameter": 0, "buffer": 0}
    byte_counts = {"parameter": 0, "buffer": 0}
    for kind, name, value in named_state:
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise NativeVAxisProbeError("frozen model state is not materialized")
        if kind == "parameter" and (value.requires_grad or value.grad is not None):
            raise NativeVAxisProbeError("frozen model parameter is trainable")
        detached = value.detach().to(device="cpu").contiguous()
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        row = {
            "kind": kind,
            "name": str(name),
            "shape": [int(item) for item in detached.shape],
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
            "byte_count": len(raw),
            "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
        }
        content.update(canonical_json_bytes(row))
        content.update(b"\0")
        content.update(raw)
        rows.append(row)
        counts[kind] += 1
        byte_counts[kind] += len(raw)
    return {
        "base_frozen": True,
        "model_eval": True,
        "adapter_modules_absent": True,
        "module_count": len(modules),
        "module_topology_sha256": object_sha256(modules),
        "parameter_tensor_count": counts["parameter"],
        "parameter_byte_count": byte_counts["parameter"],
        "buffer_tensor_count": counts["buffer"],
        "buffer_byte_count": byte_counts["buffer"],
        "state_metadata_sha256": object_sha256(rows),
        "state_content_sha256": content.hexdigest(),
        "exact_parameter_and_buffer_bytes_hashed": True,
        "device_and_storage_address_excluded": True,
    }


def _rank_zero_strong_model_freeze_certificate(
    model: Any, *, rank: int
) -> Mapping[str, Any]:
    import torch.distributed as dist

    payload: list[Any] = [None]
    if rank == 0:
        try:
            payload[0] = {
                "ok": True,
                "certificate": _strong_model_freeze_certificate(model),
            }
        except Exception as error:
            payload[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        raise NativeVAxisProbeError(f"rank-zero freeze certificate failed: {result!r}")
    certificate = result.get("certificate")
    if not isinstance(certificate, Mapping):
        raise NativeVAxisProbeError("rank-zero freeze certificate differs")
    return dict(certificate)


def _model_mutation_guard(model: Any) -> Mapping[str, Any]:
    """Detect in-place changes without copying model bytes back to host."""

    import torch

    if not isinstance(model, torch.nn.Module) or bool(model.training):
        raise NativeVAxisProbeError("mutation-guard model must be frozen eval")
    rows: list[Mapping[str, Any]] = []
    state = [
        *(("parameter", name, value) for name, value in model.named_parameters()),
        *(("buffer", name, value) for name, value in model.named_buffers()),
    ]
    for kind, name, value in state:
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise NativeVAxisProbeError("mutation-guard state is not materialized")
        if kind == "parameter" and (value.requires_grad or value.grad is not None):
            raise NativeVAxisProbeError("mutation-guard parameter is trainable")
        rows.append(
            {
                "kind": kind,
                "name": str(name),
                "shape": [int(item) for item in value.shape],
                "stride": [int(item) for item in value.stride()],
                "dtype": str(value.dtype),
                "device": str(value.device),
                "data_ptr": int(value.data_ptr()),
                "storage_offset": int(value.storage_offset()),
                "version": int(value._version),
                "requires_grad": bool(value.requires_grad),
                "gradient_absent": value.grad is None,
            }
        )
    return {
        "schema_version": "bernini-model-mutation-guard-v1",
        "state_tensor_count": len(rows),
        "process_local_storage_and_version_sha256": object_sha256(rows),
        "no_parameter_or_buffer_bytes_copied_to_host": True,
    }


def resource_lifetime_contract() -> Mapping[str, Any]:
    """Declarative 60-GiB execution contract, also copied into receipts."""

    return {
        "slurm_child_memory_bytes": 60 * 1024**3,
        "world_size": WORLD_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "rank_serialized_checkpoint_deserialize": True,
        "model_moved_to_rank_device_before_load_lock_release": True,
        "host_allocator_trim_after_each_rank_load": True,
        "prompt_embeddings_encoded_once_per_process": True,
        "text_encoder_retired_before_vae_and_sampling": True,
        "vae_instantiated_on_rank_zero_only": True,
        "correct_wrong_full_latents_encoded_on_rank_zero_only": True,
        "correct_reference_latents_encoded_on_rank_zero_only": True,
        "condition_latents_broadcast_rank_zero_to_all_ranks": True,
        "sampling_model_destroyed_without_cpu_offload_before_rank_zero_decode": True,
        "dog_human_process_trees_serial": True,
    }


def _expected_spec_contract() -> Mapping[str, Any]:
    return {
        "method": METHOD,
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "fps": FPS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_mode": "rv2v",
        "native_velocity_formula": (
            "vN=v0+1.25*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
        ),
        "v_off_velocity_formula": (
            "vOff=v0+0.0*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
        ),
        "v_vi_u_minus_v_v_term_retained_in_v_off": True,
        "arm_order": list(v_axis.ARM_ORDER),
        "canonical_correct_reference_indices": list(REFERENCE_INDICES),
        "wrong_v_replaces_full_video_condition_only": True,
        "wrong_v_keeps_correct_image_references_and_text": True,
        "same_source_instruction_seed_scheduler_within_matched_arm_set": True,
        "seed_count_per_cell": 2,
        "topology": (
            "one_world4_sp4_group_reused_serially_for_dog_then_human_on_one_"
            "retained_8gpu_64G_holder_to_bound_host_residency"
        ),
        "resource_lifetime_contract": resource_lifetime_contract(),
        "target_initialization": native.TARGET_INITIALIZATION,
        "reference_encoding": (
            "each_selected_correct_source_RGB_frame_independently_encoded_as_Wan_VAE_T1"
        ),
        "predecode_fp32_latent_required": True,
        "mp4_exact81_required": True,
        "per_branch_per_step_call_and_digest_required": True,
        "distinct_from_native_i_axis_131497": True,
        "training": False,
        "optimizer": False,
        "feature_scorer": False,
        "reward": False,
        "ranking": False,
        "selection": False,
        "selected_before_generation": True,
    }


def _plain_file(value: str | Path, *, label: str) -> Path:
    try:
        return prior._plain_file(value, label=label)
    except Exception as error:
        raise NativeVAxisProbeError(str(error)) from error


def load_cell_spec(
    path: str | Path,
    *,
    expected_file_sha256: str,
    cell_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, str]:
    spec_path = _plain_file(path, label="native V-axis core2 spec")
    observed_sha = native.legacy.file_sha256(spec_path)
    if _SHA256.fullmatch(expected_file_sha256 or "") is None:
        raise NativeVAxisProbeError("native V-axis spec SHA-256 format differs")
    if observed_sha != expected_file_sha256:
        raise NativeVAxisProbeError("native V-axis spec SHA-256 differs")
    try:
        root = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeVAxisProbeError("native V-axis spec is not valid JSON") from error
    if (
        not isinstance(root, dict)
        or set(root) != {"schema_version", "contract", "cells"}
        or root.get("schema_version") != SPEC_SCHEMA_VERSION
        or root.get("contract") != _expected_spec_contract()
    ):
        raise NativeVAxisProbeError("native V-axis spec schema/contract differs")
    cells = root.get("cells")
    if (
        not isinstance(cells, list)
        or [row.get("cell_id") for row in cells if isinstance(row, Mapping)]
        != ["dog", "human"]
    ):
        raise NativeVAxisProbeError("native V-axis spec must contain dog then human")
    selected = next((row for row in cells if row.get("cell_id") == cell_id), None)
    required = {
        "cell_id", "actor_kind", "source_iid", "source_video",
        "source_video_sha256", "wrong_source_iid", "wrong_source_video",
        "wrong_source_video_sha256", "wrong_source_geometry_confound",
        "wrong_source_pure_identity_control", "action_caption",
        "action_caption_utf8_sha256", "seeds", "selected_before_generation",
    }
    if not isinstance(selected, dict) or set(selected) != required:
        raise NativeVAxisProbeError("native V-axis cell schema differs")
    caption = selected["action_caption"]
    if (
        not isinstance(caption, str)
        or not caption.strip()
        or hashlib.sha256(caption.encode("utf-8")).hexdigest()
        != selected["action_caption_utf8_sha256"]
        or selected["selected_before_generation"] is not True
        or selected["wrong_source_pure_identity_control"] is not False
    ):
        raise NativeVAxisProbeError("cell prompt/control registration differs")
    seeds = selected["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
    ):
        raise NativeVAxisProbeError("cell must have two distinct sealed seeds")
    for name in (
        "source_video_sha256", "wrong_source_video_sha256",
        "action_caption_utf8_sha256",
    ):
        if _SHA256.fullmatch(str(selected[name])) is None:
            raise NativeVAxisProbeError(f"cell {name} differs")
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
        raise NativeVAxisProbeError("native V-axis probe is fixed to exact40")
    for name in (
        "runtime_source_revision", "expected_bernini_commit", "expected_veomni_commit"
    ):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise NativeVAxisProbeError(f"{name} must be full lowercase SHA-1")
    for name in (
        "expected_cell_spec_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "runtime_source_closure_sha256",
        "launcher_source_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise NativeVAxisProbeError(f"{name} must be lowercase SHA-256")
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise NativeVAxisProbeError("unsupported Bernini revision")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise NativeVAxisProbeError("unsupported VeOmni revision")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise NativeVAxisProbeError("unsupported checkpoint tree")
    if (
        args.expected_checkpoint_content_manifest_sha256
        != native.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise NativeVAxisProbeError("unsupported checkpoint content manifest")
    requested = Path(args.output_dir).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or _SAFE_NAME.fullmatch(requested.name) is None
    ):
        raise NativeVAxisProbeError("output-dir must be absolute, non-root and safe")
    try:
        return native._resolve_fresh_output_dir(requested)
    except Exception as error:
        raise NativeVAxisProbeError(str(error)) from error


def _sampling_contract(seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract(
        "rv2v", steps=NUM_INFERENCE_STEPS, seed=seed
    )
    if value["num_frames"] != FRAME_COUNT or value["guidance_mode"] != "rv2v":
        raise NativeVAxisProbeError("native sampling contract differs")
    return value


def _candidate_key(seed: int, arm: str) -> str:
    value = f"seed-{seed}__{arm}"
    if _SAFE_NAME.fullmatch(value) is None:
        raise NativeVAxisProbeError("candidate artifact key is unsafe")
    return value


def _gather_equal(value: Any, *, world_size: int, label: str) -> Mapping[str, Any]:
    import torch.distributed as dist

    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        raise NativeVAxisProbeError(f"WORLD4 ranks disagree on {label}")
    return {"all_rank_exact": True, "value": rows[0]}


def validate_exact40_trace(trace: Mapping[str, Any], *, arm: str) -> Mapping[str, Any]:
    """Fail closed on every branch call, coefficient and scheduler handoff."""

    if arm not in v_axis.ARM_ORDER or not isinstance(trace, Mapping):
        raise NativeVAxisProbeError("trace arm/type differs")
    steps = trace.get("steps")
    expected_omega = float(v_axis.arm_contract(arm)["omega_video"])
    names = {"none_uncond", "V_uncond", "VI_uncond", "VI_cond"}
    if (
        trace.get("step_count") != NUM_INFERENCE_STEPS
        or trace.get("expected_transformer_forwards") != 4 * NUM_INFERENCE_STEPS
        or trace.get("observed_transformer_forwards") != 4 * NUM_INFERENCE_STEPS
        or not isinstance(steps, list)
        or len(steps) != NUM_INFERENCE_STEPS
        or [row.get("step_index") for row in steps]
        != list(range(NUM_INFERENCE_STEPS))
    ):
        raise NativeVAxisProbeError("exact40 trace root differs")
    target_tokens = {row.get("target_tokens") for row in steps}
    for index, row in enumerate(steps):
        hashes = row.get("branch_target_raw_sha256")
        if (
            row.get("omega_video_hex") != expected_omega.hex()
            or row.get("standalone_v_axis_active") is not (expected_omega != 0.0)
            or row.get("branch_call_counts") != {name: 1 for name in (
                "none_uncond", "V_uncond", "VI_uncond", "VI_cond"
            )}
            or not isinstance(hashes, Mapping)
            or set(hashes) != names
            or any(_SHA256.fullmatch(str(value)) is None for value in hashes.values())
            or row.get("transformer_forward_count") != 4
            or row.get("original_scheduler_call_count") != 1
            or row.get("native_formula_exact_parity") is not True
            or row.get("v_vi_u_minus_v_v_term_retained") is not True
            or _SHA256.fullmatch(str(row.get("native_velocity_raw_sha256"))) is None
            or _SHA256.fullmatch(str(row.get("executed_velocity_raw_sha256"))) is None
        ):
            raise NativeVAxisProbeError(f"exact40 step {index} closure differs")
        native_pointer = row.get("scheduler_received_original_model_output_object")
        same_digest = (
            row.get("native_velocity_raw_sha256")
            == row.get("executed_velocity_raw_sha256")
        )
        if arm == "V-off":
            if native_pointer is not False:
                raise NativeVAxisProbeError("V-off did not replace scheduler output")
        elif native_pointer is not True or not same_digest:
            raise NativeVAxisProbeError("native-coefficient arm lost exact parity")
    if len(target_tokens) != 1 or next(iter(target_tokens), 0) in (None, 0):
        raise NativeVAxisProbeError("target geometry changed across exact40")
    unsigned = {
        "passed": True,
        "arm": arm,
        "step_count": NUM_INFERENCE_STEPS,
        "omega_video": expected_omega,
        "target_tokens": next(iter(target_tokens)),
        "four_native_branch_calls_per_step": True,
        "one_original_unipc_call_per_step": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


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
    if (
        native.legacy.file_sha256(checkpoint_manifest)
        != args.expected_checkpoint_content_manifest_sha256
    ):
        raise NativeVAxisProbeError("checkpoint manifest SHA-256 differs")
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
        raise NativeVAxisProbeError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % SP_SIZE:
        raise NativeVAxisProbeError("checkpoint heads are not SP4-compatible")
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
        raise NativeVAxisProbeError("native negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise NativeVAxisProbeError("probe requires AUH WORLD4/SP4 ROCm")
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
        if (
            not isinstance(checkpoint_rows[0], Mapping)
            or checkpoint_rows[0].get("ok") is not True
        ):
            raise NativeVAxisProbeError(
                f"checkpoint validation failed: {checkpoint_rows[0]}"
            )
        checkpoint_identity = dict(checkpoint_rows[0]["identity"])

        source_path = _plain_file(cell["source_video"], label="correct source")
        wrong_source_path = _plain_file(cell["wrong_source_video"], label="wrong source")
        source_payload: list[Any] = [None]
        source_tensor = None
        wrong_tensor = None
        if distributed.rank == 0:
            try:
                source_tensor, source_metadata, source_sha = (
                    native.source_audit.prepare_hashed_source_snapshot(source_path)
                )
                if source_sha != cell["source_video_sha256"]:
                    raise NativeVAxisProbeError("correct source SHA-256 differs")
                bucket_hw = tuple(
                    int(item)
                    for item in source_metadata["source_derived_bucket_hw"]
                )
                wrong_tensor, wrong_metadata, wrong_sha = (
                    prior._prepare_source_snapshot_at_bucket(
                        wrong_source_path, bucket_hw=bucket_hw
                    )
                )
                if wrong_sha != cell["wrong_source_video_sha256"]:
                    raise NativeVAxisProbeError("wrong source SHA-256 differs")
                observed_wrong_confound = not bool(
                    wrong_metadata["native_bucket_matches_target_cell_bucket"]
                )
                if observed_wrong_confound is not bool(
                    cell["wrong_source_geometry_confound"]
                ):
                    raise NativeVAxisProbeError(
                        "wrong-source geometry confound differs"
                    )
                source_payload[0] = {
                    "ok": True,
                    "source_metadata": source_metadata,
                    "source_sha": source_sha,
                    "wrong_metadata": wrong_metadata,
                    "wrong_sha": wrong_sha,
                    "bucket_hw": list(bucket_hw),
                }
            except Exception as error:
                source_payload[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(source_payload, src=0)
        source_record = source_payload[0]
        if not isinstance(source_record, Mapping) or source_record.get("ok") is not True:
            raise NativeVAxisProbeError(f"rank-zero source preparation failed: {source_record!r}")
        source_metadata = dict(source_record["source_metadata"])
        source_sha = str(source_record["source_sha"])
        wrong_metadata = dict(source_record["wrong_metadata"])
        wrong_sha = str(source_record["wrong_sha"])
        bucket_hw = tuple(int(item) for item in source_record["bucket_hw"])

        full_prompt = native.build_task_prompt(
            "rv2v", cell["action_caption"], prompt_cleaner=prompt_clean
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            subfolder="tokenizer",
            **native.legacy.tokenizer_load_kwargs(),
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
        native.legacy.trainer.validate_renderer_config_mapping(
            config.to_dict(), checkpoint
        )
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            raise NativeVAxisProbeError("renderer is not native UniPC shift5")
        with _serialized_host_checkpoint_load():
            model = BerniniRendererModel(config)
            model.eval().requires_grad_(False)
            model.to(device)
            host_allocator_trim_invoked = _trim_host_allocator()
        full_freeze_certificate = _rank_zero_strong_model_freeze_certificate(
            model, rank=distributed.rank
        )
        model.t5_text_encoder.to(device)
        full_guard_before_prompt = _model_mutation_guard(model)
        with torch.inference_mode():
            positive_embeds = model.encode_prompt(
                positive_ids.to(device), positive_mask.to(device)
            ).detach()
            negative_embeds = model.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach()
        full_guard_after_prompt = _model_mutation_guard(model)
        if full_guard_after_prompt != full_guard_before_prompt:
            raise NativeVAxisProbeError("frozen model changed during prompt encoding")
        retired_text_encoder = model.t5_text_encoder
        model.t5_text_encoder = None
        del retired_text_encoder, tokenizer
        del positive_ids, positive_mask, negative_ids, negative_mask
        _trim_host_allocator()
        torch.cuda.empty_cache()

        # The 64-GiB holder permits one VAE only.  Rank zero encodes both full
        # videos and all four correct references, then broadcasts exact FP32
        # latents; nonzero ranks never instantiate AutoencoderKLWan.
        vae = None
        geometry_payload: list[Any] = [None]
        if distributed.rank == 0:
            if source_tensor is None or wrong_tensor is None:
                raise NativeVAxisProbeError("rank-zero source tensor lifetime differs")
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
                full_correct_latent = _vae_encode(vae, correct_pixels).contiguous()
                full_wrong_latent = _vae_encode(vae, wrong_pixels).contiguous()
                correct_refs = {
                    index: _vae_encode(
                        vae,
                        correct_pixels[:, :, index : index + 1].contiguous(),
                    ).contiguous()
                    for index in REFERENCE_INDICES
                }
            geometry = native._latent_geometry_receipt(
                bucket_hw=bucket_hw, z_dim=int(vae.config.z_dim)
            )
            video_shape = tuple(int(item) for item in geometry["video_latent_shape"])
            ref_shape = tuple(int(item) for item in geometry["reference_latent_shape"])
            if (
                tuple(full_correct_latent.shape) != video_shape
                or tuple(full_wrong_latent.shape) != video_shape
                or video_shape[:3] != (1, 16, LATENT_PHASES)
                or any(tuple(value.shape) != ref_shape for value in correct_refs.values())
            ):
                raise NativeVAxisProbeError("source/reference exact81 geometry differs")
            geometry_payload[0] = {
                "video_shape": list(video_shape),
                "reference_shape": list(ref_shape),
                "geometry": geometry,
            }
            del correct_pixels, wrong_pixels, source_tensor, wrong_tensor
            vae.to("cpu")
            _trim_host_allocator()
            torch.cuda.empty_cache()
        dist.broadcast_object_list(geometry_payload, src=0)
        geometry_record = geometry_payload[0]
        if not isinstance(geometry_record, Mapping):
            raise NativeVAxisProbeError("rank-zero latent geometry differs")
        video_shape = tuple(int(item) for item in geometry_record["video_shape"])
        ref_shape = tuple(int(item) for item in geometry_record["reference_shape"])
        geometry = dict(geometry_record["geometry"])
        if distributed.rank != 0:
            full_correct_latent = torch.empty(
                video_shape, device=device, dtype=torch.float32
            )
            full_wrong_latent = torch.empty(
                video_shape, device=device, dtype=torch.float32
            )
            correct_refs = {
                index: torch.empty(ref_shape, device=device, dtype=torch.float32)
                for index in REFERENCE_INDICES
            }
        broadcasts = {
            "full_correct_video": native._broadcast_condition_from_rank_zero(
                full_correct_latent,
                label="full_correct_video",
                world_size=WORLD_SIZE,
            ),
            "full_wrong_video": native._broadcast_condition_from_rank_zero(
                full_wrong_latent,
                label="full_wrong_video",
                world_size=WORLD_SIZE,
            ),
            "correct_references": {
                str(index): native._broadcast_condition_from_rank_zero(
                    value,
                    label=f"correct_reference_{index}",
                    world_size=WORLD_SIZE,
                )
                for index, value in correct_refs.items()
            },
        }
        if (
            tuple(full_correct_latent.shape) != video_shape
            or tuple(full_wrong_latent.shape) != video_shape
            or video_shape[:3] != (1, 16, LATENT_PHASES)
            or any(tuple(value.shape) != ref_shape for value in correct_refs.values())
        ):
            raise NativeVAxisProbeError("source/reference exact81 geometry differs")
        condition_identities = {
            "full_correct_video": native._all_rank_tensor_identity(
                full_correct_latent,
                label="full_correct_video",
                world_size=WORLD_SIZE,
            ),
            "full_wrong_video": native._all_rank_tensor_identity(
                full_wrong_latent,
                label="full_wrong_video",
                world_size=WORLD_SIZE,
            ),
            "correct_references": {
                str(index): native._all_rank_tensor_identity(
                    value,
                    label=f"correct_reference_{index}",
                    world_size=WORLD_SIZE,
                )
                for index, value in correct_refs.items()
            },
            "rank_zero_broadcasts": broadcasts,
        }

        diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
        wan_source_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        sampler_contract._validate_scheduler_contract(
            diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
        )
        sampling_guard_before = _model_mutation_guard(model)

        generated: dict[str, Any] = {}
        generated_identities: dict[str, Any] = {}
        captures: dict[str, Any] = {}
        capture_rank_identities: dict[str, Any] = {}
        traces: dict[str, Any] = {}
        trace_rank_evidence: dict[str, Any] = {}
        candidate_rows: list[Mapping[str, Any]] = []
        selected_refs = tuple(correct_refs[index] for index in REFERENCE_INDICES)
        correct_ref_digests = [
            condition_identities["correct_references"][str(index)]["identity"][
                "raw_storage_sha256"
            ]
            for index in REFERENCE_INDICES
        ]
        for seed in cell["seeds"]:
            for arm in v_axis.ARM_ORDER:
                key = _candidate_key(seed, arm)
                contract = v_axis.arm_contract(arm)
                selected_video = (
                    full_wrong_latent
                    if contract["full_video_condition_role"] == "wrong"
                    else full_correct_latent
                )
                hook = v_axis.NativeVAxisGuidanceHook(
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
                            sample_fn=(
                                lambda video=selected_video, selected_seed=seed: diffusion.sample(
                                    prompt_embeds=positive_embeds,
                                    uncond_prompt_embeds=negative_embeds,
                                    image_vae_latents=None,
                                    multi_video_vae_latents=[video],
                                    multi_image_vae_latents=list(selected_refs),
                                    width=bucket_hw[1],
                                    height=bucket_hw[0],
                                    device=device,
                                    **_sampling_contract(selected_seed),
                                )
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
                    raise NativeVAxisProbeError(f"{key} native endpoint differs")
                if hook.sample_calls != 1 or not hook.restored:
                    raise NativeVAxisProbeError(f"{key} hook lifecycle differs")
                trace = dict(hook.trace)
                trace_gate = validate_exact40_trace(trace, arm=arm)
                generated[key] = endpoint.detach().cpu().contiguous()
                generated_identities[key] = native._all_rank_tensor_identity(
                    generated[key],
                    label=f"generated_{key}",
                    world_size=WORLD_SIZE,
                )
                captures[key] = capture
                capture_rank_identities[key] = native._all_rank_tensor_identity(
                    capture.tensor,
                    label=f"official_initial_gaussian_{key}",
                    world_size=WORLD_SIZE,
                )
                traces[key] = trace
                trace_rank_evidence[key] = _gather_equal(
                    trace["trace_digest"],
                    world_size=WORLD_SIZE,
                    label=f"trace_{key}",
                )
                video_identity_key = (
                    "full_wrong_video"
                    if arm == "wrong-V"
                    else "full_correct_video"
                )
                unsigned_candidate = {
                    "candidate_key": key,
                    "seed": seed,
                    "arm": arm,
                    "arm_contract": dict(contract),
                    "full_video_raw_storage_sha256": condition_identities[
                        video_identity_key
                    ]["identity"]["raw_storage_sha256"],
                    "correct_reference_raw_storage_sha256_in_order": (
                        correct_ref_digests
                    ),
                    "action_caption_utf8_sha256": cell[
                        "action_caption_utf8_sha256"
                    ],
                    "trace_digest": trace["trace_digest"],
                    "exact40_trace_gate": trace_gate,
                    "official_initial_gaussian_raw_value_sha256": (
                        capture.raw_value_sha256
                    ),
                    "generated_identity": generated_identities[key],
                    "score": None,
                    "rank": None,
                    "selected": False,
                }
                candidate_rows.append(
                    {
                        **unsigned_candidate,
                        "candidate_receipt_digest": object_sha256(
                            unsigned_candidate
                        ),
                    }
                )
                del endpoint
                torch.cuda.empty_cache()

        if len(candidate_rows) != 2 * len(v_axis.ARM_ORDER):
            raise NativeVAxisProbeError("candidate count differs")
        for seed in cell["seeds"]:
            seed_hashes = {
                captures[_candidate_key(seed, arm)].raw_value_sha256
                for arm in v_axis.ARM_ORDER
            }
            if len(seed_hashes) != 1:
                raise NativeVAxisProbeError("same-seed arms lost common Gaussian")
            seed_rows = [row for row in candidate_rows if row["seed"] == seed]
            if len({tuple(row["correct_reference_raw_storage_sha256_in_order"]) for row in seed_rows}) != 1:
                raise NativeVAxisProbeError("same-seed arms changed correct I refs")
            if len({row["action_caption_utf8_sha256"] for row in seed_rows}) != 1:
                raise NativeVAxisProbeError("same-seed arms changed instruction")
        seed_parent_hashes = {
            captures[_candidate_key(seed, "V-on")].raw_value_sha256
            for seed in cell["seeds"]
        }
        if len(seed_parent_hashes) != 2:
            raise NativeVAxisProbeError("two sealed seeds produced same Gaussian")

        sampling_guard_after = _model_mutation_guard(model)
        if sampling_guard_after != sampling_guard_before or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise NativeVAxisProbeError("frozen model changed during V-axis sampling")
        # Never create four new CPU transformer copies.  The full byte
        # certificate and process-local guards are already closed, so destroy
        # the sampling graph before rank-zero VAE decode.
        del diffusion, model, positive_embeds, negative_embeds
        gc.collect()
        torch.cuda.empty_cache()
        if distributed.rank != 0:
            del full_correct_latent, full_wrong_latent, correct_refs, selected_refs
            gc.collect()
            torch.cuda.empty_cache()

        if distributed.rank == 0:
            stage = prior._output_staging_directory(output_dir)
            correct_source_snapshot = stage / "source-correct.mp4"
            wrong_source_snapshot = stage / "source-wrong-V.mp4"
            shutil.copyfile(source_path, correct_source_snapshot)
            shutil.copyfile(wrong_source_path, wrong_source_snapshot)
            if (
                native.legacy.file_sha256(correct_source_snapshot) != source_sha
                or native.legacy.file_sha256(wrong_source_snapshot) != wrong_sha
            ):
                raise NativeVAxisProbeError("source MP4 snapshot differs")
            correct_source_artifact = native._save_normalized_clean_latent_atomically(
                stage / "source-correct.normalized-clean-latent.safetensors",
                full_correct_latent,
                artifact_role="source_video_condition",
            )
            wrong_source_artifact = native._save_normalized_clean_latent_atomically(
                stage / "source-wrong-V.normalized-clean-latent.safetensors",
                full_wrong_latent,
                artifact_role="source_video_condition",
            )
            ref_artifacts = {
                str(index): prior._save_tensor_artifact(
                    stage / f"correct-reference-{index:03d}.safetensors",
                    value,
                    key="reference_latent",
                    metadata={
                        "coordinate": "independent_RGB_frame_to_Wan_VAE_T1",
                        "source_role": "correct_for_all_three_arms",
                        "frame_index": str(index),
                    },
                )
                for index, value in correct_refs.items()
            }
            del full_correct_latent, full_wrong_latent, correct_refs, selected_refs
            gc.collect()
            torch.cuda.empty_cache()
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
            if vae is None:
                raise NativeVAxisProbeError("rank-zero VAE lifetime differs")
            outputs = native._save_outputs(
                output_dir=stage,
                generated=generated_device,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
            generated_device.clear()
            torch.cuda.empty_cache()
            unsigned_receipt: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "stage": "stage_A0_native_full_video_axis_exact40_exact81",
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
                    "content_validated_at_start": checkpoint_identity,
                    "opened_read_only": True,
                },
                "correct_source": {
                    "path": str(source_path),
                    "sha256": source_sha,
                    "snapshot_mp4": str(correct_source_snapshot),
                    "metadata": source_metadata,
                    "normalized_clean_latent_artifact": correct_source_artifact,
                },
                "wrong_V_source": {
                    "path": str(wrong_source_path),
                    "sha256": wrong_sha,
                    "snapshot_mp4": str(wrong_source_snapshot),
                    "metadata": wrong_metadata,
                    "normalized_clean_latent_artifact": wrong_source_artifact,
                    "used_only_as_full_video_condition_in_wrong_V": True,
                    "used_as_image_reference": False,
                    "pure_identity_control": False,
                    "geometry_confound_present": bool(
                        cell["wrong_source_geometry_confound"]
                    ),
                },
                "correct_image_references": {
                    "indices": list(REFERENCE_INDICES),
                    "artifacts": ref_artifacts,
                    "same_tensor_objects_across_all_three_arms_within_seed": True,
                },
                "condition_identities": condition_identities,
                "prompt": {
                    "action_caption": cell["action_caption"],
                    "action_caption_utf8_sha256": cell[
                        "action_caption_utf8_sha256"
                    ],
                    "full_native_prompt_utf8_sha256": hashlib.sha256(
                        full_prompt.encode("utf-8")
                    ).hexdigest(),
                    "same_across_all_arms_and_seeds": True,
                    "positive_embedding_encoded_once_per_process": True,
                    "negative_embedding_encoded_once_per_process": True,
                    "input_ids_not_reencoded_per_arm": True,
                    "text_encoder_retired_before_sampling": True,
                },
                "sampling": {
                    "exact40": True,
                    "exact81": True,
                    "frame_count": FRAME_COUNT,
                    "latent_phases": LATENT_PHASES,
                    "fps": FPS,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "seeds": list(cell["seeds"]),
                    "arm_order": list(v_axis.ARM_ORDER),
                    "same_official_gaussian_within_seed": True,
                    "same_x_t_t_target_geometry_within_seed": True,
                    "hook_contract": v_axis.hook_contract(),
                },
                "candidates": candidate_rows,
                "traces": traces,
                "trace_all_rank_evidence": trace_rank_evidence,
                "generated_identities": generated_identities,
                "initial_noise_artifacts": initial_noise_artifacts,
                "outputs": outputs,
                "freeze_certificate": {
                    "rank_zero_exact_full_model_before_prompt": (
                        full_freeze_certificate
                    ),
                    "exact_full_model_bytes_hashed_on_rank_zero_only": True,
                    "all_ranks_prompt_guard_before": full_guard_before_prompt,
                    "all_ranks_prompt_guard_after": full_guard_after_prompt,
                    "all_ranks_model_unchanged_during_prompt_encoding": True,
                    "text_encoder_retired_before_vae_and_sampling": True,
                    "all_ranks_sampling_guard_before": sampling_guard_before,
                    "all_ranks_sampling_guard_after": sampling_guard_after,
                    "all_ranks_sampling_model_unchanged": True,
                },
                "resource_lifetime": {
                    **resource_lifetime_contract(),
                    "serialized_load_lock_path": os.environ[
                        "NATIVE_V_AXIS_LOAD_LOCK"
                    ],
                    "host_allocator_trim_invoked_after_rank_load": (
                        host_allocator_trim_invoked
                    ),
                    "rank_zero_source_video_decode": True,
                    "rank_zero_only_vae_observed": vae is not None,
                },
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                },
                "interpretation": {
                    "probe_question": (
                        "does_the_native_standalone_full_video_axis_causally_"
                        "control_preservation_under_fixed_image_refs_and_text"
                    ),
                    "candidate_count": len(candidate_rows),
                    "distinct_from_native_i_axis_131497": True,
                    "training_performed": False,
                    "trainer_instantiated": False,
                    "optimizer": None,
                    "backward": False,
                    "model_weights_written": False,
                    "adapter_loaded": False,
                    "apg": False,
                    "target_video": False,
                    "feature_scorer_consumed": False,
                    "reward_computed": False,
                    "score_computed": False,
                    "ranking_performed": False,
                    "best_arm_selected": False,
                    "visual_selection_performed": False,
                    "wrong_V_changes_only_full_video_condition": True,
                    "V_off_zeros_only_standalone_vV_minus_v0_coefficient": True,
                    "V_off_retains_vVIu_minus_vV_term": True,
                    "action_success_evaluated": False,
                    "preservation_success_evaluated": False,
                    "scientific_claim_authorized_before_blind_review": False,
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
        del generated, captures, vae
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NativeVAxisProbeError",
    "SPEC_SCHEMA_VERSION",
    "_expected_spec_contract",
    "build_parser",
    "load_cell_spec",
    "main",
    "validate_cli",
    "validate_exact40_trace",
]
