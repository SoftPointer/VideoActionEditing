#!/usr/bin/env python3
"""Canonical 81-frame Bernini CPMR full-video engineering oracle.

This runner closes the gap between the single-block Ulysses smoke and a real
Bernini trajectory.  It produces two frozen, same-source/same-noise proposals,
builds the V11 carrier from their patch embeddings, and renders three paired
arms from the same independent render seed:

``B0``
    Official source + semantic-no-op Bernini output, without a CPMR patch.
``Z0``
    The identical call with the complete CPMR patch installed at gate zero.
``C10``
    The identical call with the correct carrier at gate 0.10.

The external interface is source video plus action instruction.  No target,
mask, flow, pose, track, trajectory, edited keyframe, or reference is accepted.
This first runner is an engineering/causal oracle, not a training or quality
claim.
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

import counterfactual_proposal_motion_branch as motion_branch  # noqa: E402
import counterfactual_proposal_motion_rebinding as carrier_core  # noqa: E402
import counterfactual_proposal_motion_runtime as motion_runtime  # noqa: E402
import infer_lora as legacy  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402
import source_kv_route_batches as route_batches  # noqa: E402


RECEIPT_SCHEMA = "bernini-cpmr-v11-full-video-engineering-oracle-v1"
EXPECTED_FRAMES = 81
EXPECTED_STEPS = 40
EXPECTED_ULYSSES_SIZE = 4
EXPECTED_BUCKET_HW = (496, 480)
EXPECTED_SOURCE_TOKENS = 19_530
EXPECTED_LATENT_SHAPE = (1, 16, 21, 62, 60)
EXPECTED_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
EXPECTED_INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
PROPOSAL_SEED = 2027
RENDER_SEED = 2028
ARM_ORDER = ("B0", "Z0", "C10")
ARM_GATES = {"B0": None, "Z0": 0.0, "C10": 0.10}
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CPMRFullVideoOracleError(RuntimeError):
    """Raised before an ambiguous full-video artifact is published."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--original-source-path", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument(
        "--expected-bernini-commit",
        default=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--proposal-seed", type=int, default=PROPOSAL_SEED)
    parser.add_argument("--render-seed", type=int, default=RENDER_SEED)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.instruction != EXPECTED_INSTRUCTION:
        raise CPMRFullVideoOracleError("canonical instruction differs")
    if args.expected_source_sha256 != EXPECTED_SOURCE_SHA256:
        raise CPMRFullVideoOracleError("canonical source SHA256 differs")
    if args.num_inference_steps != EXPECTED_STEPS:
        raise CPMRFullVideoOracleError("oracle is fixed to 40 solver steps")
    if args.proposal_seed != PROPOSAL_SEED or args.render_seed != RENDER_SEED:
        raise CPMRFullVideoOracleError("proposal/render seeds are frozen")
    for name in ("proposal_seed", "render_seed"):
        value = getattr(args, name)
        if type(value) is not int or not 0 <= value < 2**63:
            raise CPMRFullVideoOracleError(f"{name} must be in [0,2^63)")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
            raise CPMRFullVideoOracleError(f"{name} must be a full lowercase SHA-1")
    for name in (
        "expected_source_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise CPMRFullVideoOracleError(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit != legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise CPMRFullVideoOracleError("unsupported Bernini source revision")
    if args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise CPMRFullVideoOracleError("unsupported VeOmni source revision")
    if args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise CPMRFullVideoOracleError("unsupported checkpoint tree")
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output.suffix:
        raise CPMRFullVideoOracleError("output-dir must be an absolute directory path")


def _tensor_bytes_equal(left: Any, right: Any) -> bool:
    import torch

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _all_rank_identities(value: Any, *, label: str, world_size: int) -> dict[str, Any]:
    import torch.distributed as dist

    local = value_audit.tensor_identity(value, label=label)
    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, local)
    if any(item != rows[0] for item in rows[1:]):
        raise CPMRFullVideoOracleError(f"{label} differs across Ulysses ranks")
    return {"all_rank_exact": True, "identity": dict(rows[0])}


def _patch_field(transformer: Any, latent: Any) -> Any:
    import torch

    if tuple(int(item) for item in latent.shape) != EXPECTED_LATENT_SHAPE:
        raise CPMRFullVideoOracleError("proposal latent shape differs")
    try:
        transformer_dtype = next(transformer.parameters()).dtype
    except (AttributeError, StopIteration) as error:
        raise CPMRFullVideoOracleError(
            "could not resolve the proposal transformer's compute dtype"
        ) from error
    with torch.no_grad():
        # GEN_Wanx22 returns the scheduler state in FP32, while the pinned
        # transformer's patch embedding is BF16.  Match the official
        # ``patch_vae_latent`` path before invoking the convolution; otherwise
        # ROCm correctly rejects the mixed input/weight dtypes.
        embedded = transformer.patch_embedding(latent.to(dtype=transformer_dtype))
        expected = (1, carrier_core.HIDDEN_SIZE, 21, 31, 30)
        if tuple(int(item) for item in embedded.shape) != expected:
            raise CPMRFullVideoOracleError("proposal patch embedding shape differs")
        field = embedded.permute(0, 2, 3, 4, 1).contiguous()
    if tuple(int(item) for item in field.shape) != (1, 21, 31, 30, 1536):
        raise CPMRFullVideoOracleError("proposal patch field shape differs")
    return field


def _sample_kwargs(
    *,
    input_ids: Any,
    attention_mask: Any,
    negative_ids: Any,
    negative_mask: Any,
    source_latent: Any,
    bucket: tuple[int, int],
    device: Any,
    seed: int,
) -> dict[str, Any]:
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "uncond_input_ids": negative_ids.to(device),
        "uncond_attention_mask": negative_mask.to(device),
        "image_vae_latents": None,
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": None,
        "width": int(bucket[1]),
        "height": int(bucket[0]),
        "device": device,
        **legacy.sampler_contract(steps=EXPECTED_STEPS, seed=seed),
    }


def _save_outputs(
    *,
    output_dir: Path,
    values: Mapping[str, Any],
    vae: Any,
    device: Any,
    save_output_fn: Any,
) -> dict[str, Any]:
    from bernini.pipeline import _vae_decode
    from tools import materialize_vae

    outputs: dict[str, Any] = {}
    vae.to(device)
    for name in ("proposal_action", "proposal_noop", *ARM_ORDER):
        latent = values[name]
        with __import__("torch").no_grad():
            decoded = _vae_decode(vae, latent)
        if tuple(int(item) for item in decoded.shape) != (81, 496, 480, 3):
            raise CPMRFullVideoOracleError(f"{name} decoded shape differs")
        path = output_dir / f"{name}.mp4"
        if path.exists() or path.is_symlink():
            raise CPMRFullVideoOracleError(f"refusing to overwrite {path}")
        value_audit.save_video_atomically(
            decoded, path, fps=int(legacy.FPS), save_output_fn=save_output_fn
        )
        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(path)
        legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        if tuple(encoded_hw) != EXPECTED_BUCKET_HW:
            raise CPMRFullVideoOracleError(f"{name} encoded geometry differs")
        outputs[name] = {
            "path": str(path),
            "mp4_sha256": legacy.file_sha256(path),
            "latent": value_audit.tensor_identity(latent, label=f"{name} latent"),
        }
    vae.to("cpu")
    return outputs


def _build_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    runtime_versions: Mapping[str, str],
    freeze_certificate: Mapping[str, Any],
    proposal_identities: Mapping[str, Any],
    arm_identities: Mapping[str, Any],
    carrier_receipt: Mapping[str, Any],
    z0_byte_exact: bool,
    c10_differs: bool,
    z0_trace: Mapping[str, Any],
    c10_trace: Mapping[str, Any],
    patch_receipt: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "method": carrier_core.METHOD_NAME,
        "method_revision": args.method_source_revision,
        "method_archive_sha256": args.method_source_archive_sha256,
        "scientific_claim": False,
        "video_quality_claim": False,
        "training_claim": False,
        "lora_claim": False,
        "full_video_engineering_claim": True,
        "source_instruction_only_inference": True,
        "forbidden_inputs": [
            "target",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "edited_first_frame",
            "reference_image",
            "reference_video",
        ],
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "metadata": dict(source_metadata),
        },
        "instruction": args.instruction,
        "instruction_sha256": hashlib.sha256(args.instruction.encode()).hexdigest(),
        "seeds": {"proposal": PROPOSAL_SEED, "render": RENDER_SEED},
        "schedule": {
            "frames": EXPECTED_FRAMES,
            "steps": EXPECTED_STEPS,
            "flow_shift": 5.0,
            "proposal_action_noop_same_seed": True,
            "render_arms_same_seed": True,
        },
        "arms": {"order": list(ARM_ORDER), "gates": ARM_GATES},
        "verified_claims": {
            "proposal_latents_all_rank_exact": True,
            "arm_latents_all_rank_exact": True,
            "z0_full_latent_byte_exact_b0": z0_byte_exact,
            "c10_full_latent_differs_from_z0": c10_differs,
            "z0_complete_40_step_binding": z0_trace.get("all_bindings_complete") is True,
            "c10_complete_40_step_binding": c10_trace.get("all_bindings_complete") is True,
            "phase_zero_carrier_exact_zero": str(carrier_receipt["activity_bitset"][0]).startswith("0"),
        },
        "proposal_latents": dict(proposal_identities),
        "arm_latents": dict(arm_identities),
        "carrier": dict(carrier_receipt),
        "runtime_traces": {"Z0": dict(z0_trace), "C10": dict(c10_trace)},
        "patch": dict(patch_receipt),
        "outputs": dict(outputs),
        "checkpoint": dict(checkpoint_identity),
        "source_revisions": {
            "bernini": bernini_revision,
            "veomni": veomni_revision,
        },
        "runtime_versions": dict(runtime_versions),
        "freeze_certificate": dict(freeze_certificate),
    }
    if not all(payload["verified_claims"].values()):
        raise CPMRFullVideoOracleError("full-video engineering claims did not all pass")
    payload["receipt_digest"] = legacy.object_sha256(payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise CPMRFullVideoOracleError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % EXPECTED_ULYSSES_SIZE:
        raise CPMRFullVideoOracleError("attention heads are not divisible by Ulysses=4")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise CPMRFullVideoOracleError("runtime mv2v prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise CPMRFullVideoOracleError("runtime negative prompt differs")
    route_batches.validate_noop_instruction(route_batches.EXACT_NOOP_INSTRUCTION)
    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise CPMRFullVideoOracleError("oracle requires AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_results: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_results[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, Path(args.checkpoint_content_manifest).expanduser()
                ),
            }
        except Exception as error:
            checkpoint_results[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_results, src=0)
    checkpoint_result = checkpoint_results[0]
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
        raise CPMRFullVideoOracleError(
            f"rank-zero checkpoint validation failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])

    source_path = Path(args.source_video).expanduser().resolve(strict=True)
    original_source = Path(args.original_source_path).expanduser().resolve(strict=True)
    if source_path != original_source:
        raise CPMRFullVideoOracleError("staged and canonical source paths differ")
    source_tensor, source_metadata, source_sha256 = source_audit.prepare_hashed_source_snapshot(
        source_path
    )
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise CPMRFullVideoOracleError("source SHA256 differs")
    if tuple(source_metadata["source_derived_bucket_hw"]) != EXPECTED_BUCKET_HW:
        raise CPMRFullVideoOracleError("source-derived dog bucket differs")

    action_prompt = legacy.build_training_prompt(args.instruction, prompt_cleaner=prompt_clean)
    noop_prompt = legacy.build_training_prompt(
        route_batches.EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    action_ids, action_mask = legacy._tokenize_training_prompt(tokenizer, action_prompt)
    noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise CPMRFullVideoOracleError(str(error)) from error
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = source_audit.model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    with torch.no_grad():
        source_latent = _vae_encode(vae, source_tensor.to(device=device, dtype=torch.float32))
    if tuple(int(item) for item in source_latent.shape) != EXPECTED_LATENT_SHAPE:
        raise CPMRFullVideoOracleError("source latent shape differs")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()
    model.to(device)

    common = dict(
        negative_ids=negative_ids,
        negative_mask=negative_mask,
        source_latent=source_latent,
        bucket=EXPECTED_BUCKET_HW,
        device=device,
    )
    with torch.no_grad():
        proposal_action = model.sample(
            **_sample_kwargs(
                input_ids=action_ids,
                attention_mask=action_mask,
                seed=PROPOSAL_SEED,
                **common,
            )
        )
        proposal_noop = model.sample(
            **_sample_kwargs(
                input_ids=noop_ids,
                attention_mask=noop_mask,
                seed=PROPOSAL_SEED,
                **common,
            )
        )
    if _tensor_bytes_equal(proposal_action, proposal_noop):
        raise CPMRFullVideoOracleError("action/no-op proposals are byte-identical")
    transformer = motion_branch.resolve_wan_transformer(model)
    action_field = _patch_field(transformer, proposal_action)
    noop_field = _patch_field(transformer, proposal_noop)
    carrier_result = carrier_core.build_motion_carrier(action_field, noop_field)
    carrier = carrier_result.bfloat16().reshape(1, 1_344, 1_536)
    activity = carrier_result.activity
    del action_field, noop_field

    with torch.no_grad():
        b0 = model.sample(
            **_sample_kwargs(
                input_ids=noop_ids,
                attention_mask=noop_mask,
                seed=RENDER_SEED,
                **common,
            )
        )
    with motion_branch.install_cpmr_motion_branch(model) as patch_handle:
        with motion_runtime.cpmr_final_render_hook(
            model,
            patch_handle=patch_handle,
            carrier=carrier,
            activity=activity,
            gate=0.0,
        ) as z0_hook:
            with torch.no_grad():
                z0 = model.sample(
                    **_sample_kwargs(
                        input_ids=noop_ids,
                        attention_mask=noop_mask,
                        seed=RENDER_SEED,
                        **common,
                    )
                )
        z0_trace = z0_hook.trace.receipt()
        with motion_runtime.cpmr_final_render_hook(
            model,
            patch_handle=patch_handle,
            carrier=carrier,
            activity=activity,
            gate=0.10,
        ) as c10_hook:
            with torch.no_grad():
                c10 = model.sample(
                    **_sample_kwargs(
                        input_ids=noop_ids,
                        attention_mask=noop_mask,
                        seed=RENDER_SEED,
                        **common,
                    )
                )
        c10_trace = c10_hook.trace.receipt()
        patch_receipt = patch_handle.receipt()
    patch_receipt["restored_after_context"] = patch_handle.restored

    for name, value in {
        "proposal_action": proposal_action,
        "proposal_noop": proposal_noop,
        "B0": b0,
        "Z0": z0,
        "C10": c10,
    }.items():
        if tuple(int(item) for item in value.shape) != EXPECTED_LATENT_SHAPE:
            raise CPMRFullVideoOracleError(f"{name} latent shape differs")
    z0_byte_exact = _tensor_bytes_equal(b0, z0)
    c10_differs = not _tensor_bytes_equal(z0, c10)
    if not z0_byte_exact:
        raise CPMRFullVideoOracleError("Z0 differs bytewise from B0")
    if not c10_differs:
        raise CPMRFullVideoOracleError("C10 is byte-identical to Z0")

    proposal_identities = {
        name: _all_rank_identities(value, label=name, world_size=EXPECTED_ULYSSES_SIZE)
        for name, value in {
            "proposal_action": proposal_action,
            "proposal_noop": proposal_noop,
        }.items()
    }
    arm_identities = {
        name: _all_rank_identities(value, label=name, world_size=EXPECTED_ULYSSES_SIZE)
        for name, value in {"B0": b0, "Z0": z0, "C10": c10}.items()
    }
    freeze_after = source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before:
        raise CPMRFullVideoOracleError("model freeze certificate changed")

    runtime_versions = {
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    if distributed.rank == 0:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir.exists() or output_dir.is_symlink():
            raise CPMRFullVideoOracleError("refusing to reuse output directory")
        output_dir.mkdir(parents=True, exist_ok=False)
        values = {
            "proposal_action": proposal_action,
            "proposal_noop": proposal_noop,
            "B0": b0,
            "Z0": z0,
            "C10": c10,
        }
        outputs = _save_outputs(
            output_dir=output_dir,
            values=values,
            vae=vae,
            device=device,
            save_output_fn=save_output,
        )
        receipt = _build_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            checkpoint_identity=checkpoint_identity,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            runtime_versions=runtime_versions,
            freeze_certificate=freeze_after,
            proposal_identities=proposal_identities,
            arm_identities=arm_identities,
            carrier_receipt=carrier_result.audit_receipt(),
            z0_byte_exact=z0_byte_exact,
            c10_differs=c10_differs,
            z0_trace=z0_trace,
            c10_trace=c10_trace,
            patch_receipt=patch_receipt,
            outputs=outputs,
        )
        receipt_path = output_dir / "receipt.json"
        value_audit.write_receipt_atomically(receipt_path, receipt)
        print(legacy.canonical_json_bytes(receipt).decode(), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_GATES",
    "ARM_ORDER",
    "CPMRFullVideoOracleError",
    "EXPECTED_INSTRUCTION",
    "EXPECTED_SOURCE_SHA256",
    "PROPOSAL_SEED",
    "RENDER_SEED",
    "build_parser",
    "main",
    "validate_cli",
]
