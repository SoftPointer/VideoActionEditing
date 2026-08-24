#!/usr/bin/env python3
"""Frozen exact81 role-reversed owner/source-identity rebinding diagnostic.

This canary deliberately tests a narrow counterfactual that is different from
Q-MOSAIC's allowed owner channel: a successful pure-T2V owner is used as the
native Bernini full-video condition, while independently encoded frames from
the original source are used as image references.  The model is frozen.

Three matched arms share one action prompt, official target Gaussian, seed,
native RV2V guidance and exact40 schedule:

``owner-only``
    Measures how much of the owner's action/appearance survives native RV2V.
``owner-source-refs``
    The role-reversed hypothesis: owner temporal scaffold plus source identity.
``source-source-refs``
    Matched source-preservation endpoint and old-motion control.

This is an isolated primal-channel diagnostic, not an approved Q-MOSAIC owner
channel, pseudo-target, training item, optimizer step, or action-edit result.
Pending job-131524 receipts and their signed full81 action audit are
authenticated but never upgraded in place.  That audit authorizes only the
detached Q-MOSAIC quotient; it is evidence of owner motion here, not authority
for this deliberately isolated clean-latent condition.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_action_scaffold_identity_rebinding_canary as scaffold  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_native_multivideo_motion_donor_oracle as donor  # noqa: E402
import materialize_self_imagined_owner_core2_v1 as owner_inputs  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "frozen-bernini-role-reversed-owner-identity-rebinding-canary"
SCHEMA_VERSION = "bernini-role-reversed-owner-identity-rebinding-receipt-v1"
FRAME_COUNT = 81
LATENT_PHASES = 21
REFERENCE_INDICES = (0, 27, 53, 80)
FPS = 25
ULYSSES_SIZE = 4
NUM_INFERENCE_STEPS = 40
TARGET_SEED = 20_260_810
EXPERIMENTAL_ACK = "frozen_role_reversed_owner_primal_diagnostic_only"
ROLE_CLAUSE = (
    " Treat any full-video condition primarily as temporal action evidence. "
    "When source reference images are present, the output must use their exact "
    "subject identity, appearance, scene, camera, composition, and lighting; "
    "do not copy a conflicting identity or camera from the full-video condition."
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RoleReversedOwnerCanaryError(RuntimeError):
    """Raised before ambiguous or unauthenticated evidence can be published."""


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    video_role: str
    use_source_references: bool
    diagnostic: str


ARM_SPECS = (
    ArmSpec(
        "owner-only",
        "pure_t2v_owner_predecode_clean_latent",
        False,
        "owner_action_and_appearance_survival_endpoint",
    ),
    ArmSpec(
        "owner-source-refs",
        "pure_t2v_owner_predecode_clean_latent",
        True,
        "role_reversed_motion_owner_plus_source_identity_hypothesis",
    ),
    ArmSpec(
        "source-source-refs",
        "source_video_clean_latent",
        True,
        "source_identity_camera_endpoint_and_old_motion_control",
    ),
)
ARM_ORDER = tuple(spec.arm_id for spec in ARM_SPECS)


def _require_sha(value: Any, *, bits: int, label: str) -> str:
    text = str(value)
    pattern = _SHA1 if bits == 1 else _SHA256
    if pattern.fullmatch(text) is None:
        raise RoleReversedOwnerCanaryError(
            f"{label} must be lowercase SHA-{bits if bits == 1 else 256}"
        )
    return text


def arm_plan() -> tuple[ArmSpec, ...]:
    if tuple(spec.arm_id for spec in ARM_SPECS) != ARM_ORDER:
        raise RoleReversedOwnerCanaryError("arm order differs")
    if len(set(ARM_ORDER)) != len(ARM_ORDER):
        raise RoleReversedOwnerCanaryError("arm IDs are not unique")
    if sum(spec.use_source_references for spec in ARM_SPECS) != 2:
        raise RoleReversedOwnerCanaryError("reference marginal design differs")
    if tuple(spec.video_role for spec in ARM_SPECS[:2]) != (
        "pure_t2v_owner_predecode_clean_latent",
        "pure_t2v_owner_predecode_clean_latent",
    ):
        raise RoleReversedOwnerCanaryError("owner reference marginal differs")
    return ARM_SPECS


def bucket_and_patch_geometry(
    latent_shape: Sequence[int],
) -> tuple[tuple[int, int], int, int]:
    shape = tuple(int(item) for item in latent_shape)
    if (
        len(shape) != 5
        or shape[:3] != (1, 16, LATENT_PHASES)
        or shape[3] <= 0
        or shape[4] <= 0
        or shape[3] % 2
        or shape[4] % 2
    ):
        raise RoleReversedOwnerCanaryError("latent geometry differs")
    spatial_tokens = (shape[3] // 2) * (shape[4] // 2)
    return (shape[3] * 8, shape[4] * 8), LATENT_PHASES * spatial_tokens, spatial_tokens


def renderer_body(action_caption: str) -> str:
    if not isinstance(action_caption, str) or not action_caption.strip():
        raise RoleReversedOwnerCanaryError("action caption differs")
    return action_caption.strip() + ROLE_CLAUSE


def condition_plan(spec: ArmSpec) -> dict[str, Any]:
    if spec not in arm_plan():
        raise RoleReversedOwnerCanaryError("unknown arm")
    refs = REFERENCE_INDICES if spec.use_source_references else ()
    return {
        "video_role": spec.video_role,
        "reference_video_role": "source_video" if refs else None,
        "source_reference_indices_in_order": list(refs),
        "first_video_alone_enters_v": True,
        "video_and_references_enter_vi": True,
        "target_source_id": 0,
        "video_source_id": 1.0,
        "vi_reference_source_ids": [float(index) for index in range(2, 2 + len(refs))],
        "image_only_reference_source_ids": [
            float(index) for index in range(1, 1 + len(refs))
        ],
        "native_source_id_extrapolation_used": False,
    }


def scaffold_spec(spec: ArmSpec) -> scaffold.ArmSpec:
    refs = REFERENCE_INDICES if spec.use_source_references else ()
    return scaffold.ArmSpec(
        arm_id=spec.arm_id,
        video_roles=(spec.video_role,),
        donor_branch=None,
        source_reference_indices=refs,
        reference_video_role="source_video" if refs else None,
        privileged_v_role=spec.video_role,
        diagnostic=spec.diagnostic,
    )


@contextmanager
def dynamic_scaffold_audit_contract(
    *, target_patch_tokens: int, reference_patch_tokens: int, seed: int
) -> Iterator[None]:
    """Bind the reusable native observer to one cell's exact geometry.

    The two cells execute in separate OS processes.  Values are nevertheless
    restored in ``finally`` so unit tests and any later in-process use cannot
    inherit a modified audit contract.
    """

    if min(target_patch_tokens, reference_patch_tokens) <= 0 or seed < 0:
        raise RoleReversedOwnerCanaryError("dynamic audit contract differs")
    names = ("PATCH_TOKENS", "REFERENCE_PATCH_TOKENS", "TARGET_SEED")
    previous = {name: getattr(scaffold, name) for name in names}
    scaffold.PATCH_TOKENS = int(target_patch_tokens)
    scaffold.REFERENCE_PATCH_TOKENS = int(reference_patch_tokens)
    scaffold.TARGET_SEED = int(seed)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(scaffold, name, value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--owner-root", required=True)
    parser.add_argument("--owner-master-receipt", required=True)
    parser.add_argument("--expected-owner-master-receipt-sha256", required=True)
    parser.add_argument("--audit-sidecar", required=True)
    parser.add_argument("--expected-audit-sidecar-sha256", required=True)
    parser.add_argument("--audit-evidence", required=True)
    parser.add_argument("--audit-public-key", required=True)
    parser.add_argument("--expected-audit-public-key-sha256", required=True)
    parser.add_argument("--cell-id", required=True, choices=("dog", "human"))
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--experimental-owner-primal-ack", required=True)
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
    arm_plan()
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise RoleReversedOwnerCanaryError("scientific canary requires exact40")
    if args.experimental_owner_primal_ack != EXPERIMENTAL_ACK:
        raise RoleReversedOwnerCanaryError("explicit frozen primal diagnostic acknowledgement differs")
    for name in (
        "expected_registry_sha256",
        "expected_owner_master_receipt_sha256",
        "expected_audit_sidecar_sha256",
        "expected_audit_public_key_sha256",
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _require_sha(getattr(args, name), bits=256, label=name)
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        _require_sha(getattr(args, name), bits=1, label=name)
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise RoleReversedOwnerCanaryError("Bernini commit differs")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise RoleReversedOwnerCanaryError("VeOmni commit differs")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise RoleReversedOwnerCanaryError("checkpoint tree differs")


def _load_owner_clean_latent(child: Mapping[str, Any], *, expected_shape: Sequence[int]) -> Any:
    import torch
    from safetensors import safe_open

    artifact = child.get("artifacts", {}).get("predecode_clean_latent")
    if not isinstance(artifact, Mapping):
        raise RoleReversedOwnerCanaryError("owner predecode artifact differs")
    path = Path(str(artifact.get("path", "")))
    tensor_key = artifact.get("tensor_key")
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RoleReversedOwnerCanaryError("owner predecode artifact path differs")
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [tensor_key]:
            raise RoleReversedOwnerCanaryError("owner tensor key differs")
        value = opened.get_tensor(str(tensor_key)).contiguous()
        metadata = dict(opened.metadata() or {})
    shape = tuple(int(item) for item in expected_shape)
    if (
        value.dtype != torch.float32
        or tuple(int(item) for item in value.shape) != shape
        or not bool(torch.isfinite(value).all().item())
        or value.requires_grad
        or value.grad_fn is not None
        or metadata.get("coordinate") != "bernini_normalized_clean_vae_latent"
        or metadata.get("source") != "native_sampler_before_vae_decode"
    ):
        raise RoleReversedOwnerCanaryError("owner clean latent contract differs")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_cli(args)
    output_dir = donor._fresh_output_directory(args.output_dir)

    try:
        pending = owner_inputs.load_pending_owner_generation_inputs(
            registry=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
            owner_root=args.owner_root,
            owner_master_receipt=args.owner_master_receipt,
            expected_owner_master_receipt_sha256=args.expected_owner_master_receipt_sha256,
        )
        audited = owner_inputs.load_authorized_owner_inputs(
            registry=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
            owner_root=args.owner_root,
            owner_master_receipt=args.owner_master_receipt,
            expected_owner_master_receipt_sha256=args.expected_owner_master_receipt_sha256,
            audit_sidecar=args.audit_sidecar,
            expected_audit_sidecar_sha256=args.expected_audit_sidecar_sha256,
            audit_evidence=args.audit_evidence,
            audit_public_key=args.audit_public_key,
            expected_audit_public_key_sha256=args.expected_audit_public_key_sha256,
        )
    except Exception as error:
        raise RoleReversedOwnerCanaryError(str(error)) from error
    if (
        pending.semantic_action_audit_complete is not False
        or pending.template_materialization_authorized is not False
        or pending.clean_latent_editor_input_authorized is not False
    ):
        raise RoleReversedOwnerCanaryError("pending owner authority boundary differs")
    if (
        audited.registry_file_sha256 != pending.registry_file_sha256
        or audited.master_file_sha256 != pending.master_file_sha256
        or audited.master_receipt_digest != pending.master_receipt_digest
        or dict(audited.child_file_sha256) != dict(pending.child_file_sha256)
    ):
        raise RoleReversedOwnerCanaryError("signed owner audit/pending artifact binding differs")
    cell = pending.registry.cell(args.cell_id)
    child = pending.cell(args.cell_id)
    latent_shape = tuple(int(item) for item in cell.latent_shape)
    bucket_hw, target_patch_tokens, reference_patch_tokens = bucket_and_patch_geometry(
        latent_shape
    )
    owner_clean_cpu = _load_owner_clean_latent(child, expected_shape=latent_shape)

    source_requested = Path(cell.source_video)
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise RoleReversedOwnerCanaryError("registry source path differs")
    source_path = source_requested.resolve(strict=True)
    if (
        source_path != source_requested
        or not source_path.is_file()
        or donor.file_sha256(source_path) != cell.source_video_sha256
    ):
        raise RoleReversedOwnerCanaryError("registry source bytes changed")

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
        raise RoleReversedOwnerCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise RoleReversedOwnerCanaryError("attention heads do not divide Ulysses4")
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
    import bernini.models.transformer_wan as transformer_wan
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != native.legacy.MV2V_SYSTEM_PROMPT:
        raise RoleReversedOwnerCanaryError("runtime mv2v prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise RoleReversedOwnerCanaryError("runtime negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if distributed.world_size != ULYSSES_SIZE or distributed.ulysses_size != ULYSSES_SIZE:
        raise RoleReversedOwnerCanaryError("runtime requires WORLD4/Ulysses4")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RoleReversedOwnerCanaryError("runtime requires AUH ROCm")
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
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise RoleReversedOwnerCanaryError(f"checkpoint validation failed: {checkpoint_rows[0]}")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = (
        native.source_audit.prepare_hashed_source_snapshot(source_path)
    )
    if (
        source_sha != cell.source_video_sha256
        or source_metadata.get("frame_count") != FRAME_COUNT
        or tuple(source_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
    ):
        raise RoleReversedOwnerCanaryError("source exact81 geometry differs")

    body = renderer_body(cell.action_caption)
    full_prompt = native.legacy.build_training_prompt(body, prompt_cleaner=prompt_clean)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    target_ids, target_mask = native.legacy._tokenize_training_prompt(tokenizer, full_prompt)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except Exception as error:
        raise RoleReversedOwnerCanaryError(str(error)) from error
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise RoleReversedOwnerCanaryError("renderer scheduler differs")
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = native.source_audit.model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    reference_shape = (1, 16, 1, latent_shape[3], latent_shape[4])
    # ROCm VAE kernels are not promised to be byte-identical across four
    # independent processes.  The source condition is one shared native
    # artifact, so rank 0 encodes it once and broadcasts the exact latent
    # values.  All ranks still hash/validate the source file above.
    if distributed.rank == 0:
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            source_latent = _vae_encode(vae, source_pixels).contiguous()
            source_references = {
                index: _vae_encode(
                    vae, source_pixels[:, :, index : index + 1].contiguous()
                ).contiguous()
                for index in REFERENCE_INDICES
            }
        del source_pixels
    else:
        source_latent = torch.empty(latent_shape, device=device, dtype=torch.float32)
        source_references = {
            index: torch.empty(reference_shape, device=device, dtype=torch.float32)
            for index in REFERENCE_INDICES
        }
    dist.broadcast(source_latent, src=0)
    for index in REFERENCE_INDICES:
        dist.broadcast(source_references[index], src=0)
    if tuple(int(item) for item in source_latent.shape) != latent_shape:
        raise RoleReversedOwnerCanaryError("source latent geometry differs")
    if any(tuple(int(item) for item in value.shape) != reference_shape for value in source_references.values()):
        raise RoleReversedOwnerCanaryError("source reference geometry differs")
    owner_clean = owner_clean_cpu.to(device=device, dtype=torch.float32).contiguous()

    source_identity = native._all_rank_tensor_identity(
        source_latent, label=f"{cell.cell_id}_source_video", world_size=ULYSSES_SIZE
    )
    owner_identity = native._all_rank_tensor_identity(
        owner_clean, label=f"{cell.cell_id}_pure_t2v_owner", world_size=ULYSSES_SIZE
    )
    reference_identities = {
        str(index): native._all_rank_tensor_identity(
            value, label=f"{cell.cell_id}_source_reference_{index}", world_size=ULYSSES_SIZE
        )
        for index, value in source_references.items()
    }
    if (
        source_identity["identity"]["raw_storage_sha256"]
        == owner_identity["identity"]["raw_storage_sha256"]
    ):
        raise RoleReversedOwnerCanaryError("owner and source latent alias")
    if len(reference_identities) != len(REFERENCE_INDICES) or any(
        row.get("all_rank_exact") is not True for row in reference_identities.values()
    ):
        raise RoleReversedOwnerCanaryError("source reference rank closure differs")

    vae.to("cpu")
    del source_tensor, owner_clean_cpu
    torch.cuda.empty_cache()
    model.to(device)
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        prompt_embeds = model.encode_prompt(target_ids.to(device), target_mask.to(device))
        uncond_embeds = model.encode_prompt(negative_ids.to(device), negative_mask.to(device))
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    try:
        wan_source_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        sampler_contract._validate_scheduler_contract(
            diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
        )
    except Exception as error:
        raise RoleReversedOwnerCanaryError(str(error)) from error
    transformer_path = Path(transformer_wan.__file__).resolve()
    transformer_sha = donor.file_sha256(transformer_path)
    if transformer_sha != donor.PINNED_TRANSFORMER_WAN_SHA256:
        raise RoleReversedOwnerCanaryError("transformer source differs")

    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    initial_noise_rank_identities: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    conditions: dict[str, Any] = {}
    with dynamic_scaffold_audit_contract(
        target_patch_tokens=target_patch_tokens,
        reference_patch_tokens=reference_patch_tokens,
        seed=TARGET_SEED,
    ):
        with torch.inference_mode():
            for spec in arm_plan():
                videos = [
                    owner_clean
                    if spec.video_role == "pure_t2v_owner_predecode_clean_latent"
                    else source_latent
                ]
                refs = (
                    [source_references[index] for index in REFERENCE_INDICES]
                    if spec.use_source_references
                    else []
                )
                native_spec = scaffold_spec(spec)
                conditions[spec.arm_id] = condition_plan(spec)
                audit = scaffold.NativeRoleRebindingConditionAudit(
                    diffusion,
                    spec=native_spec,
                    video_conditions=videos,
                    image_references=refs,
                    expected_steps=NUM_INFERENCE_STEPS,
                    prompt_embeds=prompt_embeds,
                    uncond_prompt_embeds=uncond_embeds,
                )
                audit.install()
                sample_kwargs = {
                    "prompt_embeds": prompt_embeds,
                    "uncond_prompt_embeds": uncond_embeds,
                    "image_vae_latents": None,
                    "multi_video_vae_latents": videos,
                    "multi_image_vae_latents": refs if refs else None,
                    "width": bucket_hw[1],
                    "height": bucket_hw[0],
                    "device": device,
                    **native.native_sampling_contract(
                        "rv2v", steps=NUM_INFERENCE_STEPS, seed=TARGET_SEED
                    ),
                }
                try:
                    result, capture = native._sample_with_native_initial_noise_observer(
                        sample_fn=lambda kw=sample_kwargs: diffusion.sample(**kw),
                        wan_diffusion_module=wan_diffusion,
                        expected_shape=latent_shape,
                        expected_device=device,
                        expected_seed=TARGET_SEED,
                    )
                finally:
                    audit.restore()
                if (
                    not isinstance(result, torch.Tensor)
                    or result.device != device
                    or result.dtype != torch.float32
                    or result.requires_grad
                    or result.grad_fn is not None
                    or not result.is_contiguous()
                    or tuple(int(item) for item in result.shape) != latent_shape
                    or not bool(torch.isfinite(result).all().item())
                ):
                    raise RoleReversedOwnerCanaryError("native sampler return differs")
                generated_cpu = result.detach().to(device="cpu").contiguous()
                generated[spec.arm_id] = generated_cpu
                generated_identities[spec.arm_id] = native._all_rank_tensor_identity(
                    generated_cpu,
                    label=f"{cell.cell_id}_{spec.arm_id}",
                    world_size=ULYSSES_SIZE,
                )
                initial_noise[spec.arm_id] = capture
                initial_noise_rank_identities[spec.arm_id] = native._all_rank_tensor_identity(
                    capture.tensor,
                    label=f"{cell.cell_id}_{spec.arm_id}_official_initial_gaussian",
                    world_size=ULYSSES_SIZE,
                )
                audits[spec.arm_id] = dict(audit.trace)

    noise_hashes = {capture.raw_value_sha256 for capture in initial_noise.values()}
    if len(noise_hashes) != 1:
        raise RoleReversedOwnerCanaryError("arms did not share one official Gaussian")
    freeze_after = native.source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before or any(parameter.requires_grad for parameter in model.parameters()):
        raise RoleReversedOwnerCanaryError("frozen model changed")
    model.to("cpu")
    torch.cuda.empty_cache()

    after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            after_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            after_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(after_rows, src=0)
    if not isinstance(after_rows[0], Mapping) or after_rows[0].get("identity") != checkpoint_identity:
        raise RoleReversedOwnerCanaryError("checkpoint content changed")

    local_evidence = {
        "rank": distributed.rank,
        "audits_digest": scaffold.object_sha256(audits),
        "generated_digest": scaffold.object_sha256(generated_identities),
        "noise_raw_sha256": next(iter(noise_hashes)),
        "freeze_digest": scaffold.object_sha256(freeze_after),
    }
    gathered: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(gathered, local_evidence)
    if sorted(row.get("rank") for row in gathered if isinstance(row, Mapping)) != [0, 1, 2, 3]:
        raise RoleReversedOwnerCanaryError("WORLD4 rank closure differs")
    for field in ("audits_digest", "generated_digest", "noise_raw_sha256", "freeze_digest"):
        if len({row.get(field) for row in gathered if isinstance(row, Mapping)}) != 1:
            raise RoleReversedOwnerCanaryError(f"WORLD4 ranks disagree on {field}")

    if distributed.rank == 0:
        artifact_dir = donor._output_staging_directory(output_dir)
        noise_artifacts = {
            arm: native._save_initial_noise_atomically(
                artifact_dir / f"{arm}.official-initial-gaussian.safetensors",
                initial_noise[arm],
                all_rank_identity=initial_noise_rank_identities[arm],
            )
            for arm in ARM_ORDER
        }
        generated_for_decode = {
            arm: latent.to(device=device).contiguous() for arm, latent in generated.items()
        }
        try:
            outputs = native._save_outputs(
                output_dir=artifact_dir,
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
            "cell_id": cell.cell_id,
            "source_iid": cell.source_iid,
            "action_family_id": cell.action_family_id,
            "runtime_source": {
                "revision": args.runtime_source_revision,
                "archive_sha256": args.runtime_source_archive_sha256,
                "launcher_sha256": args.launcher_source_sha256,
            },
            "pinned_sources": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "wan_diffusion_sha256": wan_source_sha,
                "transformer_wan_sha256": transformer_sha,
                "bernini_inference_files": inference_file_hashes,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_before_and_after": checkpoint_identity,
                "unchanged": True,
            },
            "pending_owner_generation": {
                "registry_path": str(pending.registry_path),
                "registry_file_sha256": pending.registry_file_sha256,
                "owner_root": str(pending.owner_root),
                "master_path": str(pending.master_path),
                "master_file_sha256": pending.master_file_sha256,
                "master_receipt_digest": pending.master_receipt_digest,
                "child_path": str(pending.child_paths[cell.cell_id]),
                "child_file_sha256": pending.child_file_sha256[cell.cell_id],
                "child_receipt_digest": child["receipt_digest"],
                "predecode_clean_latent_artifact": child["artifacts"]["predecode_clean_latent"],
                "semantic_action_audit_complete_in_pending_receipt": False,
                "template_materialization_authorized_by_pending_receipt": False,
                "clean_latent_editor_input_authorized_by_pending_receipt": False,
            },
            "signed_owner_action_audit": {
                "sidecar_path": str(audited.audit_sidecar_path),
                "sidecar_file_sha256": audited.audit_sidecar_file_sha256,
                "sidecar_receipt_digest": audited.audit_sidecar_receipt_digest,
                "evidence_file_sha256": audited.audit_evidence_file_sha256,
                "public_key_file_sha256": audited.audit_public_key_file_sha256,
                "owner_exact81_action_audit_passed": True,
                "semantic_action_audit_complete": True,
                "owner_template_materialization_authorized": True,
                "approval_allows_only_detached_motion_quotient": True,
                "approval_does_not_authorize_clean_latent_editor_input": True,
            },
            "source": {
                "video_path": str(source_path),
                "video_sha256": source_sha,
                "metadata": source_metadata,
                "full_video_identity": source_identity,
                "reference_indices": list(REFERENCE_INDICES),
                "reference_identities": reference_identities,
                "references_independently_vae_encoded_from_rgb": True,
                "references_sliced_from_full_video_latent": False,
            },
            "owner_condition_identity": owner_identity,
            "prompt": {
                "action_caption_utf8_sha256": cell.action_caption_utf8_sha256,
                "role_clause_utf8_sha256": hashlib.sha256(ROLE_CLAUSE.encode("utf-8")).hexdigest(),
                "renderer_body_utf8_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "full_prompt_utf8_sha256": hashlib.sha256(full_prompt.encode("utf-8")).hexdigest(),
                "same_prompt_all_arms": True,
            },
            "matched_target": {
                "seed": TARGET_SEED,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "bucket_hw": list(bucket_hw),
                "latent_shape": list(latent_shape),
                "target_patch_tokens": target_patch_tokens,
                "reference_patch_tokens": reference_patch_tokens,
                "same_official_gaussian_all_arms": True,
                "official_gaussian_raw_sha256": next(iter(noise_hashes)),
                "target_mixed_with_owner_or_source": False,
            },
            "arms_in_order": list(ARM_ORDER),
            "arms": {
                spec.arm_id: {
                    **asdict(spec),
                    "condition_plan": conditions[spec.arm_id],
                    "native_audit": audits[spec.arm_id],
                }
                for spec in arm_plan()
            },
            "initial_noise_artifacts": noise_artifacts,
            "generated_identities": generated_identities,
            "outputs": outputs,
            "frozen_model": freeze_after,
            "world4_evidence": gathered,
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "interpretation": {
                "experimental_owner_primal_ack": EXPERIMENTAL_ACK,
                "isolated_frozen_primal_channel_diagnostic": True,
                "q_mosaic_allowed_owner_channel": False,
                "owner_rgb_consumed": False,
                "owner_mp4_decode_reencode_used": False,
                "owner_predecode_clean_latent_consumed_as_native_v_condition": True,
                "source_refs_consumed_as_native_i_conditions": True,
                "training_performed": False,
                "optimizer": None,
                "backward": False,
                "parameter_update": False,
                "checkpoint_written": False,
                "pseudo_target_distillation_performed": False,
                "action_success_evaluated": False,
                "identity_rebinding_evaluated": False,
                "camera_preservation_evaluated": False,
                "quality_claim": False,
                "scientific_or_action_editing_claim_authorized": False,
                "prior_cross_scene_role_rebinding_negative_job": "130655",
                "this_canary_differs_by_semantically_audited_source_matched_owner": True,
            },
        }
        receipt = donor._rebase_artifact_paths(
            receipt, old_root=artifact_dir, new_root=output_dir
        )
        receipt["receipt_digest"] = scaffold.object_sha256(receipt)
        donor._write_receipt(artifact_dir / "receipt.json", receipt)
        donor._commit_output_transaction(staging=artifact_dir, final=output_dir)
        print(scaffold.canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del source_latent, source_references, owner_clean, generated, initial_noise
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "ARM_SPECS",
    "EXPERIMENTAL_ACK",
    "FRAME_COUNT",
    "METHOD",
    "NUM_INFERENCE_STEPS",
    "REFERENCE_INDICES",
    "ROLE_CLAUSE",
    "RoleReversedOwnerCanaryError",
    "SCHEMA_VERSION",
    "TARGET_SEED",
    "arm_plan",
    "bucket_and_patch_geometry",
    "condition_plan",
    "dynamic_scaffold_audit_contract",
    "main",
    "renderer_body",
    "scaffold_spec",
]
