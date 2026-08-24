#!/usr/bin/env python3
"""Frozen-base 81-frame Bernini C2FR inference through official UniPC.

C2FR (Counterfactual Clean-Field Routing) obtains action, semantic-noop, and
negative branches from the pinned Bernini-R 1.3B renderer at one identical
noisy state.  The raw conditional action-minus-noop clean proposal—the exact
field used by same-state LoRA training—is routed
by ``GeneratorNativeSparseCleanCallback``.  Its temporal-DC null space is
removed before execution and denoising-step saliency is causally smoothed, so
the executed field is

``P * source + G * (source + alpha * Q_t(action_clean - noop_clean))``.

Official APG remains active as an exact integration/parity control, but its
nonlinear guided difference is not substituted for the trained raw field.

The base model is frozen: this entry point has no planner checkpoint, LoRA, or
PEFT API.  It also has no target, mask, tracker, optical flow, pose, trajectory,
or first-frame anchor input.  The tri-branch hook certifies the locally rebuilt
official action APG tensor exactly before every one of 40 original UniPC steps.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


SPT_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = SPT_ROOT.parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as base  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_lora as trainer  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402
from spt_v2 import generator_native_sparse_router as sparse_router  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = "bernini-c2fr-frozen-base-inference-receipt-v2"
METHOD_NAME = "counterfactual-clean-field-routing-c2fr-frozen-base-v2"
NUM_INFERENCE_STEPS = 40
DEFAULT_ALPHA = 1.0
DEFAULT_GENERATE_CAP = 0.12
DEFAULT_ENERGY_COVERAGE = 0.85
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class C2FRInferenceError(RuntimeError):
    """Raised before publication when a frozen-base C2FR invariant differs."""


@dataclass(frozen=True)
class RouterStepRecord:
    """Tensor-free summary of one generator-native hard P/G decision."""

    step_index: int
    timestep: float
    sigma: float
    selected_cell_count: int
    total_cell_count: int
    generate_fraction: float
    max_phase_generate_fraction: float
    active_phase_count: int
    cells_per_phase: int
    integer_capacity_per_phase: int
    per_phase_selected_counts: tuple[int, ...]
    saliency_mean: float
    saliency_max: float
    phase_activity_mean: float
    phase_activity_max: float
    support_sha256: str


@dataclass
class RouterExecutionTrace:
    """All router decisions made by one official Bernini sample call."""

    alpha: float
    config: sparse_router.GeneratorNativeSparseRouterConfig
    records: list[RouterStepRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": sparse_router.METHOD_NAME,
            "alpha": self.alpha,
            "config": asdict(self.config),
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Bernini-R 1.3B C2FR on one exact 81-frame source"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        choices=(NUM_INFERENCE_STEPS,),
        default=NUM_INFERENCE_STEPS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--max-generate-fraction", type=float, default=DEFAULT_GENERATE_CAP
    )
    parser.add_argument(
        "--energy-coverage", type=float, default=DEFAULT_ENERGY_COVERAGE
    )
    parser.add_argument(
        "--expected-bernini-commit", default=trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=trainer.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise C2FRInferenceError(f"{label} must be a finite scalar")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise C2FRInferenceError(f"{label} must be a finite scalar") from error
    if not math.isfinite(numeric):
        raise C2FRInferenceError(f"{label} must be finite")
    return numeric


def validate_cli(args: argparse.Namespace) -> None:
    if (
        not isinstance(args.instruction, str)
        or not args.instruction.strip()
        or "\x00" in args.instruction
    ):
        raise C2FRInferenceError("instruction must be non-empty text without NUL")
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise C2FRInferenceError("C2FR requires exactly 40 official UniPC steps")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise C2FRInferenceError("seed must be an integer in [0,2^63)")
    alpha = _finite_scalar(args.alpha, label="alpha")
    if alpha < 0.0:
        raise C2FRInferenceError("alpha must be non-negative")
    cap = _finite_scalar(
        args.max_generate_fraction, label="max_generate_fraction"
    )
    if not 0.0 < cap <= sparse_router.MAX_ALLOWED_GENERATE_FRACTION:
        raise C2FRInferenceError("max_generate_fraction must lie in (0,0.12]")
    coverage = _finite_scalar(args.energy_coverage, label="energy_coverage")
    if not 0.0 < coverage <= 1.0:
        raise C2FRInferenceError("energy_coverage must lie in (0,1]")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1_RE.fullmatch(value.lower()) is None:
            raise C2FRInferenceError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise C2FRInferenceError(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit.lower() != trainer.BERNINI_OFFICIAL_COMMIT:
        raise C2FRInferenceError("only the audited Bernini commit is supported")
    if args.expected_veomni_commit.lower() != trainer.VEOMNI_TESTED_COMMIT:
        raise C2FRInferenceError("only the tested VeOmni commit is supported")
    if args.expected_checkpoint_tree_sha256 != trainer.CHECKPOINT_TREE_SHA256:
        raise C2FRInferenceError("only the audited Bernini-R 1.3B checkpoint is supported")


def router_config_from_args(
    args: argparse.Namespace,
) -> sparse_router.GeneratorNativeSparseRouterConfig:
    config = sparse_router.GeneratorNativeSparseRouterConfig(
        max_generate_fraction_per_phase=float(args.max_generate_fraction),
        energy_coverage=float(args.energy_coverage),
    )
    try:
        config.validate()
    except sparse_router.GeneratorNativeSparseRouterError as error:
        raise C2FRInferenceError(str(error)) from error
    return config


def exact_sampler_contract(*, seed: int) -> dict[str, Any]:
    contract = base.sampler_contract(steps=NUM_INFERENCE_STEPS, seed=seed)
    if (
        contract["num_frames"] != 81
        or contract["num_inference_steps"] != NUM_INFERENCE_STEPS
        or contract["guidance_mode"] != "v2v_apg"
        or contract["flow_shift"] != 5.0
        or contract["omega_txt"] != 4.0
        or contract["eta"] != 0.5
    ):
        raise C2FRInferenceError("pinned Bernini sampler contract changed")
    return contract


def configure_rank_local_caches(
    environment: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Isolate ROCm compiler databases for each torchrun rank when requested."""

    source = os.environ if environment is None else environment
    root_value = source.get("BERNINI_C2FR_RANK_CACHE_ROOT")
    if not root_value:
        return {}
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise C2FRInferenceError("BERNINI_C2FR_RANK_CACHE_ROOT must be absolute")
    rank_value = source.get("LOCAL_RANK")
    try:
        rank = int(rank_value if rank_value is not None else "")
    except ValueError as error:
        raise C2FRInferenceError("LOCAL_RANK is invalid for rank-local caches") from error
    if not 0 <= rank < base.ULYSSES_SIZE:
        raise C2FRInferenceError("LOCAL_RANK is outside the four-rank C2FR world")
    rank_root = root / f"rank-{rank}"
    paths = {
        "MIOPEN_USER_DB_PATH": rank_root / "miopen-user",
        "MIOPEN_CUSTOM_CACHE_DIR": rank_root / "miopen-custom",
        "TORCH_EXTENSIONS_DIR": rank_root / "torch-extensions",
        "TRITON_CACHE_DIR": rank_root / "triton",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if environment is None:
        for name, path in paths.items():
            os.environ[name] = str(path)
    return {name: str(path) for name, path in paths.items()}


def encode_semantic_noop_prompt(
    renderer: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Construct no-op embeddings through the same frozen Bernini T5 path."""

    import torch

    if tuple(input_ids.shape) != (1, 512) or tuple(attention_mask.shape) != (1, 512):
        raise C2FRInferenceError("semantic no-op token tensors must be [1,512]")
    renderer.t5_text_encoder.to(device)
    renderer.t5_text_encoder.eval()
    try:
        with torch.no_grad():
            embeddings = renderer.encode_prompt(
                input_ids.to(device), attention_mask.to(device)
            )
    finally:
        renderer.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 3
        or int(embeddings.shape[0]) != 1
        or int(embeddings.shape[1]) <= 0
        or int(embeddings.shape[2]) <= 0
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise C2FRInferenceError("Bernini returned invalid semantic no-op embeddings")
    identity = {
        "token_shape": [1, 512],
        "nonpadding_token_count": int(attention_mask.sum().item()),
        "embedding_shape": [int(value) for value in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
        "encoder": "BerniniRendererModel.encode_prompt",
        "frozen_t5": all(
            not parameter.requires_grad
            for parameter in renderer.t5_text_encoder.parameters()
        ),
    }
    if not identity["frozen_t5"]:
        raise C2FRInferenceError("semantic no-op T5 encoder is unexpectedly trainable")
    return embeddings, identity


def _tensor_float(value: Any, *, label: str) -> float:
    try:
        numeric = float(value.detach().float().cpu().item())
    except Exception as error:
        raise C2FRInferenceError(f"cannot serialize router {label}") from error
    if not math.isfinite(numeric):
        raise C2FRInferenceError(f"router {label} is non-finite")
    return numeric


class TracedGeneratorNativeSparseCallback:
    """Inference callback that adds a tensor-free router receipt per step."""

    def __init__(
        self,
        *,
        source_clean: Any,
        layout: tri.PackedLatentLayout,
        config: sparse_router.GeneratorNativeSparseRouterConfig,
        alpha: float,
    ) -> None:
        self.inner = sparse_router.GeneratorNativeSparseCleanCallback(
            source_clean=source_clean,
            layout=layout,
            config=config,
            alpha=alpha,
        )
        self.trace = RouterExecutionTrace(alpha=float(alpha), config=config)

    def __call__(self, fields: tri.CleanFieldStep) -> Any:
        import torch

        result = self.inner(fields)
        execution = self.inner.last_execution
        if execution is None:
            raise C2FRInferenceError("sparse callback did not retain its execution")
        diagnostics = execution.plan.diagnostics
        if not isinstance(diagnostics, Mapping):
            raise C2FRInferenceError("sparse plan lacks routing diagnostics")
        support = diagnostics.get("selected_support")
        saliency = diagnostics.get("motion_saliency")
        activity = diagnostics.get("phase_activity_energy")
        if (
            not isinstance(support, torch.Tensor)
            or support.dtype != torch.bool
            or support.ndim != 4
            or int(support.shape[0]) != 1
            or int(support.shape[1]) != base.LATENT_FRAME_COUNT
            or not isinstance(saliency, torch.Tensor)
            or tuple(saliency.shape) != tuple(support.shape)
            or not isinstance(activity, torch.Tensor)
            or tuple(activity.shape) != (1, base.LATENT_FRAME_COUNT)
        ):
            raise C2FRInferenceError("sparse router diagnostics have invalid shapes")
        _, phases, height, width = map(int, support.shape)
        cells_per_phase = height * width
        selected_counts_tensor = support.sum(dim=(-1, -2))[0]
        selected_counts = tuple(
            int(value) for value in selected_counts_tensor.detach().cpu().tolist()
        )
        selected = sum(selected_counts)
        total = phases * cells_per_phase
        capacity = int(
            math.floor(
                float(self.trace.config.max_generate_fraction_per_phase)
                * cells_per_phase
            )
        )
        selected_indices = (
            torch.nonzero(support.reshape(-1), as_tuple=False)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        support_sha256 = base.object_sha256(
            {
                "shape": [int(value) for value in support.shape],
                "selected_flat_indices": [int(value) for value in selected_indices],
            }
        )
        active = diagnostics.get("active_phases")
        if not isinstance(active, torch.Tensor) or tuple(active.shape) != (1, phases):
            raise C2FRInferenceError("sparse router active-phase diagnostic differs")
        self.trace.records.append(
            RouterStepRecord(
                step_index=int(fields.step_index),
                timestep=float(fields.timestep),
                sigma=float(fields.sigma),
                selected_cell_count=selected,
                total_cell_count=total,
                generate_fraction=selected / total,
                max_phase_generate_fraction=max(selected_counts) / cells_per_phase,
                active_phase_count=int(active.sum().detach().cpu().item()),
                cells_per_phase=cells_per_phase,
                integer_capacity_per_phase=capacity,
                per_phase_selected_counts=selected_counts,
                saliency_mean=_tensor_float(saliency.mean(), label="saliency mean"),
                saliency_max=_tensor_float(saliency.max(), label="saliency max"),
                phase_activity_mean=_tensor_float(
                    activity.mean(), label="phase activity mean"
                ),
                phase_activity_max=_tensor_float(
                    activity.max(), label="phase activity max"
                ),
                support_sha256=support_sha256,
            )
        )
        return result


def validate_execution_trace(
    tri_trace: tri.TriBranchTrace,
    router_trace: RouterExecutionTrace,
) -> dict[str, Any]:
    """Require forty exact APG certificates and forty bounded router steps."""

    if not isinstance(tri_trace, tri.TriBranchTrace):
        raise C2FRInferenceError("tri_trace must be a TriBranchTrace")
    if not isinstance(router_trace, RouterExecutionTrace):
        raise C2FRInferenceError("router_trace must be a RouterExecutionTrace")
    tri_records = list(tri_trace.records)
    routing_records = list(router_trace.records)
    if tri_trace.sample_calls != 1:
        raise C2FRInferenceError("tri-branch hook must observe exactly one sample call")
    if len(tri_records) != NUM_INFERENCE_STEPS or len(routing_records) != NUM_INFERENCE_STEPS:
        raise C2FRInferenceError("C2FR must certify and route all 40 UniPC steps")
    cap = float(router_trace.config.max_generate_fraction_per_phase)
    if not 0.0 < cap <= sparse_router.MAX_ALLOWED_GENERATE_FRACTION:
        raise C2FRInferenceError("router trace exceeds the 0.12 generate cap")
    sigmas: list[float] = []
    for expected_index, (branch, routing) in enumerate(
        zip(tri_records, routing_records)
    ):
        if branch.step_index != expected_index or routing.step_index != expected_index:
            raise C2FRInferenceError("C2FR step indices are incomplete or reordered")
        if branch.model_id != "transformer_1":
            raise C2FRInferenceError("frozen Bernini-R 1.3B must remain single-expert")
        if (
            branch.transformer_forwards != 3
            or branch.shared_negative_forwards != 1
            or branch.action_forwards != 1
            or branch.noop_forwards != 1
            or branch.original_scheduler_calls != 1
        ):
            raise C2FRInferenceError("each step must use three branches and one original UniPC call")
        if (
            branch.official_action_exact_parity is not True
            or branch.official_action_parity_rms_error != 0.0
            or branch.official_action_parity_max_abs_error != 0.0
        ):
            raise C2FRInferenceError("official action APG exact certificate failed")
        if branch.effective_guidance_scale != base.OMEGA_TEXT:
            raise C2FRInferenceError("action/no-op APG guidance scale differs")
        for label, value in (
            ("sigma", branch.sigma),
            ("callback correction", branch.callback_correction_rms),
            ("raw action-noop delta", branch.raw_action_noop_delta_rms),
            ("guided action-noop delta", branch.guided_action_noop_delta_rms),
            ("guided action-noop L2", branch.guided_action_noop_delta_l2),
        ):
            if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
                raise C2FRInferenceError(f"tri-branch {label} diagnostic is invalid")
        if branch.sigma <= 0.0:
            raise C2FRInferenceError("every intercepted UniPC sigma must be positive")
        if not math.isclose(branch.sigma, routing.sigma, rel_tol=0.0, abs_tol=1e-8):
            raise C2FRInferenceError("tri-branch and router sigma traces differ")
        if not math.isclose(branch.timestep, routing.timestep, rel_tol=0.0, abs_tol=1e-6):
            raise C2FRInferenceError("tri-branch and router timestep traces differ")
        if (
            len(routing.per_phase_selected_counts) != base.LATENT_FRAME_COUNT
            or routing.total_cell_count
            != base.LATENT_FRAME_COUNT * routing.cells_per_phase
            or routing.selected_cell_count != sum(routing.per_phase_selected_counts)
            or routing.integer_capacity_per_phase
            != int(math.floor(cap * routing.cells_per_phase))
            or any(
                count < 0 or count > routing.integer_capacity_per_phase
                for count in routing.per_phase_selected_counts
            )
        ):
            raise C2FRInferenceError("router support cardinality certificate differs")
        expected_fraction = routing.selected_cell_count / routing.total_cell_count
        expected_phase_max = (
            max(routing.per_phase_selected_counts) / routing.cells_per_phase
        )
        if (
            not math.isclose(
                routing.generate_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                routing.max_phase_generate_fraction,
                expected_phase_max,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or routing.max_phase_generate_fraction > cap + 1e-12
            or not 0 <= routing.active_phase_count <= base.LATENT_FRAME_COUNT
            or _SHA256_RE.fullmatch(routing.support_sha256) is None
        ):
            raise C2FRInferenceError("router support summary is invalid")
        for value in (
            routing.saliency_mean,
            routing.saliency_max,
            routing.phase_activity_mean,
            routing.phase_activity_max,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise C2FRInferenceError("router energy summary is invalid")
        sigmas.append(float(branch.sigma))
    if any(following >= current for current, following in zip(sigmas, sigmas[1:])):
        raise C2FRInferenceError("official UniPC sigma trace must be strictly descending")
    payload = {
        "tri_branch": tri_trace.as_dict(),
        "router": router_trace.as_dict(),
        "certificate": {
            "step_count": NUM_INFERENCE_STEPS,
            "official_action_apg_exact_steps": NUM_INFERENCE_STEPS,
            "original_unipc_calls": NUM_INFERENCE_STEPS,
            "transformer_forwards": 3 * NUM_INFERENCE_STEPS,
            "generate_gate_application_count": 1,
            "max_generate_fraction_per_phase": cap,
            "custom_integrator": False,
        },
    }
    payload["trace_digest"] = base.object_sha256(payload)
    return payload


def _method_hashes() -> dict[str, str]:
    paths = {
        "spt_v2/infer_c2fr.py": SPT_ROOT / "infer_c2fr.py",
        "spt_v2/generator_native_sparse_router.py": SPT_ROOT
        / "generator_native_sparse_router.py",
        "spt_v2/counterfactual_clean_field.py": SPT_ROOT
        / "counterfactual_clean_field.py",
        "spt_v2/phase_transport.py": SPT_ROOT / "phase_transport.py",
        "tri_branch_unipc.py": METHOD_ROOT / "tri_branch_unipc.py",
        "infer_lora.py": METHOD_ROOT / "infer_lora.py",
        "motion_residual.py": METHOD_ROOT / "motion_residual.py",
    }
    return {name: base.file_sha256(path) for name, path in paths.items()}


def build_inference_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    output_path: Path,
    output_sha256: str,
    noop_identity: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    wan_diffusion_path: Path,
    wan_diffusion_sha256: str,
    runtime_versions: Mapping[str, str],
) -> dict[str, Any]:
    instruction_bytes = args.instruction.encode("utf-8")
    noop_bytes = motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
    router_config = router_config_from_args(args)
    receipt: dict[str, Any] = {
        "schema_version": INFERENCE_RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "method_files_sha256": _method_hashes(),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "bernini_inference_files": dict(inference_file_hashes),
        "wan_diffusion": {
            "path": str(wan_diffusion_path),
            "sha256": wan_diffusion_sha256,
            "validated_at_hook_installation": True,
        },
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "base_model": {
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "bernini_r_1p3b": True,
            "frozen": True,
            "planner_checkpoint_loaded": False,
            "lora_or_peft_loaded": False,
            "adapter_loaded": False,
        },
        "input": {
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "instruction_utf8_sha256": hashlib.sha256(instruction_bytes).hexdigest(),
            "instruction_utf8_bytes": len(instruction_bytes),
            "accepted_external_conditions": ["source_video", "edit_instruction"],
            "semantic_noop_is_internal_fixed_control": True,
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "external_mask_track_pose_flow_trajectory": False,
            "first_frame_anchor": False,
        },
        "preprocessing": dict(source_metadata),
        "prompt_contract": {
            "task": "mv2v",
            "action_system_prompt_sha256": hashlib.sha256(
                base.MV2V_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "semantic_noop_instruction_sha256": hashlib.sha256(noop_bytes).hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(
                base.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
            "action_noop_negative_use_frozen_t5": True,
            "semantic_noop": dict(noop_identity),
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "tokenizer_padding_side": "right",
            "fixed_token_shape": [1, 512],
        },
        "sampling": {
            **exact_sampler_contract(seed=args.seed),
            "single_expert": "transformer_1",
            "ulysses_size": base.ULYSSES_SIZE,
            "rank0_decode_and_save_only": True,
            "tri_branch_contract": tri.sampler_contract(),
            "routing_contract": sparse_router.runtime_contract(),
            "router_config": asdict(router_config),
            "alpha": float(args.alpha),
            "projection_boundary": "after_exact_action_apg_before_original_unipc_step",
            "custom_integrator": False,
        },
        "execution_trace": dict(execution_trace),
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "frame_count": base.FRAME_COUNT,
            "fps": base.FPS,
            "height": source_metadata["source_derived_bucket_hw"][0],
            "width": source_metadata["source_derived_bucket_hw"][1],
            "audio_preserved": False,
        },
        "runtime_versions": dict(runtime_versions),
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = base.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    configure_rank_local_caches()
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise C2FRInferenceError("source video must be an absolute path")
    try:
        source_path = base._plain_file(
            source_requested.resolve(strict=True), label="source video"
        )
        output_path, receipt_path = base._resolve_output(args.output)
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = trainer.validate_checkpoint(args.checkpoint)
        inference_file_hashes = base.validate_inference_source_files(bernini_root)
    except (base.InferenceContractError, trainer.TrainingContractError) as error:
        raise C2FRInferenceError(str(error)) from error
    if transformer_config["num_attention_heads"] % base.ULYSSES_SIZE:
        raise C2FRInferenceError("Bernini-R 1.3B heads are not divisible by Ulysses=4")
    wan_diffusion_path = (
        bernini_root / "bernini/models/wan_diffusion.py"
    ).resolve(strict=True)
    try:
        wan_diffusion_sha256 = tri.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
        )
    except tri.TriBranchHookError as error:
        raise C2FRInferenceError(str(error)) from error
    trainer.activate_source_trees(bernini_root, veomni_root)

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
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != base.MV2V_SYSTEM_PROMPT:
        raise C2FRInferenceError("runtime Bernini mv2v system prompt differs")
    if DEFAULT_NEG_PROMPT != base.DEFAULT_NEGATIVE_PROMPT:
        raise C2FRInferenceError("runtime Bernini negative prompt differs")
    distributed = base.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise C2FRInferenceError("C2FR requires four AUH ROCm-visible GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    try:
        source_tensor, source_metadata = base.prepare_exact_source(source_path)
    except base.InferenceContractError as error:
        raise C2FRInferenceError(str(error)) from error
    source_sha256 = base.file_sha256(source_path)
    action_prompt = base.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = base.build_training_prompt(
        motion.DEFAULT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **base.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.TrainingContractError as error:
        raise C2FRInferenceError(str(error)) from error
    if float(config.shift) != base.FLOW_SHIFT or config.use_unipc is not True:
        raise C2FRInferenceError("renderer must use official UniPC with flow shift 5")
    model = BerniniRendererModel(config)
    if any("lora_" in name.lower() for name, _ in model.named_modules()):
        raise C2FRInferenceError("frozen base unexpectedly contains LoRA modules")
    model.requires_grad_(False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **base.tokenizer_load_kwargs()
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise C2FRInferenceError("tokenizer lost fix_mistral_regex/right-padding")
    action_ids, action_mask = base._tokenize_training_prompt(tokenizer, action_prompt)
    noop_ids, noop_mask = base._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = base._tokenize_renderer_negative(
        tokenizer, base.DEFAULT_NEGATIVE_PROMPT
    )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    bucket = source_metadata["source_derived_bucket_hw"]
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        base.LATENT_FRAME_COUNT,
        int(bucket[0]) // 8,
        int(bucket[1]) // 8,
    )
    if tuple(int(value) for value in source_latent.shape) != expected_latent_shape:
        raise C2FRInferenceError("source VAE latent differs from exact 81f geometry")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    noop_embeddings, noop_identity = encode_semantic_noop_prompt(
        model, noop_ids, noop_mask, device=device
    )
    layout = tri.PackedLatentLayout.from_spatial_shape(expected_latent_shape)
    router_config = router_config_from_args(args)
    callback = TracedGeneratorNativeSparseCallback(
        source_clean=source_latent,
        layout=layout,
        config=router_config,
        alpha=float(args.alpha),
    )
    sampling = exact_sampler_contract(seed=args.seed)
    try:
        with tri.tri_branch_unipc_hook(
            model,
            noop_prompt_embeds=noop_embeddings,
            latent_shape=expected_latent_shape,
            clean_field_callback=callback,
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
            expected_steps=NUM_INFERENCE_STEPS,
            expected_flow_shift=base.FLOW_SHIFT,
        ) as tri_trace:
            with torch.no_grad():
                generated_latent = model.sample(
                    input_ids=action_ids.to(device),
                    attention_mask=action_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    image_vae_latents=None,
                    multi_video_vae_latents=[source_latent],
                    multi_image_vae_latents=None,
                    width=int(bucket[1]),
                    height=int(bucket[0]),
                    device=device,
                    **sampling,
                )
    except (
        tri.TriBranchHookError,
        sparse_router.GeneratorNativeSparseRouterError,
    ) as error:
        raise C2FRInferenceError(str(error)) from error
    execution_trace = validate_execution_trace(tri_trace, callback.trace)
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise C2FRInferenceError("generated latent differs from exact 81f geometry")
    model.to("cpu")
    del noop_embeddings, callback, source_latent
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        expected_output_shape = (
            base.FRAME_COUNT,
            int(bucket[0]),
            int(bucket[1]),
            3,
        )
        if tuple(int(value) for value in output.shape) != expected_output_shape:
            raise C2FRInferenceError("decoded output differs from exact 81f geometry")
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise C2FRInferenceError(f"stale temporary output exists: {temporary_output}")
        save_output(output, str(temporary_output), fps=int(base.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
            output_path
        )
        try:
            base.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        except base.InferenceContractError as error:
            raise C2FRInferenceError(str(error)) from error
        if tuple(encoded_hw) != tuple(bucket):
            raise C2FRInferenceError("encoded output geometry differs from source bucket")
        receipt = build_inference_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            output_path=output_path,
            output_sha256=base.file_sha256(output_path),
            noop_identity=noop_identity,
            execution_trace=execution_trace,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            inference_file_hashes=inference_file_hashes,
            wan_diffusion_path=wan_diffusion_path,
            wan_diffusion_sha256=wan_diffusion_sha256,
            runtime_versions={
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
        )
        base._atomic_write_json(receipt_path, receipt)
        print(base.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
