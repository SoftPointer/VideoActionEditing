#!/usr/bin/env python3
"""Exact-81f SPT-v2 inference through Bernini's official v2v_apg UniPC.

The deployable path has exactly two semantic conditions: the source video and
the user's raw edit instruction.  A strictly loaded ``phase_query_v2`` planner
predicts a dense plan from the clean source latent and the full unpadded T5
token sequence.  ``unipc_projection.project_unipc_steps`` executes that plan
after Bernini has completed CFG/APG and immediately before the original UniPC
``scheduler.step``.  Bernini's solver is never replaced.

This entry point intentionally has no adapter argument and never imports PEFT:
old CDF/P3T/action LoRA weights cannot enter the formal SPT evaluation.  Paired
targets and oracle plans are likewise absent.  Offline oracle diagnostics, if
ever run, must use a separate explicitly labelled harness.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


SPT_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = SPT_ROOT.parent
for root in (METHOD_ROOT, SPT_ROOT.parent):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import infer_lora as base  # noqa: E402
import train_lora as trainer  # noqa: E402
from spt_v2 import phase_query_planner as phase_query  # noqa: E402
from spt_v2 import phase_transport as spt  # noqa: E402
from spt_v2 import train_student as student_train  # noqa: E402
from spt_v2 import unipc_projection as projection  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = "bernini-spt-v2-inference-receipt-v1"
METHOD_NAME = "self-predicted-phase-transport-phase-query-v2"
NUM_INFERENCE_STEPS = 40
MAX_GENERATE_FRACTION = 0.12
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SPTInferenceError(RuntimeError):
    """Raised before output publication when an SPT contract differs."""


@dataclass(frozen=True)
class PlannerBundle:
    root: Path
    weights_path: Path
    config_path: Path
    receipt_path: Path
    optimizer_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run source+instruction-only phase_query_v2 SPT inference"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--planner-checkpoint", required=True)
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
    parser.add_argument(
        "--max-generate-fraction",
        type=float,
        default=MAX_GENERATE_FRACTION,
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


def validate_cli(args: argparse.Namespace) -> None:
    if not isinstance(args.instruction, str) or not args.instruction.strip() or "\x00" in args.instruction:
        raise SPTInferenceError("instruction must be non-empty text without NUL")
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise SPTInferenceError("formal SPT inference requires exactly 40 UniPC steps")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise SPTInferenceError("seed must be an integer in [0,2^63)")
    if args.max_generate_fraction != MAX_GENERATE_FRACTION:
        raise SPTInferenceError(
            "formal SPT inference fixes max-generate-fraction to 0.12"
        )
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1_RE.fullmatch(value.lower()) is None:
            raise SPTInferenceError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise SPTInferenceError(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit.lower() != trainer.BERNINI_OFFICIAL_COMMIT:
        raise SPTInferenceError("only the audited Bernini commit is supported")
    if args.expected_veomni_commit.lower() != trainer.VEOMNI_TESTED_COMMIT:
        raise SPTInferenceError("only the tested VeOmni commit is supported")
    if args.expected_checkpoint_tree_sha256 != trainer.CHECKPOINT_TREE_SHA256:
        raise SPTInferenceError("only the audited Bernini-R 1.3B checkpoint is supported")


def _absolute_plain_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise SPTInferenceError(f"{label} must be absolute: {path}")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise SPTInferenceError(f"cannot access {label}: {path}: {error}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise SPTInferenceError(f"{label} must be a plain non-symlink file: {path}")
    return path.resolve(strict=True)


def resolve_planner_checkpoint(value: str | Path) -> PlannerBundle:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise SPTInferenceError("planner checkpoint must be an absolute directory")
    if requested.is_symlink():
        raise SPTInferenceError("planner checkpoint directory may not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise SPTInferenceError(f"cannot resolve planner checkpoint: {error}") from error
    if not root.is_dir():
        raise SPTInferenceError("planner checkpoint is not a directory")
    forbidden = (
        root / "adapter",
        root / "adapter_config.json",
        root / "adapter_model.safetensors",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise SPTInferenceError(
            "planner checkpoint contains a LoRA/PEFT adapter; CDF/P3T adapters are forbidden"
        )
    return PlannerBundle(
        root=root,
        weights_path=_absolute_plain_file(
            root / "planner.safetensors", label="planner weights"
        ),
        config_path=_absolute_plain_file(
            root / "planner_config.json", label="planner config"
        ),
        receipt_path=_absolute_plain_file(root / "receipt.json", label="planner receipt"),
        optimizer_path=_absolute_plain_file(
            root / "optimizer.pt", label="planner optimizer provenance"
        ),
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SPTInferenceError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SPTInferenceError(f"{label} must contain one JSON object")
    return value


def validate_planner_metadata(
    raw_config: Mapping[str, Any], receipt: Mapping[str, Any]
) -> tuple[phase_query.PhaseQueryPlannerConfig, dict[str, Any]]:
    """Bind phase_query_v2 config to the hash-protected training receipt."""

    expected_config_keys = set(asdict(phase_query.PhaseQueryPlannerConfig()))
    if set(raw_config) != expected_config_keys:
        raise SPTInferenceError(
            "planner config fields differ from phase_query_v2; legacy planner is forbidden"
        )
    try:
        config = phase_query.PhaseQueryPlannerConfig(**dict(raw_config))
        config.validate()
    except (TypeError, spt.PhaseTransportError) as error:
        raise SPTInferenceError(f"invalid phase_query_v2 planner config: {error}") from error
    if (
        config.architecture != phase_query.ARCHITECTURE_NAME
        or config.latent_channels != 64
        or config.latent_phases != base.LATENT_FRAME_COUNT
        or config.cross_attention_layers != phase_query.CROSS_ATTENTION_LAYERS
        or config.source_bank_detach is not True
    ):
        raise SPTInferenceError("planner config violates the formal phase_query_v2 scope")

    candidate = dict(receipt)
    declared_receipt_digest = candidate.pop("receipt_digest", None)
    if (
        not isinstance(declared_receipt_digest, str)
        or _SHA256_RE.fullmatch(declared_receipt_digest) is None
        or base.object_sha256(candidate) != declared_receipt_digest
    ):
        raise SPTInferenceError("planner training receipt digest differs")
    if (
        receipt.get("schema_version") != student_train.RECEIPT_SCHEMA
        or receipt.get("method") != student_train.METHOD_NAME
        or receipt.get("planner", {}).get("architecture") != phase_query.ARCHITECTURE_NAME
    ):
        raise SPTInferenceError(
            "checkpoint is not a phase_query_v2 student training checkpoint"
        )
    global_step = receipt.get("global_step")
    if type(global_step) is not int or global_step <= 0:
        raise SPTInferenceError("planner global_step must be positive")

    immutable = receipt.get("immutable_contract")
    if not isinstance(immutable, dict) or set(immutable) != {"value", "digest"}:
        raise SPTInferenceError("planner receipt lacks the immutable training contract")
    immutable_value = immutable.get("value")
    if not isinstance(immutable_value, dict) or immutable.get("digest") != base.object_sha256(
        immutable_value
    ):
        raise SPTInferenceError("planner immutable-contract digest differs")
    expected_config_json = json.loads(
        base.canonical_json_bytes(asdict(config)).decode("utf-8")
    )
    immutable_expected = {
        "method": student_train.METHOD_NAME,
        "planner_architecture": phase_query.ARCHITECTURE_NAME,
        "checkpoint_tree_sha256": trainer.CHECKPOINT_TREE_SHA256,
        "planner_config": expected_config_json,
        "student_semantic_inputs": ["source_video", "edit_instruction"],
        "instruction_representation": "full_unpadded_t5_token_sequence",
        "instruction_pooling": None,
        "phase_query_count": base.LATENT_FRAME_COUNT,
        "cross_attention_layers": phase_query.CROSS_ATTENTION_LAYERS,
        "target_used_by_student": False,
        "target_used_by_training_teacher_only": True,
    }
    for key, expected in immutable_expected.items():
        if immutable_value.get(key) != expected:
            raise SPTInferenceError(
                f"planner immutable contract differs for {key}: {immutable_value.get(key)!r}"
            )

    supervision = receipt.get("supervision")
    expected_supervision = {
        "student_api": ["source", "instruction_tokens"],
        "instruction_representation": "full_unpadded_t5_token_sequence",
        "instruction_pooling": None,
        "learned_phase_queries": base.LATENT_FRAME_COUNT,
        "explicit_sinusoidal_phase_encoding": True,
        "cross_attention_layers": phase_query.CROSS_ATTENTION_LAYERS,
        "student_target_argument_exists": False,
        "target_used_by_oracle_teacher_only": True,
        "external_mask_track_pose_flow": False,
        "max_generate_fraction_per_phase": MAX_GENERATE_FRACTION,
        "generate_budget_reject_fallback": "preserve",
        "latent_phases": base.LATENT_FRAME_COUNT,
    }
    if not isinstance(supervision, dict):
        raise SPTInferenceError("planner receipt lacks supervision contract")
    for key, expected in expected_supervision.items():
        if supervision.get(key) != expected:
            raise SPTInferenceError(
                f"planner supervision contract differs for {key}: {supervision.get(key)!r}"
            )
    if receipt.get("production_claim_forbidden") is not True:
        raise SPTInferenceError("planner receipt lost production-claim restriction")
    if receipt.get("scientific_claim_authorized") is not False:
        raise SPTInferenceError("planner receipt contains an unsupported scientific claim")

    planner_section = receipt.get("planner")
    if not isinstance(planner_section, dict):
        raise SPTInferenceError("planner receipt lacks planner parameter metadata")
    parameter_names = planner_section.get("parameter_names")
    if (
        not isinstance(parameter_names, list)
        or not parameter_names
        or not all(isinstance(name, str) for name in parameter_names)
        or len(parameter_names) != len(set(parameter_names))
        or planner_section.get("parameter_names_sha256")
        != base.object_sha256(parameter_names)
    ):
        raise SPTInferenceError("planner parameter-name receipt is invalid")
    parameter_count = planner_section.get("parameter_count")
    if type(parameter_count) is not int or parameter_count <= 0:
        raise SPTInferenceError("planner parameter count must be positive")
    return config, {
        "global_step": global_step,
        "receipt_digest": declared_receipt_digest,
        "immutable_digest": immutable["digest"],
        "parameter_names": list(parameter_names),
        "parameter_count": parameter_count,
        "diagnostic_subset": bool(receipt.get("dataset", {}).get("diagnostic_subset")),
    }


def strict_load_planner(
    bundle: PlannerBundle,
    config: phase_query.PhaseQueryPlannerConfig,
    identity: Mapping[str, Any],
    *,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Strictly reload every phase_query_v2 tensor; never read optimizer state."""

    import torch
    from safetensors.torch import load_file

    planner = phase_query.PhaseQueryPlanner(config)
    expected_state = planner.state_dict()
    saved = load_file(str(bundle.weights_path), device="cpu")
    if set(saved) != set(expected_state):
        missing = sorted(set(expected_state) - set(saved))
        unexpected = sorted(set(saved) - set(expected_state))
        raise SPTInferenceError(
            f"planner state-key scope differs: missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    for key, expected in expected_state.items():
        actual = saved[key]
        if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
            raise SPTInferenceError(f"planner tensor metadata differs for {key}")
    planner.load_state_dict(saved, strict=True)
    loaded = planner.state_dict()
    unequal = [
        key for key in sorted(saved) if not bool(torch.equal(saved[key], loaded[key].cpu()))
    ]
    if unequal:
        raise SPTInferenceError(f"strict planner tensor reload differs: {unequal[:4]}")
    actual_parameter_names = [name for name, _ in planner.named_parameters()]
    actual_parameter_count = sum(int(parameter.numel()) for _, parameter in planner.named_parameters())
    if actual_parameter_names != identity["parameter_names"]:
        raise SPTInferenceError("runtime planner parameter names differ from receipt")
    if actual_parameter_count != identity["parameter_count"]:
        raise SPTInferenceError("runtime planner parameter count differs from receipt")
    planner.requires_grad_(False)
    planner.eval()
    planner.to(device)
    augmented = dict(identity)
    augmented.update(
        {
            "weights_path": str(bundle.weights_path),
            "weights_sha256": base.file_sha256(bundle.weights_path),
            "state_key_count": len(saved),
            "state_keys_sha256": base.object_sha256(sorted(saved)),
            "strictly_reloaded": True,
            "optimizer_loaded": False,
        }
    )
    return planner, augmented


def pack_clean_source(source_latent: Any) -> tuple[Any, Any, int, int]:
    """Pack normalized Wan VAE latents exactly like official ``_to_packed``."""

    from einops import rearrange

    if getattr(source_latent, "ndim", None) != 5:
        raise SPTInferenceError("source latent must be [B,C,T,H,W]")
    batch, channels, phases, latent_height, latent_width = map(int, source_latent.shape)
    if batch != 1 or channels * 4 != 64 or phases != base.LATENT_FRAME_COUNT:
        raise SPTInferenceError("source latent differs from exact Bernini 81f packed geometry")
    if latent_height <= 0 or latent_width <= 0 or latent_height % 2 or latent_width % 2:
        raise SPTInferenceError("source latent spatial dimensions must be positive and even")
    packed = rearrange(
        source_latent,
        "b c t (h ph) (w pw) -> b (t h w) (ph pw c)",
        ph=2,
        pw=2,
    ).contiguous()
    height, width = latent_height // 2, latent_width // 2
    source_video = spt.packed_to_video(packed, height=height, width=width)
    if tuple(source_video.shape) != (1, 21, height, width, 64):
        raise SPTInferenceError("SPT source video view differs from [1,21,H,W,64]")
    return packed, source_video, height, width


def encode_student_instruction(
    renderer: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    device: Any,
) -> Any:
    """Encode and retain the complete non-padding T5 sequence, without pooling."""

    import torch

    if tuple(input_ids.shape) != (1, 512) or tuple(attention_mask.shape) != (1, 512):
        raise SPTInferenceError("student instruction IDs must use Bernini's fixed [1,512] input")
    renderer.t5_text_encoder.to(device)
    renderer.t5_text_encoder.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        padded = renderer.encode_prompt(input_ids.to(device), attention_mask.to(device))
    try:
        tokens = student_train._unpadded_text_tokens([512], padded)
    except student_train.StudentTrainingError as error:
        raise SPTInferenceError(str(error)) from error
    renderer.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()
    if int(tokens.shape[0]) != 1 or int(tokens.shape[1]) <= 0:
        raise SPTInferenceError("student T5 token sequence is empty")
    return tokens.to(device)


def exact_sampler_contract(*, seed: int) -> dict[str, Any]:
    contract = base.sampler_contract(steps=NUM_INFERENCE_STEPS, seed=seed)
    if (
        contract["num_frames"] != 81
        or contract["guidance_mode"] != "v2v_apg"
        or contract["flow_shift"] != 5.0
        or contract["num_inference_steps"] != 40
    ):
        raise SPTInferenceError("base Bernini sampler contract unexpectedly changed")
    return contract


def validate_projection_trace(trace: projection.ProjectionTrace) -> dict[str, Any]:
    records = list(trace.records)
    if len(records) != NUM_INFERENCE_STEPS:
        raise SPTInferenceError(
            f"SPT projection must intercept all 40 UniPC steps, got {len(records)}"
        )
    if trace.max_generate_fraction != MAX_GENERATE_FRACTION or trace.oracle_ablation:
        raise SPTInferenceError("formal trace lost its capped student-plan contract")
    sigmas: list[float] = []
    for expected_index, record in enumerate(records):
        if record.step_index != expected_index:
            raise SPTInferenceError("UniPC trace step indices are incomplete or reordered")
        if not record.projection_applied or not math.isfinite(record.sigma) or record.sigma <= 0.0:
            raise SPTInferenceError("every official denoising step must execute a finite SPT projection")
        if not math.isfinite(record.correction_rms) or record.correction_rms < 0.0:
            raise SPTInferenceError("projection correction RMS is invalid")
        fractions = (
            record.preserve_fraction,
            record.transport_fraction,
            record.generate_fraction,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise SPTInferenceError("P/T/G trace fractions are invalid")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=2e-5):
            raise SPTInferenceError("P/T/G trace fractions do not sum to one")
        if (
            record.generate_budget != MAX_GENERATE_FRACTION
            or record.max_phase_generate_fraction > MAX_GENERATE_FRACTION + 2e-5
        ):
            raise SPTInferenceError("trace exceeds the 0.12 per-phase generate budget")
        sigmas.append(float(record.sigma))
    if any(following >= current for current, following in zip(sigmas, sigmas[1:])):
        raise SPTInferenceError("UniPC sigma trace must be strictly descending")
    payload = trace.as_dict()
    if payload.get("step_count") != NUM_INFERENCE_STEPS:
        raise SPTInferenceError("serialized SPT trace step count differs")
    return payload


def _method_hashes() -> dict[str, str]:
    paths = {
        "spt_v2/infer_spt.py": SPT_ROOT / "infer_spt.py",
        "spt_v2/phase_query_planner.py": SPT_ROOT / "phase_query_planner.py",
        "spt_v2/phase_transport.py": SPT_ROOT / "phase_transport.py",
        "spt_v2/unipc_projection.py": SPT_ROOT / "unipc_projection.py",
        "infer_lora.py": METHOD_ROOT / "infer_lora.py",
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
    planner_bundle: PlannerBundle,
    planner_identity: Mapping[str, Any],
    planner_config: phase_query.PhaseQueryPlannerConfig,
    instruction_token_count: int,
    trace: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
) -> dict[str, Any]:
    instruction_bytes = args.instruction.encode("utf-8")
    receipt: dict[str, Any] = {
        "schema_version": INFERENCE_RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "method_files_sha256": _method_hashes(),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "bernini_inference_files": dict(inference_file_hashes),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "base_model": {
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "bernini_r_1p3b": True,
            "peft_or_lora_loaded": False,
            "cdf_adapter_loaded": False,
            "p3t_adapter_loaded": False,
        },
        "planner": {
            "checkpoint_root": str(planner_bundle.root),
            "architecture": phase_query.ARCHITECTURE_NAME,
            "config": asdict(planner_config),
            **dict(planner_identity),
        },
        "input": {
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "instruction_utf8_sha256": hashlib.sha256(instruction_bytes).hexdigest(),
            "instruction_utf8_bytes": len(instruction_bytes),
            "accepted_model_conditions": ["source_video", "edit_instruction"],
            "student_planner_inputs": ["source", "full_unpadded_instruction_tokens"],
            "instruction_token_count": int(instruction_token_count),
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "oracle_plan_loaded": False,
            "external_mask_track_pose_flow_trajectory": False,
            "first_frame_anchor": False,
        },
        "preprocessing": dict(source_metadata),
        "prompt_contract": {
            "task": "mv2v",
            "raw_instruction_is_only_user_text": True,
            "system_prompt_sha256": hashlib.sha256(
                base.MV2V_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "instruction_representation": "full_unpadded_t5_token_sequence",
            "instruction_pooling": None,
        },
        "sampling": {
            **exact_sampler_contract(seed=args.seed),
            "single_expert": "transformer_1",
            "ulysses_size": base.ULYSSES_SIZE,
            "rank0_decode_and_save_only": True,
            "projection_boundary": "after_cfg_apg_before_original_unipc_step",
            "custom_euler_integrator": False,
            "max_generate_fraction_per_phase": MAX_GENERATE_FRACTION,
        },
        "projection_trace": dict(trace),
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
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise SPTInferenceError("source video must be an absolute path")
    source_path = _absolute_plain_file(source_requested, label="source video")
    try:
        output_path, receipt_path = base._resolve_output(args.output)
    except base.InferenceContractError as error:
        raise SPTInferenceError(str(error)) from error
    planner_bundle = resolve_planner_checkpoint(args.planner_checkpoint)
    planner_config_raw = _read_json(planner_bundle.config_path, label="planner config")
    planner_receipt = _read_json(planner_bundle.receipt_path, label="planner receipt")
    planner_config, planner_identity = validate_planner_metadata(
        planner_config_raw, planner_receipt
    )

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = trainer.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = trainer.validate_checkpoint(args.checkpoint)
        inference_file_hashes = base.validate_inference_source_files(bernini_root)
    except (trainer.TrainingContractError, base.InferenceContractError) as error:
        raise SPTInferenceError(str(error)) from error
    if transformer_config["num_attention_heads"] % base.ULYSSES_SIZE:
        raise SPTInferenceError("Bernini-R 1.3B attention heads are not divisible by Ulysses=4")
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
        raise SPTInferenceError("runtime Bernini mv2v prompt differs from planner training")
    if DEFAULT_NEG_PROMPT != base.DEFAULT_NEGATIVE_PROMPT:
        raise SPTInferenceError("runtime Bernini negative prompt differs")
    distributed = base.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SPTInferenceError("SPT inference requires four AUH ROCm-visible GPUs")
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
        raise SPTInferenceError(str(error)) from error
    source_sha256 = base.file_sha256(source_path)
    full_prompt = base.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
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
        raise SPTInferenceError(str(error)) from error
    if float(config.shift) != base.FLOW_SHIFT or config.use_unipc is not True:
        raise SPTInferenceError("renderer must use official UniPC with flow shift 5")
    model = BerniniRendererModel(config)
    if any("lora_" in name.lower() for name, _ in model.named_modules()):
        raise SPTInferenceError("base Bernini unexpectedly contains LoRA modules")
    model.requires_grad_(False)
    model.eval()
    planner, planner_identity = strict_load_planner(
        planner_bundle, planner_config, planner_identity, device=device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **base.tokenizer_load_kwargs()
    )
    if tokenizer.padding_side != "right" or tokenizer.init_kwargs.get("fix_mistral_regex") is not True:
        raise SPTInferenceError("tokenizer lost fix_mistral_regex/right-padding contract")
    input_ids, attention_mask = base._tokenize_training_prompt(tokenizer, full_prompt)
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
    expected_bucket = source_metadata["source_derived_bucket_hw"]
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        base.LATENT_FRAME_COUNT,
        int(expected_bucket[0]) // 8,
        int(expected_bucket[1]) // 8,
    )
    if tuple(int(value) for value in source_latent.shape) != expected_latent_shape:
        raise SPTInferenceError("source VAE latent shape differs from exact 81f geometry")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    source_packed, source_video, packed_height, packed_width = pack_clean_source(
        source_latent
    )
    instruction_tokens = encode_student_instruction(
        model, input_ids, attention_mask, device=device
    )
    if int(instruction_tokens.shape[-1]) != planner_config.text_channels:
        raise SPTInferenceError(
            "runtime full-token T5 width differs from planner checkpoint"
        )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # The formal planner call has no target/oracle argument by construction.
        plan = planner(source_video, instruction_tokens)
    planner.to("cpu")
    del planner
    torch.cuda.empty_cache()

    sampling = exact_sampler_contract(seed=args.seed)
    with projection.project_unipc_steps(
        model.diff_dec.scheduler,
        source_packed=source_packed,
        plan=plan,
        height=packed_height,
        width=packed_width,
        max_generate_fraction=args.max_generate_fraction,
    ) as projection_trace:
        with torch.no_grad():
            generated_latent = model.sample(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                uncond_input_ids=negative_ids.to(device),
                uncond_attention_mask=negative_mask.to(device),
                image_vae_latents=None,
                multi_video_vae_latents=[source_latent],
                multi_image_vae_latents=None,
                width=int(expected_bucket[1]),
                height=int(expected_bucket[0]),
                device=device,
                **sampling,
            )
    trace_payload = validate_projection_trace(projection_trace)
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise SPTInferenceError("generated latent shape differs from exact 81f geometry")
    instruction_token_count = int(instruction_tokens.shape[1])
    model.to("cpu")
    del source_latent, source_packed, source_video, instruction_tokens, plan
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        expected_output_shape = (
            base.FRAME_COUNT,
            int(expected_bucket[0]),
            int(expected_bucket[1]),
            3,
        )
        if tuple(int(value) for value in output.shape) != expected_output_shape:
            raise SPTInferenceError("decoded output shape differs from exact 81f contract")
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise SPTInferenceError(f"stale temporary output exists: {temporary_output}")
        save_output(output, str(temporary_output), fps=int(base.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded_frames, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
            output_path
        )
        try:
            base.validate_exact_video_metadata(int(encoded_frames.shape[0]), encoded_fps)
        except base.InferenceContractError as error:
            raise SPTInferenceError(str(error)) from error
        if tuple(encoded_hw) != tuple(expected_bucket):
            raise SPTInferenceError("encoded output geometry differs from source bucket")
        output_sha256 = base.file_sha256(output_path)
        receipt = build_inference_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            output_path=output_path,
            output_sha256=output_sha256,
            planner_bundle=planner_bundle,
            planner_identity=planner_identity,
            planner_config=planner_config,
            instruction_token_count=instruction_token_count,
            trace=trace_payload,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            inference_file_hashes=inference_file_hashes,
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
