#!/usr/bin/env python3
"""Fail-closed deployment preflight for the future SAIC-v2 Stage-B editor.

The public inference surface is exactly one real source video plus one natural
language instruction.  The semantic no-op prompt is method-owned and fixed;
there is no action ID, target, mask, pose, flow, track, trajectory, donor, or
offline motion code argument.

This revision validates a published Stage-B receipt and all bound adapter/base
artifacts, then refuses generation because the pinned Bernini UniPC sampler
still lacks a qualified pre-forward raw-state seam for recomputing the online
T2V action-minus-no-op field.  Falling back to an offline code would create the
exact train/inference gap SAIC-v2 is designed to remove, so no MP4 or receipt is
created until that seam and the combined Stage-A/Stage-B loader are audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as native_infer  # noqa: E402
import saic_online_motion_field_v1 as online_motion  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_saic_stage_b_v1 as stage_b  # noqa: E402


METHOD_NAME = "bernini-saic-stage-b-inference-v1"
PREFLIGHT_SCHEMA_VERSION = "bernini-saic-stage-b-inference-preflight-v1"
INFERENCE_WORLD_SIZE = 4
INFERENCE_SP_SIZE = 4
FRAME_COUNT = 81
FPS_NUMERATOR = 25
FPS_DENOMINATOR = 1
EXACT40_STEPS = 40
DEPLOYMENT_NOOP_PROMPT = (
    "Preserve the source video exactly, including every subject, appearance, "
    "action, camera, background, timing, and motion."
)
FORBIDDEN_ARGUMENTS = stage_b.FORBIDDEN_PUBLIC_ARGUMENTS

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "source_video",
        "instruction_sha256",
        "method_owned_noop_prompt_sha256",
        "artifact_identities",
        "sampling_contract",
        "public_input_names",
        "forbidden_argument_names_present",
        "runtime_capabilities",
        "runtime_blockers",
        "runtime_complete",
        "model_loaded",
        "sampling_started",
        "output_created",
        "semantic_action_editing_success_claimed",
        "preflight_digest",
    }
)


class SAICStageBInferenceError(RuntimeError):
    """Raised before an unqualified Stage-B editor can generate media."""


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
        raise SAICStageBInferenceError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_natural_instruction(value: Any) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or not 8 <= len(value) <= 2048
        or "\x00" in value
        or not any(character.isalpha() for character in value)
        or _SAFE_IDENTIFIER.fullmatch(value) is not None
    ):
        raise SAICStageBInferenceError(
            "instruction must be one canonical natural-language edit instruction"
        )
    return value


def validate_inference_environment(
    environment: Mapping[str, str] = os.environ,
) -> Mapping[str, int]:
    try:
        world = int(environment["WORLD_SIZE"])
        rank = int(environment["RANK"])
        local_rank = int(environment["LOCAL_RANK"])
    except (KeyError, ValueError) as error:
        raise SAICStageBInferenceError(
            "inference requires torchrun WORLD_SIZE/RANK/LOCAL_RANK"
        ) from error
    if (
        world != INFERENCE_WORLD_SIZE
        or not 0 <= rank < world
        or not 0 <= local_rank < world
        or local_rank != rank
    ):
        raise SAICStageBInferenceError("inference requires exactly WORLD4/SP4")
    local_world = environment.get("LOCAL_WORLD_SIZE")
    if local_world is not None:
        try:
            if int(local_world) != INFERENCE_WORLD_SIZE:
                raise SAICStageBInferenceError(
                    "inference requires all WORLD4 ranks on one node"
                )
        except ValueError as error:
            raise SAICStageBInferenceError("LOCAL_WORLD_SIZE is invalid") from error
    return {
        "world_size": world,
        "rank": rank,
        "local_rank": local_rank,
        "sequence_parallel_size": INFERENCE_SP_SIZE,
        "sequence_parallel_rank": rank,
    }


def validate_inference_accelerators(
    topology: Mapping[str, int]
) -> Mapping[str, Any]:
    try:
        import torch
    except Exception as error:  # pragma: no cover - AUH runtime only
        raise SAICStageBInferenceError("PyTorch is unavailable for GPU audit") from error
    if (
        not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
        or torch.cuda.device_count() != INFERENCE_WORLD_SIZE
    ):
        raise SAICStageBInferenceError(
            "inference requires exactly four visible AUH ROCm GPUs"
        )
    local_rank = topology.get("local_rank")
    if type(local_rank) is not int or not 0 <= local_rank < INFERENCE_WORLD_SIZE:
        raise SAICStageBInferenceError("GPU audit local rank differs")
    torch.cuda.set_device(local_rank)
    return {
        **dict(topology),
        "visible_accelerator_count": int(torch.cuda.device_count()),
        "torch_hip": str(torch.version.hip),
        "device_type": "cuda_rocm",
    }


def _probe_exact81(path: Path, *, ffprobe: str) -> Mapping[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        decoded = json.loads(subprocess.check_output(command, text=True))
        streams = decoded["streams"]
        if len(streams) != 1:
            raise ValueError("video stream count differs")
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
        frame_count = int(stream.get("nb_read_frames", stream.get("nb_frames")))
    except (
        OSError,
        subprocess.SubprocessError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise SAICStageBInferenceError(f"ffprobe failed for source video: {error}") from error
    if (
        frame_count != FRAME_COUNT
        or stream.get("r_frame_rate") != "25/1"
        or stream.get("avg_frame_rate") != "25/1"
        or width <= 0
        or height <= 0
        or width % 16
        or height % 16
    ):
        raise SAICStageBInferenceError(
            "source video must be exact81, 25 fps, and 16-aligned"
        )
    return {
        "frame_count": frame_count,
        "fps_numerator": FPS_NUMERATOR,
        "fps_denominator": FPS_DENOMINATOR,
        "width": width,
        "height": height,
    }


def resolve_create_only_media_output(value: str | Path) -> tuple[Path, Path]:
    output = Path(value).expanduser()
    if not output.is_absolute() or output == Path("/") or output.suffix.lower() != ".mp4":
        raise SAICStageBInferenceError("output must be an absolute non-root .mp4 path")
    parent = output.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise SAICStageBInferenceError("cannot resolve output parent") from error
    if parent != resolved_parent or parent.is_symlink() or not parent.is_dir():
        raise SAICStageBInferenceError("output parent must be a canonical plain directory")
    receipt = output.with_name(f"{output.name}.receipt.json")
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise SAICStageBInferenceError("output and receipt are create-only")
    return output, receipt


def runtime_capability_audit() -> Mapping[str, bool]:
    return {
        "online_motion_field_primitive": callable(
            getattr(online_motion, "build_online_motion_field", None)
        ),
        "published_stage_b_receipt_validator": callable(
            getattr(stage_b, "validate_published_checkpoint_receipt", None)
        ),
        "native_exact81_preprocessor": callable(
            getattr(native_infer, "prepare_exact_source", None)
        ),
        "native_unipc_pre_forward_raw_state_hook": False,
        "same_step_frozen_t2v_action_noop_then_source_editor_executor": False,
        "combined_stage_a_and_motion_adapter_strict_loader": False,
        "online_route_trace_and_exact40_receipt_publisher": False,
    }


def runtime_blockers(capabilities: Mapping[str, bool]) -> tuple[str, ...]:
    messages = {
        "native_unipc_pre_forward_raw_state_hook": (
            "pinned native UniPC has no qualified pre-forward raw-state hook"
        ),
        "same_step_frozen_t2v_action_noop_then_source_editor_executor": (
            "online T2V action/no-op and source-editor execution are not composed at one state/timestep"
        ),
        "combined_stage_a_and_motion_adapter_strict_loader": (
            "no byte-audited combined Stage-A plus temporal-operator loader exists"
        ),
        "online_route_trace_and_exact40_receipt_publisher": (
            "no exact40 trace proves online-field recomputation and indices 38/39 exact base"
        ),
    }
    return tuple(message for key, message in messages.items() if capabilities.get(key) is not True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256", required=True
    )
    parser.add_argument("--stage-a-adapter", required=True)
    parser.add_argument("--expected-stage-a-adapter-sha256", required=True)
    parser.add_argument("--stage-a-receipt", required=True)
    parser.add_argument("--expected-stage-a-receipt-sha256", required=True)
    parser.add_argument("--stage-b-adapter", required=True)
    parser.add_argument("--expected-stage-b-adapter-sha256", required=True)
    parser.add_argument("--stage-b-receipt", required=True)
    parser.add_argument("--expected-stage-b-receipt-sha256", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expected-source-video-sha256", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--num-inference-steps", type=int, default=EXACT40_STEPS)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--artifact-preflight-only", action="store_true")
    parser.add_argument("--runtime-preflight-only", action="store_true")
    parser.add_argument("--ack-incomplete-online-inference-preflight", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> str:
    instruction = validate_natural_instruction(args.instruction)
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_stage_a_adapter_sha256",
        "expected_stage_a_receipt_sha256",
        "expected_stage_b_adapter_sha256",
        "expected_stage_b_receipt_sha256",
        "expected_source_video_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        stage_b._sha(getattr(args, name), bits=256, label=name)  # noqa: SLF001
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        stage_b._sha(getattr(args, name), bits=160, label=name)  # noqa: SLF001
    if (
        args.expected_bernini_commit != legacy.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit != legacy.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256
    ):
        raise SAICStageBInferenceError("base source/checkpoint identity differs")
    if args.num_inference_steps != EXACT40_STEPS:
        raise SAICStageBInferenceError("inference requires exact40")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise SAICStageBInferenceError("seed must lie in [0,2^63)")
    if args.ack_incomplete_online_inference_preflight is not True:
        raise SAICStageBInferenceError(
            "explicit incomplete-online-inference preflight acknowledgement is required"
        )
    if args.artifact_preflight_only and args.runtime_preflight_only:
        raise SAICStageBInferenceError("choose at most one preflight scope")
    return instruction


def _read_stage_b_receipt(
    snapshot: stage_b.FileSnapshot,
    *,
    motion_adapter_sha256: str,
    stage_a_adapter_sha256: str,
) -> Mapping[str, Any]:
    value = stage_b._read_json_snapshot(  # noqa: SLF001 - shared trust boundary
        snapshot, label="Stage-B checkpoint receipt"
    )
    try:
        return stage_b.validate_published_checkpoint_receipt(
            value,
            motion_adapter_sha256=motion_adapter_sha256,
            stage_a_adapter_sha256=stage_a_adapter_sha256,
        )
    except stage_b.SAICStageBTrainingError as error:
        raise SAICStageBInferenceError(str(error)) from error


def build_preflight_receipt(
    *,
    source: stage_b.FileSnapshot,
    source_probe: Mapping[str, Any],
    instruction: str,
    stage_a_adapter: stage_b.FileSnapshot,
    stage_a_receipt: stage_b.FileSnapshot,
    stage_b_adapter: stage_b.FileSnapshot,
    stage_b_receipt: stage_b.FileSnapshot,
    published: Mapping[str, Any],
    checkpoint_manifest: stage_b.FileSnapshot,
    topology: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    capabilities = runtime_capability_audit()
    blockers = runtime_blockers(capabilities)
    public_names = {
        action.dest
        for action in build_parser()._actions  # noqa: SLF001 - contract audit
        if action.dest not in {"help"}
    }
    forbidden_present = sorted(public_names & FORBIDDEN_ARGUMENTS)
    if forbidden_present:
        raise SAICStageBInferenceError(
            f"inference parser exposes forbidden inputs: {forbidden_present}"
        )
    body = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "source_video": {**dict(source.receipt()), **dict(source_probe)},
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "method_owned_noop_prompt_sha256": hashlib.sha256(
            DEPLOYMENT_NOOP_PROMPT.encode("utf-8")
        ).hexdigest(),
        "artifact_identities": {
            "stage_a_adapter": dict(stage_a_adapter.receipt()),
            "stage_a_receipt": dict(stage_a_receipt.receipt()),
            "stage_b_adapter": dict(stage_b_adapter.receipt()),
            "stage_b_receipt": dict(stage_b_receipt.receipt()),
            "stage_b_publication_receipt_digest": published["receipt_digest"],
            "checkpoint_content_manifest": dict(checkpoint_manifest.receipt()),
        },
        "sampling_contract": {
            "frame_count": FRAME_COUNT,
            "latent_phases": stage_b.LATENT_PHASES,
            "exact40_steps": EXACT40_STEPS,
            "online_motion_field_recomputed_each_step": True,
            "online_motion_field_schema": online_motion.SCHEMA_VERSION,
            "method_owned_noop_prompt": True,
            "source_conditioned_native_v_and_vi": True,
            "stage_a_frozen": True,
            "motion_operator_frozen": True,
            "indices_38_39_exact_base": True,
            "output_create_only": True,
        },
        "public_input_names": sorted(public_names),
        "forbidden_argument_names_present": forbidden_present,
        "runtime_capabilities": dict(capabilities),
        "runtime_blockers": list(blockers),
        "runtime_complete": not blockers,
        "model_loaded": False,
        "sampling_started": False,
        "output_created": False,
        "semantic_action_editing_success_claimed": False,
    }
    # Topology is folded into the artifact identities only for runtime
    # preflight; keep the root schema closed across both modes.
    body["artifact_identities"]["topology"] = (
        None if topology is None else dict(topology)
    )
    value = {**body, "preflight_digest": object_sha256(body)}
    stage_b._closed(value, _PREFLIGHT_FIELDS, label="inference preflight receipt")  # noqa: SLF001
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        instruction = validate_cli(args)
        resolve_create_only_media_output(args.output)
        source = stage_b.FileSnapshot.capture(
            args.source_video,
            args.expected_source_video_sha256,
            label="source video",
        )
        source_probe = _probe_exact81(source.path, ffprobe=args.ffprobe)
        stage_a_adapter = stage_b.FileSnapshot.capture(
            args.stage_a_adapter,
            args.expected_stage_a_adapter_sha256,
            label="Stage-A adapter",
        )
        stage_a_receipt = stage_b.FileSnapshot.capture(
            args.stage_a_receipt,
            args.expected_stage_a_receipt_sha256,
            label="Stage-A receipt",
        )
        stage_b.validate_stage_a_bundle(
            adapter=stage_a_adapter, receipt=stage_a_receipt
        )
        motion_adapter = stage_b.FileSnapshot.capture(
            args.stage_b_adapter,
            args.expected_stage_b_adapter_sha256,
            label="Stage-B motion adapter",
        )
        motion_receipt = stage_b.FileSnapshot.capture(
            args.stage_b_receipt,
            args.expected_stage_b_receipt_sha256,
            label="Stage-B checkpoint receipt",
        )
        published = _read_stage_b_receipt(
            motion_receipt,
            motion_adapter_sha256=motion_adapter.sha256,
            stage_a_adapter_sha256=stage_a_adapter.sha256,
        )
        checkpoint_manifest = stage_b.FileSnapshot.capture(
            args.checkpoint_content_manifest,
            args.expected_checkpoint_content_manifest_sha256,
            label="checkpoint content manifest",
        )
        checkpoint = stage_b._canonical_directory(  # noqa: SLF001
            args.checkpoint, label="Bernini checkpoint"
        )
        if not args.artifact_preflight_only:
            bernini_root = stage_b._canonical_directory(  # noqa: SLF001
                args.bernini_root, label="Bernini root"
            )
            veomni_root = stage_b._canonical_directory(  # noqa: SLF001
                args.veomni_root, label="VeOmni root"
            )
            legacy.validate_source_trees(
                bernini_root,
                veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
            legacy.validate_checkpoint(checkpoint)
            stage_b.validate_checkpoint_content(
                checkpoint=checkpoint, manifest=checkpoint_manifest
            )
        topology = None
        if not args.artifact_preflight_only:
            topology = validate_inference_accelerators(
                validate_inference_environment()
            )
        receipt = build_preflight_receipt(
            source=source,
            source_probe=source_probe,
            instruction=instruction,
            stage_a_adapter=stage_a_adapter,
            stage_a_receipt=stage_a_receipt,
            stage_b_adapter=motion_adapter,
            stage_b_receipt=motion_receipt,
            published=published,
            checkpoint_manifest=checkpoint_manifest,
            topology=topology,
        )
        for snapshot in (
            source,
            stage_a_adapter,
            stage_a_receipt,
            motion_adapter,
            motion_receipt,
            checkpoint_manifest,
        ):
            snapshot.assert_unchanged()
        if args.artifact_preflight_only or args.runtime_preflight_only:
            if topology is None or topology.get("rank") == 0:
                print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
            return 0
        raise SAICStageBInferenceError(
            "SAIC Stage-B inference is fail-closed before model/output creation; "
            + " | ".join(receipt["runtime_blockers"])
        )
    except stage_b.SAICStageBTrainingError as error:
        raise SAICStageBInferenceError(str(error)) from error


__all__ = [
    "DEPLOYMENT_NOOP_PROMPT",
    "EXACT40_STEPS",
    "FORBIDDEN_ARGUMENTS",
    "FRAME_COUNT",
    "INFERENCE_SP_SIZE",
    "INFERENCE_WORLD_SIZE",
    "METHOD_NAME",
    "PREFLIGHT_SCHEMA_VERSION",
    "SAICStageBInferenceError",
    "build_parser",
    "build_preflight_receipt",
    "canonical_json_bytes",
    "main",
    "object_sha256",
    "resolve_create_only_media_output",
    "runtime_blockers",
    "runtime_capability_audit",
    "validate_cli",
    "validate_inference_environment",
    "validate_inference_accelerators",
    "validate_natural_instruction",
]


if __name__ == "__main__":
    raise SystemExit(main())
