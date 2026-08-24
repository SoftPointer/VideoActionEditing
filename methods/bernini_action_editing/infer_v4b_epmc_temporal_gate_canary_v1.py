#!/usr/bin/env python3
"""Render the frozen two-seed v4-B -> EPMC temporal-gating diagnostic.

The heavy path is deliberately the audited EPMC runner: official Bernini
loading, source VAE encode, action/no-op same-seed proposal carrier, 40-step
APG sampling, Ulysses-4, the first 16 real post-varlen-attention head hooks,
VAE decode, and transactional MP4 writes are reused unchanged.  This adapter
replaces only the old K=2 36D prototype with a separately sealed v4-B decoded-
residual gate state and emits these same-render-seed arms:

``B0`` / ``zero`` / ``correct`` / ``reverse`` / ``shuffle``.

Only render seeds 2028 and 2029 are legal.  The gate state itself can exist
only after the aggregate v4-B development gate was true.  This program accepts
no target or anchor video; those are detached references for the HTML review.
It is a temporal-gating diagnostic only, never action, renderer, quality, or
video-editing qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Optional, Sequence

import torch

from methods.bernini_action_editing import fewshot_motion_branch as motion_branch
from methods.bernini_action_editing import fewshot_privileged_motion_code as epmc
from methods.bernini_action_editing import infer_fewshot_motion_code as epmc_runner
from methods.bernini_action_editing import infer_lora as legacy
from methods.bernini_action_editing import infer_source_value_residual_oracle as value_audit
from methods.bernini_action_editing import materialize_v4b_epmc_gate_state_v1 as gate_materializer


RECEIPT_SCHEMA = "bernini-v4b-epmc-temporal-gate-video-canary-v1"
EXPECTED_IID = gate_materializer.EXPECTED_IID
EXPECTED_SOURCE_SHA256 = epmc_runner.EXPECTED_SOURCE_SHA256
EXPECTED_INSTRUCTION_SHA256 = epmc_runner.EXPECTED_INSTRUCTION_SHA256
PROPOSAL_SEED = 2027
RENDER_SEEDS = (2028, 2029)
ARM_ORDER = ("B0", "zero", "correct", "reverse", "shuffle")
PATCHED_ARM_ORDER = ARM_ORDER[1:]
OUTPUT_ORDER = ARM_ORDER
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_GATE_STATE_BYTES = 1 << 20


class V4BEPMCVideoCanaryError(RuntimeError):
    """The diagnostic gate state or inherited Bernini ABI differed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise V4BEPMCVideoCanaryError("value is not canonical JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise V4BEPMCVideoCanaryError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_zero(value: torch.Tensor, *, label: str) -> None:
    flat = value.detach().contiguous().reshape(-1)
    if int(torch.count_nonzero(flat).item()) != 0:
        raise V4BEPMCVideoCanaryError(f"{label} must be exact zero")
    if int(torch.count_nonzero(flat.view(torch.uint8)).item()) != 0:
        raise V4BEPMCVideoCanaryError(f"{label} must be byte-exact positive zero")


def _plain_gate_state(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise V4BEPMCVideoCanaryError("gate-state must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise V4BEPMCVideoCanaryError("cannot stat gate-state") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size <= _MAX_GATE_STATE_BYTES
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        raise V4BEPMCVideoCanaryError(
            "gate-state must be a bounded mode0444/nlink1 plain file"
        )
    return path.resolve(strict=True)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V4BEPMCVideoCanaryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise V4BEPMCVideoCanaryError(f"non-finite JSON number: {value}")


@dataclass(frozen=True)
class GateStateBundle:
    path: Path
    file_sha256: str
    receipt_digest: str
    codes: Mapping[str, epmc.MotionCode]
    payload: Mapping[str, Any]

    @property
    def motion_code_cpu(self) -> epmc.MotionCode:
        return self.codes["correct"]

    @property
    def representability_gate(self) -> str:
        # Compatibility surface consumed by the inherited runner.  The custom
        # receipt never describes this as the old K=2 representability gate.
        return "GO"

    def audit_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": gate_materializer.SCHEMA,
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "receipt_digest": self.receipt_digest,
            "iid": EXPECTED_IID,
            "outer_fold": gate_materializer.EXPECTED_OUTER_FOLD,
            "v4b_aggregate_gate_verified_true": True,
            "decoded_residual_definition": self.payload[
                "decoded_residual_contract"
            ]["definition"],
            "fit_only_p95_tensor_sha256": self.payload["fit_only_calibration"][
                "p95_tensor_sha256"
            ],
            "profile20_sha256": self.payload["temporal_mapping"][
                "profile20_sha256"
            ],
            "temporal_gating_diagnostic_only": True,
        }


def _code_from_json(value: Mapping[str, Any], *, name: str) -> epmc.MotionCode:
    if not isinstance(value, Mapping):
        raise V4BEPMCVideoCanaryError(f"{name} gate payload must be an object")
    try:
        phase = torch.tensor(value["phase_gates"], dtype=torch.float32).reshape(1, 21)
        block = torch.tensor(value["block_head_gates"], dtype=torch.float32).reshape(
            1, 16, 12
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise V4BEPMCVideoCanaryError(f"{name} gate tensors differ") from error
    phase = phase.contiguous()
    block = block.contiguous()
    try:
        code = epmc.MotionCode(phase, block)
    except epmc.PrivilegedMotionCodeContractError as error:
        raise V4BEPMCVideoCanaryError(str(error)) from error
    if (
        gate_materializer._tensor_sha256(phase)
        != value.get("phase_gates_sha256")
        or gate_materializer._tensor_sha256(block)
        != value.get("block_head_gates_sha256")
    ):
        raise V4BEPMCVideoCanaryError(f"{name} gate tensor digest differs")
    _positive_zero(block, label=f"{name} block/head gates")
    return code


def load_gate_state(
    path_value: str | Path, *, expected_sha256: str
) -> GateStateBundle:
    _required_sha256(expected_sha256, label="expected gate-state SHA256")
    path = _plain_gate_state(path_value)
    if _file_sha256(path) != expected_sha256:
        raise V4BEPMCVideoCanaryError("gate-state file SHA256 differs")
    try:
        payload = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except V4BEPMCVideoCanaryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4BEPMCVideoCanaryError("gate-state is not strict ASCII JSON") from error
    if type(payload) is not dict:
        raise V4BEPMCVideoCanaryError("gate-state must contain one object")
    digest = _required_sha256(
        payload.get("receipt_digest"), label="gate-state receipt digest"
    )
    unsigned = dict(payload)
    unsigned.pop("receipt_digest", None)
    scope = payload.get("scope")
    arms = payload.get("arms")
    detached = payload.get("detached_media_authority")
    temporal_mapping = payload.get("temporal_mapping")
    if (
        _object_sha256(unsigned) != digest
        or payload.get("schema_version") != gate_materializer.SCHEMA
        or payload.get("iid") != EXPECTED_IID
        or payload.get("outer_fold") != gate_materializer.EXPECTED_OUTER_FOLD
        or payload.get("v4b_aggregate_gate_verified_true") is not True
        or not isinstance(scope, Mapping)
        or scope.get("temporal_gating_diagnostic_only") is not True
        or scope.get("bernini_model_execution_performed") is not False
        or scope.get("heldout_action_anchor_feature_consumed") is not True
        or scope.get("heldout_action_anchor_rgb_consumed") is not False
        or scope.get("target_rgb_consumed") is not False
        or scope.get("source_plus_instruction_only_end_to_end_claim") is not False
        or scope.get("gate_state_is_derived_from_heldout_action_anchor_feature")
        is not True
        or not isinstance(detached, Mapping)
        or detached.get("source_video_sha256") != EXPECTED_SOURCE_SHA256
        or detached.get("anchor_video_sha256")
        != gate_materializer.EXPECTED_ANCHOR_VIDEO_SHA256
        or detached.get("instruction_sha256") != EXPECTED_INSTRUCTION_SHA256
        or not isinstance(temporal_mapping, Mapping)
        or temporal_mapping.get("epmc_effective_head_gate_nonzero_phase")
        != "0.5*(profile20+0)=0.5*profile20"
        or temporal_mapping.get("downstream_outer_cpmr_gate") != 0.10
        or temporal_mapping.get("total_projected_motion_residual_coefficient")
        != "0.10*0.5*profile20=0.05*profile20"
        or temporal_mapping.get("total_coefficient_scale") != 0.05
        or temporal_mapping.get("source_and_phase0_total_coefficient") != 0.0
        or not isinstance(arms, Mapping)
        or arms.get("order") != list(gate_materializer.ARM_ORDER)
        or arms.get("reverse_and_shuffle_preserve_correct_phase_multiset") is not True
        or not isinstance(arms.get("values"), Mapping)
    ):
        raise V4BEPMCVideoCanaryError(
            "gate-state does not carry the closed v4-B true-gate diagnostic scope"
        )
    codes = {
        name: _code_from_json(arms["values"].get(name), name=name)
        for name in gate_materializer.ARM_ORDER
    }
    codes["zero"].validate(require_noop=True)
    reference = torch.sort(codes["correct"].phase_gates[:, 1:], dim=1).values
    for name, indices in (
        ("reverse", epmc.REVERSE_PHASE_INDICES),
        ("shuffle", epmc.SHUFFLE_PHASE_INDICES),
    ):
        expected = codes["correct"].phase_gates[:, list(indices)]
        if not torch.equal(codes[name].phase_gates, expected):
            raise V4BEPMCVideoCanaryError(f"{name} is not the frozen phase permutation")
        actual = torch.sort(codes[name].phase_gates[:, 1:], dim=1).values
        if not torch.equal(reference, actual):
            raise V4BEPMCVideoCanaryError(f"{name} changed the phase multiset")
        if torch.equal(codes[name].phase_gates, codes["correct"].phase_gates):
            raise V4BEPMCVideoCanaryError(
                f"{name} is byte-identical to correct; causal control degenerated"
            )
    if int(torch.count_nonzero(codes["correct"].phase_gates[:, 1:]).item()) == 0:
        raise V4BEPMCVideoCanaryError("correct temporal gate degenerated to all zero")
    return GateStateBundle(path, expected_sha256, digest, codes, payload)


def validate_arm_latents(values: Mapping[str, Any]) -> dict[str, bool]:
    if set(values) != set(ARM_ORDER):
        raise V4BEPMCVideoCanaryError("arm latent set differs from frozen five arms")
    for name in ARM_ORDER:
        if tuple(int(x) for x in values[name].shape) != epmc_runner.EXPECTED_LATENT_SHAPE:
            raise V4BEPMCVideoCanaryError(f"{name} latent shape differs")
    if not epmc_runner.v11._tensor_bytes_equal(values["B0"], values["zero"]):
        raise V4BEPMCVideoCanaryError("zero differs bytewise from B0")
    return {
        "zero_full_latent_byte_exact_b0": True,
        "correct_differs_from_zero": not epmc_runner.v11._tensor_bytes_equal(
            values["correct"], values["zero"]
        ),
        "reverse_differs_from_correct": not epmc_runner.v11._tensor_bytes_equal(
            values["reverse"], values["correct"]
        ),
        "shuffle_differs_from_correct": not epmc_runner.v11._tensor_bytes_equal(
            values["shuffle"], values["correct"]
        ),
    }


def _save_arm_outputs(
    *,
    output_dir: Path,
    values: Mapping[str, Any],
    vae: Any,
    device: Any,
    save_output_fn: Any,
) -> dict[str, Any]:
    """Reuse the inherited VAE/atomic-video ABI, decoding only review arms."""

    from bernini.pipeline import _vae_decode
    from tools import materialize_vae

    expected_inputs = {"proposal_action", "proposal_noop", *ARM_ORDER}
    if set(values) != expected_inputs:
        raise V4BEPMCVideoCanaryError("latent output set differs")
    outputs: dict[str, Any] = {}
    vae.to(device)
    for name in ARM_ORDER:
        latent = values[name]
        with torch.no_grad():
            decoded = _vae_decode(vae, latent)
        expected_decoded = (
            epmc_runner.EXPECTED_FRAMES,
            *epmc_runner.EXPECTED_BUCKET_HW,
            3,
        )
        if tuple(int(x) for x in decoded.shape) != expected_decoded:
            raise V4BEPMCVideoCanaryError(f"{name} decoded shape differs")
        path = output_dir / f"{name}.mp4"
        if path.exists() or path.is_symlink():
            raise V4BEPMCVideoCanaryError(f"refusing to overwrite {path}")
        value_audit.save_video_atomically(
            decoded,
            path,
            fps=epmc_runner.EXPECTED_FPS,
            save_output_fn=save_output_fn,
        )
        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(path)
        legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        if tuple(int(x) for x in encoded_hw) != epmc_runner.EXPECTED_BUCKET_HW:
            raise V4BEPMCVideoCanaryError(f"{name} encoded geometry differs")
        outputs[name] = {
            "path": str(path),
            "mp4_sha256": legacy.file_sha256(path),
            "frames": epmc_runner.EXPECTED_FRAMES,
            "fps": epmc_runner.EXPECTED_FPS,
            "bucket_hw": list(epmc_runner.EXPECTED_BUCKET_HW),
            "latent": value_audit.tensor_identity(latent, label=f"{name} latent"),
        }
    vae.to("cpu")
    return outputs


def build_video_receipt(
    *,
    outer_args: argparse.Namespace,
    gate_bundle: GateStateBundle,
    runner_args: argparse.Namespace,
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
    arm_comparisons: Mapping[str, bool],
    arm_codes: Mapping[str, epmc.MotionCode],
    carrier_receipt: Mapping[str, Any],
    runtime_traces: Mapping[str, Mapping[str, Any]],
    patch_receipt: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        motion_branch.OUTER_CPMR_GATE != 0.10
        or set(outputs) != set(ARM_ORDER)
        or arm_comparisons.get("zero_full_latent_byte_exact_b0") is not True
        or set(runtime_traces) != set(PATCHED_ARM_ORDER)
        or not all(trace.get("all_bindings_complete") is True for trace in runtime_traces.values())
    ):
        raise V4BEPMCVideoCanaryError("video canary runtime closure differs")
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "V4B_EPMC_TEMPORAL_GATE_VIDEO_CANARY_COMPLETE_DIAGNOSTIC_ONLY",
        "method": "v4b-decoded-residual-epmc-temporal-head-gating",
        "method_revision": runner_args.method_source_revision,
        "method_archive_sha256": runner_args.method_source_archive_sha256,
        "iid": EXPECTED_IID,
        "scientific_claim": False,
        "action_representation_qualified": False,
        "renderer_qualified": False,
        "video_editing_qualified": False,
        "video_quality_claim": False,
        "temporal_gating_diagnostic_only": True,
        "v4b_aggregate_gate_verified_true": True,
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "metadata": dict(source_metadata),
        },
        "instruction": runner_args.instruction,
        "instruction_sha256": hashlib.sha256(
            runner_args.instruction.encode("utf-8")
        ).hexdigest(),
        "model_facing_input_closure": {
            "source_video": True,
            "instruction": True,
            "sealed_gate_state": True,
            "anchor_video": False,
            "target_video": False,
            "mask": False,
            "flow": False,
            "pose": False,
            "track": False,
            "trajectory": False,
        },
        "end_to_end_data_ancestry": {
            "heldout_action_anchor_feature_consumed_by_gate_materializer": True,
            "gate_state_is_privileged_action_anchor_feature_derived": True,
            "heldout_action_anchor_rgb_consumed": False,
            "target_rgb_consumed": False,
            "source_plus_instruction_only_end_to_end_claim": False,
            "source_instruction_plus_sealed_privileged_gate_state": True,
        },
        "gate_state": gate_bundle.audit_receipt(),
        "seeds": {"proposal": PROPOSAL_SEED, "render": outer_args.render_seed},
        "schedule": {
            "frames": epmc_runner.EXPECTED_FRAMES,
            "fps": epmc_runner.EXPECTED_FPS,
            "steps": epmc_runner.EXPECTED_STEPS,
            "flow_shift": 5.0,
            "proposal_action_noop_same_seed": True,
            "all_five_render_arms_same_seed": True,
        },
        "hook": {
            "implementation": "fewshot_motion_branch.install_fewshot_motion_branch",
            "location": "real post-varlen-attention projected [12,128] heads before merge/to_out",
            "bernini_blocks": list(range(16)),
            "preprojection_channel_chunk_gating": False,
            "block_head_gates_all_exact_positive_zero": True,
            "outer_cpmr_gate": motion_branch.OUTER_CPMR_GATE,
            "epmc_effective_head_gate_nonzero_phase": "0.5*(profile20+0)=0.5*profile20",
            "total_projected_motion_residual_coefficient": "0.10*0.5*profile20=0.05*profile20",
            "total_coefficient_scale": 0.05,
            "source_and_phase0_total_coefficient": 0.0,
        },
        "arms": {
            "order": list(ARM_ORDER),
            "base_prompt": "semantic_noop",
            "codes": {
                name: arm_codes[name].audit_receipt()
                for name in PATCHED_ARM_ORDER
            },
            "reverse_phase_indices": list(epmc.REVERSE_PHASE_INDICES),
            "shuffle_phase_indices": list(epmc.SHUFFLE_PHASE_INDICES),
        },
        "verified_claims": {
            "v4b_aggregate_gate_true_before_render": True,
            "gate_state_hash_pinned": True,
            "source_and_instruction_hash_pinned": True,
            "zero_full_latent_byte_exact_b0": True,
            "all_patched_arms_complete_40_step_binding": True,
            "every_output_is_81_frames_25fps": all(
                item.get("frames") == epmc_runner.EXPECTED_FRAMES
                and item.get("fps") == epmc_runner.EXPECTED_FPS
                for item in outputs.values()
            ),
            "anchor_and_target_video_not_consumed": True,
        },
        "causal_observations_not_acceptance_gates": dict(arm_comparisons),
        "proposal_latents": dict(proposal_identities),
        "arm_latents": dict(arm_identities),
        "carrier": dict(carrier_receipt),
        "runtime_traces": {
            name: dict(runtime_traces[name]) for name in PATCHED_ARM_ORDER
        },
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
        raise V4BEPMCVideoCanaryError("one or more video canary invariants failed")
    payload["receipt_digest"] = _object_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--gate-state", required=True)
    parser.add_argument("--expected-gate-state-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render-seed", type=int, required=True)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument(
        "--expected-instruction-sha256", default=EXPECTED_INSTRUCTION_SHA256
    )
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--proposal-seed", type=int, default=PROPOSAL_SEED)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.render_seed not in RENDER_SEEDS:
        raise V4BEPMCVideoCanaryError("render-seed must be exactly 2028 or 2029")
    if args.proposal_seed != PROPOSAL_SEED or args.num_inference_steps != 40:
        raise V4BEPMCVideoCanaryError("proposal seed and 40-step schedule are frozen")
    if args.expected_source_sha256 != EXPECTED_SOURCE_SHA256:
        raise V4BEPMCVideoCanaryError("preregistered source SHA256 pin differs")
    if args.expected_instruction_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise V4BEPMCVideoCanaryError("preregistered instruction SHA256 pin differs")
    if (
        type(args.instruction) is not str
        or hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()
        != EXPECTED_INSTRUCTION_SHA256
    ):
        raise V4BEPMCVideoCanaryError("instruction bytes differ")
    for name in (
        "expected_gate_state_sha256",
        "expected_source_sha256",
        "expected_instruction_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _required_sha256(getattr(args, name), label=name)
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        if type(getattr(args, name)) is not str or _SHA1.fullmatch(getattr(args, name)) is None:
            raise V4BEPMCVideoCanaryError(f"{name} must be a full lowercase SHA-1")
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output.suffix:
        raise V4BEPMCVideoCanaryError("output-dir must be an absolute suffix-free path")


def _runner_argv(args: argparse.Namespace) -> list[str]:
    # Prototype paths are parser-only compatibility placeholders.  The loader
    # is replaced before inherited main starts and never opens either path.
    placeholder_root = "/__v4b_epmc_gate_state_adapter__"
    return [
        "--bernini-root", args.bernini_root,
        "--veomni-root", args.veomni_root,
        "--checkpoint", args.checkpoint,
        "--checkpoint-content-manifest", args.checkpoint_content_manifest,
        "--source-video", args.source_video,
        "--instruction", args.instruction,
        "--prototype-state", f"{placeholder_root}/state.safetensors",
        "--prototype-receipt", f"{placeholder_root}/receipt.json",
        "--expected-prototype-state-sha256", args.expected_gate_state_sha256,
        "--expected-prototype-receipt-sha256", args.expected_gate_state_sha256,
        "--output-dir", args.output_dir,
        "--expected-source-sha256", args.expected_source_sha256,
        "--expected-instruction-sha256", args.expected_instruction_sha256,
        "--expected-bernini-commit", args.expected_bernini_commit,
        "--expected-veomni-commit", args.expected_veomni_commit,
        "--expected-checkpoint-tree-sha256", args.expected_checkpoint_tree_sha256,
        "--method-source-revision", args.method_source_revision,
        "--method-source-archive-sha256", args.method_source_archive_sha256,
        "--num-inference-steps", str(args.num_inference_steps),
        "--proposal-seed", str(args.proposal_seed),
        "--render-seed", str(args.render_seed),
    ]


def run(args: argparse.Namespace) -> int:
    validate_cli(args)
    gate_bundle = load_gate_state(
        args.gate_state, expected_sha256=args.expected_gate_state_sha256
    )
    original = {
        "RENDER_SEED": epmc_runner.RENDER_SEED,
        "ARM_ORDER": epmc_runner.ARM_ORDER,
        "PATCHED_ARM_ORDER": epmc_runner.PATCHED_ARM_ORDER,
        "OUTPUT_ORDER": epmc_runner.OUTPUT_ORDER,
        "ARM_OUTER_GATES": epmc_runner.ARM_OUTER_GATES,
        "load_prototype_bundle": epmc_runner.load_prototype_bundle,
        "build_arm_motion_codes": epmc_runner.build_arm_motion_codes,
        "validate_arm_latents": epmc_runner.validate_arm_latents,
        "_save_outputs": epmc_runner._save_outputs,
        "_build_receipt": epmc_runner._build_receipt,
    }

    def load_adapter(*_unused: Any, **_unused_kw: Any) -> GateStateBundle:
        return gate_bundle

    def codes_adapter(_unused: epmc.MotionCode) -> dict[str, epmc.MotionCode]:
        return {
            name: epmc.MotionCode(
                gate_bundle.codes[name].phase_gates.clone(),
                gate_bundle.codes[name].block_head_gates.clone(),
            )
            for name in PATCHED_ARM_ORDER
        }

    def receipt_adapter(**kwargs: Any) -> dict[str, Any]:
        inherited_bundle = kwargs.pop("prototype_bundle", None)
        if inherited_bundle is not gate_bundle:
            raise V4BEPMCVideoCanaryError(
                "inherited runner supplied a different gate-state bundle"
            )
        return build_video_receipt(
            outer_args=args,
            gate_bundle=gate_bundle,
            runner_args=kwargs.pop("args"),
            **kwargs,
        )

    epmc_runner.RENDER_SEED = args.render_seed
    epmc_runner.ARM_ORDER = ARM_ORDER
    epmc_runner.PATCHED_ARM_ORDER = PATCHED_ARM_ORDER
    epmc_runner.OUTPUT_ORDER = ("proposal_action", "proposal_noop", *ARM_ORDER)
    epmc_runner.ARM_OUTER_GATES = {
        "B0": None,
        **{name: motion_branch.OUTER_CPMR_GATE for name in PATCHED_ARM_ORDER},
    }
    epmc_runner.load_prototype_bundle = load_adapter
    epmc_runner.build_arm_motion_codes = codes_adapter
    epmc_runner.validate_arm_latents = validate_arm_latents
    epmc_runner._save_outputs = _save_arm_outputs
    epmc_runner._build_receipt = receipt_adapter
    try:
        return epmc_runner.main(_runner_argv(args))
    finally:
        for name, value in original.items():
            setattr(epmc_runner, name, value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "EXPECTED_IID",
    "GateStateBundle",
    "PATCHED_ARM_ORDER",
    "PROPOSAL_SEED",
    "RECEIPT_SCHEMA",
    "RENDER_SEEDS",
    "V4BEPMCVideoCanaryError",
    "build_parser",
    "build_video_receipt",
    "load_gate_state",
    "main",
    "run",
    "validate_arm_latents",
    "validate_cli",
]
