#!/usr/bin/env python3
"""Source-only full644 Bernini inference with generator-native camera gauge fixing.

The audited :mod:`infer_lora` runner remains authoritative for model loading,
the 240-module/480-tensor PEFT reload, ``safe_merge=True``, exact 81-frame
preprocessing, official 40-step ``v2v_apg`` UniPC sampling, decoding and video
publication.  This module only adds a fixed semantic-noop branch and invokes
the selected ``global_svd`` or ``grid_consensus`` stabilizer at the clean-field
boundary certified by :mod:`tri_branch_unipc`: after exact reconstruction of
official action APG and immediately before the one original UniPC step.  The
default remains the original ``global_svd`` path; ``grid_consensus`` uses
fixed-grid physical homography coefficients and robust per-phase consensus.

The external semantic inputs are exactly a source video and edit instruction.
The full644 adapter and beta are method parameters, not visual conditions.  No
target, mask, flow, pose, track, trajectory, reference, or edited first frame
is accepted.  ``beta=0`` is a hard byte-exact control: the callback must return
the official action clean-field object, causing the hook to pass the original
official ``model_output`` object to UniPC without a clean/velocity round trip.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generator_native_camera_stabilizer as camera  # noqa: E402
import fixed_grid_camera_consensus as grid_camera  # noqa: E402
import fixed_grid_camera_consensus_stabilizer as grid_stabilizer  # noqa: E402
import infer_lora as legacy  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_lora as trainer  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402


METHOD_NAME = "full644-generator-native-camera-tangent-stabilizer-v1"
INFERENCE_RECEIPT_SCHEMA = "bernini-full644-camera-tangent-inference-v1"
FULL644_TRAINING_STEP = 644
FULL644_TARGET_MODULE_COUNT = 240
FULL644_ADAPTER_TENSOR_COUNT = 480
FULL644_ADAPTER_CONFIG_SHA256 = (
    "b91c3a236b0e0e893e7c992be043ec28cfa05c73b7792c0b93b4013db15aef39"
)
FULL644_ADAPTER_SHA256 = (
    "9217ff653e47f915105fe8fa64856037d63811562cec1e9fd53ae9e4613a9774"
)
FULL644_TRAINING_RECEIPT_FILE_SHA256 = (
    "5931f7544d1bd185adf3fc07edb046e6bf27811b0835de8446f91c8a25c782c6"
)
FULL644_TRAINING_RECEIPT_DIGEST = (
    "6b5f2a053be048881b1426d9b7c4c380dc8b82f6098bfbda9c80034b26df17d1"
)
EXPECTED_FRAMES = 81
EXPECTED_STEPS = 40
EXPECTED_SEED = 2027
EXPECTED_FORWARDS = EXPECTED_STEPS * 3
CAMERA_ESTIMATORS = ("global_svd", "grid_consensus")
DEFAULT_CAMERA_ESTIMATOR = "global_svd"
GRID_RANK_AUTHORITY_SCHEMA = "bernini-grid-rank0-authority-v1"
GRID_RANK_AUTHORITY_EXACT_PROOF = (
    "rank0_reference_broadcast_then_torch.equal_then_all_reduce_MIN"
)
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


FORBIDDEN_EXTERNAL_CONDITIONS = (
    "target_video",
    "support_video",
    "mask",
    "optical_flow",
    "pose",
    "track",
    "swept_tube",
    "trajectory",
    "reference_image",
    "reference_video",
    "edited_first_frame",
    "first_frame_anchor",
)


class CameraStabilizerInferenceError(RuntimeError):
    """Raised before publishing an output whose method contract is ambiguous."""


@dataclass(frozen=True)
class CameraStepRecord:
    """Tensor-free evidence from one post-APG/pre-UniPC stabilizer call."""

    step_index: int
    timestep: float
    sigma: float
    beta: float
    action_passthrough_object_exact: bool
    basis_built_this_step: bool
    basis_reused_from_prior_step: bool
    core_trace: Mapping[str, Any]
    estimator: str = DEFAULT_CAMERA_ESTIMATOR
    geometry_built_this_step: bool = False
    geometry_reused_from_prior_step: bool = False
    grid_rank_authority: Optional[Mapping[str, Any]] = None


@dataclass
class CameraExecutionTrace:
    """All stabilizer callbacks from one official Bernini sample call."""

    beta: float
    estimator: str = DEFAULT_CAMERA_ESTIMATOR
    records: list[CameraStepRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "estimator": self.estimator,
            "step_count": len(self.records),
            "basis_build_count": sum(
                int(item.basis_built_this_step) for item in self.records
            ),
            "basis_reuse_count": sum(
                int(item.basis_reused_from_prior_step) for item in self.records
            ),
            "geometry_build_count": sum(
                int(item.geometry_built_this_step) for item in self.records
            ),
            "geometry_reuse_count": sum(
                int(item.geometry_reused_from_prior_step) for item in self.records
            ),
            "steps": [asdict(item) for item in self.records],
        }


def _grid_distributed_context(
    dist_module: Optional[Any] = None,
) -> tuple[Any, Optional[Any], int, int]:
    """Return the exact default process group used by one Ulysses-4 arm."""

    if dist_module is None:
        try:
            import torch.distributed as dist_module
        except ImportError:  # pragma: no cover - runtime dependency
            return None, None, 1, 0
    available = getattr(dist_module, "is_available", lambda: True)()
    initialized = getattr(dist_module, "is_initialized", lambda: False)()
    if not available or not initialized:
        return dist_module, None, 1, 0
    group = dist_module.group.WORLD
    world_size = int(dist_module.get_world_size(group=group))
    group_rank = int(dist_module.get_rank(group=group))
    if world_size != legacy.ULYSSES_SIZE:
        raise CameraStabilizerInferenceError(
            "distributed grid authority requires the exact Ulysses-4 WORLD group"
        )
    if not 0 <= group_rank < world_size:
        raise CameraStabilizerInferenceError(
            "distributed grid authority observed an invalid process-group rank"
        )
    return dist_module, group, world_size, group_rank


def _zero_beta_grid_rank_authority_evidence() -> dict[str, Any]:
    """Describe the hard beta-zero bypass without issuing any collective."""

    _, group, world_size, group_rank = _grid_distributed_context()
    return {
        "schema_version": GRID_RANK_AUTHORITY_SCHEMA,
        "mode": "zero_beta_exact_action_no_collective",
        "authority_group": (
            "torch.distributed.group.WORLD" if group is not None else "local_process"
        ),
        "world_size": world_size,
        "group_rank": group_rank,
        "process_group_rank0": 0,
        "rank0_authoritative_broadcast": False,
        "source_clean_cross_rank_exact": None,
        "source_clean_rank0_reference_exact": None,
        "source_clean_certified_this_step": False,
        "action_clean_cross_rank_exact": None,
        "action_clean_rank0_reference_exact": None,
        "local_candidate_success_all_ranks": None,
        "pre_broadcast_max_abs_disagreement": None,
        "post_broadcast_exact": None,
        "post_broadcast_proof": None,
        "exact_comparison_proof": GRID_RANK_AUTHORITY_EXACT_PROOF,
        "all_rank_participant_count": 0,
        "collective_sequence": [],
    }


class _GridRank0Authority:
    """Make rank 0's active grid result the scheduler input on every rank.

    Source/action exactness and final scheduler authority stay on device; no
    full clean field is copied to CPU and no object collective is used.
    """

    def __init__(self, source_clean_field: Any, *, dist_module: Optional[Any] = None):
        self.source_clean_field = source_clean_field
        (
            self.dist,
            self.group,
            self.world_size,
            self.group_rank,
        ) = _grid_distributed_context(dist_module)
        self._source_certified = False

    @property
    def distributed(self) -> bool:
        return self.group is not None and self.world_size > 1

    def certify_source(self) -> bool:
        """Strictly certify the immutable source once before any local SVD/MAD."""

        if self._source_certified:
            return False
        if self.distributed:
            import torch

            rank0_reference = self.source_clean_field.detach().clone()
            self.dist.broadcast(rank0_reference, src=0, group=self.group)
            exact = torch.tensor(
                [
                    int(
                        torch.equal(
                            self.source_clean_field.detach(), rank0_reference
                        )
                    )
                ],
                dtype=torch.int32,
                device=self.source_clean_field.device,
            )
            self.dist.all_reduce(
                exact,
                op=self.dist.ReduceOp.MIN,
                group=self.group,
            )
            if int(exact.item()) != 1:
                raise CameraStabilizerInferenceError(
                    "source_clean_field is not exact across the Ulysses process group"
                )
        self._source_certified = True
        return True

    def certify_action(self, action_clean_field: Any) -> bool:
        """Compare each action field exactly with a GPU-side rank-0 clone."""

        if not self.distributed:
            return True
        import torch

        rank0_reference = action_clean_field.detach().clone()
        self.dist.broadcast(rank0_reference, src=0, group=self.group)
        exact = torch.tensor(
            [int(torch.equal(action_clean_field.detach(), rank0_reference))],
            dtype=torch.int32,
            device=action_clean_field.device,
        )
        self.dist.all_reduce(exact, op=self.dist.ReduceOp.MIN, group=self.group)
        if int(exact.item()) != 1:
            raise CameraStabilizerInferenceError(
                "action_guided_clean is not exact across the Ulysses process group"
            )
        return True

    def require_all_candidates_succeeded(
        self,
        local_success: bool,
        *,
        reference: Any,
    ) -> bool:
        """Synchronize local estimator status before the next collective."""

        if not self.distributed:
            return bool(local_success)
        import torch

        status = torch.tensor(
            [int(bool(local_success))],
            dtype=torch.int32,
            device=reference.device,
        )
        self.dist.all_reduce(status, op=self.dist.ReduceOp.MIN, group=self.group)
        return int(status.item()) == 1

    def execute_rank0_authoritative(
        self,
        local_candidate: Any,
        *,
        action_exact: bool,
        source_certified_this_step: bool,
    ) -> tuple[Any, dict[str, Any]]:
        """Broadcast rank 0's candidate and return exact all-rank evidence."""

        if not self._source_certified:
            raise CameraStabilizerInferenceError(
                "source exactness was not certified before grid execution"
            )
        collective_sequence: list[str] = []
        if source_certified_this_step and self.distributed:
            collective_sequence.extend(
                [
                    "source_rank0_reference_broadcast",
                    "source_rank0_reference_exact_all_reduce_min",
                ]
            )
        if self.distributed:
            collective_sequence.extend(
                [
                    "action_rank0_reference_broadcast",
                    "action_rank0_reference_exact_all_reduce_min",
                    "candidate_success_all_reduce_min",
                    "executed_clean_broadcast_from_process_group_rank0",
                    "pre_broadcast_disagreement_all_reduce_max",
                ]
            )
            authoritative = local_candidate.detach().clone()
            self.dist.broadcast(authoritative, src=0, group=self.group)
            disagreement = (
                local_candidate.detach() - authoritative
            ).abs().amax().to(dtype=authoritative.dtype)
            self.dist.all_reduce(
                disagreement,
                op=self.dist.ReduceOp.MAX,
                group=self.group,
            )
            pre_disagreement = float(disagreement.item())
        else:
            authoritative = local_candidate
            pre_disagreement = 0.0
        if not math.isfinite(pre_disagreement) or pre_disagreement < 0.0:
            raise CameraStabilizerInferenceError(
                "grid pre-broadcast disagreement is invalid"
            )
        # ``broadcast`` is an exact-copy collective.  The immediately following
        # all-rank MAX reduction cannot complete unless every process consumed
        # that broadcast in the same collective order, so together they are the
        # post-broadcast exactness/participation proof without a 40-step D2H hash.
        post_exact = True
        evidence = {
            "schema_version": GRID_RANK_AUTHORITY_SCHEMA,
            "mode": (
                "distributed_rank0_authoritative"
                if self.distributed
                else "single_process_rank0"
            ),
            "authority_group": (
                "torch.distributed.group.WORLD"
                if self.distributed
                else "local_process"
            ),
            "world_size": self.world_size,
            "group_rank": self.group_rank,
            "process_group_rank0": 0,
            "rank0_authoritative_broadcast": self.distributed,
            "source_clean_cross_rank_exact": True,
            "source_clean_rank0_reference_exact": True,
            "source_clean_certified_this_step": source_certified_this_step,
            "action_clean_cross_rank_exact": action_exact,
            "action_clean_rank0_reference_exact": action_exact,
            "local_candidate_success_all_ranks": True,
            "pre_broadcast_max_abs_disagreement": pre_disagreement,
            "post_broadcast_exact": post_exact,
            "post_broadcast_proof": (
                "rank0_broadcast_then_all_rank_disagreement_MAX_completed"
                if self.distributed
                else "single_process_object"
            ),
            "exact_comparison_proof": GRID_RANK_AUTHORITY_EXACT_PROOF,
            "all_rank_participant_count": self.world_size,
            "collective_sequence": collective_sequence,
        }
        return authoritative, evidence


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CameraStabilizerInferenceError(
            f"value is not canonical JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _required_sha(value: Any, *, label: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise CameraStabilizerInferenceError(f"{label} has an invalid digest")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beta", required=True, type=float)
    parser.add_argument(
        "--camera-estimator",
        choices=CAMERA_ESTIMATORS,
        default=DEFAULT_CAMERA_ESTIMATOR,
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        choices=(EXPECTED_STEPS,),
        default=EXPECTED_STEPS,
    )
    parser.add_argument(
        "--seed", type=int, choices=(EXPECTED_SEED,), default=EXPECTED_SEED
    )
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-instruction-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if (
        type(args.instruction) is not str
        or not args.instruction.strip()
        or "\x00" in args.instruction
    ):
        raise CameraStabilizerInferenceError(
            "instruction must be non-empty text without NUL"
        )
    if args.num_inference_steps != EXPECTED_STEPS:
        raise CameraStabilizerInferenceError("camera stabilizer requires 40 steps")
    if type(args.seed) is not int or args.seed != EXPECTED_SEED:
        raise CameraStabilizerInferenceError("camera stabilizer seed is fixed to 2027")
    if (
        isinstance(args.beta, bool)
        or not isinstance(args.beta, (int, float))
        or not math.isfinite(float(args.beta))
        or not 0.0 <= float(args.beta) <= 1.0
    ):
        raise CameraStabilizerInferenceError("beta must be finite and in [0,1]")
    camera_estimator = getattr(
        args,
        "camera_estimator",
        DEFAULT_CAMERA_ESTIMATOR,
    )
    if camera_estimator not in CAMERA_ESTIMATORS:
        raise CameraStabilizerInferenceError("unsupported camera estimator")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        _required_sha(getattr(args, name), label=name, pattern=_SHA1_RE)
    for name in (
        "expected_checkpoint_tree_sha256",
        "expected_source_sha256",
        "expected_instruction_sha256",
        "method_source_archive_sha256",
    ):
        _required_sha(getattr(args, name), label=name, pattern=_SHA256_RE)
    if args.expected_bernini_commit != trainer.BERNINI_OFFICIAL_COMMIT:
        raise CameraStabilizerInferenceError("unsupported Bernini source revision")
    if args.expected_veomni_commit != trainer.VEOMNI_TESTED_COMMIT:
        raise CameraStabilizerInferenceError("unsupported VeOmni source revision")
    if args.expected_checkpoint_tree_sha256 != trainer.CHECKPOINT_TREE_SHA256:
        raise CameraStabilizerInferenceError("unsupported Bernini checkpoint tree")
    for name in (
        "bernini_root",
        "veomni_root",
        "checkpoint",
        "adapter_checkpoint",
        "source_video",
        "output",
    ):
        if not Path(getattr(args, name)).expanduser().is_absolute():
            raise CameraStabilizerInferenceError(f"{name} must be an absolute path")


def exact_sampler_contract() -> dict[str, Any]:
    value = legacy.sampler_contract(steps=EXPECTED_STEPS, seed=EXPECTED_SEED)
    if (
        value.get("num_frames") != EXPECTED_FRAMES
        or value.get("num_inference_steps") != EXPECTED_STEPS
        or value.get("seed") != EXPECTED_SEED
        or value.get("guidance_mode") != "v2v_apg"
        or value.get("flow_shift") != 5.0
    ):
        raise CameraStabilizerInferenceError("legacy sampler contract changed")
    return value


def validate_full644_adapter_bundle(
    adapter: legacy.AdapterBundle,
) -> dict[str, Any]:
    """Pin the exact successful full644 checkpoint before model construction."""

    identities = {
        "adapter_config_sha256": legacy.file_sha256(adapter.adapter_config_path),
        "adapter_model_sha256": legacy.file_sha256(adapter.adapter_model_path),
        "training_receipt_file_sha256": legacy.file_sha256(
            adapter.training_receipt_path
        ),
    }
    expected = {
        "adapter_config_sha256": FULL644_ADAPTER_CONFIG_SHA256,
        "adapter_model_sha256": FULL644_ADAPTER_SHA256,
        "training_receipt_file_sha256": FULL644_TRAINING_RECEIPT_FILE_SHA256,
    }
    if identities != expected:
        raise CameraStabilizerInferenceError(
            "adapter checkpoint is not the byte-pinned full644 artifact"
        )
    receipt = legacy._read_json(
        adapter.training_receipt_path, label="full644 training receipt"
    )
    if (
        receipt.get("receipt_digest") != FULL644_TRAINING_RECEIPT_DIGEST
        or receipt.get("global_step") != FULL644_TRAINING_STEP
        or receipt.get("target_module_count") != FULL644_TARGET_MODULE_COUNT
    ):
        raise CameraStabilizerInferenceError(
            "full644 training receipt digest, step, or module count differs"
        )
    training_revision = receipt.get("method_source_revision")
    training_archive = receipt.get("method_source_archive_sha256")
    _required_sha(
        training_revision,
        label="full644 training method revision",
        pattern=_SHA1_RE,
    )
    _required_sha(
        training_archive,
        label="full644 training method archive",
        pattern=_SHA256_RE,
    )
    return {
        "checkpoint_root": str(adapter.checkpoint_root),
        **identities,
        "training_receipt_digest": FULL644_TRAINING_RECEIPT_DIGEST,
        "training_global_step": FULL644_TRAINING_STEP,
        "training_method_source_revision": training_revision,
        "training_method_source_archive_sha256": training_archive,
        "target_module_count": FULL644_TARGET_MODULE_COUNT,
        "adapter_tensor_count": FULL644_ADAPTER_TENSOR_COUNT,
    }


def _encode_semantic_noop_prompt(
    renderer: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Encode the internal fixed no-op through the frozen Bernini T5 path."""

    import torch

    if tuple(input_ids.shape) != (1, 512) or tuple(attention_mask.shape) != (1, 512):
        raise CameraStabilizerInferenceError("semantic no-op tokens must be [1,512]")
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
        or not embeddings.is_floating_point()
        or not bool(torch.isfinite(embeddings).all().item())
    ):
        raise CameraStabilizerInferenceError(
            "Bernini returned invalid semantic no-op embeddings"
        )
    if any(parameter.requires_grad for parameter in renderer.t5_text_encoder.parameters()):
        raise CameraStabilizerInferenceError("semantic no-op T5 is not frozen")
    return embeddings, {
        "instruction_utf8_sha256": hashlib.sha256(
            motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "token_shape": [1, 512],
        "nonpadding_token_count": int(attention_mask.sum().item()),
        "embedding_shape": [int(value) for value in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
        "encoder": "BerniniRendererModel.encode_prompt",
        "frozen_t5": True,
        "internal_fixed_control": True,
    }


class CameraTangentCallback:
    """Thin adapter from tri-branch clean fields to the camera stabilizer API."""

    def __init__(self, *, source_clean_field: Any, beta: float) -> None:
        if not math.isfinite(float(beta)) or not 0.0 <= float(beta) <= 1.0:
            raise CameraStabilizerInferenceError("callback beta must be in [0,1]")
        self.source_clean_field = source_clean_field
        self.beta = float(beta)
        self.config = camera.CameraTangentConfig()
        self._precomputed_basis: Any = None
        self.trace = CameraExecutionTrace(self.beta)

    def __call__(self, fields: tri.CleanFieldStep) -> Any:
        try:
            basis_existed_before = self._precomputed_basis is not None
            basis_built_this_step = False
            if self.beta != 0.0 and not basis_existed_before:
                self._precomputed_basis = camera.build_camera_tangent_basis(
                    self.source_clean_field,
                    self.config,
                )
                basis_built_this_step = True
            result = camera.stabilize_camera_tangent(
                self.source_clean_field,
                fields.action_guided_clean,
                fields.noop_guided_clean,
                beta=self.beta,
                enabled=True,
                camera_edit_requested=False,
                config=self.config,
                precomputed_basis=self._precomputed_basis,
            )
            executed = result.executed_clean_field
            core_trace = result.trace.to_receipt()
        except Exception as error:
            camera_error = getattr(camera, "CameraTangentError", ())
            if camera_error and isinstance(error, camera_error):
                raise CameraStabilizerInferenceError(str(error)) from error
            raise
        if not isinstance(core_trace, Mapping):
            raise CameraStabilizerInferenceError(
                "camera stabilizer trace must serialize to a mapping"
            )
        passthrough = executed is fields.action_guided_clean
        _validate_core_step_trace(
            core_trace,
            beta=self.beta,
            action_passthrough_object_exact=passthrough,
        )
        if self.beta == 0.0 and not passthrough:
            raise CameraStabilizerInferenceError(
                "beta=0 must return the exact official action clean-field object"
            )
        self.trace.records.append(
            CameraStepRecord(
                step_index=int(fields.step_index),
                timestep=float(fields.timestep),
                sigma=float(fields.sigma),
                beta=self.beta,
                action_passthrough_object_exact=passthrough,
                basis_built_this_step=basis_built_this_step,
                basis_reused_from_prior_step=(
                    basis_existed_before and core_trace.get("basis_reused") is True
                ),
                core_trace=dict(core_trace),
            )
        )
        return executed


class CameraGridConsensusCallback:
    """Scheduler-boundary adapter for robust fixed-grid camera consensus."""

    def __init__(self, *, source_clean_field: Any, beta: float) -> None:
        if not math.isfinite(float(beta)) or not 0.0 <= float(beta) <= 1.0:
            raise CameraStabilizerInferenceError("callback beta must be in [0,1]")
        self.source_clean_field = source_clean_field
        self.beta = float(beta)
        self.config = grid_camera.CameraConsensusConfig()
        self._precomputed_geometry: Any = None
        self._rank_authority: Optional[_GridRank0Authority] = None
        self.trace = CameraExecutionTrace(
            self.beta,
            estimator="grid_consensus",
        )

    def __call__(self, fields: tri.CleanFieldStep) -> Any:
        geometry_existed_before = self._precomputed_geometry is not None
        geometry_built_this_step = False
        rank_authority_evidence: Mapping[str, Any]
        if self.beta == 0.0:
            # Hard control: no source/action replica check, geometry, correction,
            # broadcast or reduction may intervene before the exact object return.
            try:
                result = grid_stabilizer.stabilize_camera_consensus(
                    self.source_clean_field,
                    fields.action_guided_clean,
                    beta=self.beta,
                    config=self.config,
                    precomputed_geometry=None,
                )
                executed = result.executed_clean_field
                core_trace = result.trace.to_receipt()
            except Exception as error:
                grid_error = getattr(grid_camera, "CameraConsensusError", ())
                if grid_error and isinstance(error, grid_error):
                    raise CameraStabilizerInferenceError(str(error)) from error
                raise
            rank_authority_evidence = _zero_beta_grid_rank_authority_evidence()
        else:
            if self._rank_authority is None:
                self._rank_authority = _GridRank0Authority(
                    self.source_clean_field
                )
            source_certified_this_step = self._rank_authority.certify_source()
            action_exact = self._rank_authority.certify_action(
                fields.action_guided_clean
            )
            local_error: Optional[Exception] = None
            local_executed: Any = None
            core_trace: Any = None
            try:
                if not geometry_existed_before:
                    self._precomputed_geometry = (
                        grid_camera.build_fixed_grid_camera_geometry(
                            self.source_clean_field,
                            self.config,
                        )
                    )
                    geometry_built_this_step = True
                result = grid_stabilizer.stabilize_camera_consensus(
                    self.source_clean_field,
                    fields.action_guided_clean,
                    beta=self.beta,
                    config=self.config,
                    precomputed_geometry=self._precomputed_geometry,
                )
                local_executed = result.executed_clean_field
                core_trace = result.trace.to_receipt()
                if not isinstance(core_trace, Mapping):
                    raise CameraStabilizerInferenceError(
                        "grid camera stabilizer trace must serialize to a mapping"
                    )
                _validate_grid_core_step_trace(
                    core_trace,
                    beta=self.beta,
                    action_passthrough_object_exact=(
                        local_executed is fields.action_guided_clean
                    ),
                )
            except Exception as error:  # synchronize before any rank can continue
                local_error = error
            all_candidates_succeeded = (
                self._rank_authority.require_all_candidates_succeeded(
                    local_error is None,
                    reference=fields.action_guided_clean,
                )
            )
            if not all_candidates_succeeded:
                synchronized = CameraStabilizerInferenceError(
                    "one or more Ulysses ranks failed local grid consensus"
                )
                if local_error is not None:
                    raise synchronized from local_error
                raise synchronized
            if local_error is not None:  # single-process path
                grid_error = getattr(grid_camera, "CameraConsensusError", ())
                if grid_error and isinstance(local_error, grid_error):
                    raise CameraStabilizerInferenceError(
                        str(local_error)
                    ) from local_error
                raise local_error
            executed, rank_authority_evidence = (
                self._rank_authority.execute_rank0_authoritative(
                    local_executed,
                    action_exact=action_exact,
                    source_certified_this_step=source_certified_this_step,
                )
            )
        try:
            if not isinstance(core_trace, Mapping):
                raise CameraStabilizerInferenceError(
                    "grid camera stabilizer trace must serialize to a mapping"
                )
            passthrough = executed is fields.action_guided_clean
            _validate_grid_core_step_trace(
                core_trace,
                beta=self.beta,
                action_passthrough_object_exact=passthrough,
            )
        except Exception as error:
            grid_error = getattr(grid_camera, "CameraConsensusError", ())
            if grid_error and isinstance(error, grid_error):
                raise CameraStabilizerInferenceError(str(error)) from error
            raise
        if self.beta == 0.0 and not passthrough:
            raise CameraStabilizerInferenceError(
                "beta=0 must return the exact official action clean-field object"
            )
        self.trace.records.append(
            CameraStepRecord(
                step_index=int(fields.step_index),
                timestep=float(fields.timestep),
                sigma=float(fields.sigma),
                beta=self.beta,
                action_passthrough_object_exact=passthrough,
                basis_built_this_step=False,
                basis_reused_from_prior_step=False,
                core_trace=dict(core_trace),
                estimator="grid_consensus",
                geometry_built_this_step=geometry_built_this_step,
                geometry_reused_from_prior_step=(
                    geometry_existed_before
                    and core_trace.get("geometry_reused") is True
                ),
                grid_rank_authority=dict(rank_authority_evidence),
            )
        )
        return executed


def _validate_core_step_trace(
    trace: Mapping[str, Any],
    *,
    beta: float,
    action_passthrough_object_exact: bool,
) -> None:
    """Validate the core's per-step certificate without retaining tensors."""

    if (
        trace.get("schema_version") != camera.SCHEMA_VERSION
        or trace.get("method") != camera.METHOD_NAME
        or trace.get("beta_mode") != "scalar"
        or trace.get("invariant_satisfied") is not True
    ):
        raise CameraStabilizerInferenceError("camera core trace contract differs")
    beta_rows = trace.get("beta_per_phase")
    if (
        not isinstance(beta_rows, list)
        or len(beta_rows) != 1
        or not isinstance(beta_rows[0], list)
        or len(beta_rows[0]) != legacy.LATENT_FRAME_COUNT
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(float(value), beta, rel_tol=0.0, abs_tol=1e-7)
            for value in beta_rows[0]
        )
    ):
        raise CameraStabilizerInferenceError("camera core beta trace differs")
    bypassed = trace.get("bypassed")
    reason = trace.get("bypass_reason")
    basis_reused = trace.get("basis_reused")
    if type(bypassed) is not bool or type(basis_reused) is not bool:
        raise CameraStabilizerInferenceError(
            "camera bypass/reuse trace is not boolean"
        )
    if bypassed != action_passthrough_object_exact:
        raise CameraStabilizerInferenceError(
            "camera bypass trace and executed object identity differ"
        )
    if beta == 0.0 and (bypassed is not True or reason != "zero_beta"):
        raise CameraStabilizerInferenceError("beta=0 core bypass reason differs")
    if beta == 0.0 and basis_reused is not False:
        raise CameraStabilizerInferenceError("beta=0 must not consume a camera basis")
    if not bypassed and (
        trace.get("basis_built") is not True
        or basis_reused is not True
        or trace.get("source_basis_detached") is not True
        or reason is not None
    ):
        raise CameraStabilizerInferenceError(
            "active camera projection lacks detached source-basis evidence"
        )
    for name in (
        "noncamera_invariance_max_abs",
        "noncamera_invariance_rms",
        "noncamera_invariance_tolerance",
    ):
        value = trace.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise CameraStabilizerInferenceError(
                f"camera core {name} diagnostic is invalid"
            )


def _validate_grid_core_step_trace(
    trace: Mapping[str, Any],
    *,
    beta: float,
    action_passthrough_object_exact: bool,
) -> None:
    """Validate one robust-consensus trace without projection invariants."""

    if (
        trace.get("schema_version") != grid_stabilizer.SCHEMA_VERSION
        or trace.get("method") != grid_stabilizer.METHOD_NAME
        or trace.get("beta_mode") != "scalar"
        or trace.get("estimator")
        != "fixed_grid_median_MAD_trimmed_robust_consensus"
        or trace.get("consensus_scope")
        != "independent_per_batch_and_latent_phase"
        or trace.get("invalid_phases_exact_action") is not True
    ):
        raise CameraStabilizerInferenceError(
            "grid camera core trace contract differs"
        )
    beta_rows = trace.get("beta_per_phase")
    if (
        not isinstance(beta_rows, list)
        or len(beta_rows) != 1
        or not isinstance(beta_rows[0], list)
        or len(beta_rows[0]) != legacy.LATENT_FRAME_COUNT
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(float(value), beta, rel_tol=0.0, abs_tol=1e-7)
            for value in beta_rows[0]
        )
    ):
        raise CameraStabilizerInferenceError("grid camera core beta trace differs")
    bypassed = trace.get("bypassed")
    reason = trace.get("bypass_reason")
    geometry_built = trace.get("geometry_built")
    geometry_reused = trace.get("geometry_reused")
    if not all(
        type(value) is bool
        for value in (bypassed, geometry_built, geometry_reused)
    ):
        raise CameraStabilizerInferenceError(
            "grid camera bypass/geometry trace is not boolean"
        )
    if bypassed != action_passthrough_object_exact:
        raise CameraStabilizerInferenceError(
            "grid camera bypass trace and executed object identity differ"
        )
    if beta == 0.0:
        if (
            bypassed is not True
            or reason != "zero_beta"
            or geometry_built is not False
            or geometry_reused is not False
            or trace.get("consensus_valid") is not None
            or trace.get("correction_rms") is not None
            or trace.get("geometry_valid_tile_count") is not None
            or trace.get("fit_valid_tile_count") is not None
            or trace.get("inlier_tile_count") is not None
            or trace.get("spatial_coverage_valid") is not None
            or trace.get("consensus_coefficient_max_abs") is not None
            or trace.get("tile_relative_fit_residual_max") is not None
        ):
            raise CameraStabilizerInferenceError(
                "beta=0 grid camera bypass evidence differs"
            )
        return
    if (
        bypassed is not False
        or reason is not None
        or geometry_built is not False
        or geometry_reused is not True
    ):
        raise CameraStabilizerInferenceError(
            "active grid camera projection lacks reused source geometry evidence"
        )
    consensus_rows = trace.get("consensus_valid")
    correction_rows = trace.get("correction_rms")
    if (
        not isinstance(consensus_rows, list)
        or len(consensus_rows) != 1
        or not isinstance(consensus_rows[0], list)
        or len(consensus_rows[0]) != legacy.LATENT_FRAME_COUNT
        or any(type(value) is not bool for value in consensus_rows[0])
    ):
        raise CameraStabilizerInferenceError(
            "grid camera per-phase consensus evidence differs"
        )
    if (
        not isinstance(correction_rows, list)
        or len(correction_rows) != 1
        or not isinstance(correction_rows[0], list)
        or len(correction_rows[0]) != legacy.LATENT_FRAME_COUNT
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in correction_rows[0]
        )
    ):
        raise CameraStabilizerInferenceError(
            "grid camera correction diagnostic differs"
        )
    for name in (
        "geometry_valid_tile_count",
        "fit_valid_tile_count",
        "inlier_tile_count",
    ):
        rows = trace.get(name)
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], list)
            or len(rows[0]) != legacy.LATENT_FRAME_COUNT
            or any(type(value) is not int or not 0 <= value <= 16 for value in rows[0])
        ):
            raise CameraStabilizerInferenceError(
                f"grid camera {name} diagnostic differs"
            )
    coverage_rows = trace.get("spatial_coverage_valid")
    if (
        not isinstance(coverage_rows, list)
        or len(coverage_rows) != 1
        or not isinstance(coverage_rows[0], list)
        or len(coverage_rows[0]) != legacy.LATENT_FRAME_COUNT
        or any(type(value) is not bool for value in coverage_rows[0])
    ):
        raise CameraStabilizerInferenceError(
            "grid camera spatial-coverage diagnostic differs"
        )
    for name in (
        "consensus_coefficient_max_abs",
        "tile_relative_fit_residual_max",
    ):
        rows = trace.get(name)
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], list)
            or len(rows[0]) != legacy.LATENT_FRAME_COUNT
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in rows[0]
            )
        ):
            raise CameraStabilizerInferenceError(
                f"grid camera {name} diagnostic differs"
            )


def _validate_estimator_core_step_trace(
    trace: Mapping[str, Any],
    *,
    estimator: str,
    beta: float,
    action_passthrough_object_exact: bool,
) -> None:
    if estimator == "global_svd":
        _validate_core_step_trace(
            trace,
            beta=beta,
            action_passthrough_object_exact=action_passthrough_object_exact,
        )
    elif estimator == "grid_consensus":
        _validate_grid_core_step_trace(
            trace,
            beta=beta,
            action_passthrough_object_exact=action_passthrough_object_exact,
        )
    else:
        raise CameraStabilizerInferenceError("unknown camera trace estimator")


def _validate_grid_rank_authority_step(
    evidence: Any,
    *,
    beta: float,
    step_index: int,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise CameraStabilizerInferenceError(
            "grid step lacks process-group rank authority evidence"
        )
    row = dict(evidence)
    if (
        row.get("schema_version") != GRID_RANK_AUTHORITY_SCHEMA
        or row.get("exact_comparison_proof")
        != GRID_RANK_AUTHORITY_EXACT_PROOF
    ):
        raise CameraStabilizerInferenceError(
            "grid rank authority evidence schema differs"
        )
    world_size = row.get("world_size")
    group_rank = row.get("group_rank")
    if (
        type(world_size) is not int
        or world_size not in (1, legacy.ULYSSES_SIZE)
        or type(group_rank) is not int
        or not 0 <= group_rank < world_size
        or row.get("process_group_rank0") != 0
    ):
        raise CameraStabilizerInferenceError(
            "grid rank authority process-group identity differs"
        )
    collective_sequence = row.get("collective_sequence")
    if not isinstance(collective_sequence, list) or any(
        type(value) is not str for value in collective_sequence
    ):
        raise CameraStabilizerInferenceError(
            "grid rank authority collective sequence is invalid"
        )
    if beta == 0.0:
        if (
            row.get("mode") != "zero_beta_exact_action_no_collective"
            or row.get("rank0_authoritative_broadcast") is not False
            or row.get("source_clean_cross_rank_exact") is not None
            or row.get("source_clean_rank0_reference_exact") is not None
            or row.get("source_clean_certified_this_step") is not False
            or row.get("action_clean_cross_rank_exact") is not None
            or row.get("action_clean_rank0_reference_exact") is not None
            or row.get("local_candidate_success_all_ranks") is not None
            or row.get("pre_broadcast_max_abs_disagreement") is not None
            or row.get("post_broadcast_exact") is not None
            or row.get("post_broadcast_proof") is not None
            or row.get("all_rank_participant_count") != 0
            or collective_sequence
        ):
            raise CameraStabilizerInferenceError(
                "beta=0 grid path issued or claimed a rank collective"
            )
        return row
    distributed = world_size == legacy.ULYSSES_SIZE
    wanted_sequence = []
    if distributed:
        if step_index == 0:
            wanted_sequence.extend(
                [
                    "source_rank0_reference_broadcast",
                    "source_rank0_reference_exact_all_reduce_min",
                ]
            )
        wanted_sequence.extend(
            [
                "action_rank0_reference_broadcast",
                "action_rank0_reference_exact_all_reduce_min",
                "candidate_success_all_reduce_min",
                "executed_clean_broadcast_from_process_group_rank0",
                "pre_broadcast_disagreement_all_reduce_max",
            ]
        )
    disagreement = row.get("pre_broadcast_max_abs_disagreement")
    if (
        row.get("mode")
        != (
            "distributed_rank0_authoritative"
            if distributed
            else "single_process_rank0"
        )
        or row.get("authority_group")
        != (
            "torch.distributed.group.WORLD" if distributed else "local_process"
        )
        or row.get("rank0_authoritative_broadcast") is not distributed
        or row.get("source_clean_cross_rank_exact") is not True
        or row.get("source_clean_rank0_reference_exact") is not True
        or row.get("source_clean_certified_this_step") is not (step_index == 0)
        or row.get("action_clean_cross_rank_exact") is not True
        or row.get("action_clean_rank0_reference_exact") is not True
        or row.get("local_candidate_success_all_ranks") is not True
        or isinstance(disagreement, bool)
        or not isinstance(disagreement, (int, float))
        or not math.isfinite(float(disagreement))
        or float(disagreement) < 0.0
        or (not distributed and float(disagreement) != 0.0)
        or row.get("post_broadcast_exact") is not True
        or row.get("post_broadcast_proof")
        != (
            "rank0_broadcast_then_all_rank_disagreement_MAX_completed"
            if distributed
            else "single_process_object"
        )
        or row.get("all_rank_participant_count") != world_size
        or collective_sequence != wanted_sequence
    ):
        raise CameraStabilizerInferenceError(
            "active grid rank authority evidence differs"
        )
    return row


def validate_execution_trace(
    tri_trace: tri.TriBranchTrace,
    camera_trace: CameraExecutionTrace,
) -> dict[str, Any]:
    """Certify 40 exact-action APG boundaries, 120 forwards and 40 UniPC calls."""

    if not isinstance(tri_trace, tri.TriBranchTrace):
        raise CameraStabilizerInferenceError("invalid tri-branch trace type")
    if not isinstance(camera_trace, CameraExecutionTrace):
        raise CameraStabilizerInferenceError("invalid camera trace type")
    if camera_trace.estimator not in CAMERA_ESTIMATORS:
        raise CameraStabilizerInferenceError("invalid camera trace estimator")
    if tri_trace.sample_calls != 1:
        raise CameraStabilizerInferenceError("expected exactly one sample call")
    branches = list(tri_trace.records)
    controls = list(camera_trace.records)
    if len(branches) != EXPECTED_STEPS or len(controls) != EXPECTED_STEPS:
        raise CameraStabilizerInferenceError(
            "camera stabilizer must cover all 40 official steps"
        )
    sigmas: list[float] = []
    grid_rank_authority_rows: list[dict[str, Any]] = []
    for index, (branch, control) in enumerate(zip(branches, controls)):
        if branch.step_index != index or control.step_index != index:
            raise CameraStabilizerInferenceError("step indices differ or are reordered")
        if (
            branch.transformer_forwards != 3
            or branch.shared_negative_forwards != 1
            or branch.action_forwards != 1
            or branch.noop_forwards != 1
            or branch.original_scheduler_calls != 1
        ):
            raise CameraStabilizerInferenceError(
                "each step requires three forwards and one original UniPC call"
            )
        if (
            branch.official_action_exact_parity is not True
            or branch.official_action_parity_rms_error != 0.0
            or branch.official_action_parity_max_abs_error != 0.0
        ):
            raise CameraStabilizerInferenceError(
                "official full644 action APG exact parity failed"
            )
        if branch.model_id != "transformer_1":
            raise CameraStabilizerInferenceError("full644 must remain single-expert")
        if branch.effective_guidance_scale != legacy.OMEGA_TEXT:
            raise CameraStabilizerInferenceError("APG guidance scale differs")
        if control.beta != camera_trace.beta:
            raise CameraStabilizerInferenceError("camera beta changed during sampling")
        if control.estimator != camera_trace.estimator:
            raise CameraStabilizerInferenceError(
                "camera estimator changed during sampling"
            )
        if type(control.basis_built_this_step) is not bool or type(
            control.basis_reused_from_prior_step
        ) is not bool:
            raise CameraStabilizerInferenceError("camera basis trace is not boolean")
        if type(control.geometry_built_this_step) is not bool or type(
            control.geometry_reused_from_prior_step
        ) is not bool:
            raise CameraStabilizerInferenceError(
                "camera geometry trace is not boolean"
            )
        if control.basis_built_this_step and control.basis_reused_from_prior_step:
            raise CameraStabilizerInferenceError(
                "one step cannot both build and cross-step-reuse the source basis"
            )
        if (
            control.geometry_built_this_step
            and control.geometry_reused_from_prior_step
        ):
            raise CameraStabilizerInferenceError(
                "one step cannot both build and cross-step-reuse source geometry"
            )
        if camera_trace.estimator == "global_svd" and (
            control.geometry_built_this_step
            or control.geometry_reused_from_prior_step
        ):
            raise CameraStabilizerInferenceError(
                "global SVD trace unexpectedly contains grid geometry evidence"
            )
        if camera_trace.estimator == "grid_consensus" and (
            control.basis_built_this_step
            or control.basis_reused_from_prior_step
        ):
            raise CameraStabilizerInferenceError(
                "grid consensus trace unexpectedly contains SVD basis evidence"
            )
        if camera_trace.estimator == "global_svd":
            if control.grid_rank_authority is not None:
                raise CameraStabilizerInferenceError(
                    "global SVD trace unexpectedly contains grid rank authority"
                )
        else:
            grid_rank_authority_rows.append(
                _validate_grid_rank_authority_step(
                    control.grid_rank_authority,
                    beta=camera_trace.beta,
                    step_index=index,
                )
            )
        if not math.isclose(branch.sigma, control.sigma, rel_tol=0.0, abs_tol=1e-8):
            raise CameraStabilizerInferenceError("branch/camera sigma traces differ")
        if not math.isclose(
            branch.timestep, control.timestep, rel_tol=0.0, abs_tol=1e-6
        ):
            raise CameraStabilizerInferenceError("branch/camera timestep traces differ")
        if camera_trace.beta == 0.0 and not control.action_passthrough_object_exact:
            raise CameraStabilizerInferenceError(
                "beta=0 lost exact action passthrough at one or more steps"
            )
        if not isinstance(control.core_trace, Mapping):
            raise CameraStabilizerInferenceError("camera core trace is invalid")
        _validate_estimator_core_step_trace(
            control.core_trace,
            estimator=control.estimator,
            beta=control.beta,
            action_passthrough_object_exact=(
                control.action_passthrough_object_exact
            ),
        )
        if not math.isfinite(branch.sigma) or branch.sigma <= 0.0:
            raise CameraStabilizerInferenceError("official sigma trace is invalid")
        sigmas.append(float(branch.sigma))
    if any(right >= left for left, right in zip(sigmas, sigmas[1:])):
        raise CameraStabilizerInferenceError("official sigma trace is not descending")
    basis_build_count = sum(int(item.basis_built_this_step) for item in controls)
    basis_reuse_count = sum(
        int(item.basis_reused_from_prior_step) for item in controls
    )
    geometry_build_count = sum(
        int(item.geometry_built_this_step) for item in controls
    )
    geometry_reuse_count = sum(
        int(item.geometry_reused_from_prior_step) for item in controls
    )
    active = camera_trace.beta != 0.0
    expected_basis_builds = int(active and camera_trace.estimator == "global_svd")
    expected_basis_reuses = (
        EXPECTED_STEPS - 1
        if active and camera_trace.estimator == "global_svd"
        else 0
    )
    if basis_build_count != expected_basis_builds:
        raise CameraStabilizerInferenceError(
            "camera source basis build count differs for selected estimator"
        )
    if basis_reuse_count != expected_basis_reuses:
        raise CameraStabilizerInferenceError(
            "camera source basis reuse count differs for selected estimator"
        )
    expected_geometry_builds = int(
        active and camera_trace.estimator == "grid_consensus"
    )
    expected_geometry_reuses = (
        EXPECTED_STEPS - 1
        if active and camera_trace.estimator == "grid_consensus"
        else 0
    )
    if geometry_build_count != expected_geometry_builds:
        raise CameraStabilizerInferenceError(
            "camera source geometry build count differs for selected estimator"
        )
    if geometry_reuse_count != expected_geometry_reuses:
        raise CameraStabilizerInferenceError(
            "camera source geometry reuse count differs for selected estimator"
        )
    certificate: dict[str, Any] = {
        "frame_count": EXPECTED_FRAMES,
        "step_count": EXPECTED_STEPS,
        "transformer_forwards": EXPECTED_FORWARDS,
        "official_action_apg_exact_steps": EXPECTED_STEPS,
        "camera_callback_calls": EXPECTED_STEPS,
        "original_unipc_calls": EXPECTED_STEPS,
        "camera_estimator": camera_trace.estimator,
        "camera_basis_build_count": basis_build_count,
        "camera_basis_reuse_count": basis_reuse_count,
        "camera_geometry_build_count": geometry_build_count,
        "camera_geometry_reuse_count": geometry_reuse_count,
        "camera_estimator_state_build_count": (
            basis_build_count + geometry_build_count
        ),
        "camera_estimator_state_reuse_count": (
            basis_reuse_count + geometry_reuse_count
        ),
        "beta_zero_exact_full644_passthrough": (
            camera_trace.beta == 0.0
            and all(item.action_passthrough_object_exact for item in controls)
        ),
        "projection_boundary": (
            "after_exact_official_action_apg_before_original_unipc_step"
        ),
        "custom_integrator": False,
    }
    if camera_trace.estimator == "grid_consensus":
        first_authority = grid_rank_authority_rows[0]
        authority_world_size = int(first_authority["world_size"])
        authority_group_rank = int(first_authority["group_rank"])
        if any(
            int(row["world_size"]) != authority_world_size
            or int(row["group_rank"]) != authority_group_rank
            for row in grid_rank_authority_rows
        ):
            raise CameraStabilizerInferenceError(
                "grid rank authority identity changed during sampling"
            )
        active_grid = camera_trace.beta != 0.0
        disagreements = (
            [
                float(row["pre_broadcast_max_abs_disagreement"])
                for row in grid_rank_authority_rows
            ]
            if active_grid
            else []
        )
        distributed_active = (
            active_grid and authority_world_size == legacy.ULYSSES_SIZE
        )
        certificate["grid_rank_authority"] = {
            "schema_version": GRID_RANK_AUTHORITY_SCHEMA,
            "world_size": authority_world_size,
            "receipt_group_rank": authority_group_rank,
            "process_group_rank0": 0,
            "rank0_receipt_aggregated_steps": (
                EXPECTED_STEPS if authority_group_rank == 0 else 0
            ),
            "source_clean_cross_rank_exact": (
                True if active_grid else None
            ),
            "source_clean_rank0_reference_exact": (
                True if active_grid else None
            ),
            "source_clean_exact_certification_steps": sum(
                int(row["source_clean_certified_this_step"])
                for row in grid_rank_authority_rows
            ),
            "action_clean_cross_rank_exact_steps": sum(
                int(row.get("action_clean_cross_rank_exact") is True)
                for row in grid_rank_authority_rows
            ),
            "all_rank_candidate_success_steps": sum(
                int(row.get("local_candidate_success_all_ranks") is True)
                for row in grid_rank_authority_rows
            ),
            "rank0_authoritative_broadcast_steps": sum(
                int(row["rank0_authoritative_broadcast"])
                for row in grid_rank_authority_rows
            ),
            "post_broadcast_exact_steps": sum(
                int(row.get("post_broadcast_exact") is True)
                for row in grid_rank_authority_rows
            ),
            "all_rank_participant_count_per_step": (
                authority_world_size if active_grid else 0
            ),
            "pre_broadcast_max_abs_disagreement_by_step": disagreements,
            "pre_broadcast_max_abs_disagreement_peak": (
                max(disagreements) if disagreements else None
            ),
            "post_broadcast_proof": (
                "rank0_broadcast_then_all_rank_disagreement_MAX_completed"
                if distributed_active
                else ("single_process_object" if active_grid else None)
            ),
            "zero_beta_no_collective_steps": (
                EXPECTED_STEPS if not active_grid else 0
            ),
            "large_object_all_gather": False,
            "collective_order_identical_all_ranks": (
                True if distributed_active else None
            ),
        }
    payload = {
        "tri_branch": tri_trace.as_dict(),
        "camera": camera_trace.as_dict(),
        "certificate": certificate,
    }
    payload["trace_digest"] = object_sha256(payload)
    return payload


def _validate_legacy_full644_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != legacy.INFERENCE_RECEIPT_SCHEMA:
        raise CameraStabilizerInferenceError("unknown legacy inference receipt")
    adapter = receipt.get("adapter")
    model_input = receipt.get("input")
    sampling = receipt.get("sampling")
    if not all(isinstance(item, Mapping) for item in (adapter, model_input, sampling)):
        raise CameraStabilizerInferenceError("legacy receipt subcontracts are absent")
    if (
        adapter.get("adapter_model_sha256") != FULL644_ADAPTER_SHA256
        or adapter.get("training_receipt_digest")
        != FULL644_TRAINING_RECEIPT_DIGEST
        or adapter.get("training_global_step") != FULL644_TRAINING_STEP
        or adapter.get("tensor_count") != FULL644_ADAPTER_TENSOR_COUNT
        or adapter.get("strictly_reloaded") is not True
        or adapter.get("safe_merged_for_inference") is not True
    ):
        raise CameraStabilizerInferenceError("legacy full644 merge receipt differs")
    if model_input.get("accepted_model_conditions") != [
        "source_video",
        "edit_instruction",
    ]:
        raise CameraStabilizerInferenceError("legacy semantic inputs differ")
    for name in (
        "target_video_argument",
        "target_accessed_by_inference",
        "external_mask_or_swept_tube",
        "external_tracking_pose_or_trajectory",
        "reference_image_or_video",
        "external_shared_i0",
    ):
        if model_input.get(name) is not False:
            raise CameraStabilizerInferenceError(
                "legacy receipt contains a privileged external condition"
            )
    expected_sampling = exact_sampler_contract()
    for name, expected in expected_sampling.items():
        if sampling.get(name) != expected:
            raise CameraStabilizerInferenceError(
                f"legacy sampling contract differs at {name}"
            )


def _method_hashes() -> dict[str, str]:
    paths = {
        "infer_full644_camera_stabilizer.py": Path(__file__),
        "generator_native_camera_stabilizer.py": METHOD_ROOT
        / "generator_native_camera_stabilizer.py",
        "fixed_grid_camera_consensus.py": METHOD_ROOT
        / "fixed_grid_camera_consensus.py",
        "fixed_grid_camera_consensus_stabilizer.py": METHOD_ROOT
        / "fixed_grid_camera_consensus_stabilizer.py",
        "tri_branch_unipc.py": METHOD_ROOT / "tri_branch_unipc.py",
        "infer_lora.py": METHOD_ROOT / "infer_lora.py",
        "motion_residual.py": METHOD_ROOT / "motion_residual.py",
    }
    return {name: legacy.file_sha256(path) for name, path in paths.items()}


def augment_inference_receipt(
    receipt: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    adapter_identity: Mapping[str, Any],
    noop_identity: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the full644 artifact, training receipt and camera runtime trace."""

    _validate_legacy_full644_receipt(receipt)
    required_identity = {
        "adapter_config_sha256": FULL644_ADAPTER_CONFIG_SHA256,
        "adapter_model_sha256": FULL644_ADAPTER_SHA256,
        "training_receipt_file_sha256": FULL644_TRAINING_RECEIPT_FILE_SHA256,
        "training_receipt_digest": FULL644_TRAINING_RECEIPT_DIGEST,
        "training_global_step": FULL644_TRAINING_STEP,
        "target_module_count": FULL644_TARGET_MODULE_COUNT,
        "adapter_tensor_count": FULL644_ADAPTER_TENSOR_COUNT,
    }
    if any(adapter_identity.get(key) != value for key, value in required_identity.items()):
        raise CameraStabilizerInferenceError("retained full644 identity differs")
    certificate = execution_trace.get("certificate")
    if not isinstance(certificate, Mapping):
        raise CameraStabilizerInferenceError("execution certificate is absent")
    if (
        certificate.get("step_count") != EXPECTED_STEPS
        or certificate.get("transformer_forwards") != EXPECTED_FORWARDS
        or certificate.get("official_action_apg_exact_steps") != EXPECTED_STEPS
        or certificate.get("original_unipc_calls") != EXPECTED_STEPS
        or certificate.get("camera_estimator")
        != getattr(args, "camera_estimator", DEFAULT_CAMERA_ESTIMATOR)
    ):
        raise CameraStabilizerInferenceError("execution certificate differs")
    if float(args.beta) == 0.0 and (
        certificate.get("beta_zero_exact_full644_passthrough") is not True
    ):
        raise CameraStabilizerInferenceError("beta=0 passthrough is not certified")
    estimator = getattr(args, "camera_estimator", DEFAULT_CAMERA_ESTIMATOR)
    if estimator == "grid_consensus":
        authority = certificate.get("grid_rank_authority")
        if not isinstance(authority, Mapping):
            raise CameraStabilizerInferenceError(
                "grid receipt lacks rank-authority certificate"
            )
        if (
            authority.get("schema_version") != GRID_RANK_AUTHORITY_SCHEMA
            or authority.get("world_size") != legacy.ULYSSES_SIZE
            or authority.get("receipt_group_rank") != 0
            or authority.get("process_group_rank0") != 0
            or authority.get("rank0_receipt_aggregated_steps")
            != EXPECTED_STEPS
            or authority.get("large_object_all_gather") is not False
        ):
            raise CameraStabilizerInferenceError(
                "grid receipt was not aggregated by Ulysses process-group rank 0"
            )
        if float(args.beta) == 0.0:
            if (
                authority.get("zero_beta_no_collective_steps")
                != EXPECTED_STEPS
                or authority.get("rank0_authoritative_broadcast_steps") != 0
                or authority.get("source_clean_cross_rank_exact") is not None
                or authority.get("post_broadcast_exact_steps") != 0
            ):
                raise CameraStabilizerInferenceError(
                    "beta=0 grid receipt claimed a correction collective"
                )
        elif (
            authority.get("source_clean_cross_rank_exact") is not True
            or authority.get("source_clean_rank0_reference_exact") is not True
            or authority.get("source_clean_exact_certification_steps") != 1
            or authority.get("action_clean_cross_rank_exact_steps")
            != EXPECTED_STEPS
            or authority.get("all_rank_candidate_success_steps")
            != EXPECTED_STEPS
            or authority.get("rank0_authoritative_broadcast_steps")
            != EXPECTED_STEPS
            or authority.get("post_broadcast_exact_steps") != EXPECTED_STEPS
            or authority.get("all_rank_participant_count_per_step")
            != legacy.ULYSSES_SIZE
            or authority.get("collective_order_identical_all_ranks") is not True
            or not isinstance(
                authority.get("pre_broadcast_max_abs_disagreement_by_step"),
                list,
            )
            or len(authority["pre_broadcast_max_abs_disagreement_by_step"])
            != EXPECTED_STEPS
        ):
            raise CameraStabilizerInferenceError(
                "active grid receipt lacks exact all-rank scheduler authority"
            )
    try:
        if estimator == "global_svd":
            core_contract = camera.camera_stabilizer_contract_receipt()
        elif estimator == "grid_consensus":
            core_contract = (
                grid_stabilizer.camera_consensus_stabilizer_contract_receipt()
            )
        else:
            raise CameraStabilizerInferenceError("unsupported camera estimator")
    except Exception as error:
        raise CameraStabilizerInferenceError(
            "cannot serialize camera stabilizer contract"
        ) from error
    if not isinstance(core_contract, Mapping):
        raise CameraStabilizerInferenceError("camera stabilizer contract is invalid")

    value = copy.deepcopy(dict(receipt))
    value.pop("receipt_digest", None)
    value["schema_version"] = INFERENCE_RECEIPT_SCHEMA
    value["method"] = METHOD_NAME
    value["method_files_sha256"] = _method_hashes()
    value["full644_adapter"] = {
        **dict(adapter_identity),
        "strict_240_module_scope": True,
        "strict_480_tensor_reload": True,
        "safe_merge_before_camera_hook": True,
        "frozen_during_inference": True,
    }
    value["input"].update(
        {
            "accepted_external_conditions": [
                "source_video",
                "edit_instruction",
            ],
            "semantic_noop_is_internal_fixed_control": True,
            "support_accessed_by_inference": False,
            "external_mask_flow_pose_track_trajectory": False,
            "edited_first_frame_or_anchor": False,
        }
    )
    value["sampling"].update(
        {
            "tri_branch_contract": tri.sampler_contract(),
            "projection_boundary": (
                "after_exact_official_action_apg_before_original_unipc_step"
            ),
            "per_step_transformer_forwards": 3,
            "total_transformer_forwards": EXPECTED_FORWARDS,
            "custom_integrator": False,
        }
    )
    value["prompt_contract"].update(
        {
            "semantic_noop_instruction_sha256": hashlib.sha256(
                motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "semantic_noop": dict(noop_identity),
        }
    )
    value["camera_stabilizer"] = {
        "beta": float(args.beta),
        "estimator": estimator,
        "enabled": True,
        "source_and_instruction_only": True,
        "core_contract": dict(core_contract),
        "execution": dict(execution_trace),
    }
    if estimator == "global_svd":
        value["camera_stabilizer"]["camera_edit_requested"] = False
        value["camera_stabilizer"]["estimator_semantics"] = (
            "global_source_tangent_thin_svd"
        )
    else:
        value["camera_stabilizer"]["estimator_semantics"] = (
            "fixed_grid_physical_coefficients_median_MAD_trimmed_consensus_per_phase"
        )
        value["camera_stabilizer"]["nonconsensus_phase"] = (
            "exact_action_value_passthrough"
        )
    value["forbidden_external_conditions"] = list(FORBIDDEN_EXTERNAL_CONDITIONS)
    value["experimental_inference"] = True
    value["production_claim_forbidden"] = True
    value["scientific_claim_authorized"] = False
    value["receipt_digest"] = object_sha256(value)
    return value


class _CameraWrappedModel:
    """Delegate everything except the single sampling call to merged full644."""

    def __init__(self, model: Any, *, args: argparse.Namespace, state: dict[str, Any]):
        self._model = model
        self._args = args
        self._state = state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def sample(self, *sample_args: Any, **sample_kwargs: Any) -> Any:
        if "execution_trace" in self._state:
            raise CameraStabilizerInferenceError("model.sample ran more than once")
        noop_tokens = self._state.get("noop_tokens")
        if (
            not isinstance(noop_tokens, tuple)
            or len(noop_tokens) != 2
            or "device" not in sample_kwargs
        ):
            raise CameraStabilizerInferenceError(
                "semantic no-op tokens/device were not retained before sampling"
            )
        source_values = sample_kwargs.get("multi_video_vae_latents")
        if not isinstance(source_values, list) or len(source_values) != 1:
            raise CameraStabilizerInferenceError(
                "sampling must receive exactly one source-video latent"
            )
        source_clean = source_values[0]
        shape = tuple(int(value) for value in getattr(source_clean, "shape", ()))
        if len(shape) != 5 or shape[0] != 1 or shape[2] != legacy.LATENT_FRAME_COUNT:
            raise CameraStabilizerInferenceError("source latent geometry differs")
        noop_embeddings, noop_identity = _encode_semantic_noop_prompt(
            self._model,
            noop_tokens[0],
            noop_tokens[1],
            device=sample_kwargs["device"],
        )
        estimator = getattr(
            self._args,
            "camera_estimator",
            DEFAULT_CAMERA_ESTIMATOR,
        )
        if estimator == "global_svd":
            callback = CameraTangentCallback(
                source_clean_field=source_clean,
                beta=float(self._args.beta),
            )
        elif estimator == "grid_consensus":
            callback = CameraGridConsensusCallback(
                source_clean_field=source_clean,
                beta=float(self._args.beta),
            )
        else:
            raise CameraStabilizerInferenceError("unsupported camera estimator")
        wan_path = (
            Path(self._args.bernini_root).expanduser().resolve(strict=True)
            / "bernini/models/wan_diffusion.py"
        )
        try:
            with tri.tri_branch_unipc_hook(
                self._model,
                noop_prompt_embeds=noop_embeddings,
                latent_shape=shape,
                clean_field_callback=callback,
                bernini_commit=self._args.expected_bernini_commit,
                wan_diffusion_path=wan_path,
                expected_steps=EXPECTED_STEPS,
                expected_flow_shift=legacy.FLOW_SHIFT,
            ) as tri_trace:
                generated = self._model.sample(*sample_args, **sample_kwargs)
        except tri.TriBranchHookError as error:
            raise CameraStabilizerInferenceError(str(error)) from error
        self._state["noop_identity"] = noop_identity
        self._state["execution_trace"] = validate_execution_trace(
            tri_trace, callback.trace
        )
        return generated


def _legacy_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--bernini-root",
        args.bernini_root,
        "--veomni-root",
        args.veomni_root,
        "--checkpoint",
        args.checkpoint,
        "--adapter-checkpoint",
        args.adapter_checkpoint,
        "--source-video",
        args.source_video,
        "--instruction",
        args.instruction,
        "--output",
        args.output,
        "--num-inference-steps",
        str(EXPECTED_STEPS),
        "--seed",
        str(EXPECTED_SEED),
        "--expected-bernini-commit",
        args.expected_bernini_commit,
        "--expected-veomni-commit",
        args.expected_veomni_commit,
        "--expected-checkpoint-tree-sha256",
        args.expected_checkpoint_tree_sha256,
        "--method-source-revision",
        args.method_source_revision,
        "--method-source-archive-sha256",
        args.method_source_archive_sha256,
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    try:
        source_path = legacy._plain_file(
            Path(args.source_video).expanduser().resolve(strict=True),
            label="source video",
        )
        adapter = legacy.resolve_adapter_bundle(args.adapter_checkpoint)
    except legacy.InferenceContractError as error:
        raise CameraStabilizerInferenceError(str(error)) from error
    source_sha = legacy.file_sha256(source_path)
    instruction_sha = hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()
    if source_sha != args.expected_source_sha256:
        raise CameraStabilizerInferenceError("source video SHA256 differs")
    if instruction_sha != args.expected_instruction_sha256:
        raise CameraStabilizerInferenceError("instruction SHA256 differs")
    adapter_identity = validate_full644_adapter_bundle(adapter)
    exact_sampler_contract()

    state: dict[str, Any] = {"adapter_identity": adapter_identity}
    original_loader = legacy._strict_load_and_merge_adapter
    original_tokenizer = legacy._tokenize_training_prompt
    original_writer = legacy._atomic_write_json

    def strict_full644_load_and_wrap(
        base_model: Any,
        observed_adapter: legacy.AdapterBundle,
        expected_targets: Sequence[str],
    ) -> tuple[Any, int]:
        if "merged_model" in state:
            raise CameraStabilizerInferenceError("adapter loader ran more than once")
        if observed_adapter != adapter:
            raise CameraStabilizerInferenceError("resolved adapter bundle changed")
        if list(expected_targets) != legacy.expected_lora_target_modules():
            raise CameraStabilizerInferenceError("240-module target scope differs")
        merged, tensor_count = original_loader(
            base_model, observed_adapter, expected_targets
        )
        if tensor_count != FULL644_ADAPTER_TENSOR_COUNT:
            raise CameraStabilizerInferenceError("strict loader did not return 480 tensors")
        state["merged_model"] = merged
        return _CameraWrappedModel(merged, args=args, state=state), tensor_count

    def tokenize_action_and_fixed_noop(tokenizer: Any, text: str) -> tuple[Any, Any]:
        if "noop_tokens" in state:
            raise CameraStabilizerInferenceError("action prompt tokenized more than once")
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        expected_action = legacy.build_training_prompt(
            args.instruction, prompt_cleaner=prompt_clean
        )
        if text != expected_action:
            raise CameraStabilizerInferenceError("action prompt differs before tokenization")
        action_tokens = original_tokenizer(tokenizer, text)
        noop_prompt = legacy.build_training_prompt(
            motion.DEFAULT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
        )
        state["noop_tokens"] = original_tokenizer(tokenizer, noop_prompt)
        return action_tokens

    def augment_then_write(path: Path, receipt: Mapping[str, Any]) -> None:
        execution = state.get("execution_trace")
        noop_identity = state.get("noop_identity")
        if not isinstance(execution, Mapping) or not isinstance(noop_identity, Mapping):
            raise CameraStabilizerInferenceError(
                "camera execution evidence is absent before receipt publication"
            )
        augmented = augment_inference_receipt(
            receipt,
            args=args,
            adapter_identity=adapter_identity,
            noop_identity=noop_identity,
            execution_trace=execution,
        )
        if isinstance(receipt, dict):
            receipt.clear()
            receipt.update(augmented)
            original_writer(path, receipt)
        else:
            original_writer(path, augmented)

    legacy._strict_load_and_merge_adapter = strict_full644_load_and_wrap
    legacy._tokenize_training_prompt = tokenize_action_and_fixed_noop
    legacy._atomic_write_json = augment_then_write
    try:
        return legacy.main(_legacy_arguments(args))
    except (legacy.InferenceContractError, tri.TriBranchHookError) as error:
        raise CameraStabilizerInferenceError(str(error)) from error
    finally:
        legacy._strict_load_and_merge_adapter = original_loader
        legacy._tokenize_training_prompt = original_tokenizer
        legacy._atomic_write_json = original_writer


if __name__ == "__main__":
    raise SystemExit(main())
