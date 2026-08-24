#!/usr/bin/env python3
"""Source+instruction-only 81-frame Bernini V12 EPMC evaluation.

The only model-facing deployment conditions are one source video and one edit
instruction.  A hash-pinned training artifact supplies a K=2 support
prototype; support/target videos and all spatial or trajectory oracles are
deliberately absent from this program's CLI and runtime.

The runner reuses the audited V11 Bernini load, prompt, sampler, APG routing,
VAE, and transactional-video path.  It generates frozen action/no-op proposal
latents internally and then evaluates five same-render-seed arms:

``B0``
    Official Bernini no-op render with no motion branch installed.
``Z0``
    Complete few-shot branch and carrier with byte-exact zero motion code.
``PROTO``
    The training-support prototype loaded from the pinned state/receipt.
``REVERSE``
    The same prototype with only its 21 phase slots reversed.
``SHUFFLE``
    The same prototype with the frozen non-trivial phase permutation.

There is intentionally no held-out-oracle arm.  Ground-truth evaluation must
run later in a separate process after these source-only videos are committed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import counterfactual_proposal_motion_runtime as motion_runtime  # noqa: E402
import fewshot_episode_io as episode_io  # noqa: E402
import fewshot_motion_branch as motion_branch  # noqa: E402
import fewshot_privileged_motion_code as epmc  # noqa: E402
import fewshot_proposal_motion_carrier as carrier_core  # noqa: E402
import infer_counterfactual_proposal_motion_oracle as v11  # noqa: E402
import infer_lora as legacy  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402
import source_kv_route_batches as route_batches  # noqa: E402


RECEIPT_SCHEMA = "bernini-epmc-v12-source-instruction-inference-v1"
PROTOTYPE_STATE_SCHEMA = "bernini-epmc-v12-tied-prototype-state-v1"
PROTOTYPE_RECEIPT_SCHEMA = "bernini-epmc-v12-tied-prototype-training-receipt-v1"

EXPECTED_FRAMES = 81
EXPECTED_FPS = 25
EXPECTED_STEPS = 40
EXPECTED_ULYSSES_SIZE = 4
EXPECTED_BUCKET_HW = (480, 496)
EXPECTED_SOURCE_TOKENS = 19_530
EXPECTED_LATENT_SHAPE = (1, 16, 21, 60, 62)
EXPECTED_PATCH_GRID = (30, 31)
EXPECTED_SOURCE_SHA256 = (
    "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
)
PROPOSAL_SEED = 2027
RENDER_SEED = 2028
OUTER_CPMR_GATE = motion_branch.OUTER_CPMR_GATE

PROTOTYPE_TENSOR_KEYS = ("phase_gates", "block_head_gates")
EXPECTED_SUPPORT_IIDS = ("841b5e0080a1441d", "7262dd490cbf42c5")
EXTERNAL_SEMANTIC_INPUTS = ("source_video", "instruction")
FORBIDDEN_INFERENCE_ARGUMENTS = (
    "target",
    "target_video",
    "support",
    "support_video",
    "mask",
    "flow",
    "pose",
    "track",
    "trajectory",
    "reference",
    "reference_image",
    "reference_video",
    "edited_first_frame",
)
ARM_ORDER = ("B0", "Z0", "PROTO", "REVERSE", "SHUFFLE")
PATCHED_ARM_ORDER = ARM_ORDER[1:]
OUTPUT_ORDER = ("proposal_action", "proposal_noop", *ARM_ORDER)
ARM_OUTER_GATES = {
    "B0": None,
    "Z0": OUTER_CPMR_GATE,
    "PROTO": OUTER_CPMR_GATE,
    "REVERSE": OUTER_CPMR_GATE,
    "SHUFFLE": OUTER_CPMR_GATE,
}

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_STATE_BYTES = 1 << 20
_MAX_RECEIPT_BYTES = 1 << 18


class FewShotMotionInferenceError(RuntimeError):
    """Raised before an ambiguous or privileged inference artifact is used."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FewShotMotionInferenceError(
            f"value cannot be represented as canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(value: Any) -> str:
    """Hash tensor metadata plus canonical contiguous CPU payload bytes."""

    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise FewShotMotionInferenceError("digest input must be a non-meta tensor")
    detached = value.detach().reshape(-1).repeat(1).cpu()
    metadata = {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise FewShotMotionInferenceError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise FewShotMotionInferenceError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FewShotMotionInferenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise FewShotMotionInferenceError(f"non-finite JSON number: {value}")


def _decode_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except FewShotMotionInferenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FewShotMotionInferenceError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise FewShotMotionInferenceError(f"{label} must contain one JSON object")
    return dict(value)


def _plain_absolute_file(
    value: str | Path, *, label: str, maximum_bytes: Optional[int] = None
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FewShotMotionInferenceError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise FewShotMotionInferenceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FewShotMotionInferenceError(f"{label} must be a plain regular file")
    if maximum_bytes is not None and not 0 < info.st_size <= maximum_bytes:
        raise FewShotMotionInferenceError(
            f"{label} size must lie in [1,{maximum_bytes}] bytes"
        )
    return path.resolve(strict=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--prototype-state", required=True)
    parser.add_argument("--prototype-receipt", required=True)
    parser.add_argument("--expected-prototype-state-sha256", required=True)
    parser.add_argument("--expected-prototype-receipt-sha256", required=True)
    parser.add_argument("--allow-no-go-diagnostic", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument(
        "--expected-instruction-sha256", default=EXPECTED_INSTRUCTION_SHA256
    )
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
    if (
        type(args.instruction) is not str
        or not args.instruction.strip()
        or "\x00" in args.instruction
    ):
        raise FewShotMotionInferenceError(
            "instruction must be non-empty text without NUL"
        )
    instruction_sha = hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()
    if args.expected_instruction_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise FewShotMotionInferenceError("held-out instruction pin differs")
    if instruction_sha != args.expected_instruction_sha256:
        raise FewShotMotionInferenceError("instruction SHA256 differs")
    if args.expected_source_sha256 != EXPECTED_SOURCE_SHA256:
        raise FewShotMotionInferenceError("held-out source SHA256 pin differs")
    if args.num_inference_steps != EXPECTED_STEPS:
        raise FewShotMotionInferenceError("V12 inference is fixed to 40 solver steps")
    if args.proposal_seed != PROPOSAL_SEED or args.render_seed != RENDER_SEED:
        raise FewShotMotionInferenceError("proposal/render seeds are frozen")
    for name in ("proposal_seed", "render_seed"):
        value = getattr(args, name)
        if type(value) is not int or not 0 <= value < 2**63:
            raise FewShotMotionInferenceError(f"{name} must lie in [0,2^63)")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if type(value) is not str or _SHA1.fullmatch(value) is None:
            raise FewShotMotionInferenceError(f"{name} must be a full lowercase SHA-1")
    for name in (
        "expected_source_sha256",
        "expected_instruction_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_prototype_state_sha256",
        "expected_prototype_receipt_sha256",
        "method_source_archive_sha256",
    ):
        _required_sha256(getattr(args, name), label=name)
    if args.expected_bernini_commit != legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise FewShotMotionInferenceError("unsupported Bernini source revision")
    if args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise FewShotMotionInferenceError("unsupported VeOmni source revision")
    if args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise FewShotMotionInferenceError("unsupported checkpoint tree")
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output.suffix:
        raise FewShotMotionInferenceError(
            "output-dir must be an absolute directory path without a suffix"
        )
    for name in ("source_video", "prototype_state", "prototype_receipt"):
        if not Path(getattr(args, name)).expanduser().is_absolute():
            raise FewShotMotionInferenceError(f"{name} must be an absolute path")
    if Path(args.prototype_state).expanduser() == Path(args.prototype_receipt).expanduser():
        raise FewShotMotionInferenceError("prototype state and receipt paths must differ")


def validate_tied_prototype_tensors(
    tensors: Mapping[str, Any],
) -> tuple[epmc.MotionCode, Any]:
    """Validate the strict first-canary state and derive its tied 36D view."""

    import torch

    if not isinstance(tensors, Mapping):
        raise FewShotMotionInferenceError("prototype state must be a tensor mapping")
    if set(tensors) != set(PROTOTYPE_TENSOR_KEYS):
        raise FewShotMotionInferenceError(
            "prototype state tensor keys must be exactly "
            + ",".join(PROTOTYPE_TENSOR_KEYS)
        )
    phase = tensors["phase_gates"]
    block_head = tensors["block_head_gates"]
    for name, value, shape in (
        ("phase_gates", phase, (1, epmc.LATENT_PHASES)),
        (
            "block_head_gates",
            block_head,
            (1, epmc.MOTION_BLOCKS, epmc.ATTENTION_HEADS),
        ),
    ):
        if not isinstance(value, torch.Tensor):
            raise FewShotMotionInferenceError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise FewShotMotionInferenceError(f"{name} must load on CPU")
        if value.dtype != torch.float32:
            raise FewShotMotionInferenceError(f"{name} must be float32")
        if tuple(int(item) for item in value.shape) != shape:
            raise FewShotMotionInferenceError(f"{name} must have exact shape {shape}")
        if not value.is_contiguous():
            raise FewShotMotionInferenceError(f"{name} must be contiguous")
        if value.requires_grad:
            raise FewShotMotionInferenceError(f"{name} must be detached state")
        if not bool(torch.isfinite(value).all().item()):
            raise FewShotMotionInferenceError(f"{name} contains NaN or infinity")
    try:
        code = epmc.MotionCode(phase.detach().clone(), block_head.detach().clone())
    except epmc.PrivilegedMotionCodeContractError as error:
        raise FewShotMotionInferenceError(str(error)) from error

    canonical_heads = code.block_head_gates[:, :, :1].expand_as(
        code.block_head_gates
    ).contiguous()
    actual_bytes = code.block_head_gates.contiguous().view(torch.uint8)
    canonical_bytes = canonical_heads.view(torch.uint8)
    if not bool(torch.equal(actual_bytes, canonical_bytes)):
        raise FewShotMotionInferenceError(
            "first-canary state requires all 12 heads byte-exact tied per block"
        )
    tied_code = torch.cat(
        (code.phase_gates[:, 1:], code.block_head_gates[:, :, 0]), dim=1
    ).contiguous()
    if (
        tuple(int(item) for item in tied_code.shape) != (1, 36)
        or tied_code.device.type != "cpu"
        or tied_code.dtype != torch.float32
        or not tied_code.is_contiguous()
    ):
        raise FewShotMotionInferenceError("derived tied code is not CPU FP32 [1,36]")
    return code, tied_code


def build_prototype_training_receipt(
    *,
    state_filename: str,
    state_file_sha256: str,
    motion_code: epmc.MotionCode,
    tied_code_36d: Any,
    support_tied_code_36d_sha256: Sequence[str],
    training_gate_receipt_sha256: str,
    representability_gate: str = "GO",
) -> dict[str, Any]:
    """Construct the one accepted K=2 support-only prototype receipt schema."""

    motion_code.validate()
    if Path(state_filename).name != state_filename or not state_filename.endswith(
        ".safetensors"
    ):
        raise FewShotMotionInferenceError(
            "prototype receipt state filename must be a plain .safetensors basename"
        )
    _required_sha256(state_file_sha256, label="prototype state file SHA256")
    if isinstance(support_tied_code_36d_sha256, (str, bytes)):
        raise FewShotMotionInferenceError(
            "exactly two support tied-code SHA256 values are required"
        )
    support_hash_values = tuple(support_tied_code_36d_sha256)
    if len(support_hash_values) != 2:
        raise FewShotMotionInferenceError(
            "exactly two support tied-code SHA256 values are required"
        )
    support_code_hashes = [
        _required_sha256(value, label=f"support {index} tied-code SHA256")
        for index, value in enumerate(support_hash_values, start=1)
    ]
    _required_sha256(
        training_gate_receipt_sha256, label="training gate receipt SHA256"
    )
    if representability_gate not in ("GO", "NO_GO"):
        raise FewShotMotionInferenceError(
            "representability gate must be exactly GO or NO_GO"
        )
    _, canonical_tied = validate_tied_prototype_tensors(
        {
            "phase_gates": motion_code.phase_gates,
            "block_head_gates": motion_code.block_head_gates,
        }
    )
    if _tensor_sha256(tied_code_36d) != _tensor_sha256(canonical_tied):
        raise FewShotMotionInferenceError("supplied tied 36D view differs from prototype")
    state = {
        "schema_version": PROTOTYPE_STATE_SCHEMA,
        "format": "safetensors",
        "filename": state_filename,
        "file_sha256": state_file_sha256,
        "tensor_keys": list(PROTOTYPE_TENSOR_KEYS),
        "tensors": {
            "phase_gates": {
                "shape": [1, 21],
                "dtype": "torch.float32",
                "sha256": _tensor_sha256(motion_code.phase_gates),
            },
            "block_head_gates": {
                "shape": [1, 16, 12],
                "dtype": "torch.float32",
                "sha256": _tensor_sha256(motion_code.block_head_gates),
            },
            "tied_code_36d": {
                "shape": [1, 36],
                "dtype": "torch.float32",
                "sha256": _tensor_sha256(canonical_tied),
            },
        },
    }
    training = {
        "episode_config_schema": episode_io.CONFIG_SCHEMA,
        "episode_config_sha256": episode_io.REFERENCE_CONFIG_SHA256,
        "support_count": episode_io.EXPECTED_SUPPORT_COUNT,
        "support_iids": list(EXPECTED_SUPPORT_IIDS),
        "support_tied_code_36d_sha256": support_code_hashes,
        "prototype_aggregation": "arithmetic_mean_k2",
        "prototype_source": "training_support_codes_only",
        "training_gate_receipt_sha256": training_gate_receipt_sha256,
        "representability_gate": representability_gate,
        "heldout_use_definition": (
            "optimizer_or_model_tensor_use; hash_or_metadata_preflight_is_not_use"
        ),
        "heldout_hash_or_metadata_preflight_allowed": True,
        "support_targets_training_only": True,
        "heldout_source_used": False,
        "heldout_target_used": False,
        "heldout_oracle_used": False,
    }
    deployment = {
        "semantic_inputs": list(EXTERNAL_SEMANTIC_INPUTS),
        "source_instruction_only": True,
        "support_available_at_inference": False,
        "target_available_at_inference": False,
        "heldout_oracle_available_at_inference": False,
        "forbidden_inference_arguments": list(FORBIDDEN_INFERENCE_ARGUMENTS),
    }
    contracts = {
        "epmc_schema_version": epmc.SCHEMA_VERSION,
        "epmc_contract_sha256": epmc.CONTRACT_RECEIPT_SHA256,
        "motion_branch_schema_version": motion_branch.SCHEMA_VERSION,
        "proposal_carrier_schema_version": carrier_core.SCHEMA_VERSION,
        "first_canary_parameterization": (
            "20_phase_plus_16_block_tied_across_12_heads"
        ),
        "trainable_dimension": motion_branch.TIED_CODE_DIMENSION,
        "phase0_exact_positive_zero": True,
        "cpu_fp32_contiguous_state": True,
        "all_12_heads_byte_exact_tied_per_block": True,
    }
    payload = {
        "schema_version": PROTOTYPE_RECEIPT_SCHEMA,
        "prototype_state": state,
        "training_provenance": training,
        "deployment_contract": deployment,
        "implementation_contracts": contracts,
    }
    return {**payload, "receipt_digest": _object_sha256(payload)}


def _validate_prototype_receipt_envelope(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping):
        raise FewShotMotionInferenceError("prototype receipt must be an object")
    _require_exact_fields(
        receipt,
        frozenset(
            {
                "schema_version",
                "prototype_state",
                "training_provenance",
                "deployment_contract",
                "implementation_contracts",
                "receipt_digest",
            }
        ),
        label="prototype receipt",
    )
    if receipt.get("schema_version") != PROTOTYPE_RECEIPT_SCHEMA:
        raise FewShotMotionInferenceError("prototype receipt schema differs")
    declared = _required_sha256(
        receipt.get("receipt_digest"), label="prototype receipt digest"
    )
    payload = dict(receipt)
    payload.pop("receipt_digest")
    if _object_sha256(payload) != declared:
        raise FewShotMotionInferenceError("prototype receipt digest differs")
    training = receipt.get("training_provenance")
    deployment = receipt.get("deployment_contract")
    if not isinstance(training, Mapping) or not isinstance(deployment, Mapping):
        raise FewShotMotionInferenceError("prototype receipt contracts must be objects")
    if any(
        training.get(name) is not False
        for name in ("heldout_source_used", "heldout_target_used", "heldout_oracle_used")
    ):
        raise FewShotMotionInferenceError(
            "prototype receipt reports a held-out source/target/oracle dependency"
        )
    support_hashes = training.get("support_tied_code_36d_sha256")
    if (
        training.get("support_iids") != list(EXPECTED_SUPPORT_IIDS)
        or training.get("support_count") != 2
        or not isinstance(support_hashes, list)
        or len(support_hashes) != 2
        or any(_SHA256.fullmatch(value or "") is None for value in support_hashes)
        or _SHA256.fullmatch(training.get("training_gate_receipt_sha256") or "")
        is None
        or training.get("representability_gate") not in ("GO", "NO_GO")
        or training.get("heldout_use_definition")
        != "optimizer_or_model_tensor_use; hash_or_metadata_preflight_is_not_use"
        or training.get("heldout_hash_or_metadata_preflight_allowed") is not True
    ):
        raise FewShotMotionInferenceError(
            "prototype receipt lacks the exact K=2 support/GO binding"
        )
    if (
        training.get("support_targets_training_only") is not True
        or deployment.get("source_instruction_only") is not True
        or deployment.get("support_available_at_inference") is not False
        or deployment.get("target_available_at_inference") is not False
        or deployment.get("heldout_oracle_available_at_inference") is not False
    ):
        raise FewShotMotionInferenceError(
            "prototype receipt violates source+instruction-only deployment"
        )
    if deployment.get("semantic_inputs") != list(EXTERNAL_SEMANTIC_INPUTS):
        raise FewShotMotionInferenceError("prototype semantic inputs differ")
    if deployment.get("forbidden_inference_arguments") != list(
        FORBIDDEN_INFERENCE_ARGUMENTS
    ):
        raise FewShotMotionInferenceError("prototype forbidden-input registry differs")


def validate_prototype_training_receipt(
    receipt: Mapping[str, Any],
    *,
    state_filename: str,
    state_file_sha256: str,
    motion_code: epmc.MotionCode,
    tied_code_36d: Any,
) -> dict[str, Any]:
    _validate_prototype_receipt_envelope(receipt)
    training = dict(receipt["training_provenance"])
    expected = build_prototype_training_receipt(
        state_filename=state_filename,
        state_file_sha256=state_file_sha256,
        motion_code=motion_code,
        tied_code_36d=tied_code_36d,
        support_tied_code_36d_sha256=training[
            "support_tied_code_36d_sha256"
        ],
        training_gate_receipt_sha256=training["training_gate_receipt_sha256"],
        representability_gate=training["representability_gate"],
    )
    if dict(receipt) != expected:
        raise FewShotMotionInferenceError(
            "prototype receipt does not exactly bind the loaded first-canary state"
        )
    return expected


@dataclass(frozen=True)
class PrototypeBundle:
    state_path: Path
    receipt_path: Path
    state_file_sha256: str
    receipt_file_sha256: str
    motion_code_cpu: epmc.MotionCode
    tied_code_36d_cpu: Any
    training_receipt: Mapping[str, Any]
    representability_gate: str

    def audit_receipt(self) -> dict[str, Any]:
        return {
            "state_path": str(self.state_path),
            "receipt_path": str(self.receipt_path),
            "state_file_sha256": self.state_file_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "state_schema_version": PROTOTYPE_STATE_SCHEMA,
            "training_receipt_schema_version": PROTOTYPE_RECEIPT_SCHEMA,
            "motion_code": self.motion_code_cpu.audit_receipt(),
            "tied_code_36d_sha256": _tensor_sha256(self.tied_code_36d_cpu),
            "all_12_heads_byte_exact_tied_per_block": True,
            "support_training_only": True,
            "source_instruction_only_deployment": True,
            "heldout_oracle_used": False,
            "training_receipt_digest": self.training_receipt["receipt_digest"],
            "representability_gate": self.representability_gate,
        }


def _load_safetensors_cpu(path: Path) -> Mapping[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise FewShotMotionInferenceError(
            "safetensors is required to load the prototype state"
        ) from error
    try:
        return load_file(str(path), device="cpu")
    except Exception as error:
        raise FewShotMotionInferenceError("could not load prototype safetensors") from error


def load_prototype_bundle(
    state_value: str | Path,
    receipt_value: str | Path,
    *,
    expected_state_sha256: str,
    expected_receipt_sha256: str,
    allow_no_go_diagnostic: bool = False,
) -> PrototypeBundle:
    """Fail closed on receipt semantics before deserializing safetensors."""

    _required_sha256(expected_state_sha256, label="expected prototype state SHA256")
    _required_sha256(
        expected_receipt_sha256, label="expected prototype receipt SHA256"
    )
    state_path = _plain_absolute_file(
        state_value, label="prototype state", maximum_bytes=_MAX_STATE_BYTES
    )
    receipt_path = _plain_absolute_file(
        receipt_value, label="prototype receipt", maximum_bytes=_MAX_RECEIPT_BYTES
    )
    if state_path == receipt_path:
        raise FewShotMotionInferenceError("prototype state and receipt resolve identically")
    if state_path.suffix != ".safetensors":
        raise FewShotMotionInferenceError("prototype state must use .safetensors")
    state_sha = _file_sha256(state_path)
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if state_sha != expected_state_sha256:
        raise FewShotMotionInferenceError("prototype state file SHA256 differs")
    if receipt_sha != expected_receipt_sha256:
        raise FewShotMotionInferenceError("prototype receipt file SHA256 differs")
    receipt = _decode_json_object(receipt_bytes, label="prototype receipt")
    _validate_prototype_receipt_envelope(receipt)
    tensors = _load_safetensors_cpu(state_path)
    code, tied = validate_tied_prototype_tensors(tensors)
    validated_receipt = validate_prototype_training_receipt(
        receipt,
        state_filename=state_path.name,
        state_file_sha256=state_sha,
        motion_code=code,
        tied_code_36d=tied,
    )
    gate = validated_receipt["training_provenance"]["representability_gate"]
    if gate == "NO_GO" and allow_no_go_diagnostic is not True:
        raise FewShotMotionInferenceError(
            "NO_GO prototype requires explicit --allow-no-go-diagnostic"
        )
    if gate == "GO" and allow_no_go_diagnostic is True:
        raise FewShotMotionInferenceError(
            "--allow-no-go-diagnostic is valid only for a NO_GO prototype"
        )
    return PrototypeBundle(
        state_path=state_path,
        receipt_path=receipt_path,
        state_file_sha256=state_sha,
        receipt_file_sha256=receipt_sha,
        motion_code_cpu=code,
        tied_code_36d_cpu=tied,
        training_receipt=validated_receipt,
        representability_gate=gate,
    )


def build_arm_motion_codes(prototype: epmc.MotionCode) -> dict[str, epmc.MotionCode]:
    """Build the four legal patched-arm codes; no oracle code can enter."""

    prototype.validate()
    if prototype.batch_size != 1 or prototype.phase_gates.device.type != "cpu":
        raise FewShotMotionInferenceError("prototype arm code must be CPU batch-1")
    _, tied = validate_tied_prototype_tensors(
        {
            "phase_gates": prototype.phase_gates,
            "block_head_gates": prototype.block_head_gates,
        }
    )
    if tuple(int(item) for item in tied.shape) != (1, 36):
        raise FewShotMotionInferenceError("prototype does not expose the tied 36D view")
    return {
        "Z0": motion_branch.canonical_tied_noop_motion_code(device="cpu"),
        "PROTO": epmc.MotionCode(
            prototype.phase_gates.clone(), prototype.block_head_gates.clone()
        ),
        "REVERSE": epmc.permute_motion_code_phases(
            prototype, epmc.REVERSE_PHASE_INDICES
        ),
        "SHUFFLE": epmc.permute_motion_code_phases(
            prototype, epmc.SHUFFLE_PHASE_INDICES
        ),
    }


def _motion_code_to_device(code: epmc.MotionCode, device: Any) -> epmc.MotionCode:
    return epmc.MotionCode(
        code.phase_gates.to(device=device, dtype=__import__("torch").float32),
        code.block_head_gates.to(device=device, dtype=__import__("torch").float32),
    )


def validate_arm_latents(values: Mapping[str, Any]) -> dict[str, bool]:
    """Validate five full latent trajectories and the mandatory Z0 identity."""

    if set(values) != set(ARM_ORDER):
        raise FewShotMotionInferenceError("arm latent set differs from frozen five arms")
    for name in ARM_ORDER:
        value = values[name]
        if tuple(int(item) for item in value.shape) != EXPECTED_LATENT_SHAPE:
            raise FewShotMotionInferenceError(f"{name} latent shape differs")
    z0_exact = v11._tensor_bytes_equal(values["B0"], values["Z0"])
    if not z0_exact:
        raise FewShotMotionInferenceError("Z0 differs bytewise from B0")
    return {
        "z0_full_latent_byte_exact_b0": True,
        "proto_differs_from_z0": not v11._tensor_bytes_equal(
            values["PROTO"], values["Z0"]
        ),
        "reverse_differs_from_proto": not v11._tensor_bytes_equal(
            values["REVERSE"], values["PROTO"]
        ),
        "shuffle_differs_from_proto": not v11._tensor_bytes_equal(
            values["SHUFFLE"], values["PROTO"]
        ),
    }


def _validate_runtime_trace(trace: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    expected = (
        trace.get("sample_calls") == 1
        and trace.get("shared_step_calls") == 2 * EXPECTED_STEPS
        and trace.get("completed_steps") == EXPECTED_STEPS
        and trace.get("all_prompt_identity_exact") is True
        and trace.get("all_paired_state_identity_exact") is True
        and trace.get("all_bindings_complete") is True
        and isinstance(trace.get("records"), list)
        and len(trace["records"]) == EXPECTED_STEPS
    )
    if not expected:
        raise FewShotMotionInferenceError(f"{arm} runtime trace is incomplete")
    return dict(trace)


def _save_outputs(
    *,
    output_dir: Path,
    values: Mapping[str, Any],
    vae: Any,
    device: Any,
    save_output_fn: Any,
) -> dict[str, Any]:
    """Use the V11 decode/save transaction for all seven 81-frame videos."""

    from bernini.pipeline import _vae_decode
    from tools import materialize_vae

    if set(values) != set(OUTPUT_ORDER):
        raise FewShotMotionInferenceError("output latent set differs")
    outputs: dict[str, Any] = {}
    vae.to(device)
    for name in OUTPUT_ORDER:
        latent = values[name]
        with __import__("torch").no_grad():
            decoded = _vae_decode(vae, latent)
        expected_decoded = (EXPECTED_FRAMES, *EXPECTED_BUCKET_HW, 3)
        if tuple(int(item) for item in decoded.shape) != expected_decoded:
            raise FewShotMotionInferenceError(f"{name} decoded shape differs")
        path = output_dir / f"{name}.mp4"
        if path.exists() or path.is_symlink():
            raise FewShotMotionInferenceError(f"refusing to overwrite {path}")
        value_audit.save_video_atomically(
            decoded, path, fps=EXPECTED_FPS, save_output_fn=save_output_fn
        )
        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(path)
        legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        if tuple(int(item) for item in encoded_hw) != EXPECTED_BUCKET_HW:
            raise FewShotMotionInferenceError(f"{name} encoded geometry differs")
        outputs[name] = {
            "path": str(path),
            "mp4_sha256": legacy.file_sha256(path),
            "frames": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "bucket_hw": list(EXPECTED_BUCKET_HW),
            "latent": value_audit.tensor_identity(latent, label=f"{name} latent"),
        }
    vae.to("cpu")
    return outputs


def _all_rank_identity(value: Any, *, label: str) -> dict[str, Any]:
    try:
        return v11._all_rank_identities(
            value, label=label, world_size=EXPECTED_ULYSSES_SIZE
        )
    except v11.CPMRFullVideoOracleError as error:
        raise FewShotMotionInferenceError(str(error)) from error


def _build_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    prototype_bundle: PrototypeBundle,
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
    trace_claims = {
        name: runtime_traces[name].get("all_bindings_complete") is True
        for name in PATCHED_ARM_ORDER
    }
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "method": motion_branch.METHOD_NAME,
        "method_revision": args.method_source_revision,
        "method_archive_sha256": args.method_source_archive_sha256,
        "scientific_claim": False,
        "video_quality_claim": False,
        "training_claim": False,
        "source_instruction_only_inference": True,
        "semantic_inputs": list(EXTERNAL_SEMANTIC_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_INFERENCE_ARGUMENTS),
        "heldout_oracle_arm_exists": False,
        "heldout_oracle_used": False,
        "representability_gate": prototype_bundle.representability_gate,
        "diagnostic_only": prototype_bundle.representability_gate == "NO_GO",
        "no_go_diagnostic_override": args.allow_no_go_diagnostic,
        "post_generation_target_scorer_required": True,
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "metadata": dict(source_metadata),
        },
        "instruction": args.instruction,
        "instruction_sha256": hashlib.sha256(
            args.instruction.encode("utf-8")
        ).hexdigest(),
        "prototype": prototype_bundle.audit_receipt(),
        "seeds": {"proposal": PROPOSAL_SEED, "render": RENDER_SEED},
        "schedule": {
            "frames": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "steps": EXPECTED_STEPS,
            "flow_shift": 5.0,
            "proposal_action_noop_same_seed": True,
            "all_five_render_arms_same_seed": True,
        },
        "arms": {
            "order": list(ARM_ORDER),
            "outer_cpmr_gates": ARM_OUTER_GATES,
            "base_prompt": "semantic_noop",
            "codes": {
                name: arm_codes[name].audit_receipt()
                for name in PATCHED_ARM_ORDER
            },
            "reverse_phase_indices": list(epmc.REVERSE_PHASE_INDICES),
            "shuffle_phase_indices": list(epmc.SHUFFLE_PHASE_INDICES),
        },
        "verified_claims": {
            "prototype_state_and_receipt_hash_pinned": True,
            "prototype_heads_byte_exact_tied": True,
            "source_and_instruction_hash_pinned": True,
            "z0_full_latent_byte_exact_b0": arm_comparisons[
                "z0_full_latent_byte_exact_b0"
            ],
            "carrier_phase0_exact_positive_zero": str(
                carrier_receipt["activity_bitset"]
            ).startswith("0"),
            "all_patched_arms_complete_40_step_binding": all(
                trace_claims.values()
            ),
            "every_output_is_81_frames_25fps": all(
                item.get("frames") == EXPECTED_FRAMES
                and item.get("fps") == EXPECTED_FPS
                for item in outputs.values()
            ),
            "heldout_oracle_not_used": True,
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
        raise FewShotMotionInferenceError("one or more inference invariants failed")
    payload["receipt_digest"] = _object_sha256(payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    prototype_bundle = load_prototype_bundle(
        args.prototype_state,
        args.prototype_receipt,
        expected_state_sha256=args.expected_prototype_state_sha256,
        expected_receipt_sha256=args.expected_prototype_receipt_sha256,
        allow_no_go_diagnostic=args.allow_no_go_diagnostic,
    )
    arm_codes_cpu = build_arm_motion_codes(prototype_bundle.motion_code_cpu)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise FewShotMotionInferenceError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % EXPECTED_ULYSSES_SIZE:
        raise FewShotMotionInferenceError(
            "attention heads are not divisible by Ulysses=4"
        )
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
        raise FewShotMotionInferenceError("runtime mv2v prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise FewShotMotionInferenceError("runtime negative prompt differs")
    route_batches.validate_noop_instruction(route_batches.EXACT_NOOP_INSTRUCTION)
    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise FewShotMotionInferenceError("V12 inference requires AUH ROCm GPUs")
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
                    checkpoint,
                    Path(args.checkpoint_content_manifest).expanduser(),
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
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get(
        "ok"
    ) is not True:
        raise FewShotMotionInferenceError(
            f"rank-zero checkpoint validation failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])

    source_path = _plain_absolute_file(args.source_video, label="source video")
    source_tensor, source_metadata, source_sha256 = (
        source_audit.prepare_hashed_source_snapshot(source_path)
    )
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise FewShotMotionInferenceError("held-out source SHA256 differs")
    if tuple(source_metadata["source_derived_bucket_hw"]) != EXPECTED_BUCKET_HW:
        raise FewShotMotionInferenceError("held-out source bucket differs")

    action_prompt = legacy.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = legacy.build_training_prompt(
        route_batches.EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    action_ids, action_mask = legacy._tokenize_training_prompt(
        tokenizer, action_prompt
    )
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
        raise FewShotMotionInferenceError(str(error)) from error
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
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    if tuple(int(item) for item in source_latent.shape) != EXPECTED_LATENT_SHAPE:
        raise FewShotMotionInferenceError("source latent shape differs")
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
            **v11._sample_kwargs(
                input_ids=action_ids,
                attention_mask=action_mask,
                seed=PROPOSAL_SEED,
                **common,
            )
        )
        proposal_noop = model.sample(
            **v11._sample_kwargs(
                input_ids=noop_ids,
                attention_mask=noop_mask,
                seed=PROPOSAL_SEED,
                **common,
            )
        )
    if v11._tensor_bytes_equal(proposal_action, proposal_noop):
        raise FewShotMotionInferenceError("action/no-op proposals are byte-identical")
    diffusion = motion_runtime.resolve_diffusion_core(model)
    transformer = diffusion.transformer
    carrier_result = carrier_core.build_carrier_from_proposal_latents(
        transformer,
        proposal_action,
        proposal_noop,
        expected_patch_grid=EXPECTED_PATCH_GRID,
    )
    carrier = carrier_result.flattened(dtype=torch.bfloat16).to(device=device)
    activity = carrier_result.activity.to(device=device)

    with torch.no_grad():
        b0 = model.sample(
            **v11._sample_kwargs(
                input_ids=noop_ids,
                attention_mask=noop_mask,
                seed=RENDER_SEED,
                **common,
            )
        )

    patched_latents: dict[str, Any] = {}
    runtime_traces: dict[str, dict[str, Any]] = {}
    arm_codes_device = {
        name: _motion_code_to_device(code, device)
        for name, code in arm_codes_cpu.items()
    }
    with motion_branch.install_fewshot_motion_branch(model) as patch_handle:
        for arm in PATCHED_ARM_ORDER:
            with motion_branch.fewshot_motion_code_context(
                patch_handle=patch_handle,
                motion_code=arm_codes_device[arm],
            ):
                with motion_runtime.cpmr_final_render_hook(
                    model,
                    patch_handle=patch_handle,
                    carrier=carrier,
                    activity=activity,
                    gate=OUTER_CPMR_GATE,
                ) as hook:
                    with torch.no_grad():
                        patched_latents[arm] = model.sample(
                            **v11._sample_kwargs(
                                input_ids=noop_ids,
                                attention_mask=noop_mask,
                                seed=RENDER_SEED,
                                **common,
                            )
                        )
            runtime_traces[arm] = _validate_runtime_trace(
                hook.trace.receipt(), arm=arm
            )
        patch_receipt = patch_handle.receipt()
    patch_receipt["restored_after_context"] = patch_handle.restored

    arm_latents = {"B0": b0, **patched_latents}
    arm_comparisons = validate_arm_latents(arm_latents)
    for name, value in {
        "proposal_action": proposal_action,
        "proposal_noop": proposal_noop,
    }.items():
        if tuple(int(item) for item in value.shape) != EXPECTED_LATENT_SHAPE:
            raise FewShotMotionInferenceError(f"{name} latent shape differs")

    proposal_identities = {
        name: _all_rank_identity(value, label=name)
        for name, value in {
            "proposal_action": proposal_action,
            "proposal_noop": proposal_noop,
        }.items()
    }
    arm_identities = {
        name: _all_rank_identity(arm_latents[name], label=name)
        for name in ARM_ORDER
    }
    for name, code in arm_codes_device.items():
        _all_rank_identity(code.phase_gates, label=f"{name} phase gates")
        _all_rank_identity(code.block_head_gates, label=f"{name} block-head gates")
    freeze_after = source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before:
        raise FewShotMotionInferenceError("model freeze certificate changed")

    runtime_versions = {
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    if distributed.rank == 0:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir.exists() or output_dir.is_symlink():
            raise FewShotMotionInferenceError("refusing to reuse output directory")
        output_dir.mkdir(parents=True, exist_ok=False)
        output_latents = {
            "proposal_action": proposal_action,
            "proposal_noop": proposal_noop,
            **arm_latents,
        }
        outputs = _save_outputs(
            output_dir=output_dir,
            values=output_latents,
            vae=vae,
            device=device,
            save_output_fn=save_output,
        )
        receipt = _build_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            prototype_bundle=prototype_bundle,
            checkpoint_identity=checkpoint_identity,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            runtime_versions=runtime_versions,
            freeze_certificate=freeze_after,
            proposal_identities=proposal_identities,
            arm_identities=arm_identities,
            arm_comparisons=arm_comparisons,
            arm_codes=arm_codes_cpu,
            carrier_receipt=carrier_result.audit_receipt(),
            runtime_traces=runtime_traces,
            patch_receipt=patch_receipt,
            outputs=outputs,
        )
        value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(_canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "ARM_OUTER_GATES",
    "EXPECTED_BUCKET_HW",
    "EXPECTED_FRAMES",
    "EXPECTED_FPS",
    "EXPECTED_INSTRUCTION_SHA256",
    "EXPECTED_LATENT_SHAPE",
    "EXPECTED_SOURCE_SHA256",
    "EXTERNAL_SEMANTIC_INPUTS",
    "FORBIDDEN_INFERENCE_ARGUMENTS",
    "FewShotMotionInferenceError",
    "PROTOTYPE_RECEIPT_SCHEMA",
    "PROTOTYPE_STATE_SCHEMA",
    "PROTOTYPE_TENSOR_KEYS",
    "PrototypeBundle",
    "build_arm_motion_codes",
    "build_parser",
    "build_prototype_training_receipt",
    "load_prototype_bundle",
    "main",
    "validate_arm_latents",
    "validate_cli",
    "validate_prototype_training_receipt",
    "validate_tied_prototype_tensors",
]
