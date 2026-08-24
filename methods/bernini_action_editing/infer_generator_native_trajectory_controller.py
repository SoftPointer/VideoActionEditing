#!/usr/bin/env python3
"""Source+instruction-only EGNTC inference on the pinned Bernini-R 1.3B.

The learned artifact is a 36-scalar controller, not an additional visual
condition.  At every one of the official 40 UniPC steps, the audited
tri-branch hook evaluates negative, action, and fixed semantic-noop prompts at
the same current noisy state.  ``EGNTCCallback`` then executes a bounded clean
field before the original scheduler step.  The resulting deployment API is
closed to one exact 81-frame source video and one edit instruction.

Training-only support/target data and inference-time spatial or motion oracles
are absent from both the CLI and callback.  A controller whose training
receipt is not ``GO`` and deployable is rejected unless the explicit
``--allow-diagnostic-no-go`` switch is supplied; such an output is permanently
labelled diagnostic-only in its receipt.
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
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generator_native_trajectory_controller as egntc  # noqa: E402
import infer_lora as base  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_lora as trainer  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402


METHOD_NAME = "episodic-generator-native-trajectory-controller-egntc-v1"
INFERENCE_RECEIPT_SCHEMA = "bernini-egntc-source-instruction-inference-v1"
CONTROLLER_STATE_SCHEMA = "bernini-egntc-controller-state-v1"
CONTROLLER_RECEIPT_SCHEMA = "bernini-egntc-controller-training-artifact-receipt-v1"
CONTROLLER_TENSOR_KEY = "controller_raw_36d"
CONTROLLER_DIMENSION = 36
NUM_INFERENCE_STEPS = 40
MAX_CONTROLLER_BYTES = 1 << 20
MAX_RECEIPT_BYTES = 1 << 20
FORBIDDEN_INFERENCE_CONDITIONS = (
    "target",
    "target_video",
    "paired_target",
    "support",
    "support_video",
    "mask",
    "optical_flow",
    "flow",
    "pose",
    "track",
    "swept_tube",
    "trajectory",
    "reference_image",
    "reference_video",
    "first_frame_anchor",
    "edited_first_frame",
)
REPRESENTABILITY_GATES = (
    "GO",
    "NO_GO",
    "NOT_EVALUATED_ENGINEERING_SMOKE",
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EGNTCInferenceError(RuntimeError):
    """Raised before publication when an EGNTC invariant differs."""


@dataclass(frozen=True)
class ControllerBundle:
    state_path: Path
    receipt_path: Path
    state_file_sha256: str
    receipt_file_sha256: str
    raw_36d_cpu: Any
    parameters_cpu: egntc.EGNTCParameters
    training_receipt: Mapping[str, Any]
    representability_gate: str
    deployable: bool
    diagnostic_override: bool

    def audit_receipt(self) -> dict[str, Any]:
        return {
            "state_path": str(self.state_path),
            "receipt_path": str(self.receipt_path),
            "state_file_sha256": self.state_file_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "state_schema_version": CONTROLLER_STATE_SCHEMA,
            "training_receipt_schema_version": CONTROLLER_RECEIPT_SCHEMA,
            "tensor_key": CONTROLLER_TENSOR_KEY,
            "dimension": CONTROLLER_DIMENSION,
            "raw_36d_sha256": tensor_sha256(self.raw_36d_cpu),
            "training_receipt_digest": self.training_receipt["receipt_digest"],
            "representability_gate": self.representability_gate,
            "deployable": self.deployable,
            "diagnostic_override": self.diagnostic_override,
        }


@dataclass(frozen=True)
class ControllerStepRecord:
    step_index: int
    timestep: float
    sigma: float
    correction_rms: float
    proposal_rms: float
    action_noop_input_byte_exact: bool
    parity_bypass_byte_exact: bool


@dataclass
class ControllerExecutionTrace:
    controller_raw_36d_sha256: str
    records: list[ControllerStepRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "controller_raw_36d_sha256": self.controller_raw_36d_sha256,
            "step_count": len(self.records),
            "steps": [asdict(item) for item in self.records],
        }


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
        raise EGNTCInferenceError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise EGNTCInferenceError("controller digest input must be a non-meta tensor")
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
            }
        )
    )
    digest.update(b"\0")
    # Keep the small controller digest independent of the optional NumPy ABI.
    # ``controller_raw_36d`` is only 144 bytes, so a Python bytes conversion is
    # both deterministic and negligible compared with model inference.
    digest.update(bytes(tensor.view(torch.uint8).tolist()))
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise EGNTCInferenceError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_absolute_file(
    value: str | Path, *, label: str, maximum_bytes: Optional[int] = None
) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise EGNTCInferenceError(f"{label} must be an absolute path")
    try:
        info = requested.lstat()
    except OSError as error:
        raise EGNTCInferenceError(f"cannot stat {label}: {requested}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EGNTCInferenceError(f"{label} must be a plain regular file")
    if maximum_bytes is not None and not 0 < info.st_size <= maximum_bytes:
        raise EGNTCInferenceError(
            f"{label} size must lie in [1,{maximum_bytes}] bytes"
        )
    return requested.resolve(strict=True)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EGNTCInferenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise EGNTCInferenceError(f"non-finite JSON number: {value}")


def _read_strict_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except EGNTCInferenceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EGNTCInferenceError(f"cannot decode {label} as strict JSON") from error
    if not isinstance(value, Mapping):
        raise EGNTCInferenceError(f"{label} must contain one JSON object")
    return dict(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--controller-state", required=True)
    parser.add_argument("--controller-receipt", required=True)
    parser.add_argument("--expected-controller-state-sha256", required=True)
    parser.add_argument("--expected-controller-receipt-sha256", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-diagnostic-no-go",
        action="store_true",
        help="permit a non-deployable/NO_GO controller for labelled diagnostics only",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        choices=(NUM_INFERENCE_STEPS,),
        default=NUM_INFERENCE_STEPS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-instruction-sha256")
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
    if (
        type(args.instruction) is not str
        or not args.instruction.strip()
        or "\x00" in args.instruction
    ):
        raise EGNTCInferenceError("instruction must be non-empty text without NUL")
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise EGNTCInferenceError("EGNTC requires exactly 40 official UniPC steps")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise EGNTCInferenceError("seed must be an integer in [0,2^63)")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if type(value) is not str or _SHA1_RE.fullmatch(value.lower()) is None:
            raise EGNTCInferenceError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "expected_controller_state_sha256",
        "expected_controller_receipt_sha256",
        "method_source_archive_sha256",
    ):
        _required_sha256(getattr(args, name), label=name)
    for name in ("expected_source_sha256", "expected_instruction_sha256"):
        value = getattr(args, name)
        if value is not None:
            _required_sha256(value, label=name)
    if args.expected_bernini_commit.lower() != trainer.BERNINI_OFFICIAL_COMMIT:
        raise EGNTCInferenceError("only the audited Bernini commit is supported")
    if args.expected_veomni_commit.lower() != trainer.VEOMNI_TESTED_COMMIT:
        raise EGNTCInferenceError("only the tested VeOmni commit is supported")
    if args.expected_checkpoint_tree_sha256 != trainer.CHECKPOINT_TREE_SHA256:
        raise EGNTCInferenceError("only the audited Bernini-R 1.3B checkpoint is supported")
    for name in ("source_video", "controller_state", "controller_receipt", "output"):
        if not Path(getattr(args, name)).expanduser().is_absolute():
            raise EGNTCInferenceError(f"{name} must be an absolute path")


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
        raise EGNTCInferenceError("pinned Bernini sampler contract changed")
    return contract


def configure_rank_local_caches(
    environment: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    source = os.environ if environment is None else environment
    root_value = source.get("BERNINI_EGNTC_RANK_CACHE_ROOT")
    if not root_value:
        return {}
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise EGNTCInferenceError("BERNINI_EGNTC_RANK_CACHE_ROOT must be absolute")
    try:
        rank = int(source.get("LOCAL_RANK", ""))
    except ValueError as error:
        raise EGNTCInferenceError("LOCAL_RANK is invalid for rank-local caches") from error
    if not 0 <= rank < base.ULYSSES_SIZE:
        raise EGNTCInferenceError("LOCAL_RANK is outside the four-rank EGNTC world")
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
    """Use the same frozen Bernini T5 path as action/negative conditioning."""

    import torch

    if tuple(input_ids.shape) != (1, 512) or tuple(attention_mask.shape) != (1, 512):
        raise EGNTCInferenceError("semantic no-op token tensors must be [1,512]")
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
        or not bool(torch.isfinite(embeddings).all().item())
    ):
        raise EGNTCInferenceError("Bernini returned invalid semantic no-op embeddings")
    frozen = all(
        not parameter.requires_grad
        for parameter in renderer.t5_text_encoder.parameters()
    )
    if not frozen:
        raise EGNTCInferenceError("semantic no-op T5 encoder is unexpectedly trainable")
    return embeddings, {
        "token_shape": [1, 512],
        "nonpadding_token_count": int(attention_mask.sum().item()),
        "embedding_shape": [int(value) for value in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
        "encoder": "BerniniRendererModel.encode_prompt",
        "frozen_t5": True,
    }


def validate_controller_tensor(value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise EGNTCInferenceError("controller state must be a tensor")
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or tuple(value.shape) != (CONTROLLER_DIMENSION,)
        or not value.is_contiguous()
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise EGNTCInferenceError(
            "controller state must be detached contiguous finite CPU float32 [36]"
        )
    return value


def parameters_from_controller_tensor(value: Any) -> egntc.EGNTCParameters:
    tensor = validate_controller_tensor(value)
    try:
        parameters = egntc.EGNTCParameters.from_flat_tensor(tensor.clone())
        roundtrip = parameters.flat_tensor().detach().contiguous().cpu().float()
    except Exception as error:
        raise EGNTCInferenceError("cannot construct EGNTCParameters from [36]") from error
    if tuple(roundtrip.shape) != (CONTROLLER_DIMENSION,) or not __import__(
        "torch"
    ).equal(roundtrip, tensor):
        raise EGNTCInferenceError("EGNTCParameters [36] round-trip differs")
    return parameters


def build_controller_training_receipt(
    *,
    state_filename: str,
    state_file_sha256: str,
    raw_36d: Any,
    representability_gate: str,
    deployable: bool,
    training_run_receipt_sha256: str,
    support_iids: Sequence[str],
) -> dict[str, Any]:
    """Build the sole checkpoint envelope accepted by source-only inference."""

    raw = validate_controller_tensor(raw_36d)
    parameters_from_controller_tensor(raw)
    if Path(state_filename).name != state_filename or not state_filename.endswith(
        ".safetensors"
    ):
        raise EGNTCInferenceError("controller filename must be a plain .safetensors basename")
    _required_sha256(state_file_sha256, label="controller state file SHA256")
    _required_sha256(training_run_receipt_sha256, label="training run receipt SHA256")
    if representability_gate not in REPRESENTABILITY_GATES:
        raise EGNTCInferenceError("representability_gate has an unsupported value")
    if type(deployable) is not bool:
        raise EGNTCInferenceError("deployable must be boolean")
    if representability_gate != "GO" and deployable:
        raise EGNTCInferenceError("a non-GO controller cannot be deployable")
    if isinstance(support_iids, (str, bytes)) or len(tuple(support_iids)) != 2:
        raise EGNTCInferenceError("EGNTC requires exactly two training support IIDs")
    iid_values = tuple(support_iids)
    if (
        any(type(value) is not str or not value for value in iid_values)
        or len(set(iid_values)) != 2
    ):
        raise EGNTCInferenceError("support IIDs must be two distinct non-empty strings")
    payload = {
        "schema_version": CONTROLLER_RECEIPT_SCHEMA,
        "controller_state": {
            "schema_version": CONTROLLER_STATE_SCHEMA,
            "format": "safetensors",
            "filename": state_filename,
            "file_sha256": state_file_sha256,
            "tensor_key": CONTROLLER_TENSOR_KEY,
            "shape": [CONTROLLER_DIMENSION],
            "dtype": "torch.float32",
            "tensor_sha256": tensor_sha256(raw),
        },
        "training_provenance": {
            "method": METHOD_NAME,
            "support_count": 2,
            "support_iids": list(iid_values),
            "support_targets_training_only": True,
            "training_run_receipt_sha256": training_run_receipt_sha256,
            "representability_gate": representability_gate,
            "deployable": deployable,
        },
        "deployment_contract": {
            "semantic_inputs": ["source_video", "edit_instruction"],
            "source_instruction_only": True,
            "support_available_at_inference": False,
            "target_available_at_inference": False,
            "external_oracle_available_at_inference": False,
            "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
        },
    }
    return {**payload, "receipt_digest": object_sha256(payload)}


def _load_safetensors_cpu(path: Path) -> Mapping[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise EGNTCInferenceError("safetensors is required for EGNTC inference") from error
    try:
        return load_file(str(path), device="cpu")
    except Exception as error:
        raise EGNTCInferenceError("could not load controller safetensors") from error


def load_controller_bundle(
    state_value: str | Path,
    receipt_value: str | Path,
    *,
    expected_state_sha256: str,
    expected_receipt_sha256: str,
    allow_diagnostic_no_go: bool = False,
) -> ControllerBundle:
    """Authenticate the receipt before deserializing the bounded tensor file."""

    _required_sha256(expected_state_sha256, label="expected controller state SHA256")
    _required_sha256(expected_receipt_sha256, label="expected controller receipt SHA256")
    state_path = _plain_absolute_file(
        state_value, label="controller state", maximum_bytes=MAX_CONTROLLER_BYTES
    )
    receipt_path = _plain_absolute_file(
        receipt_value, label="controller receipt", maximum_bytes=MAX_RECEIPT_BYTES
    )
    if state_path == receipt_path or state_path.suffix != ".safetensors":
        raise EGNTCInferenceError("controller state/receipt paths are invalid")
    state_sha = file_sha256(state_path)
    receipt_sha = file_sha256(receipt_path)
    if state_sha != expected_state_sha256:
        raise EGNTCInferenceError("controller state file SHA256 differs")
    if receipt_sha != expected_receipt_sha256:
        raise EGNTCInferenceError("controller receipt file SHA256 differs")
    receipt = _read_strict_json(receipt_path, label="controller receipt")
    expected_top = {
        "schema_version",
        "controller_state",
        "training_provenance",
        "deployment_contract",
        "receipt_digest",
    }
    if set(receipt) != expected_top or receipt.get("schema_version") != CONTROLLER_RECEIPT_SCHEMA:
        raise EGNTCInferenceError("controller receipt envelope/schema differs")
    declared_digest = _required_sha256(
        receipt.get("receipt_digest"), label="controller receipt digest"
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared_digest:
        raise EGNTCInferenceError("controller receipt digest differs")
    state_contract = receipt.get("controller_state")
    training = receipt.get("training_provenance")
    deployment = receipt.get("deployment_contract")
    if not all(isinstance(value, Mapping) for value in (state_contract, training, deployment)):
        raise EGNTCInferenceError("controller receipt subcontracts must be objects")
    if set(state_contract) != {
        "schema_version",
        "format",
        "filename",
        "file_sha256",
        "tensor_key",
        "shape",
        "dtype",
        "tensor_sha256",
    }:
        raise EGNTCInferenceError("controller state contract fields differ")
    if set(training) != {
        "method",
        "support_count",
        "support_iids",
        "support_targets_training_only",
        "training_run_receipt_sha256",
        "representability_gate",
        "deployable",
    }:
        raise EGNTCInferenceError("controller training provenance fields differ")
    if set(deployment) != {
        "semantic_inputs",
        "source_instruction_only",
        "support_available_at_inference",
        "target_available_at_inference",
        "external_oracle_available_at_inference",
        "forbidden_inference_conditions",
    }:
        raise EGNTCInferenceError("controller deployment contract fields differ")
    if (
        state_contract.get("schema_version") != CONTROLLER_STATE_SCHEMA
        or state_contract.get("format") != "safetensors"
        or state_contract.get("filename") != state_path.name
        or state_contract.get("file_sha256") != state_sha
        or state_contract.get("tensor_key") != CONTROLLER_TENSOR_KEY
        or state_contract.get("shape") != [CONTROLLER_DIMENSION]
        or state_contract.get("dtype") != "torch.float32"
        or _SHA256_RE.fullmatch(state_contract.get("tensor_sha256") or "") is None
    ):
        raise EGNTCInferenceError("controller state contract differs")
    if (
        training.get("method") != METHOD_NAME
        or training.get("support_count") != 2
        or not isinstance(training.get("support_iids"), list)
        or len(training["support_iids"]) != 2
        or training.get("support_targets_training_only") is not True
        or _SHA256_RE.fullmatch(training.get("training_run_receipt_sha256") or "") is None
        or training.get("representability_gate") not in REPRESENTABILITY_GATES
        or type(training.get("deployable")) is not bool
    ):
        raise EGNTCInferenceError("controller training provenance differs")
    if (
        any(type(value) is not str or not value for value in training["support_iids"])
        or len(set(training["support_iids"])) != 2
    ):
        raise EGNTCInferenceError("controller support IIDs differ")
    if training["representability_gate"] != "GO" and training["deployable"]:
        raise EGNTCInferenceError("a non-GO controller cannot be deployable")
    if (
        deployment.get("semantic_inputs") != ["source_video", "edit_instruction"]
        or deployment.get("source_instruction_only") is not True
        or deployment.get("support_available_at_inference") is not False
        or deployment.get("target_available_at_inference") is not False
        or deployment.get("external_oracle_available_at_inference") is not False
        or deployment.get("forbidden_inference_conditions")
        != list(FORBIDDEN_INFERENCE_CONDITIONS)
    ):
        raise EGNTCInferenceError("controller deployment contract differs")
    ready = training["representability_gate"] == "GO" and training["deployable"] is True
    if not ready and allow_diagnostic_no_go is not True:
        raise EGNTCInferenceError(
            "controller is not GO/deployable; use --allow-diagnostic-no-go only for diagnostics"
        )
    tensors = _load_safetensors_cpu(state_path)
    if set(tensors) != {CONTROLLER_TENSOR_KEY}:
        raise EGNTCInferenceError("controller safetensors keys differ")
    raw = validate_controller_tensor(tensors[CONTROLLER_TENSOR_KEY])
    if tensor_sha256(raw) != state_contract["tensor_sha256"]:
        raise EGNTCInferenceError("controller tensor SHA256 differs")
    parameters = parameters_from_controller_tensor(raw)
    return ControllerBundle(
        state_path=state_path,
        receipt_path=receipt_path,
        state_file_sha256=state_sha,
        receipt_file_sha256=receipt_sha,
        raw_36d_cpu=raw,
        parameters_cpu=parameters,
        training_receipt=receipt,
        representability_gate=training["representability_gate"],
        deployable=training["deployable"],
        diagnostic_override=bool(allow_diagnostic_no_go),
    )


def _tensor_scalar(value: Any, *, label: str) -> float:
    try:
        result = float(value.detach().float().cpu().item())
    except Exception as error:
        raise EGNTCInferenceError(f"cannot serialize {label}") from error
    if not math.isfinite(result) or result < 0.0:
        raise EGNTCInferenceError(f"{label} must be finite and non-negative")
    return result


class TracedEGNTCCallback:
    """Thin receipt layer around the source-conditioned 36D core callback."""

    def __init__(
        self,
        *,
        source_clean: Any,
        parameters: egntc.EGNTCParameters,
        raw_36d_sha256: str,
    ) -> None:
        _required_sha256(raw_36d_sha256, label="controller raw 36D SHA256")
        self.inner = egntc.EGNTCCallback(
            source_clean=source_clean,
            parameters=parameters,
        )
        self.trace = ControllerExecutionTrace(raw_36d_sha256)

    def __call__(self, fields: tri.CleanFieldStep) -> Any:
        import torch

        executed = self.inner(fields)
        if (
            not isinstance(executed, torch.Tensor)
            or tuple(executed.shape) != tuple(fields.noop_guided_clean.shape)
            or executed.device != fields.noop_guided_clean.device
            or not executed.is_floating_point()
            or not bool(torch.isfinite(executed).all().item())
        ):
            raise EGNTCInferenceError("EGNTC callback returned an invalid clean field")
        input_exact = bool(
            torch.equal(fields.action_guided_clean, fields.noop_guided_clean)
        )
        # tri_branch_unipc only reuses Bernini's official model_output when
        # the callback returns the exact action object.  Numeric equality is
        # insufficient because it would still take a clean->velocity BF16
        # round trip at the scheduler boundary.
        bypass_exact = (not input_exact) or (
            executed is fields.action_guided_clean
        )
        if not bypass_exact:
            raise EGNTCInferenceError("EGNTC violated exact action/noop parity bypass")
        correction = executed.float() - fields.noop_guided_clean.float()
        proposal = fields.action_delta_clean.float()
        self.trace.records.append(
            ControllerStepRecord(
                step_index=int(fields.step_index),
                timestep=float(fields.timestep),
                sigma=float(fields.sigma),
                correction_rms=_tensor_scalar(
                    correction.square().mean().sqrt(), label="controller correction RMS"
                ),
                proposal_rms=_tensor_scalar(
                    proposal.square().mean().sqrt(), label="action/noop proposal RMS"
                ),
                action_noop_input_byte_exact=input_exact,
                parity_bypass_byte_exact=bypass_exact,
            )
        )
        return executed

    def finalized_core_receipt(self) -> dict[str, Any]:
        """Return the core's complete, self-authenticating trust-region trace."""

        try:
            receipt = self.inner.receipt()
            egntc.validate_controller_receipt(receipt, require_complete=True)
        except egntc.EGNTCContractError as error:
            raise EGNTCInferenceError("EGNTC core rollout receipt is incomplete") from error
        return receipt


def validate_execution_trace(
    tri_trace: tri.TriBranchTrace,
    controller_trace: ControllerExecutionTrace,
    core_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Certify the exact 40-step/120-forward official sampler execution."""

    if not isinstance(tri_trace, tri.TriBranchTrace):
        raise EGNTCInferenceError("tri_trace must be a TriBranchTrace")
    if not isinstance(controller_trace, ControllerExecutionTrace):
        raise EGNTCInferenceError("controller_trace has an invalid type")
    _required_sha256(
        controller_trace.controller_raw_36d_sha256,
        label="controller trace raw 36D SHA256",
    )
    try:
        egntc.validate_controller_receipt(core_receipt, require_complete=True)
    except egntc.EGNTCContractError as error:
        raise EGNTCInferenceError("EGNTC core rollout receipt is invalid") from error
    if tri_trace.sample_calls != 1:
        raise EGNTCInferenceError("tri-branch hook must observe exactly one sample call")
    branches = list(tri_trace.records)
    controls = list(controller_trace.records)
    core_steps = core_receipt.get("steps")
    if not isinstance(core_steps, list) or len(core_steps) != NUM_INFERENCE_STEPS:
        raise EGNTCInferenceError("EGNTC core trace must contain 40 steps")
    if len(branches) != NUM_INFERENCE_STEPS or len(controls) != NUM_INFERENCE_STEPS:
        raise EGNTCInferenceError("EGNTC must execute all 40 official UniPC steps")
    sigmas: list[float] = []
    for expected_index, (branch, control, core_step) in enumerate(
        zip(branches, controls, core_steps)
    ):
        if branch.step_index != expected_index or control.step_index != expected_index:
            raise EGNTCInferenceError("EGNTC step indices are incomplete or reordered")
        if branch.model_id != "transformer_1":
            raise EGNTCInferenceError("Bernini-R 1.3B must remain single-expert")
        if (
            branch.transformer_forwards != 3
            or branch.shared_negative_forwards != 1
            or branch.action_forwards != 1
            or branch.noop_forwards != 1
            or branch.original_scheduler_calls != 1
        ):
            raise EGNTCInferenceError(
                "each step must use three transformer forwards and one original UniPC call"
            )
        if (
            branch.official_action_exact_parity is not True
            or branch.official_action_parity_rms_error != 0.0
            or branch.official_action_parity_max_abs_error != 0.0
        ):
            raise EGNTCInferenceError("official action APG exact parity failed")
        if branch.effective_guidance_scale != base.OMEGA_TEXT:
            raise EGNTCInferenceError("action/no-op APG guidance scale differs")
        if not math.isclose(branch.sigma, control.sigma, rel_tol=0.0, abs_tol=1e-8):
            raise EGNTCInferenceError("tri-branch/controller sigma traces differ")
        if not math.isclose(branch.timestep, control.timestep, rel_tol=0.0, abs_tol=1e-6):
            raise EGNTCInferenceError("tri-branch/controller timestep traces differ")
        if (
            core_step.get("step_index") != expected_index
            or not math.isclose(
                float(core_step.get("sigma", math.nan)),
                control.sigma,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or float(core_step.get("timestep", math.nan)) != control.timestep
            or core_step.get("trust_region_satisfied") is not True
            or bool(core_step.get("action_noop_exact_parity"))
            != control.action_noop_input_byte_exact
        ):
            raise EGNTCInferenceError("EGNTC core and outer controller traces differ")
        if branch.sigma <= 0.0 or not math.isfinite(branch.sigma):
            raise EGNTCInferenceError("official UniPC sigma trace is invalid")
        if (
            not math.isfinite(control.correction_rms)
            or control.correction_rms < 0.0
            or not math.isfinite(control.proposal_rms)
            or control.proposal_rms < 0.0
            or control.parity_bypass_byte_exact is not True
        ):
            raise EGNTCInferenceError("controller step diagnostic is invalid")
        sigmas.append(float(branch.sigma))
    if any(following >= current for current, following in zip(sigmas, sigmas[1:])):
        raise EGNTCInferenceError("official UniPC sigma trace must be strictly descending")
    payload = {
        "tri_branch": tri_trace.as_dict(),
        "controller": controller_trace.as_dict(),
        "controller_core": dict(core_receipt),
        "certificate": {
            "step_count": NUM_INFERENCE_STEPS,
            "official_action_apg_exact_steps": NUM_INFERENCE_STEPS,
            "original_unipc_calls": NUM_INFERENCE_STEPS,
            "transformer_forwards": 3 * NUM_INFERENCE_STEPS,
            "controller_callback_calls": NUM_INFERENCE_STEPS,
            "custom_integrator": False,
        },
    }
    payload["trace_digest"] = object_sha256(payload)
    return payload


def _method_hashes() -> dict[str, str]:
    paths = {
        "infer_generator_native_trajectory_controller.py": Path(__file__),
        "generator_native_trajectory_controller.py": METHOD_ROOT
        / "generator_native_trajectory_controller.py",
        "tri_branch_unipc.py": METHOD_ROOT / "tri_branch_unipc.py",
        "infer_lora.py": METHOD_ROOT / "infer_lora.py",
        "motion_residual.py": METHOD_ROOT / "motion_residual.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def build_inference_receipt(
    *,
    args: argparse.Namespace,
    bundle: ControllerBundle,
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
    diagnostic_only = bool(bundle.diagnostic_override)
    bundle_audit = bundle.audit_receipt()
    traced_controller_hash = (
        execution_trace.get("controller", {}).get("controller_raw_36d_sha256")
        if isinstance(execution_trace.get("controller"), Mapping)
        else None
    )
    if traced_controller_hash != bundle_audit.get("raw_36d_sha256"):
        raise EGNTCInferenceError(
            "execution trace is not bound to the loaded controller checkpoint"
        )
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
        "base_checkpoint": {
            "path": str(Path(args.checkpoint).expanduser()),
            "tree_sha256": args.expected_checkpoint_tree_sha256,
            "bernini_r_1p3b": True,
            "frozen": True,
        },
        "controller_checkpoint": bundle_audit,
        "input": {
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "instruction_utf8_sha256": hashlib.sha256(instruction_bytes).hexdigest(),
            "instruction_utf8_bytes": len(instruction_bytes),
            "accepted_external_conditions": ["source_video", "edit_instruction"],
            "semantic_noop_is_internal_fixed_control": True,
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "support_accessed_by_inference": False,
            "external_mask_flow_pose_track_trajectory": False,
            "first_frame_anchor": False,
        },
        "preprocessing": dict(source_metadata),
        "sampling": {
            **exact_sampler_contract(seed=args.seed),
            "single_expert": "transformer_1",
            "ulysses_size": base.ULYSSES_SIZE,
            "rank0_decode_and_save_only": True,
            "tri_branch_contract": tri.sampler_contract(),
            "controller_contract": egntc.controller_contract(),
            "projection_boundary": "after_exact_action_apg_before_original_unipc_step",
            "custom_integrator": False,
        },
        "prompt_contract": {
            "task": "mv2v",
            "action_system_prompt_sha256": hashlib.sha256(
                base.MV2V_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "semantic_noop_instruction_sha256": hashlib.sha256(
                motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(
                base.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
            "action_noop_negative_use_frozen_t5": True,
            "semantic_noop": dict(noop_identity),
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
        "diagnostic_only": diagnostic_only,
        "deployable_output": bool(bundle.deployable and not diagnostic_only),
        "scientific_claim_authorized": False,
        "production_claim_forbidden": True,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    configure_rank_local_caches()
    source_requested = Path(args.source_video).expanduser()
    try:
        source_path = base._plain_file(
            source_requested.resolve(strict=True), label="source video"
        )
        output_path, receipt_path = base._resolve_output(args.output)
        bundle = load_controller_bundle(
            args.controller_state,
            args.controller_receipt,
            expected_state_sha256=args.expected_controller_state_sha256,
            expected_receipt_sha256=args.expected_controller_receipt_sha256,
            allow_diagnostic_no_go=args.allow_diagnostic_no_go,
        )
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
        raise EGNTCInferenceError(str(error)) from error
    source_sha256 = file_sha256(source_path)
    instruction_sha256 = hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()
    if args.expected_source_sha256 is not None and source_sha256 != args.expected_source_sha256:
        raise EGNTCInferenceError("source video SHA256 differs")
    if (
        args.expected_instruction_sha256 is not None
        and instruction_sha256 != args.expected_instruction_sha256
    ):
        raise EGNTCInferenceError("instruction SHA256 differs")
    if transformer_config["num_attention_heads"] % base.ULYSSES_SIZE:
        raise EGNTCInferenceError("Bernini-R 1.3B heads are not divisible by Ulysses=4")
    wan_diffusion_path = (
        bernini_root / "bernini/models/wan_diffusion.py"
    ).resolve(strict=True)
    try:
        wan_diffusion_sha256 = tri.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
        )
    except tri.TriBranchHookError as error:
        raise EGNTCInferenceError(str(error)) from error
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
        raise EGNTCInferenceError("runtime Bernini mv2v system prompt differs")
    if DEFAULT_NEG_PROMPT != base.DEFAULT_NEGATIVE_PROMPT:
        raise EGNTCInferenceError("runtime Bernini negative prompt differs")
    try:
        distributed = base.inference_distributed_contract()
    except base.InferenceContractError as error:
        raise EGNTCInferenceError(str(error)) from error
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise EGNTCInferenceError("EGNTC requires four AUH ROCm-visible GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
        device_id=torch.device("cuda", distributed.local_rank),
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    try:
        source_tensor, source_metadata = base.prepare_exact_source(source_path)
    except base.InferenceContractError as error:
        raise EGNTCInferenceError(str(error)) from error
    action_prompt = base.build_training_prompt(args.instruction, prompt_cleaner=prompt_clean)
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
        raise EGNTCInferenceError(str(error)) from error
    if float(config.shift) != base.FLOW_SHIFT or config.use_unipc is not True:
        raise EGNTCInferenceError("renderer must use official UniPC with flow shift 5")
    model = BerniniRendererModel(config)
    if any("lora_" in name.lower() for name, _ in model.named_modules()):
        raise EGNTCInferenceError("frozen base unexpectedly contains LoRA modules")
    model.requires_grad_(False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **base.tokenizer_load_kwargs()
    )
    if tokenizer.padding_side != "right" or tokenizer.init_kwargs.get(
        "fix_mistral_regex"
    ) is not True:
        raise EGNTCInferenceError("tokenizer lost fix_mistral_regex/right-padding")
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
        raise EGNTCInferenceError("source VAE latent differs from exact 81f geometry")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    noop_embeddings, noop_identity = encode_semantic_noop_prompt(
        model, noop_ids, noop_mask, device=device
    )
    parameters = bundle.parameters_cpu.to(device=device, dtype=torch.float32)
    parameters.requires_grad_(False)
    parameters.eval()
    callback = TracedEGNTCCallback(
        source_clean=source_latent,
        parameters=parameters,
        raw_36d_sha256=tensor_sha256(bundle.raw_36d_cpu),
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
    except (tri.TriBranchHookError, egntc.EGNTCContractError) as error:
        raise EGNTCInferenceError(str(error)) from error
    execution_trace = validate_execution_trace(
        tri_trace, callback.trace, callback.finalized_core_receipt()
    )
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise EGNTCInferenceError("generated latent differs from exact 81f geometry")
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
            raise EGNTCInferenceError("decoded output differs from exact 81f geometry")
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise EGNTCInferenceError(f"stale temporary output exists: {temporary_output}")
        save_output(output, str(temporary_output), fps=int(base.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(output_path)
        try:
            base.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        except base.InferenceContractError as error:
            raise EGNTCInferenceError(str(error)) from error
        if tuple(encoded_hw) != tuple(bucket):
            raise EGNTCInferenceError("encoded output geometry differs from source bucket")
        receipt = build_inference_receipt(
            args=args,
            bundle=bundle,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            output_path=output_path,
            output_sha256=file_sha256(output_path),
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
        print(canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier(device_ids=[distributed.local_rank])
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
