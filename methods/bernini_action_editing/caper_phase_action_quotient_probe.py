"""CAPER Phase Action Quotient (PAQ) frozen observation/admission probe.

This module is deliberately *not* a representation-learning or training
module.  It asks whether a frozen Bernini hidden-state path contains a
cross-identity phase/action candidate and, separately, whether a preregistered
causal intervention makes that candidate admissible.  Linear decodability is
neither computed nor accepted as evidence of admission.

Each cohort contains ``target``, ``noop``, ``reverse``, ``incomplete`` and
``wrong_action`` observations from the exact same noisy state.  Hidden tensors
must already be reduced to the target video tail with shape ``[T, D]`` or
``[T, P, D]``.  The latter is spatially pooled by this module.  Temporal
features are always anchored to the *start phase*::

    R_t = H_t - mean(H[start_begin:start_end])

Temporal-mean subtraction is intentionally absent: it would remove the
terminal/hold plateau that the probe is meant to preserve.

Realistic read-only Bernini hook locations (without modifying vendor code)
are returned by :func:`bernini_hook_plan`:

* ``attn2.to_out.0`` pre-hook: cross-attention aggregate before output linear;
* ``attn2.to_out.0`` post-hook: text-conditioned output before block residual;
* ``attn1.to_out.0`` pre-hook: joint visual self-attention aggregate before its
  output linear.

Capture all counterfactual branches at the same scheduler state, sigma,
Gaussian/noisy latent and hook call.  A runtime materializer must restore the
global target-tail token order before creating the tensor accepted here.

The final decision is fail closed.  Without held-out causal receipts proving
that target improves the requested action, reverse reverses temporal order,
noop is inert, and preservation is not worse, no code is admitted and zero
training updates are authorized or executed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping, Optional, Sequence

import torch


SCHEMA_VERSION = "caper-phase-action-quotient-observations-v1"
DECISION_SCHEMA_VERSION = "caper-phase-action-quotient-decision-v1"
CAUSAL_RECEIPT_SCHEMA_VERSION = "caper-paq-decoded-causal-receipt-v1"
PHASE_ORDER = ("start", "transition", "terminal", "hold")
REQUIRED_VARIANTS = (
    "target",
    "noop",
    "reverse",
    "incomplete",
    "wrong_action",
)
REQUIRED_SPLITS = ("discovery", "admission")
ANCHOR_FORMULA = "hidden_t-minus-start_phase_mean_v1"
NO_ADMISSION_STATUS = "NO_ADMITTED_CODE_ZERO_TRAINING"
ADMISSION_STATUS = "CAUSALLY_ADMITTED_READ_ONLY_PROBE_ZERO_UPDATES_EXECUTED"
CAUSAL_MEDIA_ROLES = ("baseline", "target", "reverse", "noop")
PRESERVATION_AXES = (
    "identity",
    "object",
    "background",
    "camera",
    "non_target",
    "quality",
)
DECODED_MEDIA_CONTRACT = "pyav-rgb24-exact81-fps25-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HOOK_SITE = re.compile(
    r"^diff_dec\.transformer\.blocks\.(\d+)\.(attn[12])\.to_out\.0:(pre|post)$"
)
_EPS = 1.0e-12


class PAQProbeError(RuntimeError):
    """Raised when a closed PAQ observation/admission contract is violated."""


@dataclass(frozen=True)
class BerniniHookPoint:
    module_path: str
    hook_kind: str
    semantic_role: str

    @property
    def site_id(self) -> str:
        return f"{self.module_path}:{self.hook_kind}"


def bernini_hook_plan(block_index: int) -> tuple[BerniniHookPoint, ...]:
    """Return the three read-only hook sites used by a PAQ scan.

    ``pre`` means ``register_forward_pre_hook`` on ``to_out[0]`` and ``post``
    means ``register_forward_hook``.  Hooks must detach observations and must
    never return a replacement input/output.
    """

    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise PAQProbeError("block_index must be an integer")
    if not 0 <= block_index < 30:
        raise PAQProbeError("Bernini-R 1.3B block_index must be in [0, 30)")
    prefix = f"diff_dec.transformer.blocks.{block_index}"
    return (
        BerniniHookPoint(
            f"{prefix}.attn2.to_out.0",
            "pre",
            "cross_attention_aggregate_before_output_projection",
        ),
        BerniniHookPoint(
            f"{prefix}.attn2.to_out.0",
            "post",
            "text_conditioned_cross_attention_output_before_block_residual",
        ),
        BerniniHookPoint(
            f"{prefix}.attn1.to_out.0",
            "pre",
            "joint_visual_self_attention_aggregate_before_output_projection",
        ),
    )


@dataclass(frozen=True)
class PhaseRanges:
    """Four ordered, non-overlapping half-open temporal intervals."""

    start: tuple[int, int]
    transition: tuple[int, int]
    terminal: tuple[int, int]
    hold: tuple[int, int]

    def as_dict(self) -> dict[str, list[int]]:
        return {name: list(getattr(self, name)) for name in PHASE_ORDER}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhaseRanges":
        if set(value) != set(PHASE_ORDER):
            raise PAQProbeError(f"phase_ranges must contain exactly {PHASE_ORDER}")
        parsed: dict[str, tuple[int, int]] = {}
        previous_stop = 0
        for index, name in enumerate(PHASE_ORDER):
            raw = value[name]
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in raw)
            ):
                raise PAQProbeError(f"phase range {name} must be two integer bounds")
            begin, stop = int(raw[0]), int(raw[1])
            if begin < 0 or stop <= begin:
                raise PAQProbeError(f"phase range {name} must be non-empty and nonnegative")
            if index and begin != previous_stop:
                raise PAQProbeError(
                    "phase ranges must be a contiguous start->transition->terminal->hold partition"
                )
            parsed[name] = (begin, stop)
            previous_stop = stop
        return cls(**parsed)

    def validate_frame_count(self, frame_count: int) -> None:
        if self.start[0] != 0:
            raise PAQProbeError("start phase must begin at temporal index zero")
        if self.hold[1] != frame_count:
            raise PAQProbeError("hold phase must terminate at the final hidden frame")


@dataclass(frozen=True)
class PAQThresholds:
    minimum_identities_per_split: int = 2
    minimum_scenes_per_split: int = 2
    minimum_seeds_per_split: int = 2
    minimum_target_contrast_norm: float = 0.25
    minimum_reverse_opposition_cosine: float = 0.70
    minimum_incomplete_transition_cosine: float = 0.70
    maximum_incomplete_hold_ratio: float = 0.70
    maximum_wrong_action_cosine: float = 0.50
    minimum_group_stability_cosine: float = 0.75
    minimum_cross_split_cosine: float = 0.75
    minimum_transition_progress_fraction: float = 0.05
    minimum_terminal_progress_margin: float = 0.05
    maximum_hold_terminal_progress_delta: float = 0.20
    minimum_hold_terminal_cosine: float = 0.90
    minimum_hold_terminal_norm_ratio: float = 0.75
    maximum_hold_terminal_norm_ratio: float = 1.25

    def __post_init__(self) -> None:
        integer_names = (
            "minimum_identities_per_split",
            "minimum_scenes_per_split",
            "minimum_seeds_per_split",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise PAQProbeError(f"{name} must be an integer >= 2")
        for name, value in asdict(self).items():
            if name in integer_names:
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise PAQProbeError(f"{name} must be finite")
        unit_interval = (
            "minimum_reverse_opposition_cosine",
            "minimum_incomplete_transition_cosine",
            "maximum_incomplete_hold_ratio",
            "maximum_wrong_action_cosine",
            "minimum_group_stability_cosine",
            "minimum_cross_split_cosine",
            "minimum_transition_progress_fraction",
            "maximum_hold_terminal_progress_delta",
            "minimum_hold_terminal_cosine",
            "minimum_hold_terminal_norm_ratio",
        )
        for name in unit_interval:
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise PAQProbeError(f"{name} must be in [0, 1]")
        if self.minimum_target_contrast_norm <= 0.0:
            raise PAQProbeError("minimum_target_contrast_norm must be positive")
        if self.minimum_terminal_progress_margin < 0.0:
            raise PAQProbeError("minimum_terminal_progress_margin must be nonnegative")
        if self.maximum_hold_terminal_norm_ratio < 1.0:
            raise PAQProbeError("maximum_hold_terminal_norm_ratio must be >= 1")
        if self.maximum_hold_terminal_norm_ratio < self.minimum_hold_terminal_norm_ratio:
            raise PAQProbeError("hold/terminal norm ratio bounds are inverted")


@dataclass(frozen=True)
class CausalThresholds:
    minimum_target_action_gain: float = 0.10
    minimum_reverse_order_score: float = 0.80
    maximum_absolute_noop_effect: float = 0.03

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise PAQProbeError(f"{name} must be finite")
            if float(value) < 0.0:
                raise PAQProbeError(f"{name} must be nonnegative")
        if not 0.0 <= self.minimum_reverse_order_score <= 1.0:
            raise PAQProbeError("minimum_reverse_order_score must be in [0, 1]")


@dataclass(frozen=True)
class CausalInterventionTrial:
    """Reference to one canonical sealed decoded causal receipt.

    No evaluator scalar is accepted through this dataclass.  Admission reads
    the actual plain receipt file, checks its canonical bytes, file hash and
    size, then re-hashes and decodes every declared exact81 media artifact.
    """

    cohort_id: str
    receipt_path: str
    receipt_file_sha256: str
    receipt_size_bytes: int


@dataclass(frozen=True)
class PhasePath:
    values: torch.Tensor
    tensor_sha256: str


@dataclass(frozen=True)
class ObservationalAudit:
    passed: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, float]
    candidate_code_sha256: Optional[str]
    candidate_code: Optional[torch.Tensor]
    candidate_is_training_eligible: bool


@dataclass(frozen=True)
class PAQDecision:
    status: str
    observational_candidate_passed: bool
    causal_intervention_passed: bool
    admitted_code: bool
    training_updates_authorized: int
    parameter_updates_executed: int
    reasons: tuple[str, ...]
    observational_metrics: Mapping[str, float]
    candidate_code_sha256: Optional[str]
    intervention_receipt_sha256: Optional[str]
    decision_receipt_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PAQProbeError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape and logical bytes, independent of strides."""

    if not isinstance(value, torch.Tensor):
        raise PAQProbeError("tensor hash input must be a torch.Tensor")
    owned = value.detach().cpu().contiguous().clone()
    header = _canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(owned.shape)}
    )
    return hashlib.sha256(header + b"\x00" + bytes(owned.untyped_storage())).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PAQProbeError("condition_text must be non-empty and stripped")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def float64_bits(value: Any, *, label: str) -> str:
    """Return the exact big-endian IEEE-754 binary64 bit pattern."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PAQProbeError(f"{label} must be a real scalar")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PAQProbeError(f"{label} must be finite")
    return struct.pack(">d", numeric).hex()


def float32_bits(value: Any, *, label: str) -> str:
    """Return the exact big-endian runtime float32 bit pattern."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PAQProbeError(f"{label} must be a real scalar")
    numeric = float(value)
    try:
        packed = struct.pack(">f", numeric)
        rounded = struct.unpack(">f", packed)[0]
    except (OverflowError, struct.error) as error:
        raise PAQProbeError(f"{label} is not representable as float32") from error
    if not math.isfinite(rounded):
        raise PAQProbeError(f"{label} must be finite float32")
    return packed.hex()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise PAQProbeError(f"{label} path must be a string")
    requested = Path(value)
    if not requested.is_absolute():
        raise PAQProbeError(f"{label} path must be absolute")
    try:
        mode = requested.lstat().st_mode
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PAQProbeError(f"{label} path is unavailable") from error
    if resolved != requested or not stat.S_ISREG(mode):
        raise PAQProbeError(f"{label} path must be canonical, regular, and non-symlink")
    return requested


def _load_canonical_receipt(path: Path) -> tuple[Mapping[str, Any], bytes]:
    raw = path.read_bytes()

    def closed_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PAQProbeError("causal receipt contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        receipt = json.loads(raw.decode("ascii"), object_pairs_hook=closed_pairs)
    except PAQProbeError:
        raise
    except Exception as error:
        raise PAQProbeError("causal receipt is not finite canonical ASCII JSON") from error
    if not isinstance(receipt, Mapping):
        raise PAQProbeError("causal receipt root must be an object")
    if raw != _canonical_json_bytes(receipt) + b"\n":
        raise PAQProbeError("causal receipt bytes are not canonical JSON plus newline")
    return receipt, raw


def _decode_exact81_media(path: Path) -> Mapping[str, Any]:
    """Decode one real media artifact and hash its logical RGB24 frames."""

    try:
        import av
    except Exception as error:
        raise PAQProbeError("PyAV is required to verify decoded causal media") from error
    digest = hashlib.sha256()
    frame_count = 0
    observed_hw: set[tuple[int, int]] = set()
    try:
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise PAQProbeError("causal media must have exactly one video stream")
            stream = streams[0]
            rate = stream.average_rate
            if rate is None or int(rate.numerator) != 25 or int(rate.denominator) != 1:
                raise PAQProbeError("causal media must decode at exact fps 25/1")
            for frame in container.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
                if tuple(rgb.shape) != (height, width, 3):
                    raise PAQProbeError("decoded causal frame is not RGB24")
                observed_hw.add((height, width))
                digest.update(struct.pack(">III", frame_count, height, width))
                digest.update(rgb.tobytes(order="C"))
                frame_count += 1
    except PAQProbeError:
        raise
    except Exception as error:
        raise PAQProbeError("causal media decode failed") from error
    if frame_count != 81 or len(observed_hw) != 1:
        raise PAQProbeError("causal media must decode to exact81 fixed-geometry frames")
    height, width = next(iter(observed_hw))
    return {
        "decoded_contract": DECODED_MEDIA_CONTRACT,
        "decoded_frame_count": frame_count,
        "decoded_fps_numerator": 25,
        "decoded_fps_denominator": 1,
        "decoded_height": height,
        "decoded_width": width,
        "decoded_rgb24_sha256": digest.hexdigest(),
    }


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PAQProbeError(f"{label} must be a lowercase SHA256")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PAQProbeError(f"{label} must be a closed non-empty identifier")
    return value


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def seal_observation_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    probe_id: str,
    checkpoint_sha256: str,
    policy_sha256: str,
    source_revision_sha256: str,
    source_exposure_registry_sha256: str,
    intervention_scale: float,
) -> dict[str, Any]:
    """Seal a canonical, sorted observation manifest.

    The function does not validate tensors; :func:`validate_observation_manifest`
    binds the sealed records to the actual hidden and noisy-state tensors.
    """

    _require_identifier(probe_id, "probe_id")
    _require_digest(checkpoint_sha256, "checkpoint_sha256")
    _require_digest(policy_sha256, "policy_sha256")
    _require_digest(source_revision_sha256, "source_revision_sha256")
    _require_digest(
        source_exposure_registry_sha256, "source_exposure_registry_sha256"
    )
    intervention_scale_bits = float64_bits(
        intervention_scale, label="intervention_scale"
    )
    if float(intervention_scale) <= 0.0:
        raise PAQProbeError("intervention_scale must be positive")
    copied = [dict(record) for record in records]
    try:
        copied.sort(key=lambda row: row["observation_id"])
    except (KeyError, TypeError) as error:
        raise PAQProbeError("every record must have a sortable observation_id") from error
    payload = {
        "schema_version": SCHEMA_VERSION,
        "probe_id": probe_id,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_sha256": policy_sha256,
        "source_revision_sha256": source_revision_sha256,
        "source_exposure_registry_sha256": source_exposure_registry_sha256,
        "intervention_scale": float(intervention_scale),
        "intervention_scale_bits": intervention_scale_bits,
        "phase_order": list(PHASE_ORDER),
        "anchor_formula": ANCHOR_FORMULA,
        "records": copied,
    }
    return {**payload, "manifest_sha256": object_sha256(payload)}


_RECORD_KEYS = {
    "observation_id",
    "cohort_id",
    "split",
    "variant",
    "identity_id",
    "scene_id",
    "seed_id",
    "requested_action_id",
    "action_family_id",
    "action_family_member_id",
    "source_exposure_id",
    "condition_text",
    "condition_text_sha256",
    "hook_site",
    "hook_block_index",
    "diffusion_step",
    "sigma",
    "sigma_float32_be_hex",
    "phase_ranges",
    "checkpoint_sha256",
    "policy_sha256",
    "source_exposure_registry_sha256",
    "hidden_key",
    "hidden_sha256",
    "noisy_state_key",
    "noisy_state_sha256",
}


def validate_observation_manifest(
    manifest: Mapping[str, Any],
    hidden_tensors: Mapping[str, torch.Tensor],
    noisy_states: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], ...]:
    """Strictly validate hashes, same-state cohorts and split isolation."""

    expected_top = {
        "schema_version",
        "probe_id",
        "checkpoint_sha256",
        "policy_sha256",
        "source_revision_sha256",
        "source_exposure_registry_sha256",
        "intervention_scale",
        "intervention_scale_bits",
        "phase_order",
        "anchor_formula",
        "records",
        "manifest_sha256",
    }
    if set(manifest) != expected_top:
        raise PAQProbeError("manifest top-level keys are not the closed PAQ schema")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise PAQProbeError("observation schema_version mismatch")
    _require_identifier(manifest["probe_id"], "probe_id")
    _require_digest(manifest["checkpoint_sha256"], "checkpoint_sha256")
    _require_digest(manifest["policy_sha256"], "policy_sha256")
    _require_digest(manifest["source_revision_sha256"], "source_revision_sha256")
    _require_digest(
        manifest["source_exposure_registry_sha256"],
        "source_exposure_registry_sha256",
    )
    if (
        float64_bits(manifest["intervention_scale"], label="intervention_scale")
        != manifest["intervention_scale_bits"]
        or float(manifest["intervention_scale"]) <= 0.0
    ):
        raise PAQProbeError("intervention_scale bits/value differ")
    if manifest["phase_order"] != list(PHASE_ORDER):
        raise PAQProbeError("manifest phase order must be start->transition->terminal->hold")
    if manifest["anchor_formula"] != ANCHOR_FORMULA:
        raise PAQProbeError("temporal mean or an unknown anchor formula is forbidden")
    declared_manifest_hash = _require_digest(
        manifest["manifest_sha256"], "manifest_sha256"
    )
    if object_sha256(_manifest_payload(manifest)) != declared_manifest_hash:
        raise PAQProbeError("manifest_sha256 does not bind the manifest payload")
    records_raw = manifest["records"]
    if not isinstance(records_raw, list) or not records_raw:
        raise PAQProbeError("manifest records must be a non-empty list")
    if records_raw != sorted(records_raw, key=lambda row: row.get("observation_id", "")):
        raise PAQProbeError("manifest records must be canonically sorted by observation_id")

    observations: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    expected_hidden_keys: set[str] = set()
    expected_state_keys: set[str] = set()
    cohort_rows: dict[str, list[dict[str, Any]]] = {}
    for raw in records_raw:
        if not isinstance(raw, dict) or set(raw) != _RECORD_KEYS:
            raise PAQProbeError("observation record does not match the closed schema")
        row = dict(raw)
        for key in (
            "observation_id",
            "cohort_id",
            "identity_id",
            "scene_id",
            "seed_id",
            "requested_action_id",
            "action_family_id",
            "action_family_member_id",
            "source_exposure_id",
            "hidden_key",
            "noisy_state_key",
        ):
            _require_identifier(row[key], key)
        if row["observation_id"] in observation_ids:
            raise PAQProbeError("observation_id values must be unique")
        observation_ids.add(row["observation_id"])
        if row["split"] not in REQUIRED_SPLITS:
            raise PAQProbeError(f"split must be one of {REQUIRED_SPLITS}")
        if row["variant"] not in REQUIRED_VARIANTS:
            raise PAQProbeError(f"variant must be one of {REQUIRED_VARIANTS}")
        if text_sha256(row["condition_text"]) != _require_digest(
            row["condition_text_sha256"], "condition_text_sha256"
        ):
            raise PAQProbeError("condition_text_sha256 mismatch")
        match = _HOOK_SITE.fullmatch(row["hook_site"])
        if match is None or int(match.group(1)) >= 30:
            raise PAQProbeError("hook_site is not a supported Bernini to_out.0 hook")
        if match.group(2) == "attn1" and match.group(3) != "pre":
            raise PAQProbeError("PAQ admits attn1.to_out.0 pre only")
        if (
            isinstance(row["hook_block_index"], bool)
            or not isinstance(row["hook_block_index"], int)
            or row["hook_block_index"] != int(match.group(1))
        ):
            raise PAQProbeError("hook_block_index does not match hook_site")
        if (
            isinstance(row["diffusion_step"], bool)
            or not isinstance(row["diffusion_step"], int)
            or row["diffusion_step"] < 0
        ):
            raise PAQProbeError("diffusion_step must be a nonnegative integer")
        if isinstance(row["sigma"], bool) or not math.isfinite(float(row["sigma"])):
            raise PAQProbeError("sigma must be finite")
        if not 0.0 <= float(row["sigma"]):
            raise PAQProbeError("sigma must be nonnegative")
        if (
            not isinstance(row["sigma_float32_be_hex"], str)
            or re.fullmatch(r"[0-9a-f]{8}", row["sigma_float32_be_hex"]) is None
            or float32_bits(row["sigma"], label="sigma")
            != row["sigma_float32_be_hex"]
        ):
            raise PAQProbeError("sigma_float32_be_hex does not bind runtime sigma bits")
        for field in (
            "checkpoint_sha256",
            "policy_sha256",
            "source_exposure_registry_sha256",
        ):
            if (
                _require_digest(row[field], field) != manifest[field]
            ):
                raise PAQProbeError(f"record {field} differs from manifest authority")
        phase_ranges = PhaseRanges.from_mapping(row["phase_ranges"])

        hidden_key = row["hidden_key"]
        state_key = row["noisy_state_key"]
        if hidden_key in expected_hidden_keys:
            raise PAQProbeError("every observation must own a unique hidden_key")
        expected_hidden_keys.add(hidden_key)
        expected_state_keys.add(state_key)
        if hidden_key not in hidden_tensors or state_key not in noisy_states:
            raise PAQProbeError("manifest references an unavailable tensor")
        hidden = hidden_tensors[hidden_key]
        state = noisy_states[state_key]
        _validate_hidden(hidden, label=hidden_key)
        _validate_finite_tensor(state, label=state_key)
        phase_ranges.validate_frame_count(int(hidden.shape[0]))
        if tensor_sha256(hidden) != _require_digest(row["hidden_sha256"], "hidden_sha256"):
            raise PAQProbeError(f"hidden tensor hash mismatch for {hidden_key}")
        if tensor_sha256(state) != _require_digest(
            row["noisy_state_sha256"], "noisy_state_sha256"
        ):
            raise PAQProbeError(f"noisy-state tensor hash mismatch for {state_key}")
        row["_phase_ranges"] = phase_ranges
        observations.append(row)
        cohort_rows.setdefault(row["cohort_id"], []).append(row)

    if set(hidden_tensors) != expected_hidden_keys:
        raise PAQProbeError("hidden tensor mapping has missing or unmanifested keys")
    if set(noisy_states) != expected_state_keys:
        raise PAQProbeError("noisy-state mapping has missing or unmanifested keys")

    state_owner: dict[str, str] = {}
    for cohort_id, rows in cohort_rows.items():
        if len(rows) != len(REQUIRED_VARIANTS):
            raise PAQProbeError("each cohort must contain exactly five counterfactuals")
        if {row["variant"] for row in rows} != set(REQUIRED_VARIANTS):
            raise PAQProbeError("cohort counterfactual variants are incomplete or duplicated")
        invariant_fields = (
            "split",
            "identity_id",
            "scene_id",
            "seed_id",
            "requested_action_id",
            "action_family_id",
            "action_family_member_id",
            "source_exposure_id",
            "hook_site",
            "hook_block_index",
            "diffusion_step",
            "sigma",
            "sigma_float32_be_hex",
            "phase_ranges",
            "checkpoint_sha256",
            "policy_sha256",
            "source_exposure_registry_sha256",
            "noisy_state_key",
            "noisy_state_sha256",
        )
        for field in invariant_fields:
            if len({_jsonable_key(row[field]) for row in rows}) != 1:
                raise PAQProbeError(f"cohort {cohort_id} does not share exact {field}")
        shapes = {tuple(hidden_tensors[row["hidden_key"]].shape) for row in rows}
        if len(shapes) != 1:
            raise PAQProbeError("same-state counterfactual hidden shapes must match")
        state_signature = rows[0]["noisy_state_sha256"]
        previous_owner = state_owner.setdefault(state_signature, cohort_id)
        if previous_owner != cohort_id:
            raise PAQProbeError("a noisy state may belong to only one cohort")

    # A PAQ candidate is one coordinate experiment, not a post-hoc mixture of
    # actions, hook coordinates, schedule cells, or authority registries.
    globally_bound_fields = (
        "requested_action_id",
        "action_family_id",
        "hook_site",
        "hook_block_index",
        "diffusion_step",
        "sigma",
        "sigma_float32_be_hex",
        "phase_ranges",
        "checkpoint_sha256",
        "policy_sha256",
        "source_exposure_registry_sha256",
    )
    for field in globally_bound_fields:
        values = {_jsonable_key(row[field]) for row in observations}
        if len(values) != 1:
            raise PAQProbeError(f"all cohorts must share global {field}")

    if {row["split"] for row in observations} != set(REQUIRED_SPLITS):
        raise PAQProbeError("both discovery and admission splits are required")
    _validate_split_isolation(observations)
    return tuple(observations)


def _jsonable_key(value: Any) -> str:
    return _canonical_json_bytes(value).decode("ascii")


def _validate_finite_tensor(value: Any, *, label: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise PAQProbeError(f"{label} must be a floating torch.Tensor")
    if value.numel() == 0 or not bool(torch.isfinite(value.detach()).all().item()):
        raise PAQProbeError(f"{label} must be non-empty and finite")


def _validate_hidden(value: Any, *, label: str) -> None:
    _validate_finite_tensor(value, label=label)
    if value.ndim not in (2, 3):
        raise PAQProbeError(f"{label} must have [T,D] or [T,P,D] shape")
    if any(int(size) <= 0 for size in value.shape):
        raise PAQProbeError(f"{label} has an empty dimension")


def _validate_split_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    """Forbid entity/scene/seed/cohort/tensor leakage across held-out split."""

    discovery = [row for row in rows if row["split"] == "discovery"]
    admission = [row for row in rows if row["split"] == "admission"]
    fields = (
        "identity_id",
        "scene_id",
        "seed_id",
        "cohort_id",
        "action_family_member_id",
        "source_exposure_id",
        "hidden_sha256",
        "noisy_state_sha256",
    )
    for field in fields:
        overlap = {row[field] for row in discovery} & {row[field] for row in admission}
        if overlap:
            raise PAQProbeError(f"split leakage detected in {field}: {sorted(overlap)!r}")


def start_anchored_phase_path(hidden: torch.Tensor, ranges: PhaseRanges) -> PhasePath:
    """Extract ``start->transition->terminal->hold`` without mean-centering time."""

    _validate_hidden(hidden, label="hidden")
    ranges.validate_frame_count(int(hidden.shape[0]))
    value = hidden.detach().to(device="cpu", dtype=torch.float64)
    if value.ndim == 3:
        value = value.mean(dim=1)
    begin, stop = ranges.start
    start_anchor = value[begin:stop].mean(dim=0, keepdim=True)
    residual = value - start_anchor
    phases = torch.stack(
        [residual[slice(*getattr(ranges, name))].mean(dim=0) for name in PHASE_ORDER],
        dim=0,
    )
    # This exact invariant catches accidental whole-trajectory centering.
    if not torch.allclose(phases[0], torch.zeros_like(phases[0]), atol=1e-10, rtol=0.0):
        raise PAQProbeError("start-anchored residual failed to make start phase zero")
    frozen = phases.to(dtype=torch.float32).contiguous()
    return PhasePath(values=frozen, tensor_sha256=tensor_sha256(frozen))


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.to(dtype=torch.float64)).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.to(dtype=torch.float64).reshape(-1)
    right64 = right.to(dtype=torch.float64).reshape(-1)
    denominator = _norm(left64) * _norm(right64)
    if denominator <= _EPS:
        return -1.0
    return float(torch.dot(left64, right64).item() / denominator)


def _minimum_leave_group_out_cosine(
    codes: Sequence[torch.Tensor], labels: Sequence[str]
) -> float:
    groups = sorted(set(labels))
    if len(groups) < 2:
        return -1.0
    values: list[float] = []
    for group in groups:
        inside = torch.stack([code for code, label in zip(codes, labels) if label == group]).mean(0)
        outside = torch.stack([code for code, label in zip(codes, labels) if label != group]).mean(0)
        values.append(_cosine(inside, outside))
    return min(values)


def _cohort_paths(
    observations: Sequence[Mapping[str, Any]],
    hidden_tensors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, PhasePath]]:
    result: dict[str, dict[str, PhasePath]] = {}
    for row in observations:
        ranges = row.get("_phase_ranges")
        if not isinstance(ranges, PhaseRanges):
            ranges = PhaseRanges.from_mapping(row["phase_ranges"])
        path = start_anchored_phase_path(hidden_tensors[row["hidden_key"]], ranges)
        result.setdefault(row["cohort_id"], {})[row["variant"]] = path
    return result


def observational_audit(
    manifest: Mapping[str, Any],
    hidden_tensors: Mapping[str, torch.Tensor],
    noisy_states: Mapping[str, torch.Tensor],
    *,
    thresholds: PAQThresholds = PAQThresholds(),
) -> ObservationalAudit:
    """Fit/audit a provisional phase quotient; never admit it for training."""

    rows = validate_observation_manifest(manifest, hidden_tensors, noisy_states)
    paths = _cohort_paths(rows, hidden_tensors)
    metadata = {row["cohort_id"]: row for row in rows}
    codes: dict[str, torch.Tensor] = {}
    noop_paths: dict[str, torch.Tensor] = {}
    reverse_codes: dict[str, torch.Tensor] = {}
    incomplete_codes: dict[str, torch.Tensor] = {}
    wrong_codes: dict[str, torch.Tensor] = {}
    for cohort_id, variants in paths.items():
        noop = variants["noop"].values
        codes[cohort_id] = variants["target"].values - noop
        noop_paths[cohort_id] = noop
        reverse_codes[cohort_id] = variants["reverse"].values - noop
        incomplete_codes[cohort_id] = variants["incomplete"].values - noop
        wrong_codes[cohort_id] = variants["wrong_action"].values - noop

    discovery_ids = sorted(
        cohort_id for cohort_id, row in metadata.items() if row["split"] == "discovery"
    )
    admission_ids = sorted(
        cohort_id for cohort_id, row in metadata.items() if row["split"] == "admission"
    )
    reasons: list[str] = []
    for split, cohort_ids in (("discovery", discovery_ids), ("admission", admission_ids)):
        for field, minimum in (
            ("identity_id", thresholds.minimum_identities_per_split),
            ("scene_id", thresholds.minimum_scenes_per_split),
            ("seed_id", thresholds.minimum_seeds_per_split),
        ):
            count = len({metadata[cohort_id][field] for cohort_id in cohort_ids})
            if count < minimum:
                reasons.append(f"{split}_{field}_coverage={count}<{minimum}")

    target_norms = [_norm(codes[key]) for key in sorted(codes)]
    noop_ratios = [
        _norm(noop_paths[key]) / max(_norm(codes[key]), _EPS) for key in sorted(codes)
    ]
    reverse_opposition = [
        _cosine(reverse_codes[key], -codes[key]) for key in sorted(codes)
    ]
    incomplete_transition = [
        _cosine(incomplete_codes[key][1], codes[key][1]) for key in sorted(codes)
    ]
    incomplete_hold_ratio = [
        _norm(incomplete_codes[key][3]) / max(_norm(codes[key][3]), _EPS)
        for key in sorted(codes)
    ]
    wrong_cosines = [_cosine(wrong_codes[key], codes[key]) for key in sorted(codes)]

    all_ids = sorted(codes)
    all_code_values = [codes[key] for key in all_ids]
    identity_stability = _minimum_leave_group_out_cosine(
        all_code_values, [metadata[key]["identity_id"] for key in all_ids]
    )
    scene_stability = _minimum_leave_group_out_cosine(
        all_code_values, [metadata[key]["scene_id"] for key in all_ids]
    )
    seed_stability = _minimum_leave_group_out_cosine(
        all_code_values, [metadata[key]["seed_id"] for key in all_ids]
    )
    discovery_template = torch.stack([codes[key] for key in discovery_ids]).mean(0)
    cross_split_cosine = min(
        _cosine(discovery_template, codes[key]) for key in admission_ids
    )

    # Phase progression is measured on the discovery template and verified on
    # every held-out code.  Hold is a retained terminal plateau, not another
    # transition and not a temporal-mean-zero artifact.
    phase_metrics: list[tuple[float, float, float, float, float]] = []
    direction = discovery_template[3]
    direction_norm_sq = float(torch.dot(direction.double(), direction.double()).item())
    if direction_norm_sq <= _EPS:
        # Keep fail-closed receipts canonical JSON (no NaN/Inf).
        phase_metrics.append((-1.0, -1.0, 1.0e30, -1.0, -1.0))
    else:
        for key in admission_ids:
            code = codes[key]
            progress = torch.mv(code.double(), direction.double()) / direction_norm_sq
            terminal_scale = max(abs(float(progress[2].item())), _EPS)
            transition_fraction = float(progress[1].item()) / terminal_scale
            terminal_margin = float(progress[2].item() - progress[1].item())
            hold_delta = abs(float(progress[3].item() - progress[2].item())) / terminal_scale
            hold_cosine = _cosine(code[3], code[2])
            hold_ratio = _norm(code[3]) / max(_norm(code[2]), _EPS)
            phase_metrics.append(
                (transition_fraction, terminal_margin, hold_delta, hold_cosine, hold_ratio)
            )

    metrics = {
        "minimum_target_contrast_norm": min(target_norms),
        # Diagnostic only.  A noop/source path may legitimately contain the
        # original video's motion; inertness is tested by decoded causal
        # intervention below, never by demanding a temporally static source.
        "diagnostic_maximum_noop_to_target_energy_ratio": max(noop_ratios),
        "minimum_reverse_opposition_cosine": min(reverse_opposition),
        "minimum_incomplete_transition_cosine": min(incomplete_transition),
        "maximum_incomplete_hold_ratio": max(incomplete_hold_ratio),
        "maximum_wrong_action_cosine": max(wrong_cosines),
        "minimum_identity_leave_group_out_cosine": identity_stability,
        "minimum_scene_leave_group_out_cosine": scene_stability,
        "minimum_seed_leave_group_out_cosine": seed_stability,
        "minimum_cross_split_cosine": cross_split_cosine,
        "minimum_transition_progress_fraction": min(item[0] for item in phase_metrics),
        "minimum_terminal_progress_margin": min(item[1] for item in phase_metrics),
        "maximum_hold_terminal_progress_delta": max(item[2] for item in phase_metrics),
        "minimum_hold_terminal_cosine": min(item[3] for item in phase_metrics),
        "minimum_hold_terminal_norm_ratio": min(item[4] for item in phase_metrics),
        "maximum_hold_terminal_norm_ratio": max(item[4] for item in phase_metrics),
    }
    checks = (
        (metrics["minimum_target_contrast_norm"] >= thresholds.minimum_target_contrast_norm,
         "target_contrast_too_small"),
        (metrics["minimum_reverse_opposition_cosine"] >= thresholds.minimum_reverse_opposition_cosine,
         "reverse_does_not_encode_opposite_order"),
        (metrics["minimum_incomplete_transition_cosine"] >= thresholds.minimum_incomplete_transition_cosine,
         "incomplete_lacks_shared_transition"),
        (metrics["maximum_incomplete_hold_ratio"] <= thresholds.maximum_incomplete_hold_ratio,
         "incomplete_contains_terminal_hold"),
        (metrics["maximum_wrong_action_cosine"] <= thresholds.maximum_wrong_action_cosine,
         "wrong_action_not_specific"),
        (identity_stability >= thresholds.minimum_group_stability_cosine,
         "identity_instability"),
        (scene_stability >= thresholds.minimum_group_stability_cosine,
         "scene_instability"),
        (seed_stability >= thresholds.minimum_group_stability_cosine,
         "seed_instability"),
        (cross_split_cosine >= thresholds.minimum_cross_split_cosine,
         "heldout_split_instability"),
        (metrics["minimum_transition_progress_fraction"] >= thresholds.minimum_transition_progress_fraction,
         "missing_transition_phase"),
        (metrics["minimum_terminal_progress_margin"] >= thresholds.minimum_terminal_progress_margin,
         "terminal_does_not_follow_transition"),
        (metrics["maximum_hold_terminal_progress_delta"] <= thresholds.maximum_hold_terminal_progress_delta,
         "hold_is_not_terminal_plateau"),
        (metrics["minimum_hold_terminal_cosine"] >= thresholds.minimum_hold_terminal_cosine,
         "hold_changes_terminal_direction"),
        (metrics["minimum_hold_terminal_norm_ratio"] >= thresholds.minimum_hold_terminal_norm_ratio,
         "hold_erases_terminal_state"),
        (metrics["maximum_hold_terminal_norm_ratio"] <= thresholds.maximum_hold_terminal_norm_ratio,
         "hold_amplifies_terminal_state"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    candidate = discovery_template.detach().cpu().to(dtype=torch.float32).contiguous()
    candidate_hash = tensor_sha256(candidate) if not reasons else None
    return ObservationalAudit(
        passed=not reasons,
        reasons=tuple(reasons),
        metrics=metrics,
        candidate_code_sha256=candidate_hash,
        candidate_code=candidate if not reasons else None,
        candidate_is_training_eligible=False,
    )


_CAUSAL_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "manifest_sha256",
    "cohort_id",
    "split",
    "identity_id",
    "scene_id",
    "seed_id",
    "requested_action_id",
    "action_family_id",
    "action_family_member_id",
    "source_exposure_id",
    "checkpoint_sha256",
    "policy_sha256",
    "source_revision_sha256",
    "source_exposure_registry_sha256",
    "hook_site",
    "hook_block_index",
    "diffusion_step",
    "sigma",
    "sigma_float32_be_hex",
    "phase_ranges",
    "candidate_code_sha256",
    "intervention_scale",
    "intervention_scale_bits",
    "decoded_media_contract",
    "decoded_media",
    "action_metrics",
    "preservation_axes",
    "no_weighted_preservation_aggregate",
    "receipt_payload_sha256",
}
_CAUSAL_MEDIA_KEYS = {
    "path",
    "file_sha256",
    "size_bytes",
    "decoded_contract",
    "decoded_frame_count",
    "decoded_fps_numerator",
    "decoded_fps_denominator",
    "decoded_height",
    "decoded_width",
    "decoded_rgb24_sha256",
}
_ACTION_METRIC_KEYS = {
    "baseline_action_score",
    "target_action_score",
    "reverse_order_score",
    "noop_effect",
}
_PRESERVATION_ROW_KEYS = {"baseline_score", "target_score"}


def _causal_receipt_payload(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }


def _validate_causal_receipt(
    reference: CausalInterventionTrial,
    *,
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_code_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if reference.cohort_id != row["cohort_id"]:
        raise PAQProbeError("causal reference cohort differs from admission row")
    expected_file_sha = _require_digest(
        reference.receipt_file_sha256, "causal receipt file SHA256"
    )
    if (
        isinstance(reference.receipt_size_bytes, bool)
        or not isinstance(reference.receipt_size_bytes, int)
        or reference.receipt_size_bytes <= 0
    ):
        raise PAQProbeError("causal receipt expected size must be a positive integer")
    receipt_path = _plain_absolute_file(
        reference.receipt_path, label="causal receipt"
    )
    actual_size = receipt_path.stat().st_size
    actual_file_sha = file_sha256(receipt_path)
    if (
        actual_size != reference.receipt_size_bytes
        or actual_file_sha != expected_file_sha
    ):
        raise PAQProbeError("causal receipt actual hash/size differs")
    receipt, raw = _load_canonical_receipt(receipt_path)
    if len(raw) != actual_size or set(receipt) != _CAUSAL_RECEIPT_KEYS:
        raise PAQProbeError("causal receipt does not match the closed schema")
    if receipt["schema_version"] != CAUSAL_RECEIPT_SCHEMA_VERSION:
        raise PAQProbeError("causal receipt schema_version differs")
    _require_identifier(receipt["receipt_id"], "causal receipt_id")
    payload_sha = _require_digest(
        receipt["receipt_payload_sha256"], "causal receipt payload SHA256"
    )
    if object_sha256(_causal_receipt_payload(receipt)) != payload_sha:
        raise PAQProbeError("causal receipt payload seal differs")

    row_bound_fields = (
        "cohort_id",
        "identity_id",
        "scene_id",
        "seed_id",
        "requested_action_id",
        "action_family_id",
        "action_family_member_id",
        "source_exposure_id",
        "hook_site",
        "hook_block_index",
        "diffusion_step",
        "sigma_float32_be_hex",
        "phase_ranges",
    )
    for field in row_bound_fields:
        if _jsonable_key(receipt[field]) != _jsonable_key(row[field]):
            raise PAQProbeError(f"causal receipt {field} differs from observation")
    manifest_bound_fields = (
        "manifest_sha256",
        "checkpoint_sha256",
        "policy_sha256",
        "source_revision_sha256",
        "source_exposure_registry_sha256",
        "intervention_scale",
        "intervention_scale_bits",
    )
    for field in manifest_bound_fields:
        if _jsonable_key(receipt[field]) != _jsonable_key(manifest[field]):
            raise PAQProbeError(f"causal receipt {field} differs from manifest")
    if (
        receipt["split"] != "admission"
        or float32_bits(receipt["sigma"], label="causal sigma")
        != receipt["sigma_float32_be_hex"]
        or float64_bits(
            receipt["intervention_scale"], label="causal intervention_scale"
        )
        != receipt["intervention_scale_bits"]
        or _require_digest(
            receipt["candidate_code_sha256"], "causal candidate_code_sha256"
        )
        != candidate_code_sha256
        or receipt["decoded_media_contract"] != DECODED_MEDIA_CONTRACT
        or receipt["no_weighted_preservation_aggregate"] is not True
    ):
        raise PAQProbeError("causal coordinate/candidate/intervention binding differs")

    media = receipt["decoded_media"]
    if not isinstance(media, Mapping) or set(media) != set(CAUSAL_MEDIA_ROLES):
        raise PAQProbeError("causal receipt must bind all four decoded media roles")
    media_paths: set[Path] = set()
    verified_media: dict[str, Any] = {}
    for role in CAUSAL_MEDIA_ROLES:
        artifact = media[role]
        if not isinstance(artifact, Mapping) or set(artifact) != _CAUSAL_MEDIA_KEYS:
            raise PAQProbeError(f"causal {role} media row differs from closed schema")
        path = _plain_absolute_file(artifact["path"], label=f"causal {role} media")
        if path in media_paths:
            raise PAQProbeError("causal media roles must use distinct artifacts")
        media_paths.add(path)
        expected_sha = _require_digest(
            artifact["file_sha256"], f"causal {role} media SHA256"
        )
        size = artifact["size_bytes"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or path.stat().st_size != size
            or file_sha256(path) != expected_sha
        ):
            raise PAQProbeError(f"causal {role} media actual hash/size differs")
        decoded = _decode_exact81_media(path)
        for field, value in decoded.items():
            if artifact[field] != value:
                raise PAQProbeError(f"causal {role} decoded exact81 evidence differs")
        verified_media[role] = {
            "path": str(path),
            "file_sha256": expected_sha,
            "size_bytes": size,
            **decoded,
        }

    metrics = receipt["action_metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != _ACTION_METRIC_KEYS:
        raise PAQProbeError("causal action metrics differ from closed schema")
    for name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -1.0 <= float(value) <= 1.0
        ):
            raise PAQProbeError(f"causal action metric {name} is invalid")

    preservation = receipt["preservation_axes"]
    if not isinstance(preservation, Mapping) or set(preservation) != set(PRESERVATION_AXES):
        raise PAQProbeError("causal preservation must contain exactly six axes")
    for axis in PRESERVATION_AXES:
        axis_row = preservation[axis]
        if not isinstance(axis_row, Mapping) or set(axis_row) != _PRESERVATION_ROW_KEYS:
            raise PAQProbeError(f"causal preservation axis {axis} differs")
        for score in axis_row.values():
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise PAQProbeError(f"causal preservation axis {axis} score invalid")
    return receipt, {
        "cohort_id": reference.cohort_id,
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": actual_file_sha,
        "receipt_size_bytes": actual_size,
        "receipt_payload_sha256": payload_sha,
        "decoded_media": verified_media,
    }


def _validate_causal_trials(
    trials: Sequence[CausalInterventionTrial],
    admission_rows: Sequence[Mapping[str, Any]],
    thresholds: CausalThresholds,
    *,
    manifest: Mapping[str, Any],
    candidate_code_sha256: Optional[str],
) -> tuple[bool, tuple[str, ...], Optional[str]]:
    expected = {row["cohort_id"]: row for row in admission_rows}
    if not trials:
        return False, ("missing_frozen_causal_intervention_trials",), None
    if candidate_code_sha256 is None:
        return False, ("observational_candidate_absent_for_causal_binding",), None
    if len(trials) != len(expected) or {trial.cohort_id for trial in trials} != set(expected):
        raise PAQProbeError("causal trials must cover every admission cohort exactly once")
    if len({trial.cohort_id for trial in trials}) != len(trials):
        raise PAQProbeError("causal trial cohort_id values must be unique")
    reasons: list[str] = []
    receipt_rows: list[dict[str, Any]] = []
    for trial in sorted(trials, key=lambda item: item.cohort_id):
        row = expected[trial.cohort_id]
        receipt, receipt_identity = _validate_causal_receipt(
            trial,
            row=row,
            manifest=manifest,
            candidate_code_sha256=candidate_code_sha256,
        )
        metrics = receipt["action_metrics"]
        target_gain = float(metrics["target_action_score"]) - float(
            metrics["baseline_action_score"]
        )
        if target_gain < thresholds.minimum_target_action_gain:
            reasons.append(f"{trial.cohort_id}:target_action_gain_failed")
        if float(metrics["reverse_order_score"]) < thresholds.minimum_reverse_order_score:
            reasons.append(f"{trial.cohort_id}:reverse_order_failed")
        if abs(float(metrics["noop_effect"])) > thresholds.maximum_absolute_noop_effect:
            reasons.append(f"{trial.cohort_id}:noop_not_inert")
        preservation_deltas: dict[str, float] = {}
        for axis in PRESERVATION_AXES:
            axis_row = receipt["preservation_axes"][axis]
            delta = float(axis_row["target_score"]) - float(
                axis_row["baseline_score"]
            )
            preservation_deltas[axis] = delta
            # Every axis is a hard noninferiority constraint.  There is no
            # weighted aggregate and no historical 0.02 degradation budget.
            if delta < 0.0:
                reasons.append(f"{trial.cohort_id}:preservation_{axis}_worse")
        receipt_rows.append(
            {
                **receipt_identity,
                "target_action_gain": target_gain,
                "preservation_target_minus_baseline": preservation_deltas,
            }
        )
    receipt = object_sha256(
        {
            "contract": "sealed_decoded_exact81_target_reverse_noop_six_axis_noninferiority_v1",
            "thresholds": asdict(thresholds),
            "trials": receipt_rows,
        }
    )
    return not reasons, tuple(reasons), receipt


def decide_phase_action_quotient(
    manifest: Mapping[str, Any],
    hidden_tensors: Mapping[str, torch.Tensor],
    noisy_states: Mapping[str, torch.Tensor],
    *,
    observational_thresholds: PAQThresholds = PAQThresholds(),
    causal_trials: Optional[Sequence[CausalInterventionTrial]] = None,
    causal_thresholds: CausalThresholds = CausalThresholds(),
) -> PAQDecision:
    """Return a fail-closed PAQ decision and execute no parameter update."""

    observations = validate_observation_manifest(manifest, hidden_tensors, noisy_states)
    audit = observational_audit(
        manifest,
        hidden_tensors,
        noisy_states,
        thresholds=observational_thresholds,
    )
    admission_rows = [row for row in observations if row["split"] == "admission"]
    causal_passed, causal_reasons, intervention_hash = _validate_causal_trials(
        tuple(causal_trials or ()),
        admission_rows,
        causal_thresholds,
        manifest=manifest,
        candidate_code_sha256=audit.candidate_code_sha256,
    )
    admitted = audit.passed and causal_passed
    reasons = tuple(audit.reasons) + tuple(causal_reasons)
    status = ADMISSION_STATUS if admitted else NO_ADMISSION_STATUS
    payload = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "status": status,
        "observational_candidate_passed": audit.passed,
        "causal_intervention_passed": causal_passed,
        "admitted_code": admitted,
        "training_updates_authorized": int(admitted),
        "parameter_updates_executed": 0,
        "reasons": list(reasons),
        "observational_metrics": dict(audit.metrics),
        "candidate_code_sha256": audit.candidate_code_sha256,
        "intervention_receipt_sha256": intervention_hash,
        "linear_decodability_is_admission_evidence": False,
    }
    return PAQDecision(
        status=status,
        observational_candidate_passed=audit.passed,
        causal_intervention_passed=causal_passed,
        admitted_code=admitted,
        training_updates_authorized=int(admitted),
        parameter_updates_executed=0,
        reasons=reasons,
        observational_metrics=audit.metrics,
        candidate_code_sha256=audit.candidate_code_sha256,
        intervention_receipt_sha256=intervention_hash,
        decision_receipt_sha256=object_sha256(payload),
    )


__all__ = [
    "ADMISSION_STATUS",
    "ANCHOR_FORMULA",
    "BerniniHookPoint",
    "CAUSAL_MEDIA_ROLES",
    "CAUSAL_RECEIPT_SCHEMA_VERSION",
    "CausalInterventionTrial",
    "CausalThresholds",
    "DECODED_MEDIA_CONTRACT",
    "NO_ADMISSION_STATUS",
    "PAQDecision",
    "PAQProbeError",
    "PAQThresholds",
    "PHASE_ORDER",
    "PRESERVATION_AXES",
    "PhasePath",
    "PhaseRanges",
    "REQUIRED_SPLITS",
    "REQUIRED_VARIANTS",
    "SCHEMA_VERSION",
    "bernini_hook_plan",
    "decide_phase_action_quotient",
    "file_sha256",
    "float32_bits",
    "float64_bits",
    "object_sha256",
    "observational_audit",
    "seal_observation_manifest",
    "start_anchored_phase_path",
    "tensor_sha256",
    "text_sha256",
    "validate_observation_manifest",
]
