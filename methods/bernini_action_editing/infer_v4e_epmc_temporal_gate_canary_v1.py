#!/usr/bin/env python3
"""Render the future-use v4-E decoded-residual EPMC video canary.

This adapter is deliberately unsealed.  Every callable execution entry point
fails before parsing a user path, importing the Bernini heavy runtime, or
creating output until the completed v4-E authority and gate materializer are
independently pinned.  Sealing does not change the experiment: exact OOF row
``7b88a1ca1f804f41`` only, proposal seed 2027, render seeds 2028/2029, and
same-render-seed ``B0``/``zero``/``correct``/``reverse``/``shuffle`` arms.

The inherited v4-B EPMC runner remains the heavy path: official Bernini load,
source VAE encode, action/no-op proposal carrier, 40-step APG sampling,
Ulysses-4, the first 16 real post-varlen-attention head hooks, VAE decode, and
transactional MP4 writes.  Only the sealed gate-state loader, arm codes, and
diagnostic receipt are replaced.

This is a privileged OOF temporal-gating diagnostic.  The action-anchor RGB is
detached and never enters Bernini, but its V-JEPA feature did create the gate.
Consequently this program makes no source+instruction-only, action
representation, renderer, video-editing, or video-quality claim.
The upstream v4-E gate covers only transform families exposed during model
development; this canary is not an unseen-transform or unseen-action result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Optional, Sequence


RECEIPT_SCHEMA = "bernini-v4e-epmc-temporal-gate-video-canary-v1"
RECEIPT_STATUS = "V4E_EPMC_TEMPORAL_GATE_VIDEO_CANARY_COMPLETE_DIAGNOSTIC_ONLY"
PIN_PLACEHOLDER = "TO_BE_PINNED"
RELEASE_SEALED = False

# These are intentionally unresolved.  They may be filled only after the v4-E
# receipt/checkpoint contract and this materializer have passed final audit.
EXPECTED_GATE_STATE_SCHEMA = PIN_PLACEHOLDER
EXPECTED_GATE_STATE_STATUS = PIN_PLACEHOLDER
EXPECTED_GATE_STATE_FILE_SHA256 = PIN_PLACEHOLDER
EXPECTED_GATE_STATE_SELF_DIGEST = PIN_PLACEHOLDER
EXPECTED_GATE_MATERIALIZER_SHA256 = PIN_PLACEHOLDER
EXPECTED_V4B_RUNTIME_IMPLEMENTATION_SHA256 = PIN_PLACEHOLDER

EXPECTED_IID = "7b88a1ca1f804f41"
EXPECTED_OUTER_FOLD = 1
EXPECTED_SOURCE_SHA256 = (
    "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
)
EXPECTED_ANCHOR_SHA256 = (
    "8234f5f35f7001134cf074263c481e3a8079c10f799370090d30e054aef02015"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
)
PROPOSAL_SEED = 2027
RENDER_SEEDS = (2028, 2029)
ARM_ORDER = ("B0", "zero", "correct", "reverse", "shuffle")
PATCHED_ARM_ORDER = ARM_ORDER[1:]
OUTPUT_ORDER = ARM_ORDER
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_GATE_STATE_BYTES = 1 << 20


class V4EEPMCVideoCanaryError(RuntimeError):
    """The release authority, gate state, or inherited Bernini ABI differed."""


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
        raise V4EEPMCVideoCanaryError("value is not canonical JSON") from error


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
        raise V4EEPMCVideoCanaryError(f"{label} must be a lowercase SHA-256")
    return value


def _require_release_sealed() -> Any:
    """Fail before any heavyweight import or user-controlled filesystem read."""

    pins = (
        EXPECTED_GATE_STATE_SCHEMA,
        EXPECTED_GATE_STATE_STATUS,
        EXPECTED_GATE_STATE_FILE_SHA256,
        EXPECTED_GATE_STATE_SELF_DIGEST,
        EXPECTED_GATE_MATERIALIZER_SHA256,
        EXPECTED_V4B_RUNTIME_IMPLEMENTATION_SHA256,
    )
    if (
        RELEASE_SEALED is not True
        or any(pin == PIN_PLACEHOLDER for pin in pins)
        or _SHA256.fullmatch(EXPECTED_GATE_STATE_FILE_SHA256) is None
        or _SHA256.fullmatch(EXPECTED_GATE_STATE_SELF_DIGEST) is None
        or _SHA256.fullmatch(EXPECTED_GATE_MATERIALIZER_SHA256) is None
        or _SHA256.fullmatch(EXPECTED_V4B_RUNTIME_IMPLEMENTATION_SHA256) is None
    ):
        raise V4EEPMCVideoCanaryError(
            "UNSEALED v4-E EPMC video canary: gate authorities are TO_BE_PINNED"
        )

    # Deliberately deferred until the local release surface is fully pinned.
    from methods.bernini_action_editing import (
        materialize_v4e_epmc_gate_state_v1 as gate_materializer,
    )

    gate_materializer._require_release_sealed()
    if (
        _file_sha256(Path(gate_materializer.__file__).resolve(strict=True))
        != EXPECTED_GATE_MATERIALIZER_SHA256
        or gate_materializer.SCHEMA != EXPECTED_GATE_STATE_SCHEMA
        or gate_materializer.STATUS != EXPECTED_GATE_STATE_STATUS
    ):
        raise V4EEPMCVideoCanaryError(
            "sealed v4-E gate materializer source/schema binding differs"
        )
    return gate_materializer


def _runtime_modules() -> tuple[Any, Any, Any, Any, Any]:
    """Import the audited v4-B adapter and EPMC stack only after release GO."""

    _require_release_sealed()
    from methods.bernini_action_editing import fewshot_motion_branch as motion_branch
    from methods.bernini_action_editing import fewshot_privileged_motion_code as epmc
    from methods.bernini_action_editing import infer_fewshot_motion_code as epmc_runner
    from methods.bernini_action_editing import (
        infer_v4b_epmc_temporal_gate_canary_v1 as v4b_runtime,
    )
    import torch

    if (
        _file_sha256(Path(v4b_runtime.__file__).resolve(strict=True))
        != EXPECTED_V4B_RUNTIME_IMPLEMENTATION_SHA256
    ):
        raise V4EEPMCVideoCanaryError("inherited v4-B canary core source differs")
    return torch, epmc, motion_branch, epmc_runner, v4b_runtime


def _plain_gate_state(path_value: str | Path) -> Path:
    _require_release_sealed()
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise V4EEPMCVideoCanaryError("gate-state must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise V4EEPMCVideoCanaryError("cannot stat gate-state") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size <= _MAX_GATE_STATE_BYTES
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        raise V4EEPMCVideoCanaryError(
            "gate-state must be a bounded mode0444/nlink1 plain file"
        )
    return path.resolve(strict=True)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V4EEPMCVideoCanaryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise V4EEPMCVideoCanaryError(f"non-finite JSON number: {value}")


@dataclass(frozen=True)
class GateStateBundle:
    path: Path
    file_sha256: str
    receipt_digest: str
    codes: Mapping[str, Any]
    payload: Mapping[str, Any]

    @property
    def motion_code_cpu(self) -> Any:
        return self.codes["correct"]

    @property
    def representability_gate(self) -> str:
        # Compatibility property consumed by the inherited EPMC runner only.
        return "GO"

    def audit_receipt(self) -> dict[str, Any]:
        _require_release_sealed()
        return {
            "schema_version": EXPECTED_GATE_STATE_SCHEMA,
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "receipt_digest": self.receipt_digest,
            "iid": EXPECTED_IID,
            "outer_fold": EXPECTED_OUTER_FOLD,
            "v4e_aggregate_gate_verified_true": True,
            "known_exposed_transform_families_only": True,
            "unseen_hostile_transform_gate": False,
            "unseen_hostile_transform_gate_evaluated": False,
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


def _positive_zero(value: Any, *, label: str) -> None:
    _require_release_sealed()
    torch, _, _, _, _ = _runtime_modules()
    flat = value.detach().contiguous().reshape(-1)
    if int(torch.count_nonzero(flat).item()) != 0:
        raise V4EEPMCVideoCanaryError(f"{label} must be exact zero")
    if int(torch.count_nonzero(flat.view(torch.uint8)).item()) != 0:
        raise V4EEPMCVideoCanaryError(f"{label} must be byte-exact positive zero")


def _code_from_json(value: Mapping[str, Any], *, name: str) -> Any:
    gate_materializer = _require_release_sealed()
    torch, epmc, _, _, _ = _runtime_modules()
    if not isinstance(value, Mapping):
        raise V4EEPMCVideoCanaryError(f"{name} gate payload must be an object")
    try:
        phase = torch.tensor(value["phase_gates"], dtype=torch.float32).reshape(1, 21)
        block = torch.tensor(value["block_head_gates"], dtype=torch.float32).reshape(
            1, 16, 12
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise V4EEPMCVideoCanaryError(f"{name} gate tensors differ") from error
    phase = phase.contiguous()
    block = block.contiguous()
    try:
        code = epmc.MotionCode(phase, block)
    except epmc.PrivilegedMotionCodeContractError as error:
        raise V4EEPMCVideoCanaryError(str(error)) from error
    if (
        gate_materializer._tensor_sha256(phase)
        != value.get("phase_gates_sha256")
        or gate_materializer._tensor_sha256(block)
        != value.get("block_head_gates_sha256")
    ):
        raise V4EEPMCVideoCanaryError(f"{name} gate tensor digest differs")
    _positive_zero(block, label=f"{name} block/head gates")
    _positive_zero(phase[:, :1], label=f"{name} phase zero")
    return code


def load_gate_state(
    path_value: str | Path, *, expected_sha256: str
) -> GateStateBundle:
    gate_materializer = _require_release_sealed()
    torch, epmc, _, _, _ = _runtime_modules()
    _required_sha256(expected_sha256, label="expected gate-state SHA256")
    if expected_sha256 != EXPECTED_GATE_STATE_FILE_SHA256:
        raise V4EEPMCVideoCanaryError("gate-state CLI SHA differs from release pin")
    path = _plain_gate_state(path_value)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1
                or not 0 < before.st_size <= _MAX_GATE_STATE_BYTES
            ):
                raise V4EEPMCVideoCanaryError("gate-state inode seal differs")
            raw = handle.read(_MAX_GATE_STATE_BYTES + 1)
            after = os.fstat(handle.fileno())
            named = path.lstat()
            identity_fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if (
                len(raw) != before.st_size
                or hashlib.sha256(raw).hexdigest() != expected_sha256
                or any(
                    getattr(before, field) != getattr(after, field)
                    for field in identity_fields
                )
                or any(
                    getattr(before, field) != getattr(named, field)
                    for field in identity_fields
                )
                or stat.S_IMODE(after.st_mode) != 0o444
                or after.st_nlink != 1
                or stat.S_ISLNK(named.st_mode)
            ):
                raise V4EEPMCVideoCanaryError(
                    "gate-state changed across single-FD read"
                )
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except V4EEPMCVideoCanaryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4EEPMCVideoCanaryError("gate-state is not strict ASCII JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(payload) is not dict:
        raise V4EEPMCVideoCanaryError("gate-state must contain one object")
    digest = _required_sha256(
        payload.get("receipt_digest"), label="gate-state receipt digest"
    )
    unsigned = dict(payload)
    unsigned.pop("receipt_digest", None)
    scope = payload.get("scope")
    arms = payload.get("arms")
    detached = payload.get("detached_media_authority")
    temporal_mapping = payload.get("temporal_mapping")
    decoded_contract = payload.get("decoded_residual_contract")
    calibration = payload.get("fit_only_calibration")
    implementation = payload.get("implementation_binding")
    v4e_receipt = payload.get("v4e_receipt")
    fold_receipt = payload.get("fold_receipt")
    fold_checkpoint = payload.get("fold_checkpoint")
    feature_record = payload.get("feature_record_authority")
    if (
        _object_sha256(unsigned) != digest
        or digest != EXPECTED_GATE_STATE_SELF_DIGEST
        or payload.get("schema_version") != EXPECTED_GATE_STATE_SCHEMA
        or payload.get("status") != EXPECTED_GATE_STATE_STATUS
        or payload.get("iid") != EXPECTED_IID
        or payload.get("outer_fold") != EXPECTED_OUTER_FOLD
        or payload.get("v4e_aggregate_gate_verified_true") is not True
        or not isinstance(implementation, Mapping)
        or implementation.get("gate_materializer_sha256")
        != EXPECTED_GATE_MATERIALIZER_SHA256
        or implementation.get("implementation_sha256")
        != gate_materializer.EXPECTED_V4E_IMPLEMENTATION_SHA256
        or not isinstance(v4e_receipt, Mapping)
        or v4e_receipt.get("file_sha256")
        != gate_materializer.EXPECTED_V4E_RECEIPT_FILE_SHA256
        or v4e_receipt.get("receipt_digest")
        != gate_materializer.EXPECTED_V4E_RECEIPT_SELF_DIGEST
        or v4e_receipt.get("exposed_five_view_codec_development_gate") is not True
        or v4e_receipt.get("known_exposed_transform_families_only") is not True
        or v4e_receipt.get("unseen_hostile_transform_gate") is not False
        or v4e_receipt.get("unseen_hostile_transform_gate_evaluated") is not False
        or not isinstance(fold_receipt, Mapping)
        or fold_receipt.get("file_sha256")
        != gate_materializer.EXPECTED_FOLD1_RECEIPT_SHA256
        or fold_receipt.get("receipt_digest")
        != gate_materializer.EXPECTED_FOLD1_RECEIPT_SELF_DIGEST
        or fold_receipt.get("aggregate_binding_exact") is not True
        or fold_receipt.get("aggregate_gate_evaluated_in_fold_receipt") is not False
        or not isinstance(fold_checkpoint, Mapping)
        or fold_checkpoint.get("outer_fold") != EXPECTED_OUTER_FOLD
        or fold_checkpoint.get("file_sha256")
        != gate_materializer.EXPECTED_FOLD1_CHECKPOINT_SHA256
        or fold_checkpoint.get("metadata_digest")
        != gate_materializer.EXPECTED_FOLD1_CHECKPOINT_METADATA_DIGEST
        or fold_checkpoint.get("model_state_sha256")
        != gate_materializer.EXPECTED_FOLD1_MODEL_STATE_SHA256
        or fold_checkpoint.get("aggregate_artifact_join_exact") is not True
        or fold_checkpoint.get("fold_receipt_artifact_join_exact") is not True
        or fold_checkpoint.get(
            "single_fd_preparse_sha_and_postparse_identity_verified"
        ) is not True
        or fold_checkpoint.get("basis_metadata_state_hash_join_verified") is not True
        or fold_checkpoint.get("model_schema_reconstructed_and_strict_loaded")
        is not True
        or not isinstance(feature_record, Mapping)
        or feature_record.get("original_ordered_contextual_sequence_sha256")
        != gate_materializer.EXPECTED_ORIGINAL_SEQUENCE_SHA256
        or feature_record.get("loaded_original_sequence_sha256_verified") is not True
        or not isinstance(scope, Mapping)
        or scope.get("temporal_gating_diagnostic_only") is not True
        or scope.get("known_exposed_transform_families_only") is not True
        or scope.get("unseen_hostile_transform_gate") is not False
        or scope.get("unseen_hostile_transform_gate_evaluated") is not False
        or scope.get("unseen_action_qualification") is not False
        or scope.get("scientific_claim") is not False
        or scope.get("latent_metric_qualified") is not False
        or scope.get("action_representation_qualified") is not False
        or scope.get("identity_disentanglement_qualified") is not False
        or scope.get("identity_preservation_qualified") is not False
        or scope.get("prior_qualified") is not False
        or scope.get("prior_generation_qualified") is not False
        or scope.get("generation_qualified") is not False
        or scope.get("renderer_qualified") is not False
        or scope.get("video_editing_qualified") is not False
        or scope.get("video_quality_claim") is not False
        or scope.get("bernini_model_execution_performed") is not False
        or scope.get("inference_authorized") is not False
        or scope.get("web_evaluation_authorized") is not False
        or scope.get("full644_refit_authorized") is not False
        or scope.get("heldout_action_anchor_feature_consumed") is not True
        or scope.get("heldout_action_anchor_rgb_consumed") is not False
        or scope.get("target_rgb_consumed") is not False
        or scope.get("source_plus_instruction_only_end_to_end_claim") is not False
        or scope.get("gate_state_is_derived_from_heldout_action_anchor_feature")
        is not True
        or scope.get("vae_necessary") is not None
        or not isinstance(detached, Mapping)
        or detached.get("source_video_sha256") != EXPECTED_SOURCE_SHA256
        or detached.get("anchor_video_sha256") != EXPECTED_ANCHOR_SHA256
        or detached.get("instruction_sha256") != EXPECTED_INSTRUCTION_SHA256
        or detached.get("source_or_anchor_rgb_opened_by_materializer") is not False
        or not isinstance(decoded_contract, Mapping)
        or decoded_contract.get("definition")
        != "R=C(D(E(C(anchor))))-C(D(0))"
        or decoded_contract.get("feature_geometry") != [32, 1024]
        or decoded_contract.get("sole_code_shape") != [12, 32]
        or decoded_contract.get("zero_code_shape") != [1, 12, 32]
        or decoded_contract.get("full_decoded_output_used_as_gate") is not False
        or decoded_contract.get("latent_code_used_directly_as_epmc_gate") is not False
        or not isinstance(calibration, Mapping)
        or calibration.get("outer_fold") != EXPECTED_OUTER_FOLD
        or type(calibration.get("model_fit_count")) is not int
        or calibration.get("model_fit_count", 0) <= 0
        or calibration.get("oof_iid_excluded") != EXPECTED_IID
        or calibration.get("inner_validation_or_oof_values_used_for_scale")
        is not False
        or calibration.get("statistic")
        != "p95 over model-fit IID x 32 of sqrt(mean_d(R**2))"
        or calibration.get("quantile") != 0.95
        or type(calibration.get("p95_value")) not in (int, float)
        or isinstance(calibration.get("p95_value"), bool)
        or not isinstance(temporal_mapping, Mapping)
        or temporal_mapping.get("scale")
        != "divide by fold1 model-fit-only p95 then clamp [0,1]"
        or temporal_mapping.get("interpolation")
        != "torch linear size=20 align_corners=True"
        or temporal_mapping.get("phase0_exact_positive_zero") is not True
        or temporal_mapping.get("all_16x12_block_head_gates_exact_positive_zero")
        is not True
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
        raise V4EEPMCVideoCanaryError(
            "gate-state does not carry the closed v4-E privileged diagnostic scope"
        )

    codes = {
        name: _code_from_json(arms["values"].get(name), name=name)
        for name in gate_materializer.ARM_ORDER
    }
    try:
        profile32 = torch.tensor(
            temporal_mapping["profile32"], dtype=torch.float32
        ).reshape(1, 32).contiguous()
        profile20 = torch.tensor(
            temporal_mapping["profile20"], dtype=torch.float32
        ).reshape(1, 20).contiguous()
        p95 = torch.tensor(
            [calibration["p95_value"]], dtype=torch.float32
        ).contiguous()
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise V4EEPMCVideoCanaryError("temporal calibration tensors differ") from error
    expected_correct_phase = torch.cat(
        (torch.zeros(1, 1, dtype=torch.float32), profile20), dim=1
    ).contiguous()
    if (
        not bool(torch.isfinite(profile32).all())
        or not bool(torch.isfinite(profile20).all())
        or bool((profile32 < 0.0).any())
        or bool((profile32 > 1.0).any())
        or bool((profile20 < 0.0).any())
        or bool((profile20 > 1.0).any())
        or not bool(torch.isfinite(p95).all())
        or float(p95) <= 0.0
        or gate_materializer._tensor_sha256(profile32)
        != temporal_mapping.get("profile32_sha256")
        or gate_materializer._tensor_sha256(profile20)
        != temporal_mapping.get("profile20_sha256")
        or gate_materializer._tensor_sha256(p95)
        != calibration.get("p95_tensor_sha256")
        or not torch.equal(codes["correct"].phase_gates, expected_correct_phase)
    ):
        raise V4EEPMCVideoCanaryError(
            "serialized profile/p95 does not reproduce the correct gate"
        )
    codes["zero"].validate(require_noop=True)
    reference = torch.sort(codes["correct"].phase_gates[:, 1:], dim=1).values
    for name, indices in (
        ("reverse", epmc.REVERSE_PHASE_INDICES),
        ("shuffle", epmc.SHUFFLE_PHASE_INDICES),
    ):
        expected = codes["correct"].phase_gates[:, list(indices)]
        if not torch.equal(codes[name].phase_gates, expected):
            raise V4EEPMCVideoCanaryError(f"{name} is not the frozen phase permutation")
        actual = torch.sort(codes[name].phase_gates[:, 1:], dim=1).values
        if not torch.equal(reference, actual):
            raise V4EEPMCVideoCanaryError(f"{name} changed the phase multiset")
        if torch.equal(codes[name].phase_gates, codes["correct"].phase_gates):
            raise V4EEPMCVideoCanaryError(
                f"{name} is byte-identical to correct; causal control degenerated"
            )
    if int(torch.count_nonzero(codes["correct"].phase_gates[:, 1:]).item()) == 0:
        raise V4EEPMCVideoCanaryError("correct temporal gate degenerated to all zero")
    return GateStateBundle(path, expected_sha256, digest, codes, payload)


def validate_arm_latents(values: Mapping[str, Any]) -> dict[str, bool]:
    _require_release_sealed()
    _, _, _, _, v4b_runtime = _runtime_modules()
    try:
        return v4b_runtime.validate_arm_latents(values)
    except v4b_runtime.V4BEPMCVideoCanaryError as error:
        raise V4EEPMCVideoCanaryError(str(error)) from error


def _save_arm_outputs(**kwargs: Any) -> dict[str, Any]:
    _require_release_sealed()
    _, _, _, _, v4b_runtime = _runtime_modules()
    try:
        return v4b_runtime._save_arm_outputs(**kwargs)
    except v4b_runtime.V4BEPMCVideoCanaryError as error:
        raise V4EEPMCVideoCanaryError(str(error)) from error


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
    arm_codes: Mapping[str, Any],
    carrier_receipt: Mapping[str, Any],
    runtime_traces: Mapping[str, Mapping[str, Any]],
    patch_receipt: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    _require_release_sealed()
    _, epmc, motion_branch, epmc_runner, _ = _runtime_modules()
    gate_scope = gate_bundle.payload.get("scope")
    gate_checkpoint = gate_bundle.payload.get("fold_checkpoint")
    gate_fold_receipt = gate_bundle.payload.get("fold_receipt")
    if (
        motion_branch.OUTER_CPMR_GATE != 0.10
        or set(outputs) != set(ARM_ORDER)
        or arm_comparisons.get("zero_full_latent_byte_exact_b0") is not True
        or set(runtime_traces) != set(PATCHED_ARM_ORDER)
        or not all(
            trace.get("all_bindings_complete") is True
            for trace in runtime_traces.values()
        )
        or gate_bundle.file_sha256 != EXPECTED_GATE_STATE_FILE_SHA256
        or gate_bundle.receipt_digest != EXPECTED_GATE_STATE_SELF_DIGEST
        or gate_bundle.payload.get("v4e_aggregate_gate_verified_true") is not True
        or not isinstance(gate_scope, Mapping)
        or gate_scope.get("known_exposed_transform_families_only") is not True
        or gate_scope.get("unseen_hostile_transform_gate") is not False
        or gate_scope.get("unseen_hostile_transform_gate_evaluated") is not False
        or not isinstance(gate_checkpoint, Mapping)
        or gate_checkpoint.get("aggregate_artifact_join_exact") is not True
        or gate_checkpoint.get("fold_receipt_artifact_join_exact") is not True
        or not isinstance(gate_fold_receipt, Mapping)
        or gate_fold_receipt.get("aggregate_binding_exact") is not True
    ):
        raise V4EEPMCVideoCanaryError("video canary runtime closure differs")
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "method": "v4e-global-codec-decoded-residual-epmc-temporal-head-gating",
        "method_revision": runner_args.method_source_revision,
        "method_archive_sha256": runner_args.method_source_archive_sha256,
        "iid": EXPECTED_IID,
        "outer_fold": EXPECTED_OUTER_FOLD,
        "scientific_claim": False,
        "known_transform_families_exposed_during_model_fit": True,
        "unseen_hostile_transform_gate": False,
        "unseen_hostile_transform_gate_evaluated": False,
        "unseen_action_qualification": False,
        "latent_metric_qualified": False,
        "action_representation_qualified": False,
        "identity_disentanglement_qualified": False,
        "identity_preservation_qualified": False,
        "prior_qualified": False,
        "prior_generation_qualified": False,
        "generation_qualified": False,
        "renderer_qualified": False,
        "video_editing_qualified": False,
        "video_quality_claim": False,
        "inference_authorized": False,
        "web_evaluation_authorized": False,
        "full644_refit_authorized": False,
        "vae_necessary": None,
        "temporal_gating_diagnostic_only": True,
        "v4e_aggregate_gate_verified_true": True,
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
            "location": (
                "real post-varlen-attention projected [12,128] heads before "
                "merge/to_out"
            ),
            "bernini_blocks": list(range(16)),
            "preprojection_channel_chunk_gating": False,
            "block_head_gates_all_exact_positive_zero": True,
            "outer_cpmr_gate": motion_branch.OUTER_CPMR_GATE,
            "epmc_effective_head_gate_nonzero_phase": (
                "0.5*(profile20+0)=0.5*profile20"
            ),
            "total_projected_motion_residual_coefficient": (
                "0.10*0.5*profile20=0.05*profile20"
            ),
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
            "v4e_aggregate_gate_true_before_render": True,
            "aggregate_fold_receipt_checkpoint_strong_join_before_render": True,
            "known_exposed_transform_boundary_preserved": True,
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
        raise V4EEPMCVideoCanaryError("one or more video canary invariants failed")
    payload["receipt_digest"] = _object_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    _require_release_sealed()
    _, _, _, _, v4b_runtime = _runtime_modules()
    return v4b_runtime.build_parser()


def validate_cli(args: argparse.Namespace) -> None:
    _require_release_sealed()
    _, _, _, _, v4b_runtime = _runtime_modules()
    try:
        v4b_runtime.validate_cli(args)
    except v4b_runtime.V4BEPMCVideoCanaryError as error:
        raise V4EEPMCVideoCanaryError(str(error)) from error


def _runner_argv(args: argparse.Namespace) -> list[str]:
    _require_release_sealed()
    _, _, _, _, v4b_runtime = _runtime_modules()
    values = v4b_runtime._runner_argv(args)
    return [value.replace("v4b_epmc", "v4e_epmc") for value in values]


def run(args: argparse.Namespace) -> int:
    _require_release_sealed()
    _, epmc, motion_branch, epmc_runner, _ = _runtime_modules()
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

    def codes_adapter(_unused: Any) -> dict[str, Any]:
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
            raise V4EEPMCVideoCanaryError(
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
    _require_release_sealed()
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "EXPECTED_IID",
    "GateStateBundle",
    "PATCHED_ARM_ORDER",
    "PIN_PLACEHOLDER",
    "PROPOSAL_SEED",
    "RECEIPT_SCHEMA",
    "RELEASE_SEALED",
    "RENDER_SEEDS",
    "V4EEPMCVideoCanaryError",
    "build_parser",
    "build_video_receipt",
    "load_gate_state",
    "main",
    "run",
    "validate_arm_latents",
    "validate_cli",
]
