#!/usr/bin/env python3
"""Materialize one fail-closed v4-D decoded-residual EPMC gate state.

This future-use bridge is intentionally unsealed.  Until every v4-D receipt,
implementation, and selected-fold checkpoint placeholder is replaced by a
reviewed immutable authority and ``RELEASE_SEALED`` is set to ``True``, every
execution entry point raises before it opens an input or creates an output.

After sealing, the only admitted row is IID ``7b88a1ca1f804f41`` in frozen
outer fold 1.  Its ordered V-JEPA2 action-anchor feature is OOF for the fold-1
codec.  The bridge computes

    R = C(D(E(C(anchor)))) - C(D(0))                    [1, 32, 1024]

and reduces it to a scalar temporal profile using feature RMS.  The sole
scale is the p95 of the same statistic over fold-1 *model-fit originals*;
inner-validation and OOF rows are excluded.  The clipped 32-step profile is
linearly interpolated to EPMC's 20 nonzero phases.  Phase zero and all 16x12
block/head gates are byte-exact positive zero.  Reverse and shuffle preserve
the exact correct-profile multiset.

No RGB path is accepted.  This artifact is a privileged OOF temporal-gating
diagnostic, never source+instruction-only inference or renderer/action/video
qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Optional, Sequence


# Kept lazy so an unsealed CLI fails with the release error even on a host
# without the tensor/Bernini environment.  ``from __future__ annotations``
# keeps the type annotations inert until the post-gate runtime is loaded.
torch: Any = None
F: Any = None
epmc: Any = None


SCHEMA = "semantic-anchor-v4d-epmc-temporal-gate-state-v1"
STATUS = "V4D_EPMC_TEMPORAL_GATE_STATE_COMPLETE_DIAGNOSTIC_ONLY"
PIN_PLACEHOLDER = "TO_BE_PINNED"
RELEASE_SEALED = False

# These values deliberately prevent use until the completed v4-D result and
# its exact selected fold-1 checkpoint have been independently audited.
EXPECTED_V4D_RECEIPT_SCHEMA = PIN_PLACEHOLDER
EXPECTED_V4D_RECEIPT_STATUS = PIN_PLACEHOLDER
EXPECTED_V4D_CHECKPOINT_SCHEMA = PIN_PLACEHOLDER
EXPECTED_V4D_IMPLEMENTATION_SHA256 = PIN_PLACEHOLDER
EXPECTED_V4D_RECEIPT_FILE_SHA256 = PIN_PLACEHOLDER
EXPECTED_V4D_RECEIPT_SELF_DIGEST = PIN_PLACEHOLDER
EXPECTED_FOLD1_CHECKPOINT_SHA256 = PIN_PLACEHOLDER
EXPECTED_FOLD1_CHECKPOINT_METADATA_DIGEST = PIN_PLACEHOLDER
EXPECTED_FOLD1_MODEL_STATE_SHA256 = PIN_PLACEHOLDER

EXPECTED_IID = "7b88a1ca1f804f41"
EXPECTED_FAMILY = "sit_down"
EXPECTED_GROUP_ID = (
    "829c459746eea3310f385b834d2848ce580b5d4f79f9f41b0c430a4bb346ca3d"
)
EXPECTED_OUTER_FOLD = 1
EXPECTED_FEATURE_ORDINAL = 465
EXPECTED_FEATURE_SHARD = 3
EXPECTED_SOURCE_VIDEO_SHA256 = (
    "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
)
EXPECTED_ANCHOR_VIDEO_SHA256 = (
    "8234f5f35f7001134cf074263c481e3a8079c10f799370090d30e054aef02015"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
)
EXPECTED_ORIGINAL_SEQUENCE_SHA256 = (
    "678985f2cf0cd0244c707e9ab7f9c6b8116aac5ede6f0a56fe78b92eeb582400"
)

EXPECTED_V4C_IMPLEMENTATION_SHA256 = (
    "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef"
)
EXPECTED_EXTRACTOR_IMPLEMENTATION_SHA256 = (
    "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc"
)
EXPECTED_V4A_IMPLEMENTATION_SHA256 = (
    "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973"
)
EXPECTED_V4A_RECEIPT_SHA256 = (
    "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2"
)
EXPECTED_V4A_RECEIPT_SELF_DIGEST = (
    "f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86"
)
EXPECTED_V4C_FRONTIER_RECEIPT_SHA256 = (
    "8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9"
)
EXPECTED_V4C_FRONTIER_RECEIPT_SELF_DIGEST = (
    "376a98dc74e30ab80a277c8866028677d56ba894073d195612a0edb0bbd74f17"
)
EXPECTED_FEATURE_RECEIPT_SHA256 = (
    "895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a"
)
EXPECTED_FEATURE_RECEIPT_SELF_DIGEST = (
    "774c48838cb0c66948f367ab080ed88514853fcafe8653d1180d17bdb1bac201"
)
EXPECTED_EXACT644_ORDERED_IID_DIGEST = (
    "8e0034fcc4a53c8220390df08e2361080dcd918990629a05c297211dd2bb6637"
)
EXPECTED_OUTER_ASSIGNMENT_DIGEST = (
    "5ab9704f456768b440c966a53328de0c1a67836548f8f8ebd92e50d21846ab5f"
)
EXPECTED_FOLD1_IID_DIGEST = (
    "18c7ad8a24f678ea93cc9d16365fcba0cb8d101667eed9542618240f3ed9c13f"
)

TIME_STEPS = 32
FEATURE_DIM = 1024
CODE_TIME = 4
CODE_CHANNELS = 96
PROFILE_SOURCE_STEPS = 32
PROFILE_TARGET_STEPS = 20
P95_QUANTILE = 0.95
ARM_ORDER = ("zero", "correct", "reverse", "shuffle")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RECEIPT_BYTES = 32 << 20
_MAX_CHECKPOINT_BYTES = 32 << 20


class V4DEPMCGateStateError(RuntimeError):
    """A release authority, codec artifact, or gate ABI differed."""


def _load_tensor_runtime() -> tuple[Any, Any, Any]:
    global F, epmc, torch
    if torch is None or F is None or epmc is None:
        torch_module = importlib.import_module("torch")
        functional_module = importlib.import_module("torch.nn.functional")
        epmc_module = importlib.import_module(
            "methods.bernini_action_editing.fewshot_privileged_motion_code"
        )
        torch, F, epmc = torch_module, functional_module, epmc_module
    return torch, F, epmc


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
        raise V4DEPMCGateStateError("value is not canonical JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    _load_tensor_runtime()
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise V4DEPMCGateStateError("digest input must be a non-meta tensor")
    tensor = value.detach().reshape(-1).repeat(1).cpu()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {"dtype": str(value.dtype), "shape": [int(x) for x in value.shape]}
        )
    )
    digest.update(b"\0")
    digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise V4DEPMCGateStateError(f"{label} must be a lowercase SHA-256")
    return value


def _release_pin_values() -> dict[str, str]:
    return {
        "v4d_receipt_schema": EXPECTED_V4D_RECEIPT_SCHEMA,
        "v4d_receipt_status": EXPECTED_V4D_RECEIPT_STATUS,
        "v4d_checkpoint_schema": EXPECTED_V4D_CHECKPOINT_SCHEMA,
        "v4d_implementation_sha256": EXPECTED_V4D_IMPLEMENTATION_SHA256,
        "v4d_receipt_file_sha256": EXPECTED_V4D_RECEIPT_FILE_SHA256,
        "v4d_receipt_self_digest": EXPECTED_V4D_RECEIPT_SELF_DIGEST,
        "fold1_checkpoint_sha256": EXPECTED_FOLD1_CHECKPOINT_SHA256,
        "fold1_checkpoint_metadata_digest": (
            EXPECTED_FOLD1_CHECKPOINT_METADATA_DIGEST
        ),
        "fold1_model_state_sha256": EXPECTED_FOLD1_MODEL_STATE_SHA256,
    }


def _require_release_sealed() -> None:
    """Fail before parsing paths, importing v4-D, or creating outputs."""

    values = _release_pin_values()
    sha_names = {
        "v4d_implementation_sha256",
        "v4d_receipt_file_sha256",
        "v4d_receipt_self_digest",
        "fold1_checkpoint_sha256",
        "fold1_checkpoint_metadata_digest",
        "fold1_model_state_sha256",
    }
    if (
        RELEASE_SEALED is not True
        or any(value == PIN_PLACEHOLDER for value in values.values())
        or any(_SHA256.fullmatch(values[name]) is None for name in sha_names)
    ):
        raise V4DEPMCGateStateError(
            "UNSEALED v4-D EPMC bridge: all v4-D pins are TO_BE_PINNED"
        )


def _plain_absolute_file(
    value: str | Path, *, label: str, maximum_bytes: int
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise V4DEPMCGateStateError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise V4DEPMCGateStateError(f"cannot stat {label}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise V4DEPMCGateStateError(f"{label} must be a plain regular file")
    if not 0 < info.st_size <= maximum_bytes:
        raise V4DEPMCGateStateError(f"{label} size is outside its frozen bound")
    if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444:
        raise V4DEPMCGateStateError(f"{label} must be mode0444/nlink1")
    return path.resolve(strict=True)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V4DEPMCGateStateError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise V4DEPMCGateStateError(f"non-finite JSON number: {value}")


def _strict_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> dict[str, Any]:
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
                or not 0 < before.st_size <= maximum_bytes
            ):
                raise V4DEPMCGateStateError(f"{label} inode seal differs")
            raw = handle.read(maximum_bytes + 1)
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
                    getattr(before, name) != getattr(after, name)
                    for name in identity_fields
                )
                or any(
                    getattr(before, name) != getattr(named, name)
                    for name in identity_fields
                )
                or stat.S_IMODE(after.st_mode) != 0o444
                or after.st_nlink != 1
                or stat.S_ISLNK(named.st_mode)
            ):
                raise V4DEPMCGateStateError(
                    f"{label} changed across single-FD read"
                )
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except V4DEPMCGateStateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4DEPMCGateStateError(f"{label} is not strict ASCII JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(value) is not dict:
        raise V4DEPMCGateStateError(f"{label} must contain one JSON object")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    digest = _required_sha256(value.get("receipt_digest"), label=f"{label} digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    if _object_sha256(unsigned) != digest:
        raise V4DEPMCGateStateError(f"{label} self-digest differs")
    return digest


def _positive_zero(value: torch.Tensor, *, label: str) -> None:
    _load_tensor_runtime()
    flat = value.detach().contiguous().reshape(-1)
    if int(torch.count_nonzero(flat).item()) != 0:
        raise V4DEPMCGateStateError(f"{label} must be exact zero")
    if int(torch.count_nonzero(flat.view(torch.uint8)).item()) != 0:
        raise V4DEPMCGateStateError(
            f"{label} must be byte-exact positive zero"
        )


def _current_source_binding(v4d: Any, v4c: Any) -> dict[str, str]:
    paths = {
        "implementation": Path(v4d.__file__).resolve(strict=True),
        "v4c_implementation": Path(v4c.__file__).resolve(strict=True),
        "extractor_implementation": Path(v4c.features.__file__).resolve(strict=True),
        "v4a_implementation": Path(v4c.v4a.__file__).resolve(strict=True),
        "gate_materializer": Path(__file__).resolve(strict=True),
    }
    result = {f"{name}_sha256": _file_sha256(path) for name, path in paths.items()}
    if (
        result["implementation_sha256"] != EXPECTED_V4D_IMPLEMENTATION_SHA256
        or result["v4c_implementation_sha256"]
        != EXPECTED_V4C_IMPLEMENTATION_SHA256
        or result["extractor_implementation_sha256"]
        != EXPECTED_EXTRACTOR_IMPLEMENTATION_SHA256
        or result["v4a_implementation_sha256"]
        != EXPECTED_V4A_IMPLEMENTATION_SHA256
    ):
        raise V4DEPMCGateStateError("v4-D bridge source binding differs")
    return result


def _fold_one_contract(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    folds = receipt.get("folds")
    if (
        type(folds) is not list
        or len(folds) != 5
        or any(not isinstance(row, Mapping) for row in folds)
    ):
        raise V4DEPMCGateStateError("v4-D receipt lacks exact-five folds")
    matches = [row for row in folds if row.get("fold_index") == EXPECTED_OUTER_FOLD]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise V4DEPMCGateStateError("v4-D fold-1 receipt is not unique")
    fold = matches[0]
    fit_iids = fold.get("model_fit_iids")
    if (
        type(fit_iids) is not list
        or not fit_iids
        or any(type(iid) is not str or not iid for iid in fit_iids)
        or len(fit_iids) != len(set(fit_iids))
        or EXPECTED_IID in fit_iids
        or fold.get("model_fit_original_count") != len(fit_iids)
        or fold.get("model_fit_iid_digest") != _object_sha256(fit_iids)
    ):
        raise V4DEPMCGateStateError(
            "v4-D fold receipt must embed ordered model-fit IIDs for p95 replay"
        )
    return fold


def validate_v4d_receipt_gate(
    receipt: Mapping[str, Any], *, expected_feature_receipt_sha256: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return fold-1 checkpoint/fold records only after the aggregate true gate."""

    digest = _verify_self_digest(receipt, label="v4-D receipt")
    metrics = receipt.get("metrics")
    scope = receipt.get("qualification_scope")
    authority = receipt.get("feature_authority")
    closure = receipt.get("oof_closure")
    selected = receipt.get("selected_fold_checkpoint_artifacts")
    implementation = receipt.get("implementation")
    frozen_split = receipt.get("frozen_split")
    upstream = receipt.get("upstream_authorities")
    fold_iid_digests = (
        frozen_split.get("fold_iid_digests")
        if isinstance(frozen_split, Mapping)
        else None
    )
    if (
        receipt.get("schema_version") != EXPECTED_V4D_RECEIPT_SCHEMA
        or receipt.get("status") != EXPECTED_V4D_RECEIPT_STATUS
        or digest != EXPECTED_V4D_RECEIPT_SELF_DIGEST
        or not isinstance(implementation, Mapping)
        or implementation.get("implementation_sha256")
        != EXPECTED_V4D_IMPLEMENTATION_SHA256
        or implementation.get("v4c_implementation_sha256")
        != EXPECTED_V4C_IMPLEMENTATION_SHA256
        or implementation.get("extractor_implementation_sha256")
        != EXPECTED_EXTRACTOR_IMPLEMENTATION_SHA256
        or implementation.get("v4a_implementation_sha256")
        != EXPECTED_V4A_IMPLEMENTATION_SHA256
        or not isinstance(metrics, Mapping)
        or metrics.get("decoded_temporal_codec_development_gate") is not True
        or not isinstance(scope, Mapping)
        or scope.get("temporal_codec_development_gate") is not True
        or scope.get("action_representation_qualified") is not False
        or scope.get("identity_preservation_qualified") is not False
        or scope.get("renderer_qualified") is not False
        or scope.get("video_editing_qualified") is not False
        or scope.get("inference_authorized") is not False
        or scope.get("web_evaluation_authorized") is not False
        or scope.get("vae_necessary") is not None
        or not isinstance(authority, Mapping)
        or authority.get("feature_receipt_sha256")
        != expected_feature_receipt_sha256
        or authority.get("feature_receipt_digest")
        != EXPECTED_FEATURE_RECEIPT_SELF_DIGEST
        or authority.get("extractor_implementation_sha256")
        != EXPECTED_EXTRACTOR_IMPLEMENTATION_SHA256
        or authority.get("exact644_ordered_iid_digest")
        != EXPECTED_EXACT644_ORDERED_IID_DIGEST
        or not isinstance(upstream, Mapping)
        or upstream.get("v4a_receipt_file_sha256")
        != EXPECTED_V4A_RECEIPT_SHA256
        or upstream.get("v4a_receipt_self_digest")
        != EXPECTED_V4A_RECEIPT_SELF_DIGEST
        or upstream.get("v4c_frontier_receipt_file_sha256")
        != EXPECTED_V4C_FRONTIER_RECEIPT_SHA256
        or upstream.get("v4c_frontier_receipt_self_digest")
        != EXPECTED_V4C_FRONTIER_RECEIPT_SELF_DIGEST
        or not isinstance(frozen_split, Mapping)
        or frozen_split.get("outer_assignment_digest")
        != EXPECTED_OUTER_ASSIGNMENT_DIGEST
        or not isinstance(fold_iid_digests, Mapping)
        or fold_iid_digests.get(str(EXPECTED_OUTER_FOLD))
        != EXPECTED_FOLD1_IID_DIGEST
        or not isinstance(closure, Mapping)
        or closure.get("unique_original_iids") != 644
        or closure.get("each_original_evaluated_exactly_once") is not True
        or not isinstance(selected, Mapping)
        or selected.get("count") != 5
        or selected.get("fold_selected_step_join_verified") is not True
        or selected.get("all_create_only_mode0444_nlink1") is not True
    ):
        raise V4DEPMCGateStateError(
            "v4-D aggregate decoded-temporal-codec gate is not a closed true gate"
        )
    evidence = closure.get("embedded_per_iid_evidence")
    if (
        type(evidence) is not list
        or len(evidence) != 644
        or any(not isinstance(row, Mapping) for row in evidence)
        or closure.get("embedded_per_iid_evidence_count") != 644
        or closure.get("embedded_per_iid_evidence_sha256")
        != _object_sha256(evidence)
        or closure.get("evidence_sufficient_to_recompute_all_gates") is not True
    ):
        raise V4DEPMCGateStateError("v4-D OOF evidence is not exact644")
    matches = [row for row in evidence if row.get("iid") == EXPECTED_IID]
    if (
        len(matches) != 1
        or matches[0].get("family") != EXPECTED_FAMILY
        or int(matches[0].get("outer_fold", -1)) != EXPECTED_OUTER_FOLD
    ):
        raise V4DEPMCGateStateError("preregistered IID is not fold-1 OOF")
    artifacts = selected.get("artifacts")
    if (
        type(artifacts) is not list
        or len(artifacts) != 5
        or selected.get("artifacts_manifest_sha256") != _object_sha256(artifacts)
        or selected.get("artifacts_reverified_immediately_before_receipt_write")
        is not True
        or selected.get(
            "artifacts_reverified_after_receipt_write_by_command_before_success_return"
        )
        is not True
    ):
        raise V4DEPMCGateStateError("v4-D selected checkpoint manifest differs")
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("mode_octal") != "0444"
            or artifact.get("nlink") != 1
            or artifact.get("selected_training_audit_state_join_verified") is not True
            or artifact.get("fresh_reload_strict_state_verified") is not True
            or artifact.get("fresh_reload_output_bit_exact") is not True
            or artifact.get("caller_model_reloaded_from_sealed_artifact_before_oof")
            is not True
        ):
            raise V4DEPMCGateStateError("v4-D checkpoint artifact audit differs")
    fold1 = [row for row in artifacts if row.get("outer_fold") == EXPECTED_OUTER_FOLD]
    if len(fold1) != 1:
        raise V4DEPMCGateStateError("v4-D fold-1 checkpoint join is not unique")
    fold_contract = _fold_one_contract(receipt)
    return dict(fold1[0]), dict(fold_contract)


def _load_checkpoint(
    path: Path,
    *,
    artifact: Mapping[str, Any],
    v4d: Any,
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor]]:
    _load_tensor_runtime()
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
                or not 0 < before.st_size <= _MAX_CHECKPOINT_BYTES
                or artifact.get("size_bytes") != before.st_size
            ):
                raise V4DEPMCGateStateError(
                    "fold-1 checkpoint inode/size/mode differs"
                )
            digest = hashlib.sha256()
            while block := handle.read(1 << 20):
                digest.update(block)
            actual_sha = digest.hexdigest()
            if (
                actual_sha != EXPECTED_FOLD1_CHECKPOINT_SHA256
                or artifact.get("file_sha256") != actual_sha
            ):
                raise V4DEPMCGateStateError(
                    "fold-1 checkpoint SHA join differs"
                )
            handle.seek(0)
            loaded = torch.load(handle, map_location="cpu", weights_only=True)
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
                any(getattr(before, name) != getattr(after, name) for name in identity_fields)
                or any(getattr(before, name) != getattr(named, name) for name in identity_fields)
                or stat.S_IMODE(after.st_mode) != 0o444
                or after.st_nlink != 1
                or stat.S_ISLNK(named.st_mode)
            ):
                raise V4DEPMCGateStateError(
                    "fold-1 checkpoint changed across single-FD load"
                )
    except V4DEPMCGateStateError:
        raise
    except Exception as error:
        raise V4DEPMCGateStateError(
            "could not safely single-FD load fold-1 checkpoint"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(loaded) is not dict or set(loaded) != {"metadata", "state_dict"}:
        raise V4DEPMCGateStateError("fold-1 checkpoint envelope differs")
    metadata, state = loaded["metadata"], loaded["state_dict"]
    if not isinstance(metadata, Mapping) or type(state) is not dict:
        raise V4DEPMCGateStateError("fold-1 checkpoint payload types differ")
    declared_digest = _required_sha256(
        metadata.get("metadata_digest"), label="checkpoint metadata digest"
    )
    unsigned = dict(metadata)
    unsigned.pop("metadata_digest", None)
    if (
        _object_sha256(unsigned) != declared_digest
        or declared_digest != EXPECTED_FOLD1_CHECKPOINT_METADATA_DIGEST
        or metadata.get("schema_version") != EXPECTED_V4D_CHECKPOINT_SCHEMA
        or metadata.get("outer_fold") != EXPECTED_OUTER_FOLD
        or metadata.get("artifact_scope")
        != "selected burned-development fold codec checkpoint; not refit or authorized inference"
        or metadata.get("refit_artifact") is not False
        or metadata.get("inference_authorized") is not False
        or metadata.get("model_state_sha256")
        != EXPECTED_FOLD1_MODEL_STATE_SHA256
        or metadata.get("model_state_sha256") != v4d._state_sha(state)
        or artifact.get("metadata_digest") != declared_digest
        or artifact.get("model_state_sha256") != metadata.get("model_state_sha256")
        or artifact.get("selected_step") != metadata.get("selected_step")
    ):
        raise V4DEPMCGateStateError("fold-1 checkpoint metadata/state join differs")
    return metadata, state


def _model_from_state(
    metadata: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    *,
    v4d: Any,
) -> Any:
    _load_tensor_runtime()
    required = {"frame_mean", "temporal_basis", "content_basis", "fit_only_rms"}
    if not required.issubset(state):
        raise V4DEPMCGateStateError("fold-1 state lacks frozen analytic buffers")
    basis = metadata.get("basis")
    if not isinstance(basis, Mapping) or (
        basis.get("frame_mean_sha256") != v4d._tensor_sha(state["frame_mean"])
        or basis.get("temporal_basis_sha256")
        != v4d._tensor_sha(state["temporal_basis"])
        or basis.get("content_basis_first96_sha256")
        != v4d._tensor_sha(state["content_basis"])
        or basis.get("fit_only_global_rms_sha256")
        != v4d._tensor_sha(state["fit_only_rms"])
    ):
        raise V4DEPMCGateStateError("checkpoint analytic-basis hashes differ")
    fitted = v4d.TuckerFit(
        frame_mean=state["frame_mean"].detach().clone(),
        temporal_basis=state["temporal_basis"].detach().clone(),
        content_basis=state["content_basis"].detach().clone(),
        fit_iid_digest=str(metadata.get("model_fit_iid_digest")),
        fit_input_sha256=str(basis.get("fixed_tucker_fit_input_sha256")),
        diagnostics={},
    )
    model = v4d.TuckerInitializedVJepaTemporalCodec(
        fitted, state["fit_only_rms"].detach().clone()
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if v4d._state_sha(model.state_dict()) != metadata.get("model_state_sha256"):
        raise V4DEPMCGateStateError("strictly reloaded model state digest differs")
    return model


def decoded_residual(
    model: Any, centered_values: torch.Tensor, zero_decode: torch.Tensor
) -> torch.Tensor:
    """Return ``C(D(E(C(x)))) - C(D(0))`` in V-JEPA feature coordinates."""

    _load_tensor_runtime()
    if (
        centered_values.ndim != 3
        or tuple(centered_values.shape[1:]) != (TIME_STEPS, FEATURE_DIM)
        or tuple(zero_decode.shape) != (1, TIME_STEPS, FEATURE_DIM)
    ):
        raise V4DEPMCGateStateError("decoded-residual input geometry differs")
    with torch.no_grad():
        decoded = model(centered_values)
    if tuple(decoded.shape) != tuple(centered_values.shape):
        raise V4DEPMCGateStateError("decoded codec output geometry differs")
    residual = (decoded - zero_decode).contiguous()
    if not bool(torch.isfinite(residual).all()):
        raise V4DEPMCGateStateError("decoded residual contains NaN or infinity")
    return residual


def residual_rms_profile(residual: torch.Tensor) -> torch.Tensor:
    _load_tensor_runtime()
    if residual.ndim != 3 or tuple(residual.shape[1:]) != (
        PROFILE_SOURCE_STEPS,
        FEATURE_DIM,
    ):
        raise V4DEPMCGateStateError("decoded residual must be [N,32,1024]")
    profile = residual.square().mean(dim=2).sqrt().contiguous()
    if not bool(torch.isfinite(profile).all()):
        raise V4DEPMCGateStateError("decoded-residual RMS is non-finite")
    return profile


def scaled_profile_32_to_20(
    held_residual: torch.Tensor, fit_residuals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the sole model-fit-only p95 scale and frozen interpolation."""

    _load_tensor_runtime()
    held_rms = residual_rms_profile(held_residual)
    fit_rms = residual_rms_profile(fit_residuals)
    if held_rms.shape[0] != 1 or fit_rms.shape[0] < 1:
        raise V4DEPMCGateStateError("held/fit residual population differs")
    p95 = torch.quantile(
        fit_rms.reshape(-1).to(torch.float64),
        P95_QUANTILE,
        interpolation="linear",
    ).to(torch.float32).reshape(1)
    if not bool(torch.isfinite(p95).all()) or float(p95) <= 0.0:
        raise V4DEPMCGateStateError("fit-only decoded-residual p95 is non-positive")
    profile32 = (held_rms / p95).clamp_(0.0, 1.0).to(torch.float32).contiguous()
    profile20 = F.interpolate(
        profile32[:, None, :],
        size=PROFILE_TARGET_STEPS,
        mode="linear",
        align_corners=True,
    )[:, 0].contiguous()
    if (
        tuple(profile32.shape) != (1, PROFILE_SOURCE_STEPS)
        or tuple(profile20.shape) != (1, PROFILE_TARGET_STEPS)
        or bool((profile20 < 0.0).any())
        or bool((profile20 > 1.0).any())
    ):
        raise V4DEPMCGateStateError("scaled temporal gate geometry/range differs")
    return p95, profile32, profile20


def build_motion_codes(profile20: torch.Tensor) -> dict[str, epmc.MotionCode]:
    _load_tensor_runtime()
    if profile20.dtype != torch.float32 or tuple(profile20.shape) != (1, 20):
        raise V4DEPMCGateStateError("correct temporal profile must be FP32 [1,20]")
    phase = torch.cat(
        (torch.zeros(1, 1, dtype=torch.float32), profile20.detach().cpu()), dim=1
    ).contiguous()
    correct = epmc.MotionCode(
        phase, torch.zeros(1, 16, 12, dtype=torch.float32)
    )
    if int(torch.count_nonzero(correct.phase_gates[:, 1:]).item()) == 0:
        raise V4DEPMCGateStateError("correct temporal gate degenerated to all zero")
    codes = {
        "zero": epmc.canonical_noop_motion_code(1, device="cpu"),
        "correct": correct,
        "reverse": epmc.permute_motion_code_phases(
            correct, epmc.REVERSE_PHASE_INDICES
        ),
        "shuffle": epmc.permute_motion_code_phases(
            correct, epmc.SHUFFLE_PHASE_INDICES
        ),
    }
    if tuple(codes) != ARM_ORDER:
        raise V4DEPMCGateStateError("derived arm order differs")
    reference = torch.sort(correct.phase_gates[:, 1:], dim=1).values
    for name in ("reverse", "shuffle"):
        actual = torch.sort(codes[name].phase_gates[:, 1:], dim=1).values
        if not torch.equal(reference, actual):
            raise V4DEPMCGateStateError(f"{name} changed the phase multiset")
        if torch.equal(codes[name].phase_gates, correct.phase_gates):
            raise V4DEPMCGateStateError(
                f"{name} is byte-identical to correct; causal control degenerated"
            )
    for name, code in codes.items():
        _positive_zero(code.block_head_gates, label=f"{name} block/head gates")
        _positive_zero(code.phase_gates[:, :1], label=f"{name} phase zero")
    return codes


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise V4DEPMCGateStateError("output must be a fresh absolute JSON child")
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    info = path.stat()
    digest = hashlib.sha256(payload).hexdigest()
    if (
        stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
        or _file_sha256(path) != digest
    ):
        raise V4DEPMCGateStateError("gate-state seal/readback differs")
    return digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4d-receipt", required=True)
    parser.add_argument("--fold1-checkpoint", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_release_sealed()
    _load_tensor_runtime()

    # Deliberately deferred until after the unsealed release gate.
    from methods.bernini_action_editing import (
        semantic_anchor_vjepa2_analytic_frontier_v4c as v4c,
    )
    from methods.bernini_action_editing import (
        semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d as v4d,
    )

    source_binding = _current_source_binding(v4d, v4c)
    if type(args.batch_size) is not int or not 1 <= args.batch_size <= 128:
        raise V4DEPMCGateStateError("batch-size must be an integer in [1,128]")
    receipt_path = _plain_absolute_file(
        args.v4d_receipt,
        label="v4-D receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    receipt = _strict_json(
        receipt_path,
        label="v4-D receipt",
        expected_sha256=EXPECTED_V4D_RECEIPT_FILE_SHA256,
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    artifact, fold_contract = validate_v4d_receipt_gate(
        receipt, expected_feature_receipt_sha256=EXPECTED_FEATURE_RECEIPT_SHA256
    )
    checkpoint_path = _plain_absolute_file(
        args.fold1_checkpoint,
        label="fold-1 checkpoint",
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
    )
    metadata, state = _load_checkpoint(checkpoint_path, artifact=artifact, v4d=v4d)
    model = _model_from_state(metadata, state, v4d=v4d)

    feature_root = Path(args.feature_root).expanduser()
    records, feature_receipt = v4c.load_v4c_features(
        feature_root, EXPECTED_FEATURE_RECEIPT_SHA256
    )
    if (
        feature_receipt.get("receipt_digest")
        != EXPECTED_FEATURE_RECEIPT_SELF_DIGEST
        or feature_receipt.get("exact644_ordered_iid_digest")
        != EXPECTED_EXACT644_ORDERED_IID_DIGEST
        or len(records) != 644
        or len({row.iid for row in records}) != 644
    ):
        raise V4DEPMCGateStateError("v4-C exact644 feature authority differs")
    by_iid = {row.iid: row for row in records}
    fit_iids = list(fold_contract["model_fit_iids"])
    if (
        any(iid not in by_iid for iid in fit_iids)
        or metadata.get("model_fit_iid_digest") != _object_sha256(fit_iids)
        or EXPECTED_IID not in by_iid
        or by_iid[EXPECTED_IID].family != EXPECTED_FAMILY
        or v4c._tensor_sha(by_iid[EXPECTED_IID].views["original"])
        != EXPECTED_ORIGINAL_SEQUENCE_SHA256
    ):
        raise V4DEPMCGateStateError("fold-1 feature/split/checkpoint join differs")

    zero_code = torch.zeros(1, CODE_TIME, CODE_CHANNELS, dtype=torch.float32)
    with torch.no_grad():
        zero_decode = model.decode(zero_code).detach().cpu().contiguous()
    if tuple(zero_decode.shape) != (1, TIME_STEPS, FEATURE_DIM):
        raise V4DEPMCGateStateError("C(D(0)) geometry differs")

    fit_residual_batches: list[torch.Tensor] = []
    for start in range(0, len(fit_iids), args.batch_size):
        values = torch.stack(
            [
                v4c.canonical_action(by_iid[iid].views["original"])
                for iid in fit_iids[start : start + args.batch_size]
            ]
        )
        fit_residual_batches.append(decoded_residual(model, values, zero_decode))
    fit_residuals = torch.cat(fit_residual_batches, dim=0).contiguous()
    held_value = v4c.canonical_action(
        by_iid[EXPECTED_IID].views["original"]
    )[None]
    held_residual = decoded_residual(model, held_value, zero_decode)
    p95, profile32, profile20 = scaled_profile_32_to_20(
        held_residual, fit_residuals
    )
    codes = build_motion_codes(profile20)

    arm_payload = {
        name: {
            "phase_gates": [float(x) for x in codes[name].phase_gates[0].tolist()],
            "block_head_gates": [
                [float(x) for x in row]
                for row in codes[name].block_head_gates[0].tolist()
            ],
            "phase_gates_sha256": _tensor_sha256(codes[name].phase_gates),
            "block_head_gates_sha256": _tensor_sha256(
                codes[name].block_head_gates
            ),
        }
        for name in ARM_ORDER
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "iid": EXPECTED_IID,
        "family": EXPECTED_FAMILY,
        "group_id": EXPECTED_GROUP_ID,
        "outer_fold": EXPECTED_OUTER_FOLD,
        "feature_record_authority": {
            "ordinal": EXPECTED_FEATURE_ORDINAL,
            "shard": EXPECTED_FEATURE_SHARD,
            "original_ordered_contextual_sequence_sha256": (
                EXPECTED_ORIGINAL_SEQUENCE_SHA256
            ),
            "loaded_original_sequence_sha256_verified": True,
        },
        "detached_media_authority": {
            "source_video_sha256": EXPECTED_SOURCE_VIDEO_SHA256,
            "anchor_video_sha256": EXPECTED_ANCHOR_VIDEO_SHA256,
            "instruction_sha256": EXPECTED_INSTRUCTION_SHA256,
            "source_or_anchor_rgb_opened_by_materializer": False,
            "source_rgb_role": "future Bernini model input and HTML reference",
            "anchor_rgb_role": (
                "detached HTML reference only; never Bernini model input"
            ),
        },
        "v4d_aggregate_gate_verified_true": True,
        "v4d_receipt": {
            "path": str(receipt_path),
            "file_sha256": EXPECTED_V4D_RECEIPT_FILE_SHA256,
            "receipt_digest": receipt["receipt_digest"],
            "decoded_temporal_codec_development_gate": True,
        },
        "implementation_binding": source_binding,
        "fold_checkpoint": {
            "path": str(checkpoint_path),
            "file_sha256": EXPECTED_FOLD1_CHECKPOINT_SHA256,
            "metadata_digest": metadata["metadata_digest"],
            "model_state_sha256": metadata["model_state_sha256"],
            "selected_step": metadata["selected_step"],
            "outer_fold": EXPECTED_OUTER_FOLD,
        },
        "feature_authority": {
            "root": str(feature_root.resolve(strict=True)),
            "receipt_sha256": EXPECTED_FEATURE_RECEIPT_SHA256,
            "receipt_digest": feature_receipt["receipt_digest"],
            "exact644_loaded": True,
        },
        "fit_only_calibration": {
            "outer_fold": EXPECTED_OUTER_FOLD,
            "model_fit_count": len(fit_iids),
            "model_fit_iid_digest": _object_sha256(fit_iids),
            "oof_iid_excluded": EXPECTED_IID,
            "inner_validation_or_oof_values_used_for_scale": False,
            "statistic": "p95 over model-fit IID x 32 of sqrt(mean_d(R**2))",
            "quantile": P95_QUANTILE,
            "p95_value": float(p95.item()),
            "p95_tensor_sha256": _tensor_sha256(p95),
            "fit_decoded_residuals_sha256": _tensor_sha256(fit_residuals),
            "fit_rms_profiles_sha256": _tensor_sha256(
                residual_rms_profile(fit_residuals)
            ),
        },
        "decoded_residual_contract": {
            "definition": "R=C(D(E(C(anchor))))-C(D(0))",
            "feature_geometry": [TIME_STEPS, FEATURE_DIM],
            "sole_code_shape": [CODE_TIME, CODE_CHANNELS],
            "zero_code_shape": [1, CODE_TIME, CODE_CHANNELS],
            "zero_code_sha256": _tensor_sha256(zero_code),
            "c_d_zero_shape": [1, TIME_STEPS, FEATURE_DIM],
            "c_d_zero_sha256": _tensor_sha256(zero_decode),
            "held_residual_shape": [1, TIME_STEPS, FEATURE_DIM],
            "held_residual_sha256": _tensor_sha256(held_residual),
            "full_decoded_output_used_as_gate": False,
            "latent_code_used_directly_as_epmc_gate": False,
        },
        "temporal_mapping": {
            "profile32": [float(x) for x in profile32[0].tolist()],
            "profile32_sha256": _tensor_sha256(profile32),
            "profile20": [float(x) for x in profile20[0].tolist()],
            "profile20_sha256": _tensor_sha256(profile20),
            "scale": "divide by fold1 model-fit-only p95 then clamp [0,1]",
            "interpolation": "torch linear size=20 align_corners=True",
            "phase0_exact_positive_zero": True,
            "all_16x12_block_head_gates_exact_positive_zero": True,
            "epmc_effective_head_gate_nonzero_phase": (
                "0.5*(profile20+0)=0.5*profile20"
            ),
            "downstream_outer_cpmr_gate": 0.10,
            "total_projected_motion_residual_coefficient": (
                "0.10*0.5*profile20=0.05*profile20"
            ),
            "total_coefficient_scale": 0.05,
            "source_and_phase0_total_coefficient": 0.0,
        },
        "arms": {
            "order": list(ARM_ORDER),
            "reverse_and_shuffle_preserve_correct_phase_multiset": True,
            "values": arm_payload,
        },
        "scope": {
            "temporal_gating_diagnostic_only": True,
            "action_representation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "video_quality_claim": False,
            "heldout_action_anchor_feature_consumed": True,
            "heldout_action_anchor_rgb_consumed": False,
            "target_rgb_consumed": False,
            "source_plus_instruction_only_end_to_end_claim": False,
            "gate_state_is_derived_from_heldout_action_anchor_feature": True,
            "bernini_model_execution_performed": False,
            "vae_necessary": None,
        },
    }
    payload["receipt_digest"] = _object_sha256(payload)
    if _current_source_binding(v4d, v4c) != source_binding:
        raise V4DEPMCGateStateError("consumer source binding changed during execution")
    output = Path(args.output).expanduser()
    file_sha = _write_json_create_only(output, payload)
    return {
        "gate_state": str(output.resolve(strict=True)),
        "gate_state_sha256": file_sha,
        "receipt_digest": payload["receipt_digest"],
        "v4d_aggregate_gate_verified_true": True,
        "temporal_gating_diagnostic_only": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _require_release_sealed()
    result = run(build_parser().parse_args(argv))
    print(_canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "EXPECTED_IID",
    "EXPECTED_OUTER_FOLD",
    "PIN_PLACEHOLDER",
    "RELEASE_SEALED",
    "SCHEMA",
    "STATUS",
    "V4DEPMCGateStateError",
    "build_motion_codes",
    "build_parser",
    "decoded_residual",
    "main",
    "residual_rms_profile",
    "run",
    "scaled_profile_32_to_20",
    "validate_v4d_receipt_gate",
]
