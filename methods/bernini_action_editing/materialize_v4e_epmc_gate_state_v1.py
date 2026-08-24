#!/usr/bin/env python3
"""Materialize one fail-closed v4-E decoded-residual EPMC gate state.

This future-use bridge is intentionally unsealed.  Until the v4-E aggregate,
fold-1 receipt, implementation, and selected checkpoint placeholders are all
replaced by reviewed immutable authorities and ``RELEASE_SEALED`` is true,
every execution entry point raises before it opens an input or creates output.

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
qualification.  The v4-E gate covers only the five transform families exposed
during development; it is explicitly not an unseen-hostile-transform result.
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


SCHEMA = "semantic-anchor-v4e-epmc-temporal-gate-state-v1"
STATUS = "V4E_EPMC_TEMPORAL_GATE_STATE_COMPLETE_DIAGNOSTIC_ONLY"
PIN_PLACEHOLDER = "TO_BE_PINNED"
RELEASE_SEALED = False

# These values deliberately prevent use until the completed v4-E result and
# its exact selected fold-1 checkpoint have been independently audited.
EXPECTED_V4E_RECEIPT_SCHEMA = PIN_PLACEHOLDER
EXPECTED_V4E_RECEIPT_STATUS = PIN_PLACEHOLDER
EXPECTED_V4E_FOLD_RECEIPT_SCHEMA = PIN_PLACEHOLDER
EXPECTED_V4E_FOLD_RECEIPT_STATUS = PIN_PLACEHOLDER
EXPECTED_V4E_CHECKPOINT_SCHEMA = PIN_PLACEHOLDER
EXPECTED_V4E_IMPLEMENTATION_SHA256 = PIN_PLACEHOLDER
EXPECTED_V4E_RECEIPT_FILE_SHA256 = PIN_PLACEHOLDER
EXPECTED_V4E_RECEIPT_SELF_DIGEST = PIN_PLACEHOLDER
EXPECTED_FOLD1_RECEIPT_SHA256 = PIN_PLACEHOLDER
EXPECTED_FOLD1_RECEIPT_SELF_DIGEST = PIN_PLACEHOLDER
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
CODE_TIME = 12
CODE_CHANNELS = 32
PROFILE_SOURCE_STEPS = 32
PROFILE_TARGET_STEPS = 20
P95_QUANTILE = 0.95
ARM_ORDER = ("zero", "correct", "reverse", "shuffle")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RECEIPT_BYTES = 32 << 20
_MAX_FOLD_RECEIPT_BYTES = 32 << 20
_MAX_CHECKPOINT_BYTES = 128 << 20
_MAX_GATE_STATE_BYTES = 1 << 20


class V4EEPMCGateStateError(RuntimeError):
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
        raise V4EEPMCGateStateError("value is not canonical JSON") from error


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
        raise V4EEPMCGateStateError("digest input must be a non-meta tensor")
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
        raise V4EEPMCGateStateError(f"{label} must be a lowercase SHA-256")
    return value


def _release_pin_values() -> dict[str, str]:
    return {
        "v4e_receipt_schema": EXPECTED_V4E_RECEIPT_SCHEMA,
        "v4e_receipt_status": EXPECTED_V4E_RECEIPT_STATUS,
        "v4e_fold_receipt_schema": EXPECTED_V4E_FOLD_RECEIPT_SCHEMA,
        "v4e_fold_receipt_status": EXPECTED_V4E_FOLD_RECEIPT_STATUS,
        "v4e_checkpoint_schema": EXPECTED_V4E_CHECKPOINT_SCHEMA,
        "v4e_implementation_sha256": EXPECTED_V4E_IMPLEMENTATION_SHA256,
        "v4e_receipt_file_sha256": EXPECTED_V4E_RECEIPT_FILE_SHA256,
        "v4e_receipt_self_digest": EXPECTED_V4E_RECEIPT_SELF_DIGEST,
        "fold1_receipt_sha256": EXPECTED_FOLD1_RECEIPT_SHA256,
        "fold1_receipt_self_digest": EXPECTED_FOLD1_RECEIPT_SELF_DIGEST,
        "fold1_checkpoint_sha256": EXPECTED_FOLD1_CHECKPOINT_SHA256,
        "fold1_checkpoint_metadata_digest": (
            EXPECTED_FOLD1_CHECKPOINT_METADATA_DIGEST
        ),
        "fold1_model_state_sha256": EXPECTED_FOLD1_MODEL_STATE_SHA256,
    }


def _require_release_sealed() -> None:
    """Fail before parsing paths, importing v4-E, or creating outputs."""

    values = _release_pin_values()
    sha_names = {
        "v4e_implementation_sha256",
        "v4e_receipt_file_sha256",
        "v4e_receipt_self_digest",
        "fold1_receipt_sha256",
        "fold1_receipt_self_digest",
        "fold1_checkpoint_sha256",
        "fold1_checkpoint_metadata_digest",
        "fold1_model_state_sha256",
    }
    if (
        RELEASE_SEALED is not True
        or any(value == PIN_PLACEHOLDER for value in values.values())
        or any(_SHA256.fullmatch(values[name]) is None for name in sha_names)
    ):
        raise V4EEPMCGateStateError(
            "UNSEALED v4-E EPMC bridge: all v4-E pins are TO_BE_PINNED"
        )


def _plain_absolute_file(
    value: str | Path, *, label: str, maximum_bytes: int
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise V4EEPMCGateStateError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise V4EEPMCGateStateError(f"cannot stat {label}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise V4EEPMCGateStateError(f"{label} must be a plain regular file")
    if not 0 < info.st_size <= maximum_bytes:
        raise V4EEPMCGateStateError(f"{label} size is outside its frozen bound")
    if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444:
        raise V4EEPMCGateStateError(f"{label} must be mode0444/nlink1")
    return path.resolve(strict=True)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V4EEPMCGateStateError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise V4EEPMCGateStateError(f"non-finite JSON number: {value}")


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
                raise V4EEPMCGateStateError(f"{label} inode seal differs")
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
                raise V4EEPMCGateStateError(
                    f"{label} changed across single-FD read"
                )
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except V4EEPMCGateStateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4EEPMCGateStateError(f"{label} is not strict ASCII JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(value) is not dict:
        raise V4EEPMCGateStateError(f"{label} must contain one JSON object")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    digest = _required_sha256(value.get("receipt_digest"), label=f"{label} digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    if _object_sha256(unsigned) != digest:
        raise V4EEPMCGateStateError(f"{label} self-digest differs")
    return digest


def _positive_zero(value: torch.Tensor, *, label: str) -> None:
    _load_tensor_runtime()
    flat = value.detach().contiguous().reshape(-1)
    if int(torch.count_nonzero(flat).item()) != 0:
        raise V4EEPMCGateStateError(f"{label} must be exact zero")
    if int(torch.count_nonzero(flat.view(torch.uint8)).item()) != 0:
        raise V4EEPMCGateStateError(
            f"{label} must be byte-exact positive zero"
        )


def _current_source_binding(v4e: Any, v4c: Any) -> dict[str, str]:
    paths = {
        "implementation": Path(v4e.__file__).resolve(strict=True),
        "v4c_implementation": Path(v4c.__file__).resolve(strict=True),
        "extractor_implementation": Path(v4c.features.__file__).resolve(strict=True),
        "v4a_implementation": Path(v4c.v4a.__file__).resolve(strict=True),
        "gate_materializer": Path(__file__).resolve(strict=True),
    }
    result = {f"{name}_sha256": _file_sha256(path) for name, path in paths.items()}
    if (
        result["implementation_sha256"] != EXPECTED_V4E_IMPLEMENTATION_SHA256
        or result["v4c_implementation_sha256"]
        != EXPECTED_V4C_IMPLEMENTATION_SHA256
        or result["extractor_implementation_sha256"]
        != EXPECTED_EXTRACTOR_IMPLEMENTATION_SHA256
        or result["v4a_implementation_sha256"]
        != EXPECTED_V4A_IMPLEMENTATION_SHA256
    ):
        raise V4EEPMCGateStateError("v4-E bridge source binding differs")
    return result


def _ordered_iids(
    value: Any, *, expected_count: Any, expected_digest: Any, label: str
) -> list[str]:
    if (
        type(value) is not list
        or not value
        or any(type(iid) is not str or not iid for iid in value)
        or len(value) != len(set(value))
        or expected_count != len(value)
        or expected_digest != _object_sha256(value)
    ):
        raise V4EEPMCGateStateError(f"{label} ordered IID closure differs")
    return list(value)


def _fold_one_contract(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    folds = receipt.get("folds")
    if (
        type(folds) is not list
        or len(folds) != 5
        or any(not isinstance(row, Mapping) for row in folds)
    ):
        raise V4EEPMCGateStateError("v4-E receipt lacks exact-five folds")
    matches = [row for row in folds if row.get("fold_index") == EXPECTED_OUTER_FOLD]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise V4EEPMCGateStateError("v4-E fold-1 receipt is not unique")
    fold = matches[0]
    fit_iids = _ordered_iids(
        fold.get("model_fit_ordered_iids"),
        expected_count=fold.get("model_fit_original_count"),
        expected_digest=fold.get("model_fit_iid_digest"),
        label="fold-1 model-fit",
    )
    training = fold.get("training")
    if not isinstance(training, Mapping):
        raise V4EEPMCGateStateError("fold-1 training audit is absent")
    inner_iids = _ordered_iids(
        training.get("inner_validation_ordered_iids"),
        expected_count=fold.get("inner_validation_original_count"),
        expected_digest=fold.get("inner_validation_iid_digest"),
        label="fold-1 inner-validation",
    )
    oof_iids = _ordered_iids(
        fold.get("oof_ordered_iids"),
        expected_count=fold.get("oof_original_count"),
        expected_digest=fold.get("oof_iid_digest"),
        label="fold-1 OOF",
    )
    if (
        EXPECTED_IID in fit_iids
        or EXPECTED_IID in inner_iids
        or EXPECTED_IID not in oof_iids
        or set(fit_iids) & set(inner_iids)
        or set(fit_iids) & set(oof_iids)
        or set(inner_iids) & set(oof_iids)
        or training.get("model_fit_ordered_iids") != fit_iids
        or training.get("model_fit_iid_digest") != _object_sha256(fit_iids)
        or training.get("inner_validation_iid_digest")
        != _object_sha256(inner_iids)
    ):
        raise V4EEPMCGateStateError("fold-1 fit/inner/OOF partition differs")
    return fold


def _scope_is_fail_closed(scope: Any, *, aggregate: bool) -> bool:
    if not isinstance(scope, Mapping):
        return False
    false_fields = (
        "latent_metric_qualified",
        "action_representation_qualified",
        "identity_disentanglement_qualified",
        "identity_preservation_qualified",
        "prior_qualified",
        "prior_generation_qualified",
        "generation_qualified",
        "renderer_qualified",
        "video_editing_qualified",
        "inference_authorized",
        "web_evaluation_authorized",
        "full644_refit_authorized",
    )
    if any(scope.get(name) is not False for name in false_fields):
        return False
    if scope.get("vae_necessary") is not None:
        return False
    if aggregate:
        return (
            scope.get("exposed_five_view_codec_development_gate") is True
            and scope.get("unseen_hostile_transform_gate") is False
            and scope.get("unseen_hostile_transform_gate_evaluated") is False
            and scope.get("video_model_training_performed") is False
        )
    return (
        scope.get("exposed_five_view_codec_development_gate") is None
        and scope.get("aggregate_gate_evaluated") is False
    )


def validate_v4e_receipt_gate(
    receipt: Mapping[str, Any], *, expected_feature_receipt_sha256: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return fold-1 checkpoint/fold records only after the aggregate true gate."""

    digest = _verify_self_digest(receipt, label="v4-E receipt")
    metrics = receipt.get("metrics")
    scope = receipt.get("qualification_scope")
    authority = receipt.get("feature_authority")
    closure = receipt.get("oof_closure")
    selected = receipt.get("selected_fold_checkpoint_artifacts")
    implementation = receipt.get("implementation")
    frozen_split = receipt.get("frozen_split")
    upstream = receipt.get("upstream_authorities")
    comparator = receipt.get("fixed_comparator_authority")
    evaluation = receipt.get("evaluation_contract")
    fold_receipts = receipt.get("fold_receipt_artifacts")
    fold_iid_digests = (
        frozen_split.get("fold_iid_digests")
        if isinstance(frozen_split, Mapping)
        else None
    )
    if (
        receipt.get("schema_version") != EXPECTED_V4E_RECEIPT_SCHEMA
        or receipt.get("status") != EXPECTED_V4E_RECEIPT_STATUS
        or digest != EXPECTED_V4E_RECEIPT_SELF_DIGEST
        or not isinstance(implementation, Mapping)
        or implementation.get("implementation_sha256")
        != EXPECTED_V4E_IMPLEMENTATION_SHA256
        or implementation.get("v4c_implementation_sha256")
        != EXPECTED_V4C_IMPLEMENTATION_SHA256
        or implementation.get("extractor_implementation_sha256")
        != EXPECTED_EXTRACTOR_IMPLEMENTATION_SHA256
        or implementation.get("v4a_implementation_sha256")
        != EXPECTED_V4A_IMPLEMENTATION_SHA256
        or not isinstance(metrics, Mapping)
        or metrics.get("exposed_five_view_codec_development_gate") is not True
        or not _scope_is_fail_closed(scope, aggregate=True)
        or not isinstance(authority, Mapping)
        or authority.get("feature_receipt_sha256")
        != expected_feature_receipt_sha256
        or authority.get("feature_receipt_digest")
        != EXPECTED_FEATURE_RECEIPT_SELF_DIGEST
        or authority.get("unique_original_iids") != 644
        or authority.get("family_count") != 28
        or authority.get("stored_views")
        != ["original", "monotone_warp", "reverse", "block_shuffle", "phase_swap"]
        or authority.get("all_five_views_are_separate_frozen_backbone_forwards")
        is not True
        or not isinstance(upstream, Mapping)
        or upstream.get("v4a_receipt_file_sha256")
        != EXPECTED_V4A_RECEIPT_SHA256
        or upstream.get("v4a_receipt_self_digest")
        != EXPECTED_V4A_RECEIPT_SELF_DIGEST
        or upstream.get("v4c_frontier_receipt_file_sha256")
        != EXPECTED_V4C_FRONTIER_RECEIPT_SHA256
        or upstream.get("v4c_frontier_receipt_self_digest")
        != EXPECTED_V4C_FRONTIER_RECEIPT_SELF_DIGEST
        or not isinstance(comparator, Mapping)
        or comparator.get("fixed_comparator_name")
        != "clip_pca_b0384_t01_r384"
        or comparator.get("v4c_burned_oof_informed_clip_pca_b384_choice")
        is not True
        or comparator.get("v4e_oof_used_to_select_comparator") is not False
        or comparator.get("single_v4e_candidate") is not True
        or comparator.get("fold_basis_fit_model_fit_original_only") is not True
        or comparator.get("inner_validation_or_oof_used_for_basis_fit") is not False
        or not isinstance(evaluation, Mapping)
        or evaluation.get("known_exposed_transform_families_only") is not True
        or evaluation.get("unseen_hostile_transform_gate_evaluated") is not False
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
        or selected.get("all_reverified_by_cpu_aggregate") is not True
        or not isinstance(fold_receipts, Mapping)
        or fold_receipts.get("count") != 5
        or fold_receipts.get("all_single_fd_mode0444_nlink1") is not True
    ):
        raise V4EEPMCGateStateError(
            "v4-E aggregate exposed-five-view gate is not a closed true gate"
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
        raise V4EEPMCGateStateError("v4-E OOF evidence is not exact644")
    matches = [row for row in evidence if row.get("iid") == EXPECTED_IID]
    if (
        len(matches) != 1
        or matches[0].get("family") != EXPECTED_FAMILY
        or int(matches[0].get("outer_fold", -1)) != EXPECTED_OUTER_FOLD
    ):
        raise V4EEPMCGateStateError("preregistered IID is not fold-1 OOF")
    fold_contract = _fold_one_contract(receipt)
    fold_evidence = [
        row for row in evidence if row.get("outer_fold") == EXPECTED_OUTER_FOLD
    ]
    if (
        [row.get("iid") for row in fold_evidence]
        != fold_contract.get("oof_ordered_iids")
        or _object_sha256(fold_evidence)
        != fold_contract.get("oof_evaluation_sha256")
    ):
        raise V4EEPMCGateStateError("aggregate fold-1 OOF evidence join differs")
    artifacts = selected.get("artifacts")
    if (
        type(artifacts) is not list
        or len(artifacts) != 5
        or selected.get("artifacts_manifest_sha256") != _object_sha256(artifacts)
    ):
        raise V4EEPMCGateStateError("v4-E selected checkpoint manifest differs")
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
            or artifact.get("single_fd_pre_post_sha256_exact") is not True
            or artifact.get("semantic_metadata_state_replay_verified") is not True
            or artifact.get("basis_metadata_state_hash_join_verified") is not True
            or artifact.get("model_schema_reconstructed_and_strict_loaded") is not True
            or artifact.get("model_forward_executed_by_loader") is not False
        ):
            raise V4EEPMCGateStateError("v4-E checkpoint artifact audit differs")
    fold1 = [row for row in artifacts if row.get("outer_fold") == EXPECTED_OUTER_FOLD]
    if len(fold1) != 1:
        raise V4EEPMCGateStateError("v4-E fold-1 checkpoint join is not unique")
    bindings = fold_receipts.get("bindings")
    if (
        type(bindings) is not list
        or len(bindings) != 5
        or any(not isinstance(row, Mapping) for row in bindings)
    ):
        raise V4EEPMCGateStateError("v4-E fold receipt bindings differ")
    fold1_bindings = [
        row for row in bindings
        if str(row.get("fold_root", "")).rstrip("/").endswith("/fold1")
    ]
    if len(fold1_bindings) != 1:
        # Do not infer by basename alone when aggregate layouts use a different
        # directory convention: join the binding through the checkpoint parent.
        checkpoint_parent = str(Path(str(fold1[0].get("path", ""))).parent)
        fold1_bindings = [
            row for row in bindings if row.get("fold_root") == checkpoint_parent
        ]
    if (
        len(fold1_bindings) != 1
        or fold1_bindings[0].get("mode_octal") != "0444"
        or fold1_bindings[0].get("nlink") != 1
        or fold1_bindings[0].get("single_fd_pre_post_bytes_and_identity_exact")
        is not True
    ):
        raise V4EEPMCGateStateError("v4-E fold-1 receipt strong binding differs")
    return dict(fold1[0]), dict(fold_contract), dict(fold1_bindings[0])


def validate_v4e_fold_receipt(
    receipt: Mapping[str, Any], *, aggregate_fold: Mapping[str, Any],
    aggregate_artifact: Mapping[str, Any], aggregate_binding: Mapping[str, Any],
    receipt_path: Path, checkpoint_path: Path,
) -> None:
    """Strongly join the separately sealed fold-1 receipt to the aggregate."""

    digest = _verify_self_digest(receipt, label="v4-E fold-1 receipt")
    implementation = receipt.get("implementation")
    fold = receipt.get("fold")
    artifact = fold.get("selected_checkpoint_artifact") if isinstance(fold, Mapping) else None
    evidence = receipt.get("oof_evidence")
    authority = receipt.get("feature_authority")
    upstream = receipt.get("upstream_authorities")
    receipt_info = receipt_path.lstat()
    receipt_physical_identity = {
        "device": receipt_info.st_dev,
        "inode": receipt_info.st_ino,
        "size_bytes": receipt_info.st_size,
    }
    if (
        receipt.get("schema_version") != EXPECTED_V4E_FOLD_RECEIPT_SCHEMA
        or receipt.get("status") != EXPECTED_V4E_FOLD_RECEIPT_STATUS
        or digest != EXPECTED_FOLD1_RECEIPT_SELF_DIGEST
        or receipt.get("authority")
        != "burned_exposed_known_transform_development_fold_only"
        or receipt.get("known_transform_families_exposed_during_model_fit") is not True
        or receipt.get("unseen_hostile_transform_gate_evaluated") is not False
        or not _scope_is_fail_closed(receipt.get("qualification_scope"), aggregate=False)
        or not isinstance(implementation, Mapping)
        or implementation.get("implementation_sha256")
        != EXPECTED_V4E_IMPLEMENTATION_SHA256
        or not isinstance(authority, Mapping)
        or authority.get("feature_receipt_sha256") != EXPECTED_FEATURE_RECEIPT_SHA256
        or authority.get("feature_receipt_digest") != EXPECTED_FEATURE_RECEIPT_SELF_DIGEST
        or not isinstance(upstream, Mapping)
        or upstream.get("v4a_receipt_sha256") != EXPECTED_V4A_RECEIPT_SHA256
        or upstream.get("v4c_frontier_receipt_sha256")
        != EXPECTED_V4C_FRONTIER_RECEIPT_SHA256
        or fold != aggregate_fold
        or artifact != aggregate_artifact
        or receipt.get("fold_root") != aggregate_binding.get("fold_root")
        or str(receipt_path) != aggregate_binding.get("path")
        or aggregate_binding.get("file_sha256") != EXPECTED_FOLD1_RECEIPT_SHA256
        or aggregate_binding.get("receipt_digest") != digest
        or aggregate_binding.get("size_bytes") != receipt_info.st_size
        or aggregate_binding.get("physical_identity") != receipt_physical_identity
        or str(checkpoint_path) != artifact.get("path")
    ):
        raise V4EEPMCGateStateError("fold-1 receipt/aggregate/checkpoint join differs")
    _fold_one_contract({"folds": [
        *({"fold_index": index} for index in range(5) if index != EXPECTED_OUTER_FOLD),
        fold,
    ]})
    oof_iids = fold["oof_ordered_iids"]
    if (
        type(evidence) is not list
        or len(evidence) != fold.get("oof_original_count")
        or any(not isinstance(row, Mapping) for row in evidence)
        or receipt.get("oof_evidence_count") != len(evidence)
        or receipt.get("oof_evidence_sha256") != _object_sha256(evidence)
        or fold.get("oof_evaluation_sha256") != _object_sha256(evidence)
        or [row.get("iid") for row in evidence if isinstance(row, Mapping)] != oof_iids
        or any(row.get("outer_fold") != EXPECTED_OUTER_FOLD for row in evidence)
        or sum(row.get("iid") == EXPECTED_IID for row in evidence) != 1
        or next(row for row in evidence if row.get("iid") == EXPECTED_IID).get("family")
        != EXPECTED_FAMILY
    ):
        raise V4EEPMCGateStateError("fold-1 OOF evidence closure differs")


def _load_checkpoint(
    path: Path,
    *,
    artifact: Mapping[str, Any],
    v4e: Any,
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor]]:
    _load_tensor_runtime()
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            physical_identity = {
                "device": before.st_dev,
                "inode": before.st_ino,
                "size_bytes": before.st_size,
            }
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1
                or not 0 < before.st_size <= _MAX_CHECKPOINT_BYTES
                or artifact.get("size_bytes") != before.st_size
                or artifact.get("mode_octal") != "0444"
                or artifact.get("nlink") != 1
                or artifact.get("path") != str(path)
                or artifact.get("outer_fold") != EXPECTED_OUTER_FOLD
                or artifact.get("implementation_sha256")
                != EXPECTED_V4E_IMPLEMENTATION_SHA256
                or artifact.get("physical_identity") != physical_identity
            ):
                raise V4EEPMCGateStateError(
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
                raise V4EEPMCGateStateError(
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
                raise V4EEPMCGateStateError(
                    "fold-1 checkpoint changed across single-FD load"
                )
    except V4EEPMCGateStateError:
        raise
    except Exception as error:
        raise V4EEPMCGateStateError(
            "could not safely single-FD load fold-1 checkpoint"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(loaded) is not dict or set(loaded) != {"metadata", "state_dict"}:
        raise V4EEPMCGateStateError("fold-1 checkpoint envelope differs")
    metadata, state = loaded["metadata"], loaded["state_dict"]
    if not isinstance(metadata, Mapping) or type(state) is not dict:
        raise V4EEPMCGateStateError("fold-1 checkpoint payload types differ")
    declared_digest = _required_sha256(
        metadata.get("metadata_digest"), label="checkpoint metadata digest"
    )
    unsigned = dict(metadata)
    unsigned.pop("metadata_digest", None)
    if (
        _object_sha256(unsigned) != declared_digest
        or declared_digest != EXPECTED_FOLD1_CHECKPOINT_METADATA_DIGEST
        or metadata.get("schema_version") != EXPECTED_V4E_CHECKPOINT_SCHEMA
        or metadata.get("outer_fold") != EXPECTED_OUTER_FOLD
        or metadata.get("artifact_scope")
        != "selected burned-development fold codec checkpoint; not refit or authorized inference"
        or metadata.get("refit_artifact") is not False
        or metadata.get("inference_authorized") is not False
        or metadata.get("model_state_sha256")
        != EXPECTED_FOLD1_MODEL_STATE_SHA256
        or metadata.get("model_state_sha256") != v4e._state_sha(state)
        or artifact.get("metadata_digest") != declared_digest
        or artifact.get("model_state_sha256") != metadata.get("model_state_sha256")
        or artifact.get("selected_step") != metadata.get("selected_step")
        or artifact.get("selected_training_audit_state_join_verified") is not True
        or artifact.get("fresh_reload_strict_state_verified") is not True
        or artifact.get("fresh_reload_output_bit_exact") is not True
        or artifact.get("caller_model_reloaded_from_sealed_artifact_before_oof")
        is not True
        or artifact.get("single_fd_pre_post_sha256_exact") is not True
        or artifact.get("semantic_metadata_state_replay_verified") is not True
        or artifact.get("basis_metadata_state_hash_join_verified") is not True
        or artifact.get("model_schema_reconstructed_and_strict_loaded") is not True
        or artifact.get("model_forward_executed_by_loader") is not False
    ):
        raise V4EEPMCGateStateError("fold-1 checkpoint metadata/state join differs")
    return metadata, state


def _model_from_state(
    metadata: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    *,
    v4e: Any,
) -> Any:
    _load_tensor_runtime()
    required = {"clip_mean", "clip_basis", "fit_only_rms"}
    if not required.issubset(state):
        raise V4EEPMCGateStateError("fold-1 state lacks frozen analytic buffers")
    basis = metadata.get("basis")
    if not isinstance(basis, Mapping) or (
        basis.get("clip_mean_sha256") != v4e._tensor_sha(state["clip_mean"])
        or basis.get("clip_basis_sha256")
        != v4e._tensor_sha(state["clip_basis"])
        or basis.get("fit_only_global_rms_sha256")
        != v4e._tensor_sha(state["fit_only_rms"])
        or tuple(state["clip_mean"].shape) != (1, TIME_STEPS * FEATURE_DIM)
        or tuple(state["clip_basis"].shape)
        != (TIME_STEPS * FEATURE_DIM, CODE_TIME * CODE_CHANNELS)
        or tuple(state["fit_only_rms"].shape) != (1,)
        or any(state[name].dtype != torch.float32 for name in required)
    ):
        raise V4EEPMCGateStateError("checkpoint analytic-basis hashes differ")
    fitted = v4e.ClipPCAFit(
        clip_mean=state["clip_mean"].detach().clone(),
        clip_basis=state["clip_basis"].detach().clone(),
        fit_iid_digest=str(metadata.get("model_fit_iid_digest")),
        fit_input_sha256=str(basis.get("fixed_clip_pca_fit_input_sha256")),
        diagnostics={},
    )
    model = v4e.VJepa2GlobalCodec(
        fitted, state["fit_only_rms"].detach().clone()
    )
    if sum(parameter.numel() for parameter in model.parameters()) != 79040:
        raise V4EEPMCGateStateError("v4-E exact parameter closure differs")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if v4e._state_sha(model.state_dict()) != metadata.get("model_state_sha256"):
        raise V4EEPMCGateStateError("strictly reloaded model state digest differs")
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
        raise V4EEPMCGateStateError("decoded-residual input geometry differs")
    with torch.no_grad():
        decoded = model(centered_values)
    if tuple(decoded.shape) != tuple(centered_values.shape):
        raise V4EEPMCGateStateError("decoded codec output geometry differs")
    residual = (decoded - zero_decode).contiguous()
    if not bool(torch.isfinite(residual).all()):
        raise V4EEPMCGateStateError("decoded residual contains NaN or infinity")
    return residual


def residual_rms_profile(residual: torch.Tensor) -> torch.Tensor:
    _load_tensor_runtime()
    if residual.ndim != 3 or tuple(residual.shape[1:]) != (
        PROFILE_SOURCE_STEPS,
        FEATURE_DIM,
    ):
        raise V4EEPMCGateStateError("decoded residual must be [N,32,1024]")
    profile = residual.square().mean(dim=2).sqrt().contiguous()
    if not bool(torch.isfinite(profile).all()):
        raise V4EEPMCGateStateError("decoded-residual RMS is non-finite")
    return profile


def scaled_profile_32_to_20(
    held_residual: torch.Tensor, fit_residuals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the sole model-fit-only p95 scale and frozen interpolation."""

    _load_tensor_runtime()
    held_rms = residual_rms_profile(held_residual)
    fit_rms = residual_rms_profile(fit_residuals)
    if held_rms.shape[0] != 1 or fit_rms.shape[0] < 1:
        raise V4EEPMCGateStateError("held/fit residual population differs")
    p95 = torch.quantile(
        fit_rms.reshape(-1).to(torch.float64),
        P95_QUANTILE,
        interpolation="linear",
    ).to(torch.float32).reshape(1)
    if not bool(torch.isfinite(p95).all()) or float(p95) <= 0.0:
        raise V4EEPMCGateStateError("fit-only decoded-residual p95 is non-positive")
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
        raise V4EEPMCGateStateError("scaled temporal gate geometry/range differs")
    return p95, profile32, profile20


def build_motion_codes(profile20: torch.Tensor) -> dict[str, epmc.MotionCode]:
    _load_tensor_runtime()
    if profile20.dtype != torch.float32 or tuple(profile20.shape) != (1, 20):
        raise V4EEPMCGateStateError("correct temporal profile must be FP32 [1,20]")
    phase = torch.cat(
        (torch.zeros(1, 1, dtype=torch.float32), profile20.detach().cpu()), dim=1
    ).contiguous()
    correct = epmc.MotionCode(
        phase, torch.zeros(1, 16, 12, dtype=torch.float32)
    )
    if int(torch.count_nonzero(correct.phase_gates[:, 1:]).item()) == 0:
        raise V4EEPMCGateStateError("correct temporal gate degenerated to all zero")
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
        raise V4EEPMCGateStateError("derived arm order differs")
    reference = torch.sort(correct.phase_gates[:, 1:], dim=1).values
    for name in ("reverse", "shuffle"):
        actual = torch.sort(codes[name].phase_gates[:, 1:], dim=1).values
        if not torch.equal(reference, actual):
            raise V4EEPMCGateStateError(f"{name} changed the phase multiset")
        if torch.equal(codes[name].phase_gates, correct.phase_gates):
            raise V4EEPMCGateStateError(
                f"{name} is byte-identical to correct; causal control degenerated"
            )
    for name, code in codes.items():
        _positive_zero(code.block_head_gates, label=f"{name} block/head gates")
        _positive_zero(code.phase_gates[:, :1], label=f"{name} phase zero")
    return codes


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise V4EEPMCGateStateError("output must be a fresh absolute JSON child")
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
        raise V4EEPMCGateStateError("gate-state seal/readback differs")
    return digest


def build_parser() -> argparse.ArgumentParser:
    _require_release_sealed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4e-receipt", required=True)
    parser.add_argument("--fold1-receipt", required=True)
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
        semantic_anchor_vjepa2_multiview_global_codec_v4e_alt as v4e,
    )

    source_binding = _current_source_binding(v4e, v4c)
    if type(args.batch_size) is not int or not 1 <= args.batch_size <= 128:
        raise V4EEPMCGateStateError("batch-size must be an integer in [1,128]")
    receipt_path = _plain_absolute_file(
        args.v4e_receipt,
        label="v4-E receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    receipt = _strict_json(
        receipt_path,
        label="v4-E receipt",
        expected_sha256=EXPECTED_V4E_RECEIPT_FILE_SHA256,
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    artifact, fold_contract, fold_binding = validate_v4e_receipt_gate(
        receipt, expected_feature_receipt_sha256=EXPECTED_FEATURE_RECEIPT_SHA256
    )
    fold_receipt_path = _plain_absolute_file(
        args.fold1_receipt,
        label="fold-1 receipt",
        maximum_bytes=_MAX_FOLD_RECEIPT_BYTES,
    )
    fold_receipt = _strict_json(
        fold_receipt_path,
        label="fold-1 receipt",
        expected_sha256=EXPECTED_FOLD1_RECEIPT_SHA256,
        maximum_bytes=_MAX_FOLD_RECEIPT_BYTES,
    )
    checkpoint_path = _plain_absolute_file(
        args.fold1_checkpoint,
        label="fold-1 checkpoint",
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
    )
    validate_v4e_fold_receipt(
        fold_receipt,
        aggregate_fold=fold_contract,
        aggregate_artifact=artifact,
        aggregate_binding=fold_binding,
        receipt_path=fold_receipt_path,
        checkpoint_path=checkpoint_path,
    )
    metadata, state = _load_checkpoint(checkpoint_path, artifact=artifact, v4e=v4e)
    model = _model_from_state(metadata, state, v4e=v4e)

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
        raise V4EEPMCGateStateError("v4-C exact644 feature authority differs")
    by_iid = {row.iid: row for row in records}
    fit_iids = list(fold_contract["model_fit_ordered_iids"])
    if (
        any(iid not in by_iid for iid in fit_iids)
        or metadata.get("model_fit_iid_digest") != _object_sha256(fit_iids)
        or EXPECTED_IID not in by_iid
        or by_iid[EXPECTED_IID].family != EXPECTED_FAMILY
        or v4c._tensor_sha(by_iid[EXPECTED_IID].views["original"])
        != EXPECTED_ORIGINAL_SEQUENCE_SHA256
    ):
        raise V4EEPMCGateStateError("fold-1 feature/split/checkpoint join differs")

    zero_code = torch.zeros(1, CODE_TIME, CODE_CHANNELS, dtype=torch.float32)
    with torch.no_grad():
        zero_decode = model.decode(zero_code).detach().cpu().contiguous()
    if tuple(zero_decode.shape) != (1, TIME_STEPS, FEATURE_DIM):
        raise V4EEPMCGateStateError("C(D(0)) geometry differs")

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
        "v4e_aggregate_gate_verified_true": True,
        "v4e_receipt": {
            "path": str(receipt_path),
            "file_sha256": EXPECTED_V4E_RECEIPT_FILE_SHA256,
            "receipt_digest": receipt["receipt_digest"],
            "exposed_five_view_codec_development_gate": True,
            "known_exposed_transform_families_only": True,
            "unseen_hostile_transform_gate": False,
            "unseen_hostile_transform_gate_evaluated": False,
        },
        "fold_receipt": {
            "path": str(fold_receipt_path),
            "file_sha256": EXPECTED_FOLD1_RECEIPT_SHA256,
            "receipt_digest": fold_receipt["receipt_digest"],
            "aggregate_binding_exact": True,
            "aggregate_gate_evaluated_in_fold_receipt": False,
        },
        "implementation_binding": source_binding,
        "fold_checkpoint": {
            "path": str(checkpoint_path),
            "file_sha256": EXPECTED_FOLD1_CHECKPOINT_SHA256,
            "metadata_digest": metadata["metadata_digest"],
            "model_state_sha256": metadata["model_state_sha256"],
            "selected_step": metadata["selected_step"],
            "outer_fold": EXPECTED_OUTER_FOLD,
            "aggregate_artifact_join_exact": True,
            "fold_receipt_artifact_join_exact": True,
            "single_fd_preparse_sha_and_postparse_identity_verified": True,
            "basis_metadata_state_hash_join_verified": True,
            "model_schema_reconstructed_and_strict_loaded": True,
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
            "known_exposed_transform_families_only": True,
            "unseen_hostile_transform_gate": False,
            "unseen_hostile_transform_gate_evaluated": False,
            "unseen_action_qualification": False,
            "scientific_claim": False,
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
            "heldout_action_anchor_feature_consumed": True,
            "heldout_action_anchor_rgb_consumed": False,
            "target_rgb_consumed": False,
            "source_plus_instruction_only_end_to_end_claim": False,
            "gate_state_is_derived_from_heldout_action_anchor_feature": True,
            "bernini_model_execution_performed": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "full644_refit_authorized": False,
            "vae_necessary": None,
        },
    }
    payload["receipt_digest"] = _object_sha256(payload)
    if _current_source_binding(v4e, v4c) != source_binding:
        raise V4EEPMCGateStateError("consumer source binding changed during execution")
    output = Path(args.output).expanduser()
    file_sha = _write_json_create_only(output, payload)
    reloaded = _strict_json(
        output.resolve(strict=True),
        label="fresh gate-state",
        expected_sha256=file_sha,
        maximum_bytes=_MAX_GATE_STATE_BYTES,
    )
    if (
        reloaded != payload
        or _verify_self_digest(reloaded, label="fresh gate-state")
        != payload["receipt_digest"]
    ):
        raise V4EEPMCGateStateError("fresh gate-state semantic readback differs")
    return {
        "gate_state": str(output.resolve(strict=True)),
        "gate_state_sha256": file_sha,
        "receipt_digest": payload["receipt_digest"],
        "v4e_aggregate_gate_verified_true": True,
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
    "V4EEPMCGateStateError",
    "build_motion_codes",
    "build_parser",
    "decoded_residual",
    "main",
    "residual_rms_profile",
    "run",
    "scaled_profile_32_to_20",
    "validate_v4e_fold_receipt",
    "validate_v4e_receipt_gate",
]
