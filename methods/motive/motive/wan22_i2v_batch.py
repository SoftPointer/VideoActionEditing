"""Recoverable, provenance-bound Wan2.2 I2V batch generation.

This runner targets the official Wan2.2 repository at ``OFFICIAL_COMMIT`` and
the Hugging Face ``Wan-AI/Wan2.2-I2V-A14B`` payload at
``MODEL_HF_REVISION``.  It deliberately follows the official eight-GPU
execution topology: all eight distributed ranks cooperate on one sample via
DiT FSDP, T5 FSDP, and Ulysses sequence parallelism.  The two Wan experts and
the text encoder are loaded once, then the manifest is processed serially.

The output protocol is fail-closed and resumable:

* the exact input JSONL bytes and every selected row are SHA-256 bound;
* anchor and source-video hashes from the manifest are revalidated;
* a deterministic seed is derived from ``(base_seed, iid)``;
* every committed sample byte-copies its source video and exact UTF-8 edit
  instruction, so the sample does not depend on the original media location;
* a sample becomes visible only after its staging directory is complete and
  is atomically renamed into ``samples/<iid>``;
* ``result.json`` is the per-sample commit marker and binds every output file;
* an existing valid sample is skipped, while an invalid committed directory
  aborts rather than being overwritten; and
* stale hidden staging directories are harmless and are never treated as
  completed samples.

Production authorization requires a source-anchored OpenSSH release verifier.
The legacy path in :mod:`motive.wan22_signed_release` still covers exactly
eight v9 rows.  The independent full-motion path in
:mod:`motive.wan22_full_motion_signed_release` lets one signed root authorize
only byte-bound contiguous eight-row shards.  No manifest row by itself,
including a legacy ``approved_generation`` row, can authorize generation.
Boolean fields and re-signed JSON are not a release signature.  The legacy
``--allow-pending-review`` option is retained only to produce an explicit
error.

For diagnostic rendering only, ``--non-production-preview`` accepts exactly
one deeply validated ``motive-goku-full-motion-generation-v6`` row or one
``motive-goku-full-motion-qwen-v16-passed-v1`` row without a signed release.
Those mutually exclusive lineages have distinct authorization modes and are
deliberately isolated from production: each run contract and output record is
marked ``production_use_forbidden=true`` and binds the input manifest, row,
executable instruction, and its lineage-specific Qwen evidence.

First-frame policy
------------------
Wan internally converts the input image to ``[-1, 1]`` and resizes it with
``torch.nn.functional.interpolate(..., mode="bicubic")``.  On rank zero, the
returned decoded video tensor's frame zero is replaced with the same operation
applied to the same RGB anchor and output size.  That float32 conditioning
frame is saved losslessly as ``conditioning_frame0_float32.npy`` and its
display-space projection is saved as PNG.  The MP4 is H.264 and therefore
lossy: this module never claims that decoding the MP4 reproduces the PNG
pixel-for-pixel.  A training loader that needs strict decoded frame-zero
identity must replace decoded frame zero from the bound PNG/tensor artifact.

Temporal policy
---------------
Wan's ``config.sample_fps`` is a model sampling configuration, not permission
to retime the output container.  Every selected source video must therefore
share one constant temporal grid, its frame count must be supported by Wan
(``4n+1``), and ``--frame-num`` must equal that source frame count.  Generated
frames are encoded at the source frame rate.  A committed target must have the
same frame count and frame rate as its source, and its container duration may
differ by at most one source-frame period.  The model sample FPS and output
container frame rate are recorded under distinct names throughout the run and
sample contracts.

Heavy dependencies are imported only inside :func:`run_batch`, so contract
and orchestration tests can import this module without Torch or Wan installed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid


OFFICIAL_COMMIT = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
MODEL_HF_REVISION = "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
GENERATION_MANIFEST_SCHEMA = "motive-goku-action-anchor-generation-v1"
APPROVAL_SCHEMA = "motive-goku-action-anchor-approval-v1"
APPROVED_MANIFEST_ROLE = "approved_generation"
SIGNED_RELEASE_SCHEMA = "motive-wan22-signed-generation-release-v1"
FULL_MOTION_SIGNED_RELEASE_SCHEMA = (
    "motive-wan22-full-motion-signed-root-release-v3"
)
SIGNED_RELEASE_VERIFIER_AVAILABLE = True
SIGNED_RELEASE_GATE_STATUS = "sshsig_qwen3_vl_32b_smoke_exact_8"
SIGNED_AUTHORIZATION_MODE = "sshsig_qwen3_vl_32b_smoke_release_v1"
FULL_MOTION_SIGNED_AUTHORIZATION_MODE = (
    "sshsig_full_motion_root_contiguous8_release_v3"
)
FULL_MOTION_GENERATION_SCHEMA = "motive-goku-full-motion-generation-v6"
FULL_MOTION_QWEN_V16_PASSED_SCHEMA = (
    "motive-goku-full-motion-qwen-v16-passed-v1"
)
NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE = (
    "unsigned_full_motion_preview_v1"
)
QWEN_V16_NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE = (
    "unsigned_full_motion_qwen_v16_preview_v1"
)
NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODES = frozenset(
    {
        NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE,
        QWEN_V16_NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE,
    }
)
SIGNED_AUTHORIZATION_MODES = frozenset(
    {SIGNED_AUTHORIZATION_MODE, FULL_MOTION_SIGNED_AUTHORIZATION_MODE}
)

RUN_SCHEMA = "motive-wan22-i2v-batch-run-v1"
SAMPLE_SCHEMA = "motive-wan22-i2v-sample-v1"
COMPLETE_SCHEMA = "motive-wan22-i2v-batch-complete-v1"
GENERATED_MANIFEST_SCHEMA = "motive-wan22-i2v-generated-target-v1"
FIRST_FRAME_POLICY = "wan22-i2v-strict-preencode-frame0-v1"
TEMPORAL_POLICY = "wan22-i2v-source-timebase-preserving-v1"

RUN_CONTRACT_NAME = "run_contract.json"
GENERATED_MANIFEST_NAME = "generated_manifest.jsonl"
RUN_COMPLETE_NAME = "run_complete.json"
SAMPLE_RESULT_NAME = "result.json"
SOURCE_VIDEO_ARTIFACT_STEM = "source_video"
EDIT_INSTRUCTION_ARTIFACT_NAME = "edit_instruction.txt"
MOTION_SPEC_ARTIFACT_NAME = "motion_spec.json"

EXPECTED_WORLD_SIZE = 8
DEFAULT_BASE_SEED = 260730
DEFAULT_FRAME_NUM = 81
DEFAULT_SAMPLE_STEPS = 40
DEFAULT_SAMPLE_SHIFT = 5.0
DEFAULT_GUIDE_SCALE = (3.5, 3.5)
DEFAULT_SIZE = "1280*720"
MODEL_SAMPLE_FPS = 16
MAX_DURATION_DELTA_FRAMES = 1
SUPPORTED_MAX_AREAS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
}

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_VIDEO_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,16}\Z")
_VERSION_PREFIX_RE = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")

# Immutable file metadata returned by the Hugging Face API for
# MODEL_HF_REVISION.  The local-dir downloader records the remote ETag in a
# sibling ``.cache/huggingface/download/<path>.metadata`` file.  Checking only
# directory names or config files is unsafe: an interrupted 126 GB download can
# otherwise look superficially complete while both expert weight directories
# are empty.
#
# Values are ``(exact byte size, exact remote ETag)``.  LFS/Xet objects use
# their SHA-256 as the ETag; ordinary Git objects use the Git blob SHA-1.
_PINNED_MODEL_FILE_SPECS: dict[str, tuple[int, str]] = {
    "Wan2.1_VAE.pth": (
        507_609_880,
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    ),
    "models_t5_umt5-xxl-enc-bf16.pth": (
        11_361_920_418,
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
    ),
    "google/umt5-xxl/special_tokens_map.json": (
        6_623,
        "14855e7052ffbb595057dfd791d293c1c940db2c",
    ),
    "google/umt5-xxl/spiece.model": (
        4_548_313,
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
    ),
    "google/umt5-xxl/tokenizer.json": (
        16_837_417,
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
    ),
    "google/umt5-xxl/tokenizer_config.json": (
        61_728,
        "4e1cc1cd85599ce0b47fd0a746af188fe4043ff2",
    ),
    "high_noise_model/config.json": (
        250,
        "17f02f93c97190f0699126f597268a5337aa6f66",
    ),
    "high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors": (
        9_994_119_944,
        "aeea563f9d38ec434b6761c497027ed2843220ee3efaf8c92c815255ade955e3",
    ),
    "high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors": (
        9_943_937_936,
        "fe3b3fcab2b50ff967d971eafbf45e24269c66adacc0881fd6e7f72cd24051e8",
    ),
    "high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors": (
        9_943_979_184,
        "c5d7ddaae0ef24ab452d852a7a53dde7bdad598cd8542e6824e65b0666f39e7f",
    ),
    "high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors": (
        9_839_059_744,
        "4e954c73022c0c8cfc09ebe1ad9aa069571af0c8954c0ed3a6b77c5c3d6b542d",
    ),
    "high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors": (
        9_839_059_744,
        "7c7328fa67ab849427db27145740a3b6531f915ae05f08c0b77c356fd1120be3",
    ),
    "high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors": (
        7_595_559_224,
        "a95d645bfdac3bf13f96d52299fac1a416c7a3bc8741a3ea58e5b7fd0eb3505f",
    ),
    "high_noise_model/diffusion_pytorch_model.safetensors.index.json": (
        96_805,
        "28ab926858c7e124b59b953754ba11284c7b3586",
    ),
    "low_noise_model/config.json": (
        250,
        "17f02f93c97190f0699126f597268a5337aa6f66",
    ),
    "low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors": (
        9_994_119_944,
        "1127e3dea8c08cd746e36d1a7047a3197449adf13d90725ae0a276aeccaf8521",
    ),
    "low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors": (
        9_943_937_936,
        "8250fff242339c31ccb55236e3e1cc25566a2c1d777d1833f3231391fd7d0006",
    ),
    "low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors": (
        9_943_979_184,
        "d0f75d2f41fdab239dbc0624c13ca1c56d79196f2aa0ac57f5f5c68cb53220ea",
    ),
    "low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors": (
        9_839_059_744,
        "aa6119d3ebf5bae827a1fe19dc8e7cba9dbc5f798635a49f80cc0629ed8a74bf",
    ),
    "low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors": (
        9_839_059_744,
        "686cedb30b1696e1ba7034c9287cf1e96471aa997815bb6c183dff9fc7994663",
    ),
    "low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors": (
        7_595_559_224,
        "8b024bda8fb709ba69ec91f1efa5edc19e173d40dcb5b0936050c985167a1be9",
    ),
    "low_noise_model/diffusion_pytorch_model.safetensors.index.json": (
        96_805,
        "28ab926858c7e124b59b953754ba11284c7b3586",
    ),
}

_EXPERT_NAMES = ("high_noise_model", "low_noise_model")
_EXPERT_SHARD_BASENAMES = tuple(
    f"diffusion_pytorch_model-{index:05d}-of-00006.safetensors"
    for index in range(1, 7)
)
_EXPERT_INDEX_BASENAME = "diffusion_pytorch_model.safetensors.index.json"
# AUH stores payloads on NFS while huggingface_hub records metadata timestamps
# from the login host.  Those clocks currently differ by roughly 39 seconds.
# Retain stale-file detection, but allow a bounded cross-host clock skew.
HF_METADATA_MTIME_TOLERANCE_SECONDS = 300.0

_FIRST_FRAME_NOTE = (
    "Frame zero was overridden in the decoded output tensor before encoding "
    "with the official bicubic-resized conditioning RGB tensor. The bound "
    "float32 NPY and PNG are lossless; preview.mp4 uses lossy H.264, so no "
    "pixel-equality claim is made for a frame decoded from the MP4."
)


class Wan22BatchError(RuntimeError):
    """The frozen input, runtime, or committed output violates the contract."""


def require_signed_generation_release() -> None:
    """Fail closed when no verified, manifest-bound release was supplied."""

    raise Wan22BatchError(
        "signed generation release gate is unavailable for unsigned inputs; "
        "a verified release is required, and no current "
        "generation manifest, legacy approved_generation record, approval "
        "JSON, boolean authorization field, or re-signed manifest can "
        "authorize Wan generation (required schema: "
        f"{SIGNED_RELEASE_SCHEMA} or {FULL_MOTION_SIGNED_RELEASE_SCHEMA})"
    )


def _is_signed_authorization(value: Any) -> bool:
    return value in SIGNED_AUTHORIZATION_MODES


def _is_non_production_preview_authorization(value: Any) -> bool:
    return value in NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value: str) -> None:
    raise Wan22BatchError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Wan22BatchError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise Wan22BatchError(f"value is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_json_bytes(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Wan22BatchError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, Wan22BatchError):
            raise
        raise Wan22BatchError(f"{context} is not strict JSON: {error}") from error


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Wan22BatchError(f"{context} must be a regular file: {path}")
    value = _parse_json_bytes(path.read_bytes(), context=context)
    if not isinstance(value, dict):
        raise Wan22BatchError(f"{context} must contain one JSON object")
    return value


def _string(
    value: Any,
    *,
    context: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise Wan22BatchError(f"{context} must be a string")
    if value != value.strip() or "\x00" in value:
        raise Wan22BatchError(f"{context} is not a canonical string")
    if not allow_empty and not value:
        raise Wan22BatchError(f"{context} must not be empty")
    return value


def _sha256_field(value: Any, *, context: str) -> str:
    digest = _string(value, context=context)
    if _SHA256_RE.fullmatch(digest) is None:
        raise Wan22BatchError(f"{context} must be a lowercase SHA-256")
    return digest


def _safe_iid(value: Any, *, context: str) -> str:
    iid = _string(value, context=context)
    if _IID_RE.fullmatch(iid) is None or iid in {".", ".."}:
        raise Wan22BatchError(
            f"{context} is not a safe output path component: {iid!r}"
        )
    return iid


def _string_list(
    value: Any,
    *,
    context: str,
    allow_empty: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise Wan22BatchError(f"{context} must be an array")
    result = [
        _string(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    ]
    if not allow_empty and not result:
        raise Wan22BatchError(f"{context} must not be empty")
    return result


def _regular_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise Wan22BatchError(
            f"{context} must be a regular non-symlink file: {expanded}"
        )
    if expanded.stat().st_size <= 0:
        raise Wan22BatchError(f"{context} is empty: {expanded}")
    return expanded.resolve(strict=True)


def _regular_directory(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise Wan22BatchError(
            f"{context} must be a non-symlink directory: {expanded}"
        )
    return expanded.resolve(strict=True)


def _resolve_manifest_media(
    row: Mapping[str, Any],
    *,
    resolved_key: str,
    fallback_key: str,
    manifest_parent: Path,
    data_root: Path | None,
    context: str,
) -> Path:
    # A present resolved path is authoritative.  We never silently fall back
    # from a stale resolved path to a different file.
    if resolved_key in row:
        raw = _string(row[resolved_key], context=f"{context} {resolved_key}")
    else:
        raw = _string(row.get(fallback_key), context=f"{context} {fallback_key}")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        base = data_root if data_root is not None else manifest_parent
        candidate = base / candidate
    return _regular_file(candidate, context=context)


def _validate_generation_row(
    raw_row: Mapping[str, Any],
    *,
    line_number: int,
    allow_pending_review: bool,
) -> dict[str, Any]:
    context = f"generation manifest row {line_number}"
    if allow_pending_review:
        raise Wan22BatchError(
            "production Wan generation forbids pending-review override; "
            "use an explicitly approved, proposal-bound manifest"
        )
    row = dict(raw_row)
    if row.get("schema_version") != GENERATION_MANIFEST_SCHEMA:
        raise Wan22BatchError(
            f"{context} schema_version must be {GENERATION_MANIFEST_SCHEMA!r}"
        )
    iid = _safe_iid(row.get("iid"), context=f"{context} iid")
    _string(row.get("group_id"), context=f"{context} group_id")
    _string(row.get("action_category"), context=f"{context} action_category")
    _string(
        row.get("target_action_verb"),
        context=f"{context} target_action_verb",
    )
    if row.get("action_change_substantive") != "yes":
        raise Wan22BatchError(
            f"{context} action_change_substantive must be exactly 'yes'"
        )
    _sha256_field(row.get("anchor_sha256"), context=f"{context} anchor_sha256")
    _sha256_field(
        row.get("source_video_sha256"),
        context=f"{context} source_video_sha256",
    )
    if "resolved_anchor_image" not in row and "anchor_image" not in row:
        raise Wan22BatchError(f"{context} has no anchor image path")
    if "resolved_source_video" not in row and "source_video" not in row:
        raise Wan22BatchError(f"{context} has no source video path")
    _string(
        row.get("absolute_target_prompt"),
        context=f"{context} absolute_target_prompt",
    )
    _string(row.get("edit_instruction"), context=f"{context} edit_instruction")
    _string_list(
        row.get("preservation_constraints"),
        context=f"{context} preservation_constraints",
        allow_empty=False,
    )
    _string_list(
        row.get("causal_stages"),
        context=f"{context} causal_stages",
        allow_empty=False,
    )

    authorization = row.get("generation_authorized")
    if type(authorization) is not bool:
        raise Wan22BatchError(
            f"{context} generation_authorized must be a JSON boolean"
        )
    review_status = _string(
        row.get("human_review_status"),
        context=f"{context} human_review_status",
    )
    if (
        authorization is not True
        or review_status != "approved"
        or row.get("manifest_role") != APPROVED_MANIFEST_ROLE
        or row.get("production_eligible") is not True
    ):
        raise Wan22BatchError(
            f"{context} is not explicitly approved for production"
        )
    approval = row.get("approval")
    if not isinstance(approval, Mapping):
        raise Wan22BatchError(
            f"{context} approval must be a closed explicit record"
        )
    required_approval = {
        "schema_version",
        "approval_digest",
        "approval_file_sha256",
        "proposal_sha256",
        "reviewer_id",
        "reviewed_at_utc",
        "decision",
        "reason",
    }
    if set(approval) != required_approval:
        raise Wan22BatchError(
            f"{context} approval is not a closed explicit record"
        )
    if approval.get("schema_version") != APPROVAL_SCHEMA:
        raise Wan22BatchError(
            f"{context} approval schema must be {APPROVAL_SCHEMA!r}"
        )
    for field in (
        "approval_digest",
        "approval_file_sha256",
        "proposal_sha256",
    ):
        _sha256_field(
            approval.get(field),
            context=f"{context} approval {field}",
        )
    for field in ("reviewer_id", "reviewed_at_utc", "reason"):
        _string(
            approval.get(field),
            context=f"{context} approval {field}",
        )
    if approval.get("decision") != "approved":
        raise Wan22BatchError(
            f"{context} approval decision must be exactly 'approved'"
        )
    authorization_mode = "legacy_approval_record_untrusted"

    row["_row_digest"] = _object_digest(raw_row)
    row["_line_number"] = line_number
    row["_authorization_mode"] = authorization_mode
    row["_iid"] = iid
    return row


def validate_generation_manifest_structure(
    manifest_path: str | Path,
    *,
    allow_pending_review: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    """Structurally validate a manifest without granting generation rights.

    This function intentionally performs no media decoding and has only
    standard-library dependencies.  Its return value is audit evidence, never
    authorization; only a future signed-release verifier may grant that.
    """

    if allow_pending_review:
        raise Wan22BatchError(
            "production Wan generation forbids pending-review override"
        )
    manifest = _regular_file(Path(manifest_path), context="generation manifest")
    raw = manifest.read_bytes()
    if not raw.endswith(b"\n"):
        raise Wan22BatchError("generation manifest must end with a newline")
    if not raw:
        raise Wan22BatchError("generation manifest is empty")

    rows: list[dict[str, Any]] = []
    seen_iids: set[str] = set()
    seen_groups: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise Wan22BatchError(
                f"generation manifest contains a blank line at {line_number}"
            )
        value = _parse_json_bytes(
            line,
            context=f"generation manifest {manifest}:{line_number}",
        )
        if not isinstance(value, dict):
            raise Wan22BatchError(
                f"generation manifest row {line_number} is not an object"
            )
        row = _validate_generation_row(
            value,
            line_number=line_number,
            allow_pending_review=allow_pending_review,
        )
        iid = row["_iid"]
        group_id = str(row["group_id"])
        if iid in seen_iids:
            raise Wan22BatchError(f"duplicate manifest iid: {iid}")
        if group_id in seen_groups:
            raise Wan22BatchError(f"duplicate manifest group_id: {group_id}")
        seen_iids.add(iid)
        seen_groups.add(group_id)
        rows.append(row)

    if max_samples is not None:
        if type(max_samples) is not int or max_samples <= 0:
            raise Wan22BatchError("max_samples must be a positive integer")
        if max_samples > len(rows):
            raise Wan22BatchError(
                f"max_samples={max_samples} exceeds manifest rows={len(rows)}"
            )
        selected = rows[:max_samples]
    else:
        selected = rows

    return {
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_bytes(raw),
        "manifest_bytes": len(raw),
        "manifest_row_count": len(rows),
        "selected_rows": selected,
        "selected_row_count": len(selected),
    }


def load_non_production_preview_manifest(
    manifest_path: str | Path,
    *,
    allow_pending_review: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    """Load one deeply validated unsigned full-motion preview row.

    This is not an authorization fallback.  It accepts one and only one row
    from either the full-motion v6 generation lineage or the Qwen v16 passed
    lineage.  The two schemas are validated by their own public validators and
    receive distinct authorization modes.  Both modes are permanently
    forbidden for production use.
    """

    if allow_pending_review:
        raise Wan22BatchError(
            "--allow-pending-review is not part of non-production preview"
        )
    if max_samples not in (None, 1):
        raise Wan22BatchError(
            "non-production preview accepts exactly one manifest row"
        )
    manifest = _regular_file(
        Path(manifest_path), context="non-production preview manifest"
    )
    raw = manifest.read_bytes()
    if not raw.endswith(b"\n"):
        raise Wan22BatchError(
            "non-production preview manifest must end with a newline"
        )
    lines = raw.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise Wan22BatchError(
            "non-production preview requires exactly one non-blank JSONL row"
        )
    value = _parse_json_bytes(
        lines[0], context=f"non-production preview manifest {manifest}:1"
    )
    if not isinstance(value, Mapping):
        raise Wan22BatchError(
            "non-production preview manifest row must be an object"
        )
    schema_version = value.get("schema_version")
    if schema_version not in {
        FULL_MOTION_GENERATION_SCHEMA,
        FULL_MOTION_QWEN_V16_PASSED_SCHEMA,
    }:
        raise Wan22BatchError(
            "non-production preview accepts only one of "
            f"{FULL_MOTION_GENERATION_SCHEMA!r} or "
            f"{FULL_MOTION_QWEN_V16_PASSED_SCHEMA!r}"
        )
    if schema_version == FULL_MOTION_GENERATION_SCHEMA:
        try:
            from motive.goku_full_motion_finalize import validate_generation_row

            validated = validate_generation_row(value)
        except Exception as error:
            raise Wan22BatchError(
                f"full-motion preview row deep validation failed: {error}"
            ) from error
        authorization_mode = NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
    else:
        try:
            from motive.goku_full_motion_qwen_v16 import validate_passed_row

            validated = validate_passed_row(value)
        except Exception as error:
            raise Wan22BatchError(
                f"full-motion Qwen v16 preview row deep validation failed: {error}"
            ) from error
        authorization_mode = QWEN_V16_NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
    if not isinstance(validated, Mapping):
        raise Wan22BatchError(
            "full-motion preview validator returned a non-object"
        )
    row = dict(validated)
    iid = _safe_iid(row.get("iid"), context="full-motion preview iid")
    instruction_sha = _sha256_field(
        row.get("edit_instruction_sha256"),
        context=f"full-motion preview iid={iid} edit_instruction_sha256",
    )
    if instruction_sha != _sha256_bytes(
        _string(
            row.get("edit_instruction"),
            context=f"full-motion preview iid={iid} edit_instruction",
        ).encode("utf-8")
    ):
        raise Wan22BatchError(
            f"full-motion preview instruction binding differs iid={iid}"
        )
    if schema_version == FULL_MOTION_GENERATION_SCHEMA:
        qwen_evidence = row.get("qwen_evidence")
        if not isinstance(qwen_evidence, Mapping):
            raise Wan22BatchError(
                f"full-motion preview Qwen evidence is missing iid={iid}"
            )
        _sha256_field(
            qwen_evidence.get("result_digest"),
            context=f"full-motion preview iid={iid} Qwen result_digest",
        )
        _sha256_field(
            qwen_evidence.get("provenance_digest"),
            context=f"full-motion preview iid={iid} Qwen provenance_digest",
        )
    else:
        _sha256_field(
            row.get("qwen_record_digest"),
            context=f"full-motion Qwen v16 iid={iid} qwen_record_digest",
        )
        if row.get("all_dynamic_subjects_covered") is not True:
            raise Wan22BatchError(
                f"full-motion Qwen v16 dynamic coverage is false iid={iid}"
            )
        if row.get("camera_covered") is not True:
            raise Wan22BatchError(
                f"full-motion Qwen v16 camera coverage is false iid={iid}"
            )
        # Runtime-compatibility fields are deliberately assigned only after
        # the exact v16 passed object has been deeply validated and digested.
        # They do not turn a pending annotation into production authorization.
        row.update(
            {
                "manifest_role": "pending_review",
                "production_eligible": False,
                "human_review_status": "pending",
                "generation_authorized": False,
                "approval": None,
                "action_change_substantive": "yes",
            }
        )
    row.update(
        {
            "_iid": iid,
            "_line_number": 1,
            "_row_digest": _object_digest(value),
            "_authorization_mode": authorization_mode,
            # Display compatibility for the existing generated-target schema;
            # neither value participates in the executable prompt.
            "action_category": "full_motion",
            "target_action_verb": "multi_entity_action_edit",
        }
    )
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_bytes(raw),
        "manifest_bytes": len(raw),
        "manifest_row_count": 1,
        "selected_rows": [row],
        "selected_row_count": 1,
        "non_production_preview": True,
    }


def load_generation_manifest(
    manifest_path: str | Path,
    *,
    allow_pending_review: bool,
    max_samples: int | None,
    signed_release_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load only rows covered by the source-anchored signed release."""

    if allow_pending_review:
        raise Wan22BatchError(
            "signed generation release gate is unavailable for a "
            "pending-review override; the override is never authorization"
        )
    if signed_release_path is None:
        require_signed_generation_release()
    release_path = _regular_file(
        Path(str(signed_release_path)), context="signed generation release"
    )
    envelope = _parse_json_bytes(
        release_path.read_bytes(),
        context=f"signed generation release {release_path}",
    )
    if not isinstance(envelope, Mapping):
        raise Wan22BatchError("signed generation release must be an object")
    if envelope.get("schema_version") == FULL_MOTION_SIGNED_RELEASE_SCHEMA:
        if max_samples not in (None, 8):
            raise Wan22BatchError(
                "a full-motion root release authorizes exactly one contiguous "
                "eight-row shard"
            )
        try:
            from motive.wan22_full_motion_signed_release import (
                verify_signed_release as verify_full_motion_signed_release,
            )

            return verify_full_motion_signed_release(
                release_path=release_path,
                manifest_path=Path(manifest_path),
                verify_media=True,
            )
        except Exception as error:
            if isinstance(error, Wan22BatchError):
                raise
            raise Wan22BatchError(
                f"full-motion signed release verification failed: {error}"
            ) from error
    if max_samples not in (None, 8):
        raise Wan22BatchError(
            "a signed eight-row release cannot be shortened with max_samples"
        )
    try:
        from motive.wan22_signed_release import verify_signed_release

        return verify_signed_release(
            release_path=release_path,
            manifest_path=Path(manifest_path),
            require_exact_manifest=False,
            verify_media=True,
        )
    except Exception as error:
        if isinstance(error, Wan22BatchError):
            raise
        raise Wan22BatchError(f"signed release verification failed: {error}") from error


def sample_seed(base_seed: int, iid: str) -> int:
    """Return an order-independent, non-negative 63-bit per-sample seed."""

    if type(base_seed) is not int or base_seed < 0:
        raise Wan22BatchError("base_seed must be a non-negative integer")
    safe_iid = _safe_iid(iid, context="iid")
    digest = hashlib.sha256(f"{base_seed}\0{safe_iid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = _VERSION_PREFIX_RE.match(value)
    if match is None:
        raise Wan22BatchError(f"cannot parse package version: {value!r}")
    return tuple(int(component or 0) for component in match.groups())


def validate_package_versions(packages: Mapping[str, str]) -> None:
    """Enforce the official Wan2.2 dependency bounds needed by this runner."""

    required = (
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "numpy",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
        "easydict",
        "einops",
        "tqdm",
        "safetensors",
        "tokenizers",
        "flash-attn",
        "ftfy",
        "regex",
        "sentencepiece",
    )
    missing = [name for name in required if not packages.get(name)]
    if missing:
        raise Wan22BatchError(f"missing required Python packages: {missing}")
    if _version_tuple(packages["torch"]) < (2, 4, 0):
        raise Wan22BatchError("official Wan2.2 requires torch>=2.4")
    transformers = _version_tuple(packages["transformers"])
    if transformers < (4, 49, 0) or transformers > (4, 51, 3):
        raise Wan22BatchError(
            "official Wan2.2 requires transformers>=4.49.0,<=4.51.3"
        )
    if _version_tuple(packages["numpy"]) >= (2, 0, 0):
        raise Wan22BatchError("official Wan2.2 requires numpy<2")
    if _version_tuple(packages["torchvision"]) < (0, 19, 0):
        raise Wan22BatchError("official Wan2.2 requires torchvision>=0.19")
    if _version_tuple(packages["diffusers"]) < (0, 31, 0):
        raise Wan22BatchError("official Wan2.2 requires diffusers>=0.31")
    if _version_tuple(packages["accelerate"]) < (1, 1, 1):
        raise Wan22BatchError("official Wan2.2 requires accelerate>=1.1.1")
    if _version_tuple(packages["tokenizers"]) < (0, 20, 3):
        raise Wan22BatchError("official Wan2.2 requires tokenizers>=0.20.3")


def inspect_python_packages() -> dict[str, Any]:
    names = (
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "numpy",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
        "easydict",
        "safetensors",
        "tokenizers",
        "einops",
        "tqdm",
        "ftfy",
        "flash-attn",
        "regex",
        "sentencepiece",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = ""
    validate_package_versions(versions)

    import_smoke = {
        "torchvision": "torchvision",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "accelerate": "accelerate",
        "numpy": "numpy",
        "Pillow": "PIL",
        "imageio": "imageio",
        "imageio-ffmpeg": "imageio_ffmpeg",
        "easydict": "easydict",
        "einops": "einops",
        "tqdm": "tqdm",
        "safetensors": "safetensors",
        "tokenizers": "tokenizers",
        "flash-attn": "flash_attn",
        "ftfy": "ftfy",
        "regex": "regex",
        "sentencepiece": "sentencepiece",
    }
    imported_paths: dict[str, str] = {}
    for distribution, module_name in import_smoke.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            detail = (
                " (flash-attn is mandatory because Wan Ulysses calls "
                "flash_attention directly; SDPA is not a valid fallback)"
                if distribution == "flash-attn"
                else ""
            )
            raise Wan22BatchError(
                f"runtime import smoke failed for {module_name}: "
                f"{type(error).__name__}: {error}{detail}"
            ) from error
        imported_paths[distribution] = str(
            getattr(module, "__file__", "<built-in>")
        )
    return {
        "python": sys.version.split()[0],
        "executable": str(Path(sys.executable).resolve()),
        "packages": versions,
        "import_smoke_paths": imported_paths,
        "attention_backend_capability": "flash_attn_required_for_ulysses",
    }


def inspect_official_checkout(
    code_root: str | Path,
    *,
    expected_commit: str = OFFICIAL_COMMIT,
) -> dict[str, Any]:
    root = _regular_directory(Path(code_root), context="official Wan code root")
    required = (
        root / "generate.py",
        root / "wan" / "image2video.py",
        root / "wan" / "__init__.py",
    )
    for path in required:
        _regular_file(path, context="official Wan source file")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Wan22BatchError(
            f"cannot validate official Wan Git checkout: {error}"
        ) from error
    if head != expected_commit:
        raise Wan22BatchError(
            f"official Wan commit mismatch: expected={expected_commit} actual={head}"
        )
    if status:
        raise Wan22BatchError(
            "official Wan checkout is dirty or has untracked files; use a "
            f"clean checkout at {expected_commit}"
        )
    return {
        "root": str(root),
        "commit": head,
        "clean": True,
        "generate_py_sha256": _sha256_file(root / "generate.py"),
        "image2video_py_sha256": _sha256_file(root / "wan" / "image2video.py"),
    }


def load_official_i2v_modules(code_root: str | Path) -> dict[str, Any]:
    """Load only Wan's I2V package graph, bypassing ``wan.__init__``.

    The official root initializer imports speech/animate stacks and therefore
    pulls unrelated optional dependencies.  A controlled namespace package
    keeps relative I2V imports intact without executing that initializer.
    """

    import importlib
    import importlib.machinery
    import types

    root = _regular_directory(Path(code_root), context="official Wan code root")
    package_dir = _regular_directory(root / "wan", context="official wan package")
    existing = sorted(
        name for name in sys.modules if name == "wan" or name.startswith("wan.")
    )
    if existing:
        raise Wan22BatchError(
            "wan modules were imported before controlled namespace setup: "
            f"{existing[:8]}"
        )
    package = types.ModuleType("wan")
    package.__package__ = "wan"
    package.__path__ = [str(package_dir)]
    package.__file__ = None
    spec = importlib.machinery.ModuleSpec(
        name="wan",
        loader=None,
        is_package=True,
    )
    spec.submodule_search_locations = [str(package_dir)]
    package.__spec__ = spec
    sys.modules["wan"] = package

    names = (
        "wan.image2video",
        "wan.configs",
        "wan.distributed.util",
        "wan.distributed.sequence_parallel",
        "wan.modules.attention",
    )
    modules: dict[str, Any] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            raise Wan22BatchError(
                f"official I2V import failed for {name}: "
                f"{type(error).__name__}: {error}"
            ) from error
        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        if not _path_is_below(module_path, package_dir):
            raise Wan22BatchError(
                f"official I2V module escaped pinned checkout: "
                f"{name} -> {module_path}"
            )
        modules[name] = module
    image2video = modules["wan.image2video"]
    if not hasattr(image2video, "WanI2V"):
        raise Wan22BatchError("pinned wan.image2video has no WanI2V")
    return {
        "WanI2V": image2video.WanI2V,
        "WAN_CONFIGS": modules["wan.configs"].WAN_CONFIGS,
        "MAX_AREA_CONFIGS": modules["wan.configs"].MAX_AREA_CONFIGS,
        "init_distributed_group": modules[
            "wan.distributed.util"
        ].init_distributed_group,
        "attention_module": modules["wan.modules.attention"],
        "module_paths": {
            name: str(Path(module.__file__).resolve())
            for name, module in modules.items()
        },
        "root_initializer_executed": False,
    }


def inspect_hf_model_directory(
    checkpoint_dir: str | Path,
    *,
    expected_revision: str = MODEL_HF_REVISION,
) -> dict[str, Any]:
    root = _regular_directory(Path(checkpoint_dir), context="Wan checkpoint")
    metadata_root = root / ".cache" / "huggingface" / "download"
    if metadata_root.is_symlink() or not metadata_root.is_dir():
        raise Wan22BatchError(
            "checkpoint lacks Hugging Face local-dir metadata needed to "
            f"verify revision {expected_revision}: {metadata_root}"
        )

    verified_files: list[dict[str, Any]] = []
    for relative, (expected_size, expected_etag) in sorted(
        _PINNED_MODEL_FILE_SPECS.items()
    ):
        payload = _regular_file(
            root / Path(*relative.split("/")),
            context=f"pinned Wan payload {relative}",
        )
        if not _path_is_below(payload, root):
            raise Wan22BatchError(
                f"pinned Wan payload escaped checkpoint root: {relative}"
            )
        actual_size = payload.stat().st_size
        if actual_size != expected_size:
            raise Wan22BatchError(
                f"Wan payload size mismatch for {relative}: "
                f"expected={expected_size} actual={actual_size}"
            )
        metadata_path = metadata_root / Path(
            *f"{relative}.metadata".split("/")
        )
        metadata = _regular_file(
            metadata_path,
            context=f"Hugging Face metadata for {relative}",
        )
        if not _path_is_below(metadata, root):
            raise Wan22BatchError(
                f"Hugging Face metadata escaped checkpoint root: {relative}"
            )
        try:
            lines = metadata.read_text(encoding="utf-8").splitlines()
        except UnicodeError as error:
            raise Wan22BatchError(
                f"Hugging Face metadata is not UTF-8 for {relative}"
            ) from error
        if len(lines) != 3:
            raise Wan22BatchError(
                f"Hugging Face metadata must have exactly three lines for "
                f"{relative}: {metadata}"
            )
        revision, etag, timestamp_text = lines
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise Wan22BatchError(
                f"cannot parse Hugging Face commit for {relative}: {revision!r}"
            )
        if revision != expected_revision:
            raise Wan22BatchError(
                f"Hugging Face model revision mismatch for {relative}: "
                f"expected={expected_revision} actual={revision}"
            )
        if etag != expected_etag:
            raise Wan22BatchError(
                f"Hugging Face ETag mismatch for {relative}: "
                f"expected={expected_etag} actual={etag}"
            )
        try:
            timestamp = float(timestamp_text)
        except ValueError as error:
            raise Wan22BatchError(
                f"invalid Hugging Face metadata timestamp for {relative}"
            ) from error
        if not math.isfinite(timestamp) or timestamp <= 0:
            raise Wan22BatchError(
                f"invalid Hugging Face metadata timestamp for {relative}: "
                f"{timestamp_text!r}"
            )
        # Metadata is written after a completed download.  On NFS, the payload
        # mtime comes from the storage server while this timestamp comes from
        # the client, so compare with one explicit bounded clock-skew budget.
        mtime_delta_seconds = payload.stat().st_mtime - timestamp
        if mtime_delta_seconds > HF_METADATA_MTIME_TOLERANCE_SECONDS:
            raise Wan22BatchError(
                f"stale Hugging Face metadata for {relative}: payload mtime "
                f"is newer by {mtime_delta_seconds:.3f}s, exceeding the "
                f"{HF_METADATA_MTIME_TOLERANCE_SECONDS:.1f}s NFS clock-skew "
                "allowance"
            )
        verified_files.append(
            {
                "path": relative,
                "bytes": actual_size,
                "etag": etag,
                "metadata_path": str(metadata.relative_to(root)),
                "metadata_timestamp": timestamp,
                "payload_mtime_delta_seconds": mtime_delta_seconds,
            }
        )

    expert_indexes: dict[str, Any] = {}
    expected_shards = set(_EXPERT_SHARD_BASENAMES)
    for expert in _EXPERT_NAMES:
        expert_root = _regular_directory(
            root / expert,
            context=f"{expert} checkpoint directory",
        )
        index_path = expert_root / _EXPERT_INDEX_BASENAME
        index = _load_json(
            index_path,
            context=f"{expert} safetensors index",
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Wan22BatchError(
                f"{expert} safetensors index has no non-empty weight_map"
            )
        referenced: set[str] = set()
        for parameter, shard in weight_map.items():
            _string(parameter, context=f"{expert} weight_map parameter")
            shard_name = _string(
                shard,
                context=f"{expert} weight_map shard for {parameter}",
            )
            if Path(shard_name).name != shard_name:
                raise Wan22BatchError(
                    f"{expert} index shard must be one basename: {shard_name!r}"
                )
            referenced.add(shard_name)
        if referenced != expected_shards:
            raise Wan22BatchError(
                f"{expert} index shard closure mismatch: "
                f"expected={sorted(expected_shards)} "
                f"actual={sorted(referenced)}"
            )
        on_disk = {
            path.name
            for path in expert_root.glob("*.safetensors")
            if path.exists()
        }
        if on_disk != expected_shards:
            raise Wan22BatchError(
                f"{expert} safetensors payload set mismatch: "
                f"expected={sorted(expected_shards)} actual={sorted(on_disk)}"
            )
        index_metadata = index.get("metadata")
        if not isinstance(index_metadata, Mapping):
            raise Wan22BatchError(f"{expert} index metadata is missing")
        total_size = index_metadata.get("total_size")
        if type(total_size) is not int or total_size <= 0:
            raise Wan22BatchError(
                f"{expert} index metadata total_size must be positive"
            )
        expert_indexes[expert] = {
            "index": f"{expert}/{_EXPERT_INDEX_BASENAME}",
            "weight_count": len(weight_map),
            "referenced_shards": sorted(referenced),
            "declared_tensor_bytes": total_size,
        }

    incomplete = sorted(root.rglob("*.incomplete"))
    if incomplete:
        raise Wan22BatchError(
            f"checkpoint has incomplete downloads: {[str(path) for path in incomplete]}"
        )
    return {
        "model_id": MODEL_ID,
        "checkpoint_dir": str(root),
        "hf_revision": expected_revision,
        "revision_binding": (
            "exact_size_and_etag_per_file_via_huggingface_local_dir_metadata"
        ),
        "metadata_file_count": len(verified_files),
        "required_payloads": [
            item["path"] for item in verified_files
        ],
        "verified_payload_bytes": sum(
            item["bytes"] for item in verified_files
        ),
        "verified_files": verified_files,
        "expert_indexes": expert_indexes,
        "incomplete_file_count": 0,
    }


def _fraction_to_float(value: Any, *, context: str) -> float:
    text = _string(value, context=context)
    try:
        fraction = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise Wan22BatchError(f"invalid rational {context}: {text!r}") from error
    result = float(fraction)
    if not math.isfinite(result) or result <= 0:
        raise Wan22BatchError(f"{context} must be positive")
    return result


def _positive_fraction(value: Any, *, context: str) -> Fraction:
    text = _string(value, context=context)
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise Wan22BatchError(f"invalid rational {context}: {text!r}") from error
    if result <= 0:
        raise Wan22BatchError(f"{context} must be positive")
    return result


def _expected_positive_fraction(value: Any, *, context: str) -> Fraction:
    if isinstance(value, Fraction):
        result = value
    elif type(value) is int:
        result = Fraction(value, 1)
    elif isinstance(value, str):
        result = _positive_fraction(value, context=context)
    else:
        raise Wan22BatchError(
            f"{context} must be an integer or rational string"
        )
    if result <= 0:
        raise Wan22BatchError(f"{context} must be positive")
    return result


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _codec_family(value: str) -> str:
    normalized = value.casefold()
    families = {
        "libx264": "h264",
        "h264": "h264",
        "libopenh264": "h264",
        "libx265": "hevc",
        "hevc": "hevc",
        "h265": "hevc",
        "libvpx-vp9": "vp9",
        "vp9": "vp9",
        "libaom-av1": "av1",
        "av1": "av1",
    }
    return families.get(normalized, normalized)


def _integerish(value: Any, *, context: str, allow_unknown: bool = False) -> int:
    if value in (None, "", "N/A"):
        if allow_unknown:
            return 0
        raise Wan22BatchError(f"{context} is missing")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise Wan22BatchError(f"{context} is not an integer: {value!r}") from error
    if result < 0:
        raise Wan22BatchError(f"{context} is negative")
    return result


def normalize_ffprobe_payload(
    payload: Mapping[str, Any],
    *,
    expected_frames: int | None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_fps: int | str | Fraction | None = None,
    expected_codec: str | None = None,
    max_nominal_duration_error_frames: int | None = None,
) -> dict[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise Wan22BatchError("ffprobe must return exactly one video stream")
    stream = streams[0]
    if not isinstance(stream, Mapping):
        raise Wan22BatchError("ffprobe video stream is not an object")
    width = _integerish(stream.get("width"), context="ffprobe width")
    height = _integerish(stream.get("height"), context="ffprobe height")
    if width <= 0 or height <= 0:
        raise Wan22BatchError("ffprobe returned an empty video size")
    if expected_width is not None and width != expected_width:
        raise Wan22BatchError(
            f"ffprobe width mismatch: expected={expected_width} actual={width}"
        )
    if expected_height is not None and height != expected_height:
        raise Wan22BatchError(
            f"ffprobe height mismatch: expected={expected_height} actual={height}"
        )
    expected_rate = (
        _expected_positive_fraction(
            expected_fps,
            context="expected frame rate",
        )
        if expected_fps is not None
        else None
    )
    frame_rate_fields: dict[str, str] = {}
    parsed_frame_rates: dict[str, Fraction] = {}
    for field in ("avg_frame_rate", "r_frame_rate"):
        if stream.get(field) not in (None, ""):
            rate_text = _string(
                stream[field],
                context=f"ffprobe {field}",
            )
            rate = _positive_fraction(
                rate_text,
                context=f"ffprobe {field}",
            )
            if expected_rate is not None and rate != expected_rate:
                raise Wan22BatchError(
                    f"ffprobe {field} mismatch: "
                    f"expected={_fraction_text(expected_rate)} "
                    f"actual={rate_text}"
                )
            frame_rate_fields[field] = rate_text
            parsed_frame_rates[field] = rate
    if not frame_rate_fields:
        raise Wan22BatchError("ffprobe frame rate is missing")
    if len(set(parsed_frame_rates.values())) != 1:
        raise Wan22BatchError(
            "ffprobe reports an incompatible variable frame-rate grid: "
            f"{frame_rate_fields}"
        )
    frame_rate_fraction = next(iter(parsed_frame_rates.values()))
    frame_rate = _fraction_text(frame_rate_fraction)
    frames = _integerish(
        stream.get("nb_read_frames") or stream.get("nb_frames"),
        context="ffprobe frame count",
        allow_unknown=expected_frames is None,
    )
    if expected_frames is not None and frames != expected_frames:
        raise Wan22BatchError(
            f"ffprobe frame mismatch: expected={expected_frames} actual={frames}"
        )
    format_payload = payload.get("format")
    if not isinstance(format_payload, Mapping):
        raise Wan22BatchError("ffprobe format payload is missing")
    try:
        duration = float(format_payload.get("duration"))
    except (TypeError, ValueError) as error:
        raise Wan22BatchError("ffprobe duration is invalid") from error
    if not math.isfinite(duration) or duration <= 0:
        raise Wan22BatchError("ffprobe duration must be positive")
    nominal_duration: float | None = None
    nominal_duration_error: float | None = None
    nominal_duration_error_frames: float | None = None
    if frames > 0:
        nominal_duration = float(Fraction(frames, 1) / frame_rate_fraction)
        nominal_duration_error = abs(duration - nominal_duration)
        nominal_duration_error_frames = (
            nominal_duration_error * float(frame_rate_fraction)
        )
    if max_nominal_duration_error_frames is not None:
        if (
            type(max_nominal_duration_error_frames) is not int
            or max_nominal_duration_error_frames < 0
        ):
            raise Wan22BatchError(
                "max_nominal_duration_error_frames must be a non-negative "
                "integer"
            )
        if nominal_duration_error_frames is None:
            raise Wan22BatchError(
                "ffprobe frame count is required for duration validation"
            )
        if (
            nominal_duration_error_frames
            > max_nominal_duration_error_frames + 1e-6
        ):
            raise Wan22BatchError(
                "ffprobe duration differs from its constant frame grid by "
                f"{nominal_duration_error_frames:.6f} frames; "
                f"maximum={max_nominal_duration_error_frames}"
            )
    codec = _string(
        stream.get("codec_name") or "unknown",
        context="ffprobe codec",
    )
    if (
        expected_codec is not None
        and _codec_family(codec) != _codec_family(expected_codec)
    ):
        raise Wan22BatchError(
            f"ffprobe codec mismatch: configured={expected_codec!r} "
            f"expected_family={_codec_family(expected_codec)!r} "
            f"actual={codec!r}"
        )
    return {
        "probe_backend": "ffprobe",
        "codec": codec,
        "configured_codec": expected_codec,
        "codec_family": _codec_family(codec),
        "pixel_format": _string(
            stream.get("pix_fmt") or "unknown",
            context="ffprobe pixel format",
        ),
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "frame_rate_fields": frame_rate_fields,
        "frames": frames,
        "duration_seconds": duration,
        "nominal_duration_seconds": nominal_duration,
        "nominal_duration_error_seconds": nominal_duration_error,
        "nominal_duration_error_frames": nominal_duration_error_frames,
        "container": _string(
            format_payload.get("format_name") or "unknown",
            context="ffprobe container",
        ),
        "container_bytes": _integerish(
            format_payload.get("size"),
            context="ffprobe container size",
            allow_unknown=True,
        ),
    }


def probe_video(
    path: Path,
    *,
    ffprobe: str,
    expected_frames: int | None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_fps: int | str | Fraction | None = None,
    expected_codec: str | None = None,
    max_nominal_duration_error_frames: int | None = None,
) -> dict[str, Any]:
    executable = shutil.which(ffprobe)
    if executable is None:
        raise Wan22BatchError(f"ffprobe executable was not found: {ffprobe}")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,pix_fmt,width,height,r_frame_rate,"
                "avg_frame_rate,nb_frames,nb_read_frames:"
                "format=duration,size,format_name"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise Wan22BatchError(
            f"ffprobe failed for {path}: {completed.stderr.strip()}"
        )
    value = _parse_json_bytes(
        completed.stdout.encode("utf-8"),
        context=f"ffprobe output for {path}",
    )
    if not isinstance(value, Mapping):
        raise Wan22BatchError(f"ffprobe output is not an object: {path}")
    return normalize_ffprobe_payload(
        value,
        expected_frames=expected_frames,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_fps=expected_fps,
        expected_codec=expected_codec,
        max_nominal_duration_error_frames=(
            max_nominal_duration_error_frames
        ),
    )


def _temporal_probe_fields(
    probe: Mapping[str, Any],
    *,
    context: str,
) -> tuple[int, Fraction, float]:
    frames = _integerish(
        probe.get("frames"),
        context=f"{context} frame count",
    )
    if frames <= 0:
        raise Wan22BatchError(f"{context} frame count must be positive")
    rate = _expected_positive_fraction(
        probe.get("frame_rate"),
        context=f"{context} frame rate",
    )
    try:
        duration = float(probe.get("duration_seconds"))
    except (TypeError, ValueError) as error:
        raise Wan22BatchError(f"{context} duration is invalid") from error
    if not math.isfinite(duration) or duration <= 0:
        raise Wan22BatchError(f"{context} duration must be positive")
    nominal_duration = float(Fraction(frames, 1) / rate)
    nominal_error_frames = abs(duration - nominal_duration) * float(rate)
    if nominal_error_frames > MAX_DURATION_DELTA_FRAMES + 1e-6:
        raise Wan22BatchError(
            f"{context} duration is incompatible with its constant time grid: "
            f"frames={frames} frame_rate={_fraction_text(rate)} "
            f"duration={duration:.9f}s nominal={nominal_duration:.9f}s "
            f"error_frames={nominal_error_frames:.6f}"
        )
    return frames, rate, duration


def validate_batch_temporal_grid(
    prepared_rows: Sequence[Mapping[str, Any]],
    *,
    expected_frame_num: int,
) -> dict[str, Any]:
    """Validate one source-derived temporal grid for a generation batch."""

    if not prepared_rows:
        raise Wan22BatchError("temporal preflight requires at least one row")
    if expected_frame_num <= 0 or (expected_frame_num - 1) % 4 != 0:
        raise Wan22BatchError(
            "expected source frame count must be positive and of the form 4n+1"
        )

    reference_rate: Fraction | None = None
    reference_duration: float | None = None
    durations: list[float] = []
    for index, row in enumerate(prepared_rows):
        iid = str(row.get("_iid") or row.get("iid") or f"index-{index}")
        media = row.get("_input_media")
        if not isinstance(media, Mapping):
            raise Wan22BatchError(f"missing input media for temporal iid={iid}")
        probe = media.get("source_video_ffprobe")
        if not isinstance(probe, Mapping):
            raise Wan22BatchError(
                f"missing source-video probe for temporal iid={iid}"
            )
        frames, rate, duration = _temporal_probe_fields(
            probe,
            context=f"source video iid={iid}",
        )
        if (frames - 1) % 4 != 0:
            raise Wan22BatchError(
                f"unsupported source frame count iid={iid}: {frames}; "
                "Wan I2V requires 4n+1 and retiming/padding is forbidden"
            )
        if frames != expected_frame_num:
            raise Wan22BatchError(
                f"source frame count differs from --frame-num iid={iid}: "
                f"source={frames} requested={expected_frame_num}; "
                "source/target frame counts must be identical"
            )
        if reference_rate is None:
            reference_rate = rate
            reference_duration = duration
        elif rate != reference_rate:
            raise Wan22BatchError(
                "batch contains incompatible source frame rates: "
                f"expected={_fraction_text(reference_rate)} "
                f"iid={iid} actual={_fraction_text(rate)}"
            )
        elif (
            reference_duration is not None
            and abs(duration - reference_duration) * float(reference_rate)
            > MAX_DURATION_DELTA_FRAMES + 1e-6
        ):
            raise Wan22BatchError(
                "batch contains incompatible source durations: "
                f"reference={reference_duration:.9f}s iid={iid} "
                f"actual={duration:.9f}s"
            )
        durations.append(duration)

    assert reference_rate is not None
    nominal_duration = float(
        Fraction(expected_frame_num, 1) / reference_rate
    )
    tolerance_seconds = float(
        Fraction(MAX_DURATION_DELTA_FRAMES, 1) / reference_rate
    )
    return {
        "policy_version": TEMPORAL_POLICY,
        "source_frame_count": expected_frame_num,
        "target_frame_count": expected_frame_num,
        "source_frame_rate": _fraction_text(reference_rate),
        "target_container_frame_rate": _fraction_text(reference_rate),
        "nominal_duration_seconds": nominal_duration,
        "source_duration_range_seconds": [min(durations), max(durations)],
        "duration_match_tolerance_frames": MAX_DURATION_DELTA_FRAMES,
        "duration_match_tolerance_seconds": tolerance_seconds,
        "model_sample_fps": MODEL_SAMPLE_FPS,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "source_frame_count_must_be_4n_plus_1": True,
        "source_target_frame_count_equal": True,
        "source_target_frame_rate_equal": True,
        "source_target_duration_within_tolerance": True,
        "batch_time_grid_uniform": True,
    }


def validate_temporal_pair(
    *,
    source_probe: Mapping[str, Any],
    target_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Return bound evidence for one source/target temporal match."""

    source_frames, source_rate, source_duration = _temporal_probe_fields(
        source_probe,
        context="source video",
    )
    target_frames, target_rate, target_duration = _temporal_probe_fields(
        target_probe,
        context="target video",
    )
    if source_frames != target_frames:
        raise Wan22BatchError(
            "source/target frame count mismatch: "
            f"source={source_frames} target={target_frames}"
        )
    if source_rate != target_rate:
        raise Wan22BatchError(
            "source/target frame rate mismatch: "
            f"source={_fraction_text(source_rate)} "
            f"target={_fraction_text(target_rate)}"
        )
    duration_delta = abs(source_duration - target_duration)
    tolerance_seconds = float(
        Fraction(MAX_DURATION_DELTA_FRAMES, 1) / source_rate
    )
    duration_delta_frames = duration_delta * float(source_rate)
    if duration_delta_frames > MAX_DURATION_DELTA_FRAMES + 1e-6:
        raise Wan22BatchError(
            "source/target duration mismatch exceeds one frame: "
            f"source={source_duration:.9f}s target={target_duration:.9f}s "
            f"delta_frames={duration_delta_frames:.6f}"
        )
    return {
        "policy_version": TEMPORAL_POLICY,
        "model_sample_fps": MODEL_SAMPLE_FPS,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "source": {
            "frame_count": source_frames,
            "frame_rate": _fraction_text(source_rate),
            "duration_seconds": source_duration,
        },
        "target": {
            "frame_count": target_frames,
            "frame_rate": _fraction_text(target_rate),
            "duration_seconds": target_duration,
        },
        "frame_count_equal": True,
        "frame_rate_equal": True,
        "duration_delta_seconds": duration_delta,
        "duration_delta_frames": duration_delta_frames,
        "duration_match_tolerance_frames": MAX_DURATION_DELTA_FRAMES,
        "duration_match_tolerance_seconds": tolerance_seconds,
        "duration_within_tolerance": True,
    }


def _path_is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _runner_sha256() -> str:
    return _sha256_file(Path(__file__).resolve(strict=True))


def _non_production_preview_bindings(
    row: Mapping[str, Any],
    *,
    manifest_sha256: Any,
) -> dict[str, Any]:
    """Return the closed provenance binding carried by preview artifacts."""

    authorization_mode = row.get("_authorization_mode")
    if not _is_non_production_preview_authorization(authorization_mode):
        raise Wan22BatchError("row is not an unsigned full-motion preview")
    common = {
        "manifest_sha256": _sha256_field(
            manifest_sha256, context="preview manifest_sha256"
        ),
        "manifest_row_digest": _sha256_field(
            row.get("_row_digest"), context="preview manifest_row_digest"
        ),
        "edit_instruction_sha256": _sha256_field(
            row.get("edit_instruction_sha256"),
            context="preview edit_instruction_sha256",
        ),
    }
    if authorization_mode == NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE:
        if row.get("schema_version") != FULL_MOTION_GENERATION_SCHEMA:
            raise Wan22BatchError("full-motion v6 preview schema differs")
        evidence = row.get("qwen_evidence")
        if not isinstance(evidence, Mapping):
            raise Wan22BatchError("full-motion preview Qwen evidence is missing")
        return {
            **common,
            "qwen_result_digest": _sha256_field(
                evidence.get("result_digest"),
                context="preview qwen_result_digest",
            ),
            "qwen_provenance_digest": _sha256_field(
                evidence.get("provenance_digest"),
                context="preview qwen_provenance_digest",
            ),
        }
    if (
        authorization_mode
        != QWEN_V16_NON_PRODUCTION_PREVIEW_AUTHORIZATION_MODE
        or row.get("schema_version") != FULL_MOTION_QWEN_V16_PASSED_SCHEMA
    ):
        raise Wan22BatchError("full-motion Qwen v16 preview lineage differs")
    media = row.get("_input_media")
    if not isinstance(media, Mapping):
        raise Wan22BatchError(
            "full-motion Qwen v16 preview media have not been prepared"
        )
    iid = _safe_iid(row.get("_iid"), context="Qwen v16 preview iid")
    source_manifest_path = _string(
        row.get("source_video"), context="Qwen v16 source_video"
    )
    source_resolved_manifest_path = _string(
        row.get("resolved_source_video"),
        context="Qwen v16 resolved_source_video",
    )
    anchor_manifest_path = _string(
        row.get("anchor_image"), context="Qwen v16 anchor_image"
    )
    anchor_resolved_manifest_path = _string(
        row.get("resolved_anchor_image"),
        context="Qwen v16 resolved_anchor_image",
    )
    if not Path(source_resolved_manifest_path).is_absolute() or not Path(
        anchor_resolved_manifest_path
    ).is_absolute():
        raise Wan22BatchError(
            "full-motion Qwen v16 resolved media paths must be absolute"
        )
    source_census = row.get("source_census")
    target_plan = row.get("target_plan")
    compiled_instruction = row.get("compiled_instruction")
    if not all(
        isinstance(value, Mapping)
        for value in (source_census, target_plan, compiled_instruction)
    ):
        raise Wan22BatchError(
            "full-motion Qwen v16 motion-plan bindings are missing"
        )
    if (
        row.get("all_dynamic_subjects_covered") is not True
        or row.get("camera_covered") is not True
    ):
        raise Wan22BatchError(
            "full-motion Qwen v16 coverage bindings are false"
        )
    return {
        **common,
        "lineage_schema_version": FULL_MOTION_QWEN_V16_PASSED_SCHEMA,
        "iid": iid,
        "source_video_manifest_path": source_manifest_path,
        "source_video_resolved_manifest_path": source_resolved_manifest_path,
        "source_video_prepared_path": _string(
            media.get("source_video_path"),
            context="Qwen v16 prepared source_video_path",
        ),
        "source_video_sha256": _sha256_field(
            row.get("source_video_sha256"),
            context="Qwen v16 source_video_sha256",
        ),
        "anchor_manifest_path": anchor_manifest_path,
        "anchor_resolved_manifest_path": anchor_resolved_manifest_path,
        "anchor_prepared_path": _string(
            media.get("anchor_path"), context="Qwen v16 prepared anchor_path"
        ),
        "anchor_sha256": _sha256_field(
            row.get("anchor_sha256"), context="Qwen v16 anchor_sha256"
        ),
        "qwen_record_digest": _sha256_field(
            row.get("qwen_record_digest"),
            context="Qwen v16 qwen_record_digest",
        ),
        "source_census_sha256": _object_digest(source_census),
        "target_plan_sha256": _object_digest(target_plan),
        "compiled_instruction_object_sha256": _object_digest(
            compiled_instruction
        ),
        "source_census": dict(source_census),
        "target_plan": dict(target_plan),
        "compiled_instruction": dict(compiled_instruction),
        "all_dynamic_subjects_covered": True,
        "camera_covered": True,
    }


def build_run_contract(
    *,
    manifest: Mapping[str, Any],
    prepared_rows: Sequence[Mapping[str, Any]],
    temporal_policy: Mapping[str, Any],
    args: argparse.Namespace,
    official: Mapping[str, Any],
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    preview_mode = _is_non_production_preview_authorization(
        prepared_rows[0].get("_authorization_mode")
    )
    selected_inputs = []
    for index, row in enumerate(prepared_rows):
        media = row["_input_media"]
        selected_input = {
                "index": index,
                "iid": row["_iid"],
                "group_id": row["group_id"],
                "row_digest": row["_row_digest"],
                "seed": sample_seed(args.base_seed, row["_iid"]),
                "anchor_sha256": row["anchor_sha256"],
                "anchor_rgb_sha256": media["anchor_rgb_sha256"],
                "anchor_width": media["anchor_width"],
                "anchor_height": media["anchor_height"],
                "source_video_sha256": row["source_video_sha256"],
                "source_video_ffprobe": media["source_video_ffprobe"],
                "authorization_mode": row["_authorization_mode"],
                "manifest_role": row["manifest_role"],
                "production_eligible": row["production_eligible"],
                "human_review_status": row["human_review_status"],
                "generation_authorized": row["generation_authorized"],
                "approval": row["approval"],
                "action_change_substantive": row["action_change_substantive"],
            }
        if _is_signed_authorization(row["_authorization_mode"]):
            selected_input["signed_release"] = row["_signed_release"]
        elif preview_mode:
            selected_input.update(
                {
                    "production_use_forbidden": True,
                    "preview_bindings": _non_production_preview_bindings(
                        row,
                        manifest_sha256=manifest["manifest_sha256"],
                    ),
                }
            )
        selected_inputs.append(selected_input)
    signed_authorization = "release" in manifest
    contract: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "manifest": {
            "path": manifest["manifest_path"],
            "sha256": manifest["manifest_sha256"],
            "bytes": manifest["manifest_bytes"],
            "row_count": manifest["manifest_row_count"],
            "selected_row_count": manifest["selected_row_count"],
            "max_samples": args.max_samples,
        },
        "selected_inputs": selected_inputs,
        "official_code": dict(official),
        "model": dict(model),
        "runtime": dict(runtime),
        "distributed_execution": {
            "world_size": args.expected_world_size,
            "cooperative_samples_per_step": 1,
            "independent_model_per_gpu": False,
            "t5_fsdp": True,
            "dit_fsdp": True,
            "ulysses_size": args.expected_world_size,
            "offload_model": False,
            "backend": "nccl",
            "rocm_required": bool(args.require_rocm),
            "expected_gpu_name_substring": args.expected_gpu_name_substring,
            "max_new_samples_per_allocation": args.max_new_samples,
        },
        "generation_parameters": {
            "task": "i2v-A14B",
            "size": args.size,
            "max_area": SUPPORTED_MAX_AREAS[args.size],
            "frame_num": args.frame_num,
            "sample_steps": args.sample_steps,
            "sample_shift": args.sample_shift,
            "sample_solver": args.sample_solver,
            "guide_scale": [
                args.sample_guide_scale_low,
                args.sample_guide_scale_high,
            ],
            "model_sample_fps": MODEL_SAMPLE_FPS,
            "output_container_frame_rate": temporal_policy[
                "target_container_frame_rate"
            ],
            "base_seed": args.base_seed,
            "negative_prompt": "official_config_default",
            "prompt_field": "edit_instruction",
            "video_codec": args.video_codec,
            "video_quality": args.video_quality,
        },
        "authorization": (
            {
                "mode": prepared_rows[0]["_authorization_mode"],
                "allow_pending_review": False,
                "legacy_approval_records_trusted": False,
                "release": dict(manifest["release"]),
            }
            if signed_authorization
            else {
                "allow_pending_review": False,
                "pending_review_override_supported": False,
                "requires_explicit_human_approval": True,
                "approved_manifest_role": APPROVED_MANIFEST_ROLE,
                "approval_schema": APPROVAL_SCHEMA,
            }
        ),
        "first_frame_policy": {
            "policy_version": FIRST_FRAME_POLICY,
            "tensor_frame0_override": True,
            "conditioning_resize": (
                "torch.nn.functional.interpolate(mode='bicubic')"
            ),
            "lossless_float32_binding": True,
            "lossless_png_binding": True,
            "mp4_codec_is_lossy": True,
            "mp4_decode_pixel_equality_claimed": False,
            "note": _FIRST_FRAME_NOTE,
        },
        "temporal_policy": dict(temporal_policy),
        "runner": {
            "module": "motive.wan22_i2v_batch",
            "sha256": _runner_sha256(),
        },
    }
    if preview_mode:
        contract.update(
            {
                "production_use_forbidden": True,
                "non_production_preview": {
                    "authorization_mode": prepared_rows[0][
                        "_authorization_mode"
                    ],
                    "production_use_forbidden": True,
                    "signed_release_supplied": False,
                    "row_count": 1,
                    "bindings": _non_production_preview_bindings(
                        prepared_rows[0],
                        manifest_sha256=manifest["manifest_sha256"],
                    ),
                },
            }
        )
        contract["authorization"] = {
            "mode": prepared_rows[0]["_authorization_mode"],
            "production_use_forbidden": True,
            "signed_release_supplied": False,
            "generation_authorized": False,
            "human_review_status": "pending",
        }
    contract["contract_digest"] = _object_digest(contract)
    return contract


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise Wan22BatchError(f"refusing to overwrite existing file: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_run_contract(output_root: Path, contract: Mapping[str, Any]) -> Path:
    existed = output_root.exists() or output_root.is_symlink()
    if output_root.is_symlink():
        raise Wan22BatchError(f"output root must not be a symlink: {output_root}")
    if existed and not output_root.is_dir():
        raise Wan22BatchError(f"output root is not a directory: {output_root}")
    contract_path = output_root / RUN_CONTRACT_NAME
    if existed and not contract_path.exists():
        entries = list(output_root.iterdir())
        if entries:
            raise Wan22BatchError(
                "existing output root has no run contract and is not empty: "
                f"{output_root}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    samples = output_root / "samples"
    staging = output_root / ".staging"
    for directory in (samples, staging):
        if directory.is_symlink():
            raise Wan22BatchError(f"output directory is a symlink: {directory}")
        directory.mkdir(exist_ok=True)
    if contract_path.exists() or contract_path.is_symlink():
        existing = _load_json(contract_path, context="run contract")
        if _canonical_bytes(existing) != _canonical_bytes(contract):
            raise Wan22BatchError(
                "existing run contract differs; use a new output root"
            )
    else:
        _atomic_create_bytes(contract_path, _json_bytes(contract))
    return contract_path


def _relative_committed_file(
    sample_dir: Path,
    relative: Any,
    *,
    context: str,
) -> Path:
    name = _string(relative, context=context)
    if Path(name).name != name or name in {".", ".."}:
        raise Wan22BatchError(f"{context} must be one basename")
    path = sample_dir / name
    return _regular_file(path, context=context)


def validate_sample_commit(
    sample_dir: Path,
    *,
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    if sample_dir.is_symlink() or not sample_dir.is_dir():
        raise Wan22BatchError(f"invalid committed sample directory: {sample_dir}")
    result_path = sample_dir / SAMPLE_RESULT_NAME
    result = _load_json(result_path, context=f"sample result iid={row['_iid']}")
    actual_digest = _sha256_field(
        result.get("result_digest"),
        context=f"sample result iid={row['_iid']} result_digest",
    )
    bound = dict(result)
    del bound["result_digest"]
    if actual_digest != _object_digest(bound):
        raise Wan22BatchError(f"sample result digest mismatch: {row['_iid']}")
    expected_equal = {
        "schema_version": SAMPLE_SCHEMA,
        "iid": row["_iid"],
        "sample_index": sample_index,
        "manifest_sha256": contract["manifest"]["sha256"],
        "manifest_row_digest": row["_row_digest"],
        "contract_digest": contract["contract_digest"],
        "seed": sample_seed(
            contract["generation_parameters"]["base_seed"],
            row["_iid"],
        ),
    }
    for field, expected in expected_equal.items():
        if result.get(field) != expected:
            raise Wan22BatchError(
                f"sample {row['_iid']} result {field} mismatch: "
                f"expected={expected!r} actual={result.get(field)!r}"
            )
    if result.get("authorization_mode") != row["_authorization_mode"]:
        raise Wan22BatchError(
            f"sample {row['_iid']} authorization provenance mismatch"
        )
    signed_mode = _is_signed_authorization(row["_authorization_mode"])
    preview_mode = _is_non_production_preview_authorization(
        row["_authorization_mode"]
    )
    if preview_mode:
        expected_preview_bindings = _non_production_preview_bindings(
            row,
            manifest_sha256=contract["manifest"]["sha256"],
        )
        authorization_differs = (
            result.get("manifest_role") != row["manifest_role"]
            or result.get("production_eligible") is not False
            or result.get("human_review_status_at_generation") != "pending"
            or result.get("generation_authorized_in_manifest") is not False
            or result.get("approval") is not None
            or result.get("production_use_forbidden") is not True
            or result.get("preview_bindings") != expected_preview_bindings
            or "signed_release" in result
        )
        contract_preview = contract.get("non_production_preview")
        if (
            contract.get("production_use_forbidden") is not True
            or not isinstance(contract_preview, Mapping)
            or contract_preview.get("production_use_forbidden") is not True
            or contract_preview.get("authorization_mode")
            != row["_authorization_mode"]
            or contract_preview.get("bindings") != expected_preview_bindings
        ):
            raise Wan22BatchError(
                f"sample {row['_iid']} preview run contract differs"
            )
    elif signed_mode:
        authorization_differs = (
            result.get("manifest_role") != "review_proposal"
            or result.get("production_eligible") is not False
            or result.get("human_review_status_at_generation") != "pending"
            or result.get("generation_authorized_in_manifest") is not False
            or result.get("approval") is not None
            or result.get("signed_release") != row["_signed_release"]
        )
    else:
        authorization_differs = (
            result.get("manifest_role") != APPROVED_MANIFEST_ROLE
            or result.get("production_eligible") is not True
            or result.get("human_review_status_at_generation") != "approved"
            or result.get("generation_authorized_in_manifest") is not True
            or _canonical_bytes(result.get("approval"))
            != _canonical_bytes(row["approval"])
        )
    if authorization_differs:
        raise Wan22BatchError(
            f"sample {row['_iid']} bound approval provenance mismatch"
        )
    if signed_mode or preview_mode:
        prompt = result.get("prompt")
        if not isinstance(prompt, Mapping):
            raise Wan22BatchError(f"sample {row['_iid']} prompt is missing")
        if (
            prompt.get("field") != "edit_instruction"
            or prompt.get("text") != row["edit_instruction"]
            or prompt.get("sha256")
            != _sha256_bytes(row["edit_instruction"].encode("utf-8"))
            or "absolute_target_prompt" in prompt
            or "edited_caption" in prompt
        ):
            raise Wan22BatchError(
                f"sample {row['_iid']} executable instruction differs"
            )
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise Wan22BatchError(f"sample {row['_iid']} outputs are missing")
    file_bindings = (
        ("source_video", "source_video_sha256"),
        ("edit_instruction_file", "edit_instruction_file_sha256"),
        ("preview_mp4", "preview_mp4_sha256"),
        ("conditioning_anchor_original", "conditioning_anchor_original_sha256"),
        (
            "conditioning_frame0_float32",
            "conditioning_frame0_float32_sha256",
        ),
        ("conditioning_frame0_png", "conditioning_frame0_png_sha256"),
    )
    for path_field, digest_field in file_bindings:
        path = _relative_committed_file(
            sample_dir,
            outputs.get(path_field),
            context=f"sample {row['_iid']} {path_field}",
        )
        expected_digest = _sha256_field(
            outputs.get(digest_field),
            context=f"sample {row['_iid']} {digest_field}",
        )
        if _sha256_file(path) != expected_digest:
            raise Wan22BatchError(
                f"sample {row['_iid']} committed file hash mismatch: {path.name}"
            )
    input_media = row.get("_input_media")
    if not isinstance(input_media, Mapping):
        raise Wan22BatchError(f"sample {row['_iid']} input media are missing")
    source_input_path = Path(
        _string(
            input_media.get("source_video_path"),
            context=f"sample {row['_iid']} source_video_path",
        )
    )
    expected_source_name = _source_video_artifact_name(
        source_input_path,
        context=f"sample {row['_iid']}",
    )
    if outputs.get("source_video") != expected_source_name:
        raise Wan22BatchError(
            f"sample {row['_iid']} committed source filename differs"
        )
    source_copy = _relative_committed_file(
        sample_dir,
        outputs.get("source_video"),
        context=f"sample {row['_iid']} source_video",
    )
    if outputs.get("source_video_sha256") != row["source_video_sha256"]:
        raise Wan22BatchError(
            f"sample {row['_iid']} committed source binding mismatch"
        )
    if outputs.get("source_video_bytes") != source_copy.stat().st_size:
        raise Wan22BatchError(
            f"sample {row['_iid']} committed source byte count mismatch"
        )

    instruction_raw = _string(
        row.get("edit_instruction"),
        context=f"sample {row['_iid']} edit_instruction",
    ).encode("utf-8")
    if outputs.get("edit_instruction_file") != EDIT_INSTRUCTION_ARTIFACT_NAME:
        raise Wan22BatchError(
            f"sample {row['_iid']} edit instruction filename differs"
        )
    instruction_path = _relative_committed_file(
        sample_dir,
        outputs.get("edit_instruction_file"),
        context=f"sample {row['_iid']} edit_instruction_file",
    )
    if instruction_path.read_bytes() != instruction_raw:
        raise Wan22BatchError(
            f"sample {row['_iid']} edit instruction content differs"
        )
    if (
        outputs.get("edit_instruction_file_sha256")
        != _sha256_bytes(instruction_raw)
        or outputs.get("edit_instruction_file_bytes") != len(instruction_raw)
    ):
        raise Wan22BatchError(
            f"sample {row['_iid']} edit instruction binding mismatch"
        )

    inputs = result.get("inputs")
    if not isinstance(inputs, Mapping):
        raise Wan22BatchError(f"sample {row['_iid']} inputs are missing")
    original_source_binding = _regular_file(
        Path(
            _string(
                inputs.get("source_video_resolved_path"),
                context=(
                    f"sample {row['_iid']} source_video_resolved_path"
                ),
            )
        ),
        context=f"sample {row['_iid']} original source provenance",
    )
    committed_source_binding = _regular_file(
        Path(
            _string(
                inputs.get("source_video_committed_path"),
                context=(
                    f"sample {row['_iid']} source_video_committed_path"
                ),
            )
        ),
        context=f"sample {row['_iid']} committed source provenance",
    )
    if (
        original_source_binding != _regular_file(
            source_input_path,
            context=f"sample {row['_iid']} prepared source provenance",
        )
        or committed_source_binding != source_copy
        or inputs.get("source_video_sha256") != row["source_video_sha256"]
    ):
        raise Wan22BatchError(
            f"sample {row['_iid']} source provenance differs"
        )

    motion_output_fields = {
        "motion_spec_json",
        "motion_spec_json_sha256",
        "motion_spec_json_bytes",
        "motion_spec_object_sha256",
    }
    optional_motion = _optional_motion_spec(
        row,
        context=f"sample {row['_iid']}",
    )
    if optional_motion is None:
        if any(field in outputs for field in motion_output_fields):
            raise Wan22BatchError(
                f"sample {row['_iid']} has an unbound motion_spec output"
            )
    else:
        motion_spec, object_digest = optional_motion
        if outputs.get("motion_spec_json") != MOTION_SPEC_ARTIFACT_NAME:
            raise Wan22BatchError(
                f"sample {row['_iid']} motion_spec filename differs"
            )
        motion_path = _relative_committed_file(
            sample_dir,
            outputs.get("motion_spec_json"),
            context=f"sample {row['_iid']} motion_spec_json",
        )
        motion_file_sha = _sha256_field(
            outputs.get("motion_spec_json_sha256"),
            context=f"sample {row['_iid']} motion_spec_json_sha256",
        )
        if (
            _sha256_file(motion_path) != motion_file_sha
            or outputs.get("motion_spec_json_bytes")
            != motion_path.stat().st_size
            or outputs.get("motion_spec_object_sha256") != object_digest
        ):
            raise Wan22BatchError(
                f"sample {row['_iid']} motion_spec file binding mismatch"
            )
        committed_motion = _load_json(
            motion_path,
            context=f"sample {row['_iid']} motion_spec_json",
        )
        if _canonical_bytes(committed_motion) != _canonical_bytes(motion_spec):
            raise Wan22BatchError(
                f"sample {row['_iid']} motion_spec content differs"
            )
    if (
        outputs.get("conditioning_anchor_original_sha256")
        != row["anchor_sha256"]
    ):
        raise Wan22BatchError(
            f"sample {row['_iid']} original anchor binding mismatch"
        )
    policy = result.get("first_frame_policy")
    if not isinstance(policy, Mapping):
        raise Wan22BatchError(f"sample {row['_iid']} frame-zero policy is missing")
    if policy.get("policy_version") != FIRST_FRAME_POLICY:
        raise Wan22BatchError(f"sample {row['_iid']} frame-zero policy mismatch")
    if policy.get("preencode_frame0_matches_png_pixels") is not True:
        raise Wan22BatchError(f"sample {row['_iid']} frame-zero match is false")
    if policy.get("mp4_decode_pixel_equality_claimed") is not False:
        raise Wan22BatchError(
            f"sample {row['_iid']} makes a forbidden MP4 equality claim"
        )
    temporal = result.get("temporal_policy")
    if not isinstance(temporal, Mapping):
        raise Wan22BatchError(
            f"sample {row['_iid']} temporal policy is missing"
        )
    source_probe = row.get("_input_media", {}).get("source_video_ffprobe")
    target_probe = outputs.get("preview_mp4_ffprobe")
    if not isinstance(source_probe, Mapping) or not isinstance(
        target_probe,
        Mapping,
    ):
        raise Wan22BatchError(
            f"sample {row['_iid']} temporal probes are missing"
        )
    expected_temporal = validate_temporal_pair(
        source_probe=source_probe,
        target_probe=target_probe,
    )
    if _canonical_bytes(temporal) != _canonical_bytes(expected_temporal):
        raise Wan22BatchError(
            f"sample {row['_iid']} temporal evidence differs"
        )
    contract_temporal = contract.get("temporal_policy")
    if not isinstance(contract_temporal, Mapping):
        raise Wan22BatchError("run contract temporal policy is missing")
    expected_contract_fields = {
        "policy_version": TEMPORAL_POLICY,
        "source_frame_count": expected_temporal["source"]["frame_count"],
        "target_frame_count": expected_temporal["target"]["frame_count"],
        "source_frame_rate": expected_temporal["source"]["frame_rate"],
        "target_container_frame_rate": expected_temporal["target"]["frame_rate"],
        "duration_match_tolerance_frames": MAX_DURATION_DELTA_FRAMES,
        "model_sample_fps": MODEL_SAMPLE_FPS,
    }
    for field, expected in expected_contract_fields.items():
        if contract_temporal.get(field) != expected:
            raise Wan22BatchError(
                f"sample {row['_iid']} contract temporal {field} mismatch"
            )
    return result


def _validate_existing_samples(
    output_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    pending: list[int] = []
    completed: dict[int, dict[str, Any]] = {}
    samples_root = output_root / "samples"
    for index, row in enumerate(rows):
        sample_dir = samples_root / row["_iid"]
        if sample_dir.exists() or sample_dir.is_symlink():
            completed[index] = validate_sample_commit(
                sample_dir,
                row=row,
                contract=contract,
                sample_index=index,
            )
        else:
            pending.append(index)
    return pending, completed


def _write_staging_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _source_video_artifact_name(source: Path, *, context: str) -> str:
    suffix = source.suffix
    if _SOURCE_VIDEO_SUFFIX_RE.fullmatch(suffix) is None:
        raise Wan22BatchError(
            f"{context} source video has no safe filename suffix: {source.name!r}"
        )
    return f"{SOURCE_VIDEO_ARTIFACT_STEM}{suffix}"


def _optional_motion_spec(
    row: Mapping[str, Any],
    *,
    context: str,
) -> tuple[dict[str, Any], str] | None:
    has_value = "motion_spec" in row
    has_digest = "motion_spec_sha256" in row
    if not has_value and not has_digest:
        return None
    if not has_value or not has_digest:
        raise Wan22BatchError(
            f"{context} motion_spec and motion_spec_sha256 must appear together"
        )
    value = row.get("motion_spec")
    if not isinstance(value, Mapping):
        raise Wan22BatchError(f"{context} motion_spec must be an object")
    motion_spec = dict(value)
    expected_digest = _sha256_field(
        row.get("motion_spec_sha256"),
        context=f"{context} motion_spec_sha256",
    )
    if _object_digest(motion_spec) != expected_digest:
        raise Wan22BatchError(f"{context} motion_spec object digest mismatch")
    return motion_spec, expected_digest


def _materialize_self_contained_inputs(
    staging: Path,
    *,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy executable inputs into one staging sample and bind their bytes."""

    iid = str(row["_iid"])
    media = row.get("_input_media")
    if not isinstance(media, Mapping):
        raise Wan22BatchError(f"sample {iid} input media are missing")
    source = _regular_file(
        Path(
            _string(
                media.get("source_video_path"),
                context=f"sample {iid} source_video_path",
            )
        ),
        context=f"sample {iid} source video",
    )
    source_name = _source_video_artifact_name(
        source,
        context=f"sample {iid}",
    )
    source_copy = staging / source_name
    shutil.copyfile(source, source_copy)
    with source_copy.open("rb") as handle:
        os.fsync(handle.fileno())
    source_sha = _sha256_file(source_copy)
    expected_source_sha = _sha256_field(
        row.get("source_video_sha256"),
        context=f"sample {iid} source_video_sha256",
    )
    if source_sha != expected_source_sha:
        raise Wan22BatchError(f"copied source video hash mismatch for iid={iid}")

    instruction = _string(
        row.get("edit_instruction"),
        context=f"sample {iid} edit_instruction",
    )
    instruction_raw = instruction.encode("utf-8")
    instruction_path = staging / EDIT_INSTRUCTION_ARTIFACT_NAME
    _write_staging_file(instruction_path, instruction_raw)
    if instruction_path.read_bytes() != instruction_raw:
        raise Wan22BatchError(
            f"copied edit instruction bytes mismatch for iid={iid}"
        )
    instruction_sha = _sha256_bytes(instruction_raw)

    outputs: dict[str, Any] = {
        "source_video": source_copy.name,
        "source_video_sha256": source_sha,
        "source_video_bytes": source_copy.stat().st_size,
        "edit_instruction_file": instruction_path.name,
        "edit_instruction_file_sha256": instruction_sha,
        "edit_instruction_file_bytes": instruction_path.stat().st_size,
    }
    optional_motion = _optional_motion_spec(
        row,
        context=f"sample {iid}",
    )
    if optional_motion is not None:
        motion_spec, object_digest = optional_motion
        motion_path = staging / MOTION_SPEC_ARTIFACT_NAME
        _write_staging_file(motion_path, _json_bytes(motion_spec))
        outputs.update(
            {
                "motion_spec_json": motion_path.name,
                "motion_spec_json_sha256": _sha256_file(motion_path),
                "motion_spec_json_bytes": motion_path.stat().st_size,
                "motion_spec_object_sha256": object_digest,
            }
        )
    return outputs


def _save_npy_new(path: Path, array: Any, *, numpy_module: Any) -> None:
    with path.open("xb") as handle:
        numpy_module.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _encode_preview_mp4(
    *,
    video: Any,
    anchor_frame_uint8: Any,
    output_path: Path,
    fps: Fraction,
    codec: str,
    quality: int,
    torch_module: Any,
    imageio_module: Any,
) -> None:
    writer = imageio_module.get_writer(
        str(output_path),
        fps=float(fps),
        codec=codec,
        quality=quality,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        frame_count = int(video.shape[1])
        for frame_index in range(frame_count):
            if frame_index == 0:
                frame = anchor_frame_uint8
            else:
                tensor = (
                    ((video[:, frame_index].detach().float().cpu() + 1.0) * 127.5)
                    .round()
                    .clamp_(0, 255)
                    .to(torch_module.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                )
                frame = tensor.numpy()
            writer.append_data(frame)
    finally:
        writer.close()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise Wan22BatchError(f"video encoder produced no output: {output_path}")
    with output_path.open("rb") as handle:
        os.fsync(handle.fileno())


def _commit_generated_sample(
    *,
    output_root: Path,
    row: Mapping[str, Any],
    sample_index: int,
    contract: Mapping[str, Any],
    video: Any,
    input_image: Any,
    args: argparse.Namespace,
    torch_module: Any,
    torchvision_tf: Any,
    numpy_module: Any,
    imageio_module: Any,
) -> dict[str, Any]:
    iid = row["_iid"]
    if video is None:
        raise Wan22BatchError(f"rank zero received no decoded video for iid={iid}")
    if getattr(video, "ndim", None) != 4:
        raise Wan22BatchError(
            f"Wan output must be C,T,H,W for iid={iid}; shape={video.shape}"
        )
    channels, frames, height, width = [int(value) for value in video.shape]
    if channels != 3 or frames != args.frame_num or height <= 0 or width <= 0:
        raise Wan22BatchError(
            f"unexpected Wan output shape for iid={iid}: {tuple(video.shape)}"
        )
    # VAE implementations may return a lower-precision tensor.  Promote the
    # preview tensor first so the frame-zero override is the exact float32
    # conditioning tensor, rather than a bfloat16 approximation.
    video = video.detach().float()

    staging_root = output_root / ".staging"
    staging = staging_root / (
        f"{iid}.{os.environ.get('SLURM_JOB_ID', 'manual')}."
        f"{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(mode=0o700)
    final_dir = output_root / "samples" / iid
    try:
        self_contained_outputs = _materialize_self_contained_inputs(
            staging,
            row=row,
        )
        original_anchor = Path(row["_input_media"]["anchor_path"])
        anchor_copy = staging / "conditioning_anchor_original.png"
        # The curated anchor contract is PNG.  Byte copying it preserves the
        # exact frame-zero input rather than re-encoding it through Pillow.
        if row["_input_media"]["anchor_format"] != "PNG":
            raise Wan22BatchError(
                f"conditioning anchor must be PNG for iid={iid}"
            )
        shutil.copyfile(original_anchor, anchor_copy)
        with anchor_copy.open("rb") as handle:
            os.fsync(handle.fileno())
        if _sha256_file(anchor_copy) != row["anchor_sha256"]:
            raise Wan22BatchError(f"copied anchor hash mismatch for iid={iid}")

        input_tensor = torchvision_tf.to_tensor(input_image)
        input_tensor = input_tensor.sub(0.5).div(0.5)
        conditioning = torch_module.nn.functional.interpolate(
            input_tensor[None].cpu(),
            size=(height, width),
            mode="bicubic",
        )[0].contiguous()
        if conditioning.dtype != torch_module.float32:
            conditioning = conditioning.float()
        video[:, 0].copy_(conditioning.to(device=video.device))
        if not torch_module.equal(video[:, 0].detach().cpu(), conditioning):
            raise Wan22BatchError(
                f"pre-encode tensor frame zero differs from conditioning iid={iid}"
            )

        conditioning_array = (
            conditioning.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        )
        float32_path = staging / "conditioning_frame0_float32.npy"
        _save_npy_new(
            float32_path,
            conditioning_array,
            numpy_module=numpy_module,
        )

        conditioning_uint8 = (
            ((conditioning.detach().float().cpu() + 1.0) * 127.5)
            .round()
            .clamp_(0, 255)
            .to(torch_module.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        from PIL import Image

        frame0_png = staging / "conditioning_frame0.png"
        Image.fromarray(conditioning_uint8, mode="RGB").save(
            frame0_png,
            format="PNG",
            compress_level=6,
        )
        with frame0_png.open("rb") as handle:
            os.fsync(handle.fileno())
        with Image.open(frame0_png) as saved:
            saved_array = numpy_module.asarray(saved.convert("RGB"))
        if not numpy_module.array_equal(saved_array, conditioning_uint8):
            raise Wan22BatchError(f"lossless frame-zero PNG mismatch for iid={iid}")

        preview = staging / "preview.mp4"
        source_probe = row["_input_media"]["source_video_ffprobe"]
        source_frame_rate = _expected_positive_fraction(
            source_probe["frame_rate"],
            context=f"source frame rate iid={iid}",
        )
        _encode_preview_mp4(
            video=video,
            anchor_frame_uint8=conditioning_uint8,
            output_path=preview,
            fps=source_frame_rate,
            codec=args.video_codec,
            quality=args.video_quality,
            torch_module=torch_module,
            imageio_module=imageio_module,
        )
        preview_probe = probe_video(
            preview,
            ffprobe=args.ffprobe,
            expected_frames=args.frame_num,
            expected_width=width,
            expected_height=height,
            expected_fps=source_frame_rate,
            expected_codec=args.video_codec,
            max_nominal_duration_error_frames=MAX_DURATION_DELTA_FRAMES,
        )
        temporal_policy = validate_temporal_pair(
            source_probe=source_probe,
            target_probe=preview_probe,
        )
        tensor_pixel_sha = _sha256_bytes(conditioning_uint8.tobytes(order="C"))
        result: dict[str, Any] = {
            "schema_version": SAMPLE_SCHEMA,
            "iid": iid,
            "group_id": row["group_id"],
            "sample_index": sample_index,
            "created_at_utc": _utc_now(),
            "manifest_sha256": contract["manifest"]["sha256"],
            "manifest_row_digest": row["_row_digest"],
            "contract_digest": contract["contract_digest"],
            "seed": sample_seed(args.base_seed, iid),
            "authorization_mode": row["_authorization_mode"],
            "manifest_role": row["manifest_role"],
            "production_eligible": row["production_eligible"],
            "approval": row["approval"],
            "action_change_substantive": row["action_change_substantive"],
            "human_review_status_at_generation": row["human_review_status"],
            "generation_authorized_in_manifest": row["generation_authorized"],
            "prompt": {
                "field": "edit_instruction",
                "text": row["edit_instruction"],
                "sha256": _sha256_bytes(
                    row["edit_instruction"].encode("utf-8")
                ),
            },
            "inputs": {
                "anchor_manifest_path": (
                    row.get("resolved_anchor_image") or row["anchor_image"]
                ),
                "anchor_resolved_path": row["_input_media"]["anchor_path"],
                "anchor_sha256": row["anchor_sha256"],
                "anchor_rgb_sha256": row["_input_media"]["anchor_rgb_sha256"],
                "anchor_width": row["_input_media"]["anchor_width"],
                "anchor_height": row["_input_media"]["anchor_height"],
                "source_video_manifest_path": (
                    row.get("resolved_source_video") or row["source_video"]
                ),
                "source_video_resolved_path": row["_input_media"][
                    "source_video_path"
                ],
                "source_video_committed_path": str(
                    final_dir / self_contained_outputs["source_video"]
                ),
                "source_video_sha256": row["source_video_sha256"],
                "source_video_ffprobe": row["_input_media"][
                    "source_video_ffprobe"
                ],
            },
            "generation_parameters": contract["generation_parameters"],
            "model": {
                "model_id": MODEL_ID,
                "hf_revision": MODEL_HF_REVISION,
                "official_code_commit": OFFICIAL_COMMIT,
            },
            "first_frame_policy": {
                "policy_version": FIRST_FRAME_POLICY,
                "tensor_frame0_overridden_before_encoding": True,
                "conditioning_tensor_shape": [3, height, width],
                "conditioning_tensor_dtype": "float32",
                "preencode_frame0_pixel_sha256": tensor_pixel_sha,
                "lossless_png_pixel_sha256": tensor_pixel_sha,
                "preencode_frame0_matches_png_pixels": True,
                "mp4_codec_is_lossy": True,
                "mp4_decode_pixel_equality_claimed": False,
                "note": _FIRST_FRAME_NOTE,
            },
            "temporal_policy": temporal_policy,
            "outputs": {
                **self_contained_outputs,
                "preview_mp4": preview.name,
                "preview_mp4_sha256": _sha256_file(preview),
                "preview_mp4_bytes": preview.stat().st_size,
                "preview_mp4_ffprobe": preview_probe,
                "conditioning_anchor_original": anchor_copy.name,
                "conditioning_anchor_original_sha256": _sha256_file(anchor_copy),
                "conditioning_frame0_float32": float32_path.name,
                "conditioning_frame0_float32_sha256": _sha256_file(float32_path),
                "conditioning_frame0_png": frame0_png.name,
                "conditioning_frame0_png_sha256": _sha256_file(frame0_png),
            },
        }
        if _is_signed_authorization(row["_authorization_mode"]):
            result["signed_release"] = row["_signed_release"]
        elif _is_non_production_preview_authorization(
            row["_authorization_mode"]
        ):
            result.update(
                {
                    "production_use_forbidden": True,
                    "preview_bindings": _non_production_preview_bindings(
                        row,
                        manifest_sha256=contract["manifest"]["sha256"],
                    ),
                }
            )
        result["result_digest"] = _object_digest(result)
        _write_staging_file(staging / SAMPLE_RESULT_NAME, _json_bytes(result))
        _fsync_directory(staging)
        if final_dir.exists() or final_dir.is_symlink():
            raise Wan22BatchError(
                f"refusing to replace committed sample directory: {final_dir}"
            )
        os.rename(staging, final_dir)
        _fsync_directory(final_dir.parent)
        return validate_sample_commit(
            final_dir,
            row=row,
            contract=contract,
            sample_index=sample_index,
        )
    except Exception:
        # Deliberately preserve staging for post-mortem evidence and safe
        # disconnect recovery.  It is never interpreted as a commit.
        raise


def _prepare_media_rows(
    manifest: Mapping[str, Any],
    *,
    data_root: Path | None,
    ffprobe: str,
    expected_frame_num: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image

    manifest_parent = Path(manifest["manifest_path"]).parent
    prepared: list[dict[str, Any]] = []
    for row in manifest["selected_rows"]:
        item = dict(row)
        iid = item["_iid"]
        anchor = _resolve_manifest_media(
            item,
            resolved_key="resolved_anchor_image",
            fallback_key="anchor_image",
            manifest_parent=manifest_parent,
            data_root=data_root,
            context=f"anchor iid={iid}",
        )
        source = _resolve_manifest_media(
            item,
            resolved_key="resolved_source_video",
            fallback_key="source_video",
            manifest_parent=manifest_parent,
            data_root=data_root,
            context=f"source video iid={iid}",
        )
        actual_anchor_sha = _sha256_file(anchor)
        if actual_anchor_sha != item["anchor_sha256"]:
            raise Wan22BatchError(
                f"anchor hash mismatch iid={iid}: "
                f"expected={item['anchor_sha256']} actual={actual_anchor_sha}"
            )
        actual_source_sha = _sha256_file(source)
        if actual_source_sha != item["source_video_sha256"]:
            raise Wan22BatchError(
                f"source video hash mismatch iid={iid}: "
                f"expected={item['source_video_sha256']} actual={actual_source_sha}"
            )
        with Image.open(anchor) as image:
            image_format = image.format
            rgb = image.convert("RGB")
            width, height = rgb.size
            rgb_bytes = rgb.tobytes()
        if width <= 0 or height <= 0:
            raise Wan22BatchError(f"anchor has invalid size iid={iid}")
        item["_input_media"] = {
            "anchor_path": str(anchor),
            "anchor_format": image_format,
            "anchor_rgb_sha256": _sha256_bytes(rgb_bytes),
            "anchor_width": width,
            "anchor_height": height,
            "source_video_path": str(source),
            "source_video_ffprobe": probe_video(
                source,
                ffprobe=ffprobe,
                expected_frames=None,
                max_nominal_duration_error_frames=(
                    MAX_DURATION_DELTA_FRAMES
                ),
            ),
        }
        prepared.append(item)
    temporal_policy = validate_batch_temporal_grid(
        prepared,
        expected_frame_num=expected_frame_num,
    )
    return prepared, temporal_policy


def _generated_manifest_rows(
    *,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        result = results[index]
        sample_dir = output_root / "samples" / row["_iid"]
        outputs = result["outputs"]
        source_copy = _relative_committed_file(
            sample_dir,
            outputs.get("source_video"),
            context=f"generated row {row['_iid']} source_video",
        )
        instruction_file = _relative_committed_file(
            sample_dir,
            outputs.get("edit_instruction_file"),
            context=f"generated row {row['_iid']} edit_instruction_file",
        )
        generated_row = {
                "schema_version": GENERATED_MANIFEST_SCHEMA,
                "iid": row["_iid"],
                "group_id": row["group_id"],
                "action_category": row["action_category"],
                "target_action_verb": row["target_action_verb"],
                "edit_instruction": row["edit_instruction"],
                "edit_instruction_sha256": _sha256_bytes(
                    row["edit_instruction"].encode("utf-8")
                ),
                "edit_instruction_file": str(instruction_file),
                "edit_instruction_file_sha256": outputs[
                    "edit_instruction_file_sha256"
                ],
                "edit_instruction_file_bytes": outputs[
                    "edit_instruction_file_bytes"
                ],
                "source_video": str(source_copy),
                "source_video_sha256": outputs["source_video_sha256"],
                "source_video_bytes": outputs["source_video_bytes"],
                "conditioning_anchor_original": str(
                    sample_dir / outputs["conditioning_anchor_original"]
                ),
                "conditioning_anchor_original_sha256": outputs[
                    "conditioning_anchor_original_sha256"
                ],
                "conditioning_frame0_float32": str(
                    sample_dir / outputs["conditioning_frame0_float32"]
                ),
                "conditioning_frame0_float32_sha256": outputs[
                    "conditioning_frame0_float32_sha256"
                ],
                "conditioning_frame0_png": str(
                    sample_dir / outputs["conditioning_frame0_png"]
                ),
                "conditioning_frame0_png_sha256": outputs[
                    "conditioning_frame0_png_sha256"
                ],
                "target_preview_mp4": str(sample_dir / outputs["preview_mp4"]),
                "target_preview_mp4_sha256": outputs["preview_mp4_sha256"],
                "result_json": str(sample_dir / SAMPLE_RESULT_NAME),
                "result_digest": result["result_digest"],
                "seed": result["seed"],
                "authorization_mode": result["authorization_mode"],
                "manifest_role": row["manifest_role"],
                "production_eligible": row["production_eligible"],
                "human_review_status": row["human_review_status"],
                "generation_authorized": row["generation_authorized"],
                "approval": row["approval"],
                "action_change_substantive": row["action_change_substantive"],
                "first_frame_policy": FIRST_FRAME_POLICY,
                "mp4_decode_pixel_equality_claimed": False,
                "temporal_policy": result["temporal_policy"],
            }
        optional_motion = _optional_motion_spec(
            row,
            context=f"generated row {row['_iid']}",
        )
        if optional_motion is not None:
            motion_path = _relative_committed_file(
                sample_dir,
                outputs.get("motion_spec_json"),
                context=f"generated row {row['_iid']} motion_spec_json",
            )
            generated_row.update(
                {
                    "motion_spec_json": str(motion_path),
                    "motion_spec_json_sha256": outputs[
                        "motion_spec_json_sha256"
                    ],
                    "motion_spec_json_bytes": outputs[
                        "motion_spec_json_bytes"
                    ],
                    "motion_spec_object_sha256": outputs[
                        "motion_spec_object_sha256"
                    ],
                }
            )
        if _is_signed_authorization(row["_authorization_mode"]):
            generated_row["signed_release"] = row["_signed_release"]
        elif _is_non_production_preview_authorization(
            row["_authorization_mode"]
        ):
            generated_row.update(
                {
                    "production_use_forbidden": True,
                    "preview_bindings": dict(result["preview_bindings"]),
                }
            )
        elif "absolute_target_prompt" in row:
            generated_row["absolute_target_prompt"] = row[
                "absolute_target_prompt"
            ]
        generated.append(generated_row)
    return generated


def publish_completion(
    output_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    results: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    generated_rows = _generated_manifest_rows(
        output_root=output_root,
        rows=rows,
        results=results,
    )
    generated_bytes = _jsonl_bytes(generated_rows)
    generated_path = output_root / GENERATED_MANIFEST_NAME
    if generated_path.exists() or generated_path.is_symlink():
        if (
            generated_path.is_symlink()
            or not generated_path.is_file()
            or generated_path.read_bytes() != generated_bytes
        ):
            raise Wan22BatchError(
                f"existing generated manifest differs: {generated_path}"
            )
    else:
        _atomic_create_bytes(generated_path, generated_bytes)
    complete: dict[str, Any] = {
        "schema_version": COMPLETE_SCHEMA,
        "contract_digest": contract["contract_digest"],
        "manifest_sha256": contract["manifest"]["sha256"],
        "selected_sample_count": len(rows),
        "completed_sample_count": len(results),
        "generated_manifest": GENERATED_MANIFEST_NAME,
        "generated_manifest_sha256": _sha256_bytes(generated_bytes),
        "temporal_policy": contract["temporal_policy"],
        "sample_result_digests": [
            results[index]["result_digest"] for index in range(len(rows))
        ],
    }
    complete["complete_digest"] = _object_digest(complete)
    complete_path = output_root / RUN_COMPLETE_NAME
    if complete_path.exists() or complete_path.is_symlink():
        existing = _load_json(complete_path, context="run completion")
        if _canonical_bytes(existing) != _canonical_bytes(complete):
            raise Wan22BatchError(f"existing completion differs: {complete_path}")
    else:
        _atomic_create_bytes(complete_path, _json_bytes(complete))
    return complete


def _broadcast_rank0_payload(
    dist: Any,
    *,
    rank: int,
    producer: Any,
    stage: str = "rank-zero operation",
) -> Any:
    if rank == 0:
        try:
            payload = {"ok": True, "value": producer()}
        except Exception as error:
            payload = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        payload = None
    container = [payload]
    dist.broadcast_object_list(container, src=0)
    received = container[0]
    if not isinstance(received, Mapping) or received.get("ok") is not True:
        raise Wan22BatchError(
            f"{stage} failed on rank zero: "
            f"{received.get('error_type') if isinstance(received, Mapping) else ''}: "
            f"{received.get('error') if isinstance(received, Mapping) else received}"
        )
    return received["value"]


def _collective_local_call(
    dist: Any,
    *,
    rank: int,
    world_size: int,
    stage: str,
    producer: Any,
    gather_values: bool = False,
) -> tuple[Any, list[Any] | None]:
    """Execute on every rank, then collectively propagate Python failures.

    This cannot recover from a killed process or a fatal GPU runtime abort, but
    it prevents an ordinary exception on one rank from leaving peers blocked at
    the next barrier.
    """

    local_value = None
    try:
        local_value = producer()
        local_status: dict[str, Any] = {
            "rank": rank,
            "ok": True,
        }
        if gather_values:
            local_status["value"] = local_value
    except Exception as error:
        local_status = {
            "rank": rank,
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    statuses: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(statuses, local_status)
    valid_ranks = {
        item.get("rank")
        for item in statuses
        if isinstance(item, Mapping)
    }
    if valid_ranks != set(range(world_size)):
        raise Wan22BatchError(
            f"{stage} returned invalid collective rank evidence: {statuses}"
        )
    failures = [
        item
        for item in statuses
        if not isinstance(item, Mapping) or item.get("ok") is not True
    ]
    if failures:
        details = "; ".join(
            (
                f"rank={item.get('rank')} "
                f"{item.get('error_type')}: {item.get('error')}"
                if isinstance(item, Mapping)
                else repr(item)
            )
            for item in failures
        )
        raise Wan22BatchError(f"{stage} failed collectively: {details}")
    gathered = (
        [item["value"] for item in statuses]
        if gather_values
        else None
    )
    return local_value, gathered


def _flash_attention_kernel_smoke(
    *,
    torch_module: Any,
    attention_module: Any,
    rank: int,
    local_rank: int,
) -> dict[str, Any]:
    """Launch one real, tiny active-backend FlashAttention kernel per rank."""

    if bool(getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False)):
        backend = "flash_attn_3"
    elif bool(getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False)):
        backend = "flash_attn_2"
    else:
        raise Wan22BatchError(
            "official Wan attention module found no FlashAttention backend"
        )
    device = torch_module.device("cuda", local_rank)
    generator = torch_module.Generator(device=device)
    generator.manual_seed(0x5A22 + rank)
    shape = (1, 8, 5, 64)
    q = torch_module.randn(
        shape,
        device=device,
        dtype=torch_module.bfloat16,
        generator=generator,
    )
    k = torch_module.randn(
        shape,
        device=device,
        dtype=torch_module.bfloat16,
        generator=generator,
    )
    v = torch_module.randn(
        shape,
        device=device,
        dtype=torch_module.bfloat16,
        generator=generator,
    )
    output = attention_module.flash_attention(
        q,
        k,
        v,
        causal=False,
        dtype=torch_module.bfloat16,
    )
    torch_module.cuda.synchronize(local_rank)
    actual_shape = tuple(int(value) for value in output.shape)
    if actual_shape != shape:
        raise Wan22BatchError(
            f"FlashAttention kernel returned shape={actual_shape}, expected={shape}"
        )
    if output.dtype != torch_module.bfloat16:
        raise Wan22BatchError(
            f"FlashAttention kernel returned dtype={output.dtype}, "
            "expected=torch.bfloat16"
        )
    finite = bool(torch_module.isfinite(output.float()).all().item())
    if not finite:
        raise Wan22BatchError("FlashAttention kernel returned non-finite values")
    del q, k, v, output
    return {
        "rank": rank,
        "local_rank": local_rank,
        "backend": backend,
        "dtype": "bfloat16",
        "input_shape": list(shape),
        "output_finite": True,
        "kernel_synchronized": True,
    }


def _distributed_runtime_preflight(
    *,
    torch_module: Any,
    dist: Any,
    rank: int,
    local_rank: int,
    world_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if world_size != args.expected_world_size or world_size != EXPECTED_WORLD_SIZE:
        raise Wan22BatchError(
            f"this runner requires exactly eight cooperative ranks; "
            f"expected={args.expected_world_size} actual={world_size}"
        )
    if not torch_module.cuda.is_available():
        raise Wan22BatchError("torch.cuda is unavailable (ROCm uses this API too)")
    if local_rank < 0 or local_rank >= torch_module.cuda.device_count():
        raise Wan22BatchError(
            f"LOCAL_RANK={local_rank} exceeds visible devices="
            f"{torch_module.cuda.device_count()}"
        )
    torch_module.cuda.set_device(local_rank)
    properties = torch_module.cuda.get_device_properties(local_rank)
    local_gpu = {
        "rank": rank,
        "local_rank": local_rank,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "visible_device_count": torch_module.cuda.device_count(),
    }
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_gpu)
    if args.expected_gpu_name_substring:
        mismatched = [
            item
            for item in gathered
            if args.expected_gpu_name_substring.casefold()
            not in str(item["name"]).casefold()
        ]
        if mismatched:
            raise Wan22BatchError(
                "GPU name preflight mismatch: "
                f"expected substring={args.expected_gpu_name_substring!r} "
                f"devices={gathered}"
            )
    hip_version = getattr(torch_module.version, "hip", None)
    if args.require_rocm and not hip_version:
        raise Wan22BatchError(
            "--require-rocm was set but torch.version.hip is empty"
        )
    return {
        "torch_version": torch_module.__version__,
        "torch_hip_version": hip_version,
        "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        "distributed_backend": dist.get_backend(),
        "world_size": world_size,
        "gpu_ranks": gathered,
    }


def _validate_args(args: argparse.Namespace) -> None:
    for label in ("manifest", "output_root", "wan_code_root", "ckpt_dir"):
        value = Path(getattr(args, label)).expanduser()
        if not value.is_absolute():
            raise Wan22BatchError(f"--{label.replace('_', '-')} must be absolute")
    if args.data_root is not None and not Path(args.data_root).is_absolute():
        raise Wan22BatchError("--data-root must be absolute")
    if args.signed_release is not None and not Path(
        args.signed_release
    ).is_absolute():
        raise Wan22BatchError("--signed-release must be absolute")
    preview_mode = bool(getattr(args, "non_production_preview", False))
    if preview_mode and args.signed_release is not None:
        raise Wan22BatchError(
            "--non-production-preview and --signed-release are mutually "
            "exclusive"
        )
    if args.allow_pending_review:
        raise Wan22BatchError(
            "--allow-pending-review is disabled for production generation; "
            "the source-anchored signed eight-row release is required"
        )
    if args.model_hf_revision != MODEL_HF_REVISION:
        raise Wan22BatchError(
            f"model revision must be pinned to {MODEL_HF_REVISION}"
        )
    if args.official_commit != OFFICIAL_COMMIT:
        raise Wan22BatchError(
            f"official commit must be pinned to {OFFICIAL_COMMIT}"
        )
    if args.size not in SUPPORTED_MAX_AREAS:
        raise Wan22BatchError(f"unsupported I2V size: {args.size}")
    if args.frame_num <= 0 or (args.frame_num - 1) % 4 != 0:
        raise Wan22BatchError("frame_num must be positive and of the form 4n+1")
    if preview_mode and args.frame_num != DEFAULT_FRAME_NUM:
        raise Wan22BatchError(
            "non-production full-motion preview requires exactly 81 frames"
        )
    if preview_mode and args.max_samples not in (None, 1):
        raise Wan22BatchError(
            "non-production preview accepts exactly one manifest row"
        )
    if args.sample_steps <= 0:
        raise Wan22BatchError("sample_steps must be positive")
    if args.sample_shift <= 0:
        raise Wan22BatchError("sample_shift must be positive")
    if args.sample_solver not in {"unipc", "dpm++"}:
        raise Wan22BatchError("sample_solver must be unipc or dpm++")
    if (
        args.sample_guide_scale_low <= 0
        or args.sample_guide_scale_high <= 0
    ):
        raise Wan22BatchError("guide scales must be positive")
    if type(args.base_seed) is not int or args.base_seed < 0:
        raise Wan22BatchError("base_seed must be non-negative")
    if args.expected_world_size != EXPECTED_WORLD_SIZE:
        raise Wan22BatchError("expected_world_size is frozen to eight")
    if args.max_new_samples is not None and args.max_new_samples <= 0:
        raise Wan22BatchError("max_new_samples must be positive when set")
    if args.video_quality < 0 or args.video_quality > 10:
        raise Wan22BatchError("video_quality must be in [0, 10]")


def _load_manifest_for_args(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "non_production_preview", False)):
        return load_non_production_preview_manifest(
            args.manifest,
            allow_pending_review=args.allow_pending_review,
            max_samples=args.max_samples,
        )
    return load_generation_manifest(
        args.manifest,
        allow_pending_review=args.allow_pending_review,
        max_samples=args.max_samples,
        signed_release_path=args.signed_release,
    )


def run_batch(args: argparse.Namespace) -> dict[str, Any] | None:
    _validate_args(args)
    # This must remain before Torch, Wan, checkpoint, CUDA, and distributed
    # initialization. No legacy manifest result may reach model loading.
    _load_manifest_for_args(args)
    # Keep the pinned official checkout clean even when a caller forgot the
    # equivalent environment variable used by the AUH Slurm scripts.
    sys.dont_write_bytecode = True
    # Import only after CLI validation; pure contract tests remain Torch-free.
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if rank < 0 or local_rank < 0 or world_size <= 0:
        raise Wan22BatchError(
            "launch with torch.distributed.run/torchrun; RANK, LOCAL_RANK, "
            "and WORLD_SIZE are required"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    pipeline = None
    try:
        gpu_runtime = _distributed_runtime_preflight(
            torch_module=torch,
            dist=dist,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            args=args,
        )
        official_runtime, _ = _collective_local_call(
            dist,
            rank=rank,
            world_size=world_size,
            stage="official checkout preflight",
            producer=lambda: inspect_official_checkout(
                args.wan_code_root,
                expected_commit=args.official_commit,
            ),
        )
        package_runtime, _ = _collective_local_call(
            dist,
            rank=rank,
            world_size=world_size,
            stage="Python dependency preflight",
            producer=inspect_python_packages,
        )
        i2v_modules, _ = _collective_local_call(
            dist,
            rank=rank,
            world_size=world_size,
            stage="official I2V module import",
            producer=lambda: load_official_i2v_modules(args.wan_code_root),
        )
        _, flash_smokes = _collective_local_call(
            dist,
            rank=rank,
            world_size=world_size,
            stage="per-rank FlashAttention kernel smoke",
            producer=lambda: _flash_attention_kernel_smoke(
                torch_module=torch,
                attention_module=i2v_modules["attention_module"],
                rank=rank,
                local_rank=local_rank,
            ),
            gather_values=True,
        )
        gpu_runtime["flash_attention_kernel_smoke"] = {
            "required_on_every_rank": True,
            "completed_rank_count": len(flash_smokes or []),
            "ranks": flash_smokes,
        }
        package_runtime["official_i2v_import"] = {
            "module_paths": i2v_modules["module_paths"],
            "root_initializer_executed": i2v_modules[
                "root_initializer_executed"
            ],
        }

        data_root = (
            _regular_directory(Path(args.data_root), context="data root")
            if args.data_root is not None
            else None
        )

        def rank0_prepare() -> dict[str, Any]:
            manifest = _load_manifest_for_args(args)
            prepared, temporal_policy = _prepare_media_rows(
                manifest,
                data_root=data_root,
                ffprobe=args.ffprobe,
                expected_frame_num=args.frame_num,
            )
            model = inspect_hf_model_directory(
                args.ckpt_dir,
                expected_revision=args.model_hf_revision,
            )
            runtime = dict(package_runtime)
            runtime.update(gpu_runtime)
            contract = build_run_contract(
                manifest=manifest,
                prepared_rows=prepared,
                temporal_policy=temporal_policy,
                args=args,
                official=official_runtime,
                model=model,
                runtime=runtime,
            )
            output_root = Path(args.output_root).resolve()
            ensure_run_contract(output_root, contract)
            pending, completed = _validate_existing_samples(
                output_root,
                rows=prepared,
                contract=contract,
            )
            return {
                "manifest": manifest,
                "prepared_rows": prepared,
                "contract": contract,
                "pending": pending,
                "completed": completed,
            }

        payload = _broadcast_rank0_payload(
            dist,
            rank=rank,
            producer=rank0_prepare,
            stage="rank-zero frozen-input preflight",
        )
        prepared_rows = payload["prepared_rows"]
        contract = payload["contract"]
        pending = [int(index) for index in payload["pending"]]
        completed: dict[int, dict[str, Any]] = {
            int(index): result
            for index, result in payload["completed"].items()
        }
        output_root = Path(args.output_root).resolve()
        if not pending:
            complete = _broadcast_rank0_payload(
                dist,
                rank=rank,
                producer=lambda: publish_completion(
                    output_root,
                    rows=prepared_rows,
                    contract=contract,
                    results=completed,
                ),
                stage="rank-zero run completion",
            )
            return complete if rank == 0 else None
        work_indices = (
            pending[: args.max_new_samples]
            if args.max_new_samples is not None
            else pending
        )

        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        import torchvision.transforms.functional as TF
        WanI2V = i2v_modules["WanI2V"]
        WAN_CONFIGS = i2v_modules["WAN_CONFIGS"]
        MAX_AREA_CONFIGS = i2v_modules["MAX_AREA_CONFIGS"]
        init_distributed_group = i2v_modules["init_distributed_group"]
        cfg = WAN_CONFIGS["i2v-A14B"]
        if cfg.num_heads % world_size != 0:
            raise Wan22BatchError(
                f"Wan heads={cfg.num_heads} are not divisible by world={world_size}"
            )
        if MAX_AREA_CONFIGS[args.size] != SUPPORTED_MAX_AREAS[args.size]:
            raise Wan22BatchError("official Wan max-area config differs")
        if int(cfg.sample_fps) != MODEL_SAMPLE_FPS:
            raise Wan22BatchError(
                f"official model sample FPS changed: "
                f"expected={MODEL_SAMPLE_FPS} "
                f"actual={cfg.sample_fps}"
            )
        init_distributed_group()
        pipeline = WanI2V(
            config=cfg,
            checkpoint_dir=str(Path(args.ckpt_dir).resolve()),
            device_id=local_rank,
            rank=rank,
            t5_fsdp=True,
            dit_fsdp=True,
            use_sp=True,
            t5_cpu=False,
            convert_model_dtype=False,
        )
        dist.barrier()

        for sample_index in work_indices:
            row = prepared_rows[sample_index]
            with Image.open(row["_input_media"]["anchor_path"]) as opened:
                input_image = opened.convert("RGB")
            seed = sample_seed(args.base_seed, row["_iid"])
            video = pipeline.generate(
                row["edit_instruction"],
                input_image,
                max_area=SUPPORTED_MAX_AREAS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=(
                    args.sample_guide_scale_low,
                    args.sample_guide_scale_high,
                ),
                n_prompt="",
                seed=seed,
                offload_model=False,
            )
            result = _broadcast_rank0_payload(
                dist,
                rank=rank,
                producer=lambda: _commit_generated_sample(
                    output_root=output_root,
                    row=row,
                    sample_index=sample_index,
                    contract=contract,
                    video=video,
                    input_image=input_image,
                    args=args,
                    torch_module=torch,
                    torchvision_tf=TF,
                    numpy_module=np,
                    imageio_module=imageio,
                ),
                stage=f"sample commit iid={row['_iid']}",
            )
            if rank == 0:
                completed[sample_index] = result
                print(
                    f"[wan22-batch] committed {sample_index + 1}/"
                    f"{len(prepared_rows)} iid={row['_iid']} seed={seed}",
                    flush=True,
                )
            del video
            torch.cuda.synchronize()
            dist.barrier()

        def rank0_finalize() -> dict[str, Any]:
            pending_after, rescanned = _validate_existing_samples(
                output_root,
                rows=prepared_rows,
                contract=contract,
            )
            if pending_after:
                partial = {
                    "schema_version": "motive-wan22-i2v-chunk-result-v1",
                    "contract_digest": contract["contract_digest"],
                    "committed_this_allocation": len(work_indices),
                    "completed_total": len(rescanned),
                    "remaining_total": len(pending_after),
                    "next_pending_index": pending_after[0],
                    "run_complete": False,
                }
                print(
                    f"[wan22-batch] chunk complete: "
                    f"committed={len(work_indices)} "
                    f"completed_total={len(rescanned)} "
                    f"remaining={len(pending_after)}",
                    flush=True,
                )
                return partial
            return publish_completion(
                output_root,
                rows=prepared_rows,
                contract=contract,
                results=rescanned,
            )

        final_result = _broadcast_rank0_payload(
            dist,
            rank=rank,
            producer=rank0_finalize,
            stage="rank-zero chunk finalization",
        )
        return final_result if rank == 0 else None
    finally:
        if pipeline is not None:
            del pipeline
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recoverable official Wan2.2 I2V batch generation with eight "
            "cooperative FSDP+Ulysses ranks."
        )
    )
    parser.add_argument("--manifest", required=True)
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument(
        "--signed-release",
        help=(
            "Absolute path to an SSH-signed legacy exact-eight release or "
            "a full-motion root release authorizing this contiguous eight-row "
            "manifest shard. "
            "Legacy approvals and manifest booleans are never accepted."
        ),
    )
    authorization.add_argument(
        "--non-production-preview",
        action="store_true",
        help=(
            "Render exactly one deeply validated full-motion v6 row without "
            "a signed release. Every artifact is permanently marked "
            "production_use_forbidden=true."
        ),
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wan-code-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--official-commit", default=OFFICIAL_COMMIT)
    parser.add_argument("--model-hf-revision", default=MODEL_HF_REVISION)
    parser.add_argument("--size", default=DEFAULT_SIZE, choices=SUPPORTED_MAX_AREAS)
    parser.add_argument(
        "--frame-num",
        type=int,
        default=DEFAULT_FRAME_NUM,
        help=(
            "Generated frame count; must equal every source frame count and "
            "be of the form 4n+1. No temporal padding or truncation is done."
        ),
    )
    parser.add_argument("--sample-steps", type=int, default=DEFAULT_SAMPLE_STEPS)
    parser.add_argument("--sample-shift", type=float, default=DEFAULT_SAMPLE_SHIFT)
    parser.add_argument(
        "--sample-solver",
        default="unipc",
        choices=("unipc", "dpm++"),
    )
    parser.add_argument(
        "--sample-guide-scale-low",
        type=float,
        default=DEFAULT_GUIDE_SCALE[0],
    )
    parser.add_argument(
        "--sample-guide-scale-high",
        type=float,
        default=DEFAULT_GUIDE_SCALE[1],
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-new-samples",
        type=int,
        help=(
            "Limit newly committed samples in this allocation without "
            "shortening the manifest contract; intended for chained jobs."
        ),
    )
    parser.add_argument(
        "--allow-pending-review",
        action="store_true",
        help=(
            "Deprecated fail-closed flag. Passing it is always an error; "
            "production generation requires the separately verified signed "
            "eight-row release."
        ),
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=EXPECTED_WORLD_SIZE,
    )
    parser.add_argument("--require-rocm", action="store_true")
    parser.add_argument("--expected-gpu-name-substring", default="")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-quality", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_batch(args)
    except Exception as error:
        print(
            f"[wan22-batch] fatal {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if result is not None:
        print(
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
