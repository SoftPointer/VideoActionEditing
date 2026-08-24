#!/usr/bin/env python3
"""Create one sealed Bernini source-RGB-frame-0 latent for SSFT diagnostics.

This program is intentionally narrower than an edit runner.  It accepts one
row from the frozen SAIC exact81 source manifest, privately decodes only that
row's source video, takes its exact RGB frame 0, and calls the pinned Bernini
``_vae_encode`` exactly once on that one-frame tensor.  It never accepts an
instruction, event bank, target video, external reference frame, adapter,
mask, pose, flow, or motion donor.

The output is a create-only, read-only safetensors artifact plus a canonical
receipt.  Both are non-authoritative inference inputs: neither is ground
truth, a selected target, evidence of action success, nor authorization to
train.  The purpose is solely to let otherwise independent visual-I0 arms
consume one byte-identical, independently encoded frame-0 coordinate without
repeating a VAE encode inside any runner.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import infer_lora as legacy  # noqa: E402


RECEIPT_SCHEMA = "bernini-saic-frame0-latent-receipt-v1"
ARTIFACT_SCHEMA = "bernini-saic-frame0-latent-artifact-v1"
METHOD = "frozen-bernini-frame0-latent-materializer"
TENSOR_KEY = "reference_frame0_latent"
FRAME_COUNT = 81
LATENT_FRAME_COUNT = 1
FPS = 25
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_CONTENT_FILE_COUNT = 23

ARTIFACT_METADATA = {
    "schema_version": ARTIFACT_SCHEMA,
    "coordinate": "bernini_source_rgb_frame0_vae_latent",
    "frame_contract": "source_rgb_index0_latent1",
    "artifact_role": "saic_common_visual_i0_reference_coordinate",
    "source": "sealed_exact81_source_rgb_frame0_wan_vae_mode",
    "posterior": "mode",
    "sampling": "false",
    "authority": "false",
}

# This is the complete local Python/asset closure reached by source-manifest
# validation and ``infer_lora.prepare_exact_source``.  Bernini and Diffusers
# vendor bytes are bound separately below.
RUNTIME_METHOD_FILES = (
    "materialize_saic_frame0_latent_v1.py",
    "build_saic_reversible_source_set_v1.py",
    "infer_lora.py",
    "train_lora.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "assets/saic_reversible_source_set_v1.json",
)
RUNTIME_ARCHIVE_MEMBERS = tuple(
    f"methods/bernini_action_editing/{relative}"
    for relative in RUNTIME_METHOD_FILES
)

ACCEPTED_INPUT_ROLES = (
    "source_manifest",
    "selected_source_video",
    "checkpoint_and_source_code_provenance",
)
FORBIDDEN_INPUT_ROLES = tuple(source_set.EXPECTED_FORBIDDEN_INPUTS) + (
    "natural_language_instruction",
    "source_caption",
    "target_caption",
    "event_bank",
    "branch",
    "rollout_seed",
    "external_reference_frame",
    "shared_i0",
    "adapter",
    "lora",
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")


class Frame0LatentMaterializationError(RuntimeError):
    """Raised before an ambiguous or partially verified artifact is kept."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise Frame0LatentMaterializationError(
            f"value is not canonical-JSON encodable: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_raw_sha256(value: Any) -> str:
    """Hash exact contiguous tensor bytes without semantic interpretation."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH runtime owns torch
        raise Frame0LatentMaterializationError(
            "PyTorch is required to hash tensors"
        ) from error
    if (
        not isinstance(value, torch.Tensor)
        or value.numel() <= 0
        or value.requires_grad
        or value.layout != torch.strided
    ):
        raise Frame0LatentMaterializationError(
            "tensor hash input must be non-empty detached strided data"
        )
    raw = value.detach().contiguous().view(torch.uint8).reshape(-1).cpu()
    digest = hashlib.sha256()
    for offset in range(0, int(raw.numel()), 1024 * 1024):
        part = raw[offset : offset + 1024 * 1024]
        try:
            payload = part.numpy().tobytes(order="C")
        except RuntimeError:
            payload = bytes(part.tolist())
        digest.update(payload)
    return digest.hexdigest()


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise Frame0LatentMaterializationError(
            f"{label} must be an absolute non-symlink file"
        )
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise Frame0LatentMaterializationError(
            f"cannot resolve {label}"
        ) from error
    if not stat.S_ISREG(mode):
        raise Frame0LatentMaterializationError(
            f"{label} must be a plain file"
        )
    return resolved


def _absolute_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise Frame0LatentMaterializationError(
            f"{label} must be an absolute non-symlink directory"
        )
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise Frame0LatentMaterializationError(
            f"cannot resolve {label}"
        ) from error
    if not resolved.is_dir():
        raise Frame0LatentMaterializationError(
            f"{label} must be a directory"
        )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--expected-source-video-sha256", required=True)
    parser.add_argument(
        "--expected-reference-frame0-tensor-raw-sha256",
        required=True,
        help=(
            "Job132387 ephemeral I0 tensor SHA used for an explicit comparison; "
            "the fresh coordinate remains valid and records a false match if "
            "ROCm convolution bytes are not reproducible"
        ),
    )
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive", required=True)
    parser.add_argument("--durable-method-source-archive", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name, None)
        if type(value) is not str or _SHA1.fullmatch(value) is None:
            raise Frame0LatentMaterializationError(
                f"{name} must be a lowercase full SHA-1"
            )
    for name in (
        "expected_source_manifest_sha256",
        "expected_source_video_sha256",
        "expected_reference_frame0_tensor_raw_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        value = getattr(args, name, None)
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise Frame0LatentMaterializationError(
                f"{name} must be a lowercase SHA-256"
            )
    if args.expected_bernini_commit != legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise Frame0LatentMaterializationError(
            "only the audited Bernini source revision is supported"
        )
    if args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise Frame0LatentMaterializationError(
            "only the tested VeOmni source revision is supported"
        )
    if args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise Frame0LatentMaterializationError(
            "only the audited Bernini-R 1.3B checkpoint is supported"
        )
    if (
        type(args.row_id) is not str
        or not args.row_id
        or _SAFE_BASENAME.fullmatch(args.row_id) is None
    ):
        raise Frame0LatentMaterializationError("row_id is unsafe")
    if type(args.device) is not str or re.fullmatch(r"cuda:[0-9]+", args.device) is None:
        raise Frame0LatentMaterializationError(
            "device must be an explicit CUDA device such as cuda:0"
        )


def resolve_output(value: str | Path) -> tuple[Path, Path]:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested.suffix != ".safetensors"
        or _SAFE_BASENAME.fullmatch(requested.name) is None
    ):
        raise Frame0LatentMaterializationError(
            "output must be an absolute safe .safetensors path"
        )
    parent = _absolute_directory(requested.parent, label="output parent")
    output = parent / requested.name
    receipt = output.with_name(f"{output.name}.receipt.json")
    for path, label in ((output, "artifact"), (receipt, "receipt")):
        if path.exists() or path.is_symlink():
            raise Frame0LatentMaterializationError(
                f"refusing to overwrite existing {label}: {path}"
            )
    return output, receipt


def load_sealed_source_row(
    manifest_value: str | Path,
    *,
    expected_raw_sha256: str,
    row_id: str,
    expected_source_video_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable source manifest and select exactly one row."""

    manifest_path = _plain_absolute_file(
        manifest_value, label="sealed source manifest"
    )
    actual_raw_sha256 = file_sha256(manifest_path)
    if actual_raw_sha256 != expected_raw_sha256:
        raise Frame0LatentMaterializationError(
            "sealed source manifest raw SHA-256 differs"
        )
    try:
        manifest = source_set.load_manifest(manifest_path)
        summary = dict(
            source_set.validate_manifest(manifest, verify_bound_files=False)
        )
    except source_set.SAICReversibleSourceSetError as error:
        raise Frame0LatentMaterializationError(str(error)) from error
    rows = [row for row in manifest["rows"] if row.get("row_id") == row_id]
    if len(rows) != 1:
        raise Frame0LatentMaterializationError(
            "row_id does not resolve to exactly one sealed source row"
        )
    row = dict(rows[0])
    if row.get("source_video_sha256") != expected_source_video_sha256:
        raise Frame0LatentMaterializationError(
            "launcher source-video SHA-256 differs from sealed row"
        )
    if (
        row.get("optimizer_eligible") is not False
        or row.get("terminal_state_contract", {}).get("terminal_event_verified")
        is not False
    ):
        raise Frame0LatentMaterializationError(
            "source row unexpectedly carries training/event authority"
        )
    sealed = {
        "accepted_roles": list(ACCEPTED_INPUT_ROLES),
        "forbidden_roles": list(FORBIDDEN_INPUT_ROLES),
        "source_manifest_path": str(manifest_path),
        "source_manifest_raw_sha256": actual_raw_sha256,
        "source_manifest_content_sha256": summary["manifest_content_sha256"],
        "source_manifest_schema_version": manifest["schema_version"],
        "source_manifest_dataset_id": manifest["dataset_id"],
        "source_manifest_bound_files_verified": False,
        "row_id": row["row_id"],
        "iid": row["iid"],
        "analysis_split": row["analysis_split"],
        "actor_family": row["actor_family"],
        "source_video_path": row["source_video"],
        "source_video_sha256": row["source_video_sha256"],
        "source_video_rehashed_after_encode": True,
        "source_manifest_terminal_events_verified": False,
        "optimizer_authorized": False,
    }
    return row, sealed


def prepare_private_exact_source(
    source_path: Path,
    *,
    prepare_fn: Callable[[Path], tuple[Any, Mapping[str, Any]]] = legacy.prepare_exact_source,
) -> tuple[Any, dict[str, Any], str]:
    """Decode an exact byte copy, then prove the named source stayed fixed."""

    source = _plain_absolute_file(source_path, label="selected source video")
    before = source.stat()
    before_sha256 = file_sha256(source)
    with tempfile.TemporaryDirectory(prefix="saic-clean-source-snapshot-") as root:
        snapshot = Path(root) / "source.mp4"
        shutil.copyfile(source, snapshot)
        if file_sha256(snapshot) != before_sha256:
            raise Frame0LatentMaterializationError(
                "private source snapshot digest differs"
            )
        try:
            source_tensor, metadata_value = prepare_fn(snapshot)
        except Exception as error:
            raise Frame0LatentMaterializationError(
                f"private exact81 source decode failed: {error}"
            ) from error
    after = source.stat()
    after_sha256 = file_sha256(source)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or after_sha256 != before_sha256:
        raise Frame0LatentMaterializationError(
            "source video changed during private decode"
        )
    metadata = dict(metadata_value)
    if (
        metadata.get("frame_count") != FRAME_COUNT
        or float(metadata.get("fps", -1)) != float(FPS)
        or metadata.get("temporal_policy")
        != "all_integer_frames_0_through_80_no_subsampling"
    ):
        raise Frame0LatentMaterializationError(
            "private decode is not the pinned exact81/25 policy"
        )
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise Frame0LatentMaterializationError("PyTorch is required") from error
    bucket = metadata.get("source_derived_bucket_hw")
    if (
        not isinstance(source_tensor, torch.Tensor)
        or source_tensor.dtype != torch.float32
        or source_tensor.requires_grad
        or source_tensor.ndim != 5
        or tuple(int(item) for item in source_tensor.shape[:3]) != (1, 3, 81)
        or type(bucket) is not list
        or len(bucket) != 2
        or tuple(int(item) for item in source_tensor.shape[-2:]) != tuple(bucket)
    ):
        raise Frame0LatentMaterializationError(
            "private decoded source tensor geometry/dtype differs"
        )
    metadata["decoded_from_private_byte_snapshot"] = True
    return source_tensor.contiguous(), metadata, before_sha256


def validate_checkpoint_content(
    checkpoint: Path,
    manifest_value: str | Path,
    *,
    expected_manifest_sha256: str = CHECKPOINT_CONTENT_MANIFEST_SHA256,
    expected_file_count: int = CHECKPOINT_CONTENT_FILE_COUNT,
) -> dict[str, Any]:
    """Hash every non-cache checkpoint file against the pinned manifest."""

    root = _absolute_directory(checkpoint, label="checkpoint content root")
    manifest = _plain_absolute_file(
        manifest_value, label="checkpoint content manifest"
    )
    manifest_sha256 = file_sha256(manifest)
    if manifest_sha256 != expected_manifest_sha256:
        raise Frame0LatentMaterializationError(
            "checkpoint content manifest SHA-256 differs"
        )
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise Frame0LatentMaterializationError(
            "cannot read checkpoint content manifest"
        ) from error
    if len(lines) != expected_file_count:
        raise Frame0LatentMaterializationError(
            "checkpoint content manifest file count differs"
        )
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise Frame0LatentMaterializationError(
                "checkpoint manifest line is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise Frame0LatentMaterializationError(
                "checkpoint manifest contains an unsafe path"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise Frame0LatentMaterializationError(
                "checkpoint manifest contains an empty/duplicate path"
            )
        expected[normalized] = digest

    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise Frame0LatentMaterializationError(
                "checkpoint contains a non-cache symlink"
            )
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise Frame0LatentMaterializationError(
                "checkpoint contains a non-regular filesystem entry"
            )
    if actual != set(expected):
        raise Frame0LatentMaterializationError(
            "checkpoint non-cache file set differs from pinned manifest"
        )
    rows = []
    for relative in sorted(expected):
        path = root / relative
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected[relative]:
            raise Frame0LatentMaterializationError(
                f"checkpoint content hash differs: {relative}"
            )
        rows.append({"path": relative, "sha256": actual_sha256})
    return {
        "manifest_path": str(manifest),
        "manifest_sha256_computed": manifest_sha256,
        "manifest_sha256_expected": expected_manifest_sha256,
        "verified_file_count": len(rows),
        "every_file_sha256_verified": True,
        "verified_entries_digest": object_sha256(rows),
    }


def _bytecode_policy() -> dict[str, Any]:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise Frame0LatentMaterializationError(
            "materializer requires PYTHONDONTWRITEBYTECODE=1 before Python starts"
        )
    configured_value = os.environ.get("PYTHONPYCACHEPREFIX")
    if not configured_value or sys.pycache_prefix is None:
        raise Frame0LatentMaterializationError(
            "materializer requires a private empty PYTHONPYCACHEPREFIX"
        )
    configured = Path(configured_value).expanduser()
    runtime = Path(sys.pycache_prefix).expanduser()
    if not configured.is_absolute() or configured != runtime:
        raise Frame0LatentMaterializationError(
            "PYTHONPYCACHEPREFIX must equal sys.pycache_prefix"
        )
    if runtime.is_symlink() or not runtime.is_dir() or any(runtime.rglob("*")):
        raise Frame0LatentMaterializationError(
            "private pycache prefix must be a non-symlink empty directory"
        )
    resolved = runtime.resolve(strict=True)
    method = METHOD_ROOT.resolve(strict=True)
    if resolved == method or method in resolved.parents or resolved in method.parents:
        raise Frame0LatentMaterializationError(
            "private pycache prefix overlaps method source"
        )
    return {
        "pythondontwritebytecode_environment": "1",
        "dont_write_bytecode": True,
        "pythonpycacheprefix_environment": configured_value,
        "runtime_pycache_prefix": str(runtime),
        "resolved_private_empty_pycache_prefix": str(resolved),
        "method_source_pycache_ignored": True,
    }


def runtime_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in RUNTIME_METHOD_FILES:
        path = METHOD_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise Frame0LatentMaterializationError(
                f"runtime source is missing/non-plain: {relative}"
            )
        result[f"methods/bernini_action_editing/{relative}"] = file_sha256(path)
    return result


def validate_method_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Bind live runtime bytes to two equal, revision-labelled safe archives."""

    bytecode_policy = _bytecode_policy()
    live = runtime_source_hashes()
    scratch = _plain_absolute_file(
        args.method_source_archive, label="scratch method archive"
    )
    durable = _plain_absolute_file(
        args.durable_method_source_archive, label="durable method archive"
    )
    scratch_sha256 = file_sha256(scratch)
    durable_sha256 = file_sha256(durable)
    if (
        scratch_sha256 != args.method_source_archive_sha256
        or durable_sha256 != scratch_sha256
    ):
        raise Frame0LatentMaterializationError(
            "scratch/durable method archive digest differs"
        )
    member_hashes: dict[str, str] = {}
    try:
        with tarfile.open(scratch, mode="r:*") as handle:
            if handle.pax_headers.get("comment") != args.method_source_revision:
                raise Frame0LatentMaterializationError(
                    "method archive revision comment differs"
                )
            members = handle.getmembers()
            seen: set[str] = set()
            for member in members:
                pure = PurePosixPath(member.name)
                name = pure.as_posix().rstrip("/")
                scoped = name in {
                    "methods",
                    "methods/bernini_action_editing",
                } or name.startswith("methods/bernini_action_editing/")
                if (
                    not name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or not scoped
                    or name in seen
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                    or (not member.isfile() and not member.isdir())
                ):
                    raise Frame0LatentMaterializationError(
                        "method archive has unsafe/duplicate/out-of-scope content"
                    )
                seen.add(name)
            for relative in RUNTIME_ARCHIVE_MEMBERS:
                matches = [item for item in members if item.name == relative]
                if len(matches) != 1 or not matches[0].isfile():
                    raise Frame0LatentMaterializationError(
                        f"method archive member differs: {relative}"
                    )
                extracted = handle.extractfile(matches[0])
                if extracted is None:
                    raise Frame0LatentMaterializationError(
                        f"cannot read method archive member: {relative}"
                    )
                member_hashes[relative] = hashlib.sha256(
                    extracted.read()
                ).hexdigest()
    except Frame0LatentMaterializationError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise Frame0LatentMaterializationError(
            "cannot validate method source archive"
        ) from error
    if member_hashes != live:
        raise Frame0LatentMaterializationError(
            "live runtime bytes differ from method archive"
        )
    return {
        "revision": args.method_source_revision,
        "scratch_archive_path": str(scratch),
        "durable_archive_path": str(durable),
        "archive_sha256": scratch_sha256,
        "archive_safe_scoped_duplicate_free_link_free": True,
        "revision_label_matches_archive_comment": True,
        "git_revision_verified_by_runner": False,
        "runtime_source_sha256": live,
        "runtime_source_index_sha256": object_sha256(live),
        "bytecode_policy": bytecode_policy,
    }


def _reject_preloaded_bernini() -> None:
    poisoned = sorted(
        name
        for name in sys.modules
        if name == "bernini" or name.startswith("bernini.")
    )
    if poisoned:
        raise Frame0LatentMaterializationError(
            f"Bernini modules were loaded before source activation: {poisoned[:4]}"
        )


def validate_encoder_callable(
    encoder: Callable[..., Any],
    bernini_root: Path,
    *,
    expected_pipeline_sha256: str = legacy.BERNINI_INFERENCE_FILE_HASHES[
        "bernini/pipeline.py"
    ],
) -> dict[str, str]:
    """Prove the callable comes from the pinned Bernini pipeline file."""

    if not callable(encoder):
        raise Frame0LatentMaterializationError("_vae_encode is not callable")
    if (
        getattr(encoder, "__module__", None) != "bernini.pipeline"
        or getattr(encoder, "__name__", None) != "_vae_encode"
        or getattr(encoder, "__qualname__", None) != "_vae_encode"
    ):
        raise Frame0LatentMaterializationError(
            "Bernini _vae_encode callable identity differs"
        )
    expected_source = (bernini_root / "bernini/pipeline.py").resolve(strict=True)
    try:
        source_file = Path(inspect.getsourcefile(encoder) or "").resolve(strict=True)
        code_file = Path(encoder.__code__.co_filename).resolve(strict=True)
        signature = inspect.signature(encoder)
        bound = signature.bind(object(), object())
    except (OSError, TypeError, ValueError) as error:
        raise Frame0LatentMaterializationError(
            "cannot bind pinned Bernini _vae_encode callable"
        ) from error
    if source_file != expected_source or code_file != expected_source:
        raise Frame0LatentMaterializationError(
            "_vae_encode source/code file is outside pinned Bernini"
        )
    if file_sha256(expected_source) != expected_pipeline_sha256:
        raise Frame0LatentMaterializationError(
            "_vae_encode pipeline source SHA-256 differs"
        )
    if len(bound.arguments) != 2 or any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    ):
        raise Frame0LatentMaterializationError(
            "_vae_encode callable signature is not the pinned two-input form"
        )
    return {
        "encoder_symbol": "bernini.pipeline._vae_encode",
        "callable_module": encoder.__module__,
        "callable_name": encoder.__name__,
        "callable_qualname": encoder.__qualname__,
        "callable_signature": str(signature),
    }


def encode_source_frame0_once(
    vae: Any,
    source_pixels: Any,
    *,
    encoder: Callable[[Any, Any], Any],
) -> tuple[Any, dict[str, Any]]:
    """Encode exact source RGB index 0 once and reject mutation/shape drift."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise Frame0LatentMaterializationError("PyTorch is required") from error
    if (
        not isinstance(source_pixels, torch.Tensor)
        or source_pixels.dtype != torch.float32
        or source_pixels.requires_grad
        or source_pixels.ndim != 5
        or tuple(int(item) for item in source_pixels.shape[:3]) != (1, 3, 81)
        or not source_pixels.is_contiguous()
    ):
        raise Frame0LatentMaterializationError(
            "encoder input must be contiguous FP32 [1,3,81,H,W]"
        )
    before_sha256 = tensor_raw_sha256(source_pixels)
    frame0_pixels = source_pixels[:, :, 0:1, :, :].contiguous()
    frame0_before_sha256 = tensor_raw_sha256(frame0_pixels)
    invocation_count = 0
    with torch.inference_mode():
        invocation_count += 1
        latent = encoder(vae, frame0_pixels)
    frame0_after_sha256 = tensor_raw_sha256(frame0_pixels)
    after_sha256 = tensor_raw_sha256(source_pixels)
    if invocation_count != 1:
        raise Frame0LatentMaterializationError(
            "source-frame0 _vae_encode invocation count differs"
        )
    if (
        before_sha256 != after_sha256
        or frame0_before_sha256 != frame0_after_sha256
    ):
        raise Frame0LatentMaterializationError(
            "_vae_encode mutated the source pixel tensor"
        )
    z_dim = int(getattr(getattr(vae, "config", None), "z_dim", -1))
    expected_shape = (
        1,
        z_dim,
        1,
        int(source_pixels.shape[-2]) // 8,
        int(source_pixels.shape[-1]) // 8,
    )
    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.requires_grad
        or latent.grad_fn is not None
        or tuple(int(item) for item in latent.shape) != expected_shape
        or tuple(int(item) for item in latent.shape[:3]) != (1, 16, 1)
    ):
        raise Frame0LatentMaterializationError(
            "frame0 latent geometry/dtype/autograd state differs"
        )
    latent = latent.detach().contiguous()
    if not bool(torch.isfinite(latent).all().item()):
        raise Frame0LatentMaterializationError(
            "frame0 latent contains non-finite values"
        )
    return latent, {
        "encoded_in_runner": False,
        "full_source_vae_encode_count": 0,
        "source_frame0_vae_encode_count": 1,
        "total_vae_encode_count": 1,
        "posterior_statistic": "latent_dist.mode",
        "sampling": False,
        "torch_inference_mode": True,
        "source_pixels_mutated": False,
        "source_pixels_before_sha256": before_sha256,
        "source_pixels_after_sha256": after_sha256,
        "source_frame0_pixels_raw_sha256": frame0_before_sha256,
        "source_frame0_pixels_after_sha256": frame0_after_sha256,
        "source_rgb_indices": [0],
        "temporal_video_latent_slice_used": False,
        "vae_dtype": "torch.float32",
        "vae_eval": getattr(vae, "training", None) is False,
        "vae_requires_grad": any(
            bool(parameter.requires_grad) for parameter in vae.parameters()
        ),
        "latent_frame_count": LATENT_FRAME_COUNT,
        "finite": True,
    }


def artifact_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Frame0LatentMaterializationError(
            "published artifact is not a plain file"
        )
    info = path.stat()
    return {
        "path": str(path),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size": int(info.st_size),
        "sha256": file_sha256(path),
    }


def unlink_owned_artifact(
    path: Path, identity: Optional[Mapping[str, Any]]
) -> bool:
    """Delete only the exact inode and bytes created by this transaction."""

    if identity is None or (not path.exists() and not path.is_symlink()):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    current = artifact_identity(path)
    keys = ("path", "device", "inode", "size", "sha256")
    if any(current.get(key) != identity.get(key) for key in keys):
        return False
    path.unlink()
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_owned_bytes(
    final_path: Path,
    payload: bytes,
    *,
    transaction_token: str,
    role: str,
) -> dict[str, Any]:
    if (
        type(transaction_token) is not str
        or _SAFE_BASENAME.fullmatch(transaction_token) is None
    ):
        raise Frame0LatentMaterializationError(
            "transaction token is unsafe"
        )
    if final_path.exists() or final_path.is_symlink():
        raise Frame0LatentMaterializationError(
            f"refusing to overwrite existing {role}"
        )
    temporary = final_path.with_name(
        f".{final_path.name}.frame0-tmp-{transaction_token}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise Frame0LatentMaterializationError(
            f"stale {role} temporary exists"
        )
    descriptor: Optional[int] = None
    temporary_identity: Optional[dict[str, Any]] = None
    linked_identity: Optional[dict[str, Any]] = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_identity = artifact_identity(temporary)
        os.link(temporary, final_path)
        linked_identity = {**temporary_identity, "path": str(final_path)}
        observed = artifact_identity(final_path)
        if observed != linked_identity:
            raise Frame0LatentMaterializationError(
                f"published {role} identity differs"
            )
        _fsync_directory(final_path.parent)
        return observed
    except BaseException:
        unlink_owned_artifact(final_path, linked_identity)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        unlink_owned_artifact(temporary, temporary_identity)


def _publish_reference_frame0_latent_owned(
    latent: Any,
    path: Path,
    *,
    transaction_token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save as save_safetensors

    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.requires_grad
        or latent.grad_fn is not None
        or latent.ndim != 5
        or tuple(int(item) for item in latent.shape[:3]) != (1, 16, 1)
    ):
        raise Frame0LatentMaterializationError(
            "published latent must be detached FP32 [1,16,1,H,W]"
        )
    stored = latent.detach().cpu().contiguous()
    if not bool(torch.isfinite(stored).all().item()):
        raise Frame0LatentMaterializationError(
            "published latent contains non-finite values"
        )
    payload = save_safetensors(
        {TENSOR_KEY: stored}, metadata=dict(ARTIFACT_METADATA)
    )
    if type(payload) is not bytes or not payload:
        raise Frame0LatentMaterializationError(
            "safetensors serializer returned invalid bytes"
        )
    owned = _write_owned_bytes(
        path,
        payload,
        transaction_token=transaction_token,
        role="frame0 latent",
    )
    try:
        os.chmod(path, 0o444)
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise Frame0LatentMaterializationError(
                "frame0 latent is not mode 0444"
            )
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [TENSOR_KEY]:
                raise Frame0LatentMaterializationError(
                    "frame0 safetensors key differs"
                )
            reopened = opened.get_tensor(TENSOR_KEY).contiguous()
            metadata = dict(opened.metadata() or {})
        raw_sha256 = tensor_raw_sha256(reopened)
        if (
            metadata != ARTIFACT_METADATA
            or reopened.dtype != torch.float32
            or tuple(reopened.shape) != tuple(stored.shape)
            or not torch.equal(reopened, stored)
            or raw_sha256 != tensor_raw_sha256(stored)
        ):
            raise Frame0LatentMaterializationError(
                "frame0 safetensors reopen differs"
            )
        sealed_owned = artifact_identity(path)
        identity_keys = ("path", "device", "inode", "size", "sha256")
        if any(sealed_owned[key] != owned[key] for key in identity_keys):
            raise Frame0LatentMaterializationError(
                "frame0 identity changed while sealing"
            )
        public = {
            "schema_version": ARTIFACT_SCHEMA,
            "path": str(path),
            "file_sha256": sealed_owned["sha256"],
            "size_bytes": sealed_owned["size"],
            "mode": "0444",
            "tensor_key": TENSOR_KEY,
            "tensor_raw_sha256": raw_sha256,
            "shape": [int(item) for item in reopened.shape],
            "dtype": str(reopened.dtype),
            "metadata": dict(metadata),
        }
        _fsync_directory(path.parent)
        return sealed_owned, public
    except BaseException:
        unlink_owned_artifact(path, owned)
        raise


def build_receipt(
    *,
    artifact: Mapping[str, Any],
    sealed_inputs: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    model_closure: Mapping[str, Any],
    encoding: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    authority = {
        "quality_claim_authorized": False,
        "semantic_action_success_authorized": False,
        "ground_truth_authorized": False,
        "training_target_authorized": False,
        "selection_authorized": False,
        "optimizer_step_authorized": False,
        "checkpoint_or_lora_artifact": False,
        "production_claim_authorized": False,
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD,
        "artifact": dict(artifact),
        "sealed_inputs": dict(sealed_inputs),
        "preprocessing": dict(preprocessing),
        "model_closure": dict(model_closure),
        "encoding": dict(encoding),
        "runtime": dict(runtime),
        "authority": authority,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def _publish_receipt_owned(
    receipt: Mapping[str, Any],
    path: Path,
    *,
    transaction_token: str,
) -> dict[str, Any]:
    payload = canonical_json_bytes(receipt) + b"\n"
    owned = _write_owned_bytes(
        path,
        payload,
        transaction_token=transaction_token,
        role="frame0 receipt",
    )
    try:
        os.chmod(path, 0o444)
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise Frame0LatentMaterializationError(
                "frame0 receipt is not mode 0444"
            )
        raw = path.read_bytes()
        reopened = json.loads(raw.decode("ascii"))
        if raw != canonical_json_bytes(reopened) + b"\n":
            raise Frame0LatentMaterializationError(
                "frame0 receipt bytes are not canonical"
            )
        unsigned = dict(reopened)
        declared = unsigned.pop("receipt_digest", None)
        if (
            type(declared) is not str
            or _SHA256.fullmatch(declared) is None
            or object_sha256(unsigned) != declared
        ):
            raise Frame0LatentMaterializationError(
                "frame0 receipt digest differs"
            )
        sealed = artifact_identity(path)
        if any(
            sealed[key] != owned[key]
            for key in ("path", "device", "inode", "size", "sha256")
        ):
            raise Frame0LatentMaterializationError(
                "receipt identity changed while sealing"
            )
        _fsync_directory(path.parent)
        return sealed
    except BaseException:
        unlink_owned_artifact(path, owned)
        raise


def reopen_published_bundle(
    output_path: Path,
    receipt_path: Path,
    *,
    expected_artifact_identity: Mapping[str, Any],
    expected_receipt_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Terminally reopen both 0444 files and verify every byte binding."""

    import torch
    from safetensors import safe_open

    observed: dict[str, dict[str, Any]] = {}
    for path, expected, label in (
        (output_path, expected_artifact_identity, "artifact"),
        (receipt_path, expected_receipt_identity, "receipt"),
    ):
        if path.is_symlink() or not path.is_file():
            raise Frame0LatentMaterializationError(
                f"terminal {label} is not a plain file"
            )
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise Frame0LatentMaterializationError(
                f"terminal {label} mode differs"
            )
        identity = artifact_identity(path)
        if any(
            identity[key] != expected[key]
            for key in ("path", "device", "inode", "size", "sha256")
        ):
            raise Frame0LatentMaterializationError(
                f"terminal {label} identity differs"
            )
        observed[label] = identity
    try:
        with safe_open(str(output_path), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [TENSOR_KEY]:
                raise Frame0LatentMaterializationError(
                    "terminal safetensors key differs"
                )
            latent = opened.get_tensor(TENSOR_KEY).contiguous()
            metadata = dict(opened.metadata() or {})
    except Frame0LatentMaterializationError:
        raise
    except Exception as error:
        raise Frame0LatentMaterializationError(
            "cannot terminal-reopen frame0 latent"
        ) from error
    raw_receipt = receipt_path.read_bytes()
    try:
        receipt = json.loads(raw_receipt.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Frame0LatentMaterializationError(
            "cannot terminal-reopen frame0 receipt"
        ) from error
    if raw_receipt != canonical_json_bytes(receipt) + b"\n":
        raise Frame0LatentMaterializationError(
            "terminal receipt is not canonical JSON"
        )
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    artifact = receipt.get("artifact", {})
    tensor_sha256 = tensor_raw_sha256(latent)
    if (
        metadata != ARTIFACT_METADATA
        or latent.dtype != torch.float32
        or object_sha256(unsigned) != declared
        or artifact.get("file_sha256") != observed["artifact"]["sha256"]
        or artifact.get("tensor_raw_sha256") != tensor_sha256
        or artifact.get("mode") != "0444"
        or artifact.get("metadata") != ARTIFACT_METADATA
    ):
        raise Frame0LatentMaterializationError(
            "terminal artifact/receipt content binding differs"
        )
    return {
        "artifact_file_sha256": observed["artifact"]["sha256"],
        "artifact_tensor_raw_sha256": tensor_sha256,
        "receipt_file_sha256": observed["receipt"]["sha256"],
        "receipt_content_sha256": declared,
        "artifact_mode": "0444",
        "receipt_mode": "0444",
        "canonical_receipt_verified": True,
        "false_authority_verified": all(
            value is False for value in receipt.get("authority", {}).values()
        ),
    }


def publish_materialization_bundle(
    latent: Any,
    output_path: Path,
    receipt_path: Path,
    *,
    sealed_inputs: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    model_closure: Mapping[str, Any],
    encoding: Mapping[str, Any],
    runtime: Mapping[str, Any],
    transaction_token: str,
) -> dict[str, Any]:
    """Atomically create, seal, and reopen the artifact/receipt pair."""

    artifact_owned: Optional[dict[str, Any]] = None
    receipt_owned: Optional[dict[str, Any]] = None
    try:
        artifact_owned, artifact_public = _publish_reference_frame0_latent_owned(
            latent, output_path, transaction_token=transaction_token
        )
        receipt = build_receipt(
            artifact=artifact_public,
            sealed_inputs=sealed_inputs,
            preprocessing=preprocessing,
            model_closure=model_closure,
            encoding=encoding,
            runtime=runtime,
        )
        receipt_owned = _publish_receipt_owned(
            receipt, receipt_path, transaction_token=transaction_token
        )
        terminal = reopen_published_bundle(
            output_path,
            receipt_path,
            expected_artifact_identity=artifact_owned,
            expected_receipt_identity=receipt_owned,
        )
        return {
            "artifact": artifact_public,
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": receipt_owned["sha256"],
            "receipt_content_sha256": receipt["receipt_digest"],
            "terminal_verification": terminal,
        }
    except BaseException:
        unlink_owned_artifact(receipt_path, receipt_owned)
        unlink_owned_artifact(output_path, artifact_owned)
        raise


def _preprocessing_receipt(
    metadata: Mapping[str, Any], source_pixels: Any
) -> dict[str, Any]:
    return {
        "decoded_from_private_byte_snapshot": bool(
            metadata["decoded_from_private_byte_snapshot"]
        ),
        "frame_count": int(metadata["frame_count"]),
        "fps": int(FPS),
        "reported_fps": float(metadata["reported_fps"]),
        "source_input_hw": list(metadata["source_input_hw"]),
        "source_derived_bucket_hw": list(metadata["source_derived_bucket_hw"]),
        "max_pixels": int(metadata["max_pixels"]),
        "stride": int(metadata["stride"]),
        "temporal_policy": metadata["temporal_policy"],
        "spatial_policy": metadata["spatial_policy"],
        "resize": metadata["resize"],
        "external_shared_i0": bool(metadata["external_shared_i0"]),
        "source_pixels_shape": [int(item) for item in source_pixels.shape],
        "source_pixels_dtype": str(source_pixels.dtype),
        "source_pixels_raw_sha256": tensor_raw_sha256(source_pixels),
    }


def materialize_reference_frame0_latent(args: argparse.Namespace) -> dict[str, Any]:
    """Launcher-facing API implementing one complete materialization."""

    validate_cli(args)
    output_path, receipt_path = resolve_output(args.output)
    method_pre = validate_method_provenance(args)
    row, sealed_inputs = load_sealed_source_row(
        args.source_manifest,
        expected_raw_sha256=args.expected_source_manifest_sha256,
        row_id=args.row_id,
        expected_source_video_sha256=args.expected_source_video_sha256,
    )
    source_path = _plain_absolute_file(
        row["source_video"], label="sealed selected source video"
    )
    # The consumer also resolves the selected source before comparing the
    # receipt, so bind the canonical plain-file path rather than an ancestor-
    # symlink spelling from the manifest.
    sealed_inputs["source_video_path"] = str(source_path)
    source_tensor, source_metadata, source_sha256 = prepare_private_exact_source(
        source_path
    )
    if source_sha256 != row["source_video_sha256"]:
        raise Frame0LatentMaterializationError(
            "selected source bytes differ from sealed row"
        )

    _reject_preloaded_bernini()
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
        bernini_inference_files = legacy.validate_inference_source_files(
            bernini_root
        )
    except (legacy.trainer.TrainingContractError, legacy.InferenceContractError) as error:
        raise Frame0LatentMaterializationError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) != 12:
        raise Frame0LatentMaterializationError(
            "checkpoint is not the pinned Bernini-R 1.3B transformer"
        )
    checkpoint_pre = validate_checkpoint_content(
        checkpoint, args.checkpoint_content_manifest
    )
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from safetensors import __version__ as safetensors_version
    from bernini.pipeline import _vae_encode

    if os.environ.get("WORLD_SIZE", "1") != "1" or dist.is_initialized():
        raise Frame0LatentMaterializationError(
            "materializer must run as one non-distributed process"
        )
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise Frame0LatentMaterializationError(
            "materializer requires an AUH ROCm-visible GPU"
        )
    device = torch.device(args.device)
    if device.index is None or device.index >= torch.cuda.device_count():
        raise Frame0LatentMaterializationError(
            "requested CUDA device is not visible"
        )
    torch.cuda.set_device(device)
    callable_identity = validate_encoder_callable(_vae_encode, bernini_root)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False).to(device)
    if vae.training or any(parameter.requires_grad for parameter in vae.parameters()):
        raise Frame0LatentMaterializationError(
            "VAE lifecycle is not frozen eval"
        )
    source_pixels = source_tensor.to(
        device=device, dtype=torch.float32
    ).contiguous()
    preprocessing = _preprocessing_receipt(source_metadata, source_pixels)
    latent, encoding_state = encode_source_frame0_once(
        vae, source_pixels, encoder=_vae_encode
    )
    latent_raw_sha256 = tensor_raw_sha256(latent)
    encoding = {
        **callable_identity,
        **encoding_state,
        "expected_job132387_frame0_tensor_raw_sha256": (
            args.expected_reference_frame0_tensor_raw_sha256
        ),
        "actual_reference_frame0_tensor_raw_sha256": latent_raw_sha256,
        "job132387_frame0_tensor_raw_sha256_match": (
            latent_raw_sha256
            == args.expected_reference_frame0_tensor_raw_sha256
        ),
    }
    if encoding["vae_requires_grad"] is not False or encoding["vae_eval"] is not True:
        raise Frame0LatentMaterializationError(
            "encoded VAE lifecycle certificate differs"
        )
    vae.to("cpu")
    del vae
    del source_tensor
    del source_pixels
    torch.cuda.empty_cache()

    # Revalidate every externally mutable byte class after the encode and
    # before either output name is created.
    if file_sha256(source_path) != source_sha256:
        raise Frame0LatentMaterializationError(
            "source video changed after VAE encode"
        )
    if file_sha256(args.source_manifest) != args.expected_source_manifest_sha256:
        raise Frame0LatentMaterializationError(
            "source manifest changed after VAE encode"
        )
    checkpoint_post = validate_checkpoint_content(
        checkpoint, args.checkpoint_content_manifest
    )
    if checkpoint_post != checkpoint_pre:
        raise Frame0LatentMaterializationError(
            "checkpoint content identity changed during encode"
        )
    try:
        post_roots = legacy.trainer.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        vendor_post = legacy.validate_inference_source_files(bernini_root)
    except (legacy.trainer.TrainingContractError, legacy.InferenceContractError) as error:
        raise Frame0LatentMaterializationError(str(error)) from error
    if (
        tuple(post_roots)
        != (bernini_root, veomni_root, bernini_revision, veomni_revision)
        or vendor_post != bernini_inference_files
    ):
        raise Frame0LatentMaterializationError(
            "vendor source identity changed during encode"
        )
    method_post = validate_method_provenance(args)
    if method_post != method_pre:
        raise Frame0LatentMaterializationError(
            "method/archive identity changed during encode"
        )

    model_closure = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_manifest_audit": checkpoint_pre,
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "bernini_inference_files": dict(bernini_inference_files),
        "bernini_inference_files_index_sha256": object_sha256(
            bernini_inference_files
        ),
        "method_source_revision": method_pre["revision"],
        "method_source_archive_sha256": method_pre["archive_sha256"],
        "runtime_source_index_sha256": method_pre[
            "runtime_source_index_sha256"
        ],
        "method_provenance": method_pre,
    }
    runtime = {
        "device_requested": args.device,
        "world_size": 1,
        "distributed_initialized": False,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hip_version": str(torch.version.hip),
        "diffusers_version": diffusers_version,
        "safetensors_version": safetensors_version,
    }
    transaction_token = hashlib.sha256(
        (row["row_id"] + source_sha256 + method_pre["archive_sha256"]).encode(
            "ascii"
        )
    ).hexdigest()[:24]
    return publish_materialization_bundle(
        latent,
        output_path,
        receipt_path,
        sealed_inputs=sealed_inputs,
        preprocessing=preprocessing,
        model_closure=model_closure,
        encoding=encoding,
        runtime=runtime,
        transaction_token=transaction_token,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_reference_frame0_latent(args)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_METADATA",
    "ARTIFACT_SCHEMA",
    "METHOD",
    "RECEIPT_SCHEMA",
    "Frame0LatentMaterializationError",
    "artifact_identity",
    "build_parser",
    "build_receipt",
    "canonical_json_bytes",
    "encode_source_frame0_once",
    "file_sha256",
    "load_sealed_source_row",
    "main",
    "materialize_reference_frame0_latent",
    "object_sha256",
    "prepare_private_exact_source",
    "publish_materialization_bundle",
    "reopen_published_bundle",
    "resolve_output",
    "runtime_source_hashes",
    "tensor_raw_sha256",
    "unlink_owned_artifact",
    "validate_checkpoint_content",
    "validate_cli",
    "validate_encoder_callable",
    "validate_method_provenance",
]
