#!/usr/bin/env python3
"""Verify a SEER eval overlay application and emit a source-binding receipt.

This verifier keeps two identities deliberately separate:

* the immutable training method source archive/revision recorded by the LoRA;
* the three-file full160 inference-runtime overlay used only for heldout decode.

It verifies exact overlay closure, validates the extracted base archive before
and runtime tree after overlay, and refuses any overlay SHA presented as the
training method archive SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT / "tools") not in os.sys.path:
    os.sys.path.insert(0, str(METHOD_ROOT / "tools"))

import build_seer_full160_eval_overlay_v1 as overlay  # noqa: E402


SCHEMA_VERSION = "bernini-seer-full160-eval-source-binding-v1"
CASE_BINDING_SCHEMA_VERSION = "bernini-seer-full160-eval-case-binding-v1"
PAIR_SCHEMA_VERSION = "bernini-self-generated-action-lora-heldout-pair-v1"
INFERENCE_SCHEMA_VERSION = "bernini-r-1p3b-action-lora-inference-receipt-v1"
TRAINING_REVISION = overlay.TRAINING_METHOD_SOURCE_REVISION
TRAINING_ARCHIVE_SHA256 = overlay.TRAINING_METHOD_SOURCE_ARCHIVE_SHA256
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SourceBindingError(RuntimeError):
    pass


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
        raise SourceBindingError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise SourceBindingError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SourceBindingError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise SourceBindingError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise SourceBindingError(f"{label} must be absolute and non-root")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SourceBindingError(f"{label} is unavailable: {error}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise SourceBindingError(f"{label} must be a plain directory")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceBindingError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SourceBindingError(f"{label} root must be an object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path == Path("/"):
        raise SourceBindingError("binding receipt must be absolute and non-root")
    if path.exists() or path.is_symlink():
        raise SourceBindingError("binding receipt target must be fresh")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _safe_archive_members(archive_path: Path) -> list[tarfile.TarInfo]:
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as error:
        raise SourceBindingError(f"cannot read training archive: {error}") from error
    if not members:
        raise SourceBindingError("training archive is empty")
    seen: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        normalized = path.as_posix()
        if normalized == "." and member.isdir():
            continue
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or normalized in seen
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            raise SourceBindingError(f"unsafe training archive member: {member.name}")
        seen.add(normalized)
    return members


def _tree_digest(root: Path) -> tuple[str, dict[str, str]]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise SourceBindingError(f"source tree has non-plain member: {relative}")
        rows[relative] = file_sha256(path)
    if not rows:
        raise SourceBindingError("source tree contains no files")
    return object_sha256(rows), rows


def _archive_file_map(archive_path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile():
                    continue
                normalized = path.as_posix()
                if normalized.startswith("./"):
                    normalized = normalized[2:]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SourceBindingError(
                        f"cannot read training archive member: {member.name}"
                    )
                if normalized in rows:
                    raise SourceBindingError(
                        f"duplicate training archive file: {normalized}"
                    )
                rows[normalized] = hashlib.sha256(extracted.read()).hexdigest()
    except (OSError, tarfile.TarError) as error:
        raise SourceBindingError(f"cannot hash training archive files: {error}") from error
    if not rows:
        raise SourceBindingError("training archive has no plain files")
    return rows


def verify_binding(
    *,
    training_archive: Path,
    overlay_archive: Path,
    manifest_path: Path,
    base_root: Path,
    runtime_root: Path,
    expected_overlay_archive_sha256: str,
    expected_overlay_manifest_sha256: str,
) -> dict[str, Any]:
    if file_sha256(training_archive) != TRAINING_ARCHIVE_SHA256:
        raise SourceBindingError("training method source archive SHA differs")
    _safe_archive_members(training_archive)
    if file_sha256(overlay_archive) != expected_overlay_archive_sha256:
        raise SourceBindingError("inference overlay archive SHA differs")
    if file_sha256(manifest_path) != expected_overlay_manifest_sha256:
        raise SourceBindingError("inference overlay manifest raw SHA differs")
    if expected_overlay_archive_sha256 == TRAINING_ARCHIVE_SHA256:
        raise SourceBindingError("overlay SHA cannot impersonate training archive SHA")
    try:
        manifest = overlay._read_manifest(manifest_path)
        checked_overlay = overlay.validate_archive(overlay_archive, manifest)
    except overlay.OverlayError as error:
        raise SourceBindingError(str(error)) from error

    base_digest, base_files = _tree_digest(base_root)
    runtime_digest, runtime_files = _tree_digest(runtime_root)
    archive_files = _archive_file_map(training_archive)
    if base_files != archive_files:
        raise SourceBindingError(
            "extracted base source tree differs from training archive bytes"
        )
    overlay_rows = {str(row["path"]): row for row in manifest["files"]}
    expected_added = {
        f"methods/bernini_action_editing/{relative}"
        for relative in manifest["added_paths"]
    }
    expected_replaced = {
        f"methods/bernini_action_editing/{relative}"
        for relative in manifest["replaced_paths"]
    }
    expected_overlay = expected_added | expected_replaced
    actual_added = set(runtime_files) - set(base_files)
    actual_removed = set(base_files) - set(runtime_files)
    actual_replaced = {
        path
        for path in set(base_files) & set(runtime_files)
        if base_files[path] != runtime_files[path]
    }
    if (
        actual_added != expected_added
        or actual_removed
        or actual_replaced != expected_replaced
    ):
        raise SourceBindingError(
            "base/runtime exact added/replaced/removed closure differs"
        )
    for relative, row in overlay_rows.items():
        rooted = f"methods/bernini_action_editing/{relative}"
        if runtime_files.get(rooted) != row["sha256"]:
            raise SourceBindingError(f"runtime overlay bytes differ: {relative}")
    if any(
        base_files[path] != runtime_files[path]
        for path in base_files
        if path not in expected_replaced
    ):
        raise SourceBindingError("non-overlay source bytes changed")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "runtime_overlay_applied_exactly_training_provenance_preserved",
        "training_method_source": {
            "revision": TRAINING_REVISION,
            "archive_path": str(training_archive),
            "archive_sha256": TRAINING_ARCHIVE_SHA256,
            "passed_to_inference_method_source_revision": TRAINING_REVISION,
            "passed_to_inference_method_source_archive_sha256": TRAINING_ARCHIVE_SHA256,
        },
        "inference_runtime_overlay": {
            "archive_path": str(overlay_archive),
            "archive_sha256": checked_overlay["archive_sha256"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_digest": manifest["manifest_digest"],
            "file_count": len(overlay_rows),
            "files": [dict(row) for row in manifest["files"]],
            "passed_as_training_method_archive": False,
        },
        "application": {
            "base_root": str(base_root),
            "base_tree_digest": base_digest,
            "runtime_root": str(runtime_root),
            "runtime_tree_digest": runtime_digest,
            "same_file_membership": False,
            "added_paths": sorted(actual_added),
            "replaced_paths": sorted(actual_replaced),
            "removed_paths": [],
            "overlay_paths_digest": object_sha256(sorted(expected_overlay)),
            "only_manifest_declared_paths_changed": True,
            "all_non_overlay_files_byte_exact": True,
        },
        "training_receipt_mutated": False,
        "training_provenance_replaced": False,
        "method_success_claimed": False,
    }


def verify_receipt(path: Path) -> dict[str, Any]:
    value = _read_json(_plain_file(path, label="binding receipt"), label="binding receipt")
    candidate = dict(value)
    declared = candidate.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(candidate) != declared:
        raise SourceBindingError("binding receipt digest differs")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("training_method_source", {}).get("revision") != TRAINING_REVISION
        or value.get("training_method_source", {}).get("archive_sha256")
        != TRAINING_ARCHIVE_SHA256
        or value.get("inference_runtime_overlay", {}).get(
            "passed_as_training_method_archive"
        )
        is not False
        or value.get("training_receipt_mutated") is not False
        or value.get("training_provenance_replaced") is not False
        or value.get("application", {}).get(
            "only_manifest_declared_paths_changed"
        )
        is not True
    ):
        raise SourceBindingError("binding receipt contract differs")
    return value


def _verified_digest_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = _plain_file(path, label=label)
    value = _read_json(path, label=label)
    candidate = dict(value)
    declared = candidate.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or object_sha256(candidate) != declared
    ):
        raise SourceBindingError(f"{label} digest differs")
    return value, file_sha256(path)


def _verify_inference_output(
    *,
    inference_path: Path,
    pair_arm: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    inference, inference_sha = _verified_digest_json(
        inference_path, label=f"{arm} inference receipt"
    )
    output = inference.get("output")
    pair_output = pair_arm.get("output")
    adapter = inference.get("adapter")
    if (
        inference.get("schema_version") != INFERENCE_SCHEMA_VERSION
        or inference.get("method_source_revision") != TRAINING_REVISION
        or inference.get("method_source_archive_sha256") != TRAINING_ARCHIVE_SHA256
        or not isinstance(output, Mapping)
        or not isinstance(pair_output, Mapping)
        or output != pair_output
        or inference.get("receipt_digest")
        != pair_arm.get("inference_receipt_digest")
        or not isinstance(adapter, Mapping)
    ):
        raise SourceBindingError(f"{arm} inference/pair cross-bind differs")
    output_path = _plain_file(output.get("path", ""), label=f"{arm} output video")
    if file_sha256(output_path) != output.get("sha256"):
        raise SourceBindingError(f"{arm} output video SHA differs")
    result: dict[str, Any] = {
        "inference_receipt_path": str(inference_path),
        "inference_receipt_sha256": inference_sha,
        "inference_receipt_digest": inference["receipt_digest"],
        "output_path": str(output_path),
        "output_sha256": output["sha256"],
    }
    if arm == "frozen_base":
        if adapter != {
            "enabled": False,
            "mode": "frozen_base_no_adapter",
            "strictly_reloaded": False,
            "safe_merged_for_inference": False,
            "tensor_count": 0,
        }:
            raise SourceBindingError("frozen inference adapter contract differs")
        result["adapter_enabled"] = False
        return result

    if (
        adapter.get("enabled") is not True
        or adapter.get("mode") != "lora_safe_merge"
        or adapter.get("strictly_reloaded") is not True
        or adapter.get("safe_merged_for_inference") is not True
        or adapter.get("training_global_step") != 160
        or adapter.get("tensor_count") != 120
        or adapter.get("adapter_model_sha256")
        != pair_arm.get("adapter_model_sha256")
        or pair_arm.get("training_global_step") != 160
    ):
        raise SourceBindingError("trained inference adapter contract differs")
    checkpoint_root = _directory(
        adapter.get("checkpoint_root", ""), label="full160 adapter checkpoint"
    )
    if checkpoint_root.name != "checkpoint-00000160":
        raise SourceBindingError("trained inference checkpoint step differs")
    adapter_model = _plain_file(
        adapter.get("adapter_model_path", ""), label="full160 adapter model"
    )
    training_receipt_path = _plain_file(
        adapter.get("training_receipt_path", ""), label="full160 training receipt"
    )
    if (
        adapter_model.parent != checkpoint_root / "adapter"
        or training_receipt_path.parent != checkpoint_root
        or file_sha256(adapter_model) != adapter.get("adapter_model_sha256")
    ):
        raise SourceBindingError("full160 adapter artifact path/SHA differs")
    training, training_sha = _verified_digest_json(
        training_receipt_path, label="full160 training receipt"
    )
    immutable = training.get("immutable_contract")
    value = immutable.get("value") if isinstance(immutable, Mapping) else None
    if (
        training.get("global_step") != 160
        or training.get("max_steps") != 160
        or training.get("receipt_digest")
        != adapter.get("training_receipt_digest")
        or not isinstance(value, Mapping)
        or value.get("method_source_revision") != TRAINING_REVISION
        or value.get("method_source_archive_sha256") != TRAINING_ARCHIVE_SHA256
    ):
        raise SourceBindingError("full160 training provenance cross-bind differs")
    result.update(
        {
            "adapter_enabled": True,
            "checkpoint_root": str(checkpoint_root),
            "adapter_model_path": str(adapter_model),
            "adapter_model_sha256": adapter["adapter_model_sha256"],
            "training_receipt_path": str(training_receipt_path),
            "training_receipt_sha256": training_sha,
            "training_receipt_digest": training["receipt_digest"],
            "training_global_step": 160,
            "training_max_steps": 160,
            "adapter_tensor_count": 120,
        }
    )
    return result


def finalize_case_binding(
    *, source_binding_path: Path, pair_receipt_path: Path
) -> dict[str, Any]:
    source = verify_receipt(source_binding_path)
    source_sha = file_sha256(source_binding_path)
    pair, pair_sha = _verified_digest_json(
        pair_receipt_path, label="paired heldout receipt"
    )
    iid = pair.get("iid")
    frozen_arm = pair.get("frozen_base")
    trained_arm = pair.get("trained_adapter")
    if (
        pair.get("schema_version") != PAIR_SCHEMA_VERSION
        or pair.get("status")
        != "decoded_pair_ready_for_blind_review_no_method_success_claim"
        or not isinstance(iid, str)
        or re.fullmatch(r"[0-9a-f]{16}", iid) is None
        or pair.get("same_source_instruction_seed_preprocessing_sampler") is not True
        or pair.get("full_video_action_and_preservation_review_complete") is not False
        or pair.get("method_success_authorized") is not False
        or not isinstance(frozen_arm, Mapping)
        or not isinstance(trained_arm, Mapping)
    ):
        raise SourceBindingError("paired heldout receipt contract differs")
    frozen_output_path = Path(str(frozen_arm.get("output", {}).get("path", "")))
    trained_output_path = Path(str(trained_arm.get("output", {}).get("path", "")))
    frozen = _verify_inference_output(
        inference_path=Path(str(frozen_output_path) + ".receipt.json"),
        pair_arm=frozen_arm,
        arm="frozen_base",
    )
    trained = _verify_inference_output(
        inference_path=Path(str(trained_output_path) + ".receipt.json"),
        pair_arm=trained_arm,
        arm="trained_adapter",
    )
    outputs_byte_identical = (
        frozen["output_sha256"] == trained["output_sha256"]
    )
    return {
        "schema_version": CASE_BINDING_SCHEMA_VERSION,
        "status": "full160_decoded_pair_source_and_output_bound_pending_blind_review",
        "iid": iid,
        "training_method_source": {
            "revision": TRAINING_REVISION,
            "archive_sha256": TRAINING_ARCHIVE_SHA256,
        },
        "source_binding": {
            "path": str(source_binding_path),
            "sha256": source_sha,
            "receipt_digest": source["receipt_digest"],
            "overlay_archive_sha256": source["inference_runtime_overlay"][
                "archive_sha256"
            ],
            "overlay_manifest_sha256": source["inference_runtime_overlay"][
                "manifest_sha256"
            ],
            "overlay_manifest_digest": source["inference_runtime_overlay"][
                "manifest_digest"
            ],
        },
        "paired_receipt": {
            "path": str(pair_receipt_path),
            "sha256": pair_sha,
            "receipt_digest": pair["receipt_digest"],
        },
        "frozen_base": frozen,
        "trained_adapter": trained,
        "decoded_outputs_byte_identical": outputs_byte_identical,
        "training_archive_and_inference_overlay_distinct": True,
        "training_receipt_mutated": False,
        "same_source_instruction_seed_preprocessing_sampler": True,
        "full_video_action_and_preservation_review_complete": False,
        "method_success_claimed": False,
        "method_success_authorized": False,
    }


def verify_case_binding_receipt(path: Path) -> dict[str, Any]:
    value, _ = _verified_digest_json(path, label="case eval binding receipt")
    if (
        value.get("schema_version") != CASE_BINDING_SCHEMA_VERSION
        or value.get("training_method_source")
        != {"revision": TRAINING_REVISION, "archive_sha256": TRAINING_ARCHIVE_SHA256}
        or value.get("training_archive_and_inference_overlay_distinct") is not True
        or value.get("training_receipt_mutated") is not False
        or value.get("same_source_instruction_seed_preprocessing_sampler") is not True
        or value.get("full_video_action_and_preservation_review_complete") is not False
        or value.get("method_success_claimed") is not False
        or value.get("method_success_authorized") is not False
        or type(value.get("decoded_outputs_byte_identical")) is not bool
    ):
        raise SourceBindingError("case eval binding receipt contract differs")
    for key in ("source_binding", "paired_receipt"):
        row = value.get(key)
        if not isinstance(row, Mapping):
            raise SourceBindingError(f"case eval {key} is absent")
        bound = _plain_file(row.get("path", ""), label=f"case eval {key}")
        if file_sha256(bound) != row.get("sha256"):
            raise SourceBindingError(f"case eval {key} SHA differs")
    for arm in ("frozen_base", "trained_adapter"):
        row = value.get(arm)
        if not isinstance(row, Mapping):
            raise SourceBindingError(f"case eval {arm} is absent")
        for stem in ("inference_receipt", "output"):
            bound = _plain_file(
                row.get(f"{stem}_path", ""), label=f"case eval {arm} {stem}"
            )
            if file_sha256(bound) != row.get(f"{stem}_sha256"):
                raise SourceBindingError(f"case eval {arm} {stem} SHA differs")
    actual_identical = (
        value["frozen_base"].get("output_sha256")
        == value["trained_adapter"].get("output_sha256")
    )
    if value["decoded_outputs_byte_identical"] is not actual_identical:
        raise SourceBindingError(
            "case eval decoded-output identity claim differs"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--training-archive", required=True)
    bind.add_argument("--overlay-archive", required=True)
    bind.add_argument("--overlay-manifest", required=True)
    bind.add_argument("--expected-overlay-archive-sha256", required=True)
    bind.add_argument("--expected-overlay-manifest-sha256", required=True)
    bind.add_argument("--base-root", required=True)
    bind.add_argument("--runtime-root", required=True)
    bind.add_argument("--output", required=True)
    verify = sub.add_parser("verify-receipt")
    verify.add_argument("--receipt", required=True)
    finalize = sub.add_parser("finalize-case")
    finalize.add_argument("--source-binding", required=True)
    finalize.add_argument("--paired-receipt", required=True)
    finalize.add_argument("--output", required=True)
    verify_case = sub.add_parser("verify-case")
    verify_case.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-receipt":
        result = verify_receipt(Path(args.receipt))
    elif args.command == "verify-case":
        result = verify_case_binding_receipt(Path(args.receipt))
    elif args.command == "finalize-case":
        unsigned = finalize_case_binding(
            source_binding_path=_plain_file(
                args.source_binding, label="source binding receipt"
            ),
            pair_receipt_path=_plain_file(
                args.paired_receipt, label="paired heldout receipt"
            ),
        )
        result = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        output = Path(args.output)
        output.parent.resolve(strict=True)
        _write_create_only(output, result)
        verify_case_binding_receipt(output)
    else:
        for value, label in (
            (args.expected_overlay_archive_sha256, "overlay archive SHA"),
            (args.expected_overlay_manifest_sha256, "overlay manifest SHA"),
        ):
            if _SHA256.fullmatch(value) is None:
                raise SourceBindingError(f"{label} is invalid")
        unsigned = verify_binding(
            training_archive=_plain_file(args.training_archive, label="training archive"),
            overlay_archive=_plain_file(args.overlay_archive, label="overlay archive"),
            manifest_path=_plain_file(args.overlay_manifest, label="overlay manifest"),
            base_root=_directory(args.base_root, label="base source root"),
            runtime_root=_directory(args.runtime_root, label="runtime source root"),
            expected_overlay_archive_sha256=args.expected_overlay_archive_sha256,
            expected_overlay_manifest_sha256=args.expected_overlay_manifest_sha256,
        )
        result = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        output = Path(args.output)
        output.parent.resolve(strict=True)
        _write_create_only(output, result)
        verify_receipt(output)
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourceBindingError as error:
        print(f"[seer-full160-source-binding] ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)
