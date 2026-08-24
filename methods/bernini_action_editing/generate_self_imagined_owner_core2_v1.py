#!/usr/bin/env python3
"""Generate the two prospective pure-T2V owners for Q-MOSAIC core2.

The source video is used by Bernini's pinned native runner only to choose the
exact81 spatial bucket.  No source pixels, latent, reference, target, mask,
flow, pose or track enters the T2V transformer.  Generation artifacts remain
proposal evidence until a detached full-video action audit is explicitly
bound; this runner never marks an owner semantically valid by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_pair_v5_t2v_calibration_bank as native_receipts  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as bank  # noqa: E402
import self_imagined_motion_cotangent_v1 as cotangent  # noqa: E402


SCHEMA_VERSION = "bernini-self-imagined-owner-generation-receipt-v1"
MASTER_SCHEMA_VERSION = "bernini-self-imagined-owner-core2-master-receipt-v1"
OWNER_RECEIPT_BASENAME = "owner-generation-receipt.json"
MASTER_RECEIPT_BASENAME = "owner-core2-master-receipt.json"
CELL_IDS = ("dog", "human")
VISIBLE_GPUS_BY_CELL = {"dog": "0,1,2,3", "human": "4,5,6,7"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class SelfImaginedOwnerGenerationError(RuntimeError):
    """Raised before an unauthenticated owner artifact is accepted."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SelfImaginedOwnerGenerationError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_dir(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise SelfImaginedOwnerGenerationError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SelfImaginedOwnerGenerationError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise SelfImaginedOwnerGenerationError(f"{label} must be lowercase full SHA-1")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def build_native_argv(args: argparse.Namespace, cell: cotangent.ProbeCellSpec) -> list[str]:
    """Build the only authorized native T2V invocation for one sealed cell."""

    _sha1(args.method_source_revision, label="method source revision")
    _sha(args.method_source_archive_sha256, label="method source archive SHA-256")
    return [
        "--bernini-root",
        str(_plain_dir(args.bernini_root, label="Bernini root")),
        "--veomni-root",
        str(_plain_dir(args.veomni_root, label="VeOmni root")),
        "--checkpoint",
        str(_plain_dir(args.checkpoint, label="Bernini checkpoint")),
        "--checkpoint-content-manifest",
        str(_plain_file(args.checkpoint_content_manifest, label="checkpoint manifest")),
        "--source-video",
        str(_plain_file(cell.source_video, label=f"{cell.cell_id} geometry source")),
        "--expected-source-sha256",
        cell.source_video_sha256,
        "--action-prompt",
        cell.action_caption,
        "--expected-action-prompt-sha256",
        cell.action_caption_utf8_sha256,
        "--output-dir",
        args.output_dir,
        "--arms",
        "t2v",
        "--num-inference-steps",
        "40",
        "--seed",
        str(cell.owner_generation_seed),
        "--method-source-revision",
        args.method_source_revision,
        "--method-source-archive-sha256",
        args.method_source_archive_sha256,
    ]


def _candidate(cell: cotangent.ProbeCellSpec) -> dict[str, Any]:
    return {
        "candidate_id": f"{cell.cell_id}-self-imagined-owner",
        "geometry_source_video": cell.source_video,
        "geometry_source_video_sha256": cell.source_video_sha256,
        "full_t2v_caption": cell.action_caption,
        "full_t2v_caption_utf8_sha256": cell.action_caption_utf8_sha256,
        "seed": cell.owner_generation_seed,
    }


def _artifact_inside(value: Mapping[str, Any], root: Path, *, label: str) -> dict[str, Any]:
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise SelfImaginedOwnerGenerationError(f"{label} path differs")
    path = _plain_file(path_value, label=label)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SelfImaginedOwnerGenerationError(f"{label} escaped output root") from error
    if file_sha256(path) != value.get("sha256"):
        raise SelfImaginedOwnerGenerationError(f"{label} SHA-256 differs")
    return dict(value)


def bind_cell_receipt(
    *,
    output_dir: Path,
    cell: cotangent.ProbeCellSpec,
    registry_path: Path,
    registry_sha256: str,
    method_source_revision: str,
    method_source_archive_sha256: str,
) -> Path:
    native_path = _plain_file(output_dir / "receipt.json", label="native receipt")
    try:
        native_value = native_receipts._load_json(native_path, "native receipt")
        verified = native_receipts._verify_native_receipt(native_value, _candidate(cell))
    except bank.PairT2VCalibrationSpecError as error:
        raise SelfImaginedOwnerGenerationError(
            "native exact81 source-free T2V receipt failed"
        ) from error
    artifacts = {
        "mp4": _artifact_inside(verified["mp4"], output_dir, label="owner MP4"),
        "predecode_clean_latent": _artifact_inside(
            verified["predecode_clean_latent"], output_dir, label="owner clean latent"
        ),
        "official_initial_gaussian": _artifact_inside(
            verified["official_initial_gaussian"], output_dir, label="owner Gaussian"
        ),
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "probe_id": "self-imagined-motion-cotangent-core2-p1",
        "cell_id": cell.cell_id,
        "registry_path": str(registry_path),
        "registry_file_sha256": registry_sha256,
        "source_iid": cell.source_iid,
        "geometry_source_video_sha256": cell.source_video_sha256,
        "geometry_source_role": "bucket_shape_only_never_transformer_condition",
        "action_family_id": cell.action_family_id,
        "action_caption_utf8_sha256": cell.action_caption_utf8_sha256,
        "owner_generation_seed": cell.owner_generation_seed,
        "native_receipt_path": str(native_path),
        "native_receipt_file_sha256": file_sha256(native_path),
        "native_receipt_digest": verified["native_receipt_digest"],
        "bucket_hw": verified["bucket_hw"],
        "latent_shape": verified["latent_shape"],
        "artifacts": artifacts,
        "method_source_revision": _sha1(
            method_source_revision, label="method source revision"
        ),
        "method_source_archive_sha256": _sha(
            method_source_archive_sha256, label="method archive SHA-256"
        ),
        "runtime_topology": {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": VISIBLE_GPUS_BY_CELL[cell.cell_id],
        },
        "owner_source_condition_used": False,
        "owner_exact81_action_audit_status": "pending_detached_full_video_review",
        "owner_template_materialization_authorized": False,
        "editor_condition_or_target_authorized": False,
        "optimizer_or_parameter_update_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": cotangent.object_sha256(unsigned)}
    path = output_dir / OWNER_RECEIPT_BASENAME
    _write_create_only(path, receipt)
    return path


def generate_cell(args: argparse.Namespace) -> int:
    registry_path = _plain_file(args.registry, label="core2 registry")
    registry_sha = file_sha256(registry_path)
    if registry_sha != _sha(args.expected_registry_sha256, label="registry SHA-256"):
        raise SelfImaginedOwnerGenerationError("core2 registry bytes changed")
    registry = cotangent.load_probe_registry(
        registry_path, expected_file_sha256=registry_sha
    )
    cell = registry.cell(args.cell_id)
    if (
        os.environ.get("WORLD_SIZE") != "4"
        or os.environ.get("ROCR_VISIBLE_DEVICES")
        != VISIBLE_GPUS_BY_CELL[cell.cell_id]
    ):
        raise SelfImaginedOwnerGenerationError(
            "cell must run in its sealed WORLD4/SP4 GPU group"
        )
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists():
        raise SelfImaginedOwnerGenerationError("output directory must be fresh and absolute")
    status = native.main(build_native_argv(args, cell))
    if status != 0:
        return status
    if int(os.environ.get("RANK", "0")) == 0:
        bind_cell_receipt(
            output_dir=output.resolve(strict=True),
            cell=cell,
            registry_path=registry_path,
            registry_sha256=registry_sha,
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
        )
    return 0


def _load_bound_cell(path: Path, *, expected_cell: str) -> dict[str, Any]:
    receipt_path = _plain_file(path / OWNER_RECEIPT_BASENAME, label=f"{expected_cell} owner receipt")
    try:
        value = json.loads(receipt_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelfImaginedOwnerGenerationError("owner receipt JSON differs") from error
    if not isinstance(value, dict):
        raise SelfImaginedOwnerGenerationError("owner receipt root differs")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("cell_id") != expected_cell
        or declared != cotangent.object_sha256(unsigned)
        or value.get("owner_exact81_action_audit_status")
        != "pending_detached_full_video_review"
        or value.get("owner_template_materialization_authorized") is not False
    ):
        raise SelfImaginedOwnerGenerationError(f"{expected_cell} owner receipt differs")
    for label, artifact in value.get("artifacts", {}).items():
        _artifact_inside(artifact, path, label=f"{expected_cell} {label}")
    return value


def audit_master(args: argparse.Namespace) -> int:
    root = _plain_dir(args.output_root, label="owner output root")
    registry_path = _plain_file(args.registry, label="core2 registry")
    registry_sha = file_sha256(registry_path)
    if registry_sha != _sha(args.expected_registry_sha256, label="registry SHA-256"):
        raise SelfImaginedOwnerGenerationError("core2 registry bytes changed")
    children = [_load_bound_cell(root / cell_id, expected_cell=cell_id) for cell_id in CELL_IDS]
    if any(row["registry_file_sha256"] != registry_sha for row in children):
        raise SelfImaginedOwnerGenerationError("child registry binding differs")
    if any(
        row.get("runtime_topology")
        != {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": VISIBLE_GPUS_BY_CELL[cell_id],
        }
        for cell_id, row in zip(CELL_IDS, children)
    ):
        raise SelfImaginedOwnerGenerationError("dual-SP4 child topology differs")
    unsigned = {
        "schema_version": MASTER_SCHEMA_VERSION,
        "probe_id": "self-imagined-motion-cotangent-core2-p1",
        "registry_path": str(registry_path),
        "registry_file_sha256": registry_sha,
        "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
        "cell_order": list(CELL_IDS),
        "children": [
            {
                "cell_id": cell_id,
                "receipt_path": str((root / cell_id / OWNER_RECEIPT_BASENAME).resolve()),
                "receipt_file_sha256": file_sha256(root / cell_id / OWNER_RECEIPT_BASENAME),
                "receipt_digest": row["receipt_digest"],
                "mp4_path": row["artifacts"]["mp4"]["path"],
                "mp4_sha256": row["artifacts"]["mp4"]["sha256"],
            }
            for cell_id, row in zip(CELL_IDS, children)
        ],
        "exact81_owner_count": 2,
        "all8_used": True,
        "semantic_action_audit_complete": False,
        "owner_template_materialization_authorized": False,
        "optimizer_or_parameter_update_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": cotangent.object_sha256(unsigned)}
    _write_create_only(root / MASTER_RECEIPT_BASENAME, receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-cell")
    generate.add_argument("--registry", required=True)
    generate.add_argument("--expected-registry-sha256", required=True)
    generate.add_argument("--cell-id", choices=CELL_IDS, required=True)
    generate.add_argument("--bernini-root", required=True)
    generate.add_argument("--veomni-root", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--checkpoint-content-manifest", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--method-source-revision", required=True)
    generate.add_argument("--method-source-archive-sha256", required=True)
    master = commands.add_parser("audit-master")
    master.add_argument("--registry", required=True)
    master.add_argument("--expected-registry-sha256", required=True)
    master.add_argument("--output-root", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-cell":
        return generate_cell(args)
    if args.command == "audit-master":
        return audit_master(args)
    raise SelfImaginedOwnerGenerationError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MASTER_RECEIPT_BASENAME",
    "OWNER_RECEIPT_BASENAME",
    "SCHEMA_VERSION",
    "SelfImaginedOwnerGenerationError",
    "audit_master",
    "bind_cell_receipt",
    "build_native_argv",
    "build_parser",
    "file_sha256",
    "generate_cell",
    "main",
]
