#!/usr/bin/env python3
"""Build the sealed, target-free source pool for the SAIC Stage-A pilot.

The builder scans only the v17 ``samples/<iid>/samples/<iid>/source_video.mp4``
closure.  Every accepted file is decoded by ``infer_lora.prepare_exact_source``
so exact81/25fps and Bernini's source-derived spatial bucket are checked by the
same implementation used at inference.  The eight preregistered action-study
IIDs are excluded before selection.

Sources are paired inside each bucket first, then 40 deterministic disjoint
pairs are selected across buckets.  Thirty-two pairs (64 identities) are
optimizer rows and eight pairs (16 identities) are held out.  Wrong-source
pairs are fixed-point-free, stay inside the same split, bucket, and DP arm,
and therefore never smuggle a held-out identity into optimization.  This file
creates metadata only; it does not encode latents or authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora  # noqa: E402


SCHEMA_VERSION = "bernini-saic-source-anchor-manifest-v1"
FRAME_COUNT = 81
FPS = 25.0
TRAIN_COUNT = 64
HOLDOUT_COUNT = 16
DP_SIZE = 2
DEFAULT_SEED = 20260809
STRICT_ACTION_IIDS = frozenset(
    {
        "7b88a1ca1f804f41",
        "841b5e0080a1441d",
        "a35b590961d24694",
        "31c34509415745ca",
        "99cde432839f4240",
        "6ea45d35943742bb",
        "311c82f83eca4a7f",
        "6d346c38cf504493",
    }
)
_IID = re.compile(r"[0-9a-f]{16}")


class SAICSourceAnchorManifestError(RuntimeError):
    """Raised when the v17 source closure is incomplete or ambiguous."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICSourceAnchorManifestError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256_stable(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SAICSourceAnchorManifestError(f"source is not a plain file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = path.stat()
        opened = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = path.stat()
    identity = lambda value: (  # noqa: E731 - compact immutable projection
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(opened) or identity(before) != identity(after):
        raise SAICSourceAnchorManifestError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def _selection_key(*, seed: int, iid: str, sha256: str) -> str:
    return hashlib.sha256(
        f"saic-source-anchor-v1\0{seed}\0{iid}\0{sha256}".encode("ascii")
    ).hexdigest()


def _scramble_seed(*, seed: int, split: str, iid: str) -> int:
    raw = hashlib.sha256(
        f"saic-source-scramble-v1\0{seed}\0{split}\0{iid}".encode("ascii")
    ).digest()
    return int.from_bytes(raw[:8], "big") & ((1 << 63) - 1)


def discover_source_paths(samples_root: Path) -> tuple[tuple[str, Path], ...]:
    root = samples_root.expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise SAICSourceAnchorManifestError("samples root must be a canonical directory")
    discovered: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*/samples/*/source_video.mp4")):
        try:
            relative = path.relative_to(root)
        except ValueError as error:  # pragma: no cover - glob roots it already
            raise SAICSourceAnchorManifestError("source escaped samples root") from error
        parts = relative.parts
        if len(parts) != 4 or parts[1] != "samples" or parts[3] != "source_video.mp4":
            raise SAICSourceAnchorManifestError(f"source layout differs: {relative}")
        outer, inner = parts[0], parts[2]
        if outer != inner or _IID.fullmatch(inner) is None:
            raise SAICSourceAnchorManifestError(f"source IID closure differs: {relative}")
        if inner in seen:
            raise SAICSourceAnchorManifestError(f"duplicate source IID: {inner}")
        seen.add(inner)
        if inner in STRICT_ACTION_IIDS:
            continue
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise SAICSourceAnchorManifestError(f"source is not a plain file: {path}")
        discovered.append((inner, path))
    if not discovered:
        raise SAICSourceAnchorManifestError("no eligible v17 source videos found")
    return tuple(discovered)


def _inspect_sources(
    sources: Sequence[tuple[str, Path]],
    *,
    prepare_source: Callable[[Path], tuple[Any, Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    inspected: list[dict[str, Any]] = []
    for iid, path in sources:
        _, metadata = prepare_source(path)
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("frame_count") != FRAME_COUNT
            or float(metadata.get("fps", -1.0)) != FPS
            or not isinstance(metadata.get("reported_fps"), (int, float))
            or abs(float(metadata["reported_fps"]) - FPS) > 1.0e-3
            or not isinstance(metadata.get("source_derived_bucket_hw"), list)
            or len(metadata["source_derived_bucket_hw"]) != 2
        ):
            raise SAICSourceAnchorManifestError(
                f"source {iid} is not exact81/25fps Bernini preprocessing"
            )
        bucket = tuple(int(item) for item in metadata["source_derived_bucket_hw"])
        if any(item <= 0 or item % 16 for item in bucket):
            raise SAICSourceAnchorManifestError(f"source {iid} bucket differs")
        inspected.append(
            {
                "iid": iid,
                "source_video_path": str(path),
                "source_video_sha256": file_sha256_stable(path),
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "reported_fps": float(metadata["reported_fps"]),
                "bucket_hw": list(bucket),
            }
        )
    return tuple(inspected)


def _rows_from_pairs(
    selected_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(selected_pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise SAICSourceAnchorManifestError("selected source pair differs")
        dp_arm = pair_index % DP_SIZE
        for correct, wrong in (pair, tuple(reversed(pair))):
            if correct["iid"] == wrong["iid"] or correct["bucket_hw"] != wrong["bucket_hw"]:
                raise SAICSourceAnchorManifestError("wrong-source derangement failed")
            value = {
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "row_index": len(rows),
                "dp_arm": dp_arm,
                "iid": correct["iid"],
                "source_video_path": correct["source_video_path"],
                "source_video_sha256": correct["source_video_sha256"],
                "wrong_iid": wrong["iid"],
                "wrong_source_video_path": wrong["source_video_path"],
                "wrong_source_video_sha256": wrong["source_video_sha256"],
                "frame_count": correct["frame_count"],
                "fps": correct["fps"],
                "reported_fps": correct["reported_fps"],
                "bucket_hw": correct["bucket_hw"],
                "scramble_seed": _scramble_seed(seed=seed, split=split, iid=correct["iid"]),
            }
            rows.append({**value, "row_digest": object_sha256(value)})
    return rows


def build_manifest(
    samples_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    train_count: int = TRAIN_COUNT,
    holdout_count: int = HOLDOUT_COUNT,
    prepare_source: Callable[[Path], tuple[Any, Mapping[str, Any]]] = infer_lora.prepare_exact_source,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise SAICSourceAnchorManifestError("seed must lie in [0,2^63)")
    if train_count != TRAIN_COUNT or holdout_count != HOLDOUT_COUNT:
        raise SAICSourceAnchorManifestError("v1 is fixed to train64/holdout16")
    root = samples_root.expanduser().resolve(strict=True)
    inspected = _inspect_sources(
        discover_source_paths(root), prepare_source=prepare_source
    )
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in inspected:
        buckets.setdefault(tuple(row["bucket_hw"]), []).append(row)
    required = train_count + holdout_count
    candidate_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for bucket, population in sorted(buckets.items()):
        ranked = sorted(
            population,
            key=lambda row: (
                _selection_key(
                    seed=seed, iid=row["iid"], sha256=row["source_video_sha256"]
                ),
                row["iid"],
            ),
        )
        candidate_pairs.extend(
            (ranked[index], ranked[index + 1])
            for index in range(0, len(ranked) - 1, 2)
        )
    candidate_pairs.sort(
        key=lambda pair: (
            hashlib.sha256(
                (
                    f"saic-source-pair-v1\0{seed}\0"
                    f"{pair[0]['iid']}\0{pair[1]['iid']}"
                ).encode("ascii")
            ).hexdigest(),
            pair[0]["iid"],
            pair[1]["iid"],
        )
    )
    if len(candidate_pairs) * 2 < required:
        summary = {f"{h}x{w}": len(rows) for (h, w), rows in sorted(buckets.items())}
        raise SAICSourceAnchorManifestError(
            f"fewer than {required} sources can form same-bucket disjoint pairs: {summary}"
        )
    selected_pairs = candidate_pairs[: required // 2]
    train_pairs = selected_pairs[: train_count // 2]
    holdout_pairs = selected_pairs[train_count // 2 :]
    train_rows = _rows_from_pairs(train_pairs, split="train", seed=seed)
    holdout_rows = _rows_from_pairs(holdout_pairs, split="holdout", seed=seed)
    train_iids = {row["iid"] for row in train_rows}
    holdout_iids = {row["iid"] for row in holdout_rows}
    if train_iids & holdout_iids or any(
        row["wrong_iid"] not in train_iids for row in train_rows
    ) or any(row["wrong_iid"] not in holdout_iids for row in holdout_rows):
        raise SAICSourceAnchorManifestError("train/holdout identity closure failed")
    selected_bucket_counts: dict[str, int] = {}
    for row in train_rows + holdout_rows:
        key = f"{row['bucket_hw'][0]}x{row['bucket_hw'][1]}"
        selected_bucket_counts[key] = selected_bucket_counts.get(key, 0) + 1
    value = {
        "schema_version": SCHEMA_VERSION,
        "optimizer_authorized": False,
        "source_root": str(root),
        "selection_seed": seed,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "train_count": train_count,
        "holdout_count": holdout_count,
        "selected_bucket_counts": dict(sorted(selected_bucket_counts.items())),
        "eligible_bucket_counts": {
            f"{height}x{width}": len(rows)
            for (height, width), rows in sorted(buckets.items())
        },
        "strict_action_iids_excluded": sorted(STRICT_ACTION_IIDS),
        "wrong_source_policy": "same_split_same_bucket_same_dp_arm_fixed_point_free",
        "holdout_used_by_optimizer": False,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "input_closure": {
            "source_video_only": True,
            "paired_target": False,
            "action_instruction": False,
            "proposal_video": False,
            "mask_pose_flow_track_trajectory": False,
        },
    }
    return {**value, "manifest_digest": object_sha256(value)}


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser()
    if (
        not destination.is_absolute()
        or not destination.parent.is_dir()
        or destination.exists()
        or destination.is_symlink()
    ):
        raise SAICSourceAnchorManifestError(
            "output must be a fresh absolute path below an existing directory"
        )
    payload = canonical_json_bytes(value) + b"\n"
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publication is atomic and, unlike os.replace, cannot
        # overwrite a path created after the preflight check.
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_manifest(args.samples_root, seed=args.seed)
    atomic_write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_digest": manifest["manifest_digest"],
                "train_count": manifest["train_count"],
                "holdout_count": manifest["holdout_count"],
                "selected_bucket_counts": manifest["selected_bucket_counts"],
                "optimizer_authorized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
