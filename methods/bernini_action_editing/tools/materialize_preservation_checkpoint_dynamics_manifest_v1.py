#!/usr/bin/env python3
"""Materialize the strict F0 checkpoint-review manifest from one fixed layout.

Expected download layout below ``--media-root``::

    training/{rank8,rank2}/dataset-receipt.json
    cells/<cell-id>/cell.json
    cells/<cell-id>/source.mp4
    cells/<cell-id>/{rank8,rank2}/step0/{video.mp4,inference-receipt.json}
    cells/<cell-id>/{rank8,rank2}/{step20,step40}/
        {video.mp4,paired-native.mp4,inference-receipt.json,training-receipt.json}

``cell.json`` contains exactly ``schema_version``, ``cell_id``, ``source_iid``,
``source_action_caption``, ``full_instruction`` and ``seed``.  This helper does
not weaken or duplicate artifact validation: the downstream HTML builder still
verifies every embedded digest, SHA, checkpoint binding and trajectory claim.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-preservation-checkpoint-dynamics-review-v2"
CELL_SCHEMA_VERSION = "bernini-preservation-checkpoint-dynamics-cell-v1"
RANKS = {"rank8": 8, "rank2": 2}
STEPS = ("step0", "step20", "step40")
AUTHORITY = {
    "feature_scalar_present": False,
    "aggregate_score_present": False,
    "reward_used": False,
    "manual_verdict_present": False,
    "method_success_claimed": False,
}
CELL_FIELDS = {
    "schema_version",
    "cell_id",
    "source_iid",
    "source_action_caption",
    "full_instruction",
    "seed",
}


class CheckpointDynamicsManifestError(RuntimeError):
    """Raised before an incomplete download tree becomes a review manifest."""


def fail(message: str) -> NoReturn:
    raise CheckpointDynamicsManifestError(message)


def _text(raw: Any, *, label: str) -> str:
    if type(raw) is not str or not raw.strip() or "\x00" in raw:
        fail(f"{label} must be non-empty text")
    return raw


def _plain_root(raw: str | Path) -> Path:
    requested = Path(raw).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("media root must be an absolute non-symlink directory")
    root = requested.resolve(strict=True)
    if root != requested or not root.is_dir():
        fail("media root directory differs")
    return root


def _plain_file(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        fail(f"{label} relative path differs")
    requested = root.joinpath(*pure.parts)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise CheckpointDynamicsManifestError(f"missing {label}: {relative}") from error
    if (
        resolved != requested
        or resolved.is_symlink()
        or not resolved.is_file()
        or root not in resolved.parents
    ):
        fail(f"{label} must be one plain file below media root")
    return resolved


def _load_cell(path: Path, *, directory_name: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        cell = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointDynamicsManifestError(f"cannot read {path}") from error
    if type(cell) is not dict or set(cell) != CELL_FIELDS:
        fail(f"{path} must contain the exact registered cell fields")
    if cell.get("schema_version") != CELL_SCHEMA_VERSION:
        fail(f"{path} schema differs")
    cell_id = _text(cell.get("cell_id"), label="cell ID")
    if cell_id != directory_name or "/" in cell_id or cell_id in {".", ".."}:
        fail(f"{path} cell ID differs from its directory")
    _text(cell.get("source_iid"), label=f"{cell_id} source IID")
    _text(
        cell.get("source_action_caption"),
        label=f"{cell_id} source action caption",
    )
    _text(cell.get("full_instruction"), label=f"{cell_id} full instruction")
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        fail(f"{cell_id} seed differs")
    return cell


def materialize(*, media_root: str | Path) -> dict[str, Any]:
    root = _plain_root(media_root)
    for rank_name in RANKS:
        _plain_file(
            root,
            f"training/{rank_name}/dataset-receipt.json",
            label=f"{rank_name} training dataset receipt",
        )
    cells_root = root / "cells"
    if (
        not cells_root.is_dir()
        or cells_root.is_symlink()
        or cells_root.resolve(strict=True) != cells_root
    ):
        fail("media root must contain one plain cells directory")
    directories = sorted(
        path
        for path in cells_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    if not directories:
        fail("download tree has no review cells")
    cells: list[dict[str, Any]] = []
    for cell_root in directories:
        if cell_root.name in {".", ".."} or "/" in cell_root.name:
            fail("unsafe cell directory")
        prefix = f"cells/{cell_root.name}"
        cell_path = _plain_file(root, f"{prefix}/cell.json", label="cell metadata")
        cell = _load_cell(cell_path, directory_name=cell_root.name)
        _plain_file(root, f"{prefix}/source.mp4", label=f"{cell_root.name} source")
        variants: dict[str, Any] = {}
        for rank_name in RANKS:
            variants[rank_name] = {}
            for step_name in STEPS:
                step_prefix = f"{prefix}/{rank_name}/{step_name}"
                item = {
                    "video": f"{step_prefix}/video.mp4",
                    "inference_receipt": f"{step_prefix}/inference-receipt.json",
                }
                _plain_file(root, item["video"], label=f"{cell_root.name} {rank_name} {step_name} video")
                _plain_file(
                    root,
                    item["inference_receipt"],
                    label=f"{cell_root.name} {rank_name} {step_name} inference receipt",
                )
                if step_name != "step0":
                    item["paired_native_video"] = (
                        f"{step_prefix}/paired-native.mp4"
                    )
                    _plain_file(
                        root,
                        item["paired_native_video"],
                        label=(
                            f"{cell_root.name} {rank_name} {step_name} "
                            "paired native video"
                        ),
                    )
                    item["training_receipt"] = (
                        f"{step_prefix}/training-receipt.json"
                    )
                    _plain_file(
                        root,
                        item["training_receipt"],
                        label=f"{cell_root.name} {rank_name} {step_name} training receipt",
                    )
                variants[rank_name][step_name] = item
        cells.append(
            {
                "cell_id": cell["cell_id"],
                "source_iid": cell["source_iid"],
                "source_video": f"{prefix}/source.mp4",
                "source_action_caption": cell["source_action_caption"],
                "full_instruction": cell["full_instruction"],
                "seed": cell["seed"],
                "variants": variants,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": dict(AUTHORITY),
        "ranks": {
            rank_name: {
                "adapter_rank": rank,
                "training_dataset_receipt": (
                    f"training/{rank_name}/dataset-receipt.json"
                ),
            }
            for rank_name, rank in RANKS.items()
        },
        "cells": cells,
    }


def write_manifest(*, media_root: str | Path, output: str | Path) -> Path:
    manifest = materialize(media_root=media_root)
    target = Path(output).expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists():
        fail("output must be an absolute fresh non-symlink JSON path")
    target = target.absolute()
    if (
        target.suffix.lower() != ".json"
        or not target.parent.is_dir()
        or target.parent.is_symlink()
        or target.parent != _plain_root(media_root)
        or target.parent.resolve(strict=True) != _plain_root(media_root)
    ):
        fail("output must be a fresh .json directly inside the plain media root")
    payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(write_manifest(media_root=args.media_root, output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
