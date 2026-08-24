#!/usr/bin/env python3
"""Materialize a closed PAIR-v5 T2V bank without hand-authored hashes.

The authoring file contains one exact-81 geometry path and ten textual branch
descriptions per cell.  This tool hashes each geometry video, composes complete
standalone captions, computes every caption digest, expands each cell in exact
MACE order, and validates the final closed bank before writing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_calibration_bank_spec as contract  # noqa: E402


_AUTHORING_FIELDS = {"schema_version", "bank_id", "expected_cell_count", "cells"}
_SELECTION_FIELDS = {
    "schema_version",
    "bank_id",
    "expected_cell_count",
    "registry_file",
    "registry_raw_sha256",
    "selected_iids",
    "first_gpu_job_default",
}
_CELL_FIELDS = {
    "iid",
    "analysis_split",
    "action_family_id",
    "actor_group_id",
    "scene_group_id",
    "action_group_id",
    "execution_group",
    "geometry_source_video",
    "seed",
    "scene_caption",
    "branch_descriptions",
    "camera_caption",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PairT2VAuthoringError(RuntimeError):
    pass


def _reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PairT2VAuthoringError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _closed(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PairT2VAuthoringError(
            f"{label} keys differ: expected={sorted(fields)!r}, actual={actual!r}"
        )
    return value


def _load_authoring(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise PairT2VAuthoringError("authoring input must be an absolute plain file")
    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise PairT2VAuthoringError("expected authoring SHA-256 is invalid")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PairT2VAuthoringError("authoring raw SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PairT2VAuthoringError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairT2VAuthoringError("authoring input is not UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise PairT2VAuthoringError("authoring input root must be an object")
    if value.get("schema_version") == contract.AUTHORING_SCHEMA_VERSION:
        return _closed(value, _AUTHORING_FIELDS, "authoring root")
    selection = _closed(value, _SELECTION_FIELDS, "authoring selection")
    if selection["schema_version"] != contract.AUTHORING_SELECTION_SCHEMA_VERSION:
        raise PairT2VAuthoringError("authoring schema_version differs")
    if type(selection["first_gpu_job_default"]) is not bool:
        raise PairT2VAuthoringError("first_gpu_job_default must be boolean")
    registry_file = selection["registry_file"]
    if (
        not isinstance(registry_file, str)
        or not registry_file
        or Path(registry_file).name != registry_file
    ):
        raise PairT2VAuthoringError("selection registry_file must be one safe basename")
    registry_digest = selection["registry_raw_sha256"]
    if not isinstance(registry_digest, str) or _SHA256_RE.fullmatch(registry_digest) is None:
        raise PairT2VAuthoringError("selection registry_raw_sha256 is invalid")
    registry_path = source.parent / registry_file
    if not registry_path.is_file() or registry_path.is_symlink():
        raise PairT2VAuthoringError("selection registry is absent or not plain")
    registry_raw = registry_path.read_bytes()
    if hashlib.sha256(registry_raw).hexdigest() != registry_digest:
        raise PairT2VAuthoringError("selection registry raw SHA-256 differs")
    try:
        registry_value = json.loads(
            registry_raw,
            object_pairs_hook=_reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PairT2VAuthoringError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairT2VAuthoringError("selection registry is not UTF-8 JSON") from error
    registry = _closed(registry_value, _AUTHORING_FIELDS, "selection registry")
    if registry["schema_version"] != contract.AUTHORING_SCHEMA_VERSION:
        raise PairT2VAuthoringError("selection registry schema differs")
    selected = selection["selected_iids"]
    expected_count = selection["expected_cell_count"]
    if (
        type(expected_count) is not int
        or not isinstance(selected, list)
        or any(
            not isinstance(iid, str) or contract._SAFE_ID_RE.fullmatch(iid) is None
            for iid in selected
        )
        or len(selected) != expected_count
        or len(set(selected)) != len(selected)
    ):
        raise PairT2VAuthoringError("selection IID count differs")
    registry_cells = registry["cells"]
    if not isinstance(registry_cells, list) or any(
        not isinstance(cell, Mapping) for cell in registry_cells
    ):
        raise PairT2VAuthoringError("selection registry cells differ")
    registry_iids = [cell.get("iid") for cell in registry_cells]
    if len(set(registry_iids)) != len(registry_iids):
        raise PairT2VAuthoringError("selection registry repeats an IID")
    by_iid = {cell["iid"]: cell for cell in registry_cells}
    missing = [iid for iid in selected if iid not in by_iid]
    if missing:
        raise PairT2VAuthoringError(f"selection IIDs are absent from registry: {missing!r}")
    return {
        "schema_version": contract.AUTHORING_SCHEMA_VERSION,
        "bank_id": selection["bank_id"],
        "expected_cell_count": expected_count,
        "cells": [by_iid[iid] for iid in selected],
    }


def _text(value: Any, label: str, *, minimum_words: int = 4) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise PairT2VAuthoringError(f"{label} must be text without NUL")
    result = value.strip()
    if len(result.split()) < minimum_words or "{" in result or "}" in result:
        raise PairT2VAuthoringError(f"{label} is incomplete or contains a placeholder")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bank(authoring: Mapping[str, Any]) -> dict[str, Any]:
    root = _closed(authoring, _AUTHORING_FIELDS, "authoring root")
    if root["schema_version"] != contract.AUTHORING_SCHEMA_VERSION:
        raise PairT2VAuthoringError("authoring schema_version differs")
    bank_id = contract._safe_id(root["bank_id"], "bank_id")
    expected_count = root["expected_cell_count"]
    cells = root["cells"]
    if (
        type(expected_count) is not int
        or expected_count < 2
        or not isinstance(cells, list)
        or len(cells) != expected_count
    ):
        raise PairT2VAuthoringError("authoring cell count differs")
    groups = {
        group_id: {"group_id": group_id, "visible_gpus": gpus, "candidates": []}
        for group_id, gpus in contract.GROUP_LAYOUT
    }
    seen_iids: set[str] = set()
    for cell_index, raw_cell in enumerate(cells):
        cell = _closed(raw_cell, _CELL_FIELDS, f"cell[{cell_index}]")
        iid = contract._safe_id(cell["iid"], f"cell[{cell_index}].iid")
        if iid in seen_iids:
            raise PairT2VAuthoringError("authoring IIDs must be unique")
        seen_iids.add(iid)
        split = cell["analysis_split"]
        if split not in contract.ANALYSIS_SPLITS:
            raise PairT2VAuthoringError("authoring split differs")
        execution_group = cell["execution_group"]
        if execution_group not in groups:
            raise PairT2VAuthoringError("authoring execution_group differs")
        geometry_path = Path(str(cell["geometry_source_video"]))
        if (
            not geometry_path.is_absolute()
            or not geometry_path.is_file()
            or geometry_path.is_symlink()
        ):
            raise PairT2VAuthoringError(
                f"geometry video for {iid} must be an absolute plain file"
            )
        seed = cell["seed"]
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise PairT2VAuthoringError(f"seed for {iid} is outside [0,2^63)")
        action_family = contract._safe_id(
            cell["action_family_id"], f"{iid}.action_family_id"
        )
        actor_group = contract._safe_id(cell["actor_group_id"], f"{iid}.actor_group_id")
        scene_group = contract._safe_id(cell["scene_group_id"], f"{iid}.scene_group_id")
        action_group = contract._safe_id(cell["action_group_id"], f"{iid}.action_group_id")
        prompt_group = f"{actor_group}--{scene_group}"
        contract._safe_id(prompt_group, f"{iid}.prompt_group_id")
        scene = _text(cell["scene_caption"], f"{iid}.scene_caption", minimum_words=8)
        camera = _text(cell["camera_caption"], f"{iid}.camera_caption", minimum_words=6)
        descriptions = _closed(
            cell["branch_descriptions"], set(contract.MACE_BRANCH_ORDER),
            f"{iid}.branch_descriptions",
        )
        geometry_sha = _file_sha256(geometry_path)
        calibration_group = f"cell-{iid}-s{seed}"
        contract._safe_id(calibration_group, f"{iid}.calibration_group_id")
        for branch in contract.MACE_BRANCH_ORDER:
            description = _text(
                descriptions[branch], f"{iid}.{branch}", minimum_words=6
            )
            caption = " ".join((scene, description, camera))
            candidate_id = f"{bank_id}-{iid}-{branch}"
            if len(candidate_id) > 96:
                candidate_id = f"{iid}-{branch}"
            candidate = {
                "candidate_id": candidate_id,
                "analysis_split": split,
                "action_family_id": action_family,
                "calibration_group_id": calibration_group,
                "prompt_group_id": prompt_group,
                "action_family_group_id": action_group,
                "actor_group_id": actor_group,
                "scene_group_id": scene_group,
                "action_group_id": action_group,
                "geometry_source_video": str(geometry_path),
                "geometry_source_video_sha256": geometry_sha,
                "geometry_contract": contract.GEOMETRY_CONTRACT,
                "semantic_branch": branch,
                "full_t2v_caption": caption,
                "full_t2v_caption_utf8_sha256": hashlib.sha256(
                    caption.encode("utf-8")
                ).hexdigest(),
                "caption_contract": contract.CAPTION_CONTRACT,
                "seed": seed,
            }
            groups[execution_group]["candidates"].append(candidate)
    bank = {
        "schema_version": contract.SCHEMA_VERSION,
        "sampling_contract": contract.SAMPLING_CONTRACT,
        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
        "artifact_use_contract": contract.ARTIFACT_USE_CONTRACT,
        "split_contract": contract.SPLIT_CONTRACT,
        "groups": [groups[group_id] for group_id, _ in contract.GROUP_LAYOUT],
    }
    try:
        return contract.validate_root_spec(bank)
    except contract.PairT2VCalibrationSpecError as error:
        raise PairT2VAuthoringError(str(error)) from error


def inspect_authoring(
    *, authoring_path: str | Path, expected_authoring_sha256: str
) -> dict[str, Any]:
    """Resolve a full registry or sealed selection without touching videos."""

    value = _load_authoring(authoring_path, expected_authoring_sha256)
    cells = value["cells"]
    return {
        "bank_id": value["bank_id"],
        "cell_count": len(cells),
        "candidate_count": len(cells) * len(contract.MACE_BRANCH_ORDER),
        "selected_iids": [cell["iid"] for cell in cells],
    }


def write_bank(
    *, authoring_path: str | Path, expected_authoring_sha256: str, output_path: str | Path
) -> dict[str, Any]:
    authoring = _load_authoring(authoring_path, expected_authoring_sha256)
    bank = build_bank(authoring)
    output = Path(output_path)
    if (
        not output.is_absolute()
        or output == Path("/")
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or output.exists()
        or output.is_symlink()
    ):
        raise PairT2VAuthoringError("output must be a fresh absolute file in a plain directory")
    raw = contract.canonical_json_bytes(bank) + b"\n"
    output.write_bytes(raw)
    os.chmod(output, 0o400)
    return {
        "schema_version": "pair-v5-pure-t2v-calibration-authoring-receipt-v1",
        "authoring_raw_sha256": expected_authoring_sha256,
        "output_path": str(output),
        "output_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "cell_count": authoring["expected_cell_count"],
        "candidate_count": sum(len(group["candidates"]) for group in bank["groups"]),
        "caption_hashes_computed": True,
        "geometry_hashes_computed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoring", required=True)
    parser.add_argument("--expected-authoring-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = write_bank(
        authoring_path=args.authoring,
        expected_authoring_sha256=args.expected_authoring_sha256,
        output_path=args.output,
    )
    print(contract.canonical_json_bytes(receipt).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
