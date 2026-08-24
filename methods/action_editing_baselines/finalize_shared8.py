#!/usr/bin/env python3
"""Audit all shared-8 outputs, then materialize a post-inference review gallery."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shared8_contract import (
    INPUT_SCHEMA,
    RECEIPT_SCHEMA,
    Shared8ContractError,
    atomic_write_json,
    file_sha256,
    load_input_manifest,
    object_sha256,
    probe_video,
    require_81f25,
    require_sha256,
)
from run_shared8 import MODEL_IDS


REFERENCE_SCHEMA = "action-editing-shared8-reference-v1"
REFERENCE_KEYS = {"schema_version", "index", "iid", "target_video"}
SUMMARY_SCHEMA = "action-editing-shared8-final-audit-v1"


def _file(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser()
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise Shared8ContractError(f"cannot resolve {label} {source}: {error}") from error
    if not resolved.is_file():
        raise Shared8ContractError(f"{label} is not a file: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_file(path, label=label).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Shared8ContractError(f"invalid {label} JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise Shared8ContractError(f"{label} is not a JSON object: {path}")
    return value


def _verify_digest(value: dict[str, Any], *, label: str) -> str:
    payload = dict(value)
    stored = payload.pop("receipt_digest", None)
    require_sha256(stored, label=f"{label} receipt_digest")
    observed = object_sha256(payload)
    if observed != stored:
        raise Shared8ContractError(f"{label} digest mismatch: {observed} != {stored}")
    return stored


def load_references(
    path: str | Path, *, expected_sha256: str, rows: list[Any]
) -> tuple[Path, list[dict[str, Any]]]:
    reference_path = _file(path, label="reference manifest")
    require_sha256(expected_sha256, label="reference-manifest SHA-256")
    observed = file_sha256(reference_path)
    if observed != expected_sha256:
        raise Shared8ContractError(
            f"reference-manifest SHA-256 mismatch: {observed} != {expected_sha256}"
        )
    values: list[dict[str, Any]] = []
    with reference_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise Shared8ContractError(
                    f"invalid reference JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict) or set(value) != REFERENCE_KEYS:
                raise Shared8ContractError(
                    f"reference line {line_number} violates the closed schema"
                )
            index = len(values)
            if (
                value["schema_version"] != REFERENCE_SCHEMA
                or type(value["index"]) is not int
                or value["index"] != index
                or index >= len(rows)
                or value["iid"] != rows[index].iid
            ):
                raise Shared8ContractError(
                    f"reference line {line_number} is not aligned with the input manifest"
                )
            target = Path(value["target_video"])
            if not target.is_absolute():
                raise Shared8ContractError(f"reference target is not absolute: {target}")
            values.append(value)
    if len(values) != len(rows):
        raise Shared8ContractError(
            f"reference count mismatch: {len(values)} != {len(rows)}"
        )
    return reference_path, values


def _validate_model_receipt(
    *, output_root: Path, model_id: str, row: Any, manifest_sha256: str, ffprobe: str
) -> dict[str, Any]:
    sample_dir = output_root / model_id / f"sample_{row.index:03d}_{row.iid}"
    receipt_path = sample_dir / "receipt.json"
    receipt = _read_json(receipt_path, label=f"{model_id}/{row.iid} receipt")
    digest = _verify_digest(receipt, label=f"{model_id}/{row.iid}")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise Shared8ContractError(f"unexpected receipt schema: {receipt_path}")
    if receipt.get("model_id") != model_id or receipt.get("sample") != asdict(row):
        raise Shared8ContractError(f"receipt sample/model mismatch: {receipt_path}")
    contract = receipt.get("input_contract")
    if not isinstance(contract, dict):
        raise Shared8ContractError(f"receipt has no input contract: {receipt_path}")
    if contract.get("manifest_sha256") != manifest_sha256:
        raise Shared8ContractError(f"receipt manifest mismatch: {receipt_path}")
    if contract.get("accepted_model_conditions") != [
        "source_video",
        "edit_instruction",
    ]:
        raise Shared8ContractError(f"receipt condition set mismatch: {receipt_path}")
    for key in (
        "target_video_argument",
        "target_video_accessed",
        "external_mask_or_swept_tube",
        "external_tracking_pose_or_trajectory",
        "reference_media",
        "external_shared_i0",
    ):
        if contract.get(key) is not False:
            raise Shared8ContractError(f"receipt does not prove {key}=false: {receipt_path}")
    source = _file(row.source_video, label=f"source {row.iid}")
    if contract.get("source_video_sha256") != file_sha256(source):
        raise Shared8ContractError(f"source hash mismatch: {receipt_path}")
    output_value = receipt.get("output")
    if not isinstance(output_value, dict):
        raise Shared8ContractError(f"receipt has no output object: {receipt_path}")
    output = _file(output_value.get("path", ""), label=f"output {model_id}/{row.iid}")
    expected_output = (sample_dir / "output.mp4").resolve(strict=True)
    if output != expected_output:
        raise Shared8ContractError(f"non-canonical output path: {output} != {expected_output}")
    output_sha256 = file_sha256(output)
    if output_value.get("sha256") != output_sha256:
        raise Shared8ContractError(f"output hash mismatch: {receipt_path}")
    output_probe = probe_video(output, ffprobe=ffprobe)
    require_81f25(output_probe, label=f"output {model_id}/{row.iid}")
    return {
        "model_id": model_id,
        "receipt_path": str(receipt_path.resolve(strict=True)),
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_digest": digest,
        "output_path": str(output),
        "output_sha256": output_sha256,
        "output_probe": asdict(output_probe),
        "model_identity": receipt.get("model_identity"),
        "sampler": receipt.get("sampler"),
        "geometry": receipt.get("geometry"),
    }


def _copy_new(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise Shared8ContractError(f"refusing to overwrite gallery media: {destination}")
    shutil.copy2(source, destination)


def _materialize_gallery(
    *, gallery: Path, rows: list[Any], references: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> Path:
    if gallery.exists() or gallery.is_symlink():
        raise Shared8ContractError(f"refusing existing gallery directory: {gallery}")
    media = gallery / "media"
    media.mkdir(parents=True)
    by_index = {sample["index"]: sample for sample in samples}
    sections: list[str] = []
    for row, reference in zip(rows, references):
        source_name = f"{row.index:03d}_{row.iid}__source.mp4"
        target_name = f"{row.index:03d}_{row.iid}__target.mp4"
        _copy_new(Path(row.source_video), media / source_name)
        _copy_new(Path(reference["target_video"]), media / target_name)
        cells = [
            ("source", source_name),
            ("reference (evaluation only)", target_name),
        ]
        sample = by_index[row.index]
        for model_id in sorted(MODEL_IDS):
            name = f"{row.index:03d}_{row.iid}__{model_id}.mp4"
            _copy_new(Path(sample["models"][model_id]["output_path"]), media / name)
            cells.append((model_id, name))
        videos = "".join(
            "<td><div class='label'>"
            + html.escape(label)
            + "</div><video controls muted loop preload='metadata' src='media/"
            + html.escape(name, quote=True)
            + "'></video></td>"
            for label, name in cells
        )
        sections.append(
            f"<section><h2>{row.index:03d} · {html.escape(row.iid)} · "
            f"{html.escape(row.split)}</h2><p>{html.escape(row.instruction)}</p>"
            f"<table><tr>{videos}</tr></table></section>"
        )
    document = """<!doctype html><meta charset='utf-8'>
<title>Action-editing shared-8 review</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
section{margin:0 0 36px} table{border-collapse:collapse;width:100%} td{padding:5px;vertical-align:top}
video{width:100%;max-width:360px}.label{font:12px monospace;margin-bottom:4px;color:#bbb}
</style>
<h1>Action-editing shared-8 · post-inference review</h1>
<p>All model calls used source video plus edit instruction only. Targets first become available here.</p>
""" + "\n".join(sections)
    index = gallery / "index.html"
    with index.open("x", encoding="utf-8") as handle:
        handle.write(document)
        handle.flush()
        os.fsync(handle.fileno())
    return index


def run(args: argparse.Namespace) -> int:
    require_sha256(args.manifest_sha256, label="input-manifest SHA-256")
    manifest, rows = load_input_manifest(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        require_media=True,
    )
    output_root = Path(args.output_root).expanduser().resolve(strict=True)
    if not output_root.is_dir():
        raise Shared8ContractError(f"output root is not a directory: {output_root}")

    # All inference receipts are closed and validated before target paths are read.
    samples: list[dict[str, Any]] = []
    for row in rows:
        models = {
            model_id: _validate_model_receipt(
                output_root=output_root,
                model_id=model_id,
                row=row,
                manifest_sha256=args.manifest_sha256,
                ffprobe=args.ffprobe,
            )
            for model_id in sorted(MODEL_IDS)
        }
        samples.append(
            {
                "index": row.index,
                "iid": row.iid,
                "split": row.split,
                "instruction": row.instruction,
                "source_video": row.source_video,
                "source_video_sha256": file_sha256(row.source_video),
                "models": models,
            }
        )

    reference_path, references = load_references(
        args.references,
        expected_sha256=args.references_sha256,
        rows=rows,
    )
    targets: list[dict[str, Any]] = []
    for value in references:
        target = _file(value["target_video"], label=f"target {value['iid']}")
        target_probe = probe_video(target, ffprobe=args.ffprobe)
        require_81f25(target_probe, label=f"target {value['iid']}")
        targets.append(
            {
                "index": value["index"],
                "iid": value["iid"],
                "path": str(target),
                "sha256": file_sha256(target),
                "probe": asdict(target_probe),
            }
        )

    gallery_path: str | None = None
    if args.gallery:
        gallery_index = _materialize_gallery(
            gallery=Path(args.gallery).expanduser(),
            rows=rows,
            references=references,
            samples=samples,
        )
        gallery_path = str(gallery_index.resolve(strict=True))

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "input_contract": {
            "schema_version": INPUT_SCHEMA,
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "model_conditions": ["source_video", "edit_instruction"],
            "target_available_during_inference": False,
            "external_mask_track_pose_trajectory_or_shared_i0": False,
        },
        "reference_contract": {
            "manifest_path": str(reference_path),
            "manifest_sha256": file_sha256(reference_path),
            "first_access_stage": "post_inference_evaluation",
        },
        "frame_contract": {"frame_count": 81, "fps": 25.0},
        "models": sorted(MODEL_IDS),
        "sample_count": len(samples),
        "output_count": sum(len(item["models"]) for item in samples),
        "samples": samples,
        "targets": targets,
        "gallery_index": gallery_path,
        "claim_limits": {
            "engineering_diagnostic_only": True,
            "content_disjoint_split": False,
            "scientific_superiority_claim_authorized": False,
        },
    }
    atomic_write_json(args.summary, summary)
    print(
        json.dumps(
            {
                "summary": str(Path(args.summary).resolve(strict=True)),
                "summary_sha256": file_sha256(args.summary),
                "samples": len(samples),
                "outputs": summary["output_count"],
                "gallery": gallery_path,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--gallery")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
