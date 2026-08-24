#!/usr/bin/env python3
"""Render blinded primary and named-reference sheets from a final shared-8 audit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared8_contract import Shared8ContractError, file_sha256, object_sha256


FRAME_INDICES = (0, 20, 40, 60, 80)
MODEL_ORDER = (
    "lucy_official_base",
    "bernini_full644_lora_step644",
    "omnivideo2_official_base",
)
MODEL_LABELS = {
    "lucy_official_base": "Lucy-Edit 1.1 official base",
    "bernini_full644_lora_step644": "Bernini-R 1.3B + full644 LoRA",
    "omnivideo2_official_base": "OmniVideo2 1.3B official base",
}
PRIMARY_LABELS = ("source", "candidate A", "candidate B", "candidate C")
BLIND_PERMUTATION_DOMAIN = "action-editing-shared8-blind-permutation-v1"
_MODEL_PERMUTATIONS = tuple(itertools.permutations(MODEL_ORDER))


@dataclass(frozen=True)
class RenderRow:
    label: str
    video: Path


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Shared8ContractError(f"cannot read final audit {path}: {error}") from error
    if not isinstance(value, dict):
        raise Shared8ContractError("final audit is not a JSON object")
    payload = dict(value)
    stored = payload.pop("receipt_digest", None)
    if not isinstance(stored, str) or object_sha256(payload) != stored:
        raise Shared8ContractError("final-audit receipt digest mismatch")
    if value.get("schema_version") != "action-editing-shared8-final-audit-v1":
        raise Shared8ContractError("unexpected final-audit schema")
    if value.get("sample_count") != 8 or value.get("output_count") != 24:
        raise Shared8ContractError("final audit does not contain the complete 3x8 run")
    _validate_summary_layout(value)
    return value


def _validate_summary_layout(summary: dict[str, Any]) -> None:
    samples = summary.get("samples")
    targets = summary.get("targets")
    if not isinstance(samples, list) or not isinstance(targets, list):
        raise Shared8ContractError("final audit samples/targets are not lists")
    if summary.get("models") != sorted(MODEL_ORDER):
        raise Shared8ContractError("final audit model set/order mismatch")
    if len(samples) != 8 or len(targets) != 8:
        raise Shared8ContractError("final audit does not have exactly eight samples/targets")
    expected_indices = list(range(8))
    if [sample.get("index") for sample in samples if isinstance(sample, dict)] != expected_indices:
        raise Shared8ContractError("final audit sample indexes are not exactly 0..7")
    if [target.get("index") for target in targets if isinstance(target, dict)] != expected_indices:
        raise Shared8ContractError("final audit target indexes are not exactly 0..7")
    for sample, target in zip(samples, targets):
        if not isinstance(sample, dict) or not isinstance(target, dict):
            raise Shared8ContractError("final audit sample/target entry is not an object")
        if sample.get("iid") != target.get("iid"):
            raise Shared8ContractError("final audit sample/target IID mismatch")
        models = sample.get("models")
        if not isinstance(models, dict) or set(models) != set(MODEL_ORDER):
            raise Shared8ContractError(
                f"final audit has an incomplete model set for sample {sample.get('index')}"
            )


def _blind_model_order(*, audit_digest: str, index: int, iid: str) -> tuple[str, ...]:
    """Select one of the six model permutations using a stable per-sample SHA-256."""
    material = f"{BLIND_PERMUTATION_DOMAIN}\0{audit_digest}\0{index}\0{iid}".encode("utf-8")
    selector = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return _MODEL_PERMUTATIONS[selector % len(_MODEL_PERMUTATIONS)]


def _primary_blind_rows(
    sample: dict[str, Any], *, audit_digest: str
) -> tuple[list[RenderRow], dict[str, str]]:
    order = _blind_model_order(
        audit_digest=audit_digest,
        index=int(sample["index"]),
        iid=str(sample["iid"]),
    )
    rows = [RenderRow("source", Path(sample["source_video"]))]
    mapping: dict[str, str] = {}
    for letter, model_id in zip("ABC", order):
        rows.append(
            RenderRow(
                f"candidate {letter}",
                Path(sample["models"][model_id]["output_path"]),
            )
        )
        mapping[letter] = model_id
    return rows, mapping


def _reference_named_rows(
    sample: dict[str, Any], target: dict[str, Any]
) -> list[RenderRow]:
    rows = [
        RenderRow("source", Path(sample["source_video"])),
        RenderRow("reference target (evaluation only)", Path(target["path"])),
    ]
    rows.extend(
        RenderRow(MODEL_LABELS[model_id], Path(sample["models"][model_id]["output_path"]))
        for model_id in MODEL_ORDER
    )
    return rows


def _require_media(rows: list[RenderRow], *, label: str) -> None:
    for row in rows:
        if not row.video.is_file():
            raise Shared8ContractError(f"{label} review media is missing: {row.video}")


def _verify_bound_media(sample: dict[str, Any], target: dict[str, Any]) -> None:
    source_path = Path(sample["source_video"])
    if not source_path.is_file() or file_sha256(source_path) != sample.get(
        "source_video_sha256"
    ):
        raise Shared8ContractError(f"source changed after final audit: {source_path}")
    target_path = Path(target["path"])
    if not target_path.is_file() or file_sha256(target_path) != target.get("sha256"):
        raise Shared8ContractError(f"target changed after final audit: {target_path}")
    for model_id in MODEL_ORDER:
        model = sample["models"][model_id]
        output = Path(model["output_path"])
        if not output.is_file() or file_sha256(output) != model.get("output_sha256"):
            raise Shared8ContractError(f"model output changed after final audit: {output}")


def _extract_frames(video: Path, destination: Path, *, ffmpeg: str) -> list[Path]:
    destination.mkdir()
    expression = "+".join(f"eq(n\\,{index})" for index in FRAME_INDICES)
    pattern = destination / "frame_%02d.png"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select={expression}",
        "-vsync",
        "0",
        str(pattern),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise Shared8ContractError(
            f"ffmpeg frame extraction failed for {video}: {error}"
        ) from error
    frames = sorted(destination.glob("frame_*.png"))
    if len(frames) != len(FRAME_INDICES):
        raise Shared8ContractError(
            f"expected {len(FRAME_INDICES)} review frames for {video}, got {len(frames)}"
        )
    return frames


def _fit_frame(path: Path, *, width: int, height: int) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        frame = source.convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    frame.thumbnail((width, height), resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "black")
    left = (width - frame.width) // 2
    top = (height - frame.height) // 2
    canvas.paste(frame, (left, top))
    return canvas


def _render_sheet(
    *,
    sample: dict[str, Any],
    rows: list[RenderRow],
    output: Path,
    temporary_root: Path,
    ffmpeg: str,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    index = int(sample["index"])
    iid = str(sample["iid"])
    cell_w, cell_h = 320, 224
    label_w, header_h = 270, 112
    row_h = cell_h + 34
    sheet = Image.new(
        "RGB",
        (label_w + cell_w * len(FRAME_INDICES), header_h + row_h * len(rows)),
        "#111111",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 10), f"{index:03d} · {iid} · {sample['split']}", fill="white", font=font)
    instruction_lines = textwrap.wrap(str(sample["instruction"]), width=150)
    draw.multiline_text(
        (12, 32), "\n".join(instruction_lines[:4]), fill="#dddddd", font=font, spacing=3
    )
    for column, frame_index in enumerate(FRAME_INDICES):
        draw.text(
            (label_w + column * cell_w + 8, header_h - 22),
            f"frame {frame_index}",
            fill="#bbbbbb",
            font=font,
        )

    for row_index, row in enumerate(rows):
        row_top = header_h + row_index * row_h
        draw.text((12, row_top + 10), row.label, fill="white", font=font)
        extracted = _extract_frames(
            row.video,
            temporary_root / f"row_{row_index:02d}",
            ffmpeg=ffmpeg,
        )
        for column, frame_path in enumerate(extracted):
            frame = _fit_frame(frame_path, width=cell_w, height=cell_h)
            sheet.paste(frame, (label_w + column * cell_w, row_top))

    if output.exists() or output.is_symlink():
        raise Shared8ContractError(f"refusing to overwrite contact sheet: {output}")
    sheet.save(output, format="JPEG", quality=92, optimize=True)
    return {"index": index, "iid": iid, "path": output.name, "sha256": file_sha256(output)}


def _primary_manifest(
    *, audit_sha256: str, rendered: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the public manifest without target or candidate-to-model information."""
    return {
        "schema_version": "action-editing-shared8-primary-blind-contact-sheets-v1",
        "source_final_audit_sha256": audit_sha256,
        "frame_indices": list(FRAME_INDICES),
        "row_labels": list(PRIMARY_LABELS),
        "sheets": rendered,
    }


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> int:
    summary_path = Path(args.summary).expanduser().resolve(strict=True)
    summary = _load_summary(summary_path)
    audit_digest = str(summary["receipt_digest"])
    audit_sha256 = file_sha256(summary_path)
    targets = {int(value["index"]): value for value in summary["targets"]}

    plans: list[tuple[dict[str, Any], list[RenderRow], list[RenderRow], dict[str, str]]] = []
    for sample in summary["samples"]:
        target = targets[int(sample["index"])]
        _verify_bound_media(sample, target)
        primary_rows, mapping = _primary_blind_rows(sample, audit_digest=audit_digest)
        reference_rows = _reference_named_rows(sample, target)
        _require_media(primary_rows, label="primary-blind")
        _require_media(reference_rows, label="reference-named")
        plans.append((sample, primary_rows, reference_rows, mapping))

    requested_output_dir = Path(args.output_dir).expanduser()
    if requested_output_dir.exists() or requested_output_dir.is_symlink():
        raise Shared8ContractError(
            f"refusing existing contact-sheet directory: {requested_output_dir}"
        )
    output_dir = requested_output_dir.resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise Shared8ContractError(f"refusing existing contact-sheet directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging_", dir=output_dir.parent))
    try:
        primary_dir = staging / "primary_blind"
        reference_dir = staging / "reference_named"
        primary_dir.mkdir()
        reference_dir.mkdir()
        primary_rendered: list[dict[str, Any]] = []
        reference_rendered: list[dict[str, Any]] = []
        blind_mappings: list[dict[str, Any]] = []
        for sample, primary_rows, reference_rows, mapping in plans:
            index = int(sample["index"])
            iid = str(sample["iid"])
            filename = f"sample_{index:03d}_{iid}.jpg"
            with tempfile.TemporaryDirectory(prefix=f".{index:03d}_primary_", dir=staging) as temp:
                primary_rendered.append(
                    _render_sheet(
                        sample=sample,
                        rows=primary_rows,
                        output=primary_dir / filename,
                        temporary_root=Path(temp),
                        ffmpeg=args.ffmpeg,
                    )
                )
            with tempfile.TemporaryDirectory(
                prefix=f".{index:03d}_reference_", dir=staging
            ) as temp:
                reference_rendered.append(
                    _render_sheet(
                        sample=sample,
                        rows=reference_rows,
                        output=reference_dir / filename,
                        temporary_root=Path(temp),
                        ffmpeg=args.ffmpeg,
                    )
                )
            blind_mappings.append(
                {"index": index, "iid": iid, "candidates": mapping}
            )

        _write_json_new(
            primary_dir / "contact_sheets.json",
            _primary_manifest(audit_sha256=audit_sha256, rendered=primary_rendered),
        )
        _write_json_new(
            reference_dir / "contact_sheets.json",
            {
                "schema_version": "action-editing-shared8-reference-named-contact-sheets-v1",
                "source_final_audit": str(summary_path),
                "source_final_audit_sha256": audit_sha256,
                "frame_indices": list(FRAME_INDICES),
                "row_labels": [
                    "source",
                    "reference target (evaluation only)",
                    *[MODEL_LABELS[model_id] for model_id in MODEL_ORDER],
                ],
                "model_order": list(MODEL_ORDER),
                "sheets": reference_rendered,
            },
        )
        blind_key_payload = {
            "schema_version": "action-editing-shared8-blind-key-v1",
            "source_final_audit": str(summary_path),
            "source_final_audit_sha256": audit_sha256,
            "permutation_algorithm": "sha256-domain-audit_digest-index-iid-mod-6-v1",
            "permutation_domain": BLIND_PERMUTATION_DOMAIN,
            "mappings": blind_mappings,
        }
        _write_json_new(
            staging / "blind_key.json",
            {**blind_key_payload, "receipt_digest": object_sha256(blind_key_payload)},
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise Shared8ContractError(f"refusing existing contact-sheet directory: {output_dir}")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "primary_blind_sheets": len(plans),
                "reference_named_sheets": len(plans),
                "blind_key": str(output_dir / "blind_key.json"),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
