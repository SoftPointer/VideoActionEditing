"""Prepare a closed, media-qualified input pool for full-motion annotation.

This stage deliberately reuses only the source-media prefilter.  It does not
reuse any v8 Qwen decision or executable instruction.  Selected rows are
copied byte-for-byte from the prefilter manifest so the later visual audit can
bind its evidence to the original source/anchor hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "motive-goku-full-motion-prepare-v1"
PREFILTER_SCHEMA = "motive-goku-action-anchor-prefilter-v1"
CANDIDATES_NAME = "candidates.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class FullMotionPrepareError(ValueError):
    """The source prefilter or requested candidate set is not closed."""


def _reject_constant(value: str) -> None:
    raise FullMotionPrepareError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullMotionPrepareError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FullMotionPrepareError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, FullMotionPrepareError):
            raise
        raise FullMotionPrepareError(
            f"{context} is not strict JSON: {error}"
        ) from error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
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


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _regular_file(path: Path, *, context: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise FullMotionPrepareError(
            f"{context} must be a regular non-symlink file: {path}"
        )
    return resolved


def _regular_directory(path: Path, *, context: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_dir():
        raise FullMotionPrepareError(
            f"{context} must be a non-symlink directory: {path}"
        )
    return resolved


def _strict_object(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    regular = _regular_file(path, context=context)
    raw = regular.read_bytes()
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise FullMotionPrepareError(f"{context} must contain one object")
    return value, raw


def _strict_jsonl(
    path: Path,
    *,
    context: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes], bytes]:
    regular = _regular_file(path, context=context)
    raw = regular.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise FullMotionPrepareError(
            f"{context} must be non-empty and newline terminated"
        )
    rows: list[dict[str, Any]] = []
    lines_by_iid: dict[str, bytes] = {}
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if line in {b"", b"\n", b"\r\n"}:
            raise FullMotionPrepareError(
                f"{context} has a blank line at {line_number}"
            )
        value = _parse_json(line, context=f"{context} line {line_number}")
        if not isinstance(value, dict):
            raise FullMotionPrepareError(
                f"{context} line {line_number} is not an object"
            )
        iid = value.get("iid")
        if (
            not isinstance(iid, str)
            or _SAFE_IID_RE.fullmatch(iid) is None
            or iid in {".", ".."}
        ):
            raise FullMotionPrepareError(
                f"{context} line {line_number} has an unsafe IID"
            )
        if iid in lines_by_iid:
            raise FullMotionPrepareError(f"duplicate IID in {context}: {iid}")
        rows.append(value)
        lines_by_iid[iid] = line
    return rows, lines_by_iid, raw


def _sha256_field(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FullMotionPrepareError(f"{context} is not a lowercase SHA-256")
    return value


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullMotionPrepareError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FullMotionPrepareError(f"{context} must be finite")
    return result


def _validate_prefilter_closure(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes], bytes, dict[str, Any]]:
    expected_entries = {
        "anchors",
        "evaluated.jsonl",
        "selected.jsonl",
        "summary.json",
        "done.json",
    }
    actual_entries = {entry.name for entry in root.iterdir()}
    if actual_entries != expected_entries:
        raise FullMotionPrepareError(
            "prefilter artifact closure differs: "
            f"{sorted(actual_entries ^ expected_entries)}"
        )
    anchors = _regular_directory(root / "anchors", context="anchor directory")
    selected_rows, lines_by_iid, selected_raw = _strict_jsonl(
        root / "selected.jsonl",
        context="prefilter selected manifest",
    )
    evaluated = _regular_file(
        root / "evaluated.jsonl",
        context="prefilter evaluated manifest",
    )
    summary, summary_raw = _strict_object(
        root / "summary.json",
        context="prefilter summary",
    )
    done, _ = _strict_object(root / "done.json", context="prefilter done")
    for value, label in ((summary, "summary"), (done, "done")):
        if value.get("schema_version") != PREFILTER_SCHEMA:
            raise FullMotionPrepareError(f"prefilter {label} schema differs")
        if value.get("status") != "complete":
            raise FullMotionPrepareError(f"prefilter {label} is not complete")
    counts = summary.get("counts")
    artifacts = done.get("artifacts")
    anchor_sha256 = done.get("anchor_sha256")
    if not isinstance(counts, Mapping) or not isinstance(artifacts, Mapping):
        raise FullMotionPrepareError("prefilter summary/done closure is malformed")
    if not isinstance(anchor_sha256, Mapping):
        raise FullMotionPrepareError("prefilter anchor digest map is malformed")
    if counts.get("selected") != len(selected_rows):
        raise FullMotionPrepareError("prefilter selected count differs")
    if done.get("selected_rows") != len(selected_rows):
        raise FullMotionPrepareError("prefilter done selected count differs")
    expected_hashes = {
        "selected.jsonl": _sha256_bytes(selected_raw),
        "evaluated.jsonl": _sha256_bytes(evaluated.read_bytes()),
        "summary.json": _sha256_bytes(summary_raw),
    }
    for name, digest in expected_hashes.items():
        if artifacts.get(name) != digest:
            raise FullMotionPrepareError(f"prefilter {name} digest differs")
    observed_anchors: dict[str, str] = {}
    for row in selected_rows:
        relative = row.get("anchor_image")
        expected = _sha256_field(
            row.get("anchor_sha256"),
            context=f"iid={row.get('iid')} anchor SHA",
        )
        if not isinstance(relative, str) or not relative.startswith("anchors/"):
            raise FullMotionPrepareError("selected anchor path is not canonical")
        candidate = (root / relative).resolve(strict=True)
        if candidate.parent != anchors or candidate.is_symlink() or not candidate.is_file():
            raise FullMotionPrepareError("selected anchor escapes anchor directory")
        actual = _sha256_bytes(candidate.read_bytes())
        if actual != expected:
            raise FullMotionPrepareError(
                f"selected anchor digest differs for iid={row.get('iid')}"
            )
        observed_anchors[relative] = actual
    if dict(sorted(observed_anchors.items())) != dict(sorted(anchor_sha256.items())):
        raise FullMotionPrepareError("prefilter anchor digest map differs")
    if artifacts.get("anchors") != _object_digest(observed_anchors):
        raise FullMotionPrepareError("prefilter aggregate anchor digest differs")
    return selected_rows, lines_by_iid, selected_raw, summary


def _eligible_row(
    row: Mapping[str, Any],
    *,
    min_short_side: int,
) -> bool:
    if row.get("schema_version") != PREFILTER_SCHEMA:
        raise FullMotionPrepareError(
            f"iid={row.get('iid')} prefilter row schema differs"
        )
    if row.get("eligible") is not True or row.get("selected") is not True:
        raise FullMotionPrepareError(
            f"iid={row.get('iid')} is not a selected eligible row"
        )
    for field in ("group_id", "resolved_src_video", "resolved_anchor_image"):
        value = row.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise FullMotionPrepareError(
                f"iid={row.get('iid')} has invalid {field}"
            )
    for field in ("source_video_sha256", "anchor_sha256"):
        _sha256_field(
            row.get(field),
            context=f"iid={row.get('iid')} {field}",
        )
    media = row.get("media")
    motion = row.get("motion")
    if not isinstance(media, Mapping) or not isinstance(motion, Mapping):
        raise FullMotionPrepareError(
            f"iid={row.get('iid')} lacks media/motion objects"
        )
    rank = row.get("selection_rank")
    if type(rank) is not int or rank <= 0:
        raise FullMotionPrepareError(
            f"iid={row.get('iid')} selection_rank is invalid"
        )
    frames = media.get("frame_count")
    if frames != 81:
        return False
    fps = _finite_number(media.get("fps"), context="media.fps")
    duration = _finite_number(
        media.get("duration_seconds"),
        context="media.duration_seconds",
    )
    short_side = media.get("short_side")
    scene_cut = _finite_number(
        motion.get("scene_cut_ratio"),
        context="motion.scene_cut_ratio",
    )
    if type(short_side) is not int:
        raise FullMotionPrepareError("media.short_side must be an integer")
    return (
        short_side >= min_short_side
        and math.isclose(fps, 25.0, rel_tol=0.0, abs_tol=1e-6)
        and 3.15 <= duration <= 3.30
        and scene_cut <= 0.0
    )


def prepare_candidates(
    *,
    prefilter_dir: str | Path,
    output_dir: str | Path,
    sample_size: int = 768,
    min_short_side: int = 640,
    required_iids: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate the media artifact and publish a fixed annotation manifest."""

    if type(sample_size) is not int or sample_size <= 0:
        raise FullMotionPrepareError("sample_size must be a positive integer")
    if type(min_short_side) is not int or min_short_side <= 0:
        raise FullMotionPrepareError("min_short_side must be positive")
    required = list(required_iids)
    if len(set(required)) != len(required):
        raise FullMotionPrepareError("required_iids contains duplicates")
    if any(
        not isinstance(iid, str)
        or _SAFE_IID_RE.fullmatch(iid) is None
        or iid in {".", ".."}
        for iid in required
    ):
        raise FullMotionPrepareError("required_iids contains an unsafe IID")

    root = _regular_directory(Path(prefilter_dir), context="prefilter root")
    rows, lines_by_iid, selected_raw, prefilter_summary = (
        _validate_prefilter_closure(root)
    )
    eligible = [
        row
        for row in rows
        if _eligible_row(row, min_short_side=min_short_side)
    ]
    eligible.sort(key=lambda row: (int(row["selection_rank"]), str(row["iid"])))
    by_iid = {str(row["iid"]): row for row in eligible}
    missing_required = [iid for iid in required if iid not in by_iid]
    if missing_required:
        raise FullMotionPrepareError(
            f"required IIDs are not media eligible: {missing_required}"
        )
    chosen = [by_iid[iid] for iid in required]
    chosen.extend(row for row in eligible if str(row["iid"]) not in set(required))
    chosen = chosen[:sample_size]
    if len(chosen) != sample_size:
        raise FullMotionPrepareError(
            f"eligible candidate shortfall: {len(chosen)}/{sample_size}"
        )
    groups = [str(row["group_id"]) for row in chosen]
    if len(set(groups)) != len(groups):
        raise FullMotionPrepareError("candidate group_id values are not unique")
    candidate_raw = b"".join(lines_by_iid[str(row["iid"])] for row in chosen)

    target = Path(output_dir).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise FullMotionPrepareError("output parent is not a plain directory")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    )
    try:
        candidate_path = stage / CANDIDATES_NAME
        candidate_path.write_bytes(candidate_raw)
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "semantics": {
                "source_media_only": True,
                "legacy_qwen_decisions_reused": False,
                "legacy_instructions_authoritative": False,
                "next_stage": "qwen3_vl_32b_full_motion_audit",
            },
            "input": {
                "prefilter_dir": str(root),
                "selected_path": str(root / "selected.jsonl"),
                "selected_sha256": _sha256_bytes(selected_raw),
                "selected_rows": len(rows),
                "prefilter_summary_sha256": _sha256_bytes(
                    (root / "summary.json").read_bytes()
                ),
                "prefilter_done_sha256": _sha256_bytes(
                    (root / "done.json").read_bytes()
                ),
                "prefilter_config": prefilter_summary.get("config"),
            },
            "selection": {
                "sample_size": sample_size,
                "min_short_side": min_short_side,
                "frame_count": 81,
                "fps": 25.0,
                "duration_range_seconds": [3.15, 3.30],
                "scene_cut_ratio_max": 0.0,
                "required_iids": required,
                "eligible_after_geometry": len(eligible),
                "iids": [str(row["iid"]) for row in chosen],
                "group_ids_unique": True,
            },
            "output": {
                "path": str(target / CANDIDATES_NAME),
                "rows": len(chosen),
                "bytes": len(candidate_raw),
                "sha256": _sha256_bytes(candidate_raw),
            },
        }
        summary_raw = _pretty_bytes(summary)
        (stage / SUMMARY_NAME).write_bytes(summary_raw)
        done = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "artifacts": {
                CANDIDATES_NAME: _sha256_bytes(candidate_raw),
                SUMMARY_NAME: _sha256_bytes(summary_raw),
            },
            "artifact_digest": _object_digest(
                {
                    CANDIDATES_NAME: _sha256_bytes(candidate_raw),
                    SUMMARY_NAME: _sha256_bytes(summary_raw),
                }
            ),
        }
        (stage / DONE_NAME).write_bytes(_pretty_bytes(done))
        for path in stage.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare source-only candidates for full-motion Qwen audit."
    )
    parser.add_argument("--prefilter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=768)
    parser.add_argument("--min-short-side", type=int, default=640)
    parser.add_argument("--require-iid", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_candidates(
        prefilter_dir=args.prefilter_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        min_short_side=args.min_short_side,
        required_iids=args.require_iid,
    )
    print(
        "[goku-full-motion-prepare] "
        f"rows={summary['output']['rows']} output={summary['output']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
