"""Read-only end-to-end audit for Qwen-v16 and streamed Wan previews.

Every audit bundle consists of one explicit candidate manifest, one Qwen-v16
output root, and one Wan stream/output root.  The manifest is the sole IID
authority: directory enumeration can neither add a sample nor make a bundle
complete.  Qwen artifacts are checked through their immutable terminal
receipt and v16 validators.  Wan commits are checked by the official batch
validator, followed by fresh byte, lossless-frame-zero, and ffprobe checks.

The command writes nothing.  It emits one deterministic JSON summary to
stdout and optionally exits nonzero when ``--require-complete`` is requested.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from .goku_action_anchor_qwen import validate_input_row
from .goku_full_motion_qwen_v16 import (
    RECORD_SCHEMA,
    GokuFullMotionQwenV16Error,
    _iter_jsonl,
    _strict_read_object,
    _validate_terminal_receipt,
    object_sha256,
    validate_compiled_instruction,
    validate_passed_row,
    validate_source_census,
    validate_target_plan,
)
from . import wan22_i2v_batch as wan_batch


AUDIT_SCHEMA = "motive-goku-full-motion-v16-audit-v1"
BUNDLE_SCHEMA = "motive-goku-full-motion-v16-audit-bundle-v1"
ROW_SCHEMA = "motive-goku-full-motion-v16-audit-row-v1"
STATUSES = (
    "pending_qwen",
    "qwen_error",
    "pending_wan",
    "wan_error",
    "complete",
)
FRAME_COUNT = 81
FRAME_RATE = "25/1"
CONTAINER_DURATION_SECONDS = 3.24
TIMELINE_SPAN_SECONDS = 3.2
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_QWEN_RECORD_KEYS = {
    "schema_version",
    "iid",
    "status",
    "input_digest",
    "input_row",
    "model",
    "runtime",
    "media_verification",
    "visual_input_digest",
    "source_stage",
    "target_stage",
    "source_census",
    "target_plan",
    "compiled_instruction",
    "error",
    "record_digest",
}


class GokuFullMotionV16AuditError(RuntimeError):
    """The requested audit inputs or an artifact binding are invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _plain_file(path: Path, *, context: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise GokuFullMotionV16AuditError(
            f"{context} must be a non-empty regular non-symlink file: {path}"
        )
    return path.resolve(strict=True)


def _plain_directory(path: Path, *, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise GokuFullMotionV16AuditError(
            f"{context} must be a regular non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _safe_relative_file(
    directory: Path, value: Any, *, context: str
) -> Path:
    if not isinstance(value, str) or Path(value).name != value or value in {".", ".."}:
        raise GokuFullMotionV16AuditError(f"{context} must be one basename")
    return _plain_file(directory / value, context=context)


def _strict_single_jsonl(path: Path) -> dict[str, Any]:
    raw = _plain_file(path, context="passed fragment").read_bytes()
    if not raw.endswith(b"\n") or len(raw.splitlines()) != 1:
        raise GokuFullMotionV16AuditError(
            f"passed fragment must contain exactly one terminated row: {path}"
        )
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise GokuFullMotionV16AuditError(
                    f"duplicate JSON key in passed fragment: {key!r}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.splitlines()[0].decode("utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                GokuFullMotionV16AuditError(f"non-finite JSON: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GokuFullMotionV16AuditError(
            f"passed fragment is not strict UTF-8 JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise GokuFullMotionV16AuditError("passed fragment row must be an object")
    return value


def _read_input_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = _plain_file(path, context="input manifest")
    rows = _iter_jsonl(manifest)
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item = validate_input_row(dict(row))
        iid = str(item["iid"])
        if iid in seen:
            raise GokuFullMotionV16AuditError(f"duplicate input IID: {iid}")
        seen.add(iid)
        validated.append(item)
    return validated, _sha256_file(manifest)


def _audit_qwen(
    row: Mapping[str, Any], *, qwen_root: Path
) -> tuple[str, dict[str, Any], dict[str, Any] | None, Path | None]:
    iid = str(row["iid"])
    result_path = qwen_root / "rows" / iid / "result.json"
    receipt_path = qwen_root / "terminal" / f"{iid}.receipt.json"
    passed_path = qwen_root / "passed" / f"{iid}.jsonl"
    evidence: dict[str, Any] = {
        "terminal_receipt": str(receipt_path),
        "result": str(result_path),
        "passed_fragment": str(passed_path),
    }
    if not receipt_path.exists() and not receipt_path.is_symlink():
        if result_path.exists() or result_path.is_symlink() or passed_path.exists() or passed_path.is_symlink():
            raise GokuFullMotionV16AuditError(
                f"partial Qwen output exists without terminal iid={iid}"
            )
        return "pending_qwen", evidence, None, None

    receipt = _validate_terminal_receipt(
        receipt_path,
        output_root=qwen_root,
        iid=iid,
        input_digest=object_sha256(row),
    )
    result = _strict_read_object(result_path)
    if set(result) != _QWEN_RECORD_KEYS:
        raise GokuFullMotionV16AuditError(
            f"Qwen result is not closed iid={iid}"
        )
    if (
        result.get("schema_version") != RECORD_SCHEMA
        or result.get("iid") != iid
        or result.get("input_digest") != object_sha256(row)
        or _canonical_bytes(result.get("input_row")) != _canonical_bytes(row)
        or result.get("status") != receipt["status"]
    ):
        raise GokuFullMotionV16AuditError(
            f"Qwen result identity/input binding differs iid={iid}"
        )
    record_digest = result.get("record_digest")
    if not isinstance(record_digest, str) or _SHA256_RE.fullmatch(record_digest) is None:
        raise GokuFullMotionV16AuditError(
            f"Qwen result digest is malformed iid={iid}"
        )
    bound = dict(result)
    bound["record_digest"] = None
    if record_digest != object_sha256(bound):
        raise GokuFullMotionV16AuditError(
            f"Qwen result digest differs iid={iid}"
        )
    evidence.update(
        {
            "receipt_digest": receipt["receipt_digest"],
            "record_digest": record_digest,
            "status": receipt["status"],
        }
    )
    if receipt["status"] == "error":
        error = result.get("error")
        if not isinstance(error, Mapping) or not error.get("type") or not error.get("message"):
            raise GokuFullMotionV16AuditError(
                f"Qwen error result lacks error evidence iid={iid}"
            )
        if passed_path.exists() or passed_path.is_symlink():
            raise GokuFullMotionV16AuditError(
                f"Qwen error unexpectedly has passed fragment iid={iid}"
            )
        evidence["error"] = dict(error)
        return "qwen_error", evidence, None, None
    if result.get("error") is not None:
        raise GokuFullMotionV16AuditError(
            f"successful Qwen result contains error iid={iid}"
        )
    census = validate_source_census(result["source_census"], expected_iid=iid)
    plan = validate_target_plan(
        result["target_plan"], expected_iid=iid, source_census=census
    )
    compiled = validate_compiled_instruction(
        result["compiled_instruction"],
        source_census=census,
        target_plan=plan,
    )
    passed = validate_passed_row(_strict_single_jsonl(passed_path))
    cross_bindings = (
        passed["qwen_record_digest"] == record_digest
        and _canonical_bytes(passed["source_census"]) == _canonical_bytes(census)
        and _canonical_bytes(passed["target_plan"]) == _canonical_bytes(plan)
        and _canonical_bytes(passed["compiled_instruction"])
        == _canonical_bytes(compiled)
        and passed["edit_instruction"] == compiled["instruction"]
    )
    if not cross_bindings:
        raise GokuFullMotionV16AuditError(
            f"Qwen passed/result cross-binding differs iid={iid}"
        )
    evidence["passed_fragment_sha256"] = _sha256_file(passed_path)
    evidence["edit_instruction_sha256"] = passed["edit_instruction_sha256"]
    return "ok", evidence, passed, passed_path


def _locate_wan_output_root(wan_root: Path, iid: str) -> Path:
    candidates = (
        wan_root / "samples" / iid,
        wan_root / iid,
        wan_root,
    )
    bound = [
        candidate
        for candidate in candidates
        if (candidate / wan_batch.RUN_CONTRACT_NAME).is_file()
        and not (candidate / wan_batch.RUN_CONTRACT_NAME).is_symlink()
    ]
    unique = list(dict.fromkeys(path.resolve() for path in bound))
    if len(unique) > 1:
        raise GokuFullMotionV16AuditError(
            f"multiple Wan output roots match iid={iid}: {unique}"
        )
    return unique[0] if unique else candidates[0]


def _default_wan_commit_validator(
    *,
    output_root: Path,
    sample_dir: Path,
    passed_path: Path,
    ffprobe: str,
) -> Mapping[str, Any]:
    manifest = wan_batch.load_non_production_preview_manifest(
        passed_path,
        allow_pending_review=False,
        max_samples=1,
    )
    prepared, _ = wan_batch._prepare_media_rows(
        manifest,
        data_root=None,
        ffprobe=ffprobe,
        expected_frame_num=FRAME_COUNT,
    )
    contract = wan_batch._load_json(
        output_root / wan_batch.RUN_CONTRACT_NAME,
        context="audited Wan run contract",
    )
    return wan_batch.validate_sample_commit(
        sample_dir,
        row=prepared[0],
        contract=contract,
        sample_index=0,
    )


def _latest_wan_error(wan_root: Path, iid: str) -> dict[str, Any] | None:
    status_root = wan_root / "status" / iid
    if not status_root.exists():
        return None
    if status_root.is_symlink() or not status_root.is_dir():
        raise GokuFullMotionV16AuditError(
            f"unsafe Wan status root iid={iid}"
        )
    status_paths = sorted(status_root.glob("attempt_*.json"))
    if not status_paths:
        return None
    status = _strict_read_object(status_paths[-1])
    if status.get("iid") != iid or status.get("status") not in {"success", "error"}:
        raise GokuFullMotionV16AuditError(
            f"Wan dispatch status differs iid={iid}"
        )
    return status if status["status"] == "error" else None


def _verify_lossless_conditioning(
    *, sample_dir: Path, result: Mapping[str, Any], passed: Mapping[str, Any]
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise GokuFullMotionV16AuditError("Wan result outputs are missing")
    anchor_copy = _safe_relative_file(
        sample_dir,
        outputs.get("conditioning_anchor_original"),
        context="Wan conditioning anchor",
    )
    if _sha256_file(anchor_copy) != passed["anchor_sha256"]:
        raise GokuFullMotionV16AuditError(
            "Wan conditioning anchor differs from Qwen anchor"
        )
    float32_path = _safe_relative_file(
        sample_dir,
        outputs.get("conditioning_frame0_float32"),
        context="Wan float32 frame zero",
    )
    png_path = _safe_relative_file(
        sample_dir,
        outputs.get("conditioning_frame0_png"),
        context="Wan lossless frame-zero PNG",
    )
    for path, digest_field in (
        (anchor_copy, "conditioning_anchor_original_sha256"),
        (float32_path, "conditioning_frame0_float32_sha256"),
        (png_path, "conditioning_frame0_png_sha256"),
    ):
        expected_digest = outputs.get(digest_field)
        if (
            not isinstance(expected_digest, str)
            or _SHA256_RE.fullmatch(expected_digest) is None
            or _sha256_file(path) != expected_digest
        ):
            raise GokuFullMotionV16AuditError(
                f"Wan output digest differs: {digest_field}"
            )
    conditioning = np.load(float32_path, allow_pickle=False)
    if conditioning.dtype != np.dtype("float32") or conditioning.ndim != 3 or conditioning.shape[0] != 3:
        raise GokuFullMotionV16AuditError(
            "conditioning frame zero must be float32 C,H,W"
        )
    expected_uint8 = (
        ((conditioning.astype(np.float32) + 1.0) * 127.5)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )
    with Image.open(png_path) as image:
        if image.format != "PNG":
            raise GokuFullMotionV16AuditError(
                "conditioning frame-zero artifact is not PNG"
            )
        actual_uint8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if actual_uint8.shape != expected_uint8.shape or not np.array_equal(
        actual_uint8, expected_uint8
    ):
        raise GokuFullMotionV16AuditError(
            "conditioning float32 tensor and lossless PNG differ"
        )
    pixel_sha = _sha256_bytes(actual_uint8.tobytes(order="C"))
    policy = result.get("first_frame_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("preencode_frame0_matches_png_pixels") is not True
        or policy.get("preencode_frame0_pixel_sha256") != pixel_sha
        or policy.get("lossless_png_pixel_sha256") != pixel_sha
    ):
        raise GokuFullMotionV16AuditError(
            "Wan first-frame lossless policy binding differs"
        )
    return {
        "conditioning_frame0_float32_sha256": _sha256_file(float32_path),
        "conditioning_frame0_png_sha256": _sha256_file(png_path),
        "conditioning_frame0_pixel_sha256": pixel_sha,
        "conditioning_shape": list(conditioning.shape),
    }


def _audit_wan(
    *,
    iid: str,
    passed: Mapping[str, Any],
    passed_path: Path,
    wan_root: Path,
    ffprobe: str,
    commit_validator: Callable[..., Mapping[str, Any]],
    video_probe: Callable[..., Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    output_root = _locate_wan_output_root(wan_root, iid)
    sample_dir = output_root / "samples" / iid
    result_path = sample_dir / wan_batch.SAMPLE_RESULT_NAME
    evidence: dict[str, Any] = {
        "output_root": str(output_root),
        "sample_dir": str(sample_dir),
        "result": str(result_path),
    }
    if not result_path.exists() and not result_path.is_symlink():
        if sample_dir.exists() or sample_dir.is_symlink():
            raise GokuFullMotionV16AuditError(
                f"partial Wan sample exists without result iid={iid}"
            )
        dispatch_error = _latest_wan_error(wan_root, iid)
        if dispatch_error is not None:
            evidence["dispatch_error"] = dispatch_error
            return "wan_error", evidence
        return "pending_wan", evidence

    result = dict(
        commit_validator(
            output_root=output_root,
            sample_dir=sample_dir,
            passed_path=passed_path,
            ffprobe=ffprobe,
        )
    )
    if result.get("iid") != iid:
        raise GokuFullMotionV16AuditError(f"Wan result IID differs iid={iid}")
    result_digest = result.get("result_digest")
    if not isinstance(result_digest, str) or _SHA256_RE.fullmatch(result_digest) is None:
        raise GokuFullMotionV16AuditError(
            f"Wan result digest is malformed iid={iid}"
        )
    prompt = result.get("prompt")
    if (
        not isinstance(prompt, Mapping)
        or prompt.get("field") != "edit_instruction"
        or prompt.get("text") != passed["edit_instruction"]
        or prompt.get("sha256") != passed["edit_instruction_sha256"]
    ):
        raise GokuFullMotionV16AuditError(
            f"Wan prompt differs from Qwen instruction iid={iid}"
        )
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise GokuFullMotionV16AuditError(f"Wan outputs are missing iid={iid}")
    source_copy = _safe_relative_file(
        sample_dir, outputs.get("source_video"), context="Wan source copy"
    )
    if _sha256_file(source_copy) != passed["source_video_sha256"]:
        raise GokuFullMotionV16AuditError(
            f"Wan source copy differs from Qwen source iid={iid}"
        )
    if outputs.get("source_video_sha256") != _sha256_file(source_copy):
        raise GokuFullMotionV16AuditError(
            f"Wan declared source-copy digest differs iid={iid}"
        )
    instruction_file = _safe_relative_file(
        sample_dir,
        outputs.get("edit_instruction_file"),
        context="Wan edit instruction",
    )
    expected_instruction = passed["edit_instruction"].encode("utf-8")
    if instruction_file.read_bytes() != expected_instruction:
        raise GokuFullMotionV16AuditError(
            f"Wan instruction file differs from Qwen iid={iid}"
        )
    if outputs.get("edit_instruction_file_sha256") != _sha256_file(
        instruction_file
    ):
        raise GokuFullMotionV16AuditError(
            f"Wan declared instruction digest differs iid={iid}"
        )
    preview = _safe_relative_file(
        sample_dir, outputs.get("preview_mp4"), context="Wan preview MP4"
    )
    if outputs.get("preview_mp4_sha256") != _sha256_file(preview):
        raise GokuFullMotionV16AuditError(
            f"Wan declared preview digest differs iid={iid}"
        )
    fresh_probe = dict(
        video_probe(
            preview,
            ffprobe=ffprobe,
            expected_frames=FRAME_COUNT,
            expected_fps=FRAME_RATE,
            max_nominal_duration_error_frames=0,
        )
    )
    frames = fresh_probe.get("frames")
    rate = fresh_probe.get("frame_rate")
    duration = fresh_probe.get("duration_seconds")
    if (
        frames != FRAME_COUNT
        or rate != FRAME_RATE
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isclose(
            float(duration), CONTAINER_DURATION_SECONDS, abs_tol=1e-6
        )
        or not math.isclose(
            (FRAME_COUNT - 1) / 25.0,
            TIMELINE_SPAN_SECONDS,
            abs_tol=1e-12,
        )
    ):
        raise GokuFullMotionV16AuditError(
            f"Wan temporal geometry differs iid={iid}"
        )
    conditioning = _verify_lossless_conditioning(
        sample_dir=sample_dir,
        result=result,
        passed=passed,
    )
    evidence.update(
        {
            "result_digest": result.get("result_digest"),
            "source_copy_sha256": _sha256_file(source_copy),
            "instruction_sha256": _sha256_bytes(expected_instruction),
            "preview_mp4_sha256": _sha256_file(preview),
            "fresh_ffprobe": fresh_probe,
            "container_duration_seconds": CONTAINER_DURATION_SECONDS,
            "timeline_span_seconds": TIMELINE_SPAN_SECONDS,
            **conditioning,
        }
    )
    return "complete", evidence


def audit_bundle(
    input_manifest: Path,
    qwen_root: Path,
    wan_root: Path,
    *,
    ffprobe: str = "ffprobe",
    commit_validator: Callable[..., Mapping[str, Any]] = _default_wan_commit_validator,
    video_probe: Callable[..., Mapping[str, Any]] = wan_batch.probe_video,
) -> dict[str, Any]:
    manifest = _plain_file(input_manifest, context="input manifest")
    qwen = _plain_directory(qwen_root, context="Qwen root")
    wan = _plain_directory(wan_root, context="Wan root")
    rows, manifest_sha = _read_input_manifest(manifest)
    audited_rows: list[dict[str, Any]] = []
    counts = {status: 0 for status in STATUSES}
    for input_row in rows:
        iid = str(input_row["iid"])
        row_summary: dict[str, Any] = {
            "schema_version": ROW_SCHEMA,
            "iid": iid,
            "status": None,
            "issues": [],
            "qwen": None,
            "wan": None,
        }
        try:
            qwen_status, qwen_evidence, passed, passed_path = _audit_qwen(
                input_row, qwen_root=qwen
            )
            row_summary["qwen"] = qwen_evidence
            if qwen_status != "ok":
                status = qwen_status
            else:
                assert passed is not None and passed_path is not None
                status, wan_evidence = _audit_wan(
                    iid=iid,
                    passed=passed,
                    passed_path=passed_path,
                    wan_root=wan,
                    ffprobe=ffprobe,
                    commit_validator=commit_validator,
                    video_probe=video_probe,
                )
                row_summary["wan"] = wan_evidence
        except Exception as error:
            stage = "qwen" if row_summary["qwen"] is None else "wan"
            status = "qwen_error" if stage == "qwen" else "wan_error"
            row_summary["issues"].append(
                {"stage": stage, "type": type(error).__name__, "message": str(error)}
            )
        row_summary["status"] = status
        counts[status] += 1
        audited_rows.append(row_summary)
    return {
        "schema_version": BUNDLE_SCHEMA,
        "input_manifest": str(manifest),
        "input_manifest_sha256": manifest_sha,
        "qwen_root": str(qwen),
        "wan_root": str(wan),
        "expected_iids": [str(row["iid"]) for row in rows],
        "expected_count": len(rows),
        "counts": counts,
        "all_complete": counts["complete"] == len(rows),
        "rows": audited_rows,
    }


def audit_bundles(
    bundles: Sequence[tuple[Path, Path, Path]],
    *,
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    if not bundles:
        raise GokuFullMotionV16AuditError("at least one audit bundle is required")
    results = [
        audit_bundle(input_path, qwen_root, wan_root, ffprobe=ffprobe)
        for input_path, qwen_root, wan_root in bundles
    ]
    counts = {status: 0 for status in STATUSES}
    for result in results:
        for status in STATUSES:
            counts[status] += int(result["counts"][status])
    expected_count = sum(int(result["expected_count"]) for result in results)
    return {
        "schema_version": AUDIT_SCHEMA,
        "bundle_count": len(results),
        "expected_count": expected_count,
        "counts": counts,
        "all_complete": counts["complete"] == expected_count,
        "bundles": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, action="append", required=True)
    parser.add_argument("--qwen-root", type=Path, action="append", required=True)
    parser.add_argument("--wan-root", type=Path, action="append", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (
        len(args.input_manifest) == len(args.qwen_root) == len(args.wan_root)
    ):
        raise GokuFullMotionV16AuditError(
            "--input-manifest/--qwen-root/--wan-root counts must match"
        )
    summary = audit_bundles(
        list(zip(args.input_manifest, args.qwen_root, args.wan_root, strict=True)),
        ffprobe=args.ffprobe,
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if summary["all_complete"] or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
