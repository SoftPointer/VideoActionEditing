#!/usr/bin/env python3
"""Run frozen-SAM2 action observation on an already-generated MEV840 bank.

This is a post-generation process.  Candidate MP4s must exist and be hashed
before the batch starts.  The generator is never called, and the real-target
action JSON is read only after each candidate has been reduced to the same
coordinate-free ABI.  Any extraction/correspondence failure leaves that
candidate unassigned and rejected rather than silently scoring it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import mev840_coordinate_free_action_oracle_v1 as oracle  # noqa: E402


SCHEMA = "mev840-generated-candidate-action-observer-batch-v1"
SUMMARY_SCHEMA = "mev840-generated-candidate-action-observer-summary-v1"
CANDIDATE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class CandidateActionObserverBatchError(RuntimeError):
    """The sealed post-generation candidate observer contract failed."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_exact(path_string: Any, expected: Any, label: str) -> Path:
    if not isinstance(path_string, str) or not isinstance(expected, str) or len(expected) != 64:
        raise CandidateActionObserverBatchError(f"{label} authority differs")
    path = Path(path_string).resolve(strict=True)
    if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
        raise CandidateActionObserverBatchError(f"{label} bytes differ")
    return path


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CandidateActionObserverBatchError("manifest must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CandidateActionObserverBatchError("cannot parse manifest") from error
    if not isinstance(value, dict):
        raise CandidateActionObserverBatchError("manifest must be one object")
    return value


def _validate_manifest(value: Mapping[str, Any]) -> None:
    target = value.get("target_action_oracle")
    reference = value.get("source_initial_reference")
    candidates = value.get("candidates")
    authority = value.get("authority")
    if (
        value.get("schema_version") != SCHEMA
        or value.get("case_id") != "MEV840"
        or not isinstance(value.get("output_root"), str)
        or not isinstance(target, dict)
        or not isinstance(reference, dict)
        or [reference.get(key) for key in ("frame_count", "fps", "width", "height")]
        != [81, 25.0, 656, 368]
        or not isinstance(candidates, list)
        or not candidates
        or authority
        != {
            "post_generation_only": True,
            "all_candidate_media_closed_before_observer_start": True,
            "generator_process_reads_manifest": False,
            "generator_process_reads_target_action": False,
            "generator_process_reads_real_target_media": False,
            "observer_calls_generator": False,
            "training_authorized": False,
            "optimizer_updates": 0,
            "failed_candidate_policy": "unassigned_reject",
            "appearance_quality_gate_external_required": True,
            "appearance_quality_gate_passed": None,
        }
    ):
        raise CandidateActionObserverBatchError("batch manifest semantics differ")
    derivation = reference.get("derivation")
    if (
        not isinstance(derivation, dict)
        or set(derivation)
        != {
            "original_path",
            "original_sha256",
            "original_frame_count",
            "original_fps",
            "original_width",
            "original_height",
            "target_width",
            "target_height",
            "algorithm",
            "ffmpeg_path",
            "ffmpeg_sha256",
        }
        or [
            derivation.get(key)
            for key in (
                "original_frame_count",
                "original_fps",
                "original_width",
                "original_height",
                "target_width",
                "target_height",
            )
        ]
        != [81, 25.0, 1280, 720, 656, 368]
        or derivation.get("algorithm")
        != "ffmpeg_scale_bicubic_libx264_preset_veryslow_crf1_yuv420p_r25"
        or derivation.get("ffmpeg_path")
        != "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_candidate_action_observer_v1_20260822_control/ffmpeg_4.4.2_authority"
        or derivation.get("ffmpeg_sha256")
        != "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
    ):
        raise CandidateActionObserverBatchError(
            "normalized source-reference derivation differs"
        )
    if target.get("representation_digest") is None:
        raise CandidateActionObserverBatchError("target representation pin is absent")
    identifiers = []
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or set(candidate) != {"candidate_id", "path", "sha256", "frame_count", "fps", "width", "height"}
            or not isinstance(candidate.get("candidate_id"), str)
            or CANDIDATE_RE.fullmatch(candidate["candidate_id"]) is None
            or [candidate.get(key) for key in ("frame_count", "fps", "width", "height")]
            != [81, 25.0, 656, 368]
        ):
            raise CandidateActionObserverBatchError("candidate ABI differs")
        identifiers.append(candidate["candidate_id"])
    if len(set(identifiers)) != len(identifiers):
        raise CandidateActionObserverBatchError("candidate identifiers are not unique")


def _scaled_candidate_spec(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    width = int(candidate["width"])
    height = int(candidate["height"])
    source_width = int(base["video"]["width"])
    source_height = int(base["video"]["height"])
    value = json.loads(json.dumps(base))
    value["video"] = {
        "path": candidate["path"],
        "sha256": candidate["sha256"],
        "role": "generated_candidate",
        "frame_count": 81,
        "fps": 25.0,
        "width": width,
        "height": height,
    }
    value["source_initial_reference"] = json.loads(json.dumps(reference))
    for role in value["roles"]:
        x1, y1, x2, y2 = role["box_xyxy"]
        role["box_xyxy"] = [
            float(x1) * width / source_width,
            float(y1) * height / source_height,
            float(x2) * width / source_width,
            float(y2) * height / source_height,
        ]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = _read_manifest(manifest_path)
    _validate_manifest(manifest)
    extractor = _regular_exact(
        manifest["extractor"]["path"], manifest["extractor"]["sha256"], "extractor"
    )
    oracle_program = _regular_exact(
        manifest["oracle_program"]["path"],
        manifest["oracle_program"]["sha256"],
        "oracle program",
    )
    # The imported module and explicitly pinned program must be identical.
    if Path(oracle.__file__).resolve() != oracle_program:
        raise CandidateActionObserverBatchError("imported oracle program authority differs")
    base_spec_path = _regular_exact(
        manifest["base_target_spec"]["path"],
        manifest["base_target_spec"]["sha256"],
        "base target spec",
    )
    target_path = _regular_exact(
        manifest["target_action_oracle"]["path"],
        manifest["target_action_oracle"]["sha256"],
        "target action oracle",
    )
    target = oracle.read_representation(target_path)
    if target["representation_digest"] != manifest["target_action_oracle"]["representation_digest"]:
        raise CandidateActionObserverBatchError("target action representation digest differs")
    reference = manifest["source_initial_reference"]
    _regular_exact(reference["path"], reference["sha256"], "source reference")
    _regular_exact(
        reference["derivation"]["original_path"],
        reference["derivation"]["original_sha256"],
        "original source reference",
    )
    _regular_exact(
        reference["derivation"]["ffmpeg_path"],
        reference["derivation"]["ffmpeg_sha256"],
        "source-reference ffmpeg authority",
    )
    base_spec = _read_manifest(base_spec_path)
    output = Path(manifest["output_root"]).absolute()
    if output.exists() or output.is_symlink():
        raise CandidateActionObserverBatchError("fresh batch output is required")
    for candidate in manifest["candidates"]:
        _regular_exact(candidate["path"], candidate["sha256"], candidate["candidate_id"])
    output.mkdir(mode=0o700, parents=True)
    rows = []
    for candidate in manifest["candidates"]:
        identifier = candidate["candidate_id"]
        candidate_root = output / identifier
        candidate_root.mkdir(mode=0o700)
        spec = _scaled_candidate_spec(base_spec, candidate, reference)
        spec_path = candidate_root / "observer.spec.json"
        spec_path.write_bytes(canonical_bytes(spec))
        observer_output = candidate_root / "observer"
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(extractor),
                "--spec",
                str(spec_path),
                "--output-dir",
                str(observer_output),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            reject = {
                "candidate_id": identifier,
                "status": "rejected_unassigned",
                "reason": "observer_or_frame0_correspondence_failed",
                "returncode": process.returncode,
                "stderr_tail": process.stderr[-4000:],
                "action_score_assigned": False,
                "selection_authorized": False,
            }
            (candidate_root / "reject.json").write_bytes(canonical_bytes(reject))
            rows.append(reject)
            continue
        action_path = observer_output / "action.json"
        candidate_action = oracle.read_representation(action_path)
        score = oracle.score_representations(target, candidate_action)
        score_path = candidate_root / "action.score.json"
        score_path.write_bytes(canonical_bytes(score))
        action_passed = bool(score["decision"]["action_gate_passed"])
        rows.append(
            {
                "candidate_id": identifier,
                "status": "action_scored_quality_gate_pending" if action_passed else "rejected_unassigned",
                "reason": "external_appearance_quality_gate_required" if action_passed else "action_gate_failed",
                "candidate_action_path": str(action_path),
                "candidate_action_sha256": file_sha256(action_path),
                "candidate_representation_digest": candidate_action["representation_digest"],
                "action_score_path": str(score_path),
                "action_score_sha256": file_sha256(score_path),
                "action_score": score["scores"]["action"],
                "action_gate_passed": action_passed,
                "appearance_quality_gate_external_required": True,
                "appearance_quality_gate_passed": None,
                "selection_authorized": False,
            }
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "case_id": "MEV840",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "target_representation_digest": target["representation_digest"],
        "candidate_count": len(rows),
        "candidates": rows,
        "generator_forward_calls": 0,
        "optimizer_updates": 0,
        "training_performed": False,
        "failed_candidate_policy": "unassigned_reject",
        "appearance_quality_gate_external_required": True,
        "appearance_quality_gate_passed": None,
        "selection_authorized": False,
    }
    summary["summary_sha256"] = oracle.object_sha256(summary)
    (output / "summary.json").write_bytes(canonical_bytes(summary))
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
