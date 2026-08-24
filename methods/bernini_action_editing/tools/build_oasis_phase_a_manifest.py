#!/usr/bin/env python3
"""Build one hash-bound OASIS Phase-A source-only frozen-oracle manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oasis_phase_a_manifest as manifest  # noqa: E402


class OASISManifestBuildError(RuntimeError):
    """The authoring draft cannot produce the closed Phase-A manifest."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise OASISManifestBuildError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_file() or path.is_symlink():
        raise OASISManifestBuildError(f"{label} must be an existing plain file")
    return path.resolve(strict=True)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OASISManifestBuildError(f"{label} must be a mapping")
    return value


def _load_draft(path_value: str) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label="draft")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OASISManifestBuildError("draft must be ASCII JSON") from error
    return _mapping(value, label="draft")


def _weak_decoy(value: Any) -> dict[str, Any]:
    row = dict(_mapping(value, label="weak_wrongref_diagnostic"))
    available = row.get("available")
    if type(available) is not bool:
        raise OASISManifestBuildError("weak decoy available must be bool")
    if available:
        path = _plain_absolute_file(row.get("path"), label="weak decoy")
        path_value: Optional[str] = str(path)
        sha: Optional[str] = manifest.file_sha256(path)
    else:
        path_value = None
        sha = None
    return {
        "schema_version": manifest.WEAK_DECOY_SCHEMA,
        "available": available,
        "path": path_value,
        "sha256": sha,
        "proxy_kind": row.get("proxy_kind", "none"),
        "known_confounds": row.get("known_confounds", []),
        "identity_only_claim": False,
        "used_for_authorization": False,
    }


def build_manifest(draft: Mapping[str, Any]) -> dict[str, Any]:
    expected_root = {
        "checkpoint_tree_sha256",
        "t2v_scalar_calibration",
        "seed_order",
        "samples",
    }
    if set(draft) != expected_root:
        raise OASISManifestBuildError("draft root field closure differs")
    calibration = _mapping(
        draft["t2v_scalar_calibration"], label="T2V scalar calibration"
    )
    status = calibration.get("status")
    if status == "unresolved":
        if set(calibration) != {"status"}:
            raise OASISManifestBuildError(
                "unresolved scalar calibration draft must contain only status"
            )
        scalar_binding = {
            "schema_version": manifest.SCALAR_CALIBRATION_BINDING_SCHEMA,
            "status": "unresolved",
            "path": None,
            "file_sha256": None,
            "evidence_digest": None,
        }
    elif status == "resolved":
        if set(calibration) != {"status", "path"}:
            raise OASISManifestBuildError(
                "resolved scalar calibration draft requires status/path only"
            )
        evidence_path = _plain_absolute_file(
            calibration["path"], label="dedicated scalar-calibration evidence"
        )
        try:
            evidence_value = json.loads(evidence_path.read_text(encoding="ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise OASISManifestBuildError(
                "dedicated scalar-calibration evidence must be ASCII JSON"
            ) from error
        if not isinstance(evidence_value, Mapping):
            raise OASISManifestBuildError("scalar-calibration evidence root differs")
        evidence_digest = evidence_value.get("evidence_digest")
        scalar_binding = {
            "schema_version": manifest.SCALAR_CALIBRATION_BINDING_SCHEMA,
            "status": "resolved",
            "path": str(evidence_path),
            "file_sha256": manifest.file_sha256(evidence_path),
            "evidence_digest": evidence_digest,
        }
    else:
        raise OASISManifestBuildError(
            "scalar calibration status must be unresolved or resolved"
        )
    rows = draft["samples"]
    if not isinstance(rows, list) or len(rows) != 4:
        raise OASISManifestBuildError("draft requires exactly four samples")
    sealed_samples: list[dict[str, Any]] = []
    sample_fields = {
        "sample_id",
        "family",
        "analysis_split",
        "source_video_path",
        "source_caption",
        "complete_action_caption",
        "actor_binding",
        "raw_caption_by_branch",
        "calibration_candidate_id",
        "calibration_event_receipt_digest",
        "weak_wrongref_diagnostic",
    }
    for ordinal, raw in enumerate(rows):
        row = _mapping(raw, label=f"sample[{ordinal}]")
        if set(row) != sample_fields:
            raise OASISManifestBuildError(f"sample[{ordinal}] field closure differs")
        source = _plain_absolute_file(
            row["source_video_path"], label=f"sample[{ordinal}] source video"
        )
        captions = _mapping(
            row["raw_caption_by_branch"], label=f"sample[{ordinal}] caption bank"
        )
        sample = {
            "schema_version": manifest.SAMPLE_SCHEMA,
            "sample_id": row["sample_id"],
            "family": row["family"],
            "analysis_split": row["analysis_split"],
            "source_video_path": str(source),
            "source_video_sha256": manifest.file_sha256(source),
            "source_caption": row["source_caption"],
            "complete_action_caption": row["complete_action_caption"],
            "actor_binding": row["actor_binding"],
            "raw_caption_by_branch": dict(captions),
            "raw_caption_bank_sha256": manifest.object_sha256(captions),
            "calibration_candidate_id": row["calibration_candidate_id"],
            "calibration_event_receipt_digest": row[
                "calibration_event_receipt_digest"
            ],
            "weak_wrongref_diagnostic": _weak_decoy(
                row["weak_wrongref_diagnostic"]
            ),
        }
        sealed_samples.append(manifest.seal_sample_draft(sample))
    root = {
        "schema_version": manifest.SCHEMA_VERSION,
        "checkpoint_tree_sha256": draft["checkpoint_tree_sha256"],
        "t2v_scalar_calibration": scalar_binding,
        "frame_count": manifest.FRAME_COUNT,
        "fps": manifest.FPS,
        "sampler_steps": manifest.SAMPLER_STEPS,
        "reference_indices": list(manifest.REFERENCE_INDICES),
        "seed_order": draft["seed_order"],
        "arm_order": list(manifest.ARM_ORDER),
        "topology": manifest.TOPOLOGY,
        "sample_count": len(sealed_samples),
        "samples": sealed_samples,
        "information_flow": manifest.static_contract()["information_flow"],
    }
    return manifest.seal_manifest_draft(root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    draft = _load_draft(args.draft)
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise OASISManifestBuildError("output must be a fresh absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    value = build_manifest(draft)
    payload = manifest.canonical_json_bytes(value) + b"\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise OASISManifestBuildError("temporary output path is not fresh")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    file_digest = manifest.file_sha256(output)
    checked = manifest.load_phase_a_manifest(output, file_digest, verify_files=True)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "file_sha256": file_digest,
                "manifest_digest": checked.manifest_digest,
                "sample_count": len(checked.samples),
                "planned_rollout_count": manifest.static_contract()["rollout_count"],
                "training_performed": False,
                "scientific_action_editing_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
