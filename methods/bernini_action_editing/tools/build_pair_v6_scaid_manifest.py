#!/usr/bin/env python3
"""Deterministically build the strict PAIR-v6 SCAID DP2 manifest.

The tool does not trust caller-supplied prompts, action-family labels, or
eligibility booleans.  It recomputes authoritative v3 evidence for each fit
candidate, reloads raw captions from the evidence-bound source-bank spec, and
requires official T2V prompt reconstruction to equal the evidence-bound
guidance event exactly.  The caller supplies only source identities and their
exact media bindings.  The correct source must equal the fit candidate's
geometry anchor; a wrong source additionally requires a hash-bound external
identity/class/initial-pose audit artifact.  Output is strict canonical ASCII
and create-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v6_scaid_source_coordinate as scaid  # noqa: E402
import infer_native_identity_generation_canary as native_infer  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as bank_spec  # noqa: E402
import train_pair_v5_native_flow_dpo_v2 as native_dpo  # noqa: E402
import train_pair_v5_t2v_guidance_distill as cagd_trainer  # noqa: E402
import train_pair_v6_scaid as trainer  # noqa: E402


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class PairV6SCAIDManifestBuildError(RuntimeError):
    pass


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PairV6SCAIDManifestBuildError(f"{label} must be lowercase SHA-256")
    return value


def _safe(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PairV6SCAIDManifestBuildError(f"{label} must be a safe identifier")
    return value


def _plain_file(value: str | Path, *, expected_sha256: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV6SCAIDManifestBuildError(f"{label} must be an absolute plain file")
    resolved = path.resolve(strict=True)
    if resolved != path or trainer.file_sha256(path) != _sha(expected_sha256, label=f"{label} SHA"):
        raise PairV6SCAIDManifestBuildError(f"{label} file SHA-256/canonical path differs")
    return path


@dataclass(frozen=True)
class EventInput:
    sample_id: str
    fit_candidate_id: str
    source_video_path: Path
    source_video_sha256: str
    wrong_source_video_path: Path
    wrong_source_video_sha256: str
    wrong_source_iid: str
    wrong_source_audit_path: Path
    wrong_source_audit_file_sha256: str
    wrong_source_audit_digest: str


def _build_t2v_task_prompt(raw_caption: str) -> str:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    return native_infer.build_task_prompt(
        "t2v", raw_caption, prompt_cleaner=prompt_clean
    )


def _load_authoritative_events_and_raw_captions(
    evidence_path: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Mapping[str, str]],
    Mapping[str, Mapping[str, Any]],
]:
    try:
        evidence = json.loads(evidence_path.read_text(encoding="ascii"))
        guidance_binding = evidence["guidance_manifest"]
        manifest = cagd_trainer.load_manifest(
            guidance_binding["path"], guidance_binding["file_sha256"]
        )
        if manifest.raw_sha256 != guidance_binding["file_sha256"]:
            raise PairV6SCAIDManifestBuildError(
                "guidance manifest hash binding differs"
            )
        spec_binding = evidence["source_bank_spec"]
        spec_path = _plain_file(
            spec_binding["path"],
            expected_sha256=spec_binding["file_sha256"],
            label="source bank spec",
        )
        spec, observed_spec_sha = bank_spec.load_sealed_spec(
            spec_path, spec_binding["file_sha256"]
        )
        if observed_spec_sha != spec_binding["file_sha256"]:
            raise PairV6SCAIDManifestBuildError("source bank spec hash binding differs")
    except Exception as error:
        raise PairV6SCAIDManifestBuildError(
            f"cannot reload authoritative guidance/spec closure: {error}"
        ) from error
    result = {event.event_id: event for event in manifest.events}
    if len(result) != len(manifest.events):
        raise PairV6SCAIDManifestBuildError("authoritative fit candidate IDs repeat")
    candidates = [
        candidate
        for group in spec["groups"]
        for candidate in group["candidates"]
    ]
    raw_by_event: dict[str, Mapping[str, str]] = {}
    anchor_by_event: dict[str, Mapping[str, Any]] = {}
    for event_id, event in result.items():
        anchors = [row for row in candidates if row["candidate_id"] == event_id]
        if (
            len(anchors) != 1
            or anchors[0]["analysis_split"] != "fit"
            or anchors[0]["semantic_branch"] != scaid.mace.ACTION_BRANCH
            or anchors[0]["action_family_id"] != event.action_family
        ):
            raise PairV6SCAIDManifestBuildError(
                f"guidance event {event_id!r} has no unique fit/action spec anchor"
            )
        anchor = anchors[0]
        cell = [
            row
            for row in candidates
            if (
                row["analysis_split"],
                row["action_family_id"],
                row["calibration_group_id"],
            )
            == (
                anchor["analysis_split"],
                anchor["action_family_id"],
                anchor["calibration_group_id"],
            )
        ]
        if [row["semantic_branch"] for row in cell] != list(scaid.BRANCH_ORDER):
            raise PairV6SCAIDManifestBuildError(
                f"guidance event {event_id!r} spec cell branch mapping differs"
            )
        raw = {
            row["semantic_branch"]: row["full_t2v_caption"]
            for row in cell
        }
        rebuilt = {
            branch: _build_t2v_task_prompt(raw[branch])
            for branch in scaid.BRANCH_ORDER
        }
        if (
            rebuilt != dict(event.prompt_by_branch)
            or scaid.object_sha256(rebuilt) != event.prompt_bank_sha256
        ):
            raise PairV6SCAIDManifestBuildError(
                f"guidance event {event_id!r} raw-to-T2V prompt mapping differs"
            )
        raw_by_event[event_id] = raw
        anchor_by_event[event_id] = {
            "candidate_id": anchor["candidate_id"],
            "geometry_source_video": anchor["geometry_source_video"],
            "geometry_source_video_sha256": anchor[
                "geometry_source_video_sha256"
            ],
        }
    return result, raw_by_event, anchor_by_event


def build_manifest(
    *,
    evidence_path: str | Path,
    expected_evidence_sha256: str,
    checkpoint_tree_sha256: str,
    events: Sequence[EventInput],
    output_path: str | Path,
) -> Mapping[str, Any]:
    """Validate all closure inputs and create one fresh canonical manifest."""

    evidence = _plain_file(
        evidence_path,
        expected_sha256=expected_evidence_sha256,
        label="authoritative v3 evidence",
    )
    checkpoint_sha = _sha(checkpoint_tree_sha256, label="checkpoint tree SHA")
    if checkpoint_sha != trainer.legacy.CHECKPOINT_TREE_SHA256:
        raise PairV6SCAIDManifestBuildError(
            "checkpoint tree is not the pinned complete Bernini-R 1.3B tree"
        )
    if len(events) != trainer.DP_SIZE:
        raise PairV6SCAIDManifestBuildError("exactly two event inputs are required")
    (
        authoritative,
        raw_by_candidate,
        source_anchor_by_candidate,
    ) = _load_authoritative_events_and_raw_captions(evidence)
    rows: list[dict[str, Any]] = []
    media_hashes: set[str] = set()
    families: set[str] = set()
    candidates: set[str] = set()
    for ordinal, item in enumerate(events):
        if not isinstance(item, EventInput):
            raise PairV6SCAIDManifestBuildError(f"event[{ordinal}] input type differs")
        sample_id = _safe(item.sample_id, label=f"event[{ordinal}] sample_id")
        candidate_id = _safe(
            item.fit_candidate_id, label=f"event[{ordinal}] fit_candidate_id"
        )
        source_anchor = source_anchor_by_candidate.get(candidate_id)
        if source_anchor is None or sample_id != source_anchor["candidate_id"]:
            raise PairV6SCAIDManifestBuildError(
                f"event[{ordinal}] sample ID is not the evidence-bound fit geometry anchor"
            )
        wrong_source_iid = _safe(
            item.wrong_source_iid, label=f"event[{ordinal}] wrong_source_iid"
        )
        source = _plain_file(
            item.source_video_path,
            expected_sha256=item.source_video_sha256,
            label=f"event[{ordinal}] source video",
        )
        wrong = _plain_file(
            item.wrong_source_video_path,
            expected_sha256=item.wrong_source_video_sha256,
            label=f"event[{ordinal}] wrong-source video",
        )
        source_sha = _sha(item.source_video_sha256, label="source video SHA")
        wrong_sha = _sha(item.wrong_source_video_sha256, label="wrong-source video SHA")
        authoritative_source = _plain_file(
            source_anchor["geometry_source_video"],
            expected_sha256=source_anchor["geometry_source_video_sha256"],
            label=f"event[{ordinal}] evidence-bound geometry source",
        )
        if (
            source != authoritative_source
            or source_sha != source_anchor["geometry_source_video_sha256"]
        ):
            raise PairV6SCAIDManifestBuildError(
                f"event[{ordinal}] native correct source is not the evidence-bound geometry anchor"
            )
        try:
            wrong_source_audit = trainer.load_wrong_source_audit(
                item.wrong_source_audit_path,
                item.wrong_source_audit_file_sha256,
                expected_audit_digest=item.wrong_source_audit_digest,
                candidate_sample_id=sample_id,
                candidate_source_video_sha256=source_sha,
                wrong_source_iid=wrong_source_iid,
                wrong_source_video_sha256=wrong_sha,
            )
        except trainer.PairV6SCAIDTrainingError as error:
            raise PairV6SCAIDManifestBuildError(str(error)) from error
        if source == wrong or source_sha == wrong_sha:
            raise PairV6SCAIDManifestBuildError(
                f"event[{ordinal}] correct/wrong source is not content-distinct"
            )
        if source_sha in media_hashes or wrong_sha in media_hashes:
            raise PairV6SCAIDManifestBuildError("source media is reused across DP events")
        media_hashes.update((source_sha, wrong_sha))
        source_geometry = native_dpo._ffprobe_exact81(source)
        wrong_geometry = native_dpo._ffprobe_exact81(wrong)
        geometry_fields = ("width", "height", "avg_frame_rate")
        if any(
            source_geometry.get(field) != wrong_geometry.get(field)
            for field in geometry_fields
        ):
            raise PairV6SCAIDManifestBuildError(
                f"event[{ordinal}] correct/wrong width-height-fps geometry differs"
            )

        # This is the only minting boundary.  It recomputes bank generation,
        # raw score rows, external event audits, fit and held-out confirmation.
        gate = scaid.load_authoritative_v3_authorization(
            evidence,
            expected_evidence_sha256=expected_evidence_sha256,
            checkpoint_tree_sha256=checkpoint_sha,
            fit_candidate_id=candidate_id,
        )
        event = authoritative.get(candidate_id)
        raw_captions = raw_by_candidate.get(candidate_id)
        if (
            event is None
            or raw_captions is None
            or event.analysis_split != "fit"
            or event.action_family != gate.action_family
            or event.prompt_bank_sha256 != gate.prompt_bank_sha256
            or scaid.object_sha256(scaid._prompts(event.prompt_by_branch))
            != gate.prompt_bank_sha256
        ):
            raise PairV6SCAIDManifestBuildError(
                f"event[{ordinal}] differs from authoritative fit evidence"
            )
        if candidate_id in candidates or gate.action_family in families:
            raise PairV6SCAIDManifestBuildError(
                "DP events must use different candidates and action families"
            )
        candidates.add(candidate_id)
        families.add(gate.action_family)
        row = {
            "schema_version": trainer.EVENT_SCHEMA,
            "sample_id": sample_id,
            "fit_candidate_id": candidate_id,
            "action_family": gate.action_family,
            "source_video_path": str(source),
            "source_video_sha256": source_sha,
            "wrong_source_video_path": str(wrong),
            "wrong_source_video_sha256": wrong_sha,
            "wrong_source_iid": wrong_source_iid,
            "wrong_source_audit_path": wrong_source_audit["path"],
            "wrong_source_audit_file_sha256": wrong_source_audit["file_sha256"],
            "wrong_source_audit_digest": wrong_source_audit["audit_digest"],
            "frame_count": trainer.FRAME_COUNT,
            "fps": trainer.FPS,
            "reference_indices": list(trainer.REFERENCE_INDICES),
            "raw_caption_by_branch": dict(raw_captions),
            "raw_caption_bank_sha256": scaid.object_sha256(raw_captions),
        }
        rows.append({**row, "event_digest": trainer.object_sha256(row)})
    root = {
        "schema_version": trainer.MANIFEST_SCHEMA,
        "checkpoint_tree_sha256": checkpoint_sha,
        "event_count": len(rows),
        "events": rows,
    }
    result = {**root, "manifest_digest": trainer.object_sha256(root)}
    raw = trainer.canonical_json_bytes(result) + b"\n"
    output = Path(output_path)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise PairV6SCAIDManifestBuildError("output must be a fresh absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(output, 0o400)
        # Reuse the training parser as the final schema/hash/exact81 oracle.
        trainer.load_manifest(output, hashlib.sha256(raw).hexdigest())
        directory_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # A failed fresh construction has no valid deliverable.  Only the file
        # created by this invocation can be removed here.
        output.chmod(0o600) if output.exists() else None
        output.unlink(missing_ok=True)
        raise
    return {
        "path": str(output),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_digest": result["manifest_digest"],
        "fit_candidate_ids": [row["fit_candidate_id"] for row in rows],
        "action_families": [row["action_family"] for row in rows],
        "deterministic_canonical_ascii": True,
        "fresh_create_only": True,
        "authoritative_v3_recomputed": True,
        "wrong_source_identity_class_pose_and_preprocessing_external_audit_required": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--expected-evidence-sha256", required=True)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--fit-candidate-id", action="append", required=True)
    parser.add_argument("--source-video", action="append", required=True)
    parser.add_argument("--source-video-sha256", action="append", required=True)
    parser.add_argument("--wrong-source-video", action="append", required=True)
    parser.add_argument("--wrong-source-video-sha256", action="append", required=True)
    parser.add_argument("--wrong-source-iid", action="append", required=True)
    parser.add_argument("--wrong-source-audit", action="append", required=True)
    parser.add_argument("--wrong-source-audit-file-sha256", action="append", required=True)
    parser.add_argument("--wrong-source-audit-digest", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    fields = (
        args.sample_id,
        args.fit_candidate_id,
        args.source_video,
        args.source_video_sha256,
        args.wrong_source_video,
        args.wrong_source_video_sha256,
        args.wrong_source_iid,
        args.wrong_source_audit,
        args.wrong_source_audit_file_sha256,
        args.wrong_source_audit_digest,
    )
    if any(len(values) != trainer.DP_SIZE for values in fields):
        raise PairV6SCAIDManifestBuildError(
            "every repeated event argument must occur exactly twice"
        )
    events = tuple(
        EventInput(
            args.sample_id[index],
            args.fit_candidate_id[index],
            Path(args.source_video[index]),
            args.source_video_sha256[index],
            Path(args.wrong_source_video[index]),
            args.wrong_source_video_sha256[index],
            args.wrong_source_iid[index],
            Path(args.wrong_source_audit[index]),
            args.wrong_source_audit_file_sha256[index],
            args.wrong_source_audit_digest[index],
        )
        for index in range(trainer.DP_SIZE)
    )
    receipt = build_manifest(
        evidence_path=args.evidence,
        expected_evidence_sha256=args.expected_evidence_sha256,
        checkpoint_tree_sha256=args.checkpoint_tree_sha256,
        events=events,
        output_path=args.output,
    )
    print(trainer.canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EventInput", "PairV6SCAIDManifestBuildError", "build_manifest"]
