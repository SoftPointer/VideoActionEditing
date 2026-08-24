#!/usr/bin/env python3
"""Train PAIR-v6 SCAID Action-LoRA directly on native RV2V-4.

World size is fixed to DP2 x Ulysses-SP4.  Each DP arm owns one distinct,
authoritatively validated fit action family.  All v3 evidence is recomputed
before model or optimizer construction.  Indices 38/39 invoke the SCAID
zero-update path and cannot call an optimizer.

No generated T2V RGB/latent, paired target, mask, flow, pose, track, or
trajectory is accepted by the manifest or runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dclr_runtime_contract as t2v_runtime  # noqa: E402
import infer_native_identity_generation_canary as native_infer  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import pair_v5_action_adapter as action_adapter  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v6_scaid_source_coordinate as scaid  # noqa: E402
import source_self_native_ref_contrastive_v3 as native  # noqa: E402
import source_self_runtime as distributed_runtime  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_pair_v5_action_preference as native_runtime  # noqa: E402
import train_pair_v5_native_flow_dpo_v2 as native_dpo  # noqa: E402
import train_pair_v5_t2v_guidance_distill as cagd_runtime  # noqa: E402


METHOD_NAME = "bernini-pair-v6-scaid-source-coordinate-v1"
MANIFEST_SCHEMA = "bernini-pair-v6-scaid-native-fit-manifest-v1"
EVENT_SCHEMA = "bernini-pair-v6-scaid-native-fit-event-v1"
RUN_RECEIPT_SCHEMA = "bernini-pair-v6-scaid-run-receipt-v1"
WRONG_SOURCE_AUDIT_SCHEMA = "bernini-pair-v6-scaid-wrong-source-audit-v1"
WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
FRAME_COUNT = 81
FPS = 25.0
REFERENCE_INDICES = (0, 27, 53, 80)
EXACT40_STEPS = 40
DEFAULT_SEED = 20260808
DEFAULT_LR = 1.0e-6
DEFAULT_MAX_GRAD_NORM = 1.0
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_ROOT_FIELDS = frozenset(
    {"schema_version", "checkpoint_tree_sha256", "event_count", "events", "manifest_digest"}
)
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "fit_candidate_id",
        "action_family",
        "source_video_path",
        "source_video_sha256",
        "wrong_source_video_path",
        "wrong_source_video_sha256",
        "wrong_source_iid",
        "wrong_source_audit_path",
        "wrong_source_audit_file_sha256",
        "wrong_source_audit_digest",
        "frame_count",
        "fps",
        "reference_indices",
        "raw_caption_by_branch",
        "raw_caption_bank_sha256",
        "event_digest",
    }
)
_WRONG_SOURCE_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_sample_id",
        "candidate_source_video_sha256",
        "wrong_source_iid",
        "wrong_source_video_sha256",
        "criteria",
        "reviewer",
        "review_artifact_path",
        "review_artifact_sha256",
        "preprocessing_contract",
        "audit_method_source_revision",
        "audit_method_source_archive_path",
        "audit_method_source_archive_sha256",
        "audit_digest",
    }
)
WRONG_SOURCE_AUDIT_CRITERIA = (
    "distinct_identity",
    "same_actor_class",
    "same_actor_count",
    "same_initial_pose_class",
    "same_spatial_bucket",
    "same_camera_class",
    "same_composition_class",
    "same_length",
    "manual_reviewed",
)
WRONG_SOURCE_PREPROCESSING_CONTRACT = {
    "decoder": "infer_source_kv_carrier_oracle.prepare_hashed_source_snapshot",
    "correct_and_wrong_use_identical_decoder": True,
    "raw_video_files_sha256_bound": True,
    "frame_count": FRAME_COUNT,
    "fps": FPS,
    "reference_indices": list(REFERENCE_INDICES),
}


class PairV6SCAIDTrainingError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return scaid.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return scaid.object_sha256(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PairV6SCAIDTrainingError(f"{label} must be lowercase SHA-{length}")
    return value


def _plain_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise PairV6SCAIDTrainingError(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV6SCAIDTrainingError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _strict_json(path: Path, expected_sha256: str, *, label: str) -> Mapping[str, Any]:
    if file_sha256(path) != _sha(expected_sha256, length=64, label=f"{label} SHA"):
        raise PairV6SCAIDTrainingError(f"{label} file SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV6SCAIDTrainingError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(value, Mapping):
        raise PairV6SCAIDTrainingError(f"{label} root must be an object")
    return value


@dataclass(frozen=True)
class EventSpec:
    sample_id: str
    fit_candidate_id: str
    action_family: str
    source_video: Path
    source_video_sha256: str
    wrong_source_video: Path
    wrong_source_video_sha256: str
    wrong_source_iid: str
    wrong_source_audit: Mapping[str, Any]
    raw_caption_by_branch: Mapping[str, str]
    raw_caption_bank_sha256: str
    event_digest: str


@dataclass(frozen=True)
class TrainingManifest:
    path: Path
    file_sha256: str
    checkpoint_tree_sha256: str
    events: tuple[EventSpec, EventSpec]
    manifest_digest: str

    def assert_unchanged(self) -> None:
        if file_sha256(self.path) != self.file_sha256:
            raise PairV6SCAIDTrainingError("SCAID manifest changed during training")
        for event in self.events:
            if (
                file_sha256(event.source_video) != event.source_video_sha256
                or file_sha256(event.wrong_source_video) != event.wrong_source_video_sha256
                or file_sha256(Path(event.wrong_source_audit["path"]))
                != event.wrong_source_audit["file_sha256"]
                or file_sha256(Path(event.wrong_source_audit["review_artifact_path"]))
                != event.wrong_source_audit["review_artifact_sha256"]
                or file_sha256(
                    Path(event.wrong_source_audit["audit_method_source_archive_path"])
                )
                != event.wrong_source_audit["audit_method_source_archive_sha256"]
            ):
                raise PairV6SCAIDTrainingError("source media changed during training")


def load_wrong_source_audit(
    path_value: str | Path,
    expected_sha256: str,
    *,
    expected_audit_digest: str,
    candidate_sample_id: str,
    candidate_source_video_sha256: str,
    wrong_source_iid: str,
    wrong_source_video_sha256: str,
) -> Mapping[str, Any]:
    path = _plain_file(str(path_value), label="wrong-source audit")
    value = _strict_json(path, expected_sha256, label="wrong-source audit")
    if set(value) != set(_WRONG_SOURCE_AUDIT_FIELDS):
        raise PairV6SCAIDTrainingError("wrong-source audit field closure differs")
    unsigned = dict(value)
    declared = _sha(
        unsigned.pop("audit_digest"), length=64, label="wrong-source audit digest"
    )
    if (
        declared != _sha(
            expected_audit_digest,
            length=64,
            label="expected wrong-source audit digest",
        )
        or object_sha256(unsigned) != declared
        or value["schema_version"] != WRONG_SOURCE_AUDIT_SCHEMA
        or value["candidate_sample_id"] != candidate_sample_id
        or value["candidate_source_video_sha256"]
        != candidate_source_video_sha256
        or value["wrong_source_iid"] != wrong_source_iid
        or value["wrong_source_video_sha256"] != wrong_source_video_sha256
        or candidate_sample_id == wrong_source_iid
        or candidate_source_video_sha256 == wrong_source_video_sha256
    ):
        raise PairV6SCAIDTrainingError("wrong-source audit identity binding differs")
    if (
        not isinstance(wrong_source_iid, str)
        or _SAFE_ID.fullmatch(wrong_source_iid) is None
    ):
        raise PairV6SCAIDTrainingError("wrong-source IID differs")
    criteria = value["criteria"]
    if (
        not isinstance(criteria, Mapping)
        or set(criteria) != set(WRONG_SOURCE_AUDIT_CRITERIA)
        or any(criteria[name] is not True for name in WRONG_SOURCE_AUDIT_CRITERIA)
    ):
        raise PairV6SCAIDTrainingError(
            "wrong-source identity/class/initial-pose audit is not fully accepted"
        )
    if value["preprocessing_contract"] != WRONG_SOURCE_PREPROCESSING_CONTRACT:
        raise PairV6SCAIDTrainingError("wrong-source preprocessing provenance differs")
    reviewer = value["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip() or "\x00" in reviewer:
        raise PairV6SCAIDTrainingError("wrong-source audit reviewer differs")
    _sha(
        value["audit_method_source_revision"],
        length=40,
        label="wrong-source audit method revision",
    )
    audit_archive_sha = _sha(
        value["audit_method_source_archive_sha256"],
        length=64,
        label="wrong-source audit source archive",
    )
    audit_archive = _plain_file(
        value["audit_method_source_archive_path"],
        label="wrong-source audit method source archive",
    )
    if file_sha256(audit_archive) != audit_archive_sha:
        raise PairV6SCAIDTrainingError(
            "wrong-source audit method source archive hash differs"
        )
    review_artifact = _plain_file(
        value["review_artifact_path"], label="wrong-source review artifact"
    )
    review_sha = _sha(
        value["review_artifact_sha256"],
        length=64,
        label="wrong-source review artifact SHA-256",
    )
    if file_sha256(review_artifact) != review_sha:
        raise PairV6SCAIDTrainingError("wrong-source review artifact hash differs")
    return {
        **dict(value),
        "path": str(path),
        "file_sha256": _sha(
            expected_sha256, length=64, label="wrong-source audit file SHA-256"
        ),
        "review_artifact_path": str(review_artifact),
        "audit_method_source_archive_path": str(audit_archive),
    }


@dataclass(frozen=True)
class RuntimeEvent:
    spec: EventSpec
    authorization: scaid.SCAIDAuthorization
    source_latent: Any
    source_references: tuple[Any, ...]
    wrong_source_latent: Any
    wrong_source_references: tuple[Any, ...]
    t2v_conditions: Mapping[str, Any]
    native_conditions: Mapping[str, Any]
    unconditional: Any
    prompt_construction_receipt: Mapping[str, Any]


def load_manifest(path_value: str | Path, expected_sha256: str) -> TrainingManifest:
    path = _plain_file(str(path_value), label="SCAID manifest")
    root = _strict_json(path, expected_sha256, label="SCAID manifest")
    if set(root) != set(_ROOT_FIELDS) or root.get("schema_version") != MANIFEST_SCHEMA:
        raise PairV6SCAIDTrainingError("SCAID manifest field closure differs")
    unsigned = dict(root)
    declared = _sha(unsigned.pop("manifest_digest"), length=64, label="manifest digest")
    if object_sha256(unsigned) != declared:
        raise PairV6SCAIDTrainingError("SCAID manifest embedded digest differs")
    rows = root.get("events")
    if not isinstance(rows, list) or len(rows) != DP_SIZE or root.get("event_count") != DP_SIZE:
        raise PairV6SCAIDTrainingError("SCAID requires exactly one fit event per DP arm")
    events: list[EventSpec] = []
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != set(_EVENT_FIELDS):
            raise PairV6SCAIDTrainingError(f"event[{ordinal}] field closure differs")
        row = dict(raw)
        embedded = _sha(row.pop("event_digest"), length=64, label="event digest")
        if object_sha256(row) != embedded or row["schema_version"] != EVENT_SCHEMA:
            raise PairV6SCAIDTrainingError(f"event[{ordinal}] digest/schema differs")
        for name in ("sample_id", "fit_candidate_id", "action_family"):
            if not isinstance(row[name], str) or _SAFE_ID.fullmatch(row[name]) is None:
                raise PairV6SCAIDTrainingError(f"event[{ordinal}] {name} differs")
        prompts = scaid._prompts(row["raw_caption_by_branch"])
        if object_sha256(prompts) != row["raw_caption_bank_sha256"]:
            raise PairV6SCAIDTrainingError(f"event[{ordinal}] prompt digest differs")
        if (
            row["frame_count"] != FRAME_COUNT
            or row["fps"] != FPS
            or row["reference_indices"] != list(REFERENCE_INDICES)
        ):
            raise PairV6SCAIDTrainingError(f"event[{ordinal}] exact81 contract differs")
        source = _plain_file(row["source_video_path"], label="source video")
        wrong = _plain_file(row["wrong_source_video_path"], label="wrong source video")
        source_sha = _sha(row["source_video_sha256"], length=64, label="source SHA")
        wrong_sha = _sha(row["wrong_source_video_sha256"], length=64, label="wrong source SHA")
        if (
            source == wrong
            or source_sha == wrong_sha
            or file_sha256(source) != source_sha
            or file_sha256(wrong) != wrong_sha
        ):
            raise PairV6SCAIDTrainingError("correct/wrong source provenance is not distinct")
        wrong_source_iid = row["wrong_source_iid"]
        if not isinstance(wrong_source_iid, str) or _SAFE_ID.fullmatch(wrong_source_iid) is None:
            raise PairV6SCAIDTrainingError(f"event[{ordinal}] wrong-source IID differs")
        wrong_source_audit = load_wrong_source_audit(
            row["wrong_source_audit_path"],
            row["wrong_source_audit_file_sha256"],
            expected_audit_digest=row["wrong_source_audit_digest"],
            candidate_sample_id=row["sample_id"],
            candidate_source_video_sha256=source_sha,
            wrong_source_iid=wrong_source_iid,
            wrong_source_video_sha256=wrong_sha,
        )
        source_geometry = native_dpo._ffprobe_exact81(source)
        wrong_geometry = native_dpo._ffprobe_exact81(wrong)
        if any(
            source_geometry.get(field) != wrong_geometry.get(field)
            for field in ("width", "height", "avg_frame_rate")
        ):
            raise PairV6SCAIDTrainingError(
                "correct/wrong source width-height-fps geometry differs"
            )
        events.append(
            EventSpec(
                row["sample_id"], row["fit_candidate_id"], row["action_family"],
                source, source_sha, wrong, wrong_sha,
                wrong_source_iid, wrong_source_audit, prompts,
                row["raw_caption_bank_sha256"], embedded,
            )
        )
    if len({event.action_family for event in events}) != DP_SIZE or len(
        {event.fit_candidate_id for event in events}
    ) != DP_SIZE:
        raise PairV6SCAIDTrainingError("DP arms must use distinct fit families/candidates")
    return TrainingManifest(
        path,
        _sha(expected_sha256, length=64, label="manifest SHA"),
        _sha(root["checkpoint_tree_sha256"], length=64, label="checkpoint tree"),
        (events[0], events[1]),
        declared,
    )


def exact40_schedule_index(step: int) -> int:
    if type(step) is not int or step < 0:
        raise PairV6SCAIDTrainingError("schedule step must be nonnegative")
    index = step % EXACT40_STEPS
    action_adapter.sigma_gate(index)
    return index


def expected_optimizer_updates(steps: int) -> int:
    if type(steps) is not int or steps <= 0 or steps % EXACT40_STEPS:
        raise PairV6SCAIDTrainingError("schedule must contain complete exact40 cycles")
    return steps // EXACT40_STEPS * 38


def noise_seed(*, seed: int, step: int, dp_rank: int) -> int:
    material = f"{seed}\0pair-v6-scaid-source-coordinate\0{step}\0{dp_rank}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cagd-validator-evidence", required=True)
    parser.add_argument("--expected-cagd-validator-evidence-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-schedule-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-high-sigma-only", action="store_true")
    parser.add_argument("--high-sigma-smoke-receipt")
    parser.add_argument("--expected-high-sigma-smoke-receipt-sha256")
    parser.add_argument("--ack-experimental-no-action-success-claim", action="store_true")
    return parser


def preflight(
    args: argparse.Namespace,
) -> tuple[
    TrainingManifest,
    tuple[scaid.SCAIDAuthorization, ...],
    Mapping[str, Any],
    Optional[Mapping[str, Any]],
]:
    if args.ack_experimental_no_action_success_claim is not True:
        raise PairV6SCAIDTrainingError("experimental no-success-claim acknowledgement is required")
    if args.smoke_high_sigma_only:
        if args.max_schedule_steps != 1:
            raise PairV6SCAIDTrainingError(
                "single-cell high-sigma smoke requires max_schedule_steps=1"
            )
        if (
            args.high_sigma_smoke_receipt is not None
            or args.expected_high_sigma_smoke_receipt_sha256 is not None
        ):
            raise PairV6SCAIDTrainingError(
                "smoke run cannot consume a prior smoke receipt"
            )
    else:
        expected_optimizer_updates(args.max_schedule_steps)
        if (
            args.high_sigma_smoke_receipt is None
            or args.expected_high_sigma_smoke_receipt_sha256 is None
        ):
            raise PairV6SCAIDTrainingError(
                "full exact40 training requires a successful high-sigma smoke receipt"
            )
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise PairV6SCAIDTrainingError(f"{name} must be finite positive")
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_manifest_sha256", "expected_cagd_validator_evidence_sha256",
        "expected_checkpoint_tree_sha256", "method_source_archive_sha256",
        "expected_checkpoint_content_manifest_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    try:
        legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
    except Exception as error:
        raise PairV6SCAIDTrainingError(
            f"official Bernini/VeOmni source-tree validation failed: {error}"
        ) from error
    manifest = load_manifest(args.manifest, args.expected_manifest_sha256)
    if (
        args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
        or manifest.checkpoint_tree_sha256 != args.expected_checkpoint_tree_sha256
    ):
        raise PairV6SCAIDTrainingError("manifest/checkpoint identity differs")
    try:
        checkpoint_identity = source_audit.validate_checkpoint_content(
            Path(args.checkpoint),
            Path(args.checkpoint_content_manifest),
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
    except Exception as error:
        raise PairV6SCAIDTrainingError(
            f"authoritative checkpoint content validation failed: {error}"
        ) from error
    gates = tuple(
        scaid.load_authoritative_v3_authorization(
            args.cagd_validator_evidence,
            expected_evidence_sha256=args.expected_cagd_validator_evidence_sha256,
            checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
            fit_candidate_id=event.fit_candidate_id,
        )
        for event in manifest.events
    )
    try:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        rebuilt_t2v = tuple(
            build_task_prompt_registry(
                event.raw_caption_by_branch, prompt_cleaner=prompt_clean
            )[0]
            for event in manifest.events
        )
    except Exception as error:
        raise PairV6SCAIDTrainingError(
            f"raw caption to authoritative T2V prompt reconstruction failed: {error}"
        ) from error
    if any(
        gate.action_family != event.action_family
        or gate.prompt_bank_sha256 != object_sha256(t2v_prompts)
        or event.sample_id != gate.fit_candidate_id
        or event.source_video != gate.geometry_source_video_path
        or event.source_video_sha256 != gate.geometry_source_video_sha256
        for gate, event, t2v_prompts in zip(gates, manifest.events, rebuilt_t2v)
    ):
        raise PairV6SCAIDTrainingError(
            "manifest raw captions do not reconstruct authoritative fit T2V prompts"
        )
    smoke_prerequisite = None
    if not args.smoke_high_sigma_only:
        smoke_path = _plain_file(
            args.high_sigma_smoke_receipt, label="high-sigma smoke receipt"
        )
        smoke = _strict_json(
            smoke_path,
            args.expected_high_sigma_smoke_receipt_sha256,
            label="high-sigma smoke receipt",
        )
        smoke_unsigned = dict(smoke)
        smoke_digest = _sha(
            smoke_unsigned.pop("receipt_digest"),
            length=64,
            label="high-sigma smoke receipt digest",
        )
        expected_sources = [
            {
                "sample_id": event.sample_id,
                "source_video_sha256": event.source_video_sha256,
                "wrong_source_video_sha256": event.wrong_source_video_sha256,
            }
            for event in manifest.events
        ]
        smoke_sources = [
            {
                "sample_id": item.get("sample_id"),
                "source_video_sha256": item.get("source_video_sha256"),
                "wrong_source_video_sha256": item.get("wrong_source_video_sha256"),
            }
            for item in smoke.get("dp_runtime_provenance", ())
            if isinstance(item, Mapping)
        ]
        if (
            object_sha256(smoke_unsigned) != smoke_digest
            or smoke.get("schema_version") != RUN_RECEIPT_SCHEMA
            or smoke.get("complete") is not True
            or smoke.get("run_kind") != "single_cell_high_sigma_smoke"
            or smoke.get("schedule_steps") != 1
            or smoke.get("schedule_indices")
            != [action_adapter.HIGH_SIGMA_INDICES[0]]
            or smoke.get("optimizer_updates") != 1
            or smoke.get("training_manifest")
            != {
                "file_sha256": manifest.file_sha256,
                "manifest_digest": manifest.manifest_digest,
            }
            or smoke.get("checkpoint", {}).get("tree_sha256")
            != args.expected_checkpoint_tree_sha256
            or smoke.get("checkpoint", {}).get("content_manifest_sha256")
            != args.expected_checkpoint_content_manifest_sha256
            or smoke.get("method_source")
            != {
                "revision": args.method_source_revision,
                "archive_sha256": args.method_source_archive_sha256,
            }
            or smoke.get("official_source_trees")
            != {
                "bernini_commit": args.expected_bernini_commit,
                "veomni_commit": args.expected_veomni_commit,
            }
            or smoke.get("authoritative_v3", {}).get("evidence_file_sha256")
            != args.expected_cagd_validator_evidence_sha256
            or smoke_sources != expected_sources
            or smoke.get("training_config", {}).get("seed") != args.seed
            or smoke.get("training_config", {}).get("learning_rate")
            != args.learning_rate
            or smoke.get("training_config", {}).get("max_grad_norm")
            != args.max_grad_norm
            or smoke.get("training_config", {}).get("scaid_config")
            != dict(scaid.SCAIDConfig().__dict__)
        ):
            raise PairV6SCAIDTrainingError(
                "high-sigma smoke receipt is not bound to this full run"
            )
        artifacts = smoke.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "adapter.safetensors",
            "history.json",
        }:
            raise PairV6SCAIDTrainingError("high-sigma smoke artifacts differ")
        for name, digest in artifacts.items():
            artifact = smoke_path.parent / name
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or file_sha256(artifact)
                != _sha(digest, length=64, label=f"smoke {name} SHA-256")
            ):
                raise PairV6SCAIDTrainingError(
                    f"high-sigma smoke artifact {name} changed"
                )
        smoke_prerequisite = {
            "path": str(smoke_path),
            "file_sha256": _sha(
                args.expected_high_sigma_smoke_receipt_sha256,
                length=64,
                label="high-sigma smoke receipt SHA-256",
            ),
            "receipt_digest": smoke_digest,
            "artifact_sha256": dict(artifacts),
        }
    return manifest, gates, checkpoint_identity, smoke_prerequisite


def _broadcast_sp(value: Any, *, parallel: Any) -> None:
    import torch.distributed as dist

    source_rank = distributed_runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    dist.broadcast(value, src=source_rank, group=parallel.sp_group)


def _encode_video(
    path: Path,
    expected_sha: str,
    *,
    vae: Any,
    device: Any,
    parallel: Any,
) -> tuple[Any, tuple[Any, ...], Mapping[str, Any]]:
    import torch
    from bernini.pipeline import _vae_encode

    pixels, metadata, digest = source_audit.prepare_hashed_source_snapshot(path)
    if digest != expected_sha or metadata["frame_count"] != FRAME_COUNT or float(metadata["fps"]) != FPS:
        raise PairV6SCAIDTrainingError("source decoder/file binding differs")
    pixels = pixels.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        clean = _vae_encode(vae, pixels).float().detach().contiguous()
        refs = tuple(
            _vae_encode(vae, pixels[:, :, index : index + 1].contiguous())
            .float().detach().contiguous()
            for index in REFERENCE_INDICES
        )
    for tensor in (clean, *refs):
        _broadcast_sp(tensor, parallel=parallel)
    scaid._exact81(clean, label="encoded source clean latent")
    if any(
        tuple(ref.shape) != (1, 16, 1, clean.shape[3], clean.shape[4])
        or ref.dtype != torch.float32
        or ref.requires_grad
        or ref.grad_fn is not None
        or not bool(torch.isfinite(ref).all().item())
        for ref in refs
    ):
        raise PairV6SCAIDTrainingError("encoded source reference geometry differs")
    value = {
        "decoder": "infer_source_kv_carrier_oracle.prepare_hashed_source_snapshot",
        "video_path": str(path),
        "video_sha256": digest,
        "decoded_metadata": metadata,
        "vae_encoder": "AutoencoderKLWan.encode+latent_dist.mode",
        "clean_latent_sha256": scaid.tensor_sha256(clean),
        "reference_indices": list(REFERENCE_INDICES),
        "reference_latent_sha256": [scaid.tensor_sha256(ref) for ref in refs],
        "detached_finite_fp32": True,
    }
    return clean, refs, {**value, "digest": object_sha256(value)}


def build_task_prompt_registry(
    raw_captions: Mapping[str, str],
    *,
    prompt_cleaner: Any,
) -> tuple[Mapping[str, str], Mapping[str, str], Mapping[str, Any]]:
    captions = scaid._prompts(raw_captions)
    try:
        t2v_prompts, rv2v_prompts, core_receipt = scaid.build_task_prompt_banks(
            captions, prompt_cleaner=prompt_cleaner
        )
    except scaid.PairV6SCAIDError as error:
        raise PairV6SCAIDTrainingError(str(error)) from error
    construction: dict[str, Any] = {}
    for branch in scaid.BRANCH_ORDER:
        raw = captions[branch]
        t2v_prompt = t2v_prompts[branch]
        rv2v_prompt = rv2v_prompts[branch]
        construction[branch] = {
            "raw_caption_utf8_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "t2v_task_prompt_utf8_sha256": hashlib.sha256(
                t2v_prompt.encode("utf-8")
            ).hexdigest(),
            "rv2v_task_prompt_utf8_sha256": hashlib.sha256(
                rv2v_prompt.encode("utf-8")
            ).hexdigest(),
            "t2v_prefix_count": 1,
            "rv2v_prefix_count": 1,
        }
    receipt = {
        "raw_caption_bank_sha256": object_sha256(captions),
        "branches": construction,
        "t2v_and_rv2v_task_prompts_rebuilt_from_same_sealed_raw_caption": True,
        "double_wrap_forbidden": True,
        "core_task_prompt_receipt_digest": core_receipt["digest"],
    }
    return t2v_prompts, rv2v_prompts, {
        **receipt,
        "digest": object_sha256(receipt),
    }


def _encode_conditions(
    renderer: Any,
    tokenizer: Any,
    raw_captions: Mapping[str, str],
    *,
    device: Any,
    parallel: Any,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Any,
    Mapping[str, Any],
    Mapping[str, str],
    Mapping[str, str],
]:
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    t2v_prompts, rv2v_prompts, receipt = build_task_prompt_registry(
        raw_captions, prompt_cleaner=prompt_clean
    )
    t2v: dict[str, Any] = {}
    rv2v: dict[str, Any] = {}
    for branch in scaid.BRANCH_ORDER:
        for target, text in ((t2v, t2v_prompts[branch]), (rv2v, rv2v_prompts[branch])):
            ids, mask = native_runtime._tokenize_positive(tokenizer, text)
            with torch.inference_mode():
                embedding = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
            _broadcast_sp(embedding, parallel=parallel)
            target[branch] = embedding
    ids, mask = native_runtime._tokenize_negative(tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT)
    with torch.inference_mode():
        unconditional = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
    _broadcast_sp(unconditional, parallel=parallel)
    return t2v, rv2v, unconditional, receipt, t2v_prompts, rv2v_prompts


class FrozenT2VCallback:
    def __init__(self, diffusion: Any, transformer: Any, handle: Any, conditions: Mapping[str, Any], task_prompts: Mapping[str, str], sp_rank: int) -> None:
        self.diffusion, self.transformer, self.handle = diffusion, transformer, handle
        self.conditions, self.task_prompts, self.sp_rank = conditions, task_prompts, sp_rank
        self.query_id: Optional[int] = None
        self.branch: Any = None
        self.target_tail_sha256: Optional[str] = None

    def __call__(self, request: scaid.T2VFieldRequest) -> Any:
        import torch

        if request.prompt != self.task_prompts.get(request.branch):
            raise PairV6SCAIDTrainingError("T2V callback prompt differs from encoded task prompt")
        query = request.coordinate
        if self.query_id != id(query):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                patched = self.transformer.patch_vae_latent(
                    query.x_sigma.to(dtype=self.transformer.dtype), source_id=0
                )
            self.branch = t2v_runtime.build_t2v_target_branch(patched[0], patched[1], target_source_id=0)
            self.query_id = id(query)
            self.target_tail_sha256 = distributed_runtime.tensor_sha256(
                self.branch.noisy_latents.detach().float()
            )
        branch = self.branch
        route = action_adapter.PairV5ActionRoute(
            total_tokens=branch.total_token_count, condition_tokens=0,
            sequence_parallel_rank=self.sp_rank, sequence_parallel_size=SP_SIZE,
            branch_name="none", sigma_schedule_index=query.schedule_index, enabled=False,
        )
        with self.handle.route(route), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = self.diffusion.shared_step(
                model_id="transformer_1", noisy_latents=branch.noisy_latents,
                timesteps=query.timestep, cond_embeds=self.conditions[request.branch],
                rotary_embs=branch.rotary_embs,
                batch_vae_seqlen=list(branch.batch_vae_seqlen), batch_text_seqlen=[512],
            )
        return native_bridge._unpack_spatial_velocity(packed, video_shape=query.x_sigma.shape)


class NativeSCAIDCallback:
    def __init__(self, diffusion: Any, transformer: Any, handle: Any, event: RuntimeEvent, task_prompts: Mapping[str, str], sp_rank: int) -> None:
        self.diffusion, self.transformer, self.handle = diffusion, transformer, handle
        self.event, self.task_prompts, self.sp_rank = event, task_prompts, sp_rank
        self.cache: dict[tuple[int, str], Any] = {}
        self.target_tail_sha256_by_source_role: dict[str, str] = {}

    def _pack(self, request: scaid.NativeFieldRequest) -> Any:
        key = (id(request.coordinate), request.source_role)
        if key not in self.cache:
            if request.source_role == "wrong":
                video, refs = self.event.wrong_source_latent, self.event.wrong_source_references
            else:
                video, refs = self.event.source_latent, self.event.source_references
            self.cache[key] = native_runtime._build_pack(
                self.transformer, video, refs, request.coordinate.x_sigma
            )
            pack = self.cache[key]
            target_tails = {
                branch.name: branch.latents[:, branch.condition_tokens :, :]
                for branch in (pack.none, pack.video, pack.image, pack.video_image)
            }
            reference = target_tails["none"]
            if any(
                not reference.equal(value)
                for value in target_tails.values()
            ):
                raise PairV6SCAIDTrainingError(
                    "native none/V/I/VI target tails differ in content"
                )
            self.target_tail_sha256_by_source_role[request.source_role] = (
                distributed_runtime.tensor_sha256(reference.detach().float())
            )
        return self.cache[key]

    def __call__(self, request: scaid.NativeFieldRequest) -> Any:
        import torch

        if request.prompt != self.task_prompts.get(request.branch):
            raise PairV6SCAIDTrainingError("native callback prompt differs from encoded RV2V task prompt")
        pack = self._pack(request)
        query = request.coordinate
        if request.phase == "frozen_native_reference_identity_control_dI":
            components = []
            for branch in (pack.video, pack.video_image):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    packed = native_runtime._forward_native_branch(
                        self.diffusion, branch, timestep=query.timestep,
                        text=self.event.unconditional, action_handle=self.handle,
                        cio_handle=None, sp_rank=self.sp_rank,
                        sigma_index=query.schedule_index, action_enabled=False,
                    )
                    components.append(
                        native_bridge._unpack_spatial_velocity(
                            packed, video_shape=query.x_sigma.shape
                        ).float()
                    )
            return components[1] - components[0]
        cond = self.event.native_conditions[request.branch]
        rows = native_runtime._native_rows(
            pack, cond_embeds=cond, uncond_embeds=self.event.unconditional
        )
        if tuple(
            (name, float(coefficient))
            for name, _branch, _text, coefficient in rows
        ) != scaid.NATIVE_GUIDANCE_COMPONENTS:
            raise PairV6SCAIDTrainingError(
                "native CFG measurement order/coefficients differ"
            )
        components: dict[str, Any] = {}
        for name, branch, text, _coefficient in rows:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                packed = native_runtime._forward_native_branch(
                    self.diffusion, branch, timestep=query.timestep, text=text,
                    action_handle=self.handle, cio_handle=None, sp_rank=self.sp_rank,
                    sigma_index=query.schedule_index,
                    action_enabled=request.adapter_enabled,
                )
                components[name] = native_bridge._unpack_spatial_velocity(
                    packed, video_shape=query.x_sigma.shape
                )
        return scaid.aggregate_native_guidance_components(components)

    def replay_component(
        self, request: scaid.NativeFieldRequest, component_name: str
    ) -> Any:
        """Build exactly one native CFG component graph for serial VJP."""

        import torch

        if request.prompt != self.task_prompts.get(request.branch):
            raise PairV6SCAIDTrainingError("native replay prompt differs from encoded RV2V task prompt")
        if request.phase != "native_student_component_serial_vjp_replay":
            raise PairV6SCAIDTrainingError("component replay phase differs")
        pack = self._pack(request)
        condition = self.event.native_conditions[request.branch]
        native_rows = native_runtime._native_rows(
            pack,
            cond_embeds=condition,
            uncond_embeds=self.event.unconditional,
        )
        if tuple(
            (name, float(coefficient))
            for name, _branch, _text, coefficient in native_rows
        ) != scaid.NATIVE_GUIDANCE_COMPONENTS:
            raise PairV6SCAIDTrainingError(
                "native CFG replay order/coefficients differ"
            )
        rows = {
            name: (branch, text)
            for name, branch, text, _coefficient in native_rows
        }
        if component_name not in rows:
            raise PairV6SCAIDTrainingError("native CFG replay component differs")
        branch, text = rows[component_name]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = native_runtime._forward_native_branch(
                self.diffusion,
                branch,
                timestep=request.coordinate.timestep,
                text=text,
                action_handle=self.handle,
                cio_handle=None,
                sp_rank=self.sp_rank,
                sigma_index=request.coordinate.schedule_index,
                action_enabled=True,
            )
            spatial = native_bridge._unpack_spatial_velocity(
                packed, video_shape=request.coordinate.x_sigma.shape
            )
        return spatial


def _fresh_epsilon(shape: Sequence[int], *, seed: int, device: Any) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device).detach()


def validate_native_condition_geometry(
    source: Any,
    references: Sequence[Any],
    wrong_source: Any,
    wrong_references: Sequence[Any],
) -> Mapping[str, Any]:
    """Require correct/wrong native packs to have byte-compatible geometry."""

    import torch

    tensors = (source, wrong_source, *references, *wrong_references)
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
        for value in tensors
    ):
        raise PairV6SCAIDTrainingError(
            "native correct/wrong latents and references must be detached finite FP32"
        )
    source_shape = tuple(int(item) for item in source.shape)
    wrong_shape = tuple(int(item) for item in wrong_source.shape)
    reference_shapes = [tuple(int(item) for item in value.shape) for value in references]
    wrong_reference_shapes = [
        tuple(int(item) for item in value.shape) for value in wrong_references
    ]
    if (
        source_shape != wrong_shape
        or len(reference_shapes) != len(REFERENCE_INDICES)
        or reference_shapes != wrong_reference_shapes
        or any(shape != (1, 16, 1, source_shape[3], source_shape[4]) for shape in reference_shapes)
    ):
        raise PairV6SCAIDTrainingError(
            "encoded correct/wrong full latent or reference geometry differs"
        )
    value = {
        "source_and_wrong_full_latent_shape": list(source_shape),
        "correct_reference_shapes": [list(shape) for shape in reference_shapes],
        "wrong_reference_shapes": [list(shape) for shape in wrong_reference_shapes],
        "native_pack_geometry_compatible": True,
    }
    return {**value, "digest": object_sha256(value)}


def target_tail_equality_receipt(
    t2v_callback: FrozenT2VCallback,
    native_callback: NativeSCAIDCallback,
) -> Mapping[str, Any]:
    t2v_digest = t2v_callback.target_tail_sha256
    native_digests = native_callback.target_tail_sha256_by_source_role
    if (
        not isinstance(t2v_digest, str)
        or set(native_digests) != {"correct", "wrong"}
        or set(native_digests.values()) != {t2v_digest}
    ):
        raise PairV6SCAIDTrainingError(
            "T2V/correct/wrong target-tail content equality differs"
        )
    value = {
        "t2v_target_tail_sha256": t2v_digest,
        "native_correct_target_tail_sha256": native_digests["correct"],
        "native_wrong_target_tail_sha256": native_digests["wrong"],
        "same_x_sigma_target_tail_content_all_paths": True,
        "source_condition_packs_may_differ": True,
    }
    return {**value, "digest": object_sha256(value)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, gates, checkpoint_identity, smoke_prerequisite = preflight(args)
    if args.preflight_only:
        print(json.dumps({
            "preflight_only": True, "optimizer_authorized": True,
            "topology": "DP2xSP4", "frame_count": 81,
            "fit_families": [gate.action_family for gate in gates],
            "legacy_self_seal_trusted": False,
            "checkpoint_content_receipt_digest": object_sha256(checkpoint_identity),
        }, sort_keys=True), flush=True)
        return 0
    try:
        bernini_root, veomni_root, _, _ = legacy.validate_source_trees(
            args.bernini_root, args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise PairV6SCAIDTrainingError(str(error)) from error
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    contract = distributed_runtime.distributed_contract()
    device = distributed_runtime.initialise_distributed(contract)
    parallel = distributed_runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = distributed_runtime.prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True, **legacy.renderer_config_overrides(checkpoint)
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    # Match the proven native trainer memory order: renderer/adapters stay on
    # CPU while the VAE encodes both source conditions, then VAE is released.
    renderer = BerniniRendererModel(config).requires_grad_(False).eval()
    diffusion, transformer = renderer.diff_dec, renderer.diff_dec.transformer
    cagd_runtime._disable_gradient_checkpointing(renderer, transformer)
    handle = action_adapter.install_pair_v5_action_adapter(transformer)
    trainable = handle.trainable_named_parameters()
    if not handle.base_parameters_frozen():
        raise PairV6SCAIDTrainingError("Action-LoRA/base trainability closure differs")

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    ).eval().requires_grad_(False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", padding_side="right",
        trust_remote_code=True, local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    event_index = contract.arm_index
    spec, gate = manifest.events[event_index], gates[event_index]
    source, refs, source_preprocessing_receipt = _encode_video(
        spec.source_video, spec.source_video_sha256, vae=vae, device=device, parallel=parallel
    )
    wrong, wrong_refs, wrong_preprocessing_receipt = _encode_video(
        spec.wrong_source_video, spec.wrong_source_video_sha256,
        vae=vae, device=device, parallel=parallel,
    )
    condition_geometry = validate_native_condition_geometry(
        source, refs, wrong, wrong_refs
    )
    del vae
    torch.cuda.empty_cache()
    renderer.to(device).eval()
    initial_digest = distributed_runtime.synchronize_initial_parameters(
        trainable, parallel.world_group
    )
    (
        t2v_conditions,
        native_conditions,
        unconditional,
        prompt_construction_receipt,
        t2v_task_prompts,
        rv2v_task_prompts,
    ) = _encode_conditions(
        renderer, tokenizer, spec.raw_caption_by_branch, device=device, parallel=parallel
    )
    del tokenizer
    event = RuntimeEvent(
        spec, gate, source, refs, wrong, wrong_refs,
        t2v_conditions, native_conditions, unconditional,
        prompt_construction_receipt,
    )
    scaid_config = scaid.SCAIDConfig()
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=args.learning_rate, weight_decay=0.0)
    history: list[Mapping[str, Any]] = []
    updates = 0
    schedule_indices = (
        (action_adapter.HIGH_SIGMA_INDICES[0],)
        if args.smoke_high_sigma_only
        else tuple(
            exact40_schedule_index(step)
            for step in range(args.max_schedule_steps)
        )
    )
    for step, index in enumerate(schedule_indices):
        seed_value = noise_seed(seed=args.seed, step=step, dp_rank=contract.arm_index)
        epsilon = _fresh_epsilon(source.shape, seed=seed_value, device=device)
        _broadcast_sp(epsilon, parallel=parallel)
        optimizer.zero_grad(set_to_none=True)
        before = distributed_runtime.trainable_parameters_digest(trainable)
        t2v_callback = FrozenT2VCallback(
            diffusion, transformer, handle, t2v_conditions,
            t2v_task_prompts, contract.sp_rank,
        )
        native_callback = NativeSCAIDCallback(
            diffusion, transformer, handle, event,
            rv2v_task_prompts, contract.sp_rank,
        )
        cell = scaid.run_scaid_cell(
            source,
            epsilon,
            schedule_index=index,
            authoritative_evidence_path=gate.evidence_path,
            expected_authoritative_evidence_sha256=gate.evidence_file_sha256,
            fit_candidate_id=gate.fit_candidate_id,
            raw_caption_by_branch=spec.raw_caption_by_branch,
            expected_raw_caption_bank_sha256=spec.raw_caption_bank_sha256,
            checkpoint_tree_sha256=manifest.checkpoint_tree_sha256,
            frozen_t2v_callback=t2v_callback, native_callback=native_callback,
            config=scaid_config,
            leaf_vjp_mode=True,
        )
        if index in action_adapter.LOW_SIGMA_INDICES:
            if not cell.zero_update or any(parameter.grad is not None for _, parameter in trainable):
                raise PairV6SCAIDTrainingError("low sigma constructed update/gradient")
            after = distributed_runtime.parameter_consensus(
                trainable, parallel.world_group, f"SCAID low anchor {step}"
            )
            if before != after:
                raise PairV6SCAIDTrainingError("low sigma changed Action-LoRA")
            record = {
                "step": step,
                "index": index,
                "noise_seed": seed_value,
                "optimizer_step": False,
                "loss": None,
                "cell_receipt": dict(cell.receipt),
                "source_coordinate_receipt": dict(
                    cell.receipt["source_coordinate_receipt"]
                ),
            }
        else:
            if cell.objective is None or not cell.optimizer_authorized:
                raise PairV6SCAIDTrainingError("trainable SCAID cell was not authorized")
            target_tail_receipt = target_tail_equality_receipt(
                t2v_callback, native_callback
            )
            cell.objective.loss.backward()
            replay = scaid.replay_native_student_vjp(
                cell, native_callback
            )
            grad_norm = distributed_runtime.synchronize_gradients(trainable, parallel)
            clipped = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], args.max_grad_norm)
            if not math.isfinite(float(clipped)):
                raise PairV6SCAIDTrainingError("non-finite Action-LoRA gradient")
            optimizer.step()
            updates += 1
            after = distributed_runtime.parameter_consensus(
                trainable, parallel.world_group, f"SCAID update {updates}"
            )
            record = {
                "step": step, "index": index, "optimizer_step": True,
                "noise_seed": seed_value,
                "loss": float(cell.objective.loss.detach().item()),
                "loss_components": {
                    "action_match": float(
                        cell.objective.action_match_loss.detach().item()
                    ),
                    "negative_parity": float(
                        cell.objective.negative_parity_loss.detach().item()
                    ),
                    "action_base_trust": float(
                        cell.objective.action_base_trust_loss.detach().item()
                    ),
                    "parity_by_branch": {
                        name: float(value.detach().item())
                        for name, value in cell.objective.parity_by_branch.items()
                    },
                },
                "cell_receipt": dict(cell.receipt),
                "source_coordinate_receipt": dict(
                    cell.receipt["source_coordinate_receipt"]
                ),
                "residual_survival_receipt": dict(
                    cell.objective.survival.receipt
                ),
                "residual_survival_receipt_digest": cell.objective.survival.receipt["receipt_digest"],
                "projected_survival_ratio": cell.objective.survival.projected_survival_ratio,
                "vjp_replay_max_abs": max(replay.values()),
                "preclip_gradient_norm": grad_norm, "parameter_digest": after,
                "target_tail_equality_receipt": target_tail_receipt,
            }
        projection = dict(record)
        distributed_runtime.digest_consensus(
            object_sha256(projection), group=parallel.sp_group,
            expected_count=SP_SIZE, label=f"SCAID step {step}",
        )
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, record, group=parallel.world_group)
        if contract.rank == 0:
            history.append({"step": step, "index": index, "dp_records": [gathered[0], gathered[4]]})
        del epsilon, t2v_callback, native_callback, cell
        torch.cuda.empty_cache()
    expected_updates = (
        1
        if args.smoke_high_sigma_only
        else expected_optimizer_updates(args.max_schedule_steps)
    )
    if updates != expected_updates:
        raise PairV6SCAIDTrainingError("exact40 optimizer count differs")
    final_digest = distributed_runtime.parameter_consensus(trainable, parallel.world_group, "SCAID final")
    if final_digest == initial_digest:
        raise PairV6SCAIDTrainingError("SCAID did not change Action-LoRA")
    manifest.assert_unchanged()
    try:
        legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
    except Exception as error:
        raise PairV6SCAIDTrainingError(
            f"official source trees changed during run: {error}"
        ) from error
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    for gate_item in gates:
        authoritative_t2v_prompts = build_task_prompt_registry(
            manifest.events[gates.index(gate_item)].raw_caption_by_branch,
            prompt_cleaner=prompt_clean,
        )[0]
        gate_item.validate(
            prompt_by_branch=authoritative_t2v_prompts,
            checkpoint_tree_sha256=manifest.checkpoint_tree_sha256,
        )
    trainer_checkpoint_manifest_sha = file_sha256(
        Path(args.checkpoint_content_manifest)
    )
    if trainer_checkpoint_manifest_sha != args.expected_checkpoint_content_manifest_sha256:
        raise PairV6SCAIDTrainingError("checkpoint content manifest changed during run")
    try:
        final_checkpoint_identity = source_audit.validate_checkpoint_content(
            Path(args.checkpoint),
            Path(args.checkpoint_content_manifest),
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
    except Exception as error:
        raise PairV6SCAIDTrainingError(
            f"checkpoint content changed during run: {error}"
        ) from error
    if object_sha256(final_checkpoint_identity) != object_sha256(checkpoint_identity):
        raise PairV6SCAIDTrainingError("checkpoint identity receipt changed during run")
    if smoke_prerequisite is not None:
        smoke_path = Path(smoke_prerequisite["path"])
        if file_sha256(smoke_path) != smoke_prerequisite["file_sha256"]:
            raise PairV6SCAIDTrainingError("high-sigma smoke receipt changed during run")
        for name, digest in smoke_prerequisite["artifact_sha256"].items():
            if file_sha256(smoke_path.parent / name) != digest:
                raise PairV6SCAIDTrainingError(
                    f"high-sigma smoke artifact {name} changed during run"
                )
    local_runtime_provenance = {
        "dp_rank": contract.arm_index,
        "sample_id": spec.sample_id,
        "fit_candidate_id": spec.fit_candidate_id,
        "action_family": spec.action_family,
        "source_video_sha256": spec.source_video_sha256,
        "wrong_source_video_sha256": spec.wrong_source_video_sha256,
        "wrong_source_iid": spec.wrong_source_iid,
        "wrong_source_audit_file_sha256": spec.wrong_source_audit[
            "file_sha256"
        ],
        "wrong_source_audit_digest": spec.wrong_source_audit["audit_digest"],
        "wrong_source_review_artifact_sha256": spec.wrong_source_audit[
            "review_artifact_sha256"
        ],
        "native_condition_geometry": condition_geometry,
        "prompt_construction_receipt": prompt_construction_receipt,
        "source_preprocessing_receipt": source_preprocessing_receipt,
        "wrong_source_preprocessing_receipt": wrong_preprocessing_receipt,
    }
    gathered_runtime_provenance: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_runtime_provenance,
        local_runtime_provenance,
        group=parallel.world_group,
    )
    dist.barrier(group=parallel.world_group)
    if contract.rank == 0:
        adapter = cagd_runtime._save_action_adapter(stage / "adapter.safetensors", handle)
        history_path = stage / "history.json"
        adapter_path = stage / "adapter.safetensors"
        distributed_runtime.atomic_json(history_path, {
            "schema_version": RUN_RECEIPT_SCHEMA, "records": history,
            "optimizer_updates": updates,
        })
        artifacts = {
            "adapter.safetensors": file_sha256(adapter_path),
            "history.json": file_sha256(history_path),
        }
        receipt_value = {
            "schema_version": RUN_RECEIPT_SCHEMA, "method": METHOD_NAME, "complete": True,
            "topology": "DP2xSP4", "frame_count": FRAME_COUNT,
            "run_kind": (
                "single_cell_high_sigma_smoke"
                if args.smoke_high_sigma_only
                else "full_exact40_training"
            ),
            "schedule_steps": len(schedule_indices), "optimizer_updates": updates,
            "schedule_indices": list(schedule_indices),
            "zero_update_indices": ([] if args.smoke_high_sigma_only else [38, 39]),
            "fit_families": [item.action_family for item in gates],
            "training_config": {
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "max_grad_norm": args.max_grad_norm,
                "optimizer": "torch.optim.AdamW",
                "optimizer_betas": list(optimizer.defaults["betas"]),
                "optimizer_eps": optimizer.defaults["eps"],
                "weight_decay": optimizer.defaults["weight_decay"],
                "scaid_config": dict(scaid_config.__dict__),
                "world_size": WORLD_SIZE,
                "dp_size": DP_SIZE,
                "sp_size": SP_SIZE,
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "reference_indices": list(REFERENCE_INDICES),
                "renderer_dtype": "torch.bfloat16",
                "field_and_cfg_aggregation_dtype": "torch.float32",
                "leaf_vjp_mode": True,
                "native_guidance_components": [
                    [name, coefficient]
                    for name, coefficient in scaid.NATIVE_GUIDANCE_COMPONENTS
                ],
            },
            "checkpoint": {
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_manifest_sha256": args.expected_checkpoint_content_manifest_sha256,
                "content_receipt_digest": object_sha256(checkpoint_identity),
            },
            "method_source": {
                "revision": args.method_source_revision,
                "archive_sha256": args.method_source_archive_sha256,
            },
            "official_source_trees": {
                "bernini_commit": args.expected_bernini_commit,
                "veomni_commit": args.expected_veomni_commit,
            },
            "training_manifest": {
                "file_sha256": manifest.file_sha256,
                "manifest_digest": manifest.manifest_digest,
            },
            "authoritative_v3": {
                "evidence_file_sha256": gates[0].evidence_file_sha256,
                "evidence_digest": gates[0].evidence_digest,
                "authorization_digests": [item.authorization_digest for item in gates],
                "calibration_receipt_digest": gates[0].calibration_receipt_digest,
            },
            "dp_runtime_provenance": [
                gathered_runtime_provenance[0], gathered_runtime_provenance[4]
            ],
            "legacy_self_seal_trusted": False, "pure_t2v_visual_tensor_consumed": False,
            "adapter_roundtrip": {
                key: value for key, value in adapter.items() if key != "path"
            },
            "artifacts": artifacts,
            "final_parameter_digest": final_digest,
            "semantic_action_editing_success_claimed": False,
            "high_sigma_smoke_prerequisite": smoke_prerequisite,
        }
        receipt = {
            **receipt_value,
            "receipt_digest": object_sha256(receipt_value),
        }
        distributed_runtime.atomic_json(stage / "receipt.json", receipt)
        os.replace(stage, output)
        distributed_runtime.fsync_directory(output.parent)
    dist.barrier(group=parallel.world_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVENT_SCHEMA", "MANIFEST_SCHEMA", "PairV6SCAIDTrainingError",
    "exact40_schedule_index", "expected_optimizer_updates", "load_manifest", "preflight",
    "validate_native_condition_geometry",
    "target_tail_equality_receipt",
    "build_task_prompt_registry",
]
