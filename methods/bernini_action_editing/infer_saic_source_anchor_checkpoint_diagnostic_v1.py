#!/usr/bin/env python3
"""Run the first decoded diagnostic for a formal SAIC source-anchor checkpoint.

This is a Stage-A checkpoint canary, not a Stage-B action editor.  It binds one
formal-32 Stage-A receipt to its exact safetensors file and source manifest,
selects one held-out correct/wrong pair, and runs seven matched official
Bernini ``v2v_apg`` exact81/exact40 cells on WORLD4/Ulysses4:

* frozen-base correct-source no-op;
* trained-anchor correct-source no-op;
* trained-anchor wrong-source no-op;
* trained-anchor route-drop no-op with the complete correct source retained;
* trained-anchor synthetic zero-condition no-op (an OOD diagnostic only);
* frozen-base correct-source action;
* trained-anchor correct-source action.

The frozen-base cells use the adapter's certified zero-initialized output maps
before any checkpoint tensor is loaded.  Every routed cell is enclosed by the
audited one-sample source-anchor native runtime.  The route-drop cell is
required to be byte-identical to the matched frozen-base no-op cell and keeps
the full source video plus four source-derived references in the stock sampler.

Rank zero decodes every result, produces full81/full80 camera/technical media
diagnostics, and computes a frozen DINOv2 correct-vs-wrong visual proxy from a
complete hash-manifest-bound checkpoint.  These measurements have permanently
zero training, checkpoint, identity, semantic-action, publication, and
production authority.  In particular, no qualified action-event observer is
available: action non-regression is reported only as raw visual/camera/quality
deltas and semantic action non-regression remains unavailable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import types
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_native_self_guided_action_field_canary as strong_audit  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import pair_v5_source_bound_preservation_evaluator_v1 as visual_contract  # noqa: E402
import saic_exact81_media_diagnostics_v1 as media_diagnostics  # noqa: E402
import saic_source_anchor_adapter_v1 as anchor_adapter  # noqa: E402
import saic_source_anchor_native_runtime_v1 as anchor_runtime  # noqa: E402
import score_pair_v5_source_bound_preservation_v1 as visual_scorer  # noqa: E402
import train_saic_source_anchor_v1 as anchor_trainer  # noqa: E402


SCHEMA_VERSION = "bernini-saic-source-anchor-checkpoint-diagnostic-v1"
DIAGNOSTIC_SCHEMA_VERSION = (
    "bernini-saic-source-anchor-checkpoint-decoded-diagnostics-v1"
)
METHOD = "saic-source-anchor-formal32-first-decoded-canary"
CLASSIFICATION = "stage_a_checkpoint_diagnostic/no_stage_b_or_scientific_authority"
WORLD_SIZE = 4
ULYSSES_SIZE = 4
FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
FPS = 25
REFERENCE_INDICES = (0, 27, 53, 80)
CELL_ORDER = (
    "base_correct_noop",
    "anchor_correct_noop",
    "anchor_wrong_noop",
    "anchor_route_drop_noop",
    "anchor_zero_condition_noop",
    "base_correct_action",
    "anchor_correct_action",
)
ROUTED_CELLS = frozenset(
    {
        "base_correct_noop",
        "anchor_correct_noop",
        "anchor_wrong_noop",
        "anchor_zero_condition_noop",
        "base_correct_action",
        "anchor_correct_action",
    }
)
BASE_CELLS = frozenset({"base_correct_noop", "base_correct_action"})
NOOP_CELLS = frozenset(CELL_ORDER[:5])
ACTION_CELLS = frozenset(CELL_ORDER[5:])
SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)\Z")
EXPECTED_RENDEZVOUS_GUARD_SHA256 = (
    "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
)
FORMAL_POSTFLIGHT_SCHEMA = "saic-source-anchor-formal32-terminal-admission-v1"
FORMAL_HISTORY_SCHEMA = "bernini-saic-source-anchor-history-v2"
VISUAL_RELEASE_SCHEMA = "saic-source-anchor-diagnostic-visual-release-v1"
VISUAL_EVALUATOR_SPEC_SCHEMA = (
    "saic-source-anchor-diagnostic-visual-evaluator-spec-v1"
)
RENDEZVOUS_RANK_SCHEMA = "saic-source-anchor-diagnostic-rank-admission-v1"
RENDEZVOUS_DECISION_SCHEMA = "saic-source-anchor-diagnostic-world4-admission-v1"

FORMAL_POSTFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "complete",
        "slurm",
        "formal_output",
        "bundle",
        "history",
        "heldout_gate",
        "publication",
        "authority",
        "receipt_digest",
    }
)
FORMAL_HISTORY_FIELDS = frozenset(
    {
        "schema_version",
        "complete",
        "optimizer_updates",
        "update_indices",
        "rows",
        "history_digest",
    }
)
CHECKPOINT_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "complete",
        "release_path",
        "formal_output_namespace",
        "job_id",
        "source_release_manifest_sha256",
        "submission_receipt_sha256",
        "postflight_source_sha256",
        "trainer_source_sha256",
        "history_digest",
        "heldout_gate",
        "authority",
        "artifacts",
        "payload_files_digest",
        "receipt_digest",
    }
)
VISUAL_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "checkpoint",
        "evaluator",
        "runtime_versions",
        "golden_preprocessor",
        "authority",
        "receipt_digest",
    }
)
VISUAL_EVALUATOR_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "adapter_id",
        "architecture_id",
        "checkpoint_manifest_sha256",
        "checkpoint_tree_digest",
        "checkpoint_file_count",
        "evaluator_sources",
        "evaluator_sources_digest",
        "runtime_versions",
        "golden_preprocessor",
        "authority",
        "spec_digest",
    }
)
FORMAL_POSTFLIGHT_AUTHORITY = {
    "formal_stage_a_training_completed": True,
    "stage_a_checkpoint_release": True,
    "stage_b": False,
    "semantic_action": False,
    "identity": False,
    "candidate_selection": False,
    "production": False,
}
CHECKPOINT_RELEASE_AUTHORITY = {
    "stage_a_checkpoint_release": True,
    "stage_b": False,
    "semantic_action": False,
    "identity": False,
    "candidate_selection": False,
    "production": False,
}
VISUAL_RELEASE_AUTHORITY = {
    "identity": False,
    "semantic_action": False,
    "candidate_selection": False,
    "training": False,
    "publication": False,
    "production": False,
}
RENDEZVOUS_AUTHORITY = {
    "scientific": False,
    "training": False,
    "checkpoint": False,
    "selection": False,
    "publication": False,
    "production": False,
}


class SAICSourceAnchorDiagnosticError(RuntimeError):
    """Raised before an ambiguous checkpoint, rollout, or receipt is accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICSourceAnchorDiagnosticError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, bits: int, label: str) -> str:
    pattern = SHA1 if bits == 160 else SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise SAICSourceAnchorDiagnosticError(
            f"{label} must be lowercase SHA-{'1' if bits == 160 else '256'}"
        )
    return value


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise SAICSourceAnchorDiagnosticError(
            f"{label} keys differ: missing={sorted(set(fields)-actual)} "
            f"extra={sorted(actual-set(fields))}"
        )
    return value


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/"):
        raise SAICSourceAnchorDiagnosticError(f"{label} must be absolute and non-root")
    try:
        row = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SAICSourceAnchorDiagnosticError(f"cannot resolve {label}: {path}") from error
    if (
        resolved != path
        or stat.S_ISLNK(row.st_mode)
        or not stat.S_ISREG(row.st_mode)
    ):
        raise SAICSourceAnchorDiagnosticError(
            f"{label} must be one canonical non-symlink file"
        )
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/"):
        raise SAICSourceAnchorDiagnosticError(f"{label} must be absolute and non-root")
    try:
        row = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SAICSourceAnchorDiagnosticError(f"cannot resolve {label}: {path}") from error
    if (
        resolved != path
        or stat.S_ISLNK(row.st_mode)
        or not stat.S_ISDIR(row.st_mode)
    ):
        raise SAICSourceAnchorDiagnosticError(
            f"{label} must be one canonical non-symlink directory"
        )
    return path


def _stable_file_sha256(path: Path, *, label: str) -> str:
    before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SAICSourceAnchorDiagnosticError(f"cannot open {label}: {path}") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise SAICSourceAnchorDiagnosticError(f"{label} changed while opening")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    if not (
        _file_signature(before)
        == _file_signature(opened)
        == _file_signature(after)
        == _file_signature(named)
    ):
        raise SAICSourceAnchorDiagnosticError(f"{label} changed while hashing")
    return digest.hexdigest()


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    signature: tuple[int, int, int, int, int]

    @classmethod
    def capture(
        cls, value: str | Path, expected_sha256: str, *, label: str
    ) -> "FileSnapshot":
        path = _plain_file(value, label=label)
        expected = _sha(expected_sha256, bits=256, label=f"{label} expected SHA-256")
        observed = _stable_file_sha256(path, label=label)
        if observed != expected:
            raise SAICSourceAnchorDiagnosticError(f"{label} SHA-256 differs")
        return cls(path, observed, _file_signature(path.lstat()))

    def assert_unchanged(self) -> None:
        if (
            _file_signature(self.path.lstat()) != self.signature
            or _stable_file_sha256(self.path, label="bound file") != self.sha256
        ):
            raise SAICSourceAnchorDiagnosticError(f"bound file changed: {self.path}")

    def receipt(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_size": self.signature[2],
        }


def _strict_json(
    snapshot: FileSnapshot, *, label: str, canonical_required: bool = True
) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise SAICSourceAnchorDiagnosticError(f"{label} contains {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SAICSourceAnchorDiagnosticError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        raw = snapshot.path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICSourceAnchorDiagnosticError(f"cannot decode {label}") from error
    snapshot.assert_unchanged()
    if (
        not isinstance(value, Mapping)
        or (
            canonical_required
            and raw != canonical_json_bytes(value) + b"\n"
        )
    ):
        raise SAICSourceAnchorDiagnosticError(
            f"{label} must be one canonical newline-terminated JSON object"
        )
    return value


STAGE_A_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "complete",
        "status",
        "run_contract",
        "manifest",
        "native_runtime",
        "objective",
        "adapter",
        "heldout_gate",
        "scientific_limitations",
        "artifacts",
        "model",
        "runtime",
        "method_source_revision",
        "method_source_archive_sha256",
        "trainer_source_sha256",
        "formal_full60_admission_sha256",
        "formal_full60_admission_digest",
        "source_anchor_pretext_only",
        "action_training",
        "semantic_action_editing_success",
        "decoded_rgb_appearance_preservation_success",
        "source_anchor_checkpoint_candidate_eligible",
        "source_anchor_checkpoint_publication_authorized",
        "action_stage_authorized",
        "semantic_action_authorized",
        "decoded_rgb_identity_authorized",
        "stage_a_checkpoint_release_requires_external_terminal_postflight",
        "smoke_incomplete_row_coverage",
        "receipt_digest",
    }
)


def validate_stage_a_bundle(
    *,
    adapter: FileSnapshot,
    receipt: FileSnapshot,
    manifest: FileSnapshot,
    checkpoint_manifest: FileSnapshot,
) -> Mapping[str, Any]:
    """Bind the formal publication envelope to loader metadata and all inputs."""

    value = _closed(
        _strict_json(receipt, label="Stage-A receipt"),
        STAGE_A_RECEIPT_FIELDS,
        label="Stage-A receipt",
    )
    declared = _sha(value["receipt_digest"], bits=256, label="Stage-A receipt digest")
    unsigned = {key: item for key, item in value.items() if key != "receipt_digest"}
    if object_sha256(unsigned) != declared:
        raise SAICSourceAnchorDiagnosticError("Stage-A receipt digest differs")
    run = value["run_contract"]
    adapter_row = value["adapter"]
    gate = value["heldout_gate"]
    artifacts = value["artifacts"]
    model = value["model"]
    training_runtime = value["runtime"]
    manifest_row = value["manifest"]
    limitations = value["scientific_limitations"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            run,
            adapter_row,
            gate,
            artifacts,
            model,
            training_runtime,
            manifest_row,
            limitations,
        )
    ):
        raise SAICSourceAnchorDiagnosticError("Stage-A nested receipt differs")
    roundtrip = adapter_row.get("safetensors_roundtrip")
    if not isinstance(roundtrip, Mapping):
        raise SAICSourceAnchorDiagnosticError("Stage-A safetensors receipt is absent")
    required_literals = (
        value["schema_version"] == anchor_trainer.RUN_RECEIPT_SCHEMA,
        value["method"] == anchor_trainer.METHOD_NAME,
        value["complete"] is True,
        value["status"] == "FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE",
        run.get("mode") == "formal",
        run.get("world_size") == anchor_trainer.WORLD_SIZE,
        run.get("data_parallel_size") == anchor_trainer.DP_SIZE,
        run.get("sequence_parallel_size") == anchor_trainer.SP_SIZE,
        run.get("frame_count") == FRAME_COUNT,
        run.get("optimizer_updates") == anchor_trainer.FORMAL_UPDATES,
        run.get("all_train_rows_used_once_as_clean_endpoint") is True,
        adapter_row.get("checkpoint_candidate_materialized") is True,
        adapter_row.get("checkpoint_published") is False,
        gate.get("noncompensating_all_pass") is True,
        gate.get("checkpoint_publication_allowed") is True,
        value["source_anchor_checkpoint_candidate_eligible"] is True,
        value["source_anchor_checkpoint_publication_authorized"] is False,
        value["source_anchor_pretext_only"] is True,
        value["action_training"] is False,
        value["semantic_action_editing_success"] is False,
        value["decoded_rgb_appearance_preservation_success"] is False,
        value["action_stage_authorized"] is False,
        value["semantic_action_authorized"] is False,
        value["decoded_rgb_identity_authorized"] is False,
        value["stage_a_checkpoint_release_requires_external_terminal_postflight"]
        is True,
        value["smoke_incomplete_row_coverage"] is False,
        limitations.get("future_action_stage_requires_fresh_rollout_nonregression")
        is True,
        limitations.get(
            "future_action_stage_must_test_action_and_identity_camera_background_separately"
        )
        is True,
        artifacts.get("adapter.safetensors") == adapter.sha256,
        roundtrip.get("schema_version") == anchor_trainer.SAFETENSORS_SCHEMA,
        roundtrip.get("file_sha256") == adapter.sha256,
        roundtrip.get("roundtrip_byte_exact_tensors") is True,
        roundtrip.get("metadata_closed") is True,
        manifest_row.get("file_sha256") == manifest.sha256,
        model.get("bernini_commit") == legacy.trainer.BERNINI_OFFICIAL_COMMIT,
        model.get("veomni_commit") == legacy.trainer.VEOMNI_TESTED_COMMIT,
        model.get("checkpoint_tree_sha256")
        == legacy.trainer.CHECKPOINT_TREE_SHA256,
        model.get("checkpoint_content_manifest_file_sha256")
        == checkpoint_manifest.sha256,
        model.get("checkpoint_content_post_training_revalidated") is True,
        model.get("single_expert") == "transformer_1",
    )
    if not all(required_literals):
        raise SAICSourceAnchorDiagnosticError(
            "Stage-A formal candidate or cross-artifact contract differs"
        )
    adapter_contract_digest = _sha(
        adapter_row.get("digest"), bits=256, label="adapter contract digest"
    )
    state_tensor_sha = _sha(
        roundtrip.get("state_tensor_sha256"),
        bits=256,
        label="adapter state tensor digest",
    )
    state_key_sha = _sha(
        roundtrip.get("state_key_sha256"), bits=256, label="adapter key digest"
    )
    heldout_digest = _sha(gate.get("digest"), bits=256, label="held-out gate digest")
    manifest_digest = _sha(
        manifest_row.get("manifest_digest"), bits=256, label="source manifest digest"
    )
    training_source_revision = _sha(
        value.get("method_source_revision"),
        bits=160,
        label="Stage-A method source revision",
    )
    training_source_archive_sha256 = _sha(
        value.get("method_source_archive_sha256"),
        bits=256,
        label="Stage-A method source archive",
    )
    trainer_source_sha256 = _sha(
        value.get("trainer_source_sha256"),
        bits=256,
        label="Stage-A trainer source",
    )
    formal_full60_admission_sha256 = _sha(
        value.get("formal_full60_admission_sha256"),
        bits=256,
        label="Stage-A formal full60 admission",
    )
    formal_full60_admission_digest = _sha(
        value.get("formal_full60_admission_digest"),
        bits=256,
        label="Stage-A formal full60 admission digest",
    )
    training_release_manifest_sha256 = _sha(
        training_runtime.get("release_manifest_sha256"),
        bits=256,
        label="Stage-A training release manifest",
    )
    training_submission_receipt_sha256 = _sha(
        training_runtime.get("submission_receipt_sha256"),
        bits=256,
        label="Stage-A training submission receipt",
    )
    metadata = {
        "schema_version": anchor_runtime.SAFETENSORS_SCHEMA_VERSION,
        "adapter_schema_version": anchor_adapter.SCHEMA_VERSION,
        "adapter_contract_digest": adapter_contract_digest,
        "state_tensor_sha256": state_tensor_sha,
        "state_key_sha256": state_key_sha,
        "optimizer_updates": str(anchor_trainer.FORMAL_UPDATES),
        "heldout_gate_digest": heldout_digest,
        "source_anchor_only": "true",
        "semantic_action_success": "false",
    }
    return {
        "receipt_digest": declared,
        "adapter_sha256": adapter.sha256,
        "adapter_contract_digest": adapter_contract_digest,
        "state_tensor_sha256": state_tensor_sha,
        "state_key_sha256": state_key_sha,
        "heldout_gate_digest": heldout_digest,
        "manifest_digest": manifest_digest,
        "training_source_revision": training_source_revision,
        "training_source_archive_sha256": training_source_archive_sha256,
        "trainer_source_sha256": trainer_source_sha256,
        "formal_full60_admission_sha256": formal_full60_admission_sha256,
        "formal_full60_admission_digest": formal_full60_admission_digest,
        "training_release_manifest_sha256": training_release_manifest_sha256,
        "training_submission_receipt_sha256": training_submission_receipt_sha256,
        "expected_safetensors_metadata": metadata,
        "training_receipt_is_private_candidate_only": True,
        "external_terminal_postflight_is_sole_publication_authority": True,
        "source_anchor_pretext_only": True,
        "decoded_qualification_still_required": True,
        "stage_b_authorized": False,
    }


def _validate_snapshot_row(
    value: Any, snapshot: FileSnapshot, *, label: str
) -> Mapping[str, Any]:
    fields = {"path", "sha256", "byte_size", "device", "inode"}
    row = _closed(value, fields, label=label)
    info = snapshot.path.lstat()
    if row != {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "byte_size": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
    }:
        raise SAICSourceAnchorDiagnosticError(f"{label} snapshot differs")
    return row


def validate_formal_stage_a_postflight(
    *,
    postflight: FileSnapshot,
    history: FileSnapshot,
    adapter: FileSnapshot,
    receipt: FileSnapshot,
    manifest: FileSnapshot,
    checkpoint_manifest: FileSnapshot,
    checkpoint_release: FileSnapshot,
) -> Mapping[str, Any]:
    """Accept only one externally postflighted, immutable formal32 bundle.

    The producer-side training receipt is necessary but not sufficient.  This
    independent envelope binds terminal Slurm accounting, the exact formal
    output namespace, all 32 update-history rows, and the released Stage-A
    files that this diagnostic will consume.
    """

    value = _closed(
        _strict_json(postflight, label="formal Stage-A postflight"),
        FORMAL_POSTFLIGHT_FIELDS,
        label="formal Stage-A postflight",
    )
    unsigned = dict(value)
    digest = _sha(
        unsigned.pop("receipt_digest", None),
        bits=256,
        label="formal Stage-A postflight digest",
    )
    if object_sha256(unsigned) != digest:
        raise SAICSourceAnchorDiagnosticError("formal Stage-A postflight seal differs")
    slurm = value.get("slurm")
    formal_output = value.get("formal_output")
    bundle = value.get("bundle")
    history_row = value.get("history")
    heldout_gate = value.get("heldout_gate")
    publication = value.get("publication")
    if not all(
        isinstance(item, Mapping)
        for item in (
            slurm,
            formal_output,
            bundle,
            history_row,
            heldout_gate,
            publication,
        )
    ):
        raise SAICSourceAnchorDiagnosticError(
            "formal Stage-A postflight nested contract differs"
        )
    if (
        value.get("schema_version") != FORMAL_POSTFLIGHT_SCHEMA
        or value.get("status")
        != "FORMAL_GATE_PASS_CHECKPOINT_RELEASED"
        or value.get("complete") is not True
        or slurm.get("terminal_state") != "COMPLETED"
        or slurm.get("exit_code") != "0:0"
        or slurm.get("world_size") != anchor_trainer.WORLD_SIZE
        or slurm.get("data_parallel_size") != anchor_trainer.DP_SIZE
        or slurm.get("sequence_parallel_size") != anchor_trainer.SP_SIZE
        or slurm.get("exact_submit_line_verified") is not True
        or slurm.get("terminal_log_closure_verified") is not True
        or formal_output.get("same_formal_output_namespace") is not True
        or formal_output.get("fresh_create_only_release") is not True
        or formal_output.get("directory_mode") != "0555"
        or set(bundle)
        != {
            "adapter",
            "training_receipt",
            "training_history",
            "source_manifest",
            "checkpoint_manifest",
            "checkpoint_release",
            "file_count",
            "files_digest",
        }
        or bundle.get("file_count") != 6
        or history_row.get("optimizer_updates") != anchor_trainer.FORMAL_UPDATES
        or history_row.get("all_updates_present_exactly_once") is not True
        or heldout_gate.get("noncompensating_all_pass") is not True
        or heldout_gate.get("checkpoint_publication_allowed") is not True
        or publication.get("adapter_mode") != "0444"
        or publication.get("directory_mode") != "0555"
        or publication.get("stage_a_checkpoint_release") is not True
        or publication.get("all_bundle_files_plain_single_link_0444") is not True
        or publication.get("postflight_created_after_terminal_accounting") is not True
        or value.get("authority") != FORMAL_POSTFLIGHT_AUTHORITY
    ):
        raise SAICSourceAnchorDiagnosticError(
            "formal Stage-A terminal publication contract differs"
        )
    bound = {
        "adapter": adapter,
        "training_receipt": receipt,
        "training_history": history,
        "source_manifest": manifest,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_release": checkpoint_release,
    }
    formal_root = _plain_directory(
        str(formal_output.get("path", "")), label="formal Stage-A output"
    )
    formal_root_info = formal_root.lstat()
    if (
        stat.S_IMODE(formal_root_info.st_mode) != 0o555
        or formal_root_info.st_uid != os.getuid()
        or any(snapshot.path.parent != formal_root for snapshot in bound.values())
    ):
        raise SAICSourceAnchorDiagnosticError(
            "formal Stage-A bundle is not one immutable output namespace"
        )
    rows = []
    for name, snapshot in bound.items():
        row = _validate_snapshot_row(bundle.get(name), snapshot, label=f"formal {name}")
        rows.append({"name": name, **dict(row)})
        info = snapshot.path.lstat()
        if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444:
            raise SAICSourceAnchorDiagnosticError(
                f"formal {name} is not an immutable single-link release file"
            )
    rows.sort(key=lambda item: item["name"])
    if bundle.get("files_digest") != object_sha256(rows):
        raise SAICSourceAnchorDiagnosticError("formal Stage-A bundle digest differs")

    history_value = _closed(
        _strict_json(history, label="formal Stage-A update history"),
        FORMAL_HISTORY_FIELDS,
        label="formal Stage-A update history",
    )
    history_unsigned = dict(history_value)
    history_digest = _sha(
        history_unsigned.pop("history_digest", None),
        bits=256,
        label="formal Stage-A history digest",
    )
    updates = list(range(anchor_trainer.FORMAL_UPDATES))
    history_rows = history_value.get("rows")
    if (
        object_sha256(history_unsigned) != history_digest
        or history_value.get("schema_version") != FORMAL_HISTORY_SCHEMA
        or history_value.get("complete") is not True
        or history_value.get("optimizer_updates") != anchor_trainer.FORMAL_UPDATES
        or history_value.get("update_indices") != updates
        or not isinstance(history_rows, list)
        or len(history_rows) != anchor_trainer.FORMAL_UPDATES
        or [row.get("update_index") for row in history_rows if isinstance(row, Mapping)]
        != updates
        or history_row.get("path") != str(history.path)
        or history_row.get("sha256") != history.sha256
        or history_row.get("history_digest") != history_digest
    ):
        raise SAICSourceAnchorDiagnosticError(
            "formal Stage-A exact32 update history differs"
        )
    checkpoint_value = _closed(
        _strict_json(checkpoint_release, label="Stage-A checkpoint release"),
        CHECKPOINT_RELEASE_FIELDS,
        label="Stage-A checkpoint release",
    )
    checkpoint_unsigned = dict(checkpoint_value)
    checkpoint_digest = _sha(
        checkpoint_unsigned.pop("receipt_digest", None),
        bits=256,
        label="Stage-A checkpoint release digest",
    )
    payload_artifacts = checkpoint_value.get("artifacts")
    payload_bound = {
        name: snapshot
        for name, snapshot in bound.items()
        if name != "checkpoint_release"
    }
    if not isinstance(payload_artifacts, Mapping) or set(payload_artifacts) != set(
        payload_bound
    ):
        raise SAICSourceAnchorDiagnosticError(
            "Stage-A checkpoint release payload closure differs"
        )
    payload_rows = []
    for name, snapshot in payload_bound.items():
        row = _validate_snapshot_row(
            payload_artifacts[name], snapshot, label=f"checkpoint release {name}"
        )
        payload_rows.append({"name": name, **dict(row)})
    payload_rows.sort(key=lambda item: item["name"])
    training_receipt_value = _strict_json(
        receipt, label="Stage-A training receipt linkage"
    )
    training_runtime = training_receipt_value.get("runtime")
    if not isinstance(training_runtime, Mapping):
        raise SAICSourceAnchorDiagnosticError(
            "Stage-A training receipt runtime linkage differs"
        )
    if (
        object_sha256(checkpoint_unsigned) != checkpoint_digest
        or checkpoint_value.get("schema_version")
        != "saic-source-anchor-formal32-checkpoint-release-v1"
        or checkpoint_value.get("status") != "FORMAL_GATE_PASS_CHECKPOINT_RELEASED"
        or checkpoint_value.get("complete") is not True
        or checkpoint_value.get("release_path") != str(formal_root)
        or checkpoint_value.get("formal_output_namespace") != str(formal_root)
        or str(checkpoint_value.get("job_id"))
        != str(slurm.get("job_id"))
        or checkpoint_value.get("history_digest") != history_digest
        or checkpoint_value.get("source_release_manifest_sha256")
        != training_runtime.get("release_manifest_sha256")
        or checkpoint_value.get("submission_receipt_sha256")
        != training_runtime.get("submission_receipt_sha256")
        or checkpoint_value.get("trainer_source_sha256")
        != training_receipt_value.get("trainer_source_sha256")
        or checkpoint_value.get("heldout_gate") != heldout_gate
        or checkpoint_value.get("authority") != CHECKPOINT_RELEASE_AUTHORITY
        or checkpoint_value.get("payload_files_digest")
        != object_sha256(payload_rows)
    ):
        raise SAICSourceAnchorDiagnosticError(
            "Stage-A checkpoint release seal/linkage differs"
        )
    return {
        "postflight": dict(postflight.receipt()),
        "postflight_digest": digest,
        "history": dict(history.receipt()),
        "history_digest": history_digest,
        "checkpoint_release": dict(checkpoint_release.receipt()),
        "checkpoint_release_digest": checkpoint_digest,
        "formal_slurm_job_id": str(slurm.get("job_id")),
        "formal_output": str(formal_output.get("path")),
        "bundle_files_digest": str(bundle.get("files_digest")),
        "terminal_postflight_required": True,
        "exact32_history_required": True,
    }


def _resolve_fresh_output(value: str | Path) -> tuple[Path, Path]:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or SAFE_BASENAME.fullmatch(requested.name) is None
    ):
        raise SAICSourceAnchorDiagnosticError(
            "output must be an absolute non-root path with a safe basename"
        )
    parent = _plain_directory(requested.parent, label="output parent")
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise SAICSourceAnchorDiagnosticError("output is create-only")
    staging = parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise SAICSourceAnchorDiagnosticError("staging path is not fresh")
    return output, staging


def _read_action_caption(snapshot: FileSnapshot) -> str:
    try:
        raw = snapshot.path.read_bytes()
        value = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise SAICSourceAnchorDiagnosticError("action caption is not UTF-8") from error
    if value.endswith("\n"):
        value = value[:-1]
    if (
        not value
        or value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or not 16 <= len(value) <= 2048
        or not any(character.isalpha() for character in value)
    ):
        raise SAICSourceAnchorDiagnosticError(
            "action caption must be one canonical natural-language line"
        )
    snapshot.assert_unchanged()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--source-anchor-manifest", required=True)
    parser.add_argument("--expected-source-anchor-manifest-sha256", required=True)
    parser.add_argument("--stage-a-adapter", required=True)
    parser.add_argument("--expected-stage-a-adapter-sha256", required=True)
    parser.add_argument("--stage-a-receipt", required=True)
    parser.add_argument("--expected-stage-a-receipt-sha256", required=True)
    parser.add_argument("--stage-a-formal-postflight", required=True)
    parser.add_argument("--expected-stage-a-formal-postflight-sha256", required=True)
    parser.add_argument("--stage-a-history", required=True)
    parser.add_argument("--expected-stage-a-history-sha256", required=True)
    parser.add_argument("--stage-a-checkpoint-release", required=True)
    parser.add_argument("--expected-stage-a-checkpoint-release-sha256", required=True)
    parser.add_argument("--heldout-row-index", type=int, required=True)
    parser.add_argument("--action-caption-file", required=True)
    parser.add_argument("--expected-action-caption-file-sha256", required=True)
    parser.add_argument("--visual-checkpoint", required=True)
    parser.add_argument("--visual-checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-visual-checkpoint-content-manifest-sha256", required=True
    )
    parser.add_argument("--visual-release-manifest", required=True)
    parser.add_argument("--expected-visual-release-manifest-sha256", required=True)
    parser.add_argument("--visual-evaluator-spec", required=True)
    parser.add_argument("--expected-visual-evaluator-spec-sha256", required=True)
    parser.add_argument("--rendezvous-guard", required=True)
    parser.add_argument("--expected-rendezvous-guard-sha256", required=True)
    parser.add_argument("--rendezvous-evidence-root", required=True)
    parser.add_argument("--expected-rendezvous-id", required=True)
    parser.add_argument("--gpu-visibility-source", required=True)
    parser.add_argument("--expected-gpu-visibility", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--method-source-release-manifest", required=True)
    parser.add_argument("--expected-method-source-release-manifest-sha256", required=True)
    parser.add_argument("--method-source-member-manifest-sha256", required=True)
    parser.add_argument("--method-source-member-manifest-digest", required=True)
    parser.add_argument("--method-source-member-count", type=int, required=True)
    parser.add_argument("--method-source-origin-manifest-sha256", required=True)
    parser.add_argument("--method-source-origin-manifest-digest", required=True)
    parser.add_argument("--method-source-origin-count", type=int, required=True)
    parser.add_argument("--artifact-preflight-only", action="store_true")
    parser.add_argument(
        "--ack-diagnostic-only-no-stage-b-authority", action="store_true"
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_source_anchor_manifest_sha256",
        "expected_stage_a_adapter_sha256",
        "expected_stage_a_receipt_sha256",
        "expected_stage_a_formal_postflight_sha256",
        "expected_stage_a_history_sha256",
        "expected_stage_a_checkpoint_release_sha256",
        "expected_action_caption_file_sha256",
        "expected_visual_checkpoint_content_manifest_sha256",
        "expected_visual_release_manifest_sha256",
        "expected_visual_evaluator_spec_sha256",
        "expected_rendezvous_guard_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_method_source_release_manifest_sha256",
        "method_source_member_manifest_sha256",
        "method_source_member_manifest_digest",
        "method_source_origin_manifest_sha256",
        "method_source_origin_manifest_digest",
    ):
        _sha(getattr(args, name), bits=256, label=name)
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        _sha(getattr(args, name), bits=160, label=name)
    if (
        args.expected_bernini_commit != legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != legacy.trainer.CHECKPOINT_TREE_SHA256
        or args.expected_rendezvous_guard_sha256
        != EXPECTED_RENDEZVOUS_GUARD_SHA256
    ):
        raise SAICSourceAnchorDiagnosticError("base source/checkpoint identity differs")
    if args.gpu_visibility_source not in {
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    }:
        raise SAICSourceAnchorDiagnosticError("Slurm GPU visibility source differs")
    visible = args.expected_gpu_visibility.split(",")
    if (
        len(visible) != WORLD_SIZE
        or len(set(visible)) != WORLD_SIZE
        or any(re.fullmatch(r"[0-9]+", item) is None for item in visible)
        or os.environ.get("ROCR_VISIBLE_DEVICES") != args.expected_gpu_visibility
        or os.environ.get("HIP_VISIBLE_DEVICES") is not None
        or os.environ.get("CUDA_VISIBLE_DEVICES") is not None
        or os.environ.get("GPU_DEVICE_ORDINAL") is not None
    ):
        raise SAICSourceAnchorDiagnosticError(
            "preserved Slurm GPU visibility closure differs"
        )
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise SAICSourceAnchorDiagnosticError("diagnostic requires exact40")
    if (
        type(args.method_source_member_count) is not int
        or args.method_source_member_count <= 0
        or type(args.method_source_origin_count) is not int
        or args.method_source_origin_count <= 0
    ):
        raise SAICSourceAnchorDiagnosticError(
            "source member/origin closure counts differ"
        )
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise SAICSourceAnchorDiagnosticError("seed must lie in [0,2^63)")
    if (
        type(args.heldout_row_index) is not int
        or not 0 <= args.heldout_row_index < anchor_trainer.HOLDOUT_PER_ARM * 2
    ):
        raise SAICSourceAnchorDiagnosticError("heldout row index must lie in [0,16)")
    if args.ack_diagnostic_only_no_stage_b_authority is not True:
        raise SAICSourceAnchorDiagnosticError(
            "diagnostic-only/no-Stage-B-authority acknowledgement is required"
        )
    if args.artifact_preflight_only:
        if args.slurm_job_id is not None:
            raise SAICSourceAnchorDiagnosticError(
                "artifact preflight must not claim a Slurm execution identity"
            )
    elif (
        type(args.slurm_job_id) is not str
        or re.fullmatch(r"[1-9][0-9]*", args.slurm_job_id) is None
        or os.environ.get("SLURM_JOB_ID") != args.slurm_job_id
    ):
        raise SAICSourceAnchorDiagnosticError(
            "full diagnostic requires the exact active Slurm job ID"
        )
    _resolve_fresh_output(args.output)


@dataclass(frozen=True)
class VisualCheckpointBinding:
    root: Path
    manifest: FileSnapshot
    files: tuple[FileSnapshot, ...]
    processor: Any
    evidence: Mapping[str, Any]
    release_files: tuple[FileSnapshot, ...] = ()

    def assert_unchanged(self) -> None:
        self.manifest.assert_unchanged()
        for snapshot in self.files:
            snapshot.assert_unchanged()
        for snapshot in self.release_files:
            snapshot.assert_unchanged()
        expected = {snapshot.path.relative_to(self.root).as_posix() for snapshot in self.files}
        actual: set[str] = set()
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            row = path.lstat()
            if stat.S_ISLNK(row.st_mode) or (
                not stat.S_ISREG(row.st_mode) and not stat.S_ISDIR(row.st_mode)
            ):
                raise SAICSourceAnchorDiagnosticError(
                    "visual checkpoint closure changed after validation"
                )
            if stat.S_ISREG(row.st_mode):
                actual.add(relative.as_posix())
        if actual != expected:
            raise SAICSourceAnchorDiagnosticError(
                "visual checkpoint file set changed after validation"
            )


def validate_visual_checkpoint(
    root_value: str | Path,
    manifest_value: str | Path,
    *,
    expected_manifest_sha256: str,
) -> VisualCheckpointBinding:
    """Bind every frozen DINOv2 file and the exact slow image processor."""

    root = _plain_directory(root_value, label="visual checkpoint")
    manifest = FileSnapshot.capture(
        manifest_value,
        expected_manifest_sha256,
        label="visual checkpoint content manifest",
    )
    try:
        lines = manifest.path.read_text("utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SAICSourceAnchorDiagnosticError(
            "cannot read visual checkpoint manifest"
        ) from error
    if not lines:
        raise SAICSourceAnchorDiagnosticError("visual checkpoint manifest is empty")
    expected: dict[str, str] = {}
    for line in lines:
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise SAICSourceAnchorDiagnosticError(
                "visual checkpoint manifest syntax differs"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SAICSourceAnchorDiagnosticError(
                "visual checkpoint manifest path escaped"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise SAICSourceAnchorDiagnosticError(
                "visual checkpoint manifest path is empty or repeated"
            )
        expected[normalized] = digest
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        row = path.lstat()
        if stat.S_ISLNK(row.st_mode):
            raise SAICSourceAnchorDiagnosticError(
                "visual checkpoint contains a non-cache symlink"
            )
        if stat.S_ISREG(row.st_mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(row.st_mode):
            raise SAICSourceAnchorDiagnosticError(
                "visual checkpoint contains a non-regular entry"
            )
    if actual != set(expected):
        raise SAICSourceAnchorDiagnosticError(
            "visual checkpoint file closure differs from manifest"
        )
    verified: list[Mapping[str, str]] = []
    file_snapshots: list[FileSnapshot] = []
    for relative in sorted(expected):
        snapshot = FileSnapshot.capture(
            root / relative,
            expected[relative],
            label=f"visual checkpoint {relative}",
        )
        file_snapshots.append(snapshot)
        verified.append({"path": relative, "sha256": snapshot.sha256})
    snapshots_by_relative = {
        snapshot.path.relative_to(root).as_posix(): snapshot
        for snapshot in file_snapshots
    }
    config_snapshot = snapshots_by_relative["config.json"]
    preprocessor_snapshot = snapshots_by_relative["preprocessor_config.json"]
    config = _strict_json(
        config_snapshot,
        label="visual checkpoint config",
        canonical_required=False,
    )
    if (
        config.get("model_type") not in visual_contract.SUPPORTED_ARCHITECTURES
        or config.get("image_size") != visual_contract.MODEL_NATIVE_IMAGE_SIZE
        or config.get("patch_size") != visual_contract.MODEL_PATCH_SIZE
        or config.get("num_register_tokens", 0) != 0
    ):
        raise SAICSourceAnchorDiagnosticError(
            "visual checkpoint is not the registered DINOv2 geometry"
        )
    try:
        processor_evidence = visual_scorer.inspect_official_processor(root)
    except Exception as error:
        raise SAICSourceAnchorDiagnosticError(
            f"visual checkpoint processor audit failed: {error}"
        ) from error
    processor = processor_evidence.pop("processor")
    versions = visual_scorer.runtime_versions()
    if versions.get("transformers_version") != "4.53.2":
        raise SAICSourceAnchorDiagnosticError(
            "DINOv2 proxy requires the audited Transformers 4.53.2 runtime"
        )
    evidence = {
        "adapter_id": visual_contract.MODEL_ADAPTER_ID,
        "architecture_id": config["model_type"],
        "root": str(root),
        "checkpoint_manifest": dict(manifest.receipt()),
        "checkpoint_config_sha256": config_snapshot.sha256,
        "preprocessor_config_sha256": preprocessor_snapshot.sha256,
        "checkpoint_file_count": len(verified),
        "verified_entries_digest": object_sha256(verified),
        "every_non_cache_file_sha256_verified": True,
        "exact_all_file_set_no_cache_exclusion": True,
        "num_register_tokens": 0,
        "image_size": visual_contract.MODEL_NATIVE_IMAGE_SIZE,
        "patch_size": visual_contract.MODEL_PATCH_SIZE,
        **processor_evidence,
        "runtime_versions": versions,
        "identity_authority": False,
        "measurement_label": "frozen_dinov2_identity_appearance_proxy_only",
    }
    binding = VisualCheckpointBinding(
        root, manifest, tuple(file_snapshots), processor, evidence
    )
    binding.assert_unchanged()
    return binding


def validate_visual_release(
    root_value: str | Path,
    manifest_value: str | Path,
    *,
    expected_manifest_sha256: str,
    release_manifest_value: str | Path,
    expected_release_manifest_sha256: str,
    evaluator_spec_value: str | Path,
    expected_evaluator_spec_sha256: str,
) -> VisualCheckpointBinding:
    """Bind DINO checkpoint, evaluator implementation, runtime, and golden I/O."""

    binding = validate_visual_checkpoint(
        root_value,
        manifest_value,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    release = FileSnapshot.capture(
        release_manifest_value,
        expected_release_manifest_sha256,
        label="visual release manifest",
    )
    evaluator_spec = FileSnapshot.capture(
        evaluator_spec_value,
        expected_evaluator_spec_sha256,
        label="visual evaluator spec",
    )
    value = _closed(
        _strict_json(release, label="visual release manifest"),
        VISUAL_RELEASE_FIELDS,
        label="visual release manifest",
    )
    unsigned = dict(value)
    digest = _sha(
        unsigned.pop("receipt_digest", None),
        bits=256,
        label="visual release digest",
    )
    if object_sha256(unsigned) != digest:
        raise SAICSourceAnchorDiagnosticError("visual release seal differs")
    checkpoint = value.get("checkpoint")
    evaluator = value.get("evaluator")
    golden = value.get("golden_preprocessor")
    if not all(isinstance(item, Mapping) for item in (checkpoint, evaluator, golden)):
        raise SAICSourceAnchorDiagnosticError("visual release nested contract differs")
    _closed(
        checkpoint,
        {
            "root",
            "content_manifest_path",
            "content_manifest_sha256",
            "tree_digest",
            "file_count",
            "adapter_id",
            "architecture_id",
        },
        label="visual release checkpoint",
    )
    _closed(
        evaluator,
        {"spec_path", "spec_sha256", "spec_digest", "sources", "sources_digest"},
        label="visual release evaluator",
    )
    _closed(
        golden,
        {"input_sha256", "output_sha256", "output_shape"},
        label="visual release golden preprocessor",
    )
    spec = _closed(
        _strict_json(evaluator_spec, label="visual evaluator spec"),
        VISUAL_EVALUATOR_SPEC_FIELDS,
        label="visual evaluator spec",
    )
    spec_unsigned = dict(spec)
    spec_digest = _sha(
        spec_unsigned.pop("spec_digest", None),
        bits=256,
        label="visual evaluator spec digest",
    )
    if object_sha256(spec_unsigned) != spec_digest:
        raise SAICSourceAnchorDiagnosticError("visual evaluator spec seal differs")
    source_rows = evaluator.get("sources")
    if not isinstance(source_rows, list) or len(source_rows) != 2:
        raise SAICSourceAnchorDiagnosticError("visual evaluator source closure differs")
    expected_sources = {
        "pair_v5_source_bound_preservation_evaluator_v1.py": (
            METHOD_ROOT / "pair_v5_source_bound_preservation_evaluator_v1.py"
        ),
        "score_pair_v5_source_bound_preservation_v1.py": (
            METHOD_ROOT / "score_pair_v5_source_bound_preservation_v1.py"
        ),
    }
    observed_sources = []
    source_snapshots = []
    seen_sources: set[str] = set()
    for row in source_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "sha256",
        }:
            raise SAICSourceAnchorDiagnosticError(
                "visual evaluator source row differs"
            )
        relative = row.get("relative_path")
        if relative not in expected_sources or relative in seen_sources:
            raise SAICSourceAnchorDiagnosticError(
                "visual evaluator source path differs"
            )
        seen_sources.add(str(relative))
        snapshot = FileSnapshot.capture(
            expected_sources[str(relative)],
            str(row.get("sha256")),
            label=f"visual evaluator source {relative}",
        )
        source_snapshots.append(snapshot)
        observed_sources.append(
            {"relative_path": str(relative), "sha256": snapshot.sha256}
        )
    observed_sources.sort(key=lambda item: item["relative_path"])
    processor = binding.evidence
    spec_golden = spec.get("golden_preprocessor")
    if (
        value.get("schema_version") != VISUAL_RELEASE_SCHEMA
        or value.get("status") != "sealed_diagnostic_proxy_release"
        or value.get("authority") != VISUAL_RELEASE_AUTHORITY
        or checkpoint.get("root") != str(binding.root)
        or checkpoint.get("content_manifest_path") != str(binding.manifest.path)
        or checkpoint.get("content_manifest_sha256") != binding.manifest.sha256
        or checkpoint.get("tree_digest") != processor.get("verified_entries_digest")
        or checkpoint.get("file_count") != processor.get("checkpoint_file_count")
        or checkpoint.get("adapter_id") != visual_contract.MODEL_ADAPTER_ID
        or checkpoint.get("architecture_id") != processor.get("architecture_id")
        or evaluator.get("spec_path") != str(evaluator_spec.path)
        or evaluator.get("spec_sha256") != evaluator_spec.sha256
        or evaluator.get("spec_digest") != spec_digest
        or evaluator.get("sources_digest") != object_sha256(observed_sources)
        or evaluator.get("sources") != observed_sources
        or value.get("runtime_versions") != processor.get("runtime_versions")
        or golden.get("input_sha256")
        != processor.get("preprocessor_golden_input_sha256")
        or golden.get("output_sha256")
        != processor.get("preprocessor_golden_output_sha256")
        or golden.get("output_shape")
        != processor.get("preprocessor_golden_output_shape")
        or spec.get("schema_version") != VISUAL_EVALUATOR_SPEC_SCHEMA
        or spec.get("adapter_id") != visual_contract.MODEL_ADAPTER_ID
        or spec.get("architecture_id") != processor.get("architecture_id")
        or spec.get("checkpoint_manifest_sha256") != binding.manifest.sha256
        or spec.get("checkpoint_tree_digest")
        != processor.get("verified_entries_digest")
        or spec.get("checkpoint_file_count")
        != processor.get("checkpoint_file_count")
        or spec.get("evaluator_sources") != observed_sources
        or spec.get("evaluator_sources_digest")
        != object_sha256(observed_sources)
        or spec.get("runtime_versions") != processor.get("runtime_versions")
        or spec_golden != golden
        or spec.get("authority") != VISUAL_RELEASE_AUTHORITY
    ):
        raise SAICSourceAnchorDiagnosticError(
            "visual model/tree/evaluator/runtime/golden release differs"
        )
    evidence = {
        **dict(binding.evidence),
        "visual_release_manifest": dict(release.receipt()),
        "visual_release_digest": digest,
        "evaluator_spec": dict(evaluator_spec.receipt()),
        "evaluator_spec_digest": spec_digest,
        "evaluator_sources": observed_sources,
        "evaluator_sources_digest": object_sha256(observed_sources),
        "golden_preprocessor_exact": True,
    }
    result = VisualCheckpointBinding(
        binding.root,
        binding.manifest,
        binding.files,
        binding.processor,
        evidence,
        (release, evaluator_spec, *source_snapshots),
    )
    result.assert_unchanged()
    return result


def _sampling_contract(*, seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract("rv2v", steps=NUM_INFERENCE_STEPS, seed=seed)
    if value.get("guidance_mode") != "rv2v":
        raise SAICSourceAnchorDiagnosticError("native RV2V sampling contract changed")
    result = dict(value)
    result["guidance_mode"] = anchor_runtime.EXPECTED_GUIDANCE_MODE
    if (
        result.get("omega_img") != 4.5
        or result.get("omega_vid") != 1.25
        or result.get("omega_txt") != 4.0
        or result.get("flow_shift") != 5.0
        or result.get("num_frames") != FRAME_COUNT
        or result.get("num_inference_steps") != NUM_INFERENCE_STEPS
    ):
        raise SAICSourceAnchorDiagnosticError("VI v2v_apg sampling contract differs")
    return result


def _broadcast_tensor(value: Any, *, source_rank: int = 0) -> None:
    import torch.distributed as dist

    dist.broadcast(value, src=source_rank)


def _broadcast_prompt(value: Any) -> Any:
    _broadcast_tensor(value)
    return value


def _load_rendezvous_guard(snapshot: FileSnapshot) -> Any:
    raw = snapshot.path.read_bytes()
    snapshot.assert_unchanged()
    module = types.ModuleType("sealed_saic_source_anchor_rendezvous_guard_v2")
    try:
        exec(compile(raw, str(snapshot.path), "exec"), module.__dict__)
    except Exception as error:
        raise SAICSourceAnchorDiagnosticError(
            f"cannot load sealed rendezvous guard: {error}"
        ) from error
    required = (
        "write_create_only",
        "load_sealed",
        "seal",
        "object_sha256",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        raise SAICSourceAnchorDiagnosticError("rendezvous guard interface differs")
    return module


def _admit_dynamic_world4_rendezvous(
    *,
    args: argparse.Namespace,
    distributed: Any,
    dist: Any,
) -> Mapping[str, Any]:
    """Seal rank/GPU/endpoint evidence before any model or checkpoint load."""

    import torch

    guard_snapshot = FileSnapshot.capture(
        args.rendezvous_guard,
        args.expected_rendezvous_guard_sha256,
        label="retained rendezvous guard",
    )
    guard = _load_rendezvous_guard(guard_snapshot)
    root = _plain_directory(args.rendezvous_evidence_root, label="rendezvous evidence root")
    root_info = root.lstat()
    if (
        stat.S_IMODE(root_info.st_mode) != 0o700
        or root_info.st_uid != os.getuid()
        or root.parent != Path(args.output).parent
        or root.name != f"{Path(args.output).name}.rendezvous"
    ):
        raise SAICSourceAnchorDiagnosticError(
            "rendezvous evidence namespace differs"
        )
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    visible = args.expected_gpu_visibility.split(",")
    if (
        master_addr != "127.0.0.1"
        or type(master_port) is not str
        or re.fullmatch(r"[0-9]+", master_port) is None
        or not 1024 <= int(master_port) <= 65535
        or run_id != args.expected_rendezvous_id
        or local_world_size != str(WORLD_SIZE)
        or not 0 <= distributed.local_rank < WORLD_SIZE
        or distributed.rank != distributed.local_rank
        or torch.cuda.device_count() != WORLD_SIZE
        or torch.cuda.current_device() != distributed.local_rank
    ):
        raise SAICSourceAnchorDiagnosticError(
            "kernel-selected WORLD4 rendezvous environment differs"
        )
    dist.barrier()
    if distributed.rank == 0 and list(root.iterdir()):
        raise SAICSourceAnchorDiagnosticError(
            "rendezvous evidence root is not create-only"
        )
    dist.barrier()
    rank_body = {
        "schema_version": RENDEZVOUS_RANK_SCHEMA,
        "status": "rank_admitted_before_model_load",
        "slurm_job_id": args.slurm_job_id,
        "rdzv_backend": "c10d",
        "rdzv_endpoint_request": "127.0.0.1:0",
        "rdzv_id": args.expected_rendezvous_id,
        "actual_master_addr": master_addr,
        "actual_master_port": int(master_port),
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "world_size": distributed.world_size,
        "local_world_size": WORLD_SIZE,
        "gpu_visibility_source": args.gpu_visibility_source,
        "gpu_visibility": args.expected_gpu_visibility,
        "physical_gpu_token": visible[distributed.local_rank],
        "logical_cuda_device": distributed.local_rank,
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_current_device": torch.cuda.current_device(),
        "model_loaded": False,
        "checkpoint_loaded": False,
        "generation_entered": False,
        "rendezvous_guard_sha256": guard_snapshot.sha256,
        "authority": RENDEZVOUS_AUTHORITY,
    }
    rank_packet = guard.seal(rank_body)
    guard.write_create_only(root / f"rank-{distributed.rank}.json", rank_packet)
    dist.barrier()
    decision_path = root / "admission.json"
    if distributed.rank == 0:
        packets = []
        exact_fields = set(rank_body) | {"receipt_digest"}
        for rank in range(WORLD_SIZE):
            packet = guard.load_sealed(
                root / f"rank-{rank}.json",
                schema_version=RENDEZVOUS_RANK_SCHEMA,
                exact_fields=exact_fields,
            )
            expected = {
                **rank_body,
                "rank": rank,
                "local_rank": rank,
                "physical_gpu_token": visible[rank],
                "logical_cuda_device": rank,
                "torch_cuda_device_count": WORLD_SIZE,
                "torch_cuda_current_device": rank,
            }
            expected = guard.seal(expected)
            if packet != expected:
                raise SAICSourceAnchorDiagnosticError(
                    f"rendezvous rank {rank} packet differs"
                )
            packets.append(packet)
        decision_body = {
            "schema_version": RENDEZVOUS_DECISION_SCHEMA,
            "status": "exact_world4_dynamic_rendezvous_admitted_before_model_load",
            "slurm_job_id": args.slurm_job_id,
            "rdzv_backend": "c10d",
            "rdzv_endpoint_request": "127.0.0.1:0",
            "rdzv_id": args.expected_rendezvous_id,
            "actual_master_addr": master_addr,
            "actual_master_port": int(master_port),
            "world_size": WORLD_SIZE,
            "rank_order": list(range(WORLD_SIZE)),
            "rank_packet_digests": [packet["receipt_digest"] for packet in packets],
            "gpu_visibility_source": args.gpu_visibility_source,
            "gpu_visibility": args.expected_gpu_visibility,
            "physical_gpu_tokens": visible,
            "logical_cuda_devices": list(range(WORLD_SIZE)),
            "torch_cuda_device_count": WORLD_SIZE,
            "torch_cuda_current_devices": list(range(WORLD_SIZE)),
            "all_four_ranks_admitted": True,
            "all_four_gpu_mappings_distinct": True,
            "kernel_selected_port": True,
            "numeric_port_preregistered": False,
            "model_load_authorized": True,
            "scientific_authority": False,
            "rendezvous_guard_sha256": guard_snapshot.sha256,
            "authority": RENDEZVOUS_AUTHORITY,
        }
        guard.write_create_only(decision_path, guard.seal(decision_body))
    dist.barrier()
    decision = guard.load_sealed(
        decision_path,
        schema_version=RENDEZVOUS_DECISION_SCHEMA,
        exact_fields={
            "schema_version",
            "status",
            "slurm_job_id",
            "rdzv_backend",
            "rdzv_endpoint_request",
            "rdzv_id",
            "actual_master_addr",
            "actual_master_port",
            "world_size",
            "rank_order",
            "rank_packet_digests",
            "gpu_visibility_source",
            "gpu_visibility",
            "physical_gpu_tokens",
            "logical_cuda_devices",
            "torch_cuda_device_count",
            "torch_cuda_current_devices",
            "all_four_ranks_admitted",
            "all_four_gpu_mappings_distinct",
            "kernel_selected_port",
            "numeric_port_preregistered",
            "model_load_authorized",
            "scientific_authority",
            "rendezvous_guard_sha256",
            "authority",
            "receipt_digest",
        },
    )
    if (
        decision.get("status")
        != "exact_world4_dynamic_rendezvous_admitted_before_model_load"
        or decision.get("slurm_job_id") != args.slurm_job_id
        or decision.get("rdzv_id") != args.expected_rendezvous_id
        or decision.get("actual_master_port") != int(master_port)
        or decision.get("rank_order") != list(range(WORLD_SIZE))
        or decision.get("physical_gpu_tokens") != visible
        or decision.get("logical_cuda_devices") != list(range(WORLD_SIZE))
        or decision.get("torch_cuda_device_count") != WORLD_SIZE
        or decision.get("torch_cuda_current_devices") != list(range(WORLD_SIZE))
        or decision.get("all_four_ranks_admitted") is not True
        or decision.get("all_four_gpu_mappings_distinct") is not True
        or decision.get("kernel_selected_port") is not True
        or decision.get("numeric_port_preregistered") is not False
        or decision.get("model_load_authorized") is not True
        or decision.get("scientific_authority") is not False
        or decision.get("rendezvous_guard_sha256") != guard_snapshot.sha256
        or decision.get("authority") != RENDEZVOUS_AUTHORITY
    ):
        raise SAICSourceAnchorDiagnosticError(
            "terminal dynamic rendezvous admission differs"
        )
    guard_snapshot.assert_unchanged()
    return {
        "root": str(root),
        "guard": dict(guard_snapshot.receipt()),
        "decision_path": str(decision_path),
        "decision_sha256": _stable_file_sha256(
            decision_path, label="rendezvous decision"
        ),
        "decision_digest": decision["receipt_digest"],
        "rank_packet_digests": list(decision["rank_packet_digests"]),
        "rdzv_id": args.expected_rendezvous_id,
        "actual_master_addr": master_addr,
        "actual_master_port": int(master_port),
        "gpu_visibility_source": args.gpu_visibility_source,
        "gpu_visibility": args.expected_gpu_visibility,
        "physical_gpu_tokens": visible,
        "admitted_before_model_load": True,
    }


def _zero_initial_adapter_certificate(
    handle: anchor_adapter.SAICSourceAnchorHandle,
) -> Mapping[str, Any]:
    import torch

    rows = tuple(handle.trainable_named_parameters())
    outputs = [(name, value) for name, value in rows if name.endswith("output_up.weight")]
    if (
        len(rows) != len(anchor_adapter.SOURCE_ANCHOR_BLOCK_INDICES) * 4
        or len(outputs) != len(anchor_adapter.SOURCE_ANCHOR_BLOCK_INDICES) * 2
        or any(int(torch.count_nonzero(value.detach()).item()) != 0 for _, value in outputs)
    ):
        raise SAICSourceAnchorDiagnosticError(
            "fresh source-anchor output maps are not exact zero"
        )
    body = {
        "adapter_contract_digest": handle.receipt()["digest"],
        "trainable_key_count": len(rows),
        "trainable_key_sha256": object_sha256(sorted(name for name, _ in rows)),
        "output_up_key_count": len(outputs),
        "output_up_key_sha256": object_sha256(sorted(name for name, _ in outputs)),
        "every_output_up_element_exact_zero": True,
        "residual_function_exact_zero_before_checkpoint_load": True,
        "frozen_base_label_valid_before_load_only": True,
    }
    return {**body, "digest": object_sha256(body)}


def _condition_identity(
    *, source: Any, references: Mapping[int, Any], label: str
) -> Mapping[str, Any]:
    return {
        "source_video": native._all_rank_tensor_identity(
            source, label=f"{label}_source", world_size=WORLD_SIZE
        ),
        "references": {
            str(index): native._all_rank_tensor_identity(
                references[index],
                label=f"{label}_reference_{index}",
                world_size=WORLD_SIZE,
            )
            for index in REFERENCE_INDICES
        },
    }


def _run_cell(
    *,
    cell: str,
    diffusion: Any,
    handle: anchor_adapter.SAICSourceAnchorHandle,
    prompt: Any,
    negative: Any,
    source: Any,
    references: Mapping[int, Any],
    latent_shape: tuple[int, int, int, int, int],
    bucket_hw: tuple[int, int],
    device: Any,
    seed: int,
    wan_diffusion_module: Any,
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    import torch

    if cell not in CELL_ORDER:
        raise SAICSourceAnchorDiagnosticError(f"unknown diagnostic cell: {cell}")
    if anchor_adapter.active_route() is not None:
        raise SAICSourceAnchorDiagnosticError("source-anchor route leaked between cells")
    sample_kwargs = {
        "prompt_embeds": prompt,
        "uncond_prompt_embeds": negative,
        "image_vae_latents": None,
        "multi_video_vae_latents": [source],
        "multi_image_vae_latents": [references[index] for index in REFERENCE_INDICES],
        "width": bucket_hw[1],
        "height": bucket_hw[0],
        "device": device,
        **_sampling_contract(seed=seed),
    }
    runtime_receipt: Mapping[str, Any]
    if cell in ROUTED_CELLS:
        with anchor_runtime.saic_source_anchor_native_runtime(
            diffusion,
            handle=handle,
            config=anchor_runtime.SourceAnchorNativeRuntimeConfig(
                target_latent_shape=latent_shape,
                branch_name="VI",
            ),
        ) as patch:
            result, capture = native._sample_with_native_initial_noise_observer(
                sample_fn=lambda: diffusion.sample(**sample_kwargs),
                wan_diffusion_module=wan_diffusion_module,
                expected_shape=latent_shape,
                expected_device=device,
                expected_seed=seed,
            )
        runtime_receipt = dict(patch.finalize())
    else:
        if cell != "anchor_route_drop_noop":
            raise SAICSourceAnchorDiagnosticError("only route-drop may omit runtime route")
        result, capture = native._sample_with_native_initial_noise_observer(
            sample_fn=lambda: diffusion.sample(**sample_kwargs),
            wan_diffusion_module=wan_diffusion_module,
            expected_shape=latent_shape,
            expected_device=device,
            expected_seed=seed,
        )
        runtime_receipt = {
            "schema_version": SCHEMA_VERSION,
            "cell": cell,
            "source_anchor_runtime_route_installed": False,
            "complete_correct_source_vi_condition_retained": True,
            "trained_adapter_parameters_present_but_inactive_without_route": True,
            "exact40_certificate_inherited_only_after_byte_identity_to_base": True,
            "semantic_or_scientific_authority": False,
        }
    if (
        type(result) is not torch.Tensor
        or tuple(result.shape) != latent_shape
        or result.dtype != torch.float32
        or result.requires_grad
        or result.grad_fn is not None
        or not bool(torch.isfinite(result).all().item())
        or anchor_adapter.active_route() is not None
    ):
        raise SAICSourceAnchorDiagnosticError(f"{cell} sampler result differs")
    stored = result.detach().to(device="cpu", dtype=torch.float32).contiguous()
    result_identity = native._all_rank_tensor_identity(
        stored, label=f"{cell}_generated", world_size=WORLD_SIZE
    )
    noise_identity = native._all_rank_tensor_identity(
        capture.tensor, label=f"{cell}_official_gaussian", world_size=WORLD_SIZE
    )
    condition = _condition_identity(
        source=source, references=references, label=f"{cell}_condition"
    )
    return stored, capture, runtime_receipt, result_identity, {
        "condition": condition,
        "noise": noise_identity,
    }


def _visual_model_evidence(
    binding: VisualCheckpointBinding,
    *,
    model: Any,
    loading_counts: Mapping[str, int],
) -> Mapping[str, Any]:
    rows = list(model.named_parameters())
    metadata = [
        {
            "name": name,
            "shape": [int(item) for item in parameter.shape],
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in rows
    ]
    if model.training or any(parameter.requires_grad for _, parameter in rows):
        raise SAICSourceAnchorDiagnosticError("visual proxy model is not frozen/eval")
    return {
        **dict(binding.evidence),
        "trainable_parameter_tensors": 0,
        "parameter_tensor_count": len(rows),
        "parameter_element_count": sum(int(value.numel()) for _, value in rows),
        "parameter_metadata_digest": object_sha256(metadata),
        **dict(loading_counts),
    }


def build_decoded_diagnostics(
    *,
    correct_source: FileSnapshot,
    wrong_source: FileSnapshot,
    outputs: Mapping[str, Mapping[str, Any]],
    visual_binding: VisualCheckpointBinding,
    device: Any,
    staging_root: Path,
    final_root: Path,
) -> Mapping[str, Any]:
    """Measure all decoded cells; every returned authority bit is false."""

    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint_for_loader = {
        "root": visual_binding.root,
        "architecture_id": visual_binding.evidence["architecture_id"],
    }
    try:
        visual_model, loading_counts = visual_scorer.load_frozen_model(
            checkpoint_for_loader, device=device
        )
    except Exception as error:
        raise SAICSourceAnchorDiagnosticError(
            f"cannot load frozen DINOv2 proxy: {error}"
        ) from error
    model_evidence = _visual_model_evidence(
        visual_binding, model=visual_model, loading_counts=loading_counts
    )

    def decode_features(snapshot: FileSnapshot) -> tuple[Any, Any, Any, Mapping[str, Any]]:
        frames, decode = visual_scorer.decode_exact81_rgb(
            snapshot.path, expected_sha256=snapshot.sha256
        )
        raw, normalized = visual_scorer.preprocess_selected_rgb(
            frames, visual_binding.processor
        )
        global_feature, dense_feature, feature_evidence = visual_scorer.extract_features(
            visual_model,
            normalized,
            device=device,
            num_register_tokens=0,
        )
        return raw, global_feature, dense_feature, {
            "decode": decode,
            "features": feature_evidence,
        }

    correct_raw, correct_global, correct_dense, correct_evidence = decode_features(
        correct_source
    )
    wrong_raw, wrong_global, wrong_dense, wrong_evidence = decode_features(wrong_source)
    del wrong_raw
    cells: dict[str, Any] = {}
    for cell in CELL_ORDER:
        row = outputs.get(cell)
        if not isinstance(row, Mapping):
            raise SAICSourceAnchorDiagnosticError(f"missing decoded output for {cell}")
        candidate = FileSnapshot.capture(
            row["path"], row["sha256"], label=f"{cell} MP4"
        )
        candidate_raw, candidate_global, candidate_dense, feature_evidence = (
            decode_features(candidate)
        )
        metrics = visual_scorer.compute_metrics(
            candidate_global=candidate_global,
            candidate_dense=candidate_dense,
            correct_global=correct_global,
            correct_dense=correct_dense,
            wrong_global=wrong_global,
            wrong_dense=wrong_dense,
            candidate_raw=candidate_raw,
            correct_raw=correct_raw,
        )
        media = media_diagnostics.build_diagnostic(
            source_video=correct_source.path,
            expected_source_sha256=correct_source.sha256,
            candidate_video=candidate.path,
            expected_candidate_sha256=candidate.sha256,
        )
        cells[cell] = {
            "candidate": dict(candidate.receipt()),
            "frozen_visual_proxy": {
                "metrics": metrics,
                "evidence": feature_evidence,
                "identity_or_appearance_authority": False,
                "absolute_thresholds_calibrated": False,
            },
            "full81_full80_media_diagnostic": media,
        }
    visual_model.to("cpu")
    del visual_model
    torch.cuda.empty_cache()

    def delta(left: str, right: str, metric: str) -> float:
        value = (
            cells[right]["frozen_visual_proxy"]["metrics"][metric]
            - cells[left]["frozen_visual_proxy"]["metrics"][metric]
        )
        if not math.isfinite(float(value)):
            raise SAICSourceAnchorDiagnosticError("comparative proxy delta is non-finite")
        return float(value)

    def camera(cell: str) -> float:
        value = float(
            cells[cell]["full81_full80_media_diagnostic"]["comparisons"]
            ["camera_trajectory"]["global_mean_xy_l2_difference_mean"]
        )
        if not math.isfinite(value):
            raise SAICSourceAnchorDiagnosticError("camera proxy value is non-finite")
        return value

    def finite_difference(left: float, right: float, *, label: str) -> float:
        value = float(right) - float(left)
        if not math.isfinite(value):
            raise SAICSourceAnchorDiagnosticError(f"{label} is non-finite")
        return value

    comparisons = {
        "noop_anchor_minus_base": {
            "identity_appearance_proxy": delta(
                "base_correct_noop",
                "anchor_correct_noop",
                "source_identity_appearance_proxy",
            ),
            "quality_proxy": delta(
                "base_correct_noop",
                "anchor_correct_noop",
                "decode_video_quality_diagnostic",
            ),
            "camera_error_delta_lower_is_better": finite_difference(
                camera("base_correct_noop"),
                camera("anchor_correct_noop"),
                label="no-op camera delta",
            ),
        },
        "action_anchor_minus_base": {
            "identity_appearance_proxy": delta(
                "base_correct_action",
                "anchor_correct_action",
                "source_identity_appearance_proxy",
            ),
            "quality_proxy": delta(
                "base_correct_action",
                "anchor_correct_action",
                "decode_video_quality_diagnostic",
            ),
            "camera_error_delta_lower_is_better": finite_difference(
                camera("base_correct_action"),
                camera("anchor_correct_action"),
                label="action camera delta",
            ),
            "semantic_action_observer_available": False,
            "semantic_action_nonregression_verdict": "unavailable",
        },
        "correct_vs_wrong_noop_identity_proxy": finite_difference(
            cells["anchor_wrong_noop"]["frozen_visual_proxy"]["metrics"]
            ["source_identity_appearance_proxy"],
            cells["anchor_correct_noop"]["frozen_visual_proxy"]["metrics"]
            ["source_identity_appearance_proxy"],
            label="correct-vs-wrong identity proxy",
        ),
        "correct_vs_zero_condition_noop_identity_proxy": finite_difference(
            cells["anchor_zero_condition_noop"]["frozen_visual_proxy"]["metrics"]
            ["source_identity_appearance_proxy"],
            cells["anchor_correct_noop"]["frozen_visual_proxy"]["metrics"]
            ["source_identity_appearance_proxy"],
            label="correct-vs-zero identity proxy",
        ),
        "route_drop_is_a_mechanism_control_not_a_dropped_source_claim": True,
        "zero_condition_is_synthetic_ood_not_a_real_source_claim": True,
    }
    body = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "cell_order": list(CELL_ORDER),
        "correct_source": dict(correct_source.receipt()),
        "wrong_source": dict(wrong_source.receipt()),
        "visual_model": model_evidence,
        "source_visual_evidence": {
            "correct": correct_evidence,
            "wrong": wrong_evidence,
        },
        "cells": cells,
        "comparisons": comparisons,
        "availability": {
            "exact81_decode": "available",
            "full80_camera": "diagnostic_only",
            "technical_quality": "diagnostic_only",
            "dinov2_identity_appearance": "proxy_only",
            "semantic_action_event": "unavailable",
            "background_semantics": "unavailable",
            "non_target_semantics": "unavailable",
            "inverse_cycle": "unavailable",
        },
        "authority": {
            "measurement_runtime_qualified_for_scientific_gate": False,
            "identity_authority": False,
            "semantic_action_authority": False,
            "candidate_selection_allowed": False,
            "training_allowed": False,
            "optimizer_step_allowed": False,
            "stage_b_allowed": False,
            "checkpoint_allowed": False,
            "publication_allowed": False,
            "production_allowed": False,
            "scientific_success_claimed": False,
        },
    }
    visual_binding.assert_unchanged()
    rebased = _rebase_paths(body, staging_root=staging_root, final_root=final_root)
    return {**rebased, "diagnostic_digest": object_sha256(rebased)}


def _rebase_paths(value: Any, *, staging_root: Path, final_root: Path) -> Any:
    if isinstance(value, Mapping):
        result = {
            key: _rebase_paths(item, staging_root=staging_root, final_root=final_root)
            for key, item in value.items()
        }
        if result.get("schema_version") == media_diagnostics.SCHEMA_VERSION:
            body = {key: item for key, item in result.items() if key != "diagnostic_digest"}
            result = {**body, "diagnostic_digest": media_diagnostics.object_sha256(body)}
        return result
    if isinstance(value, list):
        return [
            _rebase_paths(item, staging_root=staging_root, final_root=final_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _rebase_paths(item, staging_root=staging_root, final_root=final_root)
            for item in value
        ]
    if type(value) is str:
        prefix = str(staging_root)
        if value == prefix:
            return str(final_root)
        if value.startswith(prefix + os.sep):
            return str(final_root) + value[len(prefix) :]
    return value


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise SAICSourceAnchorDiagnosticError("JSON output is not create-only")
    payload = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                raise SAICSourceAnchorDiagnosticError("JSON write stalled")
            offset += wrote
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    return _stable_file_sha256(path, label="published JSON")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_create_only_directory(
    staging: Path, output: Path, *, commit_name: str
) -> None:
    """Publish by exclusive mkdir and hard links, with the receipt linked last."""

    if (
        staging.parent != output.parent
        or output.exists()
        or output.is_symlink()
        or not staging.is_dir()
        or staging.is_symlink()
    ):
        raise SAICSourceAnchorDiagnosticError(
            "create-only diagnostic publication precondition differs"
        )
    staged = {path.name: path for path in staging.iterdir()}
    if commit_name not in staged:
        raise SAICSourceAnchorDiagnosticError(
            "diagnostic publication commit marker is absent"
        )
    try:
        output.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as error:
        raise SAICSourceAnchorDiagnosticError(
            "create-only diagnostic output appeared before publication"
        ) from error
    _fsync_directory(output.parent)
    for name in sorted(set(staged) - {commit_name}):
        os.link(staged[name], output / name, follow_symlinks=False)
    os.link(staged[commit_name], output / commit_name, follow_symlinks=False)
    _fsync_directory(output)
    for path in staged.values():
        path.unlink()
    staging.rmdir()
    output.chmod(0o555)
    _fsync_directory(output.parent)
    published = {path.name: path for path in output.iterdir()}
    if set(published) != set(staged) or any(
        path.is_symlink()
        or not path.is_file()
        or path.lstat().st_nlink != 1
        or stat.S_IMODE(path.lstat().st_mode) != 0o444
        for path in published.values()
    ):
        raise SAICSourceAnchorDiagnosticError(
            "create-only diagnostic publication closure differs"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    output, staging = _resolve_fresh_output(args.output)
    source_release_snapshot = FileSnapshot.capture(
        args.method_source_release_manifest,
        args.expected_method_source_release_manifest_sha256,
        label="method source release manifest",
    )
    checkpoint_manifest = FileSnapshot.capture(
        args.checkpoint_content_manifest,
        args.expected_checkpoint_content_manifest_sha256,
        label="Bernini checkpoint content manifest",
    )
    source_manifest_snapshot = FileSnapshot.capture(
        args.source_anchor_manifest,
        args.expected_source_anchor_manifest_sha256,
        label="source-anchor manifest",
    )
    adapter_snapshot = FileSnapshot.capture(
        args.stage_a_adapter,
        args.expected_stage_a_adapter_sha256,
        label="Stage-A adapter",
    )
    stage_a_receipt_snapshot = FileSnapshot.capture(
        args.stage_a_receipt,
        args.expected_stage_a_receipt_sha256,
        label="Stage-A receipt",
    )
    stage_a_postflight_snapshot = FileSnapshot.capture(
        args.stage_a_formal_postflight,
        args.expected_stage_a_formal_postflight_sha256,
        label="Stage-A formal postflight",
    )
    stage_a_history_snapshot = FileSnapshot.capture(
        args.stage_a_history,
        args.expected_stage_a_history_sha256,
        label="Stage-A formal update history",
    )
    stage_a_checkpoint_release_snapshot = FileSnapshot.capture(
        args.stage_a_checkpoint_release,
        args.expected_stage_a_checkpoint_release_sha256,
        label="Stage-A checkpoint release receipt",
    )
    action_caption_snapshot = FileSnapshot.capture(
        args.action_caption_file,
        args.expected_action_caption_file_sha256,
        label="action caption",
    )
    action_caption = _read_action_caption(action_caption_snapshot)
    stage_a = validate_stage_a_bundle(
        adapter=adapter_snapshot,
        receipt=stage_a_receipt_snapshot,
        manifest=source_manifest_snapshot,
        checkpoint_manifest=checkpoint_manifest,
    )
    formal_stage_a = validate_formal_stage_a_postflight(
        postflight=stage_a_postflight_snapshot,
        history=stage_a_history_snapshot,
        adapter=adapter_snapshot,
        receipt=stage_a_receipt_snapshot,
        manifest=source_manifest_snapshot,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_release=stage_a_checkpoint_release_snapshot,
    )
    manifest = anchor_trainer.load_manifest(
        source_manifest_snapshot.path,
        expected_sha256=source_manifest_snapshot.sha256,
        verify_files=True,
    )
    if manifest.manifest_digest != stage_a["manifest_digest"]:
        raise SAICSourceAnchorDiagnosticError(
            "Stage-A receipt and source manifest digest differ"
        )
    row = manifest.holdout_rows[args.heldout_row_index]
    if (
        row.split != "holdout"
        or row.row_index != args.heldout_row_index
        or row.iid == row.wrong_iid
        or row.bucket_hw
        != next(item for item in manifest.holdout_rows if item.iid == row.wrong_iid).bucket_hw
    ):
        raise SAICSourceAnchorDiagnosticError("held-out correct/wrong row differs")
    correct_source = FileSnapshot.capture(
        row.source_path, row.source_sha256, label="held-out correct source"
    )
    wrong_source = FileSnapshot.capture(
        row.wrong_path, row.wrong_sha256, label="held-out wrong source"
    )
    if (
        correct_source.path == wrong_source.path
        or correct_source.sha256 == wrong_source.sha256
    ):
        raise SAICSourceAnchorDiagnosticError(
            "held-out correct/wrong source bytes are not distinct"
        )

    if args.artifact_preflight_only:
        visual = validate_visual_release(
            args.visual_checkpoint,
            args.visual_checkpoint_content_manifest,
            expected_manifest_sha256=(
                args.expected_visual_checkpoint_content_manifest_sha256
            ),
            release_manifest_value=args.visual_release_manifest,
            expected_release_manifest_sha256=(
                args.expected_visual_release_manifest_sha256
            ),
            evaluator_spec_value=args.visual_evaluator_spec,
            expected_evaluator_spec_sha256=(
                args.expected_visual_evaluator_spec_sha256
            ),
        )
        value = {
            "schema_version": SCHEMA_VERSION,
            "artifact_preflight_only": True,
            "stage_a": dict(stage_a),
            "formal_stage_a": dict(formal_stage_a),
            "method_source_release_manifest": dict(
                source_release_snapshot.receipt()
            ),
            "source_manifest": dict(source_manifest_snapshot.receipt()),
            "heldout_row_index": args.heldout_row_index,
            "correct_source": dict(correct_source.receipt()),
            "wrong_source": dict(wrong_source.receipt()),
            "visual_checkpoint": dict(visual.evidence),
            "cell_order": list(CELL_ORDER),
            "model_loaded": False,
            "sampling_started": False,
            "output_created": False,
            "stage_b_authorized": False,
        }
        print(canonical_json_bytes({**value, "digest": object_sha256(value)}).decode("ascii"))
        return 0

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise SAICSourceAnchorDiagnosticError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise SAICSourceAnchorDiagnosticError(
            "Bernini-R attention heads do not divide Ulysses4"
        )
    inference_file_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if (
        SYSTEM_PROMPTS.get("vr2v") != native.TASK_SYSTEM_PROMPTS["vr2v"]
        or DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT
    ):
        raise SAICSourceAnchorDiagnosticError("VR2V prompt/negative runtime differs")
    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise SAICSourceAnchorDiagnosticError(
            "diagnostic requires AUH WORLD4/Ulysses4 ROCm"
        )
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    rendezvous_admission = _admit_dynamic_world4_rendezvous(
        args=args,
        distributed=distributed,
        dist=dist,
    )

    visual_binding: Optional[VisualCheckpointBinding] = None
    visual_box: list[Any] = [None]
    if distributed.rank == 0:
        try:
            visual_binding = validate_visual_release(
                args.visual_checkpoint,
                args.visual_checkpoint_content_manifest,
                expected_manifest_sha256=(
                    args.expected_visual_checkpoint_content_manifest_sha256
                ),
                release_manifest_value=args.visual_release_manifest,
                expected_release_manifest_sha256=(
                    args.expected_visual_release_manifest_sha256
                ),
                evaluator_spec_value=args.visual_evaluator_spec,
                expected_evaluator_spec_sha256=(
                    args.expected_visual_evaluator_spec_sha256
                ),
            )
            visual_box[0] = {"ok": True, "evidence": dict(visual_binding.evidence)}
        except Exception as error:
            visual_box[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(visual_box, src=0)
    if not isinstance(visual_box[0], Mapping) or visual_box[0].get("ok") is not True:
        raise SAICSourceAnchorDiagnosticError(
            f"rank-zero visual checkpoint validation failed: {visual_box[0]}"
        )

    checkpoint_box: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_box[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint,
                    checkpoint_manifest.path,
                    expected_manifest_sha256=checkpoint_manifest.sha256,
                ),
            }
        except Exception as error:
            checkpoint_box[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_box, src=0)
    if (
        not isinstance(checkpoint_box[0], Mapping)
        or checkpoint_box[0].get("ok") is not True
    ):
        raise SAICSourceAnchorDiagnosticError(
            f"rank-zero Bernini checkpoint validation failed: {checkpoint_box[0]}"
        )
    checkpoint_identity = dict(checkpoint_box[0]["identity"])

    correct_pixels, correct_metadata, correct_sha = (
        source_audit.prepare_hashed_source_snapshot(correct_source.path)
    )
    wrong_pixels, wrong_metadata, wrong_sha = (
        source_audit.prepare_hashed_source_snapshot(wrong_source.path)
    )
    bucket_hw = tuple(int(item) for item in row.bucket_hw)
    if (
        correct_sha != correct_source.sha256
        or wrong_sha != wrong_source.sha256
        or correct_sha == wrong_sha
        or correct_metadata.get("frame_count") != FRAME_COUNT
        or wrong_metadata.get("frame_count") != FRAME_COUNT
        or tuple(correct_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
        or tuple(wrong_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
    ):
        raise SAICSourceAnchorDiagnosticError(
            "correct/wrong exact81 same-bucket preprocessing differs"
        )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise SAICSourceAnchorDiagnosticError("Bernini renderer is not UniPC shift5")
    model = BerniniRendererModel(config).eval().requires_grad_(False)
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False)
    latent_shape = (
        1,
        int(vae.config.z_dim),
        LATENT_PHASES,
        bucket_hw[0] // 8,
        bucket_hw[1] // 8,
    )
    reference_shape = (1, int(vae.config.z_dim), 1, latent_shape[3], latent_shape[4])
    vae.to(device)
    if distributed.rank == 0:
        correct_device = correct_pixels.to(device=device, dtype=torch.float32)
        wrong_device = wrong_pixels.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            correct_latent = _vae_encode(vae, correct_device).float().contiguous()
            wrong_latent = _vae_encode(vae, wrong_device).float().contiguous()
            correct_references = {
                index: _vae_encode(
                    vae, correct_device[:, :, index : index + 1].contiguous()
                ).float().contiguous()
                for index in REFERENCE_INDICES
            }
            wrong_references = {
                index: _vae_encode(
                    vae, wrong_device[:, :, index : index + 1].contiguous()
                ).float().contiguous()
                for index in REFERENCE_INDICES
            }
        del correct_device, wrong_device
    else:
        correct_latent = torch.empty(latent_shape, dtype=torch.float32, device=device)
        wrong_latent = torch.empty(latent_shape, dtype=torch.float32, device=device)
        correct_references = {
            index: torch.empty(reference_shape, dtype=torch.float32, device=device)
            for index in REFERENCE_INDICES
        }
        wrong_references = {
            index: torch.empty(reference_shape, dtype=torch.float32, device=device)
            for index in REFERENCE_INDICES
        }
    _broadcast_tensor(correct_latent)
    _broadcast_tensor(wrong_latent)
    for index in REFERENCE_INDICES:
        _broadcast_tensor(correct_references[index])
        _broadcast_tensor(wrong_references[index])
    if (
        tuple(correct_latent.shape) != latent_shape
        or tuple(wrong_latent.shape) != latent_shape
        or any(tuple(value.shape) != reference_shape for value in correct_references.values())
        or any(tuple(value.shape) != reference_shape for value in wrong_references.values())
    ):
        raise SAICSourceAnchorDiagnosticError("encoded condition geometry differs")
    zero_latent = torch.zeros_like(correct_latent)
    zero_references = {
        index: torch.zeros_like(correct_references[index]) for index in REFERENCE_INDICES
    }
    condition_identities = {
        "correct": _condition_identity(
            source=correct_latent, references=correct_references, label="correct"
        ),
        "wrong": _condition_identity(
            source=wrong_latent, references=wrong_references, label="wrong"
        ),
        "synthetic_zero": _condition_identity(
            source=zero_latent, references=zero_references, label="synthetic_zero"
        ),
    }
    correct_condition = condition_identities["correct"]
    wrong_condition = condition_identities["wrong"]
    correct_source_raw = correct_condition["source_video"]["identity"][
        "raw_storage_sha256"
    ]
    wrong_source_raw = wrong_condition["source_video"]["identity"][
        "raw_storage_sha256"
    ]
    distinct_reference_indices = []
    for index in REFERENCE_INDICES:
        key = str(index)
        correct_raw = correct_condition["references"][key]["identity"][
            "raw_storage_sha256"
        ]
        wrong_raw = wrong_condition["references"][key]["identity"][
            "raw_storage_sha256"
        ]
        if correct_raw == wrong_raw:
            raise SAICSourceAnchorDiagnosticError(
                f"correct/wrong encoded reference {index} bytes are identical"
            )
        distinct_reference_indices.append(index)
    if correct_source_raw == wrong_source_raw:
        raise SAICSourceAnchorDiagnosticError(
            "correct/wrong encoded full source conditions are identical"
        )
    condition_distinctness = {
        "source_file_sha256_distinct": True,
        "encoded_full_source_raw_sha256_distinct": True,
        "encoded_reference_raw_sha256_distinct_indices": distinct_reference_indices,
        "all_four_encoded_references_distinct": True,
    }
    vae.to("cpu")
    del correct_pixels, wrong_pixels
    torch.cuda.empty_cache()

    noop_clean_body = prompt_clean(
        native.TASK_BINDING_CLAUSES["rv2v"] + anchor_trainer.NOOP_INSTRUCTION
    )
    action_clean_body = prompt_clean(
        native.TASK_BINDING_CLAUSES["rv2v"] + action_caption
    )
    noop_prompt = native.build_task_prompt(
        "rv2v", anchor_trainer.NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    action_prompt = native.build_task_prompt(
        "rv2v", action_caption, prompt_cleaner=prompt_clean
    )
    if (
        action_caption == anchor_trainer.NOOP_INSTRUCTION
        or action_clean_body == noop_clean_body
        or action_prompt == noop_prompt
    ):
        raise SAICSourceAnchorDiagnosticError(
            "action raw/clean/full prompt collapsed to the no-op condition"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)
    action_ids, action_mask = legacy._tokenize_training_prompt(tokenizer, action_prompt)
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
    )
    noop_token_identity = native._all_rank_tensor_identity(
        noop_ids.to(device), label="noop_prompt_token_ids", world_size=WORLD_SIZE
    )
    action_token_identity = native._all_rank_tensor_identity(
        action_ids.to(device), label="action_prompt_token_ids", world_size=WORLD_SIZE
    )
    noop_mask_identity = native._all_rank_tensor_identity(
        noop_mask.to(device), label="noop_prompt_attention_mask", world_size=WORLD_SIZE
    )
    action_mask_identity = native._all_rank_tensor_identity(
        action_mask.to(device), label="action_prompt_attention_mask", world_size=WORLD_SIZE
    )
    if (
        noop_token_identity["identity"]["raw_storage_sha256"]
        == action_token_identity["identity"]["raw_storage_sha256"]
    ):
        raise SAICSourceAnchorDiagnosticError(
            "action token condition collapsed to the no-op condition"
        )
    if (
        noop_mask_identity["identity"]["raw_storage_sha256"]
        == action_mask_identity["identity"]["raw_storage_sha256"]
    ):
        raise SAICSourceAnchorDiagnosticError(
            "action attention mask collapsed to the no-op condition"
        )
    model.to(device)
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        noop_embeds = _broadcast_prompt(
            model.encode_prompt(noop_ids.to(device), noop_mask.to(device)).detach()
        )
        action_embeds = _broadcast_prompt(
            model.encode_prompt(action_ids.to(device), action_mask.to(device)).detach()
        )
        negative_embeds = _broadcast_prompt(
            model.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach()
        )
    noop_embedding_identity = native._all_rank_tensor_identity(
        noop_embeds, label="noop_prompt_embedding", world_size=WORLD_SIZE
    )
    action_embedding_identity = native._all_rank_tensor_identity(
        action_embeds, label="action_prompt_embedding", world_size=WORLD_SIZE
    )
    if (
        noop_embedding_identity["identity"]["raw_storage_sha256"]
        == action_embedding_identity["identity"]["raw_storage_sha256"]
    ):
        raise SAICSourceAnchorDiagnosticError(
            "action prompt embedding collapsed to the no-op embedding"
        )
    prompt_distinctness = {
        "raw_instruction_distinct": True,
        "clean_body_distinct": True,
        "full_prompt_distinct": True,
        "token_ids_distinct": True,
        "attention_mask_distinct": True,
        "embedding_distinct": True,
        "noop_clean_body_sha256": hashlib.sha256(
            noop_clean_body.encode("utf-8")
        ).hexdigest(),
        "action_clean_body_sha256": hashlib.sha256(
            action_clean_body.encode("utf-8")
        ).hexdigest(),
        "noop_token_ids": noop_token_identity,
        "action_token_ids": action_token_identity,
        "noop_attention_mask": noop_mask_identity,
        "action_attention_mask": action_mask_identity,
        "noop_embedding": noop_embedding_identity,
        "action_embedding": action_embedding_identity,
    }
    model.t5_text_encoder.to("cpu")
    del tokenizer
    torch.cuda.empty_cache()
    diffusion = model.diff_dec
    if diffusion.transformer is None or diffusion.transformer_2 is not None:
        raise SAICSourceAnchorDiagnosticError(
            "source-anchor diagnostic requires transformer_1 only"
        )
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if bool(getattr(diffusion.transformer, "gradient_checkpointing", False)) or bool(
        getattr(diffusion.transformer, "is_gradient_checkpointing", False)
    ):
        raise SAICSourceAnchorDiagnosticError("gradient checkpointing remains enabled")
    base_before = strong_audit._strong_model_freeze_certificate(model)

    generated: dict[str, Any] = {}
    captures: dict[str, Any] = {}
    runtime_receipts: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    cell_inputs: dict[str, Any] = {}
    handle: Optional[anchor_adapter.SAICSourceAnchorHandle] = None
    install_receipt: Optional[Mapping[str, Any]] = None
    zero_certificate: Optional[Mapping[str, Any]] = None
    load_receipt: Optional[Mapping[str, Any]] = None

    def execute(
        cell: str,
        *,
        prompt: Any,
        source: Any,
        references: Mapping[int, Any],
    ) -> None:
        if handle is None:
            raise SAICSourceAnchorDiagnosticError("source-anchor handle is absent")
        with torch.inference_mode():
            stored, capture, runtime_row, identity, input_row = _run_cell(
                cell=cell,
                diffusion=diffusion,
                handle=handle,
                prompt=prompt,
                negative=negative_embeds,
                source=source,
                references=references,
                latent_shape=latent_shape,
                bucket_hw=bucket_hw,
                device=device,
                seed=args.seed,
                wan_diffusion_module=wan_diffusion,
            )
        generated[cell] = stored
        captures[cell] = capture
        runtime_receipts[cell] = runtime_row
        generated_identities[cell] = identity
        cell_inputs[cell] = input_row

    try:
        handle = anchor_adapter.install_saic_source_anchor_adapter(
            diffusion.transformer
        )
        install_receipt = dict(handle.receipt())
        zero_certificate = _zero_initial_adapter_certificate(handle)
        execute(
            "base_correct_noop",
            prompt=noop_embeds,
            source=correct_latent,
            references=correct_references,
        )
        execute(
            "base_correct_action",
            prompt=action_embeds,
            source=correct_latent,
            references=correct_references,
        )
        load_receipt = anchor_runtime.load_saic_source_anchor_safetensors(
            handle,
            adapter_snapshot.path,
            expected_file_sha256=adapter_snapshot.sha256,
            expected_metadata=stage_a["expected_safetensors_metadata"],
        )
        execute(
            "anchor_correct_noop",
            prompt=noop_embeds,
            source=correct_latent,
            references=correct_references,
        )
        execute(
            "anchor_wrong_noop",
            prompt=noop_embeds,
            source=wrong_latent,
            references=wrong_references,
        )
        execute(
            "anchor_route_drop_noop",
            prompt=noop_embeds,
            source=correct_latent,
            references=correct_references,
        )
        execute(
            "anchor_zero_condition_noop",
            prompt=noop_embeds,
            source=zero_latent,
            references=zero_references,
        )
        execute(
            "anchor_correct_action",
            prompt=action_embeds,
            source=correct_latent,
            references=correct_references,
        )
        if not torch.equal(
            generated["anchor_route_drop_noop"], generated["base_correct_noop"]
        ):
            raise SAICSourceAnchorDiagnosticError(
                "route-drop no-op is not byte-identical to frozen base"
            )
    finally:
        if handle is not None and not handle.restored:
            if anchor_adapter.active_route() is not None:
                raise SAICSourceAnchorDiagnosticError(
                    "cannot restore after a leaked source-anchor route"
                )
            handle.restore()
    if (
        set(generated) != set(CELL_ORDER)
        or install_receipt is None
        or zero_certificate is None
        or load_receipt is None
    ):
        raise SAICSourceAnchorDiagnosticError("seven-cell execution closure differs")
    gaussian_hashes = {captures[cell].raw_value_sha256 for cell in CELL_ORDER}
    if len(gaussian_hashes) != 1:
        raise SAICSourceAnchorDiagnosticError(
            "seven cells did not share one exact official Gaussian"
        )
    base_after = strong_audit._strong_model_freeze_certificate(model)
    if base_after != base_before:
        raise SAICSourceAnchorDiagnosticError(
            "vendor model did not restore to byte-identical pre-install state"
        )
    model.to("cpu")
    torch.cuda.empty_cache()

    manifest.assert_unchanged()
    for snapshot in (
        source_release_snapshot,
        checkpoint_manifest,
        source_manifest_snapshot,
        adapter_snapshot,
        stage_a_receipt_snapshot,
        stage_a_postflight_snapshot,
        stage_a_history_snapshot,
        stage_a_checkpoint_release_snapshot,
        action_caption_snapshot,
        correct_source,
        wrong_source,
    ):
        snapshot.assert_unchanged()

    checkpoint_post_box: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_post_box[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint,
                    checkpoint_manifest.path,
                    expected_manifest_sha256=checkpoint_manifest.sha256,
                ),
            }
        except Exception as error:
            checkpoint_post_box[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_post_box, src=0)
    if (
        not isinstance(checkpoint_post_box[0], Mapping)
        or checkpoint_post_box[0].get("ok") is not True
        or checkpoint_post_box[0].get("identity") != checkpoint_identity
    ):
        raise SAICSourceAnchorDiagnosticError(
            f"post-sampling Bernini checkpoint closure differs: {checkpoint_post_box[0]}"
        )

    if distributed.rank == 0:
        if visual_binding is None:
            raise SAICSourceAnchorDiagnosticError(
                "rank zero lost the visual checkpoint binding"
            )
        staging.mkdir(mode=0o700, parents=False, exist_ok=False)
        generated_for_decode = {
            cell: generated[cell].to(device=device).contiguous() for cell in CELL_ORDER
        }
        try:
            outputs_staging = native._save_outputs(
                output_dir=staging,
                generated=generated_for_decode,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
        finally:
            generated_for_decode.clear()
        shared_noise = native._save_initial_noise_atomically(
            staging / "shared.official-initial-gaussian.safetensors",
            captures[CELL_ORDER[0]],
            all_rank_identity=cell_inputs[CELL_ORDER[0]]["noise"],
        )
        correct_latent_artifact = native._save_normalized_clean_latent_atomically(
            staging / "correct-source.normalized-clean-latent.safetensors",
            correct_latent,
            artifact_role="source_video_condition",
        )
        wrong_latent_artifact = native._save_normalized_clean_latent_atomically(
            staging / "wrong-source.normalized-clean-latent.safetensors",
            wrong_latent,
            artifact_role="source_video_condition",
        )
        decoded = build_decoded_diagnostics(
            correct_source=correct_source,
            wrong_source=wrong_source,
            outputs=outputs_staging,
            visual_binding=visual_binding,
            device=device,
            staging_root=staging,
            final_root=output,
        )
        try:
            checkpoint_final_identity = source_audit.validate_checkpoint_content(
                checkpoint,
                checkpoint_manifest.path,
                expected_manifest_sha256=checkpoint_manifest.sha256,
            )
        except Exception as error:
            raise SAICSourceAnchorDiagnosticError(
                f"final Bernini checkpoint validation failed: {error}"
            ) from error
        if checkpoint_final_identity != checkpoint_identity:
            raise SAICSourceAnchorDiagnosticError(
                "Bernini checkpoint closure changed before publication"
            )
        visual_binding.assert_unchanged()
        decoded_path = staging / "decoded-diagnostics.json"
        decoded_sha = _write_json_create_only(decoded_path, decoded)
        outputs = _rebase_paths(
            outputs_staging, staging_root=staging, final_root=output
        )
        shared_noise = _rebase_paths(
            shared_noise, staging_root=staging, final_root=output
        )
        correct_latent_artifact = _rebase_paths(
            correct_latent_artifact, staging_root=staging, final_root=output
        )
        wrong_latent_artifact = _rebase_paths(
            wrong_latent_artifact, staging_root=staging, final_root=output
        )
        receipt_body = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "classification": CLASSIFICATION,
            "complete": True,
            "first_real_runtime_status": "canary_only",
            "execution": {
                "slurm_job_id": args.slurm_job_id,
                "world_size": WORLD_SIZE,
                "ulysses_size": ULYSSES_SIZE,
                "single_node": True,
                "first_real_exact40_is_canary": True,
                "dynamic_rendezvous": rendezvous_admission,
                "gpu_visibility_source": args.gpu_visibility_source,
                "gpu_visibility": args.expected_gpu_visibility,
                "rank_to_physical_gpu": {
                    str(index): token
                    for index, token in enumerate(
                        args.expected_gpu_visibility.split(",")
                    )
                },
            },
            "cell_order": list(CELL_ORDER),
            "input": {
                "source_manifest": dict(source_manifest_snapshot.receipt()),
                "manifest_digest": manifest.manifest_digest,
                "heldout_row_index": row.row_index,
                "correct_iid": row.iid,
                "wrong_iid": row.wrong_iid,
                "correct_source": dict(correct_source.receipt()),
                "wrong_source": dict(wrong_source.receipt()),
                "same_bucket_hw": list(bucket_hw),
                "action_caption_file": dict(action_caption_snapshot.receipt()),
                "action_caption_utf8_sha256": hashlib.sha256(
                    action_caption.encode("utf-8")
                ).hexdigest(),
                "target_video_read": False,
                "mask_pose_flow_track_trajectory_read": False,
            },
            "stage_a": {
                **dict(stage_a),
                "formal_postflight": dict(formal_stage_a),
                "adapter": dict(adapter_snapshot.receipt()),
                "receipt": dict(stage_a_receipt_snapshot.receipt()),
                "history": dict(stage_a_history_snapshot.receipt()),
                "checkpoint_release": dict(
                    stage_a_checkpoint_release_snapshot.receipt()
                ),
                "install_receipt": install_receipt,
                "zero_initial_adapter_certificate": zero_certificate,
                "strict_load_receipt": load_receipt,
            },
            "conditions": {
                "task_prompt": "vr2v",
                "guidance_mode": "v2v_apg",
                "full_source_video_count": 1,
                "source_reference_count": 4,
                "reference_indices": list(REFERENCE_INDICES),
                "condition_identities": condition_identities,
                "correct_wrong_distinctness": condition_distinctness,
                "wrong_source_replaces_video_and_all_four_references": True,
                "route_drop_retains_complete_correct_source_condition": True,
                "zero_condition_is_synthetic_ood": True,
                "zero_condition_is_not_a_real_dropped_source_claim": True,
            },
            "prompts": {
                "noop_instruction_sha256": hashlib.sha256(
                    anchor_trainer.NOOP_INSTRUCTION.encode("utf-8")
                ).hexdigest(),
                "noop_full_prompt_sha256": hashlib.sha256(
                    noop_prompt.encode("utf-8")
                ).hexdigest(),
                "action_full_prompt_sha256": hashlib.sha256(
                    action_prompt.encode("utf-8")
                ).hexdigest(),
                "action_caption_raw_sha256": hashlib.sha256(
                    action_caption.encode("utf-8")
                ).hexdigest(),
                "distinctness": prompt_distinctness,
                "negative_prompt_sha256": hashlib.sha256(
                    legacy.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
                ).hexdigest(),
                "semantic_action_observer_available": False,
            },
            "sampling": {
                **dict(_sampling_contract(seed=args.seed)),
                "fps": FPS,
                "latent_phases": LATENT_PHASES,
                "ulysses_size": ULYSSES_SIZE,
                "same_official_gaussian_all_cells": True,
                "official_gaussian_raw_sha256": next(iter(gaussian_hashes)),
                "external_initial_noise_injection": False,
                "best_of_n_or_seed_selection": False,
            },
            "runtime_by_cell": runtime_receipts,
            "cell_inputs": cell_inputs,
            "generated_identities": generated_identities,
            "route_drop_byte_identical_to_base_correct_noop": True,
            "outputs": outputs,
            "artifacts": {
                "shared_official_gaussian": shared_noise,
                "correct_source_latent": correct_latent_artifact,
                "wrong_source_latent": wrong_latent_artifact,
                "decoded_diagnostics": {
                    "path": str(output / decoded_path.name),
                    "sha256": decoded_sha,
                    "diagnostic_digest": decoded["diagnostic_digest"],
                },
            },
            "checkpoint": {
                "root": str(checkpoint),
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_manifest": dict(checkpoint_manifest.receipt()),
                "content_identity": checkpoint_identity,
            },
            "vendor_restore": {
                "before": base_before,
                "after": base_after,
                "byte_identical": True,
            },
            "source_revisions": {
                "bernini": bernini_revision,
                "veomni": veomni_revision,
                "method": args.method_source_revision,
                "method_source_archive_sha256": args.method_source_archive_sha256,
                "method_source_release_manifest": dict(
                    source_release_snapshot.receipt()
                ),
                "method_source_member_manifest_sha256": (
                    args.method_source_member_manifest_sha256
                ),
                "method_source_member_manifest_digest": (
                    args.method_source_member_manifest_digest
                ),
                "method_source_member_count": args.method_source_member_count,
                "method_source_origin_manifest_sha256": (
                    args.method_source_origin_manifest_sha256
                ),
                "method_source_origin_manifest_digest": (
                    args.method_source_origin_manifest_digest
                ),
                "method_source_origin_count": args.method_source_origin_count,
                "archive_recursive_closure_preflighted_before_import": True,
                "all_import_origins_matched_extracted_tree": True,
                "bernini_inference_files": inference_file_hashes,
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "authority": {
                "stage_a_decoded_runtime_qualified": False,
                "stage_b_runtime_available": False,
                "semantic_action_nonregression_available": False,
                "identity_authority": False,
                "training_allowed": False,
                "optimizer_step_allowed": False,
                "checkpoint_allowed": False,
                "selection_allowed": False,
                "publication_allowed": False,
                "production_allowed": False,
                "scientific_success_claimed": False,
            },
        }
        receipt = {**receipt_body, "receipt_digest": object_sha256(receipt_body)}
        receipt_path = staging / "receipt.json"
        _write_json_create_only(receipt_path, receipt)
        expected_names = {
            "shared.official-initial-gaussian.safetensors",
            "correct-source.normalized-clean-latent.safetensors",
            "wrong-source.normalized-clean-latent.safetensors",
            "decoded-diagnostics.json",
            "receipt.json",
        } | {
            f"{cell}.mp4" for cell in CELL_ORDER
        } | {
            f"{cell}.normalized-clean-latent.safetensors" for cell in CELL_ORDER
        }
        observed_names = {path.name for path in staging.iterdir()}
        if observed_names != expected_names or any(
            path.is_symlink() or not path.is_file() for path in staging.iterdir()
        ):
            raise SAICSourceAnchorDiagnosticError(
                "staged diagnostic artifact closure differs"
            )
        for path in staging.iterdir():
            path.chmod(0o444)
        _fsync_directory(staging)
        _publish_create_only_directory(
            staging, output, commit_name="receipt.json"
        )
        if not output.is_dir() or output.is_symlink() or staging.exists():
            raise SAICSourceAnchorDiagnosticError("atomic output publication failed")
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    for snapshot in (
        source_release_snapshot,
        checkpoint_manifest,
        source_manifest_snapshot,
        adapter_snapshot,
        stage_a_receipt_snapshot,
        stage_a_postflight_snapshot,
        stage_a_history_snapshot,
        stage_a_checkpoint_release_snapshot,
        action_caption_snapshot,
        correct_source,
        wrong_source,
    ):
        snapshot.assert_unchanged()
    dist.destroy_process_group()
    return 0


__all__ = [
    "ACTION_CELLS",
    "BASE_CELLS",
    "CELL_ORDER",
    "CLASSIFICATION",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "METHOD",
    "NOOP_CELLS",
    "ROUTED_CELLS",
    "SAICSourceAnchorDiagnosticError",
    "SCHEMA_VERSION",
    "VisualCheckpointBinding",
    "build_decoded_diagnostics",
    "build_parser",
    "canonical_json_bytes",
    "main",
    "object_sha256",
    "validate_cli",
    "validate_formal_stage_a_postflight",
    "validate_stage_a_bundle",
    "validate_visual_checkpoint",
    "validate_visual_release",
]


if __name__ == "__main__":
    raise SystemExit(main())
