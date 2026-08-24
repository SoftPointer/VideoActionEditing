#!/usr/bin/env python3
"""Fail-closed 0817 checkpoint evaluation evidence coordinator.

This module consumes frozen checkpoints, source/instruction rows, decoded
full-video evidence, opaque review packets, and sealed human ballots.  It
re-hashes artifact/code trees and fully decodes source/output videos to EOF;
it verifies an independently signed authority root and ballot seal; and it does
not train, rank by loss, or invoke an automatic evaluator.  The formal path is
hard-bound to the in-module ffmpeg decoder.  Supplying a test decoder produces
an explicitly non-formal receipt.  This coordinator emits descriptive evidence
only: its Pareto set is always empty and its winner is always null.

The important boundary is deliberate: ``PRE_D0_ENGINEERING_ONLY`` artifacts
(including the 0817 r2 smoke) may receive an engineering closure receipt, but
they never acquire selection authority or authorize a scientific promotion.
Automatic metrics are diagnostic-only even when their calibration is qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import struct
import subprocess
import tempfile
from fractions import Fraction
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence


INPUT_SCHEMA = "bernini-action-editing-checkpoint-evaluation-input-0817-v4"
CHECKPOINT_FREEZE_SCHEMA = "bernini-action-editing-checkpoint-freeze-0817-v4"
ROW_FREEZE_SCHEMA = "bernini-action-editing-evaluation-row-freeze-0817-v4"
LOCKED_SPLIT_SCHEMA = "bernini-action-editing-locked-promotion-split-0817-v1"
CALIBRATION_SCHEMA = "bernini-action-editing-evaluator-calibration-0817-v1"
PRIVATE_MAPPING_SCHEMA = "bernini-action-editing-private-blind-mapping-0817-v2"
PUBLIC_PACKET_SCHEMA = "bernini-action-editing-public-blind-packet-0817-v2"
BALLOT_SCHEMA = "bernini-action-editing-full81-blind-ballot-0817-v3"
UNBLINDING_SCHEMA = "bernini-action-editing-ballot-unblinding-0817-v3"
DIAGNOSTIC_SCHEMA = "bernini-action-editing-automatic-diagnostic-0817-v1"
SELECTION_CONTRACT_SCHEMA = "bernini-action-editing-selection-contract-0817-v4"
BASE_RECEIPT_SCHEMA = "bernini-action-editing-frozen-base-receipt-0817-v1"
FORMAL_TRAINING_RECEIPT_SCHEMA = (
    "bernini-action-editing-formal-training-receipt-0817-v1"
)
PRE_D0_TRAINING_RECEIPT_SCHEMA = (
    "bernini-action-edit-large-lora-0817-pre-d0-receipt-v1"
)
DECODE_RECEIPT_SCHEMA = "bernini-action-editing-full81-decode-receipt-0817-v3"
KEEPER_COMMITMENT_SCHEMA = "bernini-action-editing-blind-keeper-commitment-0817-v1"
FORMAL_AUTHORITY_SCHEMA = (
    "bernini-action-editing-formal-training-authority-manifest-0817-v2"
)
AUTHORITY_ROOT_SCHEMA = (
    "bernini-action-editing-independent-authority-root-0817-v2"
)
DETACHED_SIGNATURE_SCHEMA = (
    "bernini-action-editing-external-detached-signature-0817-v1"
)
ARTIFACT_TREE_MANIFEST_SCHEMA = (
    "bernini-action-editing-checkpoint-artifact-tree-manifest-0817-v2"
)
RELEASE_MANIFEST_SCHEMA = "bernini-action-editing-code-release-manifest-0817-v2"
SOURCE_EQUIVALENCE_SCHEMA = (
    "bernini-action-editing-source-equivalence-authority-manifest-0817-v2"
)
REVIEWER_ROSTER_SCHEMA = (
    "bernini-action-editing-reviewer-roster-authority-manifest-0817-v2"
)
BALLOT_SEAL_SCHEMA = (
    "bernini-action-editing-ballot-seal-authority-manifest-0817-v1"
)
RECEIPT_SCHEMA = "bernini-action-editing-checkpoint-evidence-receipt-0817-v5"
SAFE_TENSOR_SEMANTIC_SCHEMA = (
    "bernini-action-editing-safetensors-semantic-closure-0817-v1"
)
MODEL_STATE_CONTRACT_SCHEMA = (
    "bernini-action-editing-bernini-model-state-contract-0817-v1"
)
RENDER_EXECUTION_AUTHORITY_SCHEMA = (
    "bernini-action-editing-renderer-consumer-execution-authority-0817-v1"
)
PYTHON_ENTRYPOINT_SCHEMA = "python3-executable-entrypoint-0817-v1"

PRODUCTION_DECODER_ID = "fixed-ffmpeg-full81-source-bound-0817-v2"
INJECTED_DECODER_TIER = "INJECTED_DECODER_NONFORMAL_TEST_ONLY"
PRODUCTION_DECODER_TIER = "PRODUCTION_FFMPEG_FORMAL_ELIGIBLE"
MEDIA_TOOL_TIMEOUT_SECONDS = 120
MAX_FFPROBE_JSON_BYTES = 4 * 1024 * 1024
MAX_DECODED_RGB_BYTES = 512 * 1024 * 1024
MAX_CONTROL_FILE_BYTES = 64 * 1024 * 1024
MAX_EXECUTABLE_ENTRYPOINT_BYTES = 4 * 1024 * 1024
MIN_FORMAL_SOURCE_CLUSTERS = 50
MIN_FORMAL_ACTOR_SCENE_CLUSTERS = 50
FROZEN_PERCEPTUAL_ALGORITHM = "rgb24-uniform-subsample-sha256-v1"
FROZEN_ACTOR_ALGORITHM = "rgb24-central-region-sha256-v1"
FROZEN_SCENE_ALGORITHM = "rgb24-border-region-sha256-v1"
FROZEN_CLUSTER_ALGORITHM = (
    "pre-frozen-collection-actor-scene-equivalence-semantic-authority-v1"
)
FROZEN_ALGORITHM_SHA256 = ""  # assigned from loaded implementation bytecode below
_FFMPEG_DISCOVERED = shutil.which("ffmpeg")
_FFPROBE_DISCOVERED = shutil.which("ffprobe")
_FIXED_FFMPEG_EXECUTABLE = (
    None if _FFMPEG_DISCOVERED is None else str(Path(_FFMPEG_DISCOVERED).resolve())
)
_FIXED_FFPROBE_EXECUTABLE = (
    None if _FFPROBE_DISCOVERED is None else str(Path(_FFPROBE_DISCOVERED).resolve())
)

FORMAL_MODE = "promotion_validation"
ENGINEERING_MODE = "engineering_comparison"
PRE_D0_STAGE = "PRE_D0_ENGINEERING_ONLY"

AXES = (
    "action",
    "order",
    "identity",
    "ownership",
    "background",
    "camera",
    "quality",
    "noop",
)
LABELS = ("pass", "fail", "abstain")
NOT_ASSESSABLE = "not_assessable"
MODEL_CAUSED_REASONS = (
    "blur",
    "occlusion",
    "crop",
    "missing_frames",
    "decode_failure",
)
SPLITS = (
    "seen_action_unseen_source",
    "unseen_scene_camera",
    "unseen_action_composition",
    "interaction_contact",
    "noop_preservation",
)
FORMAL_SPLIT_COUNTS = {
    "seen_action_unseen_source": 150,
    "unseen_scene_camera": 100,
    "unseen_action_composition": 100,
    "interaction_contact": 100,
    "noop_preservation": 50,
}
CALIBRATION_CATEGORIES = (
    "reverse",
    "incomplete",
    "wrong_actor",
    "wrong_object",
)
BOOTSTRAP_RESAMPLES = 10_000
ABSTAIN_LIMIT = 0.10
FIXED_BOOTSTRAP_SEED_HEX = hashlib.sha256(
    b"bernini-action-editing-0817-promotion-bootstrap-v1"
).hexdigest()

# This release has no production formal authority.  Provisioning is not a
# runtime flag, registry entry, environment variable, API argument, or CLI
# option: enabling it requires a new reviewed source release containing the
# real Bernini ABI, immutable renderer/consumer release, external trust roots,
# and real ffmpeg E2E evidence.  Keeping this as executable policy rather than
# mutable data prevents an in-process caller from filling a registry and
# manufacturing formal eligibility.
FORMAL_PRODUCTION_AUTHORITY_BLOCKER = (
    "FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED"
)

# This exact checked-in r2 receipt is a permanently registered PRE_D0 artifact.
# The external formal authority must repeat this registration.  Consequently,
# copying or renaming the receipt cannot turn it into a formal checkpoint.
KNOWN_PRE_D0_RECEIPTS = {
    "8014b7b71413318d80162fba12b73d83d6b9d9de5ea57ad295643a238b0f8c0e":
        PRE_D0_TRAINING_RECEIPT_SCHEMA,
}
KNOWN_PRE_D0_R2_REGISTRATION = {
    "training_receipt_file_sha256":
        "8014b7b71413318d80162fba12b73d83d6b9d9de5ea57ad295643a238b0f8c0e",
    "receipt_schema": PRE_D0_TRAINING_RECEIPT_SCHEMA,
    "classification": "pre_d0",
    "training_stage": PRE_D0_STAGE,
    "checkpoint_artifacts": [
        {
            "step": 0,
            "sha256": "cea3145d213d271cbae5ae458e23a861d5ca1aea95686b0176d07cd1fc3ffb94",
        },
        {
            "step": 1,
            "sha256": "34dd7b2835e38e3138f6533c7a36d3ad1f252715f2048af9f7f98d8f1eb79eb7",
        },
        {
            "step": 2,
            "sha256": "a1c01f2b9b5496e4ed54e6c4ce2fe9fb554f16432d90b77851fba3739e12c4a7",
        },
    ],
    "data_authority_digest":
        "554d8b84804b998a9e33eb7cc22cef8d803244192a711c62c368ac30484d96f8",
    "runner_code_sha256":
        "edf3d1d2a77cb2f713968f537ce85a7d92f0b7347a0474419fe5562fbd319bd9",
}
KNOWN_PRE_D0_ARTIFACTS = {
    item["sha256"] for item in KNOWN_PRE_D0_R2_REGISTRATION["checkpoint_artifacts"]
}

# These are non-compensable admission floors, not weights.  The action/order
# axes remain Pareto objectives and are never allowed to buy back preservation.
HARD_MINIMUM_PASS_RATES = {
    "action": 0.50,
    "order": 0.50,
    "identity": 0.90,
    "ownership": 0.85,
    "background": 0.90,
    "camera": 0.90,
    "quality": 0.85,
    "noop": 0.95,
}

# These are admission rules relative to the frozen base, not Pareto weights.
# Later stages are intentionally no weaker than D0.
STAGE_REFERENCE_GATES = {
    "D0": {"action_delta": 0.03, "order_delta": 0.03, "ci_lower": 0.00},
    "D1": {"action_delta": 0.10, "order_delta": 0.10, "ci_lower": 0.03},
    "D2": {"action_delta": 0.10, "order_delta": 0.10, "ci_lower": 0.03},
    "FULL_RENDERER": {
        "action_delta": 0.05,
        "order_delta": 0.05,
        "ci_lower": 0.00,
    },
    "C1": {"action_delta": 0.05, "order_delta": 0.05, "ci_lower": 0.00},
    "C2": {"action_delta": 0.05, "order_delta": 0.05, "ci_lower": 0.00},
    "FINAL": {"action_delta": 0.05, "order_delta": 0.05, "ci_lower": 0.00},
}
MAX_STRATUM_ACTION_REGRESSION = -0.05
MIN_STRATUM_PRESERVATION_PASS_RATE = 0.85

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")


class CheckpointEvaluationSelectionError(RuntimeError):
    """The supplied evidence is structurally open or internally inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointEvaluationSelectionError(
            "value is not canonical JSON: %s" % error
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_LOADED_COORDINATOR_FILE_SHA256 = file_sha256(Path(__file__).resolve())


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise CheckpointEvaluationSelectionError(
                    "%s contains a duplicate key" % label
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointEvaluationSelectionError(
            "%s is not strict JSON" % label
        ) from error
    if not isinstance(value, Mapping):
        raise CheckpointEvaluationSelectionError("%s root is not an object" % label)
    if payload != canonical_json_bytes(value) + b"\n":
        raise CheckpointEvaluationSelectionError("%s bytes are not canonical" % label)
    return dict(value)


def _read_plain_file(
    value: Any,
    *,
    expected_sha256: Any,
    expected_size: Any | None,
    label: str,
    require_read_only: bool = False,
    expected_mode: int | None = None,
    return_payload: bool = True,
) -> tuple[Path, bytes]:
    if not isinstance(value, str):
        raise CheckpointEvaluationSelectionError("%s path differs" % label)
    path = Path(value)
    if (
        not path.is_absolute()
        or value == os.path.sep
        or os.path.normpath(value) != value
    ):
        raise CheckpointEvaluationSelectionError(
            "%s path must be absolute and normalized" % label
        )
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise CheckpointEvaluationSelectionError("%s is missing" % label) from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise CheckpointEvaluationSelectionError("%s is not a plain file" % label)
    if info.st_nlink != 1:
        raise CheckpointEvaluationSelectionError(
            "%s must have exactly one hard link" % label
        )
    try:
        if path.resolve(strict=True) != path:
            raise CheckpointEvaluationSelectionError(
                "%s path contains a symlink/non-canonical component" % label
            )
    except OSError as error:
        raise CheckpointEvaluationSelectionError(
            "%s canonical path resolution failed" % label
        ) from error
    if require_read_only and info.st_mode & 0o222:
        raise CheckpointEvaluationSelectionError(
            "%s is not immutable/read-only" % label
        )
    if expected_mode is not None and stat.S_IMODE(info.st_mode) != expected_mode:
        raise CheckpointEvaluationSelectionError("%s mode differs" % label)
    if return_payload and info.st_size > MAX_CONTROL_FILE_BYTES:
        raise CheckpointEvaluationSelectionError("%s exceeds safe control-file bound" % label)
    payload = path.read_bytes() if return_payload else b""
    wanted = _sha(expected_sha256, label="%s SHA" % label)
    actual_digest = hashlib.sha256(payload).hexdigest() if return_payload else file_sha256(path)
    if actual_digest != wanted:
        raise CheckpointEvaluationSelectionError("%s SHA differs" % label)
    if expected_size is not None and (
        type(expected_size) is not int
        or expected_size < 0
        or info.st_size != expected_size
    ):
        raise CheckpointEvaluationSelectionError("%s byte count differs" % label)
    after = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_uid,
        item.st_gid,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(info) != identity(after):
        raise CheckpointEvaluationSelectionError("%s changed while read" % label)
    if require_read_only and after.st_mode & 0o222:
        raise CheckpointEvaluationSelectionError(
            "%s became writable while read" % label
        )
    if after.st_nlink != 1:
        raise CheckpointEvaluationSelectionError(
            "%s hard-link count changed while read" % label
        )
    if expected_mode is not None and stat.S_IMODE(after.st_mode) != expected_mode:
        raise CheckpointEvaluationSelectionError("%s mode changed while read" % label)
    return path, payload


def _read_external_manifest(value: Any, *, label: str) -> tuple[dict[str, Any], str]:
    descriptor = _closed(
        value,
        {"path", "file_sha256"},
        label="%s descriptor" % label,
    )
    wanted = _sha(descriptor["file_sha256"], label="%s file" % label)
    _, payload = _read_plain_file(
        descriptor["path"],
        expected_sha256=wanted,
        expected_size=None,
        label=label,
        require_read_only=True,
        expected_mode=0o444,
    )
    manifest = _strict_json_bytes(payload, label=label)
    return manifest, wanted


def _signature_message(envelope: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(
        {
            key: envelope[key]
            for key in (
                "schema_version",
                "purpose",
                "key_id",
                "trust_root_sha256",
                "payload_sha256",
                "signed_at_utc",
                "trusted_time_authority_id",
            )
        }
    )


def _verify_detached_signature(
    *,
    payload_sha256: str,
    descriptor_path: Any,
    descriptor_sha256: Any,
    purpose: str,
    verifier: ExternalSignatureVerifier | None,
    expected_trust_root_sha256: str,
    expected_key_id: str,
    expected_tsa_id: str,
    label: str,
    timestamp_verifier: TrustedTimestampVerifier | None = None,
) -> dict[str, Any]:
    if verifier is None:
        raise CheckpointEvaluationSelectionError(
            "%s requires an independent external signature verifier" % label
        )
    if (
        getattr(verifier, "trust_root_sha256", None)
        != expected_trust_root_sha256
        or getattr(verifier, "key_id", None) != expected_key_id
    ):
        raise CheckpointEvaluationSelectionError(
            "%s verifier differs from the precommitted trust root/key" % label
        )
    envelope, envelope_sha = _read_external_manifest(
        {"path": descriptor_path, "file_sha256": descriptor_sha256},
        label="%s detached signature" % label,
    )
    envelope = _closed(
        envelope,
        {
            "schema_version",
            "purpose",
            "key_id",
            "trust_root_sha256",
            "payload_sha256",
            "signed_at_utc",
            "trusted_time_authority_id",
            "signature_hex",
            "envelope_digest",
        },
        label="%s detached signature" % label,
    )
    _verify_digest(
        envelope,
        field="envelope_digest",
        label="%s detached signature" % label,
    )
    if (
        envelope["schema_version"] != DETACHED_SIGNATURE_SCHEMA
        or envelope["purpose"] != purpose
        or envelope["key_id"] != expected_key_id
        or envelope["trust_root_sha256"] != expected_trust_root_sha256
        or envelope["payload_sha256"] != payload_sha256
        or envelope["trusted_time_authority_id"] != expected_tsa_id
    ):
        raise CheckpointEvaluationSelectionError(
            "%s signed authority binding differs" % label
        )
    _utc_timestamp(envelope["signed_at_utc"], label="%s signed time" % label)
    signature_hex = envelope["signature_hex"]
    if (
        not isinstance(signature_hex, str)
        or len(signature_hex) < 64
        or len(signature_hex) > 1024
        or len(signature_hex) % 2
        or re.fullmatch(r"[0-9a-f]+", signature_hex) is None
    ):
        raise CheckpointEvaluationSelectionError("%s signature encoding differs" % label)
    try:
        verified = verifier.verify(_signature_message(envelope), signature_hex)
    except Exception as error:
        raise CheckpointEvaluationSelectionError(
            "%s external signature verifier failed" % label
        ) from error
    if verified is not True:
        raise CheckpointEvaluationSelectionError(
            "%s external detached signature is invalid" % label
        )
    tsa_verified = False
    if timestamp_verifier is not None:
        if getattr(timestamp_verifier, "tsa_id", None) != expected_tsa_id:
            raise CheckpointEvaluationSelectionError(
                "%s timestamp verifier differs from the frozen TSA" % label
            )
        try:
            tsa_verified = timestamp_verifier.verify(
                _signature_message(envelope),
                signed_at_utc=envelope["signed_at_utc"],
                payload_sha256=payload_sha256,
            ) is True
        except Exception as error:
            raise CheckpointEvaluationSelectionError(
                "%s trusted timestamp verification failed" % label
            ) from error
        if not tsa_verified:
            raise CheckpointEvaluationSelectionError(
                "%s has no valid externally verifiable TSA token" % label
            )
    envelope["_tsa_verified"] = tsa_verified
    envelope["_file_sha256"] = envelope_sha
    return envelope


def _validate_authority_root(
    pins: Mapping[str, Any],
    *,
    verifier: ExternalSignatureVerifier | None,
    timestamp_verifier: TrustedTimestampVerifier | None,
) -> dict[str, Any]:
    root, file_sha = _read_external_manifest(
        {
            "path": pins["authority_root_manifest_path"],
            "file_sha256": pins["authority_root_manifest_sha256"],
        },
        label="independent authority root",
    )
    root = _closed(
        root,
        {
            "schema_version",
            "root_id",
            "status",
            "issued_at_utc",
            "trusted_time_authority_id",
            "signing_key_id",
            "formal_training_manifest_file_sha256",
            "source_equivalence_manifest_file_sha256",
            "reviewer_roster_manifest_file_sha256",
            "inference_release_manifest_file_sha256",
            "renderer_execution_authority_signing_key_id",
            "authorized_checkpoint_ids",
            "allowed_training_receipt_schemas",
            "locked_split_digest",
            "keeper_commitment_digest",
            "frozen_algorithm_sha256",
            "production_decoder_id",
            "evaluation_coordinator_file_sha256",
            "ffmpeg_binary_sha256",
            "ffprobe_binary_sha256",
            "tensor_semantic_schema",
            "runner_entrypoint_schema",
            "reviewer_ballot_key_registry_digest",
            "reviewer_ballot_signatures_required",
            "root_digest",
        },
        label="independent authority root",
    )
    if (
        root["schema_version"] != AUTHORITY_ROOT_SCHEMA
        or root["status"] != "precommitted_and_locked"
        or root["trusted_time_authority_id"] != pins["trusted_time_authority_id"]
        or root["signing_key_id"] != pins["precommitted_key_id"]
        or root["formal_training_manifest_file_sha256"]
        != pins["formal_training_manifest_sha256"]
        or root["source_equivalence_manifest_file_sha256"]
        != pins["source_equivalence_manifest_sha256"]
        or root["reviewer_roster_manifest_file_sha256"]
        != pins["reviewer_roster_manifest_sha256"]
        or root["inference_release_manifest_file_sha256"]
        != pins["inference_release_manifest_sha256"]
        or root["renderer_execution_authority_signing_key_id"]
        != root["signing_key_id"]
        or root["authorized_checkpoint_ids"] != pins["authorized_checkpoint_ids"]
        or root["locked_split_digest"] != pins["locked_split_digest"]
        or root["keeper_commitment_digest"] != pins["keeper_commitment_digest"]
        or root["frozen_algorithm_sha256"] != FROZEN_ALGORITHM_SHA256
        or root["production_decoder_id"] != PRODUCTION_DECODER_ID
        or root["evaluation_coordinator_file_sha256"]
        != _LOADED_COORDINATOR_FILE_SHA256
        or file_sha256(Path(__file__).resolve())
        != _LOADED_COORDINATOR_FILE_SHA256
        or root["tensor_semantic_schema"] != SAFE_TENSOR_SEMANTIC_SCHEMA
        or root["runner_entrypoint_schema"] != PYTHON_ENTRYPOINT_SCHEMA
        or root["reviewer_ballot_signatures_required"] is not True
        or root["allowed_training_receipt_schemas"]
        != sorted(
            [
                BASE_RECEIPT_SCHEMA,
                FORMAL_TRAINING_RECEIPT_SCHEMA,
                PRE_D0_TRAINING_RECEIPT_SCHEMA,
            ]
        )
    ):
        raise CheckpointEvaluationSelectionError(
            "independent authority root registry differs"
        )
    _identifier(root["root_id"], label="authority root id")
    _sha(root["ffmpeg_binary_sha256"], label="authority-root ffmpeg binary")
    _sha(root["ffprobe_binary_sha256"], label="authority-root ffprobe binary")
    _sha(
        root["evaluation_coordinator_file_sha256"],
        label="authority-root coordinator source",
    )
    _sha(
        root["reviewer_ballot_key_registry_digest"],
        label="authority-root reviewer key registry",
    )
    issued_at = _utc_timestamp(root["issued_at_utc"], label="authority root issue time")
    _verify_digest(root, field="root_digest", label="independent authority root")
    signature = _verify_detached_signature(
        payload_sha256=file_sha,
        descriptor_path=pins["authority_root_signature_path"],
        descriptor_sha256=pins["authority_root_signature_sha256"],
        purpose="authority_root",
        verifier=verifier,
        expected_trust_root_sha256=pins["precommitted_trust_root_sha256"],
        expected_key_id=pins["precommitted_key_id"],
        expected_tsa_id=pins["trusted_time_authority_id"],
        label="independent authority root",
        timestamp_verifier=timestamp_verifier,
    )
    if _utc_timestamp(
        signature["signed_at_utc"], label="authority root signature time"
    ) != issued_at:
        raise CheckpointEvaluationSelectionError(
            "authority root signature time differs from precommit issue time"
        )
    root["_file_sha256"] = file_sha
    root["_signature"] = signature
    return root


def _validate_fixed_production_media_tools(authority_root: Mapping[str, Any]) -> None:
    for name, path, field in (
        ("ffmpeg", _FIXED_FFMPEG_EXECUTABLE, "ffmpeg_binary_sha256"),
        ("ffprobe", _FIXED_FFPROBE_EXECUTABLE, "ffprobe_binary_sha256"),
    ):
        if path is None:
            raise CheckpointEvaluationSelectionError(
                "formal production decoder requires fixed %s" % name
            )
        target = Path(path)
        try:
            info = target.lstat()
        except OSError as error:
            raise CheckpointEvaluationSelectionError(
                "fixed %s binary is unavailable" % name
            ) from error
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise CheckpointEvaluationSelectionError(
                "fixed %s binary is not a plain executable" % name
            )
        if not info.st_mode & 0o111:
            raise CheckpointEvaluationSelectionError(
                "fixed %s binary is not executable" % name
            )
        if file_sha256(target) != authority_root[field]:
            raise CheckpointEvaluationSelectionError(
                "fixed %s binary differs from signed authority root" % name
            )


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value)
        is None
    ):
        raise CheckpointEvaluationSelectionError("%s is not canonical UTC" % label)
    try:
        result = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise CheckpointEvaluationSelectionError(
            "%s is not a real UTC instant" % label
        ) from error
    return result


MEDIA_SIGNATURE_FIELDS = {
    "frame_count",
    "fps_num",
    "fps_den",
    "width",
    "height",
    "pixel_format",
    "duration_num",
    "duration_den",
    "pts_start_num",
    "pts_start_den",
    "pts_end_num",
    "pts_end_den",
    "pts_sha256",
    "frame_content_sha256",
    "perceptual_feature_sha256",
    "actor_feature_sha256",
    "scene_feature_sha256",
    "count_frames_verified",
    "decoded_to_eof",
}


class FullVideoDecoder(Protocol):
    """Test-only decoder seam; its evidence is permanently non-formal."""

    def __call__(self, path: Path) -> Mapping[str, Any]:
        ...


class ExternalSignatureVerifier(Protocol):
    """Independent trust service used by the formal path.

    The implementation and its root/key provisioning live outside the evidence
    wrapper.  A verifier must expose the already-precommitted trust-root digest
    and key identifier; merely changing a manifest and its caller SHA cannot
    produce a valid detached signature.
    """

    trust_root_sha256: str
    key_id: str

    def verify(self, message: bytes, signature_hex: str) -> bool:
        ...


class TrustedTimestampVerifier(Protocol):
    """Externally provisioned verifier for cryptographic TSA tokens."""

    tsa_id: str
    trust_root_sha256: str

    def verify(
        self, message: bytes, *, signed_at_utc: str, payload_sha256: str
    ) -> bool:
        ...


class OpenSSLExternalSignatureVerifier:
    """Production detached-signature verifier backed by a pinned public key."""

    def __init__(self, *, public_key_path: str, expected_sha256: str, key_id: str):
        path, payload = _read_plain_file(
            public_key_path,
            expected_sha256=expected_sha256,
            expected_size=None,
            label="precommitted signature public key",
            require_read_only=True,
            expected_mode=0o444,
        )
        self.public_key_path = path
        self.trust_root_sha256 = hashlib.sha256(payload).hexdigest()
        self.key_id = _identifier(key_id, label="precommitted signature key id")
        self._openssl = shutil.which("openssl")
        if self._openssl is None:
            raise CheckpointEvaluationSelectionError(
                "formal signature verification requires openssl"
            )

    def verify(self, message: bytes, signature_hex: str) -> bool:
        try:
            signature = bytes.fromhex(signature_hex)
        except ValueError:
            return False
        with tempfile.NamedTemporaryFile(prefix="bernini-signature-", mode="wb") as sig:
            with tempfile.NamedTemporaryFile(prefix="bernini-message-", mode="wb") as msg:
                sig.write(signature)
                sig.flush()
                msg.write(message)
                msg.flush()
                try:
                    completed = subprocess.run(
                        [
                            self._openssl,
                            "dgst",
                            "-sha256",
                            "-verify",
                            str(self.public_key_path),
                            "-signature",
                            sig.name,
                            msg.name,
                        ],
                        check=False,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    return False
        return completed.returncode == 0 and completed.stdout == b"Verified OK\n"


class OpenSSLRFC3161TimestampVerifier:
    """Production RFC-3161 verifier with a code-policy-pinned token registry."""

    def __init__(
        self,
        *,
        tsa_id: str,
        tsa_ca_path: str,
        tsa_ca_sha256: str,
        token_registry_path: str,
        token_registry_sha256: str,
    ) -> None:
        ca_path, ca_payload = _read_plain_file(
            tsa_ca_path,
            expected_sha256=tsa_ca_sha256,
            expected_size=None,
            label="trusted timestamp CA",
            require_read_only=True,
            expected_mode=0o444,
        )
        registry, registry_sha = _read_external_manifest(
            {"path": token_registry_path, "file_sha256": token_registry_sha256},
            label="trusted timestamp token registry",
        )
        registry = _closed(
            registry,
            {"schema_version", "tsa_id", "tokens", "registry_digest"},
            label="trusted timestamp token registry",
        )
        if (
            registry["schema_version"]
            != "bernini-action-editing-rfc3161-token-registry-0817-v1"
            or registry["tsa_id"] != tsa_id
            or not isinstance(registry["tokens"], list)
            or not registry["tokens"]
        ):
            raise CheckpointEvaluationSelectionError(
                "trusted timestamp token registry differs"
            )
        tokens = {}
        for index, item in enumerate(registry["tokens"]):
            current = _closed(
                item,
                {
                    "message_sha256",
                    "payload_sha256",
                    "signed_at_utc",
                    "token_path",
                    "token_sha256",
                },
                label="trusted timestamp token %d" % index,
            )
            for field in ("message_sha256", "payload_sha256", "token_sha256"):
                _sha(current[field], label="timestamp token %s" % field)
            _utc_timestamp(current["signed_at_utc"], label="timestamp token time")
            if not isinstance(current["token_path"], str) or not Path(
                current["token_path"]
            ).is_absolute():
                raise CheckpointEvaluationSelectionError(
                    "timestamp token path differs"
                )
            key = (
                current["message_sha256"],
                current["payload_sha256"],
                current["signed_at_utc"],
            )
            if key in tokens:
                raise CheckpointEvaluationSelectionError(
                    "timestamp token registry key collides"
                )
            tokens[key] = current
        _verify_digest(registry, field="registry_digest", label="timestamp registry")
        self.tsa_id = _identifier(tsa_id, label="trusted timestamp authority id")
        self.trust_root_sha256 = hashlib.sha256(ca_payload).hexdigest()
        self.registry_manifest_sha256 = registry_sha
        self._ca_path = ca_path
        self._tokens = tokens
        self._openssl = shutil.which("openssl")
        if self._openssl is None:
            raise CheckpointEvaluationSelectionError(
                "formal timestamp verification requires openssl"
            )

    def verify(
        self, message: bytes, *, signed_at_utc: str, payload_sha256: str
    ) -> bool:
        key = (hashlib.sha256(message).hexdigest(), payload_sha256, signed_at_utc)
        token = self._tokens.get(key)
        if token is None:
            return False
        try:
            token_path, _ = _read_plain_file(
                token["token_path"],
                expected_sha256=token["token_sha256"],
                expected_size=None,
                label="RFC-3161 timestamp response",
                require_read_only=True,
                expected_mode=0o444,
                return_payload=False,
            )
            with tempfile.NamedTemporaryFile(
                prefix="bernini-tsa-message-", mode="wb"
            ) as message_file:
                message_file.write(message)
                message_file.flush()
                completed = subprocess.run(
                    [
                        self._openssl,
                        "ts",
                        "-verify",
                        "-data",
                        message_file.name,
                        "-in",
                        str(token_path),
                        "-CAfile",
                        str(self._ca_path),
                    ],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
        except (OSError, subprocess.TimeoutExpired, CheckpointEvaluationSelectionError):
            return False
        return completed.returncode == 0 and b"Verification: OK" in completed.stdout


def _normalized_media_signature(value: Any, *, label: str) -> dict[str, Any]:
    row = _closed(value, MEDIA_SIGNATURE_FIELDS, label=label)
    for field in (
        "frame_count",
        "fps_num",
        "fps_den",
        "width",
        "height",
        "duration_num",
        "duration_den",
        "pts_start_num",
        "pts_start_den",
        "pts_end_num",
        "pts_end_den",
    ):
        if type(row[field]) is not int or row[field] < 0:
            raise CheckpointEvaluationSelectionError(
                "%s integer geometry differs" % label
            )
    for denominator in (
        "fps_den",
        "duration_den",
        "pts_start_den",
        "pts_end_den",
    ):
        if row[denominator] <= 0:
            raise CheckpointEvaluationSelectionError(
                "%s rational denominator differs" % label
            )
    if row["width"] <= 0 or row["height"] <= 0:
        raise CheckpointEvaluationSelectionError("%s geometry differs" % label)
    _identifier(row["pixel_format"], label="%s pixel format" % label)
    _sha(row["pts_sha256"], label="%s PTS digest" % label)
    _sha(row["frame_content_sha256"], label="%s frame digest" % label)
    _sha(row["perceptual_feature_sha256"], label="%s perceptual feature" % label)
    _sha(row["actor_feature_sha256"], label="%s actor feature" % label)
    _sha(row["scene_feature_sha256"], label="%s scene feature" % label)
    if (
        row["count_frames_verified"] is not True
        or row["decoded_to_eof"] is not True
    ):
        raise CheckpointEvaluationSelectionError(
            "%s was not count_frames + EOF verified" % label
        )
    for numerator, denominator in (
        ("fps_num", "fps_den"),
        ("duration_num", "duration_den"),
        ("pts_start_num", "pts_start_den"),
        ("pts_end_num", "pts_end_den"),
    ):
        fraction = Fraction(row[numerator], row[denominator])
        if (
            fraction.numerator != row[numerator]
            or fraction.denominator != row[denominator]
        ):
            raise CheckpointEvaluationSelectionError(
                "%s rational is not reduced" % label
            )
    return row


def _source_semantics_from_actual_media(media: Mapping[str, Any]) -> dict[str, str]:
    """Recompute media fingerprints, never statistical cluster labels.

    Cluster labels are study-design metadata and must come from the separately
    frozen collection/actor/scene/equivalence authority.  Deriving them from a
    content hash modulo a constant creates pseudo-clusters, not independent
    sampling units.
    """

    perceptual = _sha(
        media["perceptual_feature_sha256"], label="actual perceptual feature"
    )
    actor = _sha(media["actor_feature_sha256"], label="actual actor feature")
    scene = _sha(media["scene_feature_sha256"], label="actual scene feature")
    return {
        "source_perceptual_fingerprint_sha256": perceptual,
        "source_equivalence_group_id": "source-equivalence-" + perceptual,
        "actor_identity_fingerprint_sha256": actor,
        "scene_fingerprint_sha256": scene,
    }


def _read_exact_or_eof(handle: Any, byte_count: int) -> bytes:
    blocks = []
    remaining = byte_count
    while remaining:
        block = handle.read(remaining)
        if not block:
            break
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def _new_frame_feature_hashers() -> tuple[Any, Any, Any]:
    return (
        hashlib.sha256((FROZEN_PERCEPTUAL_ALGORITHM + "\0").encode("ascii")),
        hashlib.sha256((FROZEN_ACTOR_ALGORITHM + "\0").encode("ascii")),
        hashlib.sha256((FROZEN_SCENE_ALGORITHM + "\0").encode("ascii")),
    )


def _update_frame_features(
    *,
    payload: bytes,
    frame_index: int,
    width: int,
    height: int,
    hashers: tuple[Any, Any, Any],
) -> None:
    """Apply the fixed, code-owned RGB feature algorithms to one decoded frame."""

    perceptual, actor, scene = hashers
    index_bytes = frame_index.to_bytes(8, "big")
    perceptual.update(index_bytes)
    stride = max(1, len(payload) // 4096)
    perceptual.update(payload[::stride][:4096])

    row_bytes = width * 3
    x0 = width // 4
    x1 = width - x0
    y0 = height // 4
    y1 = height - y0
    actor.update(index_bytes)
    scene.update(index_bytes)
    for y in range(height):
        row = payload[y * row_bytes : (y + 1) * row_bytes]
        if y0 <= y < y1:
            center = row[x0 * 3 : x1 * 3]
            center_stride = max(1, len(center) // 64)
            actor.update(center[::center_stride][:64])
        if y < y0 or y >= y1:
            border = row
        else:
            border = row[: x0 * 3] + row[x1 * 3 :]
        border_stride = max(1, len(border) // 64)
        scene.update(border[::border_stride][:64])


def _loaded_algorithm_code_digest() -> str:
    functions = (
        _new_frame_feature_hashers,
        _update_frame_features,
        _source_semantics_from_actual_media,
    )
    specs = []
    for function in functions:
        code = function.__code__
        specs.append(
            {
                "name": function.__name__,
                "bytecode_hex": code.co_code.hex(),
                "constants_repr": repr(code.co_consts),
                "names": list(code.co_names),
                "varnames": list(code.co_varnames),
            }
        )
    return object_sha256(
        {
            "schema_version": "frozen-decoded-frame-algorithm-code-0817-v1",
            "algorithm_ids": [
                FROZEN_PERCEPTUAL_ALGORITHM,
                FROZEN_ACTOR_ALGORITHM,
                FROZEN_SCENE_ALGORITHM,
                FROZEN_CLUSTER_ALGORITHM,
            ],
            "implementations": specs,
        }
    )


FROZEN_ALGORITHM_SHA256 = _loaded_algorithm_code_digest()


def ffmpeg_full_video_decode(path: Path) -> dict[str, Any]:
    """Count one default MP4 video stream, then decode it to physical EOF."""

    ffprobe = _FIXED_FFPROBE_EXECUTABLE
    ffmpeg = _FIXED_FFMPEG_EXECUTABLE
    if ffprobe is None or ffmpeg is None:
        raise CheckpointEvaluationSelectionError(
            "production full-video verification requires ffprobe and ffmpeg"
        )
    probe_command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            probe_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=MEDIA_TOOL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckpointEvaluationSelectionError(
            "ffprobe full-frame count failed"
        ) from error
    if completed.returncode != 0:
        raise CheckpointEvaluationSelectionError(
            "ffprobe full-frame count failed: %s"
            % completed.stderr.decode("utf-8", errors="replace")[:500]
        )
    if len(completed.stdout) > MAX_FFPROBE_JSON_BYTES:
        raise CheckpointEvaluationSelectionError("ffprobe output exceeds safe bound")
    probe = _strict_json_object_without_canonical_requirement(
        completed.stdout, label="ffprobe output"
    )
    streams = probe.get("streams")
    media_format = probe.get("format")
    if not isinstance(streams, list) or not streams:
        raise CheckpointEvaluationSelectionError("stream closure differs")
    video_streams = [
        item
        for item in streams
        if isinstance(item, Mapping) and item.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise CheckpointEvaluationSelectionError(
            "MP4 must contain exactly one video stream"
        )
    if (
        not isinstance(media_format, Mapping)
        or not isinstance(media_format.get("format_name"), str)
        or not {"mov", "mp4"}.intersection(
            str(media_format["format_name"]).split(",")
        )
    ):
        raise CheckpointEvaluationSelectionError("media container is not MP4")
    stream = video_streams[0]
    if not isinstance(stream, Mapping):
        raise CheckpointEvaluationSelectionError("video stream metadata differs")
    try:
        stream_index = int(stream["index"])
        width = int(stream["width"])
        height = int(stream["height"])
        pixel_format = str(stream["pix_fmt"])
        fps = Fraction(str(stream["avg_frame_rate"]))
        time_base = Fraction(str(stream["time_base"]))
        counted = int(stream["nb_read_frames"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise CheckpointEvaluationSelectionError(
            "video stream count/geometry metadata differs"
        ) from error
    disposition = stream.get("disposition")
    if not isinstance(disposition, Mapping) or disposition.get("default") != 1:
        raise CheckpointEvaluationSelectionError(
            "the sole video stream is not the default stream"
        )
    if (
        counted != 81
        or width <= 0
        or height <= 0
        or width > 8192
        or height > 8192
        or fps <= 0
        or time_base <= 0
    ):
        raise CheckpointEvaluationSelectionError(
            "ffprobe count_frames/geometry closure differs"
        )
    frame_probe_command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=media_type,stream_index,best_effort_timestamp,pts,width,height,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    try:
        frame_completed = subprocess.run(
            frame_probe_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=MEDIA_TOOL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckpointEvaluationSelectionError(
            "ffprobe full-frame enumeration failed"
        ) from error
    if frame_completed.returncode != 0:
        raise CheckpointEvaluationSelectionError(
            "ffprobe full-frame enumeration failed: %s"
            % frame_completed.stderr.decode("utf-8", errors="replace")[:500]
        )
    if len(frame_completed.stdout) > MAX_FFPROBE_JSON_BYTES:
        raise CheckpointEvaluationSelectionError(
            "ffprobe frame enumeration exceeds safe bound"
        )
    frame_probe = _strict_json_object_without_canonical_requirement(
        frame_completed.stdout, label="ffprobe frame output"
    )
    frames = frame_probe.get("frames")
    if not isinstance(frames, list) or not frames:
        raise CheckpointEvaluationSelectionError("decoded frame list is empty")
    video_frames = [
        item
        for item in frames
        if isinstance(item, Mapping) and item.get("media_type") == "video"
    ]
    if any(int(item.get("stream_index", -1)) != stream_index for item in video_frames):
        raise CheckpointEvaluationSelectionError("video frame stream identity differs")
    if (
        counted != len(video_frames)
        or counted != 81
        or width <= 0
        or height <= 0
        or width > 8192
        or height > 8192
        or fps <= 0
        or time_base <= 0
    ):
        raise CheckpointEvaluationSelectionError(
            "ffprobe count_frames/geometry closure differs"
        )
    try:
        stream_duration = int(stream["duration_ts"]) * time_base
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointEvaluationSelectionError(
            "video stream duration authority is missing"
        ) from error
    if stream_duration != Fraction(counted, 1) / fps:
        raise CheckpointEvaluationSelectionError(
            "video stream duration differs from decoded frame cadence"
        )
    pts_values = []
    for index, frame in enumerate(video_frames):
        if not isinstance(frame, Mapping) or frame.get("media_type") != "video":
            raise CheckpointEvaluationSelectionError("non-video frame differs")
        try:
            pts_raw = int(
                frame.get("best_effort_timestamp", frame.get("pts"))
            )
        except (TypeError, ValueError) as error:
            raise CheckpointEvaluationSelectionError("frame PTS is missing") from error
        if (
            int(frame.get("width", width)) != width
            or int(frame.get("height", height)) != height
            or str(frame.get("pix_fmt", pixel_format)) != pixel_format
        ):
            raise CheckpointEvaluationSelectionError(
                "per-frame geometry/pixel format differs"
            )
        timestamp = pts_raw * time_base
        if timestamp != Fraction(index, 1) / fps:
            raise CheckpointEvaluationSelectionError(
                "frame PTS does not start at zero with exact constant FPS cadence"
            )
        pts_values.append(
            {
                "index": index,
                "pts_num": timestamp.numerator,
                "pts_den": timestamp.denominator,
            }
        )
    frame_bytes = width * height * 3
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:%d" % stream_index,
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-frames:v",
        str(counted + 1),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    expected_rgb_bytes = frame_bytes * counted
    if expected_rgb_bytes > MAX_DECODED_RGB_BYTES:
        raise CheckpointEvaluationSelectionError("decoded RGB payload exceeds safe bound")
    frame_digest = hashlib.sha256()
    feature_hashers = _new_frame_feature_hashers()
    decoded = 0
    with tempfile.TemporaryFile(prefix="bernini-full81-rgb-") as decoded_file:
        try:
            completed_decode = subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=decoded_file,
                stderr=subprocess.PIPE,
                timeout=MEDIA_TOOL_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CheckpointEvaluationSelectionError(
                "ffmpeg bounded frame-by-frame decode failed"
            ) from error
        if completed_decode.returncode != 0:
            raise CheckpointEvaluationSelectionError(
                "ffmpeg frame-by-frame decode failed: %s"
                % completed_decode.stderr.decode("utf-8", errors="replace")[:500]
            )
        decoded_size = os.fstat(decoded_file.fileno()).st_size
        if decoded_size > expected_rgb_bytes + frame_bytes:
            raise CheckpointEvaluationSelectionError(
                "ffmpeg decoded output exceeds the bounded EOF probe"
            )
        decoded_file.seek(0)
        while True:
            payload = _read_exact_or_eof(decoded_file, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise CheckpointEvaluationSelectionError(
                    "ffmpeg ended inside a decoded frame"
                )
            frame_digest.update(decoded.to_bytes(8, "big"))
            frame_digest.update(len(payload).to_bytes(8, "big"))
            frame_digest.update(payload)
            _update_frame_features(
                payload=payload,
                frame_index=decoded,
                width=width,
                height=height,
                hashers=feature_hashers,
            )
            decoded += 1
            if decoded > counted:
                raise CheckpointEvaluationSelectionError(
                    "ffmpeg decoded more frames than ffprobe count_frames"
                )
    if decoded != counted:
        raise CheckpointEvaluationSelectionError(
            "ffmpeg EOF count differs from ffprobe count_frames"
        )
    duration = Fraction(decoded, 1) / fps
    end = pts_values[-1]
    return {
        "frame_count": decoded,
        "fps_num": fps.numerator,
        "fps_den": fps.denominator,
        "width": width,
        "height": height,
        "pixel_format": pixel_format,
        "duration_num": duration.numerator,
        "duration_den": duration.denominator,
        "pts_start_num": 0,
        "pts_start_den": 1,
        "pts_end_num": end["pts_num"],
        "pts_end_den": end["pts_den"],
        "pts_sha256": object_sha256(pts_values),
        "frame_content_sha256": frame_digest.hexdigest(),
        "perceptual_feature_sha256": feature_hashers[0].hexdigest(),
        "actor_feature_sha256": feature_hashers[1].hexdigest(),
        "scene_feature_sha256": feature_hashers[2].hexdigest(),
        "count_frames_verified": True,
        "decoded_to_eof": True,
    }


# Bind the formal decoder once.  Replacing the module attribute later (a common
# unit-test technique) cannot alter the decoder used by formal validation.
# Captured once at module load.  The public ``ffmpeg_full_video_decode`` name
# may be monkeypatched by engineering tests, but the fixed CLI path never
# resolves that mutable name again.
_SEALED_PRODUCTION_MEDIA_DECODER: FullVideoDecoder = ffmpeg_full_video_decode


def _strict_json_object_without_canonical_requirement(
    payload: bytes, *, label: str
) -> dict[str, Any]:
    """Strictly parse trusted tool JSON; ffprobe controls its own formatting."""

    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise CheckpointEvaluationSelectionError(
                    "%s contains a duplicate key" % label
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointEvaluationSelectionError("%s is not strict JSON" % label) from error
    if not isinstance(value, Mapping):
        raise CheckpointEvaluationSelectionError("%s root is not an object" % label)
    return dict(value)


def _verify_actual_media(
    path: Path,
    expected: Any,
    *,
    decoder: FullVideoDecoder,
    expected_file_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected_row = _normalized_media_signature(expected, label="%s expected" % label)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_mode & 0o222
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise CheckpointEvaluationSelectionError(
            "%s must be an immutable single-link media file" % label
        )
    before_hash = file_sha256(path)
    if before_hash != expected_file_sha256:
        raise CheckpointEvaluationSelectionError("%s pre-decode SHA differs" % label)
    try:
        decoded_value = decoder(path)
    except CheckpointEvaluationSelectionError:
        raise
    except Exception as error:
        raise CheckpointEvaluationSelectionError(
            "%s decoder failed: %s" % (label, error)
        ) from error
    actual = _normalized_media_signature(decoded_value, label="%s actual" % label)
    after = path.lstat()
    after_hash = file_sha256(path)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_uid,
        item.st_gid,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or after_hash != before_hash:
        raise CheckpointEvaluationSelectionError(
            "%s inode/hash changed across decode" % label
        )
    if actual != expected_row:
        raise CheckpointEvaluationSelectionError(
            "%s actual complete decode differs from frozen signature" % label
        )
    if (
        actual["frame_count"] != 81
        or Fraction(actual["fps_num"], actual["fps_den"]) != 25
        or Fraction(actual["duration_num"], actual["duration_den"])
        != Fraction(81, 25)
        or Fraction(actual["pts_start_num"], actual["pts_start_den"]) != 0
        or Fraction(actual["pts_end_num"], actual["pts_end_den"])
        != Fraction(80, 25)
    ):
        raise CheckpointEvaluationSelectionError(
            "%s is not exact full81/25fps/PTS" % label
        )
    return actual


def _read_canonical_receipt(
    path: Any, *, expected_sha256: Any, label: str
) -> dict[str, Any]:
    immutable_formal = expected_sha256 not in KNOWN_PRE_D0_RECEIPTS
    _, payload = _read_plain_file(
        path,
        expected_sha256=expected_sha256,
        expected_size=None,
        label=label,
        require_read_only=immutable_formal,
        expected_mode=0o444 if immutable_formal else None,
    )
    value = _strict_json_bytes(payload, label=label)
    _verify_digest(value, field="receipt_digest", label=label)
    return value


def _closed(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CheckpointEvaluationSelectionError("%s field closure differs" % label)
    return dict(value)


def _verify_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha(value.get(field), label="%s digest" % label)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if object_sha256(unsigned) != digest:
        raise CheckpointEvaluationSelectionError("%s digest differs" % label)
    return digest


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CheckpointEvaluationSelectionError(
            "%s is not a lowercase SHA-256" % label
        )
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise CheckpointEvaluationSelectionError("%s is invalid" % label)
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise CheckpointEvaluationSelectionError("%s is not boolean" % label)
    return value


def _unit(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointEvaluationSelectionError("%s is not numeric" % label)
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CheckpointEvaluationSelectionError("%s lies outside [0,1]" % label)
    return result


def artifact_tree_digest(checkpoint_id: str, files: Sequence[Mapping[str, Any]]) -> str:
    return object_sha256(
        {
            "schema_version": "bernini-action-editing-artifact-tree-content-0817-v2",
            "checkpoint_id": checkpoint_id,
            "files": list(files),
        }
    )


def release_tree_digest(files: Sequence[Mapping[str, Any]]) -> str:
    return object_sha256(
        {
            "schema_version": "bernini-action-editing-code-release-content-0817-v2",
            "files": list(files),
        }
    )


def _validate_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CheckpointEvaluationSelectionError("%s differs" % label)
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise CheckpointEvaluationSelectionError("%s is not normalized relative" % label)
    if any(part in {"", "."} for part in path.parts):
        raise CheckpointEvaluationSelectionError("%s is not normalized relative" % label)
    return value


def _immutable_directory(value: Any, *, label: str) -> tuple[Path, os.stat_result]:
    if not isinstance(value, str):
        raise CheckpointEvaluationSelectionError("%s path differs" % label)
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value or value == os.path.sep:
        raise CheckpointEvaluationSelectionError(
            "%s path must be absolute and normalized" % label
        )
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise CheckpointEvaluationSelectionError("%s is missing" % label) from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise CheckpointEvaluationSelectionError("%s is not a plain directory" % label)
    try:
        if path.resolve(strict=True) != path:
            raise CheckpointEvaluationSelectionError(
                "%s path contains a symlink/non-canonical component" % label
            )
    except OSError as error:
        raise CheckpointEvaluationSelectionError(
            "%s canonical path resolution failed" % label
        ) from error
    if info.st_mode & 0o222 or stat.S_IMODE(info.st_mode) != 0o555:
        raise CheckpointEvaluationSelectionError(
            "%s is not an immutable mode-0555 directory" % label
        )
    return path, info


def _directory_identity(info: os.stat_result) -> tuple[Any, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _enumerate_plain_tree(root: Path, *, label: str) -> list[str]:
    result = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise CheckpointEvaluationSelectionError(
                    "%s contains a non-plain directory" % label
                )
            if info.st_mode & 0o222:
                raise CheckpointEvaluationSelectionError(
                    "%s contains a writable directory" % label
                )
            if stat.S_IMODE(info.st_mode) != 0o555:
                raise CheckpointEvaluationSelectionError(
                    "%s contains a directory with noncanonical mode" % label
                )
        for name in file_names:
            child = directory_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise CheckpointEvaluationSelectionError(
                    "%s contains a non-plain file" % label
                )
            if info.st_nlink != 1:
                raise CheckpointEvaluationSelectionError(
                    "%s contains a hard-linked file" % label
                )
            relative = child.relative_to(root).as_posix()
            result.append(relative)
    return sorted(result)


def _validate_manifest_files(
    *,
    root: Path,
    root_before: os.stat_result,
    files: Any,
    entry_fields: set[str],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(files, list) or not files:
        raise CheckpointEvaluationSelectionError("%s file list is empty" % label)
    normalized = []
    for index, item in enumerate(files):
        current = _closed(item, entry_fields, label="%s file %d" % (label, index))
        relative = _validate_relative_path(
            current["relative_path"], label="%s relative path" % label
        )
        if (
            type(current["size_bytes"]) is not int
            or current["size_bytes"] < 0
        ):
            raise CheckpointEvaluationSelectionError("%s file size differs" % label)
        _sha(current["sha256"], label="%s file SHA" % label)
        if type(current.get("mode")) is not int or current["mode"] not in {
            0o444,
            0o555,
        }:
            raise CheckpointEvaluationSelectionError("%s file mode differs" % label)
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise CheckpointEvaluationSelectionError(
                "%s file escapes root" % label
            ) from error
        _read_plain_file(
            str(path),
            expected_sha256=current["sha256"],
            expected_size=current["size_bytes"],
            label="%s file %s" % (label, relative),
            require_read_only=True,
            expected_mode=current["mode"],
            return_payload=False,
        )
        normalized.append(current)
    paths = [item["relative_path"] for item in normalized]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise CheckpointEvaluationSelectionError(
            "%s paths are not sorted/unique" % label
        )
    if _enumerate_plain_tree(root, label=label) != paths:
        raise CheckpointEvaluationSelectionError(
            "%s manifest does not close the on-disk tree" % label
        )
    root_after = root.lstat()
    if _directory_identity(root_before) != _directory_identity(root_after):
        raise CheckpointEvaluationSelectionError("%s root changed while hashed" % label)
    return normalized


_SAFETENSOR_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _parse_safetensors_semantics(path: Path, *, label: str) -> list[dict[str, Any]]:
    """Parse tensor names/dtypes/shapes/offsets, rather than trusting a suffix."""

    size = path.stat().st_size
    if path.suffix != ".safetensors" or size < 10:
        raise CheckpointEvaluationSelectionError(
            "%s is not a safetensors artifact" % label
        )
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise CheckpointEvaluationSelectionError("%s header is truncated" % label)
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > 64 * 1024 * 1024 or 8 + header_size > size:
            raise CheckpointEvaluationSelectionError("%s header size differs" % label)
        header_payload = handle.read(header_size)
    try:
        def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise CheckpointEvaluationSelectionError(
                        "%s tensor header contains a duplicate key" % label
                    )
                result[key] = value
            return result

        header = json.loads(
            header_payload.decode("utf-8").rstrip(" "),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointEvaluationSelectionError(
            "%s tensor header is invalid" % label
        ) from error
    if not isinstance(header, Mapping):
        raise CheckpointEvaluationSelectionError("%s tensor header differs" % label)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise CheckpointEvaluationSelectionError("%s metadata differs" % label)
    tensors = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(value, Mapping):
            raise CheckpointEvaluationSelectionError("%s tensor entry differs" % label)
        current = _closed(value, {"dtype", "shape", "data_offsets"}, label=label)
        dtype = current["dtype"]
        shape = current["shape"]
        offsets = current["data_offsets"]
        if (
            dtype not in _SAFETENSOR_DTYPE_BYTES
            or not isinstance(shape, list)
            or any(type(item) is not int or item < 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int or item < 0 for item in offsets)
            or offsets[1] < offsets[0]
        ):
            raise CheckpointEvaluationSelectionError("%s tensor semantics differ" % label)
        element_count = math.prod(shape)
        if offsets[1] - offsets[0] != element_count * _SAFETENSOR_DTYPE_BYTES[dtype]:
            raise CheckpointEvaluationSelectionError("%s tensor byte span differs" % label)
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets,
            }
        )
    if not tensors:
        raise CheckpointEvaluationSelectionError("%s carries no tensors" % label)
    by_offset = sorted(tensors, key=lambda item: (item["data_offsets"], item["name"]))
    cursor = 0
    for item in by_offset:
        if item["data_offsets"][0] != cursor:
            raise CheckpointEvaluationSelectionError(
                "%s tensor payload has a gap/overlap" % label
            )
        cursor = item["data_offsets"][1]
    if 8 + header_size + cursor != size:
        raise CheckpointEvaluationSelectionError(
            "%s tensor payload does not close the artifact" % label
        )
    return sorted(tensors, key=lambda item: item["name"])


def _tensor_semantic_digest(
    root: Path, weight_entries: Sequence[Mapping[str, Any]], *, label: str
) -> str:
    shards = []
    tensor_names: set[str] = set()
    for item in sorted(weight_entries, key=lambda row: row["shard_index"]):
        path = root.joinpath(*PurePosixPath(item["relative_path"]).parts)
        tensors = _parse_safetensors_semantics(path, label=label)
        names = {tensor["name"] for tensor in tensors}
        if tensor_names.intersection(names):
            raise CheckpointEvaluationSelectionError(
                "%s tensor names collide across shards" % label
            )
        tensor_names.update(names)
        shards.append(
            {
                "relative_path": item["relative_path"],
                "file_sha256": item["sha256"],
                "tensors": tensors,
            }
        )
    return object_sha256(
        {"schema_version": SAFE_TENSOR_SEMANTIC_SCHEMA, "shards": shards}
    )


def _load_safetensors_state(
    root: Path, weight_entries: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[list[dict[str, Any]], str]:
    """Load every tensor payload and return its exact state contract view.

    Header-only parsing is insufficient: a crafted artifact can advertise the
    expected keys/shapes while carrying unread, truncated, or substituted
    payload bytes.  This loader seeks to every advertised span, consumes it to
    EOF, and binds a content digest to each state key.  It intentionally uses
    the safetensors wire format directly so the audit does not depend on torch
    or an optional safetensors wheel on the evaluation host.
    """

    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard in sorted(weight_entries, key=lambda row: row["shard_index"]):
        path = root.joinpath(*PurePosixPath(shard["relative_path"]).parts)
        tensors = _parse_safetensors_semantics(path, label=label)
        with path.open("rb") as handle:
            prefix = _read_exact_or_eof(handle, 8)
            if len(prefix) != 8:
                raise CheckpointEvaluationSelectionError(
                    "%s state shard header is truncated" % label
                )
            header_size = struct.unpack("<Q", prefix)[0]
            payload_start = 8 + header_size
            for tensor in tensors:
                name = tensor["name"]
                if name in seen:
                    raise CheckpointEvaluationSelectionError(
                        "%s state key collides across shards" % label
                    )
                seen.add(name)
                start, end = tensor["data_offsets"]
                handle.seek(payload_start + start)
                remaining = end - start
                digest = hashlib.sha256()
                consumed = 0
                while remaining:
                    block = handle.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise CheckpointEvaluationSelectionError(
                            "%s state tensor payload is truncated" % label
                        )
                    digest.update(block)
                    consumed += len(block)
                    remaining -= len(block)
                loaded.append(
                    {
                        "name": name,
                        "dtype": tensor["dtype"],
                        "shape": tensor["shape"],
                        "parameter_count": math.prod(tensor["shape"]),
                        "loaded_byte_count": consumed,
                        "content_sha256": digest.hexdigest(),
                    }
                )
    loaded.sort(key=lambda item: item["name"])
    if not loaded:
        raise CheckpointEvaluationSelectionError("%s loaded state is empty" % label)
    state_digest = object_sha256(
        {
            "schema_version": "loaded-safetensors-state-closure-0817-v1",
            "tensors": loaded,
        }
    )
    return loaded, state_digest


def _validate_model_state_contract(
    value: Any,
    *,
    loaded: Sequence[Mapping[str, Any]],
    loaded_state_digest: str,
    label: str,
) -> dict[str, Any]:
    contract = _closed(
        value,
        {
            "schema_version",
            "architecture_id",
            "expected_tensors",
            "expected_tensor_count",
            "expected_parameter_count",
            "required_prefixes",
            "required_coverage",
            "loaded_state_digest",
            "contract_digest",
        },
        label="%s model-state contract" % label,
    )
    if contract["schema_version"] != MODEL_STATE_CONTRACT_SCHEMA:
        raise CheckpointEvaluationSelectionError(
            "%s model-state contract schema differs" % label
        )
    _identifier(contract["architecture_id"], label="%s architecture id" % label)
    expected = contract["expected_tensors"]
    if not isinstance(expected, list) or not expected:
        raise CheckpointEvaluationSelectionError(
            "%s expected model state is empty" % label
        )
    normalized = []
    for index, item in enumerate(expected):
        current = _closed(
            item,
            {"name", "dtype", "shape", "parameter_count", "content_sha256"},
            label="%s expected state tensor %d" % (label, index),
        )
        if (
            not isinstance(current["name"], str)
            or not current["name"]
            or current["dtype"] not in _SAFETENSOR_DTYPE_BYTES
            or not isinstance(current["shape"], list)
            or any(type(size) is not int or size < 0 for size in current["shape"])
            or type(current["parameter_count"]) is not int
            or current["parameter_count"] != math.prod(current["shape"])
        ):
            raise CheckpointEvaluationSelectionError(
                "%s expected model-state tensor differs" % label
            )
        _sha(current["content_sha256"], label="%s tensor content" % label)
        normalized.append(current)
    if [item["name"] for item in normalized] != sorted(
        {item["name"] for item in normalized}
    ):
        raise CheckpointEvaluationSelectionError(
            "%s expected state keys are not sorted/unique" % label
        )
    loaded_view = [
        {
            key: item[key]
            for key in ("name", "dtype", "shape", "parameter_count", "content_sha256")
        }
        for item in loaded
    ]
    prefixes = contract["required_prefixes"]
    if (
        not isinstance(prefixes, list)
        or not prefixes
        or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)
        or prefixes != sorted(set(prefixes))
    ):
        raise CheckpointEvaluationSelectionError(
            "%s required model-state prefixes differ" % label
        )
    covered = {
        item["name"]
        for item in loaded_view
        if any(item["name"].startswith(prefix) for prefix in prefixes)
    }
    coverage = len(covered) / len(normalized)
    if (
        normalized != loaded_view
        or contract["expected_tensor_count"] != len(normalized)
        or contract["expected_parameter_count"]
        != sum(item["parameter_count"] for item in normalized)
        or contract["required_coverage"] != 1.0
        or coverage != 1.0
        or contract["loaded_state_digest"] != loaded_state_digest
    ):
        raise CheckpointEvaluationSelectionError(
            "%s loaded Bernini state keys/shapes/content/coverage differ" % label
        )
    _verify_digest(contract, field="contract_digest", label="%s model state" % label)
    contract["expected_tensors"] = normalized
    return contract


def _validate_artifact_tree_manifest(
    checkpoint: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    manifest, manifest_file_sha = _read_external_manifest(
        {
            "path": checkpoint["artifact_manifest_path"],
            "file_sha256": checkpoint["artifact_manifest_file_sha256"],
        },
        label="%s artifact manifest" % label,
    )
    manifest = _closed(
        manifest,
        {
            "schema_version",
            "checkpoint_id",
            "artifact_root_path",
            "primary_artifact_relative_path",
            "files",
            "file_count",
            "total_bytes",
            "tree_digest",
            "tensor_semantic_schema",
            "tensor_semantic_digest",
            "model_state_contract",
            "manifest_digest",
        },
        label="%s artifact manifest" % label,
    )
    if (
        manifest["schema_version"] != ARTIFACT_TREE_MANIFEST_SCHEMA
        or manifest["checkpoint_id"] != checkpoint["checkpoint_id"]
        or manifest["artifact_root_path"] != checkpoint["artifact_root_path"]
    ):
        raise CheckpointEvaluationSelectionError(
            "%s artifact manifest identity differs" % label
        )
    root, root_before = _immutable_directory(
        manifest["artifact_root_path"], label="%s artifact root" % label
    )
    files = _validate_manifest_files(
        root=root,
        root_before=root_before,
        files=manifest["files"],
        entry_fields={
            "relative_path",
            "size_bytes",
            "sha256",
            "artifact_role",
            "shard_index",
            "shard_count",
            "mode",
        },
        label="%s artifact tree" % label,
    )
    weight_entries = []
    for item in files:
        if item["artifact_role"] == "weight_shard":
            if (
                type(item["shard_index"]) is not int
                or type(item["shard_count"]) is not int
                or item["shard_count"] < 1
                or not 0 <= item["shard_index"] < item["shard_count"]
            ):
                raise CheckpointEvaluationSelectionError(
                    "%s weight shard metadata differs" % label
                )
            weight_entries.append(item)
        elif item["artifact_role"] == "metadata":
            if item["shard_index"] is not None or item["shard_count"] is not None:
                raise CheckpointEvaluationSelectionError(
                    "%s metadata carries shard authority" % label
                )
        else:
            raise CheckpointEvaluationSelectionError(
                "%s artifact role differs" % label
            )
    if not weight_entries:
        raise CheckpointEvaluationSelectionError("%s has no weight shards" % label)
    shard_counts = {item["shard_count"] for item in weight_entries}
    if len(shard_counts) != 1:
        raise CheckpointEvaluationSelectionError("%s shard counts differ" % label)
    shard_count = next(iter(shard_counts))
    if sorted(item["shard_index"] for item in weight_entries) != list(
        range(shard_count)
    ):
        raise CheckpointEvaluationSelectionError(
            "%s weight shard set is incomplete" % label
        )
    expected_tree = artifact_tree_digest(checkpoint["checkpoint_id"], files)
    semantic_digest = _tensor_semantic_digest(root, weight_entries, label=label)
    loaded_state, loaded_state_digest = _load_safetensors_state(
        root, weight_entries, label=label
    )
    model_state_contract = _validate_model_state_contract(
        manifest["model_state_contract"],
        loaded=loaded_state,
        loaded_state_digest=loaded_state_digest,
        label=label,
    )
    for item in files:
        _read_plain_file(
            str(root.joinpath(*PurePosixPath(item["relative_path"]).parts)),
            expected_sha256=item["sha256"],
            expected_size=item["size_bytes"],
            label="%s post-semantic file %s" % (label, item["relative_path"]),
            require_read_only=True,
            expected_mode=item["mode"],
            return_payload=False,
        )
    primary_path = _validate_relative_path(
        manifest["primary_artifact_relative_path"],
        label="%s primary artifact" % label,
    )
    primary = [item for item in files if item["relative_path"] == primary_path]
    if (
        len(primary) != 1
        or primary[0]["artifact_role"] != "weight_shard"
        or primary[0]["sha256"] != checkpoint["artifact_sha256"]
        or type(manifest["file_count"]) is not int
        or type(manifest["total_bytes"]) is not int
        or manifest["file_count"] != len(files)
        or manifest["total_bytes"] != sum(item["size_bytes"] for item in files)
        or manifest["tree_digest"] != expected_tree
        or checkpoint["artifact_tree_sha256"] != expected_tree
        or checkpoint["artifact_manifest_file_sha256"] != manifest_file_sha
        or manifest["tensor_semantic_schema"] != SAFE_TENSOR_SEMANTIC_SCHEMA
        or manifest["tensor_semantic_digest"] != semantic_digest
        or checkpoint["tensor_semantic_digest"] != semantic_digest
        or checkpoint["model_state_contract_digest"]
        != model_state_contract["contract_digest"]
    ):
        raise CheckpointEvaluationSelectionError(
            "%s artifact tree size/hash closure differs" % label
        )
    _verify_digest(manifest, field="manifest_digest", label="%s artifact manifest" % label)
    manifest["files"] = files
    manifest["model_state_contract"] = model_state_contract
    manifest["loaded_state_digest"] = loaded_state_digest
    return manifest


def _validate_release_manifest(
    *, path: Any, expected_file_sha256: Any, expected_tree_sha256: Any, label: str
) -> dict[str, Any]:
    manifest, manifest_file_sha = _read_external_manifest(
        {"path": path, "file_sha256": expected_file_sha256}, label=label
    )
    manifest = _closed(
        manifest,
        {
            "schema_version",
            "release_id",
            "release_root_path",
            "files",
            "file_count",
            "total_bytes",
            "release_tree_digest",
            "entrypoint_relative_path",
            "entrypoint_sha256",
            "entrypoint_schema",
            "manifest_digest",
        },
        label=label,
    )
    if manifest["schema_version"] != RELEASE_MANIFEST_SCHEMA:
        raise CheckpointEvaluationSelectionError("%s schema differs" % label)
    _identifier(manifest["release_id"], label="%s release id" % label)
    root, root_before = _immutable_directory(
        manifest["release_root_path"], label="%s root" % label
    )
    files = _validate_manifest_files(
        root=root,
        root_before=root_before,
        files=manifest["files"],
        entry_fields={"relative_path", "size_bytes", "sha256", "mode"},
        label=label,
    )
    expected_tree = release_tree_digest(files)
    entrypoint_path = _validate_relative_path(
        manifest["entrypoint_relative_path"], label="%s entrypoint" % label
    )
    entrypoints = [item for item in files if item["relative_path"] == entrypoint_path]
    if len(entrypoints) != 1:
        raise CheckpointEvaluationSelectionError("%s entrypoint is not in release" % label)
    entrypoint = entrypoints[0]
    entrypoint_bytes = root.joinpath(
        *PurePosixPath(entrypoint_path).parts
    ).read_bytes()
    if len(entrypoint_bytes) > MAX_EXECUTABLE_ENTRYPOINT_BYTES:
        raise CheckpointEvaluationSelectionError("%s entrypoint exceeds safe bound" % label)
    try:
        compile(entrypoint_bytes, entrypoint_path, "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        raise CheckpointEvaluationSelectionError(
            "%s entrypoint is not executable Python" % label
        ) from error
    _read_plain_file(
        str(root.joinpath(*PurePosixPath(entrypoint_path).parts)),
        expected_sha256=entrypoint["sha256"],
        expected_size=entrypoint["size_bytes"],
        label="%s post-compile entrypoint" % label,
        require_read_only=True,
        expected_mode=entrypoint["mode"],
        return_payload=False,
    )
    if (
        type(manifest["file_count"]) is not int
        or type(manifest["total_bytes"]) is not int
        or manifest["file_count"] != len(files)
        or manifest["total_bytes"] != sum(item["size_bytes"] for item in files)
        or manifest["release_tree_digest"] != expected_tree
        or expected_tree != _sha(expected_tree_sha256, label="%s expected tree" % label)
        or manifest_file_sha != expected_file_sha256
        or manifest["entrypoint_schema"] != PYTHON_ENTRYPOINT_SCHEMA
        or manifest["entrypoint_sha256"] != entrypoint["sha256"]
        or entrypoint["mode"] != 0o555
    ):
        raise CheckpointEvaluationSelectionError(
            "%s on-disk release closure differs" % label
        )
    _verify_digest(manifest, field="manifest_digest", label=label)
    manifest["files"] = files
    return manifest


def _validate_authority_pins(value: Any) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "authority_root_manifest_path",
            "authority_root_manifest_sha256",
            "authority_root_signature_path",
            "authority_root_signature_sha256",
            "precommitted_trust_root_sha256",
            "precommitted_key_id",
            "trusted_time_authority_id",
            "formal_training_manifest_path",
            "formal_training_manifest_sha256",
            "authorized_checkpoint_ids",
            "locked_split_digest",
            "keeper_commitment_digest",
            "source_equivalence_manifest_path",
            "source_equivalence_manifest_sha256",
            "reviewer_roster_manifest_path",
            "reviewer_roster_manifest_sha256",
            "ballot_seal_manifest_path",
            "ballot_seal_manifest_sha256",
            "ballot_seal_signature_path",
            "ballot_seal_signature_sha256",
            "inference_release_manifest_path",
            "inference_release_manifest_sha256",
            "renderer_execution_manifest_path",
            "renderer_execution_manifest_sha256",
            "renderer_execution_signature_path",
            "renderer_execution_signature_sha256",
        },
        label="caller authority pins",
    )
    for prefix in (
        "authority_root",
        "formal_training",
        "source_equivalence",
        "reviewer_roster",
        "ballot_seal",
        "inference_release",
    ):
        path = row["%s_manifest_path" % prefix]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise CheckpointEvaluationSelectionError(
                "caller %s authority path differs" % prefix
            )
        _sha(
            row["%s_manifest_sha256" % prefix],
            label="caller %s authority" % prefix,
        )
    for prefix in ("authority_root_signature", "ballot_seal_signature"):
        path = row["%s_path" % prefix]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise CheckpointEvaluationSelectionError(
                "caller %s path differs" % prefix
            )
        _sha(row["%s_sha256" % prefix], label="caller %s" % prefix)
    execution_values = (
        row["renderer_execution_manifest_path"],
        row["renderer_execution_manifest_sha256"],
        row["renderer_execution_signature_path"],
        row["renderer_execution_signature_sha256"],
    )
    if any(value is not None for value in execution_values):
        if any(value is None for value in execution_values):
            raise CheckpointEvaluationSelectionError(
                "renderer execution authority descriptor is partial"
            )
        for field in (
            "renderer_execution_manifest_path",
            "renderer_execution_signature_path",
        ):
            if not isinstance(row[field], str) or not Path(row[field]).is_absolute():
                raise CheckpointEvaluationSelectionError(
                    "renderer execution authority path differs"
                )
        for field in (
            "renderer_execution_manifest_sha256",
            "renderer_execution_signature_sha256",
        ):
            _sha(row[field], label="renderer execution authority")
    _sha(
        row["precommitted_trust_root_sha256"],
        label="precommitted verifier trust root",
    )
    _identifier(row["precommitted_key_id"], label="precommitted signing key")
    _identifier(row["trusted_time_authority_id"], label="trusted time authority")
    members = row["authorized_checkpoint_ids"]
    if not isinstance(members, list) or not members:
        raise CheckpointEvaluationSelectionError(
            "caller checkpoint authority closure is empty"
        )
    for item in members:
        _identifier(item, label="caller authorized checkpoint")
    if members != sorted(set(members)):
        raise CheckpointEvaluationSelectionError(
            "caller checkpoint authority closure is not sorted/unique"
        )
    if row["locked_split_digest"] is not None:
        _sha(row["locked_split_digest"], label="caller locked split")
    _sha(row["keeper_commitment_digest"], label="caller keeper commitment")
    return row


def _validate_formal_training_authority(
    pins: Mapping[str, Any], *, authority_root: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, file_sha = _read_external_manifest(
        {
            "path": pins["formal_training_manifest_path"],
            "file_sha256": pins["formal_training_manifest_sha256"],
        },
        label="formal training authority manifest",
    )
    manifest = _closed(
        manifest,
        {
            "schema_version",
            "authority_id",
            "status",
            "issued_before_outputs",
            "known_pre_d0_receipts",
            "checkpoints",
            "authority_digest",
        },
        label="formal training authority manifest",
    )
    if (
        file_sha != authority_root["formal_training_manifest_file_sha256"]
        or manifest["schema_version"] != FORMAL_AUTHORITY_SCHEMA
        or manifest["status"] != "locked"
        or manifest["issued_before_outputs"] is not True
    ):
        raise CheckpointEvaluationSelectionError(
            "formal training authority is not locked before outputs"
        )
    _identifier(manifest["authority_id"], label="formal authority id")
    registrations = manifest["known_pre_d0_receipts"]
    if not isinstance(registrations, list):
        raise CheckpointEvaluationSelectionError("PRE_D0 receipt registry differs")
    normalized_registrations = []
    for index, item in enumerate(registrations):
        current = _closed(
            item,
            {
                "training_receipt_file_sha256",
                "receipt_schema",
                "classification",
                "training_stage",
                "checkpoint_artifacts",
                "data_authority_digest",
                "runner_code_sha256",
            },
            label="PRE_D0 registration %d" % index,
        )
        _sha(current["training_receipt_file_sha256"], label="registered receipt")
        if (
            current["receipt_schema"] != PRE_D0_TRAINING_RECEIPT_SCHEMA
            or current["classification"] != "pre_d0"
            or current["training_stage"] != PRE_D0_STAGE
        ):
            raise CheckpointEvaluationSelectionError("PRE_D0 registration differs")
        _sha(current["data_authority_digest"], label="registered PRE_D0 data")
        _sha(current["runner_code_sha256"], label="registered PRE_D0 runner")
        if not isinstance(current["checkpoint_artifacts"], list):
            raise CheckpointEvaluationSelectionError(
                "PRE_D0 artifact registration differs"
            )
        artifacts = []
        for artifact_index, artifact in enumerate(current["checkpoint_artifacts"]):
            artifact_row = _closed(
                artifact,
                {"step", "sha256"},
                label="PRE_D0 artifact registration %d" % artifact_index,
            )
            if type(artifact_row["step"]) is not int or artifact_row["step"] < 0:
                raise CheckpointEvaluationSelectionError(
                    "PRE_D0 registered artifact step differs"
                )
            _sha(artifact_row["sha256"], label="registered PRE_D0 artifact")
            artifacts.append(artifact_row)
        if [item["step"] for item in artifacts] != sorted(
            {item["step"] for item in artifacts}
        ):
            raise CheckpointEvaluationSelectionError(
                "PRE_D0 registered artifact steps are not sorted/unique"
            )
        current["checkpoint_artifacts"] = artifacts
        normalized_registrations.append(current)
    registered = {
        item["training_receipt_file_sha256"]: item["receipt_schema"]
        for item in normalized_registrations
    }
    registered_rows = {
        item["training_receipt_file_sha256"]: item
        for item in normalized_registrations
    }
    if (
        len(registered) != len(normalized_registrations)
        or any(
            registered.get(digest) != schema
            for digest, schema in KNOWN_PRE_D0_RECEIPTS.items()
        )
        or registered_rows.get(
            KNOWN_PRE_D0_R2_REGISTRATION["training_receipt_file_sha256"]
        )
        != KNOWN_PRE_D0_R2_REGISTRATION
    ):
        raise CheckpointEvaluationSelectionError(
            "known r2 PRE_D0 receipt is not externally registered"
        )
    entries = manifest["checkpoints"]
    if not isinstance(entries, list) or not entries:
        raise CheckpointEvaluationSelectionError("formal checkpoint registry is empty")
    entry_fields = {
        "checkpoint_id",
        "role",
        "receipt_class",
        "training_stage",
        "checkpoint_step",
        "training_receipt_file_sha256",
        "artifact_manifest_file_sha256",
        "checkpoint_artifact_sha256",
        "artifact_tree_sha256",
        "tensor_semantic_digest",
        "model_state_contract_digest",
        "data_authority_digest",
        "runner_release_manifest_file_sha256",
        "runner_code_tree_sha256",
        "training_manifest_sha256",
    }
    normalized_entries = []
    for index, item in enumerate(entries):
        current = _closed(item, entry_fields, label="authority checkpoint %d" % index)
        _identifier(current["checkpoint_id"], label="authority checkpoint id")
        if current["role"] not in {"base", "candidate"}:
            raise CheckpointEvaluationSelectionError("authority checkpoint role differs")
        if current["receipt_class"] not in {"base", "pre_d0", "formal"}:
            raise CheckpointEvaluationSelectionError("authority receipt class differs")
        for field in (
            "training_receipt_file_sha256",
            "artifact_manifest_file_sha256",
            "checkpoint_artifact_sha256",
            "artifact_tree_sha256",
            "tensor_semantic_digest",
            "model_state_contract_digest",
            "data_authority_digest",
            "runner_release_manifest_file_sha256",
            "runner_code_tree_sha256",
        ):
            _sha(current[field], label="authority %s" % field)
        if current["training_manifest_sha256"] is not None:
            _sha(current["training_manifest_sha256"], label="authority training manifest")
        if type(current["checkpoint_step"]) is not int or current["checkpoint_step"] < 0:
            raise CheckpointEvaluationSelectionError("authority checkpoint step differs")
        if not isinstance(current["training_stage"], str):
            raise CheckpointEvaluationSelectionError("authority training stage differs")
        expected_stage = (
            "BASE"
            if current["receipt_class"] == "base"
            else PRE_D0_STAGE
            if current["receipt_class"] == "pre_d0"
            else current["training_stage"]
        )
        if (
            current["training_stage"] != expected_stage
            or (current["receipt_class"] == "formal" and current["training_stage"] not in STAGE_REFERENCE_GATES)
            or (current["receipt_class"] == "base" and current["role"] != "base")
            or (current["receipt_class"] != "base" and current["role"] != "candidate")
        ):
            raise CheckpointEvaluationSelectionError("authority stage/class differs")
        if (
            current["training_receipt_file_sha256"] in KNOWN_PRE_D0_RECEIPTS
            and current["receipt_class"] != "pre_d0"
        ):
            raise CheckpointEvaluationSelectionError(
                "known r2 receipt was relabeled as formal"
            )
        if (
            current["checkpoint_artifact_sha256"] in KNOWN_PRE_D0_ARTIFACTS
            and current["receipt_class"] != "pre_d0"
        ):
            raise CheckpointEvaluationSelectionError(
                "known r2 checkpoint artifact was re-receipted as formal"
            )
        normalized_entries.append(current)
    ids = [item["checkpoint_id"] for item in normalized_entries]
    if ids != sorted(set(ids)):
        raise CheckpointEvaluationSelectionError(
            "formal authority checkpoint closure is not sorted/unique"
        )
    if ids != pins["authorized_checkpoint_ids"]:
        raise CheckpointEvaluationSelectionError(
            "formal authority members differ from caller closed set"
        )
    _verify_digest(manifest, field="authority_digest", label="formal authority manifest")
    manifest["known_pre_d0_receipts"] = normalized_registrations
    manifest["checkpoints"] = normalized_entries
    manifest["_file_sha256"] = file_sha
    return manifest


def _pre_d0_tainted(checkpoint: Mapping[str, Any]) -> bool:
    return checkpoint.get("_receipt_class") == "pre_d0"


def _validate_checkpoint_receipt(
    checkpoint: Mapping[str, Any], *, authority_entry: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], str, str]:
    receipt = _read_canonical_receipt(
        checkpoint["training_receipt_path"],
        expected_sha256=checkpoint["training_receipt_file_sha256"],
        label="%s training receipt" % label,
    )
    schema = receipt.get("schema_version")
    expected_step = checkpoint["expected_checkpoint_step"]
    artifact_sha = checkpoint["artifact_sha256"]
    data_digest = checkpoint["expected_data_authority_digest"]
    code_sha = checkpoint["expected_code_sha256"]
    checkpoint_id = checkpoint["checkpoint_id"]
    common_authority = {
        "checkpoint_id": checkpoint_id,
        "role": checkpoint["role"],
        "checkpoint_step": expected_step,
        "training_receipt_file_sha256": checkpoint["training_receipt_file_sha256"],
        "artifact_manifest_file_sha256": checkpoint[
            "artifact_manifest_file_sha256"
        ],
        "checkpoint_artifact_sha256": artifact_sha,
        "artifact_tree_sha256": checkpoint["artifact_tree_sha256"],
        "tensor_semantic_digest": checkpoint["tensor_semantic_digest"],
        "model_state_contract_digest": checkpoint["model_state_contract_digest"],
        "data_authority_digest": data_digest,
        "runner_release_manifest_file_sha256": checkpoint[
            "runner_release_manifest_file_sha256"
        ],
        "runner_code_tree_sha256": code_sha,
    }
    if any(authority_entry.get(key) != item for key, item in common_authority.items()):
        raise CheckpointEvaluationSelectionError(
            "%s differs from external formal training authority" % label
        )
    if (
        checkpoint["training_receipt_file_sha256"] in KNOWN_PRE_D0_RECEIPTS
        and authority_entry["receipt_class"] != "pre_d0"
    ):
        raise CheckpointEvaluationSelectionError(
            "known r2 receipt cannot be relabeled by a checkpoint wrapper"
        )
    if checkpoint["role"] == "base":
        fields = {
            "schema_version",
            "authority",
            "complete",
            "checkpoint_id",
            "checkpoint_step",
            "checkpoint_artifact_sha256",
            "data_authority_digest",
            "runner_source_sha256",
            "base_release_manifest_sha256",
            "inference_compatible",
            "receipt_digest",
        }
        row = _closed(receipt, fields, label="base checkpoint receipt")
        if (
            schema != BASE_RECEIPT_SCHEMA
            or row["authority"] != "FROZEN_BASE"
            or row["complete"] is not True
            or row["inference_compatible"] is not True
            or row["checkpoint_id"] != checkpoint_id
            or row["checkpoint_step"] != expected_step
            or row["checkpoint_artifact_sha256"] != artifact_sha
            or row["data_authority_digest"] != data_digest
            or row["runner_source_sha256"] != code_sha
            or row["base_release_manifest_sha256"]
            != checkpoint["runner_release_manifest_file_sha256"]
            or authority_entry["receipt_class"] != "base"
            or authority_entry["training_stage"] != "BASE"
            or authority_entry["training_manifest_sha256"] is not None
        ):
            raise CheckpointEvaluationSelectionError(
                "base checkpoint receipt authority differs"
            )
        _sha(row["base_release_manifest_sha256"], label="base release manifest")
        return row, "base", "BASE"

    if schema == PRE_D0_TRAINING_RECEIPT_SCHEMA:
        required = {
            "schema_version",
            "authority",
            "complete",
            "promotable",
            "formal_training_started",
            "counts_as_d0",
            "optimizer_steps",
            "checkpoint_steps",
            "checkpoints",
            "dataset",
            "provenance",
            "receipt_digest",
        }
        if not required.issubset(receipt):
            raise CheckpointEvaluationSelectionError(
                "PRE_D0 receipt required fields differ"
            )
        if (
            not isinstance(receipt["checkpoint_steps"], list)
            or any(type(item) is not int or item < 0 for item in receipt["checkpoint_steps"])
            or receipt["checkpoint_steps"] != sorted(set(receipt["checkpoint_steps"]))
            or not isinstance(receipt["checkpoints"], list)
            or not isinstance(receipt["dataset"], Mapping)
            or not isinstance(receipt["provenance"], Mapping)
        ):
            raise CheckpointEvaluationSelectionError(
                "PRE_D0 receipt list/object authority differs"
            )
        matches = [
            item
            for item in receipt["checkpoints"]
            if isinstance(item, Mapping) and item.get("step") == expected_step
        ]
        provenance = receipt["provenance"]
        if (
            receipt["authority"] != PRE_D0_STAGE
            or receipt["complete"] is not True
            or receipt["promotable"] is not False
            or receipt["formal_training_started"] is not False
            or receipt["counts_as_d0"] is not False
            or type(receipt["optimizer_steps"]) is not int
            or receipt["optimizer_steps"] < expected_step
            or expected_step not in receipt["checkpoint_steps"]
            or len(matches) != 1
            or matches[0].get("adapter_sha256") != artifact_sha
            or object_sha256(receipt["dataset"]) != data_digest
            or not isinstance(provenance, Mapping)
            or provenance.get("runner_source_sha256") != code_sha
            or authority_entry["receipt_class"] != "pre_d0"
            or authority_entry["training_stage"] != PRE_D0_STAGE
            or authority_entry["training_manifest_sha256"] is not None
        ):
            raise CheckpointEvaluationSelectionError(
                "PRE_D0 checkpoint receipt authority differs"
            )
        # The schema itself is dispositive.  Renaming the checkpoint or
        # changing optimistic wrapper flags can never promote this artifact.
        return receipt, "pre_d0", PRE_D0_STAGE

    if schema == FORMAL_TRAINING_RECEIPT_SCHEMA:
        fields = {
            "schema_version",
            "authority",
            "status",
            "complete",
            "promotable",
            "formal_training_started",
            "counts_as_d0",
            "training_stage",
            "checkpoint_id",
            "checkpoint_step",
            "checkpoint_artifact_sha256",
            "data_authority_digest",
            "training_manifest_sha256",
            "runner_source_sha256",
            "receipt_digest",
        }
        row = _closed(receipt, fields, label="formal training receipt")
        stage = row["training_stage"]
        if (
            row["authority"] != "FORMAL_0817_TRAINING"
            or row["status"] != "complete"
            or row["complete"] is not True
            or row["promotable"] is not True
            or row["formal_training_started"] is not True
            or row["counts_as_d0"] is not True
            or stage not in STAGE_REFERENCE_GATES
            or row["checkpoint_id"] != checkpoint_id
            or row["checkpoint_step"] != expected_step
            or row["checkpoint_artifact_sha256"] != artifact_sha
            or row["data_authority_digest"] != data_digest
            or row["runner_source_sha256"] != code_sha
            or authority_entry["receipt_class"] != "formal"
            or authority_entry["training_stage"] != stage
            or authority_entry["training_manifest_sha256"]
            != row["training_manifest_sha256"]
        ):
            raise CheckpointEvaluationSelectionError(
                "formal checkpoint receipt authority differs"
            )
        _sha(row["training_manifest_sha256"], label="formal training manifest")
        return row, "formal", stage

    raise CheckpointEvaluationSelectionError(
        "unknown checkpoint training receipt schema"
    )


def _validate_checkpoint_freeze(
    value: Any, *, formal_authority: Mapping[str, Any]
) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "schema_version",
            "base_checkpoint_id",
            "checkpoints",
            "formal_authority_manifest_file_sha256",
            "frozen_before_outputs",
            "freeze_digest",
        },
        label="checkpoint freeze",
    )
    if row["schema_version"] != CHECKPOINT_FREEZE_SCHEMA:
        raise CheckpointEvaluationSelectionError("checkpoint freeze schema differs")
    if (
        row["formal_authority_manifest_file_sha256"]
        != formal_authority["_file_sha256"]
    ):
        raise CheckpointEvaluationSelectionError(
            "checkpoint freeze formal authority binding differs"
        )
    base_id = _identifier(row["base_checkpoint_id"], label="base checkpoint id")
    if row["frozen_before_outputs"] is not True:
        raise CheckpointEvaluationSelectionError(
            "checkpoints were not frozen before outputs"
        )
    if not isinstance(row["checkpoints"], list) or len(row["checkpoints"]) < 2:
        raise CheckpointEvaluationSelectionError(
            "checkpoint freeze needs one base and at least one candidate"
        )
    checkpoints = []
    fields = {
        "checkpoint_id",
        "role",
        "artifact_sha256",
        "artifact_tree_sha256",
        "tensor_semantic_digest",
        "model_state_contract_digest",
        "artifact_root_path",
        "artifact_manifest_path",
        "artifact_manifest_file_sha256",
        "training_receipt_path",
        "training_receipt_file_sha256",
        "runner_release_manifest_path",
        "runner_release_manifest_file_sha256",
        "expected_checkpoint_step",
        "expected_data_authority_digest",
        "expected_code_sha256",
        "frozen",
    }
    for index, item in enumerate(row["checkpoints"]):
        checkpoint = _closed(item, fields, label="checkpoint %d" % index)
        _identifier(checkpoint["checkpoint_id"], label="checkpoint id")
        if checkpoint["role"] not in {"base", "candidate"}:
            raise CheckpointEvaluationSelectionError("checkpoint role differs")
        _sha(checkpoint["artifact_sha256"], label="checkpoint artifact")
        _sha(checkpoint["artifact_tree_sha256"], label="checkpoint artifact tree")
        _sha(checkpoint["tensor_semantic_digest"], label="checkpoint tensor semantics")
        _sha(
            checkpoint["model_state_contract_digest"],
            label="checkpoint model-state contract",
        )
        _sha(
            checkpoint["artifact_manifest_file_sha256"],
            label="checkpoint artifact manifest",
        )
        _sha(
            checkpoint["training_receipt_file_sha256"],
            label="training receipt file",
        )
        _sha(
            checkpoint["expected_data_authority_digest"],
            label="checkpoint data authority",
        )
        _sha(checkpoint["expected_code_sha256"], label="checkpoint code")
        _sha(
            checkpoint["runner_release_manifest_file_sha256"],
            label="checkpoint runner release manifest",
        )
        if (
            type(checkpoint["expected_checkpoint_step"]) is not int
            or checkpoint["expected_checkpoint_step"] < 0
        ):
            raise CheckpointEvaluationSelectionError(
                "expected checkpoint step differs"
            )
        if checkpoint["frozen"] is not True:
            raise CheckpointEvaluationSelectionError("checkpoint is not frozen")
        artifact_manifest = _validate_artifact_tree_manifest(
            checkpoint, label="checkpoint %d" % index
        )
        runner_manifest = _validate_release_manifest(
            path=checkpoint["runner_release_manifest_path"],
            expected_file_sha256=checkpoint[
                "runner_release_manifest_file_sha256"
            ],
            expected_tree_sha256=checkpoint["expected_code_sha256"],
            label="checkpoint %d runner release manifest" % index,
        )
        authority_entries = [
            item
            for item in formal_authority["checkpoints"]
            if item["checkpoint_id"] == checkpoint["checkpoint_id"]
        ]
        if len(authority_entries) != 1:
            raise CheckpointEvaluationSelectionError(
                "checkpoint is not a unique external authority member"
            )
        if authority_entries[0]["receipt_class"] != "pre_d0" and any(
            entry["sha256"] in KNOWN_PRE_D0_ARTIFACTS
            for entry in artifact_manifest["files"]
            if entry["artifact_role"] == "weight_shard"
        ):
            raise CheckpointEvaluationSelectionError(
                "known r2 weight shard was embedded in a formal artifact tree"
            )
        receipt, receipt_class, stage = _validate_checkpoint_receipt(
            checkpoint,
            authority_entry=authority_entries[0],
            label="checkpoint %d" % index,
        )
        checkpoint["_artifact_manifest"] = artifact_manifest
        checkpoint["_runner_release_manifest"] = runner_manifest
        checkpoint["_authority_entry"] = authority_entries[0]
        checkpoint["_training_receipt"] = receipt
        checkpoint["_receipt_class"] = receipt_class
        checkpoint["_training_stage"] = stage
        checkpoints.append(checkpoint)
    ids = [item["checkpoint_id"] for item in checkpoints]
    if ids != sorted(set(ids)):
        raise CheckpointEvaluationSelectionError(
            "checkpoint identifiers are not sorted/unique"
        )
    if sorted(ids) != [
        item["checkpoint_id"] for item in formal_authority["checkpoints"]
    ]:
        raise CheckpointEvaluationSelectionError(
            "checkpoint freeze differs from caller-authorized closed set"
        )
    bases = [item for item in checkpoints if item["role"] == "base"]
    candidates = [item for item in checkpoints if item["role"] == "candidate"]
    if (
        len(bases) != 1
        or bases[0]["checkpoint_id"] != base_id
        or not candidates
    ):
        raise CheckpointEvaluationSelectionError("base/candidate closure differs")
    _verify_digest(row, field="freeze_digest", label="checkpoint freeze")
    row["checkpoints"] = checkpoints
    return row


def _validate_row_freeze(value: Any, *, decoder: FullVideoDecoder) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "schema_version",
            "rows",
            "frozen_before_outputs",
            "row_set_digest",
        },
        label="row freeze",
    )
    if row["schema_version"] != ROW_FREEZE_SCHEMA:
        raise CheckpointEvaluationSelectionError("row freeze schema differs")
    if row["frozen_before_outputs"] is not True:
        raise CheckpointEvaluationSelectionError(
            "source/instruction rows were not frozen before outputs"
        )
    if not isinstance(row["rows"], list) or not row["rows"]:
        raise CheckpointEvaluationSelectionError("row freeze is empty")
    fields = {
        "row_id",
        "source_id",
        "source_cluster_id",
        "actor_scene_cluster_id",
        "source_equivalence_group_id",
        "source_perceptual_fingerprint_sha256",
        "upstream_group_id",
        "split",
        "row_kind",
        "source_video_path",
        "source_video_sha256",
        "source_byte_count",
        "source_media_signature",
        "instruction",
        "instruction_sha256",
        "output_width",
        "output_height",
        "output_pixel_format",
        "intrinsically_assessable",
    }
    rows = []
    for index, item in enumerate(row["rows"]):
        current = _closed(item, fields, label="evaluation row %d" % index)
        for name in (
            "row_id",
            "source_id",
            "source_cluster_id",
            "actor_scene_cluster_id",
            "source_equivalence_group_id",
            "upstream_group_id",
        ):
            _identifier(current[name], label="evaluation row %s" % name)
        if current["split"] not in SPLITS:
            raise CheckpointEvaluationSelectionError("evaluation split differs")
        if current["row_kind"] not in {"edit", "noop"}:
            raise CheckpointEvaluationSelectionError("row kind differs")
        if (current["split"] == "noop_preservation") is not (
            current["row_kind"] == "noop"
        ):
            raise CheckpointEvaluationSelectionError("noop split/kind differs")
        source_path, _ = _read_plain_file(
            current["source_video_path"],
            expected_sha256=current["source_video_sha256"],
            expected_size=current["source_byte_count"],
            label="canonical source bytes",
            return_payload=False,
        )
        _sha(
            current["source_perceptual_fingerprint_sha256"],
            label="source perceptual fingerprint",
        )
        source_media = _verify_actual_media(
            source_path,
            current["source_media_signature"],
            decoder=decoder,
            expected_file_sha256=current["source_video_sha256"],
            label="canonical source full81 video",
        )
        actual_semantics = _source_semantics_from_actual_media(source_media)
        for semantic_field in (
            "source_perceptual_fingerprint_sha256",
            "source_equivalence_group_id",
        ):
            if current[semantic_field] != actual_semantics[semantic_field]:
                raise CheckpointEvaluationSelectionError(
                    "source media equivalence field was not recomputed from decoded frames"
                )
        for geometry in ("output_width", "output_height"):
            if type(current[geometry]) is not int or current[geometry] <= 0:
                raise CheckpointEvaluationSelectionError(
                    "row output geometry differs"
                )
        _identifier(
            current["output_pixel_format"], label="row output pixel format"
        )
        instruction = current["instruction"]
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or instruction != instruction.strip()
            or "\x00" in instruction
            or text_sha256(instruction) != current["instruction_sha256"]
        ):
            raise CheckpointEvaluationSelectionError(
                "instruction bytes/digest differ"
            )
        mask = _closed(
            current["intrinsically_assessable"],
            set(AXES),
            label="intrinsic assessability mask",
        )
        for axis in AXES:
            _boolean(mask[axis], label="intrinsic mask %s" % axis)
        expected_mask = {
            "action": current["row_kind"] == "edit",
            "order": current["row_kind"] == "edit",
            "identity": True,
            "ownership": current["split"] == "interaction_contact",
            "background": True,
            "camera": True,
            "quality": True,
            "noop": current["row_kind"] == "noop",
        }
        if mask != expected_mask:
            raise CheckpointEvaluationSelectionError(
                "intrinsic mask is not determined by frozen row semantics"
            )
        current["intrinsically_assessable"] = mask
        current["source_media_signature"] = _normalized_media_signature(
            current["source_media_signature"], label="source media signature"
        )
        current["_actual_source_media"] = source_media
        current["_actual_source_semantics"] = actual_semantics
        rows.append(current)
    ids = [item["row_id"] for item in rows]
    if len(ids) != len(set(ids)):
        raise CheckpointEvaluationSelectionError("evaluation row ids collide")
    _verify_digest(row, field="row_set_digest", label="row freeze")
    row["rows"] = rows
    return row


def _validate_source_equivalence_authority(
    pins: Mapping[str, Any], *, row_freeze: Mapping[str, Any], authority_root: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, file_sha = _read_external_manifest(
        {
            "path": pins["source_equivalence_manifest_path"],
            "file_sha256": pins["source_equivalence_manifest_sha256"],
        },
        label="source equivalence authority manifest",
    )
    manifest = _closed(
        manifest,
        {
            "schema_version",
            "authority_id",
            "status",
            "frozen_before_outputs",
            "row_set_digest",
            "equivalence_contract",
            "rows",
            "no_equivalent_source_repeated",
            "authority_digest",
        },
        label="source equivalence authority manifest",
    )
    contract = _closed(
        manifest["equivalence_contract"],
        {
            "raw_byte_sha256_required",
            "decoded_frame_sha256_required",
            "perceptual_fingerprint_required",
            "perceptual_algorithm",
            "perceptual_algorithm_sha256",
            "source_clustering_algorithm_sha256",
            "actor_scene_clustering_algorithm_sha256",
            "cluster_assignment_authority",
            "distance_metric",
            "duplicate_threshold",
        },
        label="source equivalence contract",
    )
    if (
        file_sha != authority_root["source_equivalence_manifest_file_sha256"]
        or manifest["schema_version"] != SOURCE_EQUIVALENCE_SCHEMA
        or manifest["status"] != "locked"
        or manifest["frozen_before_outputs"] is not True
        or manifest["row_set_digest"] != row_freeze["row_set_digest"]
        or manifest["no_equivalent_source_repeated"] is not True
        or contract["raw_byte_sha256_required"] is not True
        or contract["decoded_frame_sha256_required"] is not True
        or contract["perceptual_fingerprint_required"] is not True
        or contract["perceptual_algorithm"] != FROZEN_PERCEPTUAL_ALGORITHM
        or contract["perceptual_algorithm_sha256"] != FROZEN_ALGORITHM_SHA256
        or contract["source_clustering_algorithm_sha256"]
        != FROZEN_ALGORITHM_SHA256
        or contract["actor_scene_clustering_algorithm_sha256"]
        != FROZEN_ALGORITHM_SHA256
        or contract["cluster_assignment_authority"]
        != "pre_frozen_collection_actor_scene_equivalence_semantics"
        or contract["distance_metric"]
        != "authority-assigned-semantic-equivalence-v1"
        or contract["duplicate_threshold"] != 0.0
    ):
        raise CheckpointEvaluationSelectionError(
            "source equivalence authority contract differs"
        )
    _identifier(manifest["authority_id"], label="source equivalence authority id")
    _identifier(contract["perceptual_algorithm"], label="perceptual algorithm")
    _sha(contract["perceptual_algorithm_sha256"], label="perceptual algorithm SHA")
    _sha(
        contract["source_clustering_algorithm_sha256"],
        label="source clustering algorithm SHA",
    )
    _sha(
        contract["actor_scene_clustering_algorithm_sha256"],
        label="actor/scene clustering algorithm SHA",
    )
    _identifier(contract["distance_metric"], label="perceptual distance metric")
    _unit(contract["duplicate_threshold"], label="perceptual duplicate threshold")
    if not isinstance(manifest["rows"], list):
        raise CheckpointEvaluationSelectionError("source authority rows differ")
    source_by_id = {item["row_id"]: item for item in row_freeze["rows"]}
    fields = {
        "row_id",
        "source_id",
        "source_cluster_id",
        "source_cluster_fingerprint_sha256",
        "source_video_sha256",
        "source_frame_content_sha256",
        "source_perceptual_fingerprint_sha256",
        "source_equivalence_group_id",
        "actor_scene_cluster_id",
        "actor_identity_fingerprint_sha256",
        "scene_fingerprint_sha256",
        "upstream_group_id",
        "collection_id",
        "actor_identity_id",
        "scene_id",
        "semantic_equivalence_id",
        "cluster_assignment_authority_sha256",
    }
    normalized = []
    for index, item in enumerate(manifest["rows"]):
        current = _closed(item, fields, label="source authority row %d" % index)
        source = source_by_id.get(current["row_id"])
        if source is None:
            raise CheckpointEvaluationSelectionError("source authority row is unknown")
        for field in (
            "collection_id",
            "actor_identity_id",
            "scene_id",
            "semantic_equivalence_id",
        ):
            _identifier(current[field], label="source authority %s" % field)
        for field in (
            "source_cluster_fingerprint_sha256",
            "actor_identity_fingerprint_sha256",
            "scene_fingerprint_sha256",
            "cluster_assignment_authority_sha256",
        ):
            _sha(current[field], label="source authority %s" % field)
        expected = {
            "row_id": source["row_id"],
            "source_id": source["source_id"],
            "source_cluster_id": source["source_cluster_id"],
            "source_cluster_fingerprint_sha256": object_sha256(
                {
                    "algorithm": FROZEN_CLUSTER_ALGORITHM,
                    "collection_id": current["collection_id"],
                    "semantic_equivalence_id": current["semantic_equivalence_id"],
                    "source_cluster_id": source["source_cluster_id"],
                }
            ),
            "source_video_sha256": source["source_video_sha256"],
            "source_frame_content_sha256": source["source_media_signature"][
                "frame_content_sha256"
            ],
            "source_perceptual_fingerprint_sha256": source[
                "_actual_source_semantics"
            ]["source_perceptual_fingerprint_sha256"],
            "source_equivalence_group_id": source["source_equivalence_group_id"],
            "actor_scene_cluster_id": source["actor_scene_cluster_id"],
            "actor_identity_fingerprint_sha256": source[
                "_actual_source_semantics"
            ]["actor_identity_fingerprint_sha256"],
            "scene_fingerprint_sha256": source["_actual_source_semantics"][
                "scene_fingerprint_sha256"
            ],
            "upstream_group_id": source["upstream_group_id"],
            "collection_id": current["collection_id"],
            "actor_identity_id": current["actor_identity_id"],
            "scene_id": current["scene_id"],
            "semantic_equivalence_id": current["semantic_equivalence_id"],
            "cluster_assignment_authority_sha256": object_sha256(
                {
                    "algorithm": FROZEN_CLUSTER_ALGORITHM,
                    "collection_id": current["collection_id"],
                    "actor_identity_id": current["actor_identity_id"],
                    "scene_id": current["scene_id"],
                    "semantic_equivalence_id": current["semantic_equivalence_id"],
                    "source_cluster_id": current["source_cluster_id"],
                    "actor_scene_cluster_id": current["actor_scene_cluster_id"],
                }
            ),
        }
        if current != expected:
            raise CheckpointEvaluationSelectionError(
                "source raw/decode and pre-frozen semantic cluster authority differs"
            )
        normalized.append(current)
    row_ids = [item["row_id"] for item in normalized]
    if row_ids != sorted(source_by_id) or len(row_ids) != len(set(row_ids)):
        raise CheckpointEvaluationSelectionError(
            "source equivalence authority does not close the row set"
        )
    groups = [item["source_equivalence_group_id"] for item in normalized]
    if len(groups) != len(set(groups)):
        raise CheckpointEvaluationSelectionError(
            "source equivalence authority repeats an equivalent source"
        )
    for cluster_field, fingerprint_fields in (
        ("source_cluster_id", ("source_cluster_fingerprint_sha256",)),
        (
            "actor_scene_cluster_id",
            (
                "actor_identity_fingerprint_sha256",
                "scene_fingerprint_sha256",
            ),
        ),
    ):
        cluster_to_fingerprints: dict[str, set[tuple[str, ...]]] = {}
        fingerprint_to_clusters: dict[tuple[str, ...], set[str]] = {}
        for item in normalized:
            fingerprint = tuple(item[field] for field in fingerprint_fields)
            cluster_to_fingerprints.setdefault(item[cluster_field], set()).add(
                fingerprint
            )
            fingerprint_to_clusters.setdefault(fingerprint, set()).add(
                item[cluster_field]
            )
        if any(len(values) != 1 for values in fingerprint_to_clusters.values()):
            raise CheckpointEvaluationSelectionError(
                "decoded feature fingerprints map to multiple frozen clusters"
            )
    _verify_digest(manifest, field="authority_digest", label="source equivalence authority")
    manifest["rows"] = normalized
    manifest["equivalence_contract"] = contract
    manifest["_file_sha256"] = file_sha
    return manifest


def _validate_locked_split(
    value: Any, *, row_freeze: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    if value is None:
        return None, ["LOCKED_SPLIT_MISSING"]
    row = _closed(
        value,
        {
            "schema_version",
            "split_id",
            "kind",
            "status",
            "row_count",
            "row_set_digest",
            "frozen_before_outputs",
            "lock_digest",
        },
        label="locked split",
    )
    if row["schema_version"] != LOCKED_SPLIT_SCHEMA:
        raise CheckpointEvaluationSelectionError("locked split schema differs")
    _identifier(row["split_id"], label="locked split id")
    if row["kind"] != FORMAL_MODE or row["status"] != "locked":
        reasons = ["LOCKED_SPLIT_NOT_PROMOTION_VALIDATION"]
    else:
        reasons = []
    if type(row["row_count"]) is not int or row["row_count"] < 1:
        raise CheckpointEvaluationSelectionError("locked split row count differs")
    _sha(row["row_set_digest"], label="locked split row set")
    if row["row_set_digest"] != row_freeze["row_set_digest"]:
        raise CheckpointEvaluationSelectionError(
            "locked split does not bind row freeze"
        )
    if row["row_count"] != len(row_freeze["rows"]):
        raise CheckpointEvaluationSelectionError(
            "locked split row count does not bind row freeze"
        )
    if row["frozen_before_outputs"] is not True:
        reasons.append("LOCKED_SPLIT_NOT_FROZEN_BEFORE_OUTPUTS")
    _verify_digest(row, field="lock_digest", label="locked split")
    return row, reasons


def _validate_calibration(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if value is None:
        return None, ["EVALUATOR_CALIBRATION_MISSING"]
    row = _closed(
        value,
        {
            "schema_version",
            "status",
            "independent_pair_count",
            "overall_auroc",
            "average_precision",
            "real_generated_auroc_gap",
            "failure_categories",
            "frozen_before_outputs",
            "calibration_digest",
        },
        label="evaluator calibration",
    )
    if row["schema_version"] != CALIBRATION_SCHEMA:
        raise CheckpointEvaluationSelectionError("calibration schema differs")
    if type(row["independent_pair_count"]) is not int or row[
        "independent_pair_count"
    ] < 0:
        raise CheckpointEvaluationSelectionError("calibration pair count differs")
    auroc = _unit(row["overall_auroc"], label="calibration AUROC")
    ap = _unit(row["average_precision"], label="calibration AP")
    gap = _unit(row["real_generated_auroc_gap"], label="domain AUROC gap")
    categories = _closed(
        row["failure_categories"],
        set(CALIBRATION_CATEGORIES),
        label="calibration failure categories",
    )
    normalized_categories = {}
    for name in CALIBRATION_CATEGORIES:
        current = _closed(
            categories[name], {"precision", "recall"}, label="category %s" % name
        )
        normalized_categories[name] = {
            "precision": _unit(current["precision"], label="%s precision" % name),
            "recall": _unit(current["recall"], label="%s recall" % name),
        }
    _boolean(row["frozen_before_outputs"], label="calibration freeze flag")
    _verify_digest(row, field="calibration_digest", label="calibration")
    qualified = (
        row["status"] == "qualified"
        and row["independent_pair_count"] >= 2_000
        and auroc >= 0.85
        and ap >= 0.80
        and gap <= 0.05
        and row["frozen_before_outputs"] is True
        and all(
            item["precision"] >= 0.85 and item["recall"] >= 0.75
            for item in normalized_categories.values()
        )
    )
    row["failure_categories"] = normalized_categories
    return row, [] if qualified else ["EVALUATOR_CALIBRATION_UNQUALIFIED"]


def _validate_selection_contract(
    value: Any,
    *,
    checkpoint_freeze: Mapping[str, Any],
    row_freeze: Mapping[str, Any],
    locked_split: Mapping[str, Any] | None,
    source_equivalence_authority: Mapping[str, Any],
    reviewer_roster_authority: Mapping[str, Any],
    authority_pins: Mapping[str, Any],
    authority_root: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    row = _closed(
        value,
        {
            "schema_version",
            "checkpoint_freeze_digest",
            "locked_split_digest",
            "row_set_digest",
            "source_equivalence_manifest_file_sha256",
            "reviewer_roster_manifest_file_sha256",
            "bootstrap_seed_hex",
            "bootstrap_resamples",
            "confidence",
            "inference_code_sha256",
            "inference_release_manifest_path",
            "inference_release_manifest_file_sha256",
            "full_video_contract",
            "frozen_before_outputs",
            "contract_digest",
        },
        label="selection contract",
    )
    if row["schema_version"] != SELECTION_CONTRACT_SCHEMA:
        raise CheckpointEvaluationSelectionError("selection contract schema differs")
    if row["checkpoint_freeze_digest"] != checkpoint_freeze["freeze_digest"]:
        raise CheckpointEvaluationSelectionError(
            "selection contract checkpoint freeze differs"
        )
    expected_lock = None if locked_split is None else locked_split["lock_digest"]
    if row["locked_split_digest"] != expected_lock:
        raise CheckpointEvaluationSelectionError(
            "selection contract locked split differs"
        )
    if row["row_set_digest"] != row_freeze["row_set_digest"]:
        raise CheckpointEvaluationSelectionError(
            "selection contract row freeze differs"
        )
    if (
        row["source_equivalence_manifest_file_sha256"]
        != source_equivalence_authority["_file_sha256"]
        or row["reviewer_roster_manifest_file_sha256"]
        != reviewer_roster_authority["_file_sha256"]
    ):
        raise CheckpointEvaluationSelectionError(
            "selection contract external source/reviewer authority differs"
        )
    seed = row["bootstrap_seed_hex"]
    if seed != FIXED_BOOTSTRAP_SEED_HEX:
        raise CheckpointEvaluationSelectionError("bootstrap seed differs")
    if (
        row["bootstrap_resamples"] != BOOTSTRAP_RESAMPLES
        or row["confidence"] != 0.95
        or row["frozen_before_outputs"] is not True
    ):
        raise CheckpointEvaluationSelectionError(
            "selection sampling contract differs"
        )
    _sha(row["inference_code_sha256"], label="inference code")
    _sha(
        row["inference_release_manifest_file_sha256"],
        label="inference release manifest",
    )
    if (
        row["inference_release_manifest_path"]
        != authority_pins["inference_release_manifest_path"]
        or row["inference_release_manifest_file_sha256"]
        != authority_pins["inference_release_manifest_sha256"]
        or row["inference_release_manifest_file_sha256"]
        != authority_root["inference_release_manifest_file_sha256"]
    ):
        raise CheckpointEvaluationSelectionError(
            "inference release differs from caller external pin"
        )
    inference_release = _validate_release_manifest(
        path=row["inference_release_manifest_path"],
        expected_file_sha256=row["inference_release_manifest_file_sha256"],
        expected_tree_sha256=row["inference_code_sha256"],
        label="inference code release manifest",
    )
    media = _closed(
        row["full_video_contract"],
        {
            "frame_count",
            "fps_num",
            "fps_den",
            "duration_num",
            "duration_den",
            "full_video_required",
            "count_frames_required",
            "decode_to_eof_required",
            "pts_digest_required",
            "frame_content_digest_required",
            "frozen_feature_digests_required",
            "formal_decoder_id",
        },
        label="full video contract",
    )
    if media != {
        "frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
        "duration_num": 81,
        "duration_den": 25,
        "full_video_required": True,
        "count_frames_required": True,
        "decode_to_eof_required": True,
        "pts_digest_required": True,
        "frame_content_digest_required": True,
        "frozen_feature_digests_required": True,
        "formal_decoder_id": PRODUCTION_DECODER_ID,
    }:
        raise CheckpointEvaluationSelectionError("full video contract differs")
    _verify_digest(row, field="contract_digest", label="selection contract")
    selection_input = {
        "schema_version": "bernini-action-editing-selection-seed-input-0817-v2",
        "checkpoint_freeze_digest": checkpoint_freeze["freeze_digest"],
        "locked_split_digest": expected_lock,
        "bootstrap_seed_hex": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    row["_inference_release_manifest"] = inference_release
    return row, object_sha256(selection_input)


def _validate_private_mapping(
    value: Any,
    *,
    checkpoint_freeze: Mapping[str, Any],
    row_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "schema_version",
            "rows",
            "keeper_commitment",
            "sealed_before_outputs",
            "mapping_digest",
        },
        label="private mapping",
    )
    if row["schema_version"] != PRIVATE_MAPPING_SCHEMA:
        raise CheckpointEvaluationSelectionError("private mapping schema differs")
    if row["sealed_before_outputs"] is not True:
        raise CheckpointEvaluationSelectionError(
            "private mapping was not sealed before outputs"
        )
    commitment = _closed(
        row["keeper_commitment"],
        {
            "schema_version",
            "keeper_id",
            "blinding_key_sha256",
            "mapping_algorithm",
            "sealed_before_outputs",
            "commitment_digest",
        },
        label="keeper commitment",
    )
    if (
        commitment["schema_version"] != KEEPER_COMMITMENT_SCHEMA
        or commitment["mapping_algorithm"] != "hmac-sha256-row-checkpoint-v1"
        or commitment["sealed_before_outputs"] is not True
    ):
        raise CheckpointEvaluationSelectionError("keeper commitment differs")
    _identifier(commitment["keeper_id"], label="keeper id")
    _sha(commitment["blinding_key_sha256"], label="keeper key commitment")
    _verify_digest(
        commitment, field="commitment_digest", label="keeper commitment"
    )
    if not isinstance(row["rows"], list):
        raise CheckpointEvaluationSelectionError("private mapping rows differ")
    rows = []
    fields = {
        "row_id",
        "opaque_candidate_id",
        "checkpoint_id",
        "mapping_row_digest",
    }
    for index, item in enumerate(row["rows"]):
        current = _closed(item, fields, label="private mapping row %d" % index)
        for name in ("row_id", "opaque_candidate_id", "checkpoint_id"):
            _identifier(current[name], label="private mapping %s" % name)
        _verify_digest(current, field="mapping_row_digest", label="mapping row")
        rows.append(current)
    expected = {
        (source["row_id"], checkpoint["checkpoint_id"])
        for source in row_freeze["rows"]
        for checkpoint in checkpoint_freeze["checkpoints"]
    }
    actual = {(item["row_id"], item["checkpoint_id"]) for item in rows}
    opaque = [item["opaque_candidate_id"] for item in rows]
    if actual != expected or len(actual) != len(rows):
        raise CheckpointEvaluationSelectionError(
            "private mapping does not close row/checkpoint Cartesian product"
        )
    if len(opaque) != len(set(opaque)):
        raise CheckpointEvaluationSelectionError("opaque candidate ids collide")
    _verify_digest(row, field="mapping_digest", label="private mapping")
    row["rows"] = rows
    row["keeper_commitment"] = commitment
    return row


def _validate_reviewer_roster_authority(
    pins: Mapping[str, Any], *, authority_root: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, file_sha = _read_external_manifest(
        {
            "path": pins["reviewer_roster_manifest_path"],
            "file_sha256": pins["reviewer_roster_manifest_sha256"],
        },
        label="reviewer roster authority manifest",
    )
    manifest = _closed(
        manifest,
        {
            "schema_version",
            "authority_id",
            "status",
            "frozen_before_outputs",
            "reviewers",
            "roster_digest",
        },
        label="reviewer roster authority manifest",
    )
    if (
        file_sha != authority_root["reviewer_roster_manifest_file_sha256"]
        or manifest["schema_version"] != REVIEWER_ROSTER_SCHEMA
        or manifest["status"] != "locked"
        or manifest["frozen_before_outputs"] is not True
    ):
        raise CheckpointEvaluationSelectionError(
            "reviewer roster was not externally locked before outputs"
        )
    _identifier(manifest["authority_id"], label="reviewer roster authority id")
    if not isinstance(manifest["reviewers"], list):
        raise CheckpointEvaluationSelectionError("reviewer roster differs")
    reviewers = []
    for index, item in enumerate(manifest["reviewers"]):
        current = _closed(
            item,
            {
                "reviewer_id",
                "reviewer_role",
                "independence_group_id",
                "signature_subject_id",
                "signature_key_id",
                "signature_trust_root_sha256",
                "eligible",
            },
            label="reviewer roster row %d" % index,
        )
        _identifier(current["reviewer_id"], label="roster reviewer id")
        _identifier(
            current["independence_group_id"], label="reviewer independence group"
        )
        if current["signature_subject_id"] != current["reviewer_id"]:
            raise CheckpointEvaluationSelectionError(
                "reviewer signature subject differs from roster identity"
            )
        _identifier(current["signature_key_id"], label="reviewer signature key id")
        _sha(
            current["signature_trust_root_sha256"],
            label="reviewer signature trust root",
        )
        if current["reviewer_role"] not in {"primary", "adjudicator"}:
            raise CheckpointEvaluationSelectionError("roster reviewer role differs")
        if current["eligible"] is not True:
            raise CheckpointEvaluationSelectionError("roster includes ineligible reviewer")
        reviewers.append(current)
    ids = [item["reviewer_id"] for item in reviewers]
    if ids != sorted(set(ids)):
        raise CheckpointEvaluationSelectionError(
            "reviewer roster ids are not sorted/unique"
        )
    primary = [item for item in reviewers if item["reviewer_role"] == "primary"]
    adjudicators = [
        item for item in reviewers if item["reviewer_role"] == "adjudicator"
    ]
    if len(primary) < 2 or not adjudicators:
        raise CheckpointEvaluationSelectionError(
            "reviewer roster lacks two primaries/third adjudicator"
        )
    groups = [item["independence_group_id"] for item in reviewers]
    if len(groups) != len(set(groups)):
        raise CheckpointEvaluationSelectionError(
            "reviewer independence groups collide"
        )
    reviewer_keys = [item["signature_key_id"] for item in reviewers]
    reviewer_roots = [item["signature_trust_root_sha256"] for item in reviewers]
    if len(reviewer_keys) != len(set(reviewer_keys)) or len(reviewer_roots) != len(
        set(reviewer_roots)
    ):
        raise CheckpointEvaluationSelectionError(
            "reviewers do not have independent signing keys/trust roots"
        )
    key_registry_digest = object_sha256(
        [
            {
                "reviewer_id": item["reviewer_id"],
                "signature_key_id": item["signature_key_id"],
                "signature_trust_root_sha256": item[
                    "signature_trust_root_sha256"
                ],
            }
            for item in reviewers
        ]
    )
    if key_registry_digest != authority_root["reviewer_ballot_key_registry_digest"]:
        raise CheckpointEvaluationSelectionError(
            "reviewer key registry differs from independent authority root"
        )
    _verify_digest(manifest, field="roster_digest", label="reviewer roster")
    manifest["reviewers"] = reviewers
    manifest["key_registry_digest"] = key_registry_digest
    manifest["_file_sha256"] = file_sha
    return manifest


def _validate_output(value: Any, *, index: int) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "row_id",
            "opaque_candidate_id",
            "video_path",
            "video_sha256",
            "video_byte_count",
            "decode_receipt_path",
            "decode_receipt_file_sha256",
            "decode_complete",
            "frame_count",
            "fps_num",
            "fps_den",
            "width",
            "height",
            "pixel_format",
            "duration_num",
            "duration_den",
            "pts_start_num",
            "pts_start_den",
            "pts_end_num",
            "pts_end_den",
            "pts_sha256",
            "frame_content_sha256",
            "perceptual_feature_sha256",
            "actor_feature_sha256",
            "scene_feature_sha256",
            "count_frames_verified",
            "decoded_to_eof",
            "decoder_verification_tier",
            "decoder_id",
            "model_caused_unassessable",
        },
        label="decoded output %d" % index,
    )
    _identifier(row["row_id"], label="output row id")
    _identifier(row["opaque_candidate_id"], label="output opaque id")
    _boolean(row["decode_complete"], label="decode complete")
    for field in (
        "video_byte_count",
        "frame_count",
        "fps_num",
        "fps_den",
        "width",
        "height",
        "duration_num",
        "duration_den",
        "pts_start_num",
        "pts_start_den",
        "pts_end_num",
        "pts_end_den",
    ):
        if type(row[field]) is not int or row[field] < 0:
            raise CheckpointEvaluationSelectionError("output geometry differs")
    if row["video_sha256"] is not None:
        _sha(row["video_sha256"], label="output video")
    if row["decode_receipt_file_sha256"] is not None:
        _sha(row["decode_receipt_file_sha256"], label="output receipt")
    if row["pixel_format"] is not None:
        _identifier(row["pixel_format"], label="output pixel format")
    if row["pts_sha256"] is not None:
        _sha(row["pts_sha256"], label="output PTS digest")
    if row["frame_content_sha256"] is not None:
        _sha(row["frame_content_sha256"], label="output frame digest")
    for field in (
        "perceptual_feature_sha256",
        "actor_feature_sha256",
        "scene_feature_sha256",
    ):
        if row[field] is not None:
            _sha(row[field], label="output %s" % field)
    if row["decoder_verification_tier"] not in {
        PRODUCTION_DECODER_TIER,
        INJECTED_DECODER_TIER,
    }:
        raise CheckpointEvaluationSelectionError("output decoder tier differs")
    if row["decoder_id"] not in {PRODUCTION_DECODER_ID, "injected-test-decoder"}:
        raise CheckpointEvaluationSelectionError("output decoder id differs")
    _boolean(row["count_frames_verified"], label="output count_frames verified")
    _boolean(row["decoded_to_eof"], label="output decoded to EOF")
    issues = _closed(
        row["model_caused_unassessable"],
        set(AXES),
        label="model-caused unassessable axes",
    )
    for axis in AXES:
        if issues[axis] is not None and issues[axis] not in MODEL_CAUSED_REASONS:
            raise CheckpointEvaluationSelectionError(
                "unsupported model-caused failure reason"
            )
    row["model_caused_unassessable"] = issues
    return row


def _validate_decoded_output_authority(
    *,
    outputs: Mapping[tuple[str, str], Mapping[str, Any]],
    private_mapping: Mapping[str, Any],
    checkpoint_freeze: Mapping[str, Any],
    row_freeze: Mapping[str, Any],
    selection_contract: Mapping[str, Any],
    decoder: FullVideoDecoder,
    verification_tier: str,
) -> None:
    mapping = {
        (item["row_id"], item["opaque_candidate_id"]): item["checkpoint_id"]
        for item in private_mapping["rows"]
    }
    checkpoints = {
        item["checkpoint_id"]: item for item in checkpoint_freeze["checkpoints"]
    }
    rows = {item["row_id"]: item for item in row_freeze["rows"]}
    receipt_fields = {
        "schema_version",
        "complete",
        "row_id",
        "opaque_candidate_id",
        "checkpoint_id",
        "checkpoint_artifact_sha256",
        "checkpoint_artifact_tree_sha256",
        "checkpoint_artifact_manifest_file_sha256",
        "checkpoint_tensor_semantic_digest",
        "training_receipt_file_sha256",
        "source_video_sha256",
        "source_frame_content_sha256",
        "instruction_sha256",
        "inference_code_sha256",
        "inference_release_manifest_file_sha256",
        "output_video_sha256",
        "output_byte_count",
        "frame_count",
        "fps_num",
        "fps_den",
        "width",
        "height",
        "pixel_format",
        "duration_num",
        "duration_den",
        "pts_start_num",
        "pts_start_den",
        "pts_end_num",
        "pts_end_den",
        "pts_sha256",
        "frame_content_sha256",
        "perceptual_feature_sha256",
        "actor_feature_sha256",
        "scene_feature_sha256",
        "count_frames_verified",
        "decoded_to_eof",
        "decoder_verification_tier",
        "decoder_id",
        "receipt_digest",
    }
    for key, output in outputs.items():
        checkpoint_id = mapping[key]
        checkpoint = checkpoints[checkpoint_id]
        source = rows[output["row_id"]]
        expected_decoder_id = (
            PRODUCTION_DECODER_ID
            if verification_tier == PRODUCTION_DECODER_TIER
            else "injected-test-decoder"
        )
        if (
            output["decoder_verification_tier"] != verification_tier
            or output["decoder_id"] != expected_decoder_id
        ):
            raise CheckpointEvaluationSelectionError(
                "decoded output decoder authority differs"
            )
        if output["decode_complete"] is not True:
            if any(
                output[field] is not None
                for field in (
                    "video_path",
                    "video_sha256",
                    "decode_receipt_path",
                    "decode_receipt_file_sha256",
                    "pixel_format",
                    "pts_sha256",
                    "frame_content_sha256",
                    "perceptual_feature_sha256",
                    "actor_feature_sha256",
                    "scene_feature_sha256",
                )
            ) or any(
                output[field] != 0
                for field in (
                    "video_byte_count",
                    "frame_count",
                    "fps_num",
                    "fps_den",
                    "width",
                    "height",
                    "duration_num",
                    "duration_den",
                    "pts_start_num",
                    "pts_start_den",
                    "pts_end_num",
                    "pts_end_den",
                )
            ) or output["count_frames_verified"] is not False or output[
                "decoded_to_eof"
            ] is not False:
                raise CheckpointEvaluationSelectionError(
                    "incomplete decode carries media authority"
                )
            continue
        video_path, _ = _read_plain_file(
            output["video_path"],
            expected_sha256=output["video_sha256"],
            expected_size=output["video_byte_count"],
            label="decoded full81 video",
            return_payload=False,
        )
        receipt = _read_canonical_receipt(
            output["decode_receipt_path"],
            expected_sha256=output["decode_receipt_file_sha256"],
            label="decoded output receipt",
        )
        receipt = _closed(receipt, receipt_fields, label="decoded output receipt")
        expected = {
            "schema_version": DECODE_RECEIPT_SCHEMA,
            "complete": True,
            "row_id": output["row_id"],
            "opaque_candidate_id": output["opaque_candidate_id"],
            "checkpoint_id": checkpoint_id,
            "checkpoint_artifact_sha256": checkpoint["artifact_sha256"],
            "checkpoint_artifact_tree_sha256": checkpoint[
                "artifact_tree_sha256"
            ],
            "checkpoint_artifact_manifest_file_sha256": checkpoint[
                "artifact_manifest_file_sha256"
            ],
            "checkpoint_tensor_semantic_digest": checkpoint[
                "tensor_semantic_digest"
            ],
            "training_receipt_file_sha256": checkpoint[
                "training_receipt_file_sha256"
            ],
            "source_video_sha256": source["source_video_sha256"],
            "source_frame_content_sha256": source["source_media_signature"][
                "frame_content_sha256"
            ],
            "instruction_sha256": source["instruction_sha256"],
            "inference_code_sha256": selection_contract[
                "inference_code_sha256"
            ],
            "inference_release_manifest_file_sha256": selection_contract[
                "inference_release_manifest_file_sha256"
            ],
            "output_video_sha256": output["video_sha256"],
            "output_byte_count": output["video_byte_count"],
            "frame_count": output["frame_count"],
            "fps_num": output["fps_num"],
            "fps_den": output["fps_den"],
            "width": output["width"],
            "height": output["height"],
            "pixel_format": output["pixel_format"],
            "duration_num": output["duration_num"],
            "duration_den": output["duration_den"],
            "pts_start_num": output["pts_start_num"],
            "pts_start_den": output["pts_start_den"],
            "pts_end_num": output["pts_end_num"],
            "pts_end_den": output["pts_end_den"],
            "pts_sha256": output["pts_sha256"],
            "frame_content_sha256": output["frame_content_sha256"],
            "perceptual_feature_sha256": output["perceptual_feature_sha256"],
            "actor_feature_sha256": output["actor_feature_sha256"],
            "scene_feature_sha256": output["scene_feature_sha256"],
            "count_frames_verified": output["count_frames_verified"],
            "decoded_to_eof": output["decoded_to_eof"],
            "decoder_verification_tier": verification_tier,
            "decoder_id": (
                PRODUCTION_DECODER_ID
                if verification_tier == PRODUCTION_DECODER_TIER
                else "injected-test-decoder"
            ),
        }
        unsigned = dict(receipt)
        unsigned.pop("receipt_digest")
        if unsigned != expected:
            raise CheckpointEvaluationSelectionError(
                "decoded receipt checkpoint/source/instruction/code/output binding differs"
            )
        signature = {
            field: output[field] for field in MEDIA_SIGNATURE_FIELDS
        }
        _verify_actual_media(
            video_path,
            signature,
            decoder=decoder,
            expected_file_sha256=output["video_sha256"],
            label="decoded candidate full81 video",
        )
        if (
            output["frame_count"] != 81
            or output["fps_num"] != 25
            or output["fps_den"] != 1
            or output["duration_num"] != 81
            or output["duration_den"] != 25
            or output["width"] != source["output_width"]
            or output["height"] != source["output_height"]
            or output["pixel_format"] != source["output_pixel_format"]
            or output["count_frames_verified"] is not True
            or output["decoded_to_eof"] is not True
        ):
            raise CheckpointEvaluationSelectionError(
                "decoded output full81 geometry differs"
            )
        for axis in AXES:
            if (
                source["intrinsically_assessable"][axis] is False
                and output["model_caused_unassessable"][axis] is not None
            ):
                raise CheckpointEvaluationSelectionError(
                    "model-caused failure was assigned to a non-assessable axis"
                )


def _full81(output: Mapping[str, Any] | None) -> bool:
    return bool(
        output is not None
        and output["decode_complete"] is True
        and output["video_sha256"] is not None
        and output["decode_receipt_file_sha256"] is not None
        and output["frame_count"] == 81
        and output["fps_num"] == 25
        and output["fps_den"] == 1
        and output["duration_num"] == 81
        and output["duration_den"] == 25
        and output["width"] > 0
        and output["height"] > 0
        and output["pixel_format"] is not None
        and output["pts_sha256"] is not None
        and output["frame_content_sha256"] is not None
        and output["count_frames_verified"] is True
        and output["decoded_to_eof"] is True
    )


def _validate_renderer_execution_authority(
    pins: Mapping[str, Any],
    *,
    outputs: Mapping[tuple[str, str], Mapping[str, Any]],
    private_mapping: Mapping[str, Any],
    checkpoint_freeze: Mapping[str, Any],
    row_freeze: Mapping[str, Any],
    selection_contract: Mapping[str, Any],
    authority_root: Mapping[str, Any],
    signature_verifier: ExternalSignatureVerifier | None,
    timestamp_verifier: TrustedTimestampVerifier | None,
) -> dict[str, Any] | None:
    """Validate externally sealed evidence of real renderer/consumer processes."""

    if pins["renderer_execution_manifest_path"] is None:
        return None
    manifest, file_sha = _read_external_manifest(
        {
            "path": pins["renderer_execution_manifest_path"],
            "file_sha256": pins["renderer_execution_manifest_sha256"],
        },
        label="renderer/consumer execution authority",
    )
    manifest = _closed(
        manifest,
        {
            "schema_version",
            "authority_id",
            "status",
            "frozen_renderer_consumer_release_manifest_file_sha256",
            "rows",
            "fresh_process_per_output",
            "manifest_digest",
        },
        label="renderer/consumer execution authority",
    )
    if (
        manifest["schema_version"] != RENDER_EXECUTION_AUTHORITY_SCHEMA
        or manifest["status"] != "complete_and_sealed"
        or manifest["fresh_process_per_output"] is not True
        or manifest["frozen_renderer_consumer_release_manifest_file_sha256"]
        != selection_contract["inference_release_manifest_file_sha256"]
    ):
        raise CheckpointEvaluationSelectionError(
            "renderer/consumer execution authority binding differs"
        )
    _identifier(manifest["authority_id"], label="execution authority id")
    mapping = {
        (item["row_id"], item["opaque_candidate_id"]): item["checkpoint_id"]
        for item in private_mapping["rows"]
    }
    checkpoints = {
        item["checkpoint_id"]: item
        for item in checkpoint_freeze["checkpoints"]
    }
    sources = {item["row_id"]: item for item in row_freeze["rows"]}
    normalized = []
    invocations: set[str] = set()
    latest_finished: datetime | None = None
    if not isinstance(manifest["rows"], list) or not manifest["rows"]:
        raise CheckpointEvaluationSelectionError(
            "renderer/consumer execution authority rows differ"
        )
    fields = {
        "row_id",
        "opaque_candidate_id",
        "checkpoint_id",
        "fresh_process_invocation_id",
        "process_receipt_path",
        "process_receipt_file_sha256",
        "row_digest",
    }
    process_fields = {
        "schema_version",
        "invocation_id",
        "fresh_process",
        "exit_code",
        "renderer_executed",
        "consumer_executed",
        "source_video_consumed",
        "instruction_consumed",
        "checkpoint_state_loaded",
        "checkpoint_id",
        "checkpoint_loaded_state_digest",
        "source_video_sha256",
        "instruction_sha256",
        "renderer_consumer_release_manifest_file_sha256",
        "output_video_sha256",
        "decode_receipt_file_sha256",
        "started_at_utc",
        "finished_at_utc",
        "receipt_digest",
    }
    for index, item in enumerate(manifest["rows"]):
        current = _closed(item, fields, label="execution row %d" % index)
        key = (current["row_id"], current["opaque_candidate_id"])
        checkpoint_id = mapping.get(key)
        output = outputs.get(key)
        if checkpoint_id is None or output is None or current["checkpoint_id"] != checkpoint_id:
            raise CheckpointEvaluationSelectionError(
                "execution authority row is outside the frozen Cartesian product"
            )
        invocation_id = _identifier(
            current["fresh_process_invocation_id"], label="fresh process invocation"
        )
        if invocation_id in invocations:
            raise CheckpointEvaluationSelectionError(
                "renderer/consumer process invocation was reused"
            )
        invocations.add(invocation_id)
        receipt = _read_canonical_receipt(
            current["process_receipt_path"],
            expected_sha256=current["process_receipt_file_sha256"],
            label="renderer/consumer process receipt",
        )
        receipt = _closed(receipt, process_fields, label="renderer process receipt")
        started = _utc_timestamp(receipt["started_at_utc"], label="process start")
        finished = _utc_timestamp(receipt["finished_at_utc"], label="process finish")
        latest_finished = (
            finished if latest_finished is None else max(latest_finished, finished)
        )
        checkpoint = checkpoints[checkpoint_id]
        source = sources[current["row_id"]]
        if (
            receipt["schema_version"]
            != "bernini-action-editing-renderer-consumer-process-receipt-0817-v1"
            or receipt["invocation_id"] != invocation_id
            or receipt["fresh_process"] is not True
            or receipt["exit_code"] != 0
            or receipt["renderer_executed"] is not True
            or receipt["consumer_executed"] is not True
            or receipt["source_video_consumed"] is not True
            or receipt["instruction_consumed"] is not True
            or receipt["checkpoint_state_loaded"] is not True
            or receipt["checkpoint_id"] != checkpoint_id
            or receipt["checkpoint_loaded_state_digest"]
            != checkpoint["_artifact_manifest"]["loaded_state_digest"]
            or receipt["source_video_sha256"] != source["source_video_sha256"]
            or receipt["instruction_sha256"] != source["instruction_sha256"]
            or receipt["renderer_consumer_release_manifest_file_sha256"]
            != selection_contract["inference_release_manifest_file_sha256"]
            or receipt["output_video_sha256"] != output["video_sha256"]
            or receipt["decode_receipt_file_sha256"]
            != output["decode_receipt_file_sha256"]
            or not started < finished
        ):
            raise CheckpointEvaluationSelectionError(
                "renderer/consumer was not proven as a fresh real execution"
            )
        _verify_digest(receipt, field="receipt_digest", label="process receipt")
        _verify_digest(current, field="row_digest", label="execution row")
        normalized.append(current)
    keys = [(item["row_id"], item["opaque_candidate_id"]) for item in normalized]
    if keys != sorted(outputs) or len(keys) != len(set(keys)):
        raise CheckpointEvaluationSelectionError(
            "renderer/consumer execution authority does not close every output"
        )
    _verify_digest(manifest, field="manifest_digest", label="execution authority")
    signature = _verify_detached_signature(
        payload_sha256=file_sha,
        descriptor_path=pins["renderer_execution_signature_path"],
        descriptor_sha256=pins["renderer_execution_signature_sha256"],
        purpose="renderer_execution_authority",
        verifier=signature_verifier,
        expected_trust_root_sha256=authority_root["_signature"]["trust_root_sha256"],
        expected_key_id=authority_root[
            "renderer_execution_authority_signing_key_id"
        ],
        expected_tsa_id=authority_root["trusted_time_authority_id"],
        label="renderer/consumer execution authority",
        timestamp_verifier=timestamp_verifier,
    )
    if latest_finished is None or _utc_timestamp(
        signature["signed_at_utc"], label="execution authority signature time"
    ) < latest_finished:
        raise CheckpointEvaluationSelectionError(
            "execution authority was sealed before renderer/consumer completion"
        )
    manifest["rows"] = normalized
    manifest["_file_sha256"] = file_sha
    manifest["_signature"] = signature
    return manifest


def _validate_public_packet(
    value: Any,
    *,
    private_mapping: Mapping[str, Any],
    row_freeze: Mapping[str, Any],
    outputs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "schema_version",
            "private_mapping_digest",
            "keeper_commitment_digest",
            "method_hidden",
            "checkpoint_hidden",
            "column_order_randomized",
            "rows",
            "packet_digest",
        },
        label="public blind packet",
    )
    if row["schema_version"] != PUBLIC_PACKET_SCHEMA:
        raise CheckpointEvaluationSelectionError("public packet schema differs")
    if row["private_mapping_digest"] != private_mapping["mapping_digest"]:
        raise CheckpointEvaluationSelectionError(
            "public/private mapping digest differs"
        )
    if (
        row["keeper_commitment_digest"]
        != private_mapping["keeper_commitment"]["commitment_digest"]
    ):
        raise CheckpointEvaluationSelectionError(
            "public keeper commitment binding differs"
        )
    if (
        row["method_hidden"] is not True
        or row["checkpoint_hidden"] is not True
        or row["column_order_randomized"] is not True
    ):
        raise CheckpointEvaluationSelectionError("public blinding contract differs")
    if not isinstance(row["rows"], list):
        raise CheckpointEvaluationSelectionError("public packet rows differ")
    source_by_id = {item["row_id"]: item for item in row_freeze["rows"]}
    mapping_keys = {
        (item["row_id"], item["opaque_candidate_id"])
        for item in private_mapping["rows"]
    }
    rows = []
    fields = {
        "row_id",
        "opaque_candidate_id",
        "source_video_sha256",
        "instruction",
        "instruction_sha256",
        "candidate_video_sha256",
        "blind_row_digest",
    }
    for index, item in enumerate(row["rows"]):
        current = _closed(item, fields, label="public blind row %d" % index)
        key = (current["row_id"], current["opaque_candidate_id"])
        source = source_by_id.get(current["row_id"])
        if source is None or key not in mapping_keys:
            raise CheckpointEvaluationSelectionError("public blind row is unmapped")
        if (
            current["source_video_sha256"] != source["source_video_sha256"]
            or current["instruction"] != source["instruction"]
            or current["instruction_sha256"] != source["instruction_sha256"]
        ):
            raise CheckpointEvaluationSelectionError(
                "public row source/instruction binding differs"
            )
        output = outputs.get(key)
        expected_video = None if output is None else output["video_sha256"]
        if current["candidate_video_sha256"] != expected_video:
            raise CheckpointEvaluationSelectionError(
                "public row decoded-video binding differs"
            )
        if expected_video is not None:
            _sha(expected_video, label="public candidate video")
        _verify_digest(current, field="blind_row_digest", label="public blind row")
        rows.append(current)
    actual_keys = {(item["row_id"], item["opaque_candidate_id"]) for item in rows}
    if actual_keys != mapping_keys or len(actual_keys) != len(rows):
        raise CheckpointEvaluationSelectionError(
            "public packet does not close private mapping"
        )
    _verify_digest(row, field="packet_digest", label="public packet")
    row["rows"] = rows
    return row


def _ballot_signature_message(ballot: Mapping[str, Any]) -> bytes:
    signed = dict(ballot)
    signed.pop("reviewer_signature_hex", None)
    signed.pop("ballot_digest", None)
    return canonical_json_bytes(
        {
            "schema_version": "bernini-action-editing-reviewer-ballot-signature-message-0817-v1",
            "ballot": signed,
        }
    )


def _validate_ballots(
    value: Any,
    *,
    public_packet: Mapping[str, Any],
    row_freeze: Mapping[str, Any],
    reviewer_roster: Mapping[str, Any],
    authority_root: Mapping[str, Any],
    reviewer_signature_verifiers: Mapping[str, ExternalSignatureVerifier] | None,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list):
        raise CheckpointEvaluationSelectionError("ballots are not a list")
    public_by_key = {
        (item["row_id"], item["opaque_candidate_id"]): item
        for item in public_packet["rows"]
    }
    source_by_id = {item["row_id"]: item for item in row_freeze["rows"]}
    roster_by_id = {
        item["reviewer_id"]: item for item in reviewer_roster["reviewers"]
    }
    fields = {
        "schema_version",
        "public_packet_digest",
        "row_id",
        "opaque_candidate_id",
        "blind_row_digest",
        "candidate_video_sha256",
        "reviewer_id",
        "reviewer_role",
        "independent_review",
        "full_81_reviewed",
        "committed_at_utc",
        "labels",
        "reviewer_signature_subject_id",
        "reviewer_signature_key_id",
        "reviewer_signature_trust_root_sha256",
        "reviewer_signature_hex",
        "ballot_digest",
    }
    ballots = []
    for index, item in enumerate(value):
        ballot = _closed(item, fields, label="ballot %d" % index)
        if ballot["schema_version"] != BALLOT_SCHEMA:
            raise CheckpointEvaluationSelectionError("ballot schema differs")
        if ballot["public_packet_digest"] != public_packet["packet_digest"]:
            raise CheckpointEvaluationSelectionError(
                "ballot public packet binding differs"
            )
        for name in ("row_id", "opaque_candidate_id", "reviewer_id"):
            _identifier(ballot[name], label="ballot %s" % name)
        if ballot["reviewer_role"] not in {"primary", "adjudicator"}:
            raise CheckpointEvaluationSelectionError("reviewer role differs")
        roster_entry = roster_by_id.get(ballot["reviewer_id"])
        if (
            roster_entry is None
            or roster_entry["reviewer_role"] != ballot["reviewer_role"]
            or roster_entry["eligible"] is not True
        ):
            raise CheckpointEvaluationSelectionError(
                "ballot reviewer is outside the external roster"
            )
        if (
            reviewer_signature_verifiers is None
            or ballot["reviewer_signature_subject_id"]
            != roster_entry["signature_subject_id"]
            or ballot["reviewer_signature_key_id"]
            != roster_entry["signature_key_id"]
            or ballot["reviewer_signature_trust_root_sha256"]
            != roster_entry["signature_trust_root_sha256"]
        ):
            raise CheckpointEvaluationSelectionError(
                "reviewer ballot signature authority differs"
            )
        _boolean(ballot["independent_review"], label="independent review")
        _boolean(ballot["full_81_reviewed"], label="full81 reviewed")
        _utc_timestamp(ballot["committed_at_utc"], label="ballot commitment time")
        public = public_by_key.get(
            (ballot["row_id"], ballot["opaque_candidate_id"])
        )
        if public is None:
            raise CheckpointEvaluationSelectionError("ballot row is not public")
        if (
            ballot["blind_row_digest"] != public["blind_row_digest"]
            or ballot["candidate_video_sha256"]
            != public["candidate_video_sha256"]
        ):
            raise CheckpointEvaluationSelectionError(
                "ballot blind-row/video binding differs"
            )
        labels = _closed(ballot["labels"], set(AXES), label="ballot labels")
        source = source_by_id[ballot["row_id"]]
        for axis in AXES:
            expected_values = (
                LABELS
                if source["intrinsically_assessable"][axis]
                else (NOT_ASSESSABLE,)
            )
            if labels[axis] not in expected_values:
                raise CheckpointEvaluationSelectionError(
                    "ballot label differs from frozen assessability semantics"
                )
        ballot["labels"] = labels
        signature_hex = ballot["reviewer_signature_hex"]
        if (
            not isinstance(signature_hex, str)
            or len(signature_hex) < 64
            or len(signature_hex) > 1024
            or len(signature_hex) % 2
            or re.fullmatch(r"[0-9a-f]+", signature_hex) is None
        ):
            raise CheckpointEvaluationSelectionError(
                "reviewer ballot signature encoding differs"
            )
        try:
            reviewer_verifier = reviewer_signature_verifiers[ballot["reviewer_id"]]
            if (
                getattr(reviewer_verifier, "trust_root_sha256", None)
                != roster_entry["signature_trust_root_sha256"]
                or getattr(reviewer_verifier, "key_id", None)
                != roster_entry["signature_key_id"]
            ):
                raise CheckpointEvaluationSelectionError(
                    "reviewer ballot verifier is not independently pinned"
                )
            signature_valid = reviewer_verifier.verify(
                _ballot_signature_message(ballot), signature_hex
            )
        except Exception as error:
            raise CheckpointEvaluationSelectionError(
                "reviewer ballot external signature verifier failed"
            ) from error
        if signature_valid is not True:
            raise CheckpointEvaluationSelectionError(
                "reviewer ballot external signature is invalid"
            )
        _verify_digest(ballot, field="ballot_digest", label="ballot")
        ballots.append(ballot)
    keys = [
        (
            item["row_id"],
            item["opaque_candidate_id"],
            item["reviewer_id"],
            item["reviewer_role"],
        )
        for item in ballots
    ]
    if len(keys) != len(set(keys)):
        raise CheckpointEvaluationSelectionError("duplicate reviewer ballot")
    ballot_set_digest = object_sha256(
        sorted(item["ballot_digest"] for item in ballots)
    )
    return ballots, ballot_set_digest


def _validate_ballot_seal_authority(
    pins: Mapping[str, Any],
    *,
    ballots: Sequence[Mapping[str, Any]],
    ballot_set_digest: str,
    public_packet: Mapping[str, Any],
    reviewer_roster: Mapping[str, Any],
    authority_root: Mapping[str, Any],
    signature_verifier: ExternalSignatureVerifier | None,
    timestamp_verifier: TrustedTimestampVerifier | None,
) -> dict[str, Any]:
    manifest, file_sha = _read_external_manifest(
        {
            "path": pins["ballot_seal_manifest_path"],
            "file_sha256": pins["ballot_seal_manifest_sha256"],
        },
        label="ballot seal authority manifest",
    )
    manifest = _closed(
        manifest,
        {
            "schema_version",
            "authority_id",
            "status",
            "reviewer_roster_manifest_file_sha256",
            "public_packet_digest",
            "commitments",
            "ballot_set_digest",
            "seal_closed_at_utc",
            "mapping_unblind_not_before_utc",
            "trusted_time_authority_id",
            "seal_digest",
        },
        label="ballot seal authority manifest",
    )
    if (
        manifest["schema_version"] != BALLOT_SEAL_SCHEMA
        or manifest["status"] != "sealed"
        or manifest["reviewer_roster_manifest_file_sha256"]
        != reviewer_roster["_file_sha256"]
        or manifest["public_packet_digest"] != public_packet["packet_digest"]
        or manifest["ballot_set_digest"] != ballot_set_digest
    ):
        raise CheckpointEvaluationSelectionError(
            "external ballot seal binding differs"
        )
    _identifier(manifest["authority_id"], label="ballot seal authority id")
    _identifier(
        manifest["trusted_time_authority_id"], label="trusted time authority id"
    )
    closed_at = _utc_timestamp(
        manifest["seal_closed_at_utc"], label="ballot seal close time"
    )
    not_before = _utc_timestamp(
        manifest["mapping_unblind_not_before_utc"],
        label="mapping unblind not-before time",
    )
    if not closed_at < not_before:
        raise CheckpointEvaluationSelectionError(
            "ballot seal did not close before permitted unblinding"
        )
    signature = _verify_detached_signature(
        payload_sha256=file_sha,
        descriptor_path=pins["ballot_seal_signature_path"],
        descriptor_sha256=pins["ballot_seal_signature_sha256"],
        purpose="ballot_seal",
        verifier=signature_verifier,
        expected_trust_root_sha256=pins["precommitted_trust_root_sha256"],
        expected_key_id=authority_root["signing_key_id"],
        expected_tsa_id=authority_root["trusted_time_authority_id"],
        label="ballot seal authority",
        timestamp_verifier=timestamp_verifier,
    )
    signed_at = _utc_timestamp(
        signature["signed_at_utc"], label="ballot seal signature time"
    )
    if signed_at != closed_at or signed_at >= not_before:
        raise CheckpointEvaluationSelectionError(
            "ballot seal signature/TSA time differs from closed review window"
        )
    if not isinstance(manifest["commitments"], list):
        raise CheckpointEvaluationSelectionError("ballot commitments differ")
    commitments = []
    fields = {
        "ballot_digest",
        "reviewer_id",
        "reviewer_role",
        "row_id",
        "opaque_candidate_id",
        "committed_at_utc",
    }
    for index, item in enumerate(manifest["commitments"]):
        current = _closed(item, fields, label="ballot commitment %d" % index)
        _sha(current["ballot_digest"], label="ballot commitment digest")
        committed = _utc_timestamp(
            current["committed_at_utc"], label="ballot committed time"
        )
        if committed > closed_at:
            raise CheckpointEvaluationSelectionError(
                "ballot was committed after the external seal closed"
            )
        commitments.append(current)
    expected = sorted(
        [
            {
                "ballot_digest": item["ballot_digest"],
                "reviewer_id": item["reviewer_id"],
                "reviewer_role": item["reviewer_role"],
                "row_id": item["row_id"],
                "opaque_candidate_id": item["opaque_candidate_id"],
                "committed_at_utc": item["committed_at_utc"],
            }
            for item in ballots
        ],
        key=lambda item: item["ballot_digest"],
    )
    if commitments != expected:
        raise CheckpointEvaluationSelectionError(
            "external ballot seal does not close every ballot commitment"
        )
    _verify_digest(manifest, field="seal_digest", label="ballot seal authority")
    manifest["commitments"] = commitments
    manifest["_file_sha256"] = file_sha
    manifest["_signature"] = signature
    manifest["_not_before"] = not_before
    return manifest


def _validate_unblinding(
    value: Any,
    *,
    private_mapping: Mapping[str, Any],
    ballot_set_digest: str,
    public_packet: Mapping[str, Any],
    ballot_seal: Mapping[str, Any],
) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "schema_version",
            "private_mapping_digest",
            "keeper_commitment_digest",
            "keeper_id",
            "blinding_key_hex",
            "ballot_set_digest",
            "ballot_seal_manifest_file_sha256",
            "ballots_sealed_before_unblinding",
            "mapping_opened_after_ballot_seal",
            "unblinded_at_utc",
            "unblinding_digest",
        },
        label="review unblinding",
    )
    if row["schema_version"] != UNBLINDING_SCHEMA:
        raise CheckpointEvaluationSelectionError("unblinding schema differs")
    commitment = private_mapping["keeper_commitment"]
    key_hex = row["blinding_key_hex"]
    if (
        not isinstance(key_hex, str)
        or len(key_hex) != 64
        or re.fullmatch(r"[0-9a-f]{64}", key_hex) is None
    ):
        raise CheckpointEvaluationSelectionError("revealed blinding key differs")
    key = bytes.fromhex(key_hex)
    if (
        row["private_mapping_digest"] != private_mapping["mapping_digest"]
        or row["keeper_commitment_digest"] != commitment["commitment_digest"]
        or row["keeper_id"] != commitment["keeper_id"]
        or hashlib.sha256(key).hexdigest() != commitment["blinding_key_sha256"]
        or row["ballot_set_digest"] != ballot_set_digest
        or row["ballot_seal_manifest_file_sha256"]
        != ballot_seal["_file_sha256"]
        or row["ballots_sealed_before_unblinding"] is not True
        or row["mapping_opened_after_ballot_seal"] is not True
    ):
        raise CheckpointEvaluationSelectionError(
            "ballot seal/private unblinding order differs"
        )
    unblinded_at = _utc_timestamp(
        row["unblinded_at_utc"], label="mapping unblinded time"
    )
    if unblinded_at < ballot_seal["_not_before"]:
        raise CheckpointEvaluationSelectionError(
            "mapping was opened before the external ballot seal allowed it"
        )
    expected_opaque = {}
    for mapping in private_mapping["rows"]:
        message = (
            "id\0%s\0%s" % (mapping["row_id"], mapping["checkpoint_id"])
        ).encode("utf-8")
        opaque_id = "blind-" + hmac.new(key, message, hashlib.sha256).hexdigest()[:32]
        if mapping["opaque_candidate_id"] != opaque_id:
            raise CheckpointEvaluationSelectionError(
                "private mapping is not keeper-key-derived"
            )
        expected_opaque[(mapping["row_id"], opaque_id)] = mapping
    public_keys = [
        (item["row_id"], item["opaque_candidate_id"])
        for item in public_packet["rows"]
    ]
    if set(public_keys) != set(expected_opaque):
        raise CheckpointEvaluationSelectionError(
            "public packet keeper mapping closure differs"
        )
    expected_order = sorted(
        public_keys,
        key=lambda item: hmac.new(
            key,
            ("order\0%s\0%s" % item).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest(),
    )
    if public_keys != expected_order:
        raise CheckpointEvaluationSelectionError(
            "public packet order is not keeper-key-randomized"
        )
    _verify_digest(row, field="unblinding_digest", label="review unblinding")
    return row


def _validate_automatic_diagnostics(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    row = _closed(
        value,
        {"schema_version", "evidence_digest", "used_for_selection"},
        label="automatic diagnostics",
    )
    if row["schema_version"] != DIAGNOSTIC_SCHEMA:
        raise CheckpointEvaluationSelectionError("diagnostic schema differs")
    _sha(row["evidence_digest"], label="automatic diagnostic evidence")
    if row["used_for_selection"] is not False:
        raise CheckpointEvaluationSelectionError(
            "automatic metrics may not select a checkpoint"
        )
    return row


def validate_input(
    value: Any,
    *,
    authority_pins: Any,
    decoder: FullVideoDecoder | None = None,
    signature_verifier: ExternalSignatureVerifier | None = None,
    reviewer_signature_verifiers: Mapping[
        str, ExternalSignatureVerifier
    ] | None = None,
    timestamp_verifier: TrustedTimestampVerifier | None = None,
) -> dict[str, Any]:
    row = _closed(
        value,
        {
            "schema_version",
            "evaluation_id",
            "mode",
            "checkpoint_freeze",
            "row_freeze",
            "locked_split",
            "selection_contract",
            "evaluator_calibration",
            "decoded_outputs",
            "private_mapping",
            "public_blind_packet",
            "ballots",
            "review_unblinding",
            "automatic_diagnostics",
            "input_digest",
        },
        label="evaluation input",
    )
    if row["schema_version"] != INPUT_SCHEMA:
        raise CheckpointEvaluationSelectionError("evaluation input schema differs")
    _identifier(row["evaluation_id"], label="evaluation id")
    if row["mode"] not in {FORMAL_MODE, ENGINEERING_MODE}:
        raise CheckpointEvaluationSelectionError("evaluation mode differs")
    pins = _validate_authority_pins(authority_pins)
    authority_root = _validate_authority_root(
        pins,
        verifier=signature_verifier,
        timestamp_verifier=timestamp_verifier,
    )
    if decoder is None:
        _validate_fixed_production_media_tools(authority_root)
    formal_authority = _validate_formal_training_authority(
        pins, authority_root=authority_root
    )
    media_decoder = (
        _SEALED_PRODUCTION_MEDIA_DECODER if decoder is None else decoder
    )
    verification_tier = (
        PRODUCTION_DECODER_TIER if decoder is None else INJECTED_DECODER_TIER
    )
    checkpoint_freeze = _validate_checkpoint_freeze(
        row["checkpoint_freeze"], formal_authority=formal_authority
    )
    row_freeze = _validate_row_freeze(row["row_freeze"], decoder=media_decoder)
    source_equivalence_authority = _validate_source_equivalence_authority(
        pins, row_freeze=row_freeze, authority_root=authority_root
    )
    reviewer_roster = _validate_reviewer_roster_authority(
        pins, authority_root=authority_root
    )
    expected_reviewer_ids = {
        item["reviewer_id"] for item in reviewer_roster["reviewers"]
    }
    if (
        reviewer_signature_verifiers is None
        or set(reviewer_signature_verifiers) != expected_reviewer_ids
    ):
        raise CheckpointEvaluationSelectionError(
            "reviewer verifier registry does not close the frozen roster"
        )
    locked_split, split_reasons = _validate_locked_split(
        row["locked_split"], row_freeze=row_freeze
    )
    if locked_split is not None and (
        pins["locked_split_digest"] != locked_split["lock_digest"]
    ):
        raise CheckpointEvaluationSelectionError(
            "locked split differs from caller external pin"
        )
    selection_contract, selection_input_digest = _validate_selection_contract(
        row["selection_contract"],
        checkpoint_freeze=checkpoint_freeze,
        row_freeze=row_freeze,
        locked_split=locked_split,
        source_equivalence_authority=source_equivalence_authority,
        reviewer_roster_authority=reviewer_roster,
        authority_pins=pins,
        authority_root=authority_root,
    )
    if not isinstance(row["decoded_outputs"], list):
        raise CheckpointEvaluationSelectionError("decoded outputs are not a list")
    outputs_list = [
        _validate_output(item, index=index)
        for index, item in enumerate(row["decoded_outputs"])
    ]
    output_keys = [
        (item["row_id"], item["opaque_candidate_id"])
        for item in outputs_list
    ]
    if len(output_keys) != len(set(output_keys)):
        raise CheckpointEvaluationSelectionError("decoded output keys collide")
    outputs = dict(zip(output_keys, outputs_list))
    private_mapping = _validate_private_mapping(
        row["private_mapping"],
        checkpoint_freeze=checkpoint_freeze,
        row_freeze=row_freeze,
    )
    if (
        private_mapping["keeper_commitment"]["commitment_digest"]
        != pins["keeper_commitment_digest"]
    ):
        raise CheckpointEvaluationSelectionError(
            "keeper commitment differs from caller external pin"
        )
    mapped_keys = {
        (item["row_id"], item["opaque_candidate_id"])
        for item in private_mapping["rows"]
    }
    if not set(outputs).issubset(mapped_keys):
        raise CheckpointEvaluationSelectionError("decoded output is not mapped")
    _validate_decoded_output_authority(
        outputs=outputs,
        private_mapping=private_mapping,
        checkpoint_freeze=checkpoint_freeze,
        row_freeze=row_freeze,
        selection_contract=selection_contract,
        decoder=media_decoder,
        verification_tier=verification_tier,
    )
    # The formal execution-authority validator is intentionally dormant in
    # this unprovisioned release.  In particular, a caller-provided manifest
    # full of self-reported "renderer_executed" booleans is never consumed as
    # formal evidence.  Engineering mode may still exercise the skeleton.
    renderer_execution_authority = (
        None
        if row["mode"] == FORMAL_MODE
        else _validate_renderer_execution_authority(
            pins,
            outputs=outputs,
            private_mapping=private_mapping,
            checkpoint_freeze=checkpoint_freeze,
            row_freeze=row_freeze,
            selection_contract=selection_contract,
            authority_root=authority_root,
            signature_verifier=signature_verifier,
            timestamp_verifier=timestamp_verifier,
        )
    )
    public_packet = _validate_public_packet(
        row["public_blind_packet"],
        private_mapping=private_mapping,
        row_freeze=row_freeze,
        outputs=outputs,
    )
    ballots, ballot_set_digest = _validate_ballots(
        row["ballots"],
        public_packet=public_packet,
        row_freeze=row_freeze,
        reviewer_roster=reviewer_roster,
        authority_root=authority_root,
        reviewer_signature_verifiers=reviewer_signature_verifiers,
    )
    ballot_seal = _validate_ballot_seal_authority(
        pins,
        ballots=ballots,
        ballot_set_digest=ballot_set_digest,
        public_packet=public_packet,
        reviewer_roster=reviewer_roster,
        authority_root=authority_root,
        signature_verifier=signature_verifier,
        timestamp_verifier=timestamp_verifier,
    )
    unblinding = _validate_unblinding(
        row["review_unblinding"],
        private_mapping=private_mapping,
        ballot_set_digest=ballot_set_digest,
        public_packet=public_packet,
        ballot_seal=ballot_seal,
    )
    calibration, calibration_reasons = _validate_calibration(
        row["evaluator_calibration"]
    )
    diagnostics = _validate_automatic_diagnostics(row["automatic_diagnostics"])
    _verify_digest(row, field="input_digest", label="evaluation input")
    row.update(
        {
            "checkpoint_freeze": checkpoint_freeze,
            "_authority_root": authority_root,
            "_formal_training_authority": formal_authority,
            "row_freeze": row_freeze,
            "_source_equivalence_authority": source_equivalence_authority,
            "_reviewer_roster_authority": reviewer_roster,
            "_renderer_execution_authority": renderer_execution_authority,
            "_ballot_seal_authority": ballot_seal,
            "_authority_pins": pins,
            "locked_split": locked_split,
            "selection_contract": selection_contract,
            "evaluator_calibration": calibration,
            "decoded_outputs": outputs_list,
            "private_mapping": private_mapping,
            "public_blind_packet": public_packet,
            "ballots": ballots,
            "review_unblinding": unblinding,
            "automatic_diagnostics": diagnostics,
            "_split_reasons": split_reasons,
            "_calibration_reasons": calibration_reasons,
            "_selection_input_digest": selection_input_digest,
            "_verification_tier": verification_tier,
            "_root_signature_verifier_is_production": type(
                signature_verifier
            ) is OpenSSLExternalSignatureVerifier,
            "_reviewer_signature_verifiers_are_production": bool(
                reviewer_signature_verifiers
            )
            and all(
                type(verifier) is OpenSSLExternalSignatureVerifier
                for verifier in reviewer_signature_verifiers.values()
            ),
            "_timestamp_verifier": timestamp_verifier,
            "_formal_module_api_injection": row["mode"] == FORMAL_MODE
            and any(
                item is not None
                for item in (
                    decoder,
                    signature_verifier,
                    reviewer_signature_verifiers,
                    timestamp_verifier,
                )
            ),
        }
    )
    return row


def _formal_external_trust_blockers(evidence: Mapping[str, Any]) -> list[str]:
    """This release cannot execute or authorize a formal validator.

    The parameter is intentionally unused.  There is no mutable registry to
    refill and no verifier/TSA/decoder object that can turn the result into a
    formal one.  A future production implementation requires a different,
    independently reviewed source release and fixed CLI-only process ABI.
    """

    del evidence
    return [FORMAL_PRODUCTION_AUTHORITY_BLOCKER]


def _formal_split_blockers(row_freeze: Mapping[str, Any]) -> list[str]:
    rows = row_freeze["rows"]
    blockers = []
    split_counts = {
        name: sum(item["split"] == name for item in rows) for name in SPLITS
    }
    if len(rows) != 500 or split_counts != FORMAL_SPLIT_COUNTS:
        blockers.append("PROMOTION_SPLIT_EXACT_500_STRATA_MISMATCH")
    if len({item["source_id"] for item in rows}) != len(rows):
        blockers.append("PROMOTION_SPLIT_SOURCE_NOT_UNIQUE")
    if len({item["source_video_sha256"] for item in rows}) != len(rows):
        blockers.append("PROMOTION_SPLIT_SOURCE_BYTES_NOT_UNIQUE")
    if len({item["upstream_group_id"] for item in rows}) != len(rows):
        blockers.append("PROMOTION_SPLIT_UPSTREAM_GROUP_NOT_UNIQUE")
    if len({item["source_equivalence_group_id"] for item in rows}) != len(rows):
        blockers.append("PROMOTION_SPLIT_EQUIVALENT_SOURCE_REPEATED")
    for field, minimum, blocker in (
        (
            "source_cluster_id",
            MIN_FORMAL_SOURCE_CLUSTERS,
            "PROMOTION_SPLIT_INSUFFICIENT_INDEPENDENT_SOURCE_CLUSTERS",
        ),
        (
            "actor_scene_cluster_id",
            MIN_FORMAL_ACTOR_SCENE_CLUSTERS,
            "PROMOTION_SPLIT_INSUFFICIENT_INDEPENDENT_ACTOR_SCENE_CLUSTERS",
        ),
    ):
        counts: dict[str, int] = {}
        for item in rows:
            counts[item[field]] = counts.get(item[field], 0) + 1
        if len(counts) < minimum or len(counts) == len(rows):
            blockers.append(blocker)
    for split in SPLITS:
        split_rows = [item for item in rows if item["split"] == split]
        for field, label in (
            ("source_cluster_id", "SOURCE"),
            ("actor_scene_cluster_id", "ACTOR_SCENE"),
        ):
            counts: dict[str, int] = {}
            for item in split_rows:
                counts[item[field]] = counts.get(item[field], 0) + 1
            if len(counts) < 2 or not any(count >= 2 for count in counts.values()):
                blockers.append(
                    "PROMOTION_STRATUM_%s_%s_CLUSTER_BOOTSTRAP_DEGENERATE"
                    % (split.upper(), label)
                )
    denominators = {
        axis: sum(item["intrinsically_assessable"][axis] for item in rows)
        for axis in AXES
    }
    if denominators["action"] != 450 or denominators["order"] != 450:
        blockers.append("ACTION_ORDER_ELIGIBLE_DENOMINATOR_NOT_450")
    if denominators["ownership"] != 100:
        blockers.append("OWNERSHIP_ELIGIBLE_DENOMINATOR_NOT_100")
    if denominators["noop"] != 50:
        blockers.append("NOOP_ELIGIBLE_DENOMINATOR_NOT_50")
    for axis in ("identity", "background", "camera", "quality"):
        if denominators[axis] != 500:
            blockers.append("%s_ELIGIBLE_DENOMINATOR_NOT_500" % axis.upper())
    if any(denominators[axis] == 0 for axis in AXES):
        blockers.append("EMPTY_AXIS_ELIGIBLE_DENOMINATOR")
    return blockers


def _review_resolution(
    *,
    output: Mapping[str, Any] | None,
    ballots: Sequence[Mapping[str, Any]],
    assessable: Mapping[str, bool],
) -> tuple[dict[str, str], bool, dict[str, str]]:
    """Resolve two primary ballots and, only on disagreement, one adjudicator."""

    full81 = _full81(output)
    primaries = [item for item in ballots if item["reviewer_role"] == "primary"]
    adjudicators = [
        item for item in ballots if item["reviewer_role"] == "adjudicator"
    ]
    usable_primary = [
        item
        for item in primaries
        if item["independent_review"] is True
        and item["full_81_reviewed"] is True
    ]
    usable_adjudicators = [
        item
        for item in adjudicators
        if item["independent_review"] is True
        and item["full_81_reviewed"] is True
    ]
    reviewer_ids = [item["reviewer_id"] for item in ballots]
    assessable_axes = [axis for axis in AXES if assessable[axis] is True]
    protocol_ok = (
        full81
        and len(primaries) == len(usable_primary) == 2
        and len(reviewer_ids) == len(set(reviewer_ids))
    )
    any_disagreement = bool(
        len(usable_primary) == 2
        and any(
            usable_primary[0]["labels"][axis]
            != usable_primary[1]["labels"][axis]
            for axis in assessable_axes
        )
    )
    if any_disagreement:
        protocol_ok = protocol_ok and (
            len(adjudicators) == len(usable_adjudicators) == 1
        )
    elif adjudicators:
        protocol_ok = False

    labels = {}
    reasons = {}
    for axis in AXES:
        if assessable[axis] is False:
            labels[axis] = NOT_ASSESSABLE
            reasons[axis] = "INTRINSICALLY_NOT_ASSESSABLE"
            continue
        if not full81:
            labels[axis] = "fail"
            reasons[axis] = "MISSING_OR_INVALID_FULL81_MODEL_OUTPUT"
            continue
        model_reason = output["model_caused_unassessable"][axis]
        if model_reason is not None:
            labels[axis] = "fail"
            reasons[axis] = "MODEL_CAUSED_%s" % model_reason.upper()
            continue
        if len(usable_primary) != 2:
            labels[axis] = "abstain"
            reasons[axis] = "TWO_PRIMARY_FULL81_BALLOTS_MISSING"
            continue
        left = usable_primary[0]["labels"][axis]
        right = usable_primary[1]["labels"][axis]
        if left == right:
            labels[axis] = left
            reasons[axis] = "PRIMARY_CONSENSUS"
            continue
        if len(usable_adjudicators) != 1:
            labels[axis] = "abstain"
            reasons[axis] = "THIRD_REVIEWER_REQUIRED_BUT_MISSING"
            continue
        labels[axis] = usable_adjudicators[0]["labels"][axis]
        reasons[axis] = (
            "THIRD_REVIEWER_ABSTAIN"
            if labels[axis] == "abstain"
            else "THIRD_REVIEWER_ADJUDICATION"
        )
    return labels, protocol_ok, reasons


def _build_human_matrix(
    evidence: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, dict[str, str]]],
    dict[str, dict[str, Any]],
]:
    checkpoint_ids = [
        item["checkpoint_id"] for item in evidence["checkpoint_freeze"]["checkpoints"]
    ]
    rows = evidence["row_freeze"]["rows"]
    mapping = {
        (item["row_id"], item["checkpoint_id"]): item["opaque_candidate_id"]
        for item in evidence["private_mapping"]["rows"]
    }
    outputs = {
        (item["row_id"], item["opaque_candidate_id"]): item
        for item in evidence["decoded_outputs"]
    }
    ballots_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for ballot in evidence["ballots"]:
        key = (ballot["row_id"], ballot["opaque_candidate_id"])
        ballots_by_key.setdefault(key, []).append(ballot)

    matrix = {
        checkpoint_id: {axis: {} for axis in AXES}
        for checkpoint_id in checkpoint_ids
    }
    details: dict[str, dict[str, Any]] = {}
    for checkpoint_id in checkpoint_ids:
        protocol_complete = True
        full81_count = 0
        model_forced_failure_count = 0
        resolution_rows = []
        for source in rows:
            row_id = source["row_id"]
            opaque_id = mapping[(row_id, checkpoint_id)]
            key = (row_id, opaque_id)
            output = outputs.get(key)
            full81_count += int(_full81(output))
            labels, protocol_ok, reasons = _review_resolution(
                output=output,
                ballots=ballots_by_key.get(key, []),
                assessable=source["intrinsically_assessable"],
            )
            protocol_complete = protocol_complete and protocol_ok
            for axis in AXES:
                matrix[checkpoint_id][axis][row_id] = labels[axis]
                model_forced_failure_count += int(
                    reasons[axis].startswith("MODEL_CAUSED_")
                    and source["intrinsically_assessable"][axis]
                )
            resolution_rows.append(
                {
                    "row_id": row_id,
                    "opaque_candidate_id": opaque_id,
                    "labels": labels,
                    "reasons": reasons,
                }
            )
        details[checkpoint_id] = {
            "expected_output_count": len(rows),
            "full81_output_count": full81_count,
            "all_outputs_full81": full81_count == len(rows),
            "blind_review_protocol_complete": protocol_complete,
            "model_caused_failure_axis_count": model_forced_failure_count,
            "resolution_digest": object_sha256(resolution_rows),
        }
    return matrix, details


def _axis_report(
    *,
    checkpoint_id: str,
    matrix: Mapping[str, Mapping[str, Mapping[str, str]]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for axis in AXES:
        eligible = [
            item
            for item in rows
            if item["intrinsically_assessable"][axis] is True
        ]
        labels = [matrix[checkpoint_id][axis][item["row_id"]] for item in eligible]
        denominator = len(eligible)
        passed = labels.count("pass")
        abstained = labels.count("abstain")
        failed = denominator - passed - abstained
        result[axis] = {
            "eligible_denominator": denominator,
            "pass_count": passed,
            "fail_count": failed,
            "abstain_count": abstained,
            "abstain_counted_as_fail_in_denominator": True,
            "pass_rate": None if denominator == 0 else passed / denominator,
            "abstain_rate": None if denominator == 0 else abstained / denominator,
        }
    return result


def _bootstrap_seed(*parts: str) -> int:
    return int.from_bytes(
        hashlib.sha256("\0".join(parts).encode("utf-8")).digest()[:8],
        "big",
    )


def _cluster_bootstrap_ci(
    *,
    differences: Sequence[float],
    cluster_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    if len(differences) != len(cluster_ids) or not differences:
        raise CheckpointEvaluationSelectionError(
            "paired bootstrap observations differ"
        )
    grouped: dict[str, list[float]] = {}
    for cluster_id, difference in zip(cluster_ids, differences):
        grouped.setdefault(cluster_id, []).append(float(difference))
    ordered = sorted(grouped)
    sums = [sum(grouped[item]) for item in ordered]
    counts = [len(grouped[item]) for item in ordered]
    cluster_count = len(ordered)
    plan_digest = object_sha256(
        {
            "schema_version": "paired-cluster-resample-plan-0817-v1",
            "seed": seed,
            "ordered_cluster_ids": ordered,
            "cluster_draw_count": cluster_count,
            "resamples": BOOTSTRAP_RESAMPLES,
        }
    )
    if len(set(differences)) == 1:
        # The ordinary cluster bootstrap is degenerate in this exact case;
        # returning its exact distribution avoids 10,000 redundant draws.
        constant = float(differences[0])
        return {
            "method": "paired_cluster_bootstrap_percentile",
            "resamples": BOOTSTRAP_RESAMPLES,
            "unique_cluster_count": cluster_count,
            "effective_paired_n": len(differences),
            "resample_plan_digest": plan_digest,
            "lower_95": constant,
            "upper_95": constant,
        }
    generator = random.Random(seed)
    choose = generator.randrange
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        numerator = 0.0
        denominator = 0
        for _ in range(cluster_count):
            index = choose(cluster_count)
            numerator += sums[index]
            denominator += counts[index]
        draws.append(numerator / denominator)
    draws.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(draws) - 1)
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        if lower_index == upper_index:
            return draws[lower_index]
        fraction = position - lower_index
        return draws[lower_index] + fraction * (
            draws[upper_index] - draws[lower_index]
        )

    lower = percentile(0.025)
    upper = percentile(0.975)
    return {
        "method": "paired_cluster_bootstrap_percentile",
        "resamples": BOOTSTRAP_RESAMPLES,
        "unique_cluster_count": cluster_count,
        "effective_paired_n": len(differences),
        "resample_plan_digest": plan_digest,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _paired_comparison(
    *,
    left_id: str,
    right_id: str,
    matrix: Mapping[str, Mapping[str, Mapping[str, str]]],
    rows: Sequence[Mapping[str, Any]],
    selection_input_digest: str,
) -> dict[str, Any]:
    axes = {}
    for axis in AXES:
        axes[axis] = _paired_axis_comparison(
            left_id=left_id,
            right_id=right_id,
            matrix=matrix,
            rows=rows,
            axis=axis,
            selection_input_digest=selection_input_digest,
            seed_scope="pooled",
        )
    comparison = {
        "left_checkpoint_id": left_id,
        "right_checkpoint_id": right_id,
        "axes": axes,
        "weighted_score": None,
    }
    comparison["comparison_digest"] = object_sha256(comparison)
    return comparison


def _paired_axis_comparison(
    *,
    left_id: str,
    right_id: str,
    matrix: Mapping[str, Mapping[str, Mapping[str, str]]],
    rows: Sequence[Mapping[str, Any]],
    axis: str,
    selection_input_digest: str,
    seed_scope: str,
) -> dict[str, Any] | None:
    eligible = [
        item for item in rows if item["intrinsically_assessable"][axis] is True
    ]
    if not eligible:
        return None
    differences = [
        float(matrix[left_id][axis][item["row_id"]] == "pass")
        - float(matrix[right_id][axis][item["row_id"]] == "pass")
        for item in eligible
    ]
    pair_scope = "\0".join(sorted((left_id, right_id)))
    source_ci = _cluster_bootstrap_ci(
        differences=differences,
        cluster_ids=[item["source_cluster_id"] for item in eligible],
        seed=_bootstrap_seed(
            selection_input_digest,
            pair_scope,
            axis,
            seed_scope,
            "source",
        ),
    )
    actor_scene_ci = _cluster_bootstrap_ci(
        differences=differences,
        cluster_ids=[item["actor_scene_cluster_id"] for item in eligible],
        seed=_bootstrap_seed(
            selection_input_digest,
            pair_scope,
            axis,
            seed_scope,
            "actor_scene",
        ),
    )
    return {
        "paired_point_delta": sum(differences) / len(differences),
        "left_pass_rate": sum(
            matrix[left_id][axis][item["row_id"]] == "pass" for item in eligible
        ) / len(eligible),
        "right_pass_rate": sum(
            matrix[right_id][axis][item["row_id"]] == "pass" for item in eligible
        ) / len(eligible),
        "source_cluster_ci": source_ci,
        "actor_scene_cluster_ci": actor_scene_ci,
    }


def _stratum_reference_report(
    *,
    candidate_id: str,
    base_id: str,
    matrix: Mapping[str, Mapping[str, Mapping[str, str]]],
    rows: Sequence[Mapping[str, Any]],
    selection_input_digest: str,
) -> dict[str, Any]:
    result = {}
    for split in SPLITS:
        split_rows = [item for item in rows if item["split"] == split]
        axes = {}
        for axis in AXES:
            comparison = _paired_axis_comparison(
                left_id=candidate_id,
                right_id=base_id,
                matrix=matrix,
                rows=split_rows,
                axis=axis,
                selection_input_digest=selection_input_digest,
                seed_scope="stratum:%s" % split,
            )
            if comparison is not None:
                axes[axis] = comparison
        result[split] = {
            "row_count": len(split_rows),
            "axes": axes,
        }
    return result


def _reference_gate_blockers(
    *,
    stage: str,
    comparison: Mapping[str, Any],
    strata: Mapping[str, Any],
) -> list[str]:
    policy = STAGE_REFERENCE_GATES[stage]
    blockers = []
    for axis in ("action", "order"):
        current = comparison["axes"][axis]
        if current["left_pass_rate"] < HARD_MINIMUM_PASS_RATES[axis]:
            blockers.append("STAGE_ABSOLUTE_FLOOR_FAIL_%s" % axis.upper())
        if current["paired_point_delta"] < policy["%s_delta" % axis]:
            blockers.append("BASE_DELTA_FAIL_%s" % axis.upper())
        if (
            current["source_cluster_ci"]["lower_95"] <= policy["ci_lower"]
            or current["actor_scene_cluster_ci"]["lower_95"]
            <= policy["ci_lower"]
        ):
            blockers.append("BASE_DUAL_CLUSTER_CI_FAIL_%s" % axis.upper())
    for axis in ("identity", "ownership", "background", "camera", "quality", "noop"):
        current = comparison["axes"][axis]
        if (
            current["source_cluster_ci"]["lower_95"] <= -0.03
            or current["actor_scene_cluster_ci"]["lower_95"] <= -0.03
        ):
            blockers.append("BASE_NONINFERIORITY_CI_FAIL_%s" % axis.upper())
    for split, report in strata.items():
        axes = report["axes"]
        for axis in ("action", "order"):
            if axis in axes and axes[axis]["paired_point_delta"] < MAX_STRATUM_ACTION_REGRESSION:
                blockers.append(
                    "STRATUM_REGRESSION_%s_%s" % (split.upper(), axis.upper())
                )
        for axis in ("identity", "background", "camera", "quality"):
            if axes[axis]["left_pass_rate"] < MIN_STRATUM_PRESERVATION_PASS_RATE:
                blockers.append(
                    "STRATUM_PRESERVATION_FAIL_%s_%s"
                    % (split.upper(), axis.upper())
                )
        if "ownership" in axes and axes["ownership"]["left_pass_rate"] < 0.85:
            blockers.append("STRATUM_OWNERSHIP_FAIL_%s" % split.upper())
        if "noop" in axes and axes["noop"]["left_pass_rate"] < 0.95:
            blockers.append("STRATUM_NOOP_FAIL_%s" % split.upper())
    return blockers


def _statistically_supported_dominance(comparison: Mapping[str, Any]) -> bool:
    axes = comparison["axes"]
    no_point_regression = all(
        axes[axis]["paired_point_delta"] >= 0.0 for axis in AXES
    )
    supported_strict_gain = any(
        axes[axis]["paired_point_delta"] > 0.0
        and axes[axis]["source_cluster_ci"]["lower_95"] > 0.0
        and axes[axis]["actor_scene_cluster_ci"]["lower_95"] > 0.0
        for axis in AXES
    )
    return no_point_regression and supported_strict_gain


def coordinate(
    value: Any,
    *,
    authority_pins: Any,
    decoder: FullVideoDecoder | None = None,
    signature_verifier: ExternalSignatureVerifier | None = None,
    reviewer_signature_verifiers: Mapping[
        str, ExternalSignatureVerifier
    ] | None = None,
    timestamp_verifier: TrustedTimestampVerifier | None = None,
) -> dict[str, Any]:
    if not isinstance(value, bytes):
        raise CheckpointEvaluationSelectionError(
            "coordinate requires strict canonical JSON bytes; mappings are not an input authority"
        )
    parsed = _strict_json_bytes(value, label="evaluation input")
    evidence = validate_input(
        parsed,
        authority_pins=authority_pins,
        decoder=decoder,
        signature_verifier=signature_verifier,
        reviewer_signature_verifiers=reviewer_signature_verifiers,
        timestamp_verifier=timestamp_verifier,
    )
    checkpoints = evidence["checkpoint_freeze"]["checkpoints"]
    checkpoint_by_id = {item["checkpoint_id"]: item for item in checkpoints}
    base_id = evidence["checkpoint_freeze"]["base_checkpoint_id"]
    candidate_ids = [
        item["checkpoint_id"] for item in checkpoints if item["role"] == "candidate"
    ]
    rows = evidence["row_freeze"]["rows"]
    matrix, engineering = _build_human_matrix(evidence)
    denominators = {
        axis: sum(item["intrinsically_assessable"][axis] for item in rows)
        for axis in AXES
    }

    global_blockers = list(evidence["_split_reasons"])
    if evidence["mode"] != FORMAL_MODE:
        global_blockers.append("ENGINEERING_MODE_NO_QUALITY_SELECTION")
    else:
        global_blockers.extend(_formal_split_blockers(evidence["row_freeze"]))
        global_blockers.extend(_formal_external_trust_blockers(evidence))
        if evidence["_formal_module_api_injection"]:
            global_blockers.append(
                "FORMAL_MODULE_API_INJECTION_ENGINEERING_ONLY"
            )
    if evidence["_verification_tier"] != PRODUCTION_DECODER_TIER:
        global_blockers.append("INJECTED_DECODER_NONFORMAL_TEST_ONLY")
    if not engineering[base_id]["all_outputs_full81"]:
        global_blockers.append("BASE_FULL81_EVIDENCE_INCOMPLETE")
    if not engineering[base_id]["blind_review_protocol_complete"]:
        global_blockers.append("BASE_BLIND_REVIEW_PROTOCOL_INCOMPLETE")
    global_blockers = sorted(set(global_blockers))

    base_axis_report = _axis_report(
        checkpoint_id=base_id,
        matrix=matrix,
        rows=rows,
    )
    base_report = {
        "checkpoint_id": base_id,
        "engineering_evidence": engineering[base_id],
        "human_axis_results": base_axis_report,
        "automatic_metric_results": None,
        "diagnostic_only": True,
    }

    candidate_reports = []
    quality_pool = []
    comparisons = []
    comparison_by_pair = {}
    for checkpoint_id in candidate_ids:
        checkpoint = checkpoint_by_id[checkpoint_id]
        pre_d0 = _pre_d0_tainted(checkpoint)
        blockers = []
        if pre_d0:
            blockers.append("PRE_D0_ENGINEERING_ONLY_NOT_A_QUALITY_CANDIDATE")
        if not engineering[checkpoint_id]["all_outputs_full81"]:
            blockers.append("CANDIDATE_FULL81_EVIDENCE_INCOMPLETE")
        if not engineering[checkpoint_id]["blind_review_protocol_complete"]:
            blockers.append("CANDIDATE_BLIND_REVIEW_PROTOCOL_INCOMPLETE")
        axis_report = None if pre_d0 else _axis_report(
            checkpoint_id=checkpoint_id,
            matrix=matrix,
            rows=rows,
        )
        if axis_report is not None:
            for axis in AXES:
                if axis_report[axis]["eligible_denominator"] == 0:
                    blockers.append("EMPTY_%s_DENOMINATOR" % axis.upper())
                if (
                    axis_report[axis]["abstain_rate"] is not None
                    and axis_report[axis]["abstain_rate"] > ABSTAIN_LIMIT
                ):
                    blockers.append("ABSTAIN_RATE_GT_10_PERCENT_%s" % axis.upper())
            for axis, minimum in HARD_MINIMUM_PASS_RATES.items():
                rate = axis_report[axis]["pass_rate"]
                if rate is None or rate < minimum:
                    blockers.append("HARD_MINIMUM_FAIL_%s" % axis.upper())
        base_comparison = None
        stratum_reference = None
        reference_gate_blockers = []
        candidate_evidence_complete = (
            not pre_d0
            and engineering[checkpoint_id]["all_outputs_full81"]
            and engineering[checkpoint_id]["blind_review_protocol_complete"]
            and not any(
                blocker.startswith("ABSTAIN_RATE_GT_10_PERCENT_")
                or blocker.startswith("EMPTY_")
                for blocker in blockers
            )
            and not global_blockers
            and evidence["mode"] == FORMAL_MODE
            and evidence["_verification_tier"] == PRODUCTION_DECODER_TIER
        )
        if candidate_evidence_complete:
            base_comparison = _paired_comparison(
                left_id=checkpoint_id,
                right_id=base_id,
                matrix=matrix,
                rows=rows,
                selection_input_digest=evidence["_selection_input_digest"],
            )
            stratum_reference = _stratum_reference_report(
                candidate_id=checkpoint_id,
                base_id=base_id,
                matrix=matrix,
                rows=rows,
                selection_input_digest=evidence["_selection_input_digest"],
            )
            reference_gate_blockers = _reference_gate_blockers(
                stage=checkpoint["_training_stage"],
                comparison=base_comparison,
                strata=stratum_reference,
            )
            comparisons.append(base_comparison)
            comparison_by_pair[(checkpoint_id, base_id)] = base_comparison
        blockers.extend(reference_gate_blockers)
        blockers.extend(global_blockers)
        blockers = sorted(set(blockers))
        descriptive_evidence_complete = (
            evidence["mode"] != FORMAL_MODE
            and not blockers
            and not pre_d0
            and base_comparison is not None
        )
        blockers.append(
            "COORDINATOR_DESCRIPTIVE_STATISTICS_ONLY_HUMAN_AUDIT_REQUIRED"
        )
        blockers = sorted(set(blockers))
        quality_candidate = False
        report = {
            "checkpoint_id": checkpoint_id,
            "training_stage": checkpoint["_training_stage"],
            "pre_d0_tainted": pre_d0,
            "engineering_evidence": engineering[checkpoint_id],
            "quality_candidate": quality_candidate,
            "descriptive_evidence_complete": descriptive_evidence_complete,
            "human_axis_results": axis_report,
            "base_reference_comparison_digest": (
                None
                if base_comparison is None
                else base_comparison["comparison_digest"]
            ),
            "stage_reference_gate": {
                "policy": (
                    None
                    if pre_d0
                    else dict(STAGE_REFERENCE_GATES[checkpoint["_training_stage"]])
                ),
                "blockers": reference_gate_blockers,
            },
            "stratum_reference_results": stratum_reference,
            "automatic_metric_results": None,
            "blockers": blockers,
        }
        candidate_reports.append(report)
        # This coordinator has no adjudicative authority.  A separately
        # chartered human audit may consume this report, but no code path here
        # can enrol a checkpoint into a quality pool or name a winner.
    for left_index, left_id in enumerate(quality_pool):
        for right_id in quality_pool[left_index + 1 :]:
            forward = _paired_comparison(
                left_id=left_id,
                right_id=right_id,
                matrix=matrix,
                rows=rows,
                selection_input_digest=evidence["_selection_input_digest"],
            )
            reverse = _paired_comparison(
                left_id=right_id,
                right_id=left_id,
                matrix=matrix,
                rows=rows,
                selection_input_digest=evidence["_selection_input_digest"],
            )
            comparisons.extend((forward, reverse))
            comparison_by_pair[(left_id, right_id)] = forward
            comparison_by_pair[(right_id, left_id)] = reverse

    dominated = set()
    dominance_edges = []
    for left_id in quality_pool:
        for right_id in quality_pool:
            if left_id == right_id:
                continue
            comparison = comparison_by_pair[(left_id, right_id)]
            if _statistically_supported_dominance(comparison):
                dominated.add(right_id)
                dominance_edges.append(
                    {
                        "dominant_checkpoint_id": left_id,
                        "dominated_checkpoint_id": right_id,
                        "comparison_digest": comparison["comparison_digest"],
                    }
                )
    pareto: list[str] = []
    pre_d0_in_quality = sorted(
        checkpoint_id
        for checkpoint_id in set(quality_pool) | set(pareto)
        if _pre_d0_tainted(checkpoint_by_id[checkpoint_id])
    )
    if pre_d0_in_quality:
        raise CheckpointEvaluationSelectionError(
            "internal invariant violated: PRE_D0 entered quality selection"
        )
    status = "DESCRIPTIVE_STATISTICS_ONLY"

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "evaluation_id": evidence["evaluation_id"],
        "evidence_input_digest": evidence["input_digest"],
        "selection_input_digest": evidence["_selection_input_digest"],
        "external_authority_file_sha256": {
            "independent_root": evidence["_authority_root"]["_file_sha256"],
            "independent_root_signature": evidence["_authority_root"][
                "_signature"
            ]["_file_sha256"],
            "formal_training": evidence["_formal_training_authority"][
                "_file_sha256"
            ],
            "source_equivalence": evidence["_source_equivalence_authority"][
                "_file_sha256"
            ],
            "reviewer_roster": evidence["_reviewer_roster_authority"][
                "_file_sha256"
            ],
            "ballot_seal": evidence["_ballot_seal_authority"]["_file_sha256"],
            "ballot_seal_signature": evidence["_ballot_seal_authority"][
                "_signature"
            ]["_file_sha256"],
            "inference_release": evidence["selection_contract"][
                "inference_release_manifest_file_sha256"
            ],
        },
        "caller_preregistered_commitments": {
            "locked_split_digest": evidence["_authority_pins"][
                "locked_split_digest"
            ],
            "keeper_commitment_digest": evidence["_authority_pins"][
                "keeper_commitment_digest"
            ],
        },
        "independent_signature_trust": {
            "precommitted_trust_root_sha256": evidence["_authority_pins"][
                "precommitted_trust_root_sha256"
            ],
            "precommitted_key_id": evidence["_authority_pins"][
                "precommitted_key_id"
            ],
            "trusted_time_authority_id": evidence["_authority_pins"][
                "trusted_time_authority_id"
            ],
            "authority_root_signature_verified": True,
            "reviewer_ballot_signatures_verified": True,
            "ballot_seal_signature_verified": True,
            "engineering_evidence_checks_only": True,
            "formal_eligibility_conferred": False,
        },
        "mode": evidence["mode"],
        "effective_mode": (
            evidence["mode"]
            if evidence["mode"] != FORMAL_MODE
            and evidence["_verification_tier"] == PRODUCTION_DECODER_TIER
            else ENGINEERING_MODE
        ),
        "formal_production_authority": {
            "provisioned": False,
            "blocker": "FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED",
            "formal_validator_executed": False,
            "runtime_registry_or_api_provisioning_supported": False,
            "module_api_decoder_verifier_tsa_injection_engineering_only": True,
            "future_enablement_requires_new_reviewed_cli_release": True,
        },
        "decoder_verification": {
            "tier": evidence["_verification_tier"],
            "formal_decoder_id": PRODUCTION_DECODER_ID,
            "injected_decoder_can_enter_pareto": False,
        },
        "status": status,
        "selection_semantics": "descriptive_statistics_only",
        "axis_order": list(AXES),
        "fixed_eligible_denominators": denominators,
        "abstain_policy": {
            "abstain_counted_as_fail": True,
            "maximum_rate": ABSTAIN_LIMIT,
            "two_primary_reviewers_required": True,
            "third_reviewer_required_on_disagreement": True,
            "third_reviewer_abstain_remains_in_denominator": True,
        },
        "model_caused_blur_occlusion_crop_is_fail": True,
        "bootstrap_contract": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence": 0.95,
            "clusterings": ["source_cluster_id", "actor_scene_cluster_id"],
            "minimum_independent_source_clusters": MIN_FORMAL_SOURCE_CLUSTERS,
            "minimum_independent_actor_scene_clusters": MIN_FORMAL_ACTOR_SCENE_CLUSTERS,
            "paired": True,
            "seed_bound_only_to_preregistered_selection_input": True,
        },
        "hard_minimum_pass_rates": dict(HARD_MINIMUM_PASS_RATES),
        "stage_reference_gates": {
            key: dict(value) for key, value in STAGE_REFERENCE_GATES.items()
        },
        "global_blockers": global_blockers,
        "base_report": base_report,
        "candidate_reports": candidate_reports,
        "paired_cluster_comparisons": comparisons,
        "dominance_edges": dominance_edges,
        "pareto_checkpoint_ids": pareto,
        "quality_selection_performed": bool(pareto),
        "winner_checkpoint_id": None,
        "weighted_score": None,
        "automatic_metrics_present": evidence["automatic_diagnostics"] is not None,
        "automatic_evaluator_status": (
            "missing"
            if evidence["evaluator_calibration"] is None
            else "unqualified_diagnostic_only"
            if evidence["_calibration_reasons"]
            else "qualified_but_diagnostic_only"
        ),
        "human_only_selection_allowed_when_auto_unqualified": True,
        "automatic_metrics_used_for_selection": False,
        "automatic_metric_winner": None,
        "pre_d0_quality_checkpoint_ids": pre_d0_in_quality,
        "scientific_promotion_authorized": False,
        "promotion_boundary": "HUMAN_AUDIT_CANDIDATE_EVIDENCE_ONLY",
    }
    # Final, literal formal safety invariant.  This intentionally does not
    # call a helper or read a mutable blocker constant.  Under the supported
    # threat model (authenticated source and the original ``coordinate``
    # entrypoint), monkeypatching ordinary helpers or globals cannot remove the
    # hard block or manufacture a selection-capable receipt.
    if evidence["mode"] == "promotion_validation":
        receipt["global_blockers"] = sorted(
            set(receipt["global_blockers"])
            | {"FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED"}
        )
        receipt["effective_mode"] = "engineering_comparison"
        receipt["formal_production_authority"]["blocker"] = (
            "FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED"
        )
        for report in receipt["candidate_reports"]:
            report["descriptive_evidence_complete"] = False
            report["quality_candidate"] = False
            report["blockers"] = sorted(
                set(report["blockers"])
                | {"FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED"}
            )
        receipt["winner_checkpoint_id"] = None
        receipt["pareto_checkpoint_ids"] = []
        receipt["quality_selection_performed"] = False
        receipt["scientific_promotion_authorized"] = False
        if not (
            "FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED"
            in receipt["global_blockers"]
            and receipt["effective_mode"] == "engineering_comparison"
            and receipt["formal_production_authority"]["blocker"]
            == "FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED"
            and all(
                report["descriptive_evidence_complete"] is False
                and report["quality_candidate"] is False
                for report in receipt["candidate_reports"]
            )
            and receipt["winner_checkpoint_id"] is None
            and receipt["pareto_checkpoint_ids"] == []
            and receipt["quality_selection_performed"] is False
            and receipt["scientific_promotion_authorized"] is False
        ):
            raise CheckpointEvaluationSelectionError(
                "internal formal hard-block receipt invariant violated"
            )
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def validate_receipt(
    value: Any,
    *,
    evidence: Any,
    authority_pins: Any,
    decoder: FullVideoDecoder | None = None,
    signature_verifier: ExternalSignatureVerifier | None = None,
    reviewer_signature_verifiers: Mapping[
        str, ExternalSignatureVerifier
    ] | None = None,
    timestamp_verifier: TrustedTimestampVerifier | None = None,
) -> dict[str, Any]:
    expected = coordinate(
        evidence,
        authority_pins=authority_pins,
        decoder=decoder,
        signature_verifier=signature_verifier,
        reviewer_signature_verifiers=reviewer_signature_verifiers,
        timestamp_verifier=timestamp_verifier,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise CheckpointEvaluationSelectionError(
            "selection receipt differs from frozen evidence"
        )
    _verify_digest(expected, field="receipt_digest", label="selection receipt")
    return expected


def _load(path: str) -> bytes:
    try:
        target = Path(path)
        if (
            not target.is_absolute()
            or os.path.normpath(path) != path
            or path == os.path.sep
        ):
            raise CheckpointEvaluationSelectionError(
                "evaluation input path must be absolute and normalized"
            )
        before = target.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise CheckpointEvaluationSelectionError(
                "evaluation input is not a plain file"
            )
        if before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o444:
            raise CheckpointEvaluationSelectionError(
                "evaluation input must be immutable mode-0444 with nlink=1"
            )
        if target.resolve(strict=True) != target:
            raise CheckpointEvaluationSelectionError(
                "evaluation input path contains a symlink/non-canonical component"
            )
        payload = target.read_bytes()
        after = target.lstat()
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_uid,
            item.st_gid,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise CheckpointEvaluationSelectionError(
                "evaluation input changed while read"
            )
        _strict_json_bytes(payload, label="evaluation input")
        return payload
    except CheckpointEvaluationSelectionError:
        raise
    except OSError as error:
        raise CheckpointEvaluationSelectionError(
            "cannot read canonical evaluation input: %s" % error
        ) from error


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise CheckpointEvaluationSelectionError(
                    "selection receipt write made no progress"
                )
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authority-root-manifest", required=True)
    parser.add_argument("--authority-root-sha256", required=True)
    parser.add_argument("--authority-root-signature", required=True)
    parser.add_argument("--authority-root-signature-sha256", required=True)
    parser.add_argument("--precommitted-public-key", required=True)
    parser.add_argument("--precommitted-trust-root-sha256", required=True)
    parser.add_argument("--precommitted-key-id", required=True)
    parser.add_argument("--trusted-time-authority-id", required=True)
    parser.add_argument("--formal-authority-manifest", required=True)
    parser.add_argument("--formal-authority-sha256", required=True)
    parser.add_argument(
        "--authorized-checkpoint-id", action="append", required=True
    )
    parser.add_argument("--locked-split-digest")
    parser.add_argument("--keeper-commitment-digest", required=True)
    parser.add_argument("--source-equivalence-manifest", required=True)
    parser.add_argument("--source-equivalence-sha256", required=True)
    parser.add_argument("--reviewer-roster-manifest", required=True)
    parser.add_argument("--reviewer-roster-sha256", required=True)
    parser.add_argument("--ballot-seal-manifest", required=True)
    parser.add_argument("--ballot-seal-sha256", required=True)
    parser.add_argument("--ballot-seal-signature", required=True)
    parser.add_argument("--ballot-seal-signature-sha256", required=True)
    parser.add_argument("--inference-release-manifest", required=True)
    parser.add_argument("--inference-release-sha256", required=True)
    parser.add_argument("--renderer-execution-manifest", required=True)
    parser.add_argument("--renderer-execution-sha256", required=True)
    parser.add_argument("--renderer-execution-signature", required=True)
    parser.add_argument("--renderer-execution-signature-sha256", required=True)
    parser.add_argument(
        "--reviewer-public-key",
        action="append",
        required=True,
        help="reviewer_id:key_id:public_key_sha256:absolute_public_key_path",
    )
    parser.add_argument("--tsa-ca", required=True)
    parser.add_argument("--tsa-ca-sha256", required=True)
    parser.add_argument("--tsa-token-registry", required=True)
    parser.add_argument("--tsa-token-registry-sha256", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute() or output.parent == output:
        raise CheckpointEvaluationSelectionError(
            "output must be an absolute non-root path"
        )
    authority_pins = {
        "authority_root_manifest_path": args.authority_root_manifest,
        "authority_root_manifest_sha256": args.authority_root_sha256,
        "authority_root_signature_path": args.authority_root_signature,
        "authority_root_signature_sha256": args.authority_root_signature_sha256,
        "precommitted_trust_root_sha256": args.precommitted_trust_root_sha256,
        "precommitted_key_id": args.precommitted_key_id,
        "trusted_time_authority_id": args.trusted_time_authority_id,
        "formal_training_manifest_path": args.formal_authority_manifest,
        "formal_training_manifest_sha256": args.formal_authority_sha256,
        "authorized_checkpoint_ids": sorted(args.authorized_checkpoint_id),
        "locked_split_digest": args.locked_split_digest,
        "keeper_commitment_digest": args.keeper_commitment_digest,
        "source_equivalence_manifest_path": args.source_equivalence_manifest,
        "source_equivalence_manifest_sha256": args.source_equivalence_sha256,
        "reviewer_roster_manifest_path": args.reviewer_roster_manifest,
        "reviewer_roster_manifest_sha256": args.reviewer_roster_sha256,
        "ballot_seal_manifest_path": args.ballot_seal_manifest,
        "ballot_seal_manifest_sha256": args.ballot_seal_sha256,
        "ballot_seal_signature_path": args.ballot_seal_signature,
        "ballot_seal_signature_sha256": args.ballot_seal_signature_sha256,
        "inference_release_manifest_path": args.inference_release_manifest,
        "inference_release_manifest_sha256": args.inference_release_sha256,
        "renderer_execution_manifest_path": args.renderer_execution_manifest,
        "renderer_execution_manifest_sha256": args.renderer_execution_sha256,
        "renderer_execution_signature_path": args.renderer_execution_signature,
        "renderer_execution_signature_sha256": args.renderer_execution_signature_sha256,
    }
    signature_verifier = OpenSSLExternalSignatureVerifier(
        public_key_path=args.precommitted_public_key,
        expected_sha256=args.precommitted_trust_root_sha256,
        key_id=args.precommitted_key_id,
    )
    reviewer_signature_verifiers = {}
    for descriptor in args.reviewer_public_key:
        parts = descriptor.split(":", 3)
        if len(parts) != 4:
            raise CheckpointEvaluationSelectionError(
                "reviewer public-key descriptor differs"
            )
        reviewer_id, key_id, public_key_sha256, public_key_path = parts
        reviewer_id = _identifier(reviewer_id, label="reviewer verifier id")
        if reviewer_id in reviewer_signature_verifiers:
            raise CheckpointEvaluationSelectionError(
                "reviewer verifier descriptor collides"
            )
        reviewer_signature_verifiers[reviewer_id] = OpenSSLExternalSignatureVerifier(
            public_key_path=public_key_path,
            expected_sha256=public_key_sha256,
            key_id=key_id,
        )
    timestamp_verifier = OpenSSLRFC3161TimestampVerifier(
        tsa_id=args.trusted_time_authority_id,
        tsa_ca_path=args.tsa_ca,
        tsa_ca_sha256=args.tsa_ca_sha256,
        token_registry_path=args.tsa_token_registry,
        token_registry_sha256=args.tsa_token_registry_sha256,
    )
    receipt = coordinate(
        _load(args.input),
        authority_pins=authority_pins,
        signature_verifier=signature_verifier,
        reviewer_signature_verifiers=reviewer_signature_verifiers,
        timestamp_verifier=timestamp_verifier,
    )
    _write_create_only(output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
