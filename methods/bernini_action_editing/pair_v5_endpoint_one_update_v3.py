#!/usr/bin/env python3
"""Trusted endpoint qualification for one PAIR-v5 WORLD8 update.

This module is intentionally independent from the PAIR-v5 v1/v2 selector and
training manifests.  It consumes the active-v4 native action receipt, decoded
post-video preservation evidence, and a round/policy-bound rollout sidecar.
All optimizer fields are derived here; callers cannot submit scores or flags.

The first supported canary is generation round zero under the frozen Bernini
base policy with no CIO and no Action-LoRA.  Later rounds are represented by
the same contract, but require a rollout-time policy attestation and the
previous Action-LoRA file.  A manifest always authorizes exactly one optimizer
step and becomes stale immediately after that step.
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
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence

import decoded_temporal_event_evaluator_v1 as decoded_event


ACTIVE_ACTION_SCHEMA = "bernini-pair-v5-native-rv2v-global-action-score-v4"
DECODED_PRESERVATION_SCHEMA = (
    "bernini-pair-v5-source-bound-preservation-candidate-v1"
)
TEMPORAL_COUNTERFACTUAL_SCHEMA = decoded_event.TEMPORAL_PROJECTION_SCHEMA
EVENT81_SCHEMA = decoded_event.EVENT81_PROJECTION_SCHEMA
LEGACY_TEMPORAL_COUNTERFACTUAL_SCHEMA = (
    "bernini-pair-v5-same-video-temporal-counterfactual-packet-v1"
)
LEGACY_EVENT81_SCHEMA = "bernini-pair-v5-start-transition-terminal-hold-packet-v1"
LEGACY_ROLLOUT_SCHEMA = "pair-v5-native-rv2v4-rollout-receipt-v1"
NATIVE_RECEIPT_SCHEMA = "bernini-native-identity-generation-canary-v1"

POLICY_SCHEMA = "bernini-pair-v5-parent-policy-v3"
POLICY_ATTESTATION_SCHEMA = "bernini-pair-v5-rollout-policy-attestation-v3"
ROLLOUT_EVIDENCE_SCHEMA = "bernini-pair-v5-round-rollout-evidence-v3"
GATE_POLICY_SCHEMA = "bernini-pair-v5-endpoint-gate-policy-v3"
BUILD_REQUEST_SCHEMA = "bernini-pair-v5-endpoint-build-request-v3"
CANDIDATE_SCHEMA = "bernini-pair-v5-qualified-endpoint-v3"
PAIR_SCHEMA = "bernini-pair-v5-qualified-fit-pair-v3"
MANIFEST_SCHEMA = "bernini-pair-v5-one-update-manifest-v3"

FRAME_COUNT = 81
LATENT_PHASES = 21
LATENT_CHANNELS = 16
REFERENCE_INDICES = (0, 27, 53, 80)
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2

PRIMARY_PRESERVATION_METRICS = {
    "identity": "source_identity_appearance_proxy",
    "background": "background_appearance_fixed_grid_proxy",
    "camera": "source_bound_spatial_layout_viewpoint_proxy",
    "temporal": "non_target_temporal_consistency_proxy",
    "quality": "decode_video_quality_diagnostic",
}
WRONG_SOURCE_DIAGNOSTIC_KEYS = (
    "source_identity_appearance_wrong_source_proxy",
    "source_identity_appearance_correct_minus_wrong_margin",
    "source_identity_appearance_wrong_normalized_contrast",
    "background_appearance_wrong_source_fixed_grid_proxy",
    "background_appearance_correct_minus_wrong_margin",
    "source_bound_spatial_layout_wrong_source_proxy",
    "source_bound_spatial_layout_correct_minus_wrong_margin",
    "source_bound_spatial_layout_wrong_normalized_contrast_proxy",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class PairV5EndpointV3Error(RuntimeError):
    """Raised before an untrusted or stale endpoint can reach an optimizer."""


class NoAuthorizedPairsError(PairV5EndpointV3Error):
    """No two-source, fit-only population satisfies all hard gates."""


def canonical_json_bytes(value: Any, *, ensure_ascii: bool = True) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii" if ensure_ascii else "utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5EndpointV3Error("value is not canonical finite JSON") from error


def object_sha256(value: Any, *, ensure_ascii: bool = True) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, ensure_ascii=ensure_ascii)
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise PairV5EndpointV3Error("hash target must be an absolute plain file")
    before = value.stat()
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = value.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PairV5EndpointV3Error("file changed while hashing")
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PairV5EndpointV3Error(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise PairV5EndpointV3Error(f"{label} must be a safe identifier")
    return value


def _finite(value: Any, *, label: str, unit: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairV5EndpointV3Error(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (unit and not 0.0 <= result <= 1.0):
        raise PairV5EndpointV3Error(f"{label} lies outside its contract")
    return result


def _closed(value: Any, fields: Iterable[str], *, label: str) -> Mapping[str, Any]:
    expected = set(fields)
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PairV5EndpointV3Error(
            f"{label} closure differs; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return value


def _verify_embedded(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
    ensure_ascii: bool = True,
) -> str:
    unsigned = dict(value)
    declared = _sha256(unsigned.pop(field, None), label=f"{label} {field}")
    if object_sha256(unsigned, ensure_ascii=ensure_ascii) != declared:
        raise PairV5EndpointV3Error(f"{label} embedded digest differs")
    return declared


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PairV5EndpointV3Error(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise PairV5EndpointV3Error(f"{label} contains {token}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairV5EndpointV3Error(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PairV5EndpointV3Error(f"{label} root must be an object")
    canonical_json_bytes(value)
    return value


def file_binding(path: str | Path) -> dict[str, Any]:
    value = Path(path).resolve(strict=True)
    if not value.is_file() or value.is_symlink() or value == Path("/"):
        raise PairV5EndpointV3Error("file binding target differs")
    return {"path": str(value), "sha256": file_sha256(value)}


def validate_file_binding(
    value: Any, *, label: str, verify_bytes: bool = True
) -> dict[str, str]:
    row = _closed(value, {"path", "sha256"}, label=label)
    expected = _sha256(row["sha256"], label=f"{label} SHA-256")
    if not isinstance(row["path"], str) or not row["path"] or "\x00" in row["path"]:
        raise PairV5EndpointV3Error(f"{label} path differs")
    path = Path(row["path"])
    if not path.is_absolute() or path == Path("/"):
        raise PairV5EndpointV3Error(f"{label} path must be absolute and non-root")
    if verify_bytes and file_sha256(path) != expected:
        raise PairV5EndpointV3Error(f"{label} file SHA-256 differs")
    return {"path": str(path), "sha256": expected}


def decoded_temporal_event_evaluator_binding() -> dict[str, Any]:
    """The one implementation allowed to mint both endpoint projections."""

    return file_binding(Path(decoded_event.__file__).resolve())


def load_bound_json(
    value: Any, *, label: str, verify_bytes: bool = True
) -> tuple[dict[str, Any], dict[str, str]]:
    binding = validate_file_binding(value, label=label, verify_bytes=verify_bytes)
    if not verify_bytes:
        raise PairV5EndpointV3Error("JSON evidence cannot skip byte verification")
    return _strict_json(Path(binding["path"]).read_bytes(), label=label), binding


def make_parent_policy(
    *,
    generation_round: int,
    checkpoint_tree_sha256: str,
    cio_adapter: Optional[Mapping[str, Any]] = None,
    action_lora: Optional[Mapping[str, Any]] = None,
    previous_update_receipt_digest: Optional[str] = None,
) -> dict[str, Any]:
    if type(generation_round) is not int or generation_round < 0:
        raise PairV5EndpointV3Error("generation round must be a nonnegative integer")
    checkpoint = _sha256(checkpoint_tree_sha256, label="checkpoint tree SHA-256")
    cio = None if cio_adapter is None else validate_file_binding(
        cio_adapter, label="parent CIO adapter", verify_bytes=True
    )
    action = None if action_lora is None else validate_file_binding(
        action_lora, label="parent Action-LoRA", verify_bytes=True
    )
    previous = None
    if previous_update_receipt_digest is not None:
        previous = _sha256(
            previous_update_receipt_digest,
            label="previous update receipt digest",
        )
    if generation_round == 0:
        if cio is not None or action is not None or previous is not None:
            raise PairV5EndpointV3Error(
                "round0 policy must be base Bernini with CIO=None and Action-LoRA=None"
            )
    elif action is None or previous is None:
        raise PairV5EndpointV3Error(
            "round>0 policy requires previous Action-LoRA and update receipt"
        )
    unsigned = {
        "schema_version": POLICY_SCHEMA,
        "generation_round": generation_round,
        "checkpoint_tree_sha256": checkpoint,
        "cio_adapter": cio,
        "action_lora": action,
        "previous_update_receipt_digest": previous,
        "round0_base_policy": generation_round == 0,
    }
    return {**unsigned, "policy_digest": object_sha256(unsigned)}


def validate_parent_policy(
    value: Any, *, verify_files: bool = True
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "generation_round",
        "checkpoint_tree_sha256",
        "cio_adapter",
        "action_lora",
        "previous_update_receipt_digest",
        "round0_base_policy",
        "policy_digest",
    }
    row = dict(_closed(value, fields, label="parent policy"))
    digest = _verify_embedded(row, field="policy_digest", label="parent policy")
    generation_round = row["generation_round"]
    if type(generation_round) is not int or generation_round < 0:
        raise PairV5EndpointV3Error("parent generation round differs")
    _sha256(row["checkpoint_tree_sha256"], label="checkpoint tree SHA-256")
    for name in ("cio_adapter", "action_lora"):
        if row[name] is not None:
            row[name] = validate_file_binding(
                row[name], label=f"parent {name}", verify_bytes=verify_files
            )
    if generation_round == 0:
        if (
            row["cio_adapter"] is not None
            or row["action_lora"] is not None
            or row["previous_update_receipt_digest"] is not None
            or row["round0_base_policy"] is not True
        ):
            raise PairV5EndpointV3Error("round0 parent policy differs")
    else:
        if (
            row["action_lora"] is None
            or row["previous_update_receipt_digest"] is None
            or row["round0_base_policy"] is not False
        ):
            raise PairV5EndpointV3Error("round>0 parent policy differs")
        _sha256(
            row["previous_update_receipt_digest"],
            label="previous update receipt digest",
        )
    row["policy_digest"] = digest
    return row


def make_policy_attestation(
    *,
    parent_policy: Mapping[str, Any],
    legacy_rollout_receipt_digest: str,
    candidate_id: str,
    candidate_mp4_sha256: str,
    clean_latent_sha256: str,
    rollout_seed: int,
) -> dict[str, Any]:
    policy = validate_parent_policy(parent_policy, verify_files=True)
    if policy["generation_round"] == 0:
        raise PairV5EndpointV3Error("round0 does not need a policy attestation")
    if type(rollout_seed) is not int or not 0 <= rollout_seed < 2**63:
        raise PairV5EndpointV3Error("rollout seed differs")
    unsigned = {
        "schema_version": POLICY_ATTESTATION_SCHEMA,
        "generation_round": policy["generation_round"],
        "parent_policy_digest": policy["policy_digest"],
        "legacy_rollout_receipt_digest": _sha256(
            legacy_rollout_receipt_digest, label="legacy rollout receipt digest"
        ),
        "candidate_id": _safe_id(candidate_id, label="candidate ID"),
        "candidate_mp4_sha256": _sha256(
            candidate_mp4_sha256, label="candidate MP4 SHA-256"
        ),
        "clean_latent_sha256": _sha256(
            clean_latent_sha256, label="clean latent SHA-256"
        ),
        "rollout_seed": rollout_seed,
        "adapter_loaded_before_sampling": True,
        "attested_at_rollout_time": True,
    }
    return {**unsigned, "attestation_digest": object_sha256(unsigned)}


def validate_policy_attestation(
    value: Any,
    *,
    policy: Mapping[str, Any],
    legacy_digest: str,
    candidate_id: str,
    mp4_sha256: str,
    clean_sha256: str,
    seed: int,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "generation_round",
        "parent_policy_digest",
        "legacy_rollout_receipt_digest",
        "candidate_id",
        "candidate_mp4_sha256",
        "clean_latent_sha256",
        "rollout_seed",
        "adapter_loaded_before_sampling",
        "attested_at_rollout_time",
        "attestation_digest",
    }
    row = dict(_closed(value, fields, label="rollout policy attestation"))
    digest = _verify_embedded(
        row, field="attestation_digest", label="rollout policy attestation"
    )
    checked_policy = validate_parent_policy(policy, verify_files=True)
    if (
        row["schema_version"] != POLICY_ATTESTATION_SCHEMA
        or row["generation_round"] != checked_policy["generation_round"]
        or row["parent_policy_digest"] != checked_policy["policy_digest"]
        or row["legacy_rollout_receipt_digest"] != legacy_digest
        or row["candidate_id"] != candidate_id
        or row["candidate_mp4_sha256"] != mp4_sha256
        or row["clean_latent_sha256"] != clean_sha256
        or row["rollout_seed"] != seed
        or row["adapter_loaded_before_sampling"] is not True
        or row["attested_at_rollout_time"] is not True
    ):
        raise PairV5EndpointV3Error("rollout policy attestation binding differs")
    row["attestation_digest"] = digest
    return row


def _validate_legacy_rollout(
    value: Any, *, native_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5EndpointV3Error("legacy rollout receipt must be an object")
    row = dict(value)
    digest = _verify_embedded(
        row,
        field="receipt_digest",
        label="legacy rollout receipt",
        ensure_ascii=False,
    )
    sampling = row.get("sampling_contract")
    closure = row.get("semantic_input_closure")
    candidate = row.get("candidate")
    artifacts = row.get("artifacts")
    if (
        row.get("schema_version") != LEGACY_ROLLOUT_SCHEMA
        or not isinstance(sampling, Mapping)
        or sampling.get("condition_mode") != "rv2v4"
        or sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("latent_frames") != LATENT_PHASES
        or sampling.get("fps") != 25
        or sampling.get("num_inference_steps") != 40
        or sampling.get("source_reference_indices") != list(REFERENCE_INDICES)
        or sampling.get("target_initialization")
        != "official_gen_wanx22_fresh_gaussian"
        or not isinstance(closure, Mapping)
        or closure.get("accepted") != ["source_video", "complete_caption"]
        or any(
            closure.get(name) is not False
            for name in (
                "target_video",
                "t2v_proposal_media",
                "donor_video",
                "external_reference",
                "mask",
                "flow",
                "pose",
                "track",
                "trajectory",
            )
        )
        or not isinstance(candidate, Mapping)
        or not isinstance(artifacts, Mapping)
    ):
        raise PairV5EndpointV3Error("legacy rollout exact81/input closure differs")
    if (
        native_receipt.get("schema_version") != NATIVE_RECEIPT_SCHEMA
        or native_receipt.get("receipt_digest") != row.get("native_receipt_digest")
    ):
        raise PairV5EndpointV3Error("native generation receipt binding differs")
    row["receipt_digest"] = digest
    return row


def _validate_native_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5EndpointV3Error("native generation receipt must be an object")
    row = dict(value)
    digest = _verify_embedded(
        row,
        field="receipt_digest",
        label="native generation receipt",
        ensure_ascii=True,
    )
    if row.get("schema_version") != NATIVE_RECEIPT_SCHEMA:
        raise PairV5EndpointV3Error("native generation receipt schema differs")
    row["receipt_digest"] = digest
    return row


def make_rollout_evidence(
    *,
    parent_policy: Mapping[str, Any],
    action_receipt_file: Mapping[str, Any],
    policy_attestation_file: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Bind one active-v4 score to its actual native rollout and policy."""

    policy = validate_parent_policy(parent_policy, verify_files=True)
    action_raw, action_binding = load_bound_json(
        action_receipt_file, label="active-v4 action receipt"
    )
    action = adapt_active_v4_action_receipt(action_raw)
    rollout_binding = file_binding(action["rollout_receipt_path"])
    rollout_raw, _ = load_bound_json(rollout_binding, label="legacy rollout receipt")
    native_binding = file_binding(rollout_raw["native_receipt_path"])
    native_raw, _ = load_bound_json(native_binding, label="native generation receipt")
    native = _validate_native_receipt(native_raw)
    rollout = _validate_legacy_rollout(rollout_raw, native_receipt=native)
    candidate = rollout["candidate"]
    artifacts = rollout["artifacts"]
    mp4 = artifacts["mp4"]
    clean = artifacts["predecode_clean_latent"]
    mp4_binding = file_binding(mp4["path"])
    clean_binding = file_binding(clean["path"])
    source_binding = file_binding(candidate["source_video"])
    if (
        candidate["candidate_id"] != action["candidate_id"]
        or candidate["source_video_sha256"] != action["source_video_sha256"]
        or candidate["complete_caption_sha256"] != action["prompt_sha256"]
        or candidate["seed"] != action["seed"]
        or rollout["receipt_digest"] != action["rollout_receipt_digest"]
        or action["rollout_receipt_file_sha256"] != rollout_binding["sha256"]
        or mp4_binding["sha256"] != action["candidate_mp4_sha256"]
        or clean_binding["sha256"] != action["clean_latent_sha256"]
        or source_binding["sha256"] != action["source_video_sha256"]
        or native["checkpoint"]["tree_sha256"]
        != policy["checkpoint_tree_sha256"]
    ):
        raise PairV5EndpointV3Error("action/rollout/policy/media binding differs")
    if policy["generation_round"] == 0:
        freeze = native.get("freeze_certificate", {})
        if (
            freeze.get("base_frozen") is not True
            or freeze.get("lora_module_count") != 0
            or freeze.get("trainable_parameter_tensors") != 0
        ):
            raise PairV5EndpointV3Error("round0 rollout was not frozen base policy")
        attestation = None
        if policy_attestation_file is not None:
            raise PairV5EndpointV3Error("round0 must not supply policy attestation")
    else:
        if policy_attestation_file is None:
            raise PairV5EndpointV3Error("round>0 requires rollout policy attestation")
        attestation_raw, attestation_binding = load_bound_json(
            policy_attestation_file, label="rollout policy attestation"
        )
        attestation = {
            "file": attestation_binding,
            "receipt": validate_policy_attestation(
                attestation_raw,
                policy=policy,
                legacy_digest=rollout["receipt_digest"],
                candidate_id=action["candidate_id"],
                mp4_sha256=mp4_binding["sha256"],
                clean_sha256=clean_binding["sha256"],
                seed=action["seed"],
            ),
        }
    nonce = object_sha256(
        {
            "candidate_id": action["candidate_id"],
            "seed": action["seed"],
            "mp4_sha256": mp4_binding["sha256"],
            "clean_latent_sha256": clean_binding["sha256"],
            "parent_policy_digest": policy["policy_digest"],
        }
    )
    unsigned = {
        "schema_version": ROLLOUT_EVIDENCE_SCHEMA,
        "generation_round": policy["generation_round"],
        "parent_policy": policy,
        "parent_policy_digest": policy["policy_digest"],
        "candidate_id": action["candidate_id"],
        "analysis_split": action["analysis_split"],
        "action_family_id": action["action_family_id"],
        "source_video": source_binding,
        "complete_caption": candidate["complete_caption"],
        "complete_caption_sha256": action["prompt_sha256"],
        "seed": action["seed"],
        "candidate_envelope_sha256": action["candidate_envelope_sha256"],
        "action_receipt": action_binding,
        "action_receipt_digest": action["receipt_digest"],
        "legacy_rollout_receipt": rollout_binding,
        "legacy_rollout_receipt_digest": rollout["receipt_digest"],
        "native_generation_receipt": native_binding,
        "native_generation_receipt_digest": native["receipt_digest"],
        "candidate_mp4": mp4_binding,
        "clean_latent": clean_binding,
        "clean_latent_tensor_key": clean.get("tensor_key"),
        "clean_latent_shape": clean.get("shape"),
        "policy_attestation": attestation,
        "rollout_nonce_digest": nonce,
        "fresh_rollout_required_for_next_round": True,
    }
    return {**unsigned, "evidence_digest": object_sha256(unsigned)}


def validate_rollout_evidence(
    value: Any, *, replay_files: bool = True
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "generation_round",
        "parent_policy",
        "parent_policy_digest",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "source_video",
        "complete_caption",
        "complete_caption_sha256",
        "seed",
        "candidate_envelope_sha256",
        "action_receipt",
        "action_receipt_digest",
        "legacy_rollout_receipt",
        "legacy_rollout_receipt_digest",
        "native_generation_receipt",
        "native_generation_receipt_digest",
        "candidate_mp4",
        "clean_latent",
        "clean_latent_tensor_key",
        "clean_latent_shape",
        "policy_attestation",
        "rollout_nonce_digest",
        "fresh_rollout_required_for_next_round",
        "evidence_digest",
    }
    row = dict(_closed(value, fields, label="round rollout evidence"))
    digest = _verify_embedded(
        row, field="evidence_digest", label="round rollout evidence"
    )
    policy = validate_parent_policy(row["parent_policy"], verify_files=replay_files)
    if (
        row["schema_version"] != ROLLOUT_EVIDENCE_SCHEMA
        or row["generation_round"] != policy["generation_round"]
        or row["parent_policy_digest"] != policy["policy_digest"]
        or row["fresh_rollout_required_for_next_round"] is not True
    ):
        raise PairV5EndpointV3Error("round rollout authority differs")
    _safe_id(row["candidate_id"], label="candidate ID")
    _safe_id(row["action_family_id"], label="action family")
    if row["analysis_split"] not in ("fit", "confirmation"):
        raise PairV5EndpointV3Error("analysis split differs")
    if (
        not isinstance(row["complete_caption"], str)
        or hashlib.sha256(row["complete_caption"].encode("utf-8")).hexdigest()
        != row["complete_caption_sha256"]
        or type(row["seed"]) is not int
        or not 0 <= row["seed"] < 2**63
    ):
        raise PairV5EndpointV3Error("caption/seed binding differs")
    for name in (
        "source_video",
        "action_receipt",
        "legacy_rollout_receipt",
        "native_generation_receipt",
        "candidate_mp4",
        "clean_latent",
    ):
        row[name] = validate_file_binding(
            row[name], label=name, verify_bytes=replay_files
        )
    for name in (
        "complete_caption_sha256",
        "candidate_envelope_sha256",
        "action_receipt_digest",
        "legacy_rollout_receipt_digest",
        "native_generation_receipt_digest",
        "rollout_nonce_digest",
    ):
        _sha256(row[name], label=name)
    shape = row["clean_latent_shape"]
    if (
        row["clean_latent_tensor_key"] != "normalized_clean_latent"
        or not isinstance(shape, list)
        or len(shape) != 5
        or any(type(item) is not int for item in shape)
        or tuple(shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
        or shape[3] <= 0
        or shape[4] <= 0
    ):
        raise PairV5EndpointV3Error("clean latent exact81 geometry differs")
    if replay_files:
        action_raw, _ = load_bound_json(row["action_receipt"], label="action receipt")
        action = adapt_active_v4_action_receipt(action_raw)
        rollout_raw, _ = load_bound_json(
            row["legacy_rollout_receipt"], label="legacy rollout receipt"
        )
        native_raw, _ = load_bound_json(
            row["native_generation_receipt"], label="native generation receipt"
        )
        native = _validate_native_receipt(native_raw)
        rollout = _validate_legacy_rollout(rollout_raw, native_receipt=native)
        candidate = rollout["candidate"]
        artifacts = rollout["artifacts"]
        if (
            action["receipt_digest"] != row["action_receipt_digest"]
            or rollout["receipt_digest"] != row["legacy_rollout_receipt_digest"]
            or native["receipt_digest"] != row["native_generation_receipt_digest"]
            or action["candidate_id"] != row["candidate_id"]
            or action["analysis_split"] != row["analysis_split"]
            or action["action_family_id"] != row["action_family_id"]
            or action["source_video_sha256"] != row["source_video"]["sha256"]
            or action["prompt_sha256"] != row["complete_caption_sha256"]
            or action["candidate_mp4_sha256"] != row["candidate_mp4"]["sha256"]
            or action["clean_latent_sha256"] != row["clean_latent"]["sha256"]
            or candidate.get("complete_caption") != row["complete_caption"]
            or artifacts.get("mp4", {}).get("sha256")
            != row["candidate_mp4"]["sha256"]
            or artifacts.get("predecode_clean_latent", {}).get("sha256")
            != row["clean_latent"]["sha256"]
            or native.get("checkpoint", {}).get("tree_sha256")
            != policy["checkpoint_tree_sha256"]
        ):
            raise PairV5EndpointV3Error("round rollout replay differs")
        if policy["generation_round"] == 0:
            freeze = native.get("freeze_certificate", {})
            if row["policy_attestation"] is not None or (
                freeze.get("lora_module_count") != 0
                or freeze.get("trainable_parameter_tensors") != 0
            ):
                raise PairV5EndpointV3Error("round0 rollout policy replay differs")
        else:
            attestation = row["policy_attestation"]
            if not isinstance(attestation, Mapping):
                raise PairV5EndpointV3Error("round>0 policy attestation is absent")
            attestation_raw, _ = load_bound_json(
                attestation.get("file"), label="rollout policy attestation"
            )
            checked = validate_policy_attestation(
                attestation_raw,
                policy=policy,
                legacy_digest=rollout["receipt_digest"],
                candidate_id=row["candidate_id"],
                mp4_sha256=row["candidate_mp4"]["sha256"],
                clean_sha256=row["clean_latent"]["sha256"],
                seed=row["seed"],
            )
            if checked != attestation.get("receipt"):
                raise PairV5EndpointV3Error("policy attestation replay differs")
    row["parent_policy"] = policy
    row["evidence_digest"] = digest
    return row


def adapt_active_v4_action_receipt(value: Any) -> dict[str, Any]:
    """Project the active-v4 full receipt through a narrow, replayed protocol."""

    if not isinstance(value, Mapping):
        raise PairV5EndpointV3Error("active-v4 action receipt must be an object")
    row = dict(value)
    digest = _verify_embedded(row, field="receipt_digest", label="active-v4 action receipt")
    candidate = row.get("candidate")
    source = row.get("source")
    rollout = row.get("rollout")
    calibration = row.get("calibration")
    prompts = row.get("prompts")
    artifacts = row.get("artifacts")
    mace = row.get("mace")
    if (
        row.get("schema_version") != ACTIVE_ACTION_SCHEMA
        or not all(
            isinstance(item, Mapping)
            for item in (candidate, source, rollout, calibration, prompts, artifacts, mace)
        )
        or row.get("optimizer_authorized") is not False
        or row.get("scientific_action_editing_claim") is not False
    ):
        raise PairV5EndpointV3Error("active-v4 action authority differs")
    candidate_id = _safe_id(candidate.get("candidate_id"), label="candidate ID")
    split = candidate.get("analysis_split")
    if split not in ("fit", "confirmation"):
        raise PairV5EndpointV3Error("active-v4 analysis split differs")
    family = _safe_id(candidate.get("action_family_id"), label="action family")
    source_sha = _sha256(candidate.get("source_video_sha256"), label="source SHA-256")
    prompt_sha = _sha256(
        candidate.get("complete_caption_utf8_sha256"), label="prompt SHA-256"
    )
    mapping = calibration.get("family_mapping")
    if not isinstance(mapping, Mapping):
        raise PairV5EndpointV3Error("active-v4 family map is absent")
    lower = _finite(mapping.get("lower_raw_anchor"), label="lower action anchor")
    upper = _finite(mapping.get("upper_raw_anchor"), label="upper action anchor")
    raw = _finite(mace.get("raw_global_action_energy_score"), label="raw action score")
    if (
        mapping.get("kind") != "clipped_affine_fit_only"
        or mapping.get("anchor_source_split") != "fit"
        or mapping.get("clip_min") != 0.0
        or mapping.get("clip_max") != 1.0
        or not upper > lower
    ):
        raise PairV5EndpointV3Error("active-v4 family map differs")
    expected = float(min(1.0, max(0.0, (raw - lower) / (upper - lower))))
    calibrated = _finite(
        mace.get("calibrated_family_action_score"),
        label="calibrated action score",
        unit=True,
    )
    internal_threshold = _finite(
        mace.get("decision_threshold"), label="internal action threshold", unit=True
    )
    if (
        calibrated != expected
        or mace.get("passes_calibrated_action_metric")
        is not (calibrated >= internal_threshold)
        or candidate.get("source_video_sha256")
        != source.get("source_video_sha256_recomputed")
        or source.get("source_video_sha256_declared") != source_sha
        or source.get("calibration_geometry_source_sha256") != source_sha
        or rollout.get("generated_mp4_consumed_by_scorer") is not False
        or rollout.get("native_condition_mode") != "rv2v4"
        or calibration.get("action_family_id") != family
        or prompts.get("full_t2v_caption_utf8_sha256_by_branch", {}).get("action")
        != prompt_sha
        or artifacts.get("clean_and_gaussian_are_same_candidate_artifacts") is not True
    ):
        raise PairV5EndpointV3Error("active-v4 score formula/binding differs")
    seed = candidate.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise PairV5EndpointV3Error("active-v4 seed differs")
    result = {
        "candidate_id": candidate_id,
        "analysis_split": split,
        "action_family_id": family,
        "source_video_path": source.get("source_video_path"),
        "source_video_sha256": source_sha,
        "complete_caption": prompts.get("full_t2v_caption_by_branch", {}).get("action"),
        "prompt_sha256": prompt_sha,
        "seed": seed,
        "action_score": calibrated,
        "internal_action_threshold_diagnostic": internal_threshold,
        "receipt_digest": digest,
        "rollout_receipt_path": rollout.get("pair_receipt_path"),
        "rollout_receipt_file_sha256": _sha256(
            rollout.get("pair_receipt_file_sha256"),
            label="rollout receipt file SHA-256",
        ),
        "rollout_receipt_digest": _sha256(
            rollout.get("pair_receipt_digest"), label="rollout receipt digest"
        ),
        "native_receipt_digest": _sha256(
            rollout.get("native_receipt_digest"), label="native receipt digest"
        ),
        "candidate_envelope_sha256": _sha256(
            rollout.get("candidate_envelope_sha256"), label="candidate envelope SHA-256"
        ),
        "candidate_mp4_sha256": _sha256(
            rollout.get("generated_mp4_sha256"), label="candidate MP4 SHA-256"
        ),
        "clean_latent_sha256": _sha256(
            artifacts.get("clean_latent_artifact_sha256"),
            label="clean latent artifact SHA-256",
        ),
        "checkpoint_tree_sha256": _sha256(
            rollout.get("checkpoint_tree_sha256"), label="checkpoint tree SHA-256"
        ),
    }
    if (
        not isinstance(result["source_video_path"], str)
        or not isinstance(result["complete_caption"], str)
        or hashlib.sha256(result["complete_caption"].encode("utf-8")).hexdigest()
        != prompt_sha
    ):
        raise PairV5EndpointV3Error("active-v4 prompt/source projection differs")
    return result


def adapt_decoded_preservation_receipt(value: Any) -> dict[str, Any]:
    """Read only stable primary metrics; wrong-source values are diagnostics."""

    if not isinstance(value, Mapping):
        raise PairV5EndpointV3Error("decoded preservation receipt must be an object")
    row = dict(value)
    digest = _verify_embedded(
        row, field="receipt_digest", label="decoded preservation receipt"
    )
    metrics = row.get("metrics")
    if (
        row.get("schema_version") != DECODED_PRESERVATION_SCHEMA
        or not isinstance(metrics, Mapping)
        or row.get("evidence_valid") is not True
        or row.get("absolute_source_preservation_pass_claim") is not False
    ):
        raise PairV5EndpointV3Error("decoded preservation evidence authority differs")
    primary = {
        name: _finite(metrics.get(key), label=f"preservation {name}", unit=True)
        for name, key in PRIMARY_PRESERVATION_METRICS.items()
    }
    diagnostics = {
        key: _finite(metrics[key], label=f"wrong-source diagnostic {key}")
        for key in WRONG_SOURCE_DIAGNOSTIC_KEYS
        if key in metrics
    }
    return {
        "candidate_id": _safe_id(row.get("candidate_id"), label="candidate ID"),
        "source_video_sha256": _sha256(
            row.get("correct_source_video_sha256"), label="correct source SHA-256"
        ),
        "wrong_source_video_sha256": _sha256(
            row.get("wrong_source_video_sha256"), label="wrong source SHA-256"
        ),
        "candidate_mp4_sha256": _sha256(
            row.get("candidate_mp4_sha256"), label="candidate MP4 SHA-256"
        ),
        "clean_latent_sha256": _sha256(
            row.get("predecode_clean_latent_sha256"), label="clean latent SHA-256"
        ),
        "candidate_envelope_sha256": _sha256(
            row.get("candidate_envelope_sha256"), label="candidate envelope SHA-256"
        ),
        "rollout_receipt_digest": _sha256(
            row.get("rollout_receipt_digest"), label="rollout receipt digest"
        ),
        "native_receipt_digest": _sha256(
            row.get("native_rollout_receipt_digest"), label="native receipt digest"
        ),
        "primary_scores": primary,
        "wrong_source_diagnostics": diagnostics,
        "wrong_source_authorizes_optimizer": False,
        "receipt_digest": digest,
    }


def adapt_temporal_counterfactual_receipt(
    value: Any, *, expected_evaluator: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay the v2 master before deriving the same-video contrast."""

    try:
        row = decoded_event.validate_temporal_projection(value)
    except decoded_event.DecodedTemporalEventError as error:
        raise PairV5EndpointV3Error(
            f"decoded temporal-counterfactual master replay failed: {error}"
        ) from error
    evaluator = validate_file_binding(
        row["evaluator_implementation"],
        label="temporal-counterfactual evaluator",
        verify_bytes=True,
    )
    expected = validate_file_binding(
        expected_evaluator,
        label="pinned temporal-counterfactual evaluator",
        verify_bytes=True,
    )
    energies = _closed(
        row["branch_energy_by_name"],
        {"target", "reverse", "shuffle", "freeze"},
        label="temporal-counterfactual branch energies",
    )
    checked = {
        name: _finite(value, label=f"{name} branch energy")
        for name, value in energies.items()
    }
    if any(value < 0.0 for value in checked.values()):
        raise PairV5EndpointV3Error("temporal-counterfactual energy is negative")
    target = checked["target"]
    minimum_negative = min(checked[name] for name in ("reverse", "shuffle", "freeze"))
    denominator = minimum_negative + target + 1.0e-12
    margin = max(0.0, min(1.0, (minimum_negative - target) / denominator))
    reverse_denominator = checked["reverse"] + target + 1.0e-12
    order = max(
        0.0,
        min(1.0, (checked["reverse"] - target) / reverse_denominator),
    )
    if (
        row["schema_version"] != TEMPORAL_COUNTERFACTUAL_SCHEMA
        or row["analysis_split"] != "fit"
        or row["frame_count"] != FRAME_COUNT
        or row["evidence_valid"] is not True
        or evaluator != expected
    ):
        raise PairV5EndpointV3Error(
            "temporal-counterfactual evaluator/split/evidence differs"
        )
    return {
        "candidate_id": _safe_id(row["candidate_id"], label="candidate ID"),
        "analysis_split": "fit",
        "action_family_id": _safe_id(
            row["action_family_id"], label="temporal action family ID"
        ),
        "source_video_sha256": _sha256(
            row["source_video_sha256"], label="source video SHA-256"
        ),
        "complete_caption_sha256": _sha256(
            row["complete_caption_sha256"], label="temporal caption SHA-256"
        ),
        "candidate_mp4_sha256": _sha256(
            row["candidate_mp4_sha256"], label="candidate MP4 SHA-256"
        ),
        "rollout_receipt_digest": _sha256(
            row["rollout_receipt_digest"], label="rollout receipt digest"
        ),
        "evaluator_implementation": evaluator,
        "probe_bank_digest": _sha256(
            row["probe_bank_digest"], label="probe bank digest"
        ),
        "master_receipt_file": validate_file_binding(
            row["master_receipt_file"],
            label="temporal master receipt",
            verify_bytes=True,
        ),
        "master_receipt_digest": _sha256(
            row["master_receipt_digest"], label="temporal master receipt digest"
        ),
        "branch_energy_by_name": checked,
        "counterfactual_margin_score": margin,
        "temporal_order_score": order,
        "receipt_digest": _sha256(
            row["receipt_digest"], label="temporal projection receipt digest"
        ),
    }


def adapt_event81_receipt(
    value: Any, *, expected_evaluator: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay the v2 master before deriving exact81 ordered-event gates."""

    try:
        row = decoded_event.validate_event81_projection(value)
    except decoded_event.DecodedTemporalEventError as error:
        raise PairV5EndpointV3Error(
            f"decoded exact81 event master replay failed: {error}"
        ) from error
    evaluator = validate_file_binding(
        row["evaluator_implementation"], label="event81 evaluator", verify_bytes=True
    )
    expected = validate_file_binding(
        expected_evaluator, label="pinned event81 evaluator", verify_bytes=True
    )
    arrays: dict[str, list[float]] = {}
    for output_name, field in (
        ("start", "start_probability_by_frame"),
        ("transition", "transition_probability_by_frame"),
        ("terminal", "terminal_probability_by_frame"),
        ("terminal_hold", "terminal_hold_probability_by_frame"),
    ):
        values = row[field]
        if not isinstance(values, list) or len(values) != FRAME_COUNT:
            raise PairV5EndpointV3Error(f"full81 {output_name} evidence differs")
        arrays[output_name] = [
            _finite(value, label=f"{output_name} probability", unit=True)
            for value in values
        ]
    transition_window = arrays["transition"][16:61]
    top_transition = sorted(transition_window, reverse=True)[:5]
    scores = {
        "event_start": sum(arrays["start"][:16]) / 16.0,
        "event_transition": sum(top_transition) / float(len(top_transition)),
        "event_terminal": sum(arrays["terminal"][61:]) / 20.0,
        "event_terminal_hold": min(arrays["terminal_hold"][73:]),
    }
    peaks = {
        name: max(range(FRAME_COUNT), key=lambda index: arrays[name][index])
        for name in ("start", "transition", "terminal", "terminal_hold")
    }
    ordering = bool(
        0 <= peaks["start"] <= 15
        and 16 <= peaks["transition"] <= 60
        and 61 <= peaks["terminal"] <= 80
        and 73 <= peaks["terminal_hold"] <= 80
        and peaks["start"] < peaks["transition"] < peaks["terminal"]
        and peaks["transition"] < peaks["terminal_hold"]
    )
    if (
        row["schema_version"] != EVENT81_SCHEMA
        or row["analysis_split"] != "fit"
        or row["frame_count"] != FRAME_COUNT
        or row["frame_indices"] != list(range(FRAME_COUNT))
        or row["evidence_valid"] is not True
        or evaluator != expected
        or not ordering
    ):
        raise PairV5EndpointV3Error("full81 event evaluator/order/split differs")
    return {
        "candidate_id": _safe_id(row["candidate_id"], label="candidate ID"),
        "analysis_split": "fit",
        "action_family_id": _safe_id(
            row["action_family_id"], label="event81 action family ID"
        ),
        "source_video_sha256": _sha256(
            row["source_video_sha256"], label="source video SHA-256"
        ),
        "complete_caption_sha256": _sha256(
            row["complete_caption_sha256"], label="event81 caption SHA-256"
        ),
        "candidate_mp4_sha256": _sha256(
            row["candidate_mp4_sha256"], label="candidate MP4 SHA-256"
        ),
        "rollout_receipt_digest": _sha256(
            row["rollout_receipt_digest"], label="rollout receipt digest"
        ),
        "evaluator_implementation": evaluator,
        "master_receipt_file": validate_file_binding(
            row["master_receipt_file"],
            label="event81 master receipt",
            verify_bytes=True,
        ),
        "master_receipt_digest": _sha256(
            row["master_receipt_digest"], label="event81 master receipt digest"
        ),
        "event_scores": scores,
        "event_peak_frame_by_state": peaks,
        "start_transition_terminal_ordering_pass": ordering,
        "receipt_digest": _sha256(
            row["receipt_digest"], label="event81 projection receipt digest"
        ),
    }


def make_gate_policy(
    *,
    temporal_counterfactual_evaluator: Mapping[str, Any],
    event81_evaluator: Mapping[str, Any],
    identity_min: float,
    background_min: float,
    camera_min: float,
    temporal_min: float,
    quality_min: float,
    action_min: float,
    action_margin: float,
    counterfactual_margin_min: float,
    temporal_order_min: float,
    event_start_min: float,
    event_transition_min: float,
    event_terminal_min: float,
    event_terminal_hold_min: float,
    minimum_distinct_sources: int = 2,
) -> dict[str, Any]:
    temporal_evaluator = validate_file_binding(
        temporal_counterfactual_evaluator,
        label="temporal-counterfactual evaluator",
        verify_bytes=True,
    )
    event_evaluator = validate_file_binding(
        event81_evaluator, label="event81 evaluator", verify_bytes=True
    )
    registered_evaluator = decoded_temporal_event_evaluator_binding()
    if (
        temporal_evaluator != event_evaluator
        or temporal_evaluator != registered_evaluator
    ):
        raise PairV5EndpointV3Error(
            "both endpoint projections must use the registered decoded master evaluator"
        )
    thresholds = {
        "identity": _finite(identity_min, label="identity minimum", unit=True),
        "background": _finite(background_min, label="background minimum", unit=True),
        "camera": _finite(camera_min, label="camera minimum", unit=True),
        "temporal": _finite(temporal_min, label="temporal minimum", unit=True),
        "quality": _finite(quality_min, label="quality minimum", unit=True),
        "action": _finite(action_min, label="action minimum", unit=True),
        "action_margin": _finite(
            action_margin, label="action margin", unit=True
        ),
        "counterfactual_margin": _finite(
            counterfactual_margin_min,
            label="temporal-counterfactual margin minimum",
            unit=True,
        ),
        "temporal_order": _finite(
            temporal_order_min, label="temporal order minimum", unit=True
        ),
        "event_start": _finite(
            event_start_min, label="event start minimum", unit=True
        ),
        "event_transition": _finite(
            event_transition_min, label="event transition minimum", unit=True
        ),
        "event_terminal": _finite(
            event_terminal_min, label="event terminal minimum", unit=True
        ),
        "event_terminal_hold": _finite(
            event_terminal_hold_min,
            label="event terminal-hold minimum",
            unit=True,
        ),
    }
    if thresholds["action_margin"] <= 0.0:
        raise PairV5EndpointV3Error("action margin must be positive")
    if type(minimum_distinct_sources) is not int or minimum_distinct_sources < 2:
        raise PairV5EndpointV3Error("at least two distinct sources are required")
    unsigned = {
        "schema_version": GATE_POLICY_SCHEMA,
        "evaluator_implementations": {
            "same_video_temporal_counterfactual": temporal_evaluator,
            "start_transition_terminal_hold": event_evaluator,
        },
        "thresholds": thresholds,
        "minimum_distinct_sources": minimum_distinct_sources,
        "analysis_split": "fit",
        "both_endpoints_must_pass_all_preservation_gates": True,
        "winner_must_pass_absolute_action_gate": True,
        "winner_loser_action_margin_required": True,
        "wrong_source_diagnostic_only": True,
        "confirmation_optimizer_use": False,
    }
    return {**unsigned, "gate_policy_digest": object_sha256(unsigned)}


def validate_gate_policy(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "evaluator_implementations",
        "thresholds",
        "minimum_distinct_sources",
        "analysis_split",
        "both_endpoints_must_pass_all_preservation_gates",
        "winner_must_pass_absolute_action_gate",
        "winner_loser_action_margin_required",
        "wrong_source_diagnostic_only",
        "confirmation_optimizer_use",
        "gate_policy_digest",
    }
    row = dict(_closed(value, fields, label="endpoint gate policy"))
    digest = _verify_embedded(
        row, field="gate_policy_digest", label="endpoint gate policy"
    )
    thresholds = _closed(
        row["thresholds"],
        {
            "identity", "background", "camera", "temporal", "quality",
            "action", "action_margin", "counterfactual_margin",
            "temporal_order", "event_start", "event_transition",
            "event_terminal", "event_terminal_hold",
        },
        label="endpoint thresholds",
    )
    row["thresholds"] = {
        key: _finite(value, label=f"threshold {key}", unit=True)
        for key, value in thresholds.items()
    }
    evaluators = _closed(
        row["evaluator_implementations"],
        {"same_video_temporal_counterfactual", "start_transition_terminal_hold"},
        label="endpoint evaluator implementations",
    )
    row["evaluator_implementations"] = {
        name: validate_file_binding(
            binding, label=f"pinned {name} evaluator", verify_bytes=True
        )
        for name, binding in evaluators.items()
    }
    registered_evaluator = decoded_temporal_event_evaluator_binding()
    if (
        row["schema_version"] != GATE_POLICY_SCHEMA
        or type(row["minimum_distinct_sources"]) is not int
        or row["minimum_distinct_sources"] < 2
        or row["analysis_split"] != "fit"
        or row["both_endpoints_must_pass_all_preservation_gates"] is not True
        or row["winner_must_pass_absolute_action_gate"] is not True
        or row["winner_loser_action_margin_required"] is not True
        or row["wrong_source_diagnostic_only"] is not True
        or row["confirmation_optimizer_use"] is not False
        or row["thresholds"]["action_margin"] <= 0.0
        or row["evaluator_implementations"]["same_video_temporal_counterfactual"]
        != registered_evaluator
        or row["evaluator_implementations"]["start_transition_terminal_hold"]
        != registered_evaluator
    ):
        raise PairV5EndpointV3Error("endpoint gate authority differs")
    row["gate_policy_digest"] = digest
    return row


def make_build_request(
    *,
    gate_policy: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = validate_gate_policy(gate_policy)
    rows = []
    for candidate in candidates:
        row = _closed(
            candidate,
            {
                "rollout_evidence",
                "action_receipt",
                "preservation_receipt",
                "temporal_counterfactual_receipt",
                "event81_receipt",
            },
            label="candidate build input",
        )
        rows.append(
            {
                key: validate_file_binding(
                    row[key], label=f"candidate {key}", verify_bytes=True
                )
                for key in (
                    "rollout_evidence",
                    "action_receipt",
                    "preservation_receipt",
                    "temporal_counterfactual_receipt",
                    "event81_receipt",
                )
            }
        )
    if not rows:
        raise PairV5EndpointV3Error("candidate build request is empty")
    unsigned = {
        "schema_version": BUILD_REQUEST_SCHEMA,
        "gate_policy": policy,
        "candidate_inputs": rows,
        "caller_scores_or_flags_accepted": False,
    }
    return {**unsigned, "request_digest": object_sha256(unsigned)}


def validate_build_request(value: Any) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            {
                "schema_version",
                "gate_policy",
                "candidate_inputs",
                "caller_scores_or_flags_accepted",
                "request_digest",
            },
            label="endpoint build request",
        )
    )
    digest = _verify_embedded(row, field="request_digest", label="build request")
    if (
        row["schema_version"] != BUILD_REQUEST_SCHEMA
        or row["caller_scores_or_flags_accepted"] is not False
        or not isinstance(row["candidate_inputs"], list)
        or not row["candidate_inputs"]
    ):
        raise PairV5EndpointV3Error("build request authority differs")
    row["gate_policy"] = validate_gate_policy(row["gate_policy"])
    checked = []
    for item in row["candidate_inputs"]:
        entry = _closed(
            item,
            {
                "rollout_evidence",
                "action_receipt",
                "preservation_receipt",
                "temporal_counterfactual_receipt",
                "event81_receipt",
            },
            label="candidate build input",
        )
        checked.append(
            {
                key: validate_file_binding(
                    entry[key], label=f"candidate {key}", verify_bytes=True
                )
                for key in entry
            }
        )
    row["candidate_inputs"] = checked
    row["request_digest"] = digest
    return row


def _derive_candidate(
    item: Mapping[str, Any], gate_policy: Mapping[str, Any]
) -> dict[str, Any]:
    rollout_raw, rollout_file = load_bound_json(
        item["rollout_evidence"], label="round rollout evidence"
    )
    rollout = validate_rollout_evidence(rollout_raw, replay_files=True)
    action_raw, action_file = load_bound_json(
        item["action_receipt"], label="active-v4 action receipt"
    )
    preservation_raw, preservation_file = load_bound_json(
        item["preservation_receipt"], label="decoded preservation receipt"
    )
    temporal_raw, temporal_file = load_bound_json(
        item["temporal_counterfactual_receipt"],
        label="same-video temporal-counterfactual receipt",
    )
    event_raw, event_file = load_bound_json(
        item["event81_receipt"], label="full81 event receipt"
    )
    action = adapt_active_v4_action_receipt(action_raw)
    preservation = adapt_decoded_preservation_receipt(preservation_raw)
    temporal = adapt_temporal_counterfactual_receipt(
        temporal_raw,
        expected_evaluator=gate_policy["evaluator_implementations"][
            "same_video_temporal_counterfactual"
        ],
    )
    event = adapt_event81_receipt(
        event_raw,
        expected_evaluator=gate_policy["evaluator_implementations"][
            "start_transition_terminal_hold"
        ],
    )
    if rollout["analysis_split"] != "fit":
        raise PairV5EndpointV3Error(
            "confirmation evidence is metric-only and cannot enter training"
        )
    if (
        action["analysis_split"] != "fit"
        or action["candidate_id"] != rollout["candidate_id"]
        or preservation["candidate_id"] != rollout["candidate_id"]
        or temporal["candidate_id"] != rollout["candidate_id"]
        or event["candidate_id"] != rollout["candidate_id"]
        or action["receipt_digest"] != rollout["action_receipt_digest"]
        or action_file != rollout["action_receipt"]
        or action["action_family_id"] != rollout["action_family_id"]
        or temporal["action_family_id"] != rollout["action_family_id"]
        or event["action_family_id"] != rollout["action_family_id"]
        or action["source_video_sha256"] != rollout["source_video"]["sha256"]
        or preservation["source_video_sha256"] != rollout["source_video"]["sha256"]
        or temporal["source_video_sha256"] != rollout["source_video"]["sha256"]
        or event["source_video_sha256"] != rollout["source_video"]["sha256"]
        or action["prompt_sha256"] != rollout["complete_caption_sha256"]
        or temporal["complete_caption_sha256"]
        != rollout["complete_caption_sha256"]
        or event["complete_caption_sha256"]
        != rollout["complete_caption_sha256"]
        or action["rollout_receipt_digest"]
        != rollout["legacy_rollout_receipt_digest"]
        or preservation["rollout_receipt_digest"]
        != rollout["legacy_rollout_receipt_digest"]
        or temporal["rollout_receipt_digest"]
        != rollout["legacy_rollout_receipt_digest"]
        or event["rollout_receipt_digest"]
        != rollout["legacy_rollout_receipt_digest"]
        or action["native_receipt_digest"]
        != rollout["native_generation_receipt_digest"]
        or preservation["native_receipt_digest"]
        != rollout["native_generation_receipt_digest"]
        or action["candidate_mp4_sha256"] != rollout["candidate_mp4"]["sha256"]
        or preservation["candidate_mp4_sha256"]
        != rollout["candidate_mp4"]["sha256"]
        or temporal["candidate_mp4_sha256"] != rollout["candidate_mp4"]["sha256"]
        or event["candidate_mp4_sha256"] != rollout["candidate_mp4"]["sha256"]
        or action["clean_latent_sha256"] != rollout["clean_latent"]["sha256"]
        or preservation["clean_latent_sha256"]
        != rollout["clean_latent"]["sha256"]
        or action["candidate_envelope_sha256"]
        != rollout["candidate_envelope_sha256"]
        or preservation["candidate_envelope_sha256"]
        != rollout["candidate_envelope_sha256"]
        or temporal["master_receipt_file"] != event["master_receipt_file"]
        or temporal["master_receipt_digest"] != event["master_receipt_digest"]
    ):
        raise PairV5EndpointV3Error("action/preservation/rollout evidence join differs")
    thresholds = gate_policy["thresholds"]
    preservation_pass = {
        name: score >= thresholds[name]
        for name, score in preservation["primary_scores"].items()
    }
    temporal_scores = {
        "counterfactual_margin": temporal["counterfactual_margin_score"],
        "temporal_order": temporal["temporal_order_score"],
    }
    temporal_pass = {
        name: score >= thresholds[name] for name, score in temporal_scores.items()
    }
    event_pass = {
        name: score >= thresholds[name]
        for name, score in event["event_scores"].items()
    }
    hard_pass = (
        all(preservation_pass.values())
        and all(temporal_pass.values())
        and all(event_pass.values())
        and event["start_transition_terminal_ordering_pass"]
    )
    unsigned = {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_id": rollout["candidate_id"],
        "analysis_split": "fit",
        "generation_round": rollout["generation_round"],
        "parent_policy_digest": rollout["parent_policy_digest"],
        "action_family_id": rollout["action_family_id"],
        "source_video": rollout["source_video"],
        "complete_caption": rollout["complete_caption"],
        "complete_caption_sha256": rollout["complete_caption_sha256"],
        "seed": rollout["seed"],
        "action_score": action["action_score"],
        "preservation_scores": preservation["primary_scores"],
        "preservation_gate_pass": preservation_pass,
        "all_absolute_preservation_gates_pass": all(preservation_pass.values()),
        "temporal_counterfactual_scores": temporal_scores,
        "temporal_counterfactual_gate_pass": temporal_pass,
        "event81_scores": event["event_scores"],
        "event81_gate_pass": event_pass,
        "start_transition_terminal_ordering_pass": event[
            "start_transition_terminal_ordering_pass"
        ],
        "all_endpoint_hard_gates_pass": hard_pass,
        "absolute_action_gate_pass": action["action_score"] >= thresholds["action"],
        "wrong_source_diagnostics": preservation["wrong_source_diagnostics"],
        "wrong_source_authorizes_optimizer": False,
        "rollout_evidence_file": rollout_file,
        "rollout_evidence_digest": rollout["evidence_digest"],
        "action_receipt_file": action_file,
        "action_receipt_digest": action["receipt_digest"],
        "preservation_receipt_file": preservation_file,
        "preservation_receipt_digest": preservation["receipt_digest"],
        "temporal_counterfactual_receipt_file": temporal_file,
        "temporal_counterfactual_receipt_digest": temporal["receipt_digest"],
        "event81_receipt_file": event_file,
        "event81_receipt_digest": event["receipt_digest"],
        "decoded_temporal_event_master_receipt_file": temporal[
            "master_receipt_file"
        ],
        "decoded_temporal_event_master_receipt_digest": temporal[
            "master_receipt_digest"
        ],
        "candidate_mp4": rollout["candidate_mp4"],
        "clean_latent": rollout["clean_latent"],
        "clean_latent_tensor_key": rollout["clean_latent_tensor_key"],
        "clean_latent_shape": rollout["clean_latent_shape"],
        "legacy_rollout_receipt": rollout["legacy_rollout_receipt"],
        "legacy_rollout_receipt_digest": rollout["legacy_rollout_receipt_digest"],
        "evidence_fit_for_pairing": hard_pass,
    }
    return {**unsigned, "candidate_digest": object_sha256(unsigned)}


def _pair_group_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["generation_round"],
        candidate["parent_policy_digest"],
        candidate["source_video"]["sha256"],
        candidate["complete_caption_sha256"],
        candidate["action_family_id"],
        candidate["analysis_split"],
    )


def assemble_one_update_manifest(
    value: Any, *, build_request_file: Mapping[str, Any]
) -> dict[str, Any]:
    request = validate_build_request(value)
    request_file = validate_file_binding(
        build_request_file, label="sealed build request", verify_bytes=True
    )
    loaded_request, _ = load_bound_json(
        request_file, label="sealed build request replay"
    )
    if loaded_request != value:
        raise PairV5EndpointV3Error("sealed build request bytes differ from input")
    policy = request["gate_policy"]
    candidates = [
        _derive_candidate(item, policy) for item in request["candidate_inputs"]
    ]
    ids = [item["candidate_id"] for item in candidates]
    if len(ids) != len(set(ids)):
        raise PairV5EndpointV3Error("candidate IDs are not unique")
    round_keys = {
        (item["generation_round"], item["parent_policy_digest"])
        for item in candidates
    }
    if len(round_keys) != 1:
        raise PairV5EndpointV3Error("candidate population mixes round/parent policy")
    generation_round, parent_policy_digest = next(iter(round_keys))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(_pair_group_key(candidate), []).append(candidate)
    thresholds = policy["thresholds"]
    pairs = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        eligible = [
            item for item in groups[key] if item["evidence_fit_for_pairing"]
        ]
        eligible.sort(key=lambda item: (-item["action_score"], item["candidate_id"]))
        if len(eligible) < 2:
            continue
        winner = eligible[0]
        loser = min(eligible[1:], key=lambda item: (item["action_score"], item["candidate_id"]))
        margin = winner["action_score"] - loser["action_score"]
        if not winner["absolute_action_gate_pass"] or margin < thresholds["action_margin"]:
            continue
        pair_id = f"r{generation_round}-{len(pairs):03d}-{winner['candidate_id']}"
        endpoint_projection = lambda item: {
            key: item[key]
            for key in (
                "candidate_id",
                "candidate_digest",
                "action_score",
                "preservation_scores",
                "temporal_counterfactual_scores",
                "event81_scores",
                "candidate_mp4",
                "clean_latent",
                "clean_latent_tensor_key",
                "clean_latent_shape",
                "legacy_rollout_receipt",
                "legacy_rollout_receipt_digest",
                "rollout_evidence_digest",
                "rollout_evidence_file",
                "action_receipt_digest",
                "action_receipt_file",
                "preservation_receipt_digest",
                "preservation_receipt_file",
                "temporal_counterfactual_receipt_digest",
                "temporal_counterfactual_receipt_file",
                "event81_receipt_digest",
                "event81_receipt_file",
                "decoded_temporal_event_master_receipt_digest",
                "decoded_temporal_event_master_receipt_file",
            )
        }
        unsigned_pair = {
            "schema_version": PAIR_SCHEMA,
            "pair_id": pair_id,
            "generation_round": generation_round,
            "parent_policy_digest": parent_policy_digest,
            "analysis_split": "fit",
            "action_family_id": winner["action_family_id"],
            "source_video": winner["source_video"],
            "complete_caption": winner["complete_caption"],
            "complete_caption_sha256": winner["complete_caption_sha256"],
            "winner": endpoint_projection(winner),
            "loser": endpoint_projection(loser),
            "action_margin": margin,
            "both_endpoints_pass_all_absolute_preservation_gates": True,
            "both_endpoints_pass_temporal_counterfactual_gates": True,
            "both_endpoints_pass_full81_event_gates": True,
            "both_endpoints_pass_all_endpoint_hard_gates": True,
            "winner_passes_absolute_action_gate": True,
            "same_round_parent_source_prompt_family_fit": True,
        }
        pairs.append({**unsigned_pair, "pair_digest": object_sha256(unsigned_pair)})
    distinct_sources = sorted({pair["source_video"]["sha256"] for pair in pairs})
    if not pairs or len(distinct_sources) < policy["minimum_distinct_sources"]:
        raise NoAuthorizedPairsError(
            "hard-gated fit pairs do not cover the required independent sources"
        )
    if len({pair["parent_policy_digest"] for pair in pairs}) != 1:
        raise PairV5EndpointV3Error("selected pairs mix parent policies")
    unsigned = {
        "schema_version": MANIFEST_SCHEMA,
        "optimizer_authorized": True,
        "generation_round": generation_round,
        "parent_policy_digest": parent_policy_digest,
        "gate_policy": policy,
        "gate_policy_digest": policy["gate_policy_digest"],
        "source_count": len(distinct_sources),
        "source_sha256_order": distinct_sources,
        "pair_count": len(pairs),
        "pairs": pairs,
        "optimizer_update_count": 1,
        "manifest_single_use": True,
        "fresh_rollout_required_after_update": True,
        "expected_next_generation_round": generation_round + 1,
        "round0_static_pair_one_update_canary_allowed": generation_round == 0,
        "round_greater_than_zero_requires_fresh_rollout": True,
        "confirmation_consumed_by_optimizer": False,
        "wrong_source_consumed_as_optimizer_gate": False,
        "same_video_temporal_counterfactual_required": True,
        "full81_start_transition_terminal_hold_required": True,
        "caller_scores_or_flags_consumed": False,
        "world_size": WORLD_SIZE,
        "data_parallel_size": DP_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "build_request_file": request_file,
        "build_request_digest": request["request_digest"],
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def validate_one_update_manifest(
    value: Any, *, replay_files: bool = True
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "optimizer_authorized",
        "generation_round",
        "parent_policy_digest",
        "gate_policy",
        "gate_policy_digest",
        "source_count",
        "source_sha256_order",
        "pair_count",
        "pairs",
        "optimizer_update_count",
        "manifest_single_use",
        "fresh_rollout_required_after_update",
        "expected_next_generation_round",
        "round0_static_pair_one_update_canary_allowed",
        "round_greater_than_zero_requires_fresh_rollout",
        "confirmation_consumed_by_optimizer",
        "wrong_source_consumed_as_optimizer_gate",
        "same_video_temporal_counterfactual_required",
        "full81_start_transition_terminal_hold_required",
        "caller_scores_or_flags_consumed",
        "world_size",
        "data_parallel_size",
        "sequence_parallel_size",
        "build_request_file",
        "build_request_digest",
        "manifest_digest",
    }
    row = dict(_closed(value, fields, label="one-update manifest"))
    digest = _verify_embedded(row, field="manifest_digest", label="one-update manifest")
    gate_policy = validate_gate_policy(row["gate_policy"])
    if (
        row["schema_version"] != MANIFEST_SCHEMA
        or row["optimizer_authorized"] is not True
        or row["gate_policy_digest"] != gate_policy["gate_policy_digest"]
        or row["optimizer_update_count"] != 1
        or row["manifest_single_use"] is not True
        or row["fresh_rollout_required_after_update"] is not True
        or row["expected_next_generation_round"] != row["generation_round"] + 1
        or row["round0_static_pair_one_update_canary_allowed"]
        is not (row["generation_round"] == 0)
        or row["round_greater_than_zero_requires_fresh_rollout"] is not True
        or row["confirmation_consumed_by_optimizer"] is not False
        or row["wrong_source_consumed_as_optimizer_gate"] is not False
        or row["same_video_temporal_counterfactual_required"] is not True
        or row["full81_start_transition_terminal_hold_required"] is not True
        or row["caller_scores_or_flags_consumed"] is not False
        or (row["world_size"], row["data_parallel_size"], row["sequence_parallel_size"])
        != (WORLD_SIZE, DP_SIZE, SP_SIZE)
        or not isinstance(row["pairs"], list)
        or row["pair_count"] != len(row["pairs"])
        or not row["pairs"]
    ):
        raise PairV5EndpointV3Error("one-update manifest authority differs")
    sources = set()
    for pair in row["pairs"]:
        pair_digest = _verify_embedded(pair, field="pair_digest", label="qualified pair")
        if (
            pair.get("schema_version") != PAIR_SCHEMA
            or pair.get("generation_round") != row["generation_round"]
            or pair.get("parent_policy_digest") != row["parent_policy_digest"]
            or pair.get("analysis_split") != "fit"
            or pair.get("both_endpoints_pass_all_absolute_preservation_gates") is not True
            or pair.get("both_endpoints_pass_temporal_counterfactual_gates") is not True
            or pair.get("both_endpoints_pass_full81_event_gates") is not True
            or pair.get("both_endpoints_pass_all_endpoint_hard_gates") is not True
            or pair.get("winner_passes_absolute_action_gate") is not True
            or pair.get("same_round_parent_source_prompt_family_fit") is not True
        ):
            raise PairV5EndpointV3Error("qualified pair authority differs")
        source = validate_file_binding(
            pair.get("source_video"), label="pair source", verify_bytes=replay_files
        )
        sources.add(source["sha256"])
        for endpoint_name in ("winner", "loser"):
            endpoint = pair.get(endpoint_name)
            if not isinstance(endpoint, Mapping):
                raise PairV5EndpointV3Error("qualified endpoint is absent")
            validate_file_binding(
                endpoint.get("candidate_mp4"),
                label=f"{endpoint_name} MP4",
                verify_bytes=replay_files,
            )
            validate_file_binding(
                endpoint.get("clean_latent"),
                label=f"{endpoint_name} clean latent",
                verify_bytes=replay_files,
            )
            validate_file_binding(
                endpoint.get("legacy_rollout_receipt"),
                label=f"{endpoint_name} rollout receipt",
                verify_bytes=replay_files,
            )
            for evidence_name in (
                "rollout_evidence_file",
                "action_receipt_file",
                "preservation_receipt_file",
                "temporal_counterfactual_receipt_file",
                "event81_receipt_file",
                "decoded_temporal_event_master_receipt_file",
            ):
                validate_file_binding(
                    endpoint.get(evidence_name),
                    label=f"{endpoint_name} {evidence_name}",
                    verify_bytes=replay_files,
                )
        if pair["winner"]["action_score"] - pair["loser"]["action_score"] != pair["action_margin"]:
            raise PairV5EndpointV3Error("qualified pair action margin differs")
        pair["pair_digest"] = pair_digest
    if (
        sorted(sources) != row["source_sha256_order"]
        or len(sources) != row["source_count"]
        or len(sources) < gate_policy["minimum_distinct_sources"]
    ):
        raise PairV5EndpointV3Error("manifest independent-source coverage differs")
    row["gate_policy"] = gate_policy
    row["manifest_digest"] = digest
    if replay_files:
        request_raw, request_file = load_bound_json(
            row["build_request_file"], label="sealed build request"
        )
        rebuilt = assemble_one_update_manifest(
            request_raw, build_request_file=request_file
        )
        if rebuilt != row:
            raise PairV5EndpointV3Error(
                "one-update manifest does not replay from trusted evidence"
            )
    return row


def authorize_manifest_for_single_step(
    value: Any,
    *,
    expected_generation_round: int,
    expected_parent_policy_digest: str,
    optimizer_step_index: int,
    replay_files: bool = True,
) -> dict[str, Any]:
    """Authorize only step zero of the exact bound rollout generation."""

    manifest = validate_one_update_manifest(value, replay_files=replay_files)
    if type(expected_generation_round) is not int or expected_generation_round < 0:
        raise PairV5EndpointV3Error("expected generation round differs")
    expected_parent = _sha256(
        expected_parent_policy_digest, label="expected parent policy digest"
    )
    if (
        optimizer_step_index != 0
        or manifest["optimizer_update_count"] != 1
        or manifest["generation_round"] != expected_generation_round
        or manifest["parent_policy_digest"] != expected_parent
    ):
        raise PairV5EndpointV3Error(
            "manifest is stale, already consumed, or bound to another parent policy"
        )
    return manifest


def write_create_only(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    if not destination.is_absolute() or destination == Path("/") or destination.exists():
        raise PairV5EndpointV3Error("output must be a fresh absolute non-root file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--request", required=True)
    assemble.add_argument("--expected-request-sha256", required=True)
    assemble.add_argument("--output", required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    binding = {"path": str(Path(getattr(args, "request", getattr(args, "manifest", ""))).resolve()),
               "sha256": getattr(args, "expected_request_sha256", getattr(args, "expected_manifest_sha256", ""))}
    raw, _ = load_bound_json(binding, label=args.command)
    if args.command == "assemble":
        manifest = assemble_one_update_manifest(raw, build_request_file=binding)
        write_create_only(Path(args.output).resolve(), manifest)
        result = {"manifest_digest": manifest["manifest_digest"], "pair_count": manifest["pair_count"], "source_count": manifest["source_count"]}
    else:
        manifest = validate_one_update_manifest(raw, replay_files=True)
        result = {"manifest_digest": manifest["manifest_digest"], "valid": True}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_ACTION_SCHEMA",
    "BUILD_REQUEST_SCHEMA",
    "CANDIDATE_SCHEMA",
    "DECODED_PRESERVATION_SCHEMA",
    "GATE_POLICY_SCHEMA",
    "MANIFEST_SCHEMA",
    "NoAuthorizedPairsError",
    "PAIR_SCHEMA",
    "POLICY_ATTESTATION_SCHEMA",
    "POLICY_SCHEMA",
    "PairV5EndpointV3Error",
    "ROLLOUT_EVIDENCE_SCHEMA",
    "adapt_active_v4_action_receipt",
    "adapt_decoded_preservation_receipt",
    "assemble_one_update_manifest",
    "authorize_manifest_for_single_step",
    "file_binding",
    "make_build_request",
    "make_gate_policy",
    "make_parent_policy",
    "make_policy_attestation",
    "make_rollout_evidence",
    "object_sha256",
    "validate_build_request",
    "validate_gate_policy",
    "validate_one_update_manifest",
    "validate_parent_policy",
    "validate_rollout_evidence",
]
