#!/usr/bin/env python3
"""Fail-closed preflight for the development-only v4-E target-graph binder.

This module is intentionally not a trainer.  It verifies a sealed
representation-only preregistration, the audited MEV catalog authority, the
complete evaluation-UUID exclusion registry, and (when one eventually
exists) a fully qualified group-disjoint split manifest.  It never loads a
generator, video, feature tensor, teacher checkpoint, or optimizer.

The target video is allowed only behind a frozen offline graph teacher.  The
binder may eventually learn from canonical object nodes, typed relation
lifecycle states, and calibrated uncertainty.  Target RGB, latents, flow,
teacher embeddings/hidden states, or values are forbidden.  Its online input
is a target-free self-generated Bernini *middle-layer* anchor; a decoded
anchor video is not an input.  Generator connection remains forbidden until
the separately registered representation gates pass.

The current 3,749-row catalog is expected to fail this preflight: it is
pending human qualification, has ``formal_sft_authorized=false``, and has no
actor/scene/action/media/perceptual grouping authority.  A blocked receipt is
the correct executable result; this file cannot weaken those blockers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, NoReturn, Sequence
import uuid as uuid_module


PREREGISTRATION_SCHEMA = "bernini-target-graph-binder-development-prereg-v4e"
CATALOG_RECEIPT_SCHEMA = "bernini-real-target-graph-pretrain-catalog-receipt-v1"
EXCLUSION_SCHEMA = "bernini-real-target-graph-eval-exclusion-registry-v1"
QUALIFIED_SPLIT_SCHEMA = "bernini-target-graph-binder-qualified-split-v4e"
PREFLIGHT_RECEIPT_SCHEMA = "bernini-target-graph-binder-development-preflight-v4e"

READY_STATUS = "V4E_BINDER_ONLY_LAUNCH_READY"
BLOCKED_STATUS = "V4E_BINDER_ONLY_LAUNCH_BLOCKED"
EXPECTED_EXPERIMENT_ID = "target_graph_binder_development_v4e"
EXPECTED_CATALOG_STATUS = "CATALOG_AUDITED_TRAIN_AUTHORIZED"
EXPECTED_SPLIT_STATUS = "DEVELOPMENT_GRAPH_BINDER_PRETRAIN_AUTHORIZED"

TRAINING_PARTITIONS = ("train", "validation")
GROUP_KINDS = (
    "video_uuid",
    "actor_group_id",
    "scene_group_id",
    "action_group_id",
    "source_media_sha256",
    "target_media_sha256",
    "perceptual_cluster_id",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIX_RE = re.compile(r"^[0-9a-f]{12}$")
GROUP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}$")


class TargetGraphBinderPreflightError(RuntimeError):
    """Raised before ambiguous authority can become a launch manifest."""


def fail(message: str) -> NoReturn:
    raise TargetGraphBinderPreflightError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise TargetGraphBinderPreflightError("value is not canonical ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    fail(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"JSON authority must be one regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_constant,
        )
    except TargetGraphBinderPreflightError:
        raise
    except Exception as error:
        raise TargetGraphBinderPreflightError(f"cannot parse {path}: {error}") from error
    if type(value) is not dict:
        fail(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    return value


def _array(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        fail(f"{label} must be {'a nonempty ' if nonempty else 'an '}array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        fail(f"{label} must be nonempty boundary-trimmed text")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{label} must be a positive integer")
    return value


def _canonical_uuid(value: Any, label: str) -> str:
    text = _text(value, label)
    try:
        parsed = uuid_module.UUID(text)
    except ValueError as error:
        raise TargetGraphBinderPreflightError(f"{label} must be a UUID") from error
    if str(parsed) != text:
        fail(f"{label} must use canonical lowercase UUID spelling")
    return text


def _group_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or GROUP_ID_RE.fullmatch(value) is None:
        fail(f"{label} must be an explicit safe group identifier")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def _verify_self_hash(value: Mapping[str, Any], key: str, label: str) -> str:
    expected = _sha256(value.get(key), f"{label}.{key}")
    payload = dict(value)
    payload.pop(key, None)
    actual = object_sha256(payload)
    if actual != expected:
        fail(f"{label} self hash differs")
    return actual


def _require_exact_booleans(
    value: Mapping[str, Any], expected: Mapping[str, bool], label: str
) -> None:
    for key, required in expected.items():
        if value.get(key) is not required:
            fail(f"{label}.{key} must be {required!r}")


def validate_preregistration(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if value.get("schema_version") != PREREGISTRATION_SCHEMA:
        fail("preregistration schema changed")
    _verify_self_hash(value, "preregistration_sha256", "preregistration")
    if value.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        fail("experiment id changed")
    if value.get("stage") != "development_only_representation_pretraining":
        fail("v4-E is representation-only development work")

    frozen = _mapping(value.get("frozen_base"), "preregistration.frozen_base")
    _require_exact_booleans(
        frozen,
        {
            "required_in_every_future_decoded_review": True,
            "loaded_in_this_observer_only_pilot": False,
            "generator_optimizer_created": False,
        },
        "preregistration.frozen_base",
    )
    if frozen.get("generator_forward_calls") != 0 or frozen.get("generator_parameter_updates") != 0:
        fail("observer-only v4-E must have zero generator calls and updates")

    boundary = _mapping(value.get("leakage_boundary"), "preregistration.leakage_boundary")
    _require_exact_booleans(
        boundary,
        {
            "target_video_offline_frozen_teacher_only": True,
            "target_video_is_binder_input": False,
            "target_is_generator_input": False,
            "target_gradient_allowed": False,
            "target_rgb_regression_allowed": False,
            "target_latent_regression_allowed": False,
            "target_flow_regression_allowed": False,
            "target_hidden_or_value_copy_allowed": False,
            "raw_teacher_embedding_export_allowed": False,
            "canonical_graph_metadata_only": True,
            "self_generated_anchor_middle_is_target_free": True,
            "final_anchor_video_feature_used": False,
            "generator_connection_authorized": False,
        },
        "preregistration.leakage_boundary",
    )

    model = _mapping(value.get("binder"), "preregistration.binder")
    if model.get("trainable_scope") != "object_edge_binder_only_then_freeze":
        fail("only the object/edge binder may eventually be trainable")
    if model.get("object_slots") != "competitive_variable_cardinality_with_null_abstain":
        fail("v4-E binder must support flexible object cardinality and abstention")
    if model.get("edge_topology") != "sparse_typed_phase_lifecycle_not_cartesian":
        fail("v4-E graph must use sparse typed phase lifecycle edges")
    if model.get("width") != 256 or model.get("layers") != 4:
        fail("v4-E binder architecture changed after preregistration")
    if model.get("max_slots") != 8 or model.get("max_neighbors_per_node") != 3:
        fail("v4-E slot/edge caps changed after preregistration")
    if model.get("anchor_middle_blocks") != [6, 12, 18, 24]:
        fail("v4-E anchor middle block scope changed")

    split = _mapping(value.get("split_contract"), "preregistration.split_contract")
    _require_exact_booleans(
        split,
        {
            "exclude_all_registered_evaluation_video_uuids_from_training": True,
            "exclude_all_adjacent_pairs_from_registered_evaluation_uuids": True,
            "require_complete_actor_scene_action_groups": True,
            "require_complete_media_hashes": True,
            "require_complete_perceptual_clusters": True,
            "require_group_disjoint_train_validation": True,
            "free_text_group_inference_allowed": False,
        },
        "preregistration.split_contract",
    )
    _positive_int(split.get("minimum_train_rows"), "minimum_train_rows")
    _positive_int(split.get("minimum_validation_rows"), "minimum_validation_rows")

    controls = _array(value.get("required_controls"), "required_controls", nonempty=True)
    required_controls = {
        "forward", "reverse", "noop", "static", "phase_shuffle",
        "role_swap", "drop_edge", "cross_appearance_three_way",
    }
    if set(controls) != required_controls or len(controls) != len(required_controls):
        fail("required controls changed")

    gates = _mapping(value.get("representation_admission"), "representation_admission")
    if _finite_number(gates.get("cross_appearance_cosine_min"), "cosine gate") != 0.95:
        fail("cross-appearance cosine gate changed")
    if _finite_number(gates.get("cross_appearance_distance_max"), "distance gate") != 0.15:
        fail("cross-appearance distance gate changed")
    if _finite_number(gates.get("forward_reverse_auroc_min"), "AUROC gate") != 0.90:
        fail("forward/reverse AUROC gate changed")
    if _finite_number(gates.get("noop_static_false_positive_max"), "false-positive gate") != 0.05:
        fail("noop/static false-positive gate changed")
    if gates.get("generator_connection_on_failure") != "forbidden":
        fail("failed representation may not connect to the generator")
    return value


def validate_exclusion_authority(
    registry: Mapping[str, Any], manual_contracts: Mapping[str, Any]
) -> Mapping[str, Any]:
    if registry.get("schema_version") != EXCLUSION_SCHEMA:
        fail("exclusion registry schema changed")
    cases = _array(registry.get("cases"), "exclusion cases", nonempty=True)
    if len(cases) != 16:
        fail("exclusion registry must contain exactly 16 cases")
    policy = _mapping(registry.get("policy"), "exclusion policy")
    if (
        policy.get("development_count") != 4
        or policy.get("locked_confirmation_count") != 12
        or policy.get("total_case_count") != 16
        or policy.get("exclude_entire_video_uuid") is not True
        or policy.get("exclude_all_temporally_adjacent_pairs") is not True
    ):
        fail("exclusion registry is not the sealed 4+12 entire-UUID policy")

    prefixes: set[str] = set()
    uuids: set[str] = set()
    pair_ids: set[str] = set()
    role_counts = Counter()
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"exclusion case {index}")
        prefix = _text(case.get("pair_prefix"), f"case {index} prefix")
        if PREFIX_RE.fullmatch(prefix) is None:
            fail(f"case {index} prefix differs")
        pair_id = _sha256(case.get("pair_id"), f"case {prefix} pair_id")
        video_uuid = _canonical_uuid(case.get("uuid"), f"case {prefix} uuid")
        if not pair_id.startswith(prefix):
            fail(f"case {prefix} pair ID differs")
        role = case.get("evaluation_role")
        if role not in {"development", "locked_confirmation"}:
            fail(f"case {prefix} evaluation role differs")
        if case.get("adjacent_exclusion_scope") != "entire_video_uuid":
            fail(f"case {prefix} must exclude the entire UUID")
        if prefix in prefixes or pair_id in pair_ids or video_uuid in uuids:
            fail("exclusion case identities must be unique")
        prefixes.add(prefix)
        pair_ids.add(pair_id)
        uuids.add(video_uuid)
        role_counts[role] += 1
    if role_counts != Counter({"locked_confirmation": 12, "development": 4}):
        fail("exclusion registry role counts differ")

    if manual_contracts.get("schema_version") != "mev-action-anchor-human-contracts-v2":
        fail("manual action contracts schema changed")
    samples = _array(manual_contracts.get("samples"), "manual contract samples", nonempty=True)
    manual_prefixes = {sample.get("pair_prefix") for sample in samples if type(sample) is dict}
    if len(samples) != 16 or manual_prefixes != prefixes:
        fail("manual action contracts do not exactly close the exclusion registry")
    return {
        "all_evaluation_uuids": sorted(uuids),
        "all_evaluation_pair_ids": sorted(pair_ids),
        "development_uuids": sorted(
            case["uuid"] for case in cases if case["evaluation_role"] == "development"
        ),
        "locked_confirmation_uuids": sorted(
            case["uuid"]
            for case in cases
            if case["evaluation_role"] == "locked_confirmation"
        ),
    }


def audit_catalog_receipt(
    receipt: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    exclusion_registry_sha256: str,
    manual_contracts_sha256: str,
) -> tuple[list[str], Mapping[str, Any]]:
    if receipt.get("schema_version") != CATALOG_RECEIPT_SCHEMA:
        fail("catalog receipt schema changed")
    _verify_self_hash(receipt, "receipt_sha256", "catalog receipt")
    source_pin = _mapping(preregistration.get("catalog_source_pin"), "catalog source pin")
    candidate = _mapping(receipt.get("candidate_manifest"), "catalog candidate manifest")
    for key in ("path", "sha256", "row_count"):
        if candidate.get(key) != source_pin.get(key):
            fail(f"catalog candidate manifest pin differs at {key}")
    authority = _mapping(receipt.get("authority"), "catalog authority")
    if authority.get("exclusion_registry_sha256") != exclusion_registry_sha256:
        fail("catalog receipt exclusion registry bytes differ")
    if authority.get("manual_contracts_sha256") != manual_contracts_sha256:
        fail("catalog receipt manual contract bytes differ")
    if (
        authority.get("registered_development_cases") != 4
        or authority.get("registered_locked_confirmation_cases") != 12
        or authority.get("registered_total_cases") != 16
        or authority.get("all_registered_pairs_found_exactly") is not True
    ):
        fail("catalog receipt does not close the sealed evaluation registry")

    leakage = _mapping(receipt.get("leakage_audit"), "catalog leakage audit")
    if (
        leakage.get("exclusion_granularity") != "entire_video_uuid"
        or leakage.get("evaluation_uuid_in_candidate_count") != 0
        or leakage.get("named_evaluation_pair_in_candidate_count") != 0
        or leakage.get("actor_scene_action_from_free_text_inferred") is not False
    ):
        fail("catalog evaluation exclusion or grouping boundary differs")

    blockers: list[str] = []
    if receipt.get("status") != EXPECTED_CATALOG_STATUS:
        blockers.append(f"CATALOG_STATUS_{receipt.get('status')}")
    for blocker in _array(receipt.get("authorization_blockers"), "catalog blockers"):
        blockers.append(f"CATALOG_{_text(blocker, 'catalog blocker')}")
    usage = _mapping(receipt.get("usage_contract"), "catalog usage contract")
    data_status = _mapping(receipt.get("data_status"), "catalog data status")
    if usage.get("graph_teacher_pretraining_authorized") is not True:
        blockers.append("CATALOG_GRAPH_TEACHER_PRETRAINING_NOT_AUTHORIZED")
    if usage.get("trainer_readable_split_emitted") is not True:
        blockers.append("CATALOG_TRAINER_READABLE_SPLIT_NOT_EMITTED")
    if data_status.get("human_qualification_complete") is not True:
        blockers.append("CATALOG_HUMAN_QUALIFICATION_INCOMPLETE")
    if data_status.get("formal_sft_authorized") is not True:
        blockers.append("CATALOG_FORMAL_SFT_NOT_AUTHORIZED")
    if (
        usage.get("target_is_generator_input") is not False
        or usage.get("target_gradient_allowed") is not False
        or usage.get("target_rgb_latent_flow_regression_allowed") is not False
        or usage.get("target_hidden_or_value_copy_allowed") is not False
        or usage.get("generator_training_authorized") is not False
    ):
        fail("catalog leakage boundary was weakened")

    group_audit = _mapping(
        _mapping(leakage.get("group_split_audit"), "group split audit"),
        "group split audit",
    )
    counts = _mapping(receipt.get("counts"), "catalog counts")
    eligible_count = counts.get("catalog_candidate_pending_authority")
    if not isinstance(eligible_count, int) or eligible_count < 0:
        fail("catalog eligible count differs")
    coverage = _mapping(group_audit.get("row_coverage"), "catalog group coverage")
    for kind in GROUP_KINDS:
        if coverage.get(kind) != eligible_count:
            blockers.append(f"CATALOG_INCOMPLETE_{kind.upper()}_COVERAGE")
    if group_audit.get("all_available_groups_split_disjoint") is not True:
        blockers.append("CATALOG_GROUPS_CROSS_SOURCE_SPLITS")
    return sorted(set(blockers)), {
        "catalog_rows": counts.get("catalog_rows"),
        "candidate_rows_after_evaluation_uuid_exclusion": eligible_count,
        "excluded_total_by_evaluation_uuid": counts.get("excluded_total_by_evaluation_uuid"),
        "group_coverage": dict(coverage),
    }


def audit_development_teacher_receipts(
    receipts: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    preregistration: Mapping[str, Any],
) -> tuple[list[str], list[Mapping[str, Any]]]:
    warnings: list[str] = []
    evidence: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    snapshot = _mapping(
        preregistration.get("development_evidence_snapshot"),
        "development evidence snapshot",
    )
    expected_rows = _array(
        snapshot.get("r3_teacher_receipts"), "registered r3 teacher receipts"
    )
    expected = {
        _text(row.get("case_id"), "registered teacher case_id"): _sha256(
            row.get("file_sha256"), "registered teacher file sha256"
        )
        for row in expected_rows
        if type(row) is dict
    }
    if receipts and len(expected) != len(expected_rows):
        fail("registered development teacher cases are duplicated")
    for path, receipt in receipts:
        case_id = _text(receipt.get("case_id"), f"{path} case_id")
        if case_id in seen:
            fail("development teacher receipt case IDs must be unique")
        seen.add(case_id)
        if case_id not in expected or file_sha256(path) != expected[case_id]:
            fail(f"development teacher {case_id} bytes are outside the preregistration")
        leakage = _mapping(receipt.get("leakage_boundary"), f"{case_id} leakage boundary")
        if (
            leakage.get("target_graph_authorized_for_generator") is not False
            or leakage.get("target_graph_authorized_for_renderer") is not False
            or leakage.get("target_graph_authorized_for_selection") is not False
            or leakage.get("optimizer_updates") != 0
            or leakage.get("renderer_forward_calls") != 0
        ):
            fail(f"development teacher {case_id} leakage boundary differs")
        training_authorized = leakage.get("target_graph_authorized_for_training") is True
        observer_class = receipt.get("observer_class")
        if not training_authorized:
            warnings.append(f"DEV_TEACHER_{case_id}_NOT_TRAINING_AUTHORIZED")
        if observer_class == "teacher_observation_scaffold_not_oceg":
            warnings.append(f"DEV_TEACHER_{case_id}_SCAFFOLD_NOT_OCEG")
        evidence.append(
            {
                "case_id": case_id,
                "receipt_path": str(path),
                "receipt_file_sha256": expected[case_id],
                "schema_version": receipt.get("schema_version"),
                "observer_class": observer_class,
                "target_graph_authorized_for_training": training_authorized,
                "optimizer_updates": leakage.get("optimizer_updates"),
                "renderer_forward_calls": leakage.get("renderer_forward_calls"),
            }
        )
    if receipts and seen != set(expected):
        fail("provide either no development teacher receipts or the complete registered set")
    return sorted(warnings), sorted(evidence, key=lambda row: row["case_id"])


def validate_qualified_split_manifest(
    manifest: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    evaluation_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    if manifest.get("schema_version") != QUALIFIED_SPLIT_SCHEMA:
        fail("qualified split schema changed")
    _verify_self_hash(manifest, "manifest_sha256", "qualified split manifest")
    authorization = _mapping(manifest.get("authorization"), "split authorization")
    if authorization.get("status") != EXPECTED_SPLIT_STATUS:
        fail("qualified split lacks explicit development graph-binder authorization")
    _require_exact_booleans(
        authorization,
        {
            "human_qualification_complete": True,
            "formal_sft_authorized": True,
            "graph_teacher_pretraining_authorized": True,
            "generator_training_authorized": False,
        },
        "split authorization",
    )
    _text(authorization.get("authority_id"), "split authority_id")
    _text(authorization.get("sealed_at_utc"), "split sealed_at_utc")

    rows = _array(manifest.get("rows"), "qualified split rows", nonempty=True)
    excluded_uuids = set(evaluation_authority["all_evaluation_uuids"])
    excluded_pairs = set(evaluation_authority["all_evaluation_pair_ids"])
    seen_pairs: set[str] = set()
    group_splits: dict[str, dict[str, set[str]]] = {
        kind: defaultdict(set) for kind in GROUP_KINDS
    }
    counts = Counter()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, f"qualified split row {index}")
        required = {
            "pair_id", "video_uuid", "partition", "actor_group_id",
            "scene_group_id", "action_group_id", "source_media_sha256",
            "target_media_sha256", "perceptual_cluster_id", "source_video_path",
            "target_video_path", "teacher_graph_path", "teacher_graph_sha256",
            "anchor_middle_path", "anchor_middle_sha256", "teacher_contract",
            "anchor_middle_contract",
        }
        if set(row) != required:
            fail(
                f"qualified split row {index} field closure differs: "
                f"missing={sorted(required - set(row))} extra={sorted(set(row) - required)}"
            )
        pair_id = _sha256(row["pair_id"], f"row {index} pair_id")
        video_uuid = _canonical_uuid(row["video_uuid"], f"row {index} video_uuid")
        partition = row["partition"]
        if partition not in TRAINING_PARTITIONS:
            fail(f"row {index} partition is outside {TRAINING_PARTITIONS!r}")
        if pair_id in seen_pairs:
            fail(f"qualified split pair is duplicated: {pair_id}")
        if pair_id in excluded_pairs or video_uuid in excluded_uuids:
            fail(f"row {index} leaks a registered evaluation pair or video UUID")
        seen_pairs.add(pair_id)
        counts[partition] += 1

        group_values: dict[str, str] = {
            "video_uuid": video_uuid,
            "actor_group_id": _group_id(row["actor_group_id"], f"row {index} actor"),
            "scene_group_id": _group_id(row["scene_group_id"], f"row {index} scene"),
            "action_group_id": _group_id(row["action_group_id"], f"row {index} action"),
            "source_media_sha256": _sha256(row["source_media_sha256"], f"row {index} source media"),
            "target_media_sha256": _sha256(row["target_media_sha256"], f"row {index} target media"),
            "perceptual_cluster_id": _group_id(
                row["perceptual_cluster_id"], f"row {index} perceptual cluster"
            ),
        }
        if group_values["source_media_sha256"] == group_values["target_media_sha256"]:
            fail(f"row {index} source and target media hashes are identical")
        for kind, value in group_values.items():
            group_splits[kind][value].add(partition)

        for key in ("source_video_path", "target_video_path", "teacher_graph_path", "anchor_middle_path"):
            path_text = _text(row[key], f"row {index} {key}")
            if not path_text.startswith("/"):
                fail(f"row {index} {key} must be absolute")
        _sha256(row["teacher_graph_sha256"], f"row {index} teacher graph")
        _sha256(row["anchor_middle_sha256"], f"row {index} anchor middle")

        teacher = _mapping(row["teacher_contract"], f"row {index} teacher contract")
        _require_exact_booleans(
            teacher,
            {
                "teacher_frozen": True,
                "training_authorized": True,
                "canonical_graph_metadata_only": True,
                "raw_masks_exported": False,
                "rgb_latent_flow_exported": False,
                "teacher_embeddings_exported": False,
                "physical_contact_inferred_from_proximity_only": False,
                "uncertainty_and_visibility_exported": True,
            },
            f"row {index} teacher contract",
        )
        if teacher.get("graph_schema") != "qualified-object-interaction-graph-v4e":
            fail(f"row {index} teacher graph schema differs")

        anchor = _mapping(
            row["anchor_middle_contract"], f"row {index} anchor middle contract"
        )
        _require_exact_booleans(
            anchor,
            {
                "bernini_frozen": True,
                "target_video_accessed": False,
                "target_graph_accessed_during_extraction": False,
                "self_generated_intermediate_only": True,
                "decoded_final_video_used": False,
                "generator_parameter_updates": False,
                "raw_middle_tensor_persisted": False,
            },
            f"row {index} anchor middle contract",
        )
        if anchor.get("blocks") != preregistration["binder"]["anchor_middle_blocks"]:
            fail(f"row {index} anchor middle blocks differ")
        if anchor.get("published_scope") != "reduced_role_slot_edge_statistics_only":
            fail(f"row {index} anchor middle publication scope differs")

    split_contract = preregistration["split_contract"]
    if counts["train"] < split_contract["minimum_train_rows"]:
        fail("qualified split has too few train rows")
    if counts["validation"] < split_contract["minimum_validation_rows"]:
        fail("qualified split has too few validation rows")
    collisions: dict[str, list[str]] = {}
    for kind, groups in group_splits.items():
        collisions[kind] = sorted(group_id for group_id, parts in groups.items() if len(parts) > 1)
    if any(collisions.values()):
        fail(f"qualified split has cross-partition group leakage: {collisions}")
    return {
        "row_count": len(rows),
        "partition_counts": {partition: counts[partition] for partition in TRAINING_PARTITIONS},
        "all_required_groups_complete": True,
        "all_groups_train_validation_disjoint": True,
        "registered_evaluation_pair_or_uuid_rows": 0,
        "target_video_only_reaches_frozen_canonical_graph_teacher": True,
        "anchor_middle_is_target_free_and_final_video_free": True,
    }


def build_preflight_receipt(
    *,
    preregistration: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    exclusion_registry: Mapping[str, Any],
    exclusion_registry_sha256: str,
    manual_contracts: Mapping[str, Any],
    manual_contracts_sha256: str,
    development_teacher_receipts: Sequence[tuple[Path, Mapping[str, Any]]] = (),
    qualified_split_manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    validate_preregistration(preregistration)
    evaluation = validate_exclusion_authority(exclusion_registry, manual_contracts)
    blockers, catalog_audit = audit_catalog_receipt(
        catalog_receipt,
        preregistration=preregistration,
        exclusion_registry_sha256=exclusion_registry_sha256,
        manual_contracts_sha256=manual_contracts_sha256,
    )
    warnings, teacher_evidence = audit_development_teacher_receipts(
        development_teacher_receipts,
        preregistration=preregistration,
    )
    if qualified_split_manifest is None:
        blockers.append("QUALIFIED_SPLIT_MANIFEST_MISSING")
        split_audit: Mapping[str, Any] = {
            "provided": False,
            "trainer_readable_rows": 0,
        }
    else:
        split_audit = {
            "provided": True,
            **validate_qualified_split_manifest(
                qualified_split_manifest,
                preregistration=preregistration,
                evaluation_authority=evaluation,
            ),
        }
    blockers = sorted(set(blockers))
    ready = not blockers
    receipt: dict[str, Any] = {
        "schema_version": PREFLIGHT_RECEIPT_SCHEMA,
        "status": READY_STATUS if ready else BLOCKED_STATUS,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "launch_authorized": ready,
        "training_executed": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "generator": {
            "loaded": False,
            "forward_calls": 0,
            "optimizer_created": False,
            "parameter_updates": 0,
            "connected_to_binder": False,
            "frozen_base_required_in_future_decoded_review": True,
        },
        "catalog_audit": catalog_audit,
        "evaluation_exclusion": {
            "registered_development_uuids": len(evaluation["development_uuids"]),
            "registered_locked_confirmation_uuids": len(
                evaluation["locked_confirmation_uuids"]
            ),
            "all_registered_evaluation_uuids_excluded_from_training": True,
            "exclusion_granularity": "entire_video_uuid",
        },
        "qualified_split_audit": split_audit,
        "development_teacher_evidence": teacher_evidence,
        "warnings_not_training_authority": warnings,
        "authorization_blockers": blockers,
        "claim_limits": {
            "binder_pretraining_result": False,
            "stable_transferable_action_representation": False,
            "generator_effect": False,
            "quality_preservation": False,
            "causal_action_editing": False,
        },
        "source_hashes": {
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "catalog_receipt_sha256": catalog_receipt["receipt_sha256"],
            "exclusion_registry_file_sha256": exclusion_registry_sha256,
            "manual_contracts_file_sha256": manual_contracts_sha256,
            "qualified_split_manifest_sha256": (
                None
                if qualified_split_manifest is None
                else qualified_split_manifest["manifest_sha256"]
            ),
        },
    }
    receipt["preflight_receipt_sha256"] = object_sha256(receipt)
    return receipt


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise TargetGraphBinderPreflightError(f"refusing to overwrite {path}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--catalog-receipt", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--manual-contracts", type=Path, required=True)
    parser.add_argument("--development-teacher-receipt", type=Path, action="append", default=[])
    parser.add_argument("--qualified-split-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = load_json(args.preregistration)
    catalog_receipt = load_json(args.catalog_receipt)
    exclusion_registry = load_json(args.exclusion_registry)
    manual_contracts = load_json(args.manual_contracts)
    teacher_receipts = [
        (path, load_json(path)) for path in args.development_teacher_receipt
    ]
    qualified_split = (
        None
        if args.qualified_split_manifest is None
        else load_json(args.qualified_split_manifest)
    )
    receipt = build_preflight_receipt(
        preregistration=preregistration,
        catalog_receipt=catalog_receipt,
        exclusion_registry=exclusion_registry,
        exclusion_registry_sha256=file_sha256(args.exclusion_registry),
        manual_contracts=manual_contracts,
        manual_contracts_sha256=file_sha256(args.manual_contracts),
        development_teacher_receipts=teacher_receipts,
        qualified_split_manifest=qualified_split,
    )
    if args.output is None:
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _write_json_create_only(args.output, receipt)
    return 0 if receipt["launch_authorized"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
