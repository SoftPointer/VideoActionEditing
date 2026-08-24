#!/usr/bin/env python3
"""Fail-closed validator for the PAIR-v5 discovery rollout manifest.

The module is deliberately dependency-free.  Its default mode validates the
sealed JSON bytes and hashes recorded in the manifest without requiring AUH
mounts.  ``--verify-evidence-files`` additionally recomputes every referenced
source/evidence SHA-256 and is intended to run on AUH before submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-pair-v5-action-preference-rollout-train-v1"
MANIFEST_ID = "pair-v5-real-action-preference-rollout-train-v1-20260808"
BRANCH_ORDER = (
    "action",
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
FORBIDDEN_INPUT_KEYS = (
    "paired_target_video",
    "paired_target_latent",
    "proposal_video",
    "proposal_latent",
    "motion_donor",
    "external_mask",
    "external_optical_flow",
    "external_pose",
    "external_track",
    "external_trajectory",
    "custom_target_noise",
    "first_frame_only_anchor",
)
EXPECTED_SHARED8_IIDS = (
    "1852ada01d7c43a4",
    "288545b9c031491a",
    "5ae88e1170c544b8",
    "81473c034c1b4839",
    "2766a3662fbf43d1",
    "219c4c5f56e74b86",
    "2206cde2643e470a",
    "7a2f54be92024a19",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")


class PairV5RolloutManifestError(ValueError):
    """Raised whenever a rollout manifest is open, ambiguous, or altered."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5RolloutManifestError(f"{label} must be an object")
    return value


def _closed(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    expected = set(keys)
    actual = set(mapping)
    if actual != expected:
        raise PairV5RolloutManifestError(
            f"{label} keys differ: missing={sorted(expected-actual)} "
            f"extra={sorted(actual-expected)}"
        )
    return mapping


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PairV5RolloutManifestError(f"{label} must be an array")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PairV5RolloutManifestError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
        raise PairV5RolloutManifestError(f"{label} must be lowercase SHA-1")
    return value


def _path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/vast/") or "\x00" in value:
        raise PairV5RolloutManifestError(f"{label} must be an absolute AUH /vast path")
    return value


def _false(value: Any, label: str) -> None:
    if value is not False:
        raise PairV5RolloutManifestError(f"{label} must be exactly false")


def _true(value: Any, label: str) -> None:
    if value is not True:
        raise PairV5RolloutManifestError(f"{label} must be exactly true")


def _verify_file(path: str, digest: str, label: str) -> None:
    artifact = Path(path)
    if not artifact.is_file() or artifact.is_symlink():
        raise PairV5RolloutManifestError(f"{label} is missing or symlinked: {path}")
    hasher = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise PairV5RolloutManifestError(f"{label} SHA-256 differs")


def _validate_path_digest(
    mapping: Mapping[str, Any], path_key: str, digest_key: str, label: str,
    *, verify_files: bool,
) -> None:
    path = _path(mapping.get(path_key), f"{label}.{path_key}")
    digest = _sha256(mapping.get(digest_key), f"{label}.{digest_key}")
    if verify_files:
        _verify_file(path, digest, label)


def canonical_manifest_digest(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_manifest(payload: Any, *, verify_files: bool = False) -> dict[str, Any]:
    root = _closed(
        payload,
        (
            "schema_version", "manifest_id", "status", "purpose", "claim_limits",
            "authorization", "evidence_manifests", "model", "rendering_contract",
            "candidate_grid", "sources",
        ),
        "manifest",
    )
    if root["schema_version"] != SCHEMA_VERSION or root["manifest_id"] != MANIFEST_ID:
        raise PairV5RolloutManifestError("manifest schema/id differs")
    if root["status"] != "preregistered_not_rendered":
        raise PairV5RolloutManifestError("manifest status must remain preregistered_not_rendered")
    if not isinstance(root["purpose"], str) or not root["purpose"].strip():
        raise PairV5RolloutManifestError("purpose must be non-empty")

    limits = _closed(
        root["claim_limits"],
        (
            "exploratory_research_only", "upstream_preview_only",
            "production_claim_forbidden", "scientific_generalization_claim_authorized",
            "iid_disjoint_from_shared8", "content_disjoint_from_shared8_proven",
            "action_editing_success_claimed",
        ),
        "claim_limits",
    )
    for key in ("exploratory_research_only", "upstream_preview_only", "production_claim_forbidden", "iid_disjoint_from_shared8"):
        _true(limits[key], f"claim_limits.{key}")
    for key in ("scientific_generalization_claim_authorized", "content_disjoint_from_shared8_proven", "action_editing_success_claimed"):
        _false(limits[key], f"claim_limits.{key}")

    authorization = _closed(
        root["authorization"],
        (
            "source_use", "full644_upstream_training_authorized",
            "full644_upstream_training_use_forbidden", "paired_target_runtime_access",
        ),
        "authorization",
    )
    if authorization["source_use"] != "user_authorized_exploratory_source_only_rollout":
        raise PairV5RolloutManifestError("source-use authorization differs")
    _false(authorization["full644_upstream_training_authorized"], "upstream authorization")
    _true(authorization["full644_upstream_training_use_forbidden"], "upstream use prohibition")
    _false(authorization["paired_target_runtime_access"], "paired target runtime access")

    evidence = _closed(
        root["evidence_manifests"],
        (
            "full644_raw_membership_path", "full644_raw_membership_sha256",
            "full644_vae_index_path", "full644_vae_index_sha256",
            "shared8_exclusion_path", "shared8_exclusion_sha256",
            "shared8_exposure_audit_path", "shared8_exposure_audit_sha256",
            "shared8_forbidden_iids",
        ),
        "evidence_manifests",
    )
    for path_key, digest_key in (
        ("full644_raw_membership_path", "full644_raw_membership_sha256"),
        ("full644_vae_index_path", "full644_vae_index_sha256"),
        ("shared8_exclusion_path", "shared8_exclusion_sha256"),
        ("shared8_exposure_audit_path", "shared8_exposure_audit_sha256"),
    ):
        _validate_path_digest(evidence, path_key, digest_key, "evidence_manifests", verify_files=verify_files)
    if tuple(_sequence(evidence["shared8_forbidden_iids"], "shared8_forbidden_iids")) != EXPECTED_SHARED8_IIDS:
        raise PairV5RolloutManifestError("shared8 IID exclusion set/order differs")

    model = _closed(
        root["model"],
        (
            "name", "checkpoint_hf_revision", "checkpoint_tree_sha256",
            "bernini_commit", "veomni_commit", "weights_frozen_during_rollout",
        ),
        "model",
    )
    if model["name"] != "Bernini-R-1.3B-Diffusers renderer-only":
        raise PairV5RolloutManifestError("model name differs")
    _sha1(model["checkpoint_hf_revision"], "checkpoint_hf_revision")
    _sha256(model["checkpoint_tree_sha256"], "checkpoint_tree_sha256")
    _sha1(model["bernini_commit"], "bernini_commit")
    _sha1(model["veomni_commit"], "veomni_commit")
    _true(model["weights_frozen_during_rollout"], "weights_frozen_during_rollout")

    rendering = _closed(
        root["rendering_contract"],
        (
            "native_arm", "training_task_name", "guidance_mode", "frame_count",
            "latent_frame_count", "fps", "num_inference_steps", "flow_shift",
            "target_initialization", "full_source_video_count",
            "source_reference_frame_indices", "source_reference_encode",
            "accepted_external_conditions", "forbidden_conditions",
        ),
        "rendering_contract",
    )
    exact_render = {
        "native_arm": "rv2v", "training_task_name": "vr2v", "guidance_mode": "rv2v",
        "frame_count": 81, "latent_frame_count": 21, "fps": 25,
        "num_inference_steps": 40, "flow_shift": 5.0,
        "target_initialization": "official_gen_wanx22_fresh_gaussian",
        "full_source_video_count": 1,
        "source_reference_frame_indices": [0, 27, 53, 80],
        "source_reference_encode": "four_independent_single_rgb_frame_vae_encodes",
        "accepted_external_conditions": ["source_video", "source_derived_reference_frames", "action_prompt"],
    }
    for key, expected in exact_render.items():
        if rendering[key] != expected:
            raise PairV5RolloutManifestError(f"rendering_contract.{key} differs")
    forbidden = _closed(rendering["forbidden_conditions"], FORBIDDEN_INPUT_KEYS, "forbidden_conditions")
    for key in FORBIDDEN_INPUT_KEYS:
        _false(forbidden[key], f"forbidden_conditions.{key}")

    grid = _closed(
        root["candidate_grid"],
        (
            "cross_product", "seeds", "guidance_schedules", "prompt_branch_order",
            "source_count", "branches_per_source", "seed_count",
            "guidance_schedule_count", "expected_rollout_count",
            "postseal_seed_topup_allowed", "postseal_guidance_change_allowed",
            "failed_cell_replacement_allowed",
        ),
        "candidate_grid",
    )
    if grid["cross_product"] != ["sources", "prompt_branches", "seeds", "guidance_schedules"]:
        raise PairV5RolloutManifestError("candidate-grid cross product differs")
    seeds = _sequence(grid["seeds"], "candidate_grid.seeds")
    if len(seeds) < 4 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise PairV5RolloutManifestError("at least four unique valid integer seeds are required")
    if tuple(grid["prompt_branch_order"]) != BRANCH_ORDER:
        raise PairV5RolloutManifestError("prompt branch set/order differs")
    schedules = _sequence(grid["guidance_schedules"], "guidance_schedules")
    if len(schedules) != 1:
        raise PairV5RolloutManifestError("v1 seals exactly one guidance schedule")
    schedule = _closed(
        schedules[0],
        (
            "schedule_id", "omega_vid", "omega_img", "omega_txt", "omega_scale",
            "eta", "norm_threshold", "momentum", "time_varying_text_guidance",
        ),
        "guidance_schedule",
    )
    expected_schedule = {
        "schedule_id": "native-default-fixed-v1", "omega_vid": 1.25,
        "omega_img": 4.5, "omega_txt": 4.0, "omega_scale": 0.8,
        "eta": 0.5, "norm_threshold": [50.0, 50.0], "momentum": 0.0,
        "time_varying_text_guidance": False,
    }
    if dict(schedule) != expected_schedule:
        raise PairV5RolloutManifestError("native guidance schedule differs")
    for key in ("postseal_seed_topup_allowed", "postseal_guidance_change_allowed", "failed_cell_replacement_allowed"):
        _false(grid[key], f"candidate_grid.{key}")

    sources = _sequence(root["sources"], "sources")
    if not 2 <= len(sources) <= 4:
        raise PairV5RolloutManifestError("PAIR-v5 v1 requires two to four sources")
    if grid["source_count"] != len(sources) or grid["branches_per_source"] != len(BRANCH_ORDER):
        raise PairV5RolloutManifestError("source/branch grid counts differ")
    if grid["seed_count"] != len(seeds) or grid["guidance_schedule_count"] != len(schedules):
        raise PairV5RolloutManifestError("seed/schedule grid counts differ")
    expected_rollouts = len(sources) * len(BRANCH_ORDER) * len(seeds) * len(schedules)
    if grid["expected_rollout_count"] != expected_rollouts:
        raise PairV5RolloutManifestError("expected rollout count differs")

    seen_iids: set[str] = set()
    selection_roles: set[str] = set()
    for index, raw_source in enumerate(sources):
        label = f"sources[{index}]"
        source = _closed(
            raw_source,
            (
                "iid", "split", "project_membership", "shared8_overlap",
                "selection_role", "full644_row_digest", "full644_selection_gates",
                "source_video", "source_annotation", "qwen_source_audit",
                "full644_vae_membership", "source_caption", "action_program", "prompts",
            ),
            label,
        )
        iid = source["iid"]
        if not isinstance(iid, str) or re.fullmatch(r"[0-9a-f]{16}", iid) is None:
            raise PairV5RolloutManifestError(f"{label}.iid differs")
        if iid in seen_iids or iid in EXPECTED_SHARED8_IIDS:
            raise PairV5RolloutManifestError(f"{label} duplicates or leaks shared8")
        seen_iids.add(iid)
        if source["split"] != "pair_v5_rollout_train_discovery_v1" or source["project_membership"] != "bernini_full644_train_membership":
            raise PairV5RolloutManifestError(f"{label} is not a full644 train-discovery source")
        _false(source["shared8_overlap"], f"{label}.shared8_overlap")
        if not isinstance(source["selection_role"], str) or source["selection_role"] in selection_roles:
            raise PairV5RolloutManifestError(f"{label}.selection_role is empty/duplicated")
        selection_roles.add(source["selection_role"])
        _sha256(source["full644_row_digest"], f"{label}.full644_row_digest")

        gates = _closed(
            source["full644_selection_gates"],
            (
                "single_dynamic_actor", "source_camera_locked_off",
                "source_census_high_confidence", "target_camera_locked_off",
                "target_camera_preserve_static", "target_plan_high_confidence",
            ),
            f"{label}.full644_selection_gates",
        )
        for gate in ("source_camera_locked_off", "source_census_high_confidence", "target_camera_locked_off", "target_camera_preserve_static", "target_plan_high_confidence"):
            _true(gates[gate], f"{label}.{gate}")
        if type(gates["single_dynamic_actor"]) is not bool:
            raise PairV5RolloutManifestError(f"{label}.single_dynamic_actor must be bool")

        video = _closed(
            source["source_video"],
            ("path", "full644_mirror_path", "sha256", "width", "height", "frame_count", "fps_num", "fps_den"),
            f"{label}.source_video",
        )
        primary_path = _path(video["path"], f"{label}.source_video.path")
        mirror_path = _path(video["full644_mirror_path"], f"{label}.source_video.full644_mirror_path")
        video_digest = _sha256(video["sha256"], f"{label}.source_video.sha256")
        if [video["width"], video["height"], video["frame_count"], video["fps_num"], video["fps_den"]] != [736, 704, 81, 25, 1]:
            raise PairV5RolloutManifestError(f"{label} source exact81 geometry differs")
        if verify_files:
            _verify_file(primary_path, video_digest, f"{label} primary source")
            _verify_file(mirror_path, video_digest, f"{label} full644 source mirror")

        annotation = _closed(source["source_annotation"], ("path", "sha256", "source_caption_field"), f"{label}.source_annotation")
        if annotation["source_caption_field"] != "source_caption":
            raise PairV5RolloutManifestError(f"{label} source-caption field differs")
        _validate_path_digest(annotation, "path", "sha256", f"{label}.source_annotation", verify_files=verify_files)
        qwen = _closed(source["qwen_source_audit"], ("path", "sha256"), f"{label}.qwen_source_audit")
        _validate_path_digest(qwen, "path", "sha256", f"{label}.qwen_source_audit", verify_files=verify_files)

        vae = _closed(
            source["full644_vae_membership"],
            ("bucket_hw", "posterior_parameters_shape", "parquet_path", "parquet_sha256", "receipt_path", "receipt_sha256"),
            f"{label}.full644_vae_membership",
        )
        if vae["bucket_hw"] != [480, 496] or vae["posterior_parameters_shape"] != [1, 32, 21, 60, 62]:
            raise PairV5RolloutManifestError(f"{label} full644 VAE geometry differs")
        _validate_path_digest(vae, "parquet_path", "parquet_sha256", f"{label}.vae_parquet", verify_files=verify_files)
        _validate_path_digest(vae, "receipt_path", "receipt_sha256", f"{label}.vae_receipt", verify_files=verify_files)

        caption = source["source_caption"]
        if not isinstance(caption, str) or len(caption.split()) < 25 or "\x00" in caption:
            raise PairV5RolloutManifestError(f"{label}.source_caption is not complete")
        program = _closed(
            source["action_program"],
            ("actor", "patient", "distractor_actor", "distractor_patient", "ordered_milestones", "terminal_state"),
            f"{label}.action_program",
        )
        for role in ("actor", "patient", "distractor_actor", "distractor_patient", "terminal_state"):
            if not isinstance(program[role], str) or not program[role].strip():
                raise PairV5RolloutManifestError(f"{label}.action_program.{role} is empty")
        milestones = _sequence(program["ordered_milestones"], f"{label}.ordered_milestones")
        if len(milestones) != 4 or any(not isinstance(item, str) or not item.strip() for item in milestones):
            raise PairV5RolloutManifestError(f"{label} must have four ordered milestones")

        prompts = _closed(source["prompts"], BRANCH_ORDER, f"{label}.prompts")
        prompt_texts: set[str] = set()
        for branch in BRANCH_ORDER:
            prompt = _closed(prompts[branch], ("text", "sha256"), f"{label}.prompts.{branch}")
            text = prompt["text"]
            if not isinstance(text, str) or len(text.split()) < 25 or "\x00" in text:
                raise PairV5RolloutManifestError(f"{label}.{branch} is not a full source-content prompt")
            if text in prompt_texts:
                raise PairV5RolloutManifestError(f"{label} has duplicate prompt text")
            prompt_texts.add(text)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if _sha256(prompt["sha256"], f"{label}.{branch}.sha256") != digest:
                raise PairV5RolloutManifestError(f"{label}.{branch} prompt SHA-256 differs")

    return {
        "manifest_id": MANIFEST_ID,
        "manifest_digest": canonical_manifest_digest(root),
        "source_iids": sorted(seen_iids),
        "rollout_count": expected_rollouts,
        "evidence_files_verified": bool(verify_files),
    }


def load_manifest(path: str | Path, *, verify_files: bool = False) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PairV5RolloutManifestError("manifest must be a plain file")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairV5RolloutManifestError(f"cannot decode manifest: {error}") from error
    return validate_manifest(payload, verify_files=verify_files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--verify-evidence-files", action="store_true")
    args = parser.parse_args(argv)
    result = load_manifest(args.manifest, verify_files=args.verify_evidence_files)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
