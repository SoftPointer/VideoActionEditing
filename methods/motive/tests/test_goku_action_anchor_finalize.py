from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from motive import goku_action_anchor_qwen as qwen_module
from motive.goku_action_anchor_finalize import (
    DONE_NAME,
    FAMILY_QUOTAS,
    GENERATION_NAME,
    MAX_PER_TARGET_VERB,
    PROPOSED_NAME,
    RESERVE_NAME,
    REVIEW_NAME,
    SCALE512_DONE_SCHEMA,
    SCALE512_FAMILY_QUOTAS,
    SCALE512_GENERATION_SCHEMA,
    SCALE512_MAX_PER_TARGET_VERB,
    SCALE512_POLICY_VERSION,
    SCALE512_PROFILE,
    SCALE512_PROPOSED_NAME,
    SCALE512_PROPOSED_SIZE,
    SCALE512_RESERVE_NAME,
    SCALE512_RESERVE_SIZE,
    SCALE512_REVIEW_LIMIT,
    SCALE512_ROW_SCHEMA,
    SCALE512_SUMMARY_SCHEMA,
    SUMMARY_NAME,
    TEMPORAL_GEOMETRY_SCHEMA,
    GokuActionAnchorFinalizeError,
    _frozen_run_config,
    _iid_shard,
    _object_digest,
    build_parser,
    finalize_action_anchors,
)
from motive.goku_action_anchor_qwen import (
    ANCHOR_COMPATIBILITY_SCHEMA,
    ANCHOR_OBSERVATION_SCHEMA,
    DRAFT_CONTINUITY_SCHEMA,
    SHARD_RECEIPT_SCHEMA,
    TARGET_ADMISSIBILITY_SCHEMA,
    aggregate_draft_continuity,
    aggregate_target_admissibility,
    deterministic_risk_codes,
    qwen_provenance_digest,
    qwen_result_payload,
)


ROOT = Path(__file__).resolve().parents[1]
PREFILTER_SBATCH = (
    ROOT / "scripts" / "auh_goku_action_anchor_prefilter.sbatch"
)
QWEN_SBATCH = ROOT / "scripts" / "auh_goku_action_anchor_qwen.sbatch"
SUBMIT_SCRIPT = (
    ROOT / "scripts" / "auh_submit_goku_action_anchor_curate.sh"
)


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _selected_row(
    index: int,
    *,
    category: str,
    score: float,
    target_verb_override: str | None = None,
) -> dict[str, object]:
    iid = f"anchor-{index:04d}"
    target_verb = (
        target_verb_override
        if target_verb_override is not None
        else f"{category}_verb_{index % 4}"
    )
    return {
        "iid": iid,
        "group_id": f"group-{index:04d}",
        "family": f"source_family_{category}",
        "src_video": f"videos/{iid}/source.mp4",
        "resolved_src_video": f"/frozen/videos/{iid}/source.mp4",
        "source_caption": f"A subject performs source action {index}.",
        "edited_caption": f"The subject performs {target_verb}.",
        "prompt": f"Have the subject perform {target_verb}.",
        "anchor_image": f"anchors/{iid}.png",
        "resolved_anchor_image": f"/frozen/anchors/{iid}.png",
        "anchor_sha256": hashlib.sha256(
            f"anchor-{index}".encode()
        ).hexdigest(),
        "source_video_sha256": hashlib.sha256(
            f"video-{index}".encode()
        ).hexdigest(),
        "prefilter_score": score,
        "media": {
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "frame_count": 81,
            "duration_seconds": 3.2,
        },
        "motion": {"dynamic_score": 0.8},
        "actor_motion": {"foreground_motion": 0.7},
    }


def _anchor_observation(
    *,
    dynamics: str = "strong",
) -> dict[str, object]:
    return {
        "schema_version": ANCHOR_OBSERVATION_SCHEMA,
        "source_quality": "high",
        "resolution_quality": "high",
        "initial_state_clarity": "clear",
        "subject_visibility": "clear",
        "initial_state": "The subject is upright beside a visible object.",
        "visible_entities": ["subject", "nearby object"],
        "interaction_affordances": ["The object is within reach."],
        "source_action": "The subject moves forward continuously.",
        "actor_motion": "clear",
        "motion_dynamics": dynamics,
        "camera_motion": "none",
        "background_motion": "none",
        "single_continuous_shot": "yes",
        "artifact_level": "none",
        "temporal_evidence": [
            "Ordered source frames show sustained subject displacement."
        ],
        "uncertainty_codes": [],
    }


def _compatibility(
    *,
    index: int,
    category: str,
    target_action: str | None = None,
    target_verb_override: str | None = None,
) -> dict[str, object]:
    target_verb = (
        target_verb_override
        if target_verb_override is not None
        else f"{category}_verb_{index % 4}"
    )
    normalized_target = (
        target_action
        if target_action is not None
        else f"perform {target_verb}"
    )
    return {
        "schema_version": ANCHOR_COMPATIBILITY_SCHEMA,
        "decision": "accept",
        "anchor_compatibility": "compatible",
        "caption_consistency": "consistent",
        "source_action_normalized": "move forward continuously",
        "target_action_normalized": normalized_target,
        "target_action_verb": target_verb,
        "action_change_substantive": "yes",
        "action_category": category,
        "required_entities": [],
        "prerequisites_visible_at_i0": "yes",
        "target_presupposes_prior_action": "no",
        "causal_bridge": "direct",
        "causal_bridge_description": (
            "From the visible pose, the subject begins to "
            f"{normalized_target}."
        ),
        "causal_stages": [
            f"From the visible pose, {normalized_target}."
        ],
        "complete_within_clip": "yes",
        "rewritten_edit_instruction": (
            f"Have the subject {normalized_target} from its visible pose."
        ),
        "absolute_target_prompt": (
            f"The same subject starts from the shown pose and then "
            f"{normalized_target}; preserve scene, identity, and fixed "
            "camera."
        ),
        "preservation_constraints": [
            "Preserve identity, appearance, background, and camera."
        ],
        "unrequested_changes": [],
        "reason_codes": [],
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _target_admissibility(
    row: dict[str, object],
    observation: dict[str, object],
    *,
    compatibility: dict[str, object],
    target_action: str | None = None,
    target_verb_override: str | None = None,
    confidence: str = "high",
) -> dict[str, object]:
    del row, observation
    normalized_target = (
        target_action
        if target_action is not None
        else str(compatibility["target_action_normalized"])
    )
    target_verb = (
        target_verb_override
        if target_verb_override is not None
        else str(compatibility["target_action_verb"])
    )
    return {
        "schema_version": TARGET_ADMISSIBILITY_SCHEMA,
        "target_change_class": "new_articulated_action",
        "source_target_relation": "novel_future",
        "target_action_normalized": normalized_target,
        "target_action_verb": target_verb,
        "target_already_true": "no",
        "target_start_state_visually_verifiable": "yes",
        "prerequisite_grounded": "yes",
        "novel_trajectory": "yes",
        "novel_trajectory_description": normalized_target,
        "scalar_or_endpoint_only": "no",
        "source_evidence_ref": "source_action",
        "target_evidence_ref": "instruction",
        "uncertainty_codes": [],
        "confidence": confidence,
    }


def _draft_continuity(
    compatibility: dict[str, object],
    *,
    confidence: str = "high",
) -> dict[str, object]:
    del compatibility
    return {
        "schema_version": DRAFT_CONTINUITY_SCHEMA,
        "continuity_mode": "clean_direct",
        "target_dominance": "dominant",
        "actor_entity_consistency": "consistent",
        "direction_state_consistency": "consistent",
        "unrequested_action": "none",
        "source_replay_ref": "none",
        "target_support_ref": "rewritten_edit_instruction",
        "uncertainty_codes": [],
        "confidence": confidence,
    }


def _qwen_row(
    row: dict[str, object],
    *,
    index: int,
    category: str,
    execution_manifest: Path,
    execution_manifest_sha256: str,
    config_digest: str,
    run_config_digest: str,
    implementation_digest: str,
    judge_a_confidence: str = "high",
    judge_b_confidence: str = "high",
    validated_from: str = "original",
    dynamics: str = "strong",
    judge_a_target_action: str | None = None,
    judge_a_target_verb: str | None = None,
    writer_target_action: str | None = None,
    writer_target_verb: str | None = None,
) -> dict[str, object]:
    iid = str(row["iid"])
    shard = _iid_shard(iid)
    observation = _anchor_observation(dynamics=dynamics)
    compatibility = _compatibility(
        index=index,
        category=category,
        target_action=writer_target_action,
        target_verb_override=writer_target_verb,
    )
    judge_a = _target_admissibility(
        row,
        observation,
        compatibility=compatibility,
        target_action=judge_a_target_action,
        target_verb_override=judge_a_target_verb,
        confidence=judge_a_confidence,
    )
    aggregate_a = aggregate_target_admissibility(
        judge_a,
        row=row,
        observation=observation,
    )
    judge_b = _draft_continuity(
        compatibility,
        confidence=judge_b_confidence,
    )
    aggregate_b = aggregate_draft_continuity(
        judge_b,
        compatibility=compatibility,
        observation=observation,
    )
    risks = deterministic_risk_codes(
        judge_a,
        judge_b,
        row=row,
        observation=observation,
        compatibility=compatibility,
    )
    compatibility_prompt = qwen_module.build_compatibility_prompt(
        row=row,
        observation=observation,
        judge_a=judge_a,
    )
    judge_a_prompt = qwen_module.build_target_admissibility_prompt(
        row=row,
        observation=observation,
    )
    judge_b_prompt = qwen_module.build_draft_continuity_prompt(
        row=row,
        observation=observation,
        judge_a=judge_a,
        compatibility=compatibility,
    )
    visual_input_digest = hashlib.sha256(
        f"visual-input-{index}".encode()
    ).hexdigest()
    record = {
        "iid": iid,
        "group_id": row["group_id"],
        "family": row["family"],
        "status": "ok",
        "input_digest": _object_digest(row),
        "config_digest": config_digest,
        "run_config_digest": run_config_digest,
        "implementation_digest": implementation_digest,
        "execution_manifest": str(execution_manifest.resolve(strict=True)),
        "execution_manifest_sha256": execution_manifest_sha256,
        "model_path": "/frozen/models/test-qwen",
        "model_revision": "test-qwen-revision",
        "transformers_version": "test-transformers-version",
        "shard_index": shard,
        "num_shards": 8,
        "resolved_src_video": row["resolved_src_video"],
        "resolved_anchor_image": row["resolved_anchor_image"],
        "media_verification": {
            "exact_i0": True,
            "lossless_png": True,
            "width": row["media"]["width"],
            "height": row["media"]["height"],
            "anchor_sha256": row["anchor_sha256"],
            "source_video_sha256": row["source_video_sha256"],
            "frame_zero_rgb_sha256": hashlib.sha256(
                f"frame-zero-{index}".encode()
            ).hexdigest(),
        },
        "visual_input_digest": visual_input_digest,
        "anchor_observation": observation,
        "anchor_observation_digest": _object_digest(observation),
        "anchor_observation_raw": json.dumps(observation),
        "anchor_observation_failure_stage": None,
        "anchor_observation_validated_from": validated_from,
        "anchor_observation_repairs": (
            (
                []
                if validated_from == "original"
                else [{"attempt": 1, "status": "ok"}]
            )
        ),
        "target_admissibility_raw": json.dumps(judge_a),
        "target_admissibility_prompt_digest": hashlib.sha256(
            (
                qwen_module.JUDGE_A_SYSTEM + "\n" + judge_a_prompt
            ).encode("utf-8")
        ).hexdigest(),
        "target_admissibility_visual_input_digest": visual_input_digest,
        "target_admissibility": judge_a,
        "target_admissibility_resolved_evidence": (
            qwen_module.resolve_target_admissibility_evidence(
                judge_a,
                row=row,
                observation=observation,
            )
        ),
        "target_admissibility_validated_from": "original",
        "target_admissibility_repairs": [],
        "target_admissibility_aggregate": aggregate_a,
        "target_admissibility_failure_stage": None,
        "compatibility_raw": json.dumps(compatibility),
        "compatibility": compatibility,
        "compatibility_prompt_digest": hashlib.sha256(
            (
                qwen_module.COMPATIBILITY_SYSTEM
                + "\n"
                + compatibility_prompt
            ).encode("utf-8")
        ).hexdigest(),
        "compatibility_initial_validated_from": "original",
        "compatibility_validated_from": "original",
        "compatibility_repairs": [],
        "compatibility_semantic_repairs": [],
        "compatibility_failure_stage": None,
        "draft_continuity_raw": json.dumps(judge_b),
        "draft_continuity_prompt_digest": hashlib.sha256(
            (
                qwen_module.JUDGE_B_SYSTEM
                + "\n"
                + judge_b_prompt
            ).encode("utf-8")
        ).hexdigest(),
        "draft_continuity": judge_b,
        "draft_continuity_resolved_evidence": (
            qwen_module.resolve_draft_continuity_evidence(
                judge_b,
                compatibility=compatibility,
            )
        ),
        "draft_continuity_validated_from": "original",
        "draft_continuity_repairs": [],
        "draft_continuity_aggregate": aggregate_b,
        "draft_continuity_failure_stage": None,
        "deterministic_risk_codes": risks,
        "pipeline_stage": "judge_b",
        "pipeline_decision": aggregate_b["decision"],
        "failure_stage": None,
    }
    record["result_digest"] = _object_digest(qwen_result_payload(record))
    record["provenance_digest"] = qwen_provenance_digest(record)
    return record


def _write_fixture(
    root: Path,
    *,
    row_count: int = 208,
    repaired_index: int | None = None,
    weak_index: int | None = None,
    medium_index: int | None = None,
    first_prompt: str | None = None,
    first_judge_a_target_action: str | None = None,
    first_judge_a_target_verb: str | None = None,
    first_writer_target_action: str | None = None,
    first_writer_target_verb: str | None = None,
    category_counts: dict[str, int] | None = None,
    target_verbs_per_category: int = 4,
) -> tuple[Path, Path, list[dict[str, object]]]:
    categories = ("locomotion", "posture", "interaction", "articulated")
    if category_counts is not None:
        if tuple(category_counts) != categories:
            raise AssertionError("test category_counts keys/order differs")
        if sum(category_counts.values()) != row_count:
            raise AssertionError("test category_counts do not sum to rows")
        custom_categories = [
            category
            for category in categories
            for _ in range(category_counts[category])
        ]
    else:
        custom_categories = []
    if target_verbs_per_category <= 0:
        raise AssertionError("target_verbs_per_category must be positive")
    selected_rows: list[dict[str, object]] = []
    category_by_index: list[str] = []
    target_verb_by_index: list[str | None] = []
    category_occurrences: Counter[str] = Counter()
    # The first 192 are a balanced, higher-quality core: 48 per category.
    for index in range(row_count):
        if category_counts is not None:
            category = custom_categories[index]
            score = 1.0 - index / 10_000
        elif index < 192:
            category = categories[index // 48]
            score = 1.0 - index / 10_000
        else:
            category = categories[index % len(categories)]
            score = 0.1 - (index - 192) / 10_000
        category_by_index.append(category)
        target_verb_override = None
        if category_counts is not None or target_verbs_per_category != 4:
            target_verb_override = (
                f"{category}_verb_"
                f"{category_occurrences[category] % target_verbs_per_category}"
            )
        category_occurrences[category] += 1
        target_verb_by_index.append(target_verb_override)
        selected_row = _selected_row(
            index,
            category=category,
            score=score,
            target_verb_override=target_verb_override,
        )
        iid = str(selected_row["iid"])
        video_path = root / "frozen" / "videos" / iid / "source.mp4"
        anchor_path = root / "frozen" / "anchors" / f"{iid}.png"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"video-{index}".encode("utf-8"))
        anchor_path.write_bytes(f"anchor-{index}".encode("utf-8"))
        selected_row["resolved_src_video"] = str(video_path.resolve())
        selected_row["resolved_anchor_image"] = str(anchor_path.resolve())
        selected_row["source_video_sha256"] = hashlib.sha256(
            video_path.read_bytes()
        ).hexdigest()
        selected_row["anchor_sha256"] = hashlib.sha256(
            anchor_path.read_bytes()
        ).hexdigest()
        selected_row["media"] = {
            **dict(selected_row["media"]),
            "file_size_bytes": video_path.stat().st_size,
        }
        if index == 0 and first_prompt is not None:
            selected_row["prompt"] = first_prompt
        selected_rows.append(selected_row)

    selected = root / "selected.jsonl"
    selected.write_bytes(_jsonl_bytes(selected_rows))
    selected_sha256 = hashlib.sha256(selected.read_bytes()).hexdigest()
    qwen_root = root / "qwen"
    qwen_root.mkdir()
    implementation_digest = hashlib.sha256(
        Path(qwen_module.__file__).read_bytes()
    ).hexdigest()
    model_path = "/frozen/models/test-qwen"
    model_revision = "test-qwen-revision"
    transformers_version = "test-transformers-version"
    run_config = _frozen_run_config(
        model_path=model_path,
        model_revision=model_revision,
        transformers_version=transformers_version,
        implementation_digest=implementation_digest,
    )
    run_config_digest = _object_digest(run_config)
    config_digests = [
        _object_digest(
            {
                "run_config_digest": run_config_digest,
                "execution_manifest": str(selected.resolve(strict=True)),
                "execution_manifest_sha256": selected_sha256,
                "root": str(root.resolve(strict=True)),
                "shard_index": shard_index,
                "num_shards": 8,
            }
        )
        for shard_index in range(8)
    ]
    shards: list[list[dict[str, object]]] = [[] for _ in range(8)]
    for index, row in enumerate(selected_rows):
        shard = _iid_shard(str(row["iid"]))
        fixture_target_verb = target_verb_by_index[index]
        qwen = _qwen_row(
            row,
            index=index,
            category=category_by_index[index],
            execution_manifest=selected,
            execution_manifest_sha256=selected_sha256,
            config_digest=config_digests[shard],
            run_config_digest=run_config_digest,
            implementation_digest=implementation_digest,
            judge_a_confidence=(
                "medium" if index == medium_index else "high"
            ),
            judge_b_confidence=(
                "medium" if index == medium_index else "high"
            ),
            validated_from=(
                "repair_1" if index == repaired_index else "original"
            ),
            dynamics="weak" if index == weak_index else "strong",
            judge_a_target_action=(
                first_judge_a_target_action if index == 0 else None
            ),
            judge_a_target_verb=(
                first_judge_a_target_verb
                if index == 0 and first_judge_a_target_verb is not None
                else fixture_target_verb
            ),
            writer_target_action=(
                first_writer_target_action if index == 0 else None
            ),
            writer_target_verb=(
                first_writer_target_verb
                if index == 0 and first_writer_target_verb is not None
                else fixture_target_verb
            ),
        )
        shards[int(qwen["shard_index"])].append(qwen)
    for shard_index, rows in enumerate(shards):
        output = qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
        output.write_bytes(_jsonl_bytes(rows))
        output_raw = output.read_bytes()
        status_counts = dict(
            sorted(Counter(str(row["status"]) for row in rows).items())
        )
        receipt: dict[str, object] = {
            "schema_version": SHARD_RECEIPT_SCHEMA,
            "status": "complete",
            "execution_manifest": str(selected.resolve(strict=True)),
            "execution_manifest_sha256": selected_sha256,
            "root": str(root.resolve(strict=True)),
            "shard_index": shard_index,
            "num_shards": 8,
            "assigned_iids": qwen_module.assigned_iids_for_shard(
                selected_rows,
                shard_index=shard_index,
                num_shards=8,
                max_samples=None,
            ),
            "implementation_digest": implementation_digest,
            "config_digest": config_digests[shard_index],
            "run_config_digest": run_config_digest,
            "run_config": run_config,
            "model_path": model_path,
            "model_revision": model_revision,
            "transformers_version": transformers_version,
            "output": {
                "path": str(output.resolve(strict=True)),
                "sha256": hashlib.sha256(output_raw).hexdigest(),
                "bytes": len(output_raw),
                "rows": len(rows),
                "status_counts": status_counts,
            },
        }
        receipt["receipt_digest"] = _object_digest(receipt)
        qwen_module.shard_receipt_path(output).write_text(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    return selected, qwen_root, selected_rows


def _write_real_repair_fixture(
    root: Path,
    *,
    iid: str,
) -> tuple[Path, Path, list[dict[str, object]]]:
    """Produce one genuine aggregate-driven source-preface repair."""

    from methods.motive.tests.test_goku_action_anchor_qwen import (
        _SemanticRepairingFakeBackend,
        _input_row,
    )

    row = _input_row(root, iid=iid)
    row["media"] = {
        **dict(row["media"]),
        "fps": 6.0,
        "frame_count": 13,
        "duration_seconds": 2.0,
    }
    selected = root / "selected.jsonl"
    selected.write_bytes(_jsonl_bytes([row]))
    qwen_root = root / "qwen"
    qwen_root.mkdir()
    for shard_index in range(8):
        output = qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
        args = qwen_module.build_parser().parse_args(
            [
                "--input",
                str(selected),
                "--output",
                str(output),
                "--model",
                "fake/Qwen2.5-VL",
                "--root",
                str(root),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                "8",
                "--attn-implementation",
                "sdpa",
            ]
        )
        status = qwen_module.run_audit(
            args,
            backend_factory=_SemanticRepairingFakeBackend,
        )
        if status != 0:
            raise AssertionError(
                f"aggregate-driven repair fixture failed in shard "
                f"{shard_index}: status={status}"
            )
    return selected, qwen_root, [row]


def _refresh_shard_receipt(qwen_root: Path, shard_index: int) -> None:
    """Rebind a receipt after a test intentionally mutates shard rows."""

    output = qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
    receipt_path = qwen_module.shard_receipt_path(output)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw = output.read_bytes()
    rows = _read_jsonl(output)
    receipt["output"] = {
        "path": str(output.resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows": len(rows),
        "status_counts": dict(
            sorted(Counter(str(row["status"]) for row in rows).items())
        ),
    }
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = _object_digest(receipt)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _make_judge_a_reject(
    record: dict[str, object],
    selected_row: dict[str, object],
) -> None:
    observation = record["anchor_observation"]
    assert isinstance(observation, dict)
    judge_a = {
        "schema_version": TARGET_ADMISSIBILITY_SCHEMA,
        "target_change_class": "source_action_restatement",
        "source_target_relation": "repeats_source_future",
        "target_action_normalized": "move forward continuously",
        "target_action_verb": "move_forward",
        "target_already_true": "yes",
        "target_start_state_visually_verifiable": "yes",
        "prerequisite_grounded": "yes",
        "novel_trajectory": "no",
        "novel_trajectory_description": "none",
        "scalar_or_endpoint_only": "no",
        "source_evidence_ref": "source_action",
        "target_evidence_ref": "instruction",
        "uncertainty_codes": [],
        "confidence": "high",
    }
    aggregate_a = aggregate_target_admissibility(
        judge_a,
        row=selected_row,
        observation=observation,
    )
    record["target_admissibility"] = judge_a
    record["target_admissibility_raw"] = json.dumps(judge_a)
    record["target_admissibility_resolved_evidence"] = (
        qwen_module.resolve_target_admissibility_evidence(
            judge_a,
            row=selected_row,
            observation=observation,
        )
    )
    record["target_admissibility_aggregate"] = aggregate_a
    for field in (
        "compatibility",
        "compatibility_raw",
        "compatibility_prompt_digest",
        "compatibility_initial_validated_from",
        "compatibility_validated_from",
        "compatibility_failure_stage",
        "draft_continuity",
        "draft_continuity_resolved_evidence",
        "draft_continuity_raw",
        "draft_continuity_prompt_digest",
        "draft_continuity_validated_from",
        "draft_continuity_aggregate",
        "draft_continuity_failure_stage",
    ):
        record[field] = None
    record["compatibility_repairs"] = []
    record["compatibility_semantic_repairs"] = []
    record["draft_continuity_repairs"] = []
    record["deterministic_risk_codes"] = deterministic_risk_codes(
        judge_a,
        None,
        row=selected_row,
        observation=observation,
        compatibility=None,
    )
    record["pipeline_stage"] = "judge_a"
    record["pipeline_decision"] = "reject"
    record["result_digest"] = _object_digest(qwen_result_payload(record))
    record["provenance_digest"] = qwen_provenance_digest(record)


class GokuActionAnchorFinalizeTests(unittest.TestCase):
    def test_finalizer_has_no_legacy_iid_or_fixed_smoke_route(self) -> None:
        source = Path(finalize_action_anchors.__code__.co_filename).read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "V" + "12_SMOKE",
            "v" + "12_smoke",
            "is_" + "exact_" + "v" + "12",
            "EXPECTED_" + "DIRECT_IIDS",
            "EXPECTED_" + "REPAIR_IIDS",
            "EXPECTED_" + "NEGATIVE_PREFIXES",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotRegex(
            source.replace('"0123456789abcdef"', ""),
            r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])",
        )
        for forbidden in (
            '"generation_authorized": True',
            '"production_eligible": True',
            '"human_labels_asserted": True',
            '"manifest_role": "approved_generation"',
            "def _load_generation_approval",
        ):
            self.assertNotIn(forbidden, source)

    def test_generation_instruction_is_byte_exact_frozen_input(
        self,
    ) -> None:
        frozen_instruction = (
            "Have the subject perform locomotion_verb_0；保留“原始”外观。"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(
                root,
                row_count=1,
                first_prompt=frozen_instruction,
            )
            output = root / "instruction-binding"
            finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                allow_partial=True,
            )
            [generation] = _read_jsonl(output / GENERATION_NAME)
            self.assertEqual(rows[0]["prompt"], frozen_instruction)
            self.assertEqual(
                str(generation["edit_instruction"]).encode("utf-8"),
                frozen_instruction.encode("utf-8"),
            )
            self.assertEqual(
                generation["source_instruction_provenance"],
                generation["edit_instruction"],
            )
            self.assertEqual(
                generation["instruction_contract"],
                {
                    "sole_candidate_instruction_field": (
                        "edit_instruction"
                    ),
                    "candidate_instruction_source": (
                        "frozen_selected_prompt"
                    ),
                    "writer_proposal_payload_included": False,
                    "writer_proposals_executable": False,
                    "requires_future_signed_release_verifier": True,
                },
            )
            for forbidden in (
                "writer_rewritten_edit_instruction",
                "absolute_target_prompt",
                "absolute_target_prompt_role",
                "causal_bridge",
                "causal_stages",
                "required_entities",
                "preservation_constraints",
                "target_support_evidence",
                "writer_instruction_target_support_evidence",
                "judge_a_writer_target_core_agreement_evidence",
            ):
                self.assertNotIn(forbidden, generation)
            self.assertEqual(
                generation["target_semantics_source"],
                "judge_a_instruction_bound",
            )
            self.assertEqual(
                generation["source_edited_caption_provenance_role"],
                "non_executable_provenance",
            )
            self.assertIs(generation["generation_authorized"], False)
            self.assertIs(generation["production_eligible"], False)

    def test_partial_mode_keeps_all_hard_gates_and_allows_variable_size(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root, row_count=7)
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "required=160 available=7",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "strict",
                )

            output = root / "partial"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                allow_partial=True,
            )
            self.assertEqual(len(_read_jsonl(output / REVIEW_NAME)), 7)
            self.assertEqual(len(_read_jsonl(output / PROPOSED_NAME)), 7)
            self.assertEqual(len(_read_jsonl(output / GENERATION_NAME)), 7)
            self.assertEqual((output / RESERVE_NAME).read_bytes(), b"")
            self.assertEqual(summary["selection"]["mode"], "partial_up_to_128")
            self.assertTrue(summary["selection"]["allow_partial"])
            self.assertEqual(
                summary["selection"]["effective_proposed_target"],
                7,
            )
            self.assertEqual(
                summary["selection"]["effective_reserve_target"],
                0,
            )
            self.assertEqual(
                sum(
                    summary["selection"][
                        "effective_category_quotas"
                    ].values()
                ),
                7,
            )

    def test_partial_mode_still_rejects_zero_hard_pass_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(
                root,
                row_count=1,
                weak_index=0,
            )
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "requires at least one hard-pass",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "partial",
                    allow_partial=True,
                )

    def test_strict_selection_counts_quotas_and_generation_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, selected_rows = _write_fixture(root)
            output = root / "final"

            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
            )

            review = _read_jsonl(output / REVIEW_NAME)
            proposed = _read_jsonl(output / PROPOSED_NAME)
            reserve = _read_jsonl(output / RESERVE_NAME)
            generation = _read_jsonl(output / GENERATION_NAME)
            selected_by_iid = {
                str(row["iid"]): row for row in selected_rows
            }
            self.assertEqual(len(review), 192)
            self.assertEqual(len(proposed), 128)
            self.assertEqual(len(reserve), 32)
            self.assertEqual(len(generation), 128)
            proposed_iids = [str(row["iid"]) for row in proposed]
            self.assertEqual(
                proposed_iids,
                [str(row["iid"]) for row in generation],
            )
            self.assertTrue(
                set(proposed_iids).isdisjoint(
                    str(row["iid"]) for row in reserve
                )
            )
            self.assertEqual(
                Counter(
                    row["action_anchor_finalization"]["action_category"]
                    for row in proposed
                ),
                Counter(FAMILY_QUOTAS),
            )
            self.assertEqual(
                len({str(row["group_id"]) for row in review}),
                len(review),
            )
            verb_counts = Counter(
                row["action_anchor_finalization"]["target_action_verb"]
                for row in review
            )
            self.assertLessEqual(
                max(verb_counts.values()),
                MAX_PER_TARGET_VERB,
            )
            self.assertTrue(
                all(
                    row["human_review_status"] == "pending"
                    and row["generation_authorized"] is False
                    and row["manifest_role"] == "review_proposal"
                    and row["production_eligible"] is False
                    and row["approval"] is None
                    and row["edit_instruction"]
                    == selected_by_iid[str(row["iid"])]["prompt"]
                    and row["source_instruction_provenance"]
                    == row["edit_instruction"]
                    and row["instruction_contract"][
                        "sole_candidate_instruction_field"
                    ]
                    == "edit_instruction"
                    and row["instruction_contract"][
                        "candidate_instruction_source"
                    ]
                    == "frozen_selected_prompt"
                    and row["instruction_contract"][
                        "writer_proposal_payload_included"
                    ]
                    is False
                    and row["instruction_contract"][
                        "writer_proposals_executable"
                    ]
                    is False
                    and row["instruction_contract"][
                        "requires_future_signed_release_verifier"
                    ]
                    is True
                    and row["target_semantics_source"]
                    == "judge_a_instruction_bound"
                    and row["edit_instruction_sha256"]
                    == hashlib.sha256(
                        str(row["edit_instruction"]).encode("utf-8")
                    ).hexdigest()
                    for row in generation
                )
            )
            for row in generation:
                selected_media = selected_by_iid[str(row["iid"])]["media"]
                self.assertEqual(
                    row["selected_media_evidence"],
                    selected_media,
                )
                self.assertEqual(
                    row["selected_media_evidence_sha256"],
                    _object_digest(selected_media),
                )
                geometry = row["strict_temporal_geometry"]
                self.assertEqual(
                    geometry["schema_version"],
                    TEMPORAL_GEOMETRY_SCHEMA,
                )
                self.assertEqual(geometry["source_frame_count"], 81)
                self.assertEqual(
                    geometry["required_output_frame_count"],
                    81,
                )
                self.assertEqual(geometry["source_fps"], 25.0)
                self.assertEqual(
                    geometry["required_output_fps"],
                    25.0,
                )
                self.assertEqual(
                    geometry["source_duration_seconds"],
                    geometry["required_output_duration_seconds"],
                )
                self.assertEqual(
                    geometry["maximum_duration_delta_frames"],
                    1,
                )
                self.assertEqual(geometry["frame_count_form"], "4n+1")
                verification = row[
                    "finalizer_media_file_verification"
                ]
                selected_row = selected_by_iid[str(row["iid"])]
                self.assertEqual(
                    verification["source_video"]["sha256"],
                    selected_row["source_video_sha256"],
                )
                self.assertEqual(
                    verification["anchor_image"]["sha256"],
                    selected_row["anchor_sha256"],
                )
                self.assertFalse(
                    row["authorization_interface_available"]
                )
            self.assertTrue(
                all(
                    row["action_anchor_finalization"]["manifest_role"]
                    == "review_proposal"
                    and row["action_anchor_finalization"][
                        "production_eligible"
                    ]
                    is False
                    and row["action_anchor_finalization"]["approval"] is None
                    and row["action_anchor_finalization"][
                        "generation_authorized"
                    ]
                    is False
                    for row in (*review, *proposed, *reserve)
                )
            )
            self.assertTrue(
                all(
                    row["action_anchor_finalization"][
                        "writer_instruction_target_support_evidence"
                    ]["complete_instruction_target_contract"]
                    and row["action_anchor_finalization"][
                        "judge_a_writer_target_core_agreement_evidence"
                    ]["agreement_verified"]
                    for row in proposed
                )
            )
            done = json.loads(
                (output / DONE_NAME).read_text(encoding="utf-8")
            )
            self.assertNotIn("profile", summary)
            self.assertNotIn("profile", done)
            self.assertNotIn("profile_sha256", done)
            self.assertTrue(
                all(
                    "profile" not in row["action_anchor_finalization"]
                    for row in (*review, *proposed, *reserve)
                )
            )
            self.assertTrue(
                all(
                    "finalization_profile" not in row
                    and "policy_version" not in row
                    for row in generation
                )
            )
            for name in (
                REVIEW_NAME,
                PROPOSED_NAME,
                RESERVE_NAME,
                GENERATION_NAME,
                SUMMARY_NAME,
            ):
                self.assertEqual(
                    done["output_sha256"][name],
                    hashlib.sha256((output / name).read_bytes()).hexdigest(),
                )
            self.assertEqual(summary["hard_gate"]["passed_rows"], 208)
            self.assertFalse(
                summary["semantics"]["generation_authorized"]
            )
            self.assertFalse(summary["semantics"]["production_eligible"])
            self.assertIsNone(summary["semantics"]["approval"])
            self.assertFalse(
                summary["semantics"]["authorization_interface_available"]
            )
            self.assertEqual(
                summary["input"]["selected_media_files_reverified"],
                len(selected_rows),
            )

    def test_explicit_scale512_profile_is_closed_and_deterministic(
        self,
    ) -> None:
        self.assertGreaterEqual(
            SCALE512_REVIEW_LIMIT,
            SCALE512_PROPOSED_SIZE + SCALE512_RESERVE_SIZE,
        )
        category_counts = {
            "locomotion": 160,
            "posture": 160,
            "interaction": 240,
            "articulated": 80,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(
                root,
                row_count=640,
                category_counts=category_counts,
                target_verbs_per_category=5,
            )
            first = root / "scale512-first"
            second = root / "scale512-second"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=first,
                profile=SCALE512_PROFILE,
            )
            finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=second,
                profile=SCALE512_PROFILE,
            )

            expected_names = {
                REVIEW_NAME,
                SCALE512_PROPOSED_NAME,
                SCALE512_RESERVE_NAME,
                GENERATION_NAME,
                SUMMARY_NAME,
                DONE_NAME,
            }
            self.assertEqual(
                {path.name for path in first.iterdir()},
                expected_names,
            )
            self.assertFalse((first / PROPOSED_NAME).exists())
            self.assertFalse((first / RESERVE_NAME).exists())
            for name in expected_names:
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )

            review = _read_jsonl(first / REVIEW_NAME)
            proposed = _read_jsonl(first / SCALE512_PROPOSED_NAME)
            reserve = _read_jsonl(first / SCALE512_RESERVE_NAME)
            generation = _read_jsonl(first / GENERATION_NAME)
            self.assertEqual(len(review), 640)
            self.assertEqual(len(proposed), SCALE512_PROPOSED_SIZE)
            self.assertEqual(len(reserve), SCALE512_RESERVE_SIZE)
            self.assertEqual(len(generation), SCALE512_PROPOSED_SIZE)
            self.assertEqual(
                Counter(
                    row["action_anchor_finalization"]["action_category"]
                    for row in proposed
                ),
                Counter(SCALE512_FAMILY_QUOTAS),
            )
            self.assertTrue(
                {
                    str(row["iid"]) for row in proposed
                }.isdisjoint(str(row["iid"]) for row in reserve)
            )
            self.assertLessEqual(
                max(
                    Counter(
                        row["action_anchor_finalization"][
                            "target_action_verb"
                        ]
                        for row in review
                    ).values()
                ),
                SCALE512_MAX_PER_TARGET_VERB,
            )

            profile = summary["profile"]
            self.assertEqual(
                set(profile),
                {"schema_version", "name", "config", "config_sha256"},
            )
            self.assertEqual(profile["name"], SCALE512_PROFILE)
            self.assertEqual(
                profile["config_sha256"],
                _object_digest(profile["config"]),
            )
            config = profile["config"]
            self.assertEqual(config["required_qwen_shard_count"], 8)
            self.assertEqual(config["review_limit"], SCALE512_REVIEW_LIMIT)
            self.assertEqual(config["proposed_size"], 512)
            self.assertEqual(config["reserve_size"], 128)
            self.assertEqual(config["max_per_target_verb"], 48)
            self.assertEqual(
                config["category_quotas"],
                SCALE512_FAMILY_QUOTAS,
            )
            self.assertEqual(
                config["artifacts"]["proposed"],
                SCALE512_PROPOSED_NAME,
            )
            self.assertEqual(
                config["artifacts"]["reserve"],
                SCALE512_RESERVE_NAME,
            )
            self.assertEqual(summary["schema_version"], SCALE512_SUMMARY_SCHEMA)
            self.assertEqual(summary["policy_version"], SCALE512_POLICY_VERSION)
            self.assertEqual(summary["input"]["qwen_num_shards"], 8)
            self.assertEqual(summary["diversity"]["target_verb_max"], 48)
            self.assertEqual(
                summary["selection"]["mode"],
                "strict_512_plus_128",
            )
            self.assertEqual(
                summary["selection"]["requested_category_quotas"],
                SCALE512_FAMILY_QUOTAS,
            )
            self.assertEqual(
                summary["selection"]["effective_category_quotas"],
                SCALE512_FAMILY_QUOTAS,
            )
            self.assertTrue(
                all(
                    row["action_anchor_finalization"]["schema_version"]
                    == SCALE512_ROW_SCHEMA
                    and row["action_anchor_finalization"]["profile"]
                    == profile
                    for row in (*review, *proposed, *reserve)
                )
            )
            self.assertTrue(
                all(
                    row["schema_version"] == SCALE512_GENERATION_SCHEMA
                    and row["policy_version"] == SCALE512_POLICY_VERSION
                    and row["finalization_profile"] == profile
                    for row in generation
                )
            )

            done = json.loads(
                (first / DONE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(done["schema_version"], SCALE512_DONE_SCHEMA)
            self.assertEqual(done["profile"], profile)
            self.assertEqual(
                done["profile_sha256"],
                _object_digest(profile),
            )
            self.assertEqual(
                set(done["output_sha256"]),
                expected_names - {DONE_NAME},
            )
            for name in expected_names - {DONE_NAME}:
                self.assertEqual(
                    done["output_sha256"][name],
                    hashlib.sha256((first / name).read_bytes()).hexdigest(),
                )

    def test_scale512_partial_keeps_requested_and_actual_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root, row_count=7)
            output = root / "scale512-partial"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                profile=SCALE512_PROFILE,
                allow_partial=True,
            )
            selection = summary["selection"]
            self.assertEqual(selection["mode"], "partial_up_to_512")
            self.assertEqual(selection["requested_proposed_rows"], 512)
            self.assertEqual(selection["requested_reserve_rows"], 128)
            self.assertEqual(selection["effective_proposed_target"], 7)
            self.assertEqual(selection["effective_reserve_target"], 0)
            self.assertEqual(selection["proposed_rows"], 7)
            self.assertEqual(selection["reserve_rows"], 0)
            self.assertTrue((output / SCALE512_PROPOSED_NAME).is_file())
            self.assertEqual(
                (output / SCALE512_RESERVE_NAME).read_bytes(),
                b"",
            )

    def test_scale512_profile_must_be_explicit_and_known(self) -> None:
        required = [
            "--input",
            "selected.jsonl",
            "--qwen-root",
            "qwen",
            "--output-dir",
            "final",
        ]
        self.assertIsNone(build_parser().parse_args(required).profile)
        self.assertEqual(
            build_parser().parse_args(
                [*required, "--profile", SCALE512_PROFILE]
            ).profile,
            SCALE512_PROFILE,
        )
        with self.assertRaises(SystemExit):
            build_parser().parse_args([*required, "--profile", "scale640"])
        with self.assertRaisesRegex(
            GokuActionAnchorFinalizeError,
            "unsupported finalization profile",
        ):
            finalize_action_anchors(
                selected_path="not-read.jsonl",
                qwen_root="not-read-qwen",
                output_dir="not-written",
                profile="scale640",
            )

    def test_generic_repairs_fail_closed_and_low_dynamics_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(
                root,
                row_count=2,
                repaired_index=0,
            )
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "(?:generic repair provenance|v8 semantic provenance) differs",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    allow_partial=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(
                root,
                row_count=210,
                weak_index=209,
            )
            output = root / "final"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
            )
            all_output_iids = {
                str(row["iid"])
                for name in (REVIEW_NAME, PROPOSED_NAME, RESERVE_NAME)
                for row in _read_jsonl(output / name)
            }
            self.assertNotIn(str(rows[209]["iid"]), all_output_iids)
            rejection_counts = summary["hard_gate"]["rejection_counts"]
            self.assertEqual(
                rejection_counts["anchor:motion_dynamics"],
                1,
            )

    def test_approval_input_is_unconditionally_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root, row_count=1)
            approval_path = root / "approval.json"
            approval_path.write_text(
                json.dumps(
                    {
                        "reviewer_id": "self-asserted-reviewer",
                        "proposal_sha256": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "approval input is forbidden",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    approval_path=approval_path,
                    allow_partial=True,
                )
            self.assertFalse((root / "must-not-exist").exists())

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--input",
                    "selected.jsonl",
                    "--qwen-root",
                    "qwen",
                    "--output-dir",
                    "final",
                    "--approval",
                    "approval.json",
                ]
            )

    def test_source_restatement_is_quarantined_by_judge_a(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(root, row_count=2)
            iid = str(rows[0]["iid"])
            shard_path = (
                qwen_root / f"qwen_shard_{_iid_shard(iid):03d}.jsonl"
            )
            shard_rows = _read_jsonl(shard_path)
            record = next(row for row in shard_rows if row["iid"] == iid)
            _make_judge_a_reject(record, rows[0])
            shard_path.write_bytes(_jsonl_bytes(shard_rows))
            _refresh_shard_receipt(qwen_root, _iid_shard(iid))

            output = root / "judge-a-filtered"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                allow_partial=True,
            )
            self.assertEqual(summary["hard_gate"]["passed_rows"], 1)
            self.assertEqual(
                summary["hard_gate"]["rejection_counts"][
                    "judge_a:not_pass"
                ],
                1,
            )
            self.assertNotIn(
                iid,
                {
                    str(row["iid"])
                    for row in _read_jsonl(output / PROPOSED_NAME)
                },
            )

    def test_target_core_rejects_paraphrase_and_other_action(
        self,
    ) -> None:
        instruction = (
            "Make the dog pick up the nearby bone and then stand."
        )
        judge_target = "pick up the nearby bone and then stand"
        judge_verb = "pick_up_and_stand"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(
                root,
                row_count=1,
                first_prompt=instruction,
                first_judge_a_target_action=judge_target,
                first_judge_a_target_verb=judge_verb,
                first_writer_target_action=(
                    "lift the nearby bone then rise"
                ),
                first_writer_target_verb="lift_and_rise",
            )
            output = root / "paraphrase-rejected"
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "semantic provenance differs",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=output,
                    allow_partial=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(
                root,
                row_count=2,
                first_prompt=instruction,
                first_judge_a_target_action=judge_target,
                first_judge_a_target_verb=judge_verb,
                first_writer_target_action="run away",
                first_writer_target_verb="run_away",
            )
            output = root / "different-action-rejected"
            self.assertEqual(len(rows), 2)
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "semantic provenance differs",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=output,
                    allow_partial=True,
                )

    def test_writer_argument_omission_never_enters_generation_manifest(
        self,
    ) -> None:
        instruction = (
            "Make the grey dog run ahead of the brown dog and look back "
            "at it over its shoulder."
        )
        judge_target = (
            "grey dog overtake brown dog and look back over shoulder"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(
                root,
                row_count=1,
                first_prompt=instruction,
                first_judge_a_target_action=judge_target,
                first_judge_a_target_verb="overtake_and_look_back",
                first_writer_target_action=judge_target,
                first_writer_target_verb="overtake_and_look_back",
            )
            output = root / "writer-argument-omission"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                allow_partial=True,
            )
            self.assertEqual(summary["hard_gate"]["passed_rows"], 1)
            [generation] = _read_jsonl(output / GENERATION_NAME)
            self.assertEqual(generation["edit_instruction"], instruction)
            self.assertEqual(
                generation["target_action_normalized"],
                judge_target,
            )
            self.assertEqual(
                generation["target_action_verb"],
                "overtake_and_look_back",
            )
            self.assertEqual(
                generation["instruction_contract"][
                    "writer_proposal_payload_included"
                ],
                False,
            )
            self.assertNotIn("absolute_target_prompt", generation)
            self.assertNotIn(
                "writer_rewritten_edit_instruction",
                generation,
            )

    def test_judge_b_reject_is_status_ok_but_hard_gate_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(root, row_count=2)
            iid = str(rows[0]["iid"])
            shard_path = (
                qwen_root / f"qwen_shard_{_iid_shard(iid):03d}.jsonl"
            )
            shard_rows = _read_jsonl(shard_path)
            record = next(row for row in shard_rows if row["iid"] == iid)
            observation = record["anchor_observation"]
            compatibility = record["compatibility"]
            judge_b = record["draft_continuity"]
            assert isinstance(observation, dict)
            assert isinstance(compatibility, dict)
            assert isinstance(judge_b, dict)
            judge_b["continuity_mode"] = (
                "source_dominant_or_target_changed"
            )
            judge_b["target_dominance"] = "absent_or_changed"
            judge_b["source_replay_ref"] = "rewritten_edit_instruction"
            record["draft_continuity_raw"] = json.dumps(judge_b)
            record["draft_continuity_resolved_evidence"] = (
                qwen_module.resolve_draft_continuity_evidence(
                    judge_b,
                    compatibility=compatibility,
                )
            )
            aggregate_b = aggregate_draft_continuity(
                judge_b,
                compatibility=compatibility,
                observation=observation,
            )
            record["draft_continuity_aggregate"] = aggregate_b
            record["deterministic_risk_codes"] = deterministic_risk_codes(
                record["target_admissibility"],
                judge_b,
                row=rows[0],
                observation=observation,
                compatibility=compatibility,
            )
            record["pipeline_decision"] = "reject"
            record["result_digest"] = _object_digest(
                qwen_result_payload(record)
            )
            record["provenance_digest"] = qwen_provenance_digest(record)
            shard_path.write_bytes(_jsonl_bytes(shard_rows))
            _refresh_shard_receipt(qwen_root, _iid_shard(iid))

            output = root / "judge-b-filtered"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                allow_partial=True,
            )
            self.assertEqual(record["status"], "ok")
            self.assertEqual(summary["hard_gate"]["passed_rows"], 1)
            self.assertEqual(
                summary["hard_gate"]["rejection_counts"][
                    "judge_b:not_pass"
                ],
                1,
            )
            self.assertNotIn(
                iid,
                {
                    str(row["iid"])
                    for row in _read_jsonl(output / PROPOSED_NAME)
                },
            )

    def test_qwen_execution_manifest_is_bound_to_actual_selected_file(
        self,
    ) -> None:
        mutations = (
            (
                "execution_manifest_sha256",
                "b" * 64,
                "execution_manifest_sha256 differs",
            ),
            (
                "execution_manifest",
                "/frozen/not-the-selected-input.jsonl",
                "execution_manifest path differs",
            ),
        )
        for field, value, expected in mutations:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, rows = _write_fixture(
                        root,
                        row_count=2,
                    )
                    iid = str(rows[0]["iid"])
                    shard_path = (
                        qwen_root
                        / f"qwen_shard_{_iid_shard(iid):03d}.jsonl"
                    )
                    shard_rows = _read_jsonl(shard_path)
                    record = next(
                        row for row in shard_rows if row["iid"] == iid
                    )
                    record[field] = value
                    record["provenance_digest"] = qwen_provenance_digest(
                        record
                    )
                    shard_path.write_bytes(_jsonl_bytes(shard_rows))
                    _refresh_shard_receipt(
                        qwen_root,
                        _iid_shard(iid),
                    )
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        expected,
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / "must-not-exist",
                            allow_partial=True,
                        )

    def test_qwen_cross_shard_runtime_identity_is_uniform(self) -> None:
        mutations = (
            (
                "model_path",
                "/different/model",
                "exactly one model_path",
            ),
            (
                "model_revision",
                "different-revision",
                "exactly one model_revision",
            ),
            (
                "transformers_version",
                "different-transformers",
                "exactly one transformers_version",
            ),
            (
                "run_config_digest",
                "c" * 64,
                "exactly one run_config_digest",
            ),
            (
                "implementation_digest",
                "d" * 64,
                "implementation_digest differs",
            ),
            (
                "num_shards",
                7,
                "num_shards binding differs",
            ),
        )
        for field, value, expected in mutations:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, rows = _write_fixture(
                        root,
                        row_count=2,
                    )
                    iid = str(rows[0]["iid"])
                    shard_path = (
                        qwen_root
                        / f"qwen_shard_{_iid_shard(iid):03d}.jsonl"
                    )
                    shard_rows = _read_jsonl(shard_path)
                    record = next(
                        row for row in shard_rows if row["iid"] == iid
                    )
                    record[field] = value
                    record["provenance_digest"] = qwen_provenance_digest(
                        record
                    )
                    shard_path.write_bytes(_jsonl_bytes(shard_rows))
                    _refresh_shard_receipt(
                        qwen_root,
                        _iid_shard(iid),
                    )
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        expected,
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / "must-not-exist",
                            allow_partial=True,
                        )

    def test_qwen_generic_repair_audit_is_validated_against_initial_writer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(root, row_count=2)
            iid = str(rows[0]["iid"])
            shard_path = (
                qwen_root / f"qwen_shard_{_iid_shard(iid):03d}.jsonl"
            )
            shard_rows = _read_jsonl(shard_path)
            record = next(row for row in shard_rows if row["iid"] == iid)
            record["compatibility_initial_validated_from"] = "repair_1"
            record["compatibility_repairs"] = []
            record["provenance_digest"] = qwen_provenance_digest(record)
            shard_path.write_bytes(_jsonl_bytes(shard_rows))
            _refresh_shard_receipt(qwen_root, _iid_shard(iid))

            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "generic repair provenance differs",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    allow_partial=True,
                )

    def test_qwen_config_digest_is_uniform_within_each_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root, row_count=40)
            target_path: Path | None = None
            target_rows: list[dict[str, object]] = []
            for candidate in sorted(qwen_root.glob("qwen_shard_*.jsonl")):
                rows = _read_jsonl(candidate)
                if len(rows) >= 2:
                    target_path = candidate
                    target_rows = rows
                    break
            self.assertIsNotNone(target_path)
            target_rows[0]["config_digest"] = "d" * 64
            target_rows[0]["provenance_digest"] = qwen_provenance_digest(
                target_rows[0]
            )
            assert target_path is not None
            target_path.write_bytes(_jsonl_bytes(target_rows))
            target_index = int(target_path.stem.rsplit("_", 1)[1])
            _refresh_shard_receipt(qwen_root, target_index)
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "within a shard do not share one config_digest",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    allow_partial=True,
                )

    def test_receipt_full_frozen_run_config_rejects_every_mutation(
        self,
    ) -> None:
        cases = (
            ("nframes", lambda config: config.__setitem__("nframes", 13)),
            (
                "max_pixels",
                lambda config: config.__setitem__("max_pixels", 1),
            ),
            (
                "max_new_tokens",
                lambda config: config.__setitem__("max_new_tokens", 1),
            ),
            (
                "repair_attempts",
                lambda config: config.__setitem__("repair_attempts", 0),
            ),
            (
                "allow_download",
                lambda config: config.__setitem__(
                    "allow_download",
                    True,
                ),
            ),
            (
                "attn_implementation",
                lambda config: config.__setitem__(
                    "attn_implementation",
                    "auto",
                ),
            ),
            (
                "generation_do_sample",
                lambda config: config["generation"].__setitem__(
                    "do_sample",
                    True,
                ),
            ),
            (
                "generation_visual_input",
                lambda config: config["generation"].__setitem__(
                    "visual_input",
                    "mosaic_only",
                ),
            ),
            (
                "judge_a_prompt_digest",
                lambda config: config.__setitem__(
                    "judge_a_prompt_digest",
                    "0" * 64,
                ),
            ),
            (
                "repair_schema_digest",
                lambda config: config.__setitem__(
                    "repair_schema_digest",
                    "0" * 64,
                ),
            ),
            (
                "extra_key",
                lambda config: config.__setitem__("unexpected", True),
            ),
            (
                "missing_key",
                lambda config: config.pop("blind_prompt_digest"),
            ),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, _ = _write_fixture(
                        root,
                        row_count=2,
                    )
                    receipt_path = qwen_module.shard_receipt_path(
                        qwen_root / "qwen_shard_000.jsonl"
                    )
                    receipt = json.loads(
                        receipt_path.read_text(encoding="utf-8")
                    )
                    mutate(receipt["run_config"])
                    receipt["run_config_digest"] = _object_digest(
                        receipt["run_config"]
                    )
                    receipt["config_digest"] = _object_digest(
                        {
                            "run_config_digest": receipt[
                                "run_config_digest"
                            ],
                            "execution_manifest": receipt[
                                "execution_manifest"
                            ],
                            "execution_manifest_sha256": receipt[
                                "execution_manifest_sha256"
                            ],
                            "root": receipt["root"],
                            "shard_index": receipt["shard_index"],
                            "num_shards": receipt["num_shards"],
                        }
                    )
                    receipt.pop("receipt_digest")
                    receipt["receipt_digest"] = _object_digest(receipt)
                    receipt_path.write_text(
                        json.dumps(
                            receipt,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        "full run_config differs",
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / "must-not-exist",
                            allow_partial=True,
                        )

    def test_receipts_are_required_for_empty_shards_and_bind_row_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root, row_count=2)
            empty_shard = next(
                index
                for index in range(8)
                if not (
                    qwen_root / f"qwen_shard_{index:03d}.jsonl"
                ).read_bytes()
            )
            qwen_module.shard_receipt_path(
                qwen_root / f"qwen_shard_{empty_shard:03d}.jsonl"
            ).unlink()
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "Qwen receipt set mismatch",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    allow_partial=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root, row_count=40)
            shard_path = next(
                path
                for path in sorted(qwen_root.glob("qwen_shard_*.jsonl"))
                if len(_read_jsonl(path)) >= 2
            )
            shard_index = int(shard_path.stem.rsplit("_", 1)[1])
            rows = _read_jsonl(shard_path)
            rows[0], rows[1] = rows[1], rows[0]
            shard_path.write_bytes(_jsonl_bytes(rows))
            _refresh_shard_receipt(qwen_root, shard_index)
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "IID/order differs from receipt",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    allow_partial=True,
                )

    def test_v5_semantic_only_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(root, row_count=2)
            iid = str(rows[0]["iid"])
            shard_index = _iid_shard(iid)
            shard_path = (
                qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
            )
            shard_rows = _read_jsonl(shard_path)
            record = next(row for row in shard_rows if row["iid"] == iid)
            record.pop("target_admissibility")
            record["semantic_critic"] = {"verdict": "pass"}
            shard_path.write_bytes(_jsonl_bytes(shard_rows))
            _refresh_shard_receipt(qwen_root, shard_index)
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "target_admissibility must be an object",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    allow_partial=True,
                )

    def test_medium_a_and_b_enter_review_but_remain_unauthorized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(
                root,
                row_count=2,
                medium_index=0,
            )
            output = root / "medium-review"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                allow_partial=True,
            )
            iid = str(rows[0]["iid"])
            proposed = _read_jsonl(output / PROPOSED_NAME)
            generation = _read_jsonl(output / GENERATION_NAME)
            self.assertEqual(summary["hard_gate"]["passed_rows"], 2)
            self.assertIn(iid, {str(row["iid"]) for row in proposed})
            candidate = next(
                row for row in generation if str(row["iid"]) == iid
            )
            self.assertEqual(
                candidate["target_admissibility"]["confidence"],
                "medium",
            )
            self.assertEqual(
                candidate["draft_continuity"]["confidence"],
                "medium",
            )
            self.assertIs(candidate["generation_authorized"], False)
            self.assertIs(candidate["production_eligible"], False)
            self.assertIsNone(candidate["approval"])

    def test_real_repair_route_is_aggregate_driven_not_iid_driven(
        self,
    ) -> None:
        for iid in ("renamed-repair-alpha", "renamed-repair-omega"):
            with self.subTest(iid=iid):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, _ = _write_real_repair_fixture(
                        root,
                        iid=iid,
                    )
                    audited = [
                        row
                        for shard in sorted(
                            qwen_root.glob("qwen_shard_*.jsonl")
                        )
                        for row in _read_jsonl(shard)
                    ]
                    [record] = audited
                    self.assertEqual(
                        record["compatibility_validated_from"],
                        "semantic_repair_1",
                    )
                    self.assertEqual(
                        record["draft_continuity_aggregate"]["decision"],
                        "pass",
                    )

                    output = root / "repair-proposal"
                    summary = finalize_action_anchors(
                        selected_path=selected,
                        qwen_root=qwen_root,
                        output_dir=output,
                        allow_partial=True,
                    )
                    self.assertEqual(summary["hard_gate"]["passed_rows"], 1)
                    [generation] = _read_jsonl(
                        output / GENERATION_NAME
                    )
                    self.assertEqual(generation["iid"], iid)
                    self.assertIs(
                        generation["generation_authorized"],
                        False,
                    )
                    self.assertIs(
                        generation["production_eligible"],
                        False,
                    )

    def test_low_judge_b_confidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(
                root,
                row_count=2,
            )
            iid = str(rows[0]["iid"])
            shard_index = _iid_shard(iid)
            shard_path = qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
            shard_rows = _read_jsonl(shard_path)
            record = next(row for row in shard_rows if row["iid"] == iid)
            observation = record["anchor_observation"]
            compatibility = record["compatibility"]
            judge_b = record["draft_continuity"]
            assert isinstance(observation, dict)
            assert isinstance(compatibility, dict)
            assert isinstance(judge_b, dict)
            judge_b["confidence"] = "low"
            record["draft_continuity_raw"] = json.dumps(judge_b)
            record["draft_continuity_aggregate"] = (
                aggregate_draft_continuity(
                    judge_b,
                    compatibility=compatibility,
                    observation=observation,
                )
            )
            record["deterministic_risk_codes"] = deterministic_risk_codes(
                record["target_admissibility"],
                judge_b,
                row=rows[0],
                observation=observation,
                compatibility=compatibility,
            )
            record["pipeline_decision"] = "reject"
            record["result_digest"] = _object_digest(
                qwen_result_payload(record)
            )
            record["provenance_digest"] = qwen_provenance_digest(record)
            shard_path.write_bytes(_jsonl_bytes(shard_rows))
            _refresh_shard_receipt(qwen_root, shard_index)

            output = root / "low-confidence-filtered"
            summary = finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=output,
                allow_partial=True,
            )
            self.assertEqual(summary["hard_gate"]["passed_rows"], 1)
            self.assertEqual(
                summary["hard_gate"]["rejection_counts"][
                    "judge_b:confidence"
                ],
                1,
            )

    def test_normal_rows_recompute_all_v8_prompt_digests(
        self,
    ) -> None:
        cases = (
            (
                "target_admissibility_prompt_digest",
                "judge_a:prompt_digest",
            ),
            (
                "compatibility_prompt_digest",
                "compatibility:prompt_digest",
            ),
            (
                "draft_continuity_prompt_digest",
                "judge_b:prompt_digest",
            ),
        )
        for field, failure in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, rows = _write_fixture(
                        root,
                        row_count=2,
                    )
                    iid = str(rows[0]["iid"])
                    shard_index = _iid_shard(iid)
                    shard_path = (
                        qwen_root
                        / f"qwen_shard_{shard_index:03d}.jsonl"
                    )
                    shard_rows = _read_jsonl(shard_path)
                    record = next(
                        row for row in shard_rows if row["iid"] == iid
                    )
                    record[field] = "0" * 64
                    record["provenance_digest"] = qwen_provenance_digest(
                        record
                    )
                    shard_path.write_bytes(_jsonl_bytes(shard_rows))
                    _refresh_shard_receipt(qwen_root, shard_index)
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        failure,
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / "must-not-exist",
                            allow_partial=True,
                        )

    def test_selected_manifest_requires_resolved_anchor_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root, row_count=2)
            selected_rows = _read_jsonl(selected)
            del selected_rows[0]["resolved_anchor_image"]
            selected.write_bytes(_jsonl_bytes(selected_rows))
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "missing fields:.*resolved_anchor_image",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "must-not-exist",
                    allow_partial=True,
                )

    def test_selected_media_files_are_opened_and_rehashed_fail_closed(
        self,
    ) -> None:
        bindings = (
            ("resolved_src_video", "source_video", "source_video_sha256"),
            ("resolved_anchor_image", "anchor_image", "anchor_sha256"),
        )
        for path_field, label, digest_field in bindings:
            with self.subTest(path_field=path_field, mode="missing"):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, rows = _write_fixture(
                        root,
                        row_count=1,
                    )
                    path = Path(str(rows[0][path_field]))
                    path.rename(path.with_name(path.name + ".moved"))
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        f"{path_field} is missing",
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / "must-not-exist",
                            allow_partial=True,
                        )
                    self.assertFalse((root / "must-not-exist").exists())

            with self.subTest(path_field=path_field, mode="replaced"):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, rows = _write_fixture(
                        root,
                        row_count=1,
                    )
                    path = Path(str(rows[0][path_field]))
                    path.write_bytes(b"replacement-content")
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        f"{label} SHA-256 differs from {digest_field}",
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / "must-not-exist",
                            allow_partial=True,
                        )
                    self.assertFalse((root / "must-not-exist").exists())

    def test_selected_temporal_geometry_is_strict_and_fail_closed(
        self,
    ) -> None:
        cases = (
            (
                lambda media: media.pop("frame_count"),
                "missing temporal fields",
            ),
            (
                lambda media: media.__setitem__("frame_count", 80),
                "frame_count must satisfy 4n\\+1",
            ),
            (
                lambda media: media.__setitem__(
                    "duration_seconds",
                    30.0,
                ),
                "duration differs from its frame timeline by more than one frame",
            ),
            (
                lambda media: media.__setitem__("fps", 0.0),
                "fps must be a positive finite number",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, _ = _write_fixture(
                        root,
                        row_count=1,
                    )
                    rows = _read_jsonl(selected)
                    mutate(rows[0]["media"])
                    selected.write_bytes(_jsonl_bytes(rows))
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        message,
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / "must-not-exist",
                            allow_partial=True,
                        )
                    self.assertFalse((root / "must-not-exist").exists())

    def test_input_digest_and_shard_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, rows = _write_fixture(root)
            iid = str(rows[0]["iid"])
            shard_index = _iid_shard(iid)
            shard_path = qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
            shard_rows = _read_jsonl(shard_path)
            for row in shard_rows:
                if row["iid"] == iid:
                    row["input_digest"] = "f" * 64
                    break
            shard_path.write_bytes(_jsonl_bytes(shard_rows))
            _refresh_shard_receipt(qwen_root, shard_index)
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "input_digest binding differs",
            ):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=root / "bad-digest",
                )

            # Restore the fixture, then prove that a row cannot claim another
            # execution shard.
            second = root / "second"
            second.mkdir()
            selected2, qwen_root2, _ = _write_fixture(second)
            binding_path = next(
                path
                for path in sorted(qwen_root2.glob("qwen_shard_*.jsonl"))
                if len(_read_jsonl(path)) >= 2
            )
            bound_rows = _read_jsonl(binding_path)
            bound_rows[0]["shard_index"] = (
                int(bound_rows[0]["shard_index"]) + 1
            ) % 8
            binding_path.write_bytes(_jsonl_bytes(bound_rows))
            binding_index = int(binding_path.stem.rsplit("_", 1)[1])
            _refresh_shard_receipt(qwen_root2, binding_index)
            with self.assertRaisesRegex(
                GokuActionAnchorFinalizeError,
                "shard_index binding differs",
            ):
                finalize_action_anchors(
                    selected_path=selected2,
                    qwen_root=qwen_root2,
                    output_dir=second / "bad-order",
                )

    def test_qwen_media_code_and_visual_provenance_fail_closed(self) -> None:
        cases = {
            "implementation_digest": (
                lambda qwen: qwen.__setitem__(
                    "implementation_digest",
                    "f" * 64,
                ),
                "implementation_digest differs",
            ),
            "anchor_observation_digest": (
                lambda qwen: qwen.__setitem__(
                    "anchor_observation_digest",
                    "f" * 64,
                ),
                "anchor digest differs",
            ),
            "visual_input_digest": (
                lambda qwen: qwen.__setitem__(
                    "visual_input_digest",
                    "f" * 64,
                ),
                "Judge-A visual input digest differs",
            ),
            "media_exact_i0": (
                lambda qwen: qwen["media_verification"].__setitem__(
                    "exact_i0",
                    False,
                ),
                "exact lossless I0 binding is false",
            ),
            "media_anchor_hash": (
                lambda qwen: qwen["media_verification"].__setitem__(
                    "anchor_sha256",
                    "f" * 64,
                ),
                "media anchor SHA differs",
            ),
            "provenance_digest": (
                lambda qwen: qwen.__setitem__(
                    "provenance_digest",
                    "f" * 64,
                ),
                "provenance digest differs",
            ),
        }
        for name, (mutate, expected_error) in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, rows = _write_fixture(root)
                    iid = str(rows[0]["iid"])
                    shard_index = _iid_shard(iid)
                    shard_path = (
                        qwen_root
                        / f"qwen_shard_{shard_index:03d}.jsonl"
                    )
                    shard_rows = _read_jsonl(shard_path)
                    target = next(
                        row for row in shard_rows if row["iid"] == iid
                    )
                    mutate(target)
                    shard_path.write_bytes(_jsonl_bytes(shard_rows))
                    _refresh_shard_receipt(qwen_root, shard_index)
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        expected_error,
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / f"bad-{name}",
                        )

    def test_resolved_evidence_and_judge_a_visual_binding_resist_resigning(
        self,
    ) -> None:
        cases = (
            (
                "judge_a_visual",
                lambda row: row.__setitem__(
                    "target_admissibility_visual_input_digest",
                    "0" * 64,
                ),
                "Judge-A visual input digest differs",
            ),
            (
                "judge_a_evidence",
                lambda row: row[
                    "target_admissibility_resolved_evidence"
                ].__setitem__("target_evidence", "rewritten target"),
                "Judge-A resolved evidence differs",
            ),
            (
                "judge_b_evidence",
                lambda row: row[
                    "draft_continuity_resolved_evidence"
                ].__setitem__("target_support_evidence", "rewritten target"),
                "Judge-B resolved evidence differs",
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    selected, qwen_root, rows = _write_fixture(
                        root,
                        row_count=2,
                    )
                    iid = str(rows[0]["iid"])
                    shard_index = _iid_shard(iid)
                    shard_path = (
                        qwen_root
                        / f"qwen_shard_{shard_index:03d}.jsonl"
                    )
                    shard_rows = _read_jsonl(shard_path)
                    record = next(
                        row for row in shard_rows if row["iid"] == iid
                    )
                    mutate(record)
                    record["result_digest"] = _object_digest(
                        qwen_result_payload(record)
                    )
                    record["provenance_digest"] = qwen_provenance_digest(
                        record
                    )
                    shard_path.write_bytes(_jsonl_bytes(shard_rows))
                    _refresh_shard_receipt(qwen_root, shard_index)
                    with self.assertRaisesRegex(
                        GokuActionAnchorFinalizeError,
                        expected,
                    ):
                        finalize_action_anchors(
                            selected_path=selected,
                            qwen_root=qwen_root,
                            output_dir=root / f"bad-{name}",
                            allow_partial=True,
                        )

    def test_outputs_are_deterministic_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, qwen_root, _ = _write_fixture(root)
            first = root / "first"
            second = root / "second"
            finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=first,
            )
            finalize_action_anchors(
                selected_path=selected,
                qwen_root=qwen_root,
                output_dir=second,
            )
            for name in (
                REVIEW_NAME,
                PROPOSED_NAME,
                RESERVE_NAME,
                GENERATION_NAME,
                SUMMARY_NAME,
                DONE_NAME,
            ):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )
            with self.assertRaises(FileExistsError):
                finalize_action_anchors(
                    selected_path=selected,
                    qwen_root=qwen_root,
                    output_dir=first,
                )


class GokuActionAnchorOrchestrationTests(unittest.TestCase):
    def test_scripts_have_valid_bash_syntax(self) -> None:
        for script in (PREFILTER_SBATCH, QWEN_SBATCH, SUBMIT_SCRIPT):
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{script.name}: {completed.stderr}",
            )

    def test_packed_qwen_loads_two_dual4_backends_then_finalizes(
        self,
    ) -> None:
        prefilter_text = PREFILTER_SBATCH.read_text(encoding="utf-8")
        self.assertIn(
            "#SBATCH --gres=gpu:mi210:1",
            prefilter_text,
        )
        text = QWEN_SBATCH.read_text(encoding="utf-8")
        for marker in (
            "#SBATCH --ntasks=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=128",
            "#SBATCH --gres=gpu:mi210:8",
            "--gpus-per-task=4",
            "--gpu-bind=none",
            "-m motive.goku_action_anchor_qwen",
            "--all-shards-sequential",
            "--sequential-shards",
            "--shard-index 0",
            '--num-shards "${shard_count}"',
            'max_new_tokens="${MOTIVE_GOKU_ACTION_QWEN_MAX_NEW_TOKENS:-1536}"',
            '--max-new-tokens "${max_new_tokens}"',
            'allow_partial="${MOTIVE_GOKU_ACTION_ALLOW_PARTIAL:-0}"',
            'finalizer_args+=(--allow-partial)',
            'worker_a_cache="${cache_parent}/motive-goku-anchor-qwen-${SLURM_JOB_ID}-a"',
            'worker_b_cache="${cache_parent}/motive-goku-anchor-qwen-${SLURM_JOB_ID}-b"',
            'run_worker "${worker_a_cache}" "0,1,2,3" &',
            'run_worker "${worker_b_cache}" "4,5,6,7" &',
            "--allow-errors",
            "--resume",
            "-m motive.goku_action_anchor_finalize",
            "generation_manifest.jsonl",
        ):
            self.assertIn(marker, text)
        self.assertLess(
            text.index("-m motive.goku_action_anchor_qwen"),
            text.index("-m motive.goku_action_anchor_finalize"),
        )
        self.assertNotIn("#SBATCH --array", text)
        self.assertNotIn("#SBATCH --ntasks=8", text)
        self.assertNotIn("--gpus-per-task=8", text)
        self.assertNotIn("--gpus-per-task=1", text)
        self.assertNotIn("--gpu-bind=single:1", text)
        self.assertNotIn("SLURM_PROCID", text)
        self.assertNotIn("for shard", text)
        self.assertNotIn("sbatch ", text)

    def test_submitter_has_disconnect_safe_afterok_chain(self) -> None:
        text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(text.count("\n  sbatch \\"), 2)
        self.assertIn('--dependency="afterok:${prefilter_job}"', text)
        self.assertIn("--kill-on-invalid-dep=yes", text)
        self.assertIn(
            "auh_goku_action_anchor_prefilter.sbatch",
            text,
        )
        self.assertIn("auh_goku_action_anchor_qwen.sbatch", text)
        self.assertIn('retry_script="${run_root}/retry_qwen.sh"', text)
        self.assertIn(
            "MOTIVE_GOKU_ACTION_ALLOW_PARTIAL",
            text,
        )
        self.assertIn("qwen_retry_script=", text)

if __name__ == "__main__":
    unittest.main()
