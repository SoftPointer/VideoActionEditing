#!/usr/bin/env python3
"""Build source-preserving complex-action preferences from eight score groups.

The pure-T2V videos calibrate only the frozen action critic against temporal
noop/reverse/incomplete controls.  They are never an endpoint.  Candidate
selection is lexicographic: first require every source-preservation gate, then
rank the remaining native RV2V candidates by candidate-side action reward.
No weighted action/appearance scalar is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-interaction-complex8-preference-manifest-v1"
REWARD_SCHEMA = "bernini-interaction-complex8-reward-group-v1"
PAIR_ROW_SCHEMA = "bernini-interaction-complex8-preference-row-v1"
ROLLOUT_BINDING_SCHEMA = "bernini-pair-v5-rollout-binding-v1"
FILE_BINDING_SCHEMA = "bernini-pair-v5-file-binding-v1"
MIN_GLOBAL_CRITIC_ACCURACY = 0.75
MIN_EVENT_CRITIC_ACCURACY = 0.50


class PreferenceBuildError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PreferenceBuildError(f"{label} must be an absolute plain JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreferenceBuildError(f"{label} root must be an object")
    return value


def verify_embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    declared = value.get("receipt_digest")
    if not isinstance(declared, str) or len(declared) != 64:
        raise PreferenceBuildError(f"{label} receipt digest is absent")
    unsigned = dict(value)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise PreferenceBuildError(f"{label} embedded digest differs")
    return declared


def file_binding(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PreferenceBuildError(f"bound file is not absolute/plain: {path}")
    return {
        "schema_version": FILE_BINDING_SCHEMA,
        "path": str(path),
        "sha256": file_sha256(path),
    }


def rollout_binding(rollout_root: Path, candidate_id: str) -> dict[str, Any]:
    receipt_path = rollout_root / candidate_id / "pair-v5-rollout-receipt.json"
    receipt = read_json(receipt_path, label=f"{candidate_id} rollout receipt")
    receipt_digest = verify_embedded_digest(receipt, label=f"{candidate_id} rollout receipt")
    candidate = receipt.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("candidate_id") != candidate_id:
        raise PreferenceBuildError(f"{candidate_id} receipt candidate differs")
    return {
        "schema_version": ROLLOUT_BINDING_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_digest": object_sha256(candidate),
        "receipt": file_binding(receipt_path),
        "expected_receipt_digest": receipt_digest,
    }


def finite_reward(row: Mapping[str, Any]) -> float:
    score = row.get("score")
    if not isinstance(score, Mapping):
        raise PreferenceBuildError("candidate action score is absent")
    value = score.get("phase_conjunctive_reward")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PreferenceBuildError("candidate action reward is non-finite")
    return float(value)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reward-root", required=True)
    result.add_argument("--rollout-root", required=True)
    result.add_argument("--output", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    reward_root = Path(args.reward_root).resolve(strict=True)
    rollout_root = Path(args.rollout_root).resolve(strict=True)
    output = Path(args.output)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise PreferenceBuildError("output must be a fresh absolute non-root file")

    rewards: list[tuple[Path, dict[str, Any]]] = []
    for group_index in range(8):
        path = reward_root / f"group_{group_index}" / "reward-group.json"
        value = read_json(path, label=f"reward group {group_index}")
        if value.get("schema_version") != REWARD_SCHEMA or value.get("complete") is not True:
            raise PreferenceBuildError(f"reward group {group_index} schema/completion differs")
        verify_embedded_digest(value, label=f"reward group {group_index}")
        closure = value.get("input_closure")
        if not isinstance(closure, Mapping) or any(
            closure.get(key) is not expected
            for key, expected in {
                "t2v_anchor_used_as_candidate_generation_input": False,
                "t2v_anchor_used_as_candidate_training_target": False,
                "t2v_anchor_appearance_used_by_candidate_scorer": False,
                "action_scorer_candidate_own_clean_and_noise_only": True,
                "preservation_scorer_source_and_candidate_rgb_only": True,
                "qwen_or_vlm_used": False,
                "training_performed": False,
            }.items()
        ):
            raise PreferenceBuildError(f"reward group {group_index} input closure differs")
        rewards.append((path, value))

    calibration_rows: list[Mapping[str, Any]] = []
    action_by_candidate: dict[str, Mapping[str, Any]] = {}
    preservation_by_candidate: dict[str, Mapping[str, Any]] = {}
    for _, reward in rewards:
        validation = reward.get("anchor_action_validation")
        if not isinstance(validation, Mapping) or not isinstance(validation.get("rows"), list):
            raise PreferenceBuildError("anchor validation rows are absent")
        calibration_rows.extend(validation["rows"])
        for row in reward.get("candidate_action_scores", ()):
            candidate_id = row.get("candidate_id") if isinstance(row, Mapping) else None
            if not isinstance(candidate_id, str) or candidate_id in action_by_candidate:
                raise PreferenceBuildError("candidate action score ID is invalid/duplicated")
            finite_reward(row)
            action_by_candidate[candidate_id] = row
        for row in reward.get("candidate_preservation", ()):
            candidate_id = row.get("candidate_id") if isinstance(row, Mapping) else None
            if not isinstance(candidate_id, str) or candidate_id in preservation_by_candidate:
                raise PreferenceBuildError("candidate preservation ID is invalid/duplicated")
            if not isinstance(row.get("hard_gate_checks"), Mapping):
                raise PreferenceBuildError("candidate preservation checks are absent")
            preservation_by_candidate[candidate_id] = row

    if len(calibration_rows) != 32 or len(action_by_candidate) != 32 or set(action_by_candidate) != set(preservation_by_candidate):
        raise PreferenceBuildError("complex8 calibration/candidate cardinality differs")
    per_event_calibration: dict[int, list[bool]] = {index: [] for index in range(8)}
    for row in calibration_rows:
        event = row.get("event_ordinal")
        comparisons = row.get("pairwise_pass")
        if type(event) is not int or event not in per_event_calibration or not isinstance(comparisons, Mapping):
            raise PreferenceBuildError("anchor validation event/comparison differs")
        if set(comparisons) != {"noop", "reverse", "incomplete"}:
            raise PreferenceBuildError("anchor negative comparison closure differs")
        per_event_calibration[event].extend(bool(comparisons[key]) for key in sorted(comparisons))
    if any(len(values) != 12 for values in per_event_calibration.values()):
        raise PreferenceBuildError("each event must have four anchors by three temporal negatives")
    global_passes = sum(sum(values) for values in per_event_calibration.values())
    global_total = sum(len(values) for values in per_event_calibration.values())
    global_accuracy = global_passes / global_total
    event_accuracy = {
        str(event): sum(values) / len(values)
        for event, values in per_event_calibration.items()
    }
    if global_accuracy < MIN_GLOBAL_CRITIC_ACCURACY or any(
        value < MIN_EVENT_CRITIC_ACCURACY for value in event_accuracy.values()
    ):
        raise PreferenceBuildError(
            "frozen action critic failed preregistered noop/reverse/incomplete calibration: "
            f"global={global_accuracy:.4f}, per_event={event_accuracy}"
        )

    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for event in range(8):
        candidates = sorted(
            (
                candidate_id
                for candidate_id, row in action_by_candidate.items()
                if row.get("event_ordinal") == event
            )
        )
        if len(candidates) != 4:
            raise PreferenceBuildError(f"event {event} candidate count differs")
        feasible = [
            candidate_id
            for candidate_id in candidates
            if preservation_by_candidate[candidate_id].get("hard_gate_pass") is True
            and all(
                value is True
                for value in preservation_by_candidate[candidate_id]["hard_gate_checks"].values()
            )
        ]
        ranked = sorted(feasible, key=lambda item: (finite_reward(action_by_candidate[item]), item))
        if len(ranked) < 2:
            skipped.append(
                {
                    "event_ordinal": event,
                    "reason": "fewer_than_two_source_preserving_candidates",
                    "candidate_ids": candidates,
                    "feasible_candidate_ids": feasible,
                }
            )
            continue
        rejected_id, chosen_id = ranked[0], ranked[-1]
        chosen_reward = finite_reward(action_by_candidate[chosen_id])
        rejected_reward = finite_reward(action_by_candidate[rejected_id])
        if not chosen_reward > rejected_reward:
            skipped.append(
                {
                    "event_ordinal": event,
                    "reason": "no_strict_candidate_action_margin",
                    "feasible_candidate_ids": feasible,
                    "reward": chosen_reward,
                }
            )
            continue
        chosen_receipt = read_json(
            rollout_root / chosen_id / "pair-v5-rollout-receipt.json",
            label=f"{chosen_id} rollout receipt",
        )
        candidate = chosen_receipt["candidate"]
        source = Path(candidate["source_video"])
        if file_sha256(source) != candidate["source_video_sha256"]:
            raise PreferenceBuildError(f"event {event} source bytes differ")
        row_unsigned = {
            "schema_version": PAIR_ROW_SCHEMA,
            "pair_id": f"complex8-event-{event:02d}",
            "event_ordinal": event,
            "source_video": file_binding(source),
            "complete_caption": candidate["complete_caption"],
            "complete_caption_sha256": candidate["complete_caption_sha256"],
            "chosen_rollout": rollout_binding(rollout_root, chosen_id),
            "rejected_rollout": rollout_binding(rollout_root, rejected_id),
            "selection": {
                "chosen_action_reward": chosen_reward,
                "rejected_action_reward": rejected_reward,
                "strict_action_margin": chosen_reward - rejected_reward,
                "both_endpoints_pass_all_source_preservation_gates": True,
                "weighted_action_appearance_score_used": False,
                "feasible_candidate_ids": feasible,
                "all_candidate_ids": candidates,
            },
            "sample_weight": 1.0,
        }
        pairs.append({**row_unsigned, "pair_digest": object_sha256(row_unsigned)})

    if len(pairs) < 6:
        raise PreferenceBuildError(
            f"only {len(pairs)}/8 events yielded strict source-preserving preferences; "
            "training is blocked below the preregistered six-event floor"
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "manifest_id": "interaction-complex8-source-gated-action-preferences-v1",
        "reward_groups": [
            {"path": str(path), "sha256": file_sha256(path), "receipt_digest": value["receipt_digest"]}
            for path, value in rewards
        ],
        "action_critic_validation": {
            "negative_roles": ["noop", "reverse", "incomplete"],
            "pairwise_pass_count": global_passes,
            "pairwise_comparison_count": global_total,
            "pairwise_accuracy": global_accuracy,
            "per_event_accuracy": event_accuracy,
            "minimum_global_accuracy": MIN_GLOBAL_CRITIC_ACCURACY,
            "minimum_event_accuracy": MIN_EVENT_CRITIC_ACCURACY,
            "passed": True,
        },
        "selection_policy": {
            "stage_1": "hard_require_all_source_preservation_checks",
            "stage_2": "rank_feasible_candidates_by_candidate_own_phase_conjunctive_action_reward",
            "chosen": "highest_feasible_action_reward",
            "rejected": "lowest_feasible_action_reward",
            "both_endpoints_must_preserve_source": True,
            "minimum_training_events": 6,
            "weighted_action_appearance_score_used": False,
        },
        "pairs": pairs,
        "skipped_events": skipped,
        "input_closure": {
            "pure_t2v_anchor_used_for_critic_calibration_only": True,
            "pure_t2v_anchor_used_as_training_endpoint": False,
            "pure_t2v_anchor_appearance_enters_candidate_selection": False,
            "native_rv2v_candidate_latents_are_dpo_endpoints": True,
            "source_video_is_visual_condition_and_preservation_authority": True,
            "qwen_or_vlm_used": False,
        },
    }
    manifest = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(manifest) + b"\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "pairs": len(pairs),
                "skipped": len(skipped),
                "critic_accuracy": global_accuracy,
                "manifest_digest": manifest["manifest_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
