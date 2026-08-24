from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_imagined_pareto_episode as contract  # noqa: E402
import source_caption_t2v_pass_a as native  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seal(value: dict, field: str) -> dict:
    value.pop(field, None)
    value[field] = contract.canonical_object_sha256(value)
    return value


def _bytes(value: dict) -> tuple[bytes, str]:
    payload = contract.canonical_json_bytes(value)
    return payload, hashlib.sha256(payload).hexdigest()


def _pass_a_receipt() -> tuple[dict, dict[str, dict[str, str]]]:
    entries = []
    artifacts: dict[str, dict[str, str]] = {}
    gaussian_by_seed: dict[str, str] = {}
    for seed_row in native.SEED_ROWS:
        gaussian = _sha(f"actual-pass-a-gaussian:{seed_row['seed']}")
        gaussian_by_seed[seed_row["seed_id"]] = gaussian
        for branch in native.BRANCH_ORDER:
            entry_id = f"{seed_row['seed_id']}-{branch.replace('_', '-')}"
            row = {
                "entry_id": entry_id,
                "seed_id": seed_row["seed_id"],
                "seed": seed_row["seed"],
                "execution_group": seed_row["execution_group"],
                "semantic_branch": branch,
                "native_receipt_path": f"/sealed/pass-a/{entry_id}/receipt.json",
                "native_receipt_sha256": _sha(f"native-receipt-file:{entry_id}"),
                "native_receipt_digest": _sha(f"native-receipt-object:{entry_id}"),
                "video_path": f"/sealed/pass-a/{entry_id}/output.mp4",
                "video_sha256": _sha(f"actual-video:{entry_id}"),
                "clean_latent_path": f"/sealed/pass-a/{entry_id}/clean.safetensors",
                "clean_latent_sha256": _sha(f"actual-clean-latent:{entry_id}"),
                "initial_gaussian_path": f"/sealed/pass-a/{entry_id}/noise.safetensors",
                "initial_gaussian_file_sha256": _sha(
                    f"gaussian-file:{seed_row['seed']}:{branch}"
                ),
                "initial_gaussian_value_sha256": gaussian,
                "initial_gaussian_independently_parsed": True,
                "pure_t2v_condition_audit_pass": True,
                "semantic_event_verified": False,
            }
            entries.append(row)
            artifacts[entry_id] = {
                field: row[field]
                for field in (
                    "native_receipt_sha256",
                    "native_receipt_digest",
                    "video_sha256",
                    "clean_latent_sha256",
                    "initial_gaussian_file_sha256",
                    "initial_gaussian_value_sha256",
                )
            }
    receipt = {
        "schema_version": native.BANK_RECEIPT_SCHEMA,
        "method": native.METHOD,
        "stage": "exact40-qualification-candidate",
        "manifest_path": "/sealed/pass-a/manifest.json",
        "manifest_file_sha256": _sha("actual-manifest-file"),
        "manifest_digest": _sha("actual-manifest-object"),
        "method_source_revision": "1" * 40,
        "method_source_archive_sha256": _sha("actual-method-archive"),
        "entry_count": 8,
        "seed_count": 2,
        "branch_count_per_seed": 4,
        "entries": entries,
        "initial_gaussian_contract": {
            "per_seed_value_sha256": gaussian_by_seed,
            "same_value_across_all_four_branches_within_seed": True,
            "different_values_across_the_two_seeds": True,
            "tensor_values_recomputed_from_safetensors": True,
            "posthoc_seed_selection": False,
        },
        "condition_closure": {
            "renderer_arm": "t2v",
            "guidance_mode": "t2v_apg",
            "source_video_role": "hash_verification_and_fixed_496x480_bucket_only",
            "source_pixels_forwarded_to_sampler": False,
            "source_video_latent_consumed": False,
            "source_reference_latent_consumed": False,
            "target_video_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "all_native_entry_condition_audits_pass": True,
        },
        "qualification": {
            "manifest_semantic_labels_are_not_event_acceptance": True,
            "semantic_events_verified": False,
            "exact40_manual_qualification_required": True,
            "qualification_unit": "complete_two_seed_by_four_branch_bank",
            "reject_pass_a_if_either_seed_or_any_branch_fails": True,
            "single_seed_or_branch_selection_forbidden": True,
            "reward_or_training_use_authorized": False,
            "pass_a_status": "pending_independent_manual_qualification",
        },
        "interpretation": {
            "render_complete": True,
            "pure_t2v_action_proposal_bank": True,
            "editing_result": False,
            "quality_claim": False,
            "model_training_performed": False,
            "scientific_claim_authorized": False,
        },
    }
    return _seal(receipt, "receipt_digest"), artifacts


def _expected_events(branch: str) -> dict[str, bool]:
    if branch == "full_action":
        return {axis: True for axis in contract.EVENT_AXES}
    if branch == "noop":
        return {axis: False for axis in contract.EVENT_AXES}
    if branch == "incomplete":
        return {
            "actor": True,
            "direction": True,
            "contact": True,
            "order": True,
            "terminal": False,
        }
    if branch == "reverse":
        return {
            "actor": True,
            "direction": False,
            "contact": True,
            "order": False,
            "terminal": False,
        }
    raise AssertionError(branch)


def _qualification(receipt: dict, receipt_sha: str) -> dict:
    entries = []
    for artifact in receipt["entries"]:
        entries.append(
            {
                "entry_id": artifact["entry_id"],
                "seed_id": artifact["seed_id"],
                "semantic_branch": artifact["semantic_branch"],
                "video_sha256": artifact["video_sha256"],
                "clean_latent_sha256": artifact["clean_latent_sha256"],
                "initial_gaussian_value_sha256": artifact[
                    "initial_gaussian_value_sha256"
                ],
                "calibrated": True,
                "absolute_uncertainty": 0.05,
                "event_axis_pass": _expected_events(artifact["semantic_branch"]),
                "branch_contract_pass": True,
            }
        )
    seal = {
        "schema_version": contract.QUALIFICATION_SEAL_SCHEMA,
        "seal_id": "cdf-dog-pass-a-blind-v3",
        "pass_a_receipt_file_sha256": receipt_sha,
        "pass_a_receipt_digest": receipt["receipt_digest"],
        "evaluator_sha256": _sha("one-global-pass-a-evaluator"),
        "calibration_sha256": _sha("one-global-pass-a-calibration"),
        "absolute_uncertainty_threshold": 0.10,
        "blinded_before_pass_b": True,
        "pass_b_artifacts_available": [],
        "qualification_unit": "complete_two_seed_by_four_branch_bank",
        "entries": entries,
    }
    return _seal(seal, "seal_digest")


def _source() -> dict:
    stable = "One tan dog with a black collar and one long bone are on gray concrete."
    old_action = "The dog sits still beside the bone and looks forward."
    instruction = "The dog picks up the long bone and holds it in its mouth."
    captioner = _sha("captioner:model-and-code")
    return {
        "sample_id": "cdf-dog",
        "source_video_sha256": native.CDF_DOG_SOURCE_SHA256,
        "source_video_latent_sha256": _sha("source-video-latent"),
        "source_video_latent_shape": list(native.LATENT_SHAPE),
        "source_vae_receipt_sha256": _sha("source-vae-receipt"),
        "correct_reference_tensor_sha256": [
            _sha(f"correct-ref:{index}") for index in range(4)
        ],
        "reference_frame_indices": [0, 27, 53, 80],
        "reference_vae_receipt_sha256": [
            _sha(f"reference-vae-receipt:{index}") for index in range(4)
        ],
        "stable_content_caption": stable,
        "stable_content_caption_sha256": _sha(stable),
        "observed_source_action": old_action,
        "observed_source_action_sha256": _sha(old_action),
        "captioner_artifact_sha256": captioner,
        "caption_artifact_sha256": contract.caption_artifact_sha256(
            stable, old_action, captioner
        ),
        "edit_instruction": instruction,
        "edit_instruction_sha256": _sha(instruction),
        "frame_count": 81,
    }


def _layout(layout_id: str) -> dict:
    return {
        "layout_id": layout_id,
        "renderer_mode": "native_rv2v",
        "frame_count": 81,
        "reference_frame_indices": [0, 27, 53, 80],
        "target_source_id": 0.0,
        "vi_source_count": 5,
        "image_only_source_count": 4,
        "vi_video_source_ids": [1.0],
        "vi_reference_source_ids": [2.0, 3.0, 4.0, 5.0],
        "image_only_reference_source_ids": [1.0, 2.0, 3.0, 4.0],
        "patch_source_id_order_per_step": [
            1.0,
            2.0,
            1.0,
            3.0,
            2.0,
            4.0,
            3.0,
            5.0,
            4.0,
            0.0,
        ],
        "guidance_branch_order": ["none", "V", "VI_uncond", "VI_text"],
        "native_source_id_interpolation_used": False,
    }


def _policy(source: dict) -> dict:
    prompt = contract.build_renderer_prompt(
        source["stable_content_caption"],
        source["observed_source_action"],
        source["edit_instruction"],
    )
    return {
        "policy_id": "bernini-r13-v3",
        "policy_revision": 0,
        "bernini_commit": native.BERNINI_COMMIT,
        "veomni_commit": native.VEOMNI_COMMIT,
        "checkpoint_sha256": _sha("checkpoint-tree"),
        "method_source_revision": "2" * 40,
        "method_source_archive_sha256": _sha("pass-b-method-archive"),
        "renderer_prompt": prompt,
        "renderer_prompt_sha256": _sha(prompt),
        "sampler_config_sha256": _sha("official-sampler-config"),
        "guidance_config_sha256": _sha("text-cfg-only-guidance-config"),
        "guidance_kind": "text_cfg_only_no_spatial_guidance",
        "gate_order": list(contract.RENDER_GATES),
        "evaluator_sha256_by_gate": {
            gate: _sha(f"renderer-evaluator:{gate}") for gate in contract.RENDER_GATES
        },
        "calibration_sha256_by_gate": {
            gate: _sha(f"renderer-calibration:{gate}") for gate in contract.RENDER_GATES
        },
        "absolute_uncertainty_threshold_by_gate": {
            gate: 0.10 for gate in contract.RENDER_GATES
        },
    }


def _spec(
    candidate_id: str,
    role: str,
    *,
    donor_mode: str,
    donor_branch: str,
    pass_a_entry_id: str | None,
    donor_latent: str,
    reference_mode: str,
    refs: list[str],
    renderer_seed: int,
    gaussian_label: str,
    layout_id: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_role": role,
        "donor_mode": donor_mode,
        "donor_semantic_branch": donor_branch,
        "pass_a_entry_id": pass_a_entry_id,
        "donor_latent_sha256": donor_latent,
        "reference_mode": reference_mode,
        "reference_tensor_sha256": list(refs),
        "renderer_seed": renderer_seed,
        "official_gaussian_sha256": _sha(f"renderer-gaussian:{gaussian_label}"),
        "condition_layout": _layout(layout_id),
        "max_gpu_seconds": 50.0,
    }


def _preregistration(
    receipt: dict,
    receipt_sha: str,
    qualification_sha: str,
) -> dict:
    source = _source()
    policy = _policy(source)
    entries = {row["entry_id"]: row for row in receipt["entries"]}
    full_id = "seed-20260808-full-action"
    incomplete_id = "seed-20260808-incomplete"
    full_latent = entries[full_id]["clean_latent_sha256"]
    incomplete_latent = entries[incomplete_id]["clean_latent_sha256"]
    correct = list(source["correct_reference_tensor_sha256"])
    wrong = [_sha(f"wrong-ref:{index}") for index in range(4)]
    off = [_sha(f"off-ref:{index}") for index in range(4)]
    off_donor = _sha("off-donor-tensor")
    specs = [
        _spec(
            "main-positive",
            "action_positive",
            donor_mode="proposal",
            donor_branch="full_action",
            pass_a_entry_id=full_id,
            donor_latent=full_latent,
            reference_mode="correct",
            refs=correct,
            renderer_seed=900,
            gaussian_label="pair",
            layout_id="pair-layout",
        ),
        _spec(
            "action-negative",
            "action_nearmiss",
            donor_mode="proposal",
            donor_branch="incomplete",
            pass_a_entry_id=incomplete_id,
            donor_latent=incomplete_latent,
            reference_mode="correct",
            refs=correct,
            renderer_seed=900,
            gaussian_label="pair",
            layout_id="pair-layout",
        ),
        _spec(
            "identity-negative",
            "preservation_nearmiss",
            donor_mode="proposal",
            donor_branch="full_action",
            pass_a_entry_id=full_id,
            donor_latent=full_latent,
            reference_mode="wrong",
            refs=wrong,
            renderer_seed=900,
            gaussian_label="pair",
            layout_id="pair-layout",
        ),
        _spec(
            "grid-neither",
            "factorial_neither",
            donor_mode="off",
            donor_branch="off",
            pass_a_entry_id=None,
            donor_latent=off_donor,
            reference_mode="off",
            refs=off,
            renderer_seed=901,
            gaussian_label="grid",
            layout_id="factorial-layout",
        ),
        _spec(
            "grid-refs",
            "factorial_refs_only",
            donor_mode="off",
            donor_branch="off",
            pass_a_entry_id=None,
            donor_latent=off_donor,
            reference_mode="correct",
            refs=correct,
            renderer_seed=901,
            gaussian_label="grid",
            layout_id="factorial-layout",
        ),
        _spec(
            "grid-donor",
            "factorial_donor_only",
            donor_mode="proposal",
            donor_branch="full_action",
            pass_a_entry_id=full_id,
            donor_latent=full_latent,
            reference_mode="off",
            refs=off,
            renderer_seed=901,
            gaussian_label="grid",
            layout_id="factorial-layout",
        ),
        _spec(
            "grid-both",
            "factorial_donor_refs",
            donor_mode="proposal",
            donor_branch="full_action",
            pass_a_entry_id=full_id,
            donor_latent=full_latent,
            reference_mode="correct",
            refs=correct,
            renderer_seed=901,
            gaussian_label="grid",
            layout_id="factorial-layout",
        ),
        _spec(
            "source-copy",
            "source_copy_control",
            donor_mode="source_video",
            donor_branch="source_action",
            pass_a_entry_id=None,
            donor_latent=source["source_video_latent_sha256"],
            reference_mode="correct",
            refs=correct,
            renderer_seed=900,
            gaussian_label="pair",
            layout_id="pair-layout",
        ),
        _spec(
            "quality-negative",
            "quality_control",
            donor_mode="proposal",
            donor_branch="full_action",
            pass_a_entry_id=full_id,
            donor_latent=full_latent,
            reference_mode="correct",
            refs=correct,
            renderer_seed=903,
            gaussian_label="quality",
            layout_id="quality-layout",
        ),
    ]
    seal = {
        "schema_version": contract.PREREGISTRATION_SEAL_SCHEMA,
        "seal_id": "cdf-dog-pass-b-v3",
        "sealed_before_pass_b": True,
        "topup_allowed": False,
        "pass_a_receipt_file_sha256": receipt_sha,
        "pass_a_qualification_seal_file_sha256": qualification_sha,
        "source": source,
        "renderer_policy": policy,
        "off_donor_latent_sha256": off_donor,
        "reference_tensor_sha256_by_mode": {
            "correct": correct,
            "wrong": wrong,
            "off": off,
        },
        "candidate_specs": specs,
        "causal_pairs": [
            {
                "pair_id": "same-seed-action",
                "pair_type": "action_donor_nearmiss",
                "winner_candidate_id": "main-positive",
                "loser_candidate_id": "action-negative",
                "pass_a_seed_id": "seed-20260808",
                "source_copy_candidate_id": "source-copy",
            },
            {
                "pair_id": "same-donor-identity",
                "pair_type": "identity_reference_nearmiss",
                "winner_candidate_id": "main-positive",
                "loser_candidate_id": "identity-negative",
                "pass_a_seed_id": "seed-20260808",
                "source_copy_candidate_id": "source-copy",
            },
        ],
        "counterfactual_2x2": {
            "layout_id": "factorial-layout",
            "arm_to_candidate_id": {
                "neither": "grid-neither",
                "identity_refs_only": "grid-refs",
                "donor_only": "grid-donor",
                "donor_and_identity_refs": "grid-both",
            },
            "gate_rule": (
                "donor_artifact_controls_A_reference_artifact_controls_I_"
                "with_CQ_pass_and_all_other_invocation_fields_equal"
            ),
        },
        "candidate_count": len(specs),
        "total_gpu_seconds_budget": 450.0,
    }
    return _seal(seal, "seal_digest")


def _chain(
    *,
    receipt: dict | None = None,
    artifacts: dict[str, dict[str, str]] | None = None,
    qualification: dict | None = None,
    preregistration: dict | None = None,
) -> dict:
    if receipt is None or artifacts is None:
        receipt, artifacts = _pass_a_receipt()
    _seal(receipt, "receipt_digest")
    receipt_bytes, receipt_sha = _bytes(receipt)
    if qualification is None:
        qualification = _qualification(receipt, receipt_sha)
    qualification["pass_a_receipt_file_sha256"] = receipt_sha
    qualification["pass_a_receipt_digest"] = receipt["receipt_digest"]
    _seal(qualification, "seal_digest")
    qualification_bytes, qualification_sha = _bytes(qualification)
    if preregistration is None:
        preregistration = _preregistration(
            receipt, receipt_sha, qualification_sha
        )
    preregistration["pass_a_receipt_file_sha256"] = receipt_sha
    preregistration["pass_a_qualification_seal_file_sha256"] = qualification_sha
    _seal(preregistration, "seal_digest")
    preregistration_bytes, preregistration_sha = _bytes(preregistration)
    context = contract.build_external_context(
        pass_a_receipt_bytes=receipt_bytes,
        expected_pass_a_receipt_sha256=receipt_sha,
        pass_a_artifact_hashes=artifacts,
        pass_a_qualification_seal_bytes=qualification_bytes,
        expected_pass_a_qualification_seal_sha256=qualification_sha,
        preregistration_seal_bytes=preregistration_bytes,
        expected_preregistration_seal_sha256=preregistration_sha,
    )
    return {
        "receipt": receipt,
        "receipt_bytes": receipt_bytes,
        "receipt_sha": receipt_sha,
        "artifacts": artifacts,
        "qualification": qualification,
        "qualification_bytes": qualification_bytes,
        "qualification_sha": qualification_sha,
        "preregistration": preregistration,
        "preregistration_bytes": preregistration_bytes,
        "preregistration_sha": preregistration_sha,
        "context": context,
    }


def _gate(policy: dict, source: dict, output_sha: str, gate: str, margin: float) -> dict:
    threshold = policy["absolute_uncertainty_threshold_by_gate"][gate]
    evaluator = policy["evaluator_sha256_by_gate"][gate]
    calibration = policy["calibration_sha256_by_gate"][gate]
    return {
        "evaluator_sha256": evaluator,
        "calibration_sha256": calibration,
        "absolute_uncertainty_threshold": threshold,
        "calibrated": True,
        "margin": margin,
        "absolute_uncertainty": 0.05,
        "pass": margin > 0,
        "evaluation_input_digest": contract.canonical_object_sha256(
            {
                "gate": gate,
                "output_video_sha256": output_sha,
                "source_video_sha256": source["source_video_sha256"],
                "edit_instruction_sha256": source["edit_instruction_sha256"],
                "evaluator_sha256": evaluator,
                "calibration_sha256": calibration,
                "absolute_uncertainty_threshold": threshold,
            }
        ),
    }


def _candidate(spec: dict, policy: dict, source: dict, margins: dict[str, float]) -> dict:
    output_latent = _sha(f"Pass-B-output-latent:{spec['candidate_id']}")
    output_video = _sha(f"Pass-B-output-video:{spec['candidate_id']}")
    invocation = {
        "schema_version": contract.INVOCATION_SCHEMA,
        "invocation_id": f"invoke-{spec['candidate_id']}",
        "candidate_id": spec["candidate_id"],
        "checkpoint_sha256": policy["checkpoint_sha256"],
        "donor": {
            "mode": spec["donor_mode"],
            "semantic_branch": spec["donor_semantic_branch"],
            "pass_a_entry_id": spec["pass_a_entry_id"],
            "latent_sha256": spec["donor_latent_sha256"],
        },
        "references": {
            "mode": spec["reference_mode"],
            "tensor_sha256": list(spec["reference_tensor_sha256"]),
        },
        "condition_layout": deepcopy(spec["condition_layout"]),
        "renderer_prompt": policy["renderer_prompt"],
        "renderer_prompt_sha256": policy["renderer_prompt_sha256"],
        "sampler_config_sha256": policy["sampler_config_sha256"],
        "guidance_config_sha256": policy["guidance_config_sha256"],
        "guidance_kind": policy["guidance_kind"],
        "official_gaussian": {
            "source": contract.OFFICIAL_GAUSSIAN_SOURCE,
            "seed": spec["renderer_seed"],
            "tensor_sha256": spec["official_gaussian_sha256"],
        },
        "external_inputs": [
            "checkpoint",
            "donor_latent",
            "reference_tensors",
            "renderer_prompt",
            "sampler_config",
            "guidance_config",
            "official_gaussian",
        ],
        "output_clean_latent_sha256": output_latent,
        "output_video_sha256": output_video,
        "output_frame_count": native.FRAME_COUNT,
        "output_fps": native.FPS,
        "output_height": native.VIDEO_HEIGHT,
        "output_width": native.VIDEO_WIDTH,
        "output_clean_latent_shape": list(native.LATENT_SHAPE),
        "measured_gpu_seconds": 10.0,
    }
    _seal(invocation, "invocation_digest")
    gates = {
        gate: _gate(policy, source, output_video, gate, margins[gate])
        for gate in contract.RENDER_GATES
    }
    failed = [gate for gate in contract.RENDER_GATES if not gates[gate]["pass"]]
    candidate = {
        "schema_version": contract.CANDIDATE_SCHEMA,
        "candidate_id": spec["candidate_id"],
        "native_invocation": invocation,
        "gates": gates,
        "joint_pass": not failed,
        "disposition": {
            "status": "strict_joint_accepted" if not failed else "explicitly_rejected",
            "failed_gates": failed,
        },
    }
    return _seal(candidate, "candidate_digest")


def _episode(chain: dict) -> tuple[dict, dict[str, dict[str, str]]]:
    context = chain["context"]
    if context.pass_a_reasons:
        episode = {
            "schema_version": contract.EPISODE_SCHEMA,
            "episode_id": "cdf-dog-self-imagined-v3",
            "external_seals": {
                "pass_a_receipt_file_sha256": context.pass_a_receipt_sha256,
                "pass_a_qualification_seal_file_sha256": context.pass_a_qualification_sha256,
                "pass_b_preregistration_seal_file_sha256": context.preregistration_sha256,
            },
            "candidates": [],
            "measured_gpu_seconds": 0.0,
        }
        return _seal(episode, "episode_digest"), {}
    margins = {
        "main-positive": {"A": 2.0, "I": 2.0, "C": 2.0, "Q": 2.0},
        "action-negative": {"A": -1.0, "I": 1.0, "C": 1.0, "Q": 1.0},
        "identity-negative": {"A": 1.0, "I": -1.0, "C": 1.0, "Q": 1.0},
        "grid-neither": {"A": -1.0, "I": -1.0, "C": 1.0, "Q": 1.0},
        "grid-refs": {"A": -1.0, "I": 1.0, "C": 1.0, "Q": 1.0},
        "grid-donor": {"A": 1.0, "I": -1.0, "C": 1.0, "Q": 1.0},
        "grid-both": {"A": 1.0, "I": 1.0, "C": 1.0, "Q": 1.0},
        "source-copy": {"A": -1.0, "I": 1.0, "C": 1.0, "Q": 1.0},
        "quality-negative": {"A": 1.0, "I": 1.0, "C": -1.0, "Q": 1.0},
    }
    policy = chain["preregistration"]["renderer_policy"]
    source = chain["preregistration"]["source"]
    candidates = [
        _candidate(spec, policy, source, margins[spec["candidate_id"]])
        for spec in chain["preregistration"]["candidate_specs"]
    ]
    registry = {
        candidate["candidate_id"]: {
            "invocation_digest": candidate["native_invocation"]["invocation_digest"],
            "official_gaussian_sha256": candidate["native_invocation"][
                "official_gaussian"
            ]["tensor_sha256"],
            "output_clean_latent_sha256": candidate["native_invocation"][
                "output_clean_latent_sha256"
            ],
            "output_video_sha256": candidate["native_invocation"][
                "output_video_sha256"
            ],
            "output_frame_count": candidate["native_invocation"]["output_frame_count"],
            "output_fps": candidate["native_invocation"]["output_fps"],
            "output_height": candidate["native_invocation"]["output_height"],
            "output_width": candidate["native_invocation"]["output_width"],
            "output_clean_latent_shape": candidate["native_invocation"][
                "output_clean_latent_shape"
            ],
        }
        for candidate in candidates
    }
    episode = {
        "schema_version": contract.EPISODE_SCHEMA,
        "episode_id": "cdf-dog-self-imagined-v3",
        "external_seals": {
            "pass_a_receipt_file_sha256": context.pass_a_receipt_sha256,
            "pass_a_qualification_seal_file_sha256": context.pass_a_qualification_sha256,
            "pass_b_preregistration_seal_file_sha256": context.preregistration_sha256,
        },
        "candidates": candidates,
        "measured_gpu_seconds": 90.0,
    }
    return _seal(episode, "episode_digest"), registry


def _candidate_by_id(episode: dict, candidate_id: str) -> dict:
    return next(row for row in episode["candidates"] if row["candidate_id"] == candidate_id)


def _sync_candidate(candidate: dict, *, invocation: bool = False) -> None:
    if invocation:
        _seal(candidate["native_invocation"], "invocation_digest")
    failed = [gate for gate in contract.RENDER_GATES if not candidate["gates"][gate]["pass"]]
    candidate["joint_pass"] = not failed
    candidate["disposition"] = {
        "status": "strict_joint_accepted" if not failed else "explicitly_rejected",
        "failed_gates": failed,
    }
    _seal(candidate, "candidate_digest")


def _sync_registry(episode: dict) -> dict[str, dict[str, str]]:
    return {
        candidate["candidate_id"]: {
            "invocation_digest": candidate["native_invocation"]["invocation_digest"],
            "official_gaussian_sha256": candidate["native_invocation"][
                "official_gaussian"
            ]["tensor_sha256"],
            "output_clean_latent_sha256": candidate["native_invocation"][
                "output_clean_latent_sha256"
            ],
            "output_video_sha256": candidate["native_invocation"][
                "output_video_sha256"
            ],
            "output_frame_count": candidate["native_invocation"]["output_frame_count"],
            "output_fps": candidate["native_invocation"]["output_fps"],
            "output_height": candidate["native_invocation"]["output_height"],
            "output_width": candidate["native_invocation"]["output_width"],
            "output_clean_latent_shape": candidate["native_invocation"][
                "output_clean_latent_shape"
            ],
        }
        for candidate in episode["candidates"]
    }


class ValidV3Tests(unittest.TestCase):
    def test_actual_pass_a_bytes_emit_both_preregistered_pairs(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        self.assertIs(
            contract.validate_episode(
                episode, context=chain["context"], pass_b_artifact_hashes=registry
            ),
            episode,
        )
        receipt = contract.select_upstream_qualification(
            episode, context=chain["context"], pass_b_artifact_hashes=registry
        )
        self.assertTrue(receipt["upstream_pair_qualified"])
        self.assertEqual(
            [row["causal_pair_type"] for row in receipt["qualifications"]],
            list(contract.PAIR_TYPES),
        )
        self.assertEqual(contract.PROPOSAL_BRANCHES, native.BRANCH_ORDER)
        self.assertEqual(receipt["search_accounting"]["total_model_generations"], 17)
        self.assertIs(
            contract.validate_selection_receipt(
                receipt,
                episode,
                context=chain["context"],
                pass_b_artifact_hashes=registry,
            ),
            receipt,
        )

    def test_episode_cannot_self_supply_context_or_budget(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        with self.assertRaises(TypeError):
            contract.select_upstream_qualification(episode)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            contract.validate_episode(episode, context=chain["context"])  # type: ignore[call-arg]

    def test_no_production_dclr_bridge_exists(self) -> None:
        with self.assertRaisesRegex(
            contract.ProductionDCLRBridgeUnavailable,
            "production consumption is forbidden",
        ):
            contract.to_production_dclr_preference_pair({})


class ActualPassAFailClosedTests(unittest.TestCase):
    def _build_with_receipt(
        self,
        receipt: dict,
        artifacts: dict[str, dict[str, str]],
        base: dict,
    ) -> None:
        _seal(receipt, "receipt_digest")
        receipt_bytes, receipt_sha = _bytes(receipt)
        contract.build_external_context(
            pass_a_receipt_bytes=receipt_bytes,
            expected_pass_a_receipt_sha256=receipt_sha,
            pass_a_artifact_hashes=artifacts,
            pass_a_qualification_seal_bytes=base["qualification_bytes"],
            expected_pass_a_qualification_seal_sha256=base["qualification_sha"],
            preregistration_seal_bytes=base["preregistration_bytes"],
            expected_preregistration_seal_sha256=base["preregistration_sha"],
        )

    def test_actual_branch_names_are_byte_exact_not_translated(self) -> None:
        base = _chain()
        receipt = deepcopy(base["receipt"])
        receipt["entries"][2]["semantic_branch"] = "incomplete_action"
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "exact native seed/branch order"
        ):
            self._build_with_receipt(receipt, base["artifacts"], base)

    def test_exact_two_by_four_is_mandatory(self) -> None:
        base = _chain()
        receipt = deepcopy(base["receipt"])
        receipt["entries"] = receipt["entries"][:4]
        receipt["entry_count"] = 4
        receipt["seed_count"] = 1
        receipt["initial_gaussian_contract"]["per_seed_value_sha256"].pop(
            "seed-20260809"
        )
        artifacts = {
            key: value
            for key, value in base["artifacts"].items()
            if key.startswith("seed-20260808")
        }
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "exact two-seed by four-branch"
        ):
            self._build_with_receipt(receipt, artifacts, base)

    def test_actual_receipt_bytes_and_independent_artifact_registry_are_pinned(self) -> None:
        base = _chain()
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "bytes differ from external seal"
        ):
            contract.build_external_context(
                pass_a_receipt_bytes=base["receipt_bytes"] + b" ",
                expected_pass_a_receipt_sha256=base["receipt_sha"],
                pass_a_artifact_hashes=base["artifacts"],
                pass_a_qualification_seal_bytes=base["qualification_bytes"],
                expected_pass_a_qualification_seal_sha256=base["qualification_sha"],
                preregistration_seal_bytes=base["preregistration_bytes"],
                expected_preregistration_seal_sha256=base["preregistration_sha"],
            )
        artifacts = deepcopy(base["artifacts"])
        first = next(iter(artifacts))
        artifacts[first]["clean_latent_sha256"] = _sha("forged-clean-latent")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "artifact bytes/hash registry differs"
        ):
            self._build_with_receipt(deepcopy(base["receipt"]), artifacts, base)

    def test_pass_a_gaussian_same_within_seed_distinct_between_seeds(self) -> None:
        base = _chain()
        receipt = deepcopy(base["receipt"])
        artifacts = deepcopy(base["artifacts"])
        row = receipt["entries"][1]
        row["initial_gaussian_value_sha256"] = _sha("branch-specific-gaussian")
        artifacts[row["entry_id"]]["initial_gaussian_value_sha256"] = row[
            "initial_gaussian_value_sha256"
        ]
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "within one seed"
        ):
            self._build_with_receipt(receipt, artifacts, base)

        receipt = deepcopy(base["receipt"])
        artifacts = deepcopy(base["artifacts"])
        first = receipt["entries"][0]["initial_gaussian_value_sha256"]
        for row in receipt["entries"][4:]:
            row["initial_gaussian_value_sha256"] = first
            artifacts[row["entry_id"]]["initial_gaussian_value_sha256"] = first
        receipt["initial_gaussian_contract"]["per_seed_value_sha256"][
            "seed-20260809"
        ] = first
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "distinct Gaussian values"
        ):
            self._build_with_receipt(receipt, artifacts, base)

    def test_either_full_action_failure_nulls_whole_bank_without_seed_pick(self) -> None:
        receipt, artifacts = _pass_a_receipt()
        receipt_bytes, receipt_sha = _bytes(receipt)
        qualification = _qualification(receipt, receipt_sha)
        row = next(
            item
            for item in qualification["entries"]
            if item["entry_id"] == "seed-20260809-full-action"
        )
        row["event_axis_pass"]["actor"] = False
        row["branch_contract_pass"] = False
        chain = _chain(
            receipt=receipt,
            artifacts=artifacts,
            qualification=qualification,
        )
        episode, registry = _episode(chain)
        result = contract.select_upstream_qualification(
            episode, context=chain["context"], pass_b_artifact_hashes=registry
        )
        self.assertFalse(result["upstream_pair_qualified"])
        self.assertEqual(result["search_accounting"]["pass_b_candidate_count"], 0)
        self.assertIn(
            "seed-20260809:full_action_failed", result["null_reasons"]
        )

    def test_any_bad_or_uncalibrated_branch_nulls_complete_bank(self) -> None:
        for mutation in ("bad-negative", "uncalibrated"):
            receipt, artifacts = _pass_a_receipt()
            receipt_bytes, receipt_sha = _bytes(receipt)
            qualification = _qualification(receipt, receipt_sha)
            if mutation == "bad-negative":
                row = next(
                    item
                    for item in qualification["entries"]
                    if item["entry_id"] == "seed-20260808-reverse"
                )
                row["event_axis_pass"] = {
                    axis: False for axis in contract.EVENT_AXES
                }
                row["branch_contract_pass"] = False
            else:
                qualification["entries"][-1]["calibrated"] = False
            chain = _chain(
                receipt=receipt,
                artifacts=artifacts,
                qualification=qualification,
            )
            episode, registry = _episode(chain)
            result = contract.select_upstream_qualification(
                episode,
                context=chain["context"],
                pass_b_artifact_hashes=registry,
            )
            with self.subTest(mutation=mutation):
                self.assertFalse(result["upstream_pair_qualified"])
                self.assertEqual(result["qualifications"], [])

    def test_pass_a_evaluator_cannot_vary_per_entry(self) -> None:
        receipt, artifacts = _pass_a_receipt()
        receipt_bytes, receipt_sha = _bytes(receipt)
        qualification = _qualification(receipt, receipt_sha)
        qualification["entries"][0]["evaluator_sha256"] = _sha("row-specific")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "fields are closed"
        ):
            _chain(
                receipt=receipt,
                artifacts=artifacts,
                qualification=qualification,
            )

    def test_one_step_actual_receipt_is_engineering_null(self) -> None:
        receipt, artifacts = _pass_a_receipt()
        receipt["stage"] = "engineering-one-step"
        receipt["qualification"]["exact40_manual_qualification_required"] = False
        receipt["qualification"]["pass_a_status"] = (
            "engineering_only_no_semantic_claim"
        )
        receipt_bytes, receipt_sha = _bytes(_seal(receipt, "receipt_digest"))
        qualification = _qualification(receipt, receipt_sha)
        chain = _chain(
            receipt=receipt,
            artifacts=artifacts,
            qualification=qualification,
        )
        episode, registry = _episode(chain)
        result = contract.select_upstream_qualification(
            episode, context=chain["context"], pass_b_artifact_hashes=registry
        )
        self.assertIn("pass_a_receipt_is_not_exact40", result["null_reasons"])


class ExternalSealAndBudgetTests(unittest.TestCase):
    def test_source_copy_latent_and_refs_are_bound_to_same_pass_a_source(self) -> None:
        base = _chain()
        for field, value in (
            ("source_video_sha256", _sha("other-source-video")),
            ("source_video_latent_shape", [1, 16, 11, 62, 60]),
        ):
            prereg = deepcopy(base["preregistration"])
            prereg["source"][field] = value
            with self.subTest(field=field), self.assertRaises(
                contract.SelfImaginedParetoContractError
            ):
                _chain(
                    receipt=deepcopy(base["receipt"]),
                    artifacts=deepcopy(base["artifacts"]),
                    qualification=deepcopy(base["qualification"]),
                    preregistration=prereg,
                )

    def test_episode_posthoc_topup_is_rejected_by_external_candidate_seal(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        episode["candidates"].append(deepcopy(episode["candidates"][-1]))
        _seal(episode, "episode_digest")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "exactly exhaust external"
        ):
            contract.validate_episode(
                episode, context=chain["context"], pass_b_artifact_hashes=registry
            )

    def test_episode_cannot_widen_externally_sealed_threshold(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        candidate = _candidate_by_id(episode, "main-positive")
        candidate["gates"]["A"]["absolute_uncertainty_threshold"] = 1_000_000.0
        candidate["gates"]["A"]["absolute_uncertainty"] = 999_999.0
        _sync_candidate(candidate)
        _seal(episode, "episode_digest")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "external seal"
        ):
            contract.validate_episode(
                episode, context=chain["context"], pass_b_artifact_hashes=registry
            )

    def test_preregistration_bytes_cannot_be_resealed_inside_episode(self) -> None:
        base = _chain()
        forged = deepcopy(base["preregistration"])
        forged["renderer_policy"]["absolute_uncertainty_threshold_by_gate"][
            "A"
        ] = 1_000_000.0
        _seal(forged, "seal_digest")
        forged_bytes, _ = _bytes(forged)
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "bytes differ from external seal"
        ):
            contract.build_external_context(
                pass_a_receipt_bytes=base["receipt_bytes"],
                expected_pass_a_receipt_sha256=base["receipt_sha"],
                pass_a_artifact_hashes=base["artifacts"],
                pass_a_qualification_seal_bytes=base["qualification_bytes"],
                expected_pass_a_qualification_seal_sha256=base["qualification_sha"],
                preregistration_seal_bytes=forged_bytes,
                expected_preregistration_seal_sha256=base["preregistration_sha"],
            )

    def test_renderer_seed_to_gaussian_mapping_is_one_to_one(self) -> None:
        base = _chain()
        for mutation in ("same-seed-different", "different-seed-same"):
            prereg = deepcopy(base["preregistration"])
            quality = next(
                row
                for row in prereg["candidate_specs"]
                if row["candidate_id"] == "quality-negative"
            )
            main = next(
                row
                for row in prereg["candidate_specs"]
                if row["candidate_id"] == "main-positive"
            )
            if mutation == "same-seed-different":
                quality["renderer_seed"] = main["renderer_seed"]
            else:
                quality["official_gaussian_sha256"] = main[
                    "official_gaussian_sha256"
                ]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                contract.SelfImaginedParetoContractError,
                "renderer seed|different renderer seeds",
            ):
                _chain(
                    receipt=deepcopy(base["receipt"]),
                    artifacts=deepcopy(base["artifacts"]),
                    qualification=deepcopy(base["qualification"]),
                    preregistration=prereg,
                )

    def test_pass_b_independent_artifact_hashes_are_required(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        registry["main-positive"]["output_video_sha256"] = _sha("forged-output")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "independently observed"
        ):
            contract.validate_episode(
                episode, context=chain["context"], pass_b_artifact_hashes=registry
            )

    def test_pass_b_output_cannot_silently_fall_back_to_41_frames(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        candidate = _candidate_by_id(episode, "main-positive")
        candidate["native_invocation"]["output_frame_count"] = 41
        _sync_candidate(candidate, invocation=True)
        registry = _sync_registry(episode)
        _seal(episode, "episode_digest")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "not exact81"
        ):
            contract.validate_episode(
                episode, context=chain["context"], pass_b_artifact_hashes=registry
            )

    def test_missing_measurement_and_budget_overrun_fail_closed(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        candidate = _candidate_by_id(episode, "main-positive")
        candidate["native_invocation"]["measured_gpu_seconds"] = None
        _sync_candidate(candidate, invocation=True)
        registry = _sync_registry(episode)
        _seal(episode, "episode_digest")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "finite numeric"
        ):
            contract.validate_episode(
                episode, context=chain["context"], pass_b_artifact_hashes=registry
            )

    def test_native_vi_and_i_source_id_call_order_is_sealed(self) -> None:
        chain = _chain()
        prereg = deepcopy(chain["preregistration"])
        prereg["candidate_specs"][0]["condition_layout"][
            "image_only_reference_source_ids"
        ] = [2.0, 3.0, 4.0, 5.0]
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "VI/I source-ID"
        ):
            _chain(
                receipt=deepcopy(chain["receipt"]),
                artifacts=deepcopy(chain["artifacts"]),
                qualification=deepcopy(chain["qualification"]),
                preregistration=prereg,
            )


class CausalPairRegressionTests(unittest.TestCase):
    def test_action_nearmiss_must_use_same_pass_a_seed(self) -> None:
        base = _chain()
        prereg = deepcopy(base["preregistration"])
        receipt_entries = {row["entry_id"]: row for row in base["receipt"]["entries"]}
        second = receipt_entries["seed-20260809-incomplete"]
        negative = next(
            row
            for row in prereg["candidate_specs"]
            if row["candidate_id"] == "action-negative"
        )
        negative["pass_a_entry_id"] = second["entry_id"]
        negative["donor_latent_sha256"] = second["clean_latent_sha256"]
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "same-Pass-A-seed"
        ):
            _chain(
                receipt=deepcopy(base["receipt"]),
                artifacts=deepcopy(base["artifacts"]),
                qualification=deepcopy(base["qualification"]),
                preregistration=prereg,
            )

    def test_action_pair_requires_correct_refs_and_its_own_source_copy(self) -> None:
        base = _chain()
        for mutation in ("wrong-refs", "other-source-copy"):
            prereg = deepcopy(base["preregistration"])
            if mutation == "wrong-refs":
                wrong = prereg["reference_tensor_sha256_by_mode"]["wrong"]
                for candidate_id in ("main-positive", "action-negative"):
                    spec = next(
                        row
                        for row in prereg["candidate_specs"]
                        if row["candidate_id"] == candidate_id
                    )
                    spec["reference_mode"] = "wrong"
                    spec["reference_tensor_sha256"] = list(wrong)
            else:
                prereg["causal_pairs"][0]["source_copy_candidate_id"] = (
                    "quality-negative"
                )
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                contract.SelfImaginedParetoContractError, "source-copy|same-Pass-A-seed"
            ):
                _chain(
                    receipt=deepcopy(base["receipt"]),
                    artifacts=deepcopy(base["artifacts"]),
                    qualification=deepcopy(base["qualification"]),
                    preregistration=prereg,
                )

    def test_pair_list_is_external_and_no_partial_pair_update_is_emitted(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        winner = _candidate_by_id(episode, "main-positive")
        winner["gates"]["I"]["margin"] = 0.5
        _sync_candidate(winner)
        _seal(episode, "episode_digest")
        result = contract.select_upstream_qualification(
            episode, context=chain["context"], pass_b_artifact_hashes=registry
        )
        self.assertFalse(result["upstream_pair_qualified"])
        self.assertEqual(result["qualifications"], [])
        self.assertIn(
            "same-seed-action:preregistered_causal_pair_failed",
            result["null_reasons"],
        )

    def test_source_copy_failure_nulls_selected_pairs(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        source_copy = _candidate_by_id(episode, "source-copy")
        source_copy["gates"]["I"]["margin"] = -1.0
        source_copy["gates"]["I"]["pass"] = False
        _sync_candidate(source_copy)
        _seal(episode, "episode_digest")
        result = contract.select_upstream_qualification(
            episode, context=chain["context"], pass_b_artifact_hashes=registry
        )
        self.assertFalse(result["upstream_pair_qualified"])
        self.assertTrue(
            any("bound_source_copy_control_failed" in reason for reason in result["null_reasons"])
        )

    def test_C_or_Q_failure_never_becomes_identity_route(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        identity = _candidate_by_id(episode, "identity-negative")
        identity["gates"]["I"]["margin"] = 1.0
        identity["gates"]["I"]["pass"] = True
        identity["gates"]["C"]["margin"] = -1.0
        identity["gates"]["C"]["pass"] = False
        _sync_candidate(identity)
        _seal(episode, "episode_digest")
        result = contract.select_upstream_qualification(
            episode, context=chain["context"], pass_b_artifact_hashes=registry
        )
        self.assertFalse(result["upstream_pair_qualified"])
        self.assertEqual(result["qualifications"], [])
        self.assertIn(
            "C_or_Q_only_rejections_are_diagnostic_not_identity_routes",
            result["null_reasons"],
        )

    def test_failed_2x2_or_uncertain_critic_nulls_every_pair(self) -> None:
        for mutation in ("grid", "uncertainty"):
            chain = _chain()
            episode, registry = _episode(chain)
            if mutation == "grid":
                candidate = _candidate_by_id(episode, "grid-donor")
                candidate["gates"]["A"]["margin"] = -1.0
                candidate["gates"]["A"]["pass"] = False
            else:
                candidate = _candidate_by_id(episode, "quality-negative")
                candidate["gates"]["Q"]["absolute_uncertainty"] = 0.2
                candidate["gates"]["Q"]["pass"] = False
            _sync_candidate(candidate)
            _seal(episode, "episode_digest")
            result = contract.select_upstream_qualification(
                episode,
                context=chain["context"],
                pass_b_artifact_hashes=registry,
            )
            with self.subTest(mutation=mutation):
                self.assertFalse(result["upstream_pair_qualified"])
                self.assertEqual(result["qualifications"], [])


class HardGateAndReceiptTests(unittest.TestCase):
    def test_joint_gate_is_strict_and_no_weighted_compensation_exists(self) -> None:
        source = (METHOD_ROOT / "self_imagined_pareto_episode.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("weighted_score", source)
        self.assertNotIn("argmax", source)
        self.assertIn("strict A AND I AND C AND Q", source)

    def test_receipt_cannot_be_forged_even_with_internal_digest(self) -> None:
        chain = _chain()
        episode, registry = _episode(chain)
        receipt = contract.select_upstream_qualification(
            episode, context=chain["context"], pass_b_artifact_hashes=registry
        )
        forged = deepcopy(receipt)
        forged["search_accounting"]["topup_observed"] = True
        _seal(forged, "receipt_digest")
        with self.assertRaisesRegex(
            contract.SelfImaginedParetoContractError, "recomputation"
        ):
            contract.validate_selection_receipt(
                forged,
                episode,
                context=chain["context"],
                pass_b_artifact_hashes=registry,
            )


if __name__ == "__main__":
    unittest.main()
