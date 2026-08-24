#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_energy_calibration_v3 as detached_events  # noqa: E402
import temporal_counterfactual_calibration_v1 as calibration  # noqa: E402
import temporal_counterfactual_contract_v1 as contract  # noqa: E402


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def make_same_state_proof(label: str) -> dict:
    def stages(name: str) -> dict:
        digest = sha(f"{label}-{name}")
        return {
            "before_action": digest,
            "after_action": digest,
            "after_noop": digest,
        }

    return {
        "noisy_latents_sha256_by_stage": stages("state"),
        "rotary_embs_sha256_by_stage": stages("rotary"),
        "native_timestep_sha256_by_stage": stages("timestep"),
        "same_noisy_latents_object_reused": True,
        "same_rotary_embs_object_reused": True,
        "same_native_timestep_object_reused": True,
        "post_call_tensor_bytes_unchanged": True,
    }


SPEC_SHA = contract.REQUIRED_CORE4_V2_SPEC_SHA256
BANK_DIGEST = contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
BANK_FILE_SHA = contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
MODEL_DIGEST = sha("frozen-model")
SEALED_SPEC = json.loads(
    (
        METHOD_ROOT
        / "assets"
        / "pair_v5_t2v_calibration_core4_bank_v2.json"
    ).read_text(encoding="ascii")
)
GROUP_BY_CANDIDATE = {
    candidate["candidate_id"]: group["group_id"]
    for group in SEALED_SPEC["groups"]
    for candidate in group["candidates"]
}


def make_identity(
    *,
    split: str,
    family: str,
    cell: str,
    branch: str,
) -> dict:
    stem = f"{split}-{family}-{cell}"
    return {
        "candidate_id": f"pair5-t2v-core4-v2-{stem}-{branch}",
        "analysis_split": split,
        "action_family_id": family,
        "calibration_group_id": f"cell-{stem}",
        "actor_group_id": f"actor-{stem}",
        "scene_group_id": f"scene-{stem}",
        "action_group_id": f"action-{stem}",
        "semantic_branch": branch,
    }


def make_bindings(identity: dict) -> tuple[dict, dict, dict, dict]:
    candidate_id = identity["candidate_id"]
    caption_sha = sha(identity["calibration_group_id"] + "-target-caption")
    noop_caption_sha = sha(identity["calibration_group_id"] + "-noop-caption")
    generation = {
        name: sha(f"{candidate_id}-{name}")
        for name in (
            "candidate_envelope_sha256",
            "generation_receipt_digest",
            "generation_receipt_file_sha256",
            "native_rollout_receipt_digest",
            "native_rollout_receipt_file_sha256",
            "generated_mp4_sha256",
            "geometry_source_video_sha256",
            "candidate_own_caption_utf8_sha256",
            "clean_latent_artifact_sha256",
            "clean_latent_tensor_sha256",
            "official_gaussian_artifact_sha256",
            "official_gaussian_raw_value_sha256",
            "official_gaussian_content_sha256",
        )
    }
    # The whole synthetic cell shares one official Gaussian tensor value.
    generation["official_gaussian_tensor_sha256"] = sha(
        identity["calibration_group_id"] + "-gaussian"
    )
    if identity["semantic_branch"] == contract.ACTION_BRANCH:
        generation["candidate_own_caption_utf8_sha256"] = caption_sha
    elif identity["semantic_branch"] == "noop":
        generation["candidate_own_caption_utf8_sha256"] = noop_caption_sha
    action_id = identity["candidate_id"].rsplit("-", 1)[0] + "-action"
    noop_id = identity["candidate_id"].rsplit("-", 1)[0] + "-noop"
    target = {
        "target_action_candidate_id": action_id,
        "target_noop_candidate_id": noop_id,
        "calibration_group_id": identity["calibration_group_id"],
        "target_action_caption_utf8_sha256": caption_sha,
        "target_noop_caption_utf8_sha256": noop_caption_sha,
    }
    action_prompt_sha = sha(identity["calibration_group_id"] + "-action-prompt")
    noop_prompt_sha = sha(identity["calibration_group_id"] + "-noop-prompt")
    action_condition_sha = sha(identity["calibration_group_id"] + "-action-cond")
    noop_condition_sha = sha(identity["calibration_group_id"] + "-noop-cond")
    prompt = {
        "action_raw_caption_utf8_sha256": caption_sha,
        "noop_raw_caption_utf8_sha256": noop_caption_sha,
        "action_full_prompt_utf8_sha256": action_prompt_sha,
        "noop_full_prompt_utf8_sha256": noop_prompt_sha,
        "action_condition_tensor_sha256": action_condition_sha,
        "noop_condition_tensor_sha256": noop_condition_sha,
        "prompt_builder_contract_digest": sha("prompt-builder"),
        "prompt_pair_digest": contract.object_sha256(
            {
                "action_full_prompt_utf8_sha256": action_prompt_sha,
                "noop_full_prompt_utf8_sha256": noop_prompt_sha,
                "action_condition_tensor_sha256": action_condition_sha,
                "noop_condition_tensor_sha256": noop_condition_sha,
            }
        ),
    }
    model = {
        "frozen_checkpoint_receipt_digest": MODEL_DIGEST,
        "checkpoint_content_manifest_sha256": contract.REQUIRED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "checkpoint_content_binding_digest": sha("checkpoint-binding"),
        "d541801_scorer_source_revision": contract.REQUIRED_D541801_SCORER_REVISION,
        "d541801_scorer_source_sha256": contract.REQUIRED_D541801_SCORER_SHA256,
        "bernini_revision": contract.REQUIRED_BERNINI_REVISION,
        "veomni_revision": contract.REQUIRED_VEOMNI_REVISION,
        "native_schedule_digest": contract.NATIVE_SCHEDULE_DIGEST,
    }
    return generation, target, prompt, model


def make_energy_grid(
    identity: dict,
    generation: dict,
    prompt: dict,
    *,
    margin: float,
    overrides: dict[str, tuple[float, float]] | None = None,
) -> dict:
    overrides = overrides or {}
    result = {}
    for transform_name in contract.TRANSFORM_ORDER:
        if transform_name == contract.CHRONOLOGICAL:
            action_margin, noop_margin = 0.0, 0.0
        else:
            action_margin, noop_margin = overrides.get(
                transform_name, (margin, 0.0)
            )
        rows = []
        for schedule_index, _sigma, _timestep in contract.NATIVE_SIGMA_COORDINATES:
            # Choose values so the log-energy margins are the requested values.
            action_energy = math.exp(action_margin) * (
                1.0 + contract.ENERGY_EPSILON
            ) - contract.ENERGY_EPSILON
            noop_energy = math.exp(noop_margin) * (
                2.0 + contract.ENERGY_EPSILON
            ) - contract.ENERGY_EPSILON
            pair = contract.make_prompt_pair_receipt(
                candidate_id=identity["candidate_id"],
                transform_name=transform_name,
                native_schedule_index=schedule_index,
                transformed_clean_tensor_sha256=(
                    generation["clean_latent_tensor_sha256"]
                    if transform_name == contract.CHRONOLOGICAL
                    else sha(f"{identity['candidate_id']}-{transform_name}-clean")
                ),
                official_gaussian_tensor_sha256=generation[
                    "official_gaussian_tensor_sha256"
                ],
                effective_gaussian_tensor_sha256=(
                    generation["official_gaussian_tensor_sha256"]
                ),
                x_sigma_tensor_sha256=sha(
                    f"{identity['candidate_id']}-{transform_name}-{schedule_index}-x"
                ),
                velocity_target_tensor_sha256=sha(
                    f"{identity['candidate_id']}-{transform_name}-target"
                ),
                action_velocity_tensor_sha256=sha(
                    f"{identity['candidate_id']}-{transform_name}-{schedule_index}-a"
                ),
                noop_velocity_tensor_sha256=sha(
                    f"{identity['candidate_id']}-{transform_name}-{schedule_index}-n"
                ),
                action_full_prompt_sha256=prompt[
                    "action_full_prompt_utf8_sha256"
                ],
                noop_full_prompt_sha256=prompt["noop_full_prompt_utf8_sha256"],
                action_condition_tensor_sha256=prompt[
                    "action_condition_tensor_sha256"
                ],
                noop_condition_tensor_sha256=prompt[
                    "noop_condition_tensor_sha256"
                ],
                frozen_model_receipt_digest=MODEL_DIGEST,
                same_state_execution_proof=make_same_state_proof(
                    f"{identity['candidate_id']}-{transform_name}-{schedule_index}"
                ),
            )
            rows.append(
                {
                    "native_schedule_index": schedule_index,
                    "action_energy": action_energy,
                    "noop_energy": noop_energy,
                    "prompt_pair_receipt": pair,
                }
            )
        result[transform_name] = rows
    return result


def make_score(
    identity: dict,
    *,
    margin: float,
    overrides: dict[str, tuple[float, float]] | None = None,
) -> dict:
    generation, target, prompt, model = make_bindings(identity)
    group_id = GROUP_BY_CANDIDATE.get(
        identity["candidate_id"],
        "sp4-a" if identity["action_family_id"].startswith("dog") else "sp4-b",
    )
    return contract.make_candidate_score_receipt(
        group_id=group_id,
        candidate_identity=identity,
        root_spec_raw_sha256=SPEC_SHA,
        bank_receipt_digest=BANK_DIGEST,
        bank_receipt_file_sha256=BANK_FILE_SHA,
        generation_binding=generation,
        target_action_binding=target,
        prompt_binding=prompt,
        model_binding=model,
        energy_by_transform=make_energy_grid(
            identity,
            generation,
            prompt,
            margin=margin,
            overrides=overrides,
        ),
    )


def make_audit(identity: dict, generation_digest: str, *, ambiguous: bool = False) -> dict:
    positive = identity["semantic_branch"] == contract.ACTION_BRANCH
    return detached_events.seal_event_audit_receipt(
        **identity,
        generation_receipt_digest=generation_digest,
        audit_source_kind="manual_detached",
        external_audit_artifact_sha256=sha(identity["candidate_id"] + "-audit"),
        complete_target_transition_observed=positive,
        terminal_hold_observed=positive,
        full_target_action_observed=positive,
        full_target_action_false_confirmed=(False if positive or ambiguous else True),
    )


def reseal_audit(audit: dict, **overrides: bool) -> dict:
    fields = (
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
        "generation_receipt_digest",
        "audit_source_kind",
        "external_audit_artifact_sha256",
        "complete_target_transition_observed",
        "terminal_hold_observed",
        "full_target_action_observed",
        "full_target_action_false_confirmed",
    )
    values = {name: audit[name] for name in fields}
    values.update(overrides)
    return detached_events.seal_event_audit_receipt(**values)


def make_population(
    *,
    confirmation_action_overrides: dict[str, tuple[float, float]] | None = None,
    ambiguous_candidate_id: str | None = None,
) -> tuple[list[dict], list[dict]]:
    scores = []
    audits = []
    identity_fields = (
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
    )
    for group in SEALED_SPEC["groups"]:
        for candidate in group["candidates"]:
            identity = {name: candidate[name] for name in identity_fields}
            split = identity["analysis_split"]
            branch = identity["semantic_branch"]
            positive_margin = 2.0 if split == "fit" else 1.5
            negative_margin = 0.20 if split == "fit" else 0.30
            overrides = (
                confirmation_action_overrides
                if split == "confirmation"
                and branch == contract.ACTION_BRANCH
                else None
            )
            score = make_score(
                identity,
                margin=(positive_margin if branch == contract.ACTION_BRANCH else negative_margin),
                overrides=overrides,
            )
            self_group = group["group_id"]
            if score["group_id"] != self_group:
                raise AssertionError("synthetic scorer group differs from sealed spec")
            scores.append(score)
            audits.append(
                make_audit(
                    identity,
                    score["generation_binding"]["generation_receipt_digest"],
                    ambiguous=identity["candidate_id"] == ambiguous_candidate_id,
                )
            )
    return scores, audits


def make_group_receipts(scores: list[dict]) -> list[dict]:
    by_id = {row["candidate_identity"]["candidate_id"]: row for row in scores}
    result = []
    for group in SEALED_SPEC["groups"]:
        ordered = [by_id[row["candidate_id"]] for row in group["candidates"]]
        result.append(
            contract.make_group_receipt(
                group_id=group["group_id"],
                candidate_receipts=ordered,
                root_spec_raw_sha256=SPEC_SHA,
                bank_receipt_digest=BANK_DIGEST,
                method_source_revision="3" * 40,
                method_source_archive_sha256=sha("archive"),
                scorer_source_sha256=contract.file_sha256(
                    METHOD_ROOT / "temporal_counterfactual_action_scorer_v1.py"
                ),
                contract_source_sha256=contract.file_sha256(
                    METHOD_ROOT / "temporal_counterfactual_contract_v1.py"
                ),
            )
        )
    return result


def run_calibration(
    scores: list[dict],
    audits: list[dict],
    *,
    preregistration: dict | None = None,
    groups: list[dict] | None = None,
) -> dict:
    return calibration.calibrate_temporal_counterfactual_scores(
        scores,
        audits,
        preregistration or calibration.make_preregistration(),
        groups or make_group_receipts(scores),
        source_bank_spec_sha256=SPEC_SHA,
        source_bank_receipt_digest=BANK_DIGEST,
        calibrator_source_revision="4" * 40,
        calibrator_source_archive_sha256=sha("calibrator-archive"),
        expected_calibrator_source_sha256=contract.file_sha256(
            Path(calibration.__file__)
        ),
    )


class TemporalCounterfactualContractTests(unittest.TestCase):
    def test_pinned_spec_identity_digests_match_contract_constants(self) -> None:
        identity_fields = (
            "candidate_id",
            "analysis_split",
            "action_family_id",
            "calibration_group_id",
            "actor_group_id",
            "scene_group_id",
            "action_group_id",
            "semantic_branch",
        )
        all_identities = []
        for group in SEALED_SPEC["groups"]:
            identities = [
                {name: candidate[name] for name in identity_fields}
                for candidate in group["candidates"]
            ]
            all_identities.extend(identities)
            self.assertEqual(
                contract.object_sha256(identities),
                contract.REQUIRED_CORE4_V2_GROUP_IDENTITY_DIGESTS[
                    group["group_id"]
                ],
            )
        self.assertEqual(
            contract.object_sha256(all_identities),
            contract.REQUIRED_CORE4_V2_CANDIDATE_IDENTITY_DIGEST,
        )

    def test_transform_maps_separate_exact_multisets_from_controls(self) -> None:
        source = tuple(range(contract.LATENT_PHASES))
        plan = contract.validate_transform_plan(contract.make_transform_plan())
        self.assertEqual(plan["transform_order"], list(contract.TRANSFORM_ORDER))
        for name in contract.TRANSFORM_ORDER:
            transformed = contract.apply_temporal_transform_sequence(source, name)
            self.assertEqual(len(transformed), contract.LATENT_PHASES)
            if name in contract.MULTISET_PRESERVING_TRANSFORMS:
                self.assertEqual(sorted(transformed), list(source))
            else:
                self.assertNotEqual(sorted(transformed), list(source))
        self.assertEqual(
            contract.apply_temporal_transform_sequence(source, "reverse"),
            tuple(reversed(source)),
        )
        self.assertNotIn(20, contract.temporal_index_map("transition_loop"))

    def test_transform_plan_mutation_fails_even_if_resealed(self) -> None:
        plan = contract.make_transform_plan()
        mutated = copy.deepcopy(plan)
        mutated["transform_specs"][1]["index_map"][0] = 0
        unsigned = dict(mutated)
        unsigned.pop("receipt_digest")
        mutated["receipt_digest"] = contract.object_sha256(unsigned)
        with self.assertRaisesRegex(
            contract.TemporalCounterfactualContractError, "plan differs"
        ):
            contract.validate_transform_plan(mutated)

    def test_reverse_and_shuffle_keep_exact_same_gaussian_object(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable in CPU contract environment")
        gaussian = torch.arange(
            1 * 16 * contract.LATENT_PHASES * 2 * 2, dtype=torch.float32
        ).reshape(1, 16, contract.LATENT_PHASES, 2, 2)
        clean = (gaussian + 1.0).contiguous()
        original_bytes = gaussian.numpy().tobytes()
        for transform_name in ("reverse", "phase_shuffle"):
            transformed_clean = contract.apply_temporal_transform_tensor(
                clean, transform_name
            )
            effective = contract.fixed_official_gaussian_tensor(
                gaussian, transform_name
            )
            self.assertIs(effective, gaussian)
            self.assertEqual(effective.numpy().tobytes(), original_bytes)
            self.assertFalse(torch.equal(transformed_clean, clean))
        self.assertEqual(gaussian.numpy().tobytes(), original_bytes)

    def test_candidate_receipt_reports_every_transform_and_hard_gate(self) -> None:
        identity = make_identity(
            split="fit",
            family=calibration.ACTION_FAMILY_ORDER[0],
            cell="dog-fit",
            branch="action",
        )
        receipt = contract.validate_candidate_score_receipt(
            make_score(identity, margin=1.25)
        )
        self.assertEqual(
            list(receipt["energy_by_transform"]), list(contract.TRANSFORM_ORDER)
        )
        self.assertEqual(
            list(receipt["transform_contributions"]),
            list(contract.COUNTERFACTUAL_TRANSFORMS),
        )
        self.assertTrue(receipt["hard_gates"]["candidate_hard_gate_passed"])
        self.assertTrue(
            all(
                row["chronological_rank_among_exact_multiset_arms"] == 1
                for row in receipt["chronological_action_energy_rank_by_sigma"]
            )
        )
        self.assertAlmostEqual(receipt["diagnostic_composite_score"], 1.25)
        self.assertFalse(receipt["single_scalar_authorizes_optimizer"])

    def test_candidate_receipt_survives_canonical_json_roundtrip(self) -> None:
        identity = make_identity(
            split="fit",
            family=calibration.ACTION_FAMILY_ORDER[0],
            cell="dog-fit",
            branch="action",
        )
        receipt = make_score(identity, margin=1.25)
        roundtripped = json.loads(contract.canonical_json_bytes(receipt))
        self.assertNotEqual(
            list(roundtripped["energy_by_transform"]),
            list(contract.TRANSFORM_ORDER),
        )
        self.assertEqual(
            contract.validate_candidate_score_receipt(roundtripped), receipt
        )

    def test_cell_scene_matched_noop_binding_cannot_alias_action(self) -> None:
        identity = make_identity(
            split="fit",
            family=calibration.ACTION_FAMILY_ORDER[0],
            cell="dog-fit",
            branch="action",
        )
        generation, target, prompt, model = make_bindings(identity)
        target["target_noop_candidate_id"] = target["target_action_candidate_id"]
        with self.assertRaisesRegex(
            contract.TemporalCounterfactualContractError,
            "target action belongs",
        ):
            contract.make_candidate_score_receipt(
                group_id="sp4-a",
                candidate_identity=identity,
                root_spec_raw_sha256=SPEC_SHA,
                bank_receipt_digest=BANK_DIGEST,
                bank_receipt_file_sha256=BANK_FILE_SHA,
                generation_binding=generation,
                target_action_binding=target,
                prompt_binding=prompt,
                model_binding=model,
                energy_by_transform=make_energy_grid(
                    identity, generation, prompt, margin=1.25
                ),
            )

    def test_multiset_rank_gate_ignores_nonmultiset_energy_ties(self) -> None:
        identity = make_identity(
            split="fit",
            family=calibration.ACTION_FAMILY_ORDER[0],
            cell="dog-fit",
            branch="action",
        )
        receipt = make_score(
            identity,
            margin=1.25,
            overrides={"freeze_first": (0.0, 0.0)},
        )
        self.assertTrue(
            receipt["hard_gates"][
                "chronological_rank1_among_multiset_controls_all_sigmas"
            ]
        )
        for row in receipt["chronological_action_energy_rank_by_sigma"]:
            self.assertEqual(
                row["chronological_tied_exact_multiset_transform_names"],
                [contract.CHRONOLOGICAL],
            )
            self.assertIn(
                "freeze_first", row["action_energy_order_low_to_high"][:2]
            )

    def test_energy_mutation_rejected_after_outer_reseal(self) -> None:
        identity = make_identity(
            split="fit",
            family=calibration.ACTION_FAMILY_ORDER[0],
            cell="dog-fit",
            branch="action",
        )
        receipt = make_score(identity, margin=1.25)
        mutated = copy.deepcopy(receipt)
        mutated["energy_by_transform"]["terminal_only"][0]["action_energy"] += 0.5
        unsigned = dict(mutated)
        unsigned.pop("receipt_digest")
        mutated["receipt_digest"] = contract.object_sha256(unsigned)
        with self.assertRaisesRegex(
            contract.TemporalCounterfactualContractError, "semantics differ"
        ):
            contract.validate_candidate_score_receipt(mutated)

    def test_prompt_pair_rejects_nonpreregistered_sigma(self) -> None:
        with self.assertRaisesRegex(
            contract.TemporalCounterfactualContractError, "not preregistered"
        ):
            contract.make_prompt_pair_receipt(
                candidate_id="candidate",
                transform_name="reverse",
                native_schedule_index=32,
                transformed_clean_tensor_sha256=sha("a"),
                official_gaussian_tensor_sha256=sha("b"),
                effective_gaussian_tensor_sha256=sha("b-effective"),
                x_sigma_tensor_sha256=sha("c"),
                velocity_target_tensor_sha256=sha("d"),
                action_velocity_tensor_sha256=sha("e"),
                noop_velocity_tensor_sha256=sha("f"),
                action_full_prompt_sha256=sha("g"),
                noop_full_prompt_sha256=sha("h"),
                action_condition_tensor_sha256=sha("i"),
                noop_condition_tensor_sha256=sha("j"),
                frozen_model_receipt_digest=sha("k"),
                same_state_execution_proof=make_same_state_proof("unregistered"),
            )

    def test_exact40_vector_conjunction_can_authorize(self) -> None:
        scores, audits = make_population()
        groups = make_group_receipts(scores)
        receipt = run_calibration(scores, audits, groups=groups)
        checked = calibration.validate_calibration_receipt(
            receipt,
            score_receipts=scores,
            event_audit_receipts=audits,
            preregistration=calibration.make_preregistration(),
            group_receipts=groups,
        )
        self.assertTrue(checked["optimizer_authorized"])
        self.assertEqual(
            checked["confirmation_metrics"]["overall"]["positive_recall"], 1.0
        )
        self.assertEqual(
            checked["confirmation_metrics"]["overall"][
                "diagnostic_composite_auroc"
            ],
            1.0,
        )
        self.assertTrue(
            all(
                value == 1.0
                for value in checked["confirmation_metrics"]["overall"][
                    "negative_specificity_by_branch"
                ].values()
            )
        )
        self.assertFalse(checked["single_scalar_authorizes_optimizer"])
        self.assertEqual(
            set(checked["fit_thresholds_by_family_and_transform"]),
            set(calibration.ACTION_FAMILY_ORDER),
        )
        self.assertTrue(
            all(
                set(values) == set(contract.COUNTERFACTUAL_TRANSFORMS)
                for values in checked[
                    "fit_thresholds_by_family_and_transform"
                ].values()
            )
        )
        self.assertNotIn(
            "affine", calibration.make_preregistration()["fit_threshold_rule"]
        )

    def test_go_receipt_has_no_standalone_authority_and_requires_exact_replay(self) -> None:
        scores, audits = make_population()
        preregistration = calibration.make_preregistration()
        groups = make_group_receipts(scores)
        receipt = run_calibration(
            scores, audits, preregistration=preregistration, groups=groups
        )
        with self.assertRaisesRegex(
            calibration.TemporalCounterfactualCalibrationError,
            "requires exact.*replay",
        ):
            calibration.validate_calibration_receipt(receipt)

        forged = copy.deepcopy(receipt)
        forged["confirmation_metrics"]["overall"][
            "diagnostic_composite_auroc"
        ] = 0.75
        unsigned = dict(forged)
        unsigned.pop("receipt_digest")
        forged["receipt_digest"] = contract.object_sha256(unsigned)
        with self.assertRaisesRegex(
            calibration.TemporalCounterfactualCalibrationError,
            "does not reproduce",
        ):
            calibration.validate_calibration_receipt(
                forged,
                score_receipts=scores,
                event_audit_receipts=audits,
                preregistration=preregistration,
                group_receipts=groups,
            )

    def test_go_authority_rejects_non_boolean_truthy_values(self) -> None:
        scores, audits = make_population()
        groups = make_group_receipts(scores)
        receipt = run_calibration(scores, audits, groups=groups)
        self.assertTrue(receipt["optimizer_authorized"])
        for forged_authority in ("true", 1, 2):
            with self.subTest(forged_authority=forged_authority):
                forged = copy.deepcopy(receipt)
                forged["optimizer_authorized"] = forged_authority
                unsigned = dict(forged)
                unsigned.pop("receipt_digest")
                forged["receipt_digest"] = contract.object_sha256(unsigned)
                with self.assertRaisesRegex(
                    calibration.TemporalCounterfactualCalibrationError,
                    "must be a JSON boolean",
                ):
                    calibration.validate_calibration_receipt(forged)

    def test_failure_reasons_require_canonical_string_list(self) -> None:
        scores, audits = make_population()
        groups = make_group_receipts(scores)
        receipt = run_calibration(scores, audits, groups=groups)
        for malformed in (
            "failure",
            [1],
            ["z", "a"],
            ["duplicate", "duplicate"],
        ):
            with self.subTest(malformed=malformed):
                forged = copy.deepcopy(receipt)
                forged["optimizer_authorized"] = False
                forged["failure_reasons"] = malformed
                unsigned = dict(forged)
                unsigned.pop("receipt_digest")
                forged["receipt_digest"] = contract.object_sha256(unsigned)
                with self.assertRaisesRegex(
                    calibration.TemporalCounterfactualCalibrationError,
                    "sorted unique list of nonempty strings",
                ):
                    calibration.validate_calibration_receipt(forged)

    def test_high_auroc_cannot_override_one_failed_transform(self) -> None:
        scores, audits = make_population(
            confirmation_action_overrides={"terminal_only": (0.5, 0.0)}
        )
        receipt = run_calibration(scores, audits)
        self.assertEqual(
            receipt["confirmation_metrics"]["overall"][
                "diagnostic_composite_auroc"
            ],
            1.0,
        )
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertIn("confirmation_vector_metrics", receipt["failure_reasons"])

    def test_reverse_direction_gate_is_independent_of_margin_threshold(self) -> None:
        # prompt-specific reverse margin stays 1.5, but action chronology loses
        # to reverse (-0.1), so the explicit direction gate must fail.
        scores, audits = make_population(
            confirmation_action_overrides={"reverse": (-0.1, -1.6)}
        )
        receipt = run_calibration(scores, audits)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertIn(
            "positive_candidate_hard_gates:confirmation",
            receipt["failure_reasons"],
        )

    def test_ambiguous_detached_label_fails_closed(self) -> None:
        scores, audits = make_population()
        negative = next(
            row
            for row in scores
            if row["candidate_identity"]["analysis_split"] == "confirmation"
            and row["candidate_identity"]["semantic_branch"] == "noop"
        )
        candidate_id = negative["candidate_identity"]["candidate_id"]
        audits = [
            (
                make_audit(
                    negative["candidate_identity"],
                    negative["generation_binding"]["generation_receipt_digest"],
                    ambiguous=True,
                )
                if row["candidate_id"] == candidate_id
                else row
            )
            for row in audits
        ]
        receipt = run_calibration(scores, audits)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertIn(f"event_audit:{candidate_id}", receipt["failure_reasons"])

    def test_every_negative_branch_rejects_target_transition_or_hold(self) -> None:
        scores, audits = make_population()
        by_id = {row["candidate_id"]: row for row in audits}
        for branch in contract.NEGATIVE_BRANCHES:
            score = next(
                row
                for row in scores
                if row["candidate_identity"]["analysis_split"] == "confirmation"
                and row["candidate_identity"]["semantic_branch"] == branch
            )
            audit = by_id[score["candidate_identity"]["candidate_id"]]
            contradictory = reseal_audit(
                audit,
                complete_target_transition_observed=True,
                terminal_hold_observed=True,
            )
            self.assertFalse(calibration._event_contract_passes(contradictory))

        target = next(
            row
            for row in scores
            if row["candidate_identity"]["analysis_split"] == "confirmation"
            and row["candidate_identity"]["semantic_branch"] == "noop"
        )
        target_id = target["candidate_identity"]["candidate_id"]
        contradictory = reseal_audit(
            by_id[target_id],
            complete_target_transition_observed=True,
            terminal_hold_observed=True,
        )
        mutated_audits = [
            contradictory if row["candidate_id"] == target_id else row
            for row in audits
        ]
        receipt = run_calibration(scores, mutated_audits)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertIn(f"event_audit:{target_id}", receipt["failure_reasons"])

    def test_group_receipt_binds_exact_order_and_source_hashes(self) -> None:
        scores, _audits = make_population()
        rows = [row for row in scores if row["group_id"] == "sp4-a"]
        self.assertEqual(len(rows), 20)
        receipt = contract.make_group_receipt(
            group_id="sp4-a",
            candidate_receipts=rows,
            root_spec_raw_sha256=SPEC_SHA,
            bank_receipt_digest=BANK_DIGEST,
            method_source_revision="3" * 40,
            method_source_archive_sha256=sha("archive"),
            scorer_source_sha256=sha("scorer"),
            contract_source_sha256=sha("contract"),
        )
        self.assertEqual(
            contract.validate_group_receipt(
                receipt, candidate_receipts=rows
            )["candidate_count"],
            20,
        )
        self.assertEqual(
            receipt["candidate_identity_digest"],
            contract.REQUIRED_CORE4_V2_GROUP_IDENTITY_DIGESTS["sp4-a"],
        )
        mutated = copy.deepcopy(receipt)
        mutated["candidate_order"][0], mutated["candidate_order"][1] = (
            mutated["candidate_order"][1],
            mutated["candidate_order"][0],
        )
        unsigned = dict(mutated)
        unsigned.pop("receipt_digest")
        mutated["receipt_digest"] = contract.object_sha256(unsigned)
        with self.assertRaisesRegex(
            contract.TemporalCounterfactualContractError,
            "group semantics differ|candidate order/digest join differs",
        ):
            contract.validate_group_receipt(mutated, candidate_receipts=rows)

        replacement = make_score(
            rows[0]["candidate_identity"], margin=9.0
        )
        mutated_rows = [replacement, *rows[1:]]
        with self.assertRaisesRegex(
            contract.TemporalCounterfactualContractError,
            "candidate order/digest join differs",
        ):
            contract.validate_group_receipt(
                receipt, candidate_receipts=mutated_rows
            )

    def test_group_receipt_rejects_resealed_fit_confirmation_identity_swap(self) -> None:
        scores, _audits = make_population()
        rows = [row for row in scores if row["group_id"] == "sp4-a"]
        splits_by_cell = {
            row["candidate_identity"]["calibration_group_id"]: row[
                "candidate_identity"
            ]["analysis_split"]
            for row in rows
        }
        self.assertEqual(set(splits_by_cell.values()), {"fit", "confirmation"})
        swapped_rows = []
        for row in rows:
            identity = dict(row["candidate_identity"])
            identity["analysis_split"] = (
                "confirmation"
                if identity["analysis_split"] == "fit"
                else "fit"
            )
            swapped_rows.append(make_score(identity, margin=1.0))
        self.assertEqual(
            [row["candidate_identity"]["candidate_id"] for row in swapped_rows],
            [row["candidate_identity"]["candidate_id"] for row in rows],
        )
        with self.assertRaisesRegex(
            contract.TemporalCounterfactualContractError,
            "identity mapping differs from formal core4-v2",
        ):
            contract.make_group_receipt(
                group_id="sp4-a",
                candidate_receipts=swapped_rows,
                root_spec_raw_sha256=SPEC_SHA,
                bank_receipt_digest=BANK_DIGEST,
                method_source_revision="3" * 40,
                method_source_archive_sha256=sha("archive"),
                scorer_source_sha256=sha("scorer"),
                contract_source_sha256=sha("contract"),
            )

    def test_t2v_closure_forbids_every_rv2v_use(self) -> None:
        self.assertFalse(contract.T2V_INPUT_CLOSURE["source_video_or_source_latent_consumed"])
        self.assertFalse(
            contract.T2V_INPUT_CLOSURE[
                "rv2v_reference_target_donor_or_noise_consumed"
            ]
        )
        self.assertFalse(
            contract.T2V_INPUT_CLOSURE["t2v_media_or_latent_may_enter_rv2v_training"]
        )


if __name__ == "__main__":
    unittest.main()
