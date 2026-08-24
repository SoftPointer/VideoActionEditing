#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest


try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

if TORCH_AVAILABLE:
    import graft_action_first_source_guided_aggregation_v1 as asga  # noqa: E402
else:
    asga = None  # type: ignore[assignment]


SOURCE_SHA = "1" * 64
CAPTIONER_MODEL_SHA = "2" * 64
CAPTIONER_CODE_SHA = "3" * 64
COMPILER_MODEL_SHA = "4" * 64
COMPILER_CODE_SHA = "5" * 64
TEACHER_SHA = "6" * 64
CHECKPOINT_SHA = "7" * 64
EXTRACTOR_MODEL_SHA = "8" * 64
EXTRACTOR_CODE_SHA = "9" * 64
SCORER_MODEL_SHA = "a" * 64
SCORER_CODE_SHA = "b" * 64
SCORER_CONFIG_SHA = "c" * 64

SOURCE_CAPTION = (
    "A grey dog in a black harness stands in autumn leaves while the camera "
    "and background remain static."
)
INSTRUCTION = "Make the dog sit and hold the seated pose."
TARGET_CAPTION = (
    "A grey dog in a black harness sits in autumn leaves and holds the pose "
    "while the camera and background remain static."
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seal(payload: dict) -> dict:
    owned = dict(payload)
    owned.pop("receipt_digest", None)
    owned["receipt_digest"] = asga.object_sha256(owned)
    return owned


def _captioner_receipt(
    *,
    origin: str = "source_video_frozen_captioner_v1",
    source_caption: str = SOURCE_CAPTION,
) -> dict:
    offline = origin == "sealed_dataset_source_caption_v1"
    return _seal(
        {
            "schema_version": asga.CAPTIONER_RECEIPT_SCHEMA_VERSION,
            "source_caption_origin": origin,
            "source_video_sha256": SOURCE_SHA,
            "source_caption_sha256": _sha(source_caption),
            "captioner_model_digest": CAPTIONER_MODEL_SHA,
            "captioner_code_digest": CAPTIONER_CODE_SHA,
            "source_video_read": True,
            "edit_instruction_read": False,
            "target_video_read": False,
            "proposal_media_read": False,
            "mask_pose_flow_track_read": False,
            "offline_only": offline,
            "online_inference_available": not offline,
            "semantic_correctness_authority": False,
            "same_process_security_boundary": False,
        }
    )


def _compiler_receipt(
    *,
    source_caption: str = SOURCE_CAPTION,
    instruction: str = INSTRUCTION,
    target_caption: str = TARGET_CAPTION,
) -> dict:
    return _seal(
        {
            "schema_version": asga.COMPILER_RECEIPT_SCHEMA_VERSION,
            "source_caption_sha256": _sha(source_caption),
            "instruction_sha256": _sha(instruction),
            "target_caption_sha256": _sha(target_caption),
            "compiler_model_digest": COMPILER_MODEL_SHA,
            "compiler_code_digest": COMPILER_CODE_SHA,
            "compiler_inputs": "source_retelling_and_instruction_only",
            "raw_instruction_retained": True,
            "source_video_read": False,
            "target_video_read": False,
            "proposal_media_read": False,
            "mask_pose_flow_track_read": False,
            "online_inference_available": True,
            "non_action_field_preservation_semantically_verified": False,
            "semantic_correctness_authority": False,
            "same_process_security_boundary": False,
        }
    )


def _retelling(
    *,
    origin: str = "source_video_frozen_captioner_v1",
    captioner_receipt: dict = None,
    compiler_receipt: dict = None,
) -> asga.SourceRetellingBinding:
    if captioner_receipt is None:
        captioner_receipt = _captioner_receipt(origin=origin)
    if compiler_receipt is None:
        compiler_receipt = _compiler_receipt()
    return asga.bind_source_retelling(
        source_video_sha256=SOURCE_SHA,
        source_caption=SOURCE_CAPTION,
        edit_instruction=INSTRUCTION,
        target_caption=TARGET_CAPTION,
        source_caption_origin=origin,
        captioner_receipt=captioner_receipt,
        target_compiler_receipt=compiler_receipt,
    )


def _programs() -> torch.Tensor:
    values = torch.arange(
        asga.CANDIDATE_COUNT * asga.PHASE_COUNT * asga.PROGRAM_WIDTH,
        dtype=torch.float32,
    )
    return values.reshape(
        asga.CANDIDATE_COUNT, asga.PHASE_COUNT, asga.PROGRAM_WIDTH
    ).contiguous()


def _branch_contract() -> dict:
    seeds = tuple(
        tuple(101 + candidate for _ in range(asga.BRANCH_COUNT))
        for candidate in range(asga.CANDIDATE_COUNT)
    )
    gaussians = tuple(
        tuple(_sha(f"gaussian-{candidate}") for _ in range(asga.BRANCH_COUNT))
        for candidate in range(asga.CANDIDATE_COUNT)
    )
    schedule = _sha("exact40-shift5-schedule")
    schedules = tuple(
        tuple(schedule for _ in range(asga.BRANCH_COUNT))
        for _ in range(asga.CANDIDATE_COUNT)
    )
    prompt_row = tuple(
        _sha(f"full-prompt-{branch}")
        for branch in asga.COUNTERFACTUAL_BRANCH_ORDER
    )
    prompts = tuple(prompt_row for _ in range(asga.CANDIDATE_COUNT))
    executions = tuple(
        tuple(
            _sha(f"execution-{candidate}-{branch}")
            for branch in range(asga.BRANCH_COUNT)
        )
        for candidate in range(asga.CANDIDATE_COUNT)
    )
    return {
        "branch_seed_matrix": seeds,
        "branch_gaussian_raw_sha256s": gaussians,
        "branch_schedule_digests": schedules,
        "branch_prompt_digests": prompts,
        "branch_execution_receipt_sha256s": executions,
        "shared_non_action_prompt_digest": _sha("shared-non-action-prompt"),
    }


def _extractor_receipt(
    programs: torch.Tensor,
    retelling: asga.SourceRetellingBinding,
    branch_contract: dict,
) -> dict:
    return _seal(
        {
            "schema_version": asga.EXTRACTOR_RECEIPT_SCHEMA_VERSION,
            "program_coordinate": asga.PROGRAM_COORDINATE,
            "phase_order": list(asga.PHASE_ORDER),
            "counterfactual_branch_order": list(
                asga.COUNTERFACTUAL_BRANCH_ORDER
            ),
            "output_shape": list(programs.shape),
            "output_tensor_sha256": asga.tensor_sha256(
                programs, label="test programs"
            ),
            "output_candidate_slice_sha256s": [
                asga.tensor_sha256(
                    programs[index].contiguous(),
                    label=f"test candidate {index}",
                )
                for index in range(asga.CANDIDATE_COUNT)
            ],
            "input_branch_execution_receipt_sha256s": [
                list(row)
                for row in branch_contract[
                    "branch_execution_receipt_sha256s"
                ]
            ],
            "input_retelling_digest": retelling.digest,
            "extractor_model_digest": EXTRACTOR_MODEL_SHA,
            "extractor_code_digest": EXTRACTOR_CODE_SHA,
            "allowed_inputs": list(asga.PROGRAM_EXTRACTOR_ALLOWED_INPUTS),
            "action_program_only_observation": True,
            "proposal_rgb_read": False,
            "proposal_latent_read": False,
            "raw_velocity_read": False,
            "target_video_read": False,
            "mask_read": False,
            "pose_read": False,
            "flow_read": False,
            "track_read": False,
            "semantic_correctness_authority": False,
            "same_process_security_boundary": False,
        }
    )


def _bank(
    *,
    programs: torch.Tensor = None,
    retelling: asga.SourceRetellingBinding = None,
    branch_contract: dict = None,
    extractor_receipt: dict = None,
) -> asga.AuthenticatedProposalBank:
    if programs is None:
        programs = _programs()
    if retelling is None:
        retelling = _retelling()
    if branch_contract is None:
        branch_contract = _branch_contract()
    if extractor_receipt is None:
        extractor_receipt = _extractor_receipt(
            programs, retelling, branch_contract
        )
    return asga.authenticate_proposal_bank(
        programs,
        branch_seed_matrix=branch_contract["branch_seed_matrix"],
        branch_gaussian_raw_sha256s=branch_contract[
            "branch_gaussian_raw_sha256s"
        ],
        branch_schedule_digests=branch_contract["branch_schedule_digests"],
        branch_prompt_digests=branch_contract["branch_prompt_digests"],
        branch_execution_receipt_sha256s=branch_contract[
            "branch_execution_receipt_sha256s"
        ],
        shared_non_action_prompt_digest=branch_contract[
            "shared_non_action_prompt_digest"
        ],
        retelling=retelling,
        frozen_teacher_receipt_sha256=TEACHER_SHA,
        checkpoint_digest=CHECKPOINT_SHA,
        extractor_receipt=extractor_receipt,
    )


def _raw_scores() -> torch.Tensor:
    row = torch.tensor(
        [0.8, 0.1, 0.0, 0.2, 0.6, 0.4, 0.3, 0.2],
        dtype=torch.float32,
    )
    return row.unsqueeze(0).repeat(asga.CANDIDATE_COUNT, 1).contiguous()


def _scorer_receipt(
    bank: asga.AuthenticatedProposalBank,
    raw_scores: torch.Tensor,
    compatibility: torch.Tensor,
) -> dict:
    return _seal(
        {
            "schema_version": asga.SCORER_RECEIPT_SCHEMA_VERSION,
            "proposal_bank_digest": bank.provenance.digest,
            "retelling_digest": bank.retelling.digest,
            "raw_score_names": list(asga.RAW_SCORE_NAMES),
            "margin_names": list(asga.MARGIN_NAMES),
            "margin_thresholds_float_hex": [
                float(value).hex() for value in asga.MARGIN_THRESHOLDS
            ],
            "source_compatibility_kind": asga.SOURCE_COMPATIBILITY_KIND,
            "normalized_score_range_float_hex": [
                float(asga.NORMALIZED_SCORE_MIN).hex(),
                float(asga.NORMALIZED_SCORE_MAX).hex(),
            ],
            "raw_scores_tensor_sha256": asga.tensor_sha256(
                raw_scores, label="test raw scores"
            ),
            "source_compatibility_tensor_sha256": asga.tensor_sha256(
                compatibility, label="test source compatibility"
            ),
            "scorer_model_digest": SCORER_MODEL_SHA,
            "scorer_code_digest": SCORER_CODE_SHA,
            "scorer_config_digest": SCORER_CONFIG_SHA,
            "allowed_inputs": list(asga.SCORER_ALLOWED_INPUTS),
            "target_video_read": False,
            "proposal_rgb_read": False,
            "proposal_latent_read": False,
            "raw_velocity_read": False,
            "mask_read": False,
            "pose_read": False,
            "flow_read": False,
            "track_read": False,
            "evaluator_selected_index_read": False,
            "semantic_correctness_authority": False,
            "same_process_security_boundary": False,
        }
    )


def _evidence(
    bank: asga.AuthenticatedProposalBank,
    *,
    raw_scores: torch.Tensor = None,
    compatibility: torch.Tensor = None,
    scorer_receipt: dict = None,
) -> asga.AuthenticatedASGAEvidence:
    if raw_scores is None:
        raw_scores = _raw_scores()
    if compatibility is None:
        compatibility = torch.linspace(
            0.0, 0.04, asga.CANDIDATE_COUNT, dtype=torch.float32
        ).contiguous()
    if scorer_receipt is None:
        scorer_receipt = _scorer_receipt(bank, raw_scores, compatibility)
    return asga.authenticate_asga_evidence(
        bank,
        raw_scores=raw_scores,
        source_compatibility=compatibility,
        scorer_receipt=scorer_receipt,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class GraftASGATests(unittest.TestCase):
    def test_retelling_receipts_bind_io_models_and_online_closure(self) -> None:
        binding = _retelling()
        binding.validate()
        self.assertTrue(binding.online_inference_closed)
        self.assertFalse(binding.offline_only)
        self.assertFalse(binding.non_action_field_preservation_semantically_verified)
        self.assertFalse(binding.semantic_correctness_authority)
        self.assertFalse(binding.same_process_security_boundary)

        bad_captioner = _captioner_receipt()
        bad_captioner["source_video_sha256"] = "f" * 64
        bad_captioner = _seal(bad_captioner)
        with self.assertRaises(asga.GraftASGAError):
            _retelling(captioner_receipt=bad_captioner)

        bad_compiler = _compiler_receipt()
        bad_compiler["target_caption_sha256"] = "e" * 64
        bad_compiler = _seal(bad_compiler)
        with self.assertRaises(asga.GraftASGAError):
            _retelling(compiler_receipt=bad_compiler)

    def test_dataset_caption_is_explicitly_offline_only(self) -> None:
        origin = "sealed_dataset_source_caption_v1"
        binding = _retelling(origin=origin)
        self.assertTrue(binding.offline_only)
        self.assertFalse(binding.online_inference_closed)
        captioner = json.loads(binding.captioner_receipt_json)
        self.assertTrue(captioner["offline_only"])
        self.assertFalse(captioner["online_inference_available"])

    def test_retelling_receipts_reject_forbidden_reads_after_reseal(self) -> None:
        captioner = _captioner_receipt()
        captioner["target_video_read"] = True
        captioner = _seal(captioner)
        with self.assertRaises(asga.GraftASGAError):
            _retelling(captioner_receipt=captioner)

        compiler = _compiler_receipt()
        compiler["proposal_media_read"] = True
        compiler = _seal(compiler)
        with self.assertRaises(asga.GraftASGAError):
            _retelling(compiler_receipt=compiler)

    def test_bank_binds_k5_p4_phases_branches_and_slice_hashes(self) -> None:
        bank = _bank()
        bank.validate()
        provenance = bank.provenance
        self.assertEqual(tuple(bank.tensor.shape), (5, 4, 32))
        self.assertEqual(provenance.phase_order, asga.PHASE_ORDER)
        self.assertEqual(
            provenance.counterfactual_branch_order,
            ("action", "noop", "reverse", "incomplete"),
        )
        self.assertEqual(len(provenance.candidate_slice_sha256s), 5)
        self.assertTrue(provenance.same_seed_within_each_counterfactual_group)
        self.assertTrue(provenance.same_gaussian_within_each_counterfactual_group)
        self.assertFalse(provenance.proposal_rgb_or_latent_included)
        self.assertFalse(provenance.raw_velocity_included)
        self.assertFalse(provenance.semantic_correctness_authority)

    def test_branch_seed_gaussian_schedule_prompt_and_receipt_mismatches_fail(
        self,
    ) -> None:
        cases = []

        contract = _branch_contract()
        rows = [list(row) for row in contract["branch_seed_matrix"]]
        rows[0][1] += 1
        contract["branch_seed_matrix"] = rows
        cases.append(contract)

        contract = _branch_contract()
        rows = [list(row) for row in contract["branch_seed_matrix"]]
        rows[1] = list(rows[0])
        contract["branch_seed_matrix"] = rows
        cases.append(contract)

        contract = _branch_contract()
        rows = [list(row) for row in contract["branch_gaussian_raw_sha256s"]]
        rows[0][2] = _sha("wrong-gaussian")
        contract["branch_gaussian_raw_sha256s"] = rows
        cases.append(contract)

        contract = _branch_contract()
        rows = [list(row) for row in contract["branch_gaussian_raw_sha256s"]]
        rows[1] = list(rows[0])
        contract["branch_gaussian_raw_sha256s"] = rows
        cases.append(contract)

        contract = _branch_contract()
        rows = [list(row) for row in contract["branch_schedule_digests"]]
        rows[2][3] = _sha("wrong-schedule")
        contract["branch_schedule_digests"] = rows
        cases.append(contract)

        contract = _branch_contract()
        rows = [list(row) for row in contract["branch_prompt_digests"]]
        rows[3][0] = _sha("wrong-prompt")
        contract["branch_prompt_digests"] = rows
        cases.append(contract)

        contract = _branch_contract()
        rows = [list(row) for row in contract["branch_execution_receipt_sha256s"]]
        rows[4][3] = rows[0][0]
        contract["branch_execution_receipt_sha256s"] = rows
        cases.append(contract)

        for index, bad_contract in enumerate(cases):
            with self.subTest(index=index):
                programs = _programs()
                retelling = _retelling()
                extractor = _extractor_receipt(
                    programs, retelling, bad_contract
                )
                with self.assertRaises(asga.GraftASGAError):
                    _bank(
                        programs=programs,
                        retelling=retelling,
                        branch_contract=bad_contract,
                        extractor_receipt=extractor,
                    )

    def test_extractor_receipt_binds_coordinate_slices_and_forbidden_inputs(
        self,
    ) -> None:
        programs = _programs()
        retelling = _retelling()
        contract = _branch_contract()
        for field, value in (
            ("output_tensor_sha256", "f" * 64),
            (
                "output_candidate_slice_sha256s",
                ["f" * 64] * asga.CANDIDATE_COUNT,
            ),
            ("phase_order", ["hold", "terminal", "transition", "onset"]),
            ("raw_velocity_read", True),
            ("proposal_rgb_read", True),
        ):
            with self.subTest(field=field):
                receipt = _extractor_receipt(programs, retelling, contract)
                receipt[field] = value
                receipt = _seal(receipt)
                with self.assertRaises(asga.GraftASGAError):
                    _bank(
                        programs=programs,
                        retelling=retelling,
                        branch_contract=contract,
                        extractor_receipt=receipt,
                    )

    def test_fixed_raw_score_schema_computes_all_seven_margins(self) -> None:
        bank = _bank()
        evidence = _evidence(bank)
        expected = torch.tensor(
            [0.7, 0.8, 0.6, 0.6, 0.4, 0.3, 0.2],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(evidence.margins[0], expected))
        self.assertNotIn(
            "margins",
            inspect.signature(asga.authenticate_asga_evidence).parameters,
        )

    def test_each_of_seven_axes_is_a_strict_noncompensable_gate(self) -> None:
        bank = _bank()
        raw_index = {name: index for index, name in enumerate(asga.RAW_SCORE_NAMES)}
        mutations = {
            "action_minus_noop": (
                "noop_target_event_score",
                "action_event_score",
            ),
            "action_minus_reverse": (
                "reverse_target_event_score",
                "action_event_score",
            ),
            "action_minus_incomplete": (
                "incomplete_target_event_score",
                "action_event_score",
            ),
            "terminal_hold": ("terminal_hold_score", None),
            "actor_preservation": ("actor_preservation_delta", None),
            "camera_preservation": ("camera_preservation_delta", None),
            "background_preservation": ("background_preservation_delta", None),
        }
        for margin_name, (field, copy_from) in mutations.items():
            with self.subTest(margin=margin_name):
                scores = _raw_scores()
                if copy_from is None:
                    scores[0, raw_index[field]] = 0.0
                else:
                    scores[0, raw_index[field]] = scores[0, raw_index[copy_from]]
                compatibility = torch.tensor(
                    [1.0, 0.0, -0.2, -0.3, -0.4], dtype=torch.float32
                )
                decision = asga.select_action_programs(
                    bank,
                    _evidence(
                        bank,
                        raw_scores=scores.contiguous(),
                        compatibility=compatibility,
                    ),
                )
                self.assertNotIn(0, decision.feasible_indices)
                self.assertIn(1, decision.feasible_indices)
                self.assertEqual(float(decision.weights[0]), 0.0)

    def test_source_like_noop_cannot_win_after_action_gate(self) -> None:
        bank = _bank()
        scores = _raw_scores()
        action = asga.RAW_SCORE_NAMES.index("action_event_score")
        noop = asga.RAW_SCORE_NAMES.index("noop_target_event_score")
        scores[0, noop] = scores[0, action]
        compatibility = torch.tensor(
            [1.0, 0.0, -0.2, -0.4, -0.8], dtype=torch.float32
        )
        decision = asga.select_action_programs(
            bank,
            _evidence(
                bank,
                raw_scores=scores.contiguous(),
                compatibility=compatibility,
            ),
        )
        self.assertNotIn(0, decision.feasible_indices)
        self.assertEqual(float(decision.weights[0]), 0.0)

    def test_equal_cosine_aggregates_programs_not_media_or_velocity(self) -> None:
        programs = torch.zeros(5, 4, 32, dtype=torch.float32)
        programs[0].fill_(1.0)
        programs[1].fill_(3.0)
        bank = _bank(programs=programs.contiguous())
        scores = _raw_scores()
        terminal = asga.RAW_SCORE_NAMES.index("terminal_hold_score")
        scores[2:, terminal] = -0.1
        compatibility = torch.zeros(5, dtype=torch.float32)
        decision = asga.select_action_programs(
            bank,
            _evidence(
                bank,
                raw_scores=scores.contiguous(),
                compatibility=compatibility,
            ),
        )
        self.assertEqual(decision.feasible_indices, (0, 1))
        self.assertTrue(
            torch.equal(
                decision.aggregated_program,
                torch.full((4, 32), 2.0, dtype=torch.float32),
            )
        )
        self.assertFalse(decision.receipt["proposal_rgb_aggregated"])
        self.assertFalse(decision.receipt["proposal_latent_aggregated"])
        self.assertFalse(decision.receipt["raw_velocity_aggregated"])
        self.assertTrue(decision.receipt["action_program_aggregated"])

    def test_empty_feasible_set_abstains_and_never_authorizes_update(self) -> None:
        bank = _bank()
        scores = _raw_scores()
        terminal = asga.RAW_SCORE_NAMES.index("terminal_hold_score")
        scores[:, terminal] = 0.0
        decision = asga.select_action_programs(
            bank, _evidence(bank, raw_scores=scores.contiguous())
        )
        self.assertTrue(decision.abstained)
        self.assertIsNone(decision.aggregated_program)
        self.assertTrue(torch.equal(decision.weights, torch.zeros(5)))
        self.assertFalse(decision.receipt["optimizer_update_authorized"])
        self.assertTrue(
            decision.receipt["external_sealed_training_authority_required"]
        )
        self.assertFalse(decision.receipt["best_of_k_fallback_used"])
        decision.validate()

    def test_normalized_cosine_range_and_extreme_softmax_are_stable(self) -> None:
        bank = _bank()
        compatibility = torch.tensor(
            [1.0, -1.0, 0.5, -0.5, 0.0], dtype=torch.float32
        )
        decision = asga.select_action_programs(
            bank, _evidence(bank, compatibility=compatibility)
        )
        self.assertTrue(bool(torch.isfinite(decision.weights).all().item()))
        self.assertAlmostEqual(float(decision.weights.sum()), 1.0, places=6)

        for invalid in (1.0001, -1.0001):
            with self.subTest(invalid=invalid):
                bad = compatibility.clone()
                bad[0] = invalid
                with self.assertRaises(asga.GraftASGAError):
                    _evidence(bank, compatibility=bad.contiguous())

        bad = compatibility.clone()
        bad[0] = float("nan")
        with self.assertRaises(asga.GraftASGAError):
            _evidence(bank, compatibility=bad.contiguous())

        for invalid in (1.0001, -1.0001, float("nan")):
            with self.subTest(raw_score=invalid):
                raw = _raw_scores()
                raw[0, 0] = invalid
                with self.assertRaises(asga.GraftASGAError):
                    _evidence(bank, raw_scores=raw.contiguous())

    def test_scorer_receipt_rejects_every_forbidden_channel(self) -> None:
        bank = _bank()
        raw = _raw_scores()
        compatibility = torch.zeros(5, dtype=torch.float32)
        forbidden = (
            "target_video_read",
            "proposal_rgb_read",
            "proposal_latent_read",
            "raw_velocity_read",
            "mask_read",
            "pose_read",
            "flow_read",
            "track_read",
            "evaluator_selected_index_read",
            "semantic_correctness_authority",
            "same_process_security_boundary",
        )
        for field in forbidden:
            with self.subTest(field=field):
                receipt = _scorer_receipt(bank, raw, compatibility)
                receipt[field] = True
                receipt = _seal(receipt)
                with self.assertRaises(asga.GraftASGAError):
                    _evidence(
                        bank,
                        raw_scores=raw,
                        compatibility=compatibility,
                        scorer_receipt=receipt,
                    )

        receipt = _scorer_receipt(bank, raw, compatibility)
        receipt["allowed_inputs"] = receipt["allowed_inputs"][:-1]
        receipt = _seal(receipt)
        with self.assertRaises(asga.GraftASGAError):
            _evidence(
                bank,
                raw_scores=raw,
                compatibility=compatibility,
                scorer_receipt=receipt,
            )

    def test_cross_bank_evidence_is_rejected(self) -> None:
        first = _bank()
        programs = _programs()
        programs[0, 0, 0] += 1.0
        second = _bank(programs=programs.contiguous())
        evidence = _evidence(first)
        with self.assertRaises(asga.GraftASGAError):
            asga.select_action_programs(second, evidence)

    def test_selection_uses_independent_snapshots(self) -> None:
        bank = _bank()
        evidence = _evidence(bank)
        decision = asga.select_action_programs(bank, evidence)
        bank.tensor[0, 0, 0] += 99.0
        evidence.raw_scores[0, 0] = -1.0
        decision.validate()

    def test_receipt_authority_and_program_forgery_fail_after_digest_reseal(
        self,
    ) -> None:
        for field in (
            "semantic_action_success_authority",
            "identity_preservation_authority",
            "target_video_used",
            "proposal_rgb_used",
            "raw_velocity_used",
            "same_process_security_boundary",
        ):
            with self.subTest(field=field):
                bank = _bank()
                decision = asga.select_action_programs(bank, _evidence(bank))
                decision.receipt[field] = True
                decision.receipt = _seal(decision.receipt)
                with self.assertRaises(asga.GraftASGAError):
                    decision.validate()

        bank = _bank()
        decision = asga.select_action_programs(bank, _evidence(bank))
        decision.aggregated_program.fill_(123.0)
        decision.receipt["aggregated_program_sha256"] = asga.tensor_sha256(
            decision.aggregated_program, label="forged aggregate"
        )
        decision.receipt = _seal(decision.receipt)
        with self.assertRaises(asga.GraftASGAError):
            decision.validate()

        bank = _bank()
        decision = asga.select_action_programs(bank, _evidence(bank))
        decision.weights[0] = -1.0
        decision.receipt["weights_float_hex"][0] = float(-1.0).hex()
        decision.receipt = _seal(decision.receipt)
        with self.assertRaises(asga.GraftASGAError):
            decision.validate()

    def test_live_tensor_mutation_and_invalid_tensor_forms_fail(self) -> None:
        bank = _bank()
        bank.tensor[0, 0, 0] += 1.0
        with self.assertRaises(asga.GraftASGAError):
            bank.validate()

        bank = _bank()
        evidence = _evidence(bank)
        evidence.margins[0, 0] = -9.0
        with self.assertRaises(asga.GraftASGAError):
            evidence.validate()

        programs = _programs()
        noncontiguous = programs.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaises(asga.GraftASGAError):
            _bank(programs=noncontiguous)

        raw = _raw_scores().requires_grad_(True)
        with self.assertRaises(asga.GraftASGAError):
            _evidence(_bank(), raw_scores=raw)

    def test_receipt_retains_all_candidates_retellings_and_false_authority(
        self,
    ) -> None:
        bank = _bank()
        receipt = asga.select_action_programs(bank, _evidence(bank)).receipt
        self.assertEqual(receipt["source_caption"], SOURCE_CAPTION)
        self.assertEqual(receipt["target_caption"], TARGET_CAPTION)
        self.assertEqual(len(receipt["branch_seed_matrix"]), 5)
        self.assertEqual(len(receipt["branch_execution_receipt_sha256s"]), 5)
        self.assertEqual(len(receipt["raw_score_values_float_hex"]), 5)
        self.assertEqual(len(receipt["margin_values_float_hex"]), 5)
        self.assertTrue(receipt["all_failed_candidates_retained"])
        self.assertFalse(receipt["optimizer_update_authorized"])
        self.assertFalse(receipt["upstream_receipt_semantic_authority"])
        self.assertFalse(receipt["same_process_security_boundary"])
        self.assertFalse(receipt["dynaedit_official_reproduction_claimed"])

    def test_public_api_has_no_media_velocity_or_selected_index_channels(self) -> None:
        forbidden = {
            "proposal_rgb",
            "proposal_video",
            "proposal_latent",
            "raw_velocity",
            "target_video",
            "mask",
            "pose",
            "flow",
            "track",
            "selected_index",
            "margins",
        }
        for function in (
            asga.bind_source_retelling,
            asga.authenticate_proposal_bank,
            asga.authenticate_asga_evidence,
            asga.select_action_programs,
        ):
            self.assertTrue(
                forbidden.isdisjoint(inspect.signature(function).parameters)
            )


if __name__ == "__main__":
    unittest.main()
