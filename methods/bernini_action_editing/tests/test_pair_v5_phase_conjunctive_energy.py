from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import pair_v5_phase_conjunctive_energy as phase_energy  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    phase_energy = None


def _weights():
    # Disjoint windows make failures easy to localize in tests while their
    # union covers all 21 latent phases.  The production registry may overlap.
    windows = {
        "actor": range(0, 5),
        "direction": range(5, 9),
        "contact": range(9, 13),
        "order": range(13, 17),
        "terminal": range(17, 21),
    }
    result = {}
    for milestone in phase_energy.MILESTONE_ORDER:
        indices = list(windows[milestone])
        vector = [0.0] * phase_energy.LATENT_PHASES
        for index in indices:
            vector[index] = 1.0 / len(indices)
        result[milestone] = vector
    return result


def _fixture(*, batch=1, height=2, width=2):
    clean = torch.zeros(
        batch,
        phase_energy.LATENT_CHANNELS,
        phase_energy.LATENT_PHASES,
        height,
        width,
        dtype=torch.float32,
    )
    epsilon = torch.ones_like(clean)
    sigma = torch.tensor(0.35, dtype=torch.float32)
    velocity = epsilon - clean
    predictions = {
        branch: velocity + 2.0 for branch in phase_energy.BRANCH_ORDER
    }
    predictions["action"] = velocity + 0.1
    commitment = phase_energy.make_phase_weight_commitment(_weights())
    return clean, epsilon, sigma, predictions, commitment


def _evaluate(values):
    clean, epsilon, sigma, predictions, commitment = values
    return phase_energy.evaluate_phase_conjunctive_energy(
        clean,
        epsilon,
        sigma,
        predictions,
        commitment,
        registered_phase_weight_digest=commitment["registration_digest"],
        frozen_t2v_receipt_digest="a" * 64,
    )


@unittest.skipIf(torch is None, "torch is unavailable")
class PhaseConjunctionRegressionTests(unittest.TestCase):
    def test_terminal_failure_is_not_hidden_by_global_easy_negative(self) -> None:
        values = list(_fixture())
        velocity = values[1] - values[0]
        # Action is excellent for 17/21 phases but fails throughout terminal.
        action = velocity + 0.1
        action[:, :, 17:21] = velocity[:, :, 17:21] + 2.0
        values[3]["action"] = action
        # Incomplete is globally worse (error 1 > action mean error ~.77), but
        # it explains the failed terminal better (1 < 4).
        values[3]["incomplete"] = velocity + 1.0

        result = _evaluate(values)
        terminal = phase_energy.MILESTONE_ORDER.index("terminal")
        incomplete = phase_energy.HARD_NEGATIVE_BRANCHES.index("incomplete")
        self.assertGreater(float(result.global_reward.item()), 0.0)
        self.assertLess(float(result.reward.item()), 0.0)
        self.assertEqual(int(result.hardest_milestone_index.item()), terminal)
        self.assertEqual(int(result.hardest_negative_index.item()), incomplete)
        self.assertEqual(
            result.reward.item(),
            result.milestone_negative_log_energy_ratios[
                terminal, incomplete, 0
            ].item(),
        )

    def test_reverse_order_failure_survives_good_global_margin(self) -> None:
        values = list(_fixture())
        velocity = values[1] - values[0]
        action = velocity + 0.1
        action[:, :, 13:17] = velocity[:, :, 13:17] + 1.2
        values[3]["action"] = action
        reverse = velocity + 2.0
        reverse[:, :, 13:17] = velocity[:, :, 13:17] + 0.7
        values[3]["reverse"] = reverse

        result = _evaluate(values)
        order = phase_energy.MILESTONE_ORDER.index("order")
        reverse_index = phase_energy.HARD_NEGATIVE_BRANCHES.index("reverse")
        self.assertGreater(float(result.global_reward.item()), 0.0)
        self.assertLess(float(result.reward.item()), 0.0)
        self.assertEqual(int(result.hardest_milestone_index.item()), order)
        self.assertEqual(int(result.hardest_negative_index.item()), reverse_index)

    def test_result_shapes_and_full_conjunction_are_exact(self) -> None:
        result = _evaluate(_fixture(batch=2))
        self.assertEqual(
            tuple(result.per_phase_branch_energies.shape),
            (len(phase_energy.BRANCH_ORDER), 2, 21),
        )
        self.assertEqual(
            tuple(result.milestone_branch_energies.shape),
            (len(phase_energy.MILESTONE_ORDER), len(phase_energy.BRANCH_ORDER), 2),
        )
        self.assertEqual(
            tuple(result.milestone_negative_log_energy_ratios.shape),
            (
                len(phase_energy.MILESTONE_ORDER),
                len(phase_energy.HARD_NEGATIVE_BRANCHES),
                2,
            ),
        )
        expected = result.milestone_negative_log_energy_ratios.reshape(-1, 2).min(
            dim=0
        ).values
        self.assertTrue(torch.equal(result.reward, expected))
        for tensor in (
            result.x_sigma,
            result.velocity_label,
            result.per_phase_branch_energies,
            result.milestone_branch_energies,
            result.milestone_negative_log_energy_ratios,
            result.reward,
        ):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertFalse(tensor.requires_grad)


@unittest.skipIf(torch is None, "torch is unavailable")
class ClosureAndCommitmentTests(unittest.TestCase):
    def test_required_negative_registry_is_closed(self) -> None:
        expected = (
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
        self.assertEqual(phase_energy.HARD_NEGATIVE_BRANCHES, expected)
        self.assertTrue(
            phase_energy.REQUIRED_CAUSAL_NEGATIVES.issubset(expected)
        )

        values = list(_fixture())
        values[3] = dict(values[3])
        values[3].pop("shuffle")
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "missing=.*shuffle"
        ):
            _evaluate(values)

        values = list(_fixture())
        values[3] = dict(values[3])
        values[3]["unregistered_easy"] = values[3]["noop"]
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "extra=.*unregistered_easy"
        ):
            _evaluate(values)

    def test_phase_registry_requires_closure_normalization_and_coverage(self) -> None:
        missing = _weights()
        missing.pop("contact")
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "missing=.*contact"
        ):
            phase_energy.make_phase_weight_commitment(missing)

        unnormalized = _weights()
        unnormalized["terminal"][17] += 0.2
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "must sum to one"
        ):
            phase_energy.make_phase_weight_commitment(unnormalized)

        uncovered = _weights()
        # Remove phase zero while preserving actor normalization at phase one.
        uncovered["actor"][1] += uncovered["actor"][0]
        uncovered["actor"][0] = 0.0
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "uncovered=.*0"
        ):
            phase_energy.make_phase_weight_commitment(uncovered)

        negative = _weights()
        negative["actor"][0] = -0.1
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "nonnegative"
        ):
            phase_energy.make_phase_weight_commitment(negative)

    def test_embedded_and_external_digest_both_fail_closed_on_tamper(self) -> None:
        values = list(_fixture())
        original_digest = values[4]["registration_digest"]
        replacement_weights = _weights()
        replacement_weights["terminal"][17] = 0.20
        replacement_weights["terminal"][18] = 0.30
        replacement = phase_energy.make_phase_weight_commitment(
            replacement_weights
        )
        tampered = deepcopy(replacement)
        tampered["registration_digest"] = original_digest
        values[4] = tampered
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "embedded digest mismatch"
        ):
            _evaluate(values)

        # Even a semantically valid, fully re-sealed replacement cannot pass
        # the digest the scheduler pinned before rollout.
        clean, epsilon, sigma, predictions, _ = _fixture()
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "pre-registered digest"
        ):
            phase_energy.evaluate_phase_conjunctive_energy(
                clean,
                epsilon,
                sigma,
                predictions,
                replacement,
                registered_phase_weight_digest=original_digest,
                frozen_t2v_receipt_digest="a" * 64,
            )

    def test_receipt_digest_and_inner_closure_detect_tampering(self) -> None:
        result = _evaluate(_fixture())
        receipt = dict(result.receipt)
        digest = receipt.pop("receipt_digest")
        self.assertEqual(digest, phase_energy.object_sha256(receipt))

        tampered = deepcopy(result.receipt)
        tampered["conjunction_policy"] = "mean"
        with self.assertRaises(phase_energy.PairV5PhaseEnergyError):
            phase_energy.validate_evaluation_receipt(tampered)

        extra = deepcopy(result.receipt)
        extra["source_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "extra=.*source_sha256"
        ):
            phase_energy.validate_evaluation_receipt(extra)

    def test_mapping_order_does_not_change_evaluation_receipt(self) -> None:
        values = list(_fixture())
        forward = _evaluate(values)
        values[3] = dict(reversed(list(values[3].items())))
        reverse = _evaluate(values)
        self.assertEqual(
            phase_energy.canonical_json_bytes(forward.receipt),
            phase_energy.canonical_json_bytes(reverse.receipt),
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class SharedStateAndNoPrivilegeTests(unittest.TestCase):
    def test_api_has_one_shared_noise_sigma_and_no_privileged_slots(self) -> None:
        parameters = set(
            inspect.signature(
                phase_energy.evaluate_phase_conjunctive_energy
            ).parameters
        )
        self.assertIn("epsilon", parameters)
        self.assertIn("sigma", parameters)
        self.assertNotIn("action_epsilon", parameters)
        self.assertNotIn("negative_epsilon", parameters)
        self.assertTrue(
            parameters.isdisjoint(phase_energy.FORBIDDEN_EXTERNAL_INPUT_NAMES)
        )
        values = _fixture()
        with self.assertRaises(TypeError):
            phase_energy.evaluate_phase_conjunctive_energy(
                *values,
                registered_phase_weight_digest=values[4]["registration_digest"],
                frozen_t2v_receipt_digest="a" * 64,
                proposal=torch.zeros(1),
            )

    def test_candidate_own_state_is_constructed_from_single_packet(self) -> None:
        values = _fixture(batch=2)
        result = _evaluate(values)
        clean, epsilon, sigma = values[:3]
        self.assertTrue(
            torch.equal(
                result.x_sigma,
                (1.0 - sigma) * clean + sigma * epsilon,
            )
        )
        self.assertTrue(torch.equal(result.velocity_label, epsilon - clean))
        self.assertFalse(result.receipt["proposal_visual_data_consumed"])
        self.assertFalse(result.receipt["privileged_visual_inputs_consumed"])
        self.assertEqual(
            result.receipt["frozen_t2v_receipt_digest"], "a" * 64
        )

    def test_predictions_must_be_detached_fp32_exact81(self) -> None:
        values = list(_fixture())
        values[3] = dict(values[3])
        values[3]["action"] = values[3]["action"].requires_grad_(True)
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "detached FP32"
        ):
            _evaluate(values)

        values = list(_fixture())
        values[3] = dict(values[3])
        values[3]["reverse"] = values[3]["reverse"].bfloat16()
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "detached FP32"
        ):
            _evaluate(values)

        values = list(_fixture())
        values[0] = values[0][:, :, :20]
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "exact81"
        ):
            _evaluate(values)

    def test_frozen_model_receipt_digest_is_mandatory(self) -> None:
        values = _fixture()
        with self.assertRaisesRegex(
            phase_energy.PairV5PhaseEnergyError, "lowercase SHA-256"
        ):
            phase_energy.evaluate_phase_conjunctive_energy(
                *values,
                registered_phase_weight_digest=values[4]["registration_digest"],
                frozen_t2v_receipt_digest="not-a-digest",
            )

    def test_contract_receipt_binds_min_conjunction_and_no_visual_leak(self) -> None:
        receipt = phase_energy.contract_receipt()
        digest = receipt.pop("digest")
        self.assertEqual(digest, phase_energy.object_sha256(receipt))
        self.assertEqual(
            receipt["conjunction"],
            "minimum_over_every_milestone_x_hard_negative_log_energy_margin",
        )
        self.assertFalse(receipt["proposal_visual_data_consumed"])
        self.assertFalse(receipt["privileged_visual_inputs_consumed"])


if __name__ == "__main__":
    unittest.main()
