import copy
from contextlib import redirect_stdout
import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

import train_omnivideo2_pact as training_entry
from pact.dataset import PAYLOAD_FORMAT, PAYLOAD_PROVENANCE_BINDINGS
from pact.manifest import canonical_json_bytes
from pact.training import (
    DiffSynthWanTrainingScheduler,
    budget_source_condition_preserving_first_frame,
    nonvisual_token_counts,
    pact_training_losses,
    prepare_pact_flow_batch,
    shifted_rectified_flow_sigma,
    validate_training_config,
    wan_sequence_length,
)
from tests.test_manifest import authorized_atom_fixture
from tests.test_dataset import encoder_contract
from tools.bind_latent_payloads import bind_latent_payloads


class PactBatchPreparationTests(unittest.TestCase):
    def test_local_endpoint_erasure_and_shared_noise_target(self) -> None:
        source = torch.ones(1, 2, 3, 5, 5)
        target = torch.full_like(source, 3.0)
        source_mask = torch.zeros(1, 1, 3, 5, 5)
        target_mask = torch.zeros_like(source_mask)
        source_mask[:, :, :, 2, 1] = 1.0
        target_mask[:, :, :, 2, 3] = 1.0
        prepared = prepare_pact_flow_batch(
            source,
            target,
            source_mask,
            target_mask,
            torch.zeros(1),
            dilation_radius=(0, 0, 0),
            feather_radius=(0, 0, 0),
            source_erasure_mode="zero",
            noise=torch.zeros_like(source),
        )
        self.assertTrue(torch.equal(prepared.source_condition[:, :, 0], source[:, :, 0]))
        self.assertTrue(
            torch.equal(
                prepared.source_condition[:, :, 1:, 2, 1],
                torch.zeros(1, 2, 2),
            )
        )
        self.assertTrue(torch.equal(prepared.x_t, prepared.local_x0))
        self.assertTrue(torch.equal(prepared.local_velocity, -prepared.local_x0))
        self.assertTrue(
            torch.equal(
                prepared.local_x0[:, :, :, 2, 1],
                torch.full((1, 2, 3), 3.0),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.local_x0[:, :, :, 2, 3],
                torch.full((1, 2, 3), 3.0),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.local_x0[:, :, :, 0, 0],
                torch.ones(1, 2, 3),
            )
        )

    def test_ideal_local_velocity_zeroes_flow_and_x0_losses(self) -> None:
        source = torch.randn(1, 2, 3, 7, 7)
        target = torch.randn_like(source)
        source_mask = torch.zeros(1, 1, 3, 7, 7)
        target_mask = torch.zeros_like(source_mask)
        source_mask[..., 3, 2] = 1.0
        target_mask[..., 3, 4] = 1.0
        prepared = prepare_pact_flow_batch(
            source,
            target,
            source_mask,
            target_mask,
            torch.tensor([0.65]),
            dilation_radius=(0, 0, 0),
            feather_radius=(0, 0, 0),
            noise=torch.randn_like(source),
        )
        losses = pact_training_losses(
            prepared.local_velocity,
            prepared,
            weights={"router": 0.0},
        )
        for name in (
            "total",
            "velocity_edit",
            "velocity_preserve",
            "x0_boundary",
            "x0_temporal_outside",
        ):
            self.assertAlmostEqual(float(losses[name]), 0.0, places=6)

    def test_outside_velocity_corruption_is_penalized(self) -> None:
        source = torch.zeros(1, 1, 3, 7, 7)
        target = torch.ones_like(source)
        mask = torch.zeros(1, 1, 3, 7, 7)
        mask[..., 3, 3] = 1.0
        prepared = prepare_pact_flow_batch(
            source,
            target,
            mask,
            mask,
            torch.tensor([0.5]),
            dilation_radius=(0, 0, 0),
            feather_radius=(0, 0, 0),
            noise=torch.zeros_like(source),
        )
        prediction = prepared.local_velocity.clone()
        prediction[:, :, 1:, 0, 0] = 1.0
        losses = pact_training_losses(prediction, prepared, weights={"router": 0.0})
        self.assertGreater(float(losses["velocity_preserve"]), 0.0)
        self.assertGreater(float(losses["x0_temporal_outside"]), 0.0)

    def test_flow_weight_scales_only_flow_terms(self) -> None:
        source = torch.zeros(1, 1, 3, 7, 7)
        target = torch.ones_like(source)
        mask = torch.zeros(1, 1, 3, 7, 7)
        mask[..., 3, 3] = 1.0
        prepared = prepare_pact_flow_batch(
            source,
            target,
            mask,
            mask,
            torch.tensor([0.5]),
            dilation_radius=(0, 0, 0),
            feather_radius=(0, 0, 0),
            noise=torch.zeros_like(source),
        )
        prediction = torch.ones_like(source)
        router_logits = torch.zeros_like(mask)
        unit_weights = {
            "velocity_edit": 1.0,
            "velocity_preserve": 1.0,
            "x0_boundary": 1.0,
            "x0_temporal_outside": 1.0,
            "router": 1.0,
        }
        unweighted = pact_training_losses(
            prediction,
            prepared,
            router_logits=router_logits,
            router_target_mask=mask,
            weights=unit_weights,
            flow_weight=1.0,
        )
        weighted = pact_training_losses(
            prediction,
            prepared,
            router_logits=router_logits,
            router_target_mask=mask,
            weights=unit_weights,
            flow_weight=torch.tensor(3.0),
        )
        flow_names = (
            "velocity_edit",
            "velocity_preserve",
            "x0_boundary",
            "x0_temporal_outside",
        )
        flow_total = sum(unweighted[name] for name in flow_names)
        self.assertTrue(torch.equal(weighted["router"], unweighted["router"]))
        torch.testing.assert_close(
            weighted["total"],
            3.0 * flow_total + unweighted["router"],
        )

    def test_flow_weight_validation_is_strict(self) -> None:
        source = torch.zeros(1, 1, 3, 7, 7)
        target = torch.ones_like(source)
        mask = torch.zeros(1, 1, 3, 7, 7)
        mask[..., 3, 3] = 1.0
        prepared = prepare_pact_flow_batch(
            source,
            target,
            mask,
            mask,
            torch.tensor([0.5]),
            dilation_radius=(0, 0, 0),
            feather_radius=(0, 0, 0),
            noise=torch.zeros_like(source),
        )
        prediction = torch.ones_like(source)
        for invalid in (
            torch.ones(1),
            torch.tensor(float("nan")),
            torch.tensor(-1.0),
            True,
        ):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(
                (TypeError, ValueError)
            ):
                pact_training_losses(prediction, prepared, flow_weight=invalid)


class DiffSynthWanTrainingSchedulerTests(unittest.TestCase):
    def test_tables_have_exact_diffsynth_parity(self) -> None:
        scheduler = DiffSynthWanTrainingScheduler(shift=5.0)
        unshifted = torch.linspace(1.0, 0.0, 1001)[:-1]
        expected_sigmas = 5.0 * unshifted / (1.0 + 4.0 * unshifted)
        expected_timesteps = expected_sigmas * 1000.0
        profile = torch.exp(-2.0 * ((expected_timesteps - 500.0) / 1000.0) ** 2)
        shifted_profile = profile - profile.min()
        expected_weights = shifted_profile * (1000.0 / shifted_profile.sum())

        self.assertEqual(scheduler.sigmas.shape, (1000,))
        self.assertEqual(scheduler.timesteps.shape, (1000,))
        self.assertEqual(scheduler.flow_weights.shape, (1000,))
        torch.testing.assert_close(scheduler.sigmas, expected_sigmas, rtol=0, atol=0)
        torch.testing.assert_close(
            scheduler.timesteps, expected_timesteps, rtol=0, atol=0
        )
        torch.testing.assert_close(
            scheduler.flow_weights, expected_weights, rtol=0, atol=0
        )
        self.assertEqual(float(scheduler.sigmas[0]), 1.0)
        self.assertGreater(float(scheduler.sigmas[-1]), 0.0)
        self.assertEqual(float(scheduler.flow_weights[0]), 0.0)
        # DiffSynth performs the normalization in FP32, whose reduction rounds
        # to 999.9998779 on this table even though every entry matches exactly.
        self.assertLess(abs(float(scheduler.flow_weights.sum()) - 1000.0), 2e-4)

    def test_sample_is_uniform_discrete_deterministic_and_batch_shared(self) -> None:
        scheduler = DiffSynthWanTrainingScheduler()
        expected_generator = torch.Generator().manual_seed(9182)
        expected_id = int(
            torch.randint(0, 1000, (1,), generator=expected_generator).item()
        )
        actual_generator = torch.Generator().manual_seed(9182)
        sample = scheduler.sample(
            4,
            generator=actual_generator,
            dtype=torch.bfloat16,
        )

        self.assertEqual(sample.timestep_id, expected_id)
        self.assertEqual(sample.sigma.shape, (4,))
        self.assertEqual(sample.timestep.shape, (4,))
        self.assertEqual(sample.flow_weight.shape, torch.Size([]))
        self.assertEqual(sample.flow_weight.dtype, torch.float32)
        self.assertTrue(torch.equal(sample.sigma, sample.sigma[:1].repeat(4)))
        self.assertTrue(torch.equal(sample.timestep, sample.timestep[:1].repeat(4)))
        self.assertEqual(
            sample.sigma[0], scheduler.sigmas[expected_id].to(torch.bfloat16)
        )
        self.assertEqual(
            sample.timestep[0], scheduler.timesteps[expected_id].to(torch.bfloat16)
        )
        self.assertEqual(sample.flow_weight, scheduler.flow_weights[expected_id])

    def test_scheduler_validation_rejects_non_parity_inputs(self) -> None:
        for invalid_shift in (True, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(shift=invalid_shift), self.assertRaises(ValueError):
                DiffSynthWanTrainingScheduler(invalid_shift)
        scheduler = DiffSynthWanTrainingScheduler()
        for invalid_batch_size in (True, 0, -1, 1.5):
            with self.subTest(batch_size=invalid_batch_size), self.assertRaises(
                ValueError
            ):
                scheduler.sample(invalid_batch_size)  # type: ignore[arg-type]
        for invalid_id in (True, -1, 1000, 1.5):
            with self.subTest(timestep_id=invalid_id), self.assertRaises(ValueError):
                scheduler.at(invalid_id, 1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            scheduler.sample(1, dtype=torch.int64)


class PactBudgetAndConfigTests(unittest.TestCase):
    def test_budget_preserves_exact_first_frame(self) -> None:
        source = torch.randn(1, 16, 21, 8, 8)
        output, metadata = budget_source_condition_preserving_first_frame(
            source,
            max_context_len=20,
            nonvisual_tokens=0,
            visual_patch_size=(1, 4, 4),
        )
        self.assertTrue(metadata.compressed)
        self.assertEqual(output.shape[2], 5)
        self.assertTrue(torch.equal(output[:, :, 0], source[:, :, 0]))

    def test_budget_rejects_conv_token_count_ambiguity(self) -> None:
        with self.assertRaises(ValueError):
            budget_source_condition_preserving_first_frame(
                torch.randn(1, 16, 3, 9, 8),
                max_context_len=100,
                nonvisual_tokens=0,
                visual_patch_size=(1, 4, 4),
            )

    def test_shift_context_count_and_wan_sequence_length(self) -> None:
        values = torch.tensor([0.0, 0.25, 1.0])
        shifted = shifted_rectified_flow_sigma(values, 5.0)
        self.assertEqual(float(shifted[0]), 0.0)
        self.assertEqual(float(shifted[-1]), 1.0)
        self.assertTrue(bool((shifted[1:] >= shifted[:-1]).all()))
        counts = nonvisual_token_counts(
            [torch.zeros(7, 4096)],
            [torch.zeros(11, 2048)],
            special_token_count=4,
        )
        self.assertEqual(counts, [22])
        self.assertEqual(wan_sequence_length((1, 16, 3, 8, 12), (1, 2, 2)), 72)
        with self.assertRaises(ValueError):
            wan_sequence_length((1, 16, 3, 9, 12), (1, 2, 2))

    def test_checked_in_config_is_valid_and_schema_is_closed(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "pact_1_3b.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        validated = validate_training_config(config)
        self.assertEqual(validated["model"]["max_context_len"], 6144)
        invalid = copy.deepcopy(config)
        invalid["surprise"] = True
        with self.assertRaises(ValueError):
            validate_training_config(invalid)

    def test_cpu_dry_run_closes_atom_payload_objective_chain(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "pact_1_3b.json"
        )
        config = validate_training_config(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = authorized_atom_fixture(root)[0]
            atomic_manifest = root / "atomic.jsonl"
            atomic_manifest.write_bytes(canonical_json_bytes(atom) + b"\n")
            payload_root = root / "payloads"
            payload_root.mkdir()
            shape = (16, 3, 8, 8)
            source_mask = torch.zeros(1, *shape[1:])
            target_mask = torch.zeros_like(source_mask)
            source_mask[:, :, 2:4, 2:4] = 1.0
            target_mask[:, :, 4:6, 4:6] = 1.0
            payload = {
                "format": PAYLOAD_FORMAT,
                "atom_id": atom["atom_id"],
                "encoder_contract": encoder_contract(),
                "source_latent": torch.randn(shape),
                "global_target_latent": torch.randn(shape),
                "source_component_mask": source_mask,
                "target_component_mask": target_mask,
                "text_context": torch.randn(2, 4096),
                "vlm_context": torch.randn(3, 2048),
            }
            for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS:
                payload[payload_field] = atom[atomic_field]
            torch.save(payload, payload_root / f"{atom['atom_id']}.pt")
            publication = root / "training"
            bind_latent_payloads(atomic_manifest, payload_root, publication)
            args = argparse.Namespace(
                manifest=publication / "training_manifest.jsonl",
                payload_root=payload_root,
                dry_run_samples=1,
            )
            dataset = training_entry._make_dataset(args)
            output = io.StringIO()
            with redirect_stdout(output):
                training_entry._dry_run(args, config, dataset)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "contract-ok")
            self.assertEqual(result["payloads_fully_validated"], 1)
            self.assertEqual(
                result["samples"][0]["ideal_velocity_loss"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
