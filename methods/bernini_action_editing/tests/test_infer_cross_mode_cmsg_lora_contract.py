#!/usr/bin/env python3
"""Fail-closed contracts for Bernini Cross-Mode CMSG LoRA v6 inference."""

from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
from dataclasses import fields, replace
import io
import inspect
from pathlib import Path
import random
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_cross_mode_cmsg_lora as inference  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _valid_adapter_and_receipt():
    targets = inference.expected_lora_targets()

    class Dataset:
        root = Path("/data")
        signature = "dataset-signature"

        def __len__(self):
            return 644

    class Route:
        def __init__(self, index):
            self.iid = f"iid-{index:03d}"
            self.tier = "motion_only"
            self.full_target_weight = 0.0

    class Router:
        digest = "5" * 64
        file_sha256 = inference.v6_auh.v5.STRICT_ROUTING_SHA256

        def receipt(self):
            return {
                "path": "/routing.jsonl",
                "default_tier": "reject",
                "file_sha256": self.file_sha256,
                "routing_digest": self.digest,
                "explicit_route_counts": {
                    "full_pair": 0,
                    "motion_only": 359,
                    "reject": 285,
                },
            }

    class Distributed:
        world_size = 4
        ulysses_size = 4

    class Parameter:
        def numel(self):
            return 128

    args = argparse.Namespace(
        method_source_revision=SHA1,
        method_source_archive_sha256=SHA256,
        expected_bernini_commit=inference.trainer.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=inference.trainer.VEOMNI_TESTED_COMMIT,
        expected_checkpoint_tree_sha256=inference.trainer.CHECKPOINT_TREE_SHA256,
        seed=20260807,
        weight_decay=0.0,
        max_grad_norm=1.0,
        enforce_frozen_prior_gate=True,
        max_gate_attempts=inference.v6_auh.MAX_GATE_ATTEMPTS_DEFAULT,
        max_steps=40,
    )
    dataset = Dataset()
    router = Router()
    summary = {"sha256": "3" * 64, "index_sha256": "4" * 64}
    eligible = [(index, Route(index)) for index in range(359)]
    immutable = inference.v6_auh._immutable_contract(
        args=args,
        dataset=dataset,
        dataset_summary=summary,
        router=router,
        eligible_routes=eligible,
        target_modules=targets,
        checkpoint=Path("/checkpoint"),
    )
    gate_audit = [
        {"attempt_ordinal": index, "accepted": True} for index in range(40)
    ]
    parameter_names = ["adapter.lora_A.default.weight", "adapter.lora_B.default.weight"]
    with mock.patch.object(
        inference.v6_auh.v4,
        "_optimizer_parameter_names",
        return_value=parameter_names,
    ), mock.patch.object(
        inference.v6_auh.v4,
        "_checkpoint_parameter_digest",
        return_value="9" * 64,
    ), mock.patch.object(
        inference.v6_auh.v4,
        "_stable_recursive_digest",
        return_value="a" * 64,
    ):
        receipt = inference.v6_auh._build_receipt(
            args=args,
            global_step=40,
            attempt_ordinal=40,
            rejected_count=0,
            metrics={"sigma_schedule_index": 39.0},
            gate_audit=gate_audit,
            dataset=dataset,
            dataset_summary=summary,
            router=router,
            checkpoint=Path("/checkpoint"),
            bernini_revision=inference.trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=inference.trainer.VEOMNI_TESTED_COMMIT,
            distributed=Distributed(),
            backend="nccl",
            target_modules=targets,
            named_trainable=[("adapter", Parameter())],
            initialization_digest="8" * 64,
            transformers_version="4.51.3",
            immutable=immutable,
            optimizer_payload={"state": "signed"},
        )
    if receipt.get("inference_loader_parity_pending") is not False:
        raise AssertionError("AUH trainer still emits a parity-pending receipt")
    config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": targets,
    }
    return config, receipt


def _redigest(receipt):
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = inference.trainer.object_sha256(receipt)


def _runtime_schedule_audit():
    return {
        "schedule_sha256": inference.sigma_strata.SCHEDULE_SHA256,
        "timesteps": list(inference.sigma_strata.PINNED_TIMESTEPS),
        "positive_sigmas": list(
            inference.sigma_strata.PINNED_POSITIVE_SIGMAS
        ),
        "positive_sigmas_float32_be_hex": list(
            inference.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": (
            inference.sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
        ),
    }


def _valid_trace():
    records = []
    for step, (timestep, sigma) in enumerate(
        zip(
            inference.sigma_strata.PINNED_TIMESTEPS,
            inference.sigma_strata.PINNED_POSITIVE_SIGMAS,
        )
    ):
        rho = inference.spectrum.release_rho(step)
        exact = rho == 0.0
        records.append(
            inference.CrossModeCMSGStepRecord(
                step_index=step,
                timestep=timestep,
                sigma=sigma,
                rho=rho,
                model_id="transformer_1",
                transformer_forwards=4,
                frozen_negative_forwards=1,
                frozen_noop_forwards=1,
                frozen_action_forwards=1,
                adapted_action_forwards=1,
                original_scheduler_calls=1,
                official_frozen_action_apg_exact=True,
                official_frozen_action_apg_rms_error=0.0,
                official_frozen_action_apg_max_abs_error=0.0,
                frozen_editor_direction_rms=1.0,
                adapted_editor_direction_rms=1.1,
                raw_direction_delta_rms=0.2,
                executed_direction_correction_rms=0.0 if exact else 0.1 * rho,
                executed_first_phase_max_abs=0.0,
                scheduler_boundary_correction_rms=0.0 if exact else 0.05 * rho,
                phase_cells=4,
                exact_official_model_output_object=exact,
                adapter_loaded=True,
                generator_forwards=0,
            )
        )
    return inference.CrossModeCMSGTrace(
        adapter_loaded=True,
        records=records,
        sample_calls=1,
    )


class CrossModeCMSGInferencePureTests(unittest.TestCase):
    def test_runtime_has_only_source_action_and_no_oracle(self):
        contract = inference.runtime_contract()
        self.assertEqual(
            contract["external_inference_conditions"],
            ["source_video", "action_instruction"],
        )
        self.assertEqual(contract["inference_generator_forwards"], 0)
        self.assertFalse(contract["inference_generator_loaded"])
        self.assertEqual(
            contract["per_step_editor_branches"],
            [
                "frozen_negative",
                "frozen_noop",
                "frozen_action",
                "adapted_action",
            ],
        )
        self.assertIn("target_video", contract["forbidden_conditions"])
        self.assertIn("mask", contract["forbidden_conditions"])
        self.assertIn("optical_flow", contract["forbidden_conditions"])
        self.assertIn("first_frame_anchor", contract["forbidden_conditions"])
        self.assertEqual(contract["formal_adapter_off_steps"], list(range(32, 40)))
        # Step 31 is the inclusive cosine endpoint and therefore also aliases
        # the official object even though the named adapter-off interval starts at 32.
        self.assertEqual(contract["zero_release_steps"], list(range(31, 40)))

    def test_exact_46_scope_is_cross_q_plus_mid_self_q(self):
        targets = inference.expected_lora_targets()
        self.assertEqual(len(targets), 46)
        self.assertEqual(len(set(targets)), 46)
        cross = [name for name in targets if ".attn2." in name]
        self_q = [name for name in targets if ".attn1." in name]
        self.assertEqual(len(cross), 30)
        self.assertEqual(len(self_q), 16)
        self.assertTrue(all(name.endswith(".to_q") for name in targets))
        self.assertEqual(
            {
                int(name.split(".blocks.")[1].split(".")[0])
                for name in self_q
            },
            set(range(7, 23)),
        )

    def test_raw_operator_signature_contains_no_inference_oracle(self):
        raw_fields = {item.name for item in fields(inference.RawCrossModeCMSGStep)}
        self.assertEqual(
            {
                "frozen_negative_velocity_packed",
                "frozen_noop_velocity_packed",
                "frozen_action_velocity_packed",
                "adapted_action_velocity_packed",
            },
            {name for name in raw_fields if name.endswith("velocity_packed")},
        )
        for forbidden in (
            "generator",
            "target",
            "mask",
            "flow",
            "pose",
            "trajectory",
            "anchor",
            "source_phase",
        ):
            self.assertFalse(any(forbidden in name for name in raw_fields))
        signature = inspect.signature(inference.project_cross_mode_cmsg_step)
        self.assertNotIn("generator", str(signature))
        self.assertNotIn("target", str(signature))

    def test_cli_is_explicitly_preflight_only_and_fails_closed(self):
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertEqual(destinations, {"help", "preflight_only"})
        for oracle in ("generator", "target", "mask", "flow", "pose", "anchor"):
            self.assertNotIn(oracle, destinations)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(inference.main(["--preflight-only"]), 0)
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.main([])
        preflight = inference.preflight_contract()
        self.assertTrue(preflight["strict_operator_ready"])
        self.assertTrue(preflight["four_branch_hook_ready"])
        self.assertFalse(preflight["standalone_full_cli_integrated"])
        self.assertFalse(preflight["production_inference_claim_authorized"])

    def test_receipt_accepts_only_completed_v6_exact_46_contract(self):
        config, receipt = _valid_adapter_and_receipt()
        identity = inference.validate_training_adapter_contract(config, receipt)
        self.assertEqual(identity["scope"], inference.REQUIRED_LORA_SCOPE)
        self.assertEqual(len(identity["targets"]), 46)
        self.assertEqual(receipt["method"], inference.v6_auh.METHOD_NAME)
        self.assertEqual(
            receipt["schema_version"], inference.v6_auh.RECEIPT_SCHEMA
        )
        immutable = receipt["immutable_contract"]["value"]
        self.assertNotIn("lora_scope", immutable)
        self.assertEqual(immutable["lora"]["target_module_count"], 46)
        self.assertEqual(
            immutable["forward_cell_order"],
            list(inference.v6_auh.FORWARD_CELL_ORDER),
        )

        mutations = []
        bad = copy.deepcopy(receipt)
        bad["schema_version"] = "v5"
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["adapter"]["target_modules"] = bad["adapter"]["target_modules"][:-1]
        bad["adapter"]["target_module_count"] = 45
        bad["adapter"]["target_modules_sha256"] = inference.trainer.object_sha256(
            bad["adapter"]["target_modules"]
        )
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["supervision"]["generator_loaded_at_inference"] = True
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["supervision"]["paired_target_used_at_inference"] = True
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["immutable_contract"]["value"]["release_schedule"][31] = 0.1
        bad["immutable_contract"]["digest"] = inference.trainer.object_sha256(
            bad["immutable_contract"]["value"]
        )
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["scientific_claim_authorized"] = True
        _redigest(bad)
        mutations.append(bad)
        bad = copy.deepcopy(receipt)
        bad["inference_loader_parity_pending"] = True
        _redigest(bad)
        mutations.append(bad)
        for candidate in mutations:
            with self.assertRaises(inference.CrossModeCMSGInferenceError):
                inference.validate_training_adapter_contract(config, candidate)

        # PEFT internally stores target_modules as a set, so JSON order is not
        # stable.  Set equality is valid; missing, duplicate, or extra scope is not.
        reordered_config = copy.deepcopy(config)
        reordered_targets = list(config["target_modules"])
        random.Random(20260807).shuffle(reordered_targets)
        self.assertNotEqual(reordered_targets, config["target_modules"])
        self.assertNotEqual(reordered_targets, sorted(reordered_targets))
        reordered_config["target_modules"] = reordered_targets
        reordered = inference.validate_training_adapter_contract(
            reordered_config, receipt
        )
        self.assertEqual(reordered["serialized_target_modules"], identity["targets"])
        bad_config = copy.deepcopy(config)
        bad_config["target_modules"] = config["target_modules"][:-1]
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.validate_training_adapter_contract(bad_config, receipt)
        bad_config = copy.deepcopy(config)
        bad_config["target_modules"][-1] = (
            "diff_dec.transformer.blocks.0.attn2.to_k"
        )
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.validate_training_adapter_contract(bad_config, receipt)
        bad_config = copy.deepcopy(config)
        bad_config["target_modules"][-1] = bad_config["target_modules"][0]
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.validate_training_adapter_contract(bad_config, receipt)
        bad_config = copy.deepcopy(config)
        bad_config["r"] = 16
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.validate_training_adapter_contract(bad_config, receipt)

    def test_strict_loader_validates_receipt_before_pinned_loader(self):
        config, receipt = _valid_adapter_and_receipt()
        sentinel = object()
        with mock.patch.object(
            inference.v5,
            "_strict_load_adapter",
            return_value=(sentinel, 92, 46),
        ) as loader:
            result = inference.strict_load_adapter(
                base_model=object(),
                bundle=object(),
                adapter_config=config,
                receipt=receipt,
            )
        self.assertIs(result[0], sentinel)
        self.assertEqual(result[1:3], (92, 46))
        self.assertEqual(len(result[3]["targets"]), 46)
        loader.assert_called_once()

        bad = copy.deepcopy(receipt)
        bad["supervision"]["generator_forwards_at_inference"] = 1
        _redigest(bad)
        with mock.patch.object(inference.v5, "_strict_load_adapter") as loader:
            with self.assertRaises(inference.CrossModeCMSGInferenceError):
                inference.strict_load_adapter(
                    base_model=object(),
                    bundle=object(),
                    adapter_config=config,
                    receipt=bad,
                )
            loader.assert_not_called()

    def test_trace_certifies_exact_40_step_release_and_no_generator(self):
        trace = _valid_trace()
        payload = inference.validate_execution_trace(
            trace, runtime_schedule_audit=_runtime_schedule_audit()
        )
        certificate = payload["certificate"]
        self.assertEqual(certificate["editor_transformer_forwards"], 160)
        self.assertEqual(certificate["generator_forwards"], 0)
        self.assertEqual(
            certificate["exact_official_model_output_steps"], list(range(31, 40))
        )
        self.assertEqual(certificate["formal_adapter_off_steps"], list(range(32, 40)))

        bad = copy.deepcopy(trace)
        bad.records[25] = replace(bad.records[25], rho=0.25)
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.validate_execution_trace(
                bad, runtime_schedule_audit=_runtime_schedule_audit()
            )
        bad = copy.deepcopy(trace)
        bad.records[32] = replace(
            bad.records[32], exact_official_model_output_object=False
        )
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.validate_execution_trace(
                bad, runtime_schedule_audit=_runtime_schedule_audit()
            )
        bad = copy.deepcopy(trace)
        bad.records[0] = replace(bad.records[0], generator_forwards=1)
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            inference.validate_execution_trace(
                bad, runtime_schedule_audit=_runtime_schedule_audit()
            )

    def test_hook_reuses_pinned_v5_four_branch_capture(self):
        self.assertTrue(
            issubclass(inference._InstalledCrossModeCMSG, inference.v5._InstalledFourBranch)
        )
        source = inspect.getsource(inference._InstalledCrossModeCMSG._wrapped_scheduler_step)
        self.assertIn("project_cross_mode_cmsg_step", source)
        self.assertIn("self._original_scheduler_step", source)
        self.assertNotIn("generator", source)


try:
    import torch
except ImportError:  # pragma: no cover - exercised on AUH when needed
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CrossModeCMSGInferenceTensorTests(unittest.TestCase):
    def _raw(self, *, step, adapted_scale=0.04):
        torch.manual_seed(37)
        layout = inference.tri.PackedLatentLayout.from_spatial_shape(
            (1, 1, 21, 2, 2)
        )
        sample = torch.randn(layout.packed_shape, dtype=torch.float32)
        negative = torch.randn(layout.packed_shape, dtype=torch.float32).to(
            torch.bfloat16
        )
        noop = torch.randn(layout.packed_shape, dtype=torch.float32).to(
            torch.bfloat16
        )
        action = torch.randn(layout.packed_shape, dtype=torch.float32).to(
            torch.bfloat16
        )
        phase_ramp = torch.linspace(0.0, 1.0, 21, dtype=torch.float32).reshape(
            1, 21, 1
        )
        adapted = (action.float() + adapted_scale * phase_ramp).to(torch.bfloat16)
        sigma = torch.tensor(0.5, dtype=torch.float32)
        spatial_sample = inference.tri._packed_to_spatial(sample, layout)
        spatial_negative = inference.tri._packed_to_spatial(negative, layout)
        spatial_action = inference.tri._packed_to_spatial(action, layout)
        negative_clean = inference.tri.pinned_raw_condition_clean(
            spatial_sample, spatial_negative, sigma
        )
        action_clean = inference.tri.pinned_raw_condition_clean(
            spatial_sample, spatial_action, sigma
        )
        apg = inference.tri.APGParameters(4.0, 0.8, False, 0.5, 50.0, 0.0)
        guided = inference.tri._normalized_guidance(
            action_clean,
            negative_clean,
            4.0,
            inference.tri._MomentumBuffer(0.0, branch="fixture"),
            0.5,
            50.0,
        )
        official = inference.tri._spatial_to_packed(
            (spatial_sample - guided) / sigma, layout
        ).to(torch.bfloat16)
        return inference.RawCrossModeCMSGStep(
            step_index=step,
            timestep=torch.tensor(float(1000 - step)),
            timestep_float=float(1000 - step),
            sigma=sigma,
            sigma_float=0.5,
            model_id="transformer_1",
            sample_packed=sample,
            official_model_output=official,
            frozen_negative_velocity_packed=negative,
            frozen_noop_velocity_packed=noop,
            frozen_action_velocity_packed=action,
            adapted_action_velocity_packed=adapted,
            apg=apg,
            layout=layout,
        )

    @staticmethod
    def _run(raw):
        return inference.project_cross_mode_cmsg_step(
            raw,
            frozen_action_momentum=inference.tri._MomentumBuffer(
                0.0, branch="frozen_action"
            ),
            frozen_noop_momentum=inference.tri._MomentumBuffer(
                0.0, branch="frozen_noop"
            ),
            adapted_action_momentum=inference.tri._MomentumBuffer(
                0.0, branch="adapted_action"
            ),
        )

    def test_early_adapted_and_taper_change_only_causal_direction(self):
        early_raw = self._raw(step=0)
        with mock.patch.object(
            inference,
            "phase_grid_to_packed",
            wraps=inference.phase_grid_to_packed,
        ) as packed_clean:
            early, early_record = self._run(early_raw)
        self.assertEqual(early_record.rho, 1.0)
        self.assertIsNot(early.model_output, early_raw.official_model_output)
        self.assertFalse(early_record.exact_official_model_output_object)
        self.assertGreater(early_record.raw_direction_delta_rms, 0.0)
        self.assertGreater(early_record.executed_direction_correction_rms, 0.0)
        self.assertEqual(early_record.executed_first_phase_max_abs, 0.0)

        # Inspect the clean field immediately before serialization.  Its full
        # phase-zero carrier must be bit-exactly the official frozen action,
        # while the adapted branch contributes only a Q0 direction delta.
        self.assertEqual(packed_clean.call_count, 1)
        scheduler_clean_phase = packed_clean.call_args.args[0]
        official_clean_packed = (
            early_raw.sample_packed
            - early_raw.sigma * early_raw.official_model_output
        )
        official_clean_phase = inference.packed_to_phase_grid(
            official_clean_packed.float(), layout=early_raw.layout
        )
        self.assertTrue(
            torch.equal(
                scheduler_clean_phase[:, :1], official_clean_phase[:, :1]
            )
        )

        taper_raw = self._raw(step=25)
        _, taper_record = self._run(taper_raw)
        self.assertGreater(taper_record.rho, 0.0)
        self.assertLess(taper_record.rho, 1.0)
        self.assertGreater(taper_record.executed_direction_correction_rms, 0.0)
        self.assertLess(
            taper_record.executed_direction_correction_rms,
            taper_record.raw_direction_delta_rms + 1.0e-7,
        )

    def test_step31_and_formal_32_to_39_alias_official_object_exactly(self):
        for step in (31, 32, 39):
            with self.subTest(step=step):
                raw = self._raw(step=step)
                projected, record = self._run(raw)
                self.assertEqual(record.rho, 0.0)
                self.assertIs(projected.model_output, raw.official_model_output)
                self.assertTrue(record.exact_official_model_output_object)
                self.assertEqual(record.executed_direction_correction_rms, 0.0)
                self.assertEqual(record.scheduler_boundary_correction_rms, 0.0)

    def test_apg_parity_and_geometry_fail_closed(self):
        raw = self._raw(step=0)
        bad_official = raw.official_model_output.clone()
        bad_official[0, 0, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            self._run(replace(raw, official_model_output=bad_official))
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            self._run(replace(raw, sigma_float=0.4))
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            self._run(replace(raw, adapted_action_velocity_packed=None))
        with self.assertRaises(inference.CrossModeCMSGInferenceError):
            self._run(replace(raw, model_id="transformer_2"))


if __name__ == "__main__":
    unittest.main()
