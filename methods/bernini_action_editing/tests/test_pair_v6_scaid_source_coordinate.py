from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import pair_v6_scaid_source_coordinate as scaid

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    scaid = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _prompts() -> dict[str, str]:
    return {
        branch: f"Complete standalone source-coordinate caption for {branch}."
        for branch in scaid.BRANCH_ORDER
    }


if _TORCH_AVAILABLE:
    class _ToyFields(nn.Module):
        def __init__(self, *, negative_leak: bool = False, gain: float = 0.0) -> None:
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(float(gain), dtype=torch.float32))
            shape = (1, 16, 21, 2, 2)
            self.t2v = {name: torch.zeros(shape) for name in scaid.BRANCH_ORDER}
            # nuisance: temporal varying axes 0..2; identity: axes 3..4;
            # temporal-DC: axis 5; valid action: axis 6.
            self.t2v["noop"][:, :, 1, 0, 0] = 1.0
            self.t2v["camera_only"][:, :, 1, 0, 0] = 1.0
            self.t2v["camera_only"][:, :, 2, 0, 0] = 2.0
            self.t2v["appearance_only"][:, :, 1, 0, 0] = 1.0
            self.t2v["appearance_only"][:, :, 3, 0, 0] = 2.0
            self.t2v["action"][:, :, 1, 0, 0] = 3.0
            self.t2v["action"][:, :, 2, 0, 0] = 4.0
            self.t2v["action"][:, :, 3, 0, 0] = 5.0
            self.t2v["action"][:, :, 4, 0, 0] = 6.0
            self.t2v["action"][:, :, :, 0, 1] = 3.0  # temporal DC
            self.t2v["action"][:, :, 5, 1, 0] = 7.0  # valid action
            self.correct = {name: torch.ones(shape) * 2.0 for name in scaid.BRANCH_ORDER}
            self.wrong = {
                "action": self.correct["action"].clone(),
                "noop": self.correct["noop"].clone(),
            }
            self.correct["action"][:, :, 4, 0, 0] += 4.0
            self.correct["noop"][:, :, 6, 0, 0] += 3.0
            self.identity = {
                "native_reference_dI": torch.zeros(shape, dtype=torch.float32)
            }
            self.identity["native_reference_dI"][:, :, 7, 0, 0] = 2.0
            self.t2v_requests: list[scaid.T2VFieldRequest] = []
            self.native_requests: list[scaid.NativeFieldRequest] = []
            self.negative_leak = negative_leak

        def t2v_callback(self, request: "scaid.T2VFieldRequest") -> "torch.Tensor":
            self.t2v_requests.append(request)
            return self.t2v[request.branch]

        def native_callback(self, request: "scaid.NativeFieldRequest") -> "torch.Tensor":
            self.native_requests.append(request)
            if request.phase == "frozen_native_reference_identity_control_dI":
                return self.identity["native_reference_dI"]
            base = (
                self.correct[request.branch]
                if request.source_role == "correct"
                else self.wrong[request.branch]
            )
            if not request.adapter_enabled:
                return base
            components = {
                name: self._student_component(request, name)
                for name, _coefficient in scaid.NATIVE_GUIDANCE_COMPONENTS
            }
            return scaid.aggregate_native_guidance_components(components)

        def _student_component(
            self, request: "scaid.NativeFieldRequest", component_name: str
        ) -> "torch.Tensor":
            base = self.correct[request.branch]
            feature = torch.zeros_like(base)
            if request.branch == "action":
                feature[:, :, 5, 1, 0] = 1.0
            elif self.negative_leak:
                feature.fill_(1.0)
            offsets = {
                "none_uncond": 1.0,
                "V_uncond": 2.0,
                "VI_uncond": 3.0,
                "VI_cond": 1.3125,
            }
            return (
                base
                + request.coordinate.gate_weight * self.gain * feature
                + offsets[component_name]
            ).to(torch.bfloat16)

        def __call__(self, request: "scaid.NativeFieldRequest") -> "torch.Tensor":
            return self.native_callback(request)

        def replay_component(
            self, request: "scaid.NativeFieldRequest", component_name: str
        ) -> "torch.Tensor":
            if component_name not in dict(scaid.NATIVE_GUIDANCE_COMPONENTS):
                raise RuntimeError("unknown native guidance component")
            # Distinct BF16 components reproduce the real numerical boundary;
            # their weighted offsets sum exactly to zero.
            self.native_requests.append(request)
            return self._student_component(request, component_name)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class PairV6SCAIDTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(608)
        self.source = torch.randn(1, 16, 21, 2, 2, dtype=torch.float32)
        self.epsilon = torch.randn_like(self.source)
        self.raw_prompts = _prompts()
        self.t2v_prompts, self.rv2v_prompts, _ = scaid.build_task_prompt_banks(
            self.raw_prompts, prompt_cleaner=lambda value: value
        )
        self.prompts = self.t2v_prompts
        self.checkpoint_sha = hashlib.sha256(b"test checkpoint tree").hexdigest()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # macOS exposes /var as a symlink to /private/var.  The production
        # provenance gate correctly rejects non-canonical absolute media
        # paths, so canonicalize the fixture root instead of weakening it.
        self.temp_root = Path(self.temp.name).resolve(strict=True)
        self.assertEqual(self.temp_root, self.temp_root.resolve(strict=True))
        self.evidence_path = self.temp_root / "authoritative-evidence.json"
        self.manifest_sha = hashlib.sha256(b"validated guidance manifest").hexdigest()
        self.geometry_source = self.temp_root / "fit-geometry-source.mp4"
        self.geometry_source.write_bytes(b"hash-bound fit geometry source")
        self.geometry_source_sha = hashlib.sha256(
            self.geometry_source.read_bytes()
        ).hexdigest()
        self.spec_path = self.temp_root / "source-bank-spec.json"
        self.spec_path.write_text("{}\n", encoding="ascii")
        self.spec_sha = hashlib.sha256(self.spec_path.read_bytes()).hexdigest()
        self.evidence_path.write_text(
            json.dumps(
                {
                    "guidance_manifest": {
                        "path": str(self.temp_root / "guidance.json"),
                        "file_sha256": self.manifest_sha,
                    },
                    "source_bank_spec": {
                        "path": str(self.spec_path),
                        "file_sha256": self.spec_sha,
                    },
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )
        self.evidence_sha = hashlib.sha256(self.evidence_path.read_bytes()).hexdigest()
        self.optimizer_authorized = True
        validator_patch = mock.patch.object(
            scaid.evidence_validator,
            "validate_evidence",
            side_effect=lambda *args, **kwargs: self._validated_receipt(),
        )
        validator_patch.start()
        self.addCleanup(validator_patch.stop)
        manifest_patch = mock.patch.object(
            scaid.cagd_trainer,
            "load_manifest",
            side_effect=lambda *args, **kwargs: self._guidance_manifest(),
        )
        manifest_patch.start()
        self.addCleanup(manifest_patch.stop)
        spec_patch = mock.patch.object(
            scaid.bank_spec,
            "load_sealed_spec",
            side_effect=lambda *args, **kwargs: (
                {
                    "groups": [
                        {
                            "candidates": [
                                {
                                    "candidate_id": "validated-fit-dog-sit",
                                    "analysis_split": "fit",
                                    "semantic_branch": "action",
                                    "action_family_id": "dog-sit",
                                    "geometry_source_video": str(
                                        self.geometry_source
                                    ),
                                    "geometry_source_video_sha256": self.geometry_source_sha,
                                }
                            ]
                        }
                    ]
                },
                self.spec_sha,
            ),
        )
        spec_patch.start()
        self.addCleanup(spec_patch.stop)
        prompt_patch = mock.patch.object(
            scaid,
            "_official_task_prompt_banks",
            side_effect=lambda raw: scaid.build_task_prompt_banks(
                raw, prompt_cleaner=lambda value: value
            ),
        )
        prompt_patch.start()
        self.addCleanup(prompt_patch.stop)

    def _guidance_manifest(self) -> SimpleNamespace:
        event = SimpleNamespace(
            event_id="validated-fit-dog-sit",
            action_family="dog-sit",
            analysis_split="fit",
            prompt_bank_sha256=scaid.object_sha256(self.prompts),
            prompt_by_branch=self.prompts,
        )
        return SimpleNamespace(raw_sha256=self.manifest_sha, events=(event,))

    def _validated_receipt(self) -> dict[str, object]:
        return {
            "schema_version": scaid.evidence_validator.AUTHORIZATION_SCHEMA,
            "optimizer_authorized": self.optimizer_authorized,
            "legacy_eligibility_self_declaration_trusted": False,
            "all_source_files_and_receipts_revalidated": True,
            "calibration_recomputed_from_raw_global_scores": True,
            "confirmation_event_count_for_optimizer": 0,
            "guidance_manifest_file_sha256": self.manifest_sha,
            "source_bank_spec_sha256": self.spec_sha,
            "fit_event_count": 1,
            "evidence_digest": hashlib.sha256(b"recomputed evidence").hexdigest(),
            "authorization_digest": hashlib.sha256(b"validator authorization").hexdigest(),
            "recomputed_calibration_receipt_digest": hashlib.sha256(
                b"recomputed calibration"
            ).hexdigest(),
        }

    def authorization(self, *, optimizer_authorized: bool = True) -> "scaid.SCAIDAuthorization":
        manifest = self._guidance_manifest()
        event = manifest.events[0]
        self.optimizer_authorized = optimizer_authorized
        result = scaid.load_authoritative_v3_authorization(
            self.evidence_path,
            expected_evidence_sha256=self.evidence_sha,
            checkpoint_tree_sha256=self.checkpoint_sha,
            fit_candidate_id=event.event_id,
        )
        return result

    def run_cell(self, model: "_ToyFields", *, index: int = 0) -> "scaid.SCAIDCell":
        gate = self.authorization()
        return scaid.run_scaid_cell(
            self.source,
            self.epsilon,
            schedule_index=index,
            authoritative_evidence_path=gate.evidence_path,
            expected_authoritative_evidence_sha256=gate.evidence_file_sha256,
            fit_candidate_id=gate.fit_candidate_id,
            raw_caption_by_branch=self.raw_prompts,
            expected_raw_caption_bank_sha256=scaid.object_sha256(self.raw_prompts),
            checkpoint_tree_sha256=self.checkpoint_sha,
            frozen_t2v_callback=model.t2v_callback,
            native_callback=model,
            config=scaid.SCAIDConfig(action_residual_weight=1.0, native_base_rms_ratio=10.0),
        )

    def test_every_field_uses_exact_source_coordinate_object(self) -> None:
        model = _ToyFields()
        cell = self.run_cell(model)
        sigma = cell.coordinate.sigma.reshape(1, 1, 1, 1, 1)
        self.assertTrue(
            torch.equal(
                cell.coordinate.x_sigma,
                (1.0 - sigma) * self.source + sigma * self.epsilon,
            )
        )
        requests = [*model.t2v_requests, *model.native_requests]
        self.assertEqual({id(item.coordinate.x_sigma) for item in requests}, {id(cell.coordinate.x_sigma)})
        self.assertEqual(len(model.t2v_requests), 10)
        self.assertEqual(len(model.native_requests), 23)
        self.assertTrue(
            all(
                request.prompt == self.t2v_prompts[request.branch]
                for request in model.t2v_requests
            )
        )
        self.assertTrue(
            all(
                request.prompt == self.rv2v_prompts[request.branch]
                for request in model.native_requests
            )
        )
        self.assertEqual(
            [(item.source_role, item.branch) for item in model.native_requests[11:13]],
            [("wrong", "action"), ("wrong", "noop")],
        )

    def test_projection_removes_nuisance_temporal_dc_and_identity_span(self) -> None:
        model = _ToyFields()
        safe = scaid.build_safe_action_residual(
            model.t2v, model.correct, model.wrong, model.identity,
            config=scaid.SCAIDConfig(native_base_rms_ratio=10.0),
        )
        self.assertLess(safe.temporal_dc_rms_after, 1.0e-6)
        self.assertIn("identity_binding_action", safe.accepted_directions)
        self.assertIn("identity_binding_noop", safe.accepted_directions)
        self.assertIn("native_reference_dI", safe.accepted_directions)
        self.assertTrue(all(value < 5.0e-5 for value in safe.projection_dot_after.values()))
        self.assertGreater(float(safe.vector[:, :, 5, 1, 0].abs().sum()), 0.0)
        self.assertFalse(safe.vector.requires_grad)

    def test_authoritative_fp32_cfg_avoids_old_bf16_staged_mismatch(self) -> None:
        values = (-0.01806640625, 0.11572265625, -0.0673828125, -0.059814453125)
        components = {
            name: torch.tensor([value], dtype=torch.bfloat16)
            for (name, _coefficient), value in zip(
                scaid.NATIVE_GUIDANCE_COMPONENTS, values
            )
        }
        none = components["none_uncond"]
        video = components["V_uncond"]
        image = components["VI_uncond"]
        conditional = components["VI_cond"]
        old_staged_bf16 = (
            none
            + 1.25 * (video - none)
            + 4.5 * (image - video)
            + 4.0 * (conditional - image)
        )
        authoritative = scaid.aggregate_native_guidance_components(components)
        expected = torch.zeros_like(none, dtype=torch.float32)
        for name, coefficient in scaid.NATIVE_GUIDANCE_COMPONENTS:
            expected = expected + components[name].float() * coefficient
        self.assertTrue(torch.equal(authoritative, expected))
        self.assertEqual(
            float((old_staged_bf16.float() - authoritative).abs().max()),
            0.00390625,
        )
        with self.assertRaisesRegex(scaid.PairV6SCAIDError, "authoritative order"):
            scaid.aggregate_native_guidance_components(dict(reversed(components.items())))

    def test_composite_teacher_is_native_base_plus_safe_residual(self) -> None:
        model = _ToyFields(gain=0.0)
        cell = self.run_cell(model)
        assert cell.objective is not None
        expected = model.correct["action"] + cell.objective.safe_residual.vector
        self.assertTrue(torch.allclose(cell.objective.composite_teacher, expected))
        self.assertFalse(cell.objective.composite_teacher.requires_grad)
        self.assertTrue(cell.objective.survival.optimizer_authorized)
        self.assertEqual(
            cell.objective.receipt["teacher"]["dry_run_survival_receipt_digest"],
            cell.objective.survival.receipt["receipt_digest"],
        )
        cell.objective.loss.backward()
        self.assertIsNotNone(model.gain.grad)
        self.assertTrue(torch.isfinite(model.gain.grad))
        self.assertNotEqual(float(model.gain.grad), 0.0)

    def test_negative_and_noop_base_parity_penalize_leakage(self) -> None:
        clean = self.run_cell(_ToyFields(negative_leak=False, gain=1.0)).objective
        leaky = self.run_cell(_ToyFields(negative_leak=True, gain=1.0)).objective
        assert clean is not None and leaky is not None
        self.assertEqual(float(clean.negative_parity_loss.detach()), 0.0)
        self.assertGreater(float(leaky.parity_by_branch["noop"].detach()), 0.9)
        self.assertGreater(float(leaky.negative_parity_loss.detach()), 0.9)

    def test_leaf_measurement_serial_vjp_is_memory_safe_and_reaches_lora(self) -> None:
        model = _ToyFields(negative_leak=True, gain=0.25)
        gate = self.authorization()
        cell = scaid.run_scaid_cell(
            self.source,
            self.epsilon,
            schedule_index=33,
            authoritative_evidence_path=gate.evidence_path,
            expected_authoritative_evidence_sha256=gate.evidence_file_sha256,
            fit_candidate_id=gate.fit_candidate_id,
            raw_caption_by_branch=self.raw_prompts,
            expected_raw_caption_bank_sha256=scaid.object_sha256(self.raw_prompts),
            checkpoint_tree_sha256=self.checkpoint_sha,
            frozen_t2v_callback=model.t2v_callback,
            native_callback=model,
            config=scaid.SCAIDConfig(
                action_residual_weight=1.0, native_base_rms_ratio=10.0
            ),
            leaf_vjp_mode=True,
        )
        assert cell.objective is not None
        cell.objective.loss.backward()
        self.assertIsNone(model.gain.grad)
        maxima = scaid.replay_native_student_vjp(
            cell, model
        )
        self.assertEqual(set(maxima), set(scaid.BRANCH_ORDER))
        self.assertTrue(all(value == 0.0 for value in maxima.values()))
        replay_requests = [
            request
            for request in model.native_requests
            if request.phase == "native_student_component_serial_vjp_replay"
        ]
        self.assertEqual(len(replay_requests), 4 * len(scaid.BRANCH_ORDER))
        self.assertIsNotNone(model.gain.grad)
        self.assertTrue(torch.isfinite(model.gain.grad))
        self.assertNotEqual(float(model.gain.grad), 0.0)
        serial_gradient = model.gain.grad.detach().clone()

        monolithic = _ToyFields(negative_leak=True, gain=0.25)
        monolithic_cell = self.run_cell(monolithic, index=33)
        assert monolithic_cell.objective is not None
        monolithic_cell.objective.loss.backward()
        self.assertIsNotNone(monolithic.gain.grad)
        self.assertTrue(
            torch.allclose(
                serial_gradient,
                monolithic.gain.grad,
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        )

    def test_low_sigma_38_39_are_absolute_zero_update_without_callbacks(self) -> None:
        for index in (38, 39):
            model = _ToyFields()
            cell = self.run_cell(model, index=index)
            self.assertTrue(cell.zero_update)
            self.assertFalse(cell.optimizer_authorized)
            self.assertIsNone(cell.objective)
            self.assertEqual(model.t2v_requests, [])
            self.assertEqual(model.native_requests, [])
            self.assertEqual(cell.receipt["callbacks_invoked"], 0)

    def test_no_go_validator_output_cannot_mint_authorization(self) -> None:
        self.assertFalse(hasattr(scaid, "seal_eligibility"))
        self.assertFalse(hasattr(scaid, "_AUTHORITY_TOKEN"))
        with self.assertRaisesRegex(scaid.PairV6SCAIDError, "does not authorize"):
            self.authorization(optimizer_authorized=False)

    def test_optimizer_api_has_no_caller_constructed_authorization_slot(self) -> None:
        parameters = inspect.signature(scaid.run_scaid_cell).parameters
        self.assertNotIn("authorization", parameters)
        self.assertIn("authoritative_evidence_path", parameters)
        self.assertIn("expected_authoritative_evidence_sha256", parameters)
        self.assertIn("fit_candidate_id", parameters)
        self.assertIn("raw_caption_by_branch", parameters)
        self.assertIn("expected_raw_caption_bank_sha256", parameters)
        self.assertNotIn("t2v_prompt_by_branch", parameters)
        self.assertNotIn("rv2v_prompt_by_branch", parameters)
        self.assertNotIn(
            "rv2v_prompt_by_branch",
            inspect.signature(scaid.replay_native_student_vjp).parameters,
        )

    def test_raw_caption_binding_fails_before_arbitrary_task_prompt_callbacks(self) -> None:
        changed = dict(self.raw_prompts)
        changed[scaid.mace.ACTION_BRANCH] += " Caller mutation."
        model = _ToyFields()
        with self.assertRaisesRegex(scaid.PairV6SCAIDError, "raw caption bank"):
            scaid.run_scaid_cell(
                self.source,
                self.epsilon,
                schedule_index=0,
                authoritative_evidence_path=self.evidence_path,
                expected_authoritative_evidence_sha256=self.evidence_sha,
                fit_candidate_id="validated-fit-dog-sit",
                raw_caption_by_branch=changed,
                expected_raw_caption_bank_sha256=scaid.object_sha256(
                    self.raw_prompts
                ),
                checkpoint_tree_sha256=self.checkpoint_sha,
                frozen_t2v_callback=model.t2v_callback,
                native_callback=model,
            )
        self.assertEqual(model.t2v_requests, [])
        self.assertEqual(model.native_requests, [])

    def test_forged_authorization_cannot_relabel_fit_event_or_prompt_bank(self) -> None:
        gate = self.authorization()
        forged_fit = scaid.SCAIDAuthorization(
            **{
                **gate.__dict__,
                "fit_candidate_id": "forged-fit",
                "action_family": "forged-family",
            }
        )
        with self.assertRaisesRegex(scaid.PairV6SCAIDError, "authoritative fit event"):
            forged_fit.validate(
                prompt_by_branch=self.prompts,
                checkpoint_tree_sha256=self.checkpoint_sha,
            )

        forged_prompts = {
            branch: f"Forged prompt closure for {branch}."
            for branch in scaid.BRANCH_ORDER
        }
        forged_prompt = scaid.SCAIDAuthorization(
            **{
                **gate.__dict__,
                "prompt_bank_sha256": scaid.object_sha256(forged_prompts),
            }
        )
        with self.assertRaisesRegex(
            scaid.PairV6SCAIDError, "action family or prompt bank differs"
        ):
            forged_prompt.validate(
                prompt_by_branch=forged_prompts,
                checkpoint_tree_sha256=self.checkpoint_sha,
            )

    def test_cell_receipt_claims_wrapper_identity_not_unprovable_forward_aliasing(self) -> None:
        cell = self.run_cell(_ToyFields())
        self.assertTrue(
            cell.receipt[
                "same_source_coordinate_request_wrapper_object_all_callbacks"
            ]
        )
        self.assertNotIn("same_source_coordinate_object_all_forwards", cell.receipt)

    def test_evidence_file_mutation_fails_before_callbacks(self) -> None:
        self.evidence_path.write_bytes(self.evidence_path.read_bytes() + b"\n")
        model = _ToyFields()
        with self.assertRaisesRegex(scaid.PairV6SCAIDError, "file SHA-256 differs"):
            scaid.run_scaid_cell(
                self.source,
                self.epsilon,
                schedule_index=0,
                authoritative_evidence_path=self.evidence_path,
                expected_authoritative_evidence_sha256=self.evidence_sha,
                fit_candidate_id="validated-fit-dog-sit",
                raw_caption_by_branch=self.raw_prompts,
                expected_raw_caption_bank_sha256=scaid.object_sha256(
                    self.raw_prompts
                ),
                checkpoint_tree_sha256=self.checkpoint_sha,
                frozen_t2v_callback=model.t2v_callback,
                native_callback=model,
            )
        self.assertEqual(model.t2v_requests, [])

    def test_rms_bound_and_contract_forbid_visual_shortcuts(self) -> None:
        model = _ToyFields()
        safe = scaid.build_safe_action_residual(
            model.t2v,
            model.correct,
            model.wrong,
            model.identity,
            config=scaid.SCAIDConfig(absolute_residual_rms_cap=0.01),
        )
        self.assertLessEqual(safe.bounded_rms, 0.010001)
        contract = scaid.contract_receipt()
        self.assertTrue(contract["public_api_forbidden_inputs_absent"])
        self.assertFalse(contract["t2v_to_rv2v_parameter_transfer"])
        self.assertEqual(contract["pure_t2v_generated_videos"], "calibration_receipts_only")
        self.assertEqual(contract["low_sigma_zero_update_indices"], [38, 39])
        self.assertNotIn("mask", inspect.signature(scaid.run_scaid_cell).parameters)

    def test_residual_survival_gate_rejects_projection_collapse(self) -> None:
        model = _ToyFields()
        safe = scaid.build_safe_action_residual(
            model.t2v,
            model.correct,
            model.wrong,
            model.identity,
            config=scaid.SCAIDConfig(native_base_rms_ratio=10.0),
        )
        with self.assertRaisesRegex(scaid.PairV6SCAIDError, "survival/orthogonality"):
            scaid.build_residual_survival_receipt(
                safe,
                config=scaid.SCAIDConfig(
                    native_base_rms_ratio=10.0,
                    minimum_projected_survival_ratio=0.99,
                ),
            )

    def test_residual_survival_requires_dI_and_identity_binding_basis(self) -> None:
        model = _ToyFields()
        model.identity["native_reference_dI"].zero_()
        model.wrong["action"] = model.correct["action"].clone()
        model.wrong["noop"] = model.correct["noop"].clone()
        safe = scaid.build_safe_action_residual(
            model.t2v,
            model.correct,
            model.wrong,
            model.identity,
            config=scaid.SCAIDConfig(native_base_rms_ratio=10.0),
        )
        with self.assertRaisesRegex(scaid.PairV6SCAIDError, "survival/orthogonality"):
            scaid.build_residual_survival_receipt(
                safe,
                config=scaid.SCAIDConfig(native_base_rms_ratio=10.0),
            )


if __name__ == "__main__":
    unittest.main()
