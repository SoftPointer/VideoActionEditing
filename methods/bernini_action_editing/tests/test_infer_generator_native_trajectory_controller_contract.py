from __future__ import annotations

import argparse
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generator_native_trajectory_controller as core  # noqa: E402
import infer_generator_native_trajectory_controller as inference  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _args(**overrides) -> argparse.Namespace:
    values = {
        "instruction": "Make the actor crouch.",
        "num_inference_steps": 40,
        "seed": 42,
        "expected_bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "expected_controller_state_sha256": SHA256,
        "expected_controller_receipt_sha256": "3" * 64,
        "expected_source_sha256": None,
        "expected_instruction_sha256": None,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": "4" * 64,
        "checkpoint": "/checkpoint/Bernini-R-1.3B-Diffusers",
        "source_video": "/source.mp4",
        "controller_state": "/controller.safetensors",
        "controller_receipt": "/controller.receipt.json",
        "output": "/out.mp4",
        "allow_diagnostic_no_go": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _valid_traces() -> tuple[
    tri.TriBranchTrace,
    inference.ControllerExecutionTrace,
    dict[str, object],
]:
    branch_records = []
    controller_records = []
    core_records = []
    for index in range(40):
        sigma = core.sigma_strata.PINNED_POSITIVE_SIGMAS[index]
        timestep = float(core.sigma_strata.PINNED_TIMESTEPS[index])
        branch_records.append(
            tri.TriBranchStepRecord(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                model_id="transformer_1",
                transformer_forwards=3,
                shared_negative_forwards=1,
                action_forwards=1,
                noop_forwards=1,
                original_scheduler_calls=1,
                callback_correction_rms=0.2,
                raw_action_noop_delta_rms=0.1,
                guided_action_noop_delta_rms=0.15,
                guided_action_noop_delta_l2=3.0,
                action_noop_exact_parity=False,
                effective_guidance_scale=4.0,
                official_action_parity_rms_error=0.0,
                official_action_parity_max_abs_error=0.0,
                official_action_exact_parity=True,
                sample_dtype="torch.float32",
                branch_velocity_dtype="torch.bfloat16",
                official_model_output_dtype="torch.float32",
            )
        )
        controller_records.append(
            inference.ControllerStepRecord(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                correction_rms=0.1,
                proposal_rms=0.2,
                action_noop_input_byte_exact=False,
                parity_bypass_byte_exact=True,
            )
        )
        interpolation = core.sigma_interpolation(
            step_index=index, timestep=timestep, sigma=sigma
        )
        core_records.append(
            {
                "step_index": index,
                "timestep": int(timestep),
                "sigma": sigma,
                "sigma_float32_be_hex": core.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                    index
                ],
                "upper_knot": interpolation.upper_knot,
                "lower_knot": interpolation.lower_knot,
                "lower_weight": interpolation.lower_weight,
                "kappa": 0.1,
                "rho": 0.1,
                "action_noop_exact_parity": False,
                "native_delta_rms_max": 0.2,
                "proposal_correction_rms_max": 0.1,
                "executed_correction_rms_max": 0.1,
                "trust_region_satisfied": True,
            }
        )
    parameter_payload = {
        "schema_version": core.PARAMETER_SCHEMA_VERSION,
        "method": core.METHOD_NAME,
        "trainable_dimension": 36,
        "parameter_shapes": {
            "alpha_logits": [6, 4],
            "kappa_raw": [6],
            "rho_raw": [6],
        },
        "parameter_vector_sha256": "d" * 64,
        "decoded_kappa_knots": [0.1] * 6,
        "decoded_rho_knots": [0.1] * 6,
        "kappa_monotone_nondecreasing": True,
        "rho_monotone_nondecreasing": True,
        "kappa_strict_upper_bound": core.MAX_KAPPA,
        "rho_strict_upper_bound": core.MAX_RHO,
        "schedule_sha256": core.sigma_strata.SCHEDULE_SHA256,
    }
    parameter_receipt = {
        **parameter_payload,
        "receipt_digest": core._object_sha256(parameter_payload),
    }
    core_payload = {
        "schema_version": core.SCHEMA_VERSION,
        "method": core.METHOD_NAME,
        "runtime_contract": core.controller_contract(),
        "parameters": parameter_receipt,
        "controls": {
            "phase_reverse": False,
            "sigma_shuffle": False,
            "kappa_off": False,
            "rho_off": False,
        },
        "active_controls": [],
        "state": {
            "expected_next_step": 40,
            "completed": True,
            "step_count": 40,
            "memory_present": True,
            "reset_count": 0,
        },
        "steps": core_records,
    }
    core_receipt = {
        **core_payload,
        "receipt_digest": core._object_sha256(core_payload),
    }
    return (
        tri.TriBranchTrace(records=branch_records, sample_calls=1),
        inference.ControllerExecutionTrace(
            controller_raw_36d_sha256="5" * 64,
            records=controller_records,
        ),
        core_receipt,
    )


class _FakeBundle:
    diagnostic_override = False
    deployable = True

    @staticmethod
    def audit_receipt() -> dict[str, object]:
        return {
            "state_file_sha256": "6" * 64,
            "receipt_file_sha256": "7" * 64,
            "raw_36d_sha256": "5" * 64,
            "representability_gate": "GO",
            "deployable": True,
        }


class PureEGNTCInferenceContractTests(unittest.TestCase):
    def test_cli_is_source_instruction_and_controller_only(self) -> None:
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertTrue(
            {
                "source_video",
                "instruction",
                "controller_state",
                "controller_receipt",
                "allow_diagnostic_no_go",
            }
            <= destinations
        )
        forbidden = {
            "target",
            "target_video",
            "support",
            "support_video",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "reference_video",
            "edited_first_frame",
        }
        self.assertTrue(destinations.isdisjoint(forbidden))
        parsed = parser.parse_args(
            [
                "--bernini-root",
                "/b",
                "--veomni-root",
                "/v",
                "--checkpoint",
                "/c",
                "--controller-state",
                "/controller.safetensors",
                "--controller-receipt",
                "/controller.json",
                "--expected-controller-state-sha256",
                SHA256,
                "--expected-controller-receipt-sha256",
                "3" * 64,
                "--source-video",
                "/source.mp4",
                "--instruction",
                "move",
                "--output",
                "/out.mp4",
                "--method-source-revision",
                SHA1,
                "--method-source-archive-sha256",
                "4" * 64,
            ]
        )
        self.assertEqual(parsed.num_inference_steps, 40)
        self.assertFalse(parsed.allow_diagnostic_no_go)

    def test_cli_rejects_solver_hash_and_instruction_drift(self) -> None:
        inference.validate_cli(_args())
        invalid = (
            {"num_inference_steps": 41},
            {"seed": -1},
            {"instruction": "\x00"},
            {"expected_controller_state_sha256": "bad"},
            {"expected_source_sha256": "bad"},
            {"source_video": "relative.mp4"},
        )
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaises(
                inference.EGNTCInferenceError
            ):
                inference.validate_cli(_args(**changed))

    def test_exact_sampler_is_81_frames_and_40_official_unipc_steps(self) -> None:
        contract = inference.exact_sampler_contract(seed=7)
        self.assertEqual(contract["num_frames"], 81)
        self.assertEqual(contract["num_inference_steps"], 40)
        self.assertEqual(contract["guidance_mode"], "v2v_apg")
        self.assertEqual(contract["flow_shift"], 5.0)
        self.assertEqual(contract["omega_txt"], 4.0)
        self.assertEqual(contract["eta"], 0.5)
        self.assertEqual(inference.base.ULYSSES_SIZE, 4)

    def test_callback_signatures_have_no_privileged_inputs(self) -> None:
        forbidden = {
            "target",
            "target_video",
            "support",
            "support_video",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "reference",
            "first_frame",
        }
        for function in (
            inference.TracedEGNTCCallback.__init__,
            inference.TracedEGNTCCallback.__call__,
            core.EGNTCCallback.__init__,
            core.EGNTCCallback.__call__,
        ):
            with self.subTest(function=function.__qualname__):
                self.assertTrue(
                    forbidden.isdisjoint(inspect.signature(function).parameters)
                )

    def test_trace_certifies_40_scheduler_calls_and_120_forwards(self) -> None:
        tri_trace, controller_trace, core_receipt = _valid_traces()
        payload = inference.validate_execution_trace(
            tri_trace, controller_trace, core_receipt
        )
        certificate = payload["certificate"]
        self.assertEqual(certificate["step_count"], 40)
        self.assertEqual(certificate["original_unipc_calls"], 40)
        self.assertEqual(certificate["transformer_forwards"], 120)
        self.assertEqual(certificate["official_action_apg_exact_steps"], 40)
        self.assertFalse(certificate["custom_integrator"])
        self.assertRegex(payload["trace_digest"], r"^[0-9a-f]{64}$")

        tri_trace.records[4] = replace(
            tri_trace.records[4], original_scheduler_calls=2
        )
        with self.assertRaisesRegex(inference.EGNTCInferenceError, "three transformer"):
            inference.validate_execution_trace(tri_trace, controller_trace, core_receipt)

    def test_trace_rejects_failed_official_parity_and_controller_bypass(self) -> None:
        tri_trace, controller_trace, core_receipt = _valid_traces()
        tri_trace.records[3] = replace(
            tri_trace.records[3], official_action_exact_parity=False
        )
        with self.assertRaisesRegex(inference.EGNTCInferenceError, "exact parity"):
            inference.validate_execution_trace(tri_trace, controller_trace, core_receipt)

        tri_trace, controller_trace, core_receipt = _valid_traces()
        controller_trace.records[3] = replace(
            controller_trace.records[3], parity_bypass_byte_exact=False
        )
        with self.assertRaisesRegex(inference.EGNTCInferenceError, "diagnostic"):
            inference.validate_execution_trace(tri_trace, controller_trace, core_receipt)

    def test_receipt_hashes_source_instruction_base_and_controller(self) -> None:
        tri_trace, controller_trace, core_receipt = _valid_traces()
        execution = inference.validate_execution_trace(
            tri_trace, controller_trace, core_receipt
        )
        args = _args()
        receipt = inference.build_inference_receipt(
            args=args,
            bundle=_FakeBundle(),
            source_path=Path("/source.mp4"),
            source_sha256="9" * 64,
            source_metadata={"source_derived_bucket_hw": [480, 496]},
            output_path=Path("/out.mp4"),
            output_sha256="a" * 64,
            noop_identity={"frozen_t5": True},
            execution_trace=execution,
            bernini_revision=inference.trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=inference.trainer.VEOMNI_TESTED_COMMIT,
            inference_file_hashes={},
            wan_diffusion_path=Path("/vendor/bernini/models/wan_diffusion.py"),
            wan_diffusion_sha256=tri.PINNED_WAN_DIFFUSION_SHA256,
            runtime_versions={},
        )
        self.assertEqual(receipt["input"]["source_video_sha256"], "9" * 64)
        self.assertRegex(receipt["input"]["instruction_utf8_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            receipt["base_checkpoint"]["tree_sha256"],
            inference.trainer.CHECKPOINT_TREE_SHA256,
        )
        self.assertEqual(
            receipt["controller_checkpoint"]["state_file_sha256"], "6" * 64
        )
        self.assertEqual(
            receipt["input"]["accepted_external_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertFalse(receipt["input"]["target_accessed_by_inference"])
        self.assertFalse(receipt["input"]["support_accessed_by_inference"])
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(inference.object_sha256(unsigned), declared)

    def test_main_installs_controller_at_exact_tri_branch_boundary(self) -> None:
        source = inspect.getsource(inference.main)
        self.assertIn("tri.tri_branch_unipc_hook", source)
        self.assertIn("TracedEGNTCCallback", source)
        self.assertIn("model.sample", source)
        self.assertIn("expected_steps=NUM_INFERENCE_STEPS", source)
        self.assertIn("multi_video_vae_latents=[source_latent]", source)
        self.assertNotIn("target_video=", source)
        self.assertNotIn("support_video=", source)

    def test_rank_cache_and_output_paths_are_isolated_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            caches = inference.configure_rank_local_caches(
                {"BERNINI_EGNTC_RANK_CACHE_ROOT": str(root), "LOCAL_RANK": "2"}
            )
            self.assertTrue(all("rank-2" in value for value in caches.values()))
            output, receipt = inference.base._resolve_output(root / "result.mp4")
            self.assertEqual(receipt.name, "result.mp4.receipt.json")
            output.touch()
            with self.assertRaisesRegex(
                inference.base.InferenceContractError, "refusing to overwrite"
            ):
                inference.base._resolve_output(root / "result.mp4")


try:
    import torch
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    torch = None
    save_safetensors = None


@unittest.skipIf(torch is None or save_safetensors is None, "torch/safetensors unavailable")
class TensorEGNTCInferenceContractTests(unittest.TestCase):
    def _bundle_files(
        self,
        root: Path,
        *,
        gate: str,
        deployable: bool,
    ) -> tuple[Path, Path, str, str, object]:
        parameters = core.EGNTCParameters()
        raw = parameters.parameter_vector(detach=True).cpu().float().contiguous()
        state = root / f"controller-{gate}-{int(deployable)}.safetensors"
        save_safetensors({inference.CONTROLLER_TENSOR_KEY: raw}, str(state))
        state_sha = inference.file_sha256(state)
        receipt_value = inference.build_controller_training_receipt(
            state_filename=state.name,
            state_file_sha256=state_sha,
            raw_36d=raw,
            representability_gate=gate,
            deployable=deployable,
            training_run_receipt_sha256="b" * 64,
            support_iids=("support-1", "support-2"),
        )
        receipt = root / f"controller-{gate}-{int(deployable)}.receipt.json"
        receipt.write_bytes(inference.canonical_json_bytes(receipt_value) + b"\n")
        return state, receipt, state_sha, inference.file_sha256(receipt), raw

    def test_strict_go_bundle_round_trips_exact_36d_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = self._bundle_files(Path(directory), gate="GO", deployable=True)
            state, receipt, state_sha, receipt_sha, raw = values
            bundle = inference.load_controller_bundle(
                state,
                receipt,
                expected_state_sha256=state_sha,
                expected_receipt_sha256=receipt_sha,
            )
        self.assertEqual(tuple(bundle.raw_36d_cpu.shape), (36,))
        self.assertTrue(torch.equal(bundle.raw_36d_cpu, raw))
        self.assertEqual(bundle.representability_gate, "GO")
        self.assertTrue(bundle.deployable)
        self.assertFalse(bundle.diagnostic_override)

    def test_no_go_and_go_non_deployable_are_default_rejected(self) -> None:
        for gate, deployable in (
            ("NO_GO", False),
            ("GO", False),
            ("NOT_EVALUATED_ENGINEERING_SMOKE", False),
        ):
            with self.subTest(gate=gate, deployable=deployable), tempfile.TemporaryDirectory() as directory:
                state, receipt, state_sha, receipt_sha, _ = self._bundle_files(
                    Path(directory), gate=gate, deployable=deployable
                )
                with self.assertRaisesRegex(
                    inference.EGNTCInferenceError, "not GO/deployable"
                ):
                    inference.load_controller_bundle(
                        state,
                        receipt,
                        expected_state_sha256=state_sha,
                        expected_receipt_sha256=receipt_sha,
                    )
                bundle = inference.load_controller_bundle(
                    state,
                    receipt,
                    expected_state_sha256=state_sha,
                    expected_receipt_sha256=receipt_sha,
                    allow_diagnostic_no_go=True,
                )
                self.assertTrue(bundle.diagnostic_override)

    def test_callback_hard_bypasses_exact_action_noop_parity(self) -> None:
        shape = (1, 2, 21, 2, 2)
        source = torch.randn(shape)
        noop = torch.randn(shape)
        zeros = torch.zeros_like(noop)
        fields = tri.CleanFieldStep(
            step_index=0,
            timestep=float(core.sigma_strata.PINNED_TIMESTEPS[0]),
            sigma=core.sigma_strata.PINNED_POSITIVE_SIGMAS[0],
            model_id="transformer_1",
            noisy=zeros,
            negative_velocity=zeros,
            action_velocity=zeros,
            noop_velocity=zeros,
            negative_clean=zeros,
            action_condition_clean=noop,
            noop_condition_clean=noop,
            action_guided_clean=noop,
            noop_guided_clean=noop,
            action_delta_clean=zeros,
        )
        parameters = core.EGNTCParameters()
        raw = parameters.parameter_vector(detach=True)
        callback = inference.TracedEGNTCCallback(
            source_clean=source,
            parameters=parameters,
            raw_36d_sha256=inference.tensor_sha256(raw),
        )
        executed = callback(fields)
        self.assertIs(executed, fields.action_guided_clean)
        self.assertEqual(len(callback.trace.records), 1)
        self.assertTrue(callback.trace.records[0].action_noop_input_byte_exact)
        self.assertTrue(callback.trace.records[0].parity_bypass_byte_exact)


if __name__ == "__main__":
    unittest.main()
