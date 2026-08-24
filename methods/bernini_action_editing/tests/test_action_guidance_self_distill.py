from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    import action_guidance_self_distill as distill
else:  # pragma: no cover
    distill = None


if torch is not None:
    class _ActionProjection(torch.nn.Module):
        def __init__(self, name: str, value: float) -> None:
            super().__init__()
            self.register_parameter(name, torch.nn.Parameter(torch.tensor(value)))
            self.parameter_name = name

        def forward(self, value):
            return value * getattr(self, self.parameter_name)


    class _Attention2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = _ActionProjection("action_lora_A", 0.08)
            self.to_out = torch.nn.Sequential(
                _ActionProjection("action_lora_B", -0.03)
            )


    class _Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn2 = _Attention2()


    class _Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([_Block()])


    class _DiffDec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = _Transformer()


    class _TinyActionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.diff_dec = _DiffDec()
            self.action_distill_checkpoint_sha256 = "a" * 64
            self.base_gain = torch.nn.Parameter(
                torch.tensor(0.65), requires_grad=False
            )

        def forward(self, state, timestep, condition, *, adapter_enabled):
            scale = condition["scale"].to(device=state.device, dtype=state.dtype)
            time_term = timestep.to(device=state.device, dtype=state.dtype) / 10000.0
            velocity = state * (self.base_gain + scale + time_term)
            if adapter_enabled:
                block = self.diff_dec.transformer.blocks[0]
                velocity = (
                    velocity
                    + block.attn2.to_q(state)
                    + block.attn2.to_out[0](state)
                )
            return velocity


@unittest.skipIf(torch is None, "torch is unavailable")
class ActionGuidanceSelfDistillTests(unittest.TestCase):
    checkpoint_sha = "a" * 64
    action_prompt = "A small brown dog turns its head to the right."

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.fixture_root = Path(self._temporary.name).resolve()
        self.proposal_path = self.fixture_root / "t2v.normalized-clean-latent.safetensors"
        self.proposal_sha = self._write_proposal_artifact(self.proposal_path)
        self.receipt_body = self._native_receipt_body()
        self.receipt_path, self.receipt_sha = self._write_receipt(
            "receipt.json", self.receipt_body
        )
        self.proposal_evidence = distill.load_native_t2v_proposal_evidence(
            receipt_path=self.receipt_path,
            expected_receipt_sha256=self.receipt_sha,
            proposal_artifact_path=self.proposal_path,
            expected_proposal_artifact_sha256=self.proposal_sha,
            rollout_seed=917,
            action_prompt=self.action_prompt,
            checkpoint_sha256=self.checkpoint_sha,
        )
        self._reset_runtime()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _object_sha256(value) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _write_proposal_artifact(cls, path: Path) -> str:
        shape = [1, 16, 21, 2, 2]
        data_nbytes = 4 * 16 * 21 * 2 * 2
        header = {
            "__metadata__": {
                "coordinate": "bernini_normalized_clean_vae_latent",
                "frame_contract": "exact81_latent21",
                "artifact_role": "native_sampler_proposal",
                "source": "native_sampler_before_vae_decode",
            },
            "normalized_clean_latent": {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, data_nbytes],
            },
        }
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        padded_header = raw_header + b" " * ((8 - len(raw_header) % 8) % 8)
        path.write_bytes(
            struct.pack("<Q", len(padded_header))
            + padded_header
            + bytes(data_nbytes)
        )
        return cls._file_sha256(path)

    def _native_receipt_body(self):
        artifact = {
            "path": str(self.proposal_path),
            "sha256": self.proposal_sha,
            "tensor_key": "normalized_clean_latent",
            "shape": [1, 16, 21, 2, 2],
            "stored_dtype": "torch.float32",
            "sampler_return_dtype": "torch.bfloat16",
            "coordinate": "bernini_normalized_clean_vae_latent",
            "artifact_role": "native_sampler_proposal",
            "origin": "native_sampler_before_vae_decode",
            "native_sampler_before_vae_decode": True,
            "source_video_vae_encode_before_any_decode": False,
            "mp4_decode_reencode_used": False,
            "roundtrip_byte_exact_fp32": True,
        }
        return {
            "schema_version": "bernini-native-identity-generation-canary-v1",
            "method": "frozen-bernini-native-identity-generation-canary",
            "arms": ["t2v"],
            "input": {
                "source_video_sha256": "4" * 64,
                "action_prompt_utf8_sha256": hashlib.sha256(
                    self.action_prompt.encode("utf-8")
                ).hexdigest(),
                "accepted_external_conditions": ["source_video", "action_prompt"],
                "target_video": False,
                "paired_target": False,
            },
            "checkpoint": {
                "tree_sha256": self.checkpoint_sha,
                "content": {
                    "every_file_sha256_verified": True,
                    "verified_entries_digest": "5" * 64,
                },
            },
            "conditioning": {
                "t2v": {
                    "full_source_video_count": 0,
                    "source_derived_reference_count": 0,
                    "source_frame_indices": [],
                    "reference_encoding": "none",
                }
            },
            "sampling": {
                "t2v": {
                    "seed": 917,
                    "guidance_mode": "t2v_apg",
                    "target_initialization": "official_gen_wanx22_fresh_gaussian",
                    "target_mixed_with_source_latent": False,
                    "custom_sampler_or_scheduler": False,
                }
            },
            "outputs": {"t2v": {"normalized_clean_latent": artifact}},
            "paired_target_accessed": False,
            "experimental_canary": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }

    def _write_receipt(self, name, body):
        value = json.loads(json.dumps(body))
        value["receipt_digest"] = self._object_sha256(value)
        path = self.fixture_root / name
        path.write_bytes(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        return path, self._file_sha256(path)

    def _reset_runtime(self) -> None:
        self.model = _TinyActionModel().eval()
        generator = torch.Generator(device="cpu").manual_seed(2026)
        self.state = torch.randn((2, 5, 4), generator=generator)
        self.action_condition = {"scale": torch.tensor(0.90)}
        self.noop_condition = {"scale": torch.tensor(0.15)}
        self.calls = []

    @staticmethod
    def timestep(sigma: float) -> torch.Tensor:
        return torch.tensor(sigma * 1000.0, dtype=torch.float32)

    def provenance(self, state=None, action=None, noop=None):
        return distill.bind_provenance(
            noised_state=self.state if state is None else state,
            action_condition=(
                self.action_condition if action is None else action
            ),
            noop_condition=self.noop_condition if noop is None else noop,
            proposal_evidence=self.proposal_evidence,
            proposal_iid="proposal-001",
        )

    def callback(self, model, request):
        self.calls.append(request)
        velocity = model(
            request.noised_state,
            request.timestep,
            request.condition,
            adapter_enabled=request.adapter_enabled,
        )
        return distill.bind_forward_response(request, velocity)

    def kwargs(self, *, sigma=0.72, callback=None, provenance=None, config=None):
        values = {
            "model": self.model,
            "forward_callback": callback or self.callback,
            "noised_state": self.state,
            "timestep": self.timestep(sigma),
            "sigma": sigma,
            "action_condition": self.action_condition,
            "noop_condition": self.noop_condition,
            "provenance": provenance or self.provenance(),
        }
        if config is not None:
            values["config"] = config
        return values

    def test_real_callback_three_forwards_same_query_and_backward_scope_audit(self):
        result = distill.run_action_guidance_self_distill(**self.kwargs())
        self.assertEqual(
            [request.branch for request in self.calls],
            list(distill.FORWARD_BRANCHES),
        )
        self.assertEqual({id(request.noised_state) for request in self.calls}, {id(self.state)})
        self.assertEqual(len({id(request.timestep) for request in self.calls}), 1)
        self.assertEqual({request.rollout_seed for request in self.calls}, {917})
        self.assertEqual(
            {request.proposal_sha256 for request in self.calls}, {self.proposal_sha}
        )
        self.assertEqual(
            [request.adapter_enabled for request in self.calls], [False, False, True]
        )
        self.assertEqual({request.mode for request in self.calls}, {"t2v"})
        forward = result.forward
        self.assertEqual(forward.loss.dtype, torch.float32)
        self.assertFalse(forward.teacher_residual.requires_grad)
        self.assertIsNone(forward.teacher_residual.grad_fn)
        self.assertTrue(forward.student_residual.requires_grad)
        audit = result.gradient_audit
        self.assertTrue(audit.passed)
        self.assertGreater(audit.total_fp32_gradient_energy, 0.0)
        self.assertEqual(
            set(audit.finite_gradient_names), set(audit.allowed_parameter_names)
        )
        self.assertTrue(audit.nonzero_gradient_names)
        self.assertEqual(audit.forbidden_parameter_gradients, ())
        self.assertIsNone(self.model.base_gain.grad)
        for name in audit.allowed_parameter_names:
            self.assertIn(".attn2.", name)
            self.assertTrue(".to_q." in name or ".to_out.0." in name)
            self.assertIn("action_lora", name)
        self.assertTrue(result.receipt["backward_gradient_audit_passed"])
        self.assertEqual(result.receipt["query_binding"]["mode"], "t2v")

    def test_teacher_is_detached_bounded_velocity_prior_not_pixel_target(self):
        config = distill.DistillConfig(
            raw_residual_l2_cap=0.8,
            teacher_max_reference_rms_ratio=1.25,
            student_base_trust_ratio=0.04,
        )
        forward = distill.build_distill_forward(**self.kwargs(config=config))
        teacher = forward.diagnostics["teacher"]
        for observed, bound in zip(
            teacher["teacher_residual_rms"], teacher["teacher_rms_cap"]
        ):
            self.assertLessEqual(observed, bound + 2.0e-6)
        self.assertTrue(teacher["raw_l2_clip_active"])
        for observed, radius in zip(
            forward.diagnostics["student_trusted_correction_rms"],
            forward.diagnostics["student_base_trust_radius"],
        ):
            self.assertLessEqual(observed, radius + 2.0e-6)
        receipt = forward.receipt
        self.assertTrue(receipt["model_self_generated_prior"])
        self.assertFalse(receipt["pixel_target_supervision"])
        self.assertFalse(receipt["teacher_rgb_received"])
        self.assertFalse(receipt["teacher_clean_latent_received"])
        self.assertFalse(receipt["proposal_clean_latent_received"])
        self.assertTrue(receipt["teacher_residual_detached"])
        self.assertEqual(receipt["energy"]["dtype"], "torch.float32")

    def test_high_mid_gate_and_low_sigma_rejection(self):
        self.assertEqual(distill.sigma_gate(0.8), ("high", 1.0))
        self.assertEqual(distill.sigma_gate(0.4), ("mid", 0.5))
        self.assertEqual(distill.sigma_gate(0.1), ("low_ineligible", 0.0))
        mid = distill.build_distill_forward(**self.kwargs(sigma=0.4))
        self.assertEqual(mid.sigma_stratum, "mid")
        self.assertEqual(mid.sigma_gate_weight, 0.5)
        # This forward has built a graph but has not run backward; discard the
        # model and calls before testing the ineligible branch.
        self._reset_runtime()
        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "low-sigma query is ineligible"
        ):
            distill.build_distill_forward(**self.kwargs(sigma=0.1))
        self.assertEqual(self.calls, [])

    def test_provenance_binds_checkpoint_proposal_state_conditions_and_seed(self):
        good = self.provenance()
        variants = (
            replace(good, student_base_checkpoint_sha256="c" * 64),
            replace(good, query_state_sha256="d" * 64),
            replace(good, action_condition_sha256="e" * 64),
            replace(good, proposal_origin="paired_target"),
            replace(good, rollout_seed=-1),
        )
        for provenance in variants:
            self._reset_runtime()
            with self.subTest(provenance=provenance):
                with self.assertRaises(distill.ActionGuidanceSelfDistillError):
                    distill.build_distill_forward(
                        **self.kwargs(provenance=provenance)
                    )
                self.assertEqual(self.calls, [])
            self._reset_runtime()
        self.model.action_distill_checkpoint_sha256 = "f" * 64
        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError,
            "runtime model checkpoint binding differs",
        ):
            distill.build_distill_forward(**self.kwargs())
        self.assertEqual(self.calls, [])

    def test_callback_must_return_bound_full_nonleaf_model_velocity(self):
        def raw_tensor(model, request):
            return model(
                request.noised_state,
                request.timestep,
                request.condition,
                adapter_enabled=request.adapter_enabled,
            )

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError,
            "bound ModelForwardResponse",
        ):
            distill.build_distill_forward(**self.kwargs(callback=raw_tensor))

        self._reset_runtime()

        def scalar_response(model, request):
            model(
                request.noised_state,
                request.timestep,
                request.condition,
                adapter_enabled=request.adapter_enabled,
            )
            return distill.bind_forward_response(request, torch.tensor(1.0))

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "full finite velocity"
        ):
            distill.build_distill_forward(**self.kwargs(callback=scalar_response))

        self._reset_runtime()

        def leaf_student(model, request):
            model_velocity = model(
                request.noised_state,
                request.timestep,
                request.condition,
                adapter_enabled=request.adapter_enabled,
            )
            if request.branch == distill.STUDENT_ACTION_BRANCH:
                velocity = torch.ones_like(request.noised_state, requires_grad=True)
            else:
                velocity = model_velocity
            return distill.bind_forward_response(request, velocity)

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "non-leaf model-forward tensor"
        ):
            distill.build_distill_forward(**self.kwargs(callback=leaf_student))

    def test_cross_seed_response_and_query_mutation_fail_closed(self):
        def wrong_seed(model, request):
            response = self.callback(model, request)
            return replace(response, rollout_seed=request.rollout_seed + 1)

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "provenance differs"
        ):
            distill.build_distill_forward(**self.kwargs(callback=wrong_seed))

        self._reset_runtime()

        def mutating(model, request):
            request.noised_state.add_(0.5)
            velocity = model(
                request.noised_state,
                request.timestep,
                request.condition,
                adapter_enabled=request.adapter_enabled,
            )
            return distill.bind_forward_response(request, velocity)

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "mutated the shared query state"
        ):
            distill.build_distill_forward(**self.kwargs(callback=mutating))

        self._reset_runtime()

        def cloned_state(model, request):
            velocity = model(
                request.noised_state.clone(),
                request.timestep,
                request.condition,
                adapter_enabled=request.adapter_enabled,
            )
            return distill.bind_forward_response(request, velocity)

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError,
            "exactly one model forward with the bound state",
        ):
            distill.build_distill_forward(**self.kwargs(callback=cloned_state))

    def test_forbidden_trainable_scope_and_zero_allowed_gradient_fail(self):
        self.model.base_gain.requires_grad_(True)
        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "not attn2 Q/O Action-LoRA"
        ):
            distill.build_distill_forward(**self.kwargs())

        self._reset_runtime()

        def zero_lora(model, request):
            velocity = model(
                request.noised_state,
                request.timestep,
                request.condition,
                adapter_enabled=request.adapter_enabled,
            )
            if request.branch == distill.STUDENT_ACTION_BRANCH:
                zero = sum(
                    parameter.sum() * 0.0
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                )
                velocity = velocity.detach() + zero
            return distill.bind_forward_response(request, velocity)

        forward = distill.build_distill_forward(
            **self.kwargs(callback=zero_lora)
        )
        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError,
            "aggregate gradient energy is zero",
        ):
            distill.backward_and_audit(self.model, forward)

    def test_condition_payload_rejects_teacher_rgb_and_clean_latent(self):
        for key in ("teacher_rgb", "teacher_clean_latent", "pixel_target"):
            condition = {
                "scale": torch.tensor(0.9),
                key: torch.zeros(1, 3, 8, 8),
            }
            with self.subTest(key=key), self.assertRaisesRegex(
                distill.ActionGuidanceSelfDistillError,
                "forbidden teacher pixel/clean target field",
            ):
                distill.bind_provenance(
                    noised_state=self.state,
                    action_condition=condition,
                    noop_condition=self.noop_condition,
                    proposal_evidence=self.proposal_evidence,
                    proposal_iid="proposal-001",
                )

    def test_native_t2v_receipt_is_content_addressed_and_paired_target_fails(self):
        evidence = self.proposal_evidence
        self.assertEqual(evidence.arm, "t2v")
        self.assertEqual(evidence.proposal_artifact_sha256, self.proposal_sha)
        self.assertEqual(evidence.receipt_file_sha256, self.receipt_sha)
        self.assertEqual(evidence.rollout_seed, 917)
        self.assertFalse(evidence.target_video)
        self.assertFalse(evidence.paired_target)

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "loader-validated native T2V"
        ):
            distill.bind_provenance(
                noised_state=self.state,
                action_condition=self.action_condition,
                noop_condition=self.noop_condition,
                proposal_evidence="b" * 64,
                proposal_iid="proposal-001",
            )

        paired = json.loads(json.dumps(self.receipt_body))
        paired["input"]["paired_target"] = True
        paired_path, paired_sha = self._write_receipt("paired.json", paired)
        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "target access flag"
        ):
            distill.load_native_t2v_proposal_evidence(
                receipt_path=paired_path,
                expected_receipt_sha256=paired_sha,
                proposal_artifact_path=self.proposal_path,
                expected_proposal_artifact_sha256=self.proposal_sha,
                rollout_seed=917,
                action_prompt=self.action_prompt,
                checkpoint_sha256=self.checkpoint_sha,
            )

        with self.assertRaisesRegex(
            distill.ActionGuidanceSelfDistillError, "artifact provenance differs"
        ):
            distill.load_native_t2v_proposal_evidence(
                receipt_path=self.receipt_path,
                expected_receipt_sha256=self.receipt_sha,
                proposal_artifact_path=self.proposal_path,
                expected_proposal_artifact_sha256="b" * 64,
                rollout_seed=917,
                action_prompt=self.action_prompt,
                checkpoint_sha256=self.checkpoint_sha,
            )

    def test_tensor_hash_uses_contiguous_byte_storage_without_python_int_lists(self):
        value = torch.arange(48, dtype=torch.float32).reshape(4, 12)[:, ::2]
        self.assertFalse(value.is_contiguous())
        detached = value.detach().contiguous().cpu()
        octets = detached.reshape(-1).view(torch.uint8).contiguous().clone()
        untyped_storage = getattr(octets, "untyped_storage", None)
        storage = untyped_storage() if callable(untyped_storage) else octets.storage()
        raw = bytes(storage)
        header = json.dumps(
            {
                "shape": [int(item) for item in detached.shape],
                "dtype": str(detached.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        digest = hashlib.sha256()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        self.assertEqual(distill.tensor_sha256(value), digest.hexdigest())
        self.assertEqual(distill.tensor_sha256(value), distill.tensor_sha256(value.clone()))
        self.assertNotEqual(
            distill.tensor_sha256(value), distill.tensor_sha256(value + 1.0)
        )
        self.assertRegex(distill.tensor_sha256(torch.tensor(3.5)), r"^[0-9a-f]{64}$")
        source = inspect.getsource(distill.tensor_sha256)
        self.assertNotIn(".tolist()", source)
        self.assertIn("untyped_storage", source)
        self.assertIn("storage_offset", source)
        self.assertIn("nbytes", source)
        self.assertIn("bytes(storage)", source)

    def test_public_bridge_has_no_teacher_rgb_or_clean_latent_argument(self):
        parameters = set(inspect.signature(distill.build_distill_forward).parameters)
        self.assertNotIn("teacher_rgb", parameters)
        self.assertNotIn("teacher_clean_latent", parameters)
        self.assertNotIn("teacher_target", parameters)
        result = distill.run_action_guidance_self_distill(**self.kwargs())
        serialized = json.dumps(result.receipt, sort_keys=True)
        self.assertIn('"cross_state_vector_forbidden": true', serialized)
        self.assertIn('"cross_seed_vector_forbidden": true', serialized)
        self.assertIn('"only_attn2_q_out_action_lora": true', serialized)
        self.assertIn('"backward_gradient_audit_passed": true', serialized)
        self.assertIn('"native_t2v_proposal_evidence_verified": true', serialized)
        self.assertIn('"trainer_integration_authorized": false', serialized)


if __name__ == "__main__":
    unittest.main()
