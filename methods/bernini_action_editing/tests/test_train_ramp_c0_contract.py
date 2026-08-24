from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TRAINER = METHOD_ROOT / "train_ramp_c0.py"

try:
    import torch
except ImportError:  # Local macOS contract environment is intentionally torch-free.
    torch = None

if torch is not None:
    if str(METHOD_ROOT) not in sys.path:
        sys.path.insert(0, str(METHOD_ROOT))
    import train_ramp_c0 as trainer
else:
    trainer = None


class RAMPTrainerStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = TRAINER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_exact81_world8_dp2_sp4_and_step_counts_are_frozen(self) -> None:
        for fragment in (
            "WORLD_SIZE = 8",
            "SP_SIZE = 4",
            "DP_SIZE = 2",
            "FRAME_COUNT = 81",
            "LATENT_PHASES = 21",
            "TIMESTEP = 1000",
            "SIGMA = 1.0",
            "CANARY_STEPS = 1",
            "C0_STEPS = 16",
            "SP_GROUP_RANKS = ((0, 1, 2, 3), (4, 5, 6, 7))",
            "DP_GROUP_RANKS = ((0, 4), (1, 5), (2, 6), (3, 7))",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("41", self.source)

    def test_visual_pack_is_source_program_epsilon_and_never_raw_donor(self) -> None:
        self.assertIn(
            "torch.cat([source_patches, program_patches, epsilon_patches], dim=0)",
            self.source,
        )
        model_inputs = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PreparedModelInputs"
        )
        input_names = {
            node.target.id
            for node in model_inputs.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertEqual(input_names, {"input_patches", "rotary", "layout"})
        prepared = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PreparedArm"
        )
        prepared_names = {
            node.target.id
            for node in prepared.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertFalse(any("donor" in name for name in prepared_names))
        velocity = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_velocity_prediction"
        )
        velocity_source = ast.get_source_segment(self.source, velocity) or ""
        self.assertIn("model_inputs: PreparedModelInputs", velocity_source)
        self.assertNotIn("clean_target", velocity_source)
        self.assertIn("del row, rows, blobs", self.source)
        self.assertIn('"raw_donor_tokens": 0', self.source)
        self.assertNotIn("process_renderer_sample(", self.source)
        self.assertIn("noisy_target_cuda = epsilon.unsqueeze(0).to(device)", self.source)
        self.assertNotIn("target_cuda = target_mode.unsqueeze(0)", self.source)

    def test_route_context_lexically_contains_forward_and_backward(self) -> None:
        route_with = None
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.With):
                continue
            expression = node.items[0].context_expr
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
                and isinstance(expression.func.value, ast.Name)
                and expression.func.value.id == "adapter"
                and expression.func.attr == "route"
            ):
                route_with = node
                break
        self.assertIsNotNone(route_with)
        calls = [node.func for node in ast.walk(route_with) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(isinstance(call, ast.Name) and call.id == "_velocity_prediction" for call in calls)
        )
        self.assertTrue(
            any(
                isinstance(call, ast.Attribute)
                and isinstance(call.value, ast.Name)
                and call.value.id == "backward"
                and call.attr == "backward"
                for call in calls
            )
        )

    def test_gradient_sync_is_sp_mean_then_dp_mean(self) -> None:
        ordered = (
            "dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=parallel.sp_group)",
            "parameter.grad.div_(float(SP_SIZE))",
            "dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=parallel.dp_group)",
            "parameter.grad.div_(float(DP_SIZE))",
        )
        positions = [self.source.index(fragment) for fragment in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("return 2.0 * pair_loss", self.source)
        self.assertIn("pair_loss = paired_prediction_loss(result)", self.source)

    def test_only_explicit_adapter_parameters_reach_optimizer(self) -> None:
        self.assertIn("renderer.requires_grad_(False)", self.source)
        self.assertIn("trainable = adapter.trainable_named_parameters()", self.source)
        self.assertIn(
            '"trainable_scope_exact": "role_embedding+program_projector+target_row_q_lora"',
            self.source,
        )
        self.assertIn("adapter.base_parameters_frozen()", self.source)

    def test_pair_and_same_state_inputs_fail_closed(self) -> None:
        for fragment in (
            '"source_blob_sha256"',
            '"donor_before_blob_sha256"',
            '"instruction"',
            '"bucket_hw"',
            '"posterior_shape"',
            '"vae_identity"',
            '"changed_conditioning_only": "21_token_temporal_transport"',
            "prepared.epsilon_sha256",
            'timestep_token="sigma=1"',
            'noisy_target_sha256=prepared.noisy_target_sha256',
        ):
            self.assertIn(fragment, self.source)
        self.assertIn("distinct programs produced byte-equal target posteriors", self.source)

    def test_authorized_json_is_parsed_from_the_authenticated_bytes(self) -> None:
        self.assertIn("def _strict_json_load_bytes(raw: bytes", self.source)
        self.assertIn("def _read_bound_bytes(", self.source)
        self.assertIn(
            'value = _strict_json_load_bytes(raw, label="RAMP pair config")',
            self.source,
        )
        self.assertIn(
            'value = _strict_json_load_bytes(raw, label="materialized sample receipt")',
            self.source,
        )
        self.assertNotIn("def _strict_json_load(path:", self.source)

    def test_receipt_never_claims_action_or_video_quality(self) -> None:
        for fragment in (
            '"natural_semantic_action_learned": False',
            '"action_editing_claim_authorized": False',
            '"video_quality_claim_authorized": False',
            '"scientific_claim_authorized": False',
            '"pretext_training_only": True',
            '"donor_identity_and_order_losses": "structural_zero_aliases_not_C1_controls"',
        ):
            self.assertIn(fragment, self.source)
        self.assertIn('"teacher_forced_program_reconstruction_optimized": False', self.source)
        self.assertIn(
            '"program_projector_gradient_source": "bernini_velocity_losses_only"',
            self.source,
        )
        self.assertNotIn("adapter.patch_wrapper.program_projector(", self.source)

    def test_output_is_published_as_one_atomic_directory_transaction(self) -> None:
        for fragment in (
            'staging = parent / f".{requested.name}.staging"',
            '"refusing to reuse training output or hidden staging path"',
            "actual_files != expected_files",
            '"staged output contains a non-plain artifact"',
            "_verify_staged_run_bundle(output_stage, receipt)",
            "_fsync_directory(output_stage)",
            "os.replace(output_stage, output)",
            "_fsync_directory(output.parent)",
            '"atomic output publication did not complete"',
        ):
            self.assertIn(fragment, self.source)
        self.assertIn("_durable_file_replace(temporary, path)", self.source)


@unittest.skipIf(torch is None, "AUH vace tensor runtime is required")
class RAMPTrainerTensorHelperTests(unittest.TestCase):
    def test_patch_pack_matches_exact_t_h_w_c_patch_order(self) -> None:
        value = torch.arange(16 * 21 * 4 * 6, dtype=torch.float32).reshape(16, 21, 4, 6)
        patches = trainer.pack_latent_patches(value)
        self.assertEqual(tuple(patches.shape), (21 * 2 * 3, 16, 1, 2, 2))
        first = patches[0, :, 0]
        self.assertTrue(torch.equal(first, value[:, 0, :2, :2]))
        output = trainer.packed_output_field(patches)
        self.assertEqual(tuple(output.shape), (1, 126, 64))
        self.assertTrue(torch.equal(output[0, 0], first.permute(1, 2, 0).reshape(-1)))

    def test_compatibility_logits_are_not_connected_to_program_projector(self) -> None:
        projector = torch.nn.Linear(trainer.LATENT_PHASES, 32, bias=True)
        projector.to(dtype=torch.float32)
        transport = torch.eye(trainer.LATENT_PHASES, dtype=torch.float32).contiguous()
        logits = trainer._objective_compatibility_logits(transport)
        self.assertEqual(
            tuple(logits.shape),
            (1, trainer.LATENT_PHASES, trainer.LATENT_PHASES),
        )
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(logits.requires_grad)
        self.assertTrue(logits.is_contiguous())
        logits.square().mean().backward()
        self.assertIsNone(projector.weight.grad)
        self.assertIsNone(projector.bias.grad)

    def test_remote_gradient_proxy_cannot_reach_remote_parameter(self) -> None:
        theta = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
        remote_value = torch.stack((theta.square(), theta.sin())).reshape(1, 2)
        proxy = trainer._remote_gradient_proxy(remote_value.detach().contiguous())
        proxy.square().sum().backward()
        self.assertIsNotNone(proxy.grad)
        self.assertIsNone(theta.grad)

    def test_dp_partial_autograd_equals_full_paired_parameter_gradient(self) -> None:
        clean_a = torch.tensor([[0.25, -0.50]], dtype=torch.float32)
        clean_b = torch.tensor([[1.00, 0.75]], dtype=torch.float32)
        epsilon = torch.tensor([[0.10, -0.20]], dtype=torch.float32)
        transport_a = torch.eye(trainer.LATENT_PHASES, dtype=torch.float32).contiguous()
        identity = trainer.objective.SameStateInterventionIdentity(
            source_sha256="0" * 64,
            text_sha256="1" * 64,
            epsilon_sha256="2" * 64,
            noisy_target_sha256="2" * 64,
            timestep_token="sigma=1",
            program_a_sha256="3" * 64,
            program_b_sha256="4" * 64,
        )

        def call_objective(theta, *, local_arm, remote_a=None, remote_b=None):
            pred_a_graph = torch.stack((2.0 * theta, -1.5 * theta)).reshape(1, 2)
            pred_b_graph = torch.stack((-3.0 * theta + 0.4, theta + 0.2)).reshape(1, 2)
            pred_a = pred_a_graph if local_arm in {"a", "both"} else remote_a
            pred_b = pred_b_graph if local_arm in {"b", "both"} else remote_b
            logits = trainer._objective_compatibility_logits(transport_a)
            result = trainer.objective.sigma_one_same_state_route_objective(
                pred_a,
                pred_b,
                clean_a,
                clean_b,
                epsilon,
                identity=identity,
                donor_identity_prediction_a=pred_a,
                donor_identity_prediction_b=pred_a,
                order_prediction_a=pred_b,
                order_prediction_b=pred_b,
                transport_logits=logits,
                transport_target=transport_a,
            )
            return result, logits

        initial = torch.tensor(0.3, dtype=torch.float32)
        with torch.no_grad():
            remote_a_value = torch.stack((2.0 * initial, -1.5 * initial)).reshape(1, 2)
            remote_b_value = torch.stack(
                (-3.0 * initial + 0.4, initial + 0.2)
            ).reshape(1, 2)

        theta_a = initial.clone().requires_grad_(True)
        remote_b = trainer._remote_gradient_proxy(remote_b_value.contiguous())
        arm_a, compatibility_a = call_objective(theta_a, local_arm="a", remote_b=remote_b)
        trainer.local_backward_loss(trainer.paired_prediction_loss(arm_a)).backward()
        grad_a = theta_a.grad.detach().clone()
        self.assertIsNotNone(remote_b.grad)
        self.assertIsNone(compatibility_a.grad)

        theta_b = initial.clone().requires_grad_(True)
        remote_a = trainer._remote_gradient_proxy(remote_a_value.contiguous())
        arm_b, compatibility_b = call_objective(theta_b, local_arm="b", remote_a=remote_a)
        trainer.local_backward_loss(trainer.paired_prediction_loss(arm_b)).backward()
        grad_b = theta_b.grad.detach().clone()
        self.assertIsNotNone(remote_a.grad)
        self.assertIsNone(compatibility_b.grad)
        dp_mean_gradient = 0.5 * (grad_a + grad_b)

        theta_reference = initial.clone().requires_grad_(True)
        reference, compatibility_reference = call_objective(
            theta_reference, local_arm="both"
        )
        trainer.paired_prediction_loss(reference).backward()
        self.assertIsNone(compatibility_reference.grad)
        self.assertTrue(
            torch.allclose(dp_mean_gradient, theta_reference.grad, rtol=1.0e-5, atol=1.0e-6)
        )

    def test_nonreentrant_checkpoint_recompute_keeps_same_route_context(self) -> None:
        from torch.utils.checkpoint import checkpoint

        base = torch.nn.Linear(4, 4, bias=False)
        base.requires_grad_(False)
        wrapper = trainer.route.TargetQueryLoRA(base, rank=2, alpha=2.0)
        layout = trainer.route.TokenRoleLayout.contiguous(
            source_tokens=1,
            target_tokens=2,
        )
        invocation = trainer.route.RouteInvocation(
            layout,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
        )
        observed = []

        def record_route(_module, _inputs):
            observed.append(trainer.route.active_route())

        hook = wrapper.register_forward_pre_hook(record_route)
        hidden = torch.randn(1, layout.total_tokens, 4, requires_grad=True)
        try:
            with trainer.route.activate_route(invocation):
                output = checkpoint(wrapper, hidden, use_reentrant=False)
                output.square().sum().backward()
        finally:
            hook.remove()
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(item is invocation for item in observed))
        self.assertIsNotNone(wrapper.lora_b.weight.grad)
        self.assertGreater(float(wrapper.lora_b.weight.grad.abs().sum()), 0.0)

    def test_durable_bundle_writers_roundtrip_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage"
            stage.mkdir()
            parameter = torch.nn.Parameter(torch.tensor([1.25], dtype=torch.float32))
            trainer._atomic_safetensors(stage / "adapter.safetensors", (("p", parameter),))
            trainer._atomic_torch_save(stage / "optimizer.pt", {"step": 1})
            trainer._atomic_json(stage / "history.json", {"steps": []})
            receipt = {
                "schema_version": "test-only",
                "artifacts": {
                    name: trainer.file_sha256(stage / name)
                    for name in ("adapter.safetensors", "optimizer.pt", "history.json")
                },
            }
            receipt["receipt_digest"] = trainer.object_sha256(receipt)
            trainer._atomic_json(stage / "receipt.json", receipt)
            trainer._verify_staged_run_bundle(stage, receipt)
            trainer._fsync_directory(stage)

    def test_rank_topology_assigns_two_sp4_arms_and_four_dp2_lanes(self) -> None:
        for rank in range(8):
            contract = trainer.distributed_contract(
                {
                    "WORLD_SIZE": "8",
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "LOCAL_WORLD_SIZE": "8",
                }
            )
            self.assertEqual(contract.arm_index, rank // 4)
            self.assertEqual(contract.sp_rank, rank % 4)


if __name__ == "__main__":
    unittest.main()
