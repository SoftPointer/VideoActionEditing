from __future__ import annotations

import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = METHOD_ROOT / "train_elal3_c1_simulator_overfit_v1.py"
PACKET_ROOT = METHOD_ROOT.parents[1] / "md/action_editing/20260817_box/simulator_gt_canary_v1"
EXTERNAL_AUTHORITY = (
    METHOD_ROOT.parents[1]
    / "md/action_editing/20260817_box/evidence/elal3_c1_simulator_optimizer_diagnostic_authority_v1.json"
)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_elal3_c1_simulator_overfit_v1 as trainer

try:
    import torch
    import elal3_c0_v1 as elal3_core
    import elal3_simulator_label_v1 as label_module

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    elal3_core = None  # type: ignore[assignment]
    label_module = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class ELAL3C1TrainerStaticTests(unittest.TestCase):
    def test_source_parses_and_contract_markers_exist(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for marker in (
            "LORA_AFFINES = 240",
            "LORA_RANK = 256",
            "ELAL3_FULL_W64_PARAMETERS = 9_979_934",
            "MEMORY_FRACTION_GATE = 0.5",
            "LATENT_SHAPE = (1, 16, 21, 52, 70)",
            "PACKED_TOTAL_TOKENS = 38_220",
            "LOCAL_SP_TOKENS = 9_555",
            'ACTIVATION_CHECKPOINT_PROFILE = "selective-nonreentrant-stride4-exact8"',
            "def partitioned_flow_matching_loss_v1",
            "def all_trainable_graph_zero_v1",
            "def synchronize_initial_parameters_v1",
            "def install_selective_activation_checkpointing_v1",
            "def replay_model_authority_world8_v1",
            "dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.sp_group)",
            "dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.dp_group)",
            '"simulator_signed_motion_used_as_diffusion_velocity": False',
            '"frozen_teacher_used": False',
            '"self_distillation_used": False',
            '"reward_used": False',
            '"source_instruction_inference": False',
            '"memory_gate_all_steps_all8_strictly_gt_half": True',
            '"TRAINING_RECEIPT.json"',
        ):
            self.assertIn(marker, source)
        for forbidden in (
            "Full30ActionRuntimeOutputsV1",
            "frozen_source_action_velocity",
            "saic_event_reward",
            "ActionPlanPredictorV1(",
            "runtime.synchronize_initial_parameters(",
            "gradient_checkpointing_enable(",
        ):
            self.assertNotIn(forbidden, source)

    def test_external_authority_is_exact_and_separately_issued(self) -> None:
        value = trainer.validate_external_optimizer_authority(
            EXTERNAL_AUTHORITY.resolve(),
            expected_sha256=trainer.EXTERNAL_OPTIMIZER_AUTHORITY_SHA256,
        )
        self.assertEqual(value["authorized_row_id"], trainer.ROW_ID)
        self.assertEqual(value["max_optimizer_updates_per_arm"], 20)
        self.assertTrue(value["oracle_q_teacher_forced_required"])
        with self.assertRaisesRegex(trainer.ELAL3C1TrainingError, "literal SHA"):
            trainer.validate_external_optimizer_authority(
                EXTERNAL_AUTHORITY.resolve(), expected_sha256="0" * 64
            )

    def test_cli_is_closed_to_one_ten_or_twenty_and_five_acks(self) -> None:
        digest = "1" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "elal3_c1_fresh")
            argv = [
                "--bernini-root", "/b", "--veomni-root", "/v",
                "--checkpoint", "/c", "--packet-root", "/p",
                "--latent-bundle", "/l", "--expected-latent-bundle-sha256",
                trainer.LATENT_BUNDLE_SHA256,
                "--latent-bundle-receipt", "/r",
                "--expected-latent-bundle-receipt-sha256",
                trainer.LATENT_BUNDLE_RECEIPT_SHA256,
                "--external-optimizer-authority", "/a",
                "--expected-external-optimizer-authority-sha256",
                trainer.EXTERNAL_OPTIMIZER_AUTHORITY_SHA256,
                "--model-authority", "/m",
                "--expected-model-authority-sha256", trainer.MODEL_AUTHORITY_SHA256,
                "--output", output, "--max-steps", "10", "--seed", "20260817",
                "--expected-runner-source-sha256", digest,
                "--expected-train-lora-source-sha256", digest,
                "--expected-elal3-core-source-sha256", digest,
                "--expected-elal3-label-source-sha256", digest,
                "--expected-packed-lora-source-sha256", digest,
                "--expected-runtime-source-sha256", digest,
                "--expected-sigma-source-sha256", digest,
                "--ack-simulator-oracle-q-overfit-only",
                "--ack-not-source-instruction-inference",
                "--ack-not-formal-c1", "--ack-not-exact160",
                "--ack-no-scientific-claim",
            ]
            args = trainer.parser().parse_args(argv)
            trainer.validate_args(args)
            self.assertEqual(args.max_steps, 10)
            args.expected_latent_bundle_sha256 = "0" * 64
            with self.assertRaisesRegex(trainer.ELAL3C1TrainingError, "registered v2 literal"):
                trainer.validate_args(args)
            args.expected_latent_bundle_sha256 = trainer.LATENT_BUNDLE_SHA256
            args.expected_latent_bundle_receipt_sha256 = "0" * 64
            with self.assertRaisesRegex(trainer.ELAL3C1TrainingError, "registered v2 literal"):
                trainer.validate_args(args)
            args.expected_latent_bundle_receipt_sha256 = trainer.LATENT_BUNDLE_RECEIPT_SHA256
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parser().parse_args(
                    [*argv[: argv.index("10")], "2", *argv[argv.index("10") + 1 :]]
                )
            args.ack_no_scientific_claim = False
            with self.assertRaisesRegex(trainer.ELAL3C1TrainingError, "five"):
                trainer.validate_args(args)

    def test_registered_sp4_partition_is_exact_and_non_straddling(self) -> None:
        rows = [
            trainer.registered_sp4_partition_v1(
                total_tokens=trainer.PACKED_TOTAL_TOKENS,
                condition_tokens=trainer.TOKENS_PER_ROLE,
                sp_rank=rank,
            )
            for rank in range(4)
        ]
        self.assertEqual([row["local_tokens"] for row in rows], [9555] * 4)
        self.assertEqual([row["source_only"] for row in rows], [True, True, False, False])
        self.assertEqual([row["target_only"] for row in rows], [False, False, True, True])
        with self.assertRaisesRegex(trainer.ELAL3C1TrainingError, "partition inputs"):
            trainer.registered_sp4_partition_v1(
                total_tokens=trainer.PACKED_TOTAL_TOKENS - 4,
                condition_tokens=trainer.TOKENS_PER_ROLE,
                sp_rank=0,
            )

    def test_model_authority_replay_rejects_hostile_object_change(self) -> None:
        reference = {"schema_version": "x", "files": [{"sha256": "a" * 64}]}
        accepted = trainer.require_model_authority_replay_identity_v1(
            reference, dict(reference), stage="post_deserialize"
        )
        self.assertTrue(accepted["world8_broadcast_identity_verified"])
        hostile = {"schema_version": "x", "files": [{"sha256": "b" * 64}]}
        with self.assertRaisesRegex(trainer.ELAL3C1TrainingError, "replay differs"):
            trainer.require_model_authority_replay_identity_v1(
                reference, hostile, stage="final_pre_publish"
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class ELAL3C1TrainerFunctionalTests(unittest.TestCase):
    def test_event_context_equal_partition_uses_diffusion_velocity(self) -> None:
        prediction = torch.zeros((1, 1, 4), dtype=torch.float32)
        target = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]])
        event = torch.tensor([[[True, True, False, False]]])
        context = ~event
        loss, receipt = trainer.partitioned_flow_matching_loss_v1(
            prediction, target, event, context
        )
        # event MSE=(1+9)/2=5; context=(4+16)/2=10; equal mean=7.5
        self.assertEqual(float(loss.item()), 7.5)
        self.assertEqual(receipt["fixed_partition_coefficients"], [0.5, 0.5])
        self.assertFalse(receipt["tunable_loss_weights"])
        self.assertFalse(receipt["simulator_signed_motion_used_as_diffusion_velocity"])
        with self.assertRaisesRegex(trainer.ELAL3C1TrainingError, "disjoint exhaustive"):
            trainer.partitioned_flow_matching_loss_v1(
                prediction, target, event, event
            )

    def test_pack_mask_matches_native_patch_vector_order(self) -> None:
        mask = torch.zeros((1, 1, 21, 2, 2), dtype=torch.bool)
        mask[:, :, :, 0, 0] = True
        target = torch.zeros((1, 21, 64), dtype=torch.float32)
        packed = trainer.pack_vae_partition_mask_v1(mask, target_velocity=target)
        self.assertEqual(tuple(packed.shape), (1, 21, 64))
        self.assertEqual(int(packed.sum().item()), 21 * 16)

    def test_all_trainable_graph_zero_materializes_explicit_zero_grads(self) -> None:
        named = [
            (f"p{index}", torch.nn.Parameter(torch.ones(()))) for index in range(668)
        ]
        reference = named[0][1] * 2.0
        zero = trainer.all_trainable_graph_zero_v1(named, reference=reference)
        zero.backward()
        self.assertEqual(float(zero.item()), 0.0)
        self.assertTrue(all(parameter.grad is not None for _, parameter in named))
        self.assertTrue(all(float(parameter.grad.item()) == 0.0 for _, parameter in named))

    def test_scalar_byte_digest_and_local_world8_initial_sync(self) -> None:
        class FakeDist:
            def __init__(self) -> None:
                self.broadcast_shapes = []

            def broadcast(self, tensor, *, src, group) -> None:
                self.assertions = (src, group)
                self.broadcast_shapes.append(tuple(tensor.shape))

            @staticmethod
            def all_gather_object(gathered, value, *, group) -> None:
                gathered[:] = [value] * trainer.WORLD_SIZE

        named = tuple(
            (f"scalar_{index:03d}", torch.nn.Parameter(torch.tensor(float(index))))
            for index in range(668)
        )
        before = trainer.trainable_digest_v1(named)
        scalar_tensor_digest = trainer.tensor_sha256_v1(named[0][1].detach().cpu())
        fake = FakeDist()
        synchronized = trainer.synchronize_initial_parameters_v1(
            named, "fake-world", dist_module=fake
        )
        self.assertEqual(synchronized, before)
        self.assertRegex(scalar_tensor_digest, r"^[0-9a-f]{64}$")
        self.assertEqual(len(fake.broadcast_shapes), 668)
        self.assertEqual(fake.broadcast_shapes[0], ())
        self.assertEqual(fake.assertions, (0, "fake-world"))

    def test_selective_exact8_checkpointing_replays_elal_route(self) -> None:
        class FakeBlock:
            def __init__(self, index) -> None:
                self.index = index

            def forward(self, value):
                return value + self.index

        class FakeTransformer:
            def __init__(self) -> None:
                self.blocks = [FakeBlock(index) for index in range(30)]
                self.gradient_checkpointing = False

        class FakeBase:
            def __init__(self) -> None:
                self.diff_dec = type("DiffDec", (), {"transformer": FakeTransformer()})()

        class FakeModel:
            def __init__(self) -> None:
                self.base = FakeBase()

            def get_base_model(self):
                return self.base

        captured = {}

        def fake_checkpoint(function, *args, use_reentrant, context_fn, **kwargs):
            captured["use_reentrant"] = use_reentrant
            captured["contexts"] = context_fn()
            return function(*args, **kwargs)

        memory = elal3_core.ELAL3ActionMemoryV1(
            tokens=torch.zeros((1, 210, 256), dtype=torch.float32),
            valid=torch.ones((1, 210), dtype=torch.bool),
            local_tokens=torch.zeros((1, 21, 64), dtype=torch.float32),
            local_grid=(21, 1, 1),
            variant="full",
        )
        route = elal3_core.ELAL3RouteV1(
            total_tokens=42,
            condition_tokens=21,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
            memory=memory,
            route_identity="selective-checkpoint-test",
        )
        model = FakeModel()
        chosen = trainer.install_selective_activation_checkpointing_v1(
            model,
            context_fn=elal3_core.elal3_checkpoint_context_fn_v1,
            checkpoint_fn=fake_checkpoint,
        )
        self.assertEqual(chosen, tuple(range(0, 30, 4)))
        value = torch.ones((), requires_grad=True)
        with elal3_core.activate_elal3_route_v1(route):
            self.assertEqual(float(model.base.diff_dec.transformer.blocks[0].forward(value)), 1.0)
            self.assertEqual(float(model.base.diff_dec.transformer.blocks[1].forward(value)), 2.0)
        self.assertFalse(captured["use_reentrant"])
        self.assertIsNone(elal3_core.active_elal3_route_v1())
        with captured["contexts"][1]:
            self.assertIs(elal3_core.active_elal3_route_v1(), route)

    def test_label_and_derived_authority_are_bound_before_training(self) -> None:
        label = label_module.load_oracle_q_label_v1(
            PACKET_ROOT,
            row_id=trainer.ROW_ID,
            patch_grid=(21, 6, 8),
            external_authority_path=EXTERNAL_AUTHORITY,
            external_authority_sha256=trainer.EXTERNAL_OPTIMIZER_AUTHORITY_SHA256,
            device="cpu",
            dtype=torch.float32,
        )
        authority = label_module.build_derivative_authority_v1(
            label,
            external_authority_path=EXTERNAL_AUTHORITY,
            external_authority_sha256=trainer.EXTERNAL_OPTIMIZER_AUTHORITY_SHA256,
        )
        checked = trainer.validate_derivative_authority_v1(
            authority, label_receipt=label.receipt, max_steps=10
        )
        self.assertEqual(checked["scope"]["allowed_optimizer_updates_max"], 20)
        self.assertTrue(checked["authority"]["external_optimizer_authority_verified"])
        self.assertFalse(checked["authority"]["source_instruction_inference_authorized"])


if __name__ == "__main__":
    unittest.main()
