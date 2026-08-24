from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import fewshot_episode_io as episode_io  # noqa: E402
import train_fewshot_motion_code as trainer  # noqa: E402

try:
    import torch
    import fewshot_privileged_motion_code as epmc
except ImportError:  # pragma: no cover - dependency-light environment
    torch = None
    epmc = None


def _valid_args(**updates):
    values = dict(
        bernini_root="/bernini",
        veomni_root="/veomni",
        checkpoint="/checkpoint",
        k2_config="/config.json",
        expected_k2_config_sha256="1" * 64,
        preview_manifest="/preview.jsonl",
        vae_index="/vae.jsonl",
        output="/absolute/output",
        num_frames=81,
        k_shot=2,
        steps_per_support=50,
        learning_rate=0.05,
        max_grad_norm=1.0,
        seed=20260808,
        proposal_seed=2027,
        fixed_sigma_index=20,
        held_sigma_index=32,
        full_target_fm_weight=0.0,
        engineering_smoke=False,
        posthoc_heldout_eval=False,
        ack_preview_experimental_only=True,
        expected_bernini_commit=trainer.legacy.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=trainer.legacy.VEOMNI_TESTED_COMMIT,
        expected_checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
        method_source_revision="2" * 40,
        method_source_archive_sha256="3" * 64,
    )
    values.update(updates)
    return argparse.Namespace(**values)


class StaticRunnerContractTests(unittest.TestCase):
    def test_sigma_lookup_uses_the_registered_numeric_shift(self):
        class Scalar:
            def reshape(self, *_shape):
                return self

            def numel(self):
                return 1

        timestep = Scalar()
        sigma = Scalar()
        flow = mock.Mock()
        flow.get_noise_sigma.return_value = sigma
        scheduler = types.SimpleNamespace(
            shift_config={"default": 3.0, "mv2v": 5.0},
            flow_scheduler={5.0: {"scheduler": flow}},
        )
        result = trainer._sigma_for_batch(scheduler, {"timesteps": timestep})
        self.assertIs(result, sigma)
        flow.get_noise_sigma.assert_called_once_with(timestep)

    def test_exact_episode_loader_is_the_only_config_interpreter(self):
        marker = object()
        args = _valid_args()
        with mock.patch.object(
            episode_io, "load_epmc_k2_canary", return_value=marker
        ) as loader:
            self.assertIs(trainer.load_audited_episode(args), marker)
        loader.assert_called_once_with(
            args.k2_config,
            args.preview_manifest,
            args.vae_index,
            experimental_training_acknowledged=True,
            expected_config_sha256=args.expected_k2_config_sha256,
        )

    def test_obsolete_parser_transpose_and_target_carrier_are_absent(self):
        source = inspect.getsource(trainer)
        for forbidden in (
            "load_k2_selection",
            "K2_CONFIG_SCHEMA",
            "_packed_patch_field",
            "explicit_hw_transpose",
            "build_motion_carrier(target_field",
            "expected_preview_manifest_sha256",
            "expected_vae_index_sha256",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("episode_io.load_epmc_k2_canary", source)
        self.assertIn("build_carrier_from_proposal_latents", source)
        self.assertIn("expected_patch_grid=PATCH_GRID_YX", source)
        self.assertEqual(trainer.PATCH_GRID_YX, (30, 31))

    def test_cli_is_81f_k2_acknowledged_and_full_target_zero(self):
        trainer.validate_cli(_valid_args())
        invalid = (
            {"ack_preview_experimental_only": False},
            {"num_frames": 41},
            {"k_shot": 1},
            {"full_target_fm_weight": 0.01},
            {"fixed_sigma_index": 20, "held_sigma_index": 20},
            {"engineering_smoke": True, "posthoc_heldout_eval": True},
        )
        for update in invalid:
            with self.subTest(update=update):
                with self.assertRaises(trainer.FewShotCodeTrainingError):
                    trainer.validate_cli(_valid_args(**update))

    def test_source_contains_explicit_apg_and_distributed_gradient_contracts(self):
        source = inspect.getsource(trainer)
        self.assertIn("prior._guided_clean", source)
        self.assertIn("negative_velocity=base.negative_velocity", source)
        self.assertIn("teacher_objective.fewshot_teacher_objective", source)
        self.assertIn("objective.detached_statistics()", source)
        self.assertNotIn("objective.as_dict()", source)
        self.assertIn("teacher_objective.build_held_noise_statistics", source)
        self.assertIn("teacher_objective.evaluate_teacher_go", source)
        self.assertNotIn("motion.causal_boundary_charbonnier_loss", source)
        self.assertIn("group=ulysses_group", source)
        self.assertIn("parameter.grad.div_(float(ULYSSES_SIZE))", source)
        self.assertIn("dist.all_gather(gathered, local_tied, group=dp_group)", source)
        self.assertIn("context=\"ordered K=2 held-noise controls\"", source)
        self.assertIn("context=\"ordered K=2 gradient probes\"", source)
        self.assertIn("parallel.support_index == 0", source)
        self.assertIn("episode_parallel.REFERENCE_PROBE_FAMILIES", source)
        self.assertIn("held_controls_for_aggregation = sorted", source)
        self.assertIn('compact_auxiliary["sigma"].device.type != "cpu"', source)
        self.assertIn("full_target_flow_matching_weight", source)
        self.assertIn("single_step_apg_surrogate", source)

    def test_every_training_collective_has_an_explicit_group(self):
        tree = ast.parse(inspect.getsource(trainer))
        collective_names = {
            "all_reduce",
            "all_gather",
            "all_gather_object",
            "barrier",
        }
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in collective_names:
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {
                "dist",
                "dist_module",
            }:
                continue
            calls.append((node.func.attr, node.lineno, {item.arg for item in node.keywords}))
        self.assertTrue(calls)
        self.assertEqual(
            [],
            [
                (name, line)
                for name, line, keywords in calls
                if "group" not in keywords
            ],
        )

    def test_exact_world8_dp2_sp4_environment_and_assignments(self):
        for rank in range(8):
            contract = trainer.epmc_distributed_contract(
                {
                    "WORLD_SIZE": "8",
                    "LOCAL_WORLD_SIZE": "8",
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                }
            )
            self.assertEqual(contract.support_index, rank // 4)
            self.assertEqual(contract.ulysses_rank, rank % 4)
            self.assertEqual(contract.ulysses_size, 4)
            self.assertEqual(contract.data_parallel_size, 2)
        invalid = (
            {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"},
            {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"},
            {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "RANK": "4", "LOCAL_RANK": "0"},
            {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "RANK": "8", "LOCAL_RANK": "8"},
        )
        for environment in invalid:
            with self.subTest(environment=environment):
                with self.assertRaises(trainer.FewShotCodeTrainingError):
                    trainer.epmc_distributed_contract(environment)

        supports = [
            types.SimpleNamespace(iid="841b5e0080a1441d"),
            types.SimpleNamespace(iid="7262dd490cbf42c5"),
        ]
        assignments = trainer._support_assignments(supports)
        self.assertEqual(
            assignments,
            [
                {
                    "support_index": 1,
                    "iid": "841b5e0080a1441d",
                    "dp_rank": 0,
                    "sp_ranks": [0, 1, 2, 3],
                },
                {
                    "support_index": 2,
                    "iid": "7262dd490cbf42c5",
                    "dp_rank": 1,
                    "sp_ranks": [4, 5, 6, 7],
                },
            ],
        )
        receipt = trainer._distributed_receipt(
            support_assignments=assignments, backend="nccl/rccl"
        )
        self.assertEqual(receipt["world_size"], 8)
        self.assertEqual(receipt["ulysses_size"], 4)
        self.assertEqual(receipt["data_parallel_size"], 2)
        self.assertIs(receipt["support_parallel"], True)
        self.assertEqual(receipt["sp_groups"], [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(receipt["dp_groups"], [[0, 4], [1, 5], [2, 6], [3, 7]])
        self.assertEqual(receipt["support_assignments"], assignments)
        self.assertEqual(
            receipt["within_support_gradient_sync"],
            "ulysses_group_all_reduce_sum_then_divide_by_4",
        )
        self.assertIs(receipt["cross_support_gradient_sync"], False)
        self.assertEqual(
            receipt["k2_exchange"],
            "data_parallel_lane_all_gather_then_world_digest_consensus",
        )

    def test_parallel_state_membership_is_exact_for_all_global_ranks(self):
        class FakeDist:
            group = types.SimpleNamespace(WORLD="world")

            def __init__(self, global_rank):
                self.global_rank = global_rank

            def get_world_size(self, group=None):
                if group == "world" or group is None:
                    return 8
                return 4 if group.startswith("sp") else 2

            def get_rank(self, group=None):
                if group == "world" or group is None:
                    return self.global_rank
                if group.startswith("sp"):
                    return self.global_rank % 4
                return self.global_rank // 4

            def all_gather_object(self, output, value, *, group):
                if group.startswith("sp"):
                    index = int(group[-1])
                    output[:] = list(trainer.SP_GROUP_RANKS[index])
                else:
                    lane = int(group[-1])
                    output[:] = list(trainer.DP_GROUP_RANKS[lane])

        for rank in range(8):
            contract = trainer.EPMCDistributedContract(8, rank, rank, 8, 4, 2)
            state = types.SimpleNamespace(
                world_size=8,
                rank=rank,
                ulysses_size=4,
                dp_size=2,
                ulysses_rank=rank % 4,
                dp_rank=rank // 4,
                ulysses_group=f"sp{rank // 4}",
                dp_group=f"dp{rank % 4}",
            )
            context = trainer.validate_epmc_parallel_state(
                contract, state, dist_module=FakeDist(rank)
            )
            self.assertEqual(context.support_index, rank // 4)
            self.assertEqual(context.ulysses_rank, rank % 4)

    def test_runtime_resolves_transformer_from_the_branch_module(self):
        source = inspect.getsource(trainer)
        self.assertIn(
            "import counterfactual_proposal_motion_branch as cpmr_branch", source
        )
        self.assertIn(
            "transformer = cpmr_branch.resolve_wan_transformer(renderer)", source
        )
        self.assertNotIn("cpmr.resolve_wan_transformer(renderer)", source)

    def test_representability_gate_preserves_support_one_reference_probes(self):
        support_iids = ("841b5e0080a1441d", "7262dd490cbf42c5")
        probes = [
            {"iid": support_iids[0], "family": family, "passed": True}
            for family in ("phase_only", "block_only")
        ]
        good = type("Decision", (), {"go": True})()
        self.assertEqual(
            trainer.representability_decision(
                probes, good, support_iids=support_iids
            ),
            "GO",
        )
        bad = type("Decision", (), {"go": False})()
        self.assertEqual(
            trainer.representability_decision(
                probes, bad, support_iids=support_iids
            ),
            "NO_GO",
        )
        self.assertEqual(
            trainer.representability_decision(
                probes[:1], good, support_iids=support_iids
            ),
            "NO_GO",
        )
        failed = [dict(item) for item in probes]
        failed[-1]["passed"] = False
        self.assertEqual(
            trainer.representability_decision(
                failed, good, support_iids=support_iids
            ),
            "NO_GO",
        )
        support_two_probe = [dict(item) for item in probes]
        support_two_probe[-1]["iid"] = support_iids[1]
        self.assertEqual(
            trainer.representability_decision(
                support_two_probe, good, support_iids=support_iids
            ),
            "NO_GO",
        )
        self.assertEqual(
            trainer.representability_decision(
                probes
                + [
                    {
                        "iid": support_iids[1],
                        "family": "phase_only",
                        "passed": True,
                    }
                ],
                good,
                support_iids=support_iids,
            ),
            "NO_GO",
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorAndArtifactContractTests(unittest.TestCase):
    def _code(self, value: float = 0.2):
        phase = torch.full((1, 21), value, dtype=torch.float32)
        phase[:, 0].zero_()
        block = torch.full((1, 16, 12), value, dtype=torch.float32)
        return epmc.MotionCode(phase, block)

    def test_support_digest_is_the_real_tied_36d_view(self):
        code = self._code()
        tied = trainer._tied_code_36d(code)
        self.assertEqual(tuple(tied.shape), (1, 36))
        self.assertTrue(torch.equal(tied[:, :20], code.phase_gates[:, 1:]))
        self.assertTrue(torch.equal(tied[:, 20:], code.block_head_gates[:, :, 0]))
        with mock.patch(
            "counterfactual_proposal_motion_rebinding.tensor_sha256",
            return_value="a" * 64,
        ) as digest:
            self.assertEqual(trainer._code_hash(code), "a" * 64)
        self.assertEqual(tuple(digest.call_args.args[0].shape), (1, 36))

    def test_tied_code_exchange_round_trip_and_world_consensus(self):
        local = self._code(0.2)
        remote_tied = torch.full((1, 36), 0.4, dtype=torch.float32)
        support_iids = ("841b5e0080a1441d", "7262dd490cbf42c5")

        def world_size(group=None):
            return 2 if group == "dp" else 8

        def group_rank(group=None):
            return 0

        def gather_tensors(output, value, *, group):
            self.assertEqual(group, "dp")
            output[0].copy_(value)
            output[1].copy_(remote_tied)

        def gather_objects(output, value, *, group):
            self.assertEqual(group, "world")
            output[:] = [value] * 8

        with mock.patch("torch.distributed.get_world_size", side_effect=world_size), mock.patch(
            "torch.distributed.get_rank", side_effect=group_rank
        ), mock.patch(
            "torch.distributed.all_gather", side_effect=gather_tensors
        ), mock.patch(
            "torch.distributed.all_gather_object", side_effect=gather_objects
        ):
            codes, digest = trainer._exchange_k2_codes(
                local,
                support_index=0,
                support_iids=support_iids,
                dp_group="dp",
                world_group="world",
            )
        self.assertEqual(len(digest), 64)
        self.assertTrue(torch.equal(trainer._tied_code_36d(codes[0]), trainer._tied_code_36d(local)))
        self.assertTrue(torch.equal(trainer._tied_code_36d(codes[1]), remote_tied))

    def test_object_exchange_orders_supports_then_checks_world_digest(self):
        support_iids = ("841b5e0080a1441d", "7262dd490cbf42c5")
        local_value = {"loss": 1.0}
        remote_value = {"loss": 2.0}

        def world_size(group=None):
            return 2 if group == "dp" else 8

        def group_rank(group=None):
            return 0

        def gather_objects(output, value, *, group):
            if group == "dp":
                output[:] = [
                    {
                        "support_index": 1,
                        "iid": support_iids[1],
                        "value": remote_value,
                    },
                    {
                        "support_index": 0,
                        "iid": support_iids[0],
                        "value": local_value,
                    },
                ]
            else:
                self.assertEqual(group, "world")
                output[:] = [value] * 8

        with mock.patch("torch.distributed.get_world_size", side_effect=world_size), mock.patch(
            "torch.distributed.get_rank", side_effect=group_rank
        ), mock.patch(
            "torch.distributed.all_gather_object", side_effect=gather_objects
        ):
            ordered, digest = trainer._exchange_k2_objects(
                local_value,
                support_index=0,
                local_iid=support_iids[0],
                support_iids=support_iids,
                dp_group="dp",
                world_group="world",
                context="test values",
            )
        self.assertEqual(ordered, [local_value, remote_value])
        self.assertEqual(len(digest), 64)

    def test_object_exchange_fails_closed_on_duplicate_missing_or_wrong_iid(self):
        support_iids = ("841b5e0080a1441d", "7262dd490cbf42c5")

        invalid_gathers = {
            "duplicate-implies-missing": [
                {"support_index": 0, "iid": support_iids[0], "value": "local"},
                {"support_index": 0, "iid": support_iids[0], "value": "duplicate"},
            ],
            "missing-via-invalid-index": [
                {"support_index": 0, "iid": support_iids[0], "value": "local"},
                {"support_index": 2, "iid": "unexpected", "value": "invalid"},
            ],
            "wrong-iid": [
                {"support_index": 0, "iid": support_iids[0], "value": "local"},
                {"support_index": 1, "iid": support_iids[0], "value": "wrong"},
            ],
        }

        def world_size(group=None):
            return 2 if group == "dp" else 8

        def group_rank(group=None):
            return 0

        for label, gathered in invalid_gathers.items():
            with self.subTest(label=label):
                def gather_objects(output, _value, *, group):
                    self.assertEqual(group, "dp")
                    output[:] = gathered

                with mock.patch(
                    "torch.distributed.get_world_size", side_effect=world_size
                ), mock.patch(
                    "torch.distributed.get_rank", side_effect=group_rank
                ), mock.patch(
                    "torch.distributed.all_gather_object", side_effect=gather_objects
                ):
                    with self.assertRaises(trainer.FewShotCodeTrainingError):
                        trainer._exchange_k2_objects(
                            "local",
                            support_index=0,
                            local_iid=support_iids[0],
                            support_iids=support_iids,
                            dp_group="dp",
                            world_group="world",
                            context="invalid exchange",
                        )

    def test_renderer_sample_t5_offload_is_restored_frozen_and_eval(self):
        class SimulatedRenderer:
            def __init__(self):
                self.t5_text_encoder = torch.nn.Linear(3, 4)

            def sample(self):
                self.t5_text_encoder.to("cpu")
                self.t5_text_encoder.train()
                self.t5_text_encoder.requires_grad_(True)

        renderer = SimulatedRenderer()
        renderer.sample()
        receipt = trainer._restore_frozen_text_encoder(renderer, torch.device("cpu"))
        self.assertFalse(renderer.t5_text_encoder.training)
        self.assertTrue(
            all(not item.requires_grad for item in renderer.t5_text_encoder.parameters())
        )
        self.assertTrue(
            all(item.device.type == "cpu" for item in renderer.t5_text_encoder.parameters())
        )
        self.assertTrue(receipt["restored_after_sample"])
        self.assertEqual(receipt["device_type"], "cpu")
        self.assertNotIn("device", receipt)

    def test_text_encoder_receipt_excludes_rank_local_cuda_index(self):
        source = inspect.getsource(trainer._restore_frozen_text_encoder)
        self.assertIn('"device_type": expected.type', source)
        self.assertNotIn('"device": str(expected)', source)

    def test_held_control_shared_step_keeps_authenticated_grad_mode(self):
        source = inspect.getsource(trainer._evaluate_controls)
        self.assertIn("require_code_grad=False", source)
        self.assertNotIn("torch.no_grad", source)

    def test_packed_clean_conversion_preserves_true_30x31_yx_order(self):
        video = torch.arange(
            16 * 21 * 60 * 62, dtype=torch.float32
        ).reshape(1, 16, 21, 60, 62)
        patches = video.reshape(1, 16, 21, 30, 2, 31, 2)
        packed = (
            patches.permute(0, 2, 3, 5, 4, 6, 1)
            .unsqueeze(4)
            .reshape(1, 19530, 64)
        )
        restored = trainer._packed_clean_video(packed)
        self.assertTrue(torch.equal(restored, video))

    def test_motion_loss_executes_real_teacher_result_api(self):
        phase = torch.full((1, 21), 0.2, dtype=torch.float32)
        phase[:, 0].zero_()
        phase.requires_grad_(True)
        block = torch.full((1, 16, 12), 0.2, dtype=torch.float32)
        block.requires_grad_(True)
        code = epmc.MotionCode(phase, block)
        packed = torch.zeros(1, 19530, 64, dtype=torch.float32)
        guided = torch.zeros(1, 21, 930, 64, dtype=torch.float32)
        velocity = torch.zeros(1, 19530, 64, dtype=torch.bfloat16)
        teacher = trainer.TeacherCell(
            iid="1111111111111111",
            instruction="Make the animal sit and turn its head.",
            action_batch={},
            noop_batch={},
            negative_batch={},
            auxiliary={
                "source_clean": packed,
                "target_clean": packed.clone(),
                "shared_noisy": packed.clone(),
                "sigma": torch.tensor(0.5, dtype=torch.float32),
            },
            source_latent_cpu=None,
            parquet_receipt={},
            noise_seed=1,
            sigma_stratum={},
        )
        cell = trainer.TrainingCell(
            teacher=teacher,
            carrier=None,
            activity=None,
            carrier_receipt={},
            proposal_receipt={},
        )
        base = trainer.BaseAPGFields(
            negative_velocity=velocity,
            noop_velocity=velocity,
            guided_noop_clean=guided,
        )

        def guided_clean(**kwargs):
            self.assertIs(kwargs["negative_velocity"], velocity)
            return guided, guided

        fake_prior = types.ModuleType("train_prior_tangent_lora")
        fake_prior._guided_clean = mock.Mock(side_effect=guided_clean)
        with mock.patch.dict(
            sys.modules, {"train_prior_tangent_lora": fake_prior}
        ), mock.patch.object(
            trainer,
            "_coded_velocity",
            return_value=(velocity, {"shared_step_calls": 1}),
        ) as coded:
            loss, statistics, branch, returned_velocity = trainer._motion_loss(
                renderer=object(),
                cell=cell,
                base=base,
                patch_handle=object(),
                code=code,
                require_code_grad=True,
            )
        fake_prior._guided_clean.assert_called_once()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(statistics["schema_version"], "bernini-fewshot-teacher-objective-v1")
        self.assertEqual(statistics["full_target_flow_matching_weight"], 0.0)
        self.assertAlmostEqual(statistics["total"], float(loss.detach().item()))
        self.assertEqual(branch, {"shared_step_calls": 1})
        self.assertIs(returned_velocity, velocity)
        coded.assert_called_once()
        loss.backward()
        self.assertIsNotNone(phase.grad)
        self.assertIsNotNone(block.grad)
        self.assertTrue(torch.isfinite(phase.grad).all().item())
        self.assertTrue(torch.isfinite(block.grad).all().item())

    def test_prototype_requires_cpu_fp32_positive_phase0_and_tied_heads(self):
        code = self._code()
        trainer.validate_prototype_tensors(
            code.phase_gates.contiguous(), code.block_head_gates.contiguous()
        )
        signed = code.phase_gates.clone()
        signed[:, 0] = -0.0
        with self.assertRaisesRegex(
            trainer.FewShotCodeTrainingError, "positive-zero"
        ):
            trainer.validate_prototype_tensors(signed, code.block_head_gates)
        untied = code.block_head_gates.clone()
        untied[:, 3, 7] = 0.3
        with self.assertRaisesRegex(trainer.FewShotCodeTrainingError, "tied"):
            trainer.validate_prototype_tensors(code.phase_gates, untied)

    def test_atomic_prototype_has_exactly_two_tensors(self):
        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prototype.safetensors"
            receipt = trainer._atomic_save_prototype(path, self._code())
            values = load_file(str(path), device="cpu")
        self.assertEqual(set(values), {"phase_gates", "block_head_gates"})
        self.assertEqual(tuple(values["phase_gates"].shape), (1, 21))
        self.assertEqual(tuple(values["block_head_gates"].shape), (1, 16, 12))
        self.assertEqual(receipt["tensor_keys"], ["block_head_gates", "phase_gates"])

    def test_runtime_gate_fast_path_skips_large_digests_with_same_gradients(self):
        heads_a = torch.randn(1, 4, 12, 128, dtype=torch.float32, requires_grad=True)
        heads_b = heads_a.detach().clone().requires_grad_(True)
        phase_ids = torch.tensor([-1, 0, 1, 2], dtype=torch.int64)
        base = self._code()
        phase_a = base.phase_gates.clone().requires_grad_(True)
        block_a = base.block_head_gates.clone().requires_grad_(True)
        code_a = epmc.MotionCode(phase_a, block_a)
        phase_b = base.phase_gates.clone().requires_grad_(True)
        block_b = base.block_head_gates.clone().requires_grad_(True)
        code_b = epmc.MotionCode(
            phase_b,
            block_b,
        )
        audited = epmc.gate_projected_motion_heads(
            heads_a, phase_ids, code_a, block_index=2, audit_digests=True
        )
        audited.flattened_output().sum().backward()
        with mock.patch.object(
            epmc,
            "_tensor_sha256",
            side_effect=AssertionError("runtime attempted a GPU-to-CPU digest"),
        ):
            fast = epmc.gate_projected_motion_heads(
                heads_b, phase_ids, code_b, block_index=2, audit_digests=False
            )
            fast.flattened_output().sum().backward()
        self.assertTrue(
            torch.equal(audited.flattened_output(), fast.flattened_output())
        )
        self.assertTrue(torch.equal(heads_a.grad, heads_b.grad))
        self.assertTrue(torch.equal(phase_a.grad, phase_b.grad))
        self.assertTrue(torch.equal(block_a.grad, block_b.grad))
        self.assertIsNone(fast.input_heads_sha256)
        self.assertIsNone(fast.code_sha256)
        with self.assertRaisesRegex(epmc.PrivilegedMotionCodeContractError, "no audit"):
            fast.audit_receipt()
        branch_source = (METHOD_ROOT / "fewshot_motion_branch.py").read_text()
        self.assertIn("audit_digests=False", branch_source)


if __name__ == "__main__":
    unittest.main()
