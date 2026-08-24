from __future__ import annotations

from pathlib import Path
import ast
import sys
import tempfile
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_saic_typed_source_coordinate_v1 as trainer

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _Attention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)
            self.to_v = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.attn1 = _Attention(hidden)
            self.attn2 = _Attention(hidden)


    class _Transformer(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(
                16, hidden, kernel_size=(1, 2, 2)
            )
            self.blocks = nn.ModuleList([_Block(hidden) for _ in range(30)])
            self.gradient_checkpointing = False

        @property
        def is_gradient_checkpointing(self) -> bool:
            return bool(self.gradient_checkpointing)

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


    class _NativeVI:
        def __init__(self) -> None:
            self.name = "VI"
            self.total_tokens = 13
            self.condition_tokens = 5
            self.concat_order = (
                "video",
                "ref0",
                "ref1",
                "ref2",
                "ref3",
                "target",
            )
            self.source_ids = (1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
            self.latents = torch.zeros(1, 13, 8)
            self.target_mask = torch.tensor(
                [False] * 5 + [True] * 8, dtype=torch.bool
            )


class TypedPilotContractTests(unittest.TestCase):
    def test_preflight_fails_before_io_without_explicit_pilot_ack(self) -> None:
        parser = trainer.build_parser()
        args = parser.parse_args(
            [
                "--bernini-root",
                "/missing/bernini",
                "--veomni-root",
                "/missing/veomni",
                "--checkpoint",
                "/missing/checkpoint",
                "--checkpoint-content-manifest",
                "/missing/checkpoint.json",
                "--manifest",
                "/missing/manifest.json",
                "--expected-manifest-sha256",
                "0" * 64,
                "--cagd-validator-evidence",
                "/missing/evidence.json",
                "--expected-cagd-validator-evidence-sha256",
                "1" * 64,
                "--output",
                "/missing/output",
                "--max-schedule-steps",
                "40",
                "--method-source-revision",
                "2" * 40,
                "--method-source-archive-sha256",
                "3" * 64,
            ]
        )
        with self.assertRaisesRegex(
            trainer.SAICTypedSourceCoordinateTrainingError,
            "acknowledgement",
        ):
            trainer.preflight(args)

    def test_exact40_has_38_updates_and_two_exact_base_anchors(self) -> None:
        self.assertEqual(
            [trainer.exact40_schedule_index(index) for index in range(40)],
            list(range(40)),
        )
        self.assertEqual(trainer.expected_optimizer_updates(40), 38)
        self.assertEqual(trainer.expected_optimizer_updates(80), 76)
        self.assertEqual(trainer.typed_operator.LOW_SIGMA_INDICES, (38, 39))
        with self.assertRaises(trainer.SAICTypedSourceCoordinateTrainingError):
            trainer.expected_optimizer_updates(41)

    def test_dp_arrow_mapping_is_exact_e0_dog_and_e1_human(self) -> None:
        dog = trainer.action_arrow_for_dp_arm(0)
        human = trainer.action_arrow_for_dp_arm(1)
        self.assertEqual(
            (dog.initial_state_type, dog.terminal_state_type),
            ("dog_standing", "dog_sitting"),
        )
        self.assertEqual(
            (human.initial_state_type, human.terminal_state_type),
            ("human_kneeling", "human_standing"),
        )
        self.assertEqual(dog.values, (1.0,) + (0.0,) * 31)
        self.assertEqual(human.values, (0.0, 1.0) + (0.0,) * 30)
        self.assertTrue(trainer.noop_arrow_for_dp_arm(0).is_noop)
        self.assertTrue(trainer.noop_arrow_for_dp_arm(1).is_noop)
        with self.assertRaises(trainer.SAICTypedSourceCoordinateTrainingError):
            trainer.action_arrow_for_dp_arm(2)

    def test_runtime_is_typed_operator_not_static_pair_adapter(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        self.assertIn("install_saic_typed_action_operator", calls)
        self.assertIn("bind_runtime_route", calls)
        self.assertIn("replay_native_student_vjp", calls)
        self.assertIn("save_checkpoint", calls)
        self.assertNotIn("install_pair_v5_action_adapter", calls)
        self.assertIn("leaf_vjp_mode=True", source)
        self.assertIn("parallel.sp_group", source)
        self.assertIn("native_branch.target_mask", source)
        self.assertIn("actual_sigma=coordinate.sigma", source)
        self.assertIn("_disable_gradient_checkpointing", source)
        self.assertIn("trainable_named_parameters_for_sigma", source)
        self.assertIn("inactive sigma parameter partition changed", source)
        self.assertIn('"optimizer_instances": 2', source)
        self.assertIn('"cross_sigma_optimizer_momentum": False', source)
        self.assertNotIn("optimizer = torch.optim.AdamW", source)
        self.assertEqual(source.count("dp_arm = contract.arm_index"), 1)
        self.assertIn('"event_gate_present": False', source)
        self.assertIn('"inverse_cycle_present": False', source)
        self.assertIn('"complete_saic_method": False', source)
        self.assertIn('"semantic_action_editing_success_claimed": False', source)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class TypedPilotRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transformer = _Transformer().requires_grad_(False)
        self.handle = trainer.typed_operator.install_saic_typed_action_operator(
            self.transformer
        )

    def tearDown(self) -> None:
        if not self.handle.restored:
            self.handle.restore()

    def test_route_binds_branch_mask_sp_group_and_physical_sigma(self) -> None:
        branch = _NativeVI()
        sigma = torch.tensor(
            [trainer.typed_operator.sigma_strata.PINNED_POSITIVE_SIGMAS[33]],
            dtype=torch.float32,
        )
        route = trainer.bind_native_action_route(
            handle=self.handle,
            native_branch=branch,
            actual_sigma=sigma,
            arrow=trainer.action_arrow_for_dp_arm(0),
            parallel=SimpleNamespace(sp_group=None),
        )
        receipt = route.receipt()
        self.assertTrue(receipt["route_factory_bound"])
        self.assertEqual(receipt["sigma_schedule_index"], 33)
        self.assertEqual(receipt["sequence_parallel_size"], 1)
        self.assertTrue(
            torch.equal(
                route.local_target_selector(device=torch.device("cpu")),
                branch.target_mask,
            )
        )

    def test_mutated_runtime_mask_fails_closed(self) -> None:
        branch = _NativeVI()
        branch.target_mask[0] = True
        sigma = torch.tensor(
            [trainer.typed_operator.sigma_strata.PINNED_POSITIVE_SIGMAS[0]],
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(
            trainer.SAICTypedSourceCoordinateTrainingError,
            "target mask",
        ):
            trainer.bind_native_action_route(
                handle=self.handle,
                native_branch=branch,
                actual_sigma=sigma,
                arrow=trainer.action_arrow_for_dp_arm(0),
                parallel=SimpleNamespace(sp_group=None),
            )

    def test_custom_checkpoint_is_closed_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operator.pt"
            saved = self.handle.save_checkpoint(path)
            self.assertEqual(saved["state_key_count"], 23 * 2 * 6)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(
                set(payload),
                {
                    "checkpoint_schema_version",
                    "operator_schema_version",
                    "state_tensor_sha256",
                    "state",
                },
            )
            loaded = self.handle.load_checkpoint(path)
            self.assertEqual(
                loaded["state_tensor_sha256"], saved["state_tensor_sha256"]
            )


class TypedPilotLauncherTests(unittest.TestCase):
    def test_launcher_is_world8_exact81_and_qos_is_submit_time(self) -> None:
        path = (
            METHOD_ROOT
            / "scripts"
            / "auh_train_saic_typed_source_coordinate_v1.sbatch"
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn("--nproc_per_node=8", text)
        self.assertIn("topology=WORLD8/DP2xSP4", text)
        self.assertIn("frames=exact81", text)
        self.assertIn("schedule_steps % 40", text)
        self.assertIn("low_base=38,39", text)
        self.assertIn("SAIC_TYPED_HIGH_SIGMA_SMOKE_ONLY", text)
        self.assertIn("train_saic_typed_source_coordinate_v1.py", text)
        self.assertIn("saic_typed_action_operator_v1.py", text)
        self.assertIn("train_pair_v6_scaid.py", text)
        self.assertIn("pair_v6_scaid_source_coordinate.py", text)
        self.assertIn("--ack-typed-source-coordinate-pilot-no-semantic-success-claim", text)
        self.assertIn("operator.pt", text)
        self.assertNotIn("#SBATCH --qos=", text)
        self.assertIn("sbatch --qos=gtqos", text)
        self.assertNotIn("--gradient-checkpointing", text)

    def test_launcher_archive_required_set_is_full_recursive_import_closure(self) -> None:
        path = (
            METHOD_ROOT
            / "scripts"
            / "auh_train_saic_typed_source_coordinate_v1.sbatch"
        )
        text = path.read_text(encoding="utf-8")
        start = text.index("required = {") + len("required = ")
        end = text.index("\nseen = set()", start)
        required = set(ast.literal_eval(text[start:end]))

        module_files = {item.stem: item for item in METHOD_ROOT.glob("*.py")}
        pending = [METHOD_ROOT / "train_saic_typed_source_coordinate_v1.py"]
        closure: set[str] = set()
        while pending:
            source_path = pending.pop()
            relative = source_path.relative_to(METHOD_ROOT.parent.parent).as_posix()
            if relative in closure:
                continue
            closure.add(relative)
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    imported.add(node.module.split(".")[0])
            pending.extend(
                module_files[name]
                for name in imported
                if name in module_files
            )
        self.assertEqual(required, closure)


if __name__ == "__main__":
    unittest.main()
