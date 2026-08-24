from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "packed_preservation_lora_v2.py"
SPEC = importlib.util.spec_from_file_location("packed_preservation_lora_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)

RELEASE_PATH = ROOT / "packed_preservation_release_v2.py"
RELEASE_SPEC = importlib.util.spec_from_file_location(
    "packed_preservation_release_v2", RELEASE_PATH
)
assert RELEASE_SPEC is not None and RELEASE_SPEC.loader is not None
release = importlib.util.module_from_spec(RELEASE_SPEC)
sys.modules[RELEASE_SPEC.name] = release
RELEASE_SPEC.loader.exec_module(release)


class FakeWeight:
    def __init__(self, out_features: int, in_features: int) -> None:
        self.shape = (out_features, in_features)


class FakeLinear:
    def __init__(self, out_features: int, in_features: int) -> None:
        self.weight = FakeWeight(out_features, in_features)
        self.out_features = out_features
        self.in_features = in_features


class WanAttnProcessor2_0:
    pass


class FakeAttention:
    def __init__(self) -> None:
        self.processor = WanAttnProcessor2_0()
        self.add_k_proj = None
        self.add_v_proj = None
        self.to_add_out = None


class FakeParameter:
    def __init__(self, shape: tuple[int, ...], *, requires_grad: bool = True) -> None:
        self.shape = shape
        self.requires_grad = requires_grad
        self.dtype = "torch.float32"

    def numel(self) -> int:
        value = 1
        for item in self.shape:
            value *= item
        return value


class FakeModel:
    def __init__(self) -> None:
        self.modules: list[tuple[str, object]] = [
            (
                "diff_dec.transformer.condition_embedder.text_embedder.linear_1",
                FakeLinear(1536, 4096),
            ),
            (
                "diff_dec.transformer.condition_embedder.text_embedder.linear_2",
                FakeLinear(1536, 1536),
            ),
        ]
        for block in range(30):
            for attention in (1, 2):
                self.modules.append(
                    (
                        f"diff_dec.transformer.blocks.{block}.attn{attention}",
                        FakeAttention(),
                    )
                )
                for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                    self.modules.append(
                        (
                            f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}",
                            FakeLinear(1536, 1536),
                        )
                    )

    def named_modules(self):
        return iter(self.modules)


class DirectTransformerFakeModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.modules = [
            (name[len("diff_dec.transformer.") :], module)
            for name, module in self.modules
        ]


class InstalledFakeModel(FakeModel):
    def __init__(self, scope: str) -> None:
        super().__init__()
        self.specs = core.select_projection_specs(self, scope)
        self.parameters: list[tuple[str, FakeParameter]] = []
        for spec in self.specs:
            prefix = f"base_model.model.{spec.name}"
            self.parameters.extend(
                (
                    (
                        f"{prefix}.lora_A.default.weight",
                        FakeParameter((256, spec.in_features)),
                    ),
                    (
                        f"{prefix}.lora_B.default.weight",
                        FakeParameter((spec.out_features, 256)),
                    ),
                )
            )
        patch = "base_model.model.diff_dec.transformer.patch_embedding"
        self.parameters.extend(
            (
                (f"{patch}.source_delta.weight", FakeParameter((1536, 16, 1, 2, 2))),
                (f"{patch}.source_delta.bias", FakeParameter((1536,))),
                (f"{patch}.target_delta.weight", FakeParameter((1536, 16, 1, 2, 2))),
                (f"{patch}.target_delta.bias", FakeParameter((1536,))),
                (f"{patch}.role_embedding", FakeParameter((2, 1536))),
            )
        )

    def named_parameters(self):
        return iter(self.parameters)


def dynamic_cpu_fake_model(test_case):
    try:
        import torch
    except ImportError:
        test_case.skipTest("PyTorch is available in the pinned AUH runtime")

    class Transformer(torch.nn.Module):
        def __init__(self, patch_embedding) -> None:
            super().__init__()
            self.patch_embedding = patch_embedding

    class AdapterBank(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.default = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.default.weight.fill_(value)

    class FakePeftResidual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = AdapterBank(1.0)
            self.lora_B = AdapterBank(0.0)

        def forward(self, value):
            unit = value.new_ones((1, 1))
            residual = self.lora_B.default(self.lora_A.default(unit))
            return value + residual.reshape(1, 1, 1, 1, 1)

    class FakePeftModel(torch.nn.Module):
        def __init__(self, transformer) -> None:
            super().__init__()
            self.transformer = transformer
            self.adapter = FakePeftResidual()
            self._disable_depth: ContextVar[int] = ContextVar(
                f"packed_preservation_fake_peft_disable_{id(self)}", default=0
            )

        @contextmanager
        def disable_adapter(self):
            depth = self._disable_depth.get()
            token = self._disable_depth.set(depth + 1)
            try:
                yield
            finally:
                self._disable_depth.reset(token)

        def forward(self, patches):
            value = self.transformer.patch_embedding(patches)
            if self._disable_depth.get() > 0:
                return value
            return self.adapter(value)

    native = torch.nn.Conv3d(
        core.PATCH_INPUT_CHANNELS,
        core.HIDDEN_SIZE,
        kernel_size=core.PATCH_KERNEL,
        stride=core.PATCH_KERNEL,
        bias=True,
    ).cpu()
    with torch.no_grad():
        weight_values = torch.arange(native.weight.numel(), dtype=torch.float32)
        native.weight.copy_(
            ((weight_values.remainder(23) - 11.0) / 1024.0).reshape_as(
                native.weight
            )
        )
        bias_values = torch.arange(native.bias.numel(), dtype=torch.float32)
        native.bias.copy_((bias_values.remainder(17) - 8.0) / 512.0)
        patch_values = torch.arange(
            2 * core.PATCH_INPUT_CHANNELS * 4, dtype=torch.float32
        )
        patches = (
            (patch_values.remainder(13) - 6.0) / 16.0
        ).reshape(2, core.PATCH_INPUT_CHANNELS, *core.PATCH_KERNEL)
        official = native(patches).detach().clone()

    transformer = Transformer(native)
    wrapped = core.install_typed_patch_embedding(transformer)
    model = FakePeftModel(transformer).cpu()
    return torch, native, wrapped, model, patches, official


def tensor_bytes(torch, value) -> bytes:
    octets = value.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    return bytes(octets.tolist())


def model_state_snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


class PackedPreservationCoreTests(unittest.TestCase):
    def test_official_frozen_native_only_is_byte_exact_after_u1(self) -> None:
        torch, native, wrapped, model, patches, official = dynamic_cpu_fake_model(self)
        self.assertIs(wrapped.native, native)

        def frozen_forward():
            with torch.no_grad(), model.disable_adapter():
                with core.official_frozen_native_only():
                    return model(patches).detach().clone()

        p0 = frozen_forward()
        p0_repeat = frozen_forward()
        self.assertTrue(torch.equal(official, p0))
        self.assertEqual(tensor_bytes(torch, official), tensor_bytes(torch, p0))
        self.assertTrue(torch.equal(p0, p0_repeat))
        self.assertEqual(tensor_bytes(torch, p0), tensor_bytes(torch, p0_repeat))

        typed_before = {
            name: value.detach().clone()
            for name, value in wrapped.named_parameters()
            if name.startswith(("source_delta.", "target_delta.", "role_embedding"))
        }
        lora_b_before = model.adapter.lora_B.default.weight.detach().clone()
        optimizer = torch.optim.SGD(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        optimizer.zero_grad(set_to_none=True)
        with core.packed_role_layout(1, 1):
            trainable_output = model(patches)
        trainable_output.float().sum().backward()
        optimizer.step()

        typed_after = dict(wrapped.named_parameters())
        self.assertTrue(
            all(
                not torch.equal(before, typed_after[name].detach())
                for name, before in typed_before.items()
            )
        )
        self.assertFalse(
            torch.equal(lora_b_before, model.adapter.lora_B.default.weight.detach())
        )
        with torch.no_grad(), core.packed_role_layout(1, 1):
            u1_trainable = model(patches).detach().clone()
        self.assertFalse(torch.equal(official, u1_trainable))

        state_before_frozen = model_state_snapshot(model)
        u1_frozen = frozen_forward()
        u1_frozen_repeat = frozen_forward()
        state_after_frozen = model_state_snapshot(model)
        self.assertTrue(torch.equal(official, u1_frozen))
        self.assertTrue(torch.equal(u1_frozen, u1_frozen_repeat))
        self.assertEqual(
            tensor_bytes(torch, official), tensor_bytes(torch, u1_frozen)
        )
        self.assertEqual(state_before_frozen.keys(), state_after_frozen.keys())
        self.assertTrue(
            all(
                torch.equal(state_before_frozen[name], state_after_frozen[name])
                for name in state_before_frozen
            )
        )

    def test_official_frozen_native_only_is_reentrant_and_exception_safe(self) -> None:
        torch, _, _, model, patches, official = dynamic_cpu_fake_model(self)
        with torch.no_grad(), model.disable_adapter():
            with core.official_frozen_native_only():
                outer_before = model(patches).detach().clone()
                with core.official_frozen_native_only():
                    nested = model(patches).detach().clone()
                outer_after = model(patches).detach().clone()
        self.assertTrue(torch.equal(official, outer_before))
        self.assertTrue(torch.equal(official, nested))
        self.assertTrue(torch.equal(official, outer_after))

        with self.assertRaisesRegex(RuntimeError, "native-only sentinel"):
            with core.official_frozen_native_only():
                raise RuntimeError("native-only sentinel")
        with self.assertRaises(core.PackedPreservationV2Error):
            model(patches)
        with torch.no_grad(), model.disable_adapter():
            with core.official_frozen_native_only():
                recovered = model(patches).detach().clone()
        self.assertTrue(torch.equal(official, recovered))

    def test_official_frozen_native_only_is_cross_thread_isolated(self) -> None:
        torch, _, wrapped, model, patches, official = dynamic_cpu_fake_model(self)
        with torch.no_grad():
            wrapped.role_embedding[0].fill_(0.5)
            wrapped.role_embedding[1].fill_(-0.75)

        entered_native_only = threading.Event()
        release_native_only = threading.Event()
        results = {}
        errors = []

        def native_worker() -> None:
            try:
                with torch.no_grad(), model.disable_adapter():
                    with core.official_frozen_native_only():
                        entered_native_only.set()
                        if not release_native_only.wait(timeout=5.0):
                            raise RuntimeError("native-only worker release timed out")
                        results["native"] = model(patches).detach().clone()
            except BaseException as error:  # propagate worker failures to unittest
                errors.append(error)

        worker = threading.Thread(target=native_worker)
        worker.start()
        try:
            self.assertTrue(entered_native_only.wait(timeout=5.0))
            with torch.no_grad(), model.disable_adapter():
                with core.packed_role_layout(1, 1):
                    results["ordinary"] = model(patches).detach().clone()
        finally:
            release_native_only.set()
            worker.join(timeout=5.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(official, results["native"]))
        self.assertFalse(torch.equal(official, results["ordinary"]))
        with self.assertRaises(core.PackedPreservationV2Error):
            model(patches)

    def test_exact_capacity_main_and_variant(self) -> None:
        model = FakeModel()
        main = core.select_projection_specs(model, "all-attention")
        variant = core.select_projection_specs(model, "self-attention")
        self.assertEqual(len(main), 240)
        self.assertEqual(len(variant), 120)
        self.assertEqual(sum(item.lora_parameters for item in main), 188_743_680)
        self.assertEqual(sum(item.lora_parameters for item in variant), 94_371_840)
        self.assertEqual(core.PATCH_ROLE_PARAMETER_COUNT, 202_752)
        self.assertEqual(
            core.EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNTS,
            {"all-attention": 188_946_432, "self-attention": 94_574_592},
        )

    def test_exact_official_cross_attention_names_shapes_and_route(self) -> None:
        model = FakeModel()
        specs = core.select_projection_specs(model, "all-attention")
        cross = [item for item in specs if item.attention == 2]
        self.assertEqual(len(cross), 120)
        self.assertEqual(
            {(item.block, item.projection) for item in cross},
            {
                (block, projection)
                for block in range(30)
                for projection in ("to_q", "to_k", "to_v", "to_out.0")
            },
        )
        self.assertEqual(
            {(item.in_features, item.out_features) for item in cross},
            {(1536, 1536)},
        )
        receipt = core.architecture_receipt("all-attention", specs)
        self.assertEqual(receipt["attention_affine_weight_shape"], [1536, 1536])
        self.assertEqual(
            receipt["text_preprojection_weight_shapes"],
            {"linear_1": [1536, 4096], "linear_2": [1536, 1536]},
        )
        self.assertFalse(receipt["cross_attention_added_kv_projection"])

    def test_rejects_dead_or_4096_wide_cross_attention_affines(self) -> None:
        wrong_shape = FakeModel()
        for index, (name, _) in enumerate(wrong_shape.modules):
            if name == "diff_dec.transformer.blocks.7.attn2.to_k":
                wrong_shape.modules[index] = (name, FakeLinear(1536, 4096))
                break
        with self.assertRaises(core.PackedPreservationV2Error):
            core.select_projection_specs(wrong_shape, "all-attention")

        live_added_kv = FakeModel()
        for name, module in live_added_kv.modules:
            if name == "diff_dec.transformer.blocks.11.attn2":
                module.add_k_proj = FakeLinear(1536, 4096)
                break
        with self.assertRaises(core.PackedPreservationV2Error):
            core.select_projection_specs(live_added_kv, "all-attention")

    def test_projection_regex_accepts_direct_transformer(self) -> None:
        specs = core.select_projection_specs(
            DirectTransformerFakeModel(), "all-attention"
        )
        self.assertEqual(len(specs), 240)
        self.assertTrue(specs[0].name.startswith("blocks."))

    def test_exact_installed_a_b_names_shapes_and_total(self) -> None:
        for scope, expected in core.EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNTS.items():
            model = InstalledFakeModel(scope)
            receipt = core.validate_lora_installation(model, model.specs)
            self.assertTrue(receipt["exact_one_a_and_b_per_affine"])
            self.assertEqual(receipt["selected_affines"], len(model.specs))
            self.assertEqual(core.verify_trainable_parameter_count(model, scope), expected)

    def test_exact80_mixture_is_40_20_20_20(self) -> None:
        self.assertEqual(
            core.objective_histogram(640),
            {"noop": 256, "cube": 128, "speed": 128, "tube": 128},
        )
        self.assertEqual(
            tuple(core.objective_for_logical_record(index) for index in range(10)),
            ("noop", "noop", "noop", "noop", "cube", "cube", "speed", "speed", "tube", "tube"),
        )

    def test_scopes_and_checkpoint_cadence(self) -> None:
        self.assertEqual(core.optimizer_steps("optimizer-canary-2"), 2)
        self.assertEqual(core.checkpoint_steps("optimizer-canary-2"), (0, 1, 2))
        self.assertEqual(core.optimizer_steps("exact80"), 80)
        self.assertEqual(core.checkpoint_steps("exact80"), (0, 20, 40, 60, 80))

    def test_architecture_receipt_forbids_sparse_or_target_gating(self) -> None:
        specs = core.select_projection_specs(FakeModel(), "all-attention")
        receipt = core.architecture_receipt("all-attention", specs)
        self.assertTrue(receipt["all_local_packed_tokens_receive_lora"])
        self.assertFalse(receipt["target_row_gating"])
        self.assertFalse(receipt["targetless_sp_early_return"])
        self.assertFalse(receipt["sparse_block_routing"])

    def test_runner_and_launcher_are_fail_closed(self) -> None:
        runner = (ROOT / "train_packed_preservation_lora_v2.py").read_text()
        core_text = MODULE_PATH.read_text()
        launcher = (
            ROOT / "scripts/auh_train_packed_preservation_lora_v2_job136140.sh"
        ).read_text()
        for required in (
            "optimizer-canary-2",
            "exact80",
            "synchronize_gradients_bucketed",
            "P0 != P1 != P2",
            "max_memory_allocated",
            "load_trainable_state_strict",
        ):
            self.assertIn(required, runner + core_text)
        self.assertIn("136140:auh7-1b-gpu-215:all-attention", launcher)
        self.assertIn("136141:auh7-1b-gpu-299:self-attention", launcher)
        self.assertIn("launch-job${holder_job}-on-${holder_node}", launcher)
        self.assertNotIn("scancel", launcher)
        self.assertIn('[[ -z "${PRESV2_METHOD_ROOT+x}" ]]', launcher)
        self.assertNotIn('method_root="${PRESV2_METHOD_ROOT', launcher)
        self.assertIn('release_extract_root="${run_root}/runtime-source"', launcher)
        self.assertIn('method_root="$("${python_bin}" -I -S -', launcher)
        self.assertLess(
            launcher.index('method_root="$("${python_bin}" -I -S -'),
            launcher.index('readonly rank_exec="${method_root}/scripts/'),
        )
        self.assertLess(
            runner.index("release_contract.validate_executed_release("),
            runner.index("import packed_preservation_lora_v2 as core"),
        )
        rank_wrapper = (
            ROOT / "scripts/auh_packed_preservation_rank_exec_v2.sh"
        ).read_text()
        self.assertNotIn("export HOME=", rank_wrapper)
        self.assertIn("export HF_HOME=", rank_wrapper)

        # The release intentionally carries the committed WORLD8 runtime API,
        # not unrelated dirty-worktree topology extensions.
        self.assertNotIn("runtime.parallel_topology(", runner)
        self.assertNotIn("distributed_contract(topology=", runner)
        self.assertNotIn("parallel.contract.topology", runner)
        self.assertIn("runtime.distributed_contract()", runner)
        self.assertIn("flat.div_(float(SP_SIZE))", runner)
        self.assertIn("flat.div_(float(DP_SIZE))", runner)

    def test_release_is_deterministic_exact_member_and_mode_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            first = release.build_release(
                method_root=ROOT,
                archive=temporary_root / "first.tar",
                manifest=temporary_root / "first.json",
            )
            second = release.build_release(
                method_root=ROOT,
                archive=temporary_root / "second.tar",
                manifest=temporary_root / "second.json",
            )
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            extracted = temporary_root / "extracted"
            extracted.mkdir()
            with tarfile.open(first["archive"], "r:") as bundle:
                bundle.extractall(extracted)
            executed_root = extracted / release.MEMBER_ROOT
            validated = release.validate_executed_release(
                method_root=executed_root,
                archive=first["archive"],
                manifest=first["manifest"],
                expected_archive_sha256=first["archive_sha256"],
                expected_manifest_sha256=first["manifest_sha256"],
                method_revision=first["method_revision"],
            )
            self.assertTrue(validated["archive_members_verified"])
            self.assertTrue(validated["executed_root_exact_closure_verified"])
            self.assertTrue(validated["executed_modes_verified"])
            wrapper = executed_root / "scripts/auh_packed_preservation_rank_exec_v2.sh"
            os.chmod(wrapper, 0o444)
            with self.assertRaises(release.PackedPreservationReleaseError):
                release.validate_executed_release(
                    method_root=executed_root,
                    archive=first["archive"],
                    manifest=first["manifest"],
                    expected_archive_sha256=first["archive_sha256"],
                    expected_manifest_sha256=first["manifest_sha256"],
                    method_revision=first["method_revision"],
                )
            os.chmod(wrapper, 0o555)
            root_b_member = executed_root / "packed_preservation_lora_v2.py"
            os.chmod(root_b_member, 0o644)
            root_b_member.write_bytes(root_b_member.read_bytes() + b"\n# root B tamper\n")
            os.chmod(root_b_member, 0o444)
            with self.assertRaises(release.PackedPreservationReleaseError):
                release.validate_executed_release(
                    method_root=executed_root,
                    archive=first["archive"],
                    manifest=first["manifest"],
                    expected_archive_sha256=first["archive_sha256"],
                    expected_manifest_sha256=first["manifest_sha256"],
                    method_revision=first["method_revision"],
                )

    def test_launcher_stdlib_bootstrap_materializes_authenticated_archive(self) -> None:
        launcher_path = (
            ROOT / "scripts/auh_train_packed_preservation_lora_v2_job136140.sh"
        )
        launcher = launcher_path.read_text()
        marker = '"${method_revision}" "${release_extract_root}" <<\'PY\'\n'
        bootstrap = launcher.split(marker, 1)[1].split("\nPY\n)\"", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            built = release.build_release(
                method_root=ROOT,
                archive=temporary_root / "release.tar",
                manifest=temporary_root / "release.json",
            )
            extract_root = temporary_root / "runtime-source"
            command = (
                sys.executable,
                "-I",
                "-S",
                "-",
                built["archive"],
                built["archive_sha256"],
                built["manifest"],
                built["manifest_sha256"],
                built["method_revision"],
                str(extract_root),
            )
            completed = subprocess.run(
                command,
                input=bootstrap,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            materialized = Path(completed.stdout.strip())
            self.assertEqual(materialized, extract_root / release.MEMBER_ROOT)
            validated = release.validate_executed_release(
                method_root=materialized,
                archive=built["archive"],
                manifest=built["manifest"],
                expected_archive_sha256=built["archive_sha256"],
                expected_manifest_sha256=built["manifest_sha256"],
                method_revision=built["method_revision"],
            )
            self.assertTrue(validated["executed_root_exact_closure_verified"])
            repeated = subprocess.run(
                command,
                input=bootstrap,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(repeated.returncode, 0)
            for directory in (
                materialized / "scripts",
                materialized,
                materialized.parent,
                extract_root,
            ):
                os.chmod(directory, 0o755)

    def test_checkpoint_adapter_and_optimizer_real_roundtrip_when_torch_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is available in the pinned AUH runtime")

        runner_spec = importlib.util.spec_from_file_location(
            "train_packed_preservation_lora_v2",
            ROOT / "train_packed_preservation_lora_v2.py",
        )
        assert runner_spec is not None and runner_spec.loader is not None
        runner = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner
        runner_spec.loader.exec_module(runner)
        runner.core = core

        class Slot(torch.nn.Module):
            def __init__(self, incoming: int, outgoing: int) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(outgoing, incoming))

        class LoRA(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.default = Slot(3, 2)

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_A = LoRA()
                self.lora_B = LoRA()

        class Patch(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.source_delta = Slot(3, 2)
                self.target_delta = Slot(3, 2)
                self.role_embedding = torch.nn.Parameter(torch.randn(2, 3))

        class Dummy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = Block()
                self.patch = Patch()

        model = Dummy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        named = core.trainable_named_parameters(model)
        parameter_sha = runner.tensor_digest(named)
        with tempfile.TemporaryDirectory() as temporary:
            record = runner.save_checkpoint(
                root=Path(temporary),
                step=0,
                model=model,
                optimizer=optimizer,
                metadata={"parameter_sha256": parameter_sha},
            )
            metadata = __import__("json").loads(
                (Path(record["path"]) / "metadata.json").read_text()
            )
            self.assertTrue(metadata["adapter_reload_verified"])
            self.assertTrue(metadata["optimizer_reload_verified"])
            self.assertTrue(metadata["same_architecture_strict_reload_verified"])
            self.assertFalse(metadata["fresh_official_rv2v_inference_process_verified"])
            self.assertEqual(metadata["roundtrip_parameter_sha256"], parameter_sha)


if __name__ == "__main__":
    unittest.main()
