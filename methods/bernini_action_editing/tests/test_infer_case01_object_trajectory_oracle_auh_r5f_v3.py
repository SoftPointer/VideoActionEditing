from __future__ import annotations

from contextlib import contextmanager
import ast
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
COMPOSITE_SOURCE = (
    METHOD_ROOT / "infer_case01_object_trajectory_oracle_auh_r5f_v3.py"
)
R5F_SOURCE = (
    METHOD_ROOT / "full644_exploratory_matched_infer_adapter_auh_r5f.py"
)
SEALED_METHOD_FIXTURE = Path(
    "/tmp/case01_object_trajectory_v1_sealed_methods_fixture"
)
SEALED_INFER_SHA256 = (
    "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_bytes(), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name} in {path}")
    return ast.dump(matches[0], include_attributes=False)


class _FakeAssets:
    def __init__(self) -> None:
        self.cli = types.SimpleNamespace(arm="route_off")
        self.replay_calls = 0
        self.close_calls = 0

    def producer_hashes(self) -> dict[str, str]:
        return {
            "wrapper_source_sha256": (
                "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9"
            ),
            "legacy_infer_lora_source_sha256": SEALED_INFER_SHA256,
            "projection_source_sha256": "a" * 64,
            "scaffold_source_sha256": "b" * 64,
        }

    def replay_all(self) -> None:
        self.replay_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class CompositeR5FV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SEALED_METHOD_FIXTURE.is_dir():
            raise AssertionError(
                "sealed method fixture is required for hermetic composite tests"
            )
        if _sha(SEALED_METHOD_FIXTURE / "infer_lora.py") != SEALED_INFER_SHA256:
            raise AssertionError("sealed infer_lora fixture differs")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.release_root = Path(cls.temporary.name).resolve()
        shutil.copytree(
            SEALED_METHOD_FIXTURE,
            cls.release_root,
            dirs_exist_ok=True,
        )
        cls.composite_path = cls.release_root / COMPOSITE_SOURCE.name
        shutil.copy2(COMPOSITE_SOURCE, cls.composite_path)
        cls.composite_path.chmod(0o444)

        cls.module_snapshot = dict(sys.modules)
        cls.path_snapshot = list(sys.path)
        local_names = {
            "action_preservation_decoded_eval_model_authority_v2",
            "infer_lora",
            "train_lora",
            "self_generated_action_preservation_v2",
            "tools",
            "tools.build_renderer_dataset",
            "tools.materialize_vae",
            "_case01_object_trajectory_r5f_v3_test",
            "_full644_exploratory_matched_infer_adapter_r5c_base",
        }
        for name in local_names:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(
            "_case01_object_trajectory_r5f_v3_test",
            cls.composite_path,
        )
        if spec is None or spec.loader is None:
            raise AssertionError("cannot create composite test spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.composite = module

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path[:] = cls.path_snapshot
        for name in tuple(sys.modules):
            if name not in cls.module_snapshot:
                sys.modules.pop(name, None)
        for name, module in cls.module_snapshot.items():
            sys.modules[name] = module
        cls.temporary.cleanup()

    def test_source_compiles_and_r5f_authority_core_is_unchanged(self) -> None:
        raw = COMPOSITE_SOURCE.read_bytes()
        for optimize in (0, 1, 2):
            compile(raw, str(COMPOSITE_SOURCE), "exec", optimize=optimize)
        for name in (
            "_verify_capture_through_namespace_fd",
            "read_fd_with_pread_r5f",
            "validate_inherited_fd_binding_r5f",
            "preserve_primary_context",
            "patched_rank_validation",
            "_activate_production_rank",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    _function_ast(COMPOSITE_SOURCE, name),
                    _function_ast(R5F_SOURCE, name),
                )

    def test_held_inner_alias_and_composite_are_distinct_and_replayed(self) -> None:
        module = self.composite
        with module.held_object_sources() as held:
            inner, alias, outer, inner_authority = held
            self.assertIs(module.base.infer_lora, sys.modules["infer_lora"])
            self.assertNotIn(
                "_bernini_full644_r5_infer_lora_acc46",
                sys.modules,
            )
            self.assertEqual(alias.sha256, SEALED_INFER_SHA256)
            self.assertEqual(
                inner_authority.sha256,
                module.OBJECT_WRAPPER_INNER_SHA256,
            )
            self.assertEqual(outer.sha256, _sha(COMPOSITE_SOURCE))
            self.assertEqual(inner.WRAPPER_RECEIPT_SCHEMA, module.COMPOSITE_RECEIPT_SCHEMA)
            self.assertEqual(
                inner.RUNTIME_TRACE_SCHEMA,
                module.COMPOSITE_RUNTIME_TRACE_SCHEMA,
            )
            for authority in (alias, outer, inner_authority):
                authority.replay()
        self.assertNotIn(module.OBJECT_WRAPPER_INNER_MODULE, sys.modules)

    def test_off_delegates_once_to_base_alias_and_strips_object_cli(self) -> None:
        module = self.composite
        calls: list[tuple[list[str], dict[str, object]]] = []

        def run(argv: object, **kwargs: object) -> dict[str, object]:
            calls.append((list(argv), dict(kwargs)))
            self.assertIs(kwargs["inference_main"], module.base.infer_lora.main)
            self.assertNotIn(
                "_bernini_full644_r5_infer_lora_acc46",
                sys.modules,
            )
            return {"return_code": 0}

        argv = [
            "--object-oracle-arm",
            "off",
            "--object-oracle-scaffold",
            "/must/not/be/opened.json",
            "--output",
            "/logical/task/output.mp4",
        ]
        with mock.patch.object(module, "_require_base_runtime_ready"), mock.patch.object(
            module.base, "run", side_effect=run
        ):
            result = module._run_object_trajectory(argv)
        self.assertEqual(result, {"return_code": 0})
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][0],
            ["--output", "/logical/task/output.mp4"],
        )

    def test_route_patch_targets_same_base_module_and_binds_outer_inner_hashes(self) -> None:
        module = self.composite
        actual_loader = module._load_object_wrapper_inner
        assets = _FakeAssets()
        holder: dict[str, object] = {}

        def loader() -> tuple[object, object]:
            inner, authority = actual_loader()

            def prepare(*args: object, **kwargs: object) -> _FakeAssets:
                holder["legacy_authority"] = kwargs["legacy_source"]
                return assets

            inner._prepare_oracle_assets = prepare
            return inner, authority

        original_receipt = module.base.infer_lora.build_inference_receipt

        def run(argv: object, **kwargs: object) -> dict[str, object]:
            self.assertIs(kwargs["inference_main"], module.base.infer_lora.main)
            self.assertIsNot(
                module.base.infer_lora.build_inference_receipt,
                original_receipt,
            )
            hashes = assets.producer_hashes()
            self.assertEqual(
                hashes,
                {
                    "wrapper_source_sha256": _sha(COMPOSITE_SOURCE),
                    "object_wrapper_inner_source_sha256": (
                        module.OBJECT_WRAPPER_INNER_SHA256
                    ),
                    "legacy_infer_lora_source_sha256": SEALED_INFER_SHA256,
                    "projection_source_sha256": "a" * 64,
                    "scaffold_source_sha256": "b" * 64,
                },
            )
            self.assertIs(
                holder["legacy_authority"],
                assets.__dict__.get("legacy_source", holder["legacy_authority"]),
            )
            return {"return_code": 0}

        argv = [
            "--object-oracle-arm=route_off",
            "--object-oracle-scaffold=/authority/scaffold.json",
            "--object-oracle-scaffold-sha256=" + "1" * 64,
            "--object-oracle-scaffold-digest=" + "2" * 64,
            "--object-oracle-bone-removed-video=/authority/aux.mp4",
            "--object-oracle-bone-removed-video-sha256=" + "3" * 64,
            "--output",
            "/logical/task/output.mp4",
        ]
        with mock.patch.object(module, "_require_base_runtime_ready"), mock.patch.object(
            module, "_load_object_wrapper_inner", side_effect=loader
        ), mock.patch.object(module.base, "run", side_effect=run):
            result = module._run_object_trajectory(argv)
        self.assertEqual(result, {"return_code": 0})
        self.assertIs(module.base.infer_lora.build_inference_receipt, original_receipt)
        self.assertGreaterEqual(assets.replay_calls, 2)
        self.assertEqual(assets.close_calls, 1)

    def test_primary_failure_survives_secondary_context_cleanup(self) -> None:
        module = self.composite

        @contextmanager
        def manager() -> object:
            try:
                yield object()
            finally:
                raise RuntimeError("secondary-cleanup")

        primary = ValueError("primary-body")
        with self.assertRaisesRegex(ValueError, "primary-body") as caught:
            with module.preserve_primary_context(manager()):
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_inner_loader_rejects_preimport_and_named_inode_replacement(self) -> None:
        module = self.composite
        sys.modules[module.OBJECT_WRAPPER_INNER_MODULE] = types.ModuleType(
            module.OBJECT_WRAPPER_INNER_MODULE
        )
        try:
            with self.assertRaisesRegex(
                module.MatchedInferAdapterR5FError,
                "imported before composite",
            ):
                module._load_object_wrapper_inner()
        finally:
            sys.modules.pop(module.OBJECT_WRAPPER_INNER_MODULE, None)

        manager = module.held_object_sources()
        held = manager.__enter__()
        _, _, _, inner_authority = held
        original = inner_authority.path
        replacement = original.with_name(original.name + ".replacement")
        replacement.write_bytes(original.read_bytes())
        replacement.chmod(0o444)
        original.chmod(0o644)
        os.replace(replacement, original)
        original.chmod(0o444)
        with self.assertRaisesRegex(
            module.MatchedInferAdapterR5FError,
            "replay differs",
        ):
            manager.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
