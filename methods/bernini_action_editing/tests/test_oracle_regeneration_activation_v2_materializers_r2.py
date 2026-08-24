#!/usr/bin/env python3
"""CPU/static release tests for activation-v2 materializer revision r2."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
CORE = METHOD_ROOT / "oracle_regeneration_activation_v2.py"
VAE_TOOL = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py"
)
PROMPT_TOOL = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py"
)
LAUNCHER = (
    METHOD_ROOT / "scripts/auh_materialize_oracle_regeneration_activation_v2_r2.sh"
)
MATERIALIZE_VAE = METHOD_ROOT / "tools/materialize_vae.py"
BUILD_RENDERER_DATASET = METHOD_ROOT / "tools/build_renderer_dataset.py"
TOOLS_INIT = METHOD_ROOT / "tools/__init__.py"
EXPECTED = {
    CORE: "ef97259dd181ff065267e32f1e5cca158e26ad5174457780163658a3db728bb0",
    VAE_TOOL: "07b40cdd67771d257ce546ca4166301980c6768269acc5f097fc08973656bbde",
    PROMPT_TOOL: "5bba9f977fa40e5044053baaaf73eba779b3816ef6137457a06ceac82a3463af",
    MATERIALIZE_VAE: "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    BUILD_RENDERER_DATASET: "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    TOOLS_INIT: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
PINNED_RUNTIME_CLOSURE = {
    "oracle_regeneration_canary_v1.py": "0148b137c200e426ff18571f71d373a9e6ef595c620664925dae0ab9d1d91081",
    "native_branch_homotopy_runtime_v1.py": "b81ee152e358e4d5a6638dfccf1232c4e221311ffb38937e61be3c6a799b84d5",
    "native_branch_homotopy_v1.py": "2585416e61935db62cc7534daf19b4bb851f9fdcdeb92f78e6152f55e034f3d0",
    "self_guided_action_field_v1.py": "2ad204c09f5eb60865017b1e596de25b777d8d6ed43774f4dcbc23a4ad58bc7e",
    "tri_branch_unipc.py": "58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2",
    "infer_native_identity_generation_canary.py": "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42",
    "infer_native_branch_homotopy_canary.py": "d6dab735ce52da151848c96f9e00775994dc281ace20afa6dcb9fb64709e5983",
    "infer_source_kv_carrier_oracle.py": "fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604",
    "infer_source_value_residual_oracle.py": "40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8",
    "infer_native_self_guided_action_field_canary.py": "ad591fe5bd5943fab59603400fcf70a126f3461a39670f93a78aa24b1902d313",
    "infer_lora.py": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "train_lora.py": "8e8daf422548bc29e2c18f2d2c692af2dd3109aaad1897fc31e590a69d7e593e",
    "action_preservation_decoded_eval_model_authority_v2.py": "760ed9988147a44965fd47f68a08fd353ce1d900e661b55bb818088ec9ef848e",
    "source_kv_replay.py": "45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3",
    "source_kv_route_batches.py": "7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e",
    "source_value_residual.py": "420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124",
}
E02_CAPTION = (
    "The same pale bare hand firmly grips the same red mushroom at its lower "
    "stem, twists and pulls it free from the same soil, lifts the same intact "
    "mushroom above the newly empty root hole, and holds it there. Exactly one "
    "hand and one mushroom remain visible; do not duplicate, split, or fuse "
    "either one."
)
E03_CAPTION = (
    "The same farm worker moves the same harvested root cluster over the same "
    "woven basket, lowers it past the rim, opens the same hand, and releases "
    "it. The cluster falls, bounces slightly, and settles inside while the "
    "now-empty hand withdraws. Do not duplicate the hand or cluster."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duplicate_literal_dict_keys(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    duplicates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        observed: set[str] = set()
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if key.value in observed:
                duplicates.append((node.lineno, key.value))
            observed.add(key.value)
    return duplicates


class MaterializerStaticTests(unittest.TestCase):
    def test_exact_authoring_code_tuple_and_hard_disabled_core(self) -> None:
        for path, expected in EXPECTED.items():
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(sha256(path), expected)
        namespace: dict[str, object] = {}
        core_source = CORE.read_text(encoding="utf-8")
        self.assertIn("COMPILED_AUTHORITY_PACKET_SHA256: Optional[str] = None", core_source)
        self.assertIn(
            "COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256: Optional[str] = None",
            core_source,
        )
        self.assertNotIn("20260818_regeneration_oracle_human_annotation_v1.md", core_source)
        self.assertNotIn("e02_round37_gate_proposal", core_source)
        self.assertNotIn("torch", sys.modules)
        del namespace

    def test_python_sources_have_no_duplicate_literal_dict_keys(self) -> None:
        for path in (CORE, VAE_TOOL, PROMPT_TOOL):
            self.assertEqual(duplicate_literal_dict_keys(path), [], str(path))

    def test_materialize_vae_transitive_import_closure_is_real_and_exact(self) -> None:
        bundle_entries = {
            "oracle_regeneration_activation_v2.py": EXPECTED[CORE],
            "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py": EXPECTED[VAE_TOOL],
            "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py": EXPECTED[PROMPT_TOOL],
            "tools/materialize_vae.py": EXPECTED[MATERIALIZE_VAE],
            "tools/build_renderer_dataset.py": EXPECTED[BUILD_RENDERER_DATASET],
            "tools/__init__.py": EXPECTED[TOOLS_INIT],
            **PINNED_RUNTIME_CLOSURE,
        }
        probe = (
            "import hashlib,json,os,pathlib,sys;"
            "root=pathlib.Path(sys.argv[1]).resolve(strict=True);"
            "sys.path.insert(0,str(root));"
            "import tools,tools.materialize_vae as m;"
            "import tools.materialize_oracle_regeneration_vae_refs_activation_v2_r2 as v;"
            "import tools.materialize_oracle_regeneration_prompts_activation_v2_r2 as p;"
            "loaded=set();"
            "[(loaded.add(str(pathlib.Path(x).resolve(strict=True).relative_to(root)))) "
            "for x in [getattr(q,'__file__',None) for q in tuple(sys.modules.values())] "
            "if x and str(pathlib.Path(x).resolve(strict=True)).startswith(str(root)+os.sep)];"
            "row={'materialize_vae':str(pathlib.Path(m.__file__).resolve(strict=True)),"
            "'raw_builder':str(pathlib.Path(m.raw_builder.__file__).resolve(strict=True)),"
            "'tools_init':str(pathlib.Path(tools.__file__).resolve(strict=True)),"
            "'tools_path':[str(pathlib.Path(x).resolve(strict=True)) for x in tools.__path__],"
            "'vae_r2':str(pathlib.Path(v.__file__).resolve(strict=True)),"
            "'prompt_r2':str(pathlib.Path(p.__file__).resolve(strict=True)),"
            "'loaded':sorted(loaded),"
            "'lazy_loaded':'_omnivideo2_strict_action_preview_materializer' in sys.modules};"
            "print(json.dumps(row,sort_keys=True))"
        )
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = Path(temporary).resolve(strict=True) / "sealed-method-root"
            for relative, expected in bundle_entries.items():
                source = METHOD_ROOT / relative
                self.assertEqual(sha256(source), expected, relative)
                destination = bundle_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                self.assertEqual(sha256(destination), expected, relative)
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "-c", probe, str(bundle_root)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        row = json.loads(completed.stdout)
        expected_loaded = {
            "native_branch_homotopy_runtime_v1.py",
            "native_branch_homotopy_v1.py",
            "oracle_regeneration_activation_v2.py",
            "oracle_regeneration_canary_v1.py",
            "self_guided_action_field_v1.py",
            "tools/__init__.py",
            "tools/build_renderer_dataset.py",
            "tools/materialize_vae.py",
            "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py",
            "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py",
            "tri_branch_unipc.py",
        }
        self.assertEqual(set(row["loaded"]), expected_loaded)
        self.assertEqual(
            Path(row["materialize_vae"]), bundle_root / "tools/materialize_vae.py"
        )
        self.assertEqual(
            Path(row["raw_builder"]), bundle_root / "tools/build_renderer_dataset.py"
        )
        self.assertEqual(Path(row["tools_init"]), bundle_root / "tools/__init__.py")
        self.assertEqual([Path(value) for value in row["tools_path"]], [bundle_root / "tools"])
        self.assertEqual(
            Path(row["vae_r2"]),
            bundle_root / "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py",
        )
        self.assertEqual(
            Path(row["prompt_r2"]),
            bundle_root / "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py",
        )
        self.assertFalse(row["lazy_loaded"])

    def test_vae_tool_is_source_only_and_sampler_free(self) -> None:
        source = VAE_TOOL.read_text(encoding="utf-8")
        required = (
            "rank zero",
            "alone decodes the bound source",
            "references_encoded_as_four_independent_rgb_frames",
            "references_not_sliced_from_full_source_latent",
            "reference_content_duplicates_rejected",
            "target_video_or_latent_used\": False",
            "self_generated_anchor_tensor_used\": False",
            "full_model_or_sampler_loaded\": False",
            "transformer_loaded\": False",
            "scheduler_loaded\": False",
        )
        self.assertTrue(all(token in source for token in required))
        self.assertNotIn("diffusion.sample", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward(", source)

    def test_prompt_tool_rank0_only_t5_and_exact_captions(self) -> None:
        source = PROMPT_TOOL.read_text(encoding="utf-8")
        required = (
            "rank0_only_text_encoder_load\": True",
            "nonzero_ranks_never_deserialized_text_encoder\": True",
            "rank0_only_text_encode\": True",
            "broadcast_exact\": True",
            "sampler_or_scheduler_called\": False",
            "target_video_or_latent_used\": False",
            "self_generated_anchor_tensor_used\": False",
        )
        self.assertTrue(all(token in source for token in required))
        self.assertNotIn("diffusion.sample", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward(", source)
        core_source = CORE.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(E02_CAPTION.encode()).hexdigest(), core_source)
        self.assertIn(hashlib.sha256(E03_CAPTION.encode()).hexdigest(), core_source)

    def test_launcher_is_existing_allocation_world4_and_exact_pinned(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("--nproc_per_node=4", source)
        self.assertIn("ORACLE_MATERIALIZER_KIND", source)
        self.assertIn("ORACLE_CASE_ID", source)
        self.assertIn("ORACLE_VISIBLE_GPUS", source)
        self.assertIn("unset PYTHONPATH", source)
        self.assertIn("PYTHONNOUSERSITE=1", source)
        self.assertNotIn("srun ", source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("ssh ", source)
        for path, digest in EXPECTED.items():
            self.assertIn(path.name, source)
            self.assertEqual(len(re.findall(re.escape(digest), source)), 1)
        for relative, digest in PINNED_RUNTIME_CLOSURE.items():
            self.assertIn(relative, source)
            self.assertEqual(len(re.findall(re.escape(digest), source)), 1)
        self.assertIn(sha256(Path(__file__).resolve(strict=True)), source)
        self.assertNotIn("__VAE_R2_SHA256__", source)
        self.assertNotIn("__PROMPT_R2_SHA256__", source)
        self.assertNotIn("__STATIC_R2_SHA256__", source)
        self.assertEqual(source.count(E02_CAPTION), 1)
        self.assertEqual(source.count(E03_CAPTION), 1)
        self.assertIn(
            "source-only diagnostic provenance inside an existing one-node allocation",
            source,
        )
        self.assertNotIn("human-reviewed", source.lower())
        self.assertNotIn("cryptographically signed", source.lower())

    def test_launcher_rejects_duplicate_gpu_map_and_out_of_range_port_early(self) -> None:
        base = {
            **os.environ,
            "ORACLE_ACTIVATION_V2_REPO_ROOT": "/",
            "ORACLE_ACTIVATION_V2_PYTHON_BIN": "/",
            "BERNINI_OFFICIAL_ROOT": "/",
            "BERNINI_VEOMNI_ROOT": "/",
            "BERNINI_ACTION_CHECKPOINT": "/",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST": "/",
            "ORACLE_MATERIALIZER_OUTPUT_DIR": "/not-created",
            "ORACLE_MATERIALIZER_KIND": "vae",
            "ORACLE_CASE_ID": "e02",
            "ORACLE_MASTER_PORT": "40001",
        }
        duplicate = subprocess.run(
            ["bash", str(LAUNCHER)],
            env={**base, "ORACLE_VISIBLE_GPUS": "0,0,0,0"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("visible GPU list differs", duplicate.stderr)
        invalid_port = subprocess.run(
            ["bash", str(LAUNCHER)],
            env={
                **base,
                "ORACLE_VISIBLE_GPUS": "0,1,2,3",
                "ORACLE_MASTER_PORT": "99999",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(invalid_port.returncode, 2)
        self.assertIn("master port exceeds 65535", invalid_port.stderr)

    def test_atomic_receipt_publish_never_replaces_existing_name(self) -> None:
        for index, tool_path in enumerate((VAE_TOOL, PROMPT_TOOL)):
            spec = importlib.util.spec_from_file_location(
                f"activation_v2_materializer_test_{index}", tool_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve(strict=True)
                path = directory / "receipt.json"
                module._write_receipt(path, {"first": True})
                before = path.read_bytes()
                before_stat = path.stat()
                with self.assertRaises(module.__dict__[next(
                    name
                    for name in module.__dict__
                    if name.endswith("MaterializationError")
                )]):
                    module._write_receipt(path, {"second": True})
                after_stat = path.stat()
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(stat.S_IMODE(after_stat.st_mode), 0o444)
                self.assertEqual(after_stat.st_nlink, 1)
                self.assertEqual(
                    (after_stat.st_dev, after_stat.st_ino),
                    (before_stat.st_dev, before_stat.st_ino),
                )

    def test_direct_exclusive_writer_has_no_link_unlink_transition(self) -> None:
        for tool_path in (VAE_TOOL, PROMPT_TOOL):
            source = tool_path.read_text(encoding="utf-8")
            writer = source[source.index("def _write_receipt"):source.index("def build_parser")]
            self.assertIn("os.O_EXCL", writer)
            self.assertIn("os.O_RDWR", writer)
            self.assertIn("os.fchmod(descriptor, 0o444)", writer)
            self.assertIn("st_nlink != 1", writer)
            self.assertNotIn("os.link(", writer)
            self.assertNotIn("os.replace(", writer)
            self.assertNotIn("st_mtime", writer)

    def test_two_direct_writer_algorithms_are_ast_identical(self) -> None:
        normalized = []
        for tool_path in (VAE_TOOL, PROMPT_TOOL):
            source = tool_path.read_text(encoding="utf-8")
            source = source.replace(
                "VaeReferenceMaterializationError", "MaterializationError"
            ).replace("PromptMaterializationError", "MaterializationError")
            tree = ast.parse(source)
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_write_receipt"
            )
            if (
                function.body
                and isinstance(function.body[0], ast.Expr)
                and isinstance(function.body[0].value, ast.Constant)
                and isinstance(function.body[0].value.value, str)
            ):
                function.body.pop(0)
            normalized.append(ast.dump(function, include_attributes=False))
        self.assertEqual(normalized[0], normalized[1])

    def test_partial_direct_creation_is_preserved_fail_closed(self) -> None:
        for index, tool_path in enumerate((VAE_TOOL, PROMPT_TOOL)):
            spec = importlib.util.spec_from_file_location(
                f"activation_v2_materializer_partial_test_{index}", tool_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            error_class = module.__dict__[next(
                name for name in module.__dict__ if name.endswith("MaterializationError")
            )]
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary).resolve(strict=True) / "receipt.json"
                real_write = os.write
                calls = 0

                def interrupted(descriptor: int, payload: bytes) -> int:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return real_write(descriptor, payload[: max(1, len(payload) // 2)])
                    raise OSError("injected interrupted write")

                with mock.patch.object(module.os, "write", side_effect=interrupted):
                    with self.assertRaises(OSError):
                        module._write_receipt(path, {"complete": False, "pad": "x" * 64})
                self.assertTrue(path.is_file())
                partial = path.read_bytes()
                inode = path.stat().st_ino
                with self.assertRaises(error_class):
                    module._write_receipt(path, {"complete": True})
                self.assertEqual(path.read_bytes(), partial)
                self.assertEqual(path.stat().st_ino, inode)

    def test_fchmod_and_fsync_failures_preserve_owned_final_inode(self) -> None:
        for index, tool_path in enumerate((VAE_TOOL, PROMPT_TOOL)):
            spec = importlib.util.spec_from_file_location(
                f"activation_v2_materializer_sync_test_{index}", tool_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            error_class = module.__dict__[next(
                name for name in module.__dict__ if name.endswith("MaterializationError")
            )]
            for operation in ("fchmod", "fsync"):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary).resolve(strict=True) / f"{operation}.json"
                    with mock.patch.object(
                        module.os, operation, side_effect=OSError(f"injected {operation}")
                    ):
                        with self.assertRaises(OSError):
                            module._write_receipt(path, {"operation": operation})
                    self.assertTrue(path.is_file())
                    before = path.read_bytes()
                    before_inode = path.stat().st_ino
                    with self.assertRaises(error_class):
                        module._write_receipt(path, {"retry": False})
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(path.stat().st_ino, before_inode)

    def test_short_write_and_interrupted_write_are_replayed_exactly(self) -> None:
        for index, tool_path in enumerate((VAE_TOOL, PROMPT_TOOL)):
            spec = importlib.util.spec_from_file_location(
                f"activation_v2_materializer_short_test_{index}", tool_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary).resolve(strict=True) / "receipt.json"
                expected = {"pad": "x" * 257, "short_write": True}
                real_write = os.write
                calls = 0

                def short_then_interrupt(descriptor: int, payload: bytes) -> int:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return real_write(descriptor, payload[:7])
                    if calls == 2:
                        raise InterruptedError("injected EINTR")
                    return real_write(descriptor, payload)

                with mock.patch.object(
                    module.os, "write", side_effect=short_then_interrupt
                ):
                    module._write_receipt(path, expected)
                self.assertEqual(
                    path.read_bytes(),
                    module.activation.safe_core.canonical_json_bytes_v1(expected),
                )
                self.assertGreaterEqual(calls, 3)

    def test_symlink_and_directory_destinations_are_never_replaced(self) -> None:
        for index, tool_path in enumerate((VAE_TOOL, PROMPT_TOOL)):
            spec = importlib.util.spec_from_file_location(
                f"activation_v2_materializer_type_test_{index}", tool_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            error_class = module.__dict__[next(
                name for name in module.__dict__ if name.endswith("MaterializationError")
            )]
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(strict=True)
                target = root / "target.json"
                target.write_bytes(b"target-stays")
                target_inode = target.stat().st_ino
                for name, link_target in (
                    ("live-link.json", target),
                    ("dangling-link.json", root / "absent-target.json"),
                ):
                    destination = root / name
                    destination.symlink_to(link_target)
                    with self.assertRaises(error_class):
                        module._write_receipt(destination, {"replace": False})
                    self.assertTrue(destination.is_symlink())
                directory = root / "directory.json"
                directory.mkdir()
                with self.assertRaises(error_class):
                    module._write_receipt(directory, {"replace": False})
                self.assertTrue(directory.is_dir())
                self.assertEqual(target.read_bytes(), b"target-stays")
                self.assertEqual(target.stat().st_ino, target_inode)

    def test_metadata_time_drift_is_ignored_but_inode_mismatch_fails(self) -> None:
        for index, tool_path in enumerate((VAE_TOOL, PROMPT_TOOL)):
            spec = importlib.util.spec_from_file_location(
                f"activation_v2_materializer_inode_test_{index}", tool_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            error_class = module.__dict__[next(
                name for name in module.__dict__ if name.endswith("MaterializationError")
            )]
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(strict=True)
                stable = root / "mtime-drift.json"
                real_fstat = os.fstat
                tick = 0

                def drifting_fstat(descriptor: int):
                    nonlocal tick
                    row = list(real_fstat(descriptor))
                    tick += 1
                    row[8] = float(row[8]) + tick
                    return os.stat_result(row)

                with mock.patch.object(module.os, "fstat", side_effect=drifting_fstat):
                    module._write_receipt(stable, {"mtime": "ignored"})
                self.assertEqual(stat.S_IMODE(stable.stat().st_mode), 0o444)

                mismatch = root / "inode-mismatch.json"
                real_stat = os.stat

                def mismatched_stat(path_value, *args, **kwargs):
                    row = real_stat(path_value, *args, **kwargs)
                    if path_value == mismatch.name and kwargs.get("dir_fd") is not None:
                        values = list(row)
                        values[1] = int(values[1]) + 1
                        return os.stat_result(values)
                    return row

                with mock.patch.object(module.os, "stat", side_effect=mismatched_stat):
                    with self.assertRaises(error_class):
                        module._write_receipt(mismatch, {"inode": "must-match"})


if __name__ == "__main__":
    unittest.main()
