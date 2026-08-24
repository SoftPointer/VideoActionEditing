#!/usr/bin/env python3

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import e00_r6_atomic_marker as atomic_marker
import validate_e00_three_vessel_clean_diag_r6 as validator
from tools import build_e00_three_vessel_clean_diag_r6_package as package


BOOTSTRAP = METHOD_ROOT / "tools/e00_three_vessel_clean_diag_r6_external_bootstrap.py"
ROOT = REPO_ROOT / package.ROOT_FILE
TEMPLATE = METHOD_ROOT / "scripts/auh_launch_e00_three_vessel_clean_diag_r6_external_bootstrap_template.sh"


class E00CleanDiagnosticR6ExternalBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = validator.DEFAULT_PROTOCOL
        self.protocol = copy.deepcopy(validator.load_protocol(self.protocol_path))
        self.root_sha = package._sha256(ROOT)
        self.root = json.loads(ROOT.read_text(encoding="utf-8"))

    @staticmethod
    def sha(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def make_core_tree(self, temporary: Path) -> Path:
        dfix2 = temporary / "dfix2"
        for relative in self.root["core_paths"]:
            target = dfix2 / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, target)
        return dfix2

    def build_actual_package(self, temporary: Path) -> Path:
        output = temporary / "package"
        package.build_package(
            dfix2_source_tree=self.make_core_tree(temporary),
            overlay_root=REPO_ROOT,
            output=output,
            expected_root_sha256=self.root_sha,
        )
        return output

    def run_bootstrap(self, flags: list[str], package_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *flags, str(BOOTSTRAP), "--package-root", str(package_root),
             "--root", str(package_root / package.ROOT_FILE), "--expected-root-sha256",
             self.sha((package_root / package.ROOT_FILE).read_bytes()), *extra],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
        )

    def test_fresh_protocol_exact_contract_and_diagnostic_scope(self) -> None:
        value = validator.validate_protocol(self.protocol)
        self.assertEqual(value["revision_tag"], validator.REVISION_TAG)
        self.assertEqual(self.protocol["ab_gate_contract"]["required_equalities"], validator.REQUIRED_EQUALITIES)
        self.assertEqual(
            self.protocol["ab_gate_contract"]["current_artifact_revalidation_required"],
            validator.REQUIRED_CURRENT_BINDINGS,
        )
        scope = self.protocol["diagnostic_scope"]
        self.assertTrue(scope["old_qk_route_diagnostic_only"])
        self.assertFalse(scope["property_preservation_fix_claimed"])
        self.assertTrue(scope["a_b_bit_exact_or_stop_without_c"])
        self.assertEqual(scope["b_c_only_intended_difference"], "final_pure_qk_route_application")
        placeholder = self.protocol["phase_a_placeholder_contract"]
        self.assertEqual(placeholder["classification"], "content_static_lossy_decode_reencode_placeholder")
        self.assertFalse(placeholder["decoded_bit_exact_source_video_claimed"])

    def test_required_equalities_are_order_and_content_exact(self) -> None:
        for change in ("remove", "append", "reorder"):
            altered = copy.deepcopy(self.protocol)
            rows = altered["ab_gate_contract"]["required_equalities"]
            if change == "remove":
                rows.pop()
            elif change == "append":
                rows.append("unregistered")
            else:
                rows[0], rows[1] = rows[1], rows[0]
            with self.assertRaises(validator.E00R6Error):
                validator.validate_protocol(altered)

    def test_bootstrap_is_stdlib_only_and_root_is_one_way_complete(self) -> None:
        tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertLessEqual(imported, {
            "__future__", "argparse", "hashlib", "json", "os", "pathlib", "stat", "sys", "typing",
        })
        self.assertTrue(self.root["one_way_root_pins"])
        self.assertFalse(self.root["consumers_pin_root"])
        self.assertEqual(self.root["pinned_file_count"], len(self.root["pins"]))
        self.assertNotIn(package.ROOT_FILE, self.root["pins"])
        self.assertEqual(self.root["runtime_consumers"], package.RUNTIME_CONSUMERS)
        for relative, expected in self.root["pins"].items():
            self.assertEqual(package._sha256(REPO_ROOT / relative), expected, relative)
        for relative in self.root["runtime_consumers"]:
            self.assertNotIn(self.root_sha, (REPO_ROOT / relative).read_text(encoding="utf-8"))

    def test_runtime_local_import_closure_is_root_pinned(self) -> None:
        module_by_name = {path.stem: path for path in METHOD_ROOT.glob("*.py")}
        pending = [
            METHOD_ROOT / "e00_legacy_infer_fixed_rng_wrapper_r6.py",
            METHOD_ROOT / "validate_e00_three_vessel_clean_diag_r6.py",
            METHOD_ROOT / "infer_anchor_sga_anc_event_v1.py",
        ]
        observed: set[Path] = set()
        while pending:
            path = pending.pop()
            if path in observed:
                continue
            observed.add(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module.split(".")[0])
                elif (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module" and node.args
                    and isinstance(node.args[0], ast.Str)
                ):
                    names.append(node.args[0].s.split(".")[0])
            pending.extend(module_by_name[name] for name in names if name in module_by_name)
        for path in observed:
            relative = path.relative_to(REPO_ROOT).as_posix()
            self.assertIn(relative, self.root["pins"], relative)

    def test_external_bootstrap_verifies_actual_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_actual_package(Path(directory))
            for flags in (["-I", "-S", "-B"], ["-O", "-I", "-S", "-B"]):
                result = self.run_bootstrap(flags, output, "--verify-only")
                self.assertEqual(result.returncode, 0, (flags, result.stderr))
                self.assertTrue(json.loads(result.stdout)["cache_bytecode_absent"])

    def test_pre_import_sentinel_never_executes_normal_or_optimized(self) -> None:
        for flags in (["-I", "-S", "-B"], ["-O", "-I", "-S", "-B"]):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                consumer_relative = "methods/consumer.sh"
                dependency_relative = "methods/dependency.py"
                sentinel = root / "SENTINEL_MUST_NOT_EXIST"
                consumer = root / consumer_relative
                dependency = root / dependency_relative
                consumer.parent.mkdir(parents=True)
                consumer.write_text("#!/bin/bash\nprintf executed > %s\n" % sentinel, encoding="utf-8")
                dependency.write_text("raise RuntimeError('must never import')\n", encoding="utf-8")
                root_path = root / package.ROOT_FILE
                root_path.parent.mkdir(parents=True)
                pins = {
                    consumer_relative: self.sha(consumer.read_bytes()),
                    dependency_relative: self.sha(dependency.read_bytes()),
                }
                root_value = {
                    "schema_version": "bernini-e00-clean-diagnostic-r6-external-bootstrap-root-v6",
                    "revision_tag": validator.REVISION_TAG,
                    "complete": True, "immutable": True, "one_way_root_pins": True,
                    "consumers_pin_root": False, "runtime_diagnostic_only": True,
                    "property_preservation_fix_claimed": False,
                    "runtime_consumers": [consumer_relative], "pinned_file_count": len(pins), "pins": pins,
                }
                root_path.write_text(json.dumps(root_value, sort_keys=True), encoding="utf-8")
                dependency.write_text("raise RuntimeError('changed but still must never import')\n", encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, *flags, str(BOOTSTRAP), "--package-root", str(root),
                     "--root", str(root_path), "--expected-root-sha256", self.sha(root_path.read_bytes()),
                     "--consumer-relative", consumer_relative, "--"],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pinned current bytes differ", result.stderr)
                self.assertFalse(sentinel.exists())
                self.assertFalse(any(path.name == "__pycache__" for path in root.rglob("*")))

    def test_synchronized_consumer_and_manifest_change_is_rejected_normal_and_optimized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_actual_package(Path(directory))
            target_relative = package.BRIDGE
            target = output / target_relative
            target.write_bytes(target.read_bytes() + b"\n# synchronized change\n")
            manifest_path = output / package.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in manifest["files"]:
                if row["path"] == target_relative:
                    row["sha256"] = package._sha256(target)
                    row["bytes"] = target.stat().st_size
            manifest["content_digest"] = hashlib.sha256(package._canonical(manifest["files"])).hexdigest()
            manifest_path.write_bytes(package._canonical(manifest) + b"\n")
            builder = output / "methods/bernini_action_editing/tools/build_e00_three_vessel_clean_diag_r6_package.py"
            for flags in (["-B"], ["-O", "-B"]):
                result = subprocess.run(
                    [sys.executable, *flags, str(builder), "verify", "--package-root", str(output),
                     "--manifest", str(manifest_path), "--expected-root-sha256", self.root_sha],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
                )
                self.assertNotEqual(result.returncode, 0, flags)
                self.assertIn("pinned current bytes differ", result.stderr, flags)

    def test_package_rejects_any_cache_or_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_actual_package(Path(directory))
            manifest = output / package.MANIFEST_NAME
            cache = output / "nested/__pycache__"
            cache.mkdir(parents=True)
            with self.assertRaises(package.PackageError):
                package.verify_package(package_root=output, manifest_path=manifest, expected_root_sha256=self.root_sha)
            cache.rmdir(); cache.parent.rmdir()
            bytecode = output / "payload.pyc"
            bytecode.write_bytes(b"compiled")
            with self.assertRaises(package.PackageError):
                package.verify_package(package_root=output, manifest_path=manifest, expected_root_sha256=self.root_sha)

    def test_builder_checks_legacy_helper_before_lazy_import(self) -> None:
        source = (METHOD_ROOT / "tools/build_e00_three_vessel_clean_diag_r6_package.py").read_text(encoding="utf-8")
        function = source[source.index("def _load_legacy_helper"):source.index("def _review_marker")]
        self.assertLess(function.index("_sha256(helper_path)"), function.index("spec_from_file_location"))
        top = ast.parse(source)
        local_top_imports = []
        for node in top.body:
            if isinstance(node, ast.Import):
                local_top_imports.extend(alias.name for alias in node.names if (METHOD_ROOT / (alias.name + ".py")).exists())
            elif isinstance(node, ast.ImportFrom) and node.module and (METHOD_ROOT / (node.module + ".py")).exists():
                local_top_imports.append(node.module)
        self.assertEqual(local_top_imports, [])

    def test_atomic_terminal_marker_and_launcher_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "completed.json"
            receipt = atomic_marker.write_marker(marker, {"state": "completed", "complete": True})
            self.assertTrue(receipt["reread_bit_exact"])
            self.assertTrue(receipt["atomic_rename_completed"])
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["state"], "completed")
            self.assertFalse(any("exclusive-" in path.name for path in root.iterdir()))
            with self.assertRaises(atomic_marker.MarkerError):
                atomic_marker.write_marker(marker, {"state": "overwrite"})

        bridge = (REPO_ROOT / package.BRIDGE).read_text(encoding="utf-8")
        a_launcher = (REPO_ROOT / package.A_LAUNCHER).read_text(encoding="utf-8")
        bc_launcher = (REPO_ROOT / package.BC_LAUNCHER).read_text(encoding="utf-8")
        for text in (bridge, a_launcher, bc_launcher):
            self.assertNotIn("sbatch ", text)
            self.assertNotIn("scancel ", text)
            self.assertNotIn("export CUDA_VISIBLE_DEVICES=", text)
            self.assertNotIn("export ROCR_VISIBLE_DEVICES=", text)
            self.assertIn("check_cache_free", text)
        self.assertIn("fixed_parent_job=143808", bridge)
        self.assertLess(bridge.index("bridge-capability"), bridge.index("torch.distributed.run"))
        self.assertLess(bc_launcher.index('run_arm B "$role_b"'), bc_launcher.index(' verify-ab-gate'))
        self.assertLess(bc_launcher.index(' verify-ab-gate'), bc_launcher.index('run_arm C "$role_c"'))
        self.assertIn('E00_R6_AB_GATE="$gate"', bc_launcher)
        for text in (a_launcher, bc_launcher):
            lines = [line.strip() for line in text.splitlines()]
            index = lines.index('mkdir "$lock"')
            self.assertEqual(lines[index + 1], "trap finalize EXIT")
            self.assertIn("terminal_marker_atomic_write_verified", text)

    def test_launch_template_hashes_bootstrap_before_execution(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")
        bootstrap_sha = package._sha256(BOOTSTRAP)
        self.assertIn(bootstrap_sha, text)
        self.assertIn(self.root_sha, text)
        self.assertLess(text.index('sha256sum -- "$bootstrap"'), text.index('exec "$python_bin"'))
        self.assertNotIn(str(ROOT), text)

    def test_current_native_and_four_rng_receipts_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"; video.write_bytes(b"video")
            native = root / "native.json"; native.write_bytes(b"native")
            rng_paths = []
            for rank in range(4):
                path = root / ("rng-%d.json" % rank); path.write_bytes(("rng-%d" % rank).encode("utf-8")); rng_paths.append(path)
            artifacts = {
                "video": {"path": str(video), "sha256": validator.file_sha256(video)},
                "native_receipt": {"path": str(native), "sha256": validator.file_sha256(native)},
                "rng_receipts_rank_order": [
                    {"rank": rank, "path": str(path), "sha256": validator.file_sha256(path)}
                    for rank, path in enumerate(rng_paths)
                ],
            }
            self.assertEqual(len(validator.validate_current_artifact_bytes(artifacts)["artifacts"]["rng_receipts_rank_order"]), 4)
            rng_paths[2].write_bytes(b"changed")
            with self.assertRaises(Exception):
                validator.validate_current_artifact_bytes(artifacts)

    def test_final_audit_retains_instruction_source_anchor_and_bc_binding(self) -> None:
        audits = [
            {"artifacts": {"native_receipt": {"path": "a"}}},
            {"artifacts": {"native_receipt": {"path": "b"}}},
            {"artifacts": {"native_receipt": {"path": "c"}}},
        ]
        binding = {"same_noise_latent_schedule_caption_anchor_observer": True}
        with mock.patch.object(validator.r5, "build_final_audit", return_value={"complete": True}), \
             mock.patch.object(validator, "revalidate_arm_current", side_effect=audits), \
             mock.patch.object(validator, "_load", side_effect=[{}, {}, {}]), \
             mock.patch.object(validator, "_bc_input_binding", side_effect=[binding, binding]):
            value = validator.build_final_audit(
                protocol=self.protocol, protocol_path=self.protocol_path,
                a_audit_path=Path("a"), b_audit_path=Path("b"), c_audit_path=Path("c"),
                ab_gate_path=Path("gate"),
            )
        self.assertTrue(value["old_qk_route_white_leakage_diagnostic_only"])
        self.assertFalse(value["property_preservation_fix_claimed"])
        retained = value["retained_three_object_instruction_and_source_anchor"]
        self.assertIn("three_object_editing_instruction_utf8", retained)
        self.assertIn("source_video", retained)
        self.assertIn("self_generated_anchor_video", retained)

        with mock.patch.object(validator.r5, "build_final_audit", return_value={"complete": True}), \
             mock.patch.object(validator, "revalidate_arm_current", side_effect=audits), \
             mock.patch.object(validator, "_load", side_effect=[{}, {}, {}]), \
             mock.patch.object(validator, "_bc_input_binding", side_effect=[{"x": 1}, {"x": 2}]):
            with self.assertRaises(validator.E00R6Error):
                validator.build_final_audit(
                    protocol=self.protocol, protocol_path=self.protocol_path,
                    a_audit_path=Path("a"), b_audit_path=Path("b"), c_audit_path=Path("c"),
                    ab_gate_path=Path("gate"),
                )


if __name__ == "__main__":
    unittest.main()
