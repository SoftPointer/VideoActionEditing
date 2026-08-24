from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS = METHOD_ROOT / "tools"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import bind_seer_full160_eval_source_v1 as binder  # noqa: E402
import build_seer_full160_eval_overlay_v1 as overlay  # noqa: E402


LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_eval_seer_full160_core4_array_20260813.sbatch"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(path: Path, unsigned: dict[str, object]) -> dict[str, object]:
    value = {**unsigned, "receipt_digest": binder.object_sha256(unsigned)}
    path.write_bytes(binder.canonical_json_bytes(value) + b"\n")
    return value


class SeerFull160EvalOverlayTests(unittest.TestCase):
    def _make_method_root(self, parent: Path) -> Path:
        root = parent / "methods" / "bernini_action_editing"
        root.mkdir(parents=True)
        for index, name in enumerate(overlay.OVERLAY_FILES):
            (root / name).write_bytes(f"runtime-{index}\n".encode("ascii"))
        return root

    def _make_training_archive(self, root: Path, path: Path) -> None:
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
            for source in sorted(root.rglob("*")):
                archive.add(
                    source,
                    arcname=source.relative_to(root),
                    recursive=False,
                )

    def _make_case_evidence(self, root: Path) -> tuple[Path, Path, Path, Path]:
        checkpoint = root / "checkpoint-00000160"
        adapter_dir = checkpoint / "adapter"
        adapter_dir.mkdir(parents=True)
        adapter_model = adapter_dir / "adapter_model.safetensors"
        adapter_model.write_bytes(b"adapter-160")
        training_receipt_path = checkpoint / "receipt.json"
        training = _write_receipt(
            training_receipt_path,
            {
                "global_step": 160,
                "max_steps": 160,
                "immutable_contract": {
                    "value": {
                        "method_source_revision": binder.TRAINING_REVISION,
                        "method_source_archive_sha256": binder.TRAINING_ARCHIVE_SHA256,
                    }
                },
            },
        )

        base_video = root / "frozen_base.mp4"
        trained_video = root / "trained_adapter.mp4"
        base_video.write_bytes(b"base-video")
        trained_video.write_bytes(b"trained-video")
        base_output = {"path": str(base_video), "sha256": _sha(base_video)}
        trained_output = {"path": str(trained_video), "sha256": _sha(trained_video)}
        base_inference_path = Path(str(base_video) + ".receipt.json")
        trained_inference_path = Path(str(trained_video) + ".receipt.json")
        base_inference = _write_receipt(
            base_inference_path,
            {
                "schema_version": binder.INFERENCE_SCHEMA_VERSION,
                "method_source_revision": binder.TRAINING_REVISION,
                "method_source_archive_sha256": binder.TRAINING_ARCHIVE_SHA256,
                "adapter": {
                    "enabled": False,
                    "mode": "frozen_base_no_adapter",
                    "strictly_reloaded": False,
                    "safe_merged_for_inference": False,
                    "tensor_count": 0,
                },
                "output": base_output,
            },
        )
        trained_inference = _write_receipt(
            trained_inference_path,
            {
                "schema_version": binder.INFERENCE_SCHEMA_VERSION,
                "method_source_revision": binder.TRAINING_REVISION,
                "method_source_archive_sha256": binder.TRAINING_ARCHIVE_SHA256,
                "adapter": {
                    "enabled": True,
                    "mode": "lora_safe_merge",
                    "checkpoint_root": str(checkpoint),
                    "adapter_model_path": str(adapter_model),
                    "adapter_model_sha256": _sha(adapter_model),
                    "training_receipt_path": str(training_receipt_path),
                    "training_receipt_digest": training["receipt_digest"],
                    "training_global_step": 160,
                    "strictly_reloaded": True,
                    "safe_merged_for_inference": True,
                    "tensor_count": 120,
                },
                "output": trained_output,
            },
        )
        pair_path = root / "paired-receipt.json"
        _write_receipt(
            pair_path,
            {
                "schema_version": binder.PAIR_SCHEMA_VERSION,
                "status": "decoded_pair_ready_for_blind_review_no_method_success_claim",
                "iid": "99cde432839f4240",
                "same_source_instruction_seed_preprocessing_sampler": True,
                "frozen_base": {
                    "output": base_output,
                    "inference_receipt_digest": base_inference["receipt_digest"],
                },
                "trained_adapter": {
                    "output": trained_output,
                    "inference_receipt_digest": trained_inference["receipt_digest"],
                    "adapter_model_sha256": _sha(adapter_model),
                    "training_global_step": 160,
                },
                "full_video_action_and_preservation_review_complete": False,
                "method_success_authorized": False,
            },
        )
        source_path = root / "source-binding.json"
        _write_receipt(
            source_path,
            {
                "schema_version": binder.SCHEMA_VERSION,
                "training_method_source": {
                    "revision": binder.TRAINING_REVISION,
                    "archive_sha256": binder.TRAINING_ARCHIVE_SHA256,
                },
                "inference_runtime_overlay": {
                    "archive_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                    "manifest_digest": "c" * 64,
                    "passed_as_training_method_archive": False,
                },
                "application": {"only_manifest_declared_paths_changed": True},
                "training_receipt_mutated": False,
                "training_provenance_replaced": False,
            },
        )
        return source_path, pair_path, base_video, trained_video

    def test_build_is_byte_deterministic_and_exact_three_file_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method = self._make_method_root(root / "source")
            a1, m1 = root / "one.tar", root / "one.json"
            a2, m2 = root / "two.tar", root / "two.json"
            first = overlay.build(method, a1, m1)
            second = overlay.build(method, a2, m2)
            self.assertEqual(a1.read_bytes(), a2.read_bytes())
            self.assertEqual(m1.read_bytes(), m2.read_bytes())
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertNotEqual(
                first["archive_sha256"],
                overlay.TRAINING_METHOD_SOURCE_ARCHIVE_SHA256,
            )
            with tarfile.open(a1, "r:") as archive:
                self.assertEqual(
                    archive.getnames(),
                    [
                        f"methods/bernini_action_editing/{name}"
                        for name in overlay.OVERLAY_FILES
                    ],
                )

    def test_manifest_rejects_extension_and_archive_rejects_extra_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method = self._make_method_root(root / "source")
            manifest = overlay.build_manifest(method)
            hostile = copy.deepcopy(manifest)
            hostile["files"].append(
                {"path": "extra.py", "sha256": "a" * 64, "size": 1, "mode": "0444"}
            )
            unsigned = dict(hostile)
            unsigned.pop("manifest_digest")
            hostile["manifest_digest"] = overlay.object_sha256(unsigned)
            with self.assertRaisesRegex(overlay.OverlayError, "closure"):
                overlay.validate_manifest(hostile)

            archive = root / "hostile.tar"
            payload = overlay.build_archive_bytes(method, manifest)
            archive.write_bytes(payload)
            with tarfile.open(archive, "a", format=tarfile.USTAR_FORMAT) as handle:
                info = tarfile.TarInfo("extra.py")
                info.size = 1
                handle.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(overlay.OverlayError, "closure"):
                overlay.validate_archive(archive.resolve(), manifest)

    def test_source_binding_accepts_only_declared_runtime_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            method = self._make_method_root(source)
            (source / "unchanged.txt").write_text("sealed\n", encoding="ascii")
            (method / overlay.ADDED_FILES[0]).unlink()
            overlay_archive, manifest_path = root / "overlay.tar", root / "overlay.json"
            base_root = root / "base"
            runtime_root = root / "runtime"
            shutil.copytree(source, base_root)
            training_archive = root / "training.tar"
            self._make_training_archive(base_root, training_archive)
            for index, name in enumerate(overlay.OVERLAY_FILES):
                (method / name).write_bytes(
                    f"overlay-runtime-{index}\n".encode("ascii")
                )
            built = overlay.build(method, overlay_archive, manifest_path)
            shutil.copytree(base_root, runtime_root)
            with tarfile.open(overlay_archive, "r:") as archive:
                archive.extractall(runtime_root)
            # Binder's production pin is immutable.  A fixture archive is
            # substituted only inside this local unit test.
            prior = binder.TRAINING_ARCHIVE_SHA256
            try:
                binder.TRAINING_ARCHIVE_SHA256 = _sha(training_archive)
                result = binder.verify_binding(
                    training_archive=training_archive.resolve(),
                    overlay_archive=overlay_archive.resolve(),
                    manifest_path=manifest_path.resolve(),
                    base_root=base_root.resolve(),
                    runtime_root=runtime_root.resolve(),
                    expected_overlay_archive_sha256=built["archive_sha256"],
                    expected_overlay_manifest_sha256=_sha(manifest_path),
                )
            finally:
                binder.TRAINING_ARCHIVE_SHA256 = prior
            self.assertFalse(
                result["inference_runtime_overlay"][
                    "passed_as_training_method_archive"
                ]
            )
            self.assertTrue(
                result["application"]["all_non_overlay_files_byte_exact"]
            )
            self.assertEqual(
                result["application"]["added_paths"],
                ["methods/bernini_action_editing/infer_seer_same_state_full160_lora.py"],
            )
            self.assertEqual(len(result["application"]["replaced_paths"]), 2)
            self.assertEqual(result["application"]["removed_paths"], [])

    def test_source_binding_rejects_undeclared_change_and_archive_impersonation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            method = self._make_method_root(source)
            (source / "unchanged.txt").write_text("sealed\n", encoding="ascii")
            overlay_archive, manifest_path = root / "overlay.tar", root / "overlay.json"
            built = overlay.build(method, overlay_archive, manifest_path)
            base_root = root / "base"
            runtime_root = root / "runtime"
            shutil.copytree(source, base_root)
            shutil.copytree(source, runtime_root)
            (runtime_root / "unchanged.txt").write_text("tampered\n", encoding="ascii")
            training_archive = root / "training.tar"
            self._make_training_archive(source, training_archive)
            prior = binder.TRAINING_ARCHIVE_SHA256
            try:
                binder.TRAINING_ARCHIVE_SHA256 = _sha(training_archive)
                with self.assertRaisesRegex(
                    binder.SourceBindingError, "added/replaced/removed"
                ):
                    binder.verify_binding(
                        training_archive=training_archive.resolve(),
                        overlay_archive=overlay_archive.resolve(),
                        manifest_path=manifest_path.resolve(),
                        base_root=base_root.resolve(),
                        runtime_root=runtime_root.resolve(),
                        expected_overlay_archive_sha256=built["archive_sha256"],
                        expected_overlay_manifest_sha256=_sha(manifest_path),
                    )
                binder.TRAINING_ARCHIVE_SHA256 = built["archive_sha256"]
                with self.assertRaisesRegex(
                    binder.SourceBindingError, "training method source archive"
                ):
                    binder.verify_binding(
                        training_archive=training_archive.resolve(),
                        overlay_archive=overlay_archive.resolve(),
                        manifest_path=manifest_path.resolve(),
                        base_root=base_root.resolve(),
                        runtime_root=runtime_root.resolve(),
                        expected_overlay_archive_sha256=built["archive_sha256"],
                        expected_overlay_manifest_sha256=_sha(manifest_path),
                    )
            finally:
                binder.TRAINING_ARCHIVE_SHA256 = prior

    def test_launcher_reuses_exclusive_256g_core4_and_preserves_training_pin(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for token in (
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:4",
            "#SBATCH --array=0-1%2",
            "#SBATCH --exclusive",
            "checkpoint-00000160",
            "infer_seer_same_state_full160_lora.py",
            "global_step\"] == 160",
            "max_steps\"] == 160",
            "bind_seer_full160_eval_source_v1.py",
            "tools/build_seer_full160_eval_overlay_v1.py",
            "SEER_FULL160_OVERLAY_BUILDER_SHA256",
            "--method-source-archive-sha256 \"${training_archive_sha}\"",
            "--method-source-revision \"${training_revision}\"",
        ):
            self.assertIn(token, text)
        self.assertIn(overlay.TRAINING_METHOD_SOURCE_ARCHIVE_SHA256, text)
        self.assertIn(overlay.TRAINING_METHOD_SOURCE_REVISION, text)
        self.assertIn('[[ "${overlay_archive_sha}" != "${training_archive_sha}" ]]', text)
        self.assertNotIn(
            '--method-source-archive-sha256 "${overlay_archive_sha}"', text
        )
        pre_extract = text.split(
            'tar --no-same-owner --no-same-permissions -xf "${overlay_archive}"',
            1,
        )[0]
        self.assertNotIn(
            'chmod u+w "${method_root}/infer_seer_same_state_full160_lora.py"',
            pre_extract,
        )
        self.assertLess(
            text.index("run_arm frozen_base 29611"),
            text.index("run_arm trained_adapter 29612"),
        )

    def test_flat_stage_binder_cli_imports_only_pinned_stage_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary).resolve()
            (stage / "tools").mkdir()
            staged_binder = stage / "bind_seer_full160_eval_source_v1.py"
            staged_builder = stage / "tools" / "build_seer_full160_eval_overlay_v1.py"
            shutil.copy2(Path(binder.__file__), staged_binder)
            shutil.copy2(Path(overlay.__file__), staged_builder)
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(staged_binder), "--help"],
                cwd=stage,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("finalize-case", result.stdout)
            staged_builder.unlink()
            failed = subprocess.run(
                [sys.executable, "-I", "-B", str(staged_binder), "--help"],
                cwd=stage,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("ModuleNotFoundError", failed.stderr)

    def test_case_binding_closes_outputs_receipts_adapter_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, pair, _, trained_video = self._make_case_evidence(root)
            unsigned = binder.finalize_case_binding(
                source_binding_path=source,
                pair_receipt_path=pair,
            )
            output = root / "eval-execution-binding.json"
            _write_receipt(output, unsigned)
            checked = binder.verify_case_binding_receipt(output)
            self.assertEqual(checked["trained_adapter"]["training_global_step"], 160)
            self.assertEqual(checked["trained_adapter"]["adapter_tensor_count"], 120)
            self.assertFalse(checked["method_success_authorized"])
            trained_video.write_bytes(b"tampered-after-binding")
            with self.assertRaisesRegex(
                binder.SourceBindingError, "trained_adapter output SHA"
            ):
                binder.verify_case_binding_receipt(output)

    def test_case_binding_rejects_overlay_sha_as_inference_training_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, pair, base_video, _ = self._make_case_evidence(root)
            inference_path = Path(str(base_video) + ".receipt.json")
            value = json.loads(inference_path.read_text(encoding="utf-8"))
            value.pop("receipt_digest")
            value["method_source_archive_sha256"] = "a" * 64
            _write_receipt(inference_path, value)
            with self.assertRaisesRegex(
                binder.SourceBindingError, "inference/pair cross-bind"
            ):
                binder.finalize_case_binding(
                    source_binding_path=source,
                    pair_receipt_path=pair,
                )

    def test_identical_base_and_trained_output_is_valid_zero_effect_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, pair, base_video, trained_video = self._make_case_evidence(root)
            trained_video.write_bytes(base_video.read_bytes())
            trained_receipt_path = Path(str(trained_video) + ".receipt.json")
            trained = json.loads(trained_receipt_path.read_text(encoding="utf-8"))
            trained.pop("receipt_digest")
            trained["output"]["sha256"] = _sha(trained_video)
            trained_signed = _write_receipt(trained_receipt_path, trained)
            pair_value = json.loads(pair.read_text(encoding="utf-8"))
            pair_value.pop("receipt_digest")
            pair_value["trained_adapter"]["output"]["sha256"] = _sha(trained_video)
            pair_value["trained_adapter"]["inference_receipt_digest"] = (
                trained_signed["receipt_digest"]
            )
            _write_receipt(pair, pair_value)
            unsigned = binder.finalize_case_binding(
                source_binding_path=source,
                pair_receipt_path=pair,
            )
            self.assertTrue(unsigned["decoded_outputs_byte_identical"])
            self.assertFalse(unsigned["method_success_claimed"])
            self.assertFalse(unsigned["method_success_authorized"])
            output = root / "identical-eval-binding.json"
            _write_receipt(output, unsigned)
            checked = binder.verify_case_binding_receipt(output)
            self.assertTrue(checked["decoded_outputs_byte_identical"])
            hostile = json.loads(output.read_text(encoding="utf-8"))
            hostile.pop("receipt_digest")
            hostile["decoded_outputs_byte_identical"] = False
            _write_receipt(output, hostile)
            with self.assertRaisesRegex(
                binder.SourceBindingError, "identity claim"
            ):
                binder.verify_case_binding_receipt(output)


if __name__ == "__main__":
    unittest.main()
