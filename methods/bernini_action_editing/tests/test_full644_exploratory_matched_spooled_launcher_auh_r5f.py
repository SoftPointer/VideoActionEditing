from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_spooled_launcher_auh_r5f as launcher


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R5FSpooledLauncherTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict[str, str], dict[str, str]]:
        source_root = root / "fresh-exact16"
        source_root.mkdir()
        dependency_names = (
            "full644_exploratory_matched_infer_adapter_auh_r5f.py",
            "full644_exploratory_matched_infer_adapter_v2.py",
            "action_preservation_decoded_eval_model_authority_v2.py",
            "infer_lora.py",
            "train_lora.py",
            "self_generated_action_preservation_v2.py",
        )
        for basename in dependency_names:
            target = source_root / basename
            shutil.copyfile(MODULE_ROOT / basename, target)
            target.chmod(0o444)
        (source_root / "tools").mkdir()
        for relative in (
            "tools/build_renderer_dataset.py",
            "tools/materialize_vae.py",
        ):
            target = source_root / relative
            shutil.copyfile(MODULE_ROOT / relative, target)
            target.chmod(0o444)

        files: dict[str, str] = {}
        expected: dict[str, str] = {}
        for index, role in enumerate(launcher.EXPECTED_STATIC_SHA256):
            if role == "adapter":
                path = source_root / (
                    "full644_exploratory_matched_infer_adapter_auh_r5f.py"
                )
            elif role == "base_adapter":
                path = source_root / (
                    "full644_exploratory_matched_infer_adapter_v2.py"
                )
            elif role == "model_authority":
                path = source_root / (
                    "action_preservation_decoded_eval_model_authority_v2.py"
                )
            else:
                path = (root / f"{role}.bin").resolve()
                path.write_bytes(f"r5f-{role}-{index}\n".encode("utf-8"))
                path.chmod(0o444)
            files[role] = str(path)
            expected[role] = _sha(path)

        python_path = (root / "python").resolve()
        shutil.copyfile(Path(sys.executable).resolve(strict=True), python_path)
        python_path.chmod(0o555)
        ffmpeg_path = (root / "ffmpeg").resolve()
        ffmpeg_path.write_bytes(b"r5f fixture ffmpeg\n")
        ffmpeg_path.chmod(0o555)
        plan_path = (root / "plan.json").resolve()
        plan_path.write_bytes(b'{"r5f_fixture":true}\n')
        plan_path.chmod(0o444)
        for name in ("model", "bernini", "veomni"):
            (root / name).mkdir()

        value = {
            "schema_version": launcher.INPUT_SCHEMA,
            "entry_mode": "trusted_stdin",
            **files,
            "python": str(python_path),
            "ffmpeg": str(ffmpeg_path),
            "plan": str(plan_path),
            "output_report": str((root / "report.json").resolve()),
            "runner_attestation": str((root / "attestation.json").resolve()),
            "model_root": str((root / "model").resolve()),
            "bernini_root": str((root / "bernini").resolve()),
            "veomni_root": str((root / "veomni").resolve()),
            "authority_root": str((root / "authority").resolve()),
            "rank_cache_root": str((root / "rank-cache").resolve()),
            "holder_job_id": "143812",
            "expected_node": "auh7-1b-gpu-293",
            "campaign_mode": launcher.CASE00_CANARY_CAMPAIGN,
        }
        return value, expected

    def test_exact_source_pin_cascade_uses_only_new_adapter_bytes(self) -> None:
        sources = {
            "runner": (
                "full644_exploratory_matched_runner_auh_r5.py",
                launcher.RUNNER_SHA256,
            ),
            "bridge": (
                "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
                launcher.BRIDGE_SHA256,
            ),
            "adapter": (
                "full644_exploratory_matched_infer_adapter_auh_r5f.py",
                launcher.R5F_ADAPTER_SHA256,
            ),
            "base_adapter": (
                "full644_exploratory_matched_infer_adapter_v2.py",
                launcher.BASE_ADAPTER_SHA256,
            ),
            "model_authority": (
                "action_preservation_decoded_eval_model_authority_v2.py",
                launcher.MODEL_AUTHORITY_SHA256,
            ),
            "base_launcher": (
                "full644_exploratory_matched_spooled_launcher_auh_r5.py",
                launcher.BASE_LAUNCHER_SHA256,
            ),
        }
        for label, (basename, expected) in sources.items():
            with self.subTest(label=label):
                self.assertEqual(_sha(MODULE_ROOT / basename), expected)
        self.assertEqual(
            launcher.EXPECTED_STATIC_SHA256["adapter"],
            launcher.R5F_ADAPTER_SHA256,
        )
        self.assertEqual(
            launcher.EXPECTED_STATIC_SHA256["runner"], launcher.RUNNER_SHA256
        )
        self.assertEqual(
            launcher.EXPECTED_STATIC_SHA256["bridge"], launcher.BRIDGE_SHA256
        )
        self.assertEqual(
            launcher.EXPECTED_STATIC_SHA256["base_adapter"],
            launcher.BASE_ADAPTER_SHA256,
        )
        self.assertNotEqual(
            launcher.R5F_ADAPTER_SHA256, launcher.BASE_ADAPTER_SHA256
        )

    def test_schema_transform_is_exact_and_leaves_r5c_module_on_disk(self) -> None:
        self.assertEqual(
            launcher.SCHEMA,
            "full644-exploratory-matched-root-launch-release-auh-r5f",
        )
        self.assertEqual(
            launcher.INPUT_SCHEMA,
            "full644-exploratory-matched-root-launch-input-auh-r5f",
        )
        self.assertEqual(
            launcher.RECEIPT_SCHEMA,
            "full644-exploratory-matched-root-launch-receipt-auh-r5f",
        )
        self.assertEqual(launcher.ROOT_BOOTSTRAP.count(launcher.SCHEMA), 1)
        self.assertNotIn("root-launch-release-auh-r5c", launcher.ROOT_BOOTSTRAP)
        self.assertIn("r5f exact16 release identity closure differs", launcher.ROOT_BOOTSTRAP)
        compile(launcher.ROOT_BOOTSTRAP, "<r5f-root-bootstrap>", "exec")
        self.assertEqual(
            _sha(MODULE_ROOT / "full644_exploratory_matched_spooled_launcher_auh_r5.py"),
            launcher.BASE_LAUNCHER_SHA256,
        )

    def test_release_binds_launcher_to_runner_to_bridge_to_r5f_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            self.assertEqual(
                expected["adapter"], launcher.R5F_ADAPTER_SHA256
            )
            with mock.patch.object(
                launcher.base, "EXPECTED_STATIC_SHA256", expected
            ):
                release, payload = launcher.build_release(value)
            self.assertEqual(release["schema_version"], launcher.SCHEMA)
            self.assertEqual(
                release["identities"]["runner"]["sha256"],
                expected["runner"],
            )
            self.assertEqual(
                release["identities"]["bridge"]["sha256"],
                expected["bridge"],
            )
            self.assertEqual(
                release["identities"]["adapter"]["sha256"],
                launcher.R5F_ADAPTER_SHA256,
            )
            self.assertEqual(len(release["identities"]), 16)
            self.assertEqual(
                release["identities"]["base_adapter"]["sha256"],
                launcher.BASE_ADAPTER_SHA256,
            )
            self.assertEqual(
                Path(release["identities"]["base_adapter"]["path"]),
                Path(release["identities"]["adapter"]["path"]).parent
                / "full644_exploratory_matched_infer_adapter_v2.py",
            )
            arguments = release["runner_arguments"]
            self.assertEqual(
                arguments[arguments.index("--bridge-script-sha256") + 1],
                expected["bridge"],
            )
            self.assertEqual(
                arguments[arguments.index("--adapter-script-sha256") + 1],
                launcher.R5F_ADAPTER_SHA256,
            )
            self.assertNotIn(value["base_adapter"], arguments)
            checked = subprocess.run(
                ["/bin/bash", "-n"],
                input=payload,
                check=False,
                capture_output=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_old_wrapper_replacement_is_rejected_by_fresh_r5f_pin(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            expected["adapter"] = launcher.R5F_ADAPTER_SHA256
            adapter_path = Path(value["adapter"])
            adapter_path.chmod(0o600)
            adapter_path.write_bytes(
                (MODULE_ROOT / "full644_exploratory_matched_infer_adapter_v2.py").read_bytes()
            )
            adapter_path.chmod(0o444)
            self.assertEqual(_sha(adapter_path), launcher.BASE_ADAPTER_SHA256)
            with mock.patch.object(
                launcher.base, "EXPECTED_STATIC_SHA256", expected
            ), self.assertRaisesRegex(
                launcher.RootLaunchReleaseError, "adapter.*SHA"
            ):
                launcher.build_release(value)

    def test_missing_wrong_name_wrong_directory_and_wrong_base_bytes_fail(self) -> None:
        hostile_kinds = ("missing", "wrong_name", "wrong_directory", "wrong_bytes")
        for kind in hostile_kinds:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as raw:
                root = Path(raw).resolve(strict=True)
                value, expected = self.fixture(root)
                original = Path(value["base_adapter"])
                if kind == "missing":
                    original.unlink()
                elif kind == "wrong_name":
                    replacement = original.with_name("not-the-frozen-base.py")
                    replacement.write_bytes(original.read_bytes())
                    replacement.chmod(0o444)
                    value["base_adapter"] = str(replacement)
                elif kind == "wrong_directory":
                    other = root / "other"
                    other.mkdir()
                    replacement = other / original.name
                    replacement.write_bytes(original.read_bytes())
                    replacement.chmod(0o444)
                    value["base_adapter"] = str(replacement)
                else:
                    original.chmod(0o600)
                    original.write_bytes(b"hostile frozen-base replacement\n")
                    original.chmod(0o444)
                expected_error = (
                    FileNotFoundError
                    if kind == "missing"
                    else launcher.RootLaunchReleaseError
                )
                expected_text = (
                    "No such file"
                    if kind == "missing"
                    else "adjacency differs"
                    if kind in {"wrong_name", "wrong_directory"}
                    else "base_adapter SHA differs"
                )
                with mock.patch.object(
                    launcher.base, "EXPECTED_STATIC_SHA256", expected
                ), self.assertRaisesRegex(expected_error, expected_text):
                    launcher.build_release(value)

    def test_strict_input_requires_the_independent_base_adapter_field(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            value.pop("base_adapter")
            launch_input = root / "launch-input.json"
            launch_input.write_bytes(
                launcher.canonical_json_bytes(value) + b"\n"
            )
            launch_input.chmod(0o444)
            with mock.patch.object(
                launcher.base, "EXPECTED_STATIC_SHA256", expected
            ), self.assertRaisesRegex(
                launcher.RootLaunchReleaseError, "input closure"
            ):
                launcher.materialize(
                    str(launch_input),
                    str(root / "payload.sh"),
                    str(root / "receipt.json"),
                )

    def test_materialize_emits_only_r5f_schemas_and_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            launch_input = root / "launch-input.json"
            launch_input.write_bytes(
                launcher.canonical_json_bytes(value) + b"\n"
            )
            launch_input.chmod(0o444)
            payload = root / "payload.sh"
            receipt = root / "payload.receipt.json"
            with mock.patch.object(
                launcher.base, "EXPECTED_STATIC_SHA256", expected
            ):
                observed = launcher.materialize(
                    str(launch_input), str(payload), str(receipt)
                )
                with self.assertRaises(launcher.RootLaunchReleaseError):
                    launcher.materialize(
                        str(launch_input), str(payload), str(receipt)
                    )
            self.assertEqual(observed["schema_version"], launcher.RECEIPT_SCHEMA)
            self.assertEqual(
                observed["release"]["schema_version"], launcher.SCHEMA
            )
            self.assertEqual(stat.S_IMODE(payload.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
            self.assertNotIn(b"root-launch-release-auh-r5c", payload.read_bytes())

            runtime = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    "import pathlib,sys; "
                    f"sys.path.insert(0,{str(Path(value['adapter']).parent)!r}); "
                    "import full644_exploratory_matched_infer_adapter_auh_r5f as r5f; "
                    "print(pathlib.Path(r5f.base.__file__).name); "
                    "print(r5f.BASE_ADAPTER_SHA256)",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(runtime.returncode, 0, runtime.stderr)
            self.assertEqual(
                runtime.stdout.splitlines(),
                [
                    "full644_exploratory_matched_infer_adapter_v2.py",
                    launcher.BASE_ADAPTER_SHA256,
                ],
            )

    def test_r5c_input_schema_is_rejected_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            value["schema_version"] = (
                "full644-exploratory-matched-root-launch-input-auh-r5c"
            )
            launch_input = root / "launch-input.json"
            launch_input.write_bytes(
                launcher.canonical_json_bytes(value) + b"\n"
            )
            launch_input.chmod(0o444)
            with mock.patch.object(
                launcher.base, "EXPECTED_STATIC_SHA256", expected
            ), self.assertRaisesRegex(
                launcher.RootLaunchReleaseError, "input closure"
            ):
                launcher.materialize(
                    str(launch_input),
                    str(root / "payload.sh"),
                    str(root / "receipt.json"),
                )


if __name__ == "__main__":
    unittest.main()
