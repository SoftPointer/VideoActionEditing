from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = METHOD_ROOT / "materialize_elal3_simulator_c2_vae_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from safetensors.torch import save as save_safetensors
    import materialize_elal3_simulator_c2_vae_v1 as subject

    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    save_safetensors = None  # type: ignore[assignment]
    subject = None  # type: ignore[assignment]
    RUNTIME_AVAILABLE = False


class ELAL3SimulatorC2VAEStaticTests(unittest.TestCase):
    def test_source_parses_and_contains_fail_closed_publication_abi(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for fragment in (
            'SCHEMA_VERSION = "bernini-elal3-simulator-c2-exact16-latent-bundle-v1"',
            '"elal3_simulator_c2_label_v1": {',
            '"tools.materialize_vae": {',
            '"tools.build_renderer_dataset": {',
            'value.add_argument("--expected-materializer-source-sha256", required=True)',
            "def validate_runtime_sources(",
            "def validate_imported_model_modules(",
            "def verify_bundle_payload_v1(",
            "bundle_payload = save_safetensors(tensors, metadata=bundle_metadata)",
            "_write_exact_file(bundle_path, bundle_payload, mode=0o400)",
            '"self_pin_caller_supplied_not_standalone_authority": True',
            '"trainer_consumption_requires_external_release_pin": True',
            "control_post = load_control_closure(",
            "runtime_source_post = validate_runtime_sources(",
            "final_bundle_payload, final_bundle_binding = labels.stable_read_path(",
        ):
            self.assertIn(fragment, source)

    def test_no_path_based_safetensors_writer_remains(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "safetensors.torch"
            for alias in node.names
        }
        self.assertNotIn("save_file", imported_names)
        self.assertIn("save", imported_names)


@unittest.skipUnless(RUNTIME_AVAILABLE, "torch+safetensors runtime is required")
class ELAL3SimulatorC2VAEFunctionalTests(unittest.TestCase):
    def _small_exact16(self):
        shape = (1, 1, 1, 1, 1)
        tensors = {
            key: torch.full(shape, float(index), dtype=torch.float32)
            for index, key in enumerate(subject.TENSOR_ORDER)
        }
        rows = [
            {
                "tensor_key": key,
                "tensor_sha256": subject.tensor_sha256(tensors[key]),
                "shape": list(shape),
                "dtype": "torch.float32",
            }
            for key in subject.TENSOR_ORDER
        ]
        metadata = subject.expected_safetensors_metadata()
        return shape, tensors, rows, metadata

    def test_valid_serialized_exact16_bytes_reload(self) -> None:
        shape, tensors, rows, metadata = self._small_exact16()
        payload = save_safetensors(tensors, metadata=metadata)
        with mock.patch.object(subject, "EXPECTED_LATENT_SHAPE", shape):
            result = subject.verify_bundle_payload_v1(
                payload,
                expected_tensors=tensors,
                tensor_rows=rows,
                expected_metadata=metadata,
            )
        self.assertEqual(result["serialized_payload_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertTrue(result["exact16_keys_verified"])
        self.assertTrue(result["all_tensors_reloaded_from_serialized_bytes"])
        self.assertEqual(len(result["tensor_rows"]), 16)

    def test_wrong_published_tensor_and_metadata_fail_closed(self) -> None:
        shape, tensors, rows, metadata = self._small_exact16()
        changed = dict(tensors)
        first_key = subject.TENSOR_ORDER[0]
        changed[first_key] = tensors[first_key] + 1.0
        changed_payload = save_safetensors(changed, metadata=metadata)
        with mock.patch.object(subject, "EXPECTED_LATENT_SHAPE", shape):
            with self.assertRaisesRegex(
                subject.ELAL3SimulatorC2VAEError,
                "reloaded safetensors tensor differs",
            ):
                subject.verify_bundle_payload_v1(
                    changed_payload,
                    expected_tensors=tensors,
                    tensor_rows=rows,
                    expected_metadata=metadata,
                )
            wrong_metadata = dict(metadata)
            wrong_metadata["tensor_count"] = "15"
            with self.assertRaisesRegex(
                subject.ELAL3SimulatorC2VAEError,
                "published safetensors metadata differs",
            ):
                subject.verify_bundle_payload_v1(
                    save_safetensors(tensors, metadata=wrong_metadata),
                    expected_tensors=tensors,
                    tensor_rows=rows,
                    expected_metadata=metadata,
                )

    def test_late_control_swap_replay_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            subject.ELAL3SimulatorC2VAEError,
            "derivative authority/experiment contract changed",
        ):
            subject.require_identical_replay(
                {"derivative_payload": b"before", "contract_payload": b"fixed"},
                {"derivative_payload": b"after", "contract_payload": b"fixed"},
                label="derivative authority/experiment contract",
            )

    def test_stream_hash_exception_closes_all_held_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            path = root / "source.py"
            path.write_bytes(b"bound source\n")
            path.chmod(0o644)
            expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            original_fstat = subject.os.fstat
            original_close = subject.os.close
            fstat_calls = 0
            closed: list[int] = []

            def injected_fstat(descriptor: int):
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 4:
                    raise OSError("injected named replay failure")
                return original_fstat(descriptor)

            def tracked_close(descriptor: int):
                closed.append(descriptor)
                return original_close(descriptor)

            with mock.patch.object(subject.os, "fstat", side_effect=injected_fstat), mock.patch.object(
                subject.os, "close", side_effect=tracked_close
            ):
                with self.assertRaisesRegex(OSError, "injected named replay failure"):
                    subject.stable_stream_hash_path(
                        path.resolve(strict=True),
                        label="exception cleanup probe",
                        expected_sha256=expected_sha,
                        expected_size=len(b"bound source\n"),
                        expected_mode=0o644,
                        allowed_root=root,
                    )
            self.assertEqual(len(closed), 2)
            for descriptor in closed:
                with self.assertRaises(OSError):
                    original_fstat(descriptor)


if __name__ == "__main__":
    unittest.main()
