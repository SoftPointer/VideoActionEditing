#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "materialize_flow_counterfactual_controls_v1.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "materialize_flow_counterfactual_controls_v1_test", SUBJECT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


subject = _load_subject()

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    safe_open = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TemporalPermutationTests(unittest.TestCase):
    def test_shuffle_is_deterministic_nonidentity_permutation(self) -> None:
        first = subject._temporal_permutation("shuffle", 20, 20260824)
        second = subject._temporal_permutation("shuffle", 20, 20260824)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(20)))
        self.assertNotEqual(first, list(range(20)))

    def test_shuffle_seed_changes_order(self) -> None:
        self.assertNotEqual(
            subject._temporal_permutation("shuffle", 20, 1),
            subject._temporal_permutation("shuffle", 20, 2),
        )

    def test_reverse_and_zero_permutations(self) -> None:
        self.assertEqual(
            subject._temporal_permutation("reverse", 4, None), [3, 2, 1, 0]
        )
        self.assertEqual(
            subject._temporal_permutation("zero", 4, None), [0, 1, 2, 3]
        )

    def test_seed_contract_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            subject.FlowCounterfactualControlError, "requires --seed"
        ):
            subject._temporal_permutation("shuffle", 4, None)
        with self.assertRaisesRegex(
            subject.FlowCounterfactualControlError, "only valid for shuffle"
        ):
            subject._temporal_permutation("reverse", 4, 7)

    def test_refuses_existing_outputs_before_tensor_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.safetensors"
            output_path = root / "output.safetensors"
            input_path.write_bytes(b"not needed because publication is refused first")
            output_path.write_bytes(b"sentinel")
            with self.assertRaisesRegex(
                subject.FlowCounterfactualControlError, "refusing to overwrite"
            ):
                subject.materialize(input_path, output_path, mode="zero")
            self.assertEqual(output_path.read_bytes(), b"sentinel")


@unittest.skipUnless(
    torch is not None and safe_open is not None and save_file is not None,
    "PyTorch/safetensors are unavailable in this local workspace",
)
class TensorBundleIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None and save_file is not None
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "flow.safetensors"
        raw = torch.arange(4 * 2 * 2 * 3, dtype=torch.float32).reshape(4, 2, 2, 3)
        camera = raw + 1000.0
        validity = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
        validity = validity.expand(4, 1, 2, 3).contiguous()
        save_file(
            {
                "backward_raw": raw,
                "backward_camera_residual": camera,
                "validity": validity,
            },
            str(self.input_path),
            metadata={"origin": "extractor-unit-test"},
        )
        self.input_tensors = {
            "backward_raw": raw,
            "backward_camera_residual": camera,
            "validity": validity,
        }
        self.input_sidecar = {"schema_version": "upstream-test-v1", "sample": "p0"}
        self.input_path.with_suffix(".json").write_text(
            json.dumps(self.input_sidecar) + "\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load_output(self, path: Path):
        assert safe_open is not None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            metadata = dict(handle.metadata() or {})
        return tensors, metadata

    def test_zero_zeros_all_three_tensors_and_records_provenance(self) -> None:
        assert torch is not None
        output = self.root / "zero.safetensors"
        receipt = subject.materialize(self.input_path, output, mode="zero")
        tensors, metadata = self._load_output(output)
        self.assertEqual(set(tensors), set(subject.REQUIRED_TENSORS))
        for tensor in tensors.values():
            self.assertEqual(int(torch.count_nonzero(tensor).item()), 0)
        self.assertEqual(receipt["permutation"], [0, 1, 2, 3])
        self.assertEqual(receipt["input_sha256"], _file_sha256(self.input_path))
        self.assertEqual(receipt["output_sha256"], _file_sha256(output))
        self.assertEqual(receipt["tensor_shapes"]["backward_raw"], [4, 2, 2, 3])
        self.assertEqual(receipt["tensor_dtypes"]["validity"], "float32")
        self.assertEqual(
            receipt["input_sidecar"]["document"], self.input_sidecar
        )
        self.assertEqual(metadata["origin"], "extractor-unit-test")
        self.assertEqual(metadata["bernini_counterfactual_mode"], "zero")
        parsed_time = datetime.fromisoformat(
            receipt["created_at_utc"].replace("Z", "+00:00")
        )
        self.assertIsNotNone(parsed_time.tzinfo)
        self.assertEqual(
            json.loads(output.with_suffix(".json").read_text(encoding="utf-8")),
            receipt,
        )

    def test_reverse_uses_one_consistent_time_permutation(self) -> None:
        assert torch is not None
        output = self.root / "reverse.safetensors"
        receipt = subject.materialize(self.input_path, output, mode="reverse")
        tensors, _ = self._load_output(output)
        permutation = receipt["permutation"]
        self.assertEqual(permutation, [3, 2, 1, 0])
        for name in subject.REQUIRED_TENSORS:
            self.assertTrue(
                torch.equal(tensors[name], self.input_tensors[name][permutation])
            )

    def test_shuffle_is_repeatable_and_consistent_across_tensors(self) -> None:
        assert torch is not None
        first = self.root / "shuffle-a.safetensors"
        second = self.root / "shuffle-b.safetensors"
        first_receipt = subject.materialize(
            self.input_path, first, mode="shuffle", seed=20260824
        )
        second_receipt = subject.materialize(
            self.input_path, second, mode="shuffle", seed=20260824
        )
        self.assertEqual(
            first_receipt["permutation"], second_receipt["permutation"]
        )
        first_tensors, _ = self._load_output(first)
        second_tensors, _ = self._load_output(second)
        permutation = first_receipt["permutation"]
        for name in subject.REQUIRED_TENSORS:
            self.assertTrue(torch.equal(first_tensors[name], second_tensors[name]))
            self.assertTrue(
                torch.equal(first_tensors[name], self.input_tensors[name][permutation])
            )

    def test_rejects_missing_key_bad_geometry_and_nonfinite_values(self) -> None:
        assert torch is not None and save_file is not None
        cases = {
            "missing": {
                "backward_raw": torch.zeros(4, 2, 2, 3),
                "backward_camera_residual": torch.zeros(4, 2, 2, 3),
            },
            "geometry": {
                "backward_raw": torch.zeros(4, 2, 2, 3),
                "backward_camera_residual": torch.zeros(3, 2, 2, 3),
                "validity": torch.zeros(4, 1, 2, 3),
            },
            "nonfinite": {
                "backward_raw": torch.full((4, 2, 2, 3), float("nan")),
                "backward_camera_residual": torch.zeros(4, 2, 2, 3),
                "validity": torch.zeros(4, 1, 2, 3),
            },
        }
        for label, tensors in cases.items():
            with self.subTest(label=label):
                input_path = self.root / f"bad-{label}.safetensors"
                save_file(tensors, str(input_path))
                with self.assertRaises(subject.FlowCounterfactualControlError):
                    subject.materialize(
                        input_path,
                        self.root / f"bad-{label}-out.safetensors",
                        mode="zero",
                    )

    def test_refuses_existing_sidecar_without_publishing_bundle(self) -> None:
        output = self.root / "blocked.safetensors"
        sidecar = output.with_suffix(".json")
        sidecar.write_bytes(b"sentinel")
        with self.assertRaisesRegex(
            subject.FlowCounterfactualControlError, "refusing to overwrite"
        ):
            subject.materialize(self.input_path, output, mode="reverse")
        self.assertFalse(output.exists())
        self.assertEqual(sidecar.read_bytes(), b"sentinel")


if __name__ == "__main__":
    unittest.main()
