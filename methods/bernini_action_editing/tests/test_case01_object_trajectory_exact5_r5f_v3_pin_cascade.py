from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
COMPOSITE = METHOD_ROOT / "infer_case01_object_trajectory_oracle_auh_r5f_v3.py"
EVAL = METHOD_ROOT / "case01_object_trajectory_exact5_eval_v3.py"
RUNNER = METHOD_ROOT / "case01_object_trajectory_exact5_runner_v3.py"
INNER = METHOD_ROOT / "infer_case01_object_trajectory_oracle_v1.py"
SEALED_METHOD_FIXTURE = Path(
    "/tmp/case01_object_trajectory_v1_sealed_methods_fixture"
)
PINS = {
    COMPOSITE.name: (
        "b30bba5c9cd233d412ffd88d8413311e9ffbb79d3ddf69aaf6eb2ee96183b489",
        42_184,
    ),
    EVAL.name: (
        "cfdfc5fec04243265b6c122649fed9144d89510d17184a77782c0ec0ddc5ed8a",
        116_374,
    ),
    RUNNER.name: (
        "02207e64a129444b26adf8bd92307102c4a91e85d2a029fa60030a7e9e6f45c8",
        21_716,
    ),
    INNER.name: (
        "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9",
        74_281,
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class R5FV3PinCascadeTests(unittest.TestCase):
    def test_core3_and_inner_tuples_are_exact(self) -> None:
        for path in (COMPOSITE, EVAL, RUNNER, INNER):
            with self.subTest(path=path.name):
                expected_sha, expected_size = PINS[path.name]
                self.assertEqual(_sha(path), expected_sha)
                self.assertEqual(path.stat().st_size, expected_size)
                raw = path.read_bytes()
                compile(raw, str(path), "exec", optimize=0)
                compile(raw, str(path), "exec", optimize=1)
                compile(raw, str(path), "exec", optimize=2)

    def test_eval_requires_outer_and_independent_inner_producer_triples(self) -> None:
        evaluator = _load("_case01_r5f_v3_eval_pin_test", EVAL)
        self.addCleanup(sys.modules.pop, "_case01_r5f_v3_eval_pin_test", None)
        producer = evaluator.incomplete_producer()
        self.assertEqual(
            set(producer),
            {
                "inference_receipt_schemas",
                "infer_lora_path",
                "infer_lora_sha256",
                "infer_lora_size",
                "infer_lora_role",
                "inference_wrapper_path",
                "inference_wrapper_sha256",
                "inference_wrapper_size",
                "object_wrapper_inner_path",
                "object_wrapper_inner_sha256",
                "object_wrapper_inner_size",
                "trajectory_projection_module_path",
                "trajectory_projection_module_sha256",
                "trajectory_projection_module_size",
                "trajectory_scaffold_module_path",
                "trajectory_scaffold_module_sha256",
                "trajectory_scaffold_module_size",
                "ffprobe_path",
                "ffprobe_sha256",
                "ffprobe_size",
                "method_source_revision",
                "method_source_archive_sha256",
                "pins_complete",
            },
        )
        self.assertEqual(
            evaluator._expected_oracle_producer_hashes(producer),
            {
                "wrapper_source_sha256": PINS[COMPOSITE.name][0],
                "object_wrapper_inner_source_sha256": PINS[INNER.name][0],
                "legacy_infer_lora_source_sha256": (
                    "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
                ),
                "projection_source_sha256": (
                    "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e"
                ),
                "scaffold_source_sha256": (
                    "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a"
                ),
            },
        )
        missing = dict(producer)
        missing.pop("object_wrapper_inner_sha256")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError,
            "producer schema differs",
        ):
            evaluator._validate_producer(
                missing,
                require_complete=False,
                reopen=False,
            )
        tampered = dict(producer)
        tampered["object_wrapper_inner_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError,
            "frozen producer source pin differs",
        ):
            evaluator._validate_producer(
                tampered,
                require_complete=False,
                reopen=False,
            )

    def test_complete_producer_reopens_all_five_source_authorities(self) -> None:
        if not SEALED_METHOD_FIXTURE.is_dir():
            self.fail("sealed method fixture is required")
        evaluator = _load("_case01_r5f_v3_eval_reopen_test", EVAL)
        self.addCleanup(sys.modules.pop, "_case01_r5f_v3_eval_reopen_test", None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            shutil.copytree(SEALED_METHOD_FIXTURE, root, dirs_exist_ok=True)
            for source in (COMPOSITE, EVAL, RUNNER):
                shutil.copy2(source, root / source.name)
                (root / source.name).chmod(0o444)
            ffprobe = root / "ffprobe"
            ffprobe.write_bytes(b"sealed-ffprobe-fixture\n")
            ffprobe.chmod(0o444)
            producer = evaluator.incomplete_producer()
            producer.update(
                {
                    "infer_lora_path": str(
                        root / "infer_lora_full644_r5_frozen_acc46.py"
                    ),
                    "inference_wrapper_path": str(root / COMPOSITE.name),
                    "object_wrapper_inner_path": str(root / INNER.name),
                    "trajectory_projection_module_path": str(
                        root / "object_trajectory_projection_v1.py"
                    ),
                    "trajectory_scaffold_module_path": str(
                        root / "case01_oracle_object_trajectory_v1.py"
                    ),
                    "ffprobe_path": str(ffprobe),
                    "ffprobe_sha256": _sha(ffprobe),
                    "ffprobe_size": ffprobe.stat().st_size,
                    "method_source_revision": "r5f-v3-local-pin-test",
                    "method_source_archive_sha256": "f" * 64,
                    "pins_complete": True,
                }
            )
            validated = evaluator._validate_producer(
                producer,
                require_complete=True,
                reopen=True,
            )
            self.assertEqual(validated, producer)
            aliased_inner = copy.deepcopy(producer)
            aliased_inner["object_wrapper_inner_path"] = str(root / COMPOSITE.name)
            with self.assertRaisesRegex(
                evaluator.ObjectTrajectoryEvalError,
                "producer source pin differs|frozen producer source pin differs",
            ):
                evaluator._validate_producer(
                    aliased_inner,
                    require_complete=True,
                    reopen=True,
                )

    def test_runner_pins_v3_eval_and_records_inner_runtime_identity(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(f'"{PINS[EVAL.name][0]}"', source)
        self.assertIn('_EVAL_BASENAME = "case01_object_trajectory_exact5_eval_v3.py"', source)
        self.assertIn('"object_wrapper_inner": frozen._identity(', source)
        self.assertIn('producer["object_wrapper_inner_path"]', source)
        self.assertNotIn(
            '_EVAL_BASENAME = "case01_object_trajectory_exact5_eval_v1.py"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
