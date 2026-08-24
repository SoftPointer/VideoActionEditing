#!/usr/bin/env python3
"""Static boundary tests for the fresh exact5 GPU-v4 HOLD controller."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = ROOT / "methods/bernini_action_editing"
V3_HOLD = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v3.HOLD.py"
)
V3_READY = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v3.READY.py"
)
V4_HOLD = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v4.HOLD.py"
)
V4_READY = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v4.READY.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = load(V4_HOLD, "_case01_gpu_v4_static")


class GPUV4StaticTests(unittest.TestCase):
    def test_v3_authorities_are_unchanged_and_no_v4_ready_exists(self):
        self.assertEqual(
            hashlib.sha256(V3_HOLD.read_bytes()).hexdigest(),
            "72b273b9de26caf37dca2a7c183a52152deed5c78b7ce1b0297855fc37e57022",
        )
        self.assertEqual(
            hashlib.sha256(V3_READY.read_bytes()).hexdigest(),
            "f1f2ca5b780fc412d5aafbbd913459e35e512aad78a3a6bfabfe1b0869b977a7",
        )
        self.assertFalse(V4_READY.exists())
        self.assertEqual(
            controller.CONTROLLER_STATE,
            "HOLD_PENDING_FRESH_V4_COMPOSITE_CPU_PINS",
        )

    def test_fresh_producer_and_output_namespace_is_exact(self):
        self.assertEqual(
            controller.PACKAGE_ROOT.name,
            "bernini_case01_object_trajectory_exact5_r64_canary_v3",
        )
        self.assertEqual(
            controller.SOURCE_OVERLAY_ROOT.name,
            "bernini_case01_object_trajectory_exact5_"
            "r5f_v4_source_overlay_6_20260822_r1",
        )
        self.assertEqual(controller.PACKAGE_PUBLICATION_RECEIPT_PATH.name,
                         controller.PACKAGE_ROOT.name + ".publication_receipt_v4.json")
        self.assertEqual(controller.MATERIALIZATION_REPORT_PATH.name,
                         "package_materialization_receipt_v4.json")
        self.assertEqual(controller.PACKAGE_CONTROLLER_EVIDENCE_PATH.name,
                         controller.PACKAGE_ROOT.name +
                         ".materialize_controller_evidence_v3.json")
        self.assertEqual(controller.COMPOSITE_CPU_RECEIPT_PATH.name,
                         controller.PACKAGE_ROOT.name +
                         ".composite_cpu_admission_receipt_v2.json")
        self.assertEqual(controller.COMPOSITE_CPU_EVIDENCE_PATH.name,
                         controller.PACKAGE_ROOT.name +
                         ".composite_cpu_admission_controller_evidence_v2.json")
        self.assertIn("node292-r3-rank-cache", str(controller.RANK_CACHE_ROOT))
        custom_paths = (
            controller.RUNTIME_RECEIPT_PATH,
            controller.RANK_CACHE_RECEIPT_PATH,
            controller.ATTEMPT_PATH,
            controller.DISPATCH_PATH,
            controller.EVIDENCE_PATH,
            controller.STDOUT_PATH,
            controller.STDERR_PATH,
        )
        self.assertEqual(len(set(custom_paths)), len(custom_paths))
        self.assertTrue(all("v4" in path.name for path in custom_paths))

    def test_scientific_abi_stays_v3_while_custom_runtime_is_v4(self):
        self.assertEqual(controller.READY_PLAN_SCHEMA,
                         "case01-object-trajectory-exact5-plan-v3")
        self.assertEqual(controller.REPORT_SCHEMA,
                         "case01-object-trajectory-exact5-report-v3")
        self.assertEqual(controller.RUNNER_SCHEMA,
                         "case01-object-trajectory-exact5-runner-attestation-v3")
        self.assertEqual(controller.RUNTIME_SCHEMA,
                         "case01-object-trajectory-exact5-r64-gpu-controller-v4-runtime")
        self.assertNotIn("gpu-controller-v3", controller.ROOT_BOOTSTRAP)
        self.assertNotIn("gpu-release-v3", controller.ROOT_BOOTSTRAP)
        self.assertIn("gpu-controller-v4", controller.ROOT_BOOTSTRAP)
        self.assertIn("gpu-release-v4", controller.ROOT_BOOTSTRAP)

    def test_core4_and_activation_import_join_are_exact(self):
        self.assertEqual(len(controller.CORE4_RELEASE_PINS), 5)
        self.assertEqual(
            controller.CORE4_RELEASE_PINS[
                "release/methods/bernini_action_editing/"
                "full644_exploratory_matched_infer_adapter_v3.py"
            ],
            (
                "7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120",
                124_612,
            ),
        )
        activation = controller.expected_composite_cpu_activation_import()
        self.assertEqual(activation["module"], "bernini.pipeline")
        self.assertIs(activation["finder_installed_before_callback"], True)
        self.assertEqual(activation["finder_count_per_rank"], [1, 1, 1, 1])
        self.assertEqual(activation["loader_type"], "_CapturedVendorLoader")
        self.assertIs(activation["loader_is_spec_loader"], True)
        self.assertIs(activation["cached_is_none"], True)
        self.assertEqual(activation["base_adapter_path"],
                         str(controller.BASE_ADAPTER_PATH))
        self.assertEqual(activation["base_adapter_sha256"],
                         controller.BASE_ADAPTER_SHA256)

    def test_only_fresh_cpu_authority_pins_are_blocked(self):
        self.assertEqual(
            set(controller.blocked_dynamic_pins()),
            {
                "composite_cpu_receipt_sha256",
                "composite_cpu_receipt_size",
                "composite_cpu_receipt_digest",
                "composite_cpu_evidence_sha256",
                "composite_cpu_evidence_size",
                "composite_cpu_evidence_digest",
            },
        )
        self.assertTrue(all(
            str(controller.dynamic_pin_values()[name]).startswith("BLOCKED_")
            for name in controller.blocked_dynamic_pins()
        ))

    def test_source_and_embedded_payloads_compile_without_asserts(self):
        source = V4_HOLD.read_text(encoding="utf-8")
        tree = ast.parse(source, str(V4_HOLD))
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        for optimize in (0, 1, 2):
            compile(source, str(V4_HOLD), "exec", optimize=optimize)
            compile(controller.ROOT_BOOTSTRAP, "<gpu-v4-root>", "exec",
                    optimize=optimize)
        for symbol in (
            "validate_srun_transport", "_terminate_process_group",
            "validate_internal_artifact_bindings", "validate_postflight",
            "validate_compute_package_root_identity",
        ):
            self.assertTrue(callable(getattr(controller, symbol)))


if __name__ == "__main__":
    unittest.main()
