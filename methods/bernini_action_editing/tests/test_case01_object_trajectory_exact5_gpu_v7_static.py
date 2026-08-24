#!/usr/bin/env python3
"""Static boundary tests for the fresh exact5 GPU-v7 HOLD controller."""

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
V5_HOLD = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v5.HOLD.py"
)
V5_READY = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v5.READY.py"
)
V6_HOLD = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v6.HOLD.py"
)
V6_READY = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v6.READY.py"
)
V7_HOLD = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v7.HOLD.py"
)
V7_READY = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v7.READY.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4_controller = load(V4_HOLD, "_case01_gpu_v4_frozen_static")
v5_controller = load(V5_HOLD, "_case01_gpu_v5_frozen_static")
v6_controller = load(V6_HOLD, "_case01_gpu_v6_frozen_static")
controller = load(V7_HOLD, "_case01_gpu_v7_static")
ready_controller = load(V7_READY, "_case01_gpu_v7_ready_static")


class GPUV7StaticTests(unittest.TestCase):
    def test_old_authorities_unchanged_and_v7_ready_is_state_only(self):
        self.assertEqual(
            hashlib.sha256(V3_HOLD.read_bytes()).hexdigest(),
            "72b273b9de26caf37dca2a7c183a52152deed5c78b7ce1b0297855fc37e57022",
        )
        self.assertEqual(
            hashlib.sha256(V3_READY.read_bytes()).hexdigest(),
            "f1f2ca5b780fc412d5aafbbd913459e35e512aad78a3a6bfabfe1b0869b977a7",
        )
        self.assertEqual(
            hashlib.sha256(V4_HOLD.read_bytes()).hexdigest(),
            "2b383422dcc75988d822719263ce143876b1622471cddb76172f7eb4f0342085",
        )
        self.assertEqual(
            hashlib.sha256(V5_HOLD.read_bytes()).hexdigest(),
            "7569b29f07007883627422c6f8ad98d46d3b785840e40467d7d05b962043f02d",
        )
        self.assertFalse(V5_READY.exists())
        self.assertEqual(len(V6_HOLD.read_bytes()), 213_522)
        self.assertEqual(hashlib.sha256(V6_HOLD.read_bytes()).hexdigest(),
                         "63a0de7f6d0ff558057609210adefc02885b3c600257f2deb9264c5aeb797921")
        self.assertEqual(len(V6_READY.read_bytes()), 213_528)
        self.assertEqual(hashlib.sha256(V6_READY.read_bytes()).hexdigest(),
                         "f375c7baee22f73f88b2e6e62b3f48f721b7db8a2f2619ce2284c8b609f912bb")
        hold_raw = V7_HOLD.read_bytes()
        ready_raw = V7_READY.read_bytes()
        self.assertEqual(len(hold_raw), 213_891)
        self.assertEqual(hashlib.sha256(hold_raw).hexdigest(),
                         "05ed84ef610cba0f935a23f325612e3a744ce4925508eae658bfa32fba878040")
        self.assertEqual(len(ready_raw), 213_897)
        self.assertEqual(hashlib.sha256(ready_raw).hexdigest(),
                         "0688e3a41427d0217547093bf88c352e6a6fdd25eda8a5261ff6bbc2c83ad98a")
        hold_state = (
            b'CONTROLLER_STATE = "HOLD_PENDING_FRESH_V4_COMPOSITE_CPU_PINS"'
        )
        ready_state = (
            b'CONTROLLER_STATE = '
            b'"READY_EXPLICIT_SINGLE_SRUN_EXACT_FIVE_NO_RETRY"'
        )
        self.assertEqual(hold_raw.replace(hold_state, ready_state, 1), ready_raw)
        self.assertEqual(
            controller.CONTROLLER_STATE,
            "HOLD_PENDING_FRESH_V4_COMPOSITE_CPU_PINS",
        )
        self.assertEqual(ready_controller.CONTROLLER_STATE,
                         ready_controller.READY_STATE)
        self.assertEqual(
            ready_controller.authorization_token(),
            "bf97da2919e2cf41b8b0aab59a0ac171d32172b1e093575746b0440a43538772",
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
                         ".composite_cpu_admission_receipt_v4.json")
        self.assertEqual(controller.COMPOSITE_CPU_EVIDENCE_PATH.name,
                         controller.PACKAGE_ROOT.name +
                         ".composite_cpu_admission_controller_evidence_v4.json")
        self.assertIn("node292-r3-rank-cache", str(controller.RANK_CACHE_ROOT))
        outer_paths = (
            controller.ATTEMPT_PATH,
            controller.DISPATCH_PATH,
            controller.EVIDENCE_PATH,
            controller.STDOUT_PATH,
            controller.STDERR_PATH,
        )
        compute_paths = (
            controller.RUNTIME_RECEIPT_PATH,
            controller.RANK_CACHE_RECEIPT_PATH,
        )
        self.assertEqual(len(set(outer_paths + compute_paths)),
                         len(outer_paths + compute_paths))
        self.assertTrue(all("v7" in path.name for path in outer_paths))
        self.assertTrue(all("v4" in path.name for path in compute_paths))
        self.assertFalse(
            set(outer_paths) & set(controller.REVOKED_GPU_PRIOR_OUTER_PATHS)
        )
        self.assertTrue(all("v4" in path.name
                            for path in controller.REVOKED_GPU_V4_OUTER_PATHS))
        self.assertTrue(all("v5" in path.name
                            for path in controller.REVOKED_GPU_V5_OUTER_PATHS))
        self.assertTrue(all("v6" in path.name
                            for path in controller.REVOKED_GPU_V6_OUTER_PATHS))

    def test_scientific_abi_stays_v3_while_custom_runtime_is_v4(self):
        self.assertEqual(controller.READY_PLAN_SCHEMA,
                         "case01-object-trajectory-exact5-plan-v3")
        self.assertEqual(controller.REPORT_SCHEMA,
                         "case01-object-trajectory-exact5-report-v3")
        self.assertEqual(controller.RUNNER_SCHEMA,
                         "case01-object-trajectory-exact5-runner-attestation-v3")
        self.assertEqual(controller.RUNTIME_SCHEMA,
                         "case01-object-trajectory-exact5-r64-gpu-controller-v4-runtime")
        self.assertEqual(controller.SCHEMA,
                         "case01-object-trajectory-exact5-r64-gpu-controller-v7")
        self.assertEqual(controller.ROOT_BOOTSTRAP, v4_controller.ROOT_BOOTSTRAP)
        self.assertEqual(controller.ROOT_BOOTSTRAP, v5_controller.ROOT_BOOTSTRAP)
        self.assertEqual(controller.ROOT_BOOTSTRAP, v6_controller.ROOT_BOOTSTRAP)
        self.assertEqual(
            controller._WITHDRAWN_PRE_CACHE_AUDIT_ROOT_BOOTSTRAP,
            v4_controller._WITHDRAWN_PRE_CACHE_AUDIT_ROOT_BOOTSTRAP,
        )
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

    def test_fresh_cpu_authority_pins_are_fully_resolved(self):
        self.assertEqual(controller.blocked_dynamic_pins(), ())
        self.assertEqual(ready_controller.blocked_dynamic_pins(), ())
        pins = controller.dynamic_pin_values()
        self.assertEqual(pins["composite_cpu_receipt_sha256"],
                         "90950f14897ccbaeb8172bd0e6a166abd325a3d6c660d00e183c3d8bd546f09e")
        self.assertEqual(pins["composite_cpu_receipt_size"], 8_311)
        self.assertEqual(pins["composite_cpu_receipt_digest"],
                         "9c1192858123a445375e8d76d39012412d048db83b5a8a41157b1ae11625caab")
        self.assertEqual(pins["composite_cpu_evidence_sha256"],
                         "58ea147f14d4fdf62f34a8a7a67eab41535c1e9b7d953b3bb2c629d88cb6face")
        self.assertEqual(pins["composite_cpu_evidence_size"], 1_981)
        self.assertEqual(pins["composite_cpu_evidence_digest"],
                         "d6b32589595b07922532fe7186a65b4e5a9169d523be16f46a9af5eb041bde17")
        self.assertFalse(
            {controller.COMPOSITE_CPU_RECEIPT_PATH,
             controller.COMPOSITE_CPU_EVIDENCE_PATH}
            & set(controller.REVOKED_COMPOSITE_CPU_PRIOR_PATHS)
        )
        self.assertFalse(
            {controller.COMPOSITE_CPU_SCHEMA,
             controller.COMPOSITE_CPU_EVIDENCE_SCHEMA}
            & controller.REVOKED_COMPOSITE_CPU_PRIOR_SCHEMAS
        )

    def test_source_and_embedded_payloads_compile_without_asserts(self):
        source = V7_HOLD.read_text(encoding="utf-8")
        ready_source = V7_READY.read_text(encoding="utf-8")
        tree = ast.parse(source, str(V7_HOLD))
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        ready_tree = ast.parse(ready_source, str(V7_READY))
        self.assertFalse(any(
            isinstance(node, ast.Assert) for node in ast.walk(ready_tree)
        ))
        for optimize in (0, 1, 2):
            compile(source, str(V7_HOLD), "exec", optimize=optimize)
            compile(ready_source, str(V7_READY), "exec", optimize=optimize)
            compile(controller.ROOT_BOOTSTRAP, "<gpu-v7-root>", "exec",
                    optimize=optimize)
        for symbol in (
            "validate_srun_transport", "_terminate_process_group",
            "validate_internal_artifact_bindings", "validate_postflight",
            "validate_compute_package_root_identity",
        ):
            self.assertTrue(callable(getattr(controller, symbol)))


if __name__ == "__main__":
    unittest.main()
