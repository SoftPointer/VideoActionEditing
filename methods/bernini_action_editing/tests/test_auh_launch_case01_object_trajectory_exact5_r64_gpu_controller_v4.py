#!/usr/bin/env python3
"""Hostile tests for the fresh receipt-gated exact5 GPU controller v4 HOLD."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v4.HOLD.py"
)
STATIC_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_gate_case01_object_trajectory_exact5_static_once_v2.READY.py"
)
ROOT_FAKE_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_gate_case01_object_trajectory_exact5_root_fake_once_v2.READY.py"
)
WORLD4_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_gate_case01_object_trajectory_exact5_world4_once_v3.HOLD.py"
)
WORLD4_ENGINE_PATH = ROOT / (
    "methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py"
)
REAL_MATERIALIZATION_PATH = ROOT / (
    "artifacts/case01_object_trajectory_exact5_r64_canary_v1/"
    "package_materialization_receipt_v1.json"
)
FRESH_PACKAGE_CONTROLLER_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_materialize_case01_object_trajectory_exact5_r64_"
    "overlay_package_once_v3.HOLD.py"
)
FRESH_MATERIALIZER_PATH = ROOT / (
    "methods/bernini_action_editing/tools/"
    "materialize_case01_object_trajectory_exact5_r64_overlay_package_v3.py"
)
FRESH_LAUNCHER_PATH = ROOT / (
    "methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_spooled_launcher_auh_v4.py"
)
FRESH_EVAL_PATH = ROOT / (
    "methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_eval_v4.py"
)
FRESH_COMPOSITE_PATH = ROOT / (
    "methods/bernini_action_editing/"
    "infer_case01_object_trajectory_oracle_auh_r5f_v4.py"
)
COMPOSITE_CPU_GATE_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_gate_case01_object_trajectory_exact5_r5f_v4_"
    "composite_cpu_once_v2.HOLD.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError("literal assignment missing: " + name)


controller = load(CONTROLLER_PATH, "_case01_gpu_controller_v4_test")


class DummyAuthority:
    def __init__(
        self, path: Path | str = "/authority", raw: bytes = b"x",
        mode: int = 0o444,
    ):
        self.path = Path(path); self.raw = raw; self.closed = False
        self.mode = mode

    def row(self):
        return {
            "path": str(self.path), "sha256": hashlib.sha256(self.raw).hexdigest(),
            "size": len(self.raw), "identity": [
                1, 2, 2012, 2000, 0o100000 | self.mode,
                1, 0, len(self.raw), 8, 10, 11,
            ], "mode": self.mode, "nlink": 1,
        }

    def replay(self):
        if self.closed:
            raise RuntimeError("closed authority replayed")

    def close(self):
        self.closed = True


class DummyDirectory:
    def __init__(self, path: Path):
        self.path = path; self.descriptor = 91; self.closed = False
        self.held_identity = (1, 2, 2012, 2000, 0o40700, 2, 0, 4096, 8, 9, 10)

    def replay(self):
        if self.closed:
            raise RuntimeError("closed directory replayed")

    def close(self):
        self.closed = True


class DummyGate:
    def __init__(self, report):
        self.values = {
            "materialization": report,
            "composite_cpu_evidence": {"slurm_step_id": "514"},
        }
        self.closed = False

    def evidence(self):
        return {"receipt_first_before_package_root": True}

    def replay(self):
        if self.closed:
            raise RuntimeError("closed gate replayed")

    def close(self):
        self.closed = True


class DummyPostflight:
    def __init__(self):
        self.evidence = {"all_five_arms_exactly_once": True}

    def replay(self):
        return None

    def close(self):
        return None


def plan_fixture():
    tasks = []
    for arm, task_id in zip(controller.ARM_ORDER, controller.TASK_IDS):
        tasks.append({
            "task_id": task_id, "oracle_arm": arm,
            "source_onset_policy": "hard1_every_step",
            "output": {
                "video_path": str(controller.OUTPUT_ROOT / f"{task_id}.mp4"),
                "receipt_path": str(
                    controller.OUTPUT_ROOT / f"{task_id}.mp4.receipt.json"
                ),
                "create_only": True,
            },
        })
    value = {
        "schema_version": controller.READY_PLAN_SCHEMA,
        "status": "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY",
        "production_ready": False, "launch_allowed": False,
        "hold_reasons": ["explicit_launch_release_not_granted"],
        "arms": list(controller.ARM_ORDER), "task_count": 5,
        "tasks": tasks,
    }
    value["plan_digest"] = controller.object_digest(value)
    return value


def identity_rows():
    result = {}
    for index, role in enumerate(controller.IDENTITY_ROLES):
        raw = f"role-{role}".encode()
        result[role] = {
            "path": str(Path("/sealed") / f"{index:02d}-{role}"),
            "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            "identity": [1, 100 + index, 2012, 2000, 0o100444, 1, 0,
                         len(raw), 8, 10, 11],
            "mode": 0o444, "nlink": 1,
        }
    result["plan"] = {
        **result["plan"], "path": str(controller.READY_PLAN_PATH),
    }
    return result


def launch_input_fixture():
    return {
        "output_report": str(controller.OUTPUT_REPORT_PATH),
        "runner_attestation": str(controller.RUNNER_ATTESTATION_PATH),
        "model_root": "/model", "bernini_root": "/bernini",
        "veomni_root": "/veomni", "authority_root": str(controller.AUTHORITY_ROOT),
        "rank_cache_root": str(controller.RANK_CACHE_ROOT),
    }


def real_launch_input_fixture(report):
    bernini_root = (
        "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
        "VideoEdit_experiments/bernini_graft_v1_20260810/"
        "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
        "source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591"
    )
    return {
        "schema_version":
            "case01-object-trajectory-exact5-hold-launch-input-auh-v3",
        "entry_mode": "trusted_stdin", "campaign_mode": controller.CAMPAIGN,
        "holder_job_id": controller.HOLDER_JOB_ID,
        "expected_node": controller.NODE,
        "expected_allocation_gpu_count": controller.GPU_COUNT,
        "identities": report["launch"]["release"]["identities"],
        "output_report": str(controller.OUTPUT_REPORT_PATH),
        "runner_attestation": str(controller.RUNNER_ATTESTATION_PATH),
        "model_root": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit/checkpoints/"
            "Bernini-R-1.3B-Diffusers-ff4c5d4"
        ),
        "bernini_root": bernini_root,
        "veomni_root": str(Path(bernini_root).parent / "VeOmni-f90b3dc6"),
        "authority_root": str(controller.AUTHORITY_ROOT),
        "rank_cache_root": str(controller.RANK_CACHE_ROOT),
    }


def realistic_rows_from_materialization(report):
    rows = {}
    certified = report["launch"]["release"]["identities"]
    for index, role in enumerate(controller.IDENTITY_ROLES):
        row = dict(certified[role])
        mode = 0o755 if role in controller.EXECUTABLE_ROLES else 0o444
        row.update({
            "identity": [
                48, 12_038_280_342_419_000_000 + index,
                2012, 2000, 0o100000 | mode, 1, 0, row["size"],
                max(8, (row["size"] + 511) // 512),
                1_787_357_728_000_000_000 + index,
                1_787_357_729_000_000_000 + index,
            ],
            "mode": mode, "nlink": 1,
        })
        rows[role] = row
    # The immutable READY overlay is a new inode, while all real package
    # authority paths/hashes/sizes above come from the sealed report.
    rows["plan"]["path"] = str(controller.READY_PLAN_PATH)
    rows["plan"]["sha256"] = "a" * 64
    return rows


def production_directory_fixtures():
    targets = {
        "output": DummyDirectory(controller.OUTPUT_ROOT),
        "final": DummyDirectory(controller.PACKAGE_ROOT / "final"),
        "runtime": DummyDirectory(controller.PACKAGE_ROOT / "runtime"),
    }
    site = DummyDirectory(controller.SITE_PACKAGES_ROOT)
    return site, targets


def _bare_row(authority):
    row = authority.row()
    return {
        "path": row["path"], "sha256": row["sha256"],
        "size": row["size"], "identity": row["identity"],
    }


def composite_cpu_fixture(mutator=None):
    publication = DummyAuthority("/fresh-publication", b"publication\n", 0o400)
    materialization = DummyAuthority(
        "/fresh-materialization", b"materialization\n", 0o400,
    )
    package_controller = DummyAuthority(
        "/fresh-package-controller", b"package-controller\n", 0o400,
    )
    production = {
        "identity_set_digest": "a" * 64,
        "inner_outer_crosslink": {"producer": "fresh-v4"},
    }
    report = {
        "release": {"manifest_digest": "b" * 64},
        "production": production,
    }
    package = {
        "root": str(controller.PACKAGE_ROOT),
        "root_identity": list(controller.PACKAGE_ROOT_IDENTITY),
        "publication_receipt_sha256": hashlib.sha256(publication.raw).hexdigest(),
        "publication_receipt_digest": controller.PACKAGE_PUBLICATION_RECEIPT_DIGEST,
        "materialization_receipt_sha256": hashlib.sha256(
            materialization.raw
        ).hexdigest(),
        "materialization_receipt_digest": controller.MATERIALIZATION_REPORT_DIGEST,
        "package_controller_evidence_sha256": hashlib.sha256(
            package_controller.raw
        ).hexdigest(),
        "package_controller_evidence_digest":
            controller.PACKAGE_CONTROLLER_EVIDENCE_DIGEST,
        "release_file_count": 25,
        "release_manifest_digest": report["release"]["manifest_digest"],
        "production_identity_count": 26,
        "identity_roles": list(controller.IDENTITY_ROLES),
        "identity_set_digest": production["identity_set_digest"],
        "inner_outer_crosslink": production["inner_outer_crosslink"],
    }
    rows = []
    for rank in range(4):
        row = {
            "rank": rank, "pid": 7000 + rank,
            "private_parent_fd_number": 40 + rank,
            "private_parent_replacement_inode": 9000 + rank,
            "pread_bytes_sha256":
                "08e33aedf25337c87eb15e08c32a58f6f4caa21fe073d00b53014c57f8d148e0",
            "pread_offset_before": 13, "pread_offset_after": 13,
            "activation_callback_import_module": "bernini.pipeline",
            "activation_import_before_callback_return": True,
            "captured_vendor_finder_preinstalled": True,
            "captured_vendor_finder_count": 1,
            "captured_vendor_loader_type": "_CapturedVendorLoader",
            "captured_vendor_spec_loader_type": "_CapturedVendorLoader",
            "captured_vendor_loader_is_spec_loader": True,
            "captured_vendor_cached_is_none": True,
        }
        rows.append(row)
    receipt = {
        "schema_version": controller.COMPOSITE_CPU_SCHEMA,
        "status": "PASS_COMPOSITE_CPU_EXACT26_ACTIVATION_IMPORT_HOLD",
        "holder_job_id": controller.HOLDER_JOB_ID, "node": controller.NODE,
        "slurm_step_id": "515", "package": package,
        "world_size": 4, "rank_count": 4, "rank_rows": rows,
        "isolated_runtime": {
            "python_flags": ["-I", "-S", "-B"], "isolated": 1,
            "no_site": 1, "dont_write_bytecode": True,
            "entry_via_proc_self_fd": True,
        },
        "private_parent_fd": {
            "synthetic_model_capture": True, "captured_parent_omitted": True,
            "captured_parent_closed_or_reused": True,
            "frozen_validator_rejected": True, "r5f_validator_accepted": True,
            "r5f_pread_path_exercised": True,
        },
        "shared_ofd_pread": {
            "rank_count": 4, "all_reads_exact": True,
            "offsets_unchanged": True,
        },
        "module_binding": {
            "module_name": "infer_lora", "base_infer_lora_same_object": True,
            "object_cli_applied_to_base_module": True,
            "translated_publication_applied_to_base_module": True,
            "legacy_module_instance_count": 1,
            "duplicate_legacy_module_loaded": False,
        },
        "activation_import": controller.expected_composite_cpu_activation_import(),
        "side_effects": {
            "gpu_requested": False, "torch_imported": False,
            "renderer_or_vae_loaded": False, "publication_performed": False,
        },
        "cache_lifecycle": {
            "admission_cache_root": (
                "/tmp/bernini-case01-object-trajectory-r5f-v4-composite-cpu-"
                "job143808-step515-cache"
            ),
            "admission_cache_fresh": True,
            "admission_cache_cleanup_performed": True,
            "admission_cache_absent_terminal": True,
            "production_rank_cache": str(controller.RANK_CACHE_ROOT),
            "production_rank_cache_untouched": True,
            "production_rank_cache_absent_before_and_after": True,
        },
        "process_cleanup": {
            "all_rank_returncodes_zero": True, "rank_processes_zero": True,
            "torchrun_processes_zero": True, "child_processes_terminal": True,
        },
        "launch_allowed": False,
    }
    if mutator is not None:
        mutator(receipt)
    for row in rows:
        unsigned = dict(row); unsigned.pop("rank_digest", None)
        row["rank_digest"] = controller.object_digest(unsigned)
    receipt["receipt_digest"] = controller.object_digest(receipt)
    receipt_raw = controller.canonical(receipt) + b"\n"
    receipt_authority = DummyAuthority(
        controller.COMPOSITE_CPU_RECEIPT_PATH, receipt_raw, 0o400,
    )
    receipt_authority.held_identity = tuple(receipt_authority.row()["identity"])
    empty_row = {
        "path": "/fresh-cpu.stderr", "sha256": hashlib.sha256(b"").hexdigest(),
        "size": 0, "identity": [1, 2, 2012, 2000, 0o100400, 1, 0, 0, 0, 1, 1],
    }
    stdout_row = {
        "path": "/fresh-cpu.stdout", "sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "size": len(receipt_raw),
        "identity": [1, 3, 2012, 2000, 0o100400, 1, 0,
                     len(receipt_raw), 8, 1, 1],
    }
    evidence = {
        "schema_version": controller.COMPOSITE_CPU_EVIDENCE_SCHEMA,
        "status": "PASS_FRESH_CANARY_V3_COMPOSITE_CPU_CONTROLLER",
        "holder_job_id": controller.HOLDER_JOB_ID, "node": controller.NODE,
        "slurm_step_id": "515", "single_srun_attempt": True,
        "retry_allowed": False, "srun_count": 1, "srun_ntasks": 1,
        "real_rank_process_count": 4, "cpus_per_task": 8, "gpu_count": 0,
        "srun_returncode": 0, "receipt": _bare_row(receipt_authority),
        "receipt_digest": receipt["receipt_digest"],
        "stdout": stdout_row, "stderr": empty_row, "stderr_empty": True,
        "process_group_zero": True, "launch_allowed": False,
        "renderer_or_vae_loaded": False, "publication_performed": False,
    }
    evidence["evidence_digest"] = controller.object_digest(evidence)
    evidence_authority = DummyAuthority(
        controller.COMPOSITE_CPU_EVIDENCE_PATH,
        controller.canonical(evidence) + b"\n", 0o400,
    )
    return {
        "receipt": receipt, "evidence": evidence,
        "receipt_authority": receipt_authority,
        "evidence_authority": evidence_authority,
        "publication": publication, "materialization": materialization,
        "package_controller": package_controller, "report": report,
    }


class GPUControllerSourceTests(unittest.TestCase):
    def test_only_hold_controller_is_checked_in(self):
        hold = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            hold.count(
                'CONTROLLER_STATE = '
                '"HOLD_PENDING_FRESH_V4_COMPOSITE_CPU_PINS"'
            ),
            1,
        )
        self.assertFalse(
            CONTROLLER_PATH.with_name(
                "auh_launch_case01_object_trajectory_exact5_r64_gpu_once_"
                "v4.READY.py"
            ).exists()
        )

    def test_source_compiles_without_asserts_and_root_bootstrap_compiles(self):
        source = CONTROLLER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertEqual(sum(isinstance(node, ast.Assert) for node in ast.walk(tree)), 0)
        compile(source, str(CONTROLLER_PATH), "exec", optimize=0)
        compile(source, str(CONTROLLER_PATH), "exec", optimize=2)
        compile(controller.ROOT_BOOTSTRAP, "<gpu-root-bootstrap>", "exec")
        self.assertIn(
            '"entry_method":"slurm-spooled-or-trusted-stdin-held-python-fd-v1"',
            controller.ROOT_BOOTSTRAP,
        )
        self.assertNotIn(
            '"entry_method":"receipt-gated-single-srun-held-python-fd-v2"',
            controller.ROOT_BOOTSTRAP,
        )
        self.assertNotIn("gpu-controller-v3", controller.ROOT_BOOTSTRAP)
        self.assertNotIn("gpu-release-v3", controller.ROOT_BOOTSTRAP)
        self.assertNotIn("node292-r2-rank-cache", controller.ROOT_BOOTSTRAP)
        self.assertEqual(controller.ROOT_BOOTSTRAP.count("gpu-controller-v4"), 3)
        self.assertEqual(controller.ROOT_BOOTSTRAP.count("gpu-release-v4"), 1)
        self.assertEqual(controller.ROOT_BOOTSTRAP.count("node292-r3-rank-cache"), 1)

    def test_static_v4_boundary_forbids_old_producer_authorities(self):
        self.assertNotIn("canary_v2", str(controller.PACKAGE_ROOT))
        self.assertNotIn("canary_v2", str(controller.PACKAGE_PUBLICATION_RECEIPT_PATH))
        self.assertNotIn("r5f-v3", controller.COMPOSITE_CPU_SCHEMA)
        self.assertNotIn("r5f-v3", controller.COMPOSITE_CPU_EVIDENCE_SCHEMA)
        self.assertTrue(str(controller.COMPOSITE_CPU_RECEIPT_PATH).endswith(
            "canary_v3.composite_cpu_admission_receipt_v2.json"
        ))
        self.assertTrue(str(controller.COMPOSITE_CPU_EVIDENCE_PATH).endswith(
            "canary_v3.composite_cpu_admission_controller_evidence_v2.json"
        ))
        self.assertEqual(
            controller.BASE_ADAPTER_PATH.name,
            "full644_exploratory_matched_infer_adapter_v3.py",
        )
        self.assertEqual(
            controller.BASE_ADAPTER_SHA256,
            "7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120",
        )
        self.assertNotIn(
            "full644_exploratory_matched_infer_adapter_v2.py",
            str(controller.BASE_ADAPTER_PATH),
        )

    def test_scientific_v3_custom_v4_and_off_v5_abi_split(self):
        self.assertEqual(controller.READY_PLAN_SCHEMA,
                         "case01-object-trajectory-exact5-plan-v3")
        self.assertEqual(controller.REPORT_SCHEMA,
                         "case01-object-trajectory-exact5-report-v3")
        self.assertEqual(controller.RUNNER_SCHEMA,
                         "case01-object-trajectory-exact5-runner-attestation-v3")
        self.assertEqual(controller.SCHEMA,
                         "case01-object-trajectory-exact5-r64-gpu-controller-v4")
        self.assertEqual(
            literal_assignment(FRESH_EVAL_PATH, "INFERENCE_RECEIPT_SCHEMA"),
            "bernini-r-1p3b-case01-object-trajectory-oracle-inference-"
            "receipt-v4",
        )
        self.assertEqual(
            literal_assignment(FRESH_EVAL_PATH, "LEGACY_INFERENCE_RECEIPT_SCHEMA"),
            "bernini-r-1p3b-action-lora-inference-receipt-v5",
        )
        self.assertEqual(
            literal_assignment(FRESH_COMPOSITE_PATH, "COMPOSITE_RUNTIME_TRACE_SCHEMA"),
            "bernini-case01-object-trajectory-oracle-runtime-v4",
        )

    def test_checked_in_hold_exits_before_any_io(self):
        forbidden = mock.Mock(side_effect=AssertionError("I/O occurred in HOLD"))
        with mock.patch.object(controller, "open_package_gate", forbidden), \
                mock.patch.object(controller.os, "lstat", forbidden), \
                mock.patch.object(controller.subprocess, "Popen", forbidden), \
                mock.patch.object(
                    sys, "stderr", new_callable=io.StringIO,
                ) as stderr:
            self.assertEqual(controller.main([]), 88)
        self.assertIn("state is not READY", stderr.getvalue())
        forbidden.assert_not_called()

    def test_optimized_checked_in_hold_is_equally_inert(self):
        result = subprocess.run(
            [sys.executable, "-O", str(CONTROLLER_PATH)],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 88)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"state is not READY", result.stderr)

    def test_even_forced_ready_state_blocks_on_cpu_pins_before_io(self):
        forbidden = mock.Mock(side_effect=AssertionError("I/O occurred without token"))
        with mock.patch.object(
                controller, "CONTROLLER_STATE", controller.READY_STATE,
            ), mock.patch.object(
                controller, "open_package_gate", forbidden,
            ), mock.patch.object(
                controller.os, "lstat", forbidden,
            ), mock.patch.object(
                controller.subprocess, "Popen", forbidden,
            ), mock.patch.object(
                sys, "stderr", new_callable=io.StringIO,
            ) as stderr:
            self.assertEqual(controller.main([]), 88)
        self.assertIn("blocked dynamic pins", stderr.getvalue())
        self.assertIn("composite_cpu_receipt_sha256", stderr.getvalue())
        forbidden.assert_not_called()

    def test_only_fresh_v4_cpu_pins_remain_blocked(self):
        pins = controller.dynamic_pin_values()
        self.assertEqual(len(pins), 23)
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
        self.assertEqual(pins["package_publication_receipt_size"], 2528)
        self.assertEqual(pins["materialization_report_size"], 41726)
        self.assertEqual(pins["package_controller_evidence_size"], 8099)
        self.assertEqual(pins["sealed_hold_plan_size"], 32050)
        self.assertEqual(pins["sealed_launch_input_size"], 9788)
        self.assertEqual(pins["named_hold_payload_size"], 12783)
        self.assertIs(type(pins["composite_cpu_receipt_size"]), str)
        self.assertIs(type(pins["composite_cpu_evidence_size"]), str)
        self.assertEqual(len(pins["package_root_identity"]), 11)

    def test_no_remote_or_submission_entrypoint(self):
        source = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("ssh ", source)
        self.assertNotIn("sbatch", source)
        self.assertNotIn("torchrun", controller.build_srun_argv(0))
        self.assertEqual(controller.build_srun_argv(0).count("/usr/bin/srun"), 1)

    def test_active_bootstrap_receipt_field_sets_match_login_validators(self):
        tree = ast.parse(controller.ROOT_BOOTSTRAP)

        def assigned_dict_keys(name):
            matches = []
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == name
                        for target in node.targets
                    )
                    and isinstance(node.value, ast.Dict)
                ):
                    matches.append({
                        key.value for key in node.value.keys
                        if isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                    })
            self.assertEqual(len(matches), 1, name)
            return matches[0]

        self.assertEqual(
            assigned_dict_keys("runtime") | {"receipt_digest"},
            set(controller.RUNTIME_RECEIPT_FIELDS),
        )
        self.assertEqual(
            assigned_dict_keys("cache_receipt") | {"receipt_digest"},
            set(controller.RANK_CACHE_RECEIPT_FIELDS),
        )
        self.assertEqual(
            assigned_dict_keys("compute") | {"result_digest"},
            set(controller.COMPUTE_RESULT_FIELDS),
        )


class ProducerAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load(FRESH_LAUNCHER_PATH, "_case01_gpu_v4_launcher")
        cls.cpu = load(COMPOSITE_CPU_GATE_PATH, "_case01_gpu_v4_cpu_gate")

    def test_paths_align_with_fresh_package_and_cpu_producers(self):
        self.assertEqual(controller.PACKAGE_ROOT, self.cpu.PACKAGE_ROOT)
        self.assertEqual(
            controller.PACKAGE_PUBLICATION_RECEIPT_PATH,
            self.cpu.PUBLICATION_PATH,
        )
        self.assertEqual(
            controller.MATERIALIZATION_REPORT_PATH,
            self.cpu.MATERIALIZATION_PATH,
        )
        self.assertEqual(
            controller.PACKAGE_CONTROLLER_EVIDENCE_PATH,
            self.cpu.PACKAGE_CONTROLLER_PATH,
        )
        self.assertEqual(controller.COMPOSITE_CPU_RECEIPT_PATH, self.cpu.RECEIPT_PATH)
        self.assertEqual(controller.COMPOSITE_CPU_EVIDENCE_PATH, self.cpu.EVIDENCE_PATH)
        self.assertEqual(controller.RANK_CACHE_ROOT, self.cpu.PRODUCTION_RANK_CACHE)
        self.assertEqual(controller.PACKAGE_ROOT.name,
                         "bernini_case01_object_trajectory_exact5_r64_canary_v3")
        self.assertEqual(controller.SOURCE_OVERLAY_ROOT.name,
                         "bernini_case01_object_trajectory_exact5_"
                         "r5f_v4_source_overlay_6_20260822_r1")
        self.assertIn("node292-r3-rank-cache", str(controller.RANK_CACHE_ROOT))
        self.assertEqual(str(controller.AUTHORITY_ROOT).rsplit("/", 1)[-1],
                         "model-authority-v3")

    def test_schema_roles_and_exact_field_sets_align_with_producers(self):
        self.assertEqual(controller.CAMPAIGN, self.launcher.CAMPAIGN)
        self.assertEqual(controller.TASK_IDS, self.launcher.TASK_IDS)
        self.assertEqual(controller.ARM_ORDER, self.launcher.ARM_ORDER)
        self.assertEqual(controller.IDENTITY_ROLES, self.launcher.IDENTITY_ROLES)
        self.assertEqual(controller.IDENTITY_ROLES, self.cpu.IDENTITY_ROLES)
        self.assertEqual(controller.COMPOSITE_CPU_SCHEMA, self.cpu.RECEIPT_SCHEMA)
        self.assertEqual(
            controller.COMPOSITE_CPU_EVIDENCE_SCHEMA,
            self.cpu.SCHEMA + "-evidence",
        )
        self.assertEqual(
            set(controller.COMPOSITE_CPU_RECEIPT_FIELDS),
            set(self.cpu.CPU_RECEIPT_FIELDS),
        )
        self.assertEqual(
            set(controller.COMPOSITE_CPU_RANK_ROW_FIELDS),
            set(self.cpu.CPU_RANK_ROW_FIELDS),
        )
        self.assertEqual(
            set(controller.COMPOSITE_CPU_EVIDENCE_FIELDS),
            set(self.cpu.CPU_EVIDENCE_FIELDS),
        )
        for consumer, producer in (
            (controller.PACKAGE_RECEIPT_FIELDS, "PACKAGE_RECEIPT_FIELDS"),
            (controller.MATERIALIZATION_FIELDS, "REPORT_FIELDS"),
            (controller.HOLD_LAUNCH_FIELDS, "LAUNCH_FIELDS"),
            (controller.HOLD_LAUNCH_RELEASE_FIELDS, "LAUNCH_RELEASE_FIELDS"),
        ):
            self.assertEqual(
                set(consumer),
                set(literal_assignment(FRESH_PACKAGE_CONTROLLER_PATH, producer)),
            )
        materializer = FRESH_MATERIALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn('"runtime/model-authority-v3"', materializer)
        self.assertIn('"outputs/media_v3"', materializer)
        self.assertNotIn('"runtime/model-authority-v2"', materializer)

    def test_activation_import_p0_and_base_v3_join_align_exactly(self):
        expected = controller.expected_composite_cpu_activation_import()
        self.assertEqual(expected["module"], "bernini.pipeline")
        self.assertEqual(
            expected["callback_phase"],
            "inside_original_activate_before_return",
        )
        self.assertIs(expected["finder_installed_before_callback"], True)
        self.assertEqual(expected["finder_count_per_rank"], [1, 1, 1, 1])
        self.assertEqual(expected["loader_type"], "_CapturedVendorLoader")
        self.assertEqual(expected["spec_loader_type"], "_CapturedVendorLoader")
        self.assertIs(expected["loader_is_spec_loader"], True)
        self.assertIs(expected["cached_is_none"], True)
        self.assertEqual(expected["base_adapter_path"], str(
            controller.PACKAGE_ROOT / "release/methods/bernini_action_editing/"
            "full644_exploratory_matched_infer_adapter_v3.py"
        ))
        self.assertEqual(
            expected["base_adapter_sha256"], self.cpu.BASE_ADAPTER_SHA256,
        )
        self.assertNotEqual(
            expected["base_adapter_sha256"],
            "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
        )

    def test_core4_release_tuple_matches_fresh_exact6_materializer(self):
        pins = controller.CORE4_RELEASE_PINS
        self.assertEqual(len(pins), 5)
        expected = {
            "full644_exploratory_matched_infer_adapter_v3.py":
                literal_assignment(FRESH_MATERIALIZER_PATH,
                                   "FINAL_BASE_ADAPTER_SHA256"),
            "infer_case01_object_trajectory_oracle_auh_r5f_v4.py":
                literal_assignment(FRESH_MATERIALIZER_PATH,
                                   "FINAL_WRAPPER_SHA256"),
            "case01_object_trajectory_exact5_eval_v4.py":
                literal_assignment(FRESH_MATERIALIZER_PATH,
                                   "FINAL_EVAL_SHA256"),
            "case01_object_trajectory_exact5_runner_v4.py":
                literal_assignment(FRESH_MATERIALIZER_PATH,
                                   "FINAL_RUNNER_SHA256"),
            "case01_object_trajectory_exact5_spooled_launcher_auh_v4.py":
                literal_assignment(FRESH_MATERIALIZER_PATH,
                                   "FINAL_LAUNCHER_SHA256"),
        }
        by_basename = {
            Path(relative).name: (sha256, size)
            for relative, (sha256, size) in pins.items()
        }
        self.assertEqual(set(by_basename), set(expected))
        for basename, sha256 in expected.items():
            self.assertEqual(by_basename[basename][0], sha256)
            self.assertGreater(by_basename[basename][1], 0)

    def test_gpu_consumer_accepts_exact_p0_join_and_rejects_regressions(self):
        def validate(fixture):
            with mock.patch.object(
                    controller, "COMPOSITE_CPU_RECEIPT_DIGEST",
                    fixture["receipt"]["receipt_digest"],
                ), mock.patch.object(
                    controller, "COMPOSITE_CPU_EVIDENCE_DIGEST",
                    fixture["evidence"]["evidence_digest"],
                ):
                return controller.validate_composite_cpu_admission(
                    fixture["receipt_authority"], fixture["evidence_authority"],
                    fixture["publication"], fixture["materialization"],
                    fixture["package_controller"], fixture["report"],
                )

        exact = composite_cpu_fixture()
        receipt, evidence = validate(exact)
        self.assertIs(receipt["activation_import"][
            "finder_installed_before_callback"
        ], True)
        self.assertEqual(evidence["real_rank_process_count"], 4)

        regressions = {
            "finder installed after callback": lambda value: value[
                "activation_import"
            ].__setitem__("finder_installed_before_callback", False),
            "wrong base adapter path": lambda value: value[
                "activation_import"
            ].__setitem__(
                "base_adapter_path",
                str(controller.BASE_ADAPTER_PATH).replace("_v3.py", "_v2.py"),
            ),
            "PathFinder loader": lambda value: value["rank_rows"][0].__setitem__(
                "captured_vendor_loader_type", "SourceFileLoader"
            ),
            "callback returned first": lambda value: value["rank_rows"][1].__setitem__(
                "activation_import_before_callback_return", False
            ),
            "cached bytecode": lambda value: value["rank_rows"][2].__setitem__(
                "captured_vendor_cached_is_none", False
            ),
        }
        for label, mutator in regressions.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                controller.GPUControllerError, "composite CPU",
            ):
                validate(composite_cpu_fixture(mutator))


class ReadyOverlayTests(unittest.TestCase):
    def test_ready_plan_is_exact_three_leaf_transform(self):
        hold = plan_fixture(); hold_raw = controller.canonical(hold) + b"\n"
        ready, ready_raw = controller.derive_ready_plan(hold_raw)
        self.assertEqual(ready["status"], "READY_FOR_EXPLICIT_LOCAL_LAUNCH")
        self.assertIs(ready["launch_allowed"], True)
        self.assertIs(ready["production_ready"], False)
        self.assertEqual(ready["hold_reasons"], [])
        unsigned = dict(ready); claimed = unsigned.pop("plan_digest")
        self.assertEqual(claimed, controller.object_digest(unsigned))
        self.assertEqual(ready_raw, controller.canonical(ready) + b"\n")
        changed = {
            key for key in ready if key != "plan_digest" and ready[key] != hold[key]
        }
        self.assertEqual(changed, {"status", "launch_allowed", "hold_reasons"})

    def test_ready_plan_rejects_extra_hold_reason(self):
        hold = plan_fixture(); hold["hold_reasons"].append("tampered")
        unsigned = dict(hold); unsigned.pop("plan_digest")
        hold["plan_digest"] = controller.object_digest(unsigned)
        with self.assertRaisesRegex(controller.GPUControllerError, "exact release base"):
            controller.derive_ready_plan(controller.canonical(hold) + b"\n")

    def test_runner_arguments_bind_exact_roles_and_five_arm_campaign(self):
        rows = identity_rows()
        arguments = controller.build_runner_arguments(launch_input_fixture(), rows)
        self.assertEqual(tuple(arguments[::2]), controller.RUNNER_ARGUMENT_FLAGS)
        values = dict(zip(arguments[::2], arguments[1::2]))
        self.assertEqual(values["--campaign-mode"], controller.CAMPAIGN)
        self.assertEqual(values["--plan"], str(controller.READY_PLAN_PATH))
        self.assertEqual(values["--adapter-script"], rows["adapter"]["path"])
        self.assertEqual(
            values["--model-manifest"], rows["base_model_manifest"]["path"]
        )
        self.assertEqual(values["--expected-allocation-gpu-count"], "8")

    def test_in_memory_payload_is_production_and_not_named_hold(self):
        rows = identity_rows()
        authorities = {
            role: types.SimpleNamespace(row=lambda row=row: dict(row))
            for role, row in rows.items() if role != "plan"
        }
        ready = types.SimpleNamespace(row=lambda: dict(rows["plan"]))
        site, targets = production_directory_fixtures()
        release, payload = controller.build_production_release(
            launch_input_fixture(), authorities, ready, site,
            DummyAuthority(
                controller.TORCH_PACKAGE_INIT_PATH, b"torch-init", 0o644,
            ), targets,
        )
        self.assertEqual(release["selected_task_ids"], list(controller.TASK_IDS))
        self.assertEqual(release["arm_order"], list(controller.ARM_ORDER))
        self.assertIs(release["all_arms_attempted_exactly_once_by_runner"], True)
        self.assertIs(release["retry_allowed"], False)
        self.assertIs(release["named_hold_payload_executed"], False)
        self.assertNotIn(b"exit 88", payload)
        self.assertNotIn(b"root_launch_payload_HOLD", payload)
        self.assertEqual(payload.count(b"exec -c"), 1)


class TransportWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        identities = {}
        for index, role in enumerate(controller.IDENTITY_ROLES):
            size = 12_000_000 if role == "ffmpeg" else 20_000 + index
            if role == "plan":
                size = controller.SEALED_HOLD_PLAN_SIZE
            identities[role] = {
                "path": str(controller.PACKAGE_ROOT / "release" / (role + ".py")),
                "sha256": hashlib.sha256(role.encode()).hexdigest(),
                "size": size,
            }
        identities["plan"].update({
            "path": str(controller.HOLD_PLAN_PATH),
            "sha256": controller.SEALED_HOLD_PLAN_SHA256,
        })
        cls.report = {
            "plan": {
                "path": str(controller.HOLD_PLAN_PATH),
                "sha256": controller.SEALED_HOLD_PLAN_SHA256,
                "plan_digest": controller.SEALED_HOLD_PLAN_DIGEST,
            },
            "launch": {
                "release": {"identities": identities},
                "input": {"size": controller.SEALED_LAUNCH_INPUT_SIZE},
                "payload_size": controller.NAMED_HOLD_PAYLOAD_SIZE,
                "payload_sha256": controller.NAMED_HOLD_PAYLOAD_SHA256,
            },
        }

    def test_real_sealed_plan_input_and_named_hold_payload_pins(self):
        self.assertEqual(controller.MATERIALIZATION_REPORT_SIZE, 41726)
        self.assertEqual(
            self.report["plan"],
            {
                "path": str(controller.HOLD_PLAN_PATH),
                "sha256": controller.SEALED_HOLD_PLAN_SHA256,
                "plan_digest": controller.SEALED_HOLD_PLAN_DIGEST,
            },
        )
        self.assertEqual(
            self.report["launch"]["release"]["identities"]["plan"]["size"],
            controller.SEALED_HOLD_PLAN_SIZE,
        )
        self.assertEqual(
            self.report["launch"]["input"]["size"],
            controller.SEALED_LAUNCH_INPUT_SIZE,
        )
        self.assertEqual(
            self.report["launch"]["payload_size"],
            controller.NAMED_HOLD_PAYLOAD_SIZE,
        )
        self.assertEqual(
            self.report["launch"]["payload_sha256"],
            controller.NAMED_HOLD_PAYLOAD_SHA256,
        )
        self.assertGreater(
            self.report["launch"]["release"]["identities"]["ffmpeg"]["size"],
            controller.MAX_SOURCE_SIZE,
        )
        self.assertLess(
            self.report["launch"]["release"]["identities"]["ffmpeg"]["size"],
            controller.MAX_RUNTIME_EXECUTABLE_SIZE,
        )

    def test_complete_real_package_payload_has_large_transport_headroom(self):
        rows = realistic_rows_from_materialization(self.report)
        authorities = {
            role: types.SimpleNamespace(row=lambda row=row: dict(row))
            for role, row in rows.items() if role != "plan"
        }
        ready = types.SimpleNamespace(row=lambda: dict(rows["plan"]))
        site, targets = production_directory_fixtures()
        release, payload = controller.build_production_release(
            real_launch_input_fixture(self.report), authorities, ready,
            site, DummyAuthority(
                controller.TORCH_PACKAGE_INIT_PATH, b"torch-init", 0o644,
            ), targets,
        )
        transport = controller.validate_srun_transport(
            controller.build_srun_argv(514), max_gate_step=514,
            payload=payload, release=release,
        )
        self.assertEqual(transport["exact_srun_joined_width"], 286)
        self.assertEqual(transport["exact_srun_execve_argv_width"], 287)
        self.assertEqual(transport["exact_srun_environment_width"], 84)
        self.assertEqual(transport["exact_srun_execve_total_width"], 371)
        self.assertEqual(
            transport["exact_srun_execve_headroom"],
            controller.OBSERVED_AUH_ARG_MAX - 371,
        )
        self.assertEqual(transport["composite_cpu_admission_step_floor"], 514)
        self.assertEqual(transport["held_stdin_size"], len(payload))
        self.assertEqual(
            transport["held_stdin_sha256"], hashlib.sha256(payload).hexdigest(),
        )
        self.assertGreater(len(payload), controller.NAMED_HOLD_PAYLOAD_SIZE)
        self.assertLess(len(payload), controller.MAX_HELD_STDIN_BYTES)
        self.assertLess(
            transport["nested_python_argv_upper_bound"],
            controller.MAX_NESTED_PYTHON_ARGV_BYTES,
        )
        self.assertEqual(
            transport["nested_python_argv_upper_bound"],
            controller._nested_python_argv_upper_bound(release, 514),
        )
        self.assertEqual(
            controller._nested_python_argv_upper_bound(release, 514),
            controller._nested_python_argv_upper_bound(
                release, 9_999_999_999_999_999_999,
            ),
        )
        self.assertGreater(
            controller.OBSERVED_AUH_ARG_MAX
            - transport["nested_python_argv_upper_bound"],
            controller.MIN_EXECVE_HEADROOM_BYTES,
        )

    def test_transport_rejects_argv_or_stdin_substitution_before_popen(self):
        command = controller.build_srun_argv(514)
        controller.validate_srun_transport(command, max_gate_step=514)
        for wrong in (0, 513, 515):
            with self.assertRaisesRegex(
                controller.GPUControllerError, "exact fixed GPU srun argv differs",
            ):
                controller.validate_srun_transport(
                    command, max_gate_step=wrong,
                )
        with self.assertRaisesRegex(
            controller.GPUControllerError, "exact fixed GPU srun argv differs",
        ):
            controller.validate_srun_transport(
                command + ["tampered"], max_gate_step=514,
            )
        with self.assertRaisesRegex(
            controller.GPUControllerError, "held GPU stdin exceeds",
        ):
            controller.validate_srun_transport(
                command, max_gate_step=514,
                payload=b"x" * controller.MAX_HELD_STDIN_BYTES,
                release={},
            )


class ComputeAuthorityHostileTests(unittest.TestCase):
    def test_compute_package_root_requires_full_published_identity(self):
        controller.validate_compute_package_root_identity(
            list(controller.PACKAGE_ROOT_IDENTITY)
        )
        same_device_new_inode = list(controller.PACKAGE_ROOT_IDENTITY)
        same_device_new_inode[1] += 1
        with self.assertRaisesRegex(
            controller.GPUControllerError, "package-root identity",
        ):
            controller.validate_compute_package_root_identity(
                same_device_new_inode
            )

    def test_active_bootstrap_uses_held_torch_source_under_real_no_site(self):
        tree = ast.parse(controller.ROOT_BOOTSTRAP)
        import_lines = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "torch" for alias in node.names)
        ]
        inserts = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "insert"
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sys"
            and node.func.value.attr == "path"
        ]
        self.assertEqual(import_lines, [])
        self.assertTrue(inserts)
        self.assertIn("torch_package_init_authority", controller.ROOT_BOOTSTRAP)
        self.assertIn("SourceFileLoader", controller.ROOT_BOOTSTRAP)
        with tempfile.TemporaryDirectory() as raw:
            site = Path(raw) / "site-packages"
            package = site / "torch"
            package.mkdir(parents=True)
            init_path = package / "__init__.py"
            init_path.write_text("HELD_SOURCE_MARKER = 731\n", encoding="utf-8")
            code = r'''
import importlib.machinery,importlib.util,os,sys
site=sys.argv[1];source=site+"/torch/__init__.py"
if sys.flags.isolated!=1 or sys.flags.no_site!=1 or not sys.dont_write_bytecode: raise SystemExit(10)
try:
 import torch
except ModuleNotFoundError:
 pass
else:
 raise SystemExit(11)
fd=os.open(source,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0));raw=os.read(fd,1048576)
sys.path.insert(0,site)
loader=importlib.machinery.SourceFileLoader("torch",source)
spec=importlib.util.spec_from_file_location("torch",source,loader=loader,submodule_search_locations=[site+"/torch"])
module=importlib.util.module_from_spec(spec);sys.modules["torch"]=module
exec(compile(raw,source,"exec",dont_inherit=True),module.__dict__)
if module.HELD_SOURCE_MARKER!=731 or module.__loader__ is not loader: raise SystemExit(12)
if os.pread(fd,len(raw),0)!=raw: raise SystemExit(13)
os.close(fd)
'''
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", code, str(site)],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_embedded_cache_inventory_real_exact281_and_symlink_hostile(self):
        tree = ast.parse(controller.ROOT_BOOTSTRAP)
        selected = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"ident", "read_fd", "cache_inventory"}
        ]
        namespace = {"os": os, "stat": __import__("stat"), "hashlib": hashlib}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<cache>", "exec"), namespace)
        inventory_fn = namespace["cache_inventory"]
        rank_children = (
            "miopen-user", "miopen-custom", "xdg", "tmp", "triton",
            "inductor", "extensions", "pycache", "home", "hf", "torch",
        )
        coordinator_children = ("pycache", "home", "hf", "torch", "xdg", "tmp")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            os.chmod(root, 0o700)
            for index, task_id in enumerate(controller.TASK_IDS):
                task = root / f"task-{index:02d}-{task_id}"
                task.mkdir(mode=0o700)
                coordinator = task / "coordinator"
                coordinator.mkdir(mode=0o700)
                for name in coordinator_children:
                    (coordinator / name).mkdir(mode=0o700)
                for rank_index in range(4):
                    rank = task / f"rank-{rank_index}"
                    rank.mkdir(mode=0o700)
                    for name in rank_children:
                        (rank / name).mkdir(mode=0o700)
            rows, total = inventory_fn(root, os.getuid(), os.getgid())
            self.assertEqual(len(rows), 281)
            self.assertEqual(total, 0)
            self.assertTrue(all(len(row["identity"]) == 11 for row in rows))
            self.assertEqual([row["path"] for row in rows], sorted(row["path"] for row in rows))
            hostile = root / f"task-00-{controller.TASK_IDS[0]}" / "rank-0/tmp/link"
            os.symlink("missing", hostile)
            with self.assertRaisesRegex(
                RuntimeError, "identity|special|symlink|openat",
            ):
                inventory_fn(root, os.getuid(), os.getgid())

    @staticmethod
    def _binding_fixture():
        release = {
            "identities": {
                "r64_checkpoint_manifest": {"sha256": "a" * 64},
            },
        }
        ready = {"plan_digest": "b" * 64}
        release_digest = controller.object_digest(release)
        attestation = {"status": "ok"}
        attestation["attestation_digest"] = controller.object_digest(attestation)
        attestation_raw = controller.canonical(attestation) + b"\n"
        runtime = {
            "slurm_step_id": "508", "production_release_digest": release_digest,
            "ready_plan_digest": ready["plan_digest"],
            "r64_checkpoint_manifest_sha256": "a" * 64,
            "exact26_identity_set_digest": controller.object_digest(
                release["identities"]
            ),
        }
        runtime["receipt_digest"] = controller.object_digest(runtime)
        runtime_raw = controller.canonical(runtime) + b"\n"
        rank = {
            "slurm_step_id": "508", "production_release_digest": release_digest,
            "runtime_receipt_sha256": hashlib.sha256(runtime_raw).hexdigest(),
            "runtime_receipt_digest": runtime["receipt_digest"],
            "runner_attestation_sha256": hashlib.sha256(attestation_raw).hexdigest(),
            "runner_attestation_digest": attestation["attestation_digest"],
            "internal_artifact_inventory_digest": "9" * 64,
            "model_authority_root_identity": [
                48, 901, 2012, 2000, 0o40700, 2, 0, 4096, 8, 10, 11,
            ],
        }
        rank["receipt_digest"] = controller.object_digest(rank)
        rank_raw = controller.canonical(rank) + b"\n"
        compute = {
            "slurm_step_id": "508", "production_release_digest": release_digest,
            "ready_plan_digest": ready["plan_digest"],
            "runtime_receipt_sha256": hashlib.sha256(runtime_raw).hexdigest(),
            "runtime_receipt_digest": runtime["receipt_digest"],
            "rank_cache_receipt_sha256": hashlib.sha256(rank_raw).hexdigest(),
            "rank_cache_receipt_digest": rank["receipt_digest"],
            "runner_attestation_sha256": hashlib.sha256(attestation_raw).hexdigest(),
            "runner_attestation_digest": attestation["attestation_digest"],
            "rank_processes_zero": True,
            "internal_artifact_count": 55,
            "internal_artifact_inventory_digest": "9" * 64,
            "model_authority_root_identity": list(
                rank["model_authority_root_identity"]
            ),
            "model_authority_root_empty": True,
            "model_authority_root_held_and_terminal_replayed": True,
            "rank_cache_compute_state":
            "RETAINED_COMPUTE_LOCAL_POSTFLIGHT_AUTHORITY",
        }
        compute["result_digest"] = controller.object_digest(compute)
        dispatch = {
            "production_release_digest": release_digest,
            "ready_plan_digest": ready["plan_digest"],
        }
        return {
            "runtime": runtime, "runtime_raw": runtime_raw,
            "rank_cache": rank, "rank_cache_raw": rank_raw,
            "compute": compute, "attestation": attestation,
            "attestation_raw": attestation_raw, "ready_plan": ready,
            "release": release, "dispatch": dispatch,
        }

    def test_runtime_cache_compute_step_and_release_tamper_fail_closed(self):
        baseline = self._binding_fixture()
        controller.validate_compute_binding_closure(**baseline)
        for container, field, value, digest_field in (
            ("runtime", "r64_checkpoint_manifest_sha256", "c" * 64, "receipt_digest"),
            ("runtime", "slurm_step_id", "509", "receipt_digest"),
            ("rank_cache", "runtime_receipt_sha256", "d" * 64, "receipt_digest"),
            ("compute", "production_release_digest", "e" * 64, "result_digest"),
            ("dispatch", "ready_plan_digest", "f" * 64, None),
        ):
            fixture = self._binding_fixture()
            fixture[container][field] = value
            if digest_field is not None:
                fixture[container].pop(digest_field)
                fixture[container][digest_field] = controller.object_digest(
                    fixture[container]
                )
            if container == "runtime":
                fixture["runtime_raw"] = controller.canonical(
                    fixture["runtime"]
                ) + b"\n"
            elif container == "rank_cache":
                fixture["rank_cache_raw"] = controller.canonical(
                    fixture["rank_cache"]
                ) + b"\n"
            with self.subTest(container=container, field=field):
                with self.assertRaises(controller.GPUControllerError):
                    controller.validate_compute_binding_closure(**fixture)

    @staticmethod
    def _internal_artifact_fixture():
        suffixes = {
            "model_capture": "-model-capture.json",
            "model_pre_use": "-model-pre-use.json",
            "consumption_input": "-consumption-input.json",
            "model_post_use": "-model-post-use.json",
            "eval_consumption_chain": "-eval-consumption-chain.json",
            "adapter_capture": "-adapter-capture.json",
            "adapter_pre_use": "-adapter-pre-use.json",
            "adapter_post_use": "-adapter-post-use.json",
            "adapter_final": "-adapter-final.json",
        }
        task_results = []
        artifact_replays = []
        inventory = []
        for index, task_id in enumerate(controller.TASK_IDS):
            prefix = f".matched-v2-{index:02d}-{task_id}"
            references = {}
            for role, suffix in suffixes.items():
                basename = prefix + suffix
                sha256 = hashlib.sha256(basename.encode()).hexdigest()
                references[role] = {"basename": basename, "sha256": sha256}
                inventory.append({
                    "task_index": index, "task_id": task_id, "role": role,
                    "basename": basename, "sha256": sha256,
                })
            result = {
                "task_index": index, "task_id": task_id, "return_code": 0,
                "attempt_count": 1, "retry_allowed": False,
                "log_basename": prefix + ".log",
                "authority_artifacts": references,
            }
            result["task_result_digest"] = controller.object_digest(result)
            runner_sha = hashlib.sha256(
                (prefix + "-runner-task.json").encode()
            ).hexdigest()
            artifact_replays.append({
                "task_id": task_id,
                "task_result_digest": result["task_result_digest"],
                "artifact_count": 9,
                "runner_task_file_sha256": runner_sha,
            })
            inventory.extend((
                {
                    "task_index": index, "task_id": task_id,
                    "role": "runner_task",
                    "basename": prefix + "-runner-task.json",
                    "sha256": runner_sha,
                },
                {
                    "task_index": index, "task_id": task_id,
                    "role": "task_log", "basename": prefix + ".log",
                    "sha256": hashlib.sha256((prefix + ".log").encode()).hexdigest(),
                },
            ))
            task_results.append(result)
        return {
            "task_results": task_results,
            "task_artifact_replays": artifact_replays,
        }, sorted(inventory, key=lambda row: row["basename"])

    def test_exact55_attestation_to_compute_inventory_join_rejects_tamper(self):
        attestation, inventory = self._internal_artifact_fixture()
        by_name = controller.validate_internal_artifact_bindings(
            attestation, inventory,
        )
        self.assertEqual(len(by_name), 55)
        for mutation in ("json_sha", "runner_sha", "duplicate_basename"):
            hostile = json.loads(json.dumps(inventory))
            if mutation == "json_sha":
                target = next(row for row in hostile if row["role"] == "model_capture")
                target["sha256"] = "f" * 64
            elif mutation == "runner_sha":
                target = next(row for row in hostile if row["role"] == "runner_task")
                target["sha256"] = "e" * 64
            else:
                hostile[1]["basename"] = hostile[0]["basename"]
            with self.subTest(mutation=mutation):
                with self.assertRaises(controller.GPUControllerError):
                    controller.validate_internal_artifact_bindings(
                        attestation, hostile,
                    )

    def test_video_publication_identity_uses_held_inode_after_same_bytes_swap(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            video = root / "arm.mp4"
            video.write_bytes(b"same-video-bytes")
            os.chmod(video, 0o400)
            stale_named_identity = controller.identity(os.lstat(video))
            os.rename(video, root / "displaced-arm.mp4")
            video.write_bytes(b"same-video-bytes")
            os.chmod(video, 0o400)
            held = controller.open_authority(
                video,
                expected_sha256=hashlib.sha256(b"same-video-bytes").hexdigest(),
                expected_size=len(b"same-video-bytes"), expected_mode=0o400,
                expected_uid=os.getuid(), expected_gid=os.getgid(),
            )
            try:
                observed = controller.held_publication_identity(held)
                stale = dict(zip(controller.IDENTITY_FIELD_NAMES, stale_named_identity))
                self.assertNotEqual(stale["inode"], observed["inode"])
                self.assertEqual(
                    observed,
                    dict(zip(controller.IDENTITY_FIELD_NAMES, held.held_identity)),
                )
                self.assertNotEqual(stale, observed)
            finally:
                held.close()

    def test_held_directory_replay_tolerates_nlink_then_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "held"
            path.mkdir(mode=0o700)
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            held = controller.HeldDirectory(
                path, descriptor, controller.identity(os.fstat(descriptor)),
            )
            (path / "legitimate-child").mkdir(mode=0o700)
            held.replay()
            displaced = Path(raw) / "displaced"
            os.rename(path, displaced)
            path.mkdir(mode=0o700)
            with self.assertRaisesRegex(controller.GPUControllerError, "changed"):
                held.replay()
            held.close()

    def test_postflight_directory_contract_rejects_late_extra_member(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            held = controller.HeldDirectory(
                path, descriptor, controller.identity(os.fstat(descriptor)),
            )
            closure = controller.PostflightClosure({}, [], [(held, set())])
            closure.replay()
            (path / "late-extra").write_bytes(b"x")
            with self.assertRaisesRegex(
                controller.GPUControllerError, "member closure changed",
            ):
                closure.replay()
            held.close()

    def test_postflight_directory_contract_rejects_same_name_inode_swap(self):
        with tempfile.TemporaryDirectory() as raw:
            directory_path = Path(raw)
            leaf = directory_path / "authority.json"
            leaf.write_bytes(b"sealed")
            os.chmod(leaf, 0o400)
            leaf_fd = os.open(leaf, os.O_RDONLY)
            leaf_authority = controller.HeldAuthority(
                leaf, leaf_fd, controller.identity(os.fstat(leaf_fd)), b"sealed",
            )
            directory_fd = os.open(
                directory_path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            held_directory = controller.HeldDirectory(
                directory_path, directory_fd,
                controller.identity(os.fstat(directory_fd)),
            )
            closure = controller.PostflightClosure(
                {}, [leaf_authority], [(held_directory, {leaf.name})],
            )
            closure.replay()
            displaced = directory_path / "displaced"
            os.rename(leaf, displaced)
            leaf.write_bytes(b"sealed")
            os.chmod(leaf, 0o400)
            with self.assertRaisesRegex(
                controller.GPUControllerError, "authority changed|leaf changed",
            ):
                closure.replay()
            leaf_authority.close()
            held_directory.close()


class OrderingAndCleanupTests(unittest.TestCase):
    def test_package_gate_opens_all_receipts_before_root(self):
        events = []

        def opened(path, **_kwargs):
            events.append("file:" + Path(path).name)
            return DummyAuthority(path, b"{}\n")

        def directory(path, **_kwargs):
            events.append("directory:" + Path(path).name)
            result = DummyDirectory(Path(path))
            if Path(path) == controller.PACKAGE_ROOT:
                result.held_identity = tuple(controller.PACKAGE_ROOT_IDENTITY)
            return result

        report = {"launch": {"release": {"identities": {}}, "input": {}}, "plan": {}}
        with mock.patch.object(
                controller, "open_authority", side_effect=opened,
            ), mock.patch.object(
                controller, "open_directory", side_effect=directory,
            ), mock.patch.object(
                controller, "validate_publication_receipt",
                return_value={
                    "target_root_identity": list(controller.PACKAGE_ROOT_IDENTITY),
                },
            ), mock.patch.object(
                controller, "validate_materialization_report", return_value=report,
            ), mock.patch.object(
                controller, "validate_package_controller_evidence",
                return_value={},
            ), mock.patch.object(
                controller, "validate_composite_cpu_admission",
                return_value=({}, {"slurm_step_id": "514"}),
            ):
            gate = controller.open_package_gate()
            gate.close()
        self.assertEqual(
            events[:5],
            [
                "file:" + controller.PACKAGE_PUBLICATION_RECEIPT_PATH.name,
                "file:" + controller.MATERIALIZATION_REPORT_PATH.name,
                "file:" + controller.PACKAGE_CONTROLLER_EVIDENCE_PATH.name,
                "file:" + controller.COMPOSITE_CPU_RECEIPT_PATH.name,
                "file:" + controller.COMPOSITE_CPU_EVIDENCE_PATH.name,
            ],
        )
        self.assertTrue(events[5].startswith("directory:"))

    def test_controller_mutation_order_attempt_plan_single_srun(self):
        events = []
        hold = plan_fixture(); hold_raw = controller.canonical(hold) + b"\n"
        rows = identity_rows()
        authorities = {
            role: DummyAuthority(rows[role]["path"], hold_raw if role == "plan" else role.encode())
            for role in controller.IDENTITY_ROLES
        }
        authorities["launch_input"] = DummyAuthority("/launch-input", b"{}\n")
        report = {
            "launch": {
                "payload_sha256": hashlib.sha256(b"HOLD\nexit 88\n").hexdigest(),
                "payload_size": len(b"HOLD\nexit 88\n"),
            },
            "plan": {"sha256": hashlib.sha256(hold_raw).hexdigest()},
        }
        launch_input = launch_input_fixture()
        launch_input["identities"] = {
            "plan": {"size": len(hold_raw)},
        }
        dirs = {
            label: DummyDirectory(path)
            for label, path in (
                ("evidence", controller.PACKAGE_ROOT / "evidence"),
                ("runtime", controller.PACKAGE_ROOT / "runtime"),
                ("logs", controller.PACKAGE_ROOT / "logs"),
                ("output", controller.OUTPUT_ROOT),
                ("final", controller.PACKAGE_ROOT / "final"),
                ("site_packages", controller.SITE_PACKAGES_ROOT),
            )
        }

        def open_any(path, **_kwargs):
            path = Path(path)
            if path == controller.HOLD_PAYLOAD_PATH:
                return DummyAuthority(path, b"HOLD\nexit 88\n")
            return DummyAuthority(path, b"authority")

        def make_json(_directory, path, value, _mode):
            events.append("json:" + Path(path).name)
            return controller.canonical(value) + b"\n"

        def make_raw(_directory, path, raw, _mode):
            events.append("raw:" + Path(path).name)
            return raw

        real_validate_transport = controller.validate_srun_transport

        def validate_transport(
            command, *, max_gate_step, payload=None, release=None,
        ):
            events.append("width:stdin" if payload is not None else "width:argv")
            return real_validate_transport(
                command, max_gate_step=max_gate_step,
                payload=payload, release=release,
            )

        with mock.patch.object(
                controller, "open_package_gate", return_value=DummyGate(report),
            ), mock.patch.object(
                controller, "_open_package_identities",
                return_value=(authorities, launch_input),
            ), mock.patch.object(
                controller, "validate_site_packages_layout",
                return_value=controller.SITE_PACKAGES_ROOT,
            ), mock.patch.object(
                controller, "open_authority", side_effect=open_any,
            ), mock.patch.object(
                controller, "open_observed_authority",
                return_value=DummyAuthority("/controller", b"controller"),
            ), mock.patch.object(
                controller, "open_directory",
                side_effect=lambda path, **_kw: dirs[next(
                    key for key, value in dirs.items()
                    if value.path == Path(path)
                )],
            ), mock.patch.object(
                controller, "require_fresh_outputs", return_value=None,
            ), mock.patch.object(
                controller, "_exact_names", return_value=None,
            ), mock.patch.object(
                controller, "create_immutable_json", side_effect=make_json,
            ), mock.patch.object(
                controller, "create_immutable", side_effect=make_raw,
            ), mock.patch.object(
                controller, "build_production_release",
                return_value=({"release": True}, b"PRODUCTION"),
            ), mock.patch.object(
                controller, "validate_srun_transport",
                side_effect=validate_transport,
            ), mock.patch.object(
                controller, "run_single_srun",
                side_effect=lambda *_args: (
                    events.append("srun") or (0, b"{}\n", b"", 123)
                ),
            ), mock.patch.object(
                controller, "validate_postflight",
                return_value=DummyPostflight(),
            ):
            result = controller.controller()
        self.assertEqual(events.count("srun"), 1)
        self.assertEqual(events.count("width:argv"), 1)
        self.assertEqual(events.count("width:stdin"), 1)
        self.assertLess(
            events.index("width:argv"),
            events.index("json:" + controller.ATTEMPT_PATH.name),
        )
        self.assertLess(
            events.index("json:" + controller.ATTEMPT_PATH.name),
            events.index("raw:" + controller.READY_PLAN_PATH.name),
        )
        self.assertLess(
            events.index("raw:" + controller.READY_PLAN_PATH.name),
            events.index("json:" + controller.DISPATCH_PATH.name),
        )
        self.assertLess(
            events.index("json:" + controller.DISPATCH_PATH.name),
            events.index("srun"),
        )
        self.assertTrue(result["all_five_arms_attempted_exactly_once"])
        self.assertFalse(result["retry_allowed"])

    def test_timeout_terminates_new_process_group_and_never_retries(self):
        process = mock.Mock()
        process.pid = 4141
        process.poll.return_value = None
        process.communicate.side_effect = subprocess.TimeoutExpired(["srun"], 1)
        with mock.patch.object(
                controller.subprocess, "Popen", return_value=process,
            ) as popen, mock.patch.object(
                controller, "_terminate_process_group",
            ) as terminate, mock.patch.object(
                controller, "_process_group_absent", return_value=True,
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                controller.run_single_srun(
                    controller.build_srun_argv(514), b"payload", {}, 514,
                )
        popen.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        terminate.assert_called_once_with(process)

    def test_cleanup_kills_descendants_even_after_group_leader_exited(self):
        process = mock.Mock()
        process.pid = 5151
        process.poll.return_value = 0
        with mock.patch.object(
                controller.os, "killpg",
            ) as killpg, mock.patch.object(
                controller, "_process_group_absent",
                side_effect=[False, True],
            ):
            controller._terminate_process_group(process)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(5151, controller.signal.SIGTERM),
             mock.call(5151, controller.signal.SIGKILL)],
        )

    def test_timeout_pipe_close_error_is_terminal_after_group_cleanup(self):
        process = mock.Mock()
        process.pid = 6161
        process.poll.return_value = None
        process.communicate.side_effect = subprocess.TimeoutExpired(["srun"], 1)
        good = mock.Mock(); good.closed = False
        bad = mock.Mock(); bad.closed = False
        bad.close.side_effect = OSError("hostile close")
        process.stdin, process.stdout, process.stderr = good, bad, good
        with mock.patch.object(
                controller.subprocess, "Popen", return_value=process,
            ), mock.patch.object(
                controller, "_terminate_process_group",
            ) as terminate:
            with self.assertRaisesRegex(
                controller.GPUControllerError, "pipe cleanup failed",
            ):
                controller.run_single_srun(
                    controller.build_srun_argv(514), b"payload", {}, 514,
                )
        terminate.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
