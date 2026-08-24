#!/usr/bin/env python3
"""Snapshot, launcher and external-controller adversarial V3 contracts."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import actual_target_foundation_canary_v3 as authority
import actual_target_foundation_controller_v3 as controller
import actual_target_foundation_runtime_v3 as runtime
import actual_target_foundation_snapshot_v3 as snapshot


TEST_JOB_ID = "147871"


def digested(value):
    return {**value, "digest": authority.object_sha256(value)}


def thaw_snapshot_tree(root: Path) -> None:
    if not root.exists():
        return
    for current, directories, _files in os.walk(root, topdown=False):
        for name in directories:
            os.chmod(Path(current) / name, 0o700)
        os.chmod(current, 0o700)


class SnapshotTests(unittest.TestCase):
    def test_materialized_snapshot_is_exact_immutable_and_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "snapshot"
            try:
                receipt = snapshot.materialize_snapshot(ROOT, target)
                self.assertTrue(receipt["verified"])
                self.assertEqual(receipt["snapshot_file_count"], 17)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555)
                self.assertTrue(snapshot.verify_snapshot(target, verify_original=True)["verified"])
                with self.assertRaises(snapshot.SnapshotV3Error):
                    snapshot.materialize_snapshot(ROOT, target)
            finally:
                thaw_snapshot_tree(target)

    def test_snapshot_rejects_extra_file_and_symlink_laundering(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            target = base / "snapshot"
            try:
                snapshot.materialize_snapshot(ROOT, target)
                os.chmod(target, 0o755)
                (target / "raw_sidecar.bin").write_bytes(b"forbidden")
                os.chmod(target, 0o555)
                with self.assertRaises(snapshot.SnapshotV3Error):
                    snapshot.verify_snapshot(target, verify_original=False)
            finally:
                thaw_snapshot_tree(target)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            real = base / "real"
            real.mkdir()
            (real / "payload.py").write_text("x=1\n", encoding="ascii")
            linked = base / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with mock.patch.object(snapshot, "_payload_paths", return_value=("payload.py",)):
                with self.assertRaises(snapshot.SnapshotV3Error):
                    snapshot.original_source_closure(linked)

    def test_payload_authority_is_exact_and_scripts_are_executable_contracts(self):
        paths = authority.load_authority()["snapshot_payload_relative_paths"]
        self.assertEqual(len(paths), 17)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("tests/test_actual_target_foundation_runtime_v3.py", paths)
        self.assertIn("tests/test_actual_target_foundation_controller_v3.py", paths)
        formal = (ROOT / "scripts/auh_actual_target_foundation_canary_formal_v3.sbatch").read_text()
        existing = (ROOT / "scripts/auh_actual_target_foundation_canary_existing_allocation_v3.sh").read_text()
        rank = (ROOT / "scripts/auh_actual_target_foundation_canary_rank_wrapper_v3.sh").read_text()
        controller_wrapper_path = ROOT / "scripts/auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
        controller_wrapper = controller_wrapper_path.read_text()
        self.assertIn(
            "scripts/auh_actual_target_foundation_canary_controller_wrapper_v3.sh",
            paths,
        )
        self.assertIn(
            stat.S_IMODE(controller_wrapper_path.stat().st_mode),
            (0o555, 0o755),
        )
        self.assertNotEqual(
            stat.S_IMODE(controller_wrapper_path.stat().st_mode) & 0o111, 0
        )
        self.assertEqual(
            hashlib.sha256(controller_wrapper_path.read_bytes()).hexdigest(),
            authority.load_authority()["v3r2_engineering_repair_contract"][
                "controller_wrapper_sha256"
            ],
        )
        self.assertIn('pipe_status=("${PIPESTATUS[@]}")', formal)
        self.assertIn('pipe_status=("${PIPESTATUS[@]}")', existing)
        self.assertEqual(
            authority.load_authority()["authorized_launch_mode"],
            "existing_allocation_only",
        )
        refusal = "V3 formal sbatch is not authorized"
        self.assertIn(refusal, formal)
        self.assertLess(formal.index("exit 64"), formal.index("snapshot_root="))
        self.assertIn("--jobid \"$job_id\"", existing)
        self.assertNotIn("--overlap", existing)
        self.assertIn("--exclusive", existing)
        self.assertIn("--exact", existing)
        self.assertIn("--immediate=60", existing)
        self.assertNotIn("active_compute_steps", existing)
        self.assertIn("${SLURM_JOB_ID:-}", existing)
        self.assertIn("${SLURM_STEP_ID:-}", existing)
        self.assertIn("${SLURM_STEPID:-}", existing)
        self.assertIn("147873", existing)
        self.assertIn("--mem=56G", existing)
        self.assertIn("--gres=gpu:mi210:1", existing)
        self.assertIn("controller_argv=(", existing)
        self.assertIn("controller_argv=(", formal)
        self.assertIn("--cpus-per-task=1", existing)
        self.assertIn("--mem=1G", existing)
        for launcher in (formal, existing):
            self.assertIn("--gpus=0", launcher)
            self.assertIn("--export=ALL", launcher)
            self.assertIn('"$controller_wrapper"', launcher)
            self.assertIn('controller_argv_file="${run_root}/controller_argv.nul"', launcher)
            self.assertIn(
                'controller_step_meta="${run_root}/controller_step_meta.json"',
                launcher,
            )
        self.assertIn("REAL_GPU_LAUNCH_AUTHORIZED = True", (ROOT / "actual_target_foundation_runtime_v3.py").read_text())
        self.assertIn("exec \"$python_bin\" -B \"$runtime\"", rank)
        for exact_export in (
            "export CUDA_VISIBLE_DEVICES=''",
            "export ROCR_VISIBLE_DEVICES=''",
            "export HIP_VISIBLE_DEVICES=''",
        ):
            self.assertIn(exact_export, controller_wrapper)
            self.assertLess(
                controller_wrapper.index(exact_export),
                controller_wrapper.index('exec "$python_bin" -B "$controller"'),
            )
        self.assertIn("MIOPEN_USER_DB_PATH", rank)
        self.assertIn("MIOPEN_CUSTOM_CACHE_DIR", rank)
        self.assertIn("unset MIOPEN_DISABLE_CACHE", rank)
        self.assertIn("initially empty", rank)
        self.assertIn("run_7f3c21a9_v3r4", existing)
        self.assertIn("prior_failed_run_root=", existing)
        self.assertIn('mkdir -m 700 "$run_root"', existing)
        self.assertNotIn('mkdir -p "$run_root"', existing)
        self.assertIn(
            '"$python_bin" -B "$controller" verify-prior-closures >/dev/null',
            existing,
        )
        self.assertLess(
            existing.index("verify-prior-closures"),
            existing.index('mkdir -m 700 "$run_root"'),
        )
        self.assertIn('sha256sum "$prior_formal_log"', existing)
        self.assertIn('sha256sum "$prior_attempt_ledger"', existing)
        self.assertIn('sha256sum "$prior_failure_closure"', existing)
        prior = authority.load_authority()["prior_failed_engineering_attempt"]
        self.assertEqual(len(prior["legacy_snapshot"]["rows"]), 21)
        self.assertEqual(len(prior["legacy_run_tree"]["rows"]), 8)
        self.assertEqual(
            prior["legacy_snapshot"]["canonical_tree_digest"],
            "02d7a785dc121c22ae836c7352557482d0884938ba9ce724fa907e057f6aa853",
        )
        self.assertEqual(
            prior["legacy_run_tree"]["canonical_tree_digest"],
            "eb2167b674c94202f14b3d4684c6b791309d03884c52206bce1bcd9cef5d0181",
        )
        controller_step = authority.load_authority()["existing_allocation_contract"][
            "external_controller_step"
        ]
        self.assertEqual(controller_step["gpu_count"], 0)
        self.assertFalse(controller_step["foundation_or_tensor_framework_imports_permitted"])
        self.assertTrue(controller_step["static_source_tree_file_and_config_closure_only"])

    def test_formal_entrypoint_and_nested_existing_launch_fail_closed(self):
        formal = ROOT / "scripts/auh_actual_target_foundation_canary_formal_v3.sbatch"
        result = subprocess.run(
            ["bash", str(formal)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 64)
        self.assertIn("formal sbatch is not authorized", result.stderr)

        existing = (
            ROOT
            / "scripts/auh_actual_target_foundation_canary_existing_allocation_v3.sh"
        )
        for variable in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_STEPID"):
            with self.subTest(variable):
                environment = dict(os.environ)
                for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_STEPID"):
                    environment.pop(name, None)
                environment[variable] = "nested"
                result = subprocess.run(
                    ["bash", str(existing), TEST_JOB_ID],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 64)
                self.assertIn("must run from a login shell", result.stderr)


class StrictArtifactTests(unittest.TestCase):
    @staticmethod
    def model_closure_with_pointer(*, shape, pointer):
        rows = []
        for model in ("sam2", "cotracker", "dinov2", "vjepa2"):
            rows.append(
                {
                    "model": model,
                    "kind": "buffer",
                    "name": "empty_contract",
                    "shape": list(shape),
                    "dtype": "torch.float32",
                    "device": "cuda:0",
                    "data_ptr": pointer,
                    "value_sha256": hashlib.sha256(b"").hexdigest(),
                    "requires_grad": False,
                }
            )
        state = digested({"tensor_count": len(rows), "tensors": rows})
        return digested(
            {
                "mode": "real_frozen_full_tensor_closure",
                "verified": True,
                "before": state,
                "after": state,
                "exact_before_after_equality": True,
                "binding": {},
            }
        )

    def test_cache_freeze_rejects_extra_plain_directory(self):
        pairs = [{"pair_id": f"pair{index}"} for index in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            candidate = root / "candidate.json"
            candidate.write_text("{}\n", encoding="ascii")
            cache = root / "cache"
            cache.mkdir()
            for pair in pairs:
                (cache / f'{pair["pair_id"]}.json').write_text("{}\n", encoding="ascii")
            (cache / "raw_sidecar").mkdir()
            with mock.patch.object(
                controller.authority, "load_preregistration", return_value={"pairs": pairs}
            ):
                with self.assertRaises(controller.ControllerV3Error):
                    controller.freeze_candidate_cache(candidate, cache)

    def test_decoded_media_duplicate_key_is_rejected(self):
        decoded = authority.load_decode_receipt()
        rows = [
            {
                "r1b_ordinal": row["r1b_ordinal"],
                "role": row["role"],
                "compressed_sha256": row["compressed_sha256"],
                "frame_count": row["frame_count"],
                "shape_hwc": [720, 1280, 3],
                "dtype": "uint8",
                "decoded_rgb_sha256": row["decoded_rgb_sha256"],
            }
            for row in decoded["rows"]
        ]
        rows[-1] = dict(rows[0])
        body = {
            "verified": True,
            "decode_receipt_file_sha256": authority.file_sha256(
                authority.base.DECODE_RECEIPT_PATH
            ),
            "decode_receipt_self_sha256": decoded["decode_receipt_self_sha256"],
            "rows": rows,
        }
        with self.assertRaises(controller.ControllerV3Error):
            controller._validate_media(digested(body))

    def test_model_closure_rejects_missing_foundation_model_before_binding(self):
        tensor_rows = []
        for index, model in enumerate(("sam2", "cotracker", "dinov2")):
            tensor_rows.append(
                {
                    "model": model,
                    "kind": "parameter",
                    "name": "weight",
                    "shape": [1],
                    "dtype": "torch.float32",
                    "device": "cuda:0",
                    "data_ptr": index + 1,
                    "value_sha256": hashlib.sha256(model.encode()).hexdigest(),
                    "requires_grad": False,
                }
            )
        state = digested({"tensor_count": len(tensor_rows), "tensors": tensor_rows})
        closure = digested(
            {
                "mode": "real_frozen_full_tensor_closure",
                "verified": True,
                "before": state,
                "after": state,
                "exact_before_after_equality": True,
                "binding": {},
            }
        )
        with self.assertRaisesRegex(
            controller.ControllerV3Error, "every foundation model"
        ):
            controller._validate_model_closure(closure, "0" * 64)

    def test_model_closure_accepts_zero_pointer_for_empty_tensors(self):
        closure = self.model_closure_with_pointer(shape=(0, 3), pointer=0)
        with mock.patch.object(controller, "_validate_model_binding", return_value=None):
            controller._validate_model_closure(closure, "0" * 64)

    def test_model_closure_accepts_positive_pointer_for_empty_tensors(self):
        closure = self.model_closure_with_pointer(shape=(0, 3), pointer=17)
        with mock.patch.object(controller, "_validate_model_binding", return_value=None):
            controller._validate_model_closure(closure, "0" * 64)

    def test_model_closure_rejects_zero_pointer_for_nonempty_tensor(self):
        closure = self.model_closure_with_pointer(shape=(1, 3), pointer=0)
        with mock.patch.object(controller, "_validate_model_binding", return_value=None):
            with self.assertRaisesRegex(controller.ControllerV3Error, "device/data_ptr"):
                controller._validate_model_closure(closure, "0" * 64)

    def test_raw_production_is_independently_rebuilt_and_zero_is_valid(self):
        logical = dict(runtime.EXPECTED_LOGICAL_COUNTS)
        decoded_frames = sum(
            row["frame_count"] for row in authority.load_decode_receipt()["rows"]
        )
        observed = {
            "compressed_video_hash_requests": logical["media_decode"],
            "decoded_bgr_frames": decoded_frames,
            "decoded_rgb_frames": decoded_frames,
            "sam_ann_records_before_filter": 0,
            "sam_mask_coordinate_calls": 0,
            "dino_processor_tensor_items": logical["dinov2"],
            "dino_model_output_unique_storages": logical["dinov2"],
            "dino_filtered_ann_records": 0,
            "dino_positive_support_records": 0,
            "cotracker_membership_rows": 0,
            "vjepa_processor_tensor_items": logical["vjepa2"],
            "vjepa_model_output_unique_storages": 4 * logical["vjepa2"],
            "model_tensor_hash_requests": 8,
        }
        case_count = len(authority.load_preregistration()["pairs"])
        produced = {
            "compressed_video_hash_buffer": logical["media_decode"],
            "decoded_bgr_frame": decoded_frames,
            "decoded_rgb_frame": decoded_frames,
            "sam_ann_mask_pre_filter": 0,
            "sam_mask_c_contiguous_copy": 0,
            "sam_mask_coordinate_indices": 0,
            "dino_processor_input": 2 * logical["dinov2"],
            "dino_tokens": 2 * logical["dinov2"],
            "dino_mask_input": 0,
            "dino_mask_resized": 0,
            "dino_mask_cropped": 0,
            "dino_patch_weights": 0,
            "dino_patch_support": 0,
            "dino_pooled_descriptor": 0,
            "dino_pooled_descriptor_cpu": 0,
            "node_signature": logical["sam2"] // runtime.PHASES
            + 3 * case_count,
            "cotracker_video": 3 * logical["cotracker"],
            "cotracker_tracks": logical["cotracker"],
            "cotracker_visibility": logical["cotracker"],
            "cotracker_coordinates_cpu": 2 * logical["cotracker"],
            "cotracker_visibility_cpu": 2 * logical["cotracker"],
            "cotracker_group_coordinates": 0,
            "cotracker_group_visibility": 0,
            "track_signature": logical["cotracker"] + case_count,
            "edge_signature": logical["cotracker"],
            "drop_edge_signature": logical["cotracker"],
            "vjepa_processor_input": 2 * logical["vjepa2"],
            "vjepa_hidden": 6 * logical["vjepa2"],
            "vjepa_phase_signature": 2 * logical["vjepa2"],
            "model_hash_copy": 8,
        }
        required = authority.load_authority()["raw_inventory_required_categories"]
        self.assertEqual(set(produced), set(required))
        scope = authority.load_authority()["raw_ownership_contract"]
        body = {
            "schema_version": "actual-target-raw-inventory-v3",
            "required_categories": required,
            "opportunity_by_category": dict(produced),
            "produced_by_category": dict(produced),
            "registered_by_category": dict(produced),
            "zeroized_by_category": dict(produced),
            "failure_attempts_by_category": {name: 0 for name in required},
            "observed_counts": observed,
            "opportunity_total": sum(produced.values()),
            "produced_total": sum(produced.values()),
            "registered_total": sum(produced.values()),
            "zeroized_total": sum(produced.values()),
            "outstanding_count": 0,
            "missing_required_categories": [],
            "zero_produced_categories": sorted(
                name for name, count in produced.items() if count == 0
            ),
            "zero_produced_categories_are_valid_abstention": True,
            "observed_count_keys": list(runtime.RAW_OBSERVED_COUNT_KEYS),
            "model_output_unique_storage_multipliers": dict(
                runtime.MODEL_OUTPUT_UNIQUE_STORAGE_MULTIPLIERS
            ),
            "model_output_unique_storage_evidence_digest": authority.object_sha256(
                scope["model_output_unique_storage_evidence"]
            ),
            "production_binding_rule": scope["production_binding_rule"],
            "in_scope_storage_boundary": scope["included_storage_scope"],
            "excluded_ephemeral_workspace_boundary": scope[
                "excluded_ephemeral_workspace_scope"
            ],
            "recursive_best_effort_scrub": True,
            "verified": True,
        }
        raw = digested(body)
        forward = {"logical_counts": logical}
        model = {"before": {"tensor_count": 4}}
        controller._validate_raw_inventory(raw, forward, model)
        self.assertEqual(produced["node_signature"], 24)

        nonzero_body = json.loads(json.dumps(body))
        nonzero_body["observed_counts"].update(
            {
                "sam_ann_records_before_filter": 1,
                "sam_mask_coordinate_calls": 1,
                "dino_filtered_ann_records": 1,
                "dino_positive_support_records": 1,
            }
        )
        increments = {
            "sam_ann_mask_pre_filter": 1,
            "sam_mask_c_contiguous_copy": 1,
            "sam_mask_coordinate_indices": 1,
            "dino_mask_input": 1,
            "dino_mask_resized": 1,
            "dino_mask_cropped": 1,
            "dino_patch_weights": 1,
            "dino_patch_support": 1,
            "dino_pooled_descriptor": 3,
            "dino_pooled_descriptor_cpu": 2,
        }
        for counter_name in (
            "opportunity_by_category",
            "produced_by_category",
            "registered_by_category",
            "zeroized_by_category",
        ):
            for category, increment in increments.items():
                nonzero_body[counter_name][category] += increment
        for total_name in (
            "opportunity_total",
            "produced_total",
            "registered_total",
            "zeroized_total",
        ):
            nonzero_body[total_name] += sum(increments.values())
        nonzero_body["zero_produced_categories"] = sorted(
            name
            for name, count in nonzero_body["produced_by_category"].items()
            if count == 0
        )
        controller._validate_raw_inventory(
            digested(nonzero_body), forward, model
        )
        self.assertEqual(
            nonzero_body["produced_by_category"]["sam_mask_coordinate_indices"],
            nonzero_body["observed_counts"]["sam_mask_coordinate_calls"],
        )

        forged = dict(raw)
        forged["produced_by_category"] = dict(produced)
        forged["produced_by_category"]["sam_ann_mask_pre_filter"] = 1
        forged.pop("digest")
        forged = digested(forged)
        with self.assertRaises(controller.ControllerV3Error):
            controller._validate_raw_inventory(forged, forward, model)

        typed_forgery = dict(raw)
        typed_forgery["failure_attempts_by_category"] = {
            **body["failure_attempts_by_category"],
            "sam_ann_mask_pre_filter": False,
        }
        typed_forgery.pop("digest")
        typed_forgery = digested(typed_forgery)
        with self.assertRaises(controller.ControllerV3Error):
            controller._validate_raw_inventory(typed_forgery, forward, model)

    def test_source_tree_mutation_omission_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            specs = []
            roots = []
            for index in range(3):
                root = base / f"tree{index}"
                root.mkdir()
                (root / "a.py").write_text(f"VALUE={index}\n", encoding="ascii")
                (root / "b.py").write_text(f"OTHER={index}\n", encoding="ascii")
                rows = controller._controller_source_tree_rows(root, ".py")
                specs.append(
                    {
                        "role": f"role{index}",
                        "root": str(root),
                        "suffix": ".py",
                        "file_count": len(rows),
                        "manifest_sha256": authority.object_sha256(rows),
                    }
                )
                roots.append(root)
            patched = {"foundation_source_tree_authority": specs}
            with mock.patch.object(controller.authority, "load_authority", return_value=patched):
                expected = controller._recompute_foundation_source_trees()
                self.assertEqual(expected, authority.foundation_source_tree_closure())
                self.assertTrue(expected["verified"])
                (roots[0] / "a.py").write_text("MUTATED=True\n", encoding="ascii")
                with self.assertRaises(controller.ControllerV3Error):
                    controller._recompute_foundation_source_trees()
            (roots[0] / "a.py").write_text("VALUE=0\n", encoding="ascii")
            (roots[1] / "b.py").unlink()
            with mock.patch.object(controller.authority, "load_authority", return_value=patched):
                with self.assertRaises(controller.ControllerV3Error):
                    controller._recompute_foundation_source_trees()
            (roots[1] / "b.py").write_text("OTHER=1\n", encoding="ascii")
            (roots[2] / "alias.py").symlink_to(roots[2] / "a.py")
            with mock.patch.object(controller.authority, "load_authority", return_value=patched):
                with self.assertRaises(controller.ControllerV3Error):
                    controller._recompute_foundation_source_trees()

    def test_sam_layout_source_evidence_binds_file_and_exact_line_span(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = (root / "amg.py", root / "automatic_mask_generator.py")
            payloads = (
                b"header\nallocate\nreshape\ntranspose\nfooter\n",
                b"header\nbinary-mask\nrle-to-mask\nannotation\nfooter\n",
            )
            roles = (
                "uncompressed_rle_to_mask",
                "automatic_binary_mask_return",
            )
            sources = []
            for role, path, payload in zip(roles, paths, payloads):
                path.write_bytes(payload)
                lines = payload.splitlines(keepends=True)
                start, end = 2, 4
                span = b"".join(lines[start - 1 : end])
                sources.append(
                    {
                        "role": role,
                        "path": str(path),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "line_start": start,
                        "line_end": end,
                        "line_span_sha256": hashlib.sha256(span).hexdigest(),
                    }
                )
            patched = {
                "v3r3_engineering_repair_contract": {
                    "sam_pinned_binary_mask_source_evidence": {
                        "claim_boundary": (
                            "the full-storage transpose layout is derived from "
                            "pinned SAM2 source bytes, not inferred from the "
                            "V3R2 compound failure log"
                        ),
                        "sources": sources,
                    }
                }
            }
            with mock.patch.object(
                controller.authority, "load_authority", return_value=patched
            ):
                receipt = controller._verify_v3r3_sam_layout_source_evidence()
                self.assertTrue(receipt["verified"])
                paths[0].write_bytes(payloads[0].replace(b"reshape", b"reshaped"))
                with self.assertRaisesRegex(
                    controller.ControllerV3Error, "pinned SAM source file"
                ):
                    controller._verify_v3r3_sam_layout_source_evidence()

    def test_static_module_path_handles_package_root_and_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            sam_root = base / "sam2"
            sam_source = sam_root / "modeling" / "sam2_base.py"
            sam_source.parent.mkdir(parents=True)
            sam_source.write_text("VALUE=1\n", encoding="ascii")
            cot_root = base / "co-tracker"
            cot_source = cot_root / "cotracker" / "predictor.py"
            cot_source.parent.mkdir(parents=True)
            cot_source.write_text("VALUE=2\n", encoding="ascii")
            tf_root = base / "transformers"
            tf_source = tf_root / "models" / "dinov2" / "modeling_dinov2.py"
            tf_source.parent.mkdir(parents=True)
            tf_source.write_text("VALUE=3\n", encoding="ascii")
            specs = {
                "foundation_source_tree_authority": [
                    {"role": "sam2_python_tree", "root": str(sam_root)},
                    {"role": "cotracker_python_tree", "root": str(cot_root)},
                    {"role": "transformers_python_tree", "root": str(tf_root)},
                ]
            }
            with mock.patch.object(
                controller.authority, "load_authority", return_value=specs
            ):
                self.assertEqual(
                    controller._static_module_source("sam2.modeling.sam2_base")[0],
                    sam_source,
                )
                self.assertEqual(
                    controller._static_module_source("cotracker.predictor")[0],
                    cot_source,
                )
                self.assertEqual(
                    controller._static_module_source(
                        "transformers.models.dinov2.modeling_dinov2"
                    )[0],
                    tf_source,
                )


class LegacyClosureHarness:
    """Small generic legacy snapshot/run closure; deliberately not 17 files."""

    def __init__(self, base: Path):
        self.base = base
        self.source_root = base / "legacy-source"
        self.source_root.mkdir()
        (self.source_root / "scripts").mkdir()
        self.source_files = {
            "payload.py": b"VALUE = 1\n",
            "scripts/tool.sh": b"#!/usr/bin/env bash\nexit 0\n",
        }
        for relative, payload in self.source_files.items():
            path = self.source_root / relative
            path.write_bytes(payload)
            os.chmod(path, 0o755 if relative.endswith(".sh") else 0o644)

        self.snapshot_root = base / "legacy-snapshot"
        self.snapshot_root.mkdir(mode=0o700)
        (self.snapshot_root / "scripts").mkdir(mode=0o700)
        snapshot_rows = []
        source_rows = []
        for relative, payload in self.source_files.items():
            source = self.source_root / relative
            target = self.snapshot_root / relative
            mode = 0o555 if relative.endswith(".sh") else 0o444
            runtime.create_only_bytes(target, payload, mode)
            digest = hashlib.sha256(payload).hexdigest()
            snapshot_rows.append(
                {
                    "relative_path": relative,
                    "sha256": digest,
                    "byte_count": len(payload),
                    "snapshot_mode": mode,
                }
            )
            source_rows.append(
                {
                    "relative_path": relative,
                    "original_path": str(source),
                    "original_mode": stat.S_IMODE(source.stat().st_mode),
                    "byte_count": len(payload),
                    "sha256": digest,
                    "original_path_components": [],
                }
            )
        source_body = {
            "source_root": str(self.source_root),
            "file_count": len(source_rows),
            "files": source_rows,
            "no_symlink_laundering": True,
        }
        source_closure = {
            **source_body,
            "digest": authority.object_sha256(source_body),
        }
        manifest_body = {
            "schema_version": snapshot.SCHEMA,
            "source_closure": source_closure,
            "snapshot_root": str(self.snapshot_root),
            "snapshot_file_count": len(snapshot_rows),
            "snapshot_files": snapshot_rows,
            "snapshot_directory_mode": 0o555,
            "snapshot_file_modes": {"data": 0o444, "executable": 0o555},
            "no_symlinks": True,
            "immutable_permissions_applied": True,
        }
        self.manifest_value = {
            **manifest_body,
            "manifest_self_sha256": authority.object_sha256(manifest_body),
        }
        self.manifest = self.snapshot_root / snapshot.MANIFEST_NAME
        runtime.create_only_json(self.manifest, self.manifest_value)
        os.chmod(self.snapshot_root / "scripts", 0o555)
        os.chmod(self.snapshot_root, 0o555)

        self.run_root = base / "legacy-run"
        self.run_root.mkdir(mode=0o700)
        self.cache = self.run_root / "cache"
        self.cache.mkdir(mode=0o700)
        ledger_body = {
            "failure_reasons": ["ControllerV3Error:legacy failure"],
            "engineering_failure": True,
        }
        self.ledger_value = digested(ledger_body)
        self.ledger = self.run_root / "attempt_ledger.json"
        runtime.create_only_json(self.ledger, self.ledger_value)
        self.controller_argv = self.run_root / "controller_argv.nul"
        self.rank_argv = self.run_root / "rank_argv.nul"
        self.srun_argv = self.run_root / "srun_argv.nul"
        controller.write_nul_argv(self.controller_argv, ["srun", "controller"])
        controller.write_nul_argv(self.rank_argv, ["rank-wrapper", "candidate"])
        controller.write_nul_argv(self.srun_argv, ["srun", "rank-wrapper"])
        self.formal = self.run_root / "formal.log"
        runtime.create_only_bytes(self.formal, b"legacy formal failure\n")
        self.step = self.run_root / "step_meta.json"
        runtime.create_only_json(self.step, {"legacy_step": True})
        os.chmod(self.cache, 0o555)
        os.chmod(self.run_root, 0o555)

        snapshot_tree = controller._scan_legacy_tree(self.snapshot_root)
        run_tree = controller._scan_legacy_tree(self.run_root)
        snapshot_tree_value = {
            "root": str(self.snapshot_root),
            "rows": snapshot_tree,
        }
        run_tree_value = {"root": str(self.run_root), "rows": run_tree}
        self.snapshot_spec = {
            **snapshot_tree_value,
            "canonical_tree_digest": authority.object_sha256(
                snapshot_tree_value
            ),
            "manifest_relative_path": snapshot.MANIFEST_NAME,
            "manifest_file_sha256": hashlib.sha256(
                self.manifest.read_bytes()
            ).hexdigest(),
            "manifest_self_sha256": self.manifest_value[
                "manifest_self_sha256"
            ],
            "manifest_schema_version": snapshot.SCHEMA,
            "source_closure_digest": source_closure["digest"],
            "snapshot_file_count": len(snapshot_rows),
        }
        self.run_spec = {
            **run_tree_value,
            "canonical_tree_digest": authority.object_sha256(run_tree_value),
            "candidate_absent": True,
            "completion_seal_absent": True,
        }
        self.authority_value = {
            "fixed_paths": {
                "fresh_formal_run_root": str(base / "fresh-v3r2")
            },
            "prior_failed_engineering_attempt": {
                "immutable_preservation_required": True,
                "relaunch_or_reuse_forbidden": True,
                "run_root": str(self.run_root),
                "legacy_snapshot": self.snapshot_spec,
                "legacy_run_tree": self.run_spec,
                "formal_log": {
                    "path": str(self.formal),
                    "sha256": hashlib.sha256(self.formal.read_bytes()).hexdigest(),
                    "mode": 0o444,
                },
                "attempt_ledger": {
                    "path": str(self.ledger),
                    "sha256": hashlib.sha256(self.ledger.read_bytes()).hexdigest(),
                    "mode": 0o444,
                    "digest": self.ledger_value["digest"],
                    "failure_reasons": self.ledger_value["failure_reasons"],
                },
            },
        }

    def verify(self):
        with mock.patch.object(
            controller.authority,
            "load_authority",
            return_value=self.authority_value,
        ):
            return controller._verify_v3r1_failed_attempt()

    def thaw(self):
        thaw_snapshot_tree(self.base)


class ScratchClosureTests(unittest.TestCase):
    @staticmethod
    def fixed(root: Path):
        return {
            "fixed_paths": {
                "fresh_formal_run_root": str(root),
                "miopen_user_dirname": "miopen-user",
                "miopen_custom_cache_dirname": "miopen-custom",
                "miopen_scratch_closure_filename": "miopen_scratch_closure.json",
            }
        }

    def test_scratch_freeze_is_recursive_stable_create_only_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve() / "run"
            run_root.mkdir(mode=0o700)
            user = run_root / "miopen-user"
            custom = run_root / "miopen-custom"
            user.mkdir(mode=0o700)
            custom.mkdir(mode=0o700)
            nested = user / "db"
            nested.mkdir(mode=0o700)
            (nested / "find-db.sqlite").write_bytes(b"sqlite-evidence")
            (custom / "kernel.bin").write_bytes(b"kernel-evidence")
            closure = run_root / "miopen_scratch_closure.json"
            try:
                with mock.patch.object(
                    controller.authority,
                    "load_authority",
                    return_value=self.fixed(run_root),
                ):
                    value = controller.freeze_miopen_scratch(
                        user, custom, closure
                    )
                    self.assertEqual(value["tree_count"], 2)
                    self.assertEqual(stat.S_IMODE(user.stat().st_mode), 0o555)
                    self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o555)
                    self.assertEqual(
                        stat.S_IMODE((nested / "find-db.sqlite").stat().st_mode),
                        0o444,
                    )
                    verified, record = controller._verify_miopen_scratch_closure(
                        closure
                    )
                    self.assertEqual(verified, value)
                    self.assertEqual(record["mode"], 0o444)
                    with self.assertRaises(controller.ControllerV3Error):
                        controller.freeze_miopen_scratch(user, custom, closure)

                    os.chmod(user, 0o755)
                    (user / "late-sidecar").write_bytes(b"forged")
                    os.chmod(user, 0o555)
                    with self.assertRaises(controller.ControllerV3Error):
                        controller._verify_miopen_scratch_closure(closure)
            finally:
                thaw_snapshot_tree(run_root)

    def test_scratch_rejects_wrong_root_preexisting_closure_and_symlink(self):
        scenarios = ("wrong_root", "preexisting_closure", "symlink")
        for scenario in scenarios:
            with self.subTest(scenario), tempfile.TemporaryDirectory() as directory:
                run_root = Path(directory).resolve() / "run"
                run_root.mkdir(mode=0o700)
                user = run_root / "miopen-user"
                custom = run_root / "miopen-custom"
                user.mkdir(mode=0o700)
                custom.mkdir(mode=0o700)
                closure = run_root / "miopen_scratch_closure.json"
                observed_user = user
                if scenario == "wrong_root":
                    observed_user = run_root / "old-run-miopen-user"
                    observed_user.mkdir(mode=0o700)
                elif scenario == "preexisting_closure":
                    closure.write_text("{}\n", encoding="ascii")
                else:
                    (user / "linked").symlink_to(custom, target_is_directory=True)
                with mock.patch.object(
                    controller.authority,
                    "load_authority",
                    return_value=self.fixed(run_root),
                ):
                    with self.assertRaises(controller.ControllerV3Error):
                        controller.freeze_miopen_scratch(
                            observed_user, custom, closure
                        )

    def test_rank_step_meta_requires_exact_fresh_miopen_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            snapshot_root = base / "snapshot"
            snapshot_root.mkdir()
            run_root = base / "run"
            run_root.mkdir(mode=0o700)
            user = run_root / "miopen-user"
            custom = run_root / "miopen-custom"
            user.mkdir(mode=0o700)
            custom.mkdir(mode=0o700)
            rank_argv = run_root / "rank.nul"
            controller.write_nul_argv(rank_argv, ["rank-wrapper", "arg"])
            fixed = self.fixed(run_root)
            fixed["fixed_paths"]["planned_preflip_snapshot_root"] = str(
                snapshot_root
            )
            receipt = {
                "verified": True,
                "digest": "1" * 64,
                "manifest_file_sha256": "2" * 64,
            }
            environment = {
                "SLURM_JOB_ID": TEST_JOB_ID,
                "SLURM_STEP_ID": f"{TEST_JOB_ID}.0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "ROCR_VISIBLE_DEVICES": "0",
                "MIOPEN_USER_DB_PATH": str(user),
                "MIOPEN_CUSTOM_CACHE_DIR": str(custom),
            }
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        controller.authority,
                        "load_authority",
                        return_value=fixed,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller.snapshot_v3,
                        "verify_snapshot",
                        return_value=receipt,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "__file__",
                        str(
                            snapshot_root
                            / "actual_target_foundation_controller_v3.py"
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.dict(os.environ, environment, clear=True)
                )
                result = controller.write_step_meta(
                    run_root / "step.json",
                    run_root / "candidate.json",
                    run_root / "cache",
                    rank_argv,
                    snapshot_root,
                    user,
                    custom,
                )
                self.assertTrue(result["miopen_directories_initially_empty"])
                self.assertFalse(result["miopen_disable_cache_present"])

                os.environ["MIOPEN_DISABLE_CACHE"] = "1"
                with self.assertRaises(controller.ControllerV3Error):
                    controller.write_step_meta(
                        run_root / "step-disabled.json",
                        run_root / "candidate.json",
                        run_root / "cache",
                        rank_argv,
                        snapshot_root,
                        user,
                        custom,
                    )

    def test_generic_legacy_snapshot_and_full_failed_run_close_without_current_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = LegacyClosureHarness(Path(directory).resolve())
            try:
                receipt = harness.verify()
                self.assertTrue(receipt["verified"])
                self.assertTrue(receipt["candidate_absent"])
                self.assertTrue(receipt["completion_seal_absent"])
                self.assertTrue(
                    receipt["legacy_snapshot"]["generic_legacy_schema_used"]
                )
                self.assertTrue(
                    receipt["legacy_snapshot"][
                        "current_payload_authority_not_used"
                    ]
                )
                self.assertEqual(
                    receipt["legacy_snapshot"]["snapshot_file_count"], 2
                )
            finally:
                harness.thaw()

    def test_legacy_snapshot_and_run_mutations_all_fail_closed(self):
        scenarios = (
            "same_bytes_new_inode",
            "snapshot_extra_file",
            "snapshot_missing_member",
            "snapshot_member_changed",
            "snapshot_manifest_changed",
            "snapshot_source_digest_changed",
            "snapshot_root_writable",
            "run_extra_file",
            "run_missing_controller_argv",
            "run_changed_rank_argv",
            "run_changed_step_meta",
            "run_cache_member",
            "run_root_writable",
            "run_cache_writable",
        )
        for scenario in scenarios:
            with self.subTest(scenario), tempfile.TemporaryDirectory() as directory:
                harness = LegacyClosureHarness(Path(directory).resolve())
                try:
                    if scenario == "same_bytes_new_inode":
                        target = harness.snapshot_root / "payload.py"
                        payload = target.read_bytes()
                        old_inode = target.stat().st_ino
                        os.chmod(harness.snapshot_root, 0o755)
                        replacement = harness.snapshot_root / "replacement.tmp"
                        replacement.write_bytes(payload)
                        os.chmod(replacement, 0o444)
                        self.assertNotEqual(old_inode, replacement.stat().st_ino)
                        os.replace(replacement, target)
                        os.chmod(harness.snapshot_root, 0o555)
                        self.assertNotEqual(old_inode, target.stat().st_ino)
                    elif scenario == "snapshot_extra_file":
                        os.chmod(harness.snapshot_root, 0o755)
                        runtime.create_only_bytes(
                            harness.snapshot_root / "extra.bin", b"extra"
                        )
                        os.chmod(harness.snapshot_root, 0o555)
                    elif scenario == "snapshot_missing_member":
                        os.chmod(harness.snapshot_root, 0o755)
                        (harness.snapshot_root / "payload.py").unlink()
                        os.chmod(harness.snapshot_root, 0o555)
                    elif scenario == "snapshot_member_changed":
                        target = harness.snapshot_root / "payload.py"
                        os.chmod(target, 0o644)
                        target.write_bytes(b"VALUE = 2\n")
                        os.chmod(target, 0o444)
                    elif scenario in {
                        "snapshot_manifest_changed",
                        "snapshot_source_digest_changed",
                    }:
                        value = json.loads(harness.manifest.read_text())
                        if scenario == "snapshot_manifest_changed":
                            value["snapshot_directory_mode"] = 0o755
                        else:
                            value["source_closure"]["digest"] = "0" * 64
                        os.chmod(harness.manifest, 0o644)
                        harness.manifest.write_text(
                            json.dumps(value, sort_keys=True) + "\n",
                            encoding="ascii",
                        )
                        os.chmod(harness.manifest, 0o444)
                    elif scenario == "snapshot_root_writable":
                        os.chmod(harness.snapshot_root, 0o755)
                    elif scenario == "run_extra_file":
                        os.chmod(harness.run_root, 0o755)
                        runtime.create_only_bytes(
                            harness.run_root / "extra.bin", b"extra"
                        )
                        os.chmod(harness.run_root, 0o555)
                    elif scenario == "run_missing_controller_argv":
                        os.chmod(harness.run_root, 0o755)
                        harness.controller_argv.unlink()
                        os.chmod(harness.run_root, 0o555)
                    elif scenario == "run_changed_rank_argv":
                        os.chmod(harness.rank_argv, 0o644)
                        harness.rank_argv.write_bytes(b"changed\0")
                        os.chmod(harness.rank_argv, 0o444)
                    elif scenario == "run_changed_step_meta":
                        os.chmod(harness.step, 0o644)
                        harness.step.write_text('{"changed":true}\n', encoding="ascii")
                        os.chmod(harness.step, 0o444)
                    elif scenario == "run_cache_member":
                        os.chmod(harness.cache, 0o755)
                        runtime.create_only_bytes(harness.cache / "sidecar", b"x")
                        os.chmod(harness.cache, 0o555)
                    elif scenario == "run_root_writable":
                        os.chmod(harness.run_root, 0o755)
                    else:
                        os.chmod(harness.cache, 0o755)
                    with self.assertRaises(controller.ControllerV3Error):
                        harness.verify()
                finally:
                    harness.thaw()


class SealHarness:
    def __init__(self, base: Path, diagnostic_pass: bool = True):
        self.base = base
        self.snapshot_root = base / "snapshot"
        self.snapshot_root.mkdir()
        (self.snapshot_root / "scripts").mkdir()
        self.controller_wrapper = (
            self.snapshot_root
            / "scripts"
            / "auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
        )
        self.controller_wrapper.write_text("#!/usr/bin/env bash\n", encoding="ascii")
        os.chmod(self.controller_wrapper, 0o755)
        self.run_root = base / "run"
        self.run_root.mkdir()
        self.cache = self.run_root / "cache"
        self.cache.mkdir()
        os.chmod(self.cache, 0o555)
        self.miopen_user = self.run_root / "miopen-user"
        self.miopen_custom = self.run_root / "miopen-custom"
        self.miopen_user.mkdir(mode=0o700)
        self.miopen_custom.mkdir(mode=0o700)
        self.miopen_scratch_closure = self.run_root / "miopen_scratch_closure.json"
        self.candidate = self.run_root / "candidate.json"
        runtime.create_only_json(self.candidate, {"candidate": True})
        self.seal = self.run_root / "completion_seal.json"
        self.ledger = self.run_root / "attempt_ledger.json"
        self.log = self.run_root / "formal.log"
        runtime.create_only_bytes(self.log, b"formal\n")
        self.srun_argv = self.run_root / "srun_argv.nul"
        self.rank_argv = self.run_root / "rank_argv.nul"
        self.step_meta = self.run_root / "step_meta.json"
        self.controller_argv = self.run_root / "controller_argv.nul"
        self.controller_step_meta = self.run_root / "controller_step_meta.json"
        self.wrapper = str(
            self.snapshot_root
            / "scripts"
            / "auh_actual_target_foundation_canary_rank_wrapper_v3.sh"
        )
        self.rank_values = [
            self.wrapper,
            str(self.candidate),
            str(self.cache),
            str(self.step_meta),
            str(self.rank_argv),
            str(self.snapshot_root),
            str(self.miopen_user),
            str(self.miopen_custom),
        ]
        rank_record = controller.write_nul_argv(self.rank_argv, self.rank_values)
        self.srun_values = [
            "srun",
            "--jobid",
            TEST_JOB_ID,
            "--exclusive",
            "--exact",
            "--immediate=60",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=16",
            "--gres=gpu:mi210:1",
            "--mem=56G",
            "--export=ALL,LOCAL_RANK=0,WORLD_SIZE=1",
            *self.rank_values,
        ]
        controller.write_nul_argv(self.srun_argv, self.srun_values)
        self.snapshot_receipt = {
            "verified": True,
            "digest": "1" * 64,
            "manifest_file_sha256": "2" * 64,
        }
        scratch_records = {
            "MIOPEN_USER_DB_PATH": controller._plain_directory_record(
                self.miopen_user
            ),
            "MIOPEN_CUSTOM_CACHE_DIR": controller._plain_directory_record(
                self.miopen_custom
            ),
        }
        self.scratch_closure_value = digested(
            {
                "schema_version": "actual-target-foundation-miopen-scratch-closure-v3r3",
                "trees": [
                    {
                        "root": str(path),
                        "root_device": scratch_records[name]["device"],
                        "root_inode": scratch_records[name]["inode"],
                        "root_mode": 0o555,
                    }
                    for name, path in (
                        ("MIOPEN_USER_DB_PATH", self.miopen_user),
                        ("MIOPEN_CUSTOM_CACHE_DIR", self.miopen_custom),
                    )
                ],
                "tree_count": 2,
                "all_plain_no_symlinks": True,
                "all_files_mode_0444": True,
                "all_directories_mode_0555": True,
            }
        )
        runtime.create_only_json(
            self.miopen_scratch_closure, self.scratch_closure_value
        )
        step_body = {
            "schema_version": "actual-target-foundation-step-meta-v3",
            "slurm_job_id": TEST_JOB_ID,
            "slurm_step_id": f"{TEST_JOB_ID}.0",
            "local_rank": 0,
            "world_size": 1,
            "rocr_visible_devices": "3",
            "hostname": "test-node",
            "candidate_path": str(self.candidate),
            "cache_dir": str(self.cache),
            "rank_argv_path": str(self.rank_argv),
            "rank_argv_sha256": rank_record["sha256"],
            "rank_argv_argc": rank_record["argc"],
            "snapshot_root": str(self.snapshot_root),
            "snapshot_receipt_digest": self.snapshot_receipt["digest"],
            "snapshot_manifest_file_sha256": self.snapshot_receipt[
                "manifest_file_sha256"
            ],
            "miopen_environment": {
                "MIOPEN_USER_DB_PATH": str(self.miopen_user),
                "MIOPEN_CUSTOM_CACHE_DIR": str(self.miopen_custom),
            },
            "miopen_disable_cache_present": False,
            "miopen_directories_initially_empty": True,
            "miopen_directory_records": scratch_records,
        }
        runtime.create_only_json(self.step_meta, digested(step_body))
        exact_paths = {
            "candidate": self.candidate,
            "cache": self.cache,
            "seal": self.seal,
            "attempt_ledger": self.ledger,
            "log": self.log,
            "srun_argv": self.srun_argv,
            "rank_argv": self.rank_argv,
            "step_meta": self.step_meta,
            "controller_argv": self.controller_argv,
            "controller_step_meta": self.controller_step_meta,
            "miopen_user": self.miopen_user,
            "miopen_custom": self.miopen_custom,
            "miopen_scratch_closure": self.miopen_scratch_closure,
        }
        controller_command = controller._expected_controller_command(
            exact_paths=exact_paths,
            expected_contract_digest="c" * 64,
            srun_exit_code=0,
            tee_exit_code=0,
            expected_job_id=TEST_JOB_ID,
            snapshot_root=self.snapshot_root,
        )
        self.controller_values = controller._expected_controller_srun_argv(
            command=controller_command,
            expected_job_id=TEST_JOB_ID,
        )[1]
        controller.write_nul_argv(self.controller_argv, self.controller_values)
        self.candidate_value = {
            "aggregate": {
                "diagnostic_canary_pass": diagnostic_pass,
                "passed_case_count": 4 if diagnostic_pass else 3,
            },
            "device_closure": {
                "rocr_visible_devices": "3",
                "miopen_scratch_binding": {
                    "environment": {
                        "MIOPEN_USER_DB_PATH": str(self.miopen_user),
                        "MIOPEN_CUSTOM_CACHE_DIR": str(self.miopen_custom),
                    },
                    "miopen_disable_cache_present": False,
                    "directories": scratch_records,
                },
            },
            "digest": "3" * 64,
        }

    def fixed_authority(self):
        return {
            "fixed_paths": {
                "planned_preflip_snapshot_root": str(self.snapshot_root),
                "fresh_formal_run_root": str(self.run_root),
                "cache_dirname": "cache",
                "miopen_user_dirname": "miopen-user",
                "miopen_custom_cache_dirname": "miopen-custom",
                "miopen_scratch_closure_filename": "miopen_scratch_closure.json",
                "candidate_filename": "candidate.json",
                "seal_filename": "completion_seal.json",
                "attempt_ledger_filename": "attempt_ledger.json",
                "formal_log_filename": "formal.log",
                "srun_argv_filename": "srun_argv.nul",
                "rank_argv_filename": "rank_argv.nul",
                "step_meta_filename": "step_meta.json",
                "controller_argv_filename": "controller_argv.nul",
                "controller_step_meta_filename": "controller_step_meta.json",
            },
            "existing_allocation_contract": {
                "approved_job_ids": [147871, 147873, 147881]
            },
            "authorized_launch_mode": "existing_allocation_only",
        }

    def call(
        self,
        *,
        srun=0,
        tee=0,
        env_job=TEST_JOB_ID,
        candidate=None,
        device_environment=None,
        unset_device_names=(),
        torch_imported=False,
        foundation_imports=None,
    ):
        candidate_value = self.candidate_value if candidate is None else candidate
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    controller.authority,
                    "load_authority",
                    return_value=self.fixed_authority(),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "_forbidden_foundation_imports",
                    return_value=[]
                    if foundation_imports is None
                    else list(foundation_imports),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller.snapshot_v3,
                    "verify_snapshot",
                    return_value=self.snapshot_receipt,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller.runtime,
                    "launch_contract",
                    return_value={"digest": "c" * 64},
                )
            )
            stack.enter_context(mock.patch.object(
                controller,
                "_validate_candidate",
                return_value=(
                    candidate_value,
                    controller.stable_file_record(self.candidate),
                    [],
                ),
            ))
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "_verify_miopen_scratch_closure",
                    return_value=(
                        self.scratch_closure_value,
                        controller.stable_file_record(
                            self.miopen_scratch_closure
                        ),
                    ),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "_verify_prior_failed_attempt",
                    return_value={"verified": True, "digest": "9" * 64},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "__file__",
                    str(self.snapshot_root / "actual_target_foundation_controller_v3.py"),
                )
            )
            stack.enter_context(
                mock.patch.object(controller.socket, "gethostname", return_value="test-node")
            )
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "_torch_is_imported",
                    side_effect=torch_imported
                    if isinstance(torch_imported, list)
                    else None,
                    return_value=torch_imported
                    if not isinstance(torch_imported, list)
                    else False,
                )
            )
            visibility = {
                "CUDA_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
            }
            if device_environment is not None:
                visibility.update(device_environment)
            environment = dict(os.environ)
            environment.update(
                {
                    "SLURM_JOB_ID": env_job,
                    "SLURM_STEP_ID": f"{TEST_JOB_ID}.1",
                    **visibility,
                }
            )
            for name in unset_device_names:
                environment.pop(name, None)
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    environment,
                    clear=True,
                )
            )
            return controller.seal_outcome(
                candidate_path=self.candidate,
                cache_dir=self.cache,
                seal_path=self.seal,
                expected_contract_digest="c" * 64,
                srun_exit_code=srun,
                tee_exit_code=tee,
                expected_job_id=TEST_JOB_ID,
                step_meta_path=self.step_meta,
                formal_log_path=self.log,
                srun_argv_path=self.srun_argv,
                rank_argv_path=self.rank_argv,
                controller_argv_path=self.controller_argv,
                controller_step_meta_path=self.controller_step_meta,
                miopen_scratch_closure_path=self.miopen_scratch_closure,
                snapshot_root=self.snapshot_root,
                attempt_ledger_path=self.ledger,
            )


class ControllerOutcomeTests(unittest.TestCase):
    def test_complete_null_candidate_recomputes_then_seals_valid_rejected(self):
        prereg = authority.load_preregistration()
        evidences = []
        for pair in prereg["pairs"]:
            branches = {
                branch: {field: None for field in fields}
                for branch, fields in authority.BRANCH_FIELDS.items()
            }
            branches["node"].update(
                {
                    "phase_cardinalities": [0] * runtime.PHASES,
                    "mechanically_valid_phases": 0,
                }
            )
            branches["track"].update(
                {
                    "assigned_track_count": 0,
                    "assigned_point_count": 0,
                    "minimum_same_track_member_phases_observed": 0,
                    "visible_and_member_fraction": None,
                    "per_phase_visible_member_counts": [0] * runtime.PHASES,
                }
            )
            branches["edge"].update(
                {
                    name: [0] * runtime.PHASES
                    for name in (
                        "per_phase_active_counts",
                        "per_phase_birth_counts",
                        "per_phase_persist_counts",
                        "per_phase_death_counts",
                        "per_phase_valid_velocity_counts",
                        "per_phase_qualified_lifecycle_counts",
                    )
                }
            )
            evidences.append(
                authority.CaseEvidenceV3(
                    family=pair["family"],
                    pair_id=pair["pair_id"],
                    branches=branches,
                )
            )
        rows = [authority.evaluate_case(evidence, prereg) for evidence in evidences]
        aggregate = authority.aggregate_canary(rows, evidences, prereg)
        self.assertFalse(aggregate["diagnostic_canary_pass"])
        self.assertEqual(aggregate["passed_case_count"], 0)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            candidate_root = base / "validated_candidate"
            candidate_root.mkdir()
            cache = candidate_root / "cache"
            cache.mkdir()
            mechanical = []
            for evidence, row in zip(evidences, rows):
                mapping = evidence.to_mapping()
                mechanical.append(
                    {**mapping, "digest": authority.object_sha256(mapping)}
                )
                runtime.create_only_json(
                    cache / f"{evidence.pair_id}.json",
                    runtime._case_receipt(evidence, row),
                )
            os.chmod(cache, 0o555)
            source_closure = {"mode": "test-static-source-closure"}
            forward = digested(
                {
                    "logical_counts": runtime.EXPECTED_LOGICAL_COUNTS,
                    "actual_forward_hook_counts": runtime.EXPECTED_HOOK_COUNTS,
                    "expected_logical_counts": runtime.EXPECTED_LOGICAL_COUNTS,
                    "expected_actual_forward_hook_counts": runtime.EXPECTED_HOOK_COUNTS,
                    "verified": True,
                }
            )
            hydra = digested({"verified": True})
            device = digested(
                {
                    "mode": "real_one_device",
                    "verified": True,
                    "type": "cuda",
                    "index": 0,
                    "visible_device_count": 1,
                    "name": "test-mi210",
                    "rocr_visible_devices": "3",
                }
            )
            assets = {"foundation_source_trees": {"verified": True}}
            body = {
                "schema_version": runtime.SCHEMA,
                "experiment_id": authority.EXPERIMENT_ID,
                "scope": "seen_development_only_not_locked_validation_not_scientific_evidence",
                "mechanical_case_evidence": mechanical,
                "cases": rows,
                "aggregate": aggregate,
                "forward_closure": forward,
                "raw_ownership": {"test": True},
                "model_closure": {"test": True},
                "device_closure": device,
                "hydra_config_closure": hydra,
                "asset_closure": assets,
                "decoded_media_closure": {"test": True},
                "runtime_source_closure": source_closure,
                "training_performed": False,
                "optimizer_created": False,
                "parameter_updates": 0,
                "generator_loaded": False,
                "generator_forward_calls": 0,
                "raw_teacher_payload_persisted": False,
                "representation_admission_hard_false": True,
                "scientific_evidence_claimed": False,
                "completion_authority": {
                    "candidate_file_presence_is_completion_authority": False,
                    "external_controller_required": True,
                    "external_controller_valid_outcomes": ["PASS", "REJECTED"],
                    "external_completion_seal_written_by_probe": False,
                },
                "launch_contract_digest": "c" * 64,
            }
            candidate_value = {**body, "digest": authority.object_sha256(body)}
            candidate_path = candidate_root / "candidate.json"
            runtime.create_only_json(candidate_path, candidate_value)
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        controller.runtime,
                        "launch_contract",
                        return_value={
                            "digest": "c" * 64,
                            "source_closure": source_closure,
                        },
                    )
                )
                stack.enter_context(
                    mock.patch.object(controller, "_validate_raw_inventory")
                )
                stack.enter_context(
                    mock.patch.object(controller, "_validate_model_closure")
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_static_hydra_config_closure",
                        return_value=hydra,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller.authority,
                        "verify_remote_assets",
                        return_value=assets,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_recompute_foundation_source_trees",
                        return_value=assets["foundation_source_trees"],
                    )
                )
                stack.enter_context(mock.patch.object(controller, "_validate_media"))
                validated, _, _ = controller._validate_candidate(
                    candidate_path, cache, "c" * 64
                )
            self.assertEqual(validated, candidate_value)

            seal_base = base / "external_controller"
            seal_base.mkdir()
            harness = SealHarness(seal_base, diagnostic_pass=False)
            validated = dict(validated)
            validated["device_closure"] = {
                **validated["device_closure"],
                "miopen_scratch_binding": harness.candidate_value[
                    "device_closure"
                ]["miopen_scratch_binding"],
            }
            sealed = harness.call(candidate=validated)
            self.assertEqual(sealed["outcome"], "REJECTED")
            self.assertTrue(sealed["valid_completion_seal"])
            self.assertFalse(harness.ledger.exists())

    def test_valid_pass_and_valid_rejected_are_both_sealed(self):
        for diagnostic, expected in ((True, "PASS"), (False, "REJECTED")):
            with self.subTest(expected), tempfile.TemporaryDirectory() as directory:
                harness = SealHarness(Path(directory).resolve(), diagnostic)
                result = harness.call()
                self.assertEqual(result["outcome"], expected)
                self.assertTrue(result["valid_completion_seal"])
                self.assertTrue(harness.seal.is_file())
                self.assertFalse(harness.ledger.exists())
                self.assertTrue(harness.controller_step_meta.is_file())
                step = result["launch_evidence"]["controller_step_meta"]["value"]
                self.assertEqual(step["slurm_step_id"], f"{TEST_JOB_ID}.1")
                self.assertEqual(step["cuda_visible_devices"], "")
                self.assertEqual(step["rocr_visible_devices"], "")
                self.assertEqual(step["hip_visible_devices"], "")
                self.assertFalse(step["torch_imported_at_metadata"])

    def test_engineering_failure_has_ledger_and_never_rejected_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = SealHarness(Path(directory).resolve())
            result = harness.call(srun=1)
            self.assertTrue(result["engineering_failure"])
            self.assertFalse(result["valid_completion_seal"])
            self.assertIsNone(result["completion_outcome"])
            self.assertTrue(harness.ledger.is_file())
            self.assertFalse(harness.seal.exists())

    def test_controller_step_rejects_visible_gpu_or_late_torch_import(self):
        scenarios = (
            {"device_environment": {"ROCR_VISIBLE_DEVICES": "3"}},
            {"unset_device_names": ("ROCR_VISIBLE_DEVICES",)},
            {"torch_imported": [False, True]},
            {"foundation_imports": ("sam2",)},
        )
        for kwargs in scenarios:
            with self.subTest(kwargs), tempfile.TemporaryDirectory() as directory:
                harness = SealHarness(Path(directory).resolve())
                result = harness.call(**kwargs)
                self.assertTrue(result["engineering_failure"])
                self.assertFalse(result["valid_completion_seal"])
                self.assertTrue(harness.ledger.is_file())
                self.assertFalse(harness.seal.exists())

    def test_controller_argv_must_be_exact_zero_gpu_srun(self):
        for scenario in ("missing_zero_gpu", "bypass_immutable_wrapper"):
            with self.subTest(scenario), tempfile.TemporaryDirectory() as directory:
                harness = SealHarness(Path(directory).resolve())
                if scenario == "missing_zero_gpu":
                    values = [
                        value
                        for value in harness.controller_values
                        if value != "--gpus=0"
                    ]
                else:
                    values = list(harness.controller_values)
                    wrapper_index = values.index(str(harness.controller_wrapper))
                    values[wrapper_index : wrapper_index + 2] = [
                        controller.AUH_PYTHON_BIN,
                        "-B",
                        str(
                            harness.snapshot_root
                            / "actual_target_foundation_controller_v3.py"
                        ),
                    ]
                os.chmod(harness.controller_argv, 0o644)
                harness.controller_argv.unlink()
                controller.write_nul_argv(harness.controller_argv, values)
                result = harness.call()
                self.assertTrue(result["engineering_failure"])
                self.assertFalse(result["valid_completion_seal"])
                self.assertTrue(harness.ledger.is_file())
                self.assertFalse(harness.seal.exists())

    def test_formal_and_mixed_launch_modes_cannot_seal(self):
        scenarios = (
            ("both_formal", True, True),
            ("formal_compute_existing_controller", True, False),
            ("existing_compute_formal_controller", False, True),
        )
        for name, formal_compute, formal_controller in scenarios:
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                harness = SealHarness(Path(directory).resolve())
                if formal_compute:
                    values = [
                        "srun",
                        "--nodes=1",
                        "--ntasks=1",
                        "--gpus-per-task=1",
                        "--cpus-per-task=16",
                        "--export=ALL,LOCAL_RANK=0,WORLD_SIZE=1",
                        *harness.rank_values,
                    ]
                    os.chmod(harness.srun_argv, 0o644)
                    harness.srun_argv.unlink()
                    controller.write_nul_argv(harness.srun_argv, values)
                if formal_controller:
                    exact_paths = {
                        "candidate": harness.candidate,
                        "cache": harness.cache,
                        "seal": harness.seal,
                        "attempt_ledger": harness.ledger,
                        "log": harness.log,
                        "srun_argv": harness.srun_argv,
                        "rank_argv": harness.rank_argv,
                        "step_meta": harness.step_meta,
                        "controller_argv": harness.controller_argv,
                        "controller_step_meta": harness.controller_step_meta,
                        "miopen_user": harness.miopen_user,
                        "miopen_custom": harness.miopen_custom,
                        "miopen_scratch_closure": harness.miopen_scratch_closure,
                    }
                    command = controller._expected_controller_command(
                        exact_paths=exact_paths,
                        expected_contract_digest="c" * 64,
                        srun_exit_code=0,
                        tee_exit_code=0,
                        expected_job_id=TEST_JOB_ID,
                        snapshot_root=harness.snapshot_root,
                    )
                    values = controller._expected_controller_srun_argv(
                        command=command,
                        expected_job_id=TEST_JOB_ID,
                    )[0]
                    os.chmod(harness.controller_argv, 0o644)
                    harness.controller_argv.unlink()
                    controller.write_nul_argv(harness.controller_argv, values)
                result = harness.call()
                self.assertTrue(result["engineering_failure"])
                self.assertFalse(result["valid_completion_seal"])
                self.assertTrue(harness.ledger.is_file())
                self.assertFalse(harness.seal.exists())

    def test_raw_sidecar_rocr_mismatch_outer_job_and_srun_conflict_fail_closed(self):
        scenarios = ("sidecar", "rocr", "job", "srun")
        for scenario in scenarios:
            with self.subTest(scenario), tempfile.TemporaryDirectory() as directory:
                harness = SealHarness(Path(directory).resolve())
                candidate = dict(harness.candidate_value)
                env_job = TEST_JOB_ID
                if scenario == "sidecar":
                    (harness.run_root / "raw_teacher_payload.bin").write_bytes(b"raw")
                elif scenario == "rocr":
                    candidate["device_closure"] = {"rocr_visible_devices": "4"}
                elif scenario == "job":
                    env_job = "forged"
                else:
                    os.chmod(harness.srun_argv, 0o644)
                    harness.srun_argv.unlink()
                    controller.write_nul_argv(
                        harness.srun_argv,
                        ["srun", "--nodes=2", *harness.srun_values[1:]],
                    )
                result = harness.call(env_job=env_job, candidate=candidate)
                self.assertTrue(result["engineering_failure"])
                self.assertFalse(result["valid_completion_seal"])
                self.assertFalse(harness.seal.exists())
                self.assertTrue(harness.ledger.exists())

    def test_seal_cli_dispatch_passes_explicit_snapshot_and_ledger(self):
        args = [
            "seal",
            "--candidate", "/tmp/candidate",
            "--cache-dir", "/tmp/cache",
            "--seal", "/tmp/seal",
            "--attempt-ledger", "/tmp/ledger",
            "--expected-contract-digest", "a" * 64,
            "--srun-exit-code", "1",
            "--tee-exit-code", "0",
            "--expected-job-id", "9",
            "--step-meta", "/tmp/step",
            "--formal-log", "/tmp/log",
            "--srun-argv", "/tmp/srun",
            "--rank-argv", "/tmp/rank",
            "--controller-argv", "/tmp/controller-argv",
            "--controller-step-meta", "/tmp/controller-step",
            "--miopen-scratch-closure", "/tmp/miopen-closure",
            "--snapshot-root", "/tmp/snapshot",
        ]
        with mock.patch.object(
            controller,
            "seal_outcome",
            return_value={"valid_completion_seal": False},
        ) as call:
            self.assertEqual(controller.main(args), 0)
        self.assertEqual(call.call_args.kwargs["snapshot_root"], Path("/tmp/snapshot"))
        self.assertEqual(call.call_args.kwargs["attempt_ledger_path"], Path("/tmp/ledger"))
        self.assertEqual(
            call.call_args.kwargs["controller_argv_path"],
            Path("/tmp/controller-argv"),
        )
        self.assertEqual(
            call.call_args.kwargs["controller_step_meta_path"],
            Path("/tmp/controller-step"),
        )
        self.assertEqual(
            call.call_args.kwargs["miopen_scratch_closure_path"],
            Path("/tmp/miopen-closure"),
        )


class AUHCPUClosureTests(unittest.TestCase):
    def test_live_v3r3_failed_attempt_closure(self):
        receipt_path = Path(
            authority.load_authority()["v3r3_failed_engineering_attempt"][
                "failure_closure_receipt"
            ]["path"]
        )
        if not receipt_path.is_file():
            self.skipTest("the preserved V3R3 failure receipt is available only on AUH")
        receipt = controller._verify_v3r3_failed_attempt()
        self.assertTrue(receipt["verified"])
        self.assertTrue(receipt["candidate_absent"])
        self.assertTrue(receipt["completion_seal_absent"])
        self.assertEqual(
            receipt["receipt_self_sha256"],
            "b9b8841ff85f9ff74588d6ba7b29f14362815269c776a9ff631ae49e2f21ec25",
        )

    def test_live_pinned_sam_layout_source_evidence(self):
        first_path = Path(
            authority.load_authority()["v3r3_engineering_repair_contract"][
                "sam_pinned_binary_mask_source_evidence"
            ]["sources"][0]["path"]
        )
        if not first_path.is_file():
            self.skipTest("pinned SAM source files are available only on AUH")
        receipt = controller._verify_v3r3_sam_layout_source_evidence()
        self.assertTrue(receipt["verified"])
        self.assertEqual(
            {row["role"] for row in receipt["sources"]},
            {"uncompressed_rle_to_mask", "automatic_binary_mask_return"},
        )

    def test_fresh_process_static_hydra_closure_imports_no_foundation_or_torch(self):
        config_path = Path(
            authority.load_authority()["sam_hydra_authority"]["runtime_config_path"]
        )
        if not config_path.is_file():
            self.skipTest("the frozen SAM YAML is available only on AUH")
        command = [
            sys.executable,
            "-B",
            "-c",
            (
                "import json,sys;sys.path.insert(0," + repr(str(ROOT)) + ");"
                "import actual_target_foundation_controller_v3 as c;"
                "r=c._static_hydra_config_closure();"
                "r['forbidden_imports']="
                "sorted(n for n in sys.modules if n=='torch' or n.startswith('torch.') "
                "or n=='transformers' or n.startswith('transformers.') "
                "or n=='sam2' or n.startswith('sam2.') "
                "or n=='cotracker' or n.startswith('cotracker.'));"
                "print(json.dumps(r,sort_keys=True))"
            ),
        ]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["forbidden_imports"], [])

    def test_live_foundation_source_tree_closures_match_independent_algorithm(self):
        first_root = Path(
            authority.load_authority()["foundation_source_tree_authority"][0]["root"]
        )
        if not first_root.is_dir():
            self.skipTest("foundation source trees are available only on AUH")
        runtime_rows = authority.foundation_source_tree_closure()
        controller_rows = controller._recompute_foundation_source_trees()
        self.assertEqual(runtime_rows, controller_rows)
        self.assertTrue(runtime_rows["verified"])
        self.assertEqual(
            [row["file_count"] for row in runtime_rows["trees"]], [25, 40, 2271]
        )

    def test_live_static_processor_and_class_source_files_match(self):
        availability = authority.load_availability()
        first_root = Path(
            authority.load_authority()["foundation_source_tree_authority"][0]["root"]
        )
        if not first_root.is_dir():
            self.skipTest("foundation source files are available only on AUH")
        for row in availability["runtime_class_authority"]:
            _path, digest = controller._static_module_source(row["module"])
            self.assertEqual(digest, row["source_sha256"])
        specs = authority.load_authority()[
            "preprocessor_and_nontensor_config_authority"
        ]
        for role in ("sam_build_function", "dinov2_processor", "vjepa2_processor"):
            record = controller.stable_file_record(Path(specs[role]["source_path"]))
            self.assertEqual(
                record["sha256"],
                specs[role]["source_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
