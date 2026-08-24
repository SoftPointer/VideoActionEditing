#!/usr/bin/env python3
"""Local release/trust/claim-boundary regressions for v15c-r8."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
FINALIZER_PATH = METHOD_ROOT / "finalize_source_sam2_proposal_role_probe_v15c_r8.py"
BOOTSTRAP_PATH = METHOD_ROOT / "tools/v15c_r8_external_bootstrap.py"
EVIDENCE_PATH = METHOD_ROOT / "sam2_observer_evidence_v15c_r8.py"
MATERIALIZER_PATH = METHOD_ROOT / "materialize_source_sam2_proposal_tracks_v15c_r8.py"
RUNNER_PATH = METHOD_ROOT / "run_source_object_proposal_role_probe_v15c_r8.py"
POSTFLIGHT_PATH = METHOD_ROOT / "postflight_source_sam2_proposal_role_probe_v15c_r8.py"
RELEASE_PATH = (
    METHOD_ROOT
    / "assets/e00_source_sam2_proposal_role_probe_v15c_r8_release.json"
)
TEMPLATE_PATH = (
    METHOD_ROOT
    / "scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r8_external.template.sh"
)


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIN = load_path("v15c_r8_finalizer_release_test", FINALIZER_PATH)
BOOT = load_path("v15c_r8_bootstrap_release_test", BOOTSTRAP_PATH)
EVIDENCE = load_path("v15c_r8_evidence_release_test", EVIDENCE_PATH)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thaw(root: Path) -> None:
    if not root.exists():
        return
    for current, directories, _files in os.walk(root, topdown=True):
        os.chmod(current, 0o700)
        for name in directories:
            os.chmod(Path(current) / name, 0o700)


class ReleaseClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.release_sha = digest(RELEASE_PATH)

    def test_release_is_exact_nine_member_local_schema_only(self):
        manifest = FIN.verify_release(
            REPO_ROOT, RELEASE_PATH, self.release_sha
        )
        self.assertEqual(manifest["schema_version"], FIN.RELEASE_SCHEMA)
        self.assertEqual(manifest["tag"], "v15c-r8-local-schema-replay")
        self.assertEqual(manifest["core_member_count"], 9)
        self.assertEqual(manifest["snapshot_file_count"], 10)
        self.assertEqual(manifest["local_evidence_status"], "LOCAL_SCHEMA_UNAUDITED")
        self.assertIs(manifest["observer_execution_authorized"], False)
        self.assertEqual(manifest["remote_gpu_status"], "REMOTE_GPU_UNAUDITED")
        self.assertIs(manifest["scientific_claim_authorized"], False)
        self.assertIs(manifest["route_authorized"], False)
        self.assertIs(manifest["decode_authorized"], False)
        self.assertIs(manifest["training_authorized"], False)
        paths = {row["path"] for row in manifest["members"]}
        self.assertIn(
            "methods/bernini_action_editing/sam2_observer_evidence_v15c_r8.py",
            paths,
        )

    def test_stdlib_bootstrap_and_exact_private_snapshot(self):
        tree = ast.parse(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
        allowed = {
            "__future__",
            "argparse",
            "hashlib",
            "json",
            "os",
            "pathlib",
            "re",
            "stat",
            "sys",
            "typing",
        }
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                imports.append((node.module or "").split(".", 1)[0])
        self.assertEqual(set(imports) - allowed, set())
        manifest = BOOT.verify_release_source(
            REPO_ROOT, RELEASE_PATH, self.release_sha
        )
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "sealed"
            try:
                rows = BOOT.materialize_snapshot(
                    REPO_ROOT,
                    RELEASE_PATH,
                    snapshot,
                    manifest,
                    self.release_sha,
                )
                self.assertEqual(rows["construction_phase"]["file_count"], 10)
                self.assertEqual(rows["sealed_phase"]["file_count"], 10)
                self.assertEqual(snapshot.stat().st_mode & 0o777, 0o500)
                replayed = FIN.verify_release(
                    snapshot,
                    snapshot / FIN.RELEASE_RELATIVE_PATH,
                    self.release_sha,
                )
                self.assertEqual(replayed, manifest)
                self.assertEqual(
                    len(FIN.verify_snapshot(snapshot, replayed, self.release_sha)),
                    10,
                )
                modules = FIN.load_sealed_modules(snapshot, replayed)
                self.assertEqual(
                    {
                        "builder",
                        "core",
                        "materializer",
                        "postflight",
                        "runner",
                    },
                    set(vars(modules)),
                )
                self.assertEqual(
                    len(FIN.verify_snapshot(snapshot, replayed, self.release_sha)),
                    10,
                )
            finally:
                thaw(snapshot)

    def test_verify_only_template_pins_release_and_bootstrap(self):
        source = TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn(digest(BOOTSTRAP_PATH), source)
        self.assertIn(self.release_sha, source)
        self.assertIn("V15C_R8_LOCAL_VERIFY_ONLY=1", source)
        self.assertIn('exec 7<"${BOOTSTRAP}"', source)
        self.assertIn('exec 8<"${PYTHON_AUTHORITY}"', source)
        self.assertIn("/proc/self/fd/8 -I -S -B /proc/self/fd/7", source)
        self.assertNotIn("srun ", source)
        self.assertNotIn("scancel", source)
        self.assertNotIn("materialize_source", source)
        self.assertNotIn("run_source_object", source)


class ProvenanceClosureStaticTests(unittest.TestCase):
    def test_r7_hash_only_counterexample_fails_before_tensor_runtime(self):
        old_fake = {
            "schema_version": "bernini-source-sam2-proposal-tracks-v15c-r3",
            "proposal_count": 1,
            "proposals": [
                {
                    "area": 999999999,
                    "bbox_xywh": [-1e9, -1e9, 1e9, 1e9],
                    "predicted_iou": 12345.0,
                    "stability_score": -54321.0,
                }
            ],
            "tracking_batches": [{"logits_sha256": "3" * 64}],
            "freeze_receipts": {
                "parameter_sha256_before": "1" * 64,
                "parameter_sha256_after": "1" * 64,
            },
        }
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            EVIDENCE.replay_local_evidence(
                root=Path("."),
                receipt=old_fake,
                expected_binding={
                    key: hashlib.sha256(key.encode()).hexdigest()
                    for key in EVIDENCE.LOCAL_BINDING_KEYS
                },
                admission={"maximum_distinct_proposals": 64},
                automatic_generator={},
                tracking_batch_size=8,
            )

    def test_runner_rebuilds_batches_and_freeze_from_reopened_evidence(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("observer_evidence.replay_local_evidence", source)
        self.assertIn("derived_batches", source)
        self.assertIn("derived_freezes", source)
        self.assertIn("published masks are not propagation-logit-derived", source)
        self.assertIn('row["area"] != int(prompt.sum())', source)
        self.assertNotIn('tracking_batches=published["tracking_batches"]', source)
        self.assertNotIn('freeze_receipts=published["freeze_receipts"]', source)

    def test_postflight_is_pure_and_finalizer_really_calls_it(self):
        post = POSTFLIGHT_PATH.read_text(encoding="utf-8")
        finalizer = FINALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("def replay_postflight(", post)
        self.assertIn("local_evidence_replay = runner.validate_track_bundle", post)
        self.assertIn('"observer_evidence_file_sha256"', post)
        self.assertIn('"observer_evidence_internal_sha256"', post)
        self.assertIn('"observer_evidence_replay_sha256"', post)
        self.assertIn("modules.postflight.replay_postflight(", finalizer)
        self.assertNotIn(
            "gates = {key: True for key in modules.postflight.GATE_KEYS}",
            finalizer,
        )

    def test_materializer_publishes_both_runs_raw_amg_and_every_logit(self):
        source = MATERIALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn("amg.safetensors", source)
        self.assertIn("prompt_call_{call_index:03d}.safetensors", source)
        self.assertIn("propagation_frame_{frame_index:05d}.safetensors", source)
        self.assertIn("_model_tensor_manifest", source)
        self.assertIn("source_frame0_array_sha256", source)
        self.assertIn("observer_evidence.replay_local_evidence", source)

    def test_all_success_literals_remain_local_and_unauthorized(self):
        for path in (
            EVIDENCE_PATH,
            MATERIALIZER_PATH,
            RUNNER_PATH,
            POSTFLIGHT_PATH,
            FINALIZER_PATH,
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn('"observer_execution_authorized": True', source)
            self.assertNotIn('"scientific_claim_authorized": True', source)
            self.assertNotIn('"route_authorized": True', source)
            self.assertNotIn('"decode_authorized": True', source)
            self.assertNotIn('"training_authorized": True', source)
            self.assertNotIn("assert ", source)


if __name__ == "__main__":
    unittest.main()
