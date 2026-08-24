from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import full30_action_topup4_exact8_plan_v1 as plan  # noqa: E402
import full30_action_topup4_exact8_generator_v1 as generator  # noqa: E402
import full30_action_topup4_exact8_controller_v1 as controller  # noqa: E402
import build_full30_action_topup4_exact8_release_v1 as release  # noqa: E402


SELECTION = METHOD_ROOT / release.SELECTION_MEMBER
PARENT_REGISTRY = METHOD_ROOT / release.PARENT_REGISTRY_MEMBER
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_full30_action_topup4_exact8_136140_world4_v1.sh"
)


def _build_plan(parent: Path) -> dict:
    result = plan.build_plan(
        selection=SELECTION.resolve(strict=True),
        expected_selection_sha256=release.SELECTION_SHA256,
        parent_registry=PARENT_REGISTRY.resolve(strict=True),
        expected_parent_registry_sha256=release.PARENT_REGISTRY_SHA256,
        output_dir=parent / "exact8-plan",
    )
    return result


def _sign(value: dict) -> dict:
    return {**value, "receipt_digest": controller.object_sha256(value)}


def _review(exact8: dict) -> dict:
    rows = [
        {
            "candidate_id": task["candidate_id"],
            "semantic_branch": task["semantic_branch"],
            "frame_count": 81,
            "reviewed_frame_count": 81,
            "reviewed_frame_indices_sha256": controller.FULL81_INDEX_SHA256,
            "all_81_frames_reviewed": True,
            "verdict": "pass",
            "action_or_incomplete_pass": True,
        }
        for task in exact8["admission_tasks"]
    ]
    return _sign(
        {
            "schema_version": controller.REVIEW_SCHEMA,
            "plan_digest": exact8["plan_digest"],
            "review_population": "minimal_cross_anchor_topup4_exact8",
            "reviewer_independent_of_generator": True,
            "reviewer_independent_of_materializer": True,
            "candidate_count": 8,
            "diagnostic_candidate_count": 0,
            "candidate_reviews": rows,
        }
    )


def _materializer_gate(exact8: dict) -> dict:
    rows = [
        {
            "group_id": cell["group_id"],
            "calibration_group_id": cell["calibration_group_id"],
            "admitted_candidate_ids": cell["candidate_ids"],
            "control_gates": {
                name: True for name in plan.MATERIALIZER_CONTROL_ORDER
            },
            "q_values": {"action": 1.25, "incomplete": 0.75},
            "a_min": 0.125,
            "threshold_gate_passed": True,
        }
        for cell in exact8["seed_cells"]
    ]
    return _sign(
        {
            "schema_version": controller.MATERIALIZER_GATE_SCHEMA,
            "plan_digest": exact8["plan_digest"],
            "materializer": "full30_action_psiout_materializer_v1",
            "same_real_source_state": True,
            "same_sigma": True,
            "same_noise": True,
            "official_frozen_noop_stopgrad": True,
            "all_threshold_gates_passed": True,
            "diagnostic_task_count": 0,
            "generated_media_used_by_optimizer": False,
            "optimizer_input_created": False,
            "seed_cell_gates": rows,
        }
    )


def _tar_with_extra(raw: bytes) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as source:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as target:
            for member in source.getmembers():
                handle = source.extractfile(member)
                target.addfile(member, handle)
            payload = b"forbidden\n"
            extra = tarfile.TarInfo(f"{release.MEMBER_ROOT}/forbidden.txt")
            extra.size = len(payload)
            extra.mode = 0o444
            extra.uid = extra.gid = extra.mtime = 0
            target.addfile(extra, io.BytesIO(payload))
    return output.getvalue()


class Topup4Exact8PlanTests(unittest.TestCase):
    def test_selection_projects_exact_four_pairs_and_no_diagnostic_lane(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            value = _build_plan(Path(temporary))
            self.assertEqual(value["formal_candidate_count"], 8)
            self.assertEqual(value["comparator_cell_count"], 4)
            self.assertEqual(
                [row["semantic_branch"] for row in value["admission_tasks"]],
                ["action", "incomplete"] * 4,
            )
            self.assertEqual(
                [row["candidate_count"] for row in value["shards"]], [4, 4]
            )
            self.assertNotIn("diagnostic_tasks", value)
            self.assertEqual(value["execution_contract"]["diagnostic_task_count"], 0)
            self.assertFalse(
                value["execution_contract"]["diagnostic_generation_allowed"]
            )
            self.assertEqual(
                [row["group_id"] for row in value["seed_cells"]],
                ["sp4-a", "sp4-a", "sp4-b", "sp4-b"],
            )

    def test_resigned_task_widening_and_reordering_fail_replay(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            value = _build_plan(Path(temporary))
            path = Path(value["_path"])
            loaded = json.loads(path.read_text(encoding="ascii"))
            hostile = copy.deepcopy(loaded)
            hostile["admission_tasks"][0], hostile["admission_tasks"][1] = (
                hostile["admission_tasks"][1],
                hostile["admission_tasks"][0],
            )
            unsigned = dict(hostile)
            unsigned.pop("plan_digest")
            hostile["plan_digest"] = plan.object_sha256(unsigned)
            hostile_path = Path(temporary) / "hostile.json"
            hostile_path.write_bytes(plan.canonical_json_bytes(hostile) + b"\n")
            with self.assertRaisesRegex(
                plan.Topup4Exact8PlanError, "replay differs"
            ):
                plan.load_plan(
                    hostile_path.resolve(strict=True),
                    hashlib.sha256(hostile_path.read_bytes()).hexdigest(),
                )


class Topup4Exact8GeneratorTests(unittest.TestCase):
    def test_gaussian_pair_requires_order_and_exact_tensor_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            tasks = _build_plan(Path(temporary))["admission_tasks"][:2]
        artifact = {
            "raw_value_sha256": "1" * 64,
            "content_sha256": "2" * 64,
            "shape": [1, 16, 21, 48, 80],
            "dtype": "torch.float32",
            "stored_dtype": "torch.float32",
            "generator_initial_seed": tasks[0]["seed"],
        }
        receipts = [
            {"artifacts": {"official_initial_gaussian": dict(artifact)}},
            {"artifacts": {"official_initial_gaussian": dict(artifact)}},
        ]
        proof = generator._gaussian_pair_proof(tasks, receipts)
        self.assertTrue(
            proof["action_incomplete_official_gaussian_tensor_values_byte_equal"]
        )
        receipts[1]["artifacts"]["official_initial_gaussian"]["content_sha256"] = "3" * 64
        with self.assertRaisesRegex(
            generator.Topup4Exact8GenerationError, "did not reuse"
        ):
            generator._gaussian_pair_proof(tasks, receipts)
        with self.assertRaisesRegex(
            generator.Topup4Exact8GenerationError, "ordered"
        ):
            generator._gaussian_pair_proof(list(reversed(tasks)), receipts)

    def test_resource_specialization_is_only_136141_to_136140(self) -> None:
        source = (METHOD_ROOT / release.RESOURCE_SOURCE).read_bytes()
        specialized = release.specialize_resource_bytes(source)
        self.assertEqual(len(source), len(specialized))
        self.assertEqual(source.count(b"136141"), 7)
        self.assertEqual(specialized.count(b"136141"), 0)
        self.assertEqual(specialized.count(b"136140"), 7)
        self.assertEqual(
            hashlib.sha256(specialized).hexdigest(),
            generator.RESOURCE_SPECIALIZED_SHA256,
        )
        self.assertEqual(specialized.replace(b"136140", b"136141"), source)
        with self.assertRaisesRegex(
            release.Topup4Exact8ReleaseError, "preimage differs"
        ):
            release.specialize_resource_bytes(source + b"\n")


class Topup4Exact8AdmissionTests(unittest.TestCase):
    def test_review_requires_each_action_and_incomplete_full81(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            exact8 = _build_plan(Path(temporary))
        receipt = _review(exact8)
        controller.validate_review_admission(receipt, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["candidate_reviews"][1]["reviewed_frame_count"] = 80
        with self.assertRaisesRegex(
            controller.Topup4Exact8ControllerError, "full81 review failed"
        ):
            controller.validate_review_admission(_sign(hostile), exact8)

    def test_materializer_requires_same_state_controls_and_positive_a_min(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            exact8 = _build_plan(Path(temporary))
        receipt = _materializer_gate(exact8)
        controller.validate_materializer_gate(receipt, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["seed_cell_gates"][0]["control_gates"]["camera_only"] = False
        with self.assertRaisesRegex(
            controller.Topup4Exact8ControllerError, "comparator-cell gate"
        ):
            controller.validate_materializer_gate(_sign(hostile), exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["seed_cell_gates"][0]["a_min"] = 0.0
        with self.assertRaisesRegex(
            controller.Topup4Exact8ControllerError, "comparator-cell gate"
        ):
            controller.validate_materializer_gate(_sign(hostile), exact8)

    def test_terminal_gate_requires_exact8_60_56_10ms_and_no_oom(self) -> None:
        value = _sign(
            {
                "schema_version": controller.TERMINAL_HOST_GATE_SCHEMA,
                "formal_candidate_count_at_gate": 8,
                "sample_interval_ns": 10_000_000,
                "maximum_observed_gap_ns": 20_000_000,
                "host_memory_limit_gib": 60,
                "host_memory_safe_ceiling_gib": 56,
                "sampled_peak_strictly_below_56_gib": True,
                "all_samples_zero_oom_and_oom_kill": True,
                "monitor_exit_status": 0,
                "monitor_identity_dead_at_gate": True,
                "diagnostic_task_count": 0,
                "optimizer_authorized": False,
            }
        )
        controller.validate_terminal_host_gate(value)
        hostile = dict(value)
        hostile.pop("receipt_digest")
        hostile["formal_candidate_count_at_gate"] = 40
        with self.assertRaisesRegex(
            controller.Topup4Exact8ControllerError, "authority differs"
        ):
            controller.validate_terminal_host_gate(_sign(hostile))


class Topup4Exact8ReleaseAndLauncherTests(unittest.TestCase):
    def test_release_is_deterministic_exact_25_member_closure(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        self.assertEqual(manifest["file_count"], 25)
        self.assertEqual(len(manifest["files"]), 25)
        self.assertEqual(manifest["authority"]["formal_candidate_count"], 8)
        self.assertEqual(manifest["authority"]["comparator_cell_count"], 4)
        self.assertEqual(manifest["authority"]["diagnostic_task_count"], 0)
        self.assertFalse(manifest["authority"]["diagnostic_generation_allowed"])
        self.assertEqual(
            manifest["topology"]["holder"],
            {"job_id": 136140, "node": "auh7-1b-gpu-215"},
        )
        self.assertEqual(manifest["topology"]["host_memory_request_gib"], 60)
        self.assertEqual(
            manifest["topology"]["host_sampled_current_safe_ceiling_gib"], 56
        )
        self.assertEqual(
            manifest["topology"]["host_cgroup_sample_interval_ns"], 10_000_000
        )
        self.assertEqual(manifest["topology"]["t2v_rank_gpu_memory_limit_gib"], 52)
        self.assertTrue(
            manifest["topology"]["t2v_text_encoder_rank_gpu_residency_required"]
        )
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
            names = [member.name for member in archive.getmembers()]
        self.assertEqual(
            names,
            [f"{release.MEMBER_ROOT}/{row['path']}" for row in manifest["files"]],
        )

    def test_archive_extra_member_and_payload_tamper_fail(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        raw = release.build_archive(manifest, payloads)
        with self.assertRaisesRegex(
            release.Topup4Exact8ReleaseError, "member order"
        ):
            release.verify_archive(_tar_with_extra(raw), manifest)
        tampered = bytearray(raw)
        needle = payloads["full30_action_topup4_exact8_plan_v1.py"][:32]
        offset = raw.find(needle)
        self.assertGreaterEqual(offset, 0)
        tampered[offset] ^= 1
        with self.assertRaisesRegex(
            release.Topup4Exact8ReleaseError, "content differs"
        ):
            release.verify_archive(bytes(tampered), manifest)

    def test_build_audit_create_only_and_resigned_manifest_pin(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            archive = root / "source.tar"
            manifest_path = root / "source.manifest.json"
            built = release.build(
                METHOD_ROOT.resolve(strict=True), archive, manifest_path
            )
            audited = release.audit(
                archive.resolve(strict=True),
                built["archive_sha256"],
                manifest_path.resolve(strict=True),
                built["manifest_sha256"],
            )
            self.assertEqual(audited["file_count"], 25)
            with self.assertRaisesRegex(
                release.Topup4Exact8ReleaseError, "fresh absolute"
            ):
                release.build(
                    METHOD_ROOT.resolve(strict=True), archive, manifest_path
                )
            resigned = json.loads(manifest_path.read_text(encoding="ascii"))
            unsigned = dict(resigned)
            unsigned.pop("manifest_digest")
            unsigned["release_scope"] += "-hostile"
            resigned = {**unsigned, "manifest_digest": release.object_sha256(unsigned)}
            hostile_path = root / "resigned.manifest.json"
            hostile_path.write_bytes(release.canonical_json_bytes(resigned) + b"\n")
            with self.assertRaisesRegex(
                release.Topup4Exact8ReleaseError, "manifest SHA-256 differs"
            ):
                release.audit(
                    archive.resolve(strict=True),
                    built["archive_sha256"],
                    hostile_path.resolve(strict=True),
                    built["manifest_sha256"],
                )

    def test_only_controller_and_launcher_are_release_entrypoints(self) -> None:
        self.assertEqual(
            set(release.ENTRYPOINTS),
            {
                "full30_action_topup4_exact8_controller_v1.py",
                "scripts/auh_full30_action_topup4_exact8_136140_world4_v1.sh",
            },
        )
        self.assertNotIn("train_lora.py", release.ENTRYPOINTS)
        self.assertIn("train_lora.py", release.FILES_AND_MODES)

    def test_launcher_is_exact8_resource_bound_and_non_destructive(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("readonly holder_job=136140", source)
        self.assertIn("readonly holder_node=auh7-1b-gpu-215", source)
        self.assertIn("--gpus-per-task=8", source)
        self.assertIn("--cpus-per-task=32 --mem=60G", source)
        self.assertEqual(source.count("run_exact4_shard sp4-"), 2)
        self.assertIn("formal_candidate_count=8", source)
        self.assertIn("formal_comparator_cell_count=4", source)
        self.assertIn("diagnostic_invocations=0", source)
        self.assertIn("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED=1", source)
        self.assertIn("physical_safe_open=true", source)
        self.assertIn("r10_r13_parity=true", source)
        self.assertIn("same_gaussian_action_incomplete_per_cell=true", source)
        self.assertIn("parent_136140_cancelled_released_or_requeued=false", source)
        for forbidden in (
            "--lane diagnostic",
            "scancel",
            "scontrol release",
            "scontrol requeue",
            "optimizer.step",
        ):
            self.assertNotIn(forbidden, source)
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
