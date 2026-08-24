from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import full30_action_confirmation8_136141_plan_v1 as plan  # noqa: E402
import full30_action_confirmation8_136141_generator_v1 as generator  # noqa: E402
import full30_action_confirmation8_136141_controller_v1 as controller  # noqa: E402
import build_full30_action_confirmation8_136141_release_v1 as release  # noqa: E402


LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_full30_action_confirmation8_136141_world4_v1.sh"
)


def _fake_source_tasks() -> list[dict]:
    tasks: list[dict] = []
    ordinal = 0
    for cell in plan.CONFIRMATION_CELL_REGISTRY:
        for branch in plan.bank_contract.MACE_BRANCH_ORDER:
            candidate_id = (
                f"{plan.SEED_PREFIXES[cell['seed_slot']]}{cell['iid']}-{branch}"
            )
            tasks.append(
                {
                    "seed_slot": cell["seed_slot"],
                    "root_spec_path": f"/sealed/{cell['seed_slot']}.json",
                    "root_spec_sha256": "1" * 64,
                    "candidate_spec_path": f"/sealed/{candidate_id}.json",
                    "candidate_spec_sha256": hashlib.sha256(
                        candidate_id.encode("ascii")
                    ).hexdigest(),
                    "group_id": cell["group_id"],
                    "visible_gpus": (
                        [0, 1, 2, 3]
                        if cell["group_id"] == "sp4-a"
                        else [4, 5, 6, 7]
                    ),
                    "ordinal": ordinal,
                    "candidate_id": candidate_id,
                    "analysis_split": "confirmation",
                    "calibration_group_id": (
                        f"cell-{cell['iid']}-s{cell['seed']}"
                    ),
                    "semantic_branch": branch,
                    "seed": cell["seed"],
                }
            )
            ordinal += 1
    return tasks


def _fake_plan() -> dict:
    return plan._plan_value(source_specs=[], source_tasks=_fake_source_tasks())


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
            "review_population": "confirmation_action_anchor_exact8_136141",
            "reviewer_independent_of_generator": True,
            "reviewer_independent_of_materializer": True,
            "candidate_count": 8,
            "candidate_reviews": rows,
            "diagnostic_task_count": 0,
            "diagnostic_generation_observed": False,
        }
    )


def _materializer_gate(exact8: dict) -> dict:
    rows = [
        {
            "seed_slot": cell["seed_slot"],
            "group_id": cell["group_id"],
            "calibration_group_id": cell["calibration_group_id"],
            "admitted_candidate_ids": cell["admission_candidate_ids"],
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
            "diagnostic_generation_observed": False,
            "optimizer_input_created": False,
            "seed_cell_gates": rows,
        }
    )


def _tar_with_extra(raw: bytes) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as source:
        with tarfile.open(
            fileobj=output, mode="w", format=tarfile.USTAR_FORMAT
        ) as target:
            for member in source.getmembers():
                target.addfile(member, source.extractfile(member))
            payload = b"forbidden\n"
            extra = tarfile.TarInfo(f"{release.MEMBER_ROOT}/forbidden.txt")
            extra.size = len(payload)
            extra.mode = 0o444
            extra.uid = extra.gid = extra.mtime = 0
            target.addfile(extra, io.BytesIO(payload))
    return output.getvalue()


class Confirmation8PlanTests(unittest.TestCase):
    def test_exact_partition_is_four_action_incomplete_pairs_only(self) -> None:
        exact8, cells = plan.partition_confirmation_tasks(
            _fake_source_tasks()
        )
        self.assertEqual(len(exact8), 8)
        self.assertEqual(len(cells), 4)
        self.assertEqual(
            [row["semantic_branch"] for row in exact8],
            ["action", "incomplete"] * 4,
        )
        self.assertTrue(all("diagnostic_candidate_ids" not in row for row in cells))

    def test_hostile_source_reordering_seed_and_split_fail(self) -> None:
        tasks = _fake_source_tasks()
        hostile = list(tasks)
        hostile[0], hostile[1] = hostile[1], hostile[0]
        with self.assertRaisesRegex(
            plan.Confirmation8PlanError, "order or identity"
        ):
            plan.partition_confirmation_tasks(hostile)
        hostile = copy.deepcopy(tasks)
        hostile[0]["seed"] += 1
        with self.assertRaisesRegex(plan.Confirmation8PlanError, "registry"):
            plan.partition_confirmation_tasks(hostile)
        hostile = copy.deepcopy(tasks)
        hostile[2]["analysis_split"] = "fit"
        with self.assertRaisesRegex(plan.Confirmation8PlanError, "registry"):
            plan.partition_confirmation_tasks(hostile)

    def test_plan_has_explicit_scientific_contract_and_no_diagnostic_lane(self) -> None:
        value = _fake_plan()
        contract = value["execution_contract"]
        self.assertEqual(value["admission_candidate_count"], 8)
        self.assertEqual(value["diagnostic_task_count"], 0)
        self.assertFalse(value["diagnostic_generation_allowed"])
        self.assertNotIn("diagnostic_tasks", value)
        self.assertEqual(value["inference_steps_per_clip"], 40)
        self.assertEqual(value["optimizer_steps"], 0)
        for field in (
            "purpose",
            "scientific_target",
            "learning_target",
            "numeric_target",
            "dataset",
            "frozen_baseline",
            "core_validation",
        ):
            self.assertIsInstance(value[field], str)
            self.assertTrue(value[field])
        self.assertEqual(contract["diagnostic_task_count"], 0)
        self.assertFalse(contract["diagnostic_generation_allowed"])
        self.assertEqual(
            contract["same_state_controls_source"],
            "official_full30_action_psiout_materializer",
        )


class Confirmation8GeneratorTests(unittest.TestCase):
    def test_formal_projection_rejects_widening_and_diagnostic_keys(self) -> None:
        value = _fake_plan()
        admission = generator._formal_tasks(value)
        self.assertEqual(len(admission), 8)
        hostile = copy.deepcopy(value)
        hostile["admission_tasks"].append(copy.deepcopy(admission[0]))
        with self.assertRaisesRegex(
            generator.Confirmation8GenerationError, "task closure"
        ):
            generator._formal_tasks(hostile)
        hostile = copy.deepcopy(value)
        hostile["diagnostic_tasks"] = []
        with self.assertRaisesRegex(
            generator.Confirmation8GenerationError, "task closure"
        ):
            generator._formal_tasks(hostile)

    def test_gaussian_pair_requires_order_and_exact_tensor_identity(self) -> None:
        tasks = _fake_plan()["admission_tasks"][:2]
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
        receipts[1]["artifacts"]["official_initial_gaussian"][
            "content_sha256"
        ] = "3" * 64
        with self.assertRaisesRegex(
            generator.Confirmation8GenerationError, "did not reuse"
        ):
            generator._gaussian_pair_proof(tasks, receipts)
        with self.assertRaisesRegex(
            generator.Confirmation8GenerationError, "ordered"
        ):
            generator._gaussian_pair_proof(list(reversed(tasks)), receipts)

    def test_release_specialization_is_exact_136141_identity_postimage(self) -> None:
        source = (METHOD_ROOT / release.RESOURCE_SOURCE).read_bytes()
        specialized = release.specialize_resource_bytes(source)
        self.assertEqual(len(source), len(specialized))
        self.assertEqual(source.count(b"136141"), 7)
        self.assertEqual(specialized.count(b"136141"), 7)
        self.assertEqual(specialized.count(b"136309"), 0)
        self.assertEqual(
            hashlib.sha256(specialized).hexdigest(),
            generator.RESOURCE_SPECIALIZED_SHA256,
        )
        self.assertEqual(specialized, source)
        with self.assertRaisesRegex(
            release.Confirmation8ReleaseError, "preimage differs"
        ):
            release.specialize_resource_bytes(source + b"\n")


class Confirmation8AdmissionTests(unittest.TestCase):
    def test_review_requires_each_action_and_incomplete_full81(self) -> None:
        exact8 = _fake_plan()
        receipt = _review(exact8)
        controller.validate_review_admission(receipt, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["candidate_reviews"][1]["reviewed_frame_count"] = 80
        hostile = _sign(hostile)
        with self.assertRaisesRegex(
            controller.Confirmation8ControllerError, "full81 review failed"
        ):
            controller.validate_review_admission(hostile, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["candidate_reviews"][0]["candidate_id"] = "forbidden-extra-row"
        hostile = _sign(hostile)
        with self.assertRaisesRegex(
            controller.Confirmation8ControllerError, "population differs"
        ):
            controller.validate_review_admission(hostile, exact8)

    def test_materializer_gate_is_same_state_and_no_diagnostic(self) -> None:
        exact8 = _fake_plan()
        receipt = _materializer_gate(exact8)
        controller.validate_materializer_gate(receipt, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["diagnostic_generation_observed"] = True
        hostile = _sign(hostile)
        with self.assertRaisesRegex(
            controller.Confirmation8ControllerError, "authority or population"
        ):
            controller.validate_materializer_gate(hostile, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["seed_cell_gates"][0]["control_gates"]["camera_only"] = False
        hostile = _sign(hostile)
        with self.assertRaisesRegex(
            controller.Confirmation8ControllerError, "seed-cell gate"
        ):
            controller.validate_materializer_gate(hostile, exact8)

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
                "optimizer_authorized": False,
            }
        )
        controller.validate_terminal_host_gate(value)
        hostile = dict(value)
        hostile.pop("receipt_digest")
        hostile["formal_candidate_count_at_gate"] = 40
        hostile = _sign(hostile)
        with self.assertRaisesRegex(
            controller.Confirmation8ControllerError, "authority differs"
        ):
            controller.validate_terminal_host_gate(hostile)


class Confirmation8ReleaseAndLauncherTests(unittest.TestCase):
    def test_release_is_deterministic_exact_closure_and_exact8_authority(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        self.assertEqual(
            manifest["authority"]["formal_branch_order"],
            ["action", "incomplete"],
        )
        self.assertEqual(manifest["authority"]["formal_candidate_count"], 8)
        self.assertEqual(manifest["authority"]["diagnostic_task_count"], 0)
        self.assertFalse(manifest["authority"]["diagnostic_generation_allowed"])
        self.assertEqual(manifest["authority"]["inference_steps_per_clip"], 40)
        self.assertEqual(manifest["authority"]["optimizer_steps"], 0)
        self.assertEqual(
            manifest["topology"]["holder"],
            {"job_id": 136141, "node": "auh7-1b-gpu-299"},
        )
        self.assertEqual(
            manifest["component_pins"][
                "resource_136141_specialization_sha256"
            ],
            release.RESOURCE_SPECIALIZED_SHA256,
        )
        self.assertEqual(
            payloads[release.RESOURCE_SOURCE],
            payloads[release.RESOURCE_SPECIALIZED],
        )
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
            names = [member.name for member in archive.getmembers()]
        self.assertEqual(
            names,
            [
                f"{release.MEMBER_ROOT}/{row['path']}"
                for row in manifest["files"]
            ],
        )

    def test_archive_extra_member_payload_tamper_and_resigned_manifest_fail(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        raw = release.build_archive(manifest, payloads)
        with self.assertRaisesRegex(
            release.Confirmation8ReleaseError, "member order"
        ):
            release.verify_archive(_tar_with_extra(raw), manifest)
        tampered = bytearray(raw)
        needle = payloads["full30_action_confirmation8_136141_plan_v1.py"][:32]
        offset = raw.find(needle)
        self.assertGreaterEqual(offset, 0)
        tampered[offset] ^= 1
        with self.assertRaisesRegex(
            release.Confirmation8ReleaseError, "content differs"
        ):
            release.verify_archive(bytes(tampered), manifest)
        hostile = copy.deepcopy(manifest)
        hostile.pop("manifest_digest")
        hostile["topology"]["holder"]["job_id"] = 136309
        hostile = {
            **hostile,
            "manifest_digest": release.object_sha256(hostile),
        }
        with self.assertRaisesRegex(
            release.Confirmation8ReleaseError, "authority/topology"
        ):
            release.validate_manifest(hostile)

    def test_build_audit_is_create_only_and_manifest_pin_is_exact(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            archive = root / "source.tar"
            manifest_path = root / "source.manifest.json"
            built = release.build(METHOD_ROOT, archive, manifest_path)
            audited = release.audit(
                archive.resolve(strict=True),
                built["archive_sha256"],
                manifest_path.resolve(strict=True),
                built["manifest_sha256"],
            )
            self.assertEqual(audited["file_count"], len(release.FILES_AND_MODES))
            with self.assertRaisesRegex(
                release.Confirmation8ReleaseError, "fresh absolute"
            ):
                release.build(METHOD_ROOT, archive, manifest_path)
            with self.assertRaisesRegex(
                release.Confirmation8ReleaseError, "manifest SHA-256 differs"
            ):
                release.audit(
                    archive.resolve(strict=True),
                    built["archive_sha256"],
                    manifest_path.resolve(strict=True),
                    "0" * 64,
                )

    def test_only_controller_and_launcher_are_release_entrypoints(self) -> None:
        self.assertEqual(
            set(release.ENTRYPOINTS),
            {
                "full30_action_confirmation8_136141_controller_v1.py",
                "scripts/auh_full30_action_confirmation8_136141_world4_v1.sh",
            },
        )
        self.assertNotIn("train_lora.py", release.ENTRYPOINTS)
        self.assertIn("train_lora.py", release.FILES_AND_MODES)

    def test_launcher_runs_only_four_exact2_admission_shards(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("readonly holder_job=136141", source)
        self.assertIn("readonly holder_node=auh7-1b-gpu-299", source)
        self.assertIn("--gpus-per-task=8", source)
        self.assertIn("--cpus-per-task=32 --mem=60G", source)
        self.assertEqual(source.count("run_admission_shard seed"), 4)
        self.assertIn('"${generator}" run-sp4', source)
        self.assertNotIn("--lane", source)
        self.assertIn("formal_candidate_count=8", source)
        self.assertIn("diagnostic_task_count=0", source)
        self.assertIn("inference_steps_per_clip=40", source)
        self.assertIn("optimizer_steps=0", source)
        self.assertIn("independent_full81_review=pending", source)
        self.assertIn("same_state_materializer_threshold_gate=pending", source)
        self.assertIn("physical_safe_open=true", source)
        self.assertIn("r10_r13_parity=true", source)
        self.assertIn("gpu_peak_reserved_limit_gib=52", source)
        self.assertIn("parent_136141_cancelled_released_or_requeued=false", source)
        for forbidden in (
            "scancel",
            "scontrol release",
            "scontrol requeue",
            "optimizer.step",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
