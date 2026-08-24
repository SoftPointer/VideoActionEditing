from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import pair_v5_t2v_calibration_bank_spec as bank  # noqa: E402
import full30_action_fit_repair_exact8_plan_v1 as plan  # noqa: E402
import full30_action_fit_repair_exact8_generator_v1 as generator  # noqa: E402
import full30_action_fit_repair_exact8_controller_v1 as controller  # noqa: E402
import build_full30_action_fit_repair_exact8_release_v1 as release  # noqa: E402


LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_full30_action_fit_repair_exact8_136140_world4_v1.sh"
)


def _caption(split: str, group_id: str, branch: str, seed: int) -> str:
    return (
        f"A continuous realistic medium shot for {split} {group_id} seed {seed} "
        f"performs the uniquely specified {branch} semantic branch while identity, "
        "scene, illumination, framing, and locked camera remain temporally coherent."
    )


def _candidate(
    *, slot: str, split: str, group_id: str, branch: str,
    ordinal: int, seed: int,
) -> dict:
    fit = split == "fit"
    if group_id == "sp4-a":
        iid = "00435ad621c44fac" if fit else f"confirm-arms-{slot}"
        prefix = (
            f"pair5-t2v-reserve4-{'v1' if slot == 'seed1' else 'seed2'}-"
            f"00435ad621c44fac"
            if fit else f"fixture-{slot}-confirm-arms"
        )
        actor = "woman-raised-arms-fit" if fit else f"woman-raised-arms-confirm-{slot}"
        scene = "portrait-interior-arms-fit" if fit else f"portrait-arms-confirm-{slot}"
        action = "arms-down-hands-hips-fit" if fit else f"arms-action-confirm-{slot}"
        geometry = f"/private/tmp/fixture-{slot}-arms.mp4"
    else:
        iid = "71ba57892bd043df" if fit else f"confirm-reach-{slot}"
        prefix = (
            f"pair5-t2v-reserve4-{'v1' if slot == 'seed1' else 'seed2'}-"
            f"71ba57892bd043df"
            if fit else f"fixture-{slot}-confirm-reach"
        )
        actor = "left-arm-performer-fit" if fit else f"left-arm-confirm-{slot}"
        scene = "portrait-left-arm-fit" if fit else f"portrait-reach-confirm-{slot}"
        action = "fist-to-palm-down-fit" if fit else f"reach-action-confirm-{slot}"
        geometry = f"/private/tmp/fixture-{slot}-reach.mp4"
    candidate_id = f"{prefix}-{branch}"
    text = _caption(split, group_id, branch, seed)
    return {
        "candidate_id": candidate_id,
        "analysis_split": split,
        "action_family_id": "articulated-pose-transition",
        "calibration_group_id": f"cell-{iid}-s{seed}",
        "prompt_group_id": f"{actor}--{scene}",
        "action_family_group_id": action,
        "actor_group_id": actor,
        "scene_group_id": scene,
        "action_group_id": action,
        "geometry_source_video": geometry,
        "geometry_source_video_sha256": hashlib.sha256(geometry.encode()).hexdigest(),
        "geometry_contract": bank.GEOMETRY_CONTRACT,
        "semantic_branch": branch,
        "full_t2v_caption": text,
        "full_t2v_caption_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "caption_contract": bank.CAPTION_CONTRACT,
        "seed": seed,
    }


def _root_spec(slot: str) -> dict:
    seed_base = 2026080800 if slot == "seed1" else 2026080900
    groups = []
    for group_id, visible in bank.GROUP_LAYOUT:
        fit_seed = seed_base + (21 if group_id == "sp4-a" else 24)
        confirm_seed = seed_base + (31 if group_id == "sp4-a" else 34)
        rows = [
            _candidate(
                slot=slot, split=split, group_id=group_id, branch=branch,
                ordinal=ordinal, seed=(fit_seed if split == "fit" else confirm_seed),
            )
            for split in ("fit", "confirmation")
            for ordinal, branch in enumerate(bank.MACE_BRANCH_ORDER)
        ]
        groups.append(
            {"group_id": group_id, "visible_gpus": visible, "candidates": rows}
        )
    return bank.validate_root_spec(
        {
            "schema_version": bank.SCHEMA_VERSION,
            "sampling_contract": bank.SAMPLING_CONTRACT,
            "semantic_input_closure": bank.SEMANTIC_INPUT_CLOSURE,
            "artifact_use_contract": bank.ARTIFACT_USE_CONTRACT,
            "split_contract": bank.SPLIT_CONTRACT,
            "groups": groups,
        }
    )


def _write_fixture_spec(parent: Path, slot: str) -> tuple[Path, str]:
    path = parent / f"{slot}.json"
    raw = bank.canonical_json_bytes(_root_spec(slot)) + b"\n"
    path.write_bytes(raw)
    return path.resolve(strict=True), hashlib.sha256(raw).hexdigest()


def _build_plan(parent: Path) -> dict:
    seed1, sha1 = _write_fixture_spec(parent, "seed1")
    seed2, sha2 = _write_fixture_spec(parent, "seed2")
    with mock.patch.object(
        plan, "SEED1_SPEC_SHA256", sha1
    ), mock.patch.object(plan, "SEED2_SPEC_SHA256", sha2):
        return plan.build_plan(
            seed1_spec=seed1,
            expected_seed1_spec_sha256=sha1,
            seed2_spec=seed2,
            expected_seed2_spec_sha256=sha2,
            output_dir=parent / "exact8-plan",
        )


def _load_fixture_plan(value: dict) -> dict:
    sha1 = value["source_specs"][0]["file_sha256"]
    sha2 = value["source_specs"][1]["file_sha256"]
    with mock.patch.object(
        plan, "SEED1_SPEC_SHA256", sha1
    ), mock.patch.object(plan, "SEED2_SPEC_SHA256", sha2):
        loaded, _, _ = plan.load_plan(value["_path"], value["_file_sha256"])
    return loaded


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
            "review_population": "fit_action_incomplete_repair_exact8",
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


_RECEIPT_TASK_BINDING_FIELDS = (
    "candidate_id",
    "analysis_split",
    "calibration_group_id",
    "semantic_branch",
    "seed",
    "prompt_utf8_sha256",
    "source_geometry_video",
    "source_geometry_video_sha256",
)


def _receipt_task_binding(task: dict) -> dict:
    return {field: task[field] for field in _RECEIPT_TASK_BINDING_FIELDS}


def _resign(value: dict) -> dict:
    unsigned = copy.deepcopy(value)
    unsigned.pop("receipt_digest", None)
    return {**unsigned, "receipt_digest": controller.object_sha256(unsigned)}


class _PhysicalCandidateReceiptResource:
    def __init__(self) -> None:
        self.reopened_paths: list[Path] = []

    def _validate_candidate_receipt(
        self, task: dict, receipt_path: Path
    ) -> tuple[dict, dict]:
        self.reopened_paths.append(receipt_path)
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
        if raw != controller.canonical_json_bytes(receipt) + b"\n":
            raise RuntimeError("candidate receipt is not canonical JSON")
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest", None)
        if declared != controller.object_sha256(unsigned):
            raise RuntimeError("candidate receipt digest differs")
        expected_candidate = _receipt_task_binding(task)
        if receipt.get("candidate") != expected_candidate:
            raise RuntimeError(
                f"candidate receipt/task binding differs: {task['candidate_id']}"
            )
        return receipt, receipt["candidate"]


def _generation_audit(
    parent: Path, built: dict, exact8: dict
) -> tuple[dict, list[Path]]:
    receipt_root = parent / "formal-candidate-receipts"
    receipt_root.mkdir()
    receipts: dict[str, dict] = {}
    receipt_paths: list[Path] = []
    rows = []
    for task in exact8["admission_tasks"]:
        gaussian_key = task["calibration_group_id"]
        gaussian = {
            "raw_value_sha256": hashlib.sha256(
                f"raw:{gaussian_key}".encode("ascii")
            ).hexdigest(),
            "content_sha256": hashlib.sha256(
                f"content:{gaussian_key}".encode("ascii")
            ).hexdigest(),
            "shape": [1, 16, 21, 48, 80],
            "dtype": "torch.float32",
            "stored_dtype": "torch.float32",
            "generator_initial_seed": task["seed"],
        }
        receipt = _sign(
            {
                "candidate": _receipt_task_binding(task),
                "artifacts": {
                    "mp4": {"frame_count": 81},
                    "official_initial_gaussian": gaussian,
                },
            }
        )
        candidate_dir = receipt_root / task["candidate_id"]
        candidate_dir.mkdir()
        receipt_path = (
            candidate_dir / "pair-v5-t2v-calibration-receipt.json"
        ).resolve()
        receipt_path.write_bytes(controller.canonical_json_bytes(receipt) + b"\n")
        receipts[task["candidate_id"]] = receipt
        receipt_paths.append(receipt_path)
        rows.append(
            {
                "candidate_id": task["candidate_id"],
                "calibration_group_id": task["calibration_group_id"],
                "semantic_branch": task["semantic_branch"],
                "path": str(receipt_path),
                "file_sha256": controller.file_sha256(receipt_path),
                "receipt_digest": receipt["receipt_digest"],
                "frame_count": 81,
            }
        )
    proofs = []
    for cell in exact8["seed_cells"]:
        tasks = [
            task
            for task in exact8["admission_tasks"]
            if task["calibration_group_id"] == cell["calibration_group_id"]
        ]
        proofs.append(
            generator._gaussian_pair_proof(
                tasks, [receipts[task["candidate_id"]] for task in tasks]
            )
        )
    return (
        _sign(
            {
                "schema_version": generator.AUDIT_SCHEMA,
                "plan_path": built["_path"],
                "plan_file_sha256": built["_file_sha256"],
                "plan_digest": exact8["plan_digest"],
                "dataset": "fit_action_incomplete_repair_exact8",
                "candidate_count": 8,
                "comparator_cell_count": 4,
                "branch_order_per_cell": list(plan.ADMISSION_BRANCH_ORDER),
                "candidate_receipts": rows,
                "action_incomplete_gaussian_pair_proofs": proofs,
                "all_candidates_exact81": True,
                "independent_full81_review_performed": False,
                "review_admission_authorized": False,
                "materializer_same_state_threshold_gate_present": False,
                "q_input_authorized": False,
                "a_min_input_authorized": False,
                "training_performed": False,
                "optimizer_created": False,
                "optimizer_authorized": False,
                "diagnostic_task_count": 0,
                "diagnostic_generation_observed_or_required": False,
            }
        ),
        receipt_paths,
    )


def _validate_generation_audit(
    audit: dict, built: dict, exact8: dict,
    resource: _PhysicalCandidateReceiptResource,
) -> dict:
    sha1 = built["source_specs"][0]["file_sha256"]
    sha2 = built["source_specs"][1]["file_sha256"]
    with mock.patch.object(
        plan, "SEED1_SPEC_SHA256", sha1
    ), mock.patch.object(
        plan, "SEED2_SPEC_SHA256", sha2
    ), mock.patch.object(
        generator, "load_resource_contract", return_value=resource
    ):
        return controller.validate_exact8_audit(audit, exact8)


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


class FitRepairPlanTests(unittest.TestCase):
    def test_prompt_freeze_text_sha_and_locked_failure_authority(self) -> None:
        plan._assert_prompt_freeze()
        self.assertEqual(
            plan.PROMPT_BUNDLE_SHA256,
            "72f3e046966eeeacb18f74d05675a4dcec814498b50f982f6773e64a5605b4d0",
        )
        self.assertIn("both hands firmly on her hips", plan.PROMPTS["arms_action"]["text"])
        self.assertIn("mid-torso level", plan.PROMPTS["arms_incomplete"]["text"])
        self.assertIn("Neither hand ever touches", plan.PROMPTS["arms_incomplete"]["text"])
        self.assertIn("palm flat and facing down", plan.PROMPTS["reach_action"]["text"])
        self.assertIn("never reaches full extension", plan.PROMPTS["reach_incomplete"]["text"])
        self.assertEqual(plan.LOCKED_BLIND_AUTHORITY["action_media_pass"], [2, 4])
        self.assertEqual(plan.LOCKED_BLIND_AUTHORITY["incomplete_media_pass"], [0, 4])
        self.assertEqual(plan.LOCKED_BLIND_AUTHORITY["complete_pair_pass"], [0, 4])

    def test_exact_four_fit_pairs_same_seed_geometry_and_no_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            value = _build_plan(Path(temporary))
            loaded = _load_fixture_plan(value)
        self.assertEqual(loaded["formal_candidate_count"], 8)
        self.assertEqual(loaded["comparator_cell_count"], 4)
        self.assertEqual(
            [row["semantic_branch"] for row in loaded["admission_tasks"]],
            ["action", "incomplete"] * 4,
        )
        self.assertEqual([row["candidate_count"] for row in loaded["shards"]], [4, 4])
        self.assertNotIn("diagnostic_tasks", loaded)
        self.assertEqual(loaded["execution_contract"]["diagnostic_task_count"], 0)
        self.assertEqual(loaded["execution_contract"]["num_inference_steps"], 40)
        self.assertEqual(
            [row["seed"] for row in loaded["seed_cells"]],
            [2026080821, 2026080824, 2026080921, 2026080924],
        )
        self.assertTrue(
            all(row["same_source_geometry"] for row in loaded["seed_cells"])
        )
        self.assertTrue(
            all(
                row["same_seed_and_official_gaussian_required"]
                for row in loaded["seed_cells"]
            )
        )

    def test_resigned_prompt_or_diagnostic_widening_fails_replay(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            value = _build_plan(root)
            original = json.loads(Path(value["_path"]).read_text(encoding="ascii"))
            sha1 = value["source_specs"][0]["file_sha256"]
            sha2 = value["source_specs"][1]["file_sha256"]
            for mutation in ("prompt", "diagnostic"):
                hostile = copy.deepcopy(original)
                if mutation == "prompt":
                    hostile["prompt_freeze"]["prompts"]["arms_action"]["text"] += " hostile"
                else:
                    hostile["diagnostic_tasks"] = [{"candidate_id": "forbidden"}]
                unsigned = dict(hostile)
                unsigned.pop("plan_digest")
                hostile["plan_digest"] = plan.object_sha256(unsigned)
                path = root / f"hostile-{mutation}.json"
                raw = plan.canonical_json_bytes(hostile) + b"\n"
                path.write_bytes(raw)
                with mock.patch.object(
                    plan, "SEED1_SPEC_SHA256", sha1
                ), mock.patch.object(plan, "SEED2_SPEC_SHA256", sha2):
                    with self.assertRaises(plan.FitRepairExact8PlanError):
                        plan.load_plan(
                            path.resolve(strict=True), hashlib.sha256(raw).hexdigest()
                        )


class FitRepairGenerationAndAdmissionTests(unittest.TestCase):
    def test_gaussian_pair_requires_exact_action_incomplete_identity(self) -> None:
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
            generator.FitRepairExact8GenerationError, "did not reuse"
        ):
            generator._gaussian_pair_proof(tasks, receipts)

    def test_completion_reopens_exact8_receipts_and_rejects_proof_omission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            built = _build_plan(root)
            exact8 = _load_fixture_plan(built)
            audit, receipt_paths = _generation_audit(root, built, exact8)
            resource = _PhysicalCandidateReceiptResource()
            _validate_generation_audit(audit, built, exact8, resource)
            self.assertEqual(resource.reopened_paths, receipt_paths)

            hostile = copy.deepcopy(audit)
            hostile.pop("receipt_digest")
            hostile.pop("action_incomplete_gaussian_pair_proofs")
            with self.assertRaisesRegex(
                controller.FitRepairExact8ControllerError, "field closure"
            ):
                _validate_generation_audit(
                    _sign(hostile), built, exact8,
                    _PhysicalCandidateReceiptResource(),
                )

    def test_completion_rejects_resigned_gaussian_proof_tamper(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            built = _build_plan(root)
            exact8 = _load_fixture_plan(built)
            audit, receipt_paths = _generation_audit(root, built, exact8)
            hostile = copy.deepcopy(audit)
            hostile.pop("receipt_digest")
            hostile["action_incomplete_gaussian_pair_proofs"][0][
                "official_gaussian_identity"
            ]["content_sha256"] = "f" * 64
            resource = _PhysicalCandidateReceiptResource()
            with self.assertRaisesRegex(
                controller.FitRepairExact8ControllerError,
                "Gaussian proofs do not replay",
            ):
                _validate_generation_audit(
                    _sign(hostile), built, exact8, resource
                )
            self.assertEqual(resource.reopened_paths, receipt_paths)

    def test_completion_rejects_fully_resigned_candidate_receipt_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            built = _build_plan(root)
            exact8 = _load_fixture_plan(built)
            audit, receipt_paths = _generation_audit(root, built, exact8)
            receipt_path = receipt_paths[0]
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            receipt["candidate"]["seed"] += 1
            receipt = _resign(receipt)
            receipt_path.write_bytes(
                controller.canonical_json_bytes(receipt) + b"\n"
            )

            hostile = copy.deepcopy(audit)
            hostile["candidate_receipts"][0]["file_sha256"] = (
                controller.file_sha256(receipt_path)
            )
            hostile["candidate_receipts"][0]["receipt_digest"] = receipt[
                "receipt_digest"
            ]
            hostile = _resign(hostile)
            resource = _PhysicalCandidateReceiptResource()
            with self.assertRaisesRegex(
                controller.FitRepairExact8ControllerError,
                "candidate receipt/task binding differs",
            ):
                _validate_generation_audit(hostile, built, exact8, resource)
            self.assertEqual(resource.reopened_paths, [receipt_path])

    def test_review_requires_independent_full81_pass_for_all_eight(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            exact8 = _build_plan(Path(temporary))
        receipt = _review(exact8)
        controller.validate_review_admission(receipt, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["candidate_reviews"][1]["reviewed_frame_count"] = 80
        with self.assertRaisesRegex(
            controller.FitRepairExact8ControllerError, "full81 review failed"
        ):
            controller.validate_review_admission(_sign(hostile), exact8)

    def test_materializer_same_state_control_and_amplitude_gates_remain_hard(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            exact8 = _build_plan(Path(temporary))
        receipt = _materializer_gate(exact8)
        controller.validate_materializer_gate(receipt, exact8)
        hostile = copy.deepcopy(receipt)
        hostile.pop("receipt_digest")
        hostile["seed_cell_gates"][0]["control_gates"]["camera_only"] = False
        with self.assertRaisesRegex(
            controller.FitRepairExact8ControllerError, "comparator-cell gate"
        ):
            controller.validate_materializer_gate(_sign(hostile), exact8)

    def test_terminal_gate_requires_exact8_60_56_10ms_52g_contract(self) -> None:
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
            controller.FitRepairExact8ControllerError, "authority differs"
        ):
            controller.validate_terminal_host_gate(_sign(hostile))


class FitRepairReleaseAndLauncherTests(unittest.TestCase):
    def test_release_is_deterministic_exact21_create_only_closure(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        self.assertEqual(manifest["file_count"], 21)
        self.assertEqual(manifest["authority"]["formal_candidate_count"], 8)
        self.assertEqual(manifest["authority"]["diagnostic_task_count"], 0)
        self.assertEqual(
            manifest["authority"]["locked_blind_authority"]["complete_pair_pass"],
            [0, 4],
        )
        self.assertEqual(
            manifest["topology"]["holder"],
            {"job_id": 136140, "node": "auh7-1b-gpu-215"},
        )
        self.assertFalse(
            manifest["resource_specialization"]["static_release_binds_live_child"]
        )
        self.assertEqual(manifest["topology"]["host_memory_request_gib"], 60)
        self.assertEqual(
            manifest["topology"]["host_sampled_current_safe_ceiling_gib"], 56
        )
        self.assertEqual(
            manifest["topology"]["host_cgroup_sample_interval_ns"], 10_000_000
        )
        self.assertEqual(manifest["topology"]["t2v_rank_gpu_memory_limit_gib"], 52)

    def test_archive_extra_member_and_payload_tamper_fail(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        raw = release.build_archive(manifest, payloads)
        with self.assertRaisesRegex(
            release.FitRepairExact8ReleaseError, "member order"
        ):
            release.verify_archive(_tar_with_extra(raw), manifest)
        tampered = bytearray(raw)
        needle = payloads[release.PLAN_MEMBER][:32]
        offset = raw.find(needle)
        self.assertGreaterEqual(offset, 0)
        tampered[offset] ^= 1
        with self.assertRaisesRegex(
            release.FitRepairExact8ReleaseError, "content differs"
        ):
            release.verify_archive(bytes(tampered), manifest)

    def test_build_audit_and_create_only_reuse_rejection(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            archive = root / "source.tar"
            manifest_path = root / "source.manifest.json"
            built = release.build(
                METHOD_ROOT.resolve(strict=True), archive, manifest_path
            )
            audited = release.audit(
                archive.resolve(strict=True), built["archive_sha256"],
                manifest_path.resolve(strict=True), built["manifest_sha256"],
            )
            self.assertEqual(audited["file_count"], 21)
            with self.assertRaisesRegex(
                release.FitRepairExact8ReleaseError, "fresh absolute"
            ):
                release.build(
                    METHOD_ROOT.resolve(strict=True), archive, manifest_path
                )

    def test_launcher_is_exact8_resource_bound_and_parent_safe(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("readonly holder_job=136140", source)
        self.assertIn("readonly holder_node=auh7-1b-gpu-215", source)
        self.assertIn("--gpus-per-task=8", source)
        self.assertIn("--cpus-per-task=32 --mem=60G", source)
        self.assertEqual(source.count("run_exact4_shard sp4-"), 2)
        self.assertIn("formal_candidate_count=8", source)
        self.assertIn("diagnostic_invocations=0", source)
        self.assertIn("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED=1", source)
        self.assertIn("physical_safe_open=true", source)
        self.assertIn("r10_r13_parity=true", source)
        self.assertIn("same_gaussian_action_incomplete_per_cell=true", source)
        self.assertIn("parent_136140_cancelled_released_or_requeued=false", source)
        for forbidden in (
            "--lane diagnostic", "scancel", "scontrol release",
            "scontrol requeue", "optimizer.step",
        ):
            self.assertNotIn(forbidden, source)
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
