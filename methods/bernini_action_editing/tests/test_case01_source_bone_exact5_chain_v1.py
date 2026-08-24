from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import case01_source_bone_exact5_eval_v1 as eval_v1  # noqa: E402
import case01_source_bone_exact5_runner_v1 as runner  # noqa: E402


ASSET_ROOT = (
    REPO_ROOT / "artifacts/object_grounded_case01_0821_bone_interventions_r4"
).resolve()
AUDIT_RECEIPT = (
    REPO_ROOT
    / "md/action_editing/20260821_man/evidence/"
      "case01_exact5_intervention_asset_independent_audit_v1.json"
).resolve()


def _build_plan(output_root: Path) -> dict:
    authority = eval_v1.build_asset_authority(
        ASSET_ROOT / "manifest.json", ASSET_ROOT, AUDIT_RECEIPT
    )
    return eval_v1.build_plan(
        asset_authority=authority,
        checkpoint_manifest={
            **eval_v1.EXPECTED_CHECKPOINT,
            "path": "/authority/checkpoint-00000644/checkpoint_manifest.json",
        },
        producer={
            **eval_v1.EXPECTED_PRODUCER,
            "infer_lora_path": "/release/infer_lora.py",
            "ffprobe_path": "/runtime/ffprobe",
        },
        output_root=output_root,
    )


def _redigest(plan: dict) -> dict:
    plan["plan_digest"] = eval_v1.object_sha256(
        {key: item for key, item in plan.items() if key != "plan_digest"}
    )
    return plan


class FakeFrozenV2:
    def __init__(self, *, parity_sha: str) -> None:
        self.parity_sha = parity_sha

    @staticmethod
    def validate_terminal_checkpoint_manifest(path, sha256):
        return {
            **eval_v1.EXPECTED_CHECKPOINT,
            "path": path,
            "sha256": sha256,
        }

    def verify_arm(self, task, producer, **kwargs):
        del producer, kwargs
        index = eval_v1.TASK_IDS.index(task["task_id"])
        return {
            "task_id": task["task_id"],
            "arm": "full644",
            "receipt_path": task["output"]["receipt_path"],
            "receipt_file_sha256": f"{index + 1:064x}",
            "receipt_digest": f"{index + 11:064x}",
            "output_path": task["output"]["video_path"],
            "output_sha256": (
                self.parity_sha if index == 0 else f"{index + 21:064x}"
            ),
            "output_size": 100 + index,
            "media_probe": {"frame_count": 81},
            "receipt": {
                "sampling": {"seed": 2027, "num_inference_steps": 40},
                "prompt_contract": {"task": "mv2v"},
                "model_consumption": {"model_capture_digest": "a" * 64},
            },
        }


class Exact5PlanAndRunnerTests(unittest.TestCase):
    def test_frozen_rich_audit_receipt_has_exact_schema_and_validates(self) -> None:
        raw = AUDIT_RECEIPT.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        self.assertEqual(len(eval_v1.INDEPENDENT_AUDIT_FIELDS), 32)
        self.assertEqual(set(value), eval_v1.INDEPENDENT_AUDIT_FIELDS)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), eval_v1.INDEPENDENT_AUDIT_SHA256
        )
        self.assertEqual(len(raw), eval_v1.INDEPENDENT_AUDIT_SIZE)
        validated = eval_v1.validate_independent_audit_receipt(
            value,
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )
        self.assertEqual(
            validated["audit_digest"], eval_v1.INDEPENDENT_AUDIT_DIGEST
        )

    def test_rich_audit_extra_and_missing_nested_evidence_fail_closed(self) -> None:
        baseline = json.loads(AUDIT_RECEIPT.read_text(encoding="utf-8"))
        mutations = []
        extra_top = copy.deepcopy(baseline)
        extra_top["unexpected"] = True
        mutations.append(("extra-top", extra_top))
        extra_media = copy.deepcopy(baseline)
        extra_media["videos"]["bone_removed"]["media_contract"][
            "unexpected"
        ] = True
        mutations.append(("extra-media", extra_media))
        missing_evidence = copy.deepcopy(baseline)
        del missing_evidence["translated_symmetry"]["failures"]
        mutations.append(("missing-nested", missing_evidence))
        for label, changed in mutations:
            unsigned = dict(changed)
            unsigned.pop("audit_digest", None)
            changed["audit_digest"] = eval_v1.object_sha256(unsigned)
            raw = eval_v1.canonical_json_bytes(changed) + b"\n"
            with self.subTest(label=label), mock.patch.multiple(
                eval_v1,
                INDEPENDENT_AUDIT_SHA256=hashlib.sha256(raw).hexdigest(),
                INDEPENDENT_AUDIT_SIZE=len(raw),
                INDEPENDENT_AUDIT_DIGEST=changed["audit_digest"],
            ):
                with self.assertRaises(eval_v1.Exact5EvalError):
                    eval_v1.validate_independent_audit_receipt(
                        changed,
                        raw=raw,
                        sha256=hashlib.sha256(raw).hexdigest(),
                        size=len(raw),
                    )

    def test_final_asset_manifest_and_exact5_plan_are_bound(self) -> None:
        self.assertEqual(
            hashlib.sha256((ASSET_ROOT / "manifest.json").read_bytes()).hexdigest(),
            eval_v1.ASSET_MANIFEST_SHA256,
        )
        with tempfile.TemporaryDirectory() as value:
            plan = _build_plan(Path(value).resolve())
        self.assertEqual(plan["task_count"], 5)
        self.assertEqual(
            tuple(task["task_id"] for task in plan["tasks"]), eval_v1.TASK_IDS
        )
        self.assertEqual(
            tuple(row["variant"] for row in plan["asset_authority"]["sources"]),
            eval_v1.VARIANT_ORDER,
        )
        self.assertTrue(plan["asset_authority"]["launch_allowed"])
        self.assertEqual(
            plan["asset_authority"]["independent_visual_audit_status"],
            "PASS_P0_0_P1_0",
        )

    def test_hostile_asset_and_plan_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            plan = _build_plan(Path(value).resolve())
            mutations = []
            for mutate in (
                lambda row: row["asset_authority"].update(launch_allowed=False),
                lambda row: row["asset_authority"].update(
                    independent_visual_audit_status="PENDING"
                ),
                lambda row: row["asset_authority"]["sources"][1].update(
                    sha256="0" * 64
                ),
                lambda row: row["tasks"].reverse(),
                lambda row: row["tasks"][0].update(arm="base"),
                lambda row: row.update(task_count=4),
                lambda row: row["producer"].update(unexpected=True),
                lambda row: row["checkpoint_manifest"].update(unexpected=True),
                lambda row: row["tasks"][0]["adapter"].update(unexpected=True),
                lambda row: row["tasks"][1].update(
                    output=copy.deepcopy(row["tasks"][0]["output"])
                ),
            ):
                changed = copy.deepcopy(plan)
                mutate(changed)
                mutations.append(_redigest(changed))
            for changed in mutations:
                with self.subTest(change=changed):
                    with self.assertRaises(eval_v1.Exact5EvalError):
                        eval_v1.validate_plan(changed)

    def test_publication_alias_filename_and_symlink_parent_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value).resolve()
            plan = _build_plan(root)
            video_name = plan["tasks"][0]["output"]["video_path"].rsplit("/", 1)[1]
            receipt_name = video_name + ".receipt.json"

            double_slash = copy.deepcopy(plan)
            double_slash["tasks"][0]["output"] = {
                "video_path": str(root) + "//" + video_name,
                "receipt_path": str(root) + "//" + receipt_name,
                "create_only": True,
            }
            parent_escape = copy.deepcopy(plan)
            parent_escape["tasks"][0]["output"] = {
                "video_path": str(root) + "/unused/../" + video_name,
                "receipt_path": str(root) + "/unused/../" + receipt_name,
                "create_only": True,
            }
            wrong_name = copy.deepcopy(plan)
            wrong_name["tasks"][0]["output"] = {
                "video_path": str(root / "wrong-task.mp4"),
                "receipt_path": str(root / "wrong-task.mp4.receipt.json"),
                "create_only": True,
            }
            duplicate_physical_leaf = copy.deepcopy(plan)
            duplicate_physical_leaf["tasks"][1]["output"] = copy.deepcopy(
                duplicate_physical_leaf["tasks"][0]["output"]
            )
            for label, changed in (
                ("double-slash", double_slash),
                ("dot-dot", parent_escape),
                ("wrong-name", wrong_name),
                ("duplicate-physical-leaf", duplicate_physical_leaf),
            ):
                with self.subTest(label=label):
                    with self.assertRaises(eval_v1.Exact5EvalError):
                        eval_v1.validate_plan(_redigest(changed))

            real_root = root / "real"
            real_root.mkdir()
            alias_root = root / "alias"
            alias_root.symlink_to(real_root, target_is_directory=True)
            symlink_parent = _build_plan(real_root.resolve())
            for task in symlink_parent["tasks"]:
                task["output"] = {
                    "video_path": str(alias_root / (task["task_id"] + ".mp4")),
                    "receipt_path": str(
                        alias_root / (task["task_id"] + ".mp4.receipt.json")
                    ),
                    "create_only": True,
                }
            with self.assertRaises(eval_v1.Exact5EvalError):
                eval_v1.validate_plan(_redigest(symlink_parent))

    def test_runner_rechecks_publication_internal_disjointness(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("publication_paths & internal_paths", source)
        with tempfile.TemporaryDirectory() as value:
            plan = _build_plan(Path(value).resolve())
            tasks = runner.validate_task_order(plan)
        leaves = {
            Path(task["output"][name])
            for task in tasks
            for name in ("video_path", "receipt_path")
        }
        self.assertEqual(len(leaves), 10)

    def test_deterministic_original_parity_is_a_hard_separate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            plan = _build_plan(Path(value).resolve())
            publications = {task_id: {} for task_id in eval_v1.TASK_IDS}
            report = eval_v1.verify_results(
                plan,
                frozen_v2=FakeFrozenV2(
                    parity_sha=eval_v1.REFERENCE_EXACT_ORIGINAL_R64_OUTPUT_SHA256
                ),
                publication_root_fd=123,
                ffprobe_authority={},
                publication_authorities=publications,
            )
            parity = report["deterministic_reference_parity"]
            self.assertEqual(parity["status"], "PASS")
            self.assertEqual(parity["policy"], "HARD_FAIL")
            self.assertTrue(
                parity["kept_separate_from_intervention_effect_interpretation"]
            )
            with self.assertRaisesRegex(
                eval_v1.Exact5EvalError, "deterministic parity failed"
            ):
                eval_v1.verify_results(
                    plan,
                    frozen_v2=FakeFrozenV2(parity_sha="f" * 64),
                    publication_root_fd=123,
                    ffprobe_authority={},
                    publication_authorities=publications,
                )

    def test_reused_execution_primitives_are_from_pinned_frozen_sources(self) -> None:
        frozen_path = str(METHOD_ROOT / runner._FROZEN_RUNNER_BASENAME)
        for function in (
            runner.frozen.RunnerExecution.capture_model,
            runner.frozen.RunnerExecution.execute_one,
            runner.frozen.RunnerExecution.run,
            runner.frozen._run_subprocess,
            runner.frozen.build_torchrun_argv,
            runner.frozen.create_publication_handoff,
            runner.frozen.read_sealed_publication_handoff,
            runner.frozen.build_eval_consumption_chain,
        ):
            with self.subTest(function=function.__qualname__):
                self.assertEqual(function.__code__.co_filename, frozen_path)
        bridge = (
            METHOD_ROOT / "full644_exploratory_matched_torchrun_fd_bridge_v2.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--nproc_per_node=4"', bridge)
        self.assertIn("spawned_ranks != {0, 1, 2, 3}", bridge)
        self.assertIn("--max_restarts=0", bridge)
        self.assertIn("pass_fds=pass_fds", bridge)

    def test_all_exact16_and_canary2_assumptions_are_explicitly_rebound(self) -> None:
        self.assertEqual(runner.frozen.TASK_IDS, eval_v1.TASK_IDS)
        self.assertEqual(runner.frozen.CANARY_TASK_IDS, eval_v1.TASK_IDS)
        self.assertEqual(runner.frozen.CASE00_CANARY_CAMPAIGN, eval_v1.CAMPAIGN)
        self.assertEqual(runner.frozen.FULL16_CAMPAIGN, "disabled-frozen-full16")
        with tempfile.TemporaryDirectory() as value:
            plan = _build_plan(Path(value).resolve())
            self.assertEqual(
                tuple(
                    task["task_id"]
                    for task in runner.select_campaign_tasks(plan, eval_v1.CAMPAIGN)
                ),
                eval_v1.TASK_IDS,
            )
            with self.assertRaisesRegex(
                runner.frozen.MatchedRunnerV2Error,
                "only the exact5 campaign",
            ):
                runner.select_campaign_tasks(plan, "case00-pair-canary")

    def test_task_index_wrapper_rejects_old_range_before_frozen_replay(self) -> None:
        arguments = (Path("/output"), 10, {}, {}, {}, {})
        with mock.patch.object(
            runner.frozen, "replay_task_authority_artifacts", return_value={"ok": True}
        ) as replay:
            for bad_index in (-1, 5, 15):
                row = {
                    "task_index": bad_index,
                    "task_id": eval_v1.TASK_IDS[0],
                }
                with self.assertRaisesRegex(
                    runner.frozen.MatchedRunnerV2Error,
                    "exact5 task result index differs",
                ):
                    runner.replay_task_authority_artifacts_exact5(
                        arguments[0], arguments[1], row, *arguments[3:]
                    )
            replay.assert_not_called()

    def test_final_attestation_has_no_pair_or_full16_count_semantics(self) -> None:
        task_results = [
            {
                "task_index": index,
                "task_id": task_id,
                "model_capture_digest": "a" * 64,
                "consumption_digest": f"{index + 1:064x}",
                "ffmpeg_exec_authority_digest": "b" * 64,
                "task_result_digest": f"{index + 11:064x}",
                "environment_digest": f"{index + 21:064x}",
            }
            for index, task_id in enumerate(eval_v1.TASK_IDS)
        ]
        model_final = {
            "task_count": 5,
            "model_capture_digest": "a" * 64,
            "task_consumption_digests": [
                row["consumption_digest"] for row in task_results
            ],
        }
        execution = types.SimpleNamespace(
            output_root_fd=20,
            output_root_identity={},
            output_root=Path("/output"),
            ffmpeg_exec_authority_digest="b" * 64,
            ffprobe_authority={
                "authority_digest": "c" * 64,
                "fd": 21,
                "source_path": "/ffprobe",
                "sha256": "d" * 64,
            },
            publication_authorities={
                task_id: {
                    "authority_digest": "e" * 64,
                    "receipt_fd": 30 + index,
                    "output_fd": 40 + index,
                }
                for index, task_id in enumerate(eval_v1.TASK_IDS)
            },
            publication_handoffs={
                task_id: {"authority_digest": "f" * 64, "fd": 50 + index}
                for index, task_id in enumerate(eval_v1.TASK_IDS)
            },
            run=lambda: (task_results, model_final),
        )
        tasks = [{"task_id": task_id} for task_id in eval_v1.TASK_IDS]
        report = {
            "results": [{"task_id": task_id} for task_id in eval_v1.TASK_IDS],
            "formal_full16_report": False,
            "manual_blind_review_required": True,
            "retained_publication_root_fd_replayed": True,
            "retained_ffprobe_executable_fd_replayed": True,
            "retained_publication_leaf_fds_replayed": True,
            "report_digest": "1" * 64,
            "task_count": 5,
        }
        args = types.SimpleNamespace(
            campaign_mode=eval_v1.CAMPAIGN,
            entry_authority={},
            plan=str(Path(runner.__file__).resolve()),
            plan_sha256="2" * 64,
        )
        final_parents = {
            "output_report": {
                "path": Path("/final/report.json"),
                "parent_fd": 60,
                "parent_identity": {},
            },
            "runner_attestation": {
                "path": Path("/final/attestation.json"),
                "parent_fd": 61,
                "parent_identity": {},
            },
        }
        entry = {
            "authority_digest": "3" * 64,
            "release_digest": "4" * 64,
            "bootstrap_sha256": "5" * 64,
        }
        artifact = {"native_receipt_mode": 0o400, "native_receipt_nlink": 1}
        with mock.patch.object(
            runner.frozen, "validate_captured_runner_entry", return_value=entry
        ), mock.patch.object(runner.frozen, "_validate_held_directory"), mock.patch.object(
            runner.frozen, "_validate_embedded_digest"
        ), mock.patch.object(
            runner.exact5, "verify_results", return_value=report
        ), mock.patch.object(
            runner, "replay_task_authority_artifacts_exact5", return_value=artifact
        ), mock.patch.object(
            runner.frozen, "_validate_final_parent"
        ), mock.patch.object(
            runner.frozen,
            "_write_json_at",
            side_effect=[(Path("/final/report.json"), "6" * 64), (Path("/final/attestation.json"), "7" * 64)],
        ), mock.patch.object(
            runner.frozen,
            "read_sealed_publication_handoff",
            return_value={"payload_digest": "8" * 64},
        ):
            attestation = runner._complete_execution(
                args,
                {"plan_digest": "9" * 64},
                tasks,
                {"physical_bindings_digest": "a" * 64},
                execution,
                final_parents,
            )
        self.assertEqual(attestation["task_count"], 5)
        self.assertEqual(attestation["unselected_task_count"], 0)
        self.assertEqual(attestation["unselected_task_ids"], [])
        self.assertFalse(attestation["formal_full16_report"])
        self.assertTrue(attestation["manual_blind_review_required"])
        self.assertTrue(attestation["all_model_adapter_post_use_replays_complete"])

    def test_wrapper_main_never_delegates_to_frozen_main(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("frozen.main(", source)
        self.assertIn("frozen._require_isolated_runner_startup()", source)
        self.assertIn("frozen.validate_captured_runner_entry()", source)


if __name__ == "__main__":
    unittest.main()
