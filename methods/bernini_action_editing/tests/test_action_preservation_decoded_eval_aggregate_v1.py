from __future__ import annotations

import copy
import hashlib
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_aggregate_v1 as aggregate
import action_preservation_decoded_eval_executor_v1 as executor
import action_preservation_decoded_eval_plan_v1 as plan


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def action_contract(iid: str) -> dict:
    description = f"Complete the fitted action for source {iid}, then hold the terminal pose."
    row = {
        "schema_version": plan.ACTION_REVIEW_CONTRACT_SCHEMA,
        "action_order_description": description,
        "action_order_description_sha256": plan.text_sha256(description),
        "expected_onset_frame_min": 4,
        "expected_onset_frame_max": 20,
        "terminal_hold_start_frame_min": 65,
        "terminal_hold_end_frame": 80,
        "full_video_frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
    }
    row["contract_digest"] = plan.object_sha256(row)
    return row


class AggregateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name).resolve()
        sources = []
        for index, iid in enumerate(plan.FITTED_IIDS):
            instruction = f"Perform the fitted action for source {iid}."
            sources.append(
                {
                    "iid": iid,
                    "source_video_sha256": digest(f"source:{iid}"),
                    "source_receipt_sha256": digest(f"source-receipt:{iid}"),
                    "instruction": instruction,
                    "instruction_sha256": plan.text_sha256(instruction),
                    "action_review_contract": action_contract(iid),
                    "seed": 2026081801 + index,
                }
            )
        checkpoints = [
            {
                "arm": arm,
                "checkpoint_step": step,
                "checkpoint_receipt_sha256": digest(f"receipt:{arm}:{step}"),
                "adapter_sha256": digest(f"adapter:{arm}:{step}"),
            }
            for arm in plan.ARMS
            for step in plan.CHECKPOINT_STEPS
        ]
        input_spec = plan.build_input_spec(
            evaluation_id="aggregate-contract-test",
            evaluation_root=root / "evaluation",
            pins={
                key: None if key == "calibration_digest" else digest(key)
                for key in plan.PIN_FIELDS
            },
            sources=sources,
            checkpoints=checkpoints,
        )
        self.bundle = plan.build_bundle(input_spec)
        self.bundle["publication_receipt"] = plan.build_publication_receipt(
            self.bundle
        )
        self.bindings = {
            "physical_bindings_digest": digest("physical bindings"),
            "calibration_digest": None,
            "sources": [
                {
                    "iid": item["iid"],
                    "source_video": {
                        "path": str(root / f"source-{item['iid']}.mp4"),
                        "sha256": item["source_video_sha256"],
                    },
                }
                for item in sources
            ],
        }
        self.outputs = []
        self.aggregate_capture = {
            "receipt_path": str(root / "aggregate-runtime-capture.json"),
            "receipt_sha256": digest("aggregate capture file"),
            "capture_digest": digest("aggregate capture"),
            "target": "action_preservation_decoded_eval_aggregate_v1.py",
            "target_arguments_sha256": digest("aggregate arguments"),
        }
        self.summaries = [
            {
                "job_id": holder["job_id"],
                "node": holder["node"],
                "summary_path": f"/summary/{holder['job_id']}",
                "summary_sha256": digest("summary file:" + holder["job_id"]),
                "summary_digest": digest("summary digest:" + holder["job_id"]),
                "holder_execution_digest": digest(
                    "holder execution:" + holder["job_id"]
                ),
                "executor_verified_release_capture": {
                    "receipt_path": str(
                        root / f"capture-{holder['job_id']}.json"
                    ),
                    "receipt_sha256": digest(
                        "capture file:" + holder["job_id"]
                    ),
                    "capture_digest": digest(
                        "capture digest:" + holder["job_id"]
                    ),
                    "target": (
                        "action_preservation_decoded_eval_executor_v1.py"
                    ),
                    "target_arguments_sha256": digest(
                        "capture arguments:" + holder["job_id"]
                    ),
                },
            }
            for holder in plan.HOLDER_ROWS
        ]
        for record in self.bundle["manifest"]["frozen_base_controls"]:
            self.outputs.append(
                {
                    "task_kind": "frozen_base_control",
                    "task_id": record["control_id"],
                    "record": record,
                    "output_path": str(root / "leaky" / record["output_relpath"]),
                    "output_video_sha256": digest("video:" + record["control_id"]),
                    "output_receipt_path": str(root / "receipts" / record["control_id"]),
                    "output_receipt_sha256": digest("output receipt:" + record["control_id"]),
                    "output_digest": digest("output digest:" + record["control_id"]),
                }
            )
        for record in self.bundle["manifest"]["candidates"]:
            self.outputs.append(
                {
                    "task_kind": "adapter_candidate",
                    "task_id": record["candidate_id"],
                    "record": record,
                    "output_path": str(root / "leaky" / record["output_relpath"]),
                    "output_video_sha256": digest("video:" + record["candidate_id"]),
                    "output_receipt_path": str(root / "receipts" / record["candidate_id"]),
                    "output_receipt_sha256": digest("output receipt:" + record["candidate_id"]),
                    "output_digest": digest("output digest:" + record["candidate_id"]),
                }
            )

    def test_exact256_opaque_packet_and_missing_calibration_abstain(self) -> None:
        value, private, public = aggregate.build_blind_packet(
            bundle=self.bundle,
            bindings=self.bindings,
            outputs=self.outputs,
            summaries=self.summaries,
            blinding_key=b"k" * 32,
            aggregate_verified_release_capture=self.aggregate_capture,
        )
        self.assertEqual(public["row_count"], 256)
        self.assertEqual(private["row_count"], 256)
        self.assertEqual(value["total_output_count"], 264)
        self.assertEqual(value["machine_status"], "ABSTAIN_CALIBRATION_MISSING")
        self.assertEqual(value["blind_review_status"], "WAIT_FOR_BLIND_REVIEW")
        self.assertEqual(value["next_action"], "WAIT_FOR_BLIND_REVIEW")
        self.assertFalse(value["scientific_promotion_authorized"])
        public_text = aggregate.canonical_json_bytes(public).decode("utf-8")
        for forbidden in [
            *plan.ARMS, '"checkpoint_step":', '"onset_policy":', "/leaky/"
        ]:
            self.assertNotIn(forbidden, public_text)
        self.assertNotIn((b"k" * 32).hex(), public_text)
        self.assertTrue(
            all(row["required_axes"] == list(plan.REVIEW_AXES) for row in public["rows"])
        )
        self.assertTrue(
            all(row["minimum_independent_reviewer_count"] == 2 for row in public["rows"])
        )
        self.assertTrue(
            all(
                row["instruction_sha256"]
                == plan.text_sha256(row["instruction"])
                and row["action_review_contract_digest"]
                == row["action_review_contract"]["contract_digest"]
                and row["matched_base_media_sha256"]
                and row["matched_base_output_digest"]
                and row["matched_base_full_video_receipt_sha256"]
                and row["blind_row_digest"]
                for row in public["rows"]
            )
        )
        normalized = plan.gate._closed_packet_authority(
            evaluation_aggregate=value,
            public_packet=public,
            private_mapping=private,
        )
        self.assertEqual(normalized[0]["aggregate_digest"], value["aggregate_digest"])

    def test_matched_base_seed_or_policy_mismatch_fails_closed(self) -> None:
        hostile = copy.deepcopy(self.outputs)
        candidate = next(
            item for item in hostile if item["task_kind"] == "adapter_candidate"
        )
        candidate["record"]["seed"] += 1
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError, "pairing differs"
        ):
            aggregate.build_blind_packet(
                bundle=self.bundle,
                bindings=self.bindings,
                outputs=hostile,
                summaries=self.summaries,
                blinding_key=b"k" * 32,
                aggregate_verified_release_capture=self.aggregate_capture,
            )

    def test_injected_summary_cannot_claim_evaluation_complete(self) -> None:
        shard = self.bundle["shards"][plan.HOLDER_ROWS[0]["job_id"]]
        results = [
            {
                "task_id": executor._task_id(task),
                "status": "success",
                "terminal_receipt_digest": digest(executor._task_id(task)),
                "output_relpath": task["record"]["output_relpath"],
            }
            for task in shard["tasks"]
        ]
        summary = executor.build_shard_summary(
            bundle=self.bundle, shard=shard, results=results, verify_tools=False
        )
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "holder execution authority|production execution",
        ):
            aggregate._validate_summary(summary, bundle=self.bundle, shard=shard)


if __name__ == "__main__":
    unittest.main()
